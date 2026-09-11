#include "nbody_common.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

/* =========================================================================
 * This program benchmarks the classic "all-pairs" O(N^2) gravitational
 * N-body force kernel using two different in-memory data layouts:
 *
 *   - AoS (Array of Structures): one struct per particle, all of that
 *     particle's fields stored contiguously.
 *   - SoA (Structure of Arrays): one array per physical component (all x's
 *     together, all y's together, etc.).
 *
 * The point of the benchmark is to show how memory layout affects
 * auto-vectorization (SIMD) and cache behavior for the same math.
 * ========================================================================= */

typedef struct particle_aos_s
{
  /* Array-of-Structures layout: each particle owns all its fields together.
   * This is intuitive, but the inner force loop must stride through x/y/z with
   * unused velocity/acceleration fields in between. */
  dtype x, y, z;
  dtype vx, vy, vz;
  dtype ax, ay, az;
} particle_aos_t;



typedef struct particles_soa_s
{
  /* Structure-of-Arrays layout: each physical component is contiguous. */
  size_t n;      /* number of particles */
  dtype *x, *y, *z;     /* position components, one contiguous array each */
  dtype *vx, *vy, *vz;  /* velocity components (read from file, unused in force calc) */
  dtype *ax, *ay, *az;  /* acceleration components, written by compute_soa() */
} particles_soa_t;


typedef enum layout_e
{
  /* Benchmark one layout or both from the same input file. */
  LAYOUT_SOA,
  LAYOUT_AOS,
  LAYOUT_BOTH
} layout_t;

typedef enum rsqrt_mode_e
{
  /* Keep the same exact/approx inverse-square-root choice as the hybrid solver
   * so layout results can be interpreted alongside the ablation study. */
  RSQRT_EXACT,
  RSQRT_APPROX
} rsqrt_mode_t;

/* Print an error message to stderr (printf-style) and terminate the process
 * with a failure status. Used everywhere as the single "fatal error" path. */
static _Noreturn void die(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fputc('\n', stderr);
  exit(EXIT_FAILURE);
}

/* Parse a command-line argument as an unsigned integer (size_t), rejecting
 * anything that isn't a clean, in-range, fully-consumed number. `name` is
 * only used to produce a helpful error message. */
static size_t parse_size(const char *text, const char *name)
{
  char *end = NULL;
  unsigned long long value;
  errno = 0;
  value = strtoull(text, &end, 10);
  if ((errno != 0) || (end == text) || (*end != '\0') ||
      (value > (unsigned long long)SIZE_MAX))
    die("invalid integer for %s: %s", name, text);
  return (size_t)value;
}

/* Parse a command-line argument as a floating-point value in the project's
 * `dtype` (float or double, depending on how nbody_common.h is configured),
 * with the same strict validation approach as parse_size(). */
static dtype parse_dtype(const char *text, const char *name)
{
  char *end = NULL;
  double value;
  errno = 0;
  value = strtod(text, &end);
  if ((errno != 0) || (end == text) || (*end != '\0') || !isfinite(value) ||
      (fabs(value) > (double)DTYPE_MAX_VALUE))
    die("invalid floating-point value for %s: %s", name, text);
  return (dtype)value;
}

/* Small helper for a minimal argv parser: supports both "--key value" and
 * "--key=value" forms. `i` is the current index into argv and gets advanced
 * by one if the "--key value" (separate token) form is used, so the caller's
 * for-loop naturally skips the consumed value token. Returns NULL if `arg`
 * doesn't match `key` at all (caller then tries the next known flag). */
static const char *option_value(int *i, int argc, char **argv,
                                const char *key)
{
  const size_t key_len = strlen(key);
  const char *arg = argv[*i];
  if ((strncmp(arg, key, key_len) == 0) && (arg[key_len] == '='))
    return arg + key_len + 1;
  if (strcmp(arg, key) == 0)
  {
    if (*i + 1 >= argc)
      die("missing value after %s", key);
    *i += 1;
    return argv[*i];
  }
  return NULL;
}

/* --layout=soa|aos|both -> enum. Exits with an error on anything else. */
static layout_t parse_layout(const char *text)
{
  if (strcmp(text, "soa") == 0)
    return LAYOUT_SOA;
  if (strcmp(text, "aos") == 0)
    return LAYOUT_AOS;
  if (strcmp(text, "both") == 0)
    return LAYOUT_BOTH;
  die("invalid --layout '%s' (expected soa, aos, or both)", text);
}

/* Inverse of parse_layout(), used when printing the CSV output row. */
static const char *layout_name(layout_t layout)
{
  if (layout == LAYOUT_SOA)
    return "soa";
  if (layout == LAYOUT_AOS)
    return "aos";
  return "both";
}

/* --rsqrt=exact|approx -> enum. */
static rsqrt_mode_t parse_rsqrt_mode(const char *text)
{
  if (strcmp(text, "exact") == 0)
    return RSQRT_EXACT;
  if (strcmp(text, "approx") == 0)
    return RSQRT_APPROX;
  die("invalid --rsqrt '%s' (expected exact or approx)", text);
}

/* Inverse of parse_rsqrt_mode(), used when printing the CSV output row. */
static const char *rsqrt_mode_name(rsqrt_mode_t mode)
{
  return mode == RSQRT_APPROX ? "approx" : "exact";
}

/* Computes 1/sqrt(r2), i.e. the inverse distance between two particles,
 * which is the quantity needed to turn a squared distance into a 1/r^3
 * force-law factor without ever calling a (slower) division-based sqrt.
 *
 * - RSQRT_EXACT: just calls the platform sqrt (dtype_sqrt, float or double
 *   depending on dtype) and divides. This is the numerically "correct"
 *   reference used to validate the approximate path.
 * - RSQRT_APPROX: classic fast-inverse-square-root style approximation.
 *   It seeds x with a low-precision (float) 1/sqrt(r2), then sharpens the
 *   result with two iterations of Newton-Raphson on f(x) = 1/x^2 - r2,
 *   whose update rule is x_{n+1} = x_n * (1.5 - 0.5 * r2 * x_n^2). Two
 *   iterations are enough to bring it back close to full dtype precision
 *   while staying cheaper than a hardware sqrt+division on some targets. */
static inline dtype invsqrt_force(dtype r2, rsqrt_mode_t mode)
{
  /* Exact mode is the numerical reference; approximate mode uses the same
   * low-precision seed plus Newton refinement strategy as the hybrid solver. */
  if (mode == RSQRT_EXACT)
    return (dtype)1.0 / dtype_sqrt(r2);
  else
  {
    dtype x = (dtype)(1.0f / sqrtf((float)r2)); /* single-precision seed */
    const dtype half = (dtype)0.5;
    const dtype three_halves = (dtype)1.5;
    x = x * (three_halves - half * r2 * x * x); /* Newton iteration #1 */
    x = x * (three_halves - half * r2 * x * x); /* Newton iteration #2 */
    return x;
  }
}

/* Allocates `nbytes` rounded up to NBODY_ALIGNMENT, using aligned_alloc so
 * that every buffer (AoS array or each SoA component array) starts on the
 * same alignment boundary. This matters for the benchmark's fairness: if one
 * layout's buffers happened to be better-aligned than the other's, that could
 * bias the timing independently of the layout itself. Returns NULL only for
 * a zero-byte request; any real allocation failure is fatal via die(). */
static void *checked_aligned_alloc(size_t nbytes)
{
  /* Align both AoS and SoA storage so the comparison does not accidentally favor
   * one layout because of a different base-address alignment. */
  const size_t alignment = NBODY_ALIGNMENT;
  const size_t padded = ((nbytes + alignment - 1u) / alignment) * alignment;
  void *ptr;
  if (nbytes == 0u)
    return NULL;
  ptr = aligned_alloc(alignment, padded);
  if (ptr == NULL)
    die("aligned_alloc failed for %zu bytes", padded);
  return ptr;
}

/* fread() wrapper that treats a short read (including EOF) as a fatal,
 * descriptive error instead of silently leaving `ptr` partially filled. */
static void checked_fread(void *ptr, size_t size, size_t nmemb, FILE *fp,
                          const char *path, const char *what)
{
  if (fread(ptr, size, nmemb, fp) != nmemb)
    die("failed reading %s from '%s'", what, path);
}

/* Resets a particles_soa_t to a known "empty" state: all pointers NULL
 * and count zero. Used both before first allocation and after
 * particles_soa_free() to avoid leaving dangling pointers around. */
static void particles_soa_init(particles_soa_t *p)
{
  memset(p, 0, sizeof(*p));
}

/* Allocates the nine component arrays (x/y/z, vx/vy/vz, ax/ay/az) for `n`
 * particles, each sized n * sizeof(dtype) and aligned via
 * checked_aligned_alloc(). Velocity/acceleration arrays are allocated too
 * (even though only x/y/z and a{x,y,z} participate in the force kernel)
 * so that read_particles() can populate the full particle state uniformly. */
static void particles_soa_allocate(particles_soa_t *p, size_t n)
{
  /* Allocate the SoA representation used by the optimized/reference layout. */
  const size_t bytes = n * sizeof(dtype);
  particles_soa_init(p);
  p->n = n;
  p->x = checked_aligned_alloc(bytes);
  p->y = checked_aligned_alloc(bytes);
  p->z = checked_aligned_alloc(bytes);
  p->vx = checked_aligned_alloc(bytes);
  p->vy = checked_aligned_alloc(bytes);
  p->vz = checked_aligned_alloc(bytes);
  p->ax = checked_aligned_alloc(bytes);
  p->ay = checked_aligned_alloc(bytes);
  p->az = checked_aligned_alloc(bytes);
}

/* Frees all nine SoA component arrays and resets the struct back to the
 * "empty" state via particles_soa_init(), so it's safe to reuse or to call
 * this function twice by accident. */
static void particles_soa_free(particles_soa_t *p)
{
  free(p->x);
  free(p->y);
  free(p->z);
  free(p->vx);
  free(p->vy);
  free(p->vz);
  free(p->ax);
  free(p->ay);
  free(p->az);
  particles_soa_init(p);
}

/* Allocates a single contiguous, aligned array of `n` particle_aos_t
 * structs and zero-initializes it (so acceleration fields start at 0
 * before the first force computation, and unused padding is deterministic). */
static particle_aos_t *particles_aos_allocate(size_t n)
{
  /* Allocate the AoS representation used as the contrast case. */
  particle_aos_t *p = checked_aligned_alloc(n * sizeof(*p));
  memset(p, 0, n * sizeof(*p));
  return p;
}

/* Reads a custom little binary format from `path`:
 *   [magic bytes][uint64 particle count][per-particle record]*
 * where each record is NBODY_BINARY_COMPONENTS 32-bit floats, presumably
 * (x, y, z, vx, vy, vz). The same values are written into *both* the SoA
 * and AoS representations in the same pass, so any downstream timing
 * difference between compute_soa() and compute_aos() reflects the layout
 * only, not different random/generated input data. */
static void read_particles(const char *path, particles_soa_t *soa,
                           particle_aos_t **aos_out)
{
  /* Read once and populate both layouts with identical values.  This removes
   * input-generation noise from the AoS-vs-SoA comparison. */
  FILE *fp = fopen(path, "rb");
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t i, n;
  particle_aos_t *aos;

  if (fp == NULL)
    die("cannot open input '%s'", path);
  checked_fread(magic, 1u, NBODY_BINARY_MAGIC_SIZE, fp, path, "magic");
  if (memcmp(magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die("invalid input magic in '%s'", path);
  checked_fread(&n64, sizeof n64, 1u, fp, path, "particle count");
  if ((uint64_t)(size_t)n64 != n64)
    die("input particle count is too large"); /* guards 32-bit size_t platforms */

  n = (size_t)n64;
  particles_soa_allocate(soa, n);
  aos = particles_aos_allocate(n);

  for (i = 0u; i < n; ++i)
  {
    /* Records are stored as plain 32-bit floats regardless of the build's
     * `dtype`, then converted/widened (or narrowed) to dtype on load. */
    float rec[NBODY_BINARY_COMPONENTS];
    checked_fread(rec, sizeof rec[0], NBODY_BINARY_COMPONENTS, fp, path,
                  "particle record");
    soa->x[i] = aos[i].x = (dtype)rec[0];
    soa->y[i] = aos[i].y = (dtype)rec[1];
    soa->z[i] = aos[i].z = (dtype)rec[2];
    soa->vx[i] = aos[i].vx = (dtype)rec[3];
    soa->vy[i] = aos[i].vy = (dtype)rec[4];
    soa->vz[i] = aos[i].vz = (dtype)rec[5];
  }

  fclose(fp);
  *aos_out = aos;
}

/* Core O(N^2) all-pairs gravitational force kernel, SoA version.
 *
 * Outer loop (over i, the particle receiving the force) is parallelized
 * across threads with OpenMP (`#pragma omp parallel for`), one particle's
 * full inner sum handled per iteration -> no data race, no reduction needed
 * across threads. Each thread's own accumulation only ever writes to its
 * own p->a{x,y,z}[i], so `schedule(static)` (fixed, cache-friendly chunking)
 * is safe and cheap.
 *
 * Inner loop (over j, every other particle contributing force) is hinted to
 * the compiler with `#pragma omp simd` because x[j]/y[j]/z[j] are
 * contiguous arrays here: the compiler can load 4/8 j's worth of doubles/
 * floats at once into a SIMD register, which is exactly the layout
 * advantage SoA is meant to demonstrate versus AoS below. */
static void compute_soa(particles_soa_t *p, dtype g, dtype mass, dtype eps,
                        rsqrt_mode_t rsqrt_mode)
{
  /* SoA force kernel: source x/y/z arrays are contiguous, which helps SIMD and
   * cache-line utilization in the innermost loop. */
  const dtype eps2 = eps * eps;   /* softening squared, avoids singularity at r=0 */
  const dtype gm = g * mass;      /* precompute G * m since mass is uniform */
  size_t i;

#pragma omp parallel for schedule(static)
  for (i = 0u; i < p->n; ++i)
  {
    const dtype xi = p->x[i], yi = p->y[i], zi = p->z[i];
    dtype ax = (dtype)0.0, ay = (dtype)0.0, az = (dtype)0.0;
    size_t j;
#pragma omp simd reduction(+ : ax, ay, az)
    for (j = 0u; j < p->n; ++j)
    {
      /* Vector from particle i to particle j. Note j == i is not skipped:
       * dx=dy=dz=0 there, so r2 == eps2 and the self-term contributes a
       * finite (not NaN/inf) but harmless value thanks to the softening. */
      const dtype dx = p->x[j] - xi;
      const dtype dy = p->y[j] - yi;
      const dtype dz = p->z[j] - zi;
      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = invsqrt_force(r2, rsqrt_mode);
      /* invr^3 turns 1/r into the 1/r^2 force-law magnitude combined with
       * a 1/r to renormalize (dx,dy,dz) into a unit direction vector. */
      const dtype s = gm * invr * invr * invr;
      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }
    p->ax[i] = ax;
    p->ay[i] = ay;
    p->az[i] = az;
  }
}

/* Same math as compute_soa(), but reading from an array of particle_aos_t
 * structs instead of separate component arrays. Deliberately has NO
 * `#pragma omp simd` on the inner loop: because p[j].x, p[j].y, p[j].z are
 * interleaved with vx/vy/vz/ax/ay/az inside each struct, consecutive j's
 * worth of x-coordinates are NOT contiguous in memory (they're
 * sizeof(particle_aos_t) apart). A compiler can still try to
 * auto-vectorize a strided/gather access pattern, but it's far less
 * efficient than the packed SoA loads above -- which is exactly the
 * layout cost this benchmark is meant to expose. */
static void compute_aos(particle_aos_t *p, size_t n, dtype g, dtype mass,
                        dtype eps, rsqrt_mode_t rsqrt_mode)
{
  /* AoS force kernel: source coordinates are interleaved with the other particle
   * fields, making the memory stream less compact for the same arithmetic. */
  const dtype eps2 = eps * eps;
  const dtype gm = g * mass;
  size_t i;

#pragma omp parallel for schedule(static)
  for (i = 0u; i < n; ++i)
  {
    const dtype xi = p[i].x, yi = p[i].y, zi = p[i].z;
    dtype ax = (dtype)0.0, ay = (dtype)0.0, az = (dtype)0.0;
    size_t j;
    for (j = 0u; j < n; ++j)
    {
      const dtype dx = p[j].x - xi;
      const dtype dy = p[j].y - yi;
      const dtype dz = p[j].z - zi;
      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = invsqrt_force(r2, rsqrt_mode);
      const dtype s = gm * invr * invr * invr;
      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }
    p[i].ax = ax;
    p[i].ay = ay;
    p[i].az = az;
  }
}

/* Dispatch one force-kernel evaluation for the selected concrete layout.
 * This avoids duplicating the same layout switch in both the warmup loop and
 * the measured loop. LAYOUT_BOTH is rejected here because this helper executes
 * exactly one layout at a time. */
static void compute_layout_once(layout_t layout, particles_soa_t *soa,
                                particle_aos_t *aos, dtype g, dtype mass,
                                dtype eps, rsqrt_mode_t rsqrt_mode)
{
  switch (layout)
  {
  case LAYOUT_SOA:
    compute_soa(soa, g, mass, eps, rsqrt_mode);
    break;
  case LAYOUT_AOS:
    compute_aos(aos, soa->n, g, mass, eps, rsqrt_mode);
    break;
  case LAYOUT_BOTH:
    die("internal error: LAYOUT_BOTH cannot be timed as one layout");
  }
}

/* Sums every acceleration component across all particles into one double.
 * This is a cheap correctness "fingerprint": if compute_soa() has a bug (or
 * a different rsqrt mode changes the numerics), the checksum will visibly
 * differ, without needing to dump/compare full per-particle arrays. The
 * reduction is done in double regardless of `dtype` to reduce accumulation
 * error in the checksum itself. */
static double checksum_soa(const particles_soa_t *p)
{
  /* Lightweight correctness fingerprint for the SoA acceleration field. */
  double sum = 0.0;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (i = 0u; i < p->n; ++i)
    sum += (double)p->ax[i] + (double)p->ay[i] + (double)p->az[i];
  return sum;
}

/* AoS counterpart of checksum_soa(); same idea, different source layout. */
static double checksum_aos(const particle_aos_t *p, size_t n)
{
  /* Lightweight correctness fingerprint for the AoS acceleration field. */
  double sum = 0.0;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (i = 0u; i < n; ++i)
    sum += (double)p[i].ax + (double)p[i].ay + (double)p[i].az;
  return sum;
}

/* Drives a single layout's benchmark: run `warmups` untimed iterations
 * first (to warm caches, let OpenMP thread pools spin up, and avoid
 * counting one-time costs like first-touch page faults), then time
 * `repeats` iterations back-to-back with omp_get_wtime(), then print one
 * CSV row summarizing the result.
 *
 * Gpairs/s (billions of particle-pair interactions per second) is the
 * standard throughput metric for all-pairs N-body kernels: this
 * implementation computes N*(N-1) ordered pairs per force evaluation (it
 * does NOT exploit Newton's third law / i<->j symmetry to halve the work),
 * across `repeats` evaluations, divided by elapsed seconds and by 1e9. */
static void run_one_layout(layout_t layout, particles_soa_t *soa,
                           particle_aos_t *aos, dtype g, dtype mass,
                           dtype eps, size_t warmups, size_t repeats,
                           rsqrt_mode_t rsqrt_mode, bool header)
{
  /* Time a selected layout after warmup iterations.  The reported force time is
   * the total time for `repeats` force evaluations, and Gpairs/s normalizes it
   * by the number of pair interactions. */
  double t0, elapsed, checksum;
  size_t r;

  for (r = 0u; r < warmups; ++r)
    compute_layout_once(layout, soa, aos, g, mass, eps, rsqrt_mode);

  t0 = omp_get_wtime();
  for (r = 0u; r < repeats; ++r)
    compute_layout_once(layout, soa, aos, g, mass, eps, rsqrt_mode);
  elapsed = omp_get_wtime() - t0;
  checksum = layout == LAYOUT_SOA ? checksum_soa(soa) : checksum_aos(aos, soa->n);

  if (header)
    puts("layout,N,threads,warmups,inner_repeats,rsqrt,force,gpairs,checksum");
  printf("%s,%zu,%d,%zu,%zu,%s,%.9g,%.9g,%.17g\n",
         layout_name(layout), soa->n, omp_get_max_threads(), warmups, repeats,
         rsqrt_mode_name(rsqrt_mode), elapsed,
         ((double)repeats * (double)soa->n * (double)(soa->n - 1u)) /
           (elapsed * 1.0e9),
         checksum);
}

/* Prints the --help text to stderr. */
static void print_usage(const char *program)
{
  fprintf(stderr,
          "usage: %s --input FILE [options]\n"
          "  --layout soa|aos|both     layout to benchmark (default: both)\n"
          "  --warmups N               unrecorded force evaluations (default: 1)\n"
          "  --inner-repeats N         recorded force evaluations (default: 3)\n"
          "  --rsqrt exact|approx      inverse sqrt path (default: exact)\n"
          "  --eps X                   softening length (default: 0.01)\n"
          "  --G X                     gravitational constant (default: 1)\n"
          "  --mass X                  equal particle mass (default: 1)\n"
          "  --header                  print CSV header\n",
          program);
}

int main(int argc, char **argv)
{
  /* Parse benchmark parameters, build both memory layouts, run the requested
   * layout comparison, then release all temporary arrays. */
  const char *input_path = NULL;
  layout_t layout = LAYOUT_BOTH;
  rsqrt_mode_t rsqrt_mode = RSQRT_EXACT;
  size_t warmups = 1u, repeats = 3u;
  dtype eps = (dtype)1.0e-2, g = (dtype)1.0, mass = (dtype)1.0;
  bool header = false;
  particles_soa_t soa;
  particle_aos_t *aos = NULL;

  /* Minimal hand-rolled argv parser: iterate every token, try each
   * recognized flag via option_value() (which itself handles the
   * "--flag value" vs "--flag=value" forms and advances `argi` when it
   * consumes a following token), and bail out with die() on anything
   * unrecognized. */
  for (int argi = 1; argi < argc; ++argi)
  {
    const char *value;
    if ((value = option_value(&argi, argc, argv, "--input")) != NULL)
      input_path = value;
    else if ((value = option_value(&argi, argc, argv, "--layout")) != NULL)
      layout = parse_layout(value);
    else if ((value = option_value(&argi, argc, argv, "--warmups")) != NULL)
      warmups = parse_size(value, "--warmups");
    else if ((value = option_value(&argi, argc, argv, "--inner-repeats")) != NULL)
      repeats = parse_size(value, "--inner-repeats");
    else if ((value = option_value(&argi, argc, argv, "--rsqrt")) != NULL)
      rsqrt_mode = parse_rsqrt_mode(value);
    else if ((value = option_value(&argi, argc, argv, "--eps")) != NULL)
      eps = parse_dtype(value, "--eps");
    else if ((value = option_value(&argi, argc, argv, "--G")) != NULL)
      g = parse_dtype(value, "--G");
    else if ((value = option_value(&argi, argc, argv, "--mass")) != NULL)
      mass = parse_dtype(value, "--mass");
    else if (strcmp(argv[argi], "--header") == 0)
      header = true;
    else if (strcmp(argv[argi], "--help") == 0)
    {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    }
    else
      die("unknown option: %s", argv[argi]);
  }

  if (input_path == NULL)
    die("missing required --input FILE");
  if ((eps < (dtype)0.0) || (g <= (dtype)0.0) || (mass <= (dtype)0.0) ||
      (repeats == 0u))
    die("invalid non-positive benchmark parameter");

  read_particles(input_path, &soa, &aos);
  if (layout == LAYOUT_BOTH)
  {
    /* Run SoA first (prints the CSV header if requested), then AoS with
     * header=false so the two rows land under a single header line and can
     * be concatenated/parsed as one CSV table. */
    run_one_layout(LAYOUT_SOA, &soa, aos, g, mass, eps, warmups, repeats,
                   rsqrt_mode, header);
    run_one_layout(LAYOUT_AOS, &soa, aos, g, mass, eps, warmups, repeats,
                   rsqrt_mode, false);
  }
  else
    run_one_layout(layout, &soa, aos, g, mass, eps, warmups, repeats,
                   rsqrt_mode, header);

  particles_soa_free(&soa);
  free(aos);
  return EXIT_SUCCESS;
}
