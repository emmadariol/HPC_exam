#include "nbody_common.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

typedef struct particle_aos_s
{
  dtype x, y, z;
  dtype vx, vy, vz;
  dtype ax, ay, az;
} particle_aos_t;

typedef struct particles_soa_s
{
  size_t n;
  dtype mass;
  dtype *x, *y, *z;
  dtype *vx, *vy, *vz;
  dtype *ax, *ay, *az;
} particles_soa_t;

typedef enum layout_e
{
  LAYOUT_SOA,
  LAYOUT_AOS,
  LAYOUT_BOTH
} layout_t;

typedef enum rsqrt_mode_e
{
  RSQRT_EXACT,
  RSQRT_APPROX
} rsqrt_mode_t;

static void die(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fputc('\n', stderr);
  exit(EXIT_FAILURE);
}

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

static layout_t parse_layout(const char *text)
{
  if (strcmp(text, "soa") == 0)
    return LAYOUT_SOA;
  if (strcmp(text, "aos") == 0)
    return LAYOUT_AOS;
  if (strcmp(text, "both") == 0)
    return LAYOUT_BOTH;
  die("invalid --layout '%s' (expected soa, aos, or both)", text);
  return LAYOUT_BOTH;
}

static const char *layout_name(layout_t layout)
{
  if (layout == LAYOUT_SOA)
    return "soa";
  if (layout == LAYOUT_AOS)
    return "aos";
  return "both";
}

static rsqrt_mode_t parse_rsqrt_mode(const char *text)
{
  if (strcmp(text, "exact") == 0)
    return RSQRT_EXACT;
  if (strcmp(text, "approx") == 0)
    return RSQRT_APPROX;
  die("invalid --rsqrt '%s' (expected exact or approx)", text);
  return RSQRT_EXACT;
}

static const char *rsqrt_mode_name(rsqrt_mode_t mode)
{
  return mode == RSQRT_APPROX ? "approx" : "exact";
}

static inline dtype invsqrt_force(dtype r2, rsqrt_mode_t mode)
{
  if (mode == RSQRT_EXACT)
    return (dtype)1.0 / dtype_sqrt(r2);
  else
  {
    dtype x = (dtype)(1.0f / sqrtf((float)r2));
    const dtype half = (dtype)0.5;
    const dtype three_halves = (dtype)1.5;
    x = x * (three_halves - half * r2 * x * x);
    x = x * (three_halves - half * r2 * x * x);
    return x;
  }
}

static void *checked_aligned_alloc(size_t nbytes)
{
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

static void checked_fread(void *ptr, size_t size, size_t nmemb, FILE *fp,
                          const char *path, const char *what)
{
  if (fread(ptr, size, nmemb, fp) != nmemb)
    die("failed reading %s from '%s'", what, path);
}

static void particles_soa_init(particles_soa_t *p)
{
  memset(p, 0, sizeof(*p));
  p->mass = (dtype)1.0;
}

static void particles_soa_allocate(particles_soa_t *p, size_t n, dtype mass)
{
  const size_t bytes = n * sizeof(dtype);
  particles_soa_init(p);
  p->n = n;
  p->mass = mass;
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

static particle_aos_t *particles_aos_allocate(size_t n)
{
  particle_aos_t *p = checked_aligned_alloc(n * sizeof(*p));
  memset(p, 0, n * sizeof(*p));
  return p;
}

static void read_particles(const char *path, dtype mass, particles_soa_t *soa,
                           particle_aos_t **aos_out)
{
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
    die("input particle count is too large");

  n = (size_t)n64;
  particles_soa_allocate(soa, n, mass);
  aos = particles_aos_allocate(n);

  for (i = 0u; i < n; ++i)
  {
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

static void compute_soa(particles_soa_t *p, dtype g, dtype eps,
                        rsqrt_mode_t rsqrt_mode)
{
  const dtype eps2 = eps * eps;
  const dtype gm = g * p->mass;
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
      const dtype dx = p->x[j] - xi;
      const dtype dy = p->y[j] - yi;
      const dtype dz = p->z[j] - zi;
      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = invsqrt_force(r2, rsqrt_mode);
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

static void compute_aos(particle_aos_t *p, size_t n, dtype g, dtype mass,
                        dtype eps, rsqrt_mode_t rsqrt_mode)
{
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

static double checksum_soa(const particles_soa_t *p)
{
  double sum = 0.0;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (i = 0u; i < p->n; ++i)
    sum += (double)p->ax[i] + (double)p->ay[i] + (double)p->az[i];
  return sum;
}

static double checksum_aos(const particle_aos_t *p, size_t n)
{
  double sum = 0.0;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (i = 0u; i < n; ++i)
    sum += (double)p[i].ax + (double)p[i].ay + (double)p[i].az;
  return sum;
}

static void run_one_layout(layout_t layout, particles_soa_t *soa,
                           particle_aos_t *aos, dtype g, dtype mass,
                           dtype eps, size_t warmups, size_t repeats,
                           rsqrt_mode_t rsqrt_mode, bool header)
{
  double t0, elapsed, checksum;
  size_t r;

  for (r = 0u; r < warmups; ++r)
  {
    if (layout == LAYOUT_SOA)
      compute_soa(soa, g, eps, rsqrt_mode);
    else
      compute_aos(aos, soa->n, g, mass, eps, rsqrt_mode);
  }

  t0 = omp_get_wtime();
  for (r = 0u; r < repeats; ++r)
  {
    if (layout == LAYOUT_SOA)
      compute_soa(soa, g, eps, rsqrt_mode);
    else
      compute_aos(aos, soa->n, g, mass, eps, rsqrt_mode);
  }
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
  const char *input_path = NULL;
  layout_t layout = LAYOUT_BOTH;
  rsqrt_mode_t rsqrt_mode = RSQRT_EXACT;
  size_t warmups = 1u, repeats = 3u;
  dtype eps = (dtype)1.0e-2, g = (dtype)1.0, mass = (dtype)1.0;
  bool header = false;
  particles_soa_t soa;
  particle_aos_t *aos = NULL;

  particles_soa_init(&soa);
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

  read_particles(input_path, mass, &soa, &aos);
  if (layout == LAYOUT_BOTH)
  {
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
