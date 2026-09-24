// Serial reference solver (DKD leapfrog); used for the vectorisation report.

#include "nbody_common.h"

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct particles_s  // particles in SoA layout
{
  size_t n;
  dtype mass;
  dtype *x;
  dtype *y;
  dtype *z;
  dtype *vx;
  dtype *vy;
  dtype *vz;
  dtype *ax;
  dtype *ay;
  dtype *az;
} particles_t;

typedef enum io_check_mode_e  // checked: validate every value; fast: plain casts
{
  IO_CHECKED,
  IO_FAST
} io_check_mode_t;

typedef struct io_profile_s  // time spent in file I/O
{
  double read_seconds;
  double write_seconds;
  double conversion_seconds;
} io_profile_t;

static void die(const char *format, ...)  // print an error and exit
{
  va_list args;

  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
  exit(EXIT_FAILURE);
}

static double wall_seconds(void)  // wall-clock time in seconds
{
  struct timespec ts;

  if (timespec_get(&ts, TIME_UTC) != TIME_UTC)
    die("timespec_get failed");
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static size_t parse_size(const char *text,  // string -> size_t, stops on invalid input
                         const char *name
)
{
  char *endptr;
  unsigned long long value;

  errno = 0;
  value = strtoull(text, &endptr, 10);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0'))
    die("invalid integer for %s: %s", name, text);
  if (value > (unsigned long long)SIZE_MAX)
    die("integer for %s is too large: %s", name, text);

  return (size_t)value;
}

static dtype parse_dtype(const char *text,  // string -> dtype, stops on invalid input
                         const char *name
)
{
  char *endptr;
  double value;

  errno = 0;
  value = strtod(text, &endptr);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0') || !isfinite(value))
    die("invalid floating-point value for %s: %s", name, text);
  if (fabs(value) > (double)DTYPE_MAX_VALUE)
    die("floating-point value for %s is outside the selected dtype range: %s", name, text);

  return (dtype)value;
}

static const char *option_value(int *i,  // accepts both --key value and --key=value
                                int argc,
                                char **argv,
                                const char *key
)
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

static void *checked_aligned_alloc(size_t nbytes,  // aligned allocation, size rounded up to the alignment
                                   size_t alignment
)
{
  void *ptr;
  size_t padded;

  if (nbytes == 0u)
    die("attempted zero-byte allocation");
  if (alignment == 0u)
    die("invalid zero alignment");
  if (nbytes > SIZE_MAX - alignment)
    die("allocation size overflow");

  padded = ((nbytes + alignment - 1u) / alignment) * alignment;
  ptr = aligned_alloc(alignment, padded);
  if (ptr == NULL)
    die("aligned_alloc failed for %zu bytes", padded);

  return ptr;
}

static void checked_fread(void *ptr,  // fread that stops the program on error
                          size_t size,
                          size_t nmemb,
                          FILE *fp,
                          const char *path,
                          const char *what
)
{
  const size_t got = fread(ptr, size, nmemb, fp);

  if (got != nmemb)
  {
    if (ferror(fp))
      die("read error while reading %s from '%s'", what, path);
    die("short file while reading %s from '%s'", what, path);
  }
}

static void checked_fwrite(const void *ptr,  // fwrite that stops the program on error
                           size_t size,
                           size_t nmemb,
                           FILE *fp,
                           const char *path,
                           const char *what
)
{
  const size_t written = fwrite(ptr, size, nmemb, fp);

  if (written != nmemb)
    die("write error while writing %s to '%s'", what, path);
}

static void particles_init_empty(particles_t *p
)
{
  p->n = 0u;
  p->mass = (dtype)1.0;
  p->x = NULL;
  p->y = NULL;
  p->z = NULL;
  p->vx = NULL;
  p->vy = NULL;
  p->vz = NULL;
  p->ax = NULL;
  p->ay = NULL;
  p->az = NULL;
}

static void particles_allocate(particles_t *p,  // allocate the nine SoA arrays
                               size_t n,
                               dtype mass
)
{
  const size_t bytes = n * sizeof(dtype);

  if (n == 0u)
    die("the number of particles must be positive");
  if (n > SIZE_MAX / sizeof(dtype))
    die("particle count is too large");
  if (!(mass > (dtype)0.0) || !dtype_isfinite(mass))
    die("particle mass must be positive and finite");

  particles_init_empty(p);
  p->n = n;
  p->mass = mass;
  p->x = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->y = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->z = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->vx = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->vy = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->vz = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->ax = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->ay = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
  p->az = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
}

static void particles_free(particles_t *p
)
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
  particles_init_empty(p);
}

static float dtype_to_storage_float(dtype value,  // float conversion with overflow check
                                    const char *component,
                                    size_t i
)
{
  const double as_double = (double)value;

  if (!isfinite(as_double) || (fabs(as_double) > (double)FLT_MAX))
    die("particle %zu component %s cannot be stored as a finite float", i, component);

  return (float)value;
}

static void particles_read_binary(const char *path,  // read the whole binary file
                                  dtype mass,
                                  io_check_mode_t checks,
                                  particles_t *p,
                                  io_profile_t *profile
)
{
  FILE *fp;
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t n;
  size_t i;
  float *records;
  size_t record_values;
  double started;

  fp = fopen(path, "rb");
  if (fp == NULL)
    die("cannot open input file '%s'", path);

  checked_fread(magic, sizeof magic[0], NBODY_BINARY_MAGIC_SIZE,  // header: 8-byte magic string
                fp, path, "binary magic");
  if (memcmp(magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die("input file '%s' is not an %s file", path, NBODY_BINARY_VERSION_TEXT);

  checked_fread(&n64, sizeof n64, 1u, fp, path, "particle count");  // header: particle count
  if ((n64 == 0u) || (n64 > (uint64_t)SIZE_MAX))
    die("invalid particle count in '%s'", path);
  n = (size_t)n64;

  if (n > SIZE_MAX / (NBODY_BINARY_COMPONENTS * sizeof(*records)))
    die("particle record count is too large");
  record_values = n * NBODY_BINARY_COMPONENTS;

  particles_allocate(p, n, mass);
  records = checked_aligned_alloc(record_values * sizeof(*records), NBODY_ALIGNMENT);

  started = wall_seconds();
  checked_fread(records, sizeof(*records), record_values,  // all records in one read
                fp, path, "particle records");
  profile->read_seconds += wall_seconds() - started;

  started = wall_seconds();
  for (i = 0u; i < n; ++i)  // unpack float records into SoA arrays
  {
    const float *record = records + i * NBODY_BINARY_COMPONENTS;

    if ((checks == IO_CHECKED) &&
        (!isfinite((double)record[0]) || !isfinite((double)record[1]) ||
         !isfinite((double)record[2]) || !isfinite((double)record[3]) ||
         !isfinite((double)record[4]) || !isfinite((double)record[5])))
      die("non-finite particle value in '%s' at index %zu", path, i);

    p->x[i] = (dtype)record[0];
    p->y[i] = (dtype)record[1];
    p->z[i] = (dtype)record[2];
    p->vx[i] = (dtype)record[3];
    p->vy[i] = (dtype)record[4];
    p->vz[i] = (dtype)record[5];
  }
  profile->conversion_seconds += wall_seconds() - started;
  free(records);

  if (fclose(fp) != 0)
    die("error while closing input file '%s'", path);
}

static void particles_write_binary(const char *path,  // write the whole binary file
                                   const particles_t *p,
                                   io_check_mode_t checks,
                                   io_profile_t *profile
)
{
  FILE *fp;
  const size_t n = p->n;
  uint64_t n64 = (uint64_t)n;
  size_t i;
  float *records;
  size_t record_values;
  double started;

  if ((size_t)n64 != n)
    die("particle count cannot be represented in the binary header");
  if (n > SIZE_MAX / (NBODY_BINARY_COMPONENTS * sizeof(*records)))
    die("particle record count is too large");
  record_values = n * NBODY_BINARY_COMPONENTS;
  records = checked_aligned_alloc(record_values * sizeof(*records), NBODY_ALIGNMENT);

  fp = fopen(path, "wb");
  if (fp == NULL)
    die("cannot open output file '%s'", path);

  checked_fwrite(nbody_binary_magic, sizeof nbody_binary_magic[0],
                 NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
  checked_fwrite(&n64, sizeof n64, 1u, fp, path, "particle count");

  started = wall_seconds();
  for (i = 0u; i < n; ++i)  // pack SoA arrays into float records
  {
    float *record = records + i * NBODY_BINARY_COMPONENTS;

    if (checks == IO_CHECKED)
    {
      record[0] = dtype_to_storage_float(p->x[i], "x", i);
      record[1] = dtype_to_storage_float(p->y[i], "y", i);
      record[2] = dtype_to_storage_float(p->z[i], "z", i);
      record[3] = dtype_to_storage_float(p->vx[i], "vx", i);
      record[4] = dtype_to_storage_float(p->vy[i], "vy", i);
      record[5] = dtype_to_storage_float(p->vz[i], "vz", i);
    }
    else
    {
      record[0] = (float)p->x[i];
      record[1] = (float)p->y[i];
      record[2] = (float)p->z[i];
      record[3] = (float)p->vx[i];
      record[4] = (float)p->vy[i];
      record[5] = (float)p->vz[i];
    }
  }
  profile->conversion_seconds += wall_seconds() - started;

  started = wall_seconds();
  checked_fwrite(records, sizeof(*records), record_values,
                 fp, path, "particle records");
  profile->write_seconds += wall_seconds() - started;
  free(records);

  if (fclose(fp) != 0)
    die("error while closing output file '%s'", path);
}

// ===========================================================================
// FORCE KERNEL (serial) with [SIMD] hint
// ===========================================================================
static void compute_accelerations_naive(size_t n,  // direct O(N^2) force loop
                                        dtype g,
                                        dtype mass,
                                        dtype eps,
                                        const dtype *restrict x,
                                        const dtype *restrict y,
                                        const dtype *restrict z,
                                        dtype *restrict ax,
                                        dtype *restrict ay,
                                        dtype *restrict az
)
{
  const dtype eps2 = eps * eps;
  size_t i;
  size_t j;

  for (i = 0u; i < n; ++i)  // loop over targets
  {
    const dtype xi = x[i];  // target position
    const dtype yi = y[i];
    const dtype zi = z[i];
    dtype axi = (dtype)0.0;
    dtype ayi = (dtype)0.0;
    dtype azi = (dtype)0.0;

#if defined(_OPENMP)
#pragma omp simd reduction(+ : axi, ayi, azi)  // [SIMD] ask the compiler to vectorise the inner loop
#endif
    for (j = 0u; j < n; ++j)  // loop over sources
    {
      if (j != i)  // skip the self-pair
      {
        const dtype dx = x[j] - xi;  // d = r_j - r_i
        const dtype dy = y[j] - yi;
        const dtype dz = z[j] - zi;
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;  // q = |d|^2 + eps^2
        const dtype invr = (dtype)1.0 / dtype_sqrt(r2);  // 1/sqrt(q)
        const dtype s = g * mass * invr * invr * invr;  // G*m / q^(3/2)

        axi += dx * s;  // a_i += d * s
        ayi += dy * s;
        azi += dz * s;
      }
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
}

static void drift(particles_t *p,  // drift: x += dt * v
                  dtype dt
)
{
  size_t n = p->n;
  dtype *x = p->x;
  dtype *y = p->y;
  dtype *z = p->z;
  dtype *vx = p->vx;
  dtype *vy = p->vy;
  dtype *vz = p->vz;
  size_t i;

  for (i = 0u; i < n; ++i)
  {
    x[i] += dt * vx[i];
    y[i] += dt * vy[i];
    z[i] += dt * vz[i];
  }
}

static void kick(particles_t *p,  // kick: v += dt * a
                 dtype dt
)
{
  size_t n = p->n;
  dtype *vx = p->vx;
  dtype *vy = p->vy;
  dtype *vz = p->vz;
  dtype *ax = p->ax;
  dtype *ay = p->ay;
  dtype *az = p->az;
  size_t i;

  for (i = 0u; i < n; ++i)
  {
    vx[i] += dt * ax[i];
    vy[i] += dt * ay[i];
    vz[i] += dt * az[i];
  }
}

static void leapfrog_dkd_step(particles_t *p,  // one Drift-Kick-Drift step
                              dtype g,
                              dtype eps,
                              dtype dt
)
{
  drift(p, (dtype)0.5 * dt);  // half drift
  compute_accelerations_naive(p->n, g, p->mass, eps,  // forces at the half-step positions
                              p->x, p->y, p->z,
                              p->ax, p->ay, p->az);
  kick(p, dt);  // full kick
  drift(p, (dtype)0.5 * dt);  // second half drift
}

static dtype kinetic_energy(const particles_t *p  // kinetic energy, long double sum
)
{
  size_t n = p->n;
  dtype mass = p->mass;
  long double sum = 0.0L;
  size_t i;

  for (i = 0u; i < n; ++i)
  {
    const long double vx = (long double)p->vx[i];
    const long double vy = (long double)p->vy[i];
    const long double vz = (long double)p->vz[i];

    sum += vx * vx + vy * vy + vz * vz;
  }

  return (dtype)(0.5L * (long double)mass * sum);
}

static dtype potential_energy_naive(particles_t *p,  // potential energy, each pair once
                                    dtype g,
                                    dtype eps
)
{
  size_t n = p->n;
  dtype eps2 = eps * eps;
  dtype m2 = p->mass * p->mass;
  long double sum = 0.0L;
  size_t i;
  size_t j;

  for (i = 0u; i < n; ++i)
  {
    dtype xi = p->x[i];
    dtype yi = p->y[i];
    dtype zi = p->z[i];

    for (j = i + 1u; j < n; ++j)  // only j > i
    {
      dtype dx = p->x[j] - xi;
      dtype dy = p->y[j] - yi;
      dtype dz = p->z[j] - zi;
      dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
      dtype invr = (dtype)1.0 / dtype_sqrt(r2);

      sum -= (long double)g * (long double)m2 * (long double)invr;  // U_ij = -G m^2 / sqrt(q)
    }
  }

  return (dtype)sum;
}

static dtype total_energy(particles_t *p,  // E = T + U
                          dtype g,
                          dtype eps,
                          dtype *kinetic,
                          dtype *potential
)
{
  *kinetic = kinetic_energy(p);
  *potential = potential_energy_naive(p, g, eps);

  return *kinetic + *potential;
}

static void print_usage(const char *program
)
{
  fprintf(stderr,
          "usage: %s --input FILE [options]\n"
          "\n"
          "options:\n"
          "  --input FILE              input binary particle file (%s)\n"
          "  --output FILE             optional final-state binary file\n"
          "  --nsteps N                number of DKD steps (default: 10)\n"
          "  --dt X                    time step (default: 0.001)\n"
          "  --eps X                   softening length (default: 0.01)\n"
          "  --G X                     gravitational constant (default: 1)\n"
          "  --mass X                  particle mass (default: 1)\n"
          "  --energy-every N          diagnostic period in steps (default: 1)\n"
          "  --energy-tol X            warning tolerance for max relative drift (default: 1e-4)\n"
          "  --io-mode checked|fast    validate conversions or use direct casts (default: checked)\n"
          "  --io-profile              print read/write/conversion timings\n"
          "  --quiet                   only print final summary\n"
          "  --help                    show this help message\n",
          program, NBODY_BINARY_VERSION_TEXT);
}

int main(int argc, char **argv)
{
  const char *input_path = NULL;
  const char *output_path = NULL;
  size_t nsteps = 10u;
  size_t energy_every = 1u;
  dtype dt = (dtype)1.0e-3;
  dtype eps = (dtype)1.0e-2;
  dtype g = (dtype)1.0;
  dtype mass = (dtype)1.0;
  dtype energy_tol = (dtype)1.0e-4;
  bool quiet = false;
  bool io_profile_enabled = false;
  io_check_mode_t io_checks = IO_CHECKED;
  io_profile_t io_profile = {0.0, 0.0, 0.0};
  particles_t particles;
  dtype kinetic0;
  dtype potential0;
  dtype energy0;

  particles_init_empty(&particles);

  for (int argi = 1; argi < argc; ++argi)
  {
    const char *value;

    if ((value = option_value(&argi, argc, argv, "--input")) != NULL)
      input_path = value;
    else if ((value = option_value(&argi, argc, argv, "--output")) != NULL)
      output_path = value;
    else if ((value = option_value(&argi, argc, argv, "--nsteps")) != NULL)
      nsteps = parse_size(value, "--nsteps");
    else if ((value = option_value(&argi, argc, argv, "--energy-every")) != NULL)
      energy_every = parse_size(value, "--energy-every");
    else if ((value = option_value(&argi, argc, argv, "--dt")) != NULL)
      dt = parse_dtype(value, "--dt");
    else if ((value = option_value(&argi, argc, argv, "--eps")) != NULL)
      eps = parse_dtype(value, "--eps");
    else if ((value = option_value(&argi, argc, argv, "--G")) != NULL)
      g = parse_dtype(value, "--G");
    else if ((value = option_value(&argi, argc, argv, "--mass")) != NULL)
      mass = parse_dtype(value, "--mass");
    else if ((value = option_value(&argi, argc, argv, "--energy-tol")) != NULL)
      energy_tol = parse_dtype(value, "--energy-tol");
    else if ((value = option_value(&argi, argc, argv, "--io-mode")) != NULL)
    {
      if (strcmp(value, "checked") == 0)
        io_checks = IO_CHECKED;
      else if (strcmp(value, "fast") == 0)
        io_checks = IO_FAST;
      else
        die("--io-mode must be checked or fast");
    }
    else if (strcmp(argv[argi], "--io-profile") == 0)
      io_profile_enabled = true;
    else if (strcmp(argv[argi], "--quiet") == 0)
      quiet = true;
    else if (strcmp(argv[argi], "--help") == 0)
    {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    }
    else
    {
      print_usage(argv[0]);
      die("unknown option: %s", argv[argi]);
    }
  }

  if (input_path == NULL)
  {
    print_usage(argv[0]);
    die("missing required --input FILE");
  }
  if (!(dt > (dtype)0.0))
    die("--dt must be positive");
  if (!(eps >= (dtype)0.0))
    die("--eps must be non-negative");
  if (!(g > (dtype)0.0))
    die("--G must be positive");
  if (!(mass > (dtype)0.0))
    die("--mass must be positive");
  if (energy_every == 0u)
    die("--energy-every must be positive");
  if (!(energy_tol > (dtype)0.0))
    die("--energy-tol must be positive");

  particles_read_binary(input_path, mass, io_checks, &particles, &io_profile);

  energy0 = total_energy(&particles, g, eps, &kinetic0, &potential0);  // reference energy E(0)

  if (!quiet)
  {
    printf("# serial direct N-body DKD baseline\n");
    printf("# arithmetic_dtype=%s binary_storage=float32 format=%s\n",
           DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
    printf("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
           particles.n, nsteps, (double)dt, (double)eps,
           (double)g, (double)mass);
    printf("# step time kinetic potential total rel_energy_drift\n");
    printf("%zu %.17g %.17g %.17g %.17g %.17g\n",
           (size_t)0u, 0.0, (double)kinetic0, (double)potential0,
           (double)energy0, 0.0);
  }

  double max_rel_drift = 0.0;

  for (size_t step = 1u; step <= nsteps; ++step)  // time loop
  {
    leapfrog_dkd_step(&particles, g, eps, dt);

    if (((step % energy_every) == 0u) || (step == nsteps))  // energy check every energy_every steps and at the end
    {
      dtype kinetic;
      dtype potential;
      const dtype energy = total_energy(&particles, g, eps, &kinetic, &potential);
      const double denom = fmax(fabs((double)energy0), (double)DTYPE_MIN_NORMAL);  // avoid division by zero
      const double rel = fabs((double)(energy - energy0)) / denom;  // relative energy drift

      if (rel > max_rel_drift)
        max_rel_drift = rel;
      if (!quiet)
        printf("%zu %.17g %.17g %.17g %.17g %.17g\n",
               step, (double)step * (double)dt, (double)kinetic,
               (double)potential, (double)energy, rel);
    }
  }

  if (output_path != NULL)
    particles_write_binary(output_path, &particles, io_checks, &io_profile);

  if (io_profile_enabled)
    printf("# io_profile mode=%s read_seconds=%.9g write_seconds=%.9g conversion_seconds=%.9g\n",
           (io_checks == IO_CHECKED) ? "checked" : "fast",
           io_profile.read_seconds, io_profile.write_seconds,
           io_profile.conversion_seconds);

  printf("# final: N=%zu steps=%zu arithmetic_dtype=%s max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
         particles.n, nsteps, DTYPE_NAME, max_rel_drift, (double)energy_tol,
         (max_rel_drift <= (double)energy_tol) ? "OK" : "WARNING");  // OK if the drift stays below the tolerance

  if (max_rel_drift > (double)energy_tol)
    fprintf(stderr,
            "warning: relative energy drift %.6e exceeds tolerance %.6e; "
            "try smaller --dt, larger --eps, or better initial conditions\n",
            max_rel_drift, (double)energy_tol);

  particles_free(&particles);

  return EXIT_SUCCESS;
}
