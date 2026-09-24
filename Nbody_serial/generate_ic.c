// Initial conditions: Plummer sphere or uniform ball, written in the binary particle format.

#include "nbody_common.h"

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NBODY_PI
#define NBODY_PI 3.141592653589793238462643383279502884
#endif

typedef struct rng_s  // random number generator state
{
  uint64_t state;
  bool has_spare;  // Box-Muller gives two normals: keep the second one
  dtype spare;
} rng_t;

static void die(const char *format, ...)  // print an error and exit
{
  va_list args;

  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
  exit(EXIT_FAILURE);
}

static void print_usage(const char *program
)
{
  fprintf(stderr,
          "usage: %s --n N --output FILE [options]\n"
          "\n"
          "options:\n"
          "  --model N             0=Plummer sphere, 1=uniform ball + Maxwellian (default: 0)\n"
          "  --n N                 number of particles\n"
          "  --output FILE         output binary particle file (%s)\n"
          "  --seed N              RNG seed (default: 1)\n"
          "  --scale X             Plummer scale radius a (default: 1)\n"
          "  --rmax X              optional truncation radius; <=0 disables (default: 0)\n"
          "  --radius X            uniform-ball radius for --model 1 (default: 1)\n"
          "  --sigma X             1D velocity dispersion for --model 1; negative=auto (default: -1)\n"
          "  --G X                 gravitational constant used for velocities (default: 1)\n"
          "  --mass X              particle mass used for velocities (default: 1)\n"
          "  --help                show this help message\n",
          program, NBODY_BINARY_VERSION_TEXT);
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

static dtype *allocate_array(size_t n,  // aligned array of n dtype values
                             const char *name
)
{
  const size_t bytes = n * sizeof(dtype);
  size_t padded;
  dtype *ptr;

  if (n == 0u)
    die("cannot allocate zero-length array %s", name);
  if (n > SIZE_MAX / sizeof(dtype))
    die("array %s is too large", name);

  padded = ((bytes + NBODY_ALIGNMENT - 1u) / NBODY_ALIGNMENT) * NBODY_ALIGNMENT;
  ptr = aligned_alloc(NBODY_ALIGNMENT, padded);
  if (ptr == NULL)
    die("allocation failed for array %s", name);

  return ptr;
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

static uint64_t rng_next_u64(rng_t *rng  // SplitMix64: simple, fast 64-bit generator
)
{
  uint64_t z;

  rng->state += UINT64_C(0x9e3779b97f4a7c15);
  z = rng->state;
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static double rng_uniform_open(rng_t *rng  // uniform number in (0, 1), never exactly 0 or 1
)
{
  const uint64_t bits = rng_next_u64(rng) >> 11;  // 53 random bits

  return ((double)bits + 0.5) * (1.0 / 9007199254740992.0);
}

static dtype rng_normal(rng_t *rng  // standard normal number (Box-Muller)
)
{
  dtype u1;
  dtype u2;
  dtype radius;
  dtype angle;

  if (rng->has_spare)  // use the spare value from the previous call
  {
    rng->has_spare = false;
    return rng->spare;
  }

  u1 = (dtype)rng_uniform_open(rng);
  u2 = (dtype)rng_uniform_open(rng);
  radius = dtype_sqrt((dtype)-2.0 * dtype_log(u1));  // Box-Muller radius
  angle = (dtype)(2.0 * NBODY_PI) * u2;

  rng->spare = radius * dtype_sin(angle);  // keep the second value for the next call
  rng->has_spare = true;
  return radius * dtype_cos(angle);
}

static void random_unit_vector(rng_t *rng,  // random direction, uniform on the sphere
                               dtype *ux,
                               dtype *uy,
                               dtype *uz
)
{
  const dtype cos_theta = (dtype)(2.0 * rng_uniform_open(rng) - 1.0);  // cos(theta) uniform in [-1, 1]
  const dtype phi = (dtype)(2.0 * NBODY_PI * rng_uniform_open(rng));
  const dtype sin_theta = dtype_sqrt(dtype_fmax((dtype)0.0,
                                                (dtype)1.0 - cos_theta * cos_theta));

  *ux = sin_theta * dtype_cos(phi);
  *uy = sin_theta * dtype_sin(phi);
  *uz = cos_theta;
}

static void generate_ball_maxwell(size_t n,  // uniform ball with Maxwellian (Gaussian) velocities
                                  dtype ball_radius,
                                  dtype sigma,
                                  rng_t *rng,
                                  dtype *restrict x,
                                  dtype *restrict y,
                                  dtype *restrict z,
                                  dtype *restrict vx,
                                  dtype *restrict vy,
                                  dtype *restrict vz
)
{
  long double xcm = 0.0L;
  long double ycm = 0.0L;
  long double zcm = 0.0L;
  long double vxcm = 0.0L;
  long double vycm = 0.0L;
  long double vzcm = 0.0L;

  for (size_t i = 0u; i < n; ++i)
  {
    dtype ux;
    dtype uy;
    dtype uz;
    dtype radius;

    radius = ball_radius * dtype_pow((dtype)rng_uniform_open(rng), (dtype)(1.0 / 3.0));  // r = R * u^(1/3) gives uniform density
    random_unit_vector(rng, &ux, &uy, &uz);
    x[i] = radius * ux;
    y[i] = radius * uy;
    z[i] = radius * uz;

    vx[i] = sigma * rng_normal(rng);  // Gaussian velocity components
    vy[i] = sigma * rng_normal(rng);
    vz[i] = sigma * rng_normal(rng);

    xcm += (long double)x[i];  // accumulate the centre of mass
    ycm += (long double)y[i];
    zcm += (long double)z[i];
    vxcm += (long double)vx[i];
    vycm += (long double)vy[i];
    vzcm += (long double)vz[i];
  }

  xcm /= (long double)n;  // centre-of-mass position and velocity
  ycm /= (long double)n;
  zcm /= (long double)n;
  vxcm /= (long double)n;
  vycm /= (long double)n;
  vzcm /= (long double)n;

  for (size_t i = 0u; i < n; ++i)  // move to the centre-of-mass frame
  {
    x[i] -= (dtype)xcm;
    y[i] -= (dtype)ycm;
    z[i] -= (dtype)zcm;
    vx[i] -= (dtype)vxcm;
    vy[i] -= (dtype)vycm;
    vz[i] -= (dtype)vzcm;
  }
}

static dtype sample_plummer_radius(rng_t *rng,  // Plummer radius from the inverse of the cumulative mass
                                   dtype scale,
                                   dtype rmax
)
{
  dtype radius;

  do
  {
    const dtype u = (dtype)rng_uniform_open(rng);

    radius = scale / dtype_sqrt(dtype_pow(u, (dtype)(-2.0 / 3.0)) - (dtype)1.0);  // r = a / sqrt(u^(-2/3) - 1)
  } while ((rmax > (dtype)0.0) && (radius > rmax));  // resample if beyond rmax

  return radius;
}

static dtype sample_plummer_q(rng_t *rng  // speed fraction q = v / v_escape, by rejection sampling
)
{
  for (;;)
  {
    const dtype q = (dtype)rng_uniform_open(rng);
    const dtype y = (dtype)(0.1 * rng_uniform_open(rng));
    const dtype density = q * q * dtype_pow((dtype)1.0 - q * q, (dtype)3.5);  // Plummer distribution of q: q^2 (1 - q^2)^(7/2)

    if (y <= density)
      return q;
  }
}

static void generate_plummer(size_t n,  // Plummer sphere in equilibrium
                             dtype scale,
                             dtype rmax,
                             dtype g,
                             dtype particle_mass,
                             rng_t *rng,
                             dtype *restrict x,
                             dtype *restrict y,
                             dtype *restrict z,
                             dtype *restrict vx,
                             dtype *restrict vy,
                             dtype *restrict vz
)
{
  const dtype total_mass = (dtype)n * particle_mass;
  long double xcm = 0.0L;
  long double ycm = 0.0L;
  long double zcm = 0.0L;
  long double vxcm = 0.0L;
  long double vycm = 0.0L;
  long double vzcm = 0.0L;

  if (!dtype_isfinite(total_mass) || !(total_mass > (dtype)0.0))
    die("total mass is not finite in the selected dtype");

  for (size_t i = 0u; i < n; ++i)
  {
    dtype ux;
    dtype uy;
    dtype uz;
    dtype radius;
    dtype q;
    dtype psi;
    dtype speed;

    radius = sample_plummer_radius(rng, scale, rmax);  // radius from the Plummer mass profile
    random_unit_vector(rng, &ux, &uy, &uz);  // random direction
    x[i] = radius * ux;
    y[i] = radius * uy;
    z[i] = radius * uz;

    q = sample_plummer_q(rng);
    psi = g * total_mass / dtype_sqrt(radius * radius + scale * scale);  // potential at radius r
    speed = q * dtype_sqrt((dtype)2.0 * psi);  // speed = q * escape speed
    random_unit_vector(rng, &ux, &uy, &uz);  // random velocity direction
    vx[i] = speed * ux;
    vy[i] = speed * uy;
    vz[i] = speed * uz;

    xcm += (long double)x[i];  // accumulate the centre of mass
    ycm += (long double)y[i];
    zcm += (long double)z[i];
    vxcm += (long double)vx[i];
    vycm += (long double)vy[i];
    vzcm += (long double)vz[i];
  }

  xcm /= (long double)n;  // centre-of-mass position and velocity
  ycm /= (long double)n;
  zcm /= (long double)n;
  vxcm /= (long double)n;
  vycm /= (long double)n;
  vzcm /= (long double)n;

  for (size_t i = 0u; i < n; ++i)  // move to the centre-of-mass frame
  {
    x[i] -= (dtype)xcm;
    y[i] -= (dtype)ycm;
    z[i] -= (dtype)zcm;
    vx[i] -= (dtype)vxcm;
    vy[i] -= (dtype)vycm;
    vz[i] -= (dtype)vzcm;
  }
}

static int verify_sanity(const size_t n,  // check every value can be written as a finite float
                         const dtype *x,
                         const dtype *y,
                         const dtype *z,
                         const dtype *vx,
                         const dtype *vy,
                         const dtype *vz
)
{
  size_t failures = 0u;

  for (size_t i = 0u; i < n; ++i)
    failures += (!isfinite(x[i]) || (fabs(x[i]) > DTYPE_MAX_VALUE));
  if (failures)
  {
    printf("%zu x component cannot be stored as a finite float", failures);
    return 1;
  }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
    failures += (!isfinite(y[i]) || (fabs(y[i]) > DTYPE_MAX_VALUE));
  if (failures)
  {
    printf("%zu y component cannot be stored as a finite float", failures);
    return 1;
  }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
    failures += (!isfinite(z[i]) || (fabs(z[i]) > DTYPE_MAX_VALUE));
  if (failures)
  {
    printf("%zu z component cannot be stored as a finite float", failures);
    return 1;
  }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
    failures += (!isfinite(vx[i]) || (fabs(vx[i]) > DTYPE_MAX_VALUE));
  if (failures)
  {
    printf("%zu vx component cannot be stored as a finite float", failures);
    return 1;
  }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
    failures += (!isfinite(vy[i]) || (fabs(vy[i]) > DTYPE_MAX_VALUE));
  if (failures)
  {
    printf("%zu vy component cannot be stored as a finite float", failures);
    return 1;
  }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
    failures += (!isfinite(vz[i]) || (fabs(vz[i]) > DTYPE_MAX_VALUE));
  if (failures)
  {
    printf("%zu vz component cannot be stored as a finite float", failures);
    return 1;
  }

  return 0;
}

static void write_particles_binary(const char *path,  // write the binary file: header + one float record per particle
                                   size_t n,
                                   const dtype *x,
                                   const dtype *y,
                                   const dtype *z,
                                   const dtype *vx,
                                   const dtype *vy,
                                   const dtype *vz
)
{
  FILE *fp;
  uint64_t n64 = (uint64_t)n;

  if ((size_t)n64 != n)
    die("particle count cannot be represented in the binary header");

  int uncorrect_data = verify_sanity(n, x, y, z, vx, vy, vz);

  if (!uncorrect_data)
  {
    fp = fopen(path, "wb");
    if (fp == NULL)
      die("cannot open output file '%s'", path);

    checked_fwrite(nbody_binary_magic, sizeof nbody_binary_magic[0],  // header: magic string and particle count
                   NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
    checked_fwrite(&n64, sizeof n64, 1u, fp, path, "particle count");

    for (size_t i = 0u; i < n; ++i)  // six floats per particle
    {
      float record[NBODY_BINARY_COMPONENTS];

      record[0] = x[i];
      record[1] = y[i];
      record[2] = z[i];
      record[3] = vx[i];
      record[4] = vy[i];
      record[5] = vz[i];

      checked_fwrite(record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                     fp, path, "particle record");
    }

    if (fclose(fp) != 0)
      die("error while closing output file '%s'", path);
  }
  else
    printf("Some data are not representable as finite floating points\n");
}

int main(int argc, char **argv)
{
  const char *output_path = NULL;
  int model = PLUMMER_SPHERE;
  size_t n = 0u;
  uint64_t seed = 1u;
  dtype ball_radius = (dtype)1.0;
  dtype sigma = (dtype)-1.0;
  dtype scale = (dtype)1.0;
  dtype rmax = (dtype)0.0;
  dtype g = (dtype)1.0;
  dtype particle_mass = (dtype)1.0;
  bool sigma_auto;
  dtype *x;
  dtype *y;
  dtype *z;
  dtype *vx;
  dtype *vy;
  dtype *vz;
  rng_t rng;
  int argi;

  for (argi = 1; argi < argc; ++argi)
  {
    const char *value;

    if ((value = option_value(&argi, argc, argv, "--model")) != NULL)
      model = parse_size(value, "--model");
    else if ((value = option_value(&argi, argc, argv, "--n")) != NULL)
      n = parse_size(value, "--n");
    else if ((value = option_value(&argi, argc, argv, "--output")) != NULL)
      output_path = value;
    else if ((value = option_value(&argi, argc, argv, "--seed")) != NULL)
      seed = (uint64_t)parse_size(value, "--seed");
    else if ((value = option_value(&argi, argc, argv, "--scale")) != NULL)
      scale = parse_dtype(value, "--scale");
    else if ((value = option_value(&argi, argc, argv, "--rmax")) != NULL)
      rmax = parse_dtype(value, "--rmax");
    else if ((value = option_value(&argi, argc, argv, "--radius")) != NULL)
      ball_radius = parse_dtype(value, "--radius");
    else if ((value = option_value(&argi, argc, argv, "--sigma")) != NULL)
      sigma = parse_dtype(value, "--sigma");
    else if ((value = option_value(&argi, argc, argv, "--G")) != NULL)
      g = parse_dtype(value, "--G");
    else if ((value = option_value(&argi, argc, argv, "--mass")) != NULL)
      particle_mass = parse_dtype(value, "--mass");
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

  if (n == 0u)
  {
    print_usage(argv[0]);
    die("missing or invalid --n N");
  }
  if (output_path == NULL)
  {
    print_usage(argv[0]);
    die("missing required --output FILE");
  }
  if (!(scale > (dtype)0.0))
    die("--scale must be positive");
  if ((model != PLUMMER_SPHERE) && (model != MAXWELL_BALL))
    die("--model must be 0 (Plummer sphere) or 1 (uniform ball + Maxwellian)");
  if (!(g > (dtype)0.0))
    die("--G must be positive");
  if (!(particle_mass > (dtype)0.0))
    die("--mass must be positive");
  if (!(ball_radius > (dtype)0.0))
    die("--radius must be positive");
  sigma_auto = (sigma < (dtype)0.0);  // negative sigma means: choose it automatically
  if (sigma_auto)
  {
    const dtype total_mass = (dtype)n * particle_mass;

    if (!dtype_isfinite(total_mass) || !(total_mass > (dtype)0.0))
      die("total mass is not finite in the selected dtype");
    sigma = dtype_sqrt(g * total_mass / ((dtype)5.0 * ball_radius));  // sigma from the virial estimate for a uniform ball
  }
  if (!(sigma >= (dtype)0.0) || !dtype_isfinite(sigma))
    die("--sigma must be non-negative and finite, or negative to request the auto value");

  x = allocate_array(n, "x");
  y = allocate_array(n, "y");
  z = allocate_array(n, "z");
  vx = allocate_array(n, "vx");
  vy = allocate_array(n, "vy");
  vz = allocate_array(n, "vz");

  rng.state = seed;  // the seed fixes the whole sequence
  rng.has_spare = false;
  rng.spare = (dtype)0.0;

  if (model == PLUMMER_SPHERE)
    generate_plummer(n, scale, rmax, g, particle_mass, &rng, x, y, z, vx, vy, vz);

  else
    generate_ball_maxwell(n, ball_radius, sigma, &rng, x, y, z, vx, vy, vz);

  write_particles_binary(output_path, n, x, y, z, vx, vy, vz);

  free(x);
  free(y);
  free(z);
  free(vx);
  free(vy);
  free(vz);

  return EXIT_SUCCESS;
}
