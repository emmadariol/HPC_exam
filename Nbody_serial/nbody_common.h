// Shared definitions: precision switch (dtype), math wrappers, binary file format.

#ifndef NBODY_COMMON_H
#define NBODY_COMMON_H

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef NBODY_ALIGNMENT
#define NBODY_ALIGNMENT 64u  // memory alignment in bytes (one AVX-512 register)
#endif

#define NBODY_BINARY_MAGIC_SIZE 8u  // binary file header: 8-byte magic string
#define NBODY_BINARY_COMPONENTS 6u  // x y z vx vy vz stored per particle
#define NBODY_BINARY_VERSION_TEXT "nbody-f32-v1"

#define PLUMMER_SPHERE 0  // --model 0 in generate_ic
#define MAXWELL_BALL   1  // --model 1 in generate_ic

static const unsigned char  nbody_binary_magic[NBODY_BINARY_MAGIC_SIZE] =  // first 8 bytes of every particle file
  { 'N', 'B', 'O', 'D', 'Y', 'F', '1', '\0' };

#if defined (NBODY_USE_FLOAT) && defined (NBODY_USE_DOUBLE)
#error "define only one of NBODY_USE_FLOAT and NBODY_USE_DOUBLE"
#endif

#if defined (NBODY_USE_FLOAT)  // single precision build
typedef float  dtype;
#define DTYPE_NAME "float"
#define DTYPE_MAX_VALUE FLT_MAX
#define DTYPE_MIN_NORMAL FLT_MIN
#define DTYPE_PRINTF_FORMAT "%.9g"

static inline dtype dtype_sqrt (dtype x)
{
  return sqrtf (x);
}

static inline dtype dtype_pow (dtype x,
                               dtype y)
{
  return powf (x, y);
}

static inline dtype dtype_sin (dtype x)
{
  return sinf (x);
}

static inline dtype dtype_cos (dtype x)
{
  return cosf (x);
}

static inline dtype dtype_log (dtype x)
{
  return logf (x);
}

static inline dtype dtype_fabs (dtype x)
{
  return fabsf (x);
}

static inline dtype dtype_fmax (dtype x,
                                dtype y)
{
  return fmaxf (x, y);
}

#else  // double precision build (default)
typedef double dtype;
#define DTYPE_NAME "double"
#define DTYPE_MAX_VALUE DBL_MAX
#define DTYPE_MIN_NORMAL DBL_MIN
#define DTYPE_PRINTF_FORMAT "%.17g"

static inline dtype dtype_sqrt (dtype x)
{
  return sqrt (x);
}

static inline dtype dtype_pow (dtype x,
                               dtype y)
{
  return pow (x, y);
}

static inline dtype dtype_sin (dtype x)
{
  return sin (x);
}

static inline dtype dtype_cos (dtype x)
{
  return cos (x);
}

static inline dtype dtype_log (dtype x)
{
  return log (x);
}

static inline dtype dtype_fabs (dtype x)
{
  return fabs (x);
}

static inline dtype dtype_fmax (dtype x,
                                dtype y)
{
  return fmax (x, y);
}

#endif

static inline bool dtype_isfinite (dtype x)
{
  return isfinite ((double) x);
}

#endif
