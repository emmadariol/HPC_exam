/*
 * nbody_common.h
 *
 * Shared constants and precision helpers for every executable in the project.
 * Keeping these definitions in one small header prevents the generator, the
 * serial solver, the hybrid solver, and the layout benchmark from silently
 * disagreeing on the binary file format or on the floating-point type used for
 * arithmetic.  That consistency is essential for fair benchmark comparisons:
 * all kernels must read the same input data and report the same precision mode.
 */

#ifndef NBODY_COMMON_H
#define NBODY_COMMON_H

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Cache-line alignment used for Structure-of-Arrays buffers.  A 64-byte value
 * matches the common x86 cache-line size and gives the compiler/runtime a clean
 * base address for vector loads, stores, and OpenMP first-touch placement.
 */
#ifndef NBODY_ALIGNMENT
#define NBODY_ALIGNMENT 64u
#endif

/*
 * Binary input/output format shared by generate_ic, the solvers, and the
 * layout benchmark:
 *
 *   magic[8]          fixed signature, used to reject wrong files early
 *   uint64_t n        number of particles
 *   float data[n][6]  x,y,z,vx,vy,vz stored in single precision
 *
 * The solvers may compute in double precision, but the assignment baseline
 * stores files as float to keep the datasets compact and portable.
 */
#define NBODY_BINARY_MAGIC_SIZE 8u
#define NBODY_BINARY_COMPONENTS 6u
#define NBODY_BINARY_VERSION_TEXT "nbody-f32-v1"

/* Initial-condition model identifiers accepted by generate_ic. */
#define PLUMMER_SPHERE 0
#define MAXWELL_BALL   1

static const unsigned char  nbody_binary_magic[NBODY_BINARY_MAGIC_SIZE] =
  { 'N', 'B', 'O', 'D', 'Y', 'F', '1', '\0' };

/*
 * The code is compiled in exactly one arithmetic mode.  Double precision is
 * used by default in the Makefile/report runs; float mode is left available for
 * optional sensitivity checks without duplicating the code base.
 */
#if defined (NBODY_USE_FLOAT) && defined (NBODY_USE_DOUBLE)
#error "define only one of NBODY_USE_FLOAT and NBODY_USE_DOUBLE"
#endif

#if defined (NBODY_USE_FLOAT)
typedef float  dtype;
#define DTYPE_NAME "float"
#define DTYPE_MAX_VALUE FLT_MAX
#define DTYPE_MIN_NORMAL FLT_MIN
#define DTYPE_PRINTF_FORMAT "%.9g"

/* Precision-dispatched wrappers keep numerical kernels readable while still
 * calling the correct libm function for the selected dtype. */
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

#else
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

/* One finite-value predicate used by argument parsing and sanity checks. */
static inline bool dtype_isfinite (dtype x)
{
  return isfinite ((double) x);
}

#endif
