// STREAM-like memory-bandwidth test (OpenMP).

#define _POSIX_C_SOURCE 200809L

#include <omp.h>

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef STREAM_N
#define STREAM_N 67108864UL  // 67M doubles = 512 MB per array, much larger than the caches
#endif

#ifndef STREAM_INNER
#define STREAM_INNER 10  // repetitions per kernel; the best one is kept
#endif

static double now_seconds(void)  // monotonic clock in seconds
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static void *checked_aligned_alloc(size_t alignment, size_t bytes)  // aligned allocation that stops on error
{
  void *ptr = NULL;
  if (posix_memalign(&ptr, alignment, bytes) != 0 || !ptr)
  {
    fprintf(stderr, "allocation failed for %zu bytes\n", bytes);
    exit(EXIT_FAILURE);
  }
  return ptr;
}

static size_t env_size_or_default(const char *name, size_t fallback)  // read a positive integer from the environment
{
  const char *text = getenv(name);
  if (!text || !*text)
  {
    return fallback;
  }

  errno = 0;
  char *end = NULL;
  unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0)
  {
    fprintf(stderr, "invalid %s=%s\n", name, text);
    exit(EXIT_FAILURE);
  }
  return (size_t)value;
}

static int env_int_or_default(const char *name, int fallback)
{
  const char *text = getenv(name);
  if (!text || !*text)
  {
    return fallback;
  }

  errno = 0;
  char *end = NULL;
  long value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0)
  {
    fprintf(stderr, "invalid %s=%s\n", name, text);
    exit(EXIT_FAILURE);
  }
  return (int)value;
}

// ===========================================================================
// [OpenMP] parallel first touch
// ===========================================================================
static void initialize_arrays(double *a, double *b, double *c, size_t n)  // first touch in parallel: pages go to the NUMA domain of each thread
{
#pragma omp parallel for schedule(static)  // [OpenMP]
  for (size_t i = 0; i < n; ++i)
  {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 0.0;
  }
}

static double checksum_arrays(const double *a, const double *b, const double *c, size_t n)  // sparse checksum so the compiler cannot skip the loops
{
  double checksum = 0.0;

#pragma omp parallel for reduction(+ : checksum) schedule(static)  // [OpenMP]
  for (size_t i = 0; i < n; i += 4096)
  {
    checksum += a[i] + b[i] + c[i];
  }

  return checksum;
}

int main(void)
{
  const size_t n = env_size_or_default("STREAM_N", STREAM_N);
  const int inner_repeats = env_int_or_default("STREAM_INNER", STREAM_INNER);
  const double scalar = 3.0;
  const size_t bytes = n * sizeof(double);  // bytes per array

  double *a = checked_aligned_alloc(64, bytes);
  double *b = checked_aligned_alloc(64, bytes);
  double *c = checked_aligned_alloc(64, bytes);

  initialize_arrays(a, b, c, n);

  printf("kernel,N,threads,inner_repeats,best_seconds,GBps,checksum\n");

  // ---- STREAM kernels, each an [OpenMP] parallel loop -----------------------
  for (int kernel = 0; kernel < 4; ++kernel)  // 0 copy, 1 scale, 2 add, 3 triad
  {
    const char *name = "copy";
    double bytes_moved = 2.0 * (double)n * sizeof(double);  // copy/scale: read one array, write one
    double best_seconds = INFINITY;

    if (kernel == 1)
    {
      name = "scale";
    }
    else if (kernel == 2)
    {
      name = "add";
      bytes_moved = 3.0 * (double)n * sizeof(double);  // add/triad: read two arrays, write one
    }
    else if (kernel == 3)
    {
      name = "triad";
      bytes_moved = 3.0 * (double)n * sizeof(double);
    }

    for (int rep = 0; rep < inner_repeats; ++rep)
    {
      const double t0 = now_seconds();

      if (kernel == 0)
      {
#pragma omp parallel for schedule(static)  // [OpenMP] copy: c = a
        for (size_t i = 0; i < n; ++i)
        {
          c[i] = a[i];
        }
      }
      else if (kernel == 1)
      {
#pragma omp parallel for schedule(static)  // [OpenMP] scale: b = k*c
        for (size_t i = 0; i < n; ++i)
        {
          b[i] = scalar * c[i];
        }
      }
      else if (kernel == 2)
      {
#pragma omp parallel for schedule(static)  // [OpenMP] add: c = a + b
        for (size_t i = 0; i < n; ++i)
        {
          c[i] = a[i] + b[i];
        }
      }
      else
      {
#pragma omp parallel for schedule(static)  // [OpenMP] triad: a = b + k*c
        for (size_t i = 0; i < n; ++i)
        {
          a[i] = b[i] + scalar * c[i];
        }
      }

      const double elapsed = now_seconds() - t0;
      if (elapsed < best_seconds)  // keep the fastest repetition
      {
        best_seconds = elapsed;
      }
    }

    const double gbps = bytes_moved / best_seconds / 1.0e9;  // bandwidth in GB/s
    const double checksum = checksum_arrays(a, b, c, n);
    printf("%s,%zu,%d,%d,%.9f,%.6f,%.17g\n",
           name, n, omp_get_max_threads(), inner_repeats, best_seconds, gbps, checksum);
  }

  free(a);
  free(b);
  free(c);
  return 0;
}
