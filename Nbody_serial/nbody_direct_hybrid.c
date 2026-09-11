/*
 * nbody_direct_hybrid.c
 *
 * Production solver used for the scaling experiments.  It combines MPI across
 * ranks with OpenMP inside each rank and keeps the same direct O(N^2)
 * gravitational algorithm as the serial reference.  The point of this file is
 * not to change the physics, but to expose the parallelization choices that the
 * report evaluates: rank decomposition, communication mode, force-kernel
 * variants, inverse-square-root approximation, and diagnostic overhead.
 */

#include "nbody_common.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <omp.h>

#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)
#include <immintrin.h>
#endif

typedef struct particles_s
{
  /* Local particle block owned by one MPI rank, stored in Structure-of-Arrays
   * form so the innermost force loop reads contiguous x/y/z source arrays. */
  size_t n;
  dtype mass;
  dtype *x, *y, *z;
  dtype *vx, *vy, *vz;
  dtype *ax, *ay, *az;
} particles_t;

typedef struct timings_s
{
  /* Timings are accumulated by phase and later reduced with MPI_MAX.  Reporting
   * the slowest rank is the correct wall-clock cost of a parallel step. */
  double drift;
  double force;
  double comm_wait;
  double kick;
  double energy;
  double io;
  double total;
} timings_t;

typedef struct thread_workspace_s
{
  /* Thread-private force buffers reused by the Newton ablation.  Keeping these
   * allocations outside the timed force kernel avoids malloc/calloc noise in
   * the measurement. */
  size_t n;
  int nthreads;
  dtype *tax;
  dtype *tay;
  dtype *taz;
} thread_workspace_t;

typedef enum comm_mode_e
{
  /* SENDRECV measures a simple blocking ring exchange.  OVERLAP posts
   * non-blocking communication before computing on the current source block, so
   * communication can be partially hidden by useful force work. */
  COMM_SENDRECV,
  COMM_OVERLAP
} comm_mode_t;

typedef enum kernel_mode_e
{
  /* DIRECT evaluates all source/target pairs.  NEWTON exploits action-reaction
   * symmetry but is implemented only for one rank because cross-rank symmetric
   * updates would require a different communication/reduction scheme. */
  KERNEL_DIRECT,
  KERNEL_NEWTON
} kernel_mode_t;

typedef enum rsqrt_mode_e
{
  /* EXACT uses the selected dtype sqrt.  APPROX uses a float seed plus Newton
   * refinement, trading numerical path changes for speed experiments. */
  RSQRT_EXACT,
  RSQRT_APPROX
} rsqrt_mode_t;

typedef enum accumulator_mode_e
{
  /* Multiple accumulator chains reduce loop-carried dependencies in the direct
   * kernel and can expose more instruction-level parallelism to the compiler. */
  ACCUMULATORS_ONE = 1,
  ACCUMULATORS_TWO = 2,
  ACCUMULATORS_FOUR = 4,
  ACCUMULATORS_EIGHT = 8
} accumulator_mode_t;

/* CLI functions are implemented together near the end of the file, after the
 * simulation routines. These declarations keep main() independent of their
 * physical location. */
static size_t parse_size(const char *text, const char *name);
static dtype parse_dtype(const char *text, const char *name);
static comm_mode_t parse_comm_mode(const char *text);
static kernel_mode_t parse_kernel_mode(const char *text);
static rsqrt_mode_t parse_rsqrt_mode(const char *text);
static accumulator_mode_t parse_accumulator_mode(const char *text);
static const char *option_value(int *i, int argc, char **argv,
                                const char *key);
static void print_usage(const char *program);

static void die(const char *fmt, ...)
{
  /* Abort every rank on fatal errors; otherwise one failed rank could leave the
   * rest of the MPI job hanging inside collectives. */
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fputc('\n', stderr);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static MPI_Datatype mpi_dtype(void)
{
  /* MPI communication must match the compile-time dtype selected in
   * nbody_common.h. */
#if defined(NBODY_USE_FLOAT)
  return MPI_FLOAT;
#else
  return MPI_DOUBLE;
#endif
}

static double seconds(void)
{
  /* MPI_Wtime is monotonic and is available on every MPI rank, so elapsed
   * intervals include both computation and MPI waiting. */
  return MPI_Wtime();
}

static inline dtype refine_rsqrt_newton(dtype r2, dtype x)
{
  const dtype half = (dtype)0.5;
  const dtype three_halves = (dtype)1.5;
  return x * (three_halves - half * r2 * x * x);
}

static inline dtype invsqrt_force(dtype r2, rsqrt_mode_t mode)
{
  if (mode == RSQRT_EXACT)
    return (dtype)1.0 / dtype_sqrt(r2);

  /* Portable approximate fallback: low-precision seed plus two Newton
   * refinements.  When the compiler target exposes AVX-512F, the hot direct
   * kernel uses _mm512_rsqrt14_pd instead; this scalar path remains necessary
   * for remainders, non-AVX builds and the single-rank Newton kernel. */
  {
    dtype x = (dtype)(1.0f / sqrtf((float)r2));
    x = refine_rsqrt_newton(r2, x);
    x = refine_rsqrt_newton(r2, x);
    return x;
  }
}

static void *checked_aligned_alloc(size_t nbytes)
{
  /* Cache-line aligned allocation helps vector loads/stores and avoids
   * accidental misalignment when OpenMP threads touch contiguous blocks. */
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

static void checked_fwrite(const void *ptr, size_t size, size_t nmemb,
                           FILE *fp, const char *path, const char *what)
{
  /* Detect disk-full and other write errors while the operation is identified,
   * instead of silently losing the final state. */
  if (fwrite(ptr, size, nmemb, fp) != nmemb)
    die("failed writing %s to '%s'", what, path);
}

static void particles_init_empty(particles_t *p)
{
  memset(p, 0, sizeof(*p));
  p->mass = (dtype)1.0;
}

static void particles_allocate(particles_t *p, size_t n, dtype mass)
{
  /* Allocate every coordinate/velocity/acceleration component separately.  This
   * SoA layout is the primary memory-layout optimization for the force kernel. */
  const size_t bytes = n * sizeof(dtype);
  particles_init_empty(p);
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

static void particles_free(particles_t *p)
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

static void workspace_init_empty(thread_workspace_t *ws)
{
  memset(ws, 0, sizeof(*ws));
}

static void workspace_allocate(thread_workspace_t *ws, size_t n, int nthreads)
{
  size_t items, bytes;

  if (nthreads <= 0)
    die("invalid OpenMP thread count for Newton workspace");
  if (n > SIZE_MAX / (size_t)nthreads)
    die("Newton workspace size overflow");
  items = (size_t)nthreads * n;
  if (items > SIZE_MAX / sizeof(dtype))
    die("Newton workspace byte-size overflow");
  bytes = items * sizeof(dtype);

  workspace_init_empty(ws);
  ws->n = n;
  ws->nthreads = nthreads;
  ws->tax = checked_aligned_alloc(bytes);
  ws->tay = checked_aligned_alloc(bytes);
  ws->taz = checked_aligned_alloc(bytes);
  if ((items > 0u) && ((ws->tax == NULL) || (ws->tay == NULL) ||
                       (ws->taz == NULL)))
    die("cannot allocate Newton workspace");
}

static void workspace_free(thread_workspace_t *ws)
{
  free(ws->tax);
  free(ws->tay);
  free(ws->taz);
  workspace_init_empty(ws);
}

static void block_bounds(size_t n, int rank, int nranks,
                         size_t *start, size_t *count)
{
  /* Block distribution with the remainder assigned to lower ranks.  Local
   * counts differ by at most one, which keeps the O(N^2/P) work balanced. */
  const size_t base = n / (size_t)nranks;
  const size_t rem = n % (size_t)nranks;
  if (count != NULL)
    *count = base + ((size_t)rank < rem ? 1u : 0u);
  if (start != NULL)
    *start = (size_t)rank * base + ((size_t)rank < rem ? (size_t)rank : rem);
}

static size_t max_block_count(size_t n, int nranks)
{
  size_t count;
  block_bounds(n, 0, nranks, NULL, &count);
  return count;
}

static void read_local_particles(const char *path, dtype mass, int rank,
                                 int nranks, particles_t *local,
                                 size_t *global_n, size_t *local_start,
                                 MPI_Comm comm)
{
  /* Collective MPI-IO input.  All ranks read the shared header together, then
   * each rank reads only its contiguous particle block with an independent
   * offset.  This avoids P ranks scanning the whole file through POSIX fread. */
  MPI_File fh;
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t i, n, local_n, nrecords;
  MPI_Offset offset;
  float *records = NULL;
  float dummy_record = 0.0f;

  if (MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) !=
      MPI_SUCCESS)
    die("MPI_File_open failed for input '%s'", path);

  if (MPI_File_read_all(fh, magic, NBODY_BINARY_MAGIC_SIZE, MPI_BYTE,
                        MPI_STATUS_IGNORE) != MPI_SUCCESS)
    die("MPI_File_read_all failed for magic in '%s'", path);
  if (memcmp(magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die("invalid input magic in '%s'", path);

  if (MPI_File_read_all(fh, &n64, 1, MPI_UINT64_T, MPI_STATUS_IGNORE) !=
      MPI_SUCCESS)
    die("MPI_File_read_all failed for particle count in '%s'", path);
  if ((uint64_t)(size_t)n64 != n64)
    die("input particle count is too large");

  n = (size_t)n64;
  block_bounds(n, rank, nranks, local_start, &local_n);
  particles_allocate(local, local_n, mass);

  nrecords = local_n * NBODY_BINARY_COMPONENTS;
  if (nrecords > (size_t)INT_MAX)
    die("MPI-IO read count exceeds INT_MAX");

  records = checked_aligned_alloc(nrecords * sizeof(float));
  offset = (MPI_Offset)NBODY_BINARY_MAGIC_SIZE +
           (MPI_Offset)sizeof(uint64_t) +
           (MPI_Offset)(*local_start) *
             (MPI_Offset)NBODY_BINARY_COMPONENTS * (MPI_Offset)sizeof(float);

  if (MPI_File_read_at_all(fh, offset, nrecords > 0u ? records : &dummy_record,
                           (int)nrecords, MPI_FLOAT, MPI_STATUS_IGNORE) !=
      MPI_SUCCESS)
    die("MPI_File_read_at_all failed for particle records in '%s'", path);

  for (i = 0u; i < local_n; ++i)
  {
    local->x[i] = (dtype)records[i * NBODY_BINARY_COMPONENTS + 0u];
    local->y[i] = (dtype)records[i * NBODY_BINARY_COMPONENTS + 1u];
    local->z[i] = (dtype)records[i * NBODY_BINARY_COMPONENTS + 2u];
    local->vx[i] = (dtype)records[i * NBODY_BINARY_COMPONENTS + 3u];
    local->vy[i] = (dtype)records[i * NBODY_BINARY_COMPONENTS + 4u];
    local->vz[i] = (dtype)records[i * NBODY_BINARY_COMPONENTS + 5u];
  }

  free(records);
  MPI_File_close(&fh);
  *global_n = n;
}

static float to_float_checked(dtype value, const char *path)
{
  /* The file format is float32 even when the simulation uses double. Prevent a
   * finite in-memory value from becoming Inf during serialization. */
  if (!dtype_isfinite(value) || (fabs((double)value) > (double)FLT_MAX))
    die("cannot write non-finite or overflowing value to '%s'", path);
  return (float)value;
}

static void write_output_root(const char *path, const particles_t *local,
                              size_t global_n, int rank, int nranks,
                              MPI_Comm comm)
{
  /* Optional final-state output is gathered on rank 0 and written in the same
   * binary format as the input.  Benchmark runs normally omit this to avoid I/O
   * contaminating timing data. */
  int *counts = NULL, *displs = NULL;
  dtype *x = NULL, *y = NULL, *z = NULL, *vx = NULL, *vy = NULL, *vz = NULL;
  size_t r;
  MPI_Datatype dt = mpi_dtype();

  counts = malloc((size_t)nranks * sizeof(*counts));
  displs = malloc((size_t)nranks * sizeof(*displs));
  if ((counts == NULL) || (displs == NULL))
    die("cannot allocate gather metadata");
  for (r = 0u; r < (size_t)nranks; ++r)
  {
    size_t start, count;
    block_bounds(global_n, (int)r, nranks, &start, &count);
    if ((count > (size_t)INT_MAX) || (start > (size_t)INT_MAX))
      die("MPI gather count exceeds INT_MAX");
    counts[r] = (int)count;
    displs[r] = (int)start;
  }

  if (rank == 0)
  {
    const size_t bytes = global_n * sizeof(dtype);
    FILE *fp;
    uint64_t n64 = (uint64_t)global_n;
    size_t i;
    x = checked_aligned_alloc(bytes);
    y = checked_aligned_alloc(bytes);
    z = checked_aligned_alloc(bytes);
    vx = checked_aligned_alloc(bytes);
    vy = checked_aligned_alloc(bytes);
    vz = checked_aligned_alloc(bytes);

    MPI_Gatherv(local->x, (int)local->n, dt, x, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->y, (int)local->n, dt, y, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->z, (int)local->n, dt, z, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vx, (int)local->n, dt, vx, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vy, (int)local->n, dt, vy, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vz, (int)local->n, dt, vz, counts, displs, dt, 0, comm);

    fp = fopen(path, "wb");
    if (fp == NULL)
      die("cannot open output '%s'", path);
    checked_fwrite(nbody_binary_magic, 1u, NBODY_BINARY_MAGIC_SIZE, fp, path, "magic");
    checked_fwrite(&n64, sizeof n64, 1u, fp, path, "particle count");
    for (i = 0u; i < global_n; ++i)
    {
      float rec[NBODY_BINARY_COMPONENTS];
      rec[0] = to_float_checked(x[i], path);
      rec[1] = to_float_checked(y[i], path);
      rec[2] = to_float_checked(z[i], path);
      rec[3] = to_float_checked(vx[i], path);
      rec[4] = to_float_checked(vy[i], path);
      rec[5] = to_float_checked(vz[i], path);
      checked_fwrite(rec, sizeof rec[0], NBODY_BINARY_COMPONENTS, fp, path,
                     "particle record");
    }
    fclose(fp);
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
  }
  else
  {
    MPI_Gatherv(local->x, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->y, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->z, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vx, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vy, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vz, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
  }

  free(counts);
  free(displs);
}

static void drift(particles_t *p, dtype dt)
{
  /* Drift step: update positions using current velocities.  Particles are
   * independent here, so OpenMP static scheduling is deterministic and cheap. */
  size_t i;
#pragma omp parallel for schedule(static)
  for (i = 0u; i < p->n; ++i)
  {
    /* x(t+dt) = x(t) + dt*v(t). Each particle is independent in this phase, so
     * the loop is safe to distribute with OpenMP. */
    p->x[i] += dt * p->vx[i];
    p->y[i] += dt * p->vy[i];
    p->z[i] += dt * p->vz[i];
  }
}

static void kick(particles_t *p, dtype dt)
{
  /* Kick step: update velocities using the acceleration field from the latest
   * force evaluation. */
  size_t i;
#pragma omp parallel for schedule(static)
  for (i = 0u; i < p->n; ++i)
  {
    /* v(t+dt) = v(t) + dt*a(t). KDK calls this twice with dt/2. */
    p->vx[i] += dt * p->ax[i];
    p->vy[i] += dt * p->ay[i];
    p->vz[i] += dt * p->az[i];
  }
}

static void accumulate_sources_scalar_chains(const particles_t *home,
                                             const dtype *restrict sx,
                                             const dtype *restrict sy,
                                             const dtype *restrict sz,
                                             size_t source_n, dtype g,
                                             dtype mass, dtype eps2,
                                             rsqrt_mode_t rsqrt_mode,
                                             size_t chains)
{
  /* For each target i, sum the acceleration generated by every source j:
   * a_i += G*m*(r_j-r_i) / (|r_j-r_i|^2 + eps^2)^(3/2).
   * The source arrays are read-only; the target acceleration is incremented
   * because an MPI ring visits several source blocks. */
  size_t i;
  const dtype gm = g * mass;

  if ((chains != 1u) && (chains != 2u) && (chains != 4u) && (chains != 8u))
    die("unsupported accumulator chain count %zu", chains);

#pragma omp parallel for schedule(static)
  for (i = 0u; i < home->n; ++i)
  {
    const dtype xi = home->x[i];
    const dtype yi = home->y[i];
    const dtype zi = home->z[i];
    dtype ax = (dtype)0.0, ay = (dtype)0.0, az = (dtype)0.0;
    size_t j = 0u;

    switch (chains)
    {
    case 1u:
      /* The common remainder loop below is the whole loop for one chain. */
      break;

    case 2u:
    {
      const size_t source_n_unrolled = source_n - (source_n % 2u);
      dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;
      dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;

#pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1)
      for (j = 0u; j < source_n_unrolled; j += 2u)
      {
        const dtype dx0 = sx[j] - xi;
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);
        const dtype s0 = gm * invr0 * invr0 * invr0;
        ax0 += dx0 * s0;
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        const dtype dx1 = sx[j + 1u] - xi;
        const dtype dy1 = sy[j + 1u] - yi;
        const dtype dz1 = sz[j + 1u] - zi;
        const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
        const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
        const dtype s1 = gm * invr1 * invr1 * invr1;
        ax1 += dx1 * s1;
        ay1 += dy1 * s1;
        az1 += dz1 * s1;
      }

      ax += ax0 + ax1;
      ay += ay0 + ay1;
      az += az0 + az1;
      break;
    }

    case 4u:
    {
      const size_t source_n_unrolled = source_n - (source_n % 4u);
      dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;
      dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;
      dtype ax2 = (dtype)0.0, ay2 = (dtype)0.0, az2 = (dtype)0.0;
      dtype ax3 = (dtype)0.0, ay3 = (dtype)0.0, az3 = (dtype)0.0;

#pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1) \
    reduction(+ : ax2, ay2, az2) reduction(+ : ax3, ay3, az3)
      for (j = 0u; j < source_n_unrolled; j += 4u)
      {
        const dtype dx0 = sx[j] - xi;
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);
        const dtype s0 = gm * invr0 * invr0 * invr0;
        ax0 += dx0 * s0;
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        const dtype dx1 = sx[j + 1u] - xi;
        const dtype dy1 = sy[j + 1u] - yi;
        const dtype dz1 = sz[j + 1u] - zi;
        const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
        const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
        const dtype s1 = gm * invr1 * invr1 * invr1;
        ax1 += dx1 * s1;
        ay1 += dy1 * s1;
        az1 += dz1 * s1;

        const dtype dx2 = sx[j + 2u] - xi;
        const dtype dy2 = sy[j + 2u] - yi;
        const dtype dz2 = sz[j + 2u] - zi;
        const dtype r2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
        const dtype invr2 = invsqrt_force(r2_2, rsqrt_mode);
        const dtype s2 = gm * invr2 * invr2 * invr2;
        ax2 += dx2 * s2;
        ay2 += dy2 * s2;
        az2 += dz2 * s2;

        const dtype dx3 = sx[j + 3u] - xi;
        const dtype dy3 = sy[j + 3u] - yi;
        const dtype dz3 = sz[j + 3u] - zi;
        const dtype r2_3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
        const dtype invr3 = invsqrt_force(r2_3, rsqrt_mode);
        const dtype s3 = gm * invr3 * invr3 * invr3;
        ax3 += dx3 * s3;
        ay3 += dy3 * s3;
        az3 += dz3 * s3;
      }

      ax += ax0 + ax1 + ax2 + ax3;
      ay += ay0 + ay1 + ay2 + ay3;
      az += az0 + az1 + az2 + az3;
      break;
    }

    case 8u:
    {
      const size_t source_n_unrolled = source_n - (source_n % 8u);
      dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;
      dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;
      dtype ax2 = (dtype)0.0, ay2 = (dtype)0.0, az2 = (dtype)0.0;
      dtype ax3 = (dtype)0.0, ay3 = (dtype)0.0, az3 = (dtype)0.0;
      dtype ax4 = (dtype)0.0, ay4 = (dtype)0.0, az4 = (dtype)0.0;
      dtype ax5 = (dtype)0.0, ay5 = (dtype)0.0, az5 = (dtype)0.0;
      dtype ax6 = (dtype)0.0, ay6 = (dtype)0.0, az6 = (dtype)0.0;
      dtype ax7 = (dtype)0.0, ay7 = (dtype)0.0, az7 = (dtype)0.0;

#pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1) \
    reduction(+ : ax2, ay2, az2) reduction(+ : ax3, ay3, az3)             \
    reduction(+ : ax4, ay4, az4) reduction(+ : ax5, ay5, az5)             \
    reduction(+ : ax6, ay6, az6) reduction(+ : ax7, ay7, az7)
      for (j = 0u; j < source_n_unrolled; j += 8u)
      {
        const dtype dx0 = sx[j] - xi;
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);
        const dtype s0 = gm * invr0 * invr0 * invr0;
        ax0 += dx0 * s0;
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        const dtype dx1 = sx[j + 1u] - xi;
        const dtype dy1 = sy[j + 1u] - yi;
        const dtype dz1 = sz[j + 1u] - zi;
        const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
        const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
        const dtype s1 = gm * invr1 * invr1 * invr1;
        ax1 += dx1 * s1;
        ay1 += dy1 * s1;
        az1 += dz1 * s1;

        const dtype dx2 = sx[j + 2u] - xi;
        const dtype dy2 = sy[j + 2u] - yi;
        const dtype dz2 = sz[j + 2u] - zi;
        const dtype r2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
        const dtype invr2 = invsqrt_force(r2_2, rsqrt_mode);
        const dtype s2 = gm * invr2 * invr2 * invr2;
        ax2 += dx2 * s2;
        ay2 += dy2 * s2;
        az2 += dz2 * s2;

        const dtype dx3 = sx[j + 3u] - xi;
        const dtype dy3 = sy[j + 3u] - yi;
        const dtype dz3 = sz[j + 3u] - zi;
        const dtype r2_3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
        const dtype invr3 = invsqrt_force(r2_3, rsqrt_mode);
        const dtype s3 = gm * invr3 * invr3 * invr3;
        ax3 += dx3 * s3;
        ay3 += dy3 * s3;
        az3 += dz3 * s3;

        const dtype dx4 = sx[j + 4u] - xi;
        const dtype dy4 = sy[j + 4u] - yi;
        const dtype dz4 = sz[j + 4u] - zi;
        const dtype r2_4 = dx4 * dx4 + dy4 * dy4 + dz4 * dz4 + eps2;
        const dtype invr4 = invsqrt_force(r2_4, rsqrt_mode);
        const dtype s4 = gm * invr4 * invr4 * invr4;
        ax4 += dx4 * s4;
        ay4 += dy4 * s4;
        az4 += dz4 * s4;

        const dtype dx5 = sx[j + 5u] - xi;
        const dtype dy5 = sy[j + 5u] - yi;
        const dtype dz5 = sz[j + 5u] - zi;
        const dtype r2_5 = dx5 * dx5 + dy5 * dy5 + dz5 * dz5 + eps2;
        const dtype invr5 = invsqrt_force(r2_5, rsqrt_mode);
        const dtype s5 = gm * invr5 * invr5 * invr5;
        ax5 += dx5 * s5;
        ay5 += dy5 * s5;
        az5 += dz5 * s5;

        const dtype dx6 = sx[j + 6u] - xi;
        const dtype dy6 = sy[j + 6u] - yi;
        const dtype dz6 = sz[j + 6u] - zi;
        const dtype r2_6 = dx6 * dx6 + dy6 * dy6 + dz6 * dz6 + eps2;
        const dtype invr6 = invsqrt_force(r2_6, rsqrt_mode);
        const dtype s6 = gm * invr6 * invr6 * invr6;
        ax6 += dx6 * s6;
        ay6 += dy6 * s6;
        az6 += dz6 * s6;

        const dtype dx7 = sx[j + 7u] - xi;
        const dtype dy7 = sy[j + 7u] - yi;
        const dtype dz7 = sz[j + 7u] - zi;
        const dtype r2_7 = dx7 * dx7 + dy7 * dy7 + dz7 * dz7 + eps2;
        const dtype invr7 = invsqrt_force(r2_7, rsqrt_mode);
        const dtype s7 = gm * invr7 * invr7 * invr7;
        ax7 += dx7 * s7;
        ay7 += dy7 * s7;
        az7 += dz7 * s7;
      }

      ax += ax0 + ax1 + ax2 + ax3 + ax4 + ax5 + ax6 + ax7;
      ay += ay0 + ay1 + ay2 + ay3 + ay4 + ay5 + ay6 + ay7;
      az += az0 + az1 + az2 + az3 + az4 + az5 + az6 + az7;
      break;
    }

    default:
      break;
    }

#pragma omp simd reduction(+ : ax, ay, az)
    for (; j < source_n; ++j)
    {
      /* Include the remainder when source_n is not divisible by chains. */
      const dtype dx = sx[j] - xi;
      const dtype dy = sy[j] - yi;
      const dtype dz = sz[j] - zi;
      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = invsqrt_force(r2, rsqrt_mode);
      const dtype s = gm * invr * invr * invr;
      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }

    home->ax[i] += ax;
    home->ay[i] += ay;
    home->az[i] += az;
  }
}

#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)
static inline __m512d refine_rsqrt_newton_pd(__m512d r2, __m512d x)
{
  const __m512d half = _mm512_set1_pd(0.5);
  const __m512d three = _mm512_set1_pd(3.0);
  return _mm512_mul_pd(_mm512_mul_pd(half, x),
                       _mm512_sub_pd(three, _mm512_mul_pd(r2, _mm512_mul_pd(x, x))));
}

static inline double hsum_pd(__m512d v)
{
  double tmp[8];
  _mm512_storeu_pd(tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3] +
         tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

static void accumulate_sources_rsqrt14_pd(const particles_t *home,
                                          const double *restrict sx,
                                          const double *restrict sy,
                                          const double *restrict sz,
                                          size_t source_n, double g,
                                          double mass, double eps2)
{
  const __m512d eps2_v = _mm512_set1_pd(eps2);
  const __m512d gm_v = _mm512_set1_pd(g * mass);
  size_t i;

#pragma omp parallel for schedule(static)
  for (i = 0u; i < home->n; ++i)
  {
    const __m512d xi = _mm512_set1_pd(home->x[i]);
    const __m512d yi = _mm512_set1_pd(home->y[i]);
    const __m512d zi = _mm512_set1_pd(home->z[i]);
    __m512d ax_v = _mm512_setzero_pd();
    __m512d ay_v = _mm512_setzero_pd();
    __m512d az_v = _mm512_setzero_pd();
    size_t j = 0u;

    for (; j + 7u < source_n; j += 8u)
    {
      const __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(sx + j), xi);
      const __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(sy + j), yi);
      const __m512d dz = _mm512_sub_pd(_mm512_loadu_pd(sz + j), zi);
      __m512d r2 = _mm512_fmadd_pd(dx, dx, eps2_v);
      r2 = _mm512_fmadd_pd(dy, dy, r2);
      r2 = _mm512_fmadd_pd(dz, dz, r2);

      __m512d invr = _mm512_rsqrt14_pd(r2);
      invr = refine_rsqrt_newton_pd(r2, invr);
      invr = refine_rsqrt_newton_pd(r2, invr);

      const __m512d invr2 = _mm512_mul_pd(invr, invr);
      const __m512d s = _mm512_mul_pd(gm_v, _mm512_mul_pd(invr2, invr));
      ax_v = _mm512_fmadd_pd(dx, s, ax_v);
      ay_v = _mm512_fmadd_pd(dy, s, ay_v);
      az_v = _mm512_fmadd_pd(dz, s, az_v);
    }

    double ax = hsum_pd(ax_v);
    double ay = hsum_pd(ay_v);
    double az = hsum_pd(az_v);
    for (; j < source_n; ++j)
    {
      const double dx = sx[j] - home->x[i];
      const double dy = sy[j] - home->y[i];
      const double dz = sz[j] - home->z[i];
      const double r2 = dx * dx + dy * dy + dz * dz + eps2;
      const double invr = invsqrt_force(r2, RSQRT_APPROX);
      const double s = g * mass * invr * invr * invr;
      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }

    home->ax[i] += ax;
    home->ay[i] += ay;
    home->az[i] += az;
  }
}
#endif

static void accumulate_sources(const particles_t *home,
                               const dtype *restrict sx,
                               const dtype *restrict sy,
                               const dtype *restrict sz,
                               size_t source_n,
                               dtype g, dtype mass, dtype eps,
                               rsqrt_mode_t rsqrt_mode,
                               accumulator_mode_t accumulator_mode)
{
  /* Core direct force kernel.  `home` is the local target block; sx/sy/sz is the
   * current source block, which may belong to this rank or may have arrived from
   * a neighbor in the MPI ring. */
  const dtype eps2 = eps * eps;

#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)
  if (rsqrt_mode == RSQRT_APPROX)
  {
    /* AVX-512 provides a vector reciprocal-square-root estimate. Newton
     * refinement improves it before it is used in the force formula. */
    accumulate_sources_rsqrt14_pd(home, sx, sy, sz, source_n, g, mass, eps2);
    return;
  }
#endif

  accumulate_sources_scalar_chains(home, sx, sy, sz, source_n, g, mass, eps2,
                                   rsqrt_mode, (size_t)accumulator_mode);
}

static void compute_accelerations_newton_private(particles_t *local, dtype g,
                                                 dtype eps,
                                                 rsqrt_mode_t rsqrt_mode,
                                                 thread_workspace_t *workspace)
{
  /* Single-rank Newton kernel.  Thread-private acceleration buffers avoid races
   * when applying equal-and-opposite pair contributions, then a reduction pass
   * merges those buffers into the real acceleration arrays. */
  const size_t n = local->n;
  const dtype eps2 = eps * eps;
  int nthreads;
  size_t i;

  if (workspace == NULL)
    die("invalid Newton workspace");
  nthreads = workspace->nthreads;
  if ((workspace->n < n) || (nthreads < omp_get_max_threads()) ||
      (workspace->tax == NULL) || (workspace->tay == NULL) ||
      (workspace->taz == NULL))
    die("invalid Newton workspace");

  memset(workspace->tax, 0, (size_t)nthreads * n * sizeof(dtype));
  memset(workspace->tay, 0, (size_t)nthreads * n * sizeof(dtype));
  memset(workspace->taz, 0, (size_t)nthreads * n * sizeof(dtype));

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    dtype *ax = workspace->tax + (size_t)tid * n;
    dtype *ay = workspace->tay + (size_t)tid * n;
    dtype *az = workspace->taz + (size_t)tid * n;
    size_t ii;

#pragma omp for schedule(static)
    for (ii = 0u; ii < n; ++ii)
    {
      const dtype xi = local->x[ii];
      const dtype yi = local->y[ii];
      const dtype zi = local->z[ii];
      size_t j;
      for (j = ii + 1u; j < n; ++j)
      {
        /* Evaluate each unordered pair once. The equal-and-opposite update
         * halves arithmetic, while thread-private arrays prevent data races. */
        const dtype dx = local->x[j] - xi;
        const dtype dy = local->y[j] - yi;
        const dtype dz = local->z[j] - zi;
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = invsqrt_force(r2, rsqrt_mode);
        const dtype s = g * local->mass * invr * invr * invr;
        const dtype fx = dx * s;
        const dtype fy = dy * s;
        const dtype fz = dz * s;
        ax[ii] += fx;
        ay[ii] += fy;
        az[ii] += fz;
        ax[j] -= fx;
        ay[j] -= fy;
        az[j] -= fz;
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (i = 0u; i < n; ++i)
  {
    /* Merge private pair contributions into the shared acceleration field in a
     * separate pass; direct shared updates in the pair loop would race. */
    dtype ax = (dtype)0.0;
    dtype ay = (dtype)0.0;
    dtype az = (dtype)0.0;
    int t;
    for (t = 0; t < nthreads; ++t)
    {
      ax += workspace->tax[(size_t)t * n + i];
      ay += workspace->tay[(size_t)t * n + i];
      az += workspace->taz[(size_t)t * n + i];
    }
    local->ax[i] = ax;
    local->ay[i] = ay;
    local->az[i] = az;
  }
}

static void exchange_sources_sendrecv(dtype *bx, dtype *by, dtype *bz,
                                      dtype *rx, dtype *ry, dtype *rz,
                                      size_t buf_n, size_t next_n,
                                      int rank, int nranks, int tag_base,
                                      MPI_Datatype dt, MPI_Comm comm)
{
  /* Blocking ring exchange: send the current source block clockwise and receive
   * the previous rank's block counter-clockwise, component by component. */
  const int send_to = (rank + 1) % nranks;
  const int recv_from = (rank + nranks - 1) % nranks;
  MPI_Sendrecv(bx, (int)buf_n, dt, send_to, tag_base + 0,
               rx, (int)next_n, dt, recv_from, tag_base + 0, comm,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(by, (int)buf_n, dt, send_to, tag_base + 1,
               ry, (int)next_n, dt, recv_from, tag_base + 1, comm,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(bz, (int)buf_n, dt, send_to, tag_base + 2,
               rz, (int)next_n, dt, recv_from, tag_base + 2, comm,
               MPI_STATUS_IGNORE);
}

static void post_source_exchange(dtype *bx, dtype *by, dtype *bz,
                                 dtype *rx, dtype *ry, dtype *rz,
                                 size_t buf_n, size_t next_n,
                                 int rank, int nranks, int tag_base,
                                 MPI_Datatype dt, MPI_Comm comm,
                                 MPI_Request req[6])
{
  /* Non-blocking ring exchange used by COMM_OVERLAP.  Receives are posted before
   * sends to avoid ordering hazards; the caller computes on the current block
   * before waiting for completion. */
  const int send_to = (rank + 1) % nranks;
  const int recv_from = (rank + nranks - 1) % nranks;
  MPI_Irecv(rx, (int)next_n, dt, recv_from, tag_base + 0, comm, &req[0]);
  MPI_Irecv(ry, (int)next_n, dt, recv_from, tag_base + 1, comm, &req[1]);
  MPI_Irecv(rz, (int)next_n, dt, recv_from, tag_base + 2, comm, &req[2]);
  MPI_Isend(bx, (int)buf_n, dt, send_to, tag_base + 0, comm, &req[3]);
  MPI_Isend(by, (int)buf_n, dt, send_to, tag_base + 1, comm, &req[4]);
  MPI_Isend(bz, (int)buf_n, dt, send_to, tag_base + 2, comm, &req[5]);
}

static void compute_accelerations_ring(particles_t *local, size_t global_n,
                                       dtype g, dtype eps,
                                       int rank, int nranks, comm_mode_t mode,
                                       kernel_mode_t kernel_mode,
                                       rsqrt_mode_t rsqrt_mode,
                                       accumulator_mode_t accumulator_mode,
                                       thread_workspace_t *newton_workspace,
                                       double *comm_wait,
                                       MPI_Comm comm)
{
  /* MPI ring algorithm for all-pairs direct summation.  Each rank starts with
   * its own local particles as sources, computes their contribution to local
   * targets, then circulates source blocks until every rank has seen every
   * particle. */
  /* Newton uses only the local particle arrays. Handle it before allocating
   * the temporary MPI ring buffers used by the direct kernel. */
  if (kernel_mode == KERNEL_NEWTON)
  {
    if (nranks != 1)
      die("--kernel newton is implemented for -np 1 only; use --kernel direct for MPI ring runs");
    compute_accelerations_newton_private(local, g, eps, rsqrt_mode,
                                         newton_workspace);
    return;
  }

  const size_t max_n = max_block_count(global_n, nranks);
  dtype *bx = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *by = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *bz = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rx = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *ry = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rz = checked_aligned_alloc(max_n * sizeof(dtype));
  MPI_Datatype dt = mpi_dtype();
  int owner = rank;
  size_t buf_n = local->n;
  size_t i;

  /* Start each force evaluation from zero. Every source block in the ring then
   * adds its contribution to this local target acceleration. */
#pragma omp parallel for schedule(static)
  for (i = 0u; i < local->n; ++i)
    local->ax[i] = local->ay[i] = local->az[i] = (dtype)0.0;
  memcpy(bx, local->x, local->n * sizeof(dtype));
  memcpy(by, local->y, local->n * sizeof(dtype));
  memcpy(bz, local->z, local->n * sizeof(dtype));

  for (int step = 0; step < nranks; ++step)
  {
    /* Step 0 computes the local source block; later steps compute blocks that
     * have traveled around the ring. */
    if (nranks > 1)
    {
      const int next_owner = (owner + nranks - 1) % nranks;
      size_t next_n;
      block_bounds(global_n, next_owner, nranks, NULL, &next_n);
      if (mode == COMM_OVERLAP)
      {
        MPI_Request req[6];
        post_source_exchange(bx, by, bz, rx, ry, rz, buf_n, next_n,
                             rank, nranks, 10, dt, comm, req);
        /* Compute with the old buffer while MPI transfers the next one. The
         * wait is required before rx/ry/rz can be copied into the source buffer. */
        accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,
                           rsqrt_mode, accumulator_mode);
        {
          const double comm_t0 = seconds();
          MPI_Waitall(6, req, MPI_STATUSES_IGNORE);
          *comm_wait += seconds() - comm_t0;
        }
      }
      else
      {
        /* Baseline mode computes first and performs the blocking exchange
         * afterwards, so communication cannot overlap force work. */
        accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,
                           rsqrt_mode, accumulator_mode);
        {
          const double comm_t0 = seconds();
          exchange_sources_sendrecv(bx, by, bz, rx, ry, rz, buf_n, next_n,
                                    rank, nranks, 10, dt, comm);
          *comm_wait += seconds() - comm_t0;
        }
      }
      memcpy(bx, rx, next_n * sizeof(dtype));
      memcpy(by, ry, next_n * sizeof(dtype));
      memcpy(bz, rz, next_n * sizeof(dtype));
      owner = next_owner;
      buf_n = next_n;
    }
    else
      accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,
                         rsqrt_mode, accumulator_mode);
  }

  free(bx);
  free(by);
  free(bz);
  free(rx);
  free(ry);
  free(rz);
}

static long double kinetic_energy_local(const particles_t *p)
{
  /* Kinetic energy is local to each rank and then summed globally.  Long double
   * accumulation reduces diagnostic roundoff without changing simulation dtype. */
  long double sum = 0.0L;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (i = 0u; i < p->n; ++i)
  {
    /* K = 1/2 sum_i m|v_i|^2. Long double reduces diagnostic roundoff without
     * changing the dtype used by the simulation itself. */
    const long double vx = (long double)p->vx[i];
    const long double vy = (long double)p->vy[i];
    const long double vz = (long double)p->vz[i];
    sum += vx * vx + vy * vy + vz * vz;
  }
  return 0.5L * (long double)p->mass * sum;
}

static long double potential_sources(const particles_t *home, size_t home_start,
                                     const dtype *restrict sx,
                                     const dtype *restrict sy,
                                     const dtype *restrict sz,
                                     size_t source_start, size_t source_n,
                                     dtype g, dtype eps)
{
  const dtype eps2 = eps * eps;
  const long double m2 = (long double)home->mass * (long double)home->mass;
  long double sum = 0.0L;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (i = 0u; i < home->n; ++i)
  {
    const dtype xi = home->x[i], yi = home->y[i], zi = home->z[i];
    const size_t gi = home_start + i;
    size_t j;
    for (j = 0u; j < source_n; ++j)
      if (gi < source_start + j)
      {
        /* The strict global-index test counts each unordered pair exactly once:
         * pair (i,j) is included only when global i < global j. */
        const dtype dx = sx[j] - xi;
        const dtype dy = sy[j] - yi;
        const dtype dz = sz[j] - zi;
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = (dtype)1.0 / dtype_sqrt(r2);
        sum -= (long double)g * m2 * (long double)invr;
      }
  }
  return sum;
}

static dtype total_energy_ring(const particles_t *local, size_t global_n,
                               size_t local_start, dtype g, dtype eps,
                               int rank, int nranks, MPI_Comm comm,
                               dtype *kinetic, dtype *potential)
{
  /* Potential energy uses a second ring traversal.  It is expensive O(N^2), so
   * production runs sample it periodically and report the measured overhead. */
  const size_t max_n = max_block_count(global_n, nranks);
  dtype *bx = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *by = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *bz = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rx = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *ry = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rz = checked_aligned_alloc(max_n * sizeof(dtype));
  MPI_Datatype dt = mpi_dtype();
  int owner = rank;
  size_t buf_start = local_start, buf_n = local->n;
  long double pot_local = 0.0L, kin_local = kinetic_energy_local(local);
  long double pot_global, kin_global;

  memcpy(bx, local->x, local->n * sizeof(dtype));
  memcpy(by, local->y, local->n * sizeof(dtype));
  memcpy(bz, local->z, local->n * sizeof(dtype));

  for (int step = 0; step < nranks; ++step)
  {
    /* Each rank evaluates its local targets against the current source block.
     * The global-index filter avoids double counting; Allreduce combines the
     * disjoint local sums at the end. */
    pot_local += potential_sources(local, local_start, bx, by, bz,
                                   buf_start, buf_n, g, eps);
    if (nranks > 1)
    {
      const int send_to = (rank + 1) % nranks;
      const int recv_from = (rank + nranks - 1) % nranks;
      const int next_owner = (owner + nranks - 1) % nranks;
      size_t next_start, next_n;
      block_bounds(global_n, next_owner, nranks, &next_start, &next_n);
      MPI_Sendrecv(bx, (int)buf_n, dt, send_to, 20,
                   rx, (int)next_n, dt, recv_from, 20, comm,
                   MPI_STATUS_IGNORE);
      MPI_Sendrecv(by, (int)buf_n, dt, send_to, 21,
                   ry, (int)next_n, dt, recv_from, 21, comm,
                   MPI_STATUS_IGNORE);
      MPI_Sendrecv(bz, (int)buf_n, dt, send_to, 22,
                   rz, (int)next_n, dt, recv_from, 22, comm,
                   MPI_STATUS_IGNORE);
      memcpy(bx, rx, next_n * sizeof(dtype));
      memcpy(by, ry, next_n * sizeof(dtype));
      memcpy(bz, rz, next_n * sizeof(dtype));
      owner = next_owner;
      buf_start = next_start;
      buf_n = next_n;
    }
  }

  MPI_Allreduce(&kin_local, &kin_global, 1, MPI_LONG_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&pot_local, &pot_global, 1, MPI_LONG_DOUBLE, MPI_SUM, comm);
  *kinetic = (dtype)kin_global;
  *potential = (dtype)pot_global;

  free(bx);
  free(by);
  free(bz);
  free(rx);
  free(ry);
  free(rz);
  return *kinetic + *potential;
}

/* Command-line parsing is kept together at the end of the implementation so
 * the numerical and MPI/OpenMP routines above remain easy to follow. */
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

static comm_mode_t parse_comm_mode(const char *text)
{
  if (strcmp(text, "sendrecv") == 0)
    return COMM_SENDRECV;
  if (strcmp(text, "overlap") == 0)
    return COMM_OVERLAP;
  die("invalid --comm '%s' (expected sendrecv or overlap)", text);
  return COMM_SENDRECV;
}

static kernel_mode_t parse_kernel_mode(const char *text)
{
  if (strcmp(text, "direct") == 0)
    return KERNEL_DIRECT;
  if (strcmp(text, "newton") == 0)
    return KERNEL_NEWTON;
  die("invalid --kernel '%s' (expected direct or newton)", text);
  return KERNEL_DIRECT;
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

static accumulator_mode_t parse_accumulator_mode(const char *text)
{
  if (strcmp(text, "1") == 0)
    return ACCUMULATORS_ONE;
  if (strcmp(text, "2") == 0)
    return ACCUMULATORS_TWO;
  if (strcmp(text, "4") == 0)
    return ACCUMULATORS_FOUR;
  if (strcmp(text, "8") == 0)
    return ACCUMULATORS_EIGHT;
  die("invalid --accumulators '%s' (expected 1, 2, 4 or 8)", text);
  return ACCUMULATORS_FOUR;
}

static const char *comm_mode_name(comm_mode_t mode)
{
  return mode == COMM_OVERLAP ? "overlap" : "sendrecv";
}

static const char *kernel_mode_name(kernel_mode_t mode)
{
  return mode == KERNEL_NEWTON ? "newton" : "direct";
}

/* ------------------------------------------------------------ */
/* Convert a rsqrt mode to a string.                            */
/* ------------------------------------------------------------ */

static const char *rsqrt_mode_name(rsqrt_mode_t mode)
{
  return mode == RSQRT_APPROX ? "approx" : "exact";
}

/* ------------------------------------------------------------ */
/* function to convert accumulator mode to a string             */
/* ------------------------------------------------------------ */

static const char *accumulator_mode_name(accumulator_mode_t mode)
{
  switch (mode)
  {
  case ACCUMULATORS_ONE:
    return "1";
  case ACCUMULATORS_TWO:
    return "2";
  case ACCUMULATORS_FOUR:
    return "4";
  case ACCUMULATORS_EIGHT:
    return "8";
  }
  return "unknown";
}

/* ------------------------------------------------------------ */
/* function to extract the value of a command-line option */
/* ------------------------------------------------------------ */

static const char *option_value(int *i, int argc, char **argv, const char *key)
{
  /* Accept both --key=value and --key value. The second form advances *i so
   * the outer loop does not process the value a second time. */
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

/* ------------------------------------------------------------ */
/* function to print usage information */
/* ------------------------------------------------------------ */

static void print_usage(const char *program)
{
  fprintf(stderr,
          "usage: %s --input FILE [options]\n"
          "\n"
          "options:\n"
          "  --input FILE              input binary particle file (%s)\n"
          "  --output FILE             optional final-state binary file\n"
          "  --nsteps N                KDK steps (default: 10)\n"
          "  --dt X                    time step (default: 0.001)\n"
          "  --eps X                   softening length (default: 0.01)\n"
          "  --G X                     gravitational constant (default: 1)\n"
          "  --mass X                  particle mass (default: 1)\n"
          "  --energy-every N          diagnostic period in steps (default: 1)\n"
          "  --energy-tol X            warning tolerance for max relative drift (default: 1e-4)\n"
          "  --comm sendrecv|overlap   ring exchange mode (default: sendrecv)\n"
          "  --kernel direct|newton    force kernel (default: direct; newton is -np 1 only)\n"
          "  --rsqrt exact|approx      inverse square-root mode (default: exact)\n"
          "  --accumulators 1|2|4|8    direct-kernel accumulator chains (default: 4)\n"
          "  --quiet                   final summary only\n"
          "  --help                    show this help message\n",
          program, NBODY_BINARY_VERSION_TEXT);
  fprintf(stderr, "binary format: %s\n", NBODY_BINARY_VERSION_TEXT);
}

/*                  ************************************************          */
/*                                         MAIN                               */
/*                  ************************************************          */

int main(int argc, char **argv)
{
  /* Main is intentionally linear: initialize MPI, parse configuration, read the
   * local particle block, run the selected leapfrog scheme, reduce timings, and
   * print machine-readable summary lines for run_benchmarks.sh. */
  const char *input_path = NULL, *output_path = NULL;
  size_t nsteps = 10u, energy_every = 1u, global_n = 0u, local_start = 0u;
  dtype dt = (dtype)1.0e-3, eps = (dtype)1.0e-2;
  dtype g = (dtype)1.0, mass = (dtype)1.0, energy_tol = (dtype)1.0e-4;
  comm_mode_t comm_mode = COMM_SENDRECV;
  kernel_mode_t kernel_mode = KERNEL_DIRECT;
  rsqrt_mode_t rsqrt_mode = RSQRT_EXACT;
  accumulator_mode_t accumulator_mode = ACCUMULATORS_FOUR;
  bool quiet = false;
  int rank, nranks, provided;
  particles_t local;
  thread_workspace_t newton_workspace;
  timings_t timing = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  dtype kinetic0, potential0, energy0;
  double max_rel_drift = 0.0;
  double t0, t1;

  /* MPI_THREAD_FUNNELED is enough because MPI calls are made by the main thread
   * while OpenMP is used only inside compute loops. */
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  /* FUNNELED permits MPI calls only from the main thread. OpenMP regions do
   * arithmetic only, and all MPI calls remain outside those regions. */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  particles_init_empty(&local);
  workspace_init_empty(&newton_workspace);

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
    else if ((value = option_value(&argi, argc, argv, "--comm")) != NULL)
      comm_mode = parse_comm_mode(value);
    else if ((value = option_value(&argi, argc, argv, "--kernel")) != NULL)
      kernel_mode = parse_kernel_mode(value);
    else if ((value = option_value(&argi, argc, argv, "--rsqrt")) != NULL)
      rsqrt_mode = parse_rsqrt_mode(value);
    else if ((value = option_value(&argi, argc, argv, "--accumulators")) != NULL)
      accumulator_mode = parse_accumulator_mode(value);
    else if ((value = option_value(&argi, argc, argv, "--eps")) != NULL)
      eps = parse_dtype(value, "--eps");
    else if ((value = option_value(&argi, argc, argv, "--G")) != NULL)
      g = parse_dtype(value, "--G");
    else if ((value = option_value(&argi, argc, argv, "--mass")) != NULL)
      mass = parse_dtype(value, "--mass");
    else if ((value = option_value(&argi, argc, argv, "--energy-tol")) != NULL)
      energy_tol = parse_dtype(value, "--energy-tol");
    else if (strcmp(argv[argi], "--quiet") == 0)
      quiet = true;
    else if (strcmp(argv[argi], "--help") == 0)
    {
      if (rank == 0)
        print_usage(argv[0]);
      MPI_Finalize();
      return EXIT_SUCCESS;
    }
    else
      die("unknown option: %s", argv[argi]);
  }

  if (input_path == NULL)
    die("missing required --input FILE");
  if ((dt <= (dtype)0.0) || (eps < (dtype)0.0) || (g <= (dtype)0.0) ||
      (mass <= (dtype)0.0) || (energy_every == 0u) ||
      (energy_tol <= (dtype)0.0))
    die("invalid non-positive physical or diagnostic parameter");

  /* Run one simulation using the inverse-square-root mode selected by the user. */
  timing = (timings_t){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  max_rel_drift = 0.0;

  /* Initial energy is computed before integration so every run reports a
   * relative energy-drift correctness metric. */
  timing.total = seconds();
  t0 = seconds();
  read_local_particles(input_path, mass, rank, nranks, &local,
                       &global_n, &local_start, MPI_COMM_WORLD);
  timing.io += seconds() - t0;

  if (kernel_mode == KERNEL_NEWTON)
    workspace_allocate(&newton_workspace, local.n, omp_get_max_threads());

  t0 = seconds();
  energy0 = total_energy_ring(&local, global_n, local_start, g, eps, rank,
                              nranks, MPI_COMM_WORLD, &kinetic0, &potential0);
  timing.energy += seconds() - t0;

  if ((rank == 0) && !quiet)
  {
    printf("# hybrid MPI+OpenMP direct N-body %s\n",
           "kdk");
    printf("# ranks=%d omp_max_threads=%d arithmetic_dtype=%s comm=%s "
           "kernel=%s rsqrt=%s accumulators=%s\n",
           nranks, omp_get_max_threads(), DTYPE_NAME,
           comm_mode_name(comm_mode), kernel_mode_name(kernel_mode),
           rsqrt_mode_name(rsqrt_mode),
           accumulator_mode_name(accumulator_mode));
    printf("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
           global_n, nsteps, (double)dt, (double)eps, (double)g,
           (double)mass);
    printf("# step time kinetic potential total rel_energy_drift\n");
    printf("%zu %.17g %.17g %.17g %.17g %.17g\n", (size_t)0u, 0.0,
           (double)kinetic0, (double)potential0, (double)energy0, 0.0);
  }

  /* KDK needs an initial force field before the first half-kick. */
  t0 = seconds();
  compute_accelerations_ring(&local, global_n, g, eps, rank,
                             nranks, comm_mode, kernel_mode, rsqrt_mode,
                             accumulator_mode,
                             &newton_workspace,
                             &timing.comm_wait,
                             MPI_COMM_WORLD);
  timing.force += seconds() - t0;

  for (size_t step = 1u; step <= nsteps; ++step)
  {
    /* Kick-Drift-Kick: half velocity update, full position update, force at
     * the new positions, then the second half velocity update. */
    t0 = seconds();
    kick(&local, (dtype)0.5 * dt);
    timing.kick += seconds() - t0;

    t0 = seconds();
    drift(&local, dt);
    timing.drift += seconds() - t0;

    t0 = seconds();
    compute_accelerations_ring(&local, global_n, g, eps,
                               rank, nranks, comm_mode, kernel_mode,
                               rsqrt_mode, accumulator_mode,
                               &newton_workspace,
                               &timing.comm_wait,
                               MPI_COMM_WORLD);
    timing.force += seconds() - t0;

    t0 = seconds();
    kick(&local, (dtype)0.5 * dt);
    timing.kick += seconds() - t0;

    /* Diagnostic energy is sampled periodically and always at the final step.
     * This balances correctness evidence against diagnostic overhead. */
    if (((step % energy_every) == 0u) || (step == nsteps))
    {
      dtype kinetic, potential, energy;
      const double denom = fmax(fabs((double)energy0),
                                (double)DTYPE_MIN_NORMAL);
      double rel;
      t0 = seconds();
      energy = total_energy_ring(&local, global_n, local_start, g, eps,
                                 rank, nranks, MPI_COMM_WORLD, &kinetic,
                                 &potential);
      timing.energy += seconds() - t0;
      rel = fabs((double)(energy - energy0)) / denom;
      if (rel > max_rel_drift)
        max_rel_drift = rel;
      if ((rank == 0) && !quiet)
        printf("%zu %.17g %.17g %.17g %.17g %.17g\n", step,
               (double)step * (double)dt, (double)kinetic,
               (double)potential, (double)energy, rel);
    }
  }

  if (output_path != NULL)
  {
    t0 = seconds();
    write_output_root(output_path, &local, global_n, rank, nranks,
                      MPI_COMM_WORLD);
    timing.io += seconds() - t0;
  }

  timing.total = seconds() - timing.total;
  t1 = timing.total;
  /* A parallel phase completes when its slowest rank completes. MPI_MAX gives
   * the wall-clock cost and exposes rank imbalance or local delays. */
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &timing.drift, &timing.drift, 1,
             MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &timing.force, &timing.force, 1,
             MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &timing.comm_wait, &timing.comm_wait,
             1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &timing.kick, &timing.kick, 1,
             MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &timing.energy, &timing.energy, 1,
             MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &timing.io, &timing.io, 1,
             MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &t1, &timing.total, 1,
             MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    const size_t force_evals = nsteps + 1u;
    const double interactions = (double)force_evals * (double)global_n *
                                (double)(global_n - 1u);
    const double ginteractions = interactions / (timing.force * 1.0e9);
    printf("# final: N=%zu steps=%zu ranks=%d threads_per_rank=%d "
           "integrator=%s comm=%s kernel=%s rsqrt=%s arithmetic_dtype=%s "
           "accumulators=%s "
           "max_relative_energy_drift=%.17g "
           "tolerance=%.17g status=%s\n",
           global_n, nsteps, nranks, omp_get_max_threads(),
           "kdk", comm_mode_name(comm_mode),
           kernel_mode_name(kernel_mode), rsqrt_mode_name(rsqrt_mode),
           DTYPE_NAME, accumulator_mode_name(accumulator_mode),
           max_rel_drift, (double)energy_tol,
           (max_rel_drift <= (double)energy_tol) ? "OK" : "WARNING");
    printf("# timing_max_seconds total=%.6f io=%.6f drift=%.6f "
           "force=%.6f comm_wait=%.6f kick=%.6f energy=%.6f\n",
           timing.total, timing.io, timing.drift, timing.force,
           timing.comm_wait, timing.kick, timing.energy);
    printf("# kernel_rate pair_interactions_per_second=%.6e "
           "gpair_interactions_per_second=%.6f\n",
           interactions / timing.force, ginteractions);
  }

  workspace_free(&newton_workspace);
  particles_free(&local);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
