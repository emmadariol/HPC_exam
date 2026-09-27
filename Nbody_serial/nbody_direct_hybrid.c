// Direct N-body solver: MPI ring of source blocks + OpenMP force loop, KDK leapfrog, energy check.

#include "nbody_common.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <omp.h>

#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)  // AVX-512 intrinsics only when the target CPU has them
#include <immintrin.h>
#endif

typedef struct particles_s  // particles in SoA layout: one array per component
{
  size_t n;  // number of particles owned by this rank
  dtype mass;  // same mass for every particle
  dtype *x, *y, *z;  // positions
  dtype *vx, *vy, *vz;  // velocities
  dtype *ax, *ay, *az;  // accelerations
} particles_t;

typedef struct timings_s  // time spent in each phase (seconds)
{
  double drift;
  double force;
  double comm_wait;  // time spent waiting for ring messages (part of force)
  double kick;
  double energy;
  double io;
  double total;
} timings_t;

typedef struct thread_workspace_s  // private force buffers for the Newton kernel
{
  size_t n;
  int nthreads;
  dtype *tax;  // one x-acceleration slice per thread
  dtype *tay;
  dtype *taz;
} thread_workspace_t;

typedef enum comm_mode_e
{
  COMM_SENDRECV,  // blocking ring exchange
  COMM_OVERLAP  // non-blocking ring exchange overlapped with compute
} comm_mode_t;

typedef enum kernel_mode_e
{
  KERNEL_DIRECT,  // every ordered pair, no write conflicts
  KERNEL_NEWTON  // each pair once (Newton's third law), one rank only
} kernel_mode_t;

typedef enum rsqrt_mode_e
{
  RSQRT_EXACT,  // 1/sqrt computed exactly
  RSQRT_APPROX,  // approximate 1/sqrt + two Newton-Raphson steps
  RSQRT_APPROX_NR1  // approximate 1/sqrt + one Newton-Raphson step
} rsqrt_mode_t;

typedef enum accumulator_mode_e  // number of independent partial sums in the force loop
{
  ACCUMULATORS_ONE = 1,
  ACCUMULATORS_TWO = 2,
  ACCUMULATORS_FOUR = 4,
  ACCUMULATORS_EIGHT = 8
} accumulator_mode_t;

static size_t parse_size(const char *text, const char *name);
static dtype parse_dtype(const char *text, const char *name);
static comm_mode_t parse_comm_mode(const char *text);
static kernel_mode_t parse_kernel_mode(const char *text);
static rsqrt_mode_t parse_rsqrt_mode(const char *text);
static accumulator_mode_t parse_accumulator_mode(const char *text);
static const char *option_value(int *i, int argc, char **argv,
                                const char *key);
static void print_usage(const char *program);

static void die(const char *fmt, ...)  // print an error and stop all ranks
{
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fputc('\n', stderr);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static MPI_Datatype mpi_dtype(void)  // MPI type matching dtype
{
#if defined(NBODY_USE_FLOAT)
  return MPI_FLOAT;
#else
  return MPI_DOUBLE;
#endif
}

static double seconds(void)  // wall-clock time in seconds
{
  return MPI_Wtime();
}

static inline dtype refine_rsqrt_newton(dtype r2, dtype x)  // one Newton-Raphson step for y = 1/sqrt(r2)
{
  const dtype half = (dtype)0.5;
  const dtype three_halves = (dtype)1.5;
  return x * (three_halves - half * r2 * x * x);  // y_new = y * (3 - r2*y*y) / 2
}

static inline dtype invsqrt_force(dtype r2, rsqrt_mode_t mode)  // 1/sqrt(r2) for the scalar force loop
{
  if (mode == RSQRT_EXACT)
    return (dtype)1.0 / dtype_sqrt(r2);  // exact path

  {
    dtype x = (dtype)(1.0f / sqrtf((float)r2));  // float estimate used as starting guess
    x = refine_rsqrt_newton(r2, x);  // first refinement
    if (mode != RSQRT_APPROX_NR1)
      x = refine_rsqrt_newton(r2, x);  // second refinement (approx2 only)
    return x;
  }
}

static void *checked_aligned_alloc(size_t nbytes)  // allocation aligned to NBODY_ALIGNMENT bytes
{
  const size_t alignment = NBODY_ALIGNMENT;
  const size_t padded = ((nbytes + alignment - 1u) / alignment) * alignment;  // round size up to a multiple of the alignment
  void *ptr;
  if (nbytes == 0u)
    return NULL;
  ptr = aligned_alloc(alignment, padded);  // aligned memory helps vector loads
  if (ptr == NULL)
    die("aligned_alloc failed for %zu bytes", padded);
  return ptr;
}

static void checked_fwrite(const void *ptr, size_t size, size_t nmemb,  // fwrite that stops the program on error
                           FILE *fp, const char *path, const char *what)
{
  if (fwrite(ptr, size, nmemb, fp) != nmemb)
    die("failed writing %s to '%s'", what, path);
}

static void particles_init_empty(particles_t *p)  // reset all pointers and counters
{
  memset(p, 0, sizeof(*p));
  p->mass = (dtype)1.0;
}

static void particles_allocate(particles_t *p, size_t n, dtype mass)  // allocate the nine SoA arrays of n particles
{
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

static void workspace_allocate(thread_workspace_t *ws, size_t n, int nthreads)  // T x n buffers, one slice per thread
{
  size_t items, bytes;

  if (nthreads <= 0)
    die("invalid OpenMP thread count for Newton workspace");
  if (n > SIZE_MAX / (size_t)nthreads)  // check T*n does not overflow
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
}

static void workspace_free(thread_workspace_t *ws)
{
  free(ws->tax);
  free(ws->tay);
  free(ws->taz);
  workspace_init_empty(ws);
}

static void block_bounds(size_t n, int rank, int nranks,  // equal blocks; the first n % P ranks get one extra particle
                         size_t *start, size_t *count)
{
  const size_t base = n / (size_t)nranks;  // particles per rank
  const size_t rem = n % (size_t)nranks;  // leftover particles
  if (count != NULL)
    *count = base + ((size_t)rank < rem ? 1u : 0u);  // block size of this rank
  if (start != NULL)
    *start = (size_t)rank * base + ((size_t)rank < rem ? (size_t)rank : rem);  // first global index of this rank
}

static size_t max_block_count(size_t n, int nranks)  // largest block size (rank 0)
{
  size_t count;
  block_bounds(n, 0, nranks, NULL, &count);
  return count;
}

// ===========================================================================
// [MPI-IO] PARALLEL INPUT
// Every rank reads only its own block of the shared file (collective read).
// ===========================================================================
static void read_local_particles(const char *path, dtype mass, int rank,  // each rank reads only its own block of the input file
                                 int nranks, particles_t *local,
                                 size_t *global_n, size_t *local_start,
                                 MPI_Comm comm)
{
  MPI_File fh;
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t i, n, local_n, nrecords;
  MPI_Offset offset;
  float *records = NULL;
  float dummy_record = 0.0f;

  if (MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) !=  // [MPI-IO] collective open of the shared file
      MPI_SUCCESS)
    die("MPI_File_open failed for input '%s'", path);

  if (MPI_File_read_all(fh, magic, NBODY_BINARY_MAGIC_SIZE, MPI_BYTE,  // [MPI-IO] header: 8-byte magic string
                        MPI_STATUS_IGNORE) != MPI_SUCCESS)
    die("MPI_File_read_all failed for magic in '%s'", path);
  if (memcmp(magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die("invalid input magic in '%s'", path);

  if (MPI_File_read_all(fh, &n64, 1, MPI_UINT64_T, MPI_STATUS_IGNORE) !=  // [MPI-IO] header: particle count
      MPI_SUCCESS)
    die("MPI_File_read_all failed for particle count in '%s'", path);
  if ((uint64_t)(size_t)n64 != n64)
    die("input particle count is too large");

  n = (size_t)n64;
  block_bounds(n, rank, nranks, local_start, &local_n);  // which particles belong to this rank
  particles_allocate(local, local_n, mass);

  nrecords = local_n * NBODY_BINARY_COMPONENTS;  // six floats per particle
  if (nrecords > (size_t)INT_MAX)
    die("MPI-IO read count exceeds INT_MAX");

  records = checked_aligned_alloc(nrecords * sizeof(float));
  offset = (MPI_Offset)NBODY_BINARY_MAGIC_SIZE +  // byte offset of this rank's first record
           (MPI_Offset)sizeof(uint64_t) +
           (MPI_Offset)(*local_start) *
             (MPI_Offset)NBODY_BINARY_COMPONENTS * (MPI_Offset)sizeof(float);

  if (MPI_File_read_at_all(fh, offset, nrecords > 0u ? records : &dummy_record,  // [MPI-IO] collective read of this rank's block
                           (int)nrecords, MPI_FLOAT, MPI_STATUS_IGNORE) !=
      MPI_SUCCESS)
    die("MPI_File_read_at_all failed for particle records in '%s'", path);

  for (i = 0u; i < local_n; ++i)  // unpack float records into dtype SoA arrays
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

static float to_float_checked(dtype value, const char *path)  // float conversion with overflow check
{
  if (!dtype_isfinite(value) || (fabs((double)value) > (double)FLT_MAX))
    die("cannot write non-finite or overflowing value to '%s'", path);
  return (float)value;
}

// ===========================================================================
// [MPI] OUTPUT
// The final state is gathered on rank 0 with MPI_Gatherv and written there.
// ===========================================================================
static void write_output_root(const char *path, const particles_t *local,  // gather all particles on rank 0 and write the output file
                              size_t global_n, int rank, int nranks,
                              MPI_Comm comm)
{
  int *counts = NULL, *displs = NULL;
  dtype *x = NULL, *y = NULL, *z = NULL, *vx = NULL, *vy = NULL, *vz = NULL;
  size_t r;
  MPI_Datatype dt = mpi_dtype();

  counts = malloc((size_t)nranks * sizeof(*counts));
  displs = malloc((size_t)nranks * sizeof(*displs));
  if ((counts == NULL) || (displs == NULL))
    die("cannot allocate gather metadata");
  for (r = 0u; r < (size_t)nranks; ++r)  // block sizes and offsets of every rank
  {
    size_t start, count;
    block_bounds(global_n, (int)r, nranks, &start, &count);
    if ((count > (size_t)INT_MAX) || (start > (size_t)INT_MAX))
      die("MPI gather count exceeds INT_MAX");
    counts[r] = (int)count;
    displs[r] = (int)start;
  }

  if (rank == 0)  // rank 0 receives and writes
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

    MPI_Gatherv(local->x, (int)local->n, dt, x, counts, displs, dt, 0, comm);  // [MPI] gather each component on rank 0
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
    for (i = 0u; i < global_n; ++i)  // write one float record per particle
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
    MPI_Gatherv(local->x, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);  // [MPI] other ranks only send
    MPI_Gatherv(local->y, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->z, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vx, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vy, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
    MPI_Gatherv(local->vz, (int)local->n, dt, NULL, counts, displs, dt, 0, comm);
  }

  free(counts);
  free(displs);
}

// ===========================================================================
// [OpenMP] DRIFT AND KICK
// O(N) loops: the particles are split among the threads.
// ===========================================================================
static void drift(particles_t *p, dtype dt)  // drift: x += dt * v
{
  size_t i;
#pragma omp parallel for schedule(static)  // [OpenMP] particles are independent: split them among threads
  for (i = 0u; i < p->n; ++i)
  {
    p->x[i] += dt * p->vx[i];
    p->y[i] += dt * p->vy[i];
    p->z[i] += dt * p->vz[i];
  }
}

static void kick(particles_t *p, dtype dt)  // kick: v += dt * a
{
  size_t i;
#pragma omp parallel for schedule(static)  // [OpenMP]
  for (i = 0u; i < p->n; ++i)
  {
    p->vx[i] += dt * p->ax[i];
    p->vy[i] += dt * p->ay[i];
    p->vz[i] += dt * p->az[i];
  }
}

// ===========================================================================
// FORCE KERNEL: [OpenMP] + [SIMD] + [UNROLL]
// Threads split the targets; the source loop is unrolled into 1, 2, 4 or 8 independent partial sums.
// ===========================================================================
static void accumulate_sources_scalar_chains(const particles_t *home,  // scalar force loop with 1, 2, 4 or 8 independent partial sums
                                             const dtype *restrict sx,
                                             const dtype *restrict sy,
                                             const dtype *restrict sz,
                                             size_t source_n, dtype g,
                                             dtype mass, dtype eps2,
                                             rsqrt_mode_t rsqrt_mode,
                                             size_t chains)
{
  size_t i;
  const dtype gm = g * mass;  // G*m, same for every pair

#pragma omp parallel for schedule(static)  // [OpenMP] each thread owns different targets: no atomics needed
  for (i = 0u; i < home->n; ++i)  // loop over home (target) particles
  {
    const dtype xi = home->x[i];  // target position, kept in registers
    const dtype yi = home->y[i];
    const dtype zi = home->z[i];

    dtype ax = (dtype)0.0, ay = (dtype)0.0, az = (dtype)0.0;  // final sums; also used by the tail loop

    size_t j = 0u;  // first source not yet processed

    switch (chains)  // chains = 1 has no case: everything is done by the tail loop
    {
    // ---- [UNROLL x2] two sources per iteration --------------------------------
    case 2u:  // two independent partial sums
    {
      const size_t source_n_unrolled = source_n - (source_n % 2u);  // largest multiple of 2 <= source_n
      dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;  // partial sums of chain 0
      dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;  // partial sums of chain 1

#pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1)  // [SIMD] ask the compiler to vectorise, sums are reductions

      for (j = 0u; j < source_n_unrolled; j += 2u)  // [UNROLL] two sources per iteration, one per chain
      {
        const dtype dx0 = sx[j] - xi;  // d = r_source - r_target
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;  // softened distance^2: q = |d|^2 + eps^2
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);  // 1/sqrt(q)
        const dtype s0 = gm * invr0 * invr0 * invr0;  // G*m / q^(3/2)

        ax0 += dx0 * s0;  // add to chain 0
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        const dtype dx1 = sx[j + 1u] - xi;  // chain 1: same formula for source j+1
        const dtype dy1 = sy[j + 1u] - yi;
        const dtype dz1 = sz[j + 1u] - zi;
        const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
        const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
        const dtype s1 = gm * invr1 * invr1 * invr1;
        ax1 += dx1 * s1;  // add to chain 1
        ay1 += dy1 * s1;
        az1 += dz1 * s1;
      }

      j = source_n_unrolled;  // tail loop starts after the unrolled part
      ax += ax0 + ax1;  // combine the partial sums
      ay += ay0 + ay1;
      az += az0 + az1;
      break;
    }

    // ---- [UNROLL x4] four sources per iteration -------------------------------
    case 4u:  // four independent partial sums
    {
      const size_t source_n_unrolled = source_n - (source_n % 4u);  // largest multiple of 4 <= source_n
      dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;
      dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;
      dtype ax2 = (dtype)0.0, ay2 = (dtype)0.0, az2 = (dtype)0.0;
      dtype ax3 = (dtype)0.0, ay3 = (dtype)0.0, az3 = (dtype)0.0;

#pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1) \
    reduction(+ : ax2, ay2, az2) reduction(+ : ax3, ay3, az3)  // [SIMD] vectorise, one reduction per chain
      for (j = 0u; j < source_n_unrolled; j += 4u)  // [UNROLL] four sources per iteration, one per chain
      {
        const dtype dx0 = sx[j] - xi;  // chain 0: source j
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;  // q = |d|^2 + eps^2
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);  // 1/sqrt(q)
        const dtype s0 = gm * invr0 * invr0 * invr0;  // G*m / q^(3/2)
        ax0 += dx0 * s0;
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        const dtype dx1 = sx[j + 1u] - xi;  // chain 1: source j+1
        const dtype dy1 = sy[j + 1u] - yi;
        const dtype dz1 = sz[j + 1u] - zi;
        const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
        const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
        const dtype s1 = gm * invr1 * invr1 * invr1;
        ax1 += dx1 * s1;
        ay1 += dy1 * s1;
        az1 += dz1 * s1;

        const dtype dx2 = sx[j + 2u] - xi;  // chain 2: source j+2
        const dtype dy2 = sy[j + 2u] - yi;
        const dtype dz2 = sz[j + 2u] - zi;
        const dtype r2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
        const dtype invr2 = invsqrt_force(r2_2, rsqrt_mode);
        const dtype s2 = gm * invr2 * invr2 * invr2;
        ax2 += dx2 * s2;
        ay2 += dy2 * s2;
        az2 += dz2 * s2;

        const dtype dx3 = sx[j + 3u] - xi;  // chain 3: source j+3
        const dtype dy3 = sy[j + 3u] - yi;
        const dtype dz3 = sz[j + 3u] - zi;
        const dtype r2_3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
        const dtype invr3 = invsqrt_force(r2_3, rsqrt_mode);
        const dtype s3 = gm * invr3 * invr3 * invr3;
        ax3 += dx3 * s3;
        ay3 += dy3 * s3;
        az3 += dz3 * s3;
      }

      j = source_n_unrolled;  // tail starts here
      ax += ax0 + ax1 + ax2 + ax3;  // combine the four chains
      ay += ay0 + ay1 + ay2 + ay3;
      az += az0 + az1 + az2 + az3;
      break;
    }

    // ---- [UNROLL x8] eight sources per iteration ------------------------------
    case 8u:  // eight independent partial sums
    {
      const size_t source_n_unrolled = source_n - (source_n % 8u);  // largest multiple of 8 <= source_n

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
    reduction(+ : ax6, ay6, az6) reduction(+ : ax7, ay7, az7)  // [SIMD] vectorise, one reduction per chain

      for (j = 0u; j < source_n_unrolled; j += 8u)  // [UNROLL] eight sources per iteration, one per chain
      {
        const dtype dx0 = sx[j] - xi;  // chain 0: source j
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;  // q = |d|^2 + eps^2
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);  // 1/sqrt(q)
        const dtype s0 = gm * invr0 * invr0 * invr0;  // G*m / q^(3/2)
        ax0 += dx0 * s0;
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        const dtype dx1 = sx[j + 1u] - xi;  // chain 1: source j+1
        const dtype dy1 = sy[j + 1u] - yi;
        const dtype dz1 = sz[j + 1u] - zi;
        const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
        const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
        const dtype s1 = gm * invr1 * invr1 * invr1;
        ax1 += dx1 * s1;
        ay1 += dy1 * s1;
        az1 += dz1 * s1;

        const dtype dx2 = sx[j + 2u] - xi;  // chain 2: source j+2
        const dtype dy2 = sy[j + 2u] - yi;
        const dtype dz2 = sz[j + 2u] - zi;
        const dtype r2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
        const dtype invr2 = invsqrt_force(r2_2, rsqrt_mode);
        const dtype s2 = gm * invr2 * invr2 * invr2;
        ax2 += dx2 * s2;
        ay2 += dy2 * s2;
        az2 += dz2 * s2;

        const dtype dx3 = sx[j + 3u] - xi;  // chain 3: source j+3
        const dtype dy3 = sy[j + 3u] - yi;
        const dtype dz3 = sz[j + 3u] - zi;
        const dtype r2_3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
        const dtype invr3 = invsqrt_force(r2_3, rsqrt_mode);
        const dtype s3 = gm * invr3 * invr3 * invr3;
        ax3 += dx3 * s3;
        ay3 += dy3 * s3;
        az3 += dz3 * s3;

        const dtype dx4 = sx[j + 4u] - xi;  // chain 4: source j+4
        const dtype dy4 = sy[j + 4u] - yi;
        const dtype dz4 = sz[j + 4u] - zi;
        const dtype r2_4 = dx4 * dx4 + dy4 * dy4 + dz4 * dz4 + eps2;
        const dtype invr4 = invsqrt_force(r2_4, rsqrt_mode);
        const dtype s4 = gm * invr4 * invr4 * invr4;
        ax4 += dx4 * s4;
        ay4 += dy4 * s4;
        az4 += dz4 * s4;

        const dtype dx5 = sx[j + 5u] - xi;  // chain 5: source j+5
        const dtype dy5 = sy[j + 5u] - yi;
        const dtype dz5 = sz[j + 5u] - zi;
        const dtype r2_5 = dx5 * dx5 + dy5 * dy5 + dz5 * dz5 + eps2;
        const dtype invr5 = invsqrt_force(r2_5, rsqrt_mode);
        const dtype s5 = gm * invr5 * invr5 * invr5;
        ax5 += dx5 * s5;
        ay5 += dy5 * s5;
        az5 += dz5 * s5;

        const dtype dx6 = sx[j + 6u] - xi;  // chain 6: source j+6
        const dtype dy6 = sy[j + 6u] - yi;
        const dtype dz6 = sz[j + 6u] - zi;
        const dtype r2_6 = dx6 * dx6 + dy6 * dy6 + dz6 * dz6 + eps2;
        const dtype invr6 = invsqrt_force(r2_6, rsqrt_mode);
        const dtype s6 = gm * invr6 * invr6 * invr6;
        ax6 += dx6 * s6;
        ay6 += dy6 * s6;
        az6 += dz6 * s6;

        const dtype dx7 = sx[j + 7u] - xi;  // chain 7: source j+7
        const dtype dy7 = sy[j + 7u] - yi;
        const dtype dz7 = sz[j + 7u] - zi;
        const dtype r2_7 = dx7 * dx7 + dy7 * dy7 + dz7 * dz7 + eps2;
        const dtype invr7 = invsqrt_force(r2_7, rsqrt_mode);
        const dtype s7 = gm * invr7 * invr7 * invr7;
        ax7 += dx7 * s7;
        ay7 += dy7 * s7;
        az7 += dz7 * s7;
      }

      j = source_n_unrolled;  // tail starts here
      ax += ax0 + ax1 + ax2 + ax3 + ax4 + ax5 + ax6 + ax7;  // combine the eight chains
      ay += ay0 + ay1 + ay2 + ay3 + ay4 + ay5 + ay6 + ay7;
      az += az0 + az1 + az2 + az3 + az4 + az5 + az6 + az7;
      break;
    }
    }

    // ---- tail loop: leftover sources, one at a time ---------------------------
    const size_t tail_start = j;  // remaining sources (all of them when chains = 1)
#pragma omp simd reduction(+ : ax, ay, az)  // [SIMD] single-chain vectorised loop
    for (j = tail_start; j < source_n; ++j)
    {
      const dtype dx = sx[j] - xi;  // d = r_source - r_target
      const dtype dy = sy[j] - yi;
      const dtype dz = sz[j] - zi;

      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;  // q = |d|^2 + eps^2 (self-pair gives d = 0, so zero force)
      const dtype invr = invsqrt_force(r2, rsqrt_mode);  // 1/sqrt(q)
      const dtype s = gm * invr * invr * invr;  // G*m / q^(3/2)

      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }

    home->ax[i] += ax;  // add this source block to the target acceleration
    home->ay[i] += ay;
    home->az[i] += az;
  }
}

// ===========================================================================
// [AVX-512] HAND-WRITTEN VECTOR KERNEL
// Used only for approximate rsqrt: 8 sources per instruction, FMA, rsqrt14 + Newton-Raphson.
// ===========================================================================
#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)  // compiled only when the CPU has AVX-512

static inline __m512d refine_rsqrt_newton_pd(__m512d r2, __m512d x)  // Newton-Raphson step on 8 lanes
{
  const __m512d half = _mm512_set1_pd(0.5);
  const __m512d three = _mm512_set1_pd(3.0);

  return _mm512_mul_pd(_mm512_mul_pd(half, x),  // [AVX-512] y * (3 - r2*y*y) / 2
                       _mm512_sub_pd(three, _mm512_mul_pd(r2, _mm512_mul_pd(x, x))));
}

static inline double hsum_pd(__m512d v)  // sum of the 8 lanes of a vector
{
  double tmp[8];
  _mm512_storeu_pd(tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3] +
         tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

static void accumulate_sources_rsqrt14_pd(const particles_t *home,  // force loop, 8 sources at a time with AVX-512
                                          const double *restrict sx,
                                          const double *restrict sy,
                                          const double *restrict sz,
                                          size_t source_n, double g,
                                          double mass, double eps2,
                                          rsqrt_mode_t rsqrt_mode)
{
  const __m512d eps2_v = _mm512_set1_pd(eps2);  // [AVX-512] eps^2 in all 8 lanes
  const __m512d gm_v = _mm512_set1_pd(g * mass);  // [AVX-512] G*m in all 8 lanes
  size_t i;

#pragma omp parallel for schedule(static)  // [OpenMP] each thread owns different targets
  for (i = 0u; i < home->n; ++i)
  {
    const __m512d xi = _mm512_set1_pd(home->x[i]);  // [AVX-512] target position copied to all 8 lanes
    const __m512d yi = _mm512_set1_pd(home->y[i]);
    const __m512d zi = _mm512_set1_pd(home->z[i]);

    __m512d ax_v = _mm512_setzero_pd();  // [AVX-512] vector accumulators, one per component
    __m512d ay_v = _mm512_setzero_pd();
    __m512d az_v = _mm512_setzero_pd();
    size_t j = 0u;

    for (; j + 7u < source_n; j += 8u)  // [AVX-512] 8 sources per vector instruction
    {
      const __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(sx + j), xi);  // [AVX-512] d = r_source - r_target for 8 sources
      const __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(sy + j), yi);
      const __m512d dz = _mm512_sub_pd(_mm512_loadu_pd(sz + j), zi);

      __m512d r2 = _mm512_fmadd_pd(dx, dx, eps2_v);  // [AVX-512] r2 = dx*dx + eps^2 (FMA)

      r2 = _mm512_fmadd_pd(dy, dy, r2);  // [AVX-512] r2 += dy*dy
      r2 = _mm512_fmadd_pd(dz, dz, r2);  // [AVX-512] r2 += dz*dz

      __m512d invr = _mm512_rsqrt14_pd(r2);  // [AVX-512] hardware 1/sqrt estimate (~14 bits)

      invr = refine_rsqrt_newton_pd(r2, invr);  // first refinement
      if (rsqrt_mode != RSQRT_APPROX_NR1)
        invr = refine_rsqrt_newton_pd(r2, invr);  // second refinement (approx2 only)

      const __m512d invr2 = _mm512_mul_pd(invr, invr);  // [AVX-512] 1/r^2
      const __m512d s = _mm512_mul_pd(gm_v, _mm512_mul_pd(invr2, invr));  // [AVX-512] G*m / r^3

      ax_v = _mm512_fmadd_pd(dx, s, ax_v);  // [AVX-512] a += d * s (FMA)
      ay_v = _mm512_fmadd_pd(dy, s, ay_v);
      az_v = _mm512_fmadd_pd(dz, s, az_v);
    }

    double ax = hsum_pd(ax_v);  // reduce vectors to scalars
    double ay = hsum_pd(ay_v);
    double az = hsum_pd(az_v);

    for (; j < source_n; ++j)  // scalar tail for the last < 8 sources
    {
      const double dx = sx[j] - home->x[i];
      const double dy = sy[j] - home->y[i];
      const double dz = sz[j] - home->z[i];

      const double r2 = dx * dx + dy * dy + dz * dz + eps2;
      const double invr = invsqrt_force(r2, rsqrt_mode);  // same approximation as the vector path
      const double s = g * mass * invr * invr * invr;
      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }

    home->ax[i] += ax;  // add to the target acceleration
    home->ay[i] += ay;
    home->az[i] += az;
  }
}
#endif

// ===========================================================================
// KERNEL DISPATCH
// Approximate rsqrt -> AVX-512 kernel; exact rsqrt -> scalar kernel with unrolling.
// ===========================================================================
static void accumulate_sources(const particles_t *home,  // dispatch: AVX-512 routine or scalar chains
                               const dtype *restrict sx,
                               const dtype *restrict sy,
                               const dtype *restrict sz,
                               size_t source_n,
                               dtype g, dtype mass, dtype eps,
                               rsqrt_mode_t rsqrt_mode,
                               accumulator_mode_t accumulator_mode)
{
  const dtype eps2 = eps * eps;  // softening squared

#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)
  if (rsqrt_mode != RSQRT_EXACT)  // approximate modes use the AVX-512 routine
  {
    accumulate_sources_rsqrt14_pd(home, sx, sy, sz, source_n, g, mass, eps2, rsqrt_mode);
    return;
  }
#endif

  accumulate_sources_scalar_chains(home, sx, sy, sz, source_n, g, mass, eps2,  // exact mode (or no AVX-512): scalar chains
                                   rsqrt_mode, (size_t)accumulator_mode);
}

// ===========================================================================
// NEWTON'S THIRD LAW KERNEL: [OpenMP] with private buffers
// Each pair is computed once; threads write to private copies, reduced at the end (no atomics).
// ===========================================================================
static void compute_accelerations_newton_private(particles_t *local, dtype g,  // single-rank kernel using Newton's third law and private buffers
                                                 dtype eps,
                                                 rsqrt_mode_t rsqrt_mode,
                                                 thread_workspace_t *workspace)
{
  const size_t n = local->n;
  const dtype eps2 = eps * eps;
  int nthreads;
  size_t i;

  nthreads = workspace->nthreads;

  memset(workspace->tax, 0, (size_t)nthreads * n * sizeof(dtype));  // clear every thread's private buffers
  memset(workspace->tay, 0, (size_t)nthreads * n * sizeof(dtype));
  memset(workspace->taz, 0, (size_t)nthreads * n * sizeof(dtype));

#pragma omp parallel  // [OpenMP] one team; each thread writes only its own slice
  {
    const int tid = omp_get_thread_num();
    dtype *ax = workspace->tax + (size_t)tid * n;  // this thread's slice of the private buffers
    dtype *ay = workspace->tay + (size_t)tid * n;
    dtype *az = workspace->taz + (size_t)tid * n;

    size_t ii;

#pragma omp for schedule(static)  // [OpenMP] static split of rows: row i has n-1-i pairs, so work is unbalanced

    for (ii = 0u; ii < n; ++ii)
    {
      const dtype xi = local->x[ii];
      const dtype yi = local->y[ii];
      const dtype zi = local->z[ii];

      size_t j;
      for (j = ii + 1u; j < n; ++j)  // only j > i: each pair once
      {
        const dtype dx = local->x[j] - xi;  // d = r_j - r_i
        const dtype dy = local->y[j] - yi;
        const dtype dz = local->z[j] - zi;
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;  // q = |d|^2 + eps^2
        const dtype invr = invsqrt_force(r2, rsqrt_mode);  // 1/sqrt(q)
        const dtype s = g * local->mass * invr * invr * invr;  // G*m / q^(3/2)
        const dtype fx = dx * s;  // force on i from j
        const dtype fy = dy * s;
        const dtype fz = dz * s;
        ax[ii] += fx;  // i gets +f
        ay[ii] += fy;
        az[ii] += fz;
        ax[j] -= fx;  // j gets -f (third law)
        ay[j] -= fy;
        az[j] -= fz;
      }
    }
  }

#pragma omp parallel for schedule(static)  // [OpenMP] reduction: sum the private slices of all threads
  for (i = 0u; i < n; ++i)
  {
    dtype ax = (dtype)0.0;
    dtype ay = (dtype)0.0;
    dtype az = (dtype)0.0;
    int t;

    for (t = 0; t < nthreads; ++t)  // add the contribution of every thread
    {
      ax += workspace->tax[(size_t)t * n + i];
      ay += workspace->tay[(size_t)t * n + i];
      az += workspace->taz[(size_t)t * n + i];
    }

    local->ax[i] = ax;  // final acceleration of particle i
    local->ay[i] = ay;
    local->az[i] = az;
  }
}

// ===========================================================================
// [MPI] RING COMMUNICATION
// Blocking exchange (MPI_Sendrecv) and non-blocking exchange (MPI_Irecv / MPI_Isend).
// ===========================================================================
static void exchange_sources_sendrecv(dtype *bx, dtype *by, dtype *bz,  // blocking ring step: send my block, receive the next one
                                      dtype *rx, dtype *ry, dtype *rz,
                                      size_t buf_n, size_t next_n,
                                      int rank, int nranks, int tag_base,
                                      MPI_Datatype dt, MPI_Comm comm)
{
  const int send_to = (rank + 1) % nranks;  // right neighbour in the ring
  const int recv_from = (rank + nranks - 1) % nranks;  // left neighbour in the ring
  MPI_Sendrecv(bx, (int)buf_n, dt, send_to, tag_base + 0,  // [MPI] x positions
               rx, (int)next_n, dt, recv_from, tag_base + 0, comm,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(by, (int)buf_n, dt, send_to, tag_base + 1,  // [MPI] y positions
               ry, (int)next_n, dt, recv_from, tag_base + 1, comm,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(bz, (int)buf_n, dt, send_to, tag_base + 2,  // [MPI] z positions
               rz, (int)next_n, dt, recv_from, tag_base + 2, comm,
               MPI_STATUS_IGNORE);
}

static void post_source_exchange(dtype *bx, dtype *by, dtype *bz,  // non-blocking ring step: only posts the messages
                                 dtype *rx, dtype *ry, dtype *rz,
                                 size_t buf_n, size_t next_n,
                                 int rank, int nranks, int tag_base,
                                 MPI_Datatype dt, MPI_Comm comm,
                                 MPI_Request req[6])
{
  const int send_to = (rank + 1) % nranks;  // right neighbour
  const int recv_from = (rank + nranks - 1) % nranks;  // left neighbour
  MPI_Irecv(rx, (int)next_n, dt, recv_from, tag_base + 0, comm, &req[0]);  // [MPI] receive the next block into rx, ry, rz
  MPI_Irecv(ry, (int)next_n, dt, recv_from, tag_base + 1, comm, &req[1]);
  MPI_Irecv(rz, (int)next_n, dt, recv_from, tag_base + 2, comm, &req[2]);
  MPI_Isend(bx, (int)buf_n, dt, send_to, tag_base + 0, comm, &req[3]);  // [MPI] send the current block
  MPI_Isend(by, (int)buf_n, dt, send_to, tag_base + 1, comm, &req[4]);
  MPI_Isend(bz, (int)buf_n, dt, send_to, tag_base + 2, comm, &req[5]);
}

// ===========================================================================
// [MPI] RING DRIVER
// P steps: compute with the current block, pass it on; optional overlap with non-blocking MPI.
// ===========================================================================
static void compute_accelerations_ring(particles_t *local, size_t global_n,  // accelerations of the home particles over the whole ring
                                       dtype g, dtype eps,
                                       int rank, int nranks, comm_mode_t mode,
                                       kernel_mode_t kernel_mode,
                                       rsqrt_mode_t rsqrt_mode,
                                       accumulator_mode_t accumulator_mode,
                                       thread_workspace_t *newton_workspace,
                                       double *comm_wait,
                                       MPI_Comm comm)
{
  if (kernel_mode == KERNEL_NEWTON)  // Newton kernel: single rank only
  {
    if (nranks != 1)
      die("--kernel newton is implemented for -np 1 only; use --kernel direct for MPI ring runs");
    compute_accelerations_newton_private(local, g, eps, rsqrt_mode,
                                         newton_workspace);
    return;
  }

  const size_t max_n = max_block_count(global_n, nranks);  // size of the largest block
  dtype *bx = checked_aligned_alloc(max_n * sizeof(dtype));  // current source block (b*) ...
  dtype *by = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *bz = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rx = checked_aligned_alloc(max_n * sizeof(dtype));  // ... and incoming block (r*)
  dtype *ry = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rz = checked_aligned_alloc(max_n * sizeof(dtype));
  MPI_Datatype dt = mpi_dtype();
  int owner = rank;  // rank that owns the current source block
  size_t buf_n = local->n;  // number of particles in the current block
  size_t i;

#pragma omp parallel for schedule(static)  // [OpenMP] reset accelerations before accumulating
  for (i = 0u; i < local->n; ++i)
    local->ax[i] = local->ay[i] = local->az[i] = (dtype)0.0;
  memcpy(bx, local->x, local->n * sizeof(dtype));  // start the ring with my own particles
  memcpy(by, local->y, local->n * sizeof(dtype));
  memcpy(bz, local->z, local->n * sizeof(dtype));

  for (int step = 0; step < nranks; ++step)  // P ring steps: every block visits every rank
  {
    if (nranks > 1)
    {
      const int next_owner = (owner + nranks - 1) % nranks;  // block arriving from the left neighbour
      size_t next_n;
      block_bounds(global_n, next_owner, nranks, NULL, &next_n);  // its size
      if (mode == COMM_OVERLAP)  // [MPI overlap] post messages, compute, then wait
      {
        MPI_Request req[6];
        post_source_exchange(bx, by, bz, rx, ry, rz, buf_n, next_n,  // [MPI] Irecv/Isend, returns immediately
                             rank, nranks, 10, dt, comm, req);
        accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,  // compute with the current block while messages travel
                           rsqrt_mode, accumulator_mode);
        {
          const double comm_t0 = seconds();  // only the time spent waiting is counted
          MPI_Waitall(6, req, MPI_STATUSES_IGNORE);  // [MPI] wait for the 6 messages
          *comm_wait += seconds() - comm_t0;
        }
      }
      else  // [MPI blocking] compute, then exchange
      {
        accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,
                           rsqrt_mode, accumulator_mode);
        {
          const double comm_t0 = seconds();
          exchange_sources_sendrecv(bx, by, bz, rx, ry, rz, buf_n, next_n,  // [MPI] Sendrecv; its time counted as waiting
                                    rank, nranks, 10, dt, comm);
          *comm_wait += seconds() - comm_t0;
        }
      }
      memcpy(bx, rx, next_n * sizeof(dtype));  // the received block becomes the current one
      memcpy(by, ry, next_n * sizeof(dtype));
      memcpy(bz, rz, next_n * sizeof(dtype));
      owner = next_owner;
      buf_n = next_n;
    }
    else  // one rank: only its own block
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

// ===========================================================================
// ENERGY CHECK: [OpenMP] reductions + [MPI] Allreduce
// Independent of the force optimisations: exact sqrt and long double sums.
// ===========================================================================
static long double kinetic_energy_local(const particles_t *p)  // kinetic energy of the home particles
{
  long double sum = 0.0L;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)  // [OpenMP] long double sum for accuracy
  for (i = 0u; i < p->n; ++i)
  {
    const long double vx = (long double)p->vx[i];
    const long double vy = (long double)p->vy[i];
    const long double vz = (long double)p->vz[i];
    sum += vx * vx + vy * vy + vz * vz;  // |v|^2
  }
  return 0.5L * (long double)p->mass * sum;  // T = m/2 * sum |v|^2
}

static long double potential_sources(const particles_t *home, size_t home_start,  // potential energy between home particles and one source block
                                     const dtype *restrict sx,
                                     const dtype *restrict sy,
                                     const dtype *restrict sz,
                                     size_t source_start, size_t source_n,
                                     dtype g, dtype eps)
{
  const dtype eps2 = eps * eps;  // same softening as the force
  const long double m2 = (long double)home->mass * (long double)home->mass;
  long double sum = 0.0L;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)  // [OpenMP] long double sum for accuracy
  for (i = 0u; i < home->n; ++i)
  {
    const dtype xi = home->x[i], yi = home->y[i], zi = home->z[i];
    const size_t gi = home_start + i;  // global index of the target
    size_t j;
    for (j = 0u; j < source_n; ++j)
      if (gi < source_start + j)  // count each pair once: global i < global j
      {
        const dtype dx = sx[j] - xi;
        const dtype dy = sy[j] - yi;
        const dtype dz = sz[j] - zi;
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = (dtype)1.0 / dtype_sqrt(r2);  // always the exact square root
        sum -= (long double)g * m2 * (long double)invr;  // U_ij = -G m^2 / sqrt(q)
      }
  }
  return sum;
}

static dtype total_energy_ring(const particles_t *local, size_t global_n,  // total energy E = T + U, computed with its own ring pass
                               size_t local_start, dtype g, dtype eps,
                               int rank, int nranks, MPI_Comm comm,
                               dtype *kinetic, dtype *potential)
{
  const size_t max_n = max_block_count(global_n, nranks);
  dtype *bx = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *by = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *bz = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rx = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *ry = checked_aligned_alloc(max_n * sizeof(dtype));
  dtype *rz = checked_aligned_alloc(max_n * sizeof(dtype));
  int owner = rank;
  size_t buf_start = local_start, buf_n = local->n;  // global start and size of the current block
  long double pot_local = 0.0L, kin_local = kinetic_energy_local(local);  // kinetic part needs no communication
  long double pot_global, kin_global;

  memcpy(bx, local->x, local->n * sizeof(dtype));
  memcpy(by, local->y, local->n * sizeof(dtype));
  memcpy(bz, local->z, local->n * sizeof(dtype));

  for (int step = 0; step < nranks; ++step)  // same ring as the force, blocking exchange
  {
    pot_local += potential_sources(local, local_start, bx, by, bz,
                                   buf_start, buf_n, g, eps);
    if (nranks > 1)
    {
      const int next_owner = (owner + nranks - 1) % nranks;  // block arriving from the left
      size_t next_start, next_n;
      block_bounds(global_n, next_owner, nranks, &next_start, &next_n);
      exchange_sources_sendrecv(bx, by, bz, rx, ry, rz, buf_n, next_n,  // different tags from the force ring
                                rank, nranks, 20, mpi_dtype(), comm);
      memcpy(bx, rx, next_n * sizeof(dtype));
      memcpy(by, ry, next_n * sizeof(dtype));
      memcpy(bz, rz, next_n * sizeof(dtype));
      owner = next_owner;
      buf_start = next_start;
      buf_n = next_n;
    }
  }

  MPI_Allreduce(&kin_local, &kin_global, 1, MPI_LONG_DOUBLE, MPI_SUM, comm);  // [MPI] sum over all ranks, result on every rank
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

// ===========================================================================
// COMMAND-LINE PARSING
// ===========================================================================
static size_t parse_size(const char *text, const char *name)  // string -> size_t, stops on invalid input
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

static dtype parse_dtype(const char *text, const char *name)  // string -> dtype, stops on invalid input
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
  if (strcmp(text, "approx") == 0 || strcmp(text, "approx2") == 0)  // "approx" is an alias for approx2
    return RSQRT_APPROX;
  if (strcmp(text, "approx1") == 0)
    return RSQRT_APPROX_NR1;
  die("invalid --rsqrt '%s' (expected exact, approx1 or approx2; approx aliases approx2)", text);
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

static const char *rsqrt_mode_name(rsqrt_mode_t mode)
{
  return mode == RSQRT_EXACT ? "exact" :
         mode == RSQRT_APPROX_NR1 ? "approx1" : "approx2";
}

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

static const char *option_value(int *i, int argc, char **argv, const char *key)  // accepts both --key value and --key=value
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
          "  --rsqrt exact|approx1|approx2  reciprocal sqrt: exact or 1/2 Newton steps\n"
          "                               approx is an alias for approx2\n"
          "  --accumulators 1|2|4|8    direct-kernel accumulator chains (default: 4)\n"
          "  --quiet                   accepted for benchmark-script compatibility\n"
          "  --help                    show this help message\n",
          program, NBODY_BINARY_VERSION_TEXT);
  fprintf(stderr, "binary format: %s\n", NBODY_BINARY_VERSION_TEXT);
}

// ===========================================================================
// MAIN: MPI setup, KDK leapfrog loop, timing reduction
// ===========================================================================
int main(int argc, char **argv)
{
  const char *input_path = NULL, *output_path = NULL;
  size_t nsteps = 10u, energy_every = 1u, global_n = 0u, local_start = 0u;  // defaults, overwritten by the command line
  dtype dt = (dtype)1.0e-3, eps = (dtype)1.0e-2;
  dtype g = (dtype)1.0, mass = (dtype)1.0, energy_tol = (dtype)1.0e-4;
  comm_mode_t comm_mode = COMM_SENDRECV;
  kernel_mode_t kernel_mode = KERNEL_DIRECT;
  rsqrt_mode_t rsqrt_mode = RSQRT_EXACT;
  accumulator_mode_t accumulator_mode = ACCUMULATORS_FOUR;
  int rank, nranks, provided;
  particles_t local;
  thread_workspace_t newton_workspace;
  timings_t timing = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  dtype kinetic0, potential0, energy0;
  double max_rel_drift = 0.0;
  double t0, t1;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  // [MPI] only the master thread calls MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  particles_init_empty(&local);
  workspace_init_empty(&newton_workspace);

  for (int argi = 1; argi < argc; ++argi)  // command-line parsing
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
    else if (strcmp(argv[argi], "--quiet") == 0)  // accepted and ignored
    {
    }
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
  if ((dt <= (dtype)0.0) || (eps < (dtype)0.0) || (g <= (dtype)0.0) ||  // reject non-physical parameters
      (mass <= (dtype)0.0) || (energy_every == 0u) ||
      (energy_tol <= (dtype)0.0))
    die("invalid non-positive physical or diagnostic parameter");

  timing = (timings_t){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  max_rel_drift = 0.0;

  timing.total = seconds();  // start of the total timer
  t0 = seconds();
  read_local_particles(input_path, mass, rank, nranks, &local,  // each rank reads its own block
                       &global_n, &local_start, MPI_COMM_WORLD);
  timing.io += seconds() - t0;

  if (kernel_mode == KERNEL_NEWTON)  // private buffers only for the Newton kernel
    workspace_allocate(&newton_workspace, local.n, omp_get_max_threads());

  t0 = seconds();
  energy0 = total_energy_ring(&local, global_n, local_start, g, eps, rank,  // reference energy E(0)
                              nranks, MPI_COMM_WORLD, &kinetic0, &potential0);
  timing.energy += seconds() - t0;

  t0 = seconds();
  compute_accelerations_ring(&local, global_n, g, eps, rank,  // initial accelerations a(0)
                             nranks, comm_mode, kernel_mode, rsqrt_mode,
                             accumulator_mode,
                             &newton_workspace,
                             &timing.comm_wait,
                             MPI_COMM_WORLD);
  timing.force += seconds() - t0;

  // ---- KDK leapfrog time loop -----------------------------------------------
  for (size_t step = 1u; step <= nsteps; ++step)  // KDK leapfrog loop
  {
    t0 = seconds();
    kick(&local, (dtype)0.5 * dt);  // kick: v += a * dt/2
    timing.kick += seconds() - t0;

    t0 = seconds();
    drift(&local, dt);  // drift: x += v * dt
    timing.drift += seconds() - t0;

    t0 = seconds();
    compute_accelerations_ring(&local, global_n, g, eps,  // new accelerations at the new positions
                               rank, nranks, comm_mode, kernel_mode,
                               rsqrt_mode, accumulator_mode,
                               &newton_workspace,
                               &timing.comm_wait,
                               MPI_COMM_WORLD);
    timing.force += seconds() - t0;

    t0 = seconds();
    kick(&local, (dtype)0.5 * dt);  // second half kick
    timing.kick += seconds() - t0;

    if (((step % energy_every) == 0u) || (step == nsteps))  // energy check every energy_every steps and at the end
    {
      dtype kinetic, potential, energy;
      const double denom = fmax(fabs((double)energy0),  // avoid division by zero
                                (double)DTYPE_MIN_NORMAL);
      double rel;
      t0 = seconds();
      energy = total_energy_ring(&local, global_n, local_start, g, eps,
                                 rank, nranks, MPI_COMM_WORLD, &kinetic,
                                 &potential);
      timing.energy += seconds() - t0;
      rel = fabs((double)(energy - energy0)) / denom;  // relative energy drift
      if (rel > max_rel_drift)
        max_rel_drift = rel;
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
  // ---- [MPI] timings: keep the slowest rank ---------------------------------
  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &timing.drift, &timing.drift, 1,  // [MPI] keep the slowest rank's time for each phase
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
    const size_t force_evals = nsteps + 1u;  // initial force + one per step
    const double interactions = (double)force_evals * (double)global_n *  // ordered pairs processed
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
           (max_rel_drift <= (double)energy_tol) ? "OK" : "WARNING");  // OK if the drift stays below the tolerance
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
