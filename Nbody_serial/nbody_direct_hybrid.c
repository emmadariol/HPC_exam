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


// SOA: contiguous x/y/z source arrays
typedef struct particles_s
{
  size_t n;
  dtype mass;
  dtype *x, *y, *z;
  dtype *vx, *vy, *vz;
  dtype *ax, *ay, *az;
} particles_t;


// Timings are accumulated by phase and later reduced with MPI_MAX
typedef struct timings_s
{
  double drift;
  double force;
  double comm_wait;
  double kick;
  double energy;
  double io;
  double total;
} timings_t;

// Thread-private force buffers used by the Newton 3 Law: keeping these allocations outside the timed force kernel avoids malloc/calloc noise in the measurement.
typedef struct thread_workspace_s
{
  size_t n;
  int nthreads;
  dtype *tax;
  dtype *tay;
  dtype *taz;
} thread_workspace_t;

// Communication mode
typedef enum comm_mode_e
{
  COMM_SENDRECV, // SENDRECV measures a simple blocking ring exchange
  COMM_OVERLAP // OVERLAP posts non-blocking communication before computing on the current source block
} comm_mode_t;

typedef enum kernel_mode_e
{
  KERNEL_DIRECT, // DIRECT evaluates all source/target pairs
  KERNEL_NEWTON // NEWTON exploits action-reaction symmetry but is implemented only for one rank (cross-rank symmetric updates would require a different communication/reduction scheme)
} kernel_mode_t;

typedef enum rsqrt_mode_e
{
  RSQRT_EXACT, // EXACT uses dtype sqrt
  RSQRT_APPROX, // Backward-compatible two-refinement mode
  RSQRT_APPROX_NR1 // Same seed, one Newton-Raphson refinement
} rsqrt_mode_t;


// Accumulator to exploit more ILP in the direct kernel
typedef enum accumulator_mode_e
{
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

/* Print a fatal error and abort the whole MPI job.
 *
 * MPI programs must not let only one rank exit normally: the other ranks may be
 * blocked in collectives or point-to-point calls.  Using MPI_Abort makes the
 * failure explicit and avoids leaving orphaned ranks in the Slurm allocation.
 */
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

/* Return the MPI datatype that matches the compile-time floating type.
 *
 * The code can be compiled in single or double precision through
 * nbody_common.h.  Every MPI send, receive and reduction must use the matching
 * MPI datatype or the exchanged buffers would be interpreted incorrectly.
 */
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

/* Return a portable wall-clock timestamp for timing benchmark phases. */
static double seconds(void)
{
  /* MPI_Wtime is monotonic and is available on every MPI rank, so elapsed
   * intervals include both computation and MPI waiting. */
  return MPI_Wtime();
}

/* Perform one Newton-Raphson refinement step for an approximate 1/sqrt(r2).
 *
 * This improves a cheap reciprocal-square-root seed before the value is used in
 * the gravitational force expression.  The function is intentionally small and
 * inline because it is called inside the innermost force loop.
 */
static inline dtype refine_rsqrt_newton(dtype r2, dtype x)
{
  const dtype half = (dtype)0.5;
  const dtype three_halves = (dtype)1.5;
  return x * (three_halves - half * r2 * x * x);
}

/* Compute the inverse square root used by the force kernel.
 *
 * RSQRT_EXACT uses the conservative `1/sqrt(r2)` path.  RSQRT_APPROX uses a
 * cheaper low-precision seed and Newton refinement; this is an experimental
 * math path measured by the ablation benchmark, not assumed to be universally
 * faster.
 */
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
    if (mode != RSQRT_APPROX_NR1)
      x = refine_rsqrt_newton(r2, x);
    return x;
  }
}

/* Allocate an aligned buffer and abort with context if allocation fails.
 *
 * Aligned arrays improve SIMD friendliness and make the SoA force loops less
 * sensitive to accidental allocator alignment choices.
 */
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

/* Write a binary block and fail immediately if the full block is not written. */
static void checked_fwrite(const void *ptr, size_t size, size_t nmemb,
                           FILE *fp, const char *path, const char *what)
{
  /* Detect disk-full and other write errors while the operation is identified,
   * instead of silently losing the final state. */
  if (fwrite(ptr, size, nmemb, fp) != nmemb)
    die("failed writing %s to '%s'", what, path);
}

/* Put a particle block in a safe empty state.
 *
 * This makes cleanup idempotent: particles_free() can reset the structure and
 * later calls will see NULL pointers and zero length.
 */
static void particles_init_empty(particles_t *p)
{
  memset(p, 0, sizeof(*p));
  p->mass = (dtype)1.0;
}

/* Allocate a local SoA particle block for one MPI rank. */
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

/* Release all arrays owned by a particle block and reset it to empty. */
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

/* Put the Newton thread-private workspace in an empty state. */
static void workspace_init_empty(thread_workspace_t *ws)
{
  memset(ws, 0, sizeof(*ws));
}

/* Allocate reusable thread-private acceleration buffers for the Newton kernel.
 *
 * The private buffers avoid OpenMP data races when pair (i,j) updates both
 * particles.  They are allocated outside the timed force loop so malloc/calloc
 * overhead does not pollute the ablation timings.
 */
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
}

/* Free the Newton workspace and reset it to the empty state. */
static void workspace_free(thread_workspace_t *ws)
{
  free(ws->tax);
  free(ws->tay);
  free(ws->taz);
  workspace_init_empty(ws);
}

/* Compute the contiguous global-particle block owned by one rank.
 *
 * The decomposition is balanced: ranks differ by at most one particle.  Passing
 * NULL for start or count lets callers request only the quantity they need.
 */
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

/* Return the largest local block size among all ranks.
 *
 * Ring buffers are allocated once with this size so they can hold any source
 * block that circulates through the MPI ring.
 */
static size_t max_block_count(size_t n, int nranks)
{
  size_t count;
  block_bounds(n, 0, nranks, NULL, &count);
  return count;
}

/* Read the input file collectively and load only the rank-local particle block.
 *
 * The binary file contains all particles, but each MPI rank owns a contiguous
 * subset.  MPI-IO lets all ranks read the shared header consistently and then
 * read their own records directly from the correct byte offset.
 */
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

/* Convert a simulation value to the float32 on-disk format safely. */
static float to_float_checked(dtype value, const char *path)
{
  /* The file format is float32 even when the simulation uses double. Prevent a
   * finite in-memory value from becoming Inf during serialization. */
  if (!dtype_isfinite(value) || (fabs((double)value) > (double)FLT_MAX))
    die("cannot write non-finite or overflowing value to '%s'", path);
  return (float)value;
}

/* Gather the distributed final state and write it from rank 0.
 *
 * This is optional and normally disabled for timing runs, because output I/O is
 * not part of the computational kernel being benchmarked.
 */
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

/* Advance particle positions for the drift part of the leapfrog integrator. */
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

/* Advance particle velocities for the kick part of the leapfrog integrator. */
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




/*                  ************************************************          */
/*                                     USE OF FMA IN THE UNROLLING            */
/*                  ************************************************          */

/* Accumulate direct gravitational forces from one source block.
 *
 * This is the SIMD-pragmas implementation: 1, 2, 4 or 8 independent
 * partial sums reduce loop-carried dependencies and give the compiler more ILP
 * to schedule around sqrt/FMA latency.
 */
static void accumulate_sources_scalar_chains(const particles_t *home,
                                             const dtype *restrict sx,
                                             const dtype *restrict sy,
                                             const dtype *restrict sz,
                                             size_t source_n, dtype g,
                                             dtype mass, dtype eps2,
                                             rsqrt_mode_t rsqrt_mode,
                                             size_t chains)
{
  /* For each target i, sum the acceleration generated by every source j.
   * In scalar notation, with d = r_j-r_i and q = |d|^2 + eps^2:
   *
   *   a_i += G*m*d/sqrt(q)^3 = G*m*d*(1/sqrt(q))^3.
   *
   * The three component sums are kept separate because x, y and z are stored
   * in independent SoA arrays. */
 
  size_t i;
  const dtype gm = g * mass; //constant G*m 

  /* Each OpenMP thread owns different target particles, so no atomic update is
   * needed in the innermost source loop. */
#pragma omp parallel for schedule(static)
  for (i = 0u; i < home->n; ++i)
  {
    const dtype xi = home->x[i];    // Broadcast one target particle's coordinates  
    const dtype yi = home->y[i];
    const dtype zi = home->z[i];
    
    dtype ax = (dtype)0.0, ay = (dtype)0.0, az = (dtype)0.0;    // `ax`, `ay`, `az` collect the scalar tail and the reduced chain sums. 
    
    size_t j = 0u; // first source not handled by the unrolled main loop


    switch (chains) // number of independent accumulation chains
    {
    case 2u:
    {
      const size_t source_n_unrolled = source_n - (source_n % 2u);  //Round down so the main loop always processes complete pairs
      dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;
      dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;

#pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1)
    
      for (j = 0u; j < source_n_unrolled; j += 2u) // One source pair is evaluated per iteration; the body is manually unrolled into chain 0 and chain 1
      {
        const dtype dx0 = sx[j] - xi; // d0 = r_source(j) - r_target: displacement of source j
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2; // q0 = |d0|^2 + eps^2: softened squared distance
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode); //invr0 = 1/sqrt(q0) (MODE: exact or approximate)
        const dtype s0 = gm * invr0 * invr0 * invr0; //s0 = G*m/q0^(3/2)

        ax0 += dx0 * s0; //Add source j's force contribution to chain 0
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        // Chain 1 repeats the same formula for the adjacent source j+1.
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

      j = source_n_unrolled;
      ax += ax0 + ax1; // independent chains are merged
      ay += ay0 + ay1;
      az += az0 + az1;
      break;
    }

    case 4u:
    {
     
      const size_t source_n_unrolled = source_n - (source_n % 4u); //complete groups of four and leave the tail for later
      dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;
      dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;
      dtype ax2 = (dtype)0.0, ay2 = (dtype)0.0, az2 = (dtype)0.0;
      dtype ax3 = (dtype)0.0, ay3 = (dtype)0.0, az3 = (dtype)0.0;

#pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1) \
    reduction(+ : ax2, ay2, az2) reduction(+ : ax3, ay3, az3)
      /* The four source bodies below are the explicit unrolling factor 4. */
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

      /* Combine the four partial vector components into the common sums. */
      j = source_n_unrolled;
      ax += ax0 + ax1 + ax2 + ax3;
      ay += ay0 + ay1 + ay2 + ay3;
      az += az0 + az1 + az2 + az3;
      break;
    }

    case 8u: // Eight chains are the maximum exposed scalar ILP setting
    {
      /
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
        /* Chain 0: displacement d0 and softened distance q0. */
        const dtype dx0 = sx[j] - xi;
        const dtype dy0 = sy[j] - yi;
        const dtype dz0 = sz[j] - zi;
        const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
        const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);
        const dtype s0 = gm * invr0 * invr0 * invr0;
        ax0 += dx0 * s0;
        ay0 += dy0 * s0;
        az0 += dz0 * s0;

        /* Chain 1: same force formula for source j+1. */
        const dtype dx1 = sx[j + 1u] - xi;
        const dtype dy1 = sy[j + 1u] - yi;
        const dtype dz1 = sz[j + 1u] - zi;
        const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
        const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
        const dtype s1 = gm * invr1 * invr1 * invr1;
        ax1 += dx1 * s1;
        ay1 += dy1 * s1;
        az1 += dz1 * s1;

        /* Chain 2: same force formula for source j+2. */
        const dtype dx2 = sx[j + 2u] - xi;
        const dtype dy2 = sy[j + 2u] - yi;
        const dtype dz2 = sz[j + 2u] - zi;
        const dtype r2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
        const dtype invr2 = invsqrt_force(r2_2, rsqrt_mode);
        const dtype s2 = gm * invr2 * invr2 * invr2;
        ax2 += dx2 * s2;
        ay2 += dy2 * s2;
        az2 += dz2 * s2;

        /* Chain 3: same force formula for source j+3. */
        const dtype dx3 = sx[j + 3u] - xi;
        const dtype dy3 = sy[j + 3u] - yi;
        const dtype dz3 = sz[j + 3u] - zi;
        const dtype r2_3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
        const dtype invr3 = invsqrt_force(r2_3, rsqrt_mode);
        const dtype s3 = gm * invr3 * invr3 * invr3;
        ax3 += dx3 * s3;
        ay3 += dy3 * s3;
        az3 += dz3 * s3;

        /* Chain 4: same force formula for source j+4. */
        const dtype dx4 = sx[j + 4u] - xi;
        const dtype dy4 = sy[j + 4u] - yi;
        const dtype dz4 = sz[j + 4u] - zi;
        const dtype r2_4 = dx4 * dx4 + dy4 * dy4 + dz4 * dz4 + eps2;
        const dtype invr4 = invsqrt_force(r2_4, rsqrt_mode);
        const dtype s4 = gm * invr4 * invr4 * invr4;
        ax4 += dx4 * s4;
        ay4 += dy4 * s4;
        az4 += dz4 * s4;

        /* Chain 5: same force formula for source j+5. */
        const dtype dx5 = sx[j + 5u] - xi;
        const dtype dy5 = sy[j + 5u] - yi;
        const dtype dz5 = sz[j + 5u] - zi;
        const dtype r2_5 = dx5 * dx5 + dy5 * dy5 + dz5 * dz5 + eps2;
        const dtype invr5 = invsqrt_force(r2_5, rsqrt_mode);
        const dtype s5 = gm * invr5 * invr5 * invr5;
        ax5 += dx5 * s5;
        ay5 += dy5 * s5;
        az5 += dz5 * s5;

        /* Chain 6: same force formula for source j+6. */
        const dtype dx6 = sx[j + 6u] - xi;
        const dtype dy6 = sy[j + 6u] - yi;
        const dtype dz6 = sz[j + 6u] - zi;
        const dtype r2_6 = dx6 * dx6 + dy6 * dy6 + dz6 * dz6 + eps2;
        const dtype invr6 = invsqrt_force(r2_6, rsqrt_mode);
        const dtype s6 = gm * invr6 * invr6 * invr6;
        ax6 += dx6 * s6;
        ay6 += dy6 * s6;
        az6 += dz6 * s6;

        /* Chain 7: same force formula for source j+7. */
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

      /* Horizontally combine all eight scalar chains. */
      j = source_n_unrolled;
      ax += ax0 + ax1 + ax2 + ax3 + ax4 + ax5 + ax6 + ax7;
      ay += ay0 + ay1 + ay2 + ay3 + ay4 + ay5 + ay6 + ay7;
      az += az0 + az1 + az2 + az3 + az4 + az5 + az6 + az7;
      break;
    }

    }

    // OpenMP SIMD requires an explicit initialization in its canonical loop.
    const size_t tail_start = j;
#pragma omp simd reduction(+ : ax, ay, az)
    // cleanup for source indices not covered by the selected unroll
    for (j = tail_start; j < source_n; ++j)
    {
      // d = r_source(j)-r_target: displacement of the remaining source
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

    // Add this source block's total to the target acceleration
    home->ax[i] += ax;
    home->ay[i] += ay;
    home->az[i] += az;
  }
}


/*                  ************************************************          */
/*                      AVX INSTRUCTION  +  NEWTON RAPHSON                    */
/*                  ************************************************          */


#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)

static inline __m512d refine_rsqrt_newton_pd(__m512d r2, __m512d x) // Vector Newton-Raphson refinement for eight double-precision lanes [__m512d] 
{
  const __m512d half = _mm512_set1_pd(0.5); // Broadcast the scalar constants 1/2 and 3 to all eight double lanes
  const __m512d three = _mm512_set1_pd(3.0);

  return _mm512_mul_pd(_mm512_mul_pd(half, x),
                       _mm512_sub_pd(three, _mm512_mul_pd(r2, _mm512_mul_pd(x, x)))); //Return x * (3 - r2*x*x) / 2, the Newton update for 1/sqrt(r2)
}

// reduce AVX-512 vector into a scalar double
static inline double hsum_pd(__m512d v)
{
  double tmp[8];  // Store the eight SIMD lanes to ordinary memory so they can be added
  _mm512_storeu_pd(tmp, v); // sum lanes 0...7 into one force component
  return tmp[0] + tmp[1] + tmp[2] + tmp[3] +    
         tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

/* AVX-512 implementation of the approximate inverse-square-root force path.
 *
 * `_mm512_rsqrt14_pd` provides a fast vector estimate for 1/sqrt(r2).  Two
 * Newton-Raphson refinements improve accuracy before the value is used in the
 * force expression.  This function is compiled only for double-precision
 * AVX-512 builds; portable builds use accumulate_sources_scalar_chains().
 */
static void accumulate_sources_rsqrt14_pd(const particles_t *home,
                                          const double *restrict sx,
                                          const double *restrict sy,
                                          const double *restrict sz,
                                          size_t source_n, double g,
                                          double mass, double eps2,
                                          rsqrt_mode_t rsqrt_mode)
{
  const __m512d eps2_v = _mm512_set1_pd(eps2); //Broadcast constants eps^2 and G*m
  const __m512d gm_v = _mm512_set1_pd(g * mass);
  size_t i;

#pragma omp parallel for schedule(static)
  for (i = 0u; i < home->n; ++i)
  {
    const __m512d xi = _mm512_set1_pd(home->x[i]); // Broadcast one target coordinate to all eight SIMD source lanes
    const __m512d yi = _mm512_set1_pd(home->y[i]);
    const __m512d zi = _mm512_set1_pd(home->z[i]);

    __m512d ax_v = _mm512_setzero_pd(); // Vector accumulators: each register contains eight partial sums
    __m512d ay_v = _mm512_setzero_pd();
    __m512d az_v = _mm512_setzero_pd();
    size_t j = 0u;

    for (; j + 7u < source_n; j += 8u)
    {
      
      const __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(sx + j), xi); //Load eight contiguous source x/y/z values without requiring alignment
      const __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(sy + j), yi);
      const __m512d dz = _mm512_sub_pd(_mm512_loadu_pd(sz + j), zi);
    
      __m512d r2 = _mm512_fmadd_pd(dx, dx, eps2_v); //FMA computes dx*dx + eps^2 in one vector instruction
  
      r2 = _mm512_fmadd_pd(dy, dy, r2); // Accumulate dy^2 and dz^2: r2 = dx^2 + dy^2 + dz^2 + eps^2
      r2 = _mm512_fmadd_pd(dz, dz, r2);

      __m512d invr = _mm512_rsqrt14_pd(r2); //Hardware estimate of 1/sqrt(r2)
  
      invr = refine_rsqrt_newton_pd(r2, invr); // First refinement: y <- y*(1.5 - 0.5*r2*y*y)
      if (rsqrt_mode != RSQRT_APPROX_NR1)
        invr = refine_rsqrt_newton_pd(r2, invr); // Optional second refinement

      const __m512d invr2 = _mm512_mul_pd(invr, invr); //invr2 = (1/sqrt(r2))^2 = 1/r2
      const __m512d s = _mm512_mul_pd(gm_v, _mm512_mul_pd(invr2, invr)); //s = G*m*(1/sqrt(r2))^3

      ax_v = _mm512_fmadd_pd(dx, s, ax_v); //Vector FMA performs ax_v += dx*s for all eight lanes
      ay_v = _mm512_fmadd_pd(dy, s, ay_v);
      az_v = _mm512_fmadd_pd(dz, s, az_v);
    }

    double ax = hsum_pd(ax_v); //Reduce the vector accumulators to scalar sums before handling a tail
    double ay = hsum_pd(ay_v);
    double az = hsum_pd(az_v);

    // TAIL MANAGING
    for (; j < source_n; ++j) // Scalar cleanup for fewer than eight remaining source particles
    {
      const double dx = sx[j] - home->x[i];
      const double dy = sy[j] - home->y[i];
      const double dz = sz[j] - home->z[i];
    
      const double r2 = dx * dx + dy * dy + dz * dz + eps2;
      const double invr = invsqrt_force(r2, rsqrt_mode);
      const double s = g * mass * invr * invr * invr;
      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }
    
    home->ax[i] += ax; // Merge this source block's vector-plus-tail result into the target SoA
    home->ay[i] += ay;
    home->az[i] += az;
  }
}
#endif


// This wrapper selects the AVX-512 approximate path when available and requested

static void accumulate_sources(const particles_t *home,
                               const dtype *restrict sx,
                               const dtype *restrict sy,
                               const dtype *restrict sz,
                               size_t source_n,
                               dtype g, dtype mass, dtype eps,
                               rsqrt_mode_t rsqrt_mode,
                               accumulator_mode_t accumulator_mode)
{
  /* `home` is the local target block; sx/sy/sz is the
   * current source block, which may belong to this rank or may have arrived from
   * a neighbor in the MPI ring. */
  const dtype eps2 = eps * eps;

#if defined(__AVX512F__) && !defined(NBODY_USE_FLOAT)
  if (rsqrt_mode != RSQRT_EXACT)
  {
    /* AVX-512 provides a vector reciprocal-square-root estimate. Newton
     * refinement improves it before it is used in the force formula. */
    accumulate_sources_rsqrt14_pd(home, sx, sy, sz, source_n, g, mass, eps2, rsqrt_mode);
    return;
  }
#endif

  accumulate_sources_scalar_chains(home, sx, sy, sz, source_n, g, mass, eps2,
                                   rsqrt_mode, (size_t)accumulator_mode);
}


/*                  ************************************************          */
/*                                  NEWTON'S THIRD LAW                        */
/*                  ************************************************          */

/* Compute accelerations with Newton's third law for a single MPI rank.
 *
 * Each unordered pair is evaluated once and contributes equal-and-opposite
 * acceleration updates.  Because two particles are updated per pair, OpenMP
 * threads write into private workspaces and a final reduction merges the results.
 */
static void compute_accelerations_newton_private(particles_t *local, dtype g,
                                                 dtype eps,
                                                 rsqrt_mode_t rsqrt_mode,
                                                 thread_workspace_t *workspace)
{
  
  const size_t n = local->n; // Number of particles owned by this rank; Newton requires a single rank
  const dtype eps2 = eps * eps;
  int nthreads; // Number of allocated thread-private slices, including any unused zero slices
  size_t i;

  nthreads = workspace->nthreads; // Reuse the thread count recorded when the workspace was allocated

  memset(workspace->tax, 0, (size_t)nthreads * n * sizeof(dtype)); // Clear all thread-private x accelerations; retain the existing allocation
  memset(workspace->tay, 0, (size_t)nthreads * n * sizeof(dtype));
  memset(workspace->taz, 0, (size_t)nthreads * n * sizeof(dtype));

#pragma omp parallel //OpenMP team
  {
    const int tid = omp_get_thread_num(); // Thread ID selects one private slice: tid = 0, ..., team_size-1. 
    dtype *ax = workspace->tax + (size_t)tid * n;// Point to this thread's x slice, starting at offset tid*N
    dtype *ay = workspace->tay + (size_t)tid * n;
    dtype *az = workspace->taz + (size_t)tid * n;

    size_t ii; // Target index for the triangular Newton pair loop

#pragma omp for schedule(static) //Assign target indices statically; the implicit end barrier completes all pairs

    for (ii = 0u; ii < n; ++ii)
    {

      const dtype xi = local->x[ii]; //Cache target coordinate x_ii
      const dtype yi = local->y[ii];
      const dtype zi = local->z[ii];

      size_t j;
      // Only j > ii: evaluate N*(N-1)/2 pairs, excluding self-interactions. 
      for (j = ii + 1u; j < n; ++j)
      {
        // The equal-and-opposite update halves arithmetic, while thread-private arrays prevent data races
        const dtype dx = local->x[j] - xi; //Displacement component dx = x_j - x_ii
        const dtype dy = local->y[j] - yi;
        const dtype dz = local->z[j] - zi;
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2; //q = dx^2 + dy^2 + dz^2 + epsilon^2
        /* . */
        const dtype invr = invsqrt_force(r2, rsqrt_mode); //u = 1/sqrt(q) [exact or approximate method]
        const dtype s = g * local->mass * invr * invr * invr; //s = G*m*u^3
        const dtype fx = dx * s; //Acceleration increment delta_ax = G*m*dx/q^(3/2)
        const dtype fy = dy * s;
        const dtype fz = dz * s;
        ax[ii] += fx; //Add delta_ax to particle ii in this thread's private slice
        ay[ii] += fy;
        az[ii] += fz;
        ax[j] -= fx;  //Apply opposite acceleration to j; equal masses make the magnitudes equal
        ay[j] -= fy;
        az[j] -= fz;
      }
    }
  }

#pragma omp parallel for schedule(static) // Distribute independent target particles across threads; finish with a barrier
  for (i = 0u; i < n; ++i) //Reduce the private contributions for each local particle
  {
    /* Merge private pair contributions into the shared acceleration field in a
     * separate pass; direct shared updates in the pair loop would race. */
    
    dtype ax = (dtype)0.0; //Initialize the x sum across thread-private slices
    dtype ay = (dtype)0.0;
    dtype az = (dtype)0.0;
    int t; // Index of a thread-private workspace slice
    
    for (t = 0; t < nthreads; ++t) //Sum every allocated slice; unused slices remain zero
    {
      ax += workspace->tax[(size_t)t * n + i]; //Add thread t's x acceleration for particle i
      ay += workspace->tay[(size_t)t * n + i];
      az += workspace->taz[(size_t)t * n + i];
    }

    local->ax[i] = ax; // Write the complete x acceleration; only this loop iteration owns particle i
    local->ay[i] = ay;
    local->az[i] = az;
  }
}

/* Exchange one source block with blocking MPI_Sendrecv calls.
 *
 * This is the simple communication baseline: computation and communication are
 * serialized, so the measured comm_wait exposes the cost that overlap tries to
 * hide.
 */
static void exchange_sources_sendrecv(dtype *bx, dtype *by, dtype *bz,
                                      dtype *rx, dtype *ry, dtype *rz,
                                      size_t buf_n, size_t next_n,
                                      int rank, int nranks, int tag_base,
                                      MPI_Datatype dt, MPI_Comm comm)
{
  /* Blocking ring exchange: send the current source block clockwise and receive
   * the previous rank's block counter-clockwise, component by component. */
  /* Clockwise destination; modulo wraps the last rank back to rank 0. */
  const int send_to = (rank + 1) % nranks;
  /* Counter-clockwise source; adding nranks avoids a negative remainder. */
  const int recv_from = (rank + nranks - 1) % nranks;
  /* Send buf_n x values while receiving next_n x values; distinct tags identify components. */
  MPI_Sendrecv(bx, (int)buf_n, dt, send_to, tag_base + 0,
               rx, (int)next_n, dt, recv_from, tag_base + 0, comm,
               MPI_STATUS_IGNORE);
  /* Send buf_n y values while receiving next_n y values; distinct tags identify components. */
  MPI_Sendrecv(by, (int)buf_n, dt, send_to, tag_base + 1,
               ry, (int)next_n, dt, recv_from, tag_base + 1, comm,
               MPI_STATUS_IGNORE);
  /* Send buf_n z values while receiving next_n z values; distinct tags identify components. */
  MPI_Sendrecv(bz, (int)buf_n, dt, send_to, tag_base + 2,
               rz, (int)next_n, dt, recv_from, tag_base + 2, comm,
               MPI_STATUS_IGNORE);
}

/* Post the non-blocking sends/receives used by the overlapped ring mode.
 *
 * The caller immediately computes with the current buffer, then waits on these
 * requests before rotating to the next received source block.
 */
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
  /* Clockwise destination; modulo wraps the last rank back to rank 0. */
  const int send_to = (rank + 1) % nranks;
  /* Counter-clockwise source; adding nranks avoids a negative remainder. */
  const int recv_from = (rank + nranks - 1) % nranks;
  /* Post x receive; rx may be read only after request 0 completes. */
  MPI_Irecv(rx, (int)next_n, dt, recv_from, tag_base + 0, comm, &req[0]);
  /* Post y receive; ry may be read only after request 1 completes. */
  MPI_Irecv(ry, (int)next_n, dt, recv_from, tag_base + 1, comm, &req[1]);
  /* Post z receive; rz may be read only after request 2 completes. */
  MPI_Irecv(rz, (int)next_n, dt, recv_from, tag_base + 2, comm, &req[2]);
  /* Post x send; bx must remain unchanged until request 3 completes. */
  MPI_Isend(bx, (int)buf_n, dt, send_to, tag_base + 0, comm, &req[3]);
  /* Post y send; by must remain unchanged until request 4 completes. */
  MPI_Isend(by, (int)buf_n, dt, send_to, tag_base + 1, comm, &req[4]);
  /* Post z send; bz must remain unchanged until request 5 completes. */
  MPI_Isend(bz, (int)buf_n, dt, send_to, tag_base + 2, comm, &req[5]);
}

/* Compute accelerations for the selected kernel and communication mode.
 *
 * For `direct`, this is the MPI ring: each rank keeps local target particles
 * fixed and circulates source positions until all ranks have contributed.  For
 * `newton`, the function delegates to the single-rank Newton ablation kernel.
 */
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
  /* Dispatch the single-rank Newton kernel before allocating direct-ring buffers. */
  if (kernel_mode == KERNEL_NEWTON)
  {
    /* Reject distributed Newton: returning remote reaction contributions is not implemented. */
    if (nranks != 1)
      /* Abort all ranks instead of computing incomplete local-only forces. */
      die("--kernel newton is implemented for -np 1 only; use --kernel direct for MPI ring runs");
    /* Evaluate each pair once and merge thread-private equal-mass accelerations. */
    compute_accelerations_newton_private(local, g, eps, rsqrt_mode,
                                         newton_workspace);
    /* Newton has already produced the acceleration field; skip the direct ring. */
    return;
  }

  /* Capacity of the largest rank-owned block, including uneven decompositions. */
  const size_t max_n = max_block_count(global_n, nranks);
  /* Allocate aligned x coordinates for the current source block. */
  dtype *bx = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned y coordinates for the current source block. */
  dtype *by = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned z coordinates for the current source block. */
  dtype *bz = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned x coordinates for the incoming block. */
  dtype *rx = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned y coordinates for the incoming block. */
  dtype *ry = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned z coordinates for the incoming block. */
  dtype *rz = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Use the MPI type matching dtype; dt here means datatype, not timestep. */
  MPI_Datatype dt = mpi_dtype();
  /* Initially the circulating source block belongs to this rank. */
  int owner = rank;
  /* Number of valid particles in the current circulating block. */
  size_t buf_n = local->n;
  /* Local target-particle index. */
  size_t i;

  /* Start each force evaluation from zero. Every source block in the ring then
   * adds its contribution to this local target acceleration. */
/* Distribute independent target particles across threads; finish with a barrier. */
#pragma omp parallel for schedule(static)
  /* Reset each local target before accumulating a fresh force evaluation. */
  for (i = 0u; i < local->n; ++i)
    /* Set a_i = (0,0,0); subsequent blocks add their contributions. */
    local->ax[i] = local->ay[i] = local->az[i] = (dtype)0.0;
  /* Initialize the circulating x coordinates with this rank's local particles. */
  memcpy(bx, local->x, local->n * sizeof(dtype));
  /* Initialize the circulating y coordinates with this rank's local particles. */
  memcpy(by, local->y, local->n * sizeof(dtype));
  /* Initialize the circulating z coordinates with this rank's local particles. */
  memcpy(bz, local->z, local->n * sizeof(dtype));

  /* Visit P source blocks; this is a ring stage, not an integration timestep. */
  for (int step = 0; step < nranks; ++step)
  {
    /* Step 0 computes the local source block; later steps compute blocks that
     * have traveled around the ring. */
    /* Exchange blocks only when another rank exists. */
    if (nranks > 1)
    {
      /* Identify the original owner of the next incoming block. */
      const int next_owner = (owner + nranks - 1) % nranks;
      /* Particle count in the next incoming block. */
      size_t next_n;
      /* Request only the next count; forces do not need its global start index. */
      block_bounds(global_n, next_owner, nranks, NULL, &next_n);
      /* Select non-blocking transfers with computation before the completion wait. */
      if (mode == COMM_OVERLAP)
      {
        /* Track three receives and three sends until MPI_Waitall completes them. */
        MPI_Request req[6];
        /* Post transfers using tags 10..12; current source buffers stay unmodified. */
        post_source_exchange(bx, by, bz, rx, ry, rz, buf_n, next_n,
                             rank, nranks, 10, dt, comm, req);
        /* Compute with the old buffer while MPI transfers the next one. The
         * wait is required before rx/ry/rz can be copied into the source buffer. */
        /* Add G*m*(r_j-r_i)/(distance^2+epsilon^2)^(3/2) from this source block. */
        accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,
                           rsqrt_mode, accumulator_mode);
        {
          /* Start the wall-clock timer immediately before waiting or blocking exchange. */
          const double comm_t0 = seconds();
          /* Complete all six operations before reusing send buffers or reading receives. */
          MPI_Waitall(6, req, MPI_STATUSES_IGNORE);
          /* Accumulate exposed communication time; overlap work is outside this interval. */
          *comm_wait += seconds() - comm_t0;
        }
      }
      else
      {
        /* Baseline mode computes first and performs the blocking exchange
         * afterwards, so communication cannot overlap force work. */
        /* Add G*m*(r_j-r_i)/(distance^2+epsilon^2)^(3/2) from this source block. */
        accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,
                           rsqrt_mode, accumulator_mode);
        {
          /* Start the wall-clock timer immediately before waiting or blocking exchange. */
          const double comm_t0 = seconds();
          /* Exchange x/y/z with the neighbours; tag base is supplied on the next line. */
          exchange_sources_sendrecv(bx, by, bz, rx, ry, rz, buf_n, next_n,
                                    rank, nranks, 10, dt, comm);
          /* Accumulate exposed communication time; overlap work is outside this interval. */
          *comm_wait += seconds() - comm_t0;
        }
      }
      /* Copy received x coordinates into the source buffer for the next stage. */
      memcpy(bx, rx, next_n * sizeof(dtype));
      /* Copy received y coordinates into the source buffer for the next stage. */
      memcpy(by, ry, next_n * sizeof(dtype));
      /* Copy received z coordinates into the source buffer for the next stage. */
      memcpy(bz, rz, next_n * sizeof(dtype));
      /* Record the original owner of the newly received coordinates. */
      owner = next_owner;
      /* Update the valid length; different ranks may own different particle counts. */
      buf_n = next_n;
    }
    else
      /* Add G*m*(r_j-r_i)/(distance^2+epsilon^2)^(3/2) from this source block. */
      accumulate_sources(local, bx, by, bz, buf_n, g, local->mass, eps,
                         rsqrt_mode, accumulator_mode);
  }

  /* Release the temporary source x buffer after all transfers complete. */
  free(bx);
  /* Release the temporary source y buffer after all transfers complete. */
  free(by);
  /* Release the temporary source z buffer after all transfers complete. */
  free(bz);
  /* Release the temporary receive x buffer after all transfers complete. */
  free(rx);
  /* Release the temporary receive y buffer after all transfers complete. */
  free(ry);
  /* Release the temporary receive z buffer after all transfers complete. */
  free(rz);
}

/* Compute the kinetic-energy contribution owned by this rank. */
static long double kinetic_energy_local(const particles_t *p)
{
  /* Kinetic energy is local to each rank and then summed globally.  Long double
   * accumulation reduces diagnostic roundoff without changing simulation dtype. */
  /* Initialize an extended-precision sum to reduce diagnostic accumulation error. */
  long double sum = 0.0L;
  /* Local target-particle index. */
  size_t i;
/* Give each thread a private sum, then combine all partial sums with addition. */
#pragma omp parallel for reduction(+ : sum) schedule(static)
  /* Visit every local particle exactly once for kinetic energy. */
  for (i = 0u; i < p->n; ++i)
  {
    /* K = 1/2 sum_i m|v_i|^2. Long double reduces diagnostic roundoff without
     * changing the dtype used by the simulation itself. */
    /* Promote velocity component vx_i before squaring. */
    const long double vx = (long double)p->vx[i];
    /* Promote velocity component vy_i before squaring. */
    const long double vy = (long double)p->vy[i];
    /* Promote velocity component vz_i before squaring. */
    const long double vz = (long double)p->vz[i];
    /* Accumulate |v_i|^2 = vx_i^2 + vy_i^2 + vz_i^2 in long double. */
    sum += vx * vx + vy * vy + vz * vz;
  }
  /* Return K_local = (m/2)*sum_i |v_i|^2 for equal particle masses. */
  return 0.5L * (long double)p->mass * sum;
}

/* Compute potential-energy contributions against one source block.
 *
 * The global-index check counts each unordered pair once, which avoids the
 * factor-of-two error that would appear if every rank summed both (i,j) and
 * (j,i).
 */
static long double potential_sources(const particles_t *home, size_t home_start,
                                     const dtype *restrict sx,
                                     const dtype *restrict sy,
                                     const dtype *restrict sz,
                                     size_t source_start, size_t source_n,
                                     dtype g, dtype eps)
{
  /* eps2 = epsilon^2, the softening term used in squared distances. */
  const dtype eps2 = eps * eps;
  /* m2 = m*m; potential energy involves both equal masses. */
  const long double m2 = (long double)home->mass * (long double)home->mass;
  /* Initialize an extended-precision sum to reduce diagnostic accumulation error. */
  long double sum = 0.0L;
  /* Local target-particle index. */
  size_t i;
/* Give each thread a private sum, then combine all partial sums with addition. */
#pragma omp parallel for reduction(+ : sum) schedule(static)
  /* Visit local target particles against this circulating source block. */
  for (i = 0u; i < home->n; ++i)
  {
    /* Cache the target position r_i = (xi,yi,zi). */
    const dtype xi = home->x[i], yi = home->y[i], zi = home->z[i];
    /* Convert target index i into its global particle index. */
    const size_t gi = home_start + i;
    /* Source-particle index inside the current block. */
    size_t j;
    /* Visit all sources in the current block. */
    for (j = 0u; j < source_n; ++j)
      /* Keep only global i < global j: excludes self-pairs and double counting. */
      if (gi < source_start + j)
      {
        /* The strict global-index test counts each unordered pair exactly once:
         * pair (i,j) is included only when global i < global j. */
        /* Displacement component dx = x_source - x_target. */
        const dtype dx = sx[j] - xi;
        /* Displacement component dy = y_source - y_target. */
        const dtype dy = sy[j] - yi;
        /* Displacement component dz = z_source - z_target. */
        const dtype dz = sz[j] - zi;
        /* q = dx^2 + dy^2 + dz^2 + epsilon^2, the softened squared separation. */
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
        /* Use the exact reciprocal root for energy even when forces use approx. */
        const dtype invr = (dtype)1.0 / dtype_sqrt(r2);
        /* Add U_ij = -G*m^2/sqrt(q); the attractive potential is negative. */
        sum -= (long double)g * m2 * (long double)invr;
      }
  }
  /* Return this rank/block potential contribution; the caller performs global summation. */
  return sum;
}

/* Compute total energy through a ring traversal and global reductions.
 *
 * This diagnostic is intentionally separate from the force kernel so its
 * overhead can be measured with the `energy-every` benchmark.
 */
static dtype total_energy_ring(const particles_t *local, size_t global_n,
                               size_t local_start, dtype g, dtype eps,
                               int rank, int nranks, MPI_Comm comm,
                               dtype *kinetic, dtype *potential)
{
  /* Potential energy uses a second ring traversal.  It is expensive O(N^2), so
   * production runs sample it periodically and report the measured overhead. */
  /* Capacity of the largest rank-owned block, including uneven decompositions. */
  const size_t max_n = max_block_count(global_n, nranks);
  /* Allocate aligned x coordinates for the current source block. */
  dtype *bx = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned y coordinates for the current source block. */
  dtype *by = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned z coordinates for the current source block. */
  dtype *bz = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned x coordinates for the incoming block. */
  dtype *rx = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned y coordinates for the incoming block. */
  dtype *ry = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Allocate aligned z coordinates for the incoming block. */
  dtype *rz = checked_aligned_alloc(max_n * sizeof(dtype));
  /* Initially the circulating source block belongs to this rank. */
  int owner = rank;
  /* Track the global first index and valid length needed to select unique pairs. */
  size_t buf_start = local_start, buf_n = local->n;
  /* Start U_local at zero and compute K_local from locally owned velocities. */
  long double pot_local = 0.0L, kin_local = kinetic_energy_local(local);
  /* Receive the global potential and kinetic sums on every rank. */
  long double pot_global, kin_global;

  /* Initialize the circulating x coordinates with this rank's local particles. */
  memcpy(bx, local->x, local->n * sizeof(dtype));
  /* Initialize the circulating y coordinates with this rank's local particles. */
  memcpy(by, local->y, local->n * sizeof(dtype));
  /* Initialize the circulating z coordinates with this rank's local particles. */
  memcpy(bz, local->z, local->n * sizeof(dtype));

  /* Visit P source blocks; this is a ring stage, not an integration timestep. */
  for (int step = 0; step < nranks; ++step)
  {
    /* Each rank evaluates its local targets against the current source block.
     * The global-index filter avoids double counting; Allreduce combines the
     * disjoint local sums at the end. */
    /* Accumulate unique-pair potential energy for the current source block. */
    pot_local += potential_sources(local, local_start, bx, by, bz,
                                   buf_start, buf_n, g, eps);
    /* Exchange blocks only when another rank exists. */
    if (nranks > 1)
    {
      /* Identify the original owner of the next incoming block. */
      const int next_owner = (owner + nranks - 1) % nranks;
      /* Global start and length of the next block, both needed for pair selection. */
      size_t next_start, next_n;
      /* Recover the next owner's contiguous global particle range. */
      block_bounds(global_n, next_owner, nranks, &next_start, &next_n);
      /* Exchange x/y/z with the neighbours; tag base is supplied on the next line. */
      exchange_sources_sendrecv(bx, by, bz, rx, ry, rz, buf_n, next_n,
                                rank, nranks, 20, mpi_dtype(), comm);
      /* Copy received x coordinates into the source buffer for the next stage. */
      memcpy(bx, rx, next_n * sizeof(dtype));
      /* Copy received y coordinates into the source buffer for the next stage. */
      memcpy(by, ry, next_n * sizeof(dtype));
      /* Copy received z coordinates into the source buffer for the next stage. */
      memcpy(bz, rz, next_n * sizeof(dtype));
      /* Record the original owner of the newly received coordinates. */
      owner = next_owner;
      /* Keep global indexing synchronized with the newly received coordinates. */
      buf_start = next_start;
      /* Update the valid length; different ranks may own different particle counts. */
      buf_n = next_n;
    }
  }

  /* K = sum_r K_r; collective addition returns the result to every rank. */
  MPI_Allreduce(&kin_local, &kin_global, 1, MPI_LONG_DOUBLE, MPI_SUM, comm);
  /* U = sum_r U_r; each unordered pair was counted exactly once. */
  MPI_Allreduce(&pot_local, &pot_global, 1, MPI_LONG_DOUBLE, MPI_SUM, comm);
  /* Convert global K to simulation precision and return it through the pointer. */
  *kinetic = (dtype)kin_global;
  /* Convert global U to simulation precision and return it through the pointer. */
  *potential = (dtype)pot_global;

  /* Release the temporary source x buffer after all transfers complete. */
  free(bx);
  /* Release the temporary source y buffer after all transfers complete. */
  free(by);
  /* Release the temporary source z buffer after all transfers complete. */
  free(bz);
  /* Release the temporary receive x buffer after all transfers complete. */
  free(rx);
  /* Release the temporary receive y buffer after all transfers complete. */
  free(ry);
  /* Release the temporary receive z buffer after all transfers complete. */
  free(rz);
  /* Return total mechanical energy E = K + U in simulation precision. */
  return *kinetic + *potential;
}





/*                  ************************************************          */
/*                                     PARSING FUNCTION                               */
/*                  ************************************************          */

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

/* Parse a floating-point command-line value into the simulation dtype. */
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

/* Decode the communication-mode option: blocking sendrecv or overlapped ring. */
static comm_mode_t parse_comm_mode(const char *text)
{
  if (strcmp(text, "sendrecv") == 0)
    return COMM_SENDRECV;
  if (strcmp(text, "overlap") == 0)
    return COMM_OVERLAP;
  die("invalid --comm '%s' (expected sendrecv or overlap)", text);
  return COMM_SENDRECV;
}

/* Decode the force-kernel option: full direct summation or local Newton reuse. */
static kernel_mode_t parse_kernel_mode(const char *text)
{
  if (strcmp(text, "direct") == 0)
    return KERNEL_DIRECT;
  if (strcmp(text, "newton") == 0)
    return KERNEL_NEWTON;
  die("invalid --kernel '%s' (expected direct or newton)", text);
  return KERNEL_DIRECT;
}

/* Decode the inverse-square-root option: exact or approximate/refined. */
static rsqrt_mode_t parse_rsqrt_mode(const char *text)
{
  if (strcmp(text, "exact") == 0)
    return RSQRT_EXACT;
  if (strcmp(text, "approx") == 0 || strcmp(text, "approx2") == 0)
    return RSQRT_APPROX;
  if (strcmp(text, "approx1") == 0)
    return RSQRT_APPROX_NR1;
  die("invalid --rsqrt '%s' (expected exact, approx1 or approx2; approx aliases approx2)", text);
  return RSQRT_EXACT;
}

/* Decode the number of accumulator chains used by the direct force loop. */
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

/* Convert a communication mode back to the string written in benchmark output. */
static const char *comm_mode_name(comm_mode_t mode)
{
  return mode == COMM_OVERLAP ? "overlap" : "sendrecv";
}

/* Convert a kernel mode back to the string written in benchmark output. */
static const char *kernel_mode_name(kernel_mode_t mode)
{
  return mode == KERNEL_NEWTON ? "newton" : "direct";
}

/* Convert an inverse-square-root mode back to the benchmark-output string. */
static const char *rsqrt_mode_name(rsqrt_mode_t mode)
{
  return mode == RSQRT_EXACT ? "exact" :
         mode == RSQRT_APPROX_NR1 ? "approx1" : "approx2";
}

/* Convert an accumulator-chain mode back to the benchmark-output string. */
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

/* Extract the value of one command-line option.
 *
 * Both `--key value` and `--key=value` are accepted so Slurm wrappers can pass
 * arguments in whichever form is more convenient.
 */
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

/* Print the user-facing command-line help. */
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

/*                  ************************************************          */
/*                                         MAIN                               */
/*                  ************************************************          */


int main(int argc, char **argv)
{
  const char *input_path = NULL, *output_path = NULL;
  size_t nsteps = 10u, energy_every = 1u, global_n = 0u, local_start = 0u;
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
    {
      /* Compatibility no-op: output is already summary-only. */
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
