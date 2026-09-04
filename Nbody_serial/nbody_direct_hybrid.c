#include "nbody_common.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <omp.h>

typedef struct particles_s
{
  size_t n;
  dtype mass;
  dtype *x, *y, *z;
  dtype *vx, *vy, *vz;
  dtype *ax, *ay, *az;
} particles_t;

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

typedef enum integrator_e
{
  INTEGRATOR_KDK,
  INTEGRATOR_DKD
} integrator_t;

typedef enum comm_mode_e
{
  COMM_SENDRECV,
  COMM_OVERLAP
} comm_mode_t;

typedef enum kernel_mode_e
{
  KERNEL_DIRECT,
  KERNEL_NEWTON
} kernel_mode_t;

typedef enum rsqrt_mode_e
{
  RSQRT_EXACT,
  RSQRT_APPROX
} rsqrt_mode_t;

typedef enum accumulator_mode_e
{
  ACCUMULATORS_ONE = 1,
  ACCUMULATORS_FOUR = 4
} accumulator_mode_t;

static void die(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fputc('\n', stderr);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static MPI_Datatype mpi_dtype(void)
{
#if defined(NBODY_USE_FLOAT)
  return MPI_FLOAT;
#else
  return MPI_DOUBLE;
#endif
}

static double seconds(void)
{
  return MPI_Wtime();
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

static integrator_t parse_integrator(const char *text)
{
  if (strcmp(text, "kdk") == 0)
    return INTEGRATOR_KDK;
  if (strcmp(text, "dkd") == 0)
    return INTEGRATOR_DKD;
  die("invalid --integrator '%s' (expected kdk or dkd)", text);
  return INTEGRATOR_KDK;
}

static const char *integrator_name(integrator_t integrator)
{
  return integrator == INTEGRATOR_KDK ? "kdk" : "dkd";
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

static const char *comm_mode_name(comm_mode_t mode)
{
  return mode == COMM_OVERLAP ? "overlap" : "sendrecv";
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

static const char *kernel_mode_name(kernel_mode_t mode)
{
  return mode == KERNEL_NEWTON ? "newton" : "direct";
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

static accumulator_mode_t parse_accumulator_mode(const char *text)
{
  if (strcmp(text, "1") == 0)
    return ACCUMULATORS_ONE;
  if (strcmp(text, "4") == 0)
    return ACCUMULATORS_FOUR;
  die("invalid --accumulators '%s' (expected 1 or 4)", text);
  return ACCUMULATORS_FOUR;
}

static const char *accumulator_mode_name(accumulator_mode_t mode)
{
  return mode == ACCUMULATORS_ONE ? "1" : "4";
}

static inline dtype invsqrt_force(dtype r2, rsqrt_mode_t mode)
{
  if (mode == RSQRT_EXACT)
    return (dtype)1.0 / dtype_sqrt(r2);

  /* Approximate path: low precision seed plus two Newton refinements. */
  {
    dtype x = (dtype)(1.0f / sqrtf((float)r2));
    const dtype half = (dtype)0.5;
    const dtype three_halves = (dtype)1.5;
    x = x * (three_halves - half * r2 * x * x);
    x = x * (three_halves - half * r2 * x * x);
    return x;
  }
}

static const char *option_value(int *i, int argc, char **argv, const char *key)
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

static void checked_fwrite(const void *ptr, size_t size, size_t nmemb,
                           FILE *fp, const char *path, const char *what)
{
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

static void block_bounds(size_t n, int rank, int nranks,
                         size_t *start, size_t *count)
{
  const size_t base = n / (size_t)nranks;
  const size_t rem = n % (size_t)nranks;
  *count = base + ((size_t)rank < rem ? 1u : 0u);
  *start = (size_t)rank * base + ((size_t)rank < rem ? (size_t)rank : rem);
}

static size_t max_block_count(size_t n, int nranks)
{
  size_t start, count;
  block_bounds(n, 0, nranks, &start, &count);
  return count;
}

static void read_local_particles(const char *path, dtype mass, int rank,
                                 int nranks, particles_t *local,
                                 size_t *global_n, size_t *local_start)
{
  FILE *fp = fopen(path, "rb");
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t i, n, local_n;

  if (fp == NULL)
    die("cannot open input '%s'", path);
  checked_fread(magic, 1u, NBODY_BINARY_MAGIC_SIZE, fp, path, "magic");
  if (memcmp(magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die("invalid input magic in '%s'", path);
  checked_fread(&n64, sizeof n64, 1u, fp, path, "particle count");
  if ((uint64_t)(size_t)n64 != n64)
    die("input particle count is too large");

  n = (size_t)n64;
  block_bounds(n, rank, nranks, local_start, &local_n);
  particles_allocate(local, local_n, mass);

  for (i = 0u; i < n; ++i)
  {
    float rec[NBODY_BINARY_COMPONENTS];
    checked_fread(rec, sizeof rec[0], NBODY_BINARY_COMPONENTS, fp, path,
                  "particle record");
    if ((i >= *local_start) && (i < *local_start + local_n))
    {
      const size_t k = i - *local_start;
      local->x[k] = (dtype)rec[0];
      local->y[k] = (dtype)rec[1];
      local->z[k] = (dtype)rec[2];
      local->vx[k] = (dtype)rec[3];
      local->vy[k] = (dtype)rec[4];
      local->vz[k] = (dtype)rec[5];
    }
  }
  fclose(fp);
  *global_n = n;
}

static float to_float_checked(dtype value, const char *path)
{
  if (!dtype_isfinite(value) || (fabs((double)value) > (double)FLT_MAX))
    die("cannot write non-finite or overflowing value to '%s'", path);
  return (float)value;
}

static void write_output_root(const char *path, const particles_t *local,
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
  size_t i;
#pragma omp parallel for schedule(static)
  for (i = 0u; i < p->n; ++i)
  {
    p->x[i] += dt * p->vx[i];
    p->y[i] += dt * p->vy[i];
    p->z[i] += dt * p->vz[i];
  }
}

static void kick(particles_t *p, dtype dt)
{
  size_t i;
#pragma omp parallel for schedule(static)
  for (i = 0u; i < p->n; ++i)
  {
    p->vx[i] += dt * p->ax[i];
    p->vy[i] += dt * p->ay[i];
    p->vz[i] += dt * p->az[i];
  }
}

static void accumulate_sources(const particles_t *home,
                               const dtype *restrict sx,
                               const dtype *restrict sy,
                               const dtype *restrict sz,
                               size_t source_n,
                               dtype g, dtype mass, dtype eps,
                               rsqrt_mode_t rsqrt_mode,
                               accumulator_mode_t accumulator_mode)
{
  const dtype eps2 = eps * eps;
  size_t i;

  if (accumulator_mode == ACCUMULATORS_ONE)
  {
#pragma omp parallel for schedule(static)
    for (i = 0u; i < home->n; ++i)
    {
      const dtype xi = home->x[i];
      const dtype yi = home->y[i];
      const dtype zi = home->z[i];
      dtype ax = (dtype)0.0, ay = (dtype)0.0, az = (dtype)0.0;
      size_t j;

#pragma omp simd reduction(+ : ax, ay, az)
      for (j = 0u; j < source_n; ++j)
      {
        const dtype dx = sx[j] - xi;
        const dtype dy = sy[j] - yi;
        const dtype dz = sz[j] - zi;
        const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = invsqrt_force(r2, rsqrt_mode);
        const dtype s = g * mass * invr * invr * invr;
        ax += dx * s;
        ay += dy * s;
        az += dz * s;
      }

      home->ax[i] += ax;
      home->ay[i] += ay;
      home->az[i] += az;
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (i = 0u; i < home->n; ++i)
  {
    const dtype xi = home->x[i];
    const dtype yi = home->y[i];
    const dtype zi = home->z[i];

    /* Four independent accumulators shorten the dependency chain. */
    dtype ax0 = (dtype)0.0, ay0 = (dtype)0.0, az0 = (dtype)0.0;
    dtype ax1 = (dtype)0.0, ay1 = (dtype)0.0, az1 = (dtype)0.0;
    dtype ax2 = (dtype)0.0, ay2 = (dtype)0.0, az2 = (dtype)0.0;
    dtype ax3 = (dtype)0.0, ay3 = (dtype)0.0, az3 = (dtype)0.0;

    size_t j = 0u;
    const size_t source_n_unrolled = source_n & ~(size_t)3;

#pragma omp simd
    for (j = 0u; j < source_n_unrolled; j += 4u)
    {
      const dtype dx0 = sx[j] - xi;
      const dtype dy0 = sy[j] - yi;
      const dtype dz0 = sz[j] - zi;
      const dtype r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
      const dtype invr0 = invsqrt_force(r2_0, rsqrt_mode);
      const dtype s0 = g * mass * invr0 * invr0 * invr0;
      ax0 += dx0 * s0;
      ay0 += dy0 * s0;
      az0 += dz0 * s0;

      const dtype dx1 = sx[j + 1] - xi;
      const dtype dy1 = sy[j + 1] - yi;
      const dtype dz1 = sz[j + 1] - zi;
      const dtype r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
      const dtype invr1 = invsqrt_force(r2_1, rsqrt_mode);
      const dtype s1 = g * mass * invr1 * invr1 * invr1;
      ax1 += dx1 * s1;
      ay1 += dy1 * s1;
      az1 += dz1 * s1;

      const dtype dx2 = sx[j + 2] - xi;
      const dtype dy2 = sy[j + 2] - yi;
      const dtype dz2 = sz[j + 2] - zi;
      const dtype r2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
      const dtype invr2 = invsqrt_force(r2_2, rsqrt_mode);
      const dtype s2 = g * mass * invr2 * invr2 * invr2;
      ax2 += dx2 * s2;
      ay2 += dy2 * s2;
      az2 += dz2 * s2;

      const dtype dx3 = sx[j + 3] - xi;
      const dtype dy3 = sy[j + 3] - yi;
      const dtype dz3 = sz[j + 3] - zi;
      const dtype r2_3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
      const dtype invr3 = invsqrt_force(r2_3, rsqrt_mode);
      const dtype s3 = g * mass * invr3 * invr3 * invr3;
      ax3 += dx3 * s3;
      ay3 += dy3 * s3;
      az3 += dz3 * s3;
    }

    dtype ax_rem = (dtype)0.0, ay_rem = (dtype)0.0, az_rem = (dtype)0.0;
    for (; j < source_n; ++j)
    {
      const dtype dx = sx[j] - xi;
      const dtype dy = sy[j] - yi;
      const dtype dz = sz[j] - zi;
      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = invsqrt_force(r2, rsqrt_mode);
      const dtype s = g * mass * invr * invr * invr;
      ax_rem += dx * s;
      ay_rem += dy * s;
      az_rem += dz * s;
    }

    home->ax[i] += ax0 + ax1 + ax2 + ax3 + ax_rem;
    home->ay[i] += ay0 + ay1 + ay2 + ay3 + ay_rem;
    home->az[i] += az0 + az1 + az2 + az3 + az_rem;
  }
}

static void compute_accelerations_newton_private(particles_t *local, dtype g,
                                                 dtype eps,
                                                 rsqrt_mode_t rsqrt_mode)
{
  const size_t n = local->n;
  const dtype eps2 = eps * eps;
  const int nthreads = omp_get_max_threads();
  dtype *tax = calloc((size_t)nthreads * n, sizeof(dtype));
  dtype *tay = calloc((size_t)nthreads * n, sizeof(dtype));
  dtype *taz = calloc((size_t)nthreads * n, sizeof(dtype));
  size_t i;

  if ((tax == NULL) || (tay == NULL) || (taz == NULL))
    die("cannot allocate thread-private Newton buffers");

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    dtype *ax = tax + (size_t)tid * n;
    dtype *ay = tay + (size_t)tid * n;
    dtype *az = taz + (size_t)tid * n;
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
    dtype ax = (dtype)0.0;
    dtype ay = (dtype)0.0;
    dtype az = (dtype)0.0;
    int t;
    for (t = 0; t < nthreads; ++t)
    {
      ax += tax[(size_t)t * n + i];
      ay += tay[(size_t)t * n + i];
      az += taz[(size_t)t * n + i];
    }
    local->ax[i] = ax;
    local->ay[i] = ay;
    local->az[i] = az;
  }

  free(tax);
  free(tay);
  free(taz);
}

static void exchange_sources_sendrecv(dtype *bx, dtype *by, dtype *bz,
                                      dtype *rx, dtype *ry, dtype *rz,
                                      size_t buf_n, size_t next_n,
                                      int rank, int nranks, int tag_base,
                                      MPI_Datatype dt, MPI_Comm comm)
{
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
                                       double *comm_wait,
                                       MPI_Comm comm)
{
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

  if (kernel_mode == KERNEL_NEWTON)
  {
    if (nranks != 1)
      die("--kernel newton is implemented for -np 1 only; use --kernel direct for MPI ring runs");
    compute_accelerations_newton_private(local, g, eps, rsqrt_mode);
    return;
  }

#pragma omp parallel for schedule(static)
  for (i = 0u; i < local->n; ++i)
    local->ax[i] = local->ay[i] = local->az[i] = (dtype)0.0;
  memcpy(bx, local->x, local->n * sizeof(dtype));
  memcpy(by, local->y, local->n * sizeof(dtype));
  memcpy(bz, local->z, local->n * sizeof(dtype));

  for (int step = 0; step < nranks; ++step)
  {
    if (nranks > 1)
    {
      const int next_owner = (owner + nranks - 1) % nranks;
      size_t next_start, next_n;
      block_bounds(global_n, next_owner, nranks, &next_start, &next_n);
      (void)next_start;
      if (mode == COMM_OVERLAP)
      {
        MPI_Request req[6];
        post_source_exchange(bx, by, bz, rx, ry, rz, buf_n, next_n,
                             rank, nranks, 10, dt, comm, req);
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
  long double sum = 0.0L;
  size_t i;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (i = 0u; i < p->n; ++i)
  {
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

static void print_usage(const char *program)
{
  fprintf(stderr,
          "usage: %s --input FILE [options]\n"
          "  --output FILE             optional final-state binary file\n"
          "  --nsteps N                DKD steps (default: 10)\n"
          "  --dt X                    time step (default: 0.001)\n"
          "  --integrator kdk|dkd      leapfrog form (default: kdk)\n"
          "  --comm sendrecv|overlap   ring exchange mode (default: sendrecv)\n"
          "  --kernel direct|newton    force kernel (default: direct; newton is -np 1 only)\n"
          "  --rsqrt exact|approx      force inverse sqrt path (default: exact)\n"
          "  --accumulators 1|4        direct-kernel accumulator chains (default: 4)\n"
          "  --eps X                   softening length (default: 0.01)\n"
          "  --G X                     gravitational constant (default: 1)\n"
          "  --mass X                  equal particle mass (default: 1)\n"
          "  --energy-every N          diagnostic period (default: 1)\n"
          "  --energy-tol X            warning tolerance (default: 1e-3)\n"
          "  --quiet                   final summary only\n",
          program);
  fprintf(stderr, "binary format: %s\n", NBODY_BINARY_VERSION_TEXT);
}

int main(int argc, char **argv)
{
  const char *input_path = NULL, *output_path = NULL;
  size_t nsteps = 10u, energy_every = 1u, global_n = 0u, local_start = 0u;
  dtype dt = (dtype)1.0e-3, eps = (dtype)1.0e-2;
  dtype g = (dtype)1.0, mass = (dtype)1.0, energy_tol = (dtype)1.0e-3;
  integrator_t integrator = INTEGRATOR_KDK;
  comm_mode_t comm_mode = COMM_SENDRECV;
  kernel_mode_t kernel_mode = KERNEL_DIRECT;
  rsqrt_mode_t rsqrt_mode = RSQRT_EXACT;
  accumulator_mode_t accumulator_mode = ACCUMULATORS_FOUR;
  bool quiet = false;
  int rank, nranks, provided;
  particles_t local;
  timings_t timing = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  dtype kinetic0, potential0, energy0;
  double max_rel_drift = 0.0;
  double t0, t1;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  particles_init_empty(&local);

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
    else if ((value = option_value(&argi, argc, argv, "--integrator")) != NULL)
      integrator = parse_integrator(value);
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

  timing.total = seconds();
  t0 = seconds();
  read_local_particles(input_path, mass, rank, nranks, &local,
                       &global_n, &local_start);
  timing.io += seconds() - t0;

  t0 = seconds();
  energy0 = total_energy_ring(&local, global_n, local_start, g, eps, rank,
                              nranks, MPI_COMM_WORLD, &kinetic0, &potential0);
  timing.energy += seconds() - t0;

  if ((rank == 0) && !quiet)
  {
    printf("# hybrid MPI+OpenMP direct N-body %s\n",
           integrator_name(integrator));
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

  if (integrator == INTEGRATOR_KDK)
  {
    t0 = seconds();
    compute_accelerations_ring(&local, global_n, g, eps, rank,
                               nranks, comm_mode, kernel_mode, rsqrt_mode,
                               accumulator_mode,
                               &timing.comm_wait,
                               MPI_COMM_WORLD);
    timing.force += seconds() - t0;
  }

  for (size_t step = 1u; step <= nsteps; ++step)
  {
    if (integrator == INTEGRATOR_KDK)
    {
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
                                 &timing.comm_wait,
                                 MPI_COMM_WORLD);
      timing.force += seconds() - t0;

      t0 = seconds();
      kick(&local, (dtype)0.5 * dt);
      timing.kick += seconds() - t0;
    }
    else
    {
      t0 = seconds();
      drift(&local, (dtype)0.5 * dt);
      timing.drift += seconds() - t0;

      t0 = seconds();
      compute_accelerations_ring(&local, global_n, g, eps,
                                 rank, nranks, comm_mode, kernel_mode,
                                 rsqrt_mode, accumulator_mode,
                                 &timing.comm_wait,
                                 MPI_COMM_WORLD);
      timing.force += seconds() - t0;

      t0 = seconds();
      kick(&local, dt);
      timing.kick += seconds() - t0;

      t0 = seconds();
      drift(&local, (dtype)0.5 * dt);
      timing.drift += seconds() - t0;
    }

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
    const size_t force_evals = nsteps +
                               (integrator == INTEGRATOR_KDK ? 1u : 0u);
    const double interactions = (double)force_evals * (double)global_n *
                                (double)(global_n - 1u);
    const double ginteractions = interactions / (timing.force * 1.0e9);
    printf("# final: N=%zu steps=%zu ranks=%d threads_per_rank=%d "
           "integrator=%s comm=%s kernel=%s rsqrt=%s arithmetic_dtype=%s "
           "accumulators=%s "
           "max_relative_energy_drift=%.17g "
           "tolerance=%.17g status=%s\n",
           global_n, nsteps, nranks, omp_get_max_threads(),
           integrator_name(integrator), comm_mode_name(comm_mode),
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

  particles_free(&local);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
