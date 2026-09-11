# Docker build used to create the portable container image locally.  
FROM ubuntu:22.04

# BUILD_MARCH defaults to x86-64-v3 so that the image stays portable across
# modern HPC CPU nodes while still enabling useful vector instructions.
ARG BUILD_MARCH="x86-64-v3"
ARG OSU_VERSION="7.5.2"
ARG OSU_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz"
ENV CFLAGS="-O3 -march=${BUILD_MARCH} -Wall -Wextra -Wpedantic"

# Install only the build/runtime dependencies needed by the solvers and OSU
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential openmpi-bin libopenmpi-dev ca-certificates wget && \
  rm -rf /var/lib/apt/lists/*

# Build OSU latency/bandwidth tests inside the image.  
# They are required to compare native MPI communication with 
# the same benchmark executed from the container.
RUN wget -O /tmp/osu.tar.gz "$OSU_URL" && \
  mkdir -p /tmp/osu && \
  tar -xzf /tmp/osu.tar.gz -C /tmp/osu --strip-components=1 && \
  cd /tmp/osu && \
  ./configure CC=mpicc --prefix=/opt/osu && \
  make -j"$(nproc)" && \
  make install && \
  ln -sf /opt/osu/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency /usr/local/bin/osu_latency && \
  ln -sf /opt/osu/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw /usr/local/bin/osu_bw && \
  rm -rf /tmp/osu /tmp/osu.tar.gz

# Copy the project source and compile all maintained executables in /opt/nbody.
WORKDIR /opt/nbody
COPY . /opt/nbody
RUN make clean && make

# Root execution variables are needed only for Docker smoke tests; Singularity
# runs on the cluster use the normal user identity.
ENV OMP_NUM_THREADS=1
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# The default command is non-destructive and proves that the solver exists.
CMD ["/opt/nbody/nbody_direct_hybrid", "--help"]
