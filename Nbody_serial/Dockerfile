FROM ubuntu:24.04

ARG BUILD_MARCH="x86-64-v3"
ARG OSU_VERSION="7.5.2"
ARG OSU_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz"
ENV CFLAGS="-O3 -march=${BUILD_MARCH} -Wall -Wextra -Wpedantic"
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential openmpi-bin libopenmpi-dev ca-certificates wget && \
  rm -rf /var/lib/apt/lists/*

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

WORKDIR /opt/nbody
COPY . /opt/nbody
RUN make clean && make

ENV OMP_NUM_THREADS=1
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
CMD ["/opt/nbody/nbody_direct_hybrid", "--help"]
