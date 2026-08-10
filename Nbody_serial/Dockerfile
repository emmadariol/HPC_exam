FROM ubuntu:24.04

ARG BUILD_MARCH="x86-64-v3"
ENV CFLAGS="-O3 -march=${BUILD_MARCH} -Wall -Wextra -Wpedantic"
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential openmpi-bin libopenmpi-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nbody
COPY . /opt/nbody
RUN make clean && make

ENV OMP_NUM_THREADS=1
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
CMD ["/opt/nbody/nbody_direct_hybrid", "--help"]
