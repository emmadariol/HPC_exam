#!/usr/bin/env bash

# Shared execution and parsing helpers for run_benchmarks.sh.
# This file is sourced once; it does not spawn a process per benchmark run.

detect_runtime() {
  if [[ -n "${RUNTIME:-}" ]]; then
    printf "%s" "$RUNTIME"
  elif command -v singularity >/dev/null 2>&1; then
    printf "singularity"
  elif command -v apptainer >/dev/null 2>&1; then
    printf "apptainer"
  elif command -v docker >/dev/null 2>&1; then
    printf "docker"
  else
    return 1
  fi
}

parse_solver_csv() {
  local prefix="$1"
  local format="${2:-full}"
  awk -v prefix="$prefix" -v format="$format" '
    BEGIN {
      FS = "[ =]+";
      OFS = ",";
      dtype = "unknown";
      total = io = drift = force = comm_wait = kick = energy = gpairs = max_rel_drift = status = "";
    }
    /^# final:/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "arithmetic_dtype") dtype = $(i + 1);
        else if ($i == "max_relative_energy_drift") max_rel_drift = $(i + 1);
        else if ($i == "status") status = $(i + 1);
      }
    }
    /^# timing_max_seconds/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "total") total = $(i + 1);
        else if ($i == "io") io = $(i + 1);
        else if ($i == "drift") drift = $(i + 1);
        else if ($i == "force") force = $(i + 1);
        else if ($i == "comm_wait") comm_wait = $(i + 1);
        else if ($i == "kick") kick = $(i + 1);
        else if ($i == "energy") energy = $(i + 1);
      }
    }
    /^# kernel_rate/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "gpair_interactions_per_second") gpairs = $(i + 1);
      }
    }
    END {
      if (format == "total") {
        if (total == "" || status == "") print prefix, "nan", "PARSE_FAILED", "nan";
        else print prefix, total, status, max_rel_drift;
      } else if (total == "" || force == "" || gpairs == "" || status == "") {
        print prefix, dtype, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "PARSE_FAILED", "nan";
      } else {
        print prefix, dtype, total, io, drift, force, comm_wait, kick, energy, gpairs, status, max_rel_drift;
      }
    }
  '
}

run_mpi_solver() {
  local ranks="$1"
  local threads="$2"
  shift 2
  OMP_NUM_THREADS="$threads" "$launcher" $cpu_bind --ntasks="$ranks" \
    --cpus-per-task="${SRUN_CPUS_PER_TASK:-$threads}" "$@"
}

launcher_distribution_args() {
  if [[ "$launcher" == *srun* && -n "${SRUN_NTASKS_PER_NODE:-}" ]]; then
    printf -- "--ntasks-per-node=%s\n" "$SRUN_NTASKS_PER_NODE"
  fi
}

run_container_mpi() {
  local runtime="$1"
  local image="$2"
  local ranks="$3"
  local threads="$4"
  shift 4
  local distribution_args=()
  mapfile -t distribution_args < <(launcher_distribution_args)
  case "$runtime" in
    singularity|apptainer)
      OMP_NUM_THREADS="$threads" "$launcher" $cpu_bind --ntasks="$ranks" \
        "${distribution_args[@]}" \
        --cpus-per-task="${SRUN_CPUS_PER_TASK:-$threads}" \
        "$runtime" exec "$image" "$@"
      ;;
    docker)
      docker run --rm \
        -e "OMP_NUM_THREADS=$threads" \
        -e OMPI_ALLOW_RUN_AS_ROOT=1 \
        -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
        -v "$PWD:/work" -w /work "$image" \
        mpirun --allow-run-as-root -np "$ranks" "$@"
      ;;
    *)
      echo "unknown runtime: $runtime" >&2
      return 127
      ;;
  esac
}

run_container_single() {
  local runtime="$1"
  local image="$2"
  shift 2
  case "$runtime" in
    singularity|apptainer) "$runtime" exec "$image" "$@" ;;
    docker) docker run --rm -v "$PWD:/work" -w /work "$image" "$@" ;;
    *) echo "unknown runtime: $runtime" >&2; return 127 ;;
  esac
}

run_hybrid_solver() {
  local mode="$1" runtime="$2" image="$3" ranks="$4" threads="$5" input="$6"
  shift 6
  local args=(
    --input "$input"
    --nsteps "$nsteps"
    --dt "$dt"
    --eps "$eps"
    --energy-every "$energy_every"
    --quiet
  )
  args+=("$@")
  if [[ "$mode" == "container" ]]; then
    run_container_mpi "$runtime" "$image" "$ranks" "$threads" \
      /opt/nbody/nbody_direct_hybrid "${args[@]}"
  else
    run_mpi_solver "$ranks" "$threads" ./nbody_direct_hybrid "${args[@]}"
  fi
}
