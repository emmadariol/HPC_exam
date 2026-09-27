#!/usr/bin/env bash
set -euo pipefail  # stop at the first error

usage() {  # help text
  cat <<'EOF'
usage: ./run_benchmarks.sh COMMAND [options]

Commands:
  scaling     strong/weak MPI or MPI+OpenMP scaling
  hybrid      fixed-resource P x T sweep; writes one merged summary
  ablation    compare kernel, math, communication and accumulator variants
  layout      AoS-vs-SoA memory-layout benchmark
  energy      energy-diagnostic overhead benchmark
  container   native-vs-container solver overhead plus launch overhead
  osu         OSU latency/bandwidth native, container, or both
  arch        native build target comparison: -march=native vs x86-64-v3
  perf        optional perf-stat hardware-counter snapshot

Options are passed as --key value and become upper-case environment variables.
Examples:
  ./run_benchmarks.sh scaling --ranks "1 2 4 8" --threads 1 --out results.csv
  ./run_benchmarks.sh container --image nbody.sif --runtime singularity
  ./run_benchmarks.sh osu --mode both --image nbody.sif --out osu.csv
EOF
}

cmd="${1:-}"  # first argument: which benchmark
if [[ -z "$cmd" || "$cmd" == "--help" || "$cmd" == "-h" ]]; then
  usage
  exit 0
fi
shift

while [[ $# -gt 0 ]]; do  # --key value options become KEY=value variables
  case "$1" in
    --*=*)
      key="${1%%=*}"
      val="${1#*=}"
      shift
      ;;
    --*)
      key="$1"
      val="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  key="${key#--}"
  key="${key//-/_}"
  key="$(printf "%s" "$key" | tr '[:lower:]' '[:upper:]')"
  export "$key=$val"  # export the option
done

launcher="${LAUNCHER:-srun}"  # MPI launcher
cpu_bind="${CPU_BIND:---cpu-bind=verbose,cores}"  # each rank on its own cores, masks printed
model=0  # initial condition: Plummer sphere
dt="${DT:-1e-4}"  # time step
eps="${EPS:-0.05}"  # softening length
nsteps="${NSTEPS:-100}"  # steps per run
repeats="${REPEATS:-5}"  # measured repetitions
warmups="${WARMUPS:-1}"  # warm-up runs, not recorded
energy_every="${ENERGY_EVERY:-100}"  # energy check interval

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/benchmark_common.sh"  # shared helper functions

write_failed_scaling_row() {  # CSV row for a failed run
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,unknown,nan,nan,nan,nan,nan,nan,nan,nan,RUN_FAILED,nan\n" "$@"
}

bench_scaling() {  # strong and weak scaling
  local strong_n="${STRONG_N:-100000}"
  local weak_per_rank="${WEAK_PER_RANK:-10000}"
  local ranks_list="${RANKS:-1 2 4 8 16 32 64}"
  local threads_list="${THREADS:-1}"
  local comm="${COMM:-overlap}"
  local kernel="${KERNEL:-direct}"
  local rsqrt="${RSQRT:-exact}"
  local accumulators="${ACCUMULATORS:-4}"
  local out="${OUT:-benchmark_results.csv}"
  local use_container="${USE_CONTAINER:-0}"
  local image="${IMAGE:-${CONTAINER_IMAGE:-nbody.sif}}"
  local runtime=""
  if [[ "$use_container" == "1" ]]; then
    runtime="$(detect_runtime)" || { echo "no container runtime found" >&2; exit 127; }
    [[ "$runtime" == "docker" || -f "$image" ]] || { echo "missing image: $image" >&2; exit 1; }
  fi

  make nbody_direct_hybrid generate_ic >/dev/null
  printf "kind,N,nsteps,ranks,threads,repeat,integrator,comm,kernel,rsqrt,accumulators,dtype,total,io,drift,force,comm_wait,kick,energy,gpairs,status,max_rel_drift\n" > "$out"

  run_case() {  # one solver run
    local kind="$1" n="$2" ranks="$3" threads="$4" rep="$5" record="${6:-1}"
    local input="${kind}_N${n}_P${ranks}_T${threads}_seed${rep}.bin"
    local log rc prefix
    if [[ "$kernel" == "newton" && "$ranks" != "1" ]]; then
      echo "skip kernel=newton with ranks=$ranks: this variant is single-rank only" >&2
      return 0
    fi
    ./generate_ic --model "$model" --n "$n" --seed "$((1000 + rep))" --output "$input" >/dev/null  # input with a fixed seed per repetition
    set +e
    if [[ "$use_container" == "1" ]]; then
      log="$(run_hybrid_solver container "$runtime" "$image" "$ranks" "$threads" "$input" \
        --comm "$comm" --kernel "$kernel" --rsqrt "$rsqrt" \
        --accumulators "$accumulators" 2>&1)"
    else
      log="$(run_hybrid_solver native "" "" "$ranks" "$threads" "$input" \
        --comm "$comm" --kernel "$kernel" --rsqrt "$rsqrt" \
        --accumulators "$accumulators" 2>&1)"
    fi
    rc=$?
    set -e
    prefix="$kind,$n,$nsteps,$ranks,$threads,$rep,kdk,$comm,$kernel,$rsqrt,$accumulators"  # first CSV columns of the run
    if [[ -n "${RESULT_DIR:-}" ]]; then
      printf "%s\n" "$log" > "$RESULT_DIR/${kind}_N${n}_P${ranks}_T${threads}_rep${rep}_record${record}.log"
    fi
    if (( rc != 0 )); then
      if [[ "$record" == "1" ]]; then
        write_failed_scaling_row "$kind" "$n" "$nsteps" "$ranks" "$threads" "$rep" kdk "$comm" "$kernel" "$rsqrt" "$accumulators" >> "$out"
      fi
      printf "warning: scaling failed kind=%s N=%s P=%s T=%s rep=%s rc=%s\n%s\n" "$kind" "$n" "$ranks" "$threads" "$rep" "$rc" "$log" >&2
    else
      if [[ "$record" == "1" ]]; then
        printf "%s\n" "$log" | parse_solver_csv "$prefix" >> "$out"  # solver output -> CSV row
      fi
    fi
    rm -f "$input"
  }

  for kind in ${SCALING_KINDS:-strong weak}; do  # strong and/or weak
    for ranks in $ranks_list; do
      local n="$strong_n"
      [[ "$kind" == "weak" ]] && n=$((weak_per_rank * ranks))  # weak scaling: N grows with P
      for threads in $threads_list; do
        for rep in $(seq 1 "$warmups"); do run_case "$kind" "$n" "$ranks" "$threads" "$rep" 0 >/dev/null; done  # warm-up runs
        for rep in ${REP_LIST:-$(seq 1 "$repeats")}; do run_case "$kind" "$n" "$ranks" "$threads" "$rep"; done  # REP_LIST="3 4" runs only those repeats (same seeds)
      done
    done
  done
  echo "wrote $out"
}

bench_hybrid() {  # P x T mappings on the same cores
  local prefix="${RESULT_PREFIX:-results_hybrid}"
  local summary="${SUMMARY:-${prefix}_summary.csv}"
  local pairs="${HYBRID_PAIRS:-64x1 32x2 16x4 8x8 4x16}"
  rm -f "${prefix}"_P*_T*.csv "${prefix}"_P*_T*_summary.csv "$summary"
  for pair in $pairs; do
    local ranks="${pair%x*}"
    local threads="${pair#*x}"
    local raw="${prefix}_P${ranks}_T${threads}.csv"
    local partial="${prefix}_P${ranks}_T${threads}_summary.csv"
    RANKS="$ranks" THREADS="$threads" SRUN_CPUS_PER_TASK="$threads" OUT="$raw" bench_scaling  # run this P x T pair
    python3 analyze.py summarize scaling "$raw" "$partial"  # summary of this pair
    if [[ ! -s "$summary" ]]; then cp "$partial" "$summary"; else tail -n +2 "$partial" >> "$summary"; fi
  done
  python3 analyze.py plot hybrid "$summary" "$prefix"  # plot all pairs
}

bench_ablation() {  # Newton, rsqrt, communication, partial sums
  local ranks="${RANKS:-64}"
  local n="${N:-10000}"
  local out="${OUT:-results_ablation.csv}"
  local input="ic_ablation_N${n}.bin"
  make nbody_direct_hybrid generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-123}" --output "$input" >/dev/null
  printf "Test_Type,Config,Time_Sec,N,nsteps,ranks,threads,repeat,dt,eps,energy_every,force,comm_wait,gpairs,status,max_rel_drift\n" > "$out"

  ablation_case() {  # one variant, one run
    local test_type="$1" config="$2" ranks="$3"
    shift 3
    local log rc time_sec
    set +e
    log="$(run_hybrid_solver native "" "" "$ranks" "${THREADS:-1}" "$input" "$@" 2>&1)"
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf "%s,%s,nan,%s,%s,%s,%s,%s,%s,%s,%s,nan,nan,nan,RUN_FAILED,nan\n" \
        "$test_type" "$config" "$n" "$nsteps" "$ranks" "${THREADS:-1}" "$rep" "$dt" "$eps" "$energy_every" >> "$out"
      printf "warning: ablation %s/%s failed rc=%s\n%s\n" "$test_type" "$config" "$rc" "$log" >&2
    else
      time_sec="$(printf "%s\n" "$log" | awk 'BEGIN{FS="[ =]+"} /^# timing_max_seconds/ {for(i=1;i<=NF;i++) if($i=="total") print $(i+1)}')"
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s," \
        "$test_type" "$config" "${time_sec:-nan}" "$n" "$nsteps" "$ranks" "${THREADS:-1}" "$rep" "$dt" "$eps" "$energy_every" >> "$out"
      printf "%s\n" "$log" | awk '
        BEGIN { FS="[ =]+"; force=wait=rate=drift="nan"; status="PARSE_FAILED" }
        /^# timing_max_seconds/ { for(i=1;i<NF;i++) {
          if($i=="force") force=$(i+1); if($i=="comm_wait") wait=$(i+1) } }
        /^# kernel_rate/ { for(i=1;i<NF;i++) if($i=="gpair_interactions_per_second") rate=$(i+1) }
        /^# final:/ { for(i=1;i<NF;i++) {
          if($i=="status") status=$(i+1); if($i=="max_relative_energy_drift") drift=$(i+1) } }
        END { printf "%s,%s,%s,%s,%s\n",force,wait,rate,status,drift }
      ' >> "$out"
    fi
  }

  for rep in $(seq 1 "$repeats"); do ablation_case Kernel direct 1 --kernel direct; done  # direct kernel
  for rep in $(seq 1 "$repeats"); do ablation_case Kernel newton 1 --kernel newton; done  # Newton's third law
  for rep in $(seq 1 "$repeats"); do ablation_case Math exact "$ranks" --rsqrt exact; done  # exact square root
  for rep in $(seq 1 "$repeats"); do ablation_case Math approx1 "$ranks" --rsqrt approx1; done  # rsqrt14 + 1 Newton step
  for rep in $(seq 1 "$repeats"); do ablation_case Math approx2 "$ranks" --rsqrt approx2; done  # rsqrt14 + 2 Newton steps
  for rep in $(seq 1 "$repeats"); do ablation_case Comm sendrecv "$ranks" --comm sendrecv; done  # blocking ring
  for rep in $(seq 1 "$repeats"); do ablation_case Comm overlap "$ranks" --comm overlap; done  # overlapped ring
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 1 "$ranks" --accumulators 1; done  # 1 partial sum
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 2 "$ranks" --accumulators 2; done
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 4 "$ranks" --accumulators 4; done
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 8 "$ranks" --accumulators 8; done  # 8 partial sums
  rm -f "$input"
  echo "wrote $out"
}

bench_layout() {  # AoS vs SoA
  local n="${N:-10000}"
  local threads_list="${THREADS:-1 2 4 8}"
  local inner_repeats="${INNER_REPEATS:-3}"
  local rsqrt="${RSQRT:-exact}"
  local out="${OUT:-layout_results.csv}"
  local input="${INPUT:-layout_N${n}.bin}"
  make nbody_layout_benchmark generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-4242}" --output "$input" >/dev/null
  printf "layout,N,threads,repeat,warmups,inner_repeats,rsqrt,force,gpairs,checksum\n" > "$out"
  for threads in $threads_list; do
    for rep in $(seq 1 "$repeats"); do
      OMP_NUM_THREADS="$threads" ./nbody_layout_benchmark \
        --input "$input" --layout both --warmups "$warmups" \
        --inner-repeats "$inner_repeats" --eps "$eps" --rsqrt "$rsqrt" |
        awk -F, -v rep="$rep" 'BEGIN{OFS=","}{print $1,$2,$3,rep,$4,$5,$6,$7,$8,$9}' >> "$out"
    done
  done
  rm -f "$input"
  echo "wrote $out"
}

bench_energy() {  # cost of the energy check
  local n="${N:-10000}"
  local ranks="${RANKS:-8}"
  local threads="${THREADS:-1}"
  local list="${ENERGY_LIST:-}"
  local out="${OUT:-energy_overhead.csv}"
  local input="${INPUT:-energy_N${n}.bin}"
  make nbody_direct_hybrid generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-5151}" --output "$input" >/dev/null
  printf "N,nsteps,ranks,threads,repeat,energy_every,total,force,energy,status,max_rel_drift\n" > "$out"
  if [[ -z "$list" ]]; then
    list="1 5 10 $nsteps"  # default energy-every values
  fi
  list="$(for ee in $list; do if (( ee > nsteps )); then echo "$nsteps"; else echo "$ee"; fi; done | awk '!seen[$0]++')"
  for ee in $list; do
    for rep in $(seq 1 "$warmups"); do
      energy_every="$ee" run_hybrid_solver native "" "" "$ranks" "$threads" "$input" >/dev/null 2>&1 || true
    done
    for rep in $(seq 1 "$repeats"); do
      set +e
      log="$(energy_every="$ee" run_hybrid_solver native "" "" "$ranks" "$threads" "$input" 2>&1)"
      rc=$?
      set -e
      if (( rc != 0 )); then
        printf "%s,%s,%s,%s,%s,%s,nan,nan,nan,RUN_FAILED,nan\n" "$n" "$nsteps" "$ranks" "$threads" "$rep" "$ee" >> "$out"
        printf "warning: energy failed energy_every=%s rep=%s rc=%s\n%s\n" "$ee" "$rep" "$rc" "$log" >&2
      else
        printf "%s\n" "$log" | awk -v p="$n,$nsteps,$ranks,$threads,$rep,$ee" '
          BEGIN{FS="[ =]+";OFS=",";total=force=energy=status=drift=""}
          /^# final:/ {for(i=1;i<=NF;i++){if($i=="max_relative_energy_drift") drift=$(i+1); else if($i=="status") status=$(i+1)}}
          /^# timing_max_seconds/ {for(i=1;i<=NF;i++){if($i=="total") total=$(i+1); else if($i=="force") force=$(i+1); else if($i=="energy") energy=$(i+1)}}
          END{if(total==""||force==""||energy==""||status=="") print p,"nan","nan","nan","PARSE_FAILED","nan"; else print p,total,force,energy,status,drift}
        ' >> "$out"
      fi
    done
  done
  rm -f "$input"
  echo "wrote $out"
}

bench_container() {  # native vs container
  local image="${IMAGE:-nbody.sif}"
  local out="${OUT:-container_overhead.csv}"
  local launch_out="${LAUNCH_OUT:-container_launch_overhead.csv}"
  local ranks_list="${RANKS:-1 2 4}"
  local threads="${THREADS:-1}"
  local strong_n="${N:-100000}"
  local n_per_rank="${N_PER_RANK:-10000}"
  local launch_repeats="${LAUNCH_REPEATS:-10}"
  local runtime
  runtime="$(detect_runtime)" || { echo "no container runtime found" >&2; exit 127; }
  [[ "$runtime" == "docker" || -f "$image" ]] || { echo "missing image: $image" >&2; exit 1; }
  make nbody_direct_hybrid generate_ic >/dev/null
  if [[ "$runtime" == "singularity" || "$runtime" == "apptainer" ]]; then
    write_mpi_linkage_check "$runtime" "$image" ./nbody_direct_hybrid /opt/nbody/nbody_direct_hybrid \
      "${MPI_LINKAGE_OUT:-$(dirname "$out")/mpi_linkage_check.txt}"
  fi
  printf "kind,mode,N,ranks,threads,repeat,total,status,max_rel_drift\n" > "$out"
  container_case() {
    local kind="$1" mode="$2" n="$3" ranks="$4" rep="$5"
    local input="container_${kind}_N${n}_P${ranks}_seed${rep}.bin"
    local log rc prefix
    ./generate_ic --model "$model" --n "$n" --seed "$((7700 + rep))" --output "$input" >/dev/null
    set +e
    if [[ "$mode" == "native" ]]; then
      log="$(run_hybrid_solver native "" "" "$ranks" "$threads" "$input" --comm sendrecv 2>&1)"
    else
      log="$(run_hybrid_solver container "$runtime" "$image" "$ranks" "$threads" "$input" --comm sendrecv 2>&1)"
    fi
    rc=$?
    set -e
    prefix="$kind,$mode,$n,$ranks,$threads,$rep"
    if (( rc != 0 )); then
      printf "%s,nan,RUN_FAILED,nan\n" "$prefix" >> "$out"
      printf "warning: container case failed %s rc=%s\n%s\n" "$prefix" "$rc" "$log" >&2
    else
      printf "%s\n" "$log" | parse_solver_csv "$prefix" total >> "$out"
    fi
    rm -f "$input"
  }
  for ranks in $ranks_list; do
    for rep in $(seq 1 "$repeats"); do
      container_case strong native "$strong_n" "$ranks" "$rep"  # strong, native
      container_case strong container "$strong_n" "$ranks" "$rep"  # strong, container
      container_case weak native "$((n_per_rank * ranks))" "$ranks" "$rep"  # weak, native
      container_case weak container "$((n_per_rank * ranks))" "$ranks" "$rep"  # weak, container
    done
  done
  printf "repeat,seconds\n" > "$launch_out"  # container start-up times
  for rep in $(seq 1 "$launch_repeats"); do
    seconds="$( { time -p run_container_single "$runtime" "$image" true; } 2>&1 | awk '/^real /{print $2}' )" || seconds="nan"
    printf "%s,%s\n" "$rep" "${seconds:-nan}" >> "$launch_out"
  done
  echo "wrote $out and $launch_out"
}

parse_osu() {  # OSU output -> CSV
  local mode_name="$1" bench="$2" metric="$3"
  awk -v mode="$mode_name" -v bench="$bench" -v metric="$metric" '
    BEGIN { OFS="," }
    /^[[:space:]]*[0-9]+[[:space:]]+/ { print mode, bench, metric, $1, $2 }
  '
}

resolve_command_path() {
  local tool="$1"
  if command -v "$tool" >/dev/null 2>&1; then
    command -v "$tool"
  else
    printf "%s\n" "$tool"
  fi
}

extract_lib_path() {
  local lib_regex="$1"
  awk -v lib_regex="$lib_regex" '
    $0 ~ lib_regex && $0 ~ /=>/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "=>") {
          print $(i + 1);
          exit;
        }
      }
      print $1;
      exit;
    }
  '
}

write_mpi_linkage_check() {  # compare the MPI library used native vs container
  local runtime="$1" image="$2" native_tool="$3" container_tool="$4" out="$5"
  local native_path native_ldd container_ldd native_libmpi container_libmpi
  local interesting='libmpi|libopen-rte|libopen-pal|libpmix|libucp|libucs|libuct|libucm|libfabric|libpsm|libibverbs|libhwloc'

  mkdir -p "$(dirname "$out")"
  native_path="$(resolve_command_path "$native_tool")"
  native_ldd="$(ldd "$native_path" 2>&1 || true)"  # libraries of the native program
  container_ldd="$(
    run_container_single "$runtime" "$image" sh -lc '
      tool="$1"
      if command -v "$tool" >/dev/null 2>&1; then
        resolved="$(command -v "$tool")"
      else
        resolved="$tool"
      fi
      echo "container_resolved_tool=$resolved"
      ldd "$resolved"
    ' sh "$container_tool" 2>&1 || true
  )"

  {
    printf "== native tool ==\n%s\n\n" "$native_path"
    printf "== native selected ldd ==\n"
    printf "%s\n" "$native_ldd" | grep -E "$interesting" || true
    printf "\n== container tool ==\n%s\n\n" "$container_tool"
    printf "== container selected ldd ==\n"
    printf "%s\n" "$container_ldd" | grep -E "container_resolved_tool|$interesting" || true
  } > "$out"

  native_libmpi="$(printf "%s\n" "$native_ldd" | extract_lib_path 'libmpi\.so')"  # libmpi used natively
  container_libmpi="$(printf "%s\n" "$container_ldd" | extract_lib_path 'libmpi\.so')"  # libmpi used in the container

  if printf "%s\n" "$container_ldd" | grep -qE "GLIBC_[0-9.]+.*not found|version .* not found"; then
    {
      printf "\n== linkage verdict ==\n"
      printf "FAIL: container C library is too old for the injected host MPI stack.\n"
      printf "The image must be rebuilt from a base distribution with a newer glibc, e.g. Ubuntu 24.04 for Orfeo host OpenMPI requiring GLIBC_2.38.\n"
    } >> "$out"
    printf "ERROR: container glibc is too old for host MPI. See %s\n" "$out" >&2
    return 1
  fi

  if [[ -z "$native_libmpi" || -z "$container_libmpi" ]]; then
    printf "ERROR: unable to resolve libmpi.so in native/container ldd. See %s\n" "$out" >&2
    return 1
  fi
  if [[ "$native_libmpi" != "$container_libmpi" ]]; then
    {
      printf "\n== linkage verdict ==\n"
      printf "FAIL: native and container libmpi.so do not match.\n"
      printf "native_libmpi=%s\n" "$native_libmpi"
      printf "container_libmpi=%s\n" "$container_libmpi"
    } >> "$out"
    printf "ERROR: MPI linkage mismatch. See %s\n" "$out" >&2
    return 1
  fi

  {
    printf "\n== linkage verdict ==\n"
    printf "OK: native and container libmpi.so match exactly.\n"
    printf "libmpi=%s\n" "$native_libmpi"
  } >> "$out"
}

bench_osu() {  # OSU latency and bandwidth
  local mode="${MODE:-both}"
  local image="${IMAGE:-nbody.sif}"
  local out="${OUT:-osu_microbench.csv}"
  local latency="${OSU_LATENCY:-osu_latency}"
  local bw="${OSU_BW:-osu_bw}"
  local container_latency="${CONTAINER_OSU_LATENCY:-osu_latency}"
  local container_bw="${CONTAINER_OSU_BW:-osu_bw}"
  local runtime
  runtime="$(detect_runtime 2>/dev/null || true)"
  export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"  # Open MPI point-to-point layer
  export OMPI_MCA_btl="${OMPI_MCA_btl:-self,tcp}"  # transports: self and TCP
  export OMPI_MCA_btl_vader_single_copy_mechanism="${OMPI_MCA_btl_vader_single_copy_mechanism:-none}"
  export SINGULARITYENV_OMPI_MCA_pml="$OMPI_MCA_pml"
  export APPTAINERENV_OMPI_MCA_pml="$OMPI_MCA_pml"
  export SINGULARITYENV_OMPI_MCA_btl="$OMPI_MCA_btl"
  export APPTAINERENV_OMPI_MCA_btl="$OMPI_MCA_btl"
  export SINGULARITYENV_OMPI_MCA_btl_vader_single_copy_mechanism="$OMPI_MCA_btl_vader_single_copy_mechanism"
  export APPTAINERENV_OMPI_MCA_btl_vader_single_copy_mechanism="$OMPI_MCA_btl_vader_single_copy_mechanism"
  printf "mode,benchmark,metric,bytes,value\n" > "$out"
  run_osu_one() {
    local mode_name="$1" bench="$2" metric="$3" tool="$4" record="${5:-1}"
    local log rc
    set +e
    if [[ "$mode_name" == "native" ]]; then
      local distribution_args=()
      local saved_ntasks_per_node="${SRUN_NTASKS_PER_NODE:-}"
      SRUN_NTASKS_PER_NODE="${OSU_NTASKS_PER_NODE:-}"
      mapfile -t distribution_args < <(launcher_distribution_args)
      SRUN_NTASKS_PER_NODE="$saved_ntasks_per_node"
      log="$("$launcher" $cpu_bind -n 2 "${distribution_args[@]}" "$tool" 2>&1)"  # native: 2 processes
    else
      log="$(SRUN_NTASKS_PER_NODE="${OSU_NTASKS_PER_NODE:-}" \
        run_container_mpi "$runtime" "$image" 2 1 "$tool" 2>&1)"
    fi
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf "warning: OSU %s/%s failed rc=%s\n%s\n" "$mode_name" "$bench" "$rc" "$log" >&2
    elif [[ "$record" == "1" ]]; then
      printf "%s\n" "$log" | parse_osu "$mode_name" "$bench" "$metric" >> "$out"
    fi
  }
  run_osu_repeated() {
    local mode_name="$1" bench="$2" metric="$3" tool="$4"
    local rep
    for rep in $(seq 1 "$warmups"); do
      run_osu_one "$mode_name" "$bench" "$metric" "$tool" 0
    done
    for rep in $(seq 1 "$repeats"); do
      run_osu_one "$mode_name" "$bench" "$metric" "$tool" 1
    done
  }
  if [[ "$mode" == "native" || "$mode" == "both" ]]; then
    command -v "$latency" >/dev/null 2>&1 || [[ -x "$latency" ]] || { echo "missing $latency" >&2; exit 1; }
    command -v "$bw" >/dev/null 2>&1 || [[ -x "$bw" ]] || { echo "missing $bw" >&2; exit 1; }
    if [[ "$mode" == "both" ]]; then
      [[ -n "$runtime" ]] || { echo "no container runtime found" >&2; exit 127; }
      write_mpi_linkage_check "$runtime" "$image" "$latency" "$container_latency" \
        "${MPI_LINKAGE_OUT:-$(dirname "$out")/osu_mpi_linkage_check.txt}"
    fi
    run_osu_repeated native latency latency_us "$latency"  # native latency
    run_osu_repeated native bandwidth bandwidth_MBps "$bw"  # native bandwidth
  fi
  if [[ "$mode" == "container" || "$mode" == "both" ]]; then
    [[ -n "$runtime" ]] || { echo "no container runtime found" >&2; exit 127; }
    if [[ "$mode" == "container" ]]; then
      write_mpi_linkage_check "$runtime" "$image" "${OSU_NATIVE_REFERENCE:-$latency}" "$container_latency" \
        "${MPI_LINKAGE_OUT:-$(dirname "$out")/osu_mpi_linkage_check.txt}"
    fi
    run_osu_repeated container latency latency_us "$container_latency"  # container latency
    run_osu_repeated container bandwidth bandwidth_MBps "$container_bw"  # container bandwidth
  fi
  echo "wrote $out"
}

bench_arch() {  # -march=native vs -march=x86-64-v3
  local out="${OUT:-arch_target_comparison.csv}"
  local n="${N:-100000}"
  local ranks="${RANKS:-64}"
  local threads="${THREADS:-1}"
  local base_cflags="${BASE_CFLAGS:--O3 -Wall -Wextra -Wpedantic}"
  local input="arch_compare_N${n}.bin"
  printf "target,N,nsteps,ranks,threads,repeat,total,status,max_rel_drift\n" > "$out"
  for target in native x86-64-v3; do  # the two compilation targets
    make clean >/dev/null
    CFLAGS="$base_cflags -march=$target" make nbody_direct_hybrid generate_ic >/dev/null  # build for this target
    ./generate_ic --model "$model" --n "$n" --seed "${SEED:-5151}" --output "$input" >/dev/null
    for rep in $(seq 1 "$warmups"); do
      run_hybrid_solver native "" "" "$ranks" "$threads" "$input" \
        --comm overlap --kernel direct --rsqrt "${RSQRT:-exact}" --accumulators 4 >/dev/null 2>&1 || true
    done
    for rep in $(seq 1 "$repeats"); do
      local log rc prefix
      set +e
      log="$(run_hybrid_solver native "" "" "$ranks" "$threads" "$input" \
        --comm overlap --kernel direct --rsqrt "${RSQRT:-exact}" --accumulators 4 2>&1)"
      rc=$?
      set -e
      prefix="$target,$n,$nsteps,$ranks,$threads,$rep"
      if (( rc != 0 )); then
        printf "%s,nan,RUN_FAILED,nan\n" "$prefix" >> "$out"
        printf "warning: arch target=%s failed rep=%s rc=%s\n%s\n" "$target" "$rep" "$rc" "$log" >&2
      else
        printf "%s\n" "$log" | parse_solver_csv "$prefix" total >> "$out"
      fi
    done
  done
  rm -f "$input"
  make clean >/dev/null
  make nbody_direct_hybrid generate_ic >/dev/null
  echo "wrote $out"
}

bench_perf() {  # hardware counters with perf
  local input="${INPUT:-perf_counter_input.bin}"
  local n="${N:-20000}"
  local ranks="${RANKS:-1}"
  local threads="${THREADS:-1}"
  local out="${OUT:-perf_counters.txt}"
  local events="${EVENTS:-cycles,instructions,cache-references,cache-misses,branches,branch-misses}"
  command -v perf >/dev/null 2>&1 || { echo "perf is not available in PATH" >&2; exit 127; }
  make all >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed 4242 --output "$input" >/dev/null
  if [[ "${PERF_TARGET:-solver}" == "layout" ]]; then
    for layout in aos soa; do
      for rep in $(seq 1 "$repeats"); do
        OMP_NUM_THREADS="$threads" "$launcher" --ntasks=1 --cpus-per-task="$threads" $cpu_bind \
          perf stat -x ';' -e "$events" -o "${out}_${layout}_rep${rep}.csv" \
          ./nbody_layout_benchmark --input "$input" --layout "$layout" \
          --warmups "$warmups" --inner-repeats "${INNER_REPEATS:-3}" --eps "$eps" --rsqrt exact \
          > "${out}_${layout}_rep${rep}.out"
      done
    done
  else
    OMP_NUM_THREADS="$threads" "$launcher" --ntasks="$ranks" --cpus-per-task="$threads" $cpu_bind \
      bash -c 'dest="$1"; events="$2"; shift 2; exec perf stat -e "$events" -o "${dest}_rank${SLURM_PROCID:-0}.txt" "$@"' bash "$out" "$events" \
      ./nbody_direct_hybrid --input "$input" --nsteps "$nsteps" \
      --dt "$dt" --eps "$eps" --energy-every "$nsteps" \
      --comm sendrecv --kernel direct --rsqrt exact --quiet
  fi
  rm -f "$input"
}

case "$cmd" in  # run the chosen benchmark
  scaling) bench_scaling ;;
  hybrid) bench_hybrid ;;
  ablation) bench_ablation ;;
  layout) bench_layout ;;
  energy) bench_energy ;;
  container) bench_container ;;
  osu) bench_osu ;;
  arch) bench_arch ;;
  perf) bench_perf ;;
  *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
