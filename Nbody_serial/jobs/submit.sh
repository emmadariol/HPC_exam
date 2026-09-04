#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: jobs/submit.sh --cluster orfeo|leonardo --bench BENCH [options]

BENCH:
  probe       collect hardware/software information
  scaling     MPI strong/weak scaling
  hybrid      MPI+OpenMP P x T sweep
  ablation    optimisation ablation study
  evidence    layout + energy evidence
  container   native-vs-Singularity container overhead
  osu         OSU latency/bandwidth native-vs-container

Common options:
  --partition NAME      override partition
  --account NAME        override account
  --qos NAME            override qos
  --nodes N             default: 1
  --cpus N              Slurm --cpus-per-task/--ntasks total budget helper
  --time HH:MM:SS       override time limit
  --exclusive           request full node
  --afterok JOBID       add dependency afterok:JOBID
  --result-dir DIR      output directory, default runs/<cluster>_<bench>_<timestamp>
  --name NAME           Slurm job name
  -- VAR=VALUE ...      extra environment assignments passed to the job

Examples:
  jobs/submit.sh --cluster orfeo --bench scaling --partition GENOA --cpus 64 --time 01:59:00 -- RANKS="1 2 4 8 16 32 64"
  jobs/submit.sh --cluster orfeo --bench container --partition GENOA --cpus 4 -- IMAGE=nbody.sif
EOF
}

cluster=""
bench=""
partition=""
account=""
qos=""
nodes="1"
cpus=""
time_limit=""
exclusive="0"
dependency=""
result_dir=""
job_name=""
extra_env=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster) cluster="$2"; shift 2 ;;
    --bench) bench="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --account) account="$2"; shift 2 ;;
    --qos) qos="$2"; shift 2 ;;
    --nodes) nodes="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --exclusive) exclusive="1"; shift ;;
    --afterok) dependency="$2"; shift 2 ;;
    --result-dir) result_dir="$2"; shift 2 ;;
    --name) job_name="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; extra_env=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$cluster" && -n "$bench" ]] || { usage >&2; exit 2; }

case "$cluster" in
  orfeo)
    account="${account:-dssc}"
    partition="${partition:-GENOA}"
    qos="${qos:-normal}"
    module_block='module purge; module load openMPI/4.1.6; module load singularity/4.3.1 2>/dev/null || true'
    ;;
  leonardo)
    account="${account:-uTS26_Tornator_0}"
    partition="${partition:-dcgp_usr_prod}"
    qos="${qos:-dcgp_qos_bprod}"
    module_block='module load profile/base 2>/dev/null || true; module purge; module load profile/base 2>/dev/null || true; module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2 2>/dev/null || module load openmpi 2>/dev/null || true'
    ;;
  *)
    echo "unknown cluster: $cluster" >&2
    exit 2
    ;;
esac

case "$bench" in
  probe) default_time="00:10:00"; default_cpus="2" ;;
  scaling) default_time="02:00:00"; default_cpus="64" ;;
  hybrid) default_time="02:00:00"; default_cpus="64" ;;
  ablation) default_time="01:30:00"; default_cpus="64" ;;
  evidence) default_time="01:30:00"; default_cpus="8" ;;
  container) default_time="01:00:00"; default_cpus="4" ;;
  osu) default_time="00:20:00"; default_cpus="2" ;;
  *) echo "unknown bench: $bench" >&2; usage >&2; exit 2 ;;
esac

time_limit="${time_limit:-$default_time}"
cpus="${cpus:-$default_cpus}"
job_name="${job_name:-${bench}_${cluster}}"
timestamp="$(date +%Y%m%d_%H%M%S)"
result_dir="${result_dir:-runs/${cluster}_${bench}_${timestamp}}"

mkdir -p "$result_dir"

case "$bench" in
  probe)
    payload='bash ./collect_system_info.sh "$RESULT_DIR/system_info.txt"'
    ;;
  scaling)
    payload='OUT="${RESULT_DIR}/scaling.csv" bash ./run_benchmarks.sh scaling; python3 analyze.py summarize scaling "$RESULT_DIR/scaling.csv" "$RESULT_DIR/scaling_summary.csv"; python3 analyze.py plot scaling "$RESULT_DIR/scaling_summary.csv" "$RESULT_DIR/scaling"'
    ;;
  hybrid)
    payload='RESULT_PREFIX="${RESULT_DIR}/hybrid" bash ./run_benchmarks.sh hybrid'
    ;;
  ablation)
    payload='OUT="${RESULT_DIR}/ablation.csv" bash ./run_benchmarks.sh ablation; python3 analyze.py plot ablation "$RESULT_DIR/ablation.csv" "$RESULT_DIR/ablation"'
    ;;
  evidence)
    payload='bash ./collect_system_info.sh "$RESULT_DIR/system_info.txt"; OUT="$RESULT_DIR/layout.csv" bash ./run_benchmarks.sh layout; python3 analyze.py summarize layout "$RESULT_DIR/layout.csv" "$RESULT_DIR/layout_summary.csv"; python3 analyze.py plot layout "$RESULT_DIR/layout_summary.csv" "$RESULT_DIR/layout_force_time"; OUT="$RESULT_DIR/energy.csv" bash ./run_benchmarks.sh energy; python3 analyze.py summarize energy "$RESULT_DIR/energy.csv" "$RESULT_DIR/energy_summary.csv"; python3 analyze.py plot energy "$RESULT_DIR/energy_summary.csv" "$RESULT_DIR/energy_overhead"'
    ;;
  container)
    payload='if [[ ! -f "${IMAGE:-nbody.sif}" && -n "${IMAGE_URI:-docker://memid01/nbody-hpc:latest}" ]]; then singularity pull --force "${IMAGE:-nbody.sif}" "${IMAGE_URI:-docker://memid01/nbody-hpc:latest}"; fi; OUT="$RESULT_DIR/container_overhead.csv" LAUNCH_OUT="$RESULT_DIR/container_overhead_launch.csv" bash ./run_benchmarks.sh container; python3 analyze.py summarize container "$RESULT_DIR/container_overhead.csv" "$RESULT_DIR/container_overhead_summary.csv"; python3 analyze.py plot container "$RESULT_DIR/container_overhead_summary.csv" "$RESULT_DIR/container_overhead"'
    ;;
  osu)
    payload='if [[ ! -f "${IMAGE:-nbody.sif}" && -n "${IMAGE_URI:-docker://memid01/nbody-hpc:latest}" ]]; then singularity pull --force "${IMAGE:-nbody.sif}" "${IMAGE_URI:-docker://memid01/nbody-hpc:latest}"; fi; OUT="$RESULT_DIR/osu_microbench_native_vs_container.csv" bash ./run_benchmarks.sh osu --mode both; python3 analyze.py summarize osu "$RESULT_DIR/osu_microbench_native_vs_container.csv" "$RESULT_DIR/osu_microbench_summary.csv"; python3 analyze.py plot osu "$RESULT_DIR/osu_microbench_summary.csv" "$RESULT_DIR/osu_microbench"'
    ;;
esac

printf -v quoted_result_dir "%q" "$result_dir"
env_block="RESULT_DIR=$quoted_result_dir"
for item in "${extra_env[@]}"; do
  if [[ "$item" != *=* ]]; then
    echo "extra environment item must be VAR=VALUE, got: $item" >&2
    exit 2
  fi
  env_key="${item%%=*}"
  env_val="${item#*=}"
  printf -v quoted_env_val "%q" "$env_val"
  env_block="$env_block $env_key=$quoted_env_val"
done

printf -v quoted_pwd "%q" "$PWD"
job_script="$result_dir/job_${bench}.sh"
cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd $quoted_pwd
$module_block
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
$env_block
$payload
EOF

sbatch_args=(
  --account="$account"
  --partition="$partition"
  --qos="$qos"
  --nodes="$nodes"
  --ntasks="$cpus"
  --cpus-per-task=1
  --time="$time_limit"
  --job-name="$job_name"
  --output="$result_dir/slurm_%j.out"
  --error="$result_dir/slurm_%j.err"
)

[[ "$exclusive" == "1" ]] && sbatch_args+=(--exclusive)
[[ -n "$dependency" ]] && sbatch_args+=(--dependency="afterok:$dependency")
[[ "$cluster" == "leonardo" ]] && sbatch_args+=(--gres=tmpfs:10g)

job_id="$(sbatch --parsable "${sbatch_args[@]}" "$job_script")"
printf "%s\t%s\t%s\n" "$bench" "$job_id" "$result_dir" | tee -a "$result_dir/jobs.tsv"
echo "$result_dir" > "LAST_${cluster^^}_RUN.txt"
