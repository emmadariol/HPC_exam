# Curated final results

This directory contains only the final, documented data used by
`FINAL_REPORT.md`.
The large `runs/` tree remains local execution history and is ignored by Git.

The retained material is organised as follows:

- `scaling_64*`: final MPI strong/weak scaling and runtime, speedup,
  efficiency and communication-bandwidth figures.
- `mapping_numa/`, `mapping_socket/`, `mapping_core/`: the verified
  64-core MPI/OpenMP placement comparison.
- `ablation_complete_20260922_083515/`: the complete five-repetition
  optimisation campaign, including exact/approximate rsqrt, Newton reuse,
  communication mode and accumulator-chain variants.
- `layout*` and `energy*`: final AoS/SoA and energy-diagnostic results.
- `arch_target_comparison*`: native `-march=native` versus
  `-march=x86-64-v3`.
- `required_table/`: the assignment-compliant native/container strong and
  weak scaling table, launch overhead, OSU latency/bandwidth measurements,
  two-node allocation evidence and host-MPI linkage check.
- `system_info_*.txt` and `vectorization_*.txt`: reproducibility and
  compiler-vectorisation evidence.

The report now uses `scaling_64_all_runs_summary.csv` and
`scaling_64_all_runs_*.svg`. They retain all five successful runs per point.
The earlier MAD-filtered summary and figures remain available for comparison;
the raw measurements in `scaling_64.csv` are unchanged.

To reproduce the current scaling results from the project directory:

```sh
KEEP_ALL_REPETITIONS=1 python3 analyze.py summarize scaling results_final/scaling_64.csv results_final/scaling_64_all_runs_summary.csv
python3 analyze.py plot scaling results_final/scaling_64_all_runs_summary.csv results_final/scaling_64_all_runs
```

OSU table standard deviations come from
`required_table/osu_microbench_summary.csv`. Launch statistics use all ten
entries in `required_table/container_launch_overhead.csv`; the nine later
launches are also described separately. Software checks are saved in
`software_provenance/`.
