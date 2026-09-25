# results_final

Data behind `FINAL_REPORT.md`: only the CSV files and the 16 figures used in the report.
Slurm outputs, per-run logs, job scripts and build logs were removed.

| Report experiment | Data |
|---|---|
| 1. Energy conservation | column `max_rel_drift` in every CSV; long run in `energy_long_20260923_170127/` |
| 2. Cost of the energy check | `energy.csv`, `energy_summary.csv`, `energy_overhead.svg` |
| 3. Phase breakdown and memory bandwidth | `scaling_100steps/strong_P1_*`, `scaling_100steps/strong_P8_32`, `mapping_*`, `energy_long_*`; `memory_bandwidth.csv`, `memory_bandwidth_summary.csv` |
| 4. Strong scaling (N = 100000, 100 steps) | `scaling_100steps/strong_*/` (native); summary `native_strong_summary.csv` and figures `strong_*.svg` in `scaling_100steps/` |
| 5. Weak scaling (10000 per rank, 100 steps) | `scaling_100steps/weak_*/`; summary `native_weak_summary.csv`, figures `weak_*.svg` |
| 6. MPI/OpenMP mapping | `mapping_numa/`, `mapping_socket/`, `mapping_core/` |
| 7-9. Newton, rsqrt, accumulators (T = 1) | `ablation_complete_20260922_083515/` |
| 7-9. Same tests with T = 2, 4, 8, 16 | `ablation_T{2,4,8,16}_20260923_165910/`; summary and figure in `thread_sweep/` |
| 8. rsqrt time-step convergence | `rsqrt_conv_<method>_n<steps>_20260923_170127/`; summary and figure in `rsqrt_convergence/` |
| 10. AoS vs SoA | `layout.csv`, `layout_summary.csv`, `layout_force_time.svg` |
| 11. native vs x86-64-v3 | `arch_target_comparison*.csv` (N=10000), `arch_exact_20260923_170254/`, `arch_approx1_20260924_134326/` (N=100000) |
| 12. Blocking vs overlapped ring | `overlap_20260923_132533/` (summary CSV and figure in that folder) |
| 13. Native vs container solver | `scaling_100steps/ctr_*/` (container) against the native folders; summary `container_summary.csv`, figure `container_overhead.svg` |
| 14. Container launch | `required_table/container_launch_overhead.csv` |
| 15. OSU latency/bandwidth | `required_table/osu_microbench_*` |

The folders `scaling_100steps/`, `thread_sweep/`, `rsqrt_convergence/` and
`overlap_20260923_132533/` contain a small Python script that rebuilds their summary
CSV and figures from the raw CSVs; run it from inside the folder with `python3 <script>.py`.
The scripts draw with `svgplot.py` (no matplotlib needed), which reproduces the look of the figures made by `analyze.py`.

In `scaling_100steps/` each subfolder is one Slurm job (`strong_P1_rep3` = strong scaling, P = 1, repetition 3; `ctr_` = inside the container). The long points were split over several jobs with `REP_LIST`.
