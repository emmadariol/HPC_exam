# results_final

Curated measurement data behind `FINAL_REPORT.md`. Every folder keeps the raw
per-run CSV (`scaling.csv`, `ablation.csv`, ...), the Slurm job script and, where
useful, a summary CSV and the figure used in the report.

| Report experiment | Data |
|---|---|
| 1. Energy conservation | all folders below (column `max_rel_drift`); long run: `energy_long_20260923_170127/` |
| 2. Cost of the energy check | `energy.csv`, `energy_summary.csv`, `energy_overhead.svg` |
| 3. Phase breakdown and memory bandwidth | `strong20_*`, `mapping_*`, `energy_long_*`; `memory_bandwidth.csv`, `memory_bandwidth_summary.csv` |
| 4. Strong scaling, main campaign (100 steps) | `required_table/required_container_scaling.csv` (mode `native`), `required_table/required_native_strong_*` |
| 4. Strong scaling, same executable, all phases (20 steps) | `strong20_P1_20260923_165910/`, `strong20_P4_16_64_20260923_165910/`, `overlap_20260923_132533/P*_sendrecv/`; summary and figure in `strong20_summary/` |
| 5. Weak scaling | `required_table/required_container_scaling.csv` (mode `native`, kind `weak`), `required_table/required_native_weak_*` |
| 6. MPI/OpenMP mapping | `mapping_numa/`, `mapping_socket/`, `mapping_core/` |
| 7-9. Newton, rsqrt, accumulators (T = 1) | `ablation_complete_20260922_083515/`; job accounting in `mapping_and_rsqrt_job_status.txt` |
| 7-9. Same tests with T = 2, 4, 8, 16 | `ablation_T{2,4,8,16}_20260923_165910/`; summary and figure in `thread_sweep/` |
| 8. rsqrt time-step convergence | `rsqrt_conv_<method>_n<steps>_20260923_170127/`; summary and figure in `rsqrt_convergence/` |
| 10. AoS vs SoA | `layout.csv`, `layout_summary.csv`, `layout_force_time.svg`; compiler reports `vectorization_*.txt` |
| 11. native vs x86-64-v3 | `arch_target_comparison*.csv` (N=10000), `arch_exact_20260923_170254/`, `arch_approx1_20260924_134326/` (N=100000) |
| 12. Blocking vs overlapped ring | `overlap_20260923_132533/`; summary and figure produced by `make_overlap_summary.py` in that folder |
| 13. Native vs container solver | `required_table/required_container_scaling*.csv`, `required_table/required_container_scaling_*.svg` |
| 14. Container launch | `required_table/container_launch_overhead.csv` |
| 15. OSU latency/bandwidth | `required_table/osu_microbench_*` |
| Hardware and software | `system_info_orfeo.txt`, `build_logs/`, `FINAL_JOB_STATUS.txt` (job IDs of the first campaign), `software_provenance/`, `mpi_linkage_check.txt`, `required_table/osu_mpi_linkage_check.txt` |

The summary folders (`strong20_summary/`, `thread_sweep/`, `rsqrt_convergence/`,
`overlap_20260923_132533/`) each contain the Python script that rebuilds their
CSV and SVG from the raw data; run it from inside that folder with `python3 <script>.py`.

The SHA-256 of the solver executable used for the overlap and second
strong-scaling campaigns is in `software_provenance/binary_sha256_strong20.txt`.
