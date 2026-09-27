# results_final

Data behind `FINAL_REPORT.md`.

- this folder: summary CSV files, the figures used in the report, and the scripts that rebuild them;
- `raw/`: the raw CSV of every run, one subfolder per experiment or Slurm job.

| Report experiment | Summary and figures (this folder) | Raw runs (`raw/`) |
|---|---|---|
| Energy conservation | column `max_rel_drift` in every CSV | `energy_long_20260923_170127/` |
| Cost of the energy check | `energy_summary.csv` | `energy_check/` |
| Strong and weak scaling | `native_strong_summary.csv`, `native_weak_summary.csv`, `strong_*.svg`, `weak_*.svg` | `scaling_100steps/strong_*`, `scaling_100steps/weak_*` |
| Mapping ranks and threads | - | `mapping_numa/`, `mapping_socket/`, `mapping_core/` |
| Newton, rsqrt, partial sums | `thread_sweep_summary.csv`, `newton_threads_*.svg` | `ablation_complete_20260922_083515/`, `ablation_T*_20260923_165910/` |
| rsqrt time-step convergence | `rsqrt_convergence_summary.csv` | `rsqrt_conv_*_20260923_170127/` |
| AoS vs SoA | `layout_summary.csv`, `layout_force_time.svg` | `layout/` |
| native vs x86-64-v3 | `arch_target_comparison_summary.csv` (N = 10000) | `arch_short_N10000/`, `arch_exact_20260923_170254/`, `arch_approx1_20260924_134326/` |
| Blocking vs overlapped ring | `overlap_summary.csv` | `overlap/` |
| Native vs container | `container_summary.csv` | `scaling_100steps/ctr_*` |
| Container start-up | - | `container_launch/` |
| OSU latency and bandwidth | `osu_microbench_summary.csv`, `osu_microbench_*.svg` | `osu/` |

Rebuild the summaries and figures from the raw runs:

```sh
python3 make_scaling_summary.py
python3 make_thread_sweep.py
```
