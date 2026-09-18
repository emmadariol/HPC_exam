# Curated final results

This directory contains the small, version-controlled dataset used by
`FINAL_REPORT.md`.

The large `runs/` tree is intentionally ignored by Git and should be treated as
local execution history. If new production runs are accepted, copy only the CSV,
SVG and TXT files needed by the final report into this directory.

Main groups:

- `scaling_64*`: MPI strong/weak scaling on one Orfeo GENOA node.
- `hybrid_64*`: fixed-resource MPI x OpenMP configurations.
- `ablation_64*`: kernel/math/communication/accumulator ablation evidence.
  Refreshed campaigns also include `ablation_summary.csv`, with percentage
  deltas such as communication-overlap gain relative to `sendrecv`.
- `layout*`: AoS-vs-SoA evidence and checksum comparison.
- `energy*`: energy-diagnostic overhead evidence.
- `memory_bandwidth*`: standalone STREAM-style RAM-bandwidth evidence.
- `container_overhead*`: native-vs-Singularity solver timings.
- `arch_target_comparison*`: native `-march=native` vs native
  `-march=x86-64-v3`, used to separate compilation-target effects from
  container runtime overhead.
- `osu_microbench_native_vs_container*` and `osu_microbench_*`: OSU
  latency/bandwidth evidence for native and Singularity execution. For strict
  Exercise-1 compliance, the final OSU campaign should be submitted with
  `--nodes 2 --cpus 2 --ntasks-per-node 1`.
  The solver-level native-vs-container overhead is reported in
  `container_overhead*`.
- `mpi_linkage_check.txt`: host-MPI linkage verification inside the container.
