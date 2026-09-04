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
- `layout*`: AoS-vs-SoA evidence and checksum comparison.
- `energy*`: energy-diagnostic overhead evidence.
- `container_overhead*`: native-vs-Singularity solver timings.
- `osu_microbench*`: OSU latency/bandwidth native-vs-container evidence.
- `mpi_linkage_check.txt`: host-MPI linkage verification inside the container.
