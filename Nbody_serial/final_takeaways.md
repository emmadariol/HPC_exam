# Final benchmark takeaways

## MPI strong scaling, GENOA, 64 ranks
- N = 20000, steps = 20
- Speedup at 64 ranks: 57.94x
- Efficiency at 64 ranks: 90.5%

## Hybrid scaling, GENOA, 64 cores
- All P x T configurations are close: ~0.535-0.541 s
- Best observed: 32 MPI ranks x 2 OpenMP threads

## Container
- Final folder: runs/orfeo_seq_20260817_120943/07_container_pdf_final
- Solver overhead: ~3.0-3.4%
- Launch overhead: ~0.09-0.11 s
- OSU native vs container: available, 5 repetitions
- ldd confirms host OpenMPI binding inside Singularity

## Correctness
- Energy drift always below tolerance
- AoS/SoA checksum differences negligible
