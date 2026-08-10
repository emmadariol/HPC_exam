#!/bin/bash
# sbatch_concrete_commands.sh
# Concrete sbatch commands (editable) for Orfeo and Leonardo.
# Edit ACCOUNT and PARTITION variables to match your systems.

ACCOUNT=YOUR_ACCOUNT_HERE
ORFEO_PARTITION=short
LEONARDO_PARTITION=standard

echo "# Orfeo: native quick sweep"
echo "sbatch --nodes=1 --ntasks-per-node=4 --cpus-per-task=2 --time=01:00:00 --partition=${ORFEO_PARTITION} --output=orfeo.%j.out --export=RANKS='1 2',THREADS='1 2',REPEATS=3,STRONG_N=1000,NSTEPS=20,ENERGY_EVERY=5 Nbody_serial/jobs/orfeo_job.sh"

echo "# Orfeo: container (Singularity)"
echo "sbatch --nodes=1 --ntasks-per-node=4 --cpus-per-task=2 --time=01:00:00 --partition=${ORFEO_PARTITION} --output=orfeo_container.%j.out --export=IMAGE_PATH=\"$PWD/nbody.sif\",RANKS='1 2',THREADS='1 2',REPEATS=3,STRONG_N=1000,NSTEPS=20 --wrap=\"cd $SLURM_SUBMIT_DIR && singularity exec \$IMAGE_PATH ./jobs/orfeo_job.sh\""

echo "# Leonardo: production sweep (native)"
echo "sbatch --nodes=4 --ntasks-per-node=16 --cpus-per-task=4 --time=04:00:00 --partition=${LEONARDO_PARTITION} --account=${ACCOUNT} --output=leonardo.%j.out --export=RANKS='1 2 4 8',THREADS='1 2 4',REPEATS=5,STRONG_N=4000,WEAK_PER_RANK=1000,NSTEPS=50,ENERGY_EVERY=10 Nbody_serial/jobs/leonardo_job.sh"

echo "# Leonardo: production sweep (container)"
echo "sbatch --nodes=4 --ntasks-per-node=16 --cpus-per-task=4 --time=04:00:00 --partition=${LEONARDO_PARTITION} --account=${ACCOUNT} --output=leonardo_container.%j.out --wrap=\"cd $SLURM_SUBMIT_DIR && singularity exec $PWD/nbody.sif ./jobs/leonardo_job.sh\" --export=REPEATS=5,STRONG_N=4000"

echo "# One-rank-per-socket example (edit sockets/cores to match site)"
echo "sbatch --nodes=4 --ntasks-per-node=2 --cpus-per-task=24 --time=04:00:00 --partition=${LEONARDO_PARTITION} --account=${ACCOUNT} --output=one_rank_per_socket.%j.out --wrap=\"cd $SLURM_SUBMIT_DIR && RANKS='4 8' THREADS='1' REPEATS=5 STRONG_N=4000 NSTEPS=50 ./benchmark_scaling.sh\""

echo "# Single MPI rank (node-wide threads) example"
echo "sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --time=02:00:00 --partition=${LEONARDO_PARTITION} --account=${ACCOUNT} --output=single_mpi_rank.%j.out --wrap=\"cd $SLURM_SUBMIT_DIR && export OMP_NUM_THREADS=32 && mpirun -n 1 ./nbody_direct_hybrid --input ic_2000.bin --nsteps 50 --integrator kdk --comm sendrecv --kernel direct\""

echo "# Post-process job (analysis + plots)"
echo "sbatch --time=00:10:00 --wrap=\"cd $SLURM_SUBMIT_DIR && python3 analyze_benchmark.py benchmark_results.csv benchmark_summary_final.csv && python3 plot_scaling.py benchmark_summary_final.csv final_scaling\" --output=analysis.%j.out"
