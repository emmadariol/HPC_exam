# Exercise 1 Direct N-body: Implementation, Derivables, and Scalability Guide

## 1. Scope and Goal

This document explains, in detail:

- what was required for Exercise 1,
- what is implemented in the baseline folder and in the modified folder,
- why implementation choices were made,
- how to generate every required derivable (tables, graphs, reports),
- how to interpret scalability behavior for this direct N-body solver.

Folders covered:

- `Nbody_serial_baseline`: clean serial baseline
- `Nbody_serial`: extended implementation with MPI + OpenMP + benchmarking + container workflow

---

## 2. What Exercise 1 Requires (Mapped to Deliverables)

Exercise 1 asks for a direct N-body code path suitable for HPC analysis. The practical deliverables are:

1. Correct direct all-pairs gravitational solver with softening.
2. Reproducible initial-condition generation.
3. Numerical correctness verification via energy drift.
4. Parallel implementation (MPI + OpenMP) and timing instrumentation.
5. Strong scaling analysis (speedup, efficiency).
6. Weak scaling analysis (efficiency trends).
7. Discussion of optimization choices (SoA layout, kernel structure, rsqrt variant, communication mode).
8. Container overhead comparison (native vs container runtime).
9. Final report with hardware, software, method, results, and bottleneck discussion.

All of these are implemented in `Nbody_serial` and documented by scripts + generated artifacts.

---

## 3. Baseline vs Modified: Architectural Difference

## Baseline (`Nbody_serial_baseline`)

- serial direct solver (`nbody_direct_serial.c`),
- one IC generator (`generate_ic.c`) supporting:
  - Plummer sphere (`--model 0`),
  - uniform ball + Maxwellian velocities (`--model 1`),
- common type/maths/binary definitions (`nbody_common.h`),
- minimal build (`Makefile`) and run examples (`README.md`).

Purpose: correctness-first reference and optimization starting point.

## Modified (`Nbody_serial`)

Contains baseline pieces plus:

- hybrid MPI + OpenMP solver (`nbody_direct_hybrid.c`),
- full benchmark pipeline and analyzers,
- plotting and reporting workflow,
- Docker/Singularity benchmark helpers,
- vectorization reporting helper,
- final report/template and generated result artifacts.

Purpose: complete Exercise 1 submission workflow and scalability study.

---

## 4. File-by-File Explanation

## 4.1 Core physics and numerics

### `nbody_common.h` (both folders)

Defines project-wide fundamentals:

- compile-time arithmetic type selection:
  - `NBODY_USE_DOUBLE` (default),
  - `NBODY_USE_FLOAT`.
- `dtype` abstraction and math wrappers (`dtype_sqrt`, `dtype_pow`, ...),
- binary format constants and magic header,
- model identifiers:
  - `PLUMMER_SPHERE = 0`,
  - `MAXWELL_BALL = 1`.

Choice rationale:

- keeps numerical precision switch centralized,
- avoids duplicated float/double code,
- keeps binary I/O format stable while allowing arithmetic experiments.

### `generate_ic.c` (both folders)

Generates binary initial conditions accepted by both solvers.

Major components:

- strict CLI parser (`--key value` and `--key=value`),
- deterministic SplitMix64 RNG,
- isotropic random direction sampling,
- Gaussian sampling via Box-Muller,
- Plummer radius and equilibrium speed sampling,
- uniform-ball + Maxwellian alternative model,
- center-of-mass removal for position and velocity,
- binary writer with finite-range sanity check.

Important implementation choices:

- deterministic RNG to make benchmark runs reproducible,
- binary storage as float32 to reduce I/O footprint,
- internal computation in selected `dtype` for numerical control,
- strict input validation to prevent silent bad runs.

Recent cleanup implemented:

- usage text now documents all supported options (`--model`, `--radius`, `--sigma`),
- explicit model validation (`0` or `1`) added,
- sanity-failure counter type fixed to `size_t` for format safety.

### `nbody_direct_serial.c` (both folders)

Serial O(N^2) softened direct solver using leapfrog DKD integration.

Core responsibilities:

- read particle file,
- compute all-pairs accelerations,
- integrate particle states,
- compute kinetic/potential/total energy,
- report relative energy drift,
- optionally write final state.

Data layout choice:

- structure-of-arrays (`x[]`, `y[]`, `z[]`, ...) instead of array-of-structs,
- better for cache streaming and future SIMD/OpenMP/MPI kernels.

Recent cleanup implemented:

- stale scaffold comments removed,
- force-kernel pointer qualifiers tightened (`const` + `restrict`) to help compiler optimization and code clarity.

## 4.2 Parallel and performance code (`Nbody_serial` only)

### `nbody_direct_hybrid.c`

Hybrid MPI + OpenMP implementation for Exercise 1.

Implemented features:

- MPI rank decomposition by contiguous blocks,
- ring-shift source exchange among ranks,
- OpenMP parallel loops on local target particles,
- two integrators:
  - `kdk` (default for main report),
  - `dkd` (comparison/compatibility),
- communication modes:
  - `sendrecv` (blocking),
  - `overlap` (non-blocking exchange + compute overlap),
- kernel modes:
  - `direct` (distributed scalable path),
  - `newton` (single-rank Newton third law with thread-private force buffers),
- inverse square root modes:
  - `exact`,
  - `approx` (float seed + Newton refinement),
- section timing instrumentation:
  - total, io, drift, force, kick, energy,
- kernel throughput reporting (pair interactions/s and Gpair/s),
- final summary with energy-drift status.

Key design rationale:

- ring-shift avoids full-replication all-gather memory cost,
- direct kernel avoids inner-loop atomics in distributed mode,
- thread-private buffers in Newton mode avoid race-heavy updates,
- split timing makes bottleneck attribution explicit.

Cleanup implemented:

- removed one unused local variable in the Newton helper.

---

## 5. Build, Benchmark, and Analysis Files (`Nbody_serial`)

### Build and utility files

- `Makefile`: serial + hybrid + generator builds, smoke runs, vectorization report target.
- `collect_system_info.sh`: captures hardware/software environment.
- `Dockerfile`: reproducible Docker build environment.
- `Singularity.def`: Singularity/Apptainer recipe.
- `.dockerignore`: reduces Docker build context.

### Scaling benchmark pipeline

- `benchmark_scaling.sh`
  - loops over ranks/threads/repeats,
  - generates fresh IC per repetition,
  - runs hybrid solver,
  - parses final/timing/rate lines,
  - emits `benchmark_results.csv`.

- `analyze_benchmark.py`
  - computes medians/stdev per configuration,
  - computes speedup and efficiency versus baseline resources,
  - writes `benchmark_summary.csv`.

- `plot_scaling.py`
  - renders SVG plots:
    - `scaling_strong_speedup.svg`,
    - `scaling_strong_efficiency.svg`,
    - `scaling_weak_speedup.svg`,
    - `scaling_weak_efficiency.svg`.

### Container overhead pipeline

- `benchmark_docker.sh`: native vs Docker repeated measurements.
- `benchmark_container.sh`: native vs Singularity/Apptainer repeated measurements.
- `analyze_container_overhead.py`: medians/stdev/overhead table.

### Reporting files

- `REPORT_TEMPLATE.md`: checklist for required report sections.
- `FINAL_REPORT.md`: completed narrative report with platform and results.
- `vectorization_report.txt`: compiler vectorization diagnostics from `make vec-report`.

---

## 6. Derivables: What They Are and How To Produce Them

## 6.1 Correctness derivable

Solver prints max relative energy drift:

- metric: `abs(E(t)-E(0))/max(abs(E(0)), dtype_min_normal)`
- status: `OK` if drift <= tolerance, else `WARNING`.

Command (example):

```bash
OMP_NUM_THREADS=2 mpirun -np 2 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 50 --dt 1e-4 --eps 0.05 \
  --energy-every 10 --integrator kdk --comm overlap --kernel direct --quiet
```

## 6.2 Strong and weak scaling derivables

Generate raw benchmark:

```bash
RANKS="1 2 4" THREADS="1 2" REPEATS=5 STRONG_N=4000 \
WEAK_PER_RANK=1000 NSTEPS=50 ENERGY_EVERY=10 \
INTEGRATOR=kdk COMM=overlap KERNEL=direct RSQRT=exact \
./benchmark_scaling.sh
```

Reduce + summarize:

```bash
./analyze_benchmark.py benchmark_results.csv benchmark_summary.csv
```

Generate graphs:

```bash
./plot_scaling.py benchmark_summary.csv scaling
```

Outputs:

- raw table: `benchmark_results.csv`
- reduced summary: `benchmark_summary.csv`
- strong/weak SVG charts listed above

## 6.3 Container overhead derivables

Docker path:

```bash
./docker_build.sh
./docker_smoke.sh
RANKS="1 2 4" THREADS=1 REPEATS=5 N=1000 NSTEPS=20 ./benchmark_docker.sh
./analyze_container_overhead.py docker_overhead.csv docker_overhead_summary.csv
```

Singularity/Apptainer path:

```bash
RANKS="1 2 4" THREADS=1 REPEATS=5 N=1000 NSTEPS=20 ./benchmark_container.sh
./analyze_container_overhead.py container_overhead.csv container_overhead_summary.csv
```

---

## 7. Graphs and Numerical Results Already Present

From current `benchmark_summary.csv` (kdk + overlap + direct + exact):

Strong scaling (`N=4000`, best row):

- ranks=4, threads=2, resources=8,
- median total time = 0.868922 s,
- speedup = 2.811,
- efficiency = 0.351.

Interpretation:

- scaling is real but non-ideal at high resource count,
- force kernel dominates and parallel overheads become visible,
- expected on a WSL laptop platform and moderate problem size.

Weak scaling highlights:

- baseline (1x1, N=1000): 0.144845 s
- 4x2 at N=4000: 0.986230 s

Interpretation:

- direct all-pairs complexity implies rising work per rank when total N grows,
- weak scaling cannot remain flat like nearest-neighbor stencil workloads,
- this is an algorithmic property of direct N-body, not only an implementation artifact.

From current `docker_overhead_summary.csv`:

- N=1000, ranks=1: overhead ~0.91%
- N=1000, ranks=2: overhead ~19.60%
- N=1000, ranks=4: overhead ~12.12%

Interpretation:

- very small overhead at 1 rank,
- larger variability and overhead for multi-rank local runs under Docker/WSL.

---

## 8. Scalability Discussion (Short, Exam-Oriented)

For this code, the dominant force computation is O(N^2). This has two direct consequences:

1. Strong scaling
- can show useful acceleration until communication/synchronization and serial fractions dominate.

2. Weak scaling (particles per rank fixed)
- still worsens in time for direct summation because each local set interacts with a globally growing particle set.

So the observed non-ideal weak-scaling trend is expected and physically consistent with the algorithm.

To improve asymptotic scaling substantially, one must change algorithmic family (tree/FMM/PM), which is outside this direct-kernel exercise.

---

## 9. Cleanup Performed

Implemented cleanup actions:

- Removed stale scaffold-style comments in serial solver source.
- Added stricter and clearer option documentation/validation in generator.
- Fixed one type-safety issue in generator sanity counter.
- Added `.gitignore` files in both folders.
- Removed generated transient artifacts from `Nbody_serial`:
  - executables,
  - temporary `.bin` states,
  - Python cache folder.

Kept intentionally:

- CSV summaries and SVG graphs that are required derivables,
- report files and benchmark scripts,
- container recipes and helper scripts.

---

## 10. Minimal Reproduction Checklist

Inside `Nbody_serial`:

1. `make clean && make`
2. `make run-smoke`
3. `./collect_system_info.sh system_info.txt`
4. run scaling benchmark + analysis + plotting
5. run container overhead benchmark + analysis
6. include resulting tables/plots and system info in final report

This gives a complete, reproducible Exercise 1 package.

---

## 11. Cosa Lanciare Su Orfeo E Cosa Su Leonardo

Questa e la divisione consigliata, in modo pratico e coerente con l'esame.

### 11.1 Su Orfeo (sviluppo, debug, smoke, test piccoli)

Usa Orfeo per tutto cio che serve a verificare funzionalita e correttezza veloce:

- compilazione e fix del codice;
- smoke test seriale e ibrido;
- test brevi su N piccoli;
- confronto rapido fra opzioni (`kdk/dkd`, `sendrecv/overlap`, `exact/approx`).

Comandi tipici su Orfeo:

```bash
cd Nbody_serial
make clean && make
make run-smoke

./generate_ic --model 0 --n 1000 --seed 123 --output plummer_1000.bin
OMP_NUM_THREADS=2 mpirun -np 2 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 20 --dt 1e-4 --eps 0.05 \
  --energy-every 10 --integrator kdk --comm sendrecv --kernel direct --rsqrt exact
```

Output atteso su Orfeo:

- stato `OK` sulla deriva energetica,
- tempi coerenti,
- nessun crash/errore I/O,
- configurazione pronta per il run finale.

### 11.2 Su Leonardo (misure ufficiali e derivables finali)

Usa Leonardo per i run che devono finire nel report finale:

- strong scaling completo (ripetizioni multiple);
- weak scaling completo;
- benchmark container overhead (se richiesto dal docente);
- raccolta hardware/software del nodo usato;
- produzione CSV e grafici finali.

Comandi tipici su Leonardo:

```bash
cd Nbody_serial
make clean && make

./collect_system_info.sh system_info.txt

RANKS="1 2 4" THREADS="1 2" REPEATS=5 STRONG_N=4000 \
WEAK_PER_RANK=1000 NSTEPS=50 ENERGY_EVERY=10 \
INTEGRATOR=kdk COMM=overlap KERNEL=direct RSQRT=exact \
./benchmark_scaling.sh

./analyze_benchmark.py benchmark_results.csv benchmark_summary.csv
./plot_scaling.py benchmark_summary.csv scaling
```

Se fai anche overhead container su Leonardo:

```bash
RANKS="1 2 4" THREADS=1 REPEATS=5 N=1000 NSTEPS=20 ./benchmark_container.sh
./analyze_container_overhead.py container_overhead.csv container_overhead_summary.csv
```

### 11.3 Regola pratica (importante)

- Orfeo: velocita di iterazione e validazione funzionale.
- Leonardo: numeri finali da consegnare (scaling, grafici, tabelle, overhead).

In altre parole, non usare Orfeo per i risultati prestazionali definitivi del report, e non usare Leonardo per debug minuto per minuto.

## 12. Verifica locale (passi rapidi e comandi corretti)

Prima di sottomettere run su Orfeo/Leonardo verifica tutto in locale seguendo questi passi. Copia e incolla i comandi nel terminale dalla cartella principale del repository.

1) Compilare tutto

```bash
cd Nbody_serial
make clean
make -j
```

File: [Nbody_serial/Makefile](Nbody_serial/Makefile)

2) Generare una condizione iniziale di prova (esempio 128 particelle)

```bash
./generate_ic --n 128 --model 0 --output ic_128.bin
```

File: [Nbody_serial/generate_ic](generate_ic)

3) Smoke test (target fornito dal `Makefile`)

```bash
make run-smoke
```

4) Eseguire un test seriale corto con l'IC appena creato

```bash
./nbody_direct_serial --input ic_128.bin --nsteps 5
```

File: [Nbody_serial/nbody_direct_serial](nbody_direct_serial)

5) Eseguire un test ibrido MPI+OpenMP (esempio 2 ranks × 2 threads)

```bash
export OMP_NUM_THREADS=2
export OMP_PLACES=cores   # (opzionale) mappa i threads sui core
export OMP_PROC_BIND=close
mpirun -n 2 ./nbody_direct_hybrid --input ic_128.bin --nsteps 5
```

File: [Nbody_serial/nbody_direct_hybrid](nbody_direct_hybrid)

Nota: le opzioni corrette sono `--input` per il file IC e `--nsteps` per il numero di step; `-i`/`-N` non sono accettati.

6) Eseguire un breve benchmark locale (limitato)

```bash
# esempio rapido: 2 ripetizioni, 1-2 ranks/threads
RANKS="1 2" THREADS="1 2" REPEATS=2 STRONG_N=1000 NSTEPS=10 ./benchmark_scaling.sh
```

7) Analisi e plot dei risultati

```bash
python3 analyze_benchmark.py benchmark_results.csv benchmark_summary.csv
python3 plot_scaling.py benchmark_summary.csv scaling
```

8) Raccogli informazioni di sistema (per la riproducibilita')

```bash
./collect_system_info.sh > system_info_local.txt
```

File: [Nbody_serial/collect_system_info.sh](collect_system_info.sh)

Cosa controllare dopo ogni run

- Nell'output: la riga `# final: ... status=OK` e il valore `max_relative_energy_drift` (dev'essere piccolo).
- `benchmark_results.csv` e `benchmark_summary.csv` devono essere aggiornati dopo lo script di benchmark.
- I file SVG prodotti da `plot_scaling.py` devono essere presenti nella cartella.
- In caso di errori: correggi build o opzioni locali e ripeti i test prima di lanciare su Orfeo/Leonardo.

Se vuoi, posso eseguire qui `make run-smoke` e i test seriale/ibrido e mostrarti gli output; dimmi quali eseguire ora.

## 13. Valori da provare (param sweep consigliato)

Questa sezione elenca i set di parametri che abbiamo usato per test locali e che consiglio di eseguire su Orfeo/Leonardo. Copia i singoli comandi o usali come variabili per `benchmark_scaling.sh`.

- Ambienti OMP/MPI (sempre impostare prima di lanciare):

```bash
export OMP_NUM_THREADS=⟨threads⟩
export OMP_PLACES=cores
export OMP_PROC_BIND=close
mpirun -n ⟨ranks⟩ ./nbody_direct_hybrid --input ic_⟨N⟩.bin --nsteps ⟨NSTEPS⟩ [options]
```

- Valori locali rapidi (debug / smoke):

```text
RANKS = 1, 2
THREADS = 1, 2
REPEATS = 1-3
STRONG_N = 128, 256, 512
NSTEPS = 5-20
ENERGY_EVERY = 1
INTEGRATOR = kdk, dkd
COMM = sendrecv, overlap
KERNEL = direct, newton (newton: only -np 1)
RSQRT = exact, approx
EPS = 0.01, 0.05
DT = 1e-3, 1e-4
MODEL = 0 (Plummer), 1 (uniform ball)
SEED = 1, 42, 123
```

- Valori consigliati per test su Orfeo (sviluppo, test approfonditi ma non finali):

```text
RANKS = 1, 2, 4
THREADS = 1, 2, 4
REPEATS = 3
STRONG_N = 1000, 2000
WEAK_PER_RANK = 500, 1000
NSTEPS = 20-50
ENERGY_EVERY = 5-10
INTEGRATOR = kdk
COMM = overlap
KERNEL = direct
RSQRT = exact
EPS = 0.01
DT = 1e-4
```

- Valori per run definitivi su Leonardo (produzione dei derivabili):

```text
RANKS = 1, 2, 4, 8 (o fino al limite della coda)
THREADS = 1, 2, 4, 8
REPEATS = 5
STRONG_N = 2000, 4000, 8000
WEAK_PER_RANK = 1000, 2000
NSTEPS = 50
ENERGY_EVERY = 10
INTEGRATOR = kdk
COMM = overlap
KERNEL = direct
RSQRT = exact
EPS = 0.01
DT = 1e-4
```

- Parametri container / overhead

```text
RANKS = 1, 2, 4
THREADS = 1
REPEATS = 5
N = 1000
NSTEPS = 20
```

Esempio di invocazione rapida (locale):

```bash
RANKS="1 2" THREADS="1 2" REPEATS=2 STRONG_N=1000 NSTEPS=10 ./benchmark_scaling.sh
```

Note pratiche:
- Preferisci `COMM=overlap` per misurazioni realistiche di produzione (minimizza idle di comunicazione).
- Usa `RSQRT=approx` solo per test di prestazioni; i risultati numerici possono differire leggermente.
- Quando provi `KERNEL=newton`, limita a `-np 1` (single-rank) per confronti.
- Mantieni `SEED` fisso tra ripetizioni per comparabilità.

Questa lista copre sia rapid-check locali sia la griglia da lanciare su cluster per produrre i derivabili finali del report.

### Job scripts e automazione

Ho aggiunto script pronti a essere usati per lanciare i run su Orfeo/Leonardo e uno script orchestratore che compila, esegue la griglia di benchmark, analizza e genera i plot.

- `Nbody_serial/jobs/orfeo_job.sh`: template SLURM per Orfeo (piccoli test, tempo breve).
- `Nbody_serial/jobs/leonardo_job.sh`: template SLURM per Leonardo (run di produzione, nodi multipli).
- `Nbody_serial/jobs/run_all.sh`: orchestratore locale o da job node che compila, lancia `benchmark_scaling.sh`, esegue `analyze_benchmark.py` e `plot_scaling.py`, e impacchetta gli output.

Esempi rapidi:

```bash
# Local/interactive: usa l'orchestratore
cd Nbody_serial
./jobs/run_all.sh results_local

# Submit su Orfeo
cd Nbody_serial
sbatch jobs/orfeo_job.sh

# Submit su Leonardo (aggiorna --account e --partition nel file se richiesto)
cd Nbody_serial
sbatch jobs/leonardo_job.sh
```

Nota: prima di lanciare su Leonardo modifica `--account=YOUR_ACCOUNT_HERE` in `jobs/leonardo_job.sh` e controlla la `--partition` appropriata. I job script impostano valori raccomandati ma puoi modificarli direttamente nelle variabili in cima ai file.
