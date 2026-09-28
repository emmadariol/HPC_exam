# Exercise 1 - Direct N-body gravitational simulation

High Performance Computing exam

Dariol Emma - SM3800118

---

# HPC PART

## Explanation of the experiment

This report is about a computer program that simulates how a group of bodies moves because of their gravity. The program runs in parallel on many processor cores at the same time. The goal of the work is not only to make the program fast. The goal is also to understand why it is fast or slow, to check that it still gives correct physical results when it uses many cores, and to test the same program inside a software container. All the measurements were done on the GENOA nodes of the Orfeo cluster of Area Science Park; the LEONARDO supercomputer was not available.

We have N particles in empty space, and each particle pulls every other particle with the force of gravity. If we know where the particles are and how fast they move at the beginning, we want to know where they will be later. With more than two bodies there is no simple formula for this, so the program moves forward in many small time steps. At every step it computes the total pull on each particle and then moves all the particles a little. The difficult part is that every particle interacts with every other particle. With N particles there are about N times N pairs to compute at each step, so if we double the number of particles, each step becomes about four times more expensive. This is called an O(N^2) cost. The acceleration of particle i is computed with this formula, where we use G = 1 and all masses equal to 1:

```text
a_i = sum over all j != i of  G m_j (r_j - r_i) / (|r_j - r_i|^2 + epsilon^2)^(3/2)
```

The epsilon in the formula is the softening length. In pure Newtonian gravity the force becomes infinite when two particles get very close, which creates huge and unrealistic jumps. With epsilon the force stays finite: far apart compared with epsilon it is almost exactly Newton's force, very close it becomes weak and goes to zero, as if every particle were a small soft ball of size epsilon.

Epsilon has to be compared with the typical distance between neighbouring particles. In the centre of a Plummer sphere of scale radius a (a = 1 in our runs) this distance is:

```text
d_centre = (4 pi / (3 N))^(1/3) x a        (0.075 for N = 10,000; 0.035 for N = 100,000)
```

Our epsilon = 0.05 is of the same order: the pull of the nearest neighbours is smoothed, while the pull of the rest of the cluster stays Newtonian. A much smaller epsilon would produce very large forces in close encounters and need much smaller time steps; a much larger one would flatten the dense centre.

Every run starts from a Plummer sphere. This is a classic model of a spherical cluster: it is dense in the centre and thinner towards the edge, and the velocities are chosen so that the cluster neither collapses nor explodes. A separate program creates the starting positions and velocities from a random seed. 

To move the particles forward in time we use the leapfrog method in its Kick-Drift-Kick form. Each step does:

1. kick: v += a x dt/2, with the current accelerations;
2. drift: x += v x dt;
3. new accelerations at the new positions (the expensive part);
4. kick: v += a x dt/2, with the new accelerations.

Kick and drift update every particle independently, so they are split among the threads with `#pragma omp parallel for schedule(static)`. The method needs only one force computation per step, and the energy does not slowly drift up or down during a long simulation: it only oscillates a little around the correct value.

This property is the reason why we use energy to check correctness. A group of particles that only attract each other must keep its total energy E = T + U constant, where T is the energy of motion and U is the gravitational energy, computed with the same softening as the forces. So the program computes the total energy at the start and at regular times during the run, and it reports the largest relative change, |E(t) - E(0)| / |E(0)|, which we call the energy drift. A run is considered OK only if the drift stays below 1e-4, a stricter limit than 1e-3. The energy is checked only at the sampled times and not at every moment, and computing the energy costs as much as computing the forces, so in performance runs it cannot be done very often.

One processor core needs about an hour to simulate 100 steps of 100,000 particles, so the work is split among many cores using MPI: it starts many independent copies of the program, called ranks or processes. Each rank has its own memory and cannot see the data of the others; if it needs something, it must receive it as a message. The work is divided like this:

- the N particles are split into P equal groups, one per rank; each rank owns its "home" particles for the whole run (when N is not a multiple of P, the first ranks get one extra particle);
- the ranks form a ring, and each rank holds one travelling block of particles;
- at every ring step each rank computes the forces between its home particles and the block it holds, then passes the block to one neighbour and receives a new block from the other;
- after P steps every block has visited every rank, and every rank has the complete force on its home particles;
- only the positions travel; the home particles never leave their rank, so nobody else writes their forces and no final combination of results is needed.

Inside each rank, OpenMP starts several threads that share the same memory and split the home particles among them. The two tools are used together: N particles, P ranks and T threads per rank, which gives P x T cores in total.

The positions are stored as a structure of arrays (one array for all x, one for all y, and so on), aligned to 64 bytes for AVX-512 loads; the ring sends them as three plain arrays. Each rank reads only its own block of the input file with `MPI_File_read_at_all`, and the final state, when requested, is collected on rank 0 with `MPI_Gatherv`. Compared with sending every particle to every rank (for example with `MPI_Allgather`), the ring moves the same data but needs memory only for the home block and one travelling block, and its P small exchanges can be overlapped with computation.

To understand where the time goes, the program measures itself with the MPI clock, `MPI_Wtime()`. Each rank records:

- `io`: reading the input;
- `force`: computing the forces, including `comm_wait`, the time spent waiting for ring messages;
- `kick` and `drift`: moving the particles;
- `energy`: computing the energy;
- `total`: from reading the input to the end of the simulation (without the start of the program and of MPI).

For each phase the largest value among the ranks is collected on rank 0 with `MPI_Reduce` and `MPI_MAX`, because the slowest rank decides when the job ends. Different phases can be slowest on different ranks, so the phases do not add up exactly to the total. From the force time the program also computes a throughput, the number of particle pairs processed per second:

```text
Gpairs/s = (number of steps + 1) x N x (N - 1) / (T_force x 1e9)
```

The "+ 1" is the force computation before the first step. This is a useful "speedometer" for the main computation, but it is not the same thing as floating-point operations per second.

All runs used GENOA nodes of Orfeo. Each node has two processors, called sockets, with 32 cores each, so 64 cores in total. The memory of a node is divided into 8 regions called NUMA domains, with 8 cores each. The details in the two tables below come from the `lscpu` and `numactl -H` output collected on a compute node:

| Item | Value |
|---|---|
| Nodes used | GENOA nodes between `genoa001` and `genoa013` (one node per job; two nodes for the OSU test) |
| CPU | AMD EPYC 9374F 32-Core Processor (Zen 4) |
| Sockets | 2 |
| Cores per socket | 32 |
| Hardware threads | 1 per core (SMT off) |
| Total CPUs | 64 |
| NUMA domains | 8, with 8 cores each (CPUs 0-7, 8-15, ..., 56-63) |
| NUMA distances (`numactl -H`) | 10 inside a domain, 12 between domains of the same socket, 32 across sockets |
| Memory | 503 GiB (about 478 GiB available at measurement time), no swap |
| Vector instructions | AVX2 and AVX-512 supported |
| Operating system kernel | Linux 6.13.12-200.fc41 |

| Component | Version |
|---|---|
| C compiler | GCC 14.3.1 (Red Hat 14.3.1-4) |
| MPI | Open MPI 4.1.6rc4 (module `openMPI/4.1.6`) |
| OpenMP runtime | GNU libgomp, package `libgomp-14.3.1-4.fc41` |
| Other libraries | libm, hwloc 2.12.0; no BLAS or other numerical library |
| Container runtime | SingularityCE 4.3.1 |

The native program is compiled with:

```sh
mpicc -std=c11 -DNBODY_USE_DOUBLE -O3 -march=native -Wall -Wextra -Wpedantic -fopenmp -o nbody_direct_hybrid nbody_direct_hybrid.c -lm
```

This means:

- the C11 language standard, 
- calculations in double precision,
- strong optimisation by the compiler (`-O3`)
- instructions chosen for the exact processor of the machine that builds the program
- all compiler warnings turned on
- OpenMP turned on
- No profile-guided optimisation is used

All other programs are built with the same flags, changing only the compiler, OpenMP and a few extra options:

```sh
<compiler> -std=c11 -DNBODY_USE_DOUBLE -O3 -march=<target> -Wall -Wextra -Wpedantic [-fopenmp] [extra options] -o <program> <source>.c -lm
```

Note: the vectorisation reports are obtained by adding `-fopt-info-vec-optimized -fopt-info-vec-missed -c` to this command; GCC then prints to standard error which loops were vectorised and which were not.

The OSU micro-benchmarks are built in user space with the cluster MPI:

```sh
./configure CC=mpicc --prefix=$HOME/osu && make -j && make install
```

Run configuration. Every solver run is started in the same way, with P processes (MPI ranks) and T threads per process:

```sh
export OMP_NUM_THREADS=T
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
srun --ntasks=P --cpus-per-task=T --cpu-bind=verbose,cores ./nbody_direct_hybrid ...
```

`--cpus-per-task=T` gives each rank T cores of its own, `OMP_PLACES=cores` puts each thread on one physical core and `OMP_PROC_BIND=spread` spreads the threads of a rank over its cores. `--cpu-bind=cores` is the MPI binding: it binds every rank to its cores (the Slurm equivalent of `mpirun --bind-to`, since on Orfeo the ranks are started by Slurm), and `verbose` prints the real CPU mask of every rank, so the placement is checked. Each job runs on one GENOA node (two nodes only for the OSU test) and reserves at least P x T cores, so no other job shares them. Fixing processes and threads to cores stops the operating system from moving them, which would make the timings noisy.

Statistics:

- every configuration is run five times;
- we report the median, the middle value of the five sorted runs, which is not moved much by one unusually slow run;
- next to it we give the sample standard deviation, which we call the spread:

```text
s = sqrt( sum over the n runs of (x_i - mean)^2 / (n - 1) )
```

It says how much the runs differ from each other; it is not a formal error on the median. Every run also reports its energy drift and status.

For scaling experiments we use two standard numbers, with T(P) the median time on P ranks:

```text
speedup     S(P) = T(1) / T(P)
efficiency  E(P) = S(P) / P
```

The speedup says how many times faster P ranks are than one rank. The efficiency says which fraction of the ideal speedup we reach: 100% means that doubling the cores exactly halves the time.


---

## Energy conservation

```mermaid
flowchart LR
  A["<b>Input</b><br/>every run of this report,<br/>plus one long run (N =<br/>10,000, 2000 steps, 1<br/>rank x 8 threads)"]
  B["<b>Run</b><br/>E(0) at the start, then E<br/>again every energy-every<br/>steps and after the last<br/>step (long double sums,<br/>exact square root,<br/>MPI_Allreduce over the<br/>ranks)"]
  C["<b>Measure</b><br/>largest relative drift<br/>|E(t) - E(0)| / |E(0)| of<br/>each run"]
  D["<b>Analyse</b><br/>compare with the<br/>tolerance 1e-4; compare<br/>the same seed across P,<br/>ring versions,<br/>native/container"]
  E["<b>Result</b><br/>largest drift 4.7e-6,<br/>20x below tolerance;<br/>long run 4.99e-7"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

### Energy drift across all experiments

Before looking at speed, we want to be sure that the program computes the right thing, also when it runs in parallel. The energy drift is recorded in every run of this report.

The total energy is the energy of motion T plus the gravitational energy U. U uses the same softening as the force, and each pair is counted once:

```text
T = sum over i of  (1/2) m_i |v_i|^2
U = - sum over pairs i < j of  G m_i m_j / sqrt(|r_j - r_i|^2 + epsilon^2)
E = T + U
```

At the start the program stores E(0). Every time it computes the energy again, it measures the relative change and keeps the largest value found during the run:

```text
energy drift = max over the checks of  |E(t) - E(0)| / |E(0)|
```

The energy is computed at the start, then after every `energy-every` steps and after the last step; if the drift exceeds the tolerance `energy-tol` (1e-4), the run is marked as  fail. The check works like this:

- kinetic energy: each rank sums |v|^2 over its own particles, so no communication is needed. The threads split the particles with `#pragma omp parallel for reduction(+ : sum) schedule(static)`, and the sum is kept in `long double`;
- potential energy: it needs all pairs, so it uses the same ring as the force, with blocking `MPI_Sendrecv` exchanges. Each pair is counted once, only when the global index of the home particle is smaller than that of the source particle ("i smaller than j"), always with the exact square root, and with the same OpenMP directive;
- total: `MPI_Allreduce` with `MPI_SUM` (in `MPI_LONG_DOUBLE`) adds the kinetic and potential sums of all ranks, and every rank gets E.

The rule "i smaller than j" makes the work uneven between ranks: the rank with the first particles accepts almost all its pairs, the rank with the last ones almost none. The strong-scaling section measures the effect of this.


### A longer validation run

The runs above are short (5 to 100 steps). To check energy conservation over a longer run for a Plummer sphere of 10,000 particles, we also ran one long simulation:

- N = 10,000, 2000 steps of dt = 1e-4 (twenty times longer than the scaling runs);
- one rank with 8 threads, exact square root;
- energy checked every 10 steps (201 checks);
- only the energy matters here, so the job ran on a shared node with 8 cores and its timings are not used.

| N | Steps | dt | Simulated time | Energy checks | Largest energy drift |
|---:|---:|---:|---:|---:|---:|
| 10000 | 2000 | 1e-4 | 0.2 | every 10 steps | 4.99e-7 |

Over the whole run the largest drift is 4.99e-7: about 200 times below our tolerance of 1e-4 and 2000 times below 1e-3. The time-step convergence runs, with the same N, dt and threads but only 100 steps, reach at most 2.5e-7. Running twenty times longer therefore only doubles the largest drift, instead of multiplying it by twenty. This agrees with the leapfrog method, whose energy error oscillates in a bounded range instead of growing with time.

---

## Cost of the energy check

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>10,000, same input for<br/>every run"]
  B["<b>Run</b><br/>1 rank x 8 threads, 5<br/>steps: energy_every = 1<br/>(every step) and<br/>energy_every = 5 (start<br/>and end); 5 runs each"]
  C["<b>Measure</b><br/>total, force and energy<br/>time of each run"]
  D["<b>Analyse</b><br/>medians; ratio =<br/>T(every step) / T(start<br/>and end)"]
  E["<b>Result</b><br/>1.41x longer with the<br/>energy at every step, same<br/>force time and same drift"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

The potential energy needs every pair of particles, like the force, so checking it often can change the timing of a performance run. We measured how much:

- N = 10,000, 5 steps, one rank with 8 threads, the same input for every run, five runs per setting;
- energy at every step (`energy_every = 1`, 6 energy evaluations: the start and steps 1-5) against energy only at the start and at the end (`energy_every = 5`, 2 evaluations).

With medians over the five runs:

```text
ratio                   = T_total(every step) / T_total(start and end only)
time per evaluation     = T_energy / number of energy evaluations
```

| Energy computed | Energy evaluations | Median total (s) | Median force (s) | Median energy (s) | Time per evaluation (s) | Ratio | Largest drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| every step | 6 | 0.5546 | 0.3012 | 0.2476 | 0.0413 | 1.41 | 9.62e-8 |
| start and end only | 2 | 0.3938 | 0.3019 | 0.0880 | 0.0440 | 1.00 | 9.62e-8 |

The force time is the same in both settings, so the whole difference comes from the energy phase: one evaluation costs 0.041-0.044 s, and 6 instead of 2 evaluations make the run 1.41 times longer (0.5546 s against 0.3938 s). The largest drift is the same, 9.62e-8: in this run, checking at every step finds no larger deviation than checking only at the start and at the end.

For this reason all performance runs compute the energy only at the start and at the end.

---

## Profiling: where the time goes and kernel throughput

```mermaid
flowchart LR
  A["<b>Input</b><br/>runs with all phase<br/>timers: strong scaling P<br/>= 1, 16, 32 (100 steps),<br/>mapping 8 x 8 (20 steps)"]
  B["<b>Run</b><br/>MPI_Wtime around each<br/>phase: io, force,<br/>comm_wait, kick, drift,<br/>energy"]
  C["<b>Measure</b><br/>time of each phase<br/>(slowest rank), force<br/>time"]
  D["<b>Analyse</b><br/>share of the force phase;<br/>Gpairs/s = (steps + 1) N<br/>(N - 1) / (T_force 1e9),<br/>per core"]
  E["<b>Result</b><br/>force = 90-99% of the<br/>time; 0.25-0.27 Gpairs/s<br/>per core from 1 to 64<br/>cores"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

To identify the bottleneck we need measurements. Hardware counters were not available, so we use the program's own instrumentation: the phase timers described in the methods, and the throughput of the force kernel. A memory-bound code is measured in bytes per second; for an O(N^2) kernel the unit of work is one particle pair, so we measure pairs per second:

```text
Gpairs/s = (number of steps + 1) x N x (N - 1) / (T_force x 1e9)
```

| Run | Cores | Force phase, share of total time | Gpairs/s (whole run) | Gpairs/s per core |
|---|---:|---:|---:|---:|
| N = 100,000, 100 steps, 1 rank x 1 thread | 1 | 99.1% | 0.267 | 0.267 |
| N = 100,000, 100 steps, 16 ranks x 1 thread | 16 | 98.3% | 4.27 | 0.267 |
| N = 100,000, 100 steps, 32 ranks x 1 thread | 32 | 98.3% | 8.32 | 0.260 |
| N = 100,000, 20 steps, 8 ranks x 8 threads | 64 | 90.4% | 16.27 | 0.254 |

The force phase takes 90-99% of the run in every configuration. The rest is almost only the energy check, which is also a pair computation; waiting for messages and moving the particles take very little time. The throughput per core stays between 0.25 and 0.27 billion pairs per second from 1 to 64 cores, so adding cores adds throughput almost linearly. This is what we expect when each core works on data in its own caches, not when cores compete for main memory. One pair costs about 1 / 0.267e9 = 3.7 ns on one core, mostly in the square root and the division, which is why the square-root optimisation has the largest effect. Without hardware counters this is an indication, but the timers and the throughput agree: the program is limited by the arithmetic of each pair, not by memory or communication.

---

## Strong scaling

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>100,000, seeds 1-5 (the<br/>same at every P)"]
  B["<b>Run</b><br/>P = 1, 2, 4, 8, 16, 32<br/>ranks x 1 thread, 100<br/>steps, sendrecv ring,<br/>exact sqrt, energy at<br/>start and end; 30 runs<br/>split into Slurm jobs of<br/>at most 2 hours"]
  C["<b>Measure</b><br/>total and phase times,<br/>energy drift"]
  D["<b>Analyse</b><br/>median and spread -><br/>S(P), E(P), E_energy(P);<br/>Amdahl f_eff"]
  E["<b>Result</b><br/>table + run time, speedup<br/>and efficiency plots<br/>(30.9x, 96.7% at 32<br/>ranks)"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

Strong scaling measures how much faster a problem of fixed size runs when we add cores. Ideally, 32 cores would be 32 times faster. In practice some parts do not speed up: communication between ranks, work that every rank repeats, and waiting for the slowest rank. The way the efficiency drops as cores are added shows where these limits are.

Set-up:

- N = 100,000 particles, 100 time steps of dt = 1e-4;
- P = 1, 2, 4, 8, 16 and 32 ranks, one thread each, on GENOA nodes;
- blocking ring (`sendrecv`), direct kernel, exact square root, energy checked only at the start and at the end;
- the ranks wait for each other only in the ring exchanges (`MPI_Sendrecv`), in the energy reduction (`MPI_Allreduce`) and in the final gather of the particles;
- five repetitions per point, with the same five initial conditions (seeds) at every P, no warm-up;
- the same executable for every P, with all phase timers recorded;
- 30 runs split into Slurm jobs of at most two hours (one run on one core takes about 64 minutes): one job per repetition at P = 1, fewer jobs for the faster points; P = 8, 16 and 32 in one job on the same node;
- largest energy drift 2.1e-6.

Besides the speedup and efficiency of the total time, the table gives the efficiency of the energy check, computed in the same way from the median time of that phase, the waiting time as a share of the total, and the speed of the force computation of each rank:

```text
E_energy(P)        = T_energy(1) / (P x T_energy(P))
waiting            = T_comm_wait(P) / T_total(P)
Gpairs/s per rank  = (number of steps + 1) x N x (N - 1) / (T_force(P) x 1e9 x P)
```

The force computation needs no separate efficiency: the number of pairs is fixed, so if the Gpairs/s per rank stay the same as with one rank, the force computation has an efficiency of 100%.

| MPI ranks P | N/P | Median total (s) | Spread s (s) | Speedup | Efficiency (total) | Efficiency (energy check) | Waiting for messages | Gpairs/s per rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100000 | 3823.47 | 2.27 | 1.00 | 100.0% | 100.0% | 0.00% | 0.2667 |
| 2 | 50000 | 1920.50 | 0.63 | 1.99 | 99.5% | 69.6% | 0.27% | 0.2665 |
| 4 | 25000 | 960.53 | 0.86 | 3.98 | 99.5% | 60.6% | 0.27% | 0.2670 |
| 8 | 12500 | 480.42 | 1.01 | 7.96 | 99.5% | 56.9% | 0.19% | 0.2672 |
| 16 | 6250 | 240.56 | 1.67 | 15.89 | 99.3% | 55.1% | 0.26% | 0.2670 |
| 32 | 3125 | 123.59 | 0.02 | 30.94 | 96.7% | 52.9% | 0.24% | 0.2599 |

<p align="center"><img src="results_final/strong_runtime.svg" alt="Strong-scaling run time at N=100000" width="60%"></p>

_The run time falls from about 64 minutes on one core to about 2 minutes on 32 cores, following the ideal line T(1)/P._

<p align="center"><img src="results_final/strong_speedup.svg" alt="Strong-scaling speedup at N=100000" width="49%"> <img src="results_final/strong_efficiency.svg" alt="Strong-scaling efficiency at N=100000" width="49%"></p>

_Left: the speedup reaches 30.9 at 32 ranks, against an ideal of 32. Right: the efficiency stays above 99% up to 16 ranks and falls to 96.7% at 32 ranks (note the vertical scale, from 0.90 to 1.02)._

Up to 16 ranks the efficiency stays above 99%, and 32 ranks are 30.9 times faster than one, an efficiency of 96.7%. The force computation processes 0.267 billion pairs per second per rank both with 1 and with 16 ranks. The waiting time for ring messages stays below 0.3% of the run at every P.

The only point clearly below 99% is P = 32. The waiting time there is still 0.24%, but the force computation of each rank is slower: 0.260 billion pairs per second against 0.267 at the other rank counts. This looks like a property of the node, not of the scaling. The runs with 8, 16 and 32 ranks ran in the same job on the same node (genoa004), and with 8 and 16 ranks the speed per rank is normal. So some of the extra cores used only at 32 ranks are probably slower, for example because they run at a lower clock. The timers cannot confirm the cause.

The phase timers show that the other loss comes from the energy check. Its efficiency drops to about 70% with 2 ranks and to about 55% from 8 ranks on. The reason is a load imbalance in how the energy is computed. To count each pair once, a rank adds the pair (i, j) only when the global index of i is smaller than that of j. The rank that owns the first particles accepts almost all its pairs, while the rank that owns the last particles accepts almost none. Everybody waits for the busiest rank in `MPI_Allreduce`. It is Amdahl's law inside a single routine: a part of the work that does not divide evenly limits the whole. In these runs the energy is computed only twice, so it takes 0.9% of the time on one rank and 1.7% on 32 ranks, and its effect on the total is below one percentage point. With more frequent checks it would matter more. The fix is well known, for example letting each rank handle only half of the ring so that every rank gets the same number of unique pairs; it was not needed for correctness and was not applied.

Amdahl's law can also summarise the whole curve. Solving it for the serial fraction with the measured speedups gives an effective serial fraction:

```text
f_eff = (1/S(P) - 1/P) / (1 - 1/P)          0.00044 at P = 16,  0.00111 at P = 32
```

The losses behave as if a fraction of about 0.0004-0.001 of the work were serial. This number collects every source of loss (the energy imbalance and, at 32 ranks, the slower cores); it is not a measurement of serial code.

### Limits of strong scaling

Two limits are expected as P grows with N fixed:

- too little work per rank for vector instructions: at 32 ranks each rank still has 3,125 particles, and the force speed per rank is unchanged up to 16 ranks, so this effect is not seen;
- a ring that becomes too long: each force evaluation does P exchanges (the last one only brings each block home and is kept to simplify the loop). The computation per rank shrinks like N^2/P, while the messages grow like P:

  ```text
  computation per rank   ~ c x N^2 / P
  communication per rank ~ P x (latency + (data per block) / bandwidth)
  ```

  With blocks of about 75 kB at 32 ranks (3,125 particles x 3 coordinates x 8 bytes) and waiting of at most 0.3%, the point where the two terms meet is far beyond 32 ranks on one node for N = 100,000; across nodes it would come earlier.

### Growth of the cost with N

For the direct method the time per step grows like N^2, so multiplying N by 10 should multiply the time by 10^2 = 100.

We check this with two sets of runs that differ only in N, with the same executable and settings: one rank, one thread, 100 steps, N = 10,000 (the first point of the weak-scaling series) and N = 100,000 (the first point of the strong-scaling series).

| N | Median total time (s) |
|---:|---:|
| 10,000 | 38.11 |
| 100,000 | 3823.47 |
| Ratio | 100.32 (expected 100) |

The measured ratio, 100.32, is close to the prediction of 100, a difference compatible with the different initial conditions and the different amount of data in the caches. This confirms that the program has the quadratic cost of the direct method.

---

## Weak scaling

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, 10,000<br/>particles per rank (N =<br/>10,000 x P), seeds 1-5"]
  B["<b>Run</b><br/>P = 1, 2, 4, 8, 16 ranks<br/>x 1 thread, 100 steps,<br/>same settings as strong<br/>scaling; 25 runs"]
  C["<b>Measure</b><br/>total and phase times,<br/>energy drift"]
  D["<b>Analyse</b><br/>pairs ratio R(P) -> ideal<br/>time R(P) T(1) / P, work-<br/>normalised S(P) and E(P)"]
  E["<b>Result</b><br/>table + time, speedup and<br/>efficiency plots (15.9x,<br/>99.2% at 16 ranks)"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

Weak scaling keeps the number of particles per rank fixed and adds ranks, so that the problem grows with the machine. For many applications the ideal is a constant time.

For the direct N-body method this ideal does not apply. Each rank owns a fixed number of particles, but each of them must interact with all N particles, and N grows with P. So the work per rank grows like P: with 16 ranks each rank has 16 times more pairs than with one. The right ideal is a time that grows linearly with P, not a constant time.

The set-up was:

- 10,000 particles per rank, so N = 10,000, 20,000, 40,000, 80,000 and 160,000;
- P = 1, 2, 4, 8 and 16 ranks, one thread each, 100 time steps, GENOA nodes;
- five repetitions per point, no warm-up; the same executable and settings as the strong-scaling runs (blocking ring, exact square root, energy checked only at the start and at the end).

To compare each point with the ideal, we scale the one-rank time T(1) with the number of pairs:

```text
pairs ratio  R(P) = N_P (N_P - 1) / (N_1 (N_1 - 1))
ideal time   T_ideal(P) = R(P) x T(1) / P          (about P x T(1))
speedup      S(P) = R(P) x T(1) / T(P)
efficiency   E(P) = S(P) / P
```

This speedup is a model-based estimate: the large single-core runs were not done, because at N = 160,000 one would take several hours. It assumes the same cost per pair for all problem sizes, which the growth test of the previous section confirms (ratio 100.32 against 100).

| MPI ranks P | Total particles N | Median total time (s) | Spread s (s) | Ideal time (s) | Time relative to ideal | Work-normalised speedup | Work-normalised efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10000 | 38.11 | 0.004 | 38.1 | 1.000 | 1.00 | 100.0% |
| 2 | 20000 | 76.62 | 0.03 | 76.2 | 1.005 | 1.99 | 99.5% |
| 4 | 40000 | 153.42 | 0.07 | 152.5 | 1.006 | 3.97 | 99.4% |
| 8 | 80000 | 307.16 | 0.72 | 304.9 | 1.007 | 7.94 | 99.3% |
| 16 | 160000 | 614.90 | 1.53 | 609.8 | 1.008 | 15.87 | 99.2% |

<p align="center"><img src="results_final/weak_time.svg" alt="Weak-scaling run time" width="60%"></p>

_The time grows with P even though each rank keeps the same number of particles, as expected for an all-pairs method. The measured curve lies on the ideal one: at 16 ranks the ideal is 609.8 s and the measurement 614.9 s, 1.008 times the ideal._

<p align="center"><img src="results_final/weak_speedup.svg" alt="Weak-scaling work-normalised speedup" width="49%"> <img src="results_final/weak_efficiency.svg" alt="Weak-scaling work-normalised efficiency" width="49%"></p>

_Left: the work-normalised speedup reaches 15.9 at 16 ranks, against an ideal of 16. Right: the work-normalised efficiency stays above 99% at every P._

The weak-scaling result agrees with the strong-scaling one: the time grows exactly like the number of pairs per rank, and at 16 ranks the time is 1.008 times the ideal.

This agrees with Gustafson's law: when the problem grows with the machine, the parallel part dominates. Per force evaluation each rank computes about n x N = n^2 x P pairs (n = N/P) and receives n x P particles, so computation and communication both grow linearly with P, and the waiting for messages stays a small fraction of the run at every P. The uneven density of the Plummer sphere does not cause load imbalance, because in the direct method every rank has exactly the same number of pairs.

### The energy check

At 16 ranks the run takes 1.008 times the ideal time. The force computation is almost exactly at the ideal (604.5 s against 603.9 s) and the waiting for messages does not grow, so almost all of the difference comes from the energy check, which is computed twice per run (start and end):

| P | N | Energy time (s) | Share of total | Energy / force, per evaluation |
|---:|---:|---:|---:|---:|
| 1 | 10000 | 0.36 | 0.94% | 0.48 |
| 16 | 160000 | 10.4 | 1.70% | 0.87 |

With one rank an energy evaluation costs about half a force evaluation, as expected, because it counts each pair once. With 16 ranks it costs 1.8 times more than that. This is the same "i smaller than j" imbalance seen in strong scaling: the rank with the lowest indices has the most pairs, and `MPI_Allreduce` waits for it. A balanced energy check would take about 6 s instead of 10.4 s at P = 16: the 4.4 s of difference are almost all of the 5.1 s between the measured time (614.9 s) and the ideal one (609.8 s). The loss could be removed by computing the energy less often or by dividing the pairs evenly.

These are single-node measurements: they cannot predict the behaviour over several nodes, where the network would matter.

---

## Mapping MPI ranks and OpenMP threads onto the node

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>100,000"]
  B["<b>Run</b><br/>the same 64 cores as 8<br/>ranks x 8 threads, 2 x 32<br/>and 64 x 1; 20 steps,<br/>overlapped ring; 1 warm-<br/>up + 5 runs each; CPU<br/>masks printed by Slurm"]
  C["<b>Measure</b><br/>total, force, waiting and<br/>energy time; Gpairs/s"]
  D["<b>Analyse</b><br/>median, mean and spread;<br/>compare the phases of the<br/>three mappings"]
  E["<b>Result</b><br/>one rank per socket<br/>fastest (13.99 s against<br/>14.28 s); the difference<br/>is in the energy phase"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

The same 64 cores can be used in many ways: 64 ranks with one thread, 8 ranks with 8 threads, 2 ranks with 32 threads, and so on. Fewer ranks mean fewer and shorter ring exchanges, but more threads sharing memory. A common choice is one rank per NUMA domain, so that the threads of each rank use nearby memory. We compared it with one rank per socket and one rank per core, using the same 64 cores:

- one rank per NUMA domain: P = 8 ranks x T = 8 threads;
- one rank per socket: P = 2 x T = 32;
- one rank per core: P = 64 x T = 1.

Set-up:

- N = 100,000, 20 steps, overlapped ring, exact square root, four partial sums, double precision;
- inside each rank the loop over the home particles is split among the threads with `#pragma omp parallel for schedule(static)`, which gives each thread an equal, contiguous block of particles;
- MPI is started with `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`: only the main thread calls MPI, never inside an OpenMP parallel region;
- no `critical` or `atomic` sections and no locks: each thread writes only its own particles, and all sums between threads are OpenMP reductions;
- one warm-up and five measured runs per mapping;
- the CPU masks printed by Slurm confirm that each rank received its own, non-overlapping set of cores (in the NUMA mapping each rank owns exactly one block of 8 cores), and that consecutive ranks are placed alternately on the two sockets.

| Mapping | Ranks P | Threads T | Median total (s) | Mean total (s) | Spread s (s) | Median force (s) | Median waiting for messages (s) | Median Gpairs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| One rank per NUMA domain | 8 | 8 | 14.275566 | 14.283084 | 0.027015 | 12.909107 | 0.126690 | 16.267422 |
| One rank per socket | 2 | 32 | 13.986056 | 13.988024 | 0.012181 | 12.905388 | 0.051432 | 16.272110 |
| One rank per core | 64 | 1 | 14.025092 | 14.045257 | 0.054429 | 12.852373 | 0.101077 | 16.339232 |

The three mappings are very close: one rank per socket is the fastest, with 13.99 s against 14.28 s for one rank per NUMA domain and 14.03 s for one rank per core.

- The force computation is the same in all three: about 16.3 billion pairs per second and about 12.9 s in all three mappings.
- Waiting for messages is lowest with the socket mapping (only 2 ranks in the ring), but it is at most 0.13 s in all cases, so it cannot explain the 0.29 s difference.
- Most of the difference is in the energy check: 1.34 s with the NUMA mapping, 1.07 s with the socket mapping and 1.09 s with one rank per core. From these data we cannot tell where the extra 0.3 s of the NUMA mapping come from.

So one rank per NUMA domain is not the fastest mapping in this test. For a program limited by arithmetic, whose data fit in the caches, memory locality matters less than expected, and the best mapping has to be measured. The largest energy drift of these runs is 4.94e-7.

---

## Newton's third law: computing each pair only once

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>10,000, 5 steps, 1 rank"]
  B["<b>Run</b><br/>direct kernel vs Newton<br/>kernel (private force<br/>copies), T = 1, 2, 4, 8,<br/>16 threads, 5 runs each"]
  C["<b>Measure</b><br/>force time, energy drift"]
  D["<b>Analyse</b><br/>ratio Newton / direct;<br/>cost of imbalance and<br/>private copies"]
  E["<b>Result</b><br/>Newton faster with 1-2<br/>threads, slower from 4<br/>threads (1.65x slower<br/>with 16)"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

The force computation takes most of the run time, so it is the obvious place to look for speed. The next sections test several classic optimisations one at a time, and report the result also when an idea does not pay off.

For this section and the next two, unless stated otherwise, the tests use a small, quick configuration so that many variants can be compared under the same conditions:

- N = 10,000 particles, 5 steps, dt = 1e-4, epsilon = 0.05, energy only at the start and end;
- one rank, first with one thread and then with 2, 4, 8 and 16 threads;
- five repetitions per variant, no warm-up; 55 runs per thread count (11 variants x 5), 275 runs in total.

A single rank keeps communication out of the picture, and starting from one thread shows the effect of each idea on the computation before thread effects come in.

Newton's third law says that the force of particle j on particle i is equal and opposite to the force of i on j. The direct loop computes both, so in principle we could compute each pair once and use the result twice, halving the arithmetic.

In the direct loop each thread writes only the forces of its own particles. With the third law, a thread that handles the pair (i, j) must also update particle j, which may belong to another thread. Two threads could then update the same particle at the same time, and one update would be lost. This must be prevented, and that has a cost.

Our implementation:

- each thread has its own private copy of the force arrays;
- inside one `#pragma omp parallel` region, the rows i are split among the threads with `#pragma omp for schedule(static)`; each thread computes the pairs (i, j) with j > i once and writes into its own copy;
- a second `#pragma omp parallel for` adds all the copies together;
- no locks or atomic operations in the pair loop.

The price is extra memory (three arrays per thread) and the work of clearing and summing the copies.

The first test uses 1 rank and 1 thread.

| Kernel | Median total (s) | Mean total (s) | Spread s (s) | Median force (s) |
|---|---:|---:|---:|---:|
| Direct (every pair twice) | 2.601807 | 2.605978 | 0.007693 | 2.235868 |
| Newton (every pair once) | 1.767468 | 1.768040 | 0.003336 | 1.401407 |

The third law makes the total time 1.47 times shorter and the force phase 1.60 times shorter. This is less than the ideal factor of 2 because only the arithmetic is halved: the loop still has to read the particles, and the extra bookkeeping adds some work. Both kernels give the same energy drift (1.2974058e-7).

With one thread there is no conflict, so the comparison was repeated with 2, 4, 8 and 16 threads (force time).

| Threads | Direct force (s) | Newton force (s) | Newton / direct |
|---:|---:|---:|---:|
| 1 | 2.2359 | 1.4014 | 0.63 |
| 2 | 1.1174 | 1.0514 | 0.94 |
| 4 | 0.5589 | 0.6144 | 1.10 |
| 8 | 0.2801 | 0.3401 | 1.21 |
| 16 | 0.1407 | 0.2326 | 1.65 |

<p align="center"><img src="results_final/newton_threads_time.svg" alt="Newton's third law vs direct kernel: force time" width="49%"> <img src="results_final/newton_threads_ratio.svg" alt="Newton's third law vs direct kernel: time ratio" width="49%"></p>

_Left: the direct kernel follows the ideal line, halving its time at every doubling of threads, while Newton's kernel improves much less. Right: Newton is faster with 1 and 2 threads and slower from 4 threads on, 1.65 times slower with 16._

The main reason is how the work is divided. In Newton's kernel, row i contains only the pairs with j > i, so the first rows are long and the last rows almost empty. `schedule(static)` gives each thread an equal number of rows, so the thread with the first rows has much more work than the others, and everybody waits for it. The energy check shows the same kind of imbalance between MPI ranks.

### When the saving outweighs the cost of the conflict, and when it does not

Three effects decide the result:

- saved arithmetic: with one thread the Newton force time is 0.63 of the direct one (1.4014 s against 2.2359 s). It is not 0.5, because each pair now also writes the force of particle j back to memory, while the direct loop keeps its sums in registers;
- load imbalance: with `schedule(static)` on the triangular loop j > i the first thread gets the longest rows, and the imbalance grows with the number of threads;
- conflict resolution: the T private copies must be cleared before the loop and added after it. Each thread works on arrays of size N, work that does not shrink when threads are added, while its share of pairs, N^2 / T, does.

The direct kernel has none of these costs, and its force time follows D(1) / T. So the saving of the one-thread run is lost as threads are added: Newton wins with 1 and 2 threads (0.63 and 0.94) and loses from 4 threads on (1.10, 1.21, 1.65).

The saving therefore outweighs the conflict resolution only with few threads writing into the same arrays, a balanced split of the rows (for example `schedule(dynamic)`, or pairing a long row with a short one), a large N per thread and an expensive pair computation. It does not pay off with many threads on a small N, with `atomic` or `critical` updates (every pair would pay a synchronised write), with a cheap vectorised pair computation, whose scattered writes to particle j do not become faster, or across MPI ranks, where the force on j would have to be sent back to its owner.

For our production runs the direct kernel is the better choice: with many threads Newton loses, as measured, and with many ranks most pairs involve particles of other ranks, where the third law would require sending forces back.

Note also that the Newton kernel computes N(N-1)/2 pairs, while the Gpairs/s formula counts N(N-1), so its throughput is an effective value.

---

## A faster inverse square root

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>10,000, 5 steps, 1 rank"]
  B["<b>Run</b><br/>exact, approx1 (rsqrt14 +<br/>1 Newton step), approx2<br/>(+ 2 steps); T = 1-16<br/>threads, 5 runs each"]
  C["<b>Measure</b><br/>force and total time,<br/>energy drift"]
  D["<b>Analyse</b><br/>median force and total<br/>time; drift difference<br/>from exact"]
  E["<b>Result</b><br/>force time 2.24 s (exact),<br/>0.27 s (approx1), 0.35 s<br/>(approx2) with 1 thread"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

For every pair the force needs 1 / sqrt(r^2 + epsilon^2), and an exact square root followed by a division is one of the slowest operations in the loop. Modern processors have a special instruction that gives a rough approximation of 1/sqrt(x) very quickly. On our AVX-512 processor it is `_mm512_rsqrt14_pd`, accurate to about 14 bits (4 significant digits). The approximation can be improved with one or two cheap Newton-Raphson correction steps; each step roughly doubles the number of correct digits:

```text
y_new = y x (1.5 - 0.5 x q x y^2)      where y is the estimate of 1/sqrt(q)
```

The solver offers three options:

- `exact`: the standard square root and division;
- `approx1`: hardware approximation plus one correction step;
- `approx2`: hardware approximation plus two correction steps.

Implementation:

- approximate options: a separate hand-written loop that processes eight pairs at once with AVX-512 intrinsics (`_mm512_rsqrt14_pd` for the estimate, `_mm512_fmadd_pd` for the fused multiply-adds);
- exact option: the ordinary loop.

So the comparison is between two complete implementations, not only between two square-root instructions.

| Method | Median total (s) | Mean total (s) | Spread s (s) | Median force (s) | Largest energy drift |
|---|---:|---:|---:|---:|---:|
| exact | 2.602213 | 2.603644 | 0.004365 | 2.235830 | 1.2974058018291037e-7 |
| approx1 | 0.635762 | 0.637167 | 0.003193 | 0.270813 | 1.2974022806770565e-7 |
| approx2 | 0.723951 | 0.722741 | 0.002251 | 0.354722 | 1.2974058018291037e-7 |

The force phase becomes 8.3 times faster with one correction step and 6.3 times faster with two. The whole solver becomes about 4 times faster; it cannot gain as much as the force phase because the remaining 0.37 s (reading input, energy checks, moving particles) is not accelerated. The second correction step costs about 0.08 s more, as expected from the extra arithmetic.

Most of the gain comes from the fully vectorised loop, which handles eight pairs per instruction, while the exact path does one pair at a time. The timings do not allow us to split the gain between "faster square root" and "vector instructions".

The same comparison with 2, 4, 8 and 16 threads gives:

| Threads | Exact force (s) | approx1 force (s) | approx2 force (s) | Exact total (s) | approx1 total (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.2358 | 0.2708 | 0.3547 | 2.6022 | 0.6358 |
| 2 | 1.1175 | 0.1356 | 0.1774 | 1.3790 | 0.3960 |
| 4 | 0.5590 | 0.0680 | 0.0888 | 0.7122 | 0.2203 |
| 8 | 0.2803 | 0.0345 | 0.0450 | 0.3649 | 0.1194 |
| 16 | 0.1413 | 0.0178 | 0.0231 | 0.1874 | 0.0638 |

The gain in the force computation stays at about 8 times (one correction) and 6 times (two corrections) at every thread count, so the fast version parallelises in the same way as the exact one. The gain on the total time, however, falls from 4.1 to 2.9 times as threads are added. This is Amdahl's law on a small scale: the force phase shrinks by a factor of 8, but the other parts of the run (reading the input and the energy check, which always uses the exact square root) are not accelerated, so they become a larger share of what remains.

One correction step brings the approximation to roughly 8 significant digits, not the 16 of full double precision; two steps come much closer but do not guarantee identical results. In our runs the energy drift of `approx1` differs from the exact one by only 3.5e-13, and `approx2` gives the same printed value as `exact`.

The energy routine always uses the exact square root, so the check cannot hide the error of the approximation. These runs are short (5 steps), so the stronger test is the convergence study below.

### Time-step convergence study

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>10,000, 3 seeds, fixed<br/>physical time 0.01"]
  B["<b>Run</b><br/>dt = 1e-4, 5e-5, 2.5e-5<br/>(100, 200, 400 steps) x<br/>exact, approx1, approx2;<br/>1 rank x 8 threads,<br/>energy at every step; 27<br/>runs"]
  C["<b>Measure</b><br/>largest energy drift of<br/>each run"]
  D["<b>Analyse</b><br/>drop = drift(dt) /<br/>drift(dt/2);<br/>|drift(approx) -<br/>drift(exact)|"]
  E["<b>Result</b><br/>drop 4.02 and 4.00<br/>(second order); approx1<br/>adds a constant 3.8e-12,<br/>approx2 adds nothing"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

The leapfrog method is second order: if the time step is halved, its error should become about four times smaller. So we simulate the same stretch of physical time with smaller and smaller time steps and watch the energy drift fall. For each halving of dt we compute:

```text
drop = drift(dt) / drift(dt/2)          (ideal value 4 for a second-order method)
difference vs exact = | drift(approx) - drift(exact) |   (same initial condition)
```

An approximate square root adds an extra error that does not shrink with the time step: if an approximate version stopped improving while the exact one kept falling, the approximation would have become the limiting error.

Set-up:

- N = 10,000 Plummer particles, one rank with 8 threads, fixed physical time 0.01;
- dt = 1e-4 (100 steps), dt/2 = 5e-5 (200 steps), dt/4 = 2.5e-5 (400 steps);
- energy checked at every step, three initial conditions (seeds) per point;
- `exact`, `approx1` and `approx2`: 27 runs in total;
- 8 cores of a shared node, so the timings are not used.

| dt | Steps | exact: largest drift | Drop vs previous dt | approx1: largest drift | approx2: largest drift | Largest difference approx1 vs exact |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 100 | 2.5305e-7 | - | 2.5304e-7 | 2.5305e-7 | 3.7e-12 |
| 5e-5 | 200 | 6.2991e-8 | 4.02x | 6.2989e-8 | 6.2991e-8 | 3.8e-12 |
| 2.5e-5 | 400 | 1.5731e-8 | 4.00x | 1.5728e-8 | 1.5731e-8 | 3.8e-12 |


The results lead to three conclusions:

- The integrator behaves as theory says. Halving the time step reduces the drift by 4.02 and then 4.00 times, which confirms that the leapfrog is second order and that the energy check really measures the integration error.
- The energy check cannot tell `approx2` apart from `exact`. In every run its drift agrees with the exact one to all printed digits (about 16 significant figures), so for energy conservation two correction steps are as good as the standard square root. This does not prove that positions and velocities are identical, because the drift is a global number and small differences in single particles could cancel in it.
- `approx1` leaves a small, constant mark. Its drift differs from the exact one by about 3.8e-12 at every time step, an error that does not shrink with dt, as expected from a fixed approximation error. It is 60,000 times smaller than the integration error at dt = 1e-4 and still more than 4000 times smaller at dt/4. Following the factor-of-four trend, the two would only become comparable at a time step about 60 times smaller than our smallest one (around 4e-7).

A direct comparison of the final positions with the exact version would be the natural next check, because the energy drift alone cannot detect errors that cancel out.

---

## Independent partial sums and the limits of the processor - FMA

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>10,000, 5 steps, 1 rank"]
  B["<b>Run</b><br/>exact loop with 1, 2, 4,<br/>8 independent partial<br/>sums; T = 1-16 threads, 5<br/>runs each"]
  C["<b>Measure</b><br/>force time"]
  D["<b>Analyse</b><br/>median force time;<br/>theoretical peak = cores<br/>x frequency x FLOP per<br/>cycle"]
  E["<b>Result</b><br/>chains: almost no<br/>change; threads: time<br/>halves at every doubling;<br/>loop limited by sqrt<br/>and division"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

For each particle the force is a long sum, a_x = a_x + (source 1) + (source 2) + ..., and each addition must wait for the previous one (a latency of a few cycles). With k independent partial sums, k chains advance at the same time and are added at the end. For a loop made only of multiply-adds the gain grows until all arithmetic units are busy.

The exact force loop is unrolled by hand into 1, 2, 4 or 8 partial sums (`--accumulators`). The version with two sums per component:

```c
#pragma omp parallel for schedule(static)                 // threads split the targets i
for (i = 0; i < home->n; ++i)
{
  dtype ax0 = 0, ay0 = 0, az0 = 0, ax1 = 0, ay1 = 0, az1 = 0;
  #pragma omp simd reduction(+ : ax0, ay0, az0) reduction(+ : ax1, ay1, az1)
  for (j = 0; j < source_n_unrolled; j += 2)              // two sources per iteration
  {
    /* pair (i, j):   dx0, dy0, dz0, r2_0, invr0 = 1/sqrt(r2_0), s0 = G m invr0^3 */
    ax0 += dx0 * s0;  ay0 += dy0 * s0;  az0 += dz0 * s0;  // chain 0
    /* pair (i, j+1): same formulas */
    ax1 += dx1 * s1;  ay1 += dy1 * s1;  az1 += dz1 * s1;  // chain 1
  }
  ax += ax0 + ax1;  ay += ay0 + ay1;  az += az0 + az1;    // combine; then a tail loop
}
```

- threads: `#pragma omp parallel for schedule(static)` gives each thread different targets, so no synchronisation is needed;
- vectorisation: `#pragma omp simd` with the partial sums in `reduction` asks the compiler to process several sources at once;
- partial sums: k independent chains of additions per component.

Set-up: N = 10,000, 5 steps, one rank, exact square root, energy only at the start and at the end; 1, 2, 4, 8 partial sums with 1, 2, 4, 8, 16 threads; five runs per case, same energy drift in every case.

| Threads | 1 chain (s) | 2 chains (s) | 4 chains (s) | 8 chains (s) |
|---:|---:|---:|---:|---:|
| 1 | 2.2566 | 2.2340 | 2.2355 | 2.2488 |
| 2 | 1.1274 | 1.1165 | 1.1174 | 1.1240 |
| 4 | 0.5638 | 0.5581 | 0.5588 | 0.5621 |
| 8 | 0.2828 | 0.2799 | 0.2800 | 0.2819 |
| 16 | 0.1420 | 0.1406 | 0.1412 | 0.1416 |

Along a row the time hardly changes: two and four chains are slightly faster than one, and eight chains are slightly slower than two and four. Down a column the time halves at every doubling of the threads.

Why more threads help and more partial sums do not:

- each thread runs on its own core, with its own square-root and division unit; the pairs of each thread are N x N / T, so the limiting work is divided by T. Partial sums stay inside one core and add no such unit;
- one pair costs about 14 cycles (0.266 billion pairs per second per core: 6 x 10,000 x 9,999 pairs in 2.2566 s at 3.85 GHz), mostly in the exact square root and division, against a few cycles for one addition. A new term reaches the sum only every 14 cycles, so even one chain waits for the pair computation, not for the previous addition;
- with one partial sum there are already three independent chains (ax, ay, az), and the next pairs do not depend on the sums, so the core already overlaps them;
- the loop runs one pair at a time with separate multiplications and additions: with `-std=c11` GCC does not fuse them into FMAs.

The last point was checked with a separate test: the same solver compiled once more with `-ffp-contract=fast`, which lets GCC use FMAs. Same set-up as above, one thread, one partial sum, five runs each, same node:

| Build | Median force (s) | Spread s (s) | Median total (s) | Largest energy drift |
|---|---:|---:|---:|---:|
| `-std=c11` (used in this report) | 2.2603 | 0.0012 | 2.6254 | 1.7300258730569728e-8 |
| `-std=c11 -ffp-contract=fast` | 2.0813 | 0.0009 | 2.4419 | 1.7300258730569728e-8 |

With FMAs the force time drops from 2.26 to 2.08 s (ratio 0.92), much more than the spread, and the energy drift is identical to all printed digits. In the exact loop, r^2 = dx dx + dy dy + dz dz + eps^2 before the square root and a += d s after it are multiplications followed by additions; with `-ffp-contract=fast` GCC fuses each of these pairs into one FMA, so each particle pair needs fewer instructions and a shorter chain before the square root. The square root and the division are not affected and remain the largest cost, but they are not the only one. Partial sums still do not help, because the chain of additions is not the limit in either build.

Partial sums would matter only in a loop where one iteration costs less than the latency of an addition.

### Peak floating-point speed of the kernel

The machine peak is set by the fused multiply-add (FMA, a x b + c, two operations). Each Zen 4 core issues two FMAs per cycle on 4 doubles (AVX-512 runs as two 256-bit halves), 16 operations per cycle, so one socket at 3.85 GHz gives

```text
machine peak FP64 = 32 cores x 3.85e9 x 16 = 1.97 TFLOP/s
```

A kernel reaches this only if it is made of independent vector FMAs. Our kernel is limited by the slowest resource it uses: kernel peak = cores x (pairs per second per core) x (operations per pair).

- memory is not the limit: each source is reused for all targets, the data fit in the caches, and the throughput per core is the same from 1 to 32 cores (profiling section);
- exact kernel: the square root and the division, scalar, about 14 cycles per pair;
- approximate kernel: `rsqrt14` plus Newton steps in an AVX-512 loop (8 pairs per instruction), mostly multiplications and FMAs, so the limit moves towards the FMA and multiply units; this kernel is 8.3 times faster.

With the conventional count of about 20 operations per pair (square root and division counted as one each; only a convention) and the one-core throughput multiplied by 32:

| Kernel | Pairs/s per core | Pairs/s per socket | Equivalent FLOP/s | Fraction of machine peak |
|---|---:|---:|---:|---:|
| exact | 0.266e9 | 8.5e9 | 0.17 TFLOP/s | about 0.09 |
| approx1 (AVX-512) | about 2.2e9 | about 71e9 | about 1.4 TFLOP/s | about 0.7 |

On one socket the peak of our kernel is therefore set by its bottleneck: the square-root and division unit for the exact version, the vector FMA and multiply units for the approximate one.

---

## AoS versus SoA

```mermaid
flowchart LR
  A["<b>Input</b><br/>separate OpenMP benchmark<br/>(no MPI), N = 10,000"]
  B["<b>Run</b><br/>AoS and SoA layouts, T =<br/>1, 2, 4, 8; 1 warm-up + 3<br/>timed force evaluations<br/>per run; 5 runs"]
  C["<b>Measure</b><br/>force time (OpenMP<br/>clock), checksum of the<br/>accelerations"]
  D["<b>Analyse</b><br/>median, Gpairs/s, time<br/>ratio AoS / SoA"]
  E["<b>Result</b><br/>SoA slower at every<br/>thread count (T(SoA) /<br/>T(AoS) = 1.10-1.12)"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

There are two natural ways to store particles. The array of structures (AoS) keeps all data of one particle together: {x, y, z, vx, vy, vz, m}, then the next particle. The structure of arrays (SoA) keeps one array for all x values, one for all y values, and so on. SoA is usually recommended for vector instructions: consecutive x values are adjacent, and one instruction loads 4 (AVX2) or 8 (AVX-512) of them, while with AoS they are 72 bytes apart and must be collected one by one ("gather"). If the loop is not vectorised, both layouts process one pair at a time and SoA has no special advantage.

A separate small benchmark computes the same forces with both layouts:

- loop over the target particles: `#pragma omp parallel for schedule(static)` in both layouts;
- loop over the sources: `#pragma omp simd reduction(+ : ax, ay, az)` in the SoA version, no `simd` pragma in the AoS version;
- one process (OpenMP only, no MPI) with 1, 2, 4 and 8 threads;
- N = 10,000, exact square root;
- one warm-up and three timed force evaluations per run; five runs per layout and thread count;
- only the force computation is timed, with the OpenMP clock; same compiler and flags as the solver.

At the end the program prints a checksum, the sum of all acceleration components. By Newton's third law the forces cancel, so the total is close to zero (about 8.5e-10) whatever the single forces are: equal checksums exclude gross errors, such as missing particles or invalid numbers, but do not prove identical forces.

| Layout | N | Threads | Median force time (s) | Median Gpairs/s | Checksum difference vs AoS |
|---|---:|---:|---:|---:|---|
| AoS | 10000 | 1 | 1.010227 | 0.297 | baseline |
| SoA | 10000 | 1 | 1.129805 | 0.266 | 0.0 |
| AoS | 10000 | 2 | 0.509064 | 0.589 | baseline |
| SoA | 10000 | 2 | 0.565051 | 0.531 | 0.0 |
| AoS | 10000 | 4 | 0.254240 | 1.180 | baseline |
| SoA | 10000 | 4 | 0.282475 | 1.062 | 0.0 |
| AoS | 10000 | 8 | 0.130199 | 2.304 | baseline |
| SoA | 10000 | 8 | 0.143438 | 2.091 | 0.0 |

<p align="center"><img src="results_final/layout_force_time.svg" alt="AoS vs SoA layout benchmark" width="60%"></p>

_Against the usual expectation, the SoA layout is slower than AoS at every thread count (T(SoA) / T(AoS) = 1.10-1.12). The two layouts give the same checksum, which excludes gross errors but does not prove identical forces._

To see what really happens, we looked at the compiler report (`-fopt-info-vec`):

- in neither layout is the force loop vectorised: both process one pair at a time;
- the cause is the branches inside the loop, such as the `if` that chooses between the exact and the approximate square root. The compiler reports "control flow in loop";
- the report does say "loop vectorized" for the SoA version, but that message refers to a small helper loop created by `#pragma omp simd` (clearing the partial sums), not to the force loop;
- the failed `#pragma omp simd reduction(+ : ax, ay, az)` leaves a cost behind. In the AoS loop the three sums ax, ay, az stay in registers. In the SoA loop they are kept in memory: at every pair each sum is read, updated and written back.

This explains the result. Both layouts run the same scalar arithmetic, and SoA is slower (T(SoA) / T(AoS) = 1.10-1.12 at every thread count) because every pair waits for three sums to go through memory instead of staying in registers. The benchmark therefore does not measure an advantage of AoS over SoA: it measures a vectorisation request that fails and leaves extra work behind.

In summary, SoA is faster only when the loop is really vectorised. The main solver shows this case: it uses SoA with a hand-written AVX-512 loop and the approximate inverse square root (`_mm512_rsqrt14_pd`), loads 8 consecutive x, y and z values at once, and makes the force computation about eight times faster. With AoS that loop would need gathers. To see the SoA advantage in this benchmark, the branches would have to be removed from the inner loop.

Hardware counters of the vector instructions actually executed (with `perf` or PAPI) would be needed to prove this explanation, but neither tool is available on Orfeo, so this report has no measured instruction counts, cache misses or branch misses.

---

## Compiling for this exact processor or for any modern processor

```mermaid
flowchart LR
  A["<b>Input</b><br/>the same source built<br/>twice: -march=native and<br/>-march=x86-64-v3"]
  B["<b>Run</b><br/>test A: N = 10,000, 5<br/>steps, 8 ranks, exact<br/>sqrt test B: N = 100,000,<br/>10 steps, 32 ranks, exact<br/>and approx1; 1 warm-up +<br/>5 runs"]
  C["<b>Measure</b><br/>total time, energy drift"]
  D["<b>Analyse</b><br/>portable vs native =<br/>T(x86-64-v3) / T(native)<br/>- 1"]
  E["<b>Result</b><br/>exact: same time;<br/>approx1: portable build<br/>4.83x slower (no AVX-512 loop)"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

The option `-march=native` lets the compiler use every instruction of the build machine, including AVX-512 (8 doubles per vector). The option `-march=x86-64-v3` targets a common baseline supported by most processors of the last decade, with AVX2 (4 doubles per vector) but without AVX-512. The container uses the portable option, so we want to know how much performance it loses. The last column of the table is:

```text
portable vs native = T(x86-64-v3) / T(native) - 1
```

The same solver was built both ways and compared in two tests:

- short test: N = 10,000, 5 steps, 8 ranks with one thread each, exact square root;
- larger test: N = 100,000, 10 steps, 32 ranks with one thread each, exact square root and approx1; one warm-up and five measured runs per target;

| Test | Square root | native median ± s (s) | x86-64-v3 median ± s (s) | Portable vs native |
|---|---|---:|---:|---:|
| N = 10,000, 5 steps, 8 ranks | exact | 0.385524 ± 0.002109 | 0.384304 ± 0.006533 | -0.3% |
| N = 100,000, 10 steps, 32 ranks | exact | 15.010416 ± 0.106892 | 15.061305 ± 0.011998 | +0.3% |
| N = 100,000, 10 steps, 32 ranks | approximate (approx1) | 3.723728 ± 0.005836 | 17.980670 ± 0.006151 | +383% (4.83 times slower) |

Results:

- exact square root: the two builds are equally fast within the spread, for both problem sizes (0.386 s against 0.384 s, 15.01 s against 15.06 s). In both builds the exact loop is ordinary scalar code, so the wider AVX-512 vectors have nothing to improve;
- approximate square root (approx1): the portable build is 4.8 times slower (17.98 s against 3.72 s), and even slower than its own exact version (17.98 s against 15.06 s). The speed of approx1 comes from the hand-written AVX-512 loop, which processes eight pairs at once and exists only when the compiler targets AVX-512. In the portable build it is replaced by a scalar loop (single-precision estimate plus correction steps), which is slower than the exact square root;
- energy drift: the exact runs agree to all printed digits, and the approximate runs differ by about 3e-12.

What this experiment measures, and what it does not:

- it measures how much the portable build loses for our code as it is written: the same source, compiler and node, with only `-march` changed;
- it does not measure the value of AVX-512 over AVX2 for the same vector code. The portable build has no AVX2 version of the fast loop, so the approx1 comparison is between a vector loop and a scalar one, and its result was predictable from the code. Measuring it would need the same kernel in an AVX2 version (4 doubles per vector) and an AVX-512 version (8 doubles per vector), timed on the same node; in theory the gain is at most a factor of 2, and less on Zen 4, which executes AVX-512 as two 256-bit halves.

---

## Hiding communication behind computation

```mermaid
flowchart LR
  A["<b>Input</b><br/>Plummer sphere, N =<br/>100,000, 20 steps, 1<br/>thread per rank"]
  B["<b>Run</b><br/>sendrecv vs overlap at P<br/>= 2, 8, 32; 1 warm-up + 5<br/>runs each, both versions<br/>on the same node"]
  C["<b>Measure</b><br/>total time, waiting time<br/>(comm_wait), energy drift"]
  D["<b>Analyse</b><br/>waiting time of the two<br/>versions; change of<br/>total time +- combined<br/>spread"]
  E["<b>Result</b><br/>no measurable gain;<br/>waiting times become more<br/>regular"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

The ring exchange exists in two versions in the program:

- blocking (`sendrecv`): the rank computes with its current block, then calls `MPI_Sendrecv`, which sends the block to the right neighbour and receives the next one from the left neighbour; the rank is idle while the message travels;
- overlapped (`overlap`): the rank starts the exchange of the next block with `MPI_Irecv` and `MPI_Isend`, computes with the current block, and at the end waits with `MPI_Waitall`; ideally the message travels while the rank is busy and has already arrived when the computation ends.

We want to measure how much of the communication is really hidden. The waiting time (`comm_wait`) is the time spent inside the exchange call: `MPI_Sendrecv` for the blocking version, `MPI_Waitall` for the overlapped one. We compare the total time and the waiting time of the two versions.

The set-up was:

- N = 100,000 particles, 20 time steps, P = 2, 8 and 32 ranks, one thread each;
- direct kernel, exact square root, four partial sums, double precision; energy computed only at the start and at the end;
- one warm-up and five measured runs for each version and each P;
- the two versions at the same P ran one after the other on the same node (P = 2 on genoa001, P = 8 on genoa003, P = 32 on genoa004), with the same executable, so each pair is a like-for-like comparison.

For each initial condition the energy drift is identical in the two versions, consistent with the overlapped exchange not changing the results (both versions add the blocks in the same order).

| P | Blocking total (s) | Overlapped total (s) | Overlapped / blocking | Blocking wait (s) | Overlapped wait (s) | Blocking wait as % of total |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 418.119 ± 0.701 | 417.872 ± 0.402 | 0.9994 | 0.435 ± 0.422 | 0.534 ± 0.111 | 0.10% |
| 8 | 106.249 ± 0.478 | 105.798 ± 0.410 | 0.9958 | 0.235 ± 0.389 | 0.188 ± 0.040 | 0.22% |
| 32 | 28.063 ± 0.390 | 27.967 ± 0.484 | 0.9966 | 0.507 ± 0.218 | 0.506 ± 0.007 | 1.81% |

Values are median ± spread s over five runs.

There is no measurable speedup. The ratio of the overlapped to the blocking total time is 0.9994, 0.9958 and 0.9966: the differences are smaller than the normal variation between runs. The force phase is also unchanged.

The waiting time does not decrease either. At 2 and 8 ranks waiting is only 0.1-0.2% of the run, and the blocking waiting time varies between runs as much as its own median, so the differences between the two versions (0.435 against 0.534 s, 0.235 against 0.188 s) are within the noise. At 32 ranks, where waiting grows to 1.8% of the total, the median waiting time is the same in both versions (0.507 s and 0.506 s): practically nothing is hidden.

What overlap does change is the regularity of the waiting time. With the blocking exchange some runs show occasional spikes of waiting (1.31 s at P = 2, 1.06 s at P = 8, 0.99 s at P = 32). With the overlapped exchange these spikes disappear, and the spread of the waiting time drops from 0.42 to 0.11 s, from 0.39 to 0.04 s, and from 0.22 to 0.007 s. Starting the exchange early absorbs occasional delays of a neighbour, but does not reduce the typical waiting.

The waiting time is longer than the transfer time alone. At 32 ranks each rank exchanges blocks of 3,125 particles, about 75 kB. Inside one node such a message should take a few tens of microseconds. Yet the measured waiting is about 0.75 ms per ring step (0.507 s divided by 21 force evaluations x 32 exchanges), more than ten times what the transfer alone needs. Each rank spends roughly 38 ms computing per ring step, so the waiting is still small. Two mechanisms explain why this time cannot be hidden:

1. Ranks wait for each other, not for data. The ring moves in lock-step, and if some cores are slightly slower (clock speed, shared caches and memory, two sockets), the faster ranks wait at every exchange. Overlap can hide the travel time of a message, not the wait for a slower neighbour.
2. Messages of tens of kilobytes use the "rendezvous" protocol, and without a background progress thread Open MPI moves the data only inside an MPI call. In our loop the next call after starting the exchange is `MPI_Waitall`, so much of the transfer still happens there. Calling `MPI_Test` during the computation, or enabling asynchronous progress, would be the next thing to try.

Even a perfect overlap could save at most the waiting time itself, which here is only 0.1-1.8% of the run. On one node this program is dominated by computation, so a small benefit was expected. Overlap would matter more with many more ranks, fewer particles per rank, or communication between nodes.

In these runs reading the input varies from 0.04 to 1.04 s at 32 ranks, more than the effect of overlap itself.

---

# CLOUD PART

---

## Building and running the container

HPC centres use Singularity, which can run images made with Docker. Our image:

- starts from Ubuntu 24.04 and installs the compiler, the OpenMPI development packages, the OSU micro-benchmarks (version 7.5.2) and our source code, then compiles the program inside the image;
- is sent to Docker Hub and converted on Orfeo into a Singularity image file;
- uses Ubuntu 24.04 and not 22.04 because the Orfeo MPI libraries need glibc 2.38 or newer: Ubuntu 22.04 has glibc 2.35 and could not load the cluster MPI, Ubuntu 24.04 has glibc 2.39 and works.

We chose a plain Ubuntu image and not a vendor HPC image, such as the NVIDIA HPC SDK images on `nvcr.io`, for four reasons:

- The program runs only on CPUs. Vendor HPC images are made for GPUs: they contain CUDA, GPU libraries and the vendor compiler, and weigh several gigabytes. The GENOA nodes have AMD CPUs and no GPU, so we need only GCC, OpenMP and the MPI headers. A small image is faster to download, to convert into a Singularity file and to start.
- The cluster MPI must replace the MPI of the image. This works because the image is built with Ubuntu's OpenMPI 4.1.6, which is compatible with Orfeo's OpenMPI 4.1.6. A vendor image brings its own MPI stack (for example HPC-X with UCX, in other versions). We would have to either use that MPI inside the container, losing the link with Slurm and the cluster network, or replace it with the cluster MPI and risk incompatible libraries.
- We control and document every component. The image contains only what the Dockerfile installs (see the table below), and anyone can rebuild it with `docker build`, without an account on a vendor registry.
- The comparison with the native build stays fair. Both builds use GCC (13.3 in the image, 14.3 on Orfeo). A vendor image would use a different compiler, and the native-versus-container comparison would mostly measure the compiler instead of the container.

A vendor image would be the better choice for a GPU code, or for multi-node runs on InfiniBand without a cluster MPI to mount, where its tuned MPI and network libraries are ready to use. The price is a much larger image tied to one vendor.

OpenMPI is installed inside the image because the `mpicc` compiler wrapper and the MPI header files are needed to build the program, but at run time the cluster's own MPI library replaces it. This is the key point for MPI programs in containers: MPI must talk to the cluster's launcher and network, which the container does not know, so the program is built with the container's MPI and run with the cluster's MPI, mounted inside the container. Finally, the container program is compiled with `-march=x86-64-v3` instead of `-march=native`, because the image should run on other machines too (see "Portability versus performance" below). The container build command, checked with `make -nB` inside the image, is:

```sh
mpicc -std=c11 -DNBODY_USE_DOUBLE -O3 -march=x86-64-v3 -Wall -Wextra -Wpedantic -fopenmp -o nbody_direct_hybrid nbody_direct_hybrid.c -lm
```

The image itself is built with Docker on a personal computer, sent to Docker Hub, and then converted into a Singularity image on Orfeo. The recipe uses `x86-64-v3` as its default target, and inside the image the OSU micro-benchmarks are compiled with the image's `mpicc`, in the same way as the native ones. The commands are:

```sh
# on a personal computer
docker build -t .../nbody-hpc:latest .
docker push .../nbody-hpc:latest

# on Orfeo
module load singularity/4.3.1
singularity pull --force nbody.sif docker://.../nbody-hpc:latest
```

So the native and the container environments are not identical:

| Component | Native | Container |
|---|---|---|
| Build environment | Orfeo (Fedora 41 based) | Ubuntu 24.04.5 LTS |
| Compiler | GCC 14.3.1 | GCC 13.3.0 |
| Compilation target | `-march=native` (AVX-512) | `-march=x86-64-v3` (AVX2) |
| MPI used to build | Orfeo Open MPI 4.1.6rc4 | Ubuntu OpenMPI 4.1.6-7ubuntu2 |
| MPI used to run | Orfeo Open MPI 4.1.6rc4 | the same Orfeo Open MPI, mounted into the image |
| OpenMP runtime | libgomp 14.3.1 (Orfeo) | libgomp 14.2.0 (Ubuntu) |
| Other libraries | libm, hwloc 2.12.0 | Ubuntu glibc 2.39 and libm, Orfeo hwloc 2.12.0 |

The cluster's MPI and hwloc folders are mounted inside the container with `SINGULARITY_BINDPATH`, and `SINGULARITYENV_LD_LIBRARY_PATH` tells the program to look for libraries there first:

```sh
export SINGULARITY_BINDPATH=/opt/programs/openMPI/4.1.6:/opt/programs/openMPI/4.1.6,/opt/programs/hwloc/2.12.0:/opt/programs/hwloc/2.12.0
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/programs/openMPI/4.1.6/lib:/opt/programs/hwloc/2.12.0/lib
```

We checked the result with `ldd`, which lists the libraries a program really loads. Inside the container the program loads `libmpi.so.40`, `libopen-rte.so.40` and `libopen-pal.so.40` from `/opt/programs/openMPI/4.1.6/lib` and `libhwloc.so.15` from `/opt/programs/hwloc/2.12.0/lib`, which are exactly the same files used by the native program, while OpenMP (`libgomp.so.1`) comes from the image. If the container had silently used its own MPI, runs on several nodes could fail or use slow communication without any clear error. The first images failed because they did not have glibc 2.38, and later a part of the cluster MPI loaded an incompatible version of the UCX communication library from the image. 

#### Portability versus performance

The choice of the compilation target is a trade-off between portability and performance:

- `-march=native`: the fastest build, because it can use every instruction of the build machine (AVX-512 on Zen 4). But it runs only on processors with the same instructions: on an older or different CPU it stops with an "illegal instruction" error. It must be rebuilt on each machine.
- `-march=x86-64-v3`: one build that runs on almost every x86 processor of the last decade. This is what a container needs, because the same image must run on a laptop, in the cloud and on Orfeo. The price is that AVX-512 and the code written for it are not available.

---

## Native versus container

```mermaid
flowchart LR
  A["<b>Input</b><br/>the same inputs (seeds)<br/>as the strong- and weak-<br/>scaling runs"]
  B["<b>Run</b><br/>strong P = 1-32 and weak<br/>P = 1-16 inside<br/>Singularity (x86-64-v3<br/>image, cluster MPI<br/>mounted); 5 runs each"]
  C["<b>Measure</b><br/>total time, energy drift"]
  D["<b>Analyse</b><br/>overhead = container<br/>median / native median -<br/>1; combined spread"]
  E["<b>Result</b><br/>+0.19 to +0.36% in 10 of<br/>11 configurations; P = 32<br/>explained by a slower<br/>node"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

This is the main container test: does running the program inside Singularity make it slower? The N-body program is a useful test case, because almost all its time is spent computing and communication is small. So a slowdown could not come from the network: it would have to come from the container itself, or from the different software inside it.

We ran the strong- and weak-scaling experiments inside Singularity with the same settings, the same inputs (the same seeds) and five repetitions per point:

- strong scaling: N = 100,000, 100 steps, P = 1, 2, 4, 8, 16, 32 (the long points split over several Slurm jobs, as for the native runs);
- weak scaling: 10,000 particles per rank, 100 steps, P = 1, 2, 4, 8, 16;
- one thread per rank;
- image built from the same source code as the native executable, with the container's compiler and the portable target x86-64-v3, using the cluster's MPI library;
- native values: the runs of the strong- and weak-scaling experiments;

The times are the program's internal totals, so they do not include the start-up of the container, which is measured in the next section. The overhead is:

```text
overhead = 100 x (container median / native median - 1)      [%]
```

A positive value means the container is slower.

Strong scaling (the container speedup and efficiency use the container one-rank time, S(P) = T(1) / T(P) and E(P) = S(P) / P):

| P | N | Native median ± s (s) | Container median ± s (s) | Overhead | Container speedup | Container efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100000 | 3823.47 ± 2.27 | 3834.80 ± 0.54 | +0.30% | 1.00 | 100.0% |
| 2 | 100000 | 1920.50 ± 0.63 | 1925.21 ± 1.40 | +0.25% | 1.99 | 99.6% |
| 4 | 100000 | 960.53 ± 0.86 | 962.36 ± 1.31 | +0.19% | 3.98 | 99.6% |
| 8 | 100000 | 480.42 ± 1.01 | 481.46 ± 0.89 | +0.22% | 7.96 | 99.6% |
| 16 | 100000 | 240.56 ± 1.67 | 241.05 ± 1.17 | +0.20% | 15.91 | 99.4% |
| 32 | 100000 | 123.59 ± 0.02 | 120.94 ± 0.03 | -2.15% | 31.71 | 99.1% |

Weak scaling:

| P | N | Native median ± s (s) | Container median ± s (s) | Overhead |
|---:|---:|---:|---:|---:|
| 1 | 10000 | 38.11 ± 0.004 | 38.21 ± 0.01 | +0.27% |
| 2 | 20000 | 76.62 ± 0.03 | 76.84 ± 0.02 | +0.28% |
| 4 | 40000 | 153.42 ± 0.07 | 153.97 ± 0.07 | +0.36% |
| 8 | 80000 | 307.16 ± 0.72 | 308.26 ± 0.19 | +0.36% |
| 16 | 160000 | 614.90 ± 1.53 | 617.04 ± 1.61 | +0.35% |

In ten of the eleven configurations the container is slower by 0.19-0.36%. The difference is small but systematic: at P = 1 and 2 and in weak scaling it is larger than the run-to-run spread. Inside the container the strong scaling behaves like the native one: the efficiency stays between 99.1% and 99.6% up to 32 ranks. For this compute-bound program the overhead of the container is below 0.4% of the run time. The container also computes the same results: for every initial condition the energy drift is identical in the native and in the container run.

The two environments differ in more than the container itself: the compiler (GCC 13.3 against 14.3), the OpenMP runtime and the compilation target (x86-64-v3 against native). The compilation-target experiment measured 15.06 s against 15.01 s between the two targets with the exact square root on 32 ranks, a difference of the same size as the one measured here. So the small overhead is consistent with the different compilation, without any cost of Singularity itself; these measurements cannot separate the two effects.

The only exception is strong scaling at P = 32, where the container is 2.15% faster, about a hundred times the run-to-run spread. This is the node effect described in the strong-scaling section: the native 32-rank runs ran on genoa004, where the speed per rank at 32 ranks was lower (0.260 against 0.267 billion pairs per second), while the container runs ran on another node (genoa006). Measured against the container's own one-rank time, the container run at 32 ranks has an efficiency of 99.1%, in line with the other points. Running native and container alternately on the same node would remove this effect.

---

## Container start-up time

```mermaid
flowchart LR
  A["<b>Input</b><br/>the Singularity image"]
  B["<b>Run</b><br/>singularity exec <image><br/>true, 10 times"]
  C["<b>Measure</b><br/>wall time of each launch"]
  D["<b>Analyse</b><br/>median and spread"]
  E["<b>Result</b><br/>0.09 s median (first,<br/>cold launch 0.58 s)"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

Before the program starts, Singularity has to open the image and set up the container. This is a fixed cost per launch: negligible for a long run, but it could dominate a very short test. We measured it by launching a container that does nothing (`singularity exec <image> true`) ten times.

The ten launch times were 0.58 s for the first launch and 0.09 s for each of the other nine. With the median, the mean and the sample standard deviation s:

| Launches | Median (s) | Mean (s) | Standard deviation s (s) |
|---|---:|---:|---:|
| all 10 | 0.09 | 0.14 | 0.155 |
| 2-10 (after the first) | 0.09 | 0.09 | < 0.01 |

The first launch is probably a "cold start", when the image file is read from disk for the first time; after that it stays in memory. It is kept in the statistics of all ten launches, and it alone produces the whole standard deviation of 0.155 s. The other nine launches all took 0.09 s: their standard deviation is below the 0.01 s resolution of the timer, so identical values do not mean that the start-up time is perfectly constant.

Compared with runs that last from tens of seconds to over an hour, 0.1 s is negligible, which is why it does not show up in the solver comparison.

---

## Communication inside the container

```mermaid
flowchart LR
  A["<b>Input</b><br/>OSU micro-benchmarks, 2<br/>processes on 2 nodes, TCP<br/>forced (ob1, self,tcp)"]
  B["<b>Run</b><br/>osu_latency and osu_bw,<br/>native and inside the<br/>container, 5 runs per<br/>message size"]
  C["<b>Measure</b><br/>latency (us) and<br/>bandwidth (MB/s) for each<br/>message size"]
  D["<b>Analyse</b><br/>median +- s; difference =<br/>container / native - 1"]
  E["<b>Result</b><br/>latency 2.6-6.5% higher,<br/>bandwidth for large<br/>messages 4-5% lower"]
  A --> B --> C --> D --> E
  classDef io fill:#e8f1fb,stroke:#1f77b4,color:#111;
  classDef step fill:#f7f7f7,stroke:#555,color:#111;
  class A,E io;
  class B,C,D step;
```

To look at communication alone, without computation, we used the OSU micro-benchmarks, a standard tool for measuring MPI performance between two processes:

- `osu_latency` measures the time for a message to go from one process to the other: the message is sent back and forth many times, and the latency is half of the average round-trip time;
- `osu_bw` measures how much data per second one process can stream to the other: bandwidth = bytes sent / time, in MB/s.

Set-up:

- native and inside the container, two processes with one thread each, on two different nodes (`genoa012` and `genoa013`, confirmed by the Slurm accounting);
- five runs for each message size;
- the same OSU executables and the same host MPI library in both cases (checked with `ldd`);
- the same communication path, forced with `OMPI_MCA_pml=ob1` and `OMPI_MCA_btl=self,tcp`, that is ordinary TCP; this avoided the incompatible UCX library found in earlier attempts.

The last column of the table is 100 x (container / native - 1).

| Benchmark | Message size (bytes) | Native median ± s | Container median ± s | Container difference |
|---|---:|---:|---:|---:|
| latency | 1 | 15.91 ± 0.556 us | 16.47 ± 0.493 us | +3.5% |
| latency | 1024 | 18.32 ± 0.591 us | 19.05 ± 0.500 us | +4.0% |
| latency | 1048576 | 356.32 ± 20.817 us | 365.52 ± 30.700 us | +2.6% |
| latency | 4194304 | 1109.62 ± 73.417 us | 1182.27 ± 44.455 us | +6.5% |
| bandwidth | 1 | 0.46 ± 0.011 MB/s | 0.46 ± 0.005 MB/s | +0.0% |
| bandwidth | 1024 | 369.16 ± 6.048 MB/s | 383.59 ± 1.683 MB/s | +3.9% |
| bandwidth | 1048576 | 3739.03 ± 82.883 MB/s | 3550.77 ± 183.128 MB/s | -5.0% |
| bandwidth | 4194304 | 3809.27 ± 213.087 MB/s | 3652.44 ± 162.964 MB/s | -4.1% |

For latency a positive difference means slower communication in the container; for bandwidth a negative difference means lower throughput.

<p align="center"><img src="results_final/osu_microbench_latency.svg" alt="OSU native-vs-container latency" width="49%"> <img src="results_final/osu_microbench_bandwidth.svg" alt="OSU native-vs-container bandwidth" width="49%"></p>

_Left: latency inside the container is a few percent higher at all selected sizes. Right: for large messages the container reaches about 4-5% less bandwidth._

Communication through the container is a few percent slower: latency is 2.6-6.5% higher and the bandwidth for large messages 4-5% lower, differences similar to or a little larger than the run-to-run spread. The benchmark does not tell which layer is responsible. Because the N-body solver spends only a small fraction of its time communicating, a few percent slower communication gives a much smaller difference in total run time, consistent with the full-solver comparison.

---

## Conclusions and limitations

The energy drift stays below the chosen tolerance in every run, also over a long run, and the drift does not depend on how the work is divided: it is the same for every number of ranks, for both versions of the ring and inside the container. Halving the time step reduces the energy error as expected for a second-order method, and the cost grows with the square of the number of bodies, as expected for the direct method.

The program is limited by arithmetic rather than by memory or communication. The force computation takes almost all of the run time, each core processes the same number of pairs per second whatever the number of cores, and the time spent waiting for messages stays small. As a consequence, the parallel efficiency on one node stays above 96% in both strong and weak scaling. The small losses come mainly from the energy check, whose work is divided unevenly among the ranks, and at the largest rank count from slower cores on one node. The way the cores of a node are split between ranks and threads changes the run time only slightly.

The optimisations behave differently: Newton's third law saves work on a single core, but with several threads the uneven distribution of the pairs makes it slower than the direct kernel. The approximate inverse square root, written by hand with AVX-512 instructions, is the largest improvement, without a visible effect on energy conservation; its benefit, however, depends on that instruction set, and in a portable build it is lost. Independent partial sums do not help, because the loop is limited by the square root and the division rather than by the chain of additions, while more threads help because each one brings its own arithmetic units. Letting the compiler fuse multiplications and additions into FMAs (`-ffp-contract=fast`, not used in the other runs) shortens the exact force computation from 2.26 to 2.08 s. The structure-of-arrays layout gives no gain as long as the compiler cannot vectorise the loop, and overlapping communication with computation cannot save much when communication is already a small part of the run.

Inside Singularity the program runs almost as fast as natively, and the small difference is consistent with the different compiler and compilation target rather than with a cost of the container itself, once the container uses the cluster's own MPI library. The choice of a portable target costs little for code that the compiler generates by itself, and a lot for code written for one specific instruction set.

So we can say that these results have some limits. All solver runs used a single node, so the behaviour over a real network between nodes was not measured; only the OSU test crosses two nodes. No hardware counters were available, so statements about vector instructions and caches rely on timings, phase timers and compiler reports. The energy drift is a single global number and cannot prove that trajectories are identical. The load imbalance of the energy check and of the Newton kernel was explained but not fixed. Some runs of the same experiment ran on different nodes, one of which was slower. Finally, the compilation-target experiment measures the loss of the portable build for our code, not the value of AVX-512 over AVX2, and the container was built only for the portable target, so the cost of Singularity alone cannot be fully separated from the effect of the compiler and the target.
