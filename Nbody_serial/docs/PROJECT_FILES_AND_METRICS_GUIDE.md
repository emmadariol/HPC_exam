# Guida ai file del progetto e alle metriche

Questa guida spiega a cosa serve ogni file importante del progetto N-body, come
sono prodotti i benchmark e come leggere le metriche nei CSV finali. È pensata
come supporto per relazione, discussione orale e debug dei run su Orfeo.

## 1. Idea generale del progetto

Il progetto implementa una simulazione N-body diretta. Dato un insieme di
particelle, il codice calcola l'accelerazione gravitazionale su ogni particella
dovuta a tutte le altre.

Il costo dominante è il calcolo delle coppie:

```text
for i in particles:
  for j in particles:
    compute force(i, j)
```

Quindi il costo algoritmico è circa:

```text
O(N^2)
```

dove `N` è il numero totale di particelle.

Il progetto valuta:

- scaling MPI;
- configurazioni ibride MPI + OpenMP;
- layout dati AoS vs SoA;
- correttezza numerica tramite drift energetico;
- costo della diagnostica energetica;
- bandwidth memoria;
- effetto delle ottimizzazioni del kernel;
- overhead Singularity/container;
- microbenchmark MPI OSU nativo vs container;
- evidenza di vettorizzazione.

## 2. File sorgente C

### `nbody_common.h`

Header condiviso tra i programmi C.

Contiene:

- definizione di `dtype`;
- scelta `float`/`double`;
- magic number del formato binario;
- costanti di formato;
- wrapper matematici per `sqrt`, `pow`, `sin`, `cos`, `log`;
- macro di allineamento.

Nel progetto attuale la compilazione usa:

```text
-DNBODY_USE_DOUBLE
```

quindi:

```c
typedef double dtype;
```

Questo è importante perché influenza:

- precisione numerica;
- drift energetico;
- byte comunicati;
- byte caricati in memoria;
- possibilità di vettorizzazione.

### `generate_ic.c`

Genera le condizioni iniziali.

Produce un file binario nel formato:

```text
nbody-f32-v1
```

Il file contiene, per ogni particella:

```text
x, y, z, vx, vy, vz
```

Serve a rendere i benchmark riproducibili. Ogni run genera input deterministici
usando un seed controllato.

Modelli principali:

- `PLUMMER_SPHERE`;
- `MAXWELL_BALL`.

Nei benchmark viene usato normalmente il modello `0`, cioè Plummer.

Perché è importante:

- evita input casuali non riproducibili;
- permette di confrontare nativo/container sullo stesso stato iniziale;
- consente di misurare il drift energetico su condizioni fisiche coerenti.

### `nbody_direct_serial.c`

Versione seriale del solver.

Serve soprattutto come:

- baseline concettuale;
- smoke test;
- riferimento per correttezza;
- implementazione semplice del calcolo diretto.

Implementa:

- lettura del file binario;
- integrazione leapfrog;
- calcolo accelerazioni diretto;
- drift e kick;
- energia cinetica;
- energia potenziale;
- drift energetico.

La versione seriale non è il target principale dei benchmark di scaling, ma è
utile per dimostrare che il modello numerico è comprensibile e verificabile.

### `nbody_direct_hybrid.c`

È il cuore del progetto.

Implementa il solver N-body parallelo MPI + OpenMP.

Responsabilità principali:

- decomposizione delle particelle tra rank MPI;
- lettura parallela con MPI-IO;
- calcolo delle forze;
- scambio ad anello dei blocchi di particelle;
- kernel diretto;
- kernel Newton single-rank;
- modalità comunicazione `sendrecv` e `overlap`;
- uso di OpenMP nel loop sulle particelle locali;
- accumulatori multipli;
- opzione `rsqrt exact` o `rsqrt approx`;
- misure temporali dettagliate;
- riduzione MPI dei tempi con `MPI_MAX`;
- stampa delle righe parsate dagli script.

Le righe più importanti prodotte dal programma sono:

```text
# final: ...
# timing_max_seconds ...
# kernel_rate ...
```

Esempio:

```text
# final: N=10000 steps=5 ranks=8 threads_per_rank=1 ...
# timing_max_seconds total=... io=... drift=... force=... comm_wait=... kick=... energy=...
# kernel_rate pair_interactions_per_second=... gpair_interactions_per_second=...
```

I tempi paralleli sono ridotti con:

```text
MPI_Reduce(..., MPI_MAX, ...)
```

Quindi il tempo riportato è il massimo tra i rank, cioè il vero wall-clock
della fase parallela. Questo è corretto perché un'applicazione MPI finisce una
fase solo quando il rank più lento ha finito.

#### Fasi temporali misurate

| Campo | Cosa misura |
|---|---|
| `total` | tempo wall-clock totale del solver |
| `io` | lettura/scrittura file |
| `drift` | aggiornamento posizioni |
| `force` | calcolo accelerazioni |
| `comm_wait` | attesa/scambio MPI nel ring |
| `kick` | aggiornamento velocità |
| `energy` | calcolo diagnostica energetica |

#### Kernel diretto

Il kernel diretto calcola tutte le interazioni:

```text
F_i = sum_j F_ij
```

La stima delle interazioni stampata dal codice è:

```text
interactions = (nsteps + 1) * N * (N - 1)
```

Il `+1` nasce dallo schema KDK: serve una valutazione della forza iniziale e poi
una valutazione per ogni step.

La metrica:

```text
gpair_interactions_per_second
```

è:

```text
interactions / force_time / 1e9
```

Misura quante miliardi di interazioni particella-particella al secondo vengono
calcolate.

#### `--comm sendrecv` e `--comm overlap`

`sendrecv` usa uno scambio MPI più diretto: comunica un blocco e poi calcola.

`overlap` prova a sovrapporre comunicazione e calcolo:

```text
posta comunicazione non bloccante
calcola mentre la comunicazione procede
attende completamento
```

L'ablation confronta queste due modalità.

#### `--kernel direct` e `--kernel newton`

`direct` calcola ogni coppia nel modo standard.

`newton` sfrutta la terza legge di Newton:

```text
F_ij = -F_ji
```

Nel progetto attuale `newton` è usato come ottimizzazione single-rank. In
multi-rank rimane complesso perché richiede reinstradare anche le forze
accumulate.

#### `--rsqrt exact` e `--rsqrt approx`

`exact` usa il calcolo matematico standard:

```text
1 / sqrt(r2)
```

`approx` usa una strada più veloce quando disponibile, con approssimazione e
raffinamento Newton-Raphson.

L'ablation misura quanto questa scelta impatta il tempo.

#### Accumulatori

L'opzione:

```text
--accumulators 1|2|4|8
```

serve a testare catene di accumulo indipendenti nel kernel delle forze.

Motivazione:

```text
ax += dx * s
```

crea una dipendenza sullo stesso registro. Più accumulatori possono aumentare
l'ILP, cioè instruction-level parallelism.

Nel benchmark finale questa ottimizzazione non dà un guadagno netto evidente:
questo è comunque un risultato utile, perché indica che il compilatore o altri
colli di bottiglia dominano.

### `nbody_layout_benchmark.c`

Benchmark dedicato al layout dati.

Confronta:

- AoS, Array of Structures;
- SoA, Structure of Arrays.

AoS:

```c
struct particle {
  x, y, z;
  vx, vy, vz;
  ax, ay, az;
};
```

SoA:

```c
x[], y[], z[]
vx[], vy[], vz[]
ax[], ay[], az[]
```

Misura:

- tempo del kernel forza;
- Gpairs/s;
- checksum.

Il checksum serve a verificare che AoS e SoA producano lo stesso risultato
numerico. Nel risultato finale il checksum coincide, quindi il confronto di
performance è corretto.

### `memory_bandwidth.c`

Benchmark tipo STREAM.

Misura la bandwidth memoria con kernel semplici:

- `copy`;
- `scale`;
- `add`;
- `triad`.

Esempi concettuali:

```c
copy:  a[i] = b[i]
scale: a[i] = scalar * b[i]
add:   a[i] = b[i] + c[i]
triad: a[i] = b[i] + scalar * c[i]
```

Serve come riferimento hardware. Se il solver fosse memory-bound, la sua
prestazione sarebbe limitata dai GB/s disponibili. Nel caso N-body diretto, il
kernel è invece molto più compute-bound perché ogni coppia richiede molti FLOP.

## 3. File di orchestrazione

### `Makefile`

Compila i programmi e definisce target comodi.

Target principali:

| Target | Funzione |
|---|---|
| `make` | compila tutti i programmi principali |
| `make clean` | rimuove binari e file temporanei |
| `make run-smoke` | test rapido locale |
| `make vec-report` | produce report di vettorizzazione |
| `make bench-scaling` | lancia benchmark scaling |
| `make bench-hybrid` | lancia benchmark ibrido |
| `make bench-ablation` | lancia ablation |
| `make bench-layout` | lancia layout |
| `make bench-energy` | lancia energy |
| `make bench-memory` | lancia memory bandwidth |
| `make bench-container` | lancia container overhead |
| `make bench-osu` | lancia OSU |
| `make bench-arch` | confronta target architetturali |
| `make bench-perf` | snapshot opzionale con `perf` |

Flag importanti:

```text
-std=c11
-DNBODY_USE_DOUBLE
-O3
-march=native
-Wall -Wextra -Wpedantic
-fopenmp
```

`-march=native` abilita ottimizzazioni specifiche della CPU del nodo. Per
questo esiste anche il benchmark `arch`, che confronta con un target più
portabile.

### `benchmark_common.sh`

Contiene funzioni Bash condivise.

Gestisce:

- rilevamento runtime container;
- invocazione nativa;
- invocazione Singularity/Docker;
- costruzione dei comandi MPI;
- distribuzione `srun`;
- binding e variabili OpenMP.

È il livello che evita di duplicare codice tra benchmark diversi.

### `run_benchmarks.sh`

È il runner principale.

Comandi supportati:

```text
scaling
hybrid
ablation
layout
energy
memory
container
osu
arch
perf
```

Ogni comando produce un CSV raw. Poi `analyze.py` produce summary e grafici.

#### `scaling`

Produce:

```text
kind,N,nsteps,ranks,threads,repeat,...,total,force,comm_wait,gpairs,status,max_rel_drift
```

`kind` è:

- `strong`;
- `weak`.

Strong scaling:

```text
N fisso
P variabile
```

Weak scaling:

```text
N/P fisso
P variabile
```

#### `hybrid`

Esegue più chiamate a `scaling` variando coppie:

```text
P x T
```

Esempio:

```text
64x1
32x2
16x4
8x8
4x16
2x32
1x64
```

Qui il totale risorse resta circa costante, ma cambia il rapporto MPI/OpenMP.

#### `ablation`

Misura singole scelte progettuali:

- kernel `direct` vs `newton`;
- `rsqrt exact` vs `approx`;
- comunicazione `sendrecv` vs `overlap`;
- accumulatori `1,2,4,8`.

Output:

```text
Test_Type,Config,repeat,total,status
```

Il summary calcola tempi mediani e speedup rispetto a una baseline.

#### `layout`

Misura AoS vs SoA:

```text
layout,N,threads,repeat,warmups,inner_repeats,rsqrt,force,gpairs,checksum
```

Il summary calcola:

- mediana tempo forza;
- mediana Gpairs/s;
- speedup SoA vs AoS;
- differenza checksum.

#### `energy`

Misura il costo della diagnostica energetica.

Varia:

```text
energy_every
```

Se `energy_every=1`, l'energia è calcolata a ogni step.

Se `energy_every=nsteps`, l'energia è calcolata più raramente.

Output principale:

```text
total, force, energy, max_rel_drift
```

#### `memory`

Esegue `memory_bandwidth`.

Output:

```text
kernel,N,threads,repeat,inner_repeats,best_seconds,GBps,checksum,status
```

#### `container`

Confronta nativo e container sul solver N-body.

Output:

```text
kind,mode,N,ranks,threads,repeat,total,status,max_rel_drift
```

dove `mode` è:

- `native`;
- `container`.

Produce anche:

```text
container_overhead_launch.csv
```

con il tempo di lancio di:

```text
singularity exec nbody.sif true
```

#### `osu`

Esegue OSU Micro-Benchmarks.

Misura:

- `osu_latency`;
- `osu_bw`.

Output:

```text
mode,benchmark,metric,bytes,value
```

dove:

- `mode` è `native` o `container`;
- `benchmark` è `latency` o `bandwidth`;
- `bytes` è la dimensione del messaggio;
- `value` è microsecondi o MB/s.

#### `arch`

Confronta compilazioni diverse:

- `-march=native`;
- `-march=x86-64-v3`.

Serve a discutere portabilità vs ottimizzazione specifica del nodo.

#### `perf`

Benchmark opzionale con contatori hardware, se disponibili.

Eventi tipici:

```text
cycles
instructions
cache-references
cache-misses
branches
branch-misses
```

Non è sempre utilizzabile sui cluster, perché dipende dai permessi del sistema.

### `jobs/submit.sh`

Wrapper Slurm unificato.

Serve a non avere uno script diverso per ogni cluster o benchmark.

Parametri principali:

```text
--cluster orfeo|leonardo
--bench scaling|hybrid|...
--partition GENOA|EPYC|...
--cpus N
--nodes N
--time HH:MM:SS
--afterok JOBID
--result-dir DIR
```

Il wrapper:

- imposta account/partition/qos;
- carica moduli;
- crea directory risultato;
- genera uno script Slurm;
- sottomette con `sbatch`;
- passa variabili a `run_benchmarks.sh`.

È il punto principale per riprodurre i run.

### `collect_system_info.sh`

Raccoglie informazioni di sistema per la riproducibilità:

- hostname;
- CPU;
- NUMA;
- memoria;
- moduli caricati;
- compilatore;
- MPI;
- Slurm;
- ambiente OpenMP.

Serve per soddisfare la richiesta di identificazione hardware/software.

## 4. Analisi e grafici

### `analyze.py`

È lo strumento unico di analisi.

Comandi principali:

```bash
python3 analyze.py summarize KIND input.csv output_summary.csv
python3 analyze.py plot KIND summary.csv output_prefix
```

`KIND` può essere:

```text
scaling
layout
energy
memory
container
ablation
arch
osu
```

### Come vengono calcolate le statistiche

Per ogni gruppo sperimentale, `analyze.py`:

1. legge il CSV raw;
2. converte campi numerici;
3. raggruppa per configurazione;
4. conta run falliti;
5. tiene solo run con valori finiti;
6. rimuove outlier tramite MAD;
7. calcola mediana;
8. calcola deviazione standard;
9. calcola metriche derivate;
10. scrive un summary CSV.

La mediana è preferita alla media perché è più robusta a outlier e jitter del
cluster.

### Outlier

L'outlier filter usa la Median Absolute Deviation:

```text
MAD = median(|x_i - median(x)|)
```

I punti troppo lontani dalla mediana possono essere esclusi dal calcolo finale.
Il numero escluso finisce nella colonna:

```text
outliers
```

### Strong scaling

Per lo strong scaling:

```text
N fisso
risorse variabili
```

Speedup:

```text
speedup(P) = T(P_base) / T(P)
```

Efficiency:

```text
efficiency(P) = speedup(P) / (resources(P) / resources(P_base))
```

Con baseline tipicamente `P=1`.

Interpretazione:

- `speedup` ideale cresce linearmente;
- `efficiency` ideale è circa 1;
- se efficiency crolla, comunicazione/overhead dominano.

### Weak scaling

Nel tuo caso:

```text
N/P = costante
```

Per un algoritmo N-body diretto, il lavoro totale cresce come `N^2`, ma il
lavoro per rank cresce comunque con `P`, perché ogni rank deve interagire con
il sistema globale.

Per questo `analyze.py` normalizza il tempo weak come:

```text
weak_normalized_time(P) = T(P) / (resource_ratio * T(base))
```

Valori vicini a 1 indicano che il tempo cresce come previsto dal modello
algoritmico. Valori sopra 1 indicano overhead parallelo aggiuntivo.

### Communication bandwidth stimata

Nel summary scaling:

```text
comm_bandwidth_GBps
```

è una stima:

```text
bytes_per_rank / comm_wait_median / 1e9
```

dove `bytes_per_rank` considera:

- numero di force evaluations;
- numero di scambi nel ring;
- particelle locali;
- coordinate `x,y,z`;
- dimensione di `dtype`.

Non è una misura hardware pura come OSU: è una stima applicativa basata sul
tempo passato in comunicazione.

### Gpairs/s

Nel solver:

```text
interactions = (nsteps + 1) * N * (N - 1)
```

Poi:

```text
Gpairs/s = interactions / force_time / 1e9
```

Misura throughput del kernel di forza.

### Drift energetico

Il codice calcola:

```text
max_relative_energy_drift = max |E(t) - E(0)| / |E(0)|
```

Se:

```text
max_relative_energy_drift <= energy_tol
```

lo status è:

```text
OK
```

altrimenti:

```text
WARNING
```

`WARNING` non significa necessariamente crash o dato inutilizzabile. Significa
che la conservazione energetica supera la tolleranza impostata. Per confronti
nativo/container, se il drift è identico nei due modi, il confronto di tempo
resta significativo.

### Container overhead

Nel summary container:

```text
overhead_percent = 100 * (container_median - native_median) / native_median
```

Interpretazione:

- positivo: container più lento;
- negativo: container leggermente più veloce, di solito rumore sperimentale;
- vicino a 0: overhead trascurabile.

### Launch overhead

Il file:

```text
container_overhead_launch.csv
```

misura solo il costo di avviare il container:

```bash
time singularity exec nbody.sif true
```

Questo non misura il solver. Misura solo il costo fisso di lancio.

### OSU latency

OSU latency misura il tempo medio per mandare e ricevere un messaggio piccolo
tra due processi MPI.

Unità:

```text
microsecondi
```

Più basso è meglio.

### OSU bandwidth

OSU bandwidth misura il throughput MPI per messaggi più grandi.

Unità:

```text
MB/s
```

Più alto è meglio.

OSU è indipendente dal solver N-body. Serve a isolare il costo della
comunicazione MPI.

## 5. Container e immagini

### `Dockerfile`

Definisce l'immagine Docker usata per costruire il container.

Contiene:

- sistema base Ubuntu;
- compilatori;
- OpenMPI;
- OSU Micro-Benchmarks;
- sorgenti del progetto;
- build del codice.

Il workflow usato è:

```text
Docker build locale
Docker push su Docker Hub
Singularity pull su Orfeo
```

### `Singularity.def`

Definizione Singularity/Apptainer.

È utile per documentare l'ambiente container, ma su Orfeo la build diretta può
non essere permessa senza `fakeroot`/`proot`. Per questo è stato usato Docker
Hub come ponte per ottenere `nbody.sif`.

### `nbody.sif`

Immagine Singularity finale.

Non deve essere committata su GitHub.

Serve per:

- benchmark container del solver;
- OSU container;
- verifica riproducibilità ambiente.

## 6. Directory risultati

### `runs/`

Contiene run grezzi.

Non va committata integralmente.

Può contenere:

- tentativi falliti;
- log Slurm;
- output intermedi;
- CSV parziali;
- directory con timestamp.

Serve per debug, non come risultato finale.

### `results_final/`

Contiene i risultati selezionati per il report.

Dovrebbe contenere solo:

- CSV finali;
- summary CSV;
- SVG;
- system info;
- vectorization report;
- job status finale;
- eventuale README.

Non dovrebbe contenere:

- `runs/`;
- `.sif`;
- `.bin`;
- cache Singularity;
- output temporanei;
- tentativi vecchi non usati.

### `docs/`

Contiene documentazione:

- guida dettagliata esercizio;
- template relazione;
- report vettorizzazione;
- questa guida.

## 7. File risultati principali

### Scaling

| File | Contenuto |
|---|---|
| `scaling.csv` | misure raw, una riga per run |
| `scaling_summary.csv` | mediane, speedup, efficiency |
| `scaling_strong_speedup.svg` | grafico speedup strong |
| `scaling_strong_efficiency.svg` | grafico efficiency strong |
| `scaling_weak_time.svg` | tempo weak |
| `scaling_weak_normalized_time.svg` | tempo weak normalizzato |
| `scaling_strong_comm_bandwidth.svg` | stima bandwidth comunicazione |

### Hybrid

| File | Contenuto |
|---|---|
| `hybrid_P*_T*.csv` | raw per coppia MPI/OpenMP |
| `hybrid_P*_T*_summary.csv` | summary per coppia |
| `hybrid_summary.csv` | summary aggregato |
| `hybrid_time_by_config.svg` | tempi per configurazione |
| `hybrid_gpairs_by_config.svg` | throughput per configurazione |

### Ablation

| File | Contenuto |
|---|---|
| `ablation.csv` | raw ablation |
| `ablation_summary.csv` | confronto contro baseline |
| `ablation.svg` | grafico comparativo |

### Layout

| File | Contenuto |
|---|---|
| `layout.csv` | raw AoS/SoA |
| `layout_summary.csv` | mediana, Gpairs/s, checksum |
| `layout_force_time.svg` | grafico tempi layout |

### Energy

| File | Contenuto |
|---|---|
| `energy.csv` | raw diagnostica energetica |
| `energy_summary.csv` | overhead energy |
| `energy_overhead.svg` | grafico overhead |

### Memory

| File | Contenuto |
|---|---|
| `memory_bandwidth.csv` | raw STREAM-like |
| `memory_bandwidth_summary.csv` | GB/s mediani |
| `memory_bandwidth.svg` | grafico bandwidth |

### Container

| File | Contenuto |
|---|---|
| `container_overhead.csv` | raw native/container |
| `container_overhead_summary.csv` | overhead percentuale |
| `container_overhead_strong.svg` | overhead strong |
| `container_overhead_weak.svg` | overhead weak |
| `container_overhead_launch.csv` | costo di lancio Singularity |

### OSU

| File | Contenuto |
|---|---|
| `osu_microbench_native_vs_container.csv` | raw OSU |
| `osu_microbench_summary.csv` | mediane/stdev OSU |
| `osu_microbench_latency.svg` | latenza |
| `osu_microbench_bandwidth.svg` | bandwidth |

### Arch

| File | Contenuto |
|---|---|
| `arch_target_comparison.csv` | raw confronto target |
| `arch_target_comparison_summary.csv` | overhead vs native |
| `arch_target_comparison.svg` | grafico |

## 8. Campagna “required table”

La tabella richiesta dall'esercizio container richiede:

```text
Strong native:    N=100000, 100 steps, P=1,2,4,8,16,32
Strong container: N=100000, 100 steps, P=1,2,4,8,16,32
Weak native:      N/P=10000, 100 steps, P=1,2,4,8,16
Weak container:   N/P=10000, 100 steps, P=1,2,4,8,16
Launch overhead:  10 lanci
OSU:              native vs container, 2 processi
```

Per rispettare “5 run per punto” con il limite Slurm `01:59:00`, i punti più
lenti devono essere lanciati come job array serializzati:

```bash
--array=1-5%1
```

Il `%1` è fondamentale: assicura che le cinque ripetizioni non partano in
parallelo sullo stesso nodo. Senza `%1`, i run si disturbano tra loro e la
statistica diventa meno pulita.

La directory della campagna richiesta è indicata da:

```text
LAST_REQUIRED_TABLE_RUN.txt
```

Il file:

```text
jobs_required.tsv
```

contiene la lista dei job sottomessi.

## 9. Come spiegare i benchmark all'orale

Una risposta sintetica ma completa può essere:

```text
Lo scaling misura come cambia il tempo aumentando i rank MPI. Il caso strong
tiene N fisso e guarda speedup/efficienza; il caso weak tiene N/P fisso e guarda
quanto il tempo cresce rispetto al modello atteso. Hybrid tiene fisse le risorse
totali e cambia il rapporto MPI/OpenMP. Ablation isola singole scelte del
kernel, come rsqrt, overlap e accumulatori. Layout confronta AoS e SoA e usa un
checksum per la correttezza. Energy misura il costo della diagnostica fisica.
Memory dà la bandwidth massima della macchina. Container confronta il solver
nativo e Singularity. OSU misura invece la comunicazione MPI pura, indipendente
dal solver.
```

## 10. Caveat importanti da dichiarare

Se alcuni risultati hanno `WARNING`, significa che il drift supera la tolleranza
impostata, non che il programma sia crashato.

Se native e container hanno lo stesso drift, il confronto prestazionale resta
valido, perché stanno eseguendo la stessa dinamica numerica.

Se OSU mostra overhead container molto più alto del solver N-body, non è una
contraddizione: OSU misura comunicazione pura, mentre il solver N-body diretto è
dominato dal calcolo.

Se un punto richiede molto tempo, è corretto spezzare le ripetizioni in job
separati purché:

- i parametri siano identici;
- il seed sia controllato;
- il numero di ripetizioni sia dichiarato;
- il merge finale mantenga `repeat` e configurazione.

## 11. Checklist finale prima del push

Controlli consigliati:

```bash
grep -RniE "RUN_FAILED|PARSE_FAILED|,nan,|,nan$|Traceback|missing|FATAL" results_final
```

Controllo Git:

```bash
git status --short
git diff --cached --stat
```

Non committare:

```text
runs/
*.sif
*.bin
singularity_cache/
__pycache__/
```

Committare invece:

```text
*.c
*.h
Makefile
run_benchmarks.sh
benchmark_common.sh
analyze.py
jobs/submit.sh
docs/
results_final/
```

