# Guida completa al progetto - Esercizio 1: simulazione N-body diretta MPI + OpenMP

Questa guida serve per capire davvero il progetto, non solo per sapere quali comandi lanciare. È scritta pensando a tre domande:

1. Che cosa chiede l'esercizio?
2. Come è stato implementato nel codice?
3. Come vanno letti e difesi i risultati?

Il progetto implementa un solutore gravitazionale N-body diretto: ogni particella interagisce con tutte le altre. Il codice finale usa MPI per distribuire il lavoro tra processi e OpenMP per parallelizzare il lavoro dentro ogni processo. In più contiene benchmark di scalabilità, test di correttezza, analisi delle ottimizzazioni e confronto native vs container.

---

## 1. Idea generale del progetto

Il problema N-body consiste nel simulare l'evoluzione di `N` particelle soggette alla gravità. Ogni particella sente la forza generata da tutte le altre.

In forma molto semplice:

```text
per ogni step temporale:
    per ogni particella i:
        calcola la forza totale su i dovuta a tutte le particelle j
    aggiorna velocità e posizione
    opzionalmente calcola l'energia totale per controllare la correttezza
```

Il cuore del progetto è quindi il calcolo delle forze.

```text
particella i
    |
    | interagisce con
    v
particelle j = 0, 1, 2, ..., N-1
```

Poiché questo succede per ogni particella `i`, il costo è:

```text
N particelle target * N particelle sorgente = O(N^2)
```

Questo è importantissimo per interpretare i risultati: il codice è computazionalmente pesante, ma regolare. È una buona situazione per studiare il parallelismo, perché il lavoro è abbondante e simile per tutti.

---

## 2. Che cosa richiede l'esercizio

L'esercizio richiede una versione HPC del codice N-body diretto, con analisi delle prestazioni e del container.

I requisiti principali sono:

| Richiesta | Dove viene soddisfatta |
|---|---|
| Solutore N-body diretto | `nbody_direct_serial.c`, `nbody_direct_hybrid.c` |
| Parallelizzazione MPI + OpenMP | `nbody_direct_hybrid.c` |
| Strong scaling | `results_final/scaling_64_summary.csv` |
| Weak scaling | `results_final/scaling_64_summary.csv` |
| Studio MPI/OpenMP ibrido | `results_final/hybrid_64_summary.csv` |
| Almeno 5 ripetizioni/statistiche | CSV finali e script `analyze.py` |
| Correttezza numerica | deriva energetica e checksum AoS/SoA |
| Discussione delle ottimizzazioni | ablation, layout, rsqrt, comunicazione |
| Container overhead | `results_final/container_overhead_summary.csv + results_final/osu_microbench_summary.csv` |
| OSU Micro-Benchmarks native vs container | `run_benchmarks.sh osu`, CSV OSU finali |
| Verifica host MPI nel container | `mpi_linkage_check.txt` |

Il risultato finale non è solo "il codice gira", ma:

```text
codice corretto
    +
parallelizzazione sensata
    +
misure ripetute
    +
grafici interpretabili
    +
spiegazione dei bottleneck
    +
container confrontato con native
```

---

## 3. Mappa dei file importanti

### 3.1 Codice numerico

| File | Ruolo |
|---|---|
| `generate_ic.c` | genera condizioni iniziali riproducibili |
| `nbody_common.h` | definizioni comuni: tipi, formato binario, costanti |
| `nbody_direct_serial.c` | versione seriale di riferimento |
| `nbody_direct_hybrid.c` | versione parallela MPI + OpenMP |
| `nbody_layout_benchmark.c` | benchmark AoS vs SoA |

### 3.2 Benchmark e analisi

| File | Ruolo |
|---|---|
| `run_benchmarks.sh scaling` | lancia strong e weak scaling |
| `analyze.py summarize scaling` | calcola mediane, speedup, efficiency |
| `analyze.py plot scaling` | produce grafici scaling |
| `analyze.py plot hybrid` | produce grafici P x T per MPI/OpenMP |
| `run_benchmarks.sh layout` | misura AoS vs SoA |
| `analyze.py summarize layout` | riassume layout e checksum |
| `run_benchmarks.sh energy` | misura costo del controllo energia |
| `analyze.py summarize energy` | riassume overhead energia |
| `run_benchmarks.sh container` | confronta native e Singularity |
| `run_benchmarks.sh osu` | esegue OSU latency/bandwidth |
| `analyze.py plot evidence` | produce grafici esplicativi per il report |

### 3.3 Container e job

| File/cartella | Ruolo |
|---|---|
| `Dockerfile` | immagine Docker usata per costruire il container |
| `Singularity.def` | recipe Singularity/Apptainer |
| `jobs/submit.sh` | wrapper Slurm unico per Orfeo e Leonardo |

### 3.4 Report e risultati

| File/cartella | Ruolo |
|---|---|
| `../FINAL_REPORT.md` | report finale completo |
| `results_final` | risultati principali a 64 core su GENOA |
| `results_final/layout_summary.csv + results_final/energy_overhead_summary.csv` | layout, energia, info sistema |
| `results_final/container_overhead_summary.csv + results_final/osu_microbench_summary.csv` | risultati container finali |

### 3.5 Dove sono implementati i concetti nel codice

Questa tabella è utile per l'orale: collega il concetto teorico alla parte concreta del codice.

| Concetto | Dove guardare |
|---|---|
| Parsing opzioni `--integrator`, `--comm`, `--kernel`, `--rsqrt` | `nbody_direct_hybrid.c`, funzioni `parse_*` |
| Radice inversa esatta/approssimata | `nbody_direct_hybrid.c`, funzione `invsqrt_force` |
| Lettura file binario iniziale | `nbody_direct_hybrid.c`, `read_local_particles`; `nbody_direct_serial.c` |
| Scrittura stato finale | `nbody_direct_hybrid.c`, `write_output_root` |
| Kernel diretto | `nbody_direct_hybrid.c`, accumulo accelerazioni nel ring |
| Kernel Newton single-rank | `nbody_direct_hybrid.c`, `compute_accelerations_newton_private` |
| Scambio MPI bloccante | `nbody_direct_hybrid.c`, `exchange_sources_sendrecv` |
| Ring MPI | `nbody_direct_hybrid.c`, `compute_accelerations_ring` |
| OpenMP sul loop delle particelle | `nbody_direct_hybrid.c`, loop paralleli nel force kernel |
| Energia e deriva energetica | `nbody_direct_serial.c`, `nbody_direct_hybrid.c` |
| Generazione condizioni iniziali | `generate_ic.c` |
| Layout AoS/SoA | `nbody_layout_benchmark.c` |
| Analisi speedup/efficiency | `analyze.py summarize scaling` |
| Plot con riferimento ideale | `analyze.py plot scaling` |
| Plot evidence/bottleneck | `analyze.py plot evidence` |

---

## 4. Concetti fisici e numerici

### 4.1 Che cos'è una simulazione N-body

Una simulazione N-body descrive l'evoluzione di molte particelle che interagiscono fra loro. In questo progetto l'interazione è gravitazionale.

Ogni particella ha:

```text
posizione:  x, y, z
velocità:   vx, vy, vz
massa:      m
```

Il codice calcola l'accelerazione di ogni particella e poi aggiorna posizione e velocità.

Schema:

```text
posizioni e velocità al tempo t
        |
        v
calcolo accelerazioni gravitazionali
        |
        v
integrazione temporale
        |
        v
posizioni e velocità al tempo t + dt
```

### 4.2 Forza gravitazionale diretta

Per ogni coppia di particelle `i` e `j`, la forza dipende dalla distanza tra le due.

Nel codice si usa una forma ammorbidita:

```text
a_i = somma_j G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
```

Dove:

- `a_i` è l'accelerazione della particella `i`;
- `G` è la costante gravitazionale;
- `m_j` è la massa della particella sorgente `j`;
- `r_i` e `r_j` sono le posizioni;
- `eps` è il softening.

### 4.3 Che cos'è il softening

Il softening evita che la forza diventi infinita quando due particelle sono estremamente vicine.

Senza softening:

```text
se distanza -> 0
    forza -> infinito
```

Con softening:

```text
distanza_effettiva^2 = distanza^2 + eps^2
```

Quindi la forza resta finita.

Perché serve:

- evita instabilità numeriche;
- rende più stabile la simulazione;
- permette di usare timestep ragionevoli;
- rende significativa la verifica tramite energia.

### 4.4 Integrazione temporale: leapfrog

Il codice usa uno schema leapfrog, cioè un integratore semplice, stabile e adatto a sistemi Hamiltoniani come la gravità.

Nel report finale si usa principalmente la variante `kdk`, cioè kick-drift-kick.

Schema KDK:

```text
KICK mezzo step:
    aggiorna velocità usando accelerazione a(t)

DRIFT step intero:
    aggiorna posizione usando velocità a metà step

ricalcola accelerazione a(t + dt)

KICK mezzo step:
    aggiorna velocità usando nuova accelerazione
```

In forma compatta:

```text
v(t + dt/2) = v(t) + a(t) * dt/2
x(t + dt)   = x(t) + v(t + dt/2) * dt
v(t + dt)   = v(t + dt/2) + a(t + dt) * dt/2
```

Perché si usa:

- è di ordine 2;
- conserva bene l'energia su tempi lunghi;
- è molto comune nelle simulazioni gravitazionali;
- ha una struttura semplice da parallelizzare.

---

## 5. Correttezza numerica

### 5.1 Perché non basta che il codice finisca

In HPC un codice veloce ma sbagliato non vale nulla. Per questo il progetto controlla la correttezza in due modi:

1. deriva dell'energia totale;
2. confronto dei checksum nei benchmark AoS/SoA.

### 5.2 Energia totale

L'energia totale è:

```text
energia totale = energia cinetica + energia potenziale
```

Energia cinetica:

```text
K = 1/2 * m * v^2
```

Energia potenziale gravitazionale:

```text
U = somma sulle coppie -G * m_i * m_j / distanza
```

In una simulazione ideale di un sistema isolato, l'energia totale dovrebbe restare circa costante.

### 5.3 Deriva energetica

Il codice misura:

```text
max_relative_energy_drift =
    max_t |E(t) - E(0)| / max(|E(0)|, tiny)
```

Tradotto:

```text
quanto cambia al massimo l'energia rispetto all'energia iniziale
```

Se la deriva è piccola, la simulazione è numericamente coerente.

Nel progetto la tolleranza usata è:

```text
1e-3
```

I run finali hanno stato `OK`.

### 5.4 Energia come costo extra

Calcolare l'energia non è gratis. Anche l'energia potenziale richiede confrontare molte coppie di particelle, quindi costa circa `O(N^2)`.

Per questo il progetto misura anche l'overhead del controllo energetico.

Schema:

```text
simulazione senza energia frequente
        |
        v
tempo minore

simulazione con energia a ogni step
        |
        v
tempo maggiore, ma controllo più dettagliato
```

Risultato finale:

- controllare l'energia a ogni step è utile per validare;
- ma nei benchmark prestazionali bisogna farlo più raramente;
- altrimenti si misura il costo della diagnostica, non solo quello del solver.

---

## 6. Complessità computazionale

### 6.1 Che cosa significa `O(N^2)`

Se raddoppio il numero di particelle, il numero di interazioni cresce circa di quattro volte.

```text
N       interazioni circa
1000    1,000,000
2000    4,000,000
4000    16,000,000
```

Questo succede perché:

```text
ogni particella interagisce con tutte le altre
```

Schema:

```text
             sorgenti j
          0  1  2  3  ... N-1
target 0  x  x  x  x      x
target 1  x  x  x  x      x
target 2  x  x  x  x      x
...
target N  x  x  x  x      x
```

Questa matrice ha dimensione `N x N`.

### 6.2 Perché questo è importante per la scalabilità

Il codice diretto scala bene in strong scaling perché c'è tanto lavoro aritmetico da dividere.

Però in weak scaling non bisogna aspettarsi tempo costante come in uno stencil locale.

Perché?

Se tengo fisse le particelle per rank:

```text
N_per_rank = costante
N_totale = P * N_per_rank
```

Ogni rank ha sempre `N_per_rank` particelle target, ma ogni target interagisce con `N_totale` particelle sorgente.

Quindi il lavoro per rank è:

```text
N_per_rank * N_totale
= N_per_rank * P * N_per_rank
= P * N_per_rank^2
```

Quindi cresce con `P`.

Questo è un punto chiave da spiegare nel report.

---

## 7. Parallelizzazione MPI

### 7.1 Che cos'è MPI

MPI significa Message Passing Interface. È il modello usato quando più processi separati collaborano e si scambiano messaggi.

Ogni processo MPI si chiama rank.

```text
rank 0
rank 1
rank 2
rank 3
```

Ogni rank ha la propria memoria. Se un rank vuole dati da un altro rank, deve riceverli tramite comunicazione MPI.

### 7.2 Decomposizione usata nel progetto

Il progetto divide le particelle in blocchi contigui. Ogni rank possiede un pezzo permanente del sistema, detto home chunk.

Esempio con 16 particelle e 4 rank:

```text
rank 0: particelle  0  1  2  3
rank 1: particelle  4  5  6  7
rank 2: particelle  8  9 10 11
rank 3: particelle 12 13 14 15
```

Ogni rank aggiorna solo le proprie particelle home.

### 7.3 Problema: ogni rank deve vedere tutte le particelle

Per calcolare la forza sulle proprie particelle, un rank deve conoscere anche le particelle possedute dagli altri rank.

Soluzione semplice ma pesante:

```text
allgather di tutte le particelle su tutti i rank
```

Questo però replica tutto ovunque.

Soluzione usata:

```text
ring-shift
```

### 7.4 Ring-shift communication

Nel ring-shift ogni rank manda il proprio blocco al vicino e riceve un blocco da un altro vicino. A ogni fase calcola le interazioni tra le sue particelle home e il blocco sorgente corrente.

Schema con 4 rank:

```text
fase 0:
rank 0 usa blocco 0
rank 1 usa blocco 1
rank 2 usa blocco 2
rank 3 usa blocco 3

fase 1:
rank 0 riceve blocco 3
rank 1 riceve blocco 0
rank 2 riceve blocco 1
rank 3 riceve blocco 2

fase 2:
rank 0 riceve blocco 2
rank 1 riceve blocco 3
rank 2 riceve blocco 0
rank 3 riceve blocco 1

fase 3:
rank 0 riceve blocco 1
rank 1 riceve blocco 2
rank 2 riceve blocco 3
rank 3 riceve blocco 0
```

Visualmente:

```text
rank 0 ----> rank 1
  ^            |
  |            v
rank 3 <---- rank 2
```

Alla fine ogni rank ha visto tutti i blocchi sorgente, ma senza tenere permanentemente tutto il sistema in memoria.

### 7.5 Perché ring-shift

Vantaggi:

- memoria più controllata;
- comunicazione regolare;
- buon bilanciamento del lavoro;
- naturale per un algoritmo all-pairs.

Svantaggi:

- servono `P - 1` scambi;
- a molti rank la latenza può diventare visibile;
- su più nodi la rete potrebbe diventare un collo di bottiglia.

Nel progetto finale single-node, la comunicazione non domina: il force kernel resta il costo principale.

---

## 8. Parallelizzazione OpenMP

### 8.1 Che cos'è OpenMP

OpenMP serve a usare più thread dentro lo stesso processo. A differenza di MPI, i thread condividono la memoria.

Schema:

```text
processo MPI
    |
    +-- thread 0
    +-- thread 1
    +-- thread 2
    +-- thread 3
```

### 8.2 Dove viene usato nel progetto

OpenMP viene applicato al loop sulle particelle home.

Forma concettuale:

```c
#pragma omp parallel for
for (i = 0; i < n_local; ++i) {
    calcola accelerazione della particella locale i
}
```

Ogni thread lavora su particelle target diverse.

### 8.3 Perché parallelizzare il loop esterno

Il loop interno accumula la forza sulla particella `i`. Se più thread aggiornassero la stessa particella, servirebbero atomiche o riduzioni complicate.

La scelta fatta è:

```text
thread diversi -> particelle target diverse
```

Così ogni thread aggiorna solo i propri accumuli locali.

Schema:

```text
thread 0: target 0, 1, 2
thread 1: target 3, 4, 5
thread 2: target 6, 7, 8
thread 3: target 9, 10, 11
```

Vantaggi:

- niente atomiche nel loop più costoso;
- parallelismo semplice;
- buona scalabilità;
- meno rischio di race condition.

---

## 9. Parallelismo ibrido MPI + OpenMP

### 9.1 Che cosa significa ibrido

Un codice ibrido usa sia MPI sia OpenMP.

```text
nodo di calcolo
    |
    +-- rank MPI 0
    |       +-- thread OpenMP
    |       +-- thread OpenMP
    |
    +-- rank MPI 1
    |       +-- thread OpenMP
    |       +-- thread OpenMP
    |
    +-- rank MPI 2
            +-- thread OpenMP
            +-- thread OpenMP
```

Nel progetto si studiano configurazioni `P x T`, dove:

- `P` = numero di processi MPI;
- `T` = thread OpenMP per processo.

Esempio:

```text
P x T = 32 x 2
```

significa:

```text
32 rank MPI, 2 thread per rank, totale 64 core
```

### 9.2 Perché studiare più configurazioni P x T

Con lo stesso numero totale di core posso scegliere combinazioni diverse:

```text
64 x 1  = tanti rank MPI, un thread ciascuno
32 x 2  = rank MPI medi, due thread ciascuno
16 x 4
8  x 8
4  x 16
2  x 32
1  x 64 = un solo rank MPI, tanti thread
```

Queste configurazioni hanno compromessi diversi:

| Configurazione | Pro | Contro |
|---|---|---|
| molti rank, pochi thread | più decomposizione MPI, chunk piccoli | più fasi/anelli/comunicazione |
| pochi rank, molti thread | meno comunicazione MPI | più pressione su memoria condivisa/threading |

Risultato finale del progetto:

- tutte le configurazioni a 64 core sono molto vicine;
- la migliore mediana è `32 x 2`;
- la differenza tra migliore e peggiore è circa 1.26%.

Interpretazione:

```text
il codice non dipende in modo fragile da una sola configurazione MPI/OpenMP
```

---

## 10. Comunicazione: `sendrecv` e `overlap`

### 10.1 Comunicazione bloccante: `sendrecv`

La modalità `sendrecv` fa una comunicazione esplicita e poi calcola.

Schema:

```text
comunica blocco sorgente
        |
        v
calcola forze con quel blocco
        |
        v
comunica blocco successivo
        |
        v
calcola ...
```

È semplice e robusta.

### 10.2 Comunicazione sovrapposta: `overlap`

La modalità `overlap` prova a iniziare la comunicazione del prossimo blocco mentre il rank calcola sul blocco corrente.

Schema ideale:

```text
tempo --->

calcolo:       [ compute block A ][ compute block B ][ compute block C ]
comunicazione:      [ recv B ]        [ recv C ]        [ recv D ]
```

L'idea è nascondere parte del tempo di comunicazione dietro il calcolo.

### 10.3 Perché nel progetto overlap aiuta poco

Nel run finale single-node:

- il calcolo delle forze domina;
- la comunicazione è piccola rispetto al totale;
- MPI deve comunque fare progresso nella comunicazione;
- la differenza misurata tra `sendrecv` e `overlap` è sotto il rumore sperimentale.

Quindi la conclusione corretta è:

```text
overlap non è sbagliato, ma in questo caso non è decisivo
```

Su run multi-nodo potrebbe diventare più importante, perché la rete avrebbe latenza maggiore.

---

## 11. Newton's third law

### 11.1 Idea teorica

La terza legge di Newton dice:

```text
F_ij = -F_ji
```

Quindi, se calcolo la forza tra `i` e `j`, potrei aggiornare sia `i` sia `j` con un solo calcolo.

Senza Newton:

```text
calcolo i <- j
calcolo j <- i
```

Con Newton:

```text
calcolo coppia (i, j) una volta sola
aggiorno sia i sia j
```

### 11.2 Perché non è la soluzione principale distribuita

In seriale o shared-memory può aiutare. In MPI, però, `i` e `j` possono stare su rank diversi.

Esempio:

```text
rank 0 possiede i
rank 1 possiede j
```

Se rank 0 calcola la coppia `(i, j)`, può aggiornare facilmente la forza su `i`, ma la forza opposta su `j` appartiene a rank 1.

Allora servirebbe:

```text
calcolo su rank 0
        |
        v
invio contributo opposto a rank 1
        |
        v
rank 1 accumula forza ricevuta
```

Questo aggiunge comunicazione, buffering e complessità.

Per questo:

- il kernel distribuito principale usa il metodo diretto senza Newton;
- Newton viene misurato come ablation single-rank.

Risultato:

- Newton è più veloce nel test single-rank;
- ma non è automaticamente migliore nella versione MPI distribuita.

---

## 12. Exact vs approximate inverse square root

### 12.1 Dove compare la radice inversa

Nel calcolo della forza serve:

```text
1 / (r^2 + eps^2)^(3/2)
```

Il codice può calcolarlo in modo:

- `exact`: usando funzioni matematiche standard;
- `approx`: usando una strada approssimata con raffinamento.

### 12.2 Perché provare una versione approssimata

In molti codici HPC, approssimare `1/sqrt(x)` può essere più veloce perché certe CPU hanno istruzioni specializzate.

Ma non è garantito.

Dipende da:

- compilatore;
- tipo `float` o `double`;
- vettorizzazione;
- conversioni;
- costo del raffinamento;
- layout dati.

### 12.3 Cosa è successo nel progetto

Nel progetto la versione `approx` è più lenta della `exact`.

Questa non è una sconfitta: è una dimostrazione importante.

Conclusione:

```text
un'ottimizzazione teorica va misurata; non va assunta
```

---

## 13. Layout dati: AoS vs SoA

### 13.1 Che cosa sono AoS e SoA

AoS significa Array of Structures.

```text
particle[0] = {x, y, z, vx, vy, vz}
particle[1] = {x, y, z, vx, vy, vz}
particle[2] = {x, y, z, vx, vy, vz}
```

SoA significa Structure of Arrays.

```text
x[0],  x[1],  x[2],  ...
y[0],  y[1],  y[2],  ...
z[0],  z[1],  z[2],  ...
vx[0], vx[1], vx[2], ...
```

Schema:

```text
AoS:
| x y z vx vy vz | x y z vx vy vz | x y z vx vy vz |

SoA:
| x x x x x ... |
| y y y y y ... |
| z z z z z ... |
```

### 13.2 Perché SoA spesso è utile

SoA può aiutare perché il loop sulle forze legge molte posizioni consecutive.

La CPU può caricare meglio:

```text
x[j], x[j+1], x[j+2], ...
```

Questo può favorire:

- cache;
- prefetching;
- SIMD/vectorization.

### 13.3 Cosa dice il risultato del progetto

Nel benchmark isolato del progetto, SoA non è più veloce di AoS.

Questo può succedere perché:

- il compilatore non vettorizza come sperato;
- il kernel specifico ha dipendenze o accessi non ideali;
- il costo complessivo non è dominato solo dal layout;
- la macchina e le flag di compilazione contano.

La cosa importante è che è stato aggiunto il checksum:

```text
AoS checksum ≈ SoA checksum
```

Quindi il confronto è corretto: non stiamo confrontando due codici che calcolano cose diverse.

---

## 14. Strong scaling

### 14.1 Definizione

Strong scaling significa:

```text
mantengo fisso il problema
aumento le risorse
misuro quanto diminuisce il tempo
```

Nel progetto:

```text
N = 20000 fisso
ranks = 1, 2, 4, 8, 16, 32, 64
threads/rank = 1
```

### 14.2 Speedup

Lo speedup misura quanto vado più veloce rispetto al caso base.

Formula:

```text
S(P) = T(1) / T(P)
```

Dove:

- `T(1)` è il tempo con una risorsa;
- `T(P)` è il tempo con `P` risorse.

Esempio:

```text
T(1)  = 31.186 s
T(64) = 0.538 s

S(64) = 31.186 / 0.538 ≈ 57.94
```

### 14.3 Efficiency

L'efficienza misura quanto bene sto usando le risorse.

Formula:

```text
E(P) = S(P) / P
```

Se `E(P) = 1`, lo scaling è ideale.

Nel progetto:

```text
E(64) = 57.94 / 64 ≈ 0.905 = 90.5%
```

### 14.4 Perché non basta plottare i tempi

Il tempo assoluto scende in modo non lineare. L'occhio umano non giudica bene curve non lineari.

Per questo nel report si usano:

- speedup;
- efficiency;
- linea ideale `S(P)=P`;
- linea ideale `E(P)=1`.

Schema:

```text
tempo assoluto:
    utile, ma difficile da giudicare

speedup:
    ideale = linea dritta
    facile vedere il gap

efficiency:
    ideale = linea piatta a 1
    facile vedere la perdita
```

### 14.5 Che cosa significa il riferimento a 16x

Nelle note di scalabilità viene detto che, per un buon codice, una perdita forte di efficienza dovrebbe comparire solo oltre un fattore di scaling abbastanza grande, indicativamente oltre `16x`.

Non è una legge matematica. È una regola pratica.

Nel nostro caso:

```text
16x = passare da 1 rank a 16 rank
```

Il risultato del progetto è:

```text
E(16) = 93.3%
E(64) = 90.5%
```

Quindi il codice si comporta bene: l'efficienza resta alta anche oltre 16x.

### 14.6 Overhead nello strong scaling

Overhead significa costo extra dovuto alla parallelizzazione.

Esempi:

- comunicazione MPI;
- sincronizzazione;
- gestione dei thread;
- attesa tra rank;
- costo del runtime;
- sbilanciamenti;
- rumore del sistema operativo.

Idealmente:

```text
T_ideale(64) = T(1) / 64 = 31.186 / 64 = 0.487 s
```

Misurato:

```text
T_reale(64) = 0.538 s
```

Differenza:

```text
0.538 - 0.487 = 0.051 s
```

Questa differenza è overhead + parti non perfettamente parallelizzabili.

Nel progetto è piccola, quindi la scalabilità è buona.

---

## 15. Weak scaling

### 15.1 Definizione generale

Weak scaling significa:

```text
aumento le risorse
aumento proporzionalmente anche il problema
misuro se il tempo resta controllato
```

In molti problemi locali, per esempio stencil, il tempo ideale resta circa costante.

### 15.2 Perché N-body diretto è diverso

Nel direct N-body ogni particella interagisce con tutte le altre.

Nel progetto:

```text
N per rank = 2000
N totale = P * 2000
```

Però ogni particella locale vede tutto `N totale`.

Quindi il lavoro per rank cresce con `P`.

Schema:

```text
P = 1:
rank 0 ha 2000 target
ogni target vede 2000 sorgenti

P = 2:
ogni rank ha 2000 target
ogni target vede 4000 sorgenti

P = 64:
ogni rank ha 2000 target
ogni target vede 128000 sorgenti
```

Quindi il tempo debba aumentare non è un fallimento. È una conseguenza dell'algoritmo.

### 15.3 Come viene interpretato nel report

Il report mostra:

- speedup/efficiency secondo definizione standard;
- tempo assoluto con riferimento ideale `O(P)`;
- tempo normalizzato `T(P) / (P*T(1))`.

Il tempo normalizzato serve a chiedere:

```text
il codice segue circa la crescita O(P) attesa?
```

Risposta:

```text
sì, a 64 rank è solo circa 6.6% sopra il riferimento O(P)
```

Quindi il weak scaling non è "piatto", ma è coerente con il direct all-pairs.

---

## 16. Bottleneck

### 16.1 Che cos'è un bottleneck

Un bottleneck è la parte del codice che limita le prestazioni.

Se una parte prende quasi tutto il tempo, migliorare altre parti cambia poco.

Schema:

```text
tempo totale
|------------------------------------------------|
| force kernel                                   | comunicazione | altro |
|------------------------------------------------|
```

Se il force kernel domina, il collo di bottiglia è il calcolo delle forze.

### 16.2 Come viene misurato nel progetto

Il solver stampa tempi separati:

```text
total
io
drift
force
comm_wait
kick
energy
```

Nel run strong a 64 rank:

```text
total      = 0.538247 s
force      = 0.466024 s
comm_wait  = 0.007174 s
```

Quindi:

```text
force / total ≈ 86.6%
comm / total  ≈ 1.3%
```

Conclusione:

```text
il bottleneck è il force kernel, non la comunicazione
```

---

## 17. Benchmark e statistica

### 17.1 Perché servono ripetizioni

Un singolo tempo non è affidabile. In HPC i tempi possono cambiare per:

- rumore del sistema;
- scheduling;
- cache;
- filesystem;
- interferenze del nodo;
- variazioni del runtime MPI/OpenMP.

Per questo ogni punto finale usa 5 ripetizioni.

### 17.2 Perché usare la mediana

La mediana è più robusta della media quando ci sono outlier.

Esempio:

```text
tempi = 1.00, 1.01, 1.00, 1.02, 4.80
```

La media viene sporcata dal `4.80`.

La mediana resta circa `1.01`.

Per questo il report usa:

```text
median time
standard deviation
MAD/outlier count
failed runs
```

### 17.3 CSV grezzi e CSV riassunti

Gli script producono due livelli:

```text
benchmark_results.csv
    contiene ogni singola ripetizione

benchmark_summary.csv
    contiene mediana, stdev, speedup, efficiency
```

Questo rende il risultato riproducibile e verificabile.

---

## 18. Container

### 18.1 Perché usare un container

Un container rende l'ambiente più riproducibile.

Senza container:

```text
dipendo dai moduli installati sul cluster
```

Con container:

```text
porto con me sistema base, librerie, tool e binari
```

### 18.2 Perché MPI nei container è delicato

MPI non è una libreria qualsiasi: deve parlare con Slurm, rete, process manager e librerie del cluster.

Se dentro il container uso un MPI diverso da quello del cluster, posso avere:

- crash;
- performance sbagliate;
- warning;
- fallback su trasporti lenti;
- risultati non confrontabili.

Per questo il progetto usa host MPI injection:

```text
container contiene MPI per compilare
runtime usa MPI del cluster
```

Schema:

```text
build container:
    OpenMPI dentro immagine -> compila /opt/nbody/nbody_direct_hybrid

run su cluster:
    bind /opt/programs/openMPI dal sistema host
    LD_LIBRARY_PATH punta al MPI host
    ldd verifica che libmpi venga dal cluster
```

### 18.3 Che cosa viene misurato

La parte container misura:

1. solver native vs solver Singularity;
2. launch overhead;
3. OSU latency native vs container;
4. OSU bandwidth native vs container;
5. linkage MPI con `ldd`.

### 18.4 Overhead del container

Formula:

```text
overhead % = (T_container - T_native) / T_native * 100
```

Risultato finale:

```text
circa 3.0% - 3.4%
```

Interpretazione:

- il container aggiunge un piccolo costo;
- il costo è stabile;
- non c'è collasso MPI;
- il binding del MPI host è corretto.

### 18.5 Launch overhead

Il launch overhead è il costo di avviare il container.

Esempio:

```text
singularity exec nbody.sif true
```

Nel progetto è circa:

```text
0.09 - 0.11 s
```

Per run lunghi è trascurabile. Per run piccolissimi può dominare.

### 18.6 OSU Micro-Benchmarks

OSU misura comunicazione MPI elementare.

Nel progetto si usano:

- `osu_latency`: latenza messaggio piccolo;
- `osu_bw`: banda per messaggi più grandi.

Se native e container sono simili, significa che il container non sta rompendo il layer MPI.

Risultato:

```text
latency e bandwidth native/container sono molto vicine
```

---

## 19. Risultati finali da ricordare

### 19.1 Strong scaling

Dataset:

```text
results_final/scaling_64_summary.csv
```

Risultati principali:

| ranks | tempo mediano (s) | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 31.186382 | 1.00 | 100.0% |
| 2 | 16.048572 | 1.94 | 97.2% |
| 4 | 8.147579 | 3.83 | 95.7% |
| 8 | 4.116640 | 7.58 | 94.7% |
| 16 | 2.089044 | 14.93 | 93.3% |
| 32 | 1.047827 | 29.76 | 93.0% |
| 64 | 0.538247 | 57.94 | 90.5% |

Messaggio da difendere:

```text
lo strong scaling è quasi ideale fino al full node GENOA
```

### 19.2 Weak scaling

Dataset:

```text
results_final/scaling_64_summary.csv
```

Risultato chiave:

```text
T(P)/(P*T(1)) ≈ 1.066 a 64 rank
```

Messaggio da difendere:

```text
il tempo cresce perché il direct N-body ha costo globale O(N^2);
normalizzando rispetto alla crescita O(P) attesa, il comportamento è buono
```

### 19.3 Hybrid MPI/OpenMP

Dataset:

```text
results_final/hybrid_64_summary.csv
```

Risultato:

```text
migliore: 32 MPI ranks x 2 OpenMP threads
differenza tra migliore e peggiore: circa 1.26%
```

Messaggio da difendere:

```text
la decomposizione MPI/OpenMP non è un parametro fragile nel range testato
```

### 19.4 Bottleneck

Risultato a 64 rank:

```text
force ≈ 86.6% del tempo totale
communication wait ≈ 1.3%
```

Messaggio:

```text
il collo di bottiglia è il calcolo delle forze
```

### 19.5 Container

Dataset:

```text
results_final/container_overhead_summary.csv + results_final/osu_microbench_summary.csv
```

Risultato:

```text
overhead solver ≈ 3.0% - 3.4%
launch overhead ≈ 0.09 - 0.11 s
OSU native/container molto simili
host MPI verificato con ldd
```

Messaggio:

```text
il container è riproducibile e introduce overhead piccolo
```

---

## 20. Come leggere i grafici

### 20.1 Strong speedup

Grafico:

```text
results_final/scaling_64_strong_speedup.svg
```

Come leggerlo:

- asse x: risorse, qui MPI ranks/core sullo stesso nodo;
- asse y: speedup;
- linea tratteggiata: ideale;
- più la curva misurata è vicina all'ideale, meglio è.

### 20.2 Strong efficiency

Grafico:

```text
results_final/scaling_64_strong_efficiency.svg
```

Come leggerlo:

- ideale = 1;
- valori sopra 0.9 sono molto buoni;
- la discesa indica overhead crescente.

### 20.3 Weak time

Grafico:

```text
results_final/scaling_64_weak_time.svg
```

Come leggerlo:

- non deve essere piatto;
- il riferimento giusto per questo algoritmo è `O(P)`;
- se segue la linea `O(P)`, il comportamento è coerente.

### 20.4 Hybrid configuration

Grafici:

```text
results_final/hybrid_64_time_by_config.svg
results_final/hybrid_64_gpairs_by_config.svg
```

Come leggerli:

- tutte le barre/punti sono vicini;
- non bisogna usare speedup perché il totale risorse è fisso;
- si confrontano tempo e throughput tra configurazioni `P x T`.

### 20.5 Bottleneck breakdown

Grafico:

```text
results_final/scaling_64_strong_breakdown.svg
```

Come leggerlo:

- mostra quanto pesa force, comm e altro;
- serve a giustificare che il force kernel domina.

### 20.6 Container e OSU

Grafici:

```text
results_final/container_overhead_strong.svg
results_final/container_overhead_weak.svg
results_final/osu_microbench_latency.svg
results_final/osu_microbench_bandwidth.svg
```

Come leggerli:

- container leggermente più lento del native;
- differenza stabile e piccola;
- OSU conferma che la comunicazione MPI non peggiora drasticamente.

---

## 21. Workflow del progetto

### 21.1 Pipeline generale

```text
1. Compila codice
        |
        v
2. Genera condizioni iniziali
        |
        v
3. Esegui solver seriale/ibrido
        |
        v
4. Raccogli output testuali
        |
        v
5. Parsifica in CSV
        |
        v
6. Calcola statistiche
        |
        v
7. Genera grafici
        |
        v
8. Scrivi report
```

### 21.2 Comandi base

Da dentro `Nbody_serial`:

```bash
make clean
make
```

Smoke test:

```bash
bash preflight.sh
```

Run seriale manuale:

```bash
./generate_ic --model 0 --n 128 --seed 1 --output test_128.bin
./nbody_direct_serial --input test_128.bin --nsteps 5
```

Run MPI/OpenMP manuale:

```bash
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun -np 2 ./nbody_direct_hybrid \
  --input test_128.bin \
  --nsteps 5 \
  --integrator kdk \
  --comm overlap \
  --kernel direct \
  --rsqrt exact \
  --quiet
```

---

## 22. Job su Orfeo

La produzione finale è stata fatta su Orfeo/GENOA perché la disponibilità CPU su Leonardo DCGP non permetteva di completare i run richiesti. Questo non cambia la logica HPC del progetto: si usa comunque Slurm, MPI, OpenMP, Singularity e host MPI injection.

Script utili:

```text
jobs/submit.sh --cluster orfeo --bench probe
jobs/submit.sh --cluster orfeo --bench scaling
jobs/submit.sh --cluster orfeo --bench hybrid
jobs/submit.sh --cluster orfeo --bench ablation
jobs/submit.sh --cluster orfeo --bench container
jobs/submit.sh --cluster orfeo --bench evidence
```

File finali:

```text
results_final/scaling_64_summary.csv
results_final/hybrid_64_summary.csv
results_final/ablation_64.csv
results_final/layout_summary.csv + results_final/energy_overhead_summary.csv
results_final/container_overhead_summary.csv + results_final/osu_microbench_summary.csv
```

---

## 23. Cosa dire se chiedono "perché non Leonardo?"

Risposta breve e difendibile:

```text
L'esercizio cita Leonardo, ma durante i test l'account CPU DCGP non accettava i job.
Per evitare misure parziali o non riproducibili, i run finali sono stati eseguiti su un singolo nodo Orfeo GENOA omogeneo.
La metodologia HPC richiesta resta la stessa: Slurm, MPI+OpenMP, scaling, Singularity, OSU e verifica host MPI.
```

Punto importante:

```text
non mischiare GENOA, EPYC e Leonardo nello stesso grafico principale
```

Perché mischiare architetture renderebbe ambiguo il confronto.

---

## 24. Come difendere le scelte principali

### 24.1 Perché direct O(N^2)?

Perché è richiesto dall'esercizio e rende chiaro il comportamento HPC.

Non si usa Barnes-Hut/FMM perché cambierebbe algoritmo.

### 24.2 Perché MPI ring?

Perché ogni rank deve vedere tutti i blocchi sorgente, ma non vogliamo replicare sempre tutto.

### 24.3 Perché OpenMP sul loop esterno?

Per evitare race condition e atomiche nel loop interno.

### 24.4 Perché KDK?

Perché leapfrog KDK è stabile, semplice e adatto a sistemi gravitazionali.

### 24.5 Perché energia non a ogni step nei benchmark?

Perché l'energia costa molto e può falsare la misura prestazionale.

### 24.6 Perché usare mediana?

Perché i tempi HPC hanno outlier.

### 24.7 Perché container con host MPI?

Perché MPI deve essere compatibile con Slurm e librerie del cluster.

### 24.8 Perché SoA non migliora?

Perché un'ottimizzazione teorica dipende da implementazione, compilatore e architettura. Il benchmark misura il caso reale.

---

## 25. Glossario essenziale

| Termine | Significato semplice |
|---|---|
| Rank MPI | processo MPI separato |
| Thread OpenMP | flusso di esecuzione dentro un processo |
| Strong scaling | stesso problema, più risorse |
| Weak scaling | problema più grande proporzionale alle risorse |
| Speedup | quanto vado più veloce rispetto al caso base |
| Efficiency | speedup diviso risorse |
| Overhead | costo extra del parallelismo |
| Bottleneck | parte che limita la prestazione |
| Softening | modifica che evita forza infinita a distanza zero |
| Leapfrog | integratore temporale stabile per sistemi fisici |
| KDK | kick-drift-kick |
| AoS | array di strutture |
| SoA | struttura di array |
| OSU | microbenchmark MPI per latenza/banda |
| SIF | immagine Singularity |
| Host MPI injection | usare MPI del cluster dentro il container |

---

## 26. Mini schema mentale finale

```text
Problema:
    simulare gravità N-body diretta

Costo:
    O(N^2), force kernel dominante

Parallelizzazione:
    MPI divide particelle tra rank
    OpenMP divide loop locale tra thread
    ring-shift fa circolare i blocchi sorgente

Correttezza:
    energia quasi conservata
    checksum layout coerenti

Scalabilità:
    strong scaling quasi ideale fino a 64 rank
    weak scaling cresce come atteso per direct all-pairs
    hybrid P x T stabile

Ottimizzazioni:
    Newton utile single-rank ma complicato in MPI
    rsqrt approx non conviene qui
    overlap poco rilevante single-node
    layout va misurato, non assunto

Container:
    Docker/Singularity riproducibile
    host MPI verificato
    overhead circa 3%
```

---

## 27. Frasi pronte per il report/orale

### Strong scaling

```text
The strong-scaling curve is close to the ideal reference. At 64 ranks the solver reaches a speedup of 57.94x and an efficiency of 90.5%, showing that the direct force kernel provides enough work to amortize MPI and OpenMP overhead within one GENOA node.
```

### Weak scaling

```text
The weak-scaling time is not expected to remain constant for a direct all-pairs N-body solver. With fixed particles per rank, each local particle still interacts with the globally growing particle set, so the expected per-rank work grows approximately linearly with the number of ranks.
```

### Bottleneck

```text
The timing breakdown shows that the force kernel accounts for about 86.6% of the total runtime at 64 ranks, while communication wait is about 1.3%. Therefore, the main bottleneck is computation rather than MPI communication in the tested single-node regime.
```

### Container

```text
The Singularity container uses host MPI injection, verified through ldd. Native and container OSU latency/bandwidth are close, and the solver overhead is stable around 3%, indicating that the container preserves MPI performance reasonably well.
```

### Ottimizzazioni

```text
The ablation study shows that optimisations must be measured on the actual implementation and architecture. Newton reuse improves the single-rank kernel but is not directly free in distributed MPI; approximate rsqrt is slower here; communication overlap brings little benefit because communication is already small.
```

---

## 28. Controlli finali prima della consegna

Prima di consegnare, controllare:

- `../FINAL_REPORT.md` contiene tutte le sezioni richieste;
- i grafici referenziati esistono;
- i CSV finali non contengono `RUN_FAILED`, `PARSE_FAILED`, `nan`;
- il report non usa cartelle vecchie o tentativi falliti;
- il container finale è quello in `results_final`;
- i risultati principali sono quelli a 64 core GENOA;
- la guida non confonde run esplorativi EPYC/128 con dataset ufficiale;
- i comandi riproducibili sono presenti;
- la scelta Orfeo/GENOA è spiegata.

Comando utile per controllare errori nei CSV:

```bash
grep -RIn "RUN_FAILED\\|PARSE_FAILED\\|nan" results_final results_final/layout_summary.csv + results_final/energy_overhead_summary.csv results_final/container_overhead_summary.csv + results_final/osu_microbench_summary.csv
```

Comando utile per rigenerare grafici esplicativi:

```bash
python3 analyze.py plot evidence .
```

---

## 29. In una frase

Il progetto dimostra che un solutore gravitazionale diretto `O(N^2)` può scalare molto bene su un singolo nodo quando il force kernel domina, che il modello MPI+OpenMP è stabile rispetto alla decomposizione `P x T`, e che il container Singularity introduce un overhead piccolo se si usa correttamente il MPI del cluster.
