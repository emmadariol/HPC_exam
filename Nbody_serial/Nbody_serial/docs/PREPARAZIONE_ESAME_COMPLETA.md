# Preparazione completa all'esame HPC - Progetto N-body MPI + OpenMP

## Come usare questo documento

Questo file non e un riassunto da imparare a memoria. E una guida per arrivare all'orale sapendo:

1. identificare il problema;
2. ricavare una soluzione sensata;
3. spiegare il codice che la implementa;
4. individuare i punti critici;
5. dimostrare le proprie affermazioni con misure;
6. riconoscere un risultato insoddisfacente;
7. proporre e verificare un miglioramento.

Per ogni argomento viene usato, quando applicabile, lo schema:

```text
Che cosa e -> Perche serve -> Come e implementato -> Rischio
-> Evidenza -> Interpretazione -> Miglioramento -> Domanda orale
```

Le fonti principali del progetto sono:

- `../../assignment/Final_exams---Exercises.pdf`: consegna ufficiale;
- `../../assignment/A_few_words_on_scalability.pdf`: indicazioni su strong e weak scaling;
- `../FINAL_REPORT.md`: risultati e configurazioni finali;
- `EXERCISE1_DETAILED_GUIDE.md`: spiegazione tecnica estesa;
- `nbody_direct_hybrid.c`: implementazione MPI + OpenMP;
- `nbody_direct_serial.c`: riferimento seriale;
- `nbody_layout_benchmark.c`: confronto AoS/SoA;
- script `run_benchmarks.sh`, `analyze.py` e `analyze.py plot`: pipeline sperimentale.

---

## 1. La storia completa del progetto in due minuti

Il progetto simula un sistema gravitazionale di `N` particelle. Il metodo diretto calcola tutte le interazioni, quindi costa `O(N^2)` per valutazione delle forze. Il tempo viene integrato con leapfrog KDK, di ordine due e simplettico. La correttezza viene controllata tramite la massima deriva relativa dell'energia totale.

MPI distribuisce le particelle: ogni rank possiede permanentemente un home chunk e riceve, uno dopo l'altro, i chunk sorgente degli altri rank attraverso un ring-shift. OpenMP divide le particelle target locali fra i thread; ogni thread accumula la forza di target distinti, evitando atomiche nel loop interno.

Il progetto misura strong scaling, weak scaling specifico dell'algoritmo `O(N^2)`, diverse decomposizioni MPI/OpenMP, costo della diagnostica energetica, legge di Newton, radice quadrata inversa, layout AoS/SoA, sovrapposizione comunicazione-calcolo e overhead Singularity.

Il risultato centrale e uno speedup di `57.94x` su 64 core, con efficienza `90.5%`. Il force kernel occupa circa `86.6%` del tempo e l'attesa MPI circa `1.3%`: nel regime single-node misurato il collo di bottiglia e il calcolo, non la comunicazione. Alcune ottimizzazioni suggerite non hanno funzionato: `rsqrt` approssimata e SoA sono risultati piu lenti; overlap e sendrecv sono indistinguibili entro la variabilita. Questi risultati non vanno nascosti: mostrano che le ottimizzazioni sono state verificate sperimentalmente.

---

## 2. Che cosa richiede davvero la consegna

La consegna non richiede soltanto un eseguibile parallelo. Richiede una catena completa di prove.

| Requisito | Domanda a cui risponde | Evidenza nel progetto |
|---|---|---|
| Identificazione hardware | Su quale macchina sono valide le misure? | system info e sezione 4 del report |
| Stack software | Quali compilatore, MPI e runtime sono stati usati? | `collect_system_info.sh`, report |
| Flag esatti | Quale codice macchina e stato generato? | `Makefile`, Dockerfile, report |
| Configurazione dei run | Come sono stati collocati rank e thread? | script Slurm, variabili OMP |
| Verifica quantitativa | Il risultato fisico e corretto? | energy drift, checksum AoS/SoA |
| Almeno 5 ripetizioni | Il risultato e robusto al rumore? | CSV grezzi e summary |
| Strong e weak scaling | Il codice usa efficientemente piu risorse? | grafici e tabelle |
| Profiling/instrumentation | Qual e il collo di bottiglia? | tempi force, comm, energy, I/O |
| Container comparison | Il container altera le prestazioni? | native/container, launch, OSU, `ldd` |

Errore tipico: mostrare un buon grafico senza spiegare la causa della forma della curva. Un grafico e un'osservazione, non ancora una spiegazione.

---

## 3. Modello fisico

### 3.1 Stato di una particella

Ogni particella ha posizione `r_i = (x_i,y_i,z_i)`, velocita `v_i`, massa `m_i` e accelerazione `a_i`. Nel progetto le masse sono uguali e passate da riga di comando; non sono memorizzate nel file binario.

Questa scelta semplifica dati e comunicazione, ma e una limitazione: il codice non descrive direttamente sistemi multi-massa. Per generalizzarlo servirebbe un array delle masse, includerlo nel formato di input e ruotarlo insieme alle posizioni.

### 3.2 Accelerazione gravitazionale ammorbidita

La formula implementata e:

```text
a_i = sum_{j != i} G m_j (r_j-r_i) /
      (|r_j-r_i|^2 + eps^2)^(3/2)
```

Per ogni coppia si calcolano le differenze coordinate, `r2`, la radice inversa e i tre contributi all'accelerazione.

Il softening `eps`:

- mantiene finita la forza per incontri ravvicinati;
- modifica realmente il potenziale e quindi la fisica;
- riduce la necessita di timestep minuscoli;
- deve comparire coerentemente sia nella forza sia nell'energia potenziale.

Non e corretto dire soltanto “serve a evitare divisioni per zero”. La sua scelta modifica le scale spaziali risolte. Se e troppo piccolo aumentano accelerazioni estreme ed errore temporale; se e troppo grande vengono cancellate interazioni fisiche a piccola scala.

Domanda orale: “Cosa succede aumentando `eps` di dieci volte?”

Risposta ragionata: diminuisce la forza alle piccole distanze, il sistema e numericamente meno rigido e puo mostrare una deriva energetica minore a `dt` fisso, ma la dinamica non rappresenta piu lo stesso modello fisico. Il costo asintotico resta `O(N^2)`; il tempo puo cambiare poco perche si esegue quasi la stessa sequenza di istruzioni.

### 3.3 Condizioni iniziali

`generate_ic.c` produce:

- una sfera di Plummer;
- una sfera uniforme con velocita Maxwelliane.

Il seed rende l'esperimento riproducibile. A parita di seed, parametri e precisione di memorizzazione, tutti i confronti partono dallo stesso stato.

Criticita: condizioni iniziali diverse possono cambiare la dinamica e la deriva energetica. Nei confronti prestazionali vanno quindi mantenute fisse, altrimenti si confonde l'effetto della variante software con quello del caso fisico.

---

## 4. Metodo numerico: leapfrog KDK

### 4.1 Sequenza

```text
v(t+dt/2) = v(t) + a(t) dt/2        primo kick
r(t+dt)   = r(t) + v(t+dt/2) dt     drift
a(t+dt)   = force(r(t+dt))           nuova forza
v(t+dt)   = v(t+dt/2) + a(t+dt)dt/2 secondo kick
```

Il solver hybrid usa esclusivamente KDK, come richiesto dalla consegna.

### 4.2 Perche leapfrog

- e di ordine due: l'errore globale decresce circa come `dt^2` nel regime asintotico;
- e simplettico: conserva la struttura Hamiltoniana e tende a produrre errore energetico limitato e oscillante;
- e time-reversible in aritmetica esatta;
- richiede una valutazione nuova delle forze per step dopo l'inizializzazione.

“Simplettico” non significa energia esattamente costante. Significa che l'integrazione segue un Hamiltoniano modificato vicino a quello reale e, in genere, evita una deriva secolare tipica di altri schemi.

### 4.3 Come verificare l'ordine

Un esperimento convincente usa lo stesso stato iniziale e simula lo stesso intervallo fisico con `dt`, `dt/2`, `dt/4`. Se si e nel regime di convergenza, un errore appropriato dovrebbe ridursi di circa quattro volte quando `dt` viene dimezzato.

Se non accade, possibili cause sono:

- errore di roundoff dominante;
- softening/timestep non nel regime asintotico;
- confronto di traiettorie caotiche troppo lungo;
- bug nella sequenza KDK;
- forza ed energia che usano potenziali diversi.

---

## 5. Correttezza: che cosa dimostra davvero ogni test

### 5.1 Deriva dell'energia

```text
delta_E_max = max_t |E(t)-E(0)| / max(|E(0)|, tiny)
```

La tolleranza dei benchmark e `1e-3`; i run finali risultano `OK`. Nello studio dell'overhead energetico la deriva osservata e dell'ordine di `10^-7`.

Questo test rileva errori nella forza, nell'integratore o nella coerenza del potenziale. Non dimostra da solo che ogni particella sia corretta: errori compensanti potrebbero mantenere quasi invariata l'energia.

### 5.2 Checksum AoS/SoA

Il benchmark confronta checksum delle accelerazioni. Le differenze sono `0` o circa `1.75e-10`, compatibili con riordinamenti floating-point.

Il checksum e utile per confrontare kernel, ma puo avere collisioni. Un test piu forte confronterebbe componente per componente con norme:

```text
errore_rel_L2 = ||a_test-a_ref||_2 / ||a_ref||_2
errore_Linf   = max_k |a_test_k-a_ref_k|
```

### 5.3 Riferimento seriale

Per piccoli `N`, il modo migliore di verificare la versione parallela e confrontarla con `nbody_direct_serial.c`, usando gli stessi parametri e tenendo conto che l'ordine di somma cambia. Il confronto deve usare tolleranze, non uguaglianza bit-a-bit.

### 5.4 Test minimi che devi saper proporre

1. `N=1`: accelerazione zero e moto rettilineo uniforme.
2. Due particelle simmetriche: accelerazioni opposte e centro di massa fermo.
3. `N` non divisibile per `P`: verifica dei blocchi sbilanciati di una particella.
4. `P>N`: rank con blocco vuoto, se supportato; altrimenti errore esplicito.
5. Confronto 1 rank / molti rank.
6. Confronto 1 thread / molti thread.
7. Confronto sendrecv / overlap.
8. Riduzione di `dt` e studio della convergenza.

---

## 6. Complessita e modello dei costi

### 6.1 Computazione

Una valutazione diretta delle forze costa `Theta(N^2)`. Con `P` rank e bilanciamento ideale:

```text
T_compute(P,N) ~ c N^2 / P
```

Con `nsteps` KDK, il termine dominante e proporzionale a `nsteps*N^2/P`.

Se `N` aumenta di un fattore 10, il tempo atteso del kernel aumenta circa di un fattore 100, finche non cambiano regime di cache, frequenza, vettorizzazione o bilanciamento.

### 6.2 Comunicazione del ring

Ogni rank possiede circa `N/P` sorgenti e compie `P-1` scambi. Il volume totale per rank per valutazione e `Theta(N)`. Un modello semplice e:

```text
T_comm ~ (P-1)*alpha + beta*N
```

dove `alpha` e la latenza per messaggio e `beta` il costo inverso di banda per byte. A grande `P`, il termine di latenza cresce anche se il volume totale resta simile.

### 6.3 Memoria

Il ring evita di replicare permanentemente tutte le `N` particelle su ogni rank. Ogni rank conserva home chunk e buffer di dimensione circa `N/P`, quindi la memoria per rank e `Theta(N/P)`, oltre ai buffer temporanei.

### 6.4 Perche non Barnes-Hut

Barnes-Hut ridurrebbe tipicamente il costo a circa `O(N log N)`, ma non e consentito dalla consegna. Il direct solver e scelto per rendere visibili SIMD, FMA, dipendenze di accumulo, layout e rapporto computazione/comunicazione. Questa e una scelta didattica, non l'affermazione che `O(N^2)` sia l'algoritmo migliore in produzione.

---

## 7. Decomposizione MPI

### 7.1 Ownership

`block_bounds` assegna blocchi contigui. Se `N` non e divisibile per `P`, i primi rank ricevono una particella in piu. Ogni rank aggiorna solo le proprie particelle home.

Invariante fondamentale:

```text
ogni particella e posseduta e aggiornata da un solo rank;
ogni home chunk vede una volta tutti i chunk sorgente.
```

### 7.2 Ring-shift

In ogni fase:

1. il rank accumula le forze dal buffer sorgente corrente;
2. invia quel buffer al vicino;
3. riceve il successivo dall'altro vicino;
4. ripete fino ad aver visto tutti i proprietari.

Vantaggi:

- volume regolare;
- memoria distribuita;
- solo comunicazioni fra vicini logici;
- nessun `Allgather` dell'intero stato.

Svantaggi:

- `P-1` fasi sequenziali;
- latenza crescente con `P`;
- tutti i rank devono avanzare in modo sufficientemente coordinato;
- gestire blocchi di dimensione diversa richiede conteggi corretti.

### 7.3 Alternativa Allgather

`MPI_Allgatherv` renderebbe il kernel piu semplice: ogni rank riceverebbe tutto `N`. Pero la memoria per rank diventerebbe `Theta(N)` e ogni valutazione richiederebbe una collettiva globale. Puo essere competitivo per piccoli `N/P` e pochi rank, ma scala peggio in memoria.

Un buon miglioramento sperimentale e implementare entrambe le strategie e cercare il crossover in funzione di `N`, `P` e numero di nodi.

### 7.4 Sendrecv e overlap

`sendrecv` separa comunicazione e calcolo. `overlap` pubblica `MPI_Irecv`/`MPI_Isend` del prossimo chunk, calcola sul chunk corrente e poi attende.

Perche l'overlap reale puo essere scarso:

- MPI potrebbe non progredire senza chiamate MPI;
- il trasferimento e gia molto breve;
- la comunicazione condivide risorse con i thread;
- i buffer o la memoria possono creare contesa;
- il calcolo corrente puo essere troppo breve per nascondere la latenza.

Risultato misurato:

```text
sendrecv = 0.938524 s
overlap  = 0.937171 s
differenza = 0.001353 s, inferiore alla deviazione standard
```

Conclusione difendibile: non e stata dimostrata una differenza statisticamente significativa nel caso single-node testato. Non si puo generalizzare ai run multi-node.

---

## 8. Parallelizzazione OpenMP

### 8.1 Scelta del loop

Il `parallel for` e sul loop dei target locali. Ogni iterazione possiede accumulatori `ax`, `ay`, `az` privati e scrive un solo elemento finale.

Questa scelta evita:

- race condition sugli accumulatori;
- atomiche nel loop `j`;
- false sharing significativo durante l'accumulo, perche la scrittura avviene alla fine.

### 8.2 Scheduling statico

Il costo di ogni target e quasi identico nel direct solver: visita tutte le sorgenti. `schedule(static)` ha overhead minimo ed e quindi appropriato. `dynamic` sarebbe utile con lavoro irregolare, ma qui introdurrebbe gestione della coda senza un reale vantaggio.

### 8.3 Variabili condivise e private

- condivise in sola lettura: coordinate home e sorgenti, masse, `eps`, `G`;
- private: indici, differenze, distanza, fattore e accumulatori;
- condivise in scrittura ma disgiunta: `ax[i]`, `ay[i]`, `az[i]`.

Domanda orale: “Perche non serve `atomic`?”

Risposta: ogni iterazione `i` e assegnata a un solo thread e solo quel thread scrive le accelerazioni di `i`; le sorgenti sono lette soltanto. Un'atomica sarebbe necessaria se thread diversi aggiornassero lo stesso target, come in una parallelizzazione ingenua delle coppie con la terza legge di Newton.

### 8.4 Binding e NUMA

I run usano:

```text
OMP_PLACES=cores
OMP_PROC_BIND=spread
srun --cpu-bind=verbose,cores
```

Il binding evita migrazioni. `spread` distribuisce thread fra i core/place disponibili; puo migliorare banda e uso delle risorse, ma su otto domini NUMA va verificato insieme al first-touch della memoria. Un confronto con `close` e con un rank per NUMA domain renderebbe piu solida la scelta.

---

## 9. Configurazioni ibride MPI x OpenMP

Con 64 core sono state provate `64x1`, `32x2`, `16x4`, `8x8`, `4x16`, `2x32`, `1x64`.

| P x T | Tempo mediano (s) |
|---|---:|
| 64x1 | 0.541323 |
| 32x2 | 0.534608 |
| 16x4 | 0.540644 |
| 8x8 | 0.537852 |
| 4x16 | 0.536850 |
| 2x32 | 0.536742 |
| 1x64 | 0.540497 |

Il migliore e `32x2`, ma lo spread fra migliore e peggiore e solo circa `1.26%`. Non bisogna proclamare `32x2` vincitore assoluto senza un test statistico: il dato piu solido e che il programma non e molto sensibile alla decomposizione nel caso testato.

Possibili cause della stabilita:

- il kernel aritmetico domina;
- il ring single-node costa poco;
- tutti i layout usano gli stessi 64 core;
- la banda di memoria non e il limite principale del force kernel.

Per estendere lo studio: ripetere multi-node, confrontare un rank per NUMA domain/socket/core e registrare affinity effettiva.

---

## 10. Layout dei dati: AoS contro SoA

### 10.1 Differenza

AoS:

```c
particle p[N]; // x,y,z,... vicini per particella
```

SoA:

```c
dtype x[N], y[N], z[N], vx[N], vy[N], vz[N];
```

SoA dovrebbe favorire accessi contigui alle sole componenti necessarie e vettorizzazione. AoS puo invece offrire buona localita per operazioni che usano tutti i campi di una particella.

### 10.2 Risultato reale

| Layout | Thread | Force time (s) | Gpairs/s |
|---|---:|---:|---:|
| AoS | 1 | 25.338622 | 0.296 |
| SoA | 1 | 28.315078 | 0.265 |
| AoS | 8 | 3.224015 | 2.326 |
| SoA | 8 | 3.607846 | 2.079 |

SoA e circa 11-12% piu lento nel microbenchmark. Il risultato e insoddisfacente rispetto all'ipotesi, ma numericamente valido grazie al checksum.

### 10.3 Diagnosi corretta

Non basta dire “il compilatore non ha ottimizzato”. Bisogna controllare:

1. report di vettorizzazione (`-fopt-info-vec-all`);
2. assembly del loop;
3. istruzioni FP e larghezza SIMD con contatori hardware;
4. cache miss e bandwidth;
5. eventuali differenze fra i due kernel oltre al layout;
6. aliasing e uso di `restrict`/`const`;
7. allineamento reale degli array;
8. frequenza CPU e variabilita dei run.

Possibile miglioramento: costruire due kernel il piu possibile isomorfi, aggiungere `restrict`, garantire allineamento, usare `omp simd reduction`, verificare il report del compilatore e poi ripetere. Finche questo non e fatto, la spiegazione resta un'ipotesi.

---

## 11. Terza legge di Newton

### 11.1 Opportunita

Per una coppia `(i,j)`, `F_ij=-F_ji`. Calcolando ogni coppia una volta si dimezza quasi il numero teorico di interazioni.

### 11.2 Conflitto shared-memory

Se un thread elabora `(i,j)`, deve aggiornare sia `a[i]` sia `a[j]`. Thread diversi possono condividere `i` o `j`, creando race. Soluzioni:

- atomiche: semplici ma costose;
- lock: inaccettabili nel kernel fine-grained;
- buffer privati per thread: veloci ma memoria `O(TN)` e riduzione finale;
- decomposizione a blocchi/coloring: piu complessa.

### 11.3 Problema distribuito

Nel ring, `j` puo appartenere a un altro rank. Il contributo opposto deve tornare al proprietario, aggiungendo comunicazione e accumulo. Per questo la variante implementata e intenzionalmente single-rank.

### 11.4 Risultato

```text
direct = 44.123917 s
newton = 34.708724 s
miglioramento = 21.3%
```

Il miglioramento e inferiore al 50% perche dimezzare le coppie non dimezza ogni costo: restano gestione buffer, riduzione, accessi e dipendenze. La domanda giusta non e “Newton e piu veloce?”, ma “in quale regime il risparmio aritmetico supera sincronizzazione, memoria e comunicazione?”.

---

## 12. Radice quadrata inversa

Il kernel usa `1/(r2)^(3/2)`. Una `rsqrt` approssimata con raffinamento Newton puo essere vantaggiosa solo se:

- il processore fornisce una buona istruzione vettoriale;
- il compilatore la genera;
- conversioni e raffinamento costano meno della strada esatta;
- l'errore resta accettabile.

Risultato:

```text
exact  = 0.943831 s
approx = 1.166320 s
approx e 23.6% piu lenta
```

Diagnosi da eseguire prima di concludere:

- controllare assembly e report SIMD;
- verificare se l'approssimazione usa conversioni float/double;
- separare throughput della sola funzione dal kernel completo;
- confrontare precisione e deriva energetica;
- provare flag e intrinsics specifiche in un benchmark controllato.

Non si deve mantenere una variante approssimata solo perche “in teoria e veloce”. La versione di produzione deve restare `exact` finche non esiste evidenza contraria.

---

## 13. Dipendenze di accumulo, SIMD e FMA

Nel loop interno:

```text
ax += dx * factor
ay += dy * factor
az += dz * factor
```

ogni aggiornamento di `ax` dipende dal precedente. Questa catena puo limitare il throughput anche con FMA vettoriali.

Una tecnica e usare piu accumulatori indipendenti:

```text
ax0,ax1,ax2,ax3 ...
```

e ridurli alla fine. Questo aumenta instruction-level parallelism ma usa piu registri; oltre un certo numero si rischia register spilling. Il numero ottimo va misurato.

Una prova completa comprende:

1. baseline con un accumulatore;
2. 2, 4, 8 accumulatori;
3. report SIMD;
4. Gpairs/s e, se disponibili, FLOP/s hardware;
5. controllo della correttezza;
6. ispezione dello spill nell'assembly.

---

## 14. Strong scaling

### 14.1 Definizioni

Con problema fisso:

```text
S(P) = T(1)/T(P)
E(P) = S(P)/P
```

Ideale: `S(P)=P`, `E(P)=1`.

### 14.2 Dati finali

| Rank | Tempo (s) | Speedup | Efficienza |
|---:|---:|---:|---:|
| 1 | 31.186382 | 1.00 | 100.0% |
| 2 | 16.048572 | 1.94 | 97.2% |
| 4 | 8.147579 | 3.83 | 95.7% |
| 8 | 4.116640 | 7.58 | 94.7% |
| 16 | 2.089044 | 14.93 | 93.3% |
| 32 | 1.047827 | 29.76 | 93.0% |
| 64 | 0.538247 | 57.94 | 90.5% |

Configurazione: `N=20000`, 20 step, 1 thread/rank, KDK, overlap, direct, exact, cinque ripetizioni.

### 14.3 Interpretazione

A 64 rank il tempo ideale e `31.186382/64 = 0.487287 s`; quello misurato e `0.538247 s`. Il gap contiene comunicazione, sincronizzazione, lavoro locale finito, runtime e rumore.

La frazione effettiva in stile Amdahl e circa `0.0017`, ma non e una pura frazione seriale: ingloba overhead dipendenti da `P`. Presentarla come “la percentuale seriale del codice” sarebbe scorretto.

### 14.4 Quando crollera

Continuando ad aumentare `P` a `N` fisso:

- `N/P` diventa troppo piccolo;
- le fasi del ring aumentano;
- latenza e sincronizzazione smettono di essere trascurabili;
- i loop corti usano peggio SIMD e thread;
- il rumore diventa una frazione grande del tempo.

---

## 15. Weak scaling: il punto piu delicato

### 15.1 Definizione generale

Il weak scaling classico mantiene costante il lavoro per risorsa. Per un problema locale questo spesso significa mantenere costante la dimensione locale e aspettarsi tempo costante.

### 15.2 Perche qui il tempo non deve essere costante

Con `n=N/P` particelle home fisse:

```text
lavoro per rank = (N/P)*N = n*(P*n) = P*n^2
```

Ogni rank conserva `n` target, ma il numero globale di sorgenti cresce come `P`. Quindi l'ideale specifico dell'algoritmo e un tempo proporzionale a `P`, non costante.

### 15.3 Dati

| Rank | N totale | Tempo (s) | T(P)/(P*T1) |
|---:|---:|---:|---:|
| 1 | 2000 | 0.312026 | 1.000 |
| 2 | 4000 | 0.646512 | 1.036 |
| 4 | 8000 | 1.312515 | 1.052 |
| 8 | 16000 | 2.640154 | 1.058 |
| 16 | 32000 | 5.294914 | 1.061 |
| 32 | 64000 | 10.617243 | 1.063 |
| 64 | 128000 | 21.289183 | 1.066 |

Il punto a 64 rank e solo `6.6%` sopra la crescita `O(P)` attesa.

### 15.4 Come parlarne senza cadere in contraddizione

Frase corretta: “Con particelle per rank fisse, la dimensione dei dati locali e costante ma il lavoro locale non lo e, perche l'interazione e globale. Mostro quindi sia il tempo assoluto con riferimento `O(P)`, sia `T(P)/(P*T1)`.”

Frase scorretta: “Il weak scaling e quasi perfetto perche il throughput cresce.” Il throughput e utile, ma non sostituisce la definizione del carico mantenuto costante.

Possibile critica del docente: la consegna afferma che il rapporto compute/communication resta costante. La risposta e distinguere volume di dati, numero di interazioni e definizione operativa del test; esplicitare sempre quale quantita e stata mantenuta fissa e quale baseline teorica si usa.

---

## 16. Amdahl e Gustafson

### 16.1 Amdahl

```text
S(P) = 1 / (f + (1-f)/P)
```

Descrive il limite dello speedup a problema fisso quando una frazione `f` non e parallelizzabile. Nei dati reali, il gap contiene anche overhead che la formula semplice non modella.

### 16.2 Gustafson

Gustafson osserva che con piu risorse spesso aumenta la dimensione del problema. Nel progetto, 64 rank permettono di simulare 128000 particelle nel test weak invece di 2000, ma il direct solver rende il costo globale superlineare nella dimensione.

### 16.3 Cosa non fare

- non adattare Amdahl e chiamare `f` “seriale” senza riserve;
- non usare Gustafson come slogan per ignorare il tempo crescente;
- non confrontare strong e weak efficiency come se misurassero lo stesso esperimento.

---

## 17. Instrumentation e bottleneck

Il solver misura `total`, `io`, `drift`, `force`, `comm_wait`, `kick`, `energy` e Gpairs/s.

A 64 rank strong:

```text
total       = 0.538247 s
force       = 0.466024 s = 86.6%
comm_wait   = 0.007174 s = 1.3%
```

Conclusione: il force kernel e il collo di bottiglia nel regime testato.

Limitazioni della misura:

- sezioni sovrapposte non sono sempre additive;
- il massimo fra rank e necessario per il wall time parallelo, ma puo nascondere la distribuzione;
- `comm_wait` non misura tutto il costo MPI: copie e progresso possono avvenire altrove;
- instrumentation e wall clock non provano da sole compute-bound vs memory-bound.

Per rafforzare la diagnosi:

- roofline con arithmetic intensity e FLOP/s;
- contatori per istruzioni FP, cache e bandwidth;
- tempi per rank, non solo massimo;
- scaling del kernel isolato;
- confronto con peak sostenibile, non solo teorico.

---

## 18. Statistica dei benchmark

### 18.1 Protocollo usato

- almeno cinque ripetizioni;
- mediana come stima centrale;
- deviazione standard;
- outlier MAD;
- run grezzi conservati;
- warm-up non mescolati ai punti misurati;
- fallimenti e `nan` controllati.

### 18.2 Mediana e deviazione standard

La mediana e robusta agli outlier. La deviazione standard, pero, e centrata naturalmente sulla media; associare “mediana +/- sigma” e comune nella consegna ma statisticamente non perfetto. Una presentazione piu coerente puo riportare mediana e intervallo interquartile/MAD, oppure trimmed mean e deviazione standard.

Con solo cinque campioni, ogni stima di dispersione e fragile. Se due varianti differiscono meno della dispersione, bisogna dire che non e stata dimostrata una differenza.

### 18.3 Outlier

Un outlier non va cancellato automaticamente. Bisogna:

1. conservarlo nel CSV;
2. identificare un criterio definito prima dell'analisi;
3. cercare una causa, come OS jitter o contention;
4. mostrare se la conclusione cambia includendolo.

### 18.4 Disegno sperimentale

Per confrontare A e B:

- stessa macchina e allocazione;
- stessi input e parametri;
- stesso binding;
- ordine alternato o randomizzato per ridurre drift temporale;
- compilazioni documentate;
- nessuna diagnostica diversa fra le varianti.

---

## 19. Costo della diagnostica energetica

| energy_every | Tempo (s) | Frazione energia | Overhead |
|---:|---:|---:|---:|
| 1 | 107.530 | 47.6% | 84.2% |
| 5 | 67.393 | 16.4% | 15.4% |
| 10 | 62.410 | 9.7% | 6.9% |
| 50 | 58.391 | 3.4% | baseline |

L'energia potenziale e anch'essa `O(N^2)`. Calcolarla a ogni step quasi raddoppia il tempo. Il benchmark prestazionale deve usare controlli abbastanza radi, ma la validazione deve restare sufficiente a catturare la massima deriva.

Criticita: campionare raramente puo perdere un picco di errore. Una strategia migliore separa:

- run di validazione con energia frequente;
- run di performance con diagnostica rara o esclusa dal timer;
- stessi parametri fisici e verifica finale.

---

## 20. Container Singularity e host MPI

### 20.1 Build e runtime sono diversi

Il container include OpenMPI per compilare. Sul cluster, pero, l'eseguibile deve usare l'MPI dell'host, compatibile con launcher, rete e scheduler. `ldd` dentro e fuori conferma il caricamento delle stesse librerie host.

### 20.2 Target di compilazione

Native:

```text
-O3 -march=native
```

Container:

```text
-O3 -march=x86-64-v3
```

Quindi il confronto misura insieme:

- overhead del runtime/container;
- possibile penalita del target portabile;
- differenze di libreria/loader;
- startup e mount.

Chiamarlo “puro overhead Singularity” sarebbe troppo forte. Per isolarlo, bisogna compilare native e container con lo stesso target e, separatamente, confrontare i target ISA.

### 20.3 Risultati solver

L'overhead e circa `2.93-3.44%` nelle sei configurazioni strong/weak a 1, 2 e 4 rank. E compatibile con l'intervallo atteso dalla consegna, ma il dataset e limitato a pochi rank single-node.

### 20.4 Launch overhead

Dieci lanci di `singularity exec ... true` richiedono circa `0.09-0.11 s`. E un costo fisso rilevante per programmi corti e trascurabile per run lunghi.

### 20.5 OSU

Latenza e bandwidth native/container sono molto vicine. Differenze positive e negative sotto pochi punti percentuali indicano assenza di un degrado sistematico evidente, non che il container “migliori” la rete.

### 20.6 Criticita da riconoscere

Nel run finale sono forzati `ob1` e `tcp` e disabilitato il meccanismo vader single-copy. Questo rende il setup robusto, ma puo non rappresentare il path MPI piu performante del cluster. Prima di generalizzare a multi-node bisogna verificare il trasporto site-recommended e la compatibilita ABI/PMIx.

---

## 21. Hardware e riproducibilita

Macchina principale:

| Voce | Valore |
|---|---|
| CPU | AMD EPYC 9374F, 32 core/socket |
| Socket | 2 |
| Core totali | 64 |
| SMT | 1 hardware thread/core |
| NUMA | 8 domini, 8 CPU ciascuno |
| RAM | 503 GiB |
| GCC | 14.3.1 |
| Open MPI | 4.1.6rc4 |
| Singularity CE | 4.3.1 |

Perche non mescolare nodi/CPU diverse nella stessa curva: cambiando architettura cambiano frequenza, cache, NUMA, compilazione e rete. Non si saprebbe piu se la curva misura scaling o eterogeneita hardware.

Limite attuale: il dataset principale e single-node. La comunicazione ring non e stata stressata come in un vero scaling multi-node. Questa limitazione va dichiarata apertamente.

---

## 22. Risultati insoddisfacenti: come presentarli bene

### 22.1 SoA piu lento

- Osservazione: SoA perde circa 11-12%.
- Ipotesi: il compilatore non sfrutta la vettorizzazione attesa o i kernel non sono perfettamente equivalenti.
- Evidenza mancante: report SIMD/counter/assembly.
- Azione: rendere i kernel isomorfi, aggiungere qualifiers e misurare.
- Decisione: non rivendicare SoA come ottimizzazione riuscita.

### 22.2 `rsqrt` approssimata piu lenta

- Osservazione: `+23.6%` di tempo.
- Ipotesi: conversioni e raffinamento superano il risparmio.
- Evidenza mancante: sequenza di istruzioni e precisione effettiva.
- Azione: microbenchmark e assembly.
- Decisione: mantenere `exact` nella produzione.

### 22.3 Overlap senza vantaggio

- Osservazione: differenza piu piccola della variabilita.
- Ipotesi: comunicazione gia piccola o progresso MPI insufficiente.
- Azione: multi-node, messaggi maggiori, timeline e test di progresso.
- Decisione: non dichiarare speedup; dire “nessuna differenza dimostrata”.

### 22.4 Container al 3%

- Osservazione: overhead stabile circa 3%.
- Ipotesi: target ISA portabile piu costi fissi/runtime.
- Problema: target di compilazione non uguale, quindi causalita non isolata.
- Azione: matrice 2x2 native/container per `native` e `x86-64-v3`.

### 22.5 Weak time crescente

- Osservazione: il tempo cresce quasi linearmente con `P`.
- Non e un fallimento di implementazione: deriva dall'interazione globale.
- Azione comunicativa: mostrare la derivazione `P*n^2` e il tempo normalizzato.
- Miglioramento algoritmico: un metodo gerarchico cambierebbe complessita, ma e fuori consegna.

---

## 23. Matrice ipotesi-esperimento

| Ipotesi | Esperimento che la verifica | Metrica | Esito atteso |
|---|---|---|---|
| Il kernel e compute-bound | roofline/counter + scaling kernel | FLOP/s, BW | FLOP vicino al limite sostenibile, BW non saturo |
| Il ring non domina single-node | tempi sezionati | comm/total | quota piccola |
| Overlap nasconde MPI | sendrecv vs overlap multi-node | wall, comm wait | gap oltre rumore |
| SoA aiuta SIMD | report vec + counter | vector width, Gpairs/s | piu SIMD e throughput |
| Newton conviene | sweep N,T e strategia update | tempo, memoria | crossover misurabile |
| `rsqrt` conserva accuratezza | confronto energia/error norms | drift, L2 | errore sotto soglia |
| Binding conta | spread/close/no bind | tempo, variance | binding stabile e piu rapido |
| Container rete e near-native | OSU e solver | latency, BW, wall | nessun gap sistematico grande |

---

## 24. Domande orali con risposta ragionata

### “Perche `O(N^2)`?”

Perche per ciascuno degli `N` target si visitano tutte le `N` sorgenti. La consegna impone il metodo diretto per studiare il kernel regolare e la parallelizzazione, non perche sia il miglior algoritmo fisico a grande `N`.

### “Perche il ring?”

Evita la replica globale permanente, mantiene memoria per rank proporzionale a `N/P` e offre comunicazione regolare. Il prezzo sono `P-1` fasi e una latenza che emergera a grande `P`.

### “Dove sono le race condition?”

Nel direct kernel non ce ne sono perche ogni thread possiede un target. Comparirebbero usando Newton e aggiornando contemporaneamente entrambi gli estremi della coppia. In MPI, inoltre, uno dei due estremi puo essere remoto.

### “Come sai che e corretto?”

Uso tre livelli: confronto con riferimento seriale su casi piccoli, checksum/norme fra varianti di kernel e deriva energetica sulla traiettoria. Nessuno dei tre, isolatamente, prova tutto.

### “Perche l'energia costa cosi tanto?”

La parte potenziale visita tutte le coppie ed e `O(N^2)`, come la forza. Misurarla a ogni step aggiunge quasi un secondo kernel all-pairs.

### “Perche non hai ottenuto il 50% con Newton?”

Si dimezza il numero teorico di coppie, non tutti i costi. Restano aggiornamenti doppi, buffer privati, riduzione, memoria, dipendenze e overhead. In distribuito si aggiungerebbe anche il ritorno dei contributi remoti.

### “Il 90.5% a 64 core e buono?”

Si, nel regime single-node misurato, perche la curva resta vicina all'ideale e il force kernel domina. Non prova che il codice scalera altrettanto su piu nodi.

### “Qual e il bottleneck?”

Nel punto strong a 64 rank, force e l'86.6% del totale e comm wait l'1.3%. Quindi il kernel di forza. Per distinguere compute-bound da instruction/latency/memory-bound servirebbero counter o roofline.

### “Perche il weak scaling non e piatto?”

Fissare `N/P=n` non fissa il lavoro per rank: ciascuno dei suoi `n` target interagisce con `P*n` sorgenti, quindi il lavoro locale cresce come `P*n^2`.

### “Perche l'overlap non migliora?”

La comunicazione e gia una quota minima e la differenza misurata e inferiore alla deviazione. Inoltre MPI potrebbe non garantire progresso asincrono sufficiente. Serve un test multi-node prima di generalizzare.

### “Perche SoA e peggiore?”

So soltanto che nel kernel misurato e peggiore e che i risultati numerici coincidono. La causa e ancora un'ipotesi finche non controllo report di vettorizzazione, assembly e counter. Questo e il modo corretto di delimitare la conclusione.

### “Cosa miglioreresti per primo?”

Prima acquisirei evidenza sul kernel dominante: report SIMD e contatori. Poi proverei accumulatori multipli e qualifiers/allineamento. E una priorita motivata dall'86.6% di tempo nella forza; ottimizzare I/O o MPI single-node avrebbe un limite di beneficio molto basso.

### “Come separeresti overhead container e ISA?”

Con quattro casi sulla stessa macchina: native/x86-64-v3, native/march-native, container/x86-64-v3, container/march-native. Il confronto a target uguale isola meglio il runtime; quello tra target uguali in ambiente uguale isola l'ISA.

---

## 25. Passeggiata guidata nel codice

Quando studi `nbody_direct_hybrid.c`, segui quest'ordine:

1. `particles_t`: capisci quali array esistono e chi li possiede.
2. enum integrator/comm/kernel/rsqrt: collega ogni opzione all'esperimento.
3. parsing: identifica default e controlli di validita.
4. `particles_allocate`: allineamento, inizializzazione e gestione errori.
5. `block_bounds`: dimostra copertura completa e assenza di sovrapposizioni.
6. `read_local_particles`: capisci offset e formato binario.
7. `drift` e `kick`: spiega perche `schedule(static)` e sicuro.
8. `accumulate_sources`: ricostruisci una singola interazione e il trattamento del self term.
9. `compute_accelerations_newton_private`: individua memoria privata e riduzione.
10. funzioni di exchange: disegna mittente, destinatario, tag e lifetime dei buffer.
11. `compute_accelerations_ring`: dimostra che ogni sorgente e visitata una volta.
12. energia: verifica conteggio delle coppie senza doppio conteggio.
13. `main`: ricostruisci ordine KDK, timing e riduzioni MPI.
14. cleanup: verifica che ogni allocazione venga liberata e MPI finalizzato.

Per ogni funzione devi saper rispondere:

- precondizioni;
- dati letti e scritti;
- complessita;
- parallelismo;
- possibili errori;
- test minimo;
- metrica prestazionale pertinente.

---

## 26. Piano di studio pratico

### Sessione 1 - Fisica e numerica

- derivare forza, softening, energia e KDK;
- spiegare ordine due e simpletticita;
- eseguire mentalmente un sistema a una e due particelle;
- preparare il test di convergenza in `dt`.

### Sessione 2 - Codice seriale

- riscrivere pseudocodice del force kernel;
- calcolare complessita e accessi;
- seguire lettura input, step ed energia;
- identificare i punti da parallelizzare.

### Sessione 3 - MPI

- disegnare un ring a 4 rank fase per fase;
- gestire `N%P != 0`;
- spiegare deadlock e ruolo di Sendrecv/nonblocking;
- confrontare ring e Allgather.

### Sessione 4 - OpenMP e ibrido

- classificare variabili shared/private;
- spiegare assenza di race;
- discutere static scheduling, affinity e NUMA;
- interpretare la tabella `P x T`.

### Sessione 5 - Scaling

- calcolare a mano speedup ed efficienza per 2, 16 e 64 rank;
- derivare il weak scaling specifico;
- spiegare Amdahl senza confondere serial fraction e overhead;
- leggere ogni grafico senza testo preparato.

### Sessione 6 - Ottimizzazioni

- difendere Newton, SoA, rsqrt, overlap e accumulatori multipli;
- per ciascuno separare aspettativa, misura e spiegazione;
- progettare l'esperimento successivo.

### Sessione 7 - Container e riproducibilita

- spiegare build MPI vs host MPI;
- leggere `ldd`;
- spiegare launch, OSU e target ISA;
- dichiarare correttamente cosa il 3% include.

### Sessione 8 - Orale simulato

- presentazione di 5 minuti senza slide;
- 15 minuti di domande sul codice;
- 10 minuti di critica ai risultati;
- chiusura con tre miglioramenti prioritari.

---

## 27. Checklist prima dell'orale

### Comprensione

- [ ] So derivare il costo `O(N^2)`.
- [ ] So scrivere KDK e spiegare il softening.
- [ ] So disegnare il ring e provarne la correttezza.
- [ ] So classificare le variabili OpenMP.
- [ ] So spiegare ogni opzione importante del programma.

### Risultati

- [ ] Ricordo `57.94x` e `90.5%` a 64 rank.
- [ ] Ricordo `86.6%` force e `1.3%` comm wait.
- [ ] So derivare il riferimento weak `O(P)`.
- [ ] So dire perche `32x2` non e un vincitore forte.
- [ ] Ricordo gli esiti di Newton, rsqrt, AoS/SoA e overlap.
- [ ] So spiegare il circa 3% container con le sue ambiguita.

### Metodo scientifico

- [ ] Distinguo osservazione, ipotesi e verifica.
- [ ] Non confondo correlazione e causa.
- [ ] Non chiamo significativa una differenza sotto il rumore.
- [ ] Dichiaro i limiti single-node.
- [ ] Per ogni difetto propongo un esperimento concreto.

### Codice

- [ ] Posso ricostruire il flusso di `main`.
- [ ] Posso spiegare ownership e lifetime dei buffer.
- [ ] Posso indicare dove nascerebbero race/deadlock.
- [ ] Posso giustificare timer e riduzioni MPI.
- [ ] Posso descrivere un test per ogni funzione critica.

---

## 28. Scheda finale da usare per qualsiasi nuova osservazione

```text
Fenomeno osservato:
Dato quantitativo e incertezza:
Configurazione esatta:
Aspettativa teorica:
La misura conferma l'aspettativa? Perche?
Ipotesi causali alternative:
Quale evidenza separa le ipotesi?
Limite della misura attuale:
Esperimento successivo:
Decisione pratica sul codice:
```

Se sai compilare questa scheda senza inventare spiegazioni, stai mostrando esattamente la competenza richiesta: non soltanto ottenere un risultato, ma comprenderlo, criticarlo e sapere come migliorarlo.
