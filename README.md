# Progetto di Sistemi Digitali A.A. 2024/25

Questo repository contiene il codice per il mio progetto relativo all'esame del corso di Sistemi Digitali del Corso di Laurea Magistrale in Ingegneria Informatica dell'Università degli Studi di Bologna.

## Descrizione

Il progetto consiste in varie implementazioni dell'algoritmo di clustering K-Means in linguaggio C++ su vari target che riguardano quanto studiato durante il corso.

## Struttura del repository

Il repository è strutturato come segue:

- `src/`: contiene il codice delle varie implementazioni dell'algoritmo K-Means
- `test/`: contiene il codice che esegue i test sulle varie implementazioni

## Build

Il progetto utilizza CMake come sistema di build. Per compilare il progetto è sufficiente eseguire i seguenti comandi:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

Sono disponibili diverse variabili di CMake per configurare il progetto:

| Variabile     | Descrizione                       | Valore predefinito | Dipendenze                      |
| ------------- | --------------------------------- | ------------------ | ------------------------------- |
| KMEANS_USM    | Usa l'implementazione SYCL USM    | ON                 | Compilatore con supporto a SYCL |
| KMEANS_BUFFER | Usa l'implementazione SYCL buffer | ON                 | Compilatore con supporto a SYCL |
| KMEANS_OCV    | Usa l'implementazione OpenCV      | ON                 | OpenCV                          |
| KMEANS_SIMD   | Usa l'implementazione x86 SIMD    | ON                 | SSE4                            |
| KMEANS_TBB    | Usa l'implementazione TBB         | ON                 | Thread Building Blocks          |
| KMEANS_CUDA   | Usa l'implementazione CUDA        | OFF                | CUDA                            |
| KMEANS_ONEAPI | Usa l'implementazione Intel SYCL  | OFF                | Compilatore DPC++               |

Per esempio, per compilare il progetto senza l'implementazione SYCL USM è sufficiente eseguire:

```bash
cmake -B build -DKMEANS_USM=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```
