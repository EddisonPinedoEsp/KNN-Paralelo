# KNN Paralelo con MPI

Este proyecto implementa una versión **paralela** del algoritmo **K-Nearest Neighbors (KNN)** utilizando **MPI (Message Passing Interface)** con la biblioteca [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/). El objetivo es **distribuir el cálculo de las distancias** y la predicción de clases entre múltiples procesos para mejorar el rendimiento en comparación con una implementación secuencial.

---

## 📂 Descripción General

El script `knnParalelo.py` permite:
- Cargar datasets **Digits** o **MNIST**.
- Dividir el dataset entre procesos MPI.
- Calcular distancias Euclidianas de forma distribuida.
- Combinar resultados en el proceso maestro para predecir usando votación mayoritaria.
- Medir precisión, tiempos de cómputo y métricas de rendimiento (**FLOPs**, **GFLOPS**).

---

## ⚙️ Requisitos

Asegúrate de tener instalados:

- **Python 3**  
  👉 [Descargar](https://www.python.org/)

- **Implementación de MPI**  
  Ejemplo: [Open MPI](https://www.open-mpi.org/) o [MPICH](https://www.mpich.org/)

- **Bibliotecas de Python**:
  - `mpi4py`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` (opcional, si deseas graficar resultados)

Instala las dependencias con `pip`:

```bash
pip install mpi4py numpy scikit-learn matplotlib
```

---

## ▶️ Cómo Ejecutar el Script

Ejecuta el archivo `knnParalelo.py` usando `mpiexec` o `mpirun`:

```bash
mpiexec -n <numero_de_procesos> python knnParalelo.py [dataset] [fracción]
```

**Argumentos opcionales:**
- `dataset` → `digits` o `mnist` (por defecto: `digits`)
- `fracción` → número entre 0.0 y 1.0 para definir qué porcentaje del dataset usar. Ejemplo: `0.1` para usar 10 % del dataset.

---

### 📌 Ejemplos

**Ejecutar usando 4 procesos con el dataset Digits (100 %):**
```bash
mpiexec -n 4 python knnParalelo.py
```

**Ejecutar usando 8 procesos con MNIST y solo el 10 % de los datos:**
```bash
mpiexec -n 8 python knnParalelo.py mnist 0.1
```

---

## 🔬 Detalles Técnicos

- Cada proceso recibe una **partición** de los datos de entrenamiento.
- Cada punto de prueba se evalúa calculando la **distancia Euclidiana** a todos los puntos de entrenamiento de su partición.
- Los `k` vecinos más cercanos se envían al **proceso maestro**, que combina resultados, selecciona los `k` vecinos más cercanos globales y decide la clase final por **votación mayoritaria**.
- El script mide:
  - Exactitud (**accuracy**)
  - Tiempo total, de cómputo y de comunicación.
  - **FLOPs** y **GFLOPS/segundo** estimados.

---

## 📑 Archivos Clave

- `knnParalelo.py` — Script principal para correr el algoritmo paralelo.

---

## 📖 Referencias

- [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/)
- [Scikit-learn](https://scikit-learn.org/stable/)

---

## 🏷️ Licencia

Este proyecto es de uso académico. Modifícalo y experiméntalo libremente para entender y mejorar algoritmos paralelos.
