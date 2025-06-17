# KNN Paralelo con MPI

Este proyecto implementa una versión paralela del algoritmo K-Nearest Neighbors (KNN) utilizando MPI (Message Passing Interface) con la biblioteca `mpi4py` en Python. El objetivo es distribuir el cálculo de las predicciones de KNN entre múltiples procesos para acelerar el rendimiento en comparación con una implementación secuencial.

## Requisitos

Asegúrate de tener instalados los siguientes componentes:

1.  **Python 3**: (https://www.python.org/)
2.  **Una implementación de MPI**:
    *   MPI for Python (https://mpi4py.readthedocs.io/en/stable/index.html)
3.  **Bibliotecas de Python**:
    *   `mpi4py`
    *   `numpy`
    *   `scikit-learn`
    *   `matplotlib`

Puedes instalar las bibliotecas de Python usando pip (preferiblemente dentro de un entorno virtual):

```bash
python -m pip install mpi4py
```

```bash
pip install numpy scikit-learn matplotlib
```

## Cómo Ejecutar el Código en Paralelo

El script principal para la ejecución paralela es `knnParalelo.py`. Para ejecutarlo, utilizarás el comando `mpiexec` (o `mpirun`, dependiendo de tu implementación de MPI y configuración).

**Comando General:**

```bash
mpiexec -n <numero_de_procesos> python knnParalelo.py
```

**Donde:** 

- `<numero_de_procesos>` es el número de procesos que deseas utilizar para la ejecución paralela. Por ejemplo, si quieres usar 4 procesos, el comando sería:

**Ejemplo:**

```bash
mpiexec -n 4 python knnParalelo.py
```
