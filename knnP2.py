from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict_optimized(test_point, X_train, y_train, k):
    # Optimización: usar numpy para cálculos vectorizados
    distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
    k_indices = np.argpartition(distances, k)[:k]
    k_labels = y_train[k_indices]
    unique, counts = np.unique(k_labels, return_counts=True)
    return unique[np.argmax(counts)]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parámetros iniciales
    k = 3
    
    if rank == 0:
        # Cargar datos
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        print(f"Dataset loaded: {len(X_test)} test samples")
        print(f"Training set size: {len(X_train)}")
        print(f"Using {size} processes")
        
        # Mejor distribución de trabajo
        total_samples = len(X_test)
        samples_per_process = total_samples // size
        remainder = total_samples % size
        
        # Crear índices para cada proceso
        indices = []
        start_idx = 0
        for i in range(size):
            end_idx = start_idx + samples_per_process + (1 if i < remainder else 0)
            indices.append((start_idx, end_idx))
            start_idx = end_idx
        
        start_time = time.time()
    else:
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        indices = None
        start_time = None
    
    # Broadcast: distribuir datos de entrenamiento y parámetros
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    k = comm.bcast(k, root=0)
    
    # Scatter: distribuir índices de trabajo
    local_indices = comm.scatter(indices, root=0)
    
    # Procesamiento local con mejor distribución
    local_predictions = []
    if local_indices:
        start_idx, end_idx = local_indices
        local_X_test = X_test[start_idx:end_idx]
        
        for test_point in local_X_test:
            pred = knn_predict_optimized(test_point, X_train, y_train, k)
            local_predictions.append(pred)
        
        print(f"Process {rank}: processed {len(local_predictions)} samples")
    
    # Gather: recolectar predicciones con orden preservado
    all_predictions = comm.gather(local_predictions, root=0)
    
    if rank == 0:
        # Reconstruir predicciones en orden correcto
        y_pred = []
        for pred_chunk in all_predictions:
            y_pred.extend(pred_chunk)
        
        accuracy = np.mean(np.array(y_pred) == y_test)
        end_time = time.time()
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Execution time (MPI v2): {end_time - start_time:.4f} sec")
        print(f"Total predictions: {len(y_pred)}")

if __name__ == "__main__":
    main()