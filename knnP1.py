from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Inicialización en el proceso maestro
    if rank == 0:
        # Cargar y dividir los datos
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        print(f"Dataset loaded: {len(X_test)} test samples")
        print(f"Using {size} processes")
        
        # Dividir datos de prueba entre procesos
        chunk_size = len(X_test) // size
        chunks = [X_test[i:i + chunk_size] for i in range(0, len(X_test), chunk_size)]
        
        # Ajustar si hay residuo
        if len(chunks) > size:
            chunks[-2].extend(chunks[-1])
            chunks = chunks[:-1]
        
        start_time = time.time()
    else:
        X_train = None
        y_train = None
        chunks = None
        y_test = None
        start_time = None
    
    # Broadcast de datos de entrenamiento
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    k = comm.bcast(3, root=0)  # Parámetro k
    
    # Scatter: distribuir chunks de datos de prueba
    local_X_test = comm.scatter(chunks, root=0)
    
    # Procesamiento local
    local_predictions = []
    if local_X_test is not None:
        for test_point in local_X_test:
            pred = knn_predict(test_point, X_train, y_train, k)
            local_predictions.append(pred)
    
    # Gather: recolectar predicciones
    all_predictions = comm.gather(local_predictions, root=0)
    
    # Evaluación en el proceso maestro
    if rank == 0:
        # Flatten predicciones
        y_pred = []
        for pred_chunk in all_predictions:
            y_pred.extend(pred_chunk)
        
        accuracy = np.mean(np.array(y_pred) == y_test)
        end_time = time.time()
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Execution time (MPI v1): {end_time - start_time:.4f} sec")
        print(f"Total predictions: {len(y_pred)}")

if __name__ == "__main__":
    main()