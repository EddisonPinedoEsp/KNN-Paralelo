from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

def euclidean_distance(x1, x2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict_parallel(X_train, y_train, X_test_chunk, k, rank, size, comm):
    """Predicción KNN paralela para un chunk de datos de prueba"""
    predictions = []
    
    for test_point in X_test_chunk:
        # Calcular distancias a todos los puntos de entrenamiento
        distances = []
        for i, train_point in enumerate(X_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))
        
        # Ordenar por distancia y tomar los k más cercanos
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        # Votación mayoritaria
        labels = [label for _, label in k_nearest]
        prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)
    
    return np.array(predictions)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parámetros
    k = 5
    test_size = 0.2
    random_state = 42
    
    # Variables para almacenar datos
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    X_test_chunks = None
    
    # El proceso 0 carga y prepara los datos
    if rank == 0:
        print(f"Ejecutando KNN paralelo con {size} procesos")
        print("Cargando dataset...")
        
        # Cargar datos
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Datos de entrenamiento: {X_train.shape}")
        print(f"Datos de prueba: {X_test.shape}")
        print(f"Usando k = {k}")
        
        # Dividir datos de prueba entre procesos
        n_test = len(X_test)
        chunk_size = n_test // size
        remainder = n_test % size
        
        X_test_chunks = []
        start_idx = 0
        
        for i in range(size):
            # Distribuir el resto entre los primeros procesos
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            X_test_chunks.append(X_test[start_idx:end_idx])
            start_idx = end_idx
        
        print(f"Dividiendo {n_test} muestras de prueba entre {size} procesos")
        for i, chunk in enumerate(X_test_chunks):
            print(f"Proceso {i}: {len(chunk)} muestras")
    
    # Broadcast de datos de entrenamiento a todos los procesos
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    
    # Scatter de chunks de datos de prueba
    X_test_chunk = comm.scatter(X_test_chunks, root=0)
    
    if rank == 0:
        start_time = time.time()
        print("\nIniciando predicción paralela...")
    
    # Cada proceso procesa su chunk
    local_predictions = knn_predict_parallel(X_train, y_train, X_test_chunk, k, rank, size, comm)
    
    # Gather todas las predicciones en el proceso 0
    all_predictions = comm.gather(local_predictions, root=0)
    
    # El proceso 0 evalúa los resultados
    if rank == 0:
        end_time = time.time()
        
        # Concatenar todas las predicciones
        y_pred = np.concatenate(all_predictions)
        
        # Evaluar precisión
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTiempo de ejecución: {end_time - start_time:.4f} segundos")
        print(f"Precisión: {accuracy:.4f}")
        
        # Reporte detallado
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()