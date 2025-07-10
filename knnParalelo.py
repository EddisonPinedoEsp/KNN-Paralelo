from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(distances, labels, k):
    k_labels = labels[:k]
    label_counts = Counter(k_labels)
    most_common = label_counts.most_common(1)[0][0]
    
    return most_common

def process_calc(test_point, pX_train, pY_train, k):
    data = []
    
    for i, train_point in enumerate(pX_train):
        distance = euclidean_distance(test_point, train_point)
        label = pY_train[i]
        data.append((distance, label))
    
    data.sort(key=lambda x: x[0])
    
    k_nearest = data[:k]
    
    k_distances = np.array([x[0] for x in k_nearest])
    k_labels = np.array([x[1] for x in k_nearest])
    
    return k_distances, k_labels

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    k = 3
    
    if rank == 0: # Maestro
        print(f"=== KNN Paralelo ===")
        print(f"Procesos: {size}, k = {k}")
        
        # Carga datos de entrenamiento
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        print(f"Entrenamiento: {X_train.shape}")
        print(f"Prueba: {X_test.shape}")
        
        # Ajustar el dataset para que sea divisible por el número de procesos
        X_train_size = len(X_train)
        remainder = X_train_size % size
        
        if remainder != 0:
            new_size = X_train_size - remainder
            X_train = X_train[:new_size]
            y_train = y_train[:new_size]
    
    # Enviar los datos de prueba a todos los procesos
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)
    k = comm.bcast(k, root=0)
    
    if rank == 0:
        nrows = len(X_train) // size
        
        # Dividir la data en particiones
        partition_X = []
        partition_Y = []
        for i in range(size):
            start = i * nrows
            end = (i + 1) * nrows
            partition_X.append(X_train[start:end])
            partition_Y.append(y_train[start:end])
    else:
        partition_X = None
        partition_Y = None
    
    # Enviar a cada proceso una particion
    pX_train = comm.scatter(partition_X, root=0)
    pY_train = comm.scatter(partition_Y, root=0)
    
    comm.barrier()
    start_time = time.time()
    
    predictions = []
    
    # Enviar la cantidad de puntos de testeo a cada proceso
    num_test_points = comm.bcast(len(X_test), root=0)
    
    for i in range(num_test_points):
        test_point = X_test[i]
        
        # Cada proceso calcula distancias, ordena y selecciona k-nearest
        pk_distances, pk_labels = process_calc(
            test_point, pX_train, pY_train, k
        )
        
        # Recibir solo los k vecinos más cercanos de cada proceso
        k_distances = comm.gather(pk_distances, root=0)
        k_labels = comm.gather(pk_labels, root=0)
        
        if rank == 0:
            # Combinar los k vecinos más cercanos de todos los procesos
            k_neighbors = []
            
            for pdist, plabels in zip(k_distances, k_labels):
                for d, l in zip(pdist, plabels):
                    k_neighbors.append((d, l))
            
            # Ordenar todos los k*p vecinos y tomar los k mejores globales
            k_neighbors.sort(key=lambda x: x[0])
            
            global_k_neighbors = k_neighbors[:k]
            final_distances = [x[0] for x in global_k_neighbors]
            final_labels = [x[1] for x in global_k_neighbors]
            
            # Prediccion Final
            prediction = knn_predict(
                final_distances, final_labels, k
            )
            
            predictions.append(prediction)
            
    
    comm.barrier()
    
    # Analisis de Resultados
    if rank == 0:
        end_time = time.time()
        
        accuracy = np.mean(np.array(predictions) == y_test)
        
        print(f"\n=== RESULTADOS FINALES ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Execution time: {end_time - start_time:.4f} sec")
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        for i, ax in enumerate(axes.flat):
            if i < min(10, num_test_points):
                ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
                ax.set_title(f"Pred: {predictions[i]}\nTrue: {y_test[i]}")
                ax.axis('off')
        plt.suptitle(f"KNN Paralelo - k={k}, {size} procesos")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
