from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time
import sys

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(distances, labels, k):
    k_labels = labels[:k]
    label_counts = Counter(k_labels)
    most_common = label_counts.most_common(1)[0][0]
    return most_common

def process_calc(test_point, pX_train, pY_train, k):
    data = []
    flop_count = 0 
    
    for i, train_point in enumerate(pX_train):
        distance = euclidean_distance(test_point, train_point)
        label = pY_train[i]
        data.append((distance, label))
        
        # 64 restas + 64 multiplicaciones + 63 sumas + 1 sqrt = 192 FLOPs
        flop_count += len(test_point) * 3 + 1  
    
    data.sort(key=lambda x: x[0])
    k_nearest = data[:k]
    
    k_distances = np.array([x[0] for x in k_nearest])
    k_labels = np.array([x[1] for x in k_nearest])
    
    return k_distances, k_labels, flop_count

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    k = 3
    
    communication_time = 0
    computation_time = 0
    total_flops = 0 
    
    if rank == 0:
        print(f"=== KNN PARALELO ===")
        print(f"Procesos: {size}, k = {k}")
        
        # Cargar datos
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        
        # Ajustar para divisibilidad
        train_size = len(X_train)
        remainder = train_size % size
        if remainder != 0:
            new_size = train_size - remainder
            X_train = X_train[:new_size]
            y_train = y_train[:new_size]
    else:
        X_train = None
        y_train = None
        X_test = None
        y_test = None
    
    # === COMUNICACIÓN ===
    comm.barrier()
    comm_start = time.time()
    
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)
    k = comm.bcast(k, root=0)
    
    communication_time += time.time() - comm_start
    
    # === COMUNICACIÓN ===
    comm_start = time.time()
    
    if rank == 0:
        nrows = len(X_train) // size
        partition_X = []
        partition_Y = []
        for i in range(size):
            start_idx = i * nrows
            end_idx = (i + 1) * nrows
            partition_X.append(X_train[start_idx:end_idx])
            partition_Y.append(y_train[start_idx:end_idx])
    else:
        partition_X = None
        partition_Y = None
    
    pX_train = comm.scatter(partition_X, root=0)
    pY_train = comm.scatter(partition_Y, root=0)
    
    communication_time += time.time() - comm_start
    
    if rank == 0:
        print(f"Partición de {len(pX_train)} muestras de entrenamiento")
    
    comm.barrier()
    total_start = time.time()
    
    predictions = []
    num_test_points = len(X_test)
    
    
    for i in range(num_test_points):
        test_point = X_test[i]
        
        # Cómputo local
        comp_start = time.time()
        pk_distances, pk_labels, local_flops = process_calc(test_point, pX_train, pY_train, k)
        computation_time += time.time() - comp_start
        total_flops += local_flops
        
        # Comunicación
        comm_start = time.time()
        k_distances = comm.gather(pk_distances, root=0)
        k_labels = comm.gather(pk_labels, root=0)
        communication_time += time.time() - comm_start
        
        # Cómputo
        if rank == 0:
            comp_start = time.time()
            k_neighbors = []
            for pdist, plabels in zip(k_distances, k_labels):
                for d, l in zip(pdist, plabels):
                    k_neighbors.append((d, l))
            
            k_neighbors.sort(key=lambda x: x[0])
            global_k_neighbors = k_neighbors[:k]
            final_distances = [x[0] for x in global_k_neighbors]
            final_labels = [x[1] for x in global_k_neighbors]
            
            prediction = knn_predict(final_distances, final_labels, k)
            predictions.append(prediction)
            computation_time += time.time() - comp_start
    
    comm.barrier()
    total_time = time.time() - total_start
    
    # === RESULTADOS ===
    if rank == 0:
        accuracy = np.mean(np.array(predictions) == y_test[:num_test_points])
        
        flops_per_second = total_flops / computation_time
        gflops_per_second = flops_per_second / 1e9
        
        print(f"\n=== RESULTADOS ===")
        print(f"Procesos utilizados: {size}")
        print(f"")
        print(f"=== TIEMPOS (segundos) ===")
        print(f"Tiempo de Ejecucion: {total_time:.4f}")
        print(f"  - Computo: {computation_time:.4f} ({100*computation_time/total_time:.1f}%)")
        print(f"  - Comunicacion: {communication_time:.4f} ({100*communication_time/total_time:.1f}%)")
        print(f"")
        print(f"=== METRICAS DE FLOPs ===")
        print(f"Total FLOPs: {total_flops:,}")
        print(f"FLOPS/segundo: {flops_per_second:.2e}")
        print(f"GFLOPS/segundo: {gflops_per_second:.3f}")
        print(f"")
        print(f"=== METRICA DE PRECISION ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"")

if __name__ == "__main__":
    main()
