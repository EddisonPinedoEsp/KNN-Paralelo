from mpi4py import MPI
from sklearn.datasets import load_digits, fetch_openml
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

def load_dataset(dataset_type, dataset_fraction=1.0):
    
    if dataset_type == "digits":
        digits = load_digits()
        X, y = digits.data, digits.target
        
    elif dataset_type == "mnist":
        print("Descargando MNIST... (puede tomar tiempo la primera vez)")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        
        X = mnist.data.values if hasattr(mnist.data, 'values') else np.array(mnist.data)
        y = mnist.target.values if hasattr(mnist.target, 'values') else np.array(mnist.target)
        
        y = y.astype(int)
        X = X / 255.0 
        
    else:
        raise ValueError("dataset_type debe ser 'digits' o 'mnist'")
    
    if dataset_fraction < 1.0:
        total_samples = len(X)
        subset_size = int(total_samples * dataset_fraction)
        X = X[:subset_size]
        y = y[:subset_size]
        print(f"Usando {dataset_fraction*100:.1f}% del dataset: {subset_size} de {total_samples} muestras")
        
    return X, y

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1]  
    else:
        dataset_type = "digits"
        
    if len(sys.argv) > 2:
        dataset_fraction = float(sys.argv[2]) 
        if dataset_fraction <= 0 or dataset_fraction > 1:
            if rank == 0:
                print("ERROR: El porcentaje debe estar entre 0.0 y 1.0")
            return
    else:
        dataset_fraction = 1.0
    
    k = 3
    
    if rank == 0: 
        # Cargar dataset seleccionado
        X, y = load_dataset(dataset_type, dataset_fraction)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
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
    
    # Comunicación inicial
    comm_start = time.time()
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)
    k = comm.bcast(k, root=0)
    
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
    comm_time = time.time() - comm_start
    
    comm.barrier()
    total_start = time.time()
    
    predictions = []
    num_test_points = len(X_test)
    comp_time = 0
    gather_time = 0
    total_flops = 0
    
    for i in range(num_test_points):
        test_point = X_test[i]
        
        comp_start = time.time()
        pk_distances, pk_labels, local_flops = process_calc(test_point, pX_train, pY_train, k)
        comp_time += time.time() - comp_start
        total_flops += local_flops
        
        gather_start = time.time()
        k_distances = comm.gather(pk_distances, root=0)
        k_labels = comm.gather(pk_labels, root=0)
        gather_time += time.time() - gather_start
        
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
            comp_time += time.time() - comp_start
    
    comm.barrier()
    total_time = time.time() - total_start
    
    if rank == 0:
        accuracy = np.mean(np.array(predictions) == y_test[:num_test_points])
        total_comm_time = comm_time + gather_time
        samples = len(pX_train) * size
        
        flops_per_second = total_flops / comp_time
        gflops_per_second = flops_per_second / 1e9
        
        print(f"\n=== RESULTADOS ===")
        print(f"Dataset: {dataset_type}")
        print(f"Procesos: {size}")
        print(f"Muestras entrenamiento total: {samples:,}")
        print(f"")
        print(f"=== TIEMPOS ===")
        print(f"Tiempo de Ejecucion: {total_time:.4f}s")
        print(f"  - Tiempo cómputo: {comp_time:.4f}s ({100*comp_time/total_time:.1f}%)")
        print(f"  - Tiempo comunicación: {total_comm_time:.4f}s ({100*total_comm_time/total_time:.1f}%)")
        print(f"")
        print(f"=== METRICAS DE FLOPs ===")
        print(f"Total FLOPs: {total_flops:,}")
        print(f"FLOPS/segundo: {flops_per_second:.2e}")
        print(f"GFLOPS/segundo: {gflops_per_second:.3f}")
        print(f"")
        print(f"=== MÉTRICAS ===")
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
