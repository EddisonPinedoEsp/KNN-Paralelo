from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time
import pickle
import os

def mpi_initialise():
    """Inicializar MPI y obtener rank y size"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def save_model_weights(X_train, y_train, X_test, y_test, model_info, filename="knn_weights"):
    """Guardar los 'pesos' del modelo KNN (datos de entrenamiento y configuración)"""
    
    # Crear directorio si no existe
    weights_dir = "knn_weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # Información del modelo
    model_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'model_info': model_info,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'version': 'knnP4_c_style'
    }
    
    # Guardar en formato pickle
    filepath = os.path.join(weights_dir, f"{filename}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Guardar información legible
    info_filepath = os.path.join(weights_dir, f"{filename}_info.txt")
    with open(info_filepath, 'w') as f:
        f.write("KNN Model Weights Information\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {model_data['timestamp']}\n")
        f.write(f"Version: {model_data['version']}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
        f.write(f"Classes: {len(np.unique(y_train))}\n")
        f.write(f"K parameter: {model_info.get('k', 'N/A')}\n")
        f.write(f"Accuracy achieved: {model_info.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"Execution time: {model_info.get('execution_time', 'N/A'):.4f} sec\n")
        f.write(f"Processes used: {model_info.get('processes', 'N/A')}\n")
        f.write(f"Random state: {model_info.get('random_state', 42)}\n")
    
    return filepath, info_filepath

def init_features_and_labels():
    """Cargar datos como en el código C (equivalente a initFeatures/initLabels)"""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    return X_train.astype(np.float32), X_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32)

def predict(distance, labels, k=3, nclasses=10, verbose=False):
    """Función de predicción equivalente a predict() en C"""
    neighbor_count = np.zeros(nclasses, dtype=np.float32)
    
    # Contar vecinos más cercanos (primeros k)
    for i in range(k):
        neighbor_count[int(labels[i])] += 1
    
    # Calcular probabilidades
    probability = neighbor_count / float(k)
    
    # Obtener clase predicha (índice del máximo)
    predicted_class = np.argmax(neighbor_count)
    
    # Mostrar probabilidades solo si verbose=True
    if verbose:
        print("Probability:")
        for i in range(min(5, nclasses)):  # TOPN equivalente
            print(f"Class {i}\t{probability[i]:.4f}")
    
    return predicted_class

def calc_distance(ndata_per_process, pdata, x, nfeatures):
    """Cálculo de distancias equivalente a calcDistance() en C"""
    nrows_local = ndata_per_process // nfeatures
    pdistance = np.zeros(nrows_local, dtype=np.float32)
    
    index = 0
    for i in range(0, ndata_per_process, nfeatures):
        pdistance[index] = 0.0
        for j in range(nfeatures):
            diff = pdata[i + j] - x[j]
            pdistance[index] += diff * diff
        index += 1
    
    return pdistance

def merge_sort_with_labels(distances, labels):
    """Ordenamiento que mantiene asociación entre distancias y etiquetas"""
    # Crear pares (distancia, etiqueta)
    pairs = list(zip(distances, labels))
    # Ordenar por distancia
    pairs.sort(key=lambda x: x[0])
    # Separar de nuevo
    sorted_distances = np.array([pair[0] for pair in pairs], dtype=np.float32)
    sorted_labels = np.array([pair[1] for pair in pairs], dtype=np.float32)
    return sorted_distances, sorted_labels

def calculate_metrics(y_true, y_pred, nclasses=10):
    """Calcular métricas de evaluación"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Accuracy general
    accuracy = np.mean(y_true == y_pred)
    
    # Accuracy por clase
    class_accuracy = {}
    class_count = {}
    
    for class_id in range(nclasses):
        mask = (y_true == class_id)
        if np.sum(mask) > 0:
            class_accuracy[class_id] = np.mean(y_true[mask] == y_pred[mask])
            class_count[class_id] = np.sum(mask)
        else:
            class_accuracy[class_id] = 0.0
            class_count[class_id] = 0
    
    # Matriz de confusión simple
    confusion_matrix = np.zeros((nclasses, nclasses), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[int(true_label)][int(pred_label)] += 1
    
    return accuracy, class_accuracy, class_count, confusion_matrix

def fit(X_train, y_train, X_test, y_test, comm, rank, size):
    """Función principal de entrenamiento/predicción equivalente a fit() en C"""
    NTRAIN = len(X_train) if X_train is not None else 0
    NTEST = len(X_test)
    NFEATURES = X_train.shape[1] if X_train is not None else X_test.shape[1]
    K = 3  # Parámetro k
    
    # Broadcast de dimensiones desde proceso 0
    NTRAIN = comm.bcast(NTRAIN, root=0)
    NFEATURES = comm.bcast(NFEATURES, root=0)
    K = comm.bcast(K, root=0)
    
    # Verificar divisibilidad (como en C)
    if NTRAIN % size != 0:
        if rank == 0:
            print(f"Number of rows in training dataset ({NTRAIN}) should be divisible by number of processors ({size})")
        MPI.Finalize()
        exit(0)
    
    # Calcular distribución de datos
    nrows_per_process = NTRAIN // size
    ndata_per_process = nrows_per_process * NFEATURES
    
    if rank == 0:
        print(f"Process distribution: {nrows_per_process} training samples per process")
    
    # Preparar arrays locales
    pdata = np.zeros(ndata_per_process, dtype=np.float32)
    plabels = np.zeros(nrows_per_process, dtype=np.float32)
    
    # Arrays para recolectar resultados (solo en proceso 0)
    if rank == 0:
        distance = np.zeros(NTRAIN, dtype=np.float32)
        labels = np.zeros(NTRAIN, dtype=np.float32)
        # Arrays para almacenar todas las predicciones
        all_predictions = []
        all_true_labels = []
    else:
        distance = None
        labels = None
    
    # SCATTER inicial de datos de entrenamiento (UNA SOLA VEZ)
    if rank == 0:
        X_train_flat = X_train.flatten().astype(np.float32)
    else:
        X_train_flat = None
    
    comm.Scatter(X_train_flat, pdata, root=0)
    
    # Bucle principal: UNA ITERACIÓN POR MUESTRA DE PRUEBA (como en C)
    for i in range(NTEST):
        # SCATTER de etiquetas CADA VEZ (como en C - muy importante!)
        comm.Scatter(y_train, plabels, root=0)
        
        # Obtener muestra de prueba actual
        x = X_test[i].astype(np.float32)
        
        # Calcular distancias locales
        pdistance = calc_distance(ndata_per_process, pdata, x, NFEATURES)
        
        # Ordenar distancias locales con sus etiquetas
        pdistance, plabels = merge_sort_with_labels(pdistance, plabels)
        
        # GATHER de distancias y etiquetas ordenadas
        comm.Gather(pdistance, distance, root=0)
        comm.Gather(plabels, labels, root=0)
        
        # Predicción final en proceso 0
        if rank == 0:
            # Ordenamiento final global
            distance, labels = merge_sort_with_labels(distance, labels)
            
            # Hacer predicción (sin verbose para no saturar output)
            predicted_class = predict(distance, labels, k=K, verbose=False)
            true_class = int(y_test[i])
            
            # Almacenar para métricas
            all_predictions.append(predicted_class)
            all_true_labels.append(true_class)
            
            # Mostrar progreso cada 50 muestras
            if (i + 1) % 50 == 0 or i == 0:
                print(f"Processed {i+1}/{NTEST} samples... Current: Pred={predicted_class}, True={true_class}")
    
    # Calcular y mostrar métricas finales - SOLO EN PROCESO 0
    if rank == 0:
        print(f"\n" + "="*60)
        print(f"           FINAL RESULTS - KNN MPI (C-Style)")
        print(f"="*60)
        
        accuracy, class_accuracy, class_count, confusion_matrix = calculate_metrics(
            all_true_labels, all_predictions
        )
        
        # IMPRIMIR ACCURACY PRINCIPAL DE FORMA PROMINENTE
        print(f"\n🎯 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Correct predictions: {int(accuracy * len(all_predictions))}/{len(all_predictions)}")
        print(f"   Total test samples: {len(all_predictions)}")
        print(f"   K parameter: {K}")
        print(f"   Processes used: {size}")
        
        print(f"\n📊 ACCURACY BY CLASS:")
        print("-" * 40)
        for class_id in range(10):
            if class_count[class_id] > 0:
                print(f"   Class {class_id}: {class_accuracy[class_id]:.4f} ({class_count[class_id]} samples)")
        
        print(f"\n📈 CONFUSION MATRIX (rows=true, cols=predicted):")
        print("-" * 50)
        print("    ", end="")
        for j in range(10):
            print(f"{j:>4}", end="")
        print()
        
        for i in range(10):
            print(f"{i}: ", end="")
            for j in range(10):
                print(f"{confusion_matrix[i][j]:>4}", end="")
            print()
        
        # Mostrar algunas predicciones detalladas
        print(f"\n🔍 SAMPLE PREDICTIONS (first 10):")
        print("-" * 40)
        for i in range(min(10, len(all_predictions))):
            status = "✓" if all_predictions[i] == all_true_labels[i] else "✗"
            print(f"   {status} Sample {i}: Predicted={all_predictions[i]}, True={all_true_labels[i]}")
        
        print(f"\n" + "="*60)
        
        # Retornar métricas para guardar
        return accuracy, K, size
    
    return None, None, None

def knn_c_style():
    """Función principal equivalente a knn() en C"""
    comm, rank, size = mpi_initialise()
    
    # Cargar datos solo en proceso 0 (como en C)
    if rank == 0:
        X_train, X_test, y_train, y_test = init_features_and_labels()
        print(f"🚀 KNN MPI (C-Style Implementation with Weights Saving)")
        print(f"="*60)
        print(f"📊 Dataset Information:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {len(np.unique(y_train))}")
        print(f"   Processes: {size}")
        print(f"   K parameter: 3")
        print(f"="*60)
        t1 = MPI.Wtime()
    else:
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        t1 = None
    
    # Todos los procesos cargan X_test y y_test (como en C)
    if rank != 0:
        _, X_test, _, y_test = init_features_and_labels()
    
    # Ejecutar algoritmo
    accuracy, k, processes = fit(X_train, y_train, X_test, y_test, comm, rank, size)
    
    # Medir tiempo final y guardar pesos
    if rank == 0:
        t2 = MPI.Wtime()
        execution_time = t2 - t1
        
        print(f"⏱️  EXECUTION TIME: {execution_time:.4f} seconds ({size} processors)")
        
        # Información del modelo para guardar
        model_info = {
            'accuracy': accuracy,
            'k': k,
            'processes': processes,
            'execution_time': execution_time,
            'random_state': 42,
            'algorithm': 'KNN_C_Style_MPI',
            'communication_pattern': 'scatter_gather_per_sample'
        }
        
        # Guardar pesos/modelo
        weight_file, info_file = save_model_weights(
            X_train, y_train, X_test, y_test, 
            model_info, 
            filename=f"knn_c_style_p{size}_k{k}"
        )
        
        print(f"💾 MODEL WEIGHTS SAVED:")
        print(f"   Weights file: {weight_file}")
        print(f"   Info file: {info_file}")
        print(f"="*60)

if __name__ == "__main__":
    knn_c_style()