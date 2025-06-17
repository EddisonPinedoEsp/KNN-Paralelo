from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import time

# --- Funciones auxiliares (igual que en el secuencial) ---
def euclidean_distance(a, b):
    """Calcula la distancia euclidiana entre dos puntos a y b."""
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict_for_point(test_point, X_train_data, y_train_data, k_val):
    """
    Predice la etiqueta para un único punto de prueba usando KNN.
    X_train_data e y_train_data son los datos de entrenamiento completos.
    """
    distances = [euclidean_distance(test_point, x) for x in X_train_data]
    # Encuentra los k índices más cercanos
    k_indices = np.argsort(distances)[:k_val]
    # Obtiene las etiquetas de esos k vecinos
    k_labels = [y_train_data[i] for i in k_indices]
    # Devuelve la etiqueta más común
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# --- Función principal MPI ---
def main_parallel_knn():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parámetros
    k = 3
    X_train, X_test, y_train, y_test_root = None, None, None, None # y_test_root solo en rank 0
    root_start_time = 0

    if rank == 0:
        print(f"--- Ejecutando KNN Paralelo con {size} procesos ---")
        # Cargar y dividir los datos solo en el proceso raíz
        digits = load_digits()
        X_train, X_test, y_train, y_test_root = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        root_start_time = time.time()

    # 1. Broadcast de los datos de entrenamiento (X_train, y_train) y el parámetro k
    #    Estos son necesarios para todos los procesos para hacer predicciones.
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    k = comm.bcast(k, root=0)

    # 2. Dividir X_test y distribuirlo (scatter) entre los procesos
    #    El proceso raíz prepara los trozos (chunks) de X_test.
    local_X_test_chunk = None
    if rank == 0:
        # np.array_split maneja bien los casos donde la división no es perfecta
        chunks_X_test = np.array_split(X_test, size, axis=0)
    else:
        chunks_X_test = None

    # Distribuir los trozos de X_test a cada proceso
    local_X_test_chunk = comm.scatter(chunks_X_test, root=0)

    # 3. Cada proceso realiza predicciones para su porción de X_test
    local_y_pred = []
    if local_X_test_chunk is not None and len(local_X_test_chunk) > 0:
        for test_point in local_X_test_chunk:
            prediction = knn_predict_for_point(test_point, X_train, y_train, k)
            local_y_pred.append(prediction)
    
    # Sincronizar antes de medir el tiempo de cómputo si se desea medir solo la predicción
    # comm.Barrier() # Opcional aquí, el gather ya sincroniza

    # 4. Recopilar todas las predicciones locales en el proceso raíz
    #    comm.gather recopila listas de cada proceso en una lista de listas en el raíz.
    gathered_y_pred_lists = comm.gather(local_y_pred, root=0)

    # 5. El proceso raíz combina las predicciones, evalúa y muestra resultados
    if rank == 0:
        # Combinar la lista de listas en una sola lista de predicciones
        # El orden de gather preserva el orden de los rangos, y np.array_split
        # mantiene el orden original de los datos, por lo que y_pred estará en el orden correcto.
        y_pred_final = []
        if gathered_y_pred_lists:
            for lst in gathered_y_pred_lists:
                y_pred_final.extend(lst)
        
        y_pred_final = np.array(y_pred_final)

        if len(y_pred_final) == len(y_test_root):
            accuracy = np.mean(y_pred_final == y_test_root)
            root_end_time = time.time()

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Execution time (parallel): {root_end_time - root_start_time:.4f} sec")

            # Visualizar algunas predicciones (igual que en el secuencial)
            # Importar matplotlib solo en el proceso raíz y cuando se necesite
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, ax in enumerate(axes.flat):
                if i < len(X_test) and i < len(y_pred_final): # Usar X_test original del raíz
                    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
                    ax.set_title(f"Pred: {y_pred_final[i]}\nTrue: {y_test_root[i]}")
                    ax.axis('off')
                else:
                    ax.axis('off') # Ocultar ejes si no hay datos para mostrar
            plt.suptitle(f"Sample Predictions (Parallel KNN with {size} processes)")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Error: La longitud de y_pred_final ({len(y_pred_final)}) no coincide con y_test_root ({len(y_test_root)}).")
            print("Esto podría indicar un problema con la división o recolección de datos.")

if __name__ == '__main__':
    main_parallel_knn()