import pickle
import numpy as np
import os
from collections import Counter

def load_model_weights(filename):
    """Cargar los pesos/datos del modelo KNN guardado"""
    
    weights_dir = "knn_weights"
    filepath = os.path.join(weights_dir, f"{filename}.pkl")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def list_available_weights():
    """Listar todos los archivos de pesos disponibles"""
    weights_dir = "knn_weights"
    
    if not os.path.exists(weights_dir):
        print("No weights directory found.")
        return []
    
    files = [f for f in os.listdir(weights_dir) if f.endswith('.pkl')]
    
    print("Available weight files:")
    print("-" * 40)
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
        
        # Mostrar info si existe
        info_file = file.replace('.pkl', '_info.txt')
        info_path = os.path.join(weights_dir, info_file)
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                lines = f.readlines()
                # Mostrar líneas clave
                for line in lines:
                    if any(keyword in line for keyword in ['Accuracy', 'Execution time', 'Processes']):
                        print(f"   {line.strip()}")
        print()
    
    return files

def test_with_loaded_weights(weight_filename, test_samples=10):
    """Probar el modelo usando pesos cargados (versión secuencial simple)"""
    
    print(f"🔄 Testing with loaded weights: {weight_filename}")
    print("="*50)
    
    # Cargar modelo
    model_data = load_model_weights(weight_filename)
    
    X_train = model_data['X_train']
    y_train = model_data['y_train'] 
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    model_info = model_data['model_info']
    
    print(f"📊 Loaded Model Info:")
    print(f"   Original accuracy: {model_info.get('accuracy', 'N/A'):.4f}")
    print(f"   K parameter: {model_info.get('k', 3)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Original processes: {model_info.get('processes', 'N/A')}")
    
    # KNN simple para verificar
    def simple_knn_predict(test_point, X_train, y_train, k=3):
        # Calcular distancias
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        # Obtener k vecinos más cercanos
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        # Votación mayoritaria
        most_common = Counter(k_labels).most_common(1)
        return int(most_common[0][0])
    
    # Probar con algunas muestras
    k = model_info.get('k', 3)
    predictions = []
    true_labels = []
    
    print(f"\n🧪 Testing {test_samples} samples with loaded weights:")
    print("-" * 40)
    
    for i in range(min(test_samples, len(X_test))):
        pred = simple_knn_predict(X_test[i], X_train, y_train, k)
        true = int(y_test[i])
        
        predictions.append(pred)
        true_labels.append(true)
        
        status = "✓" if pred == true else "✗"
        print(f"   {status} Sample {i}: Predicted={pred}, True={true}")
    
    # Calcular accuracy de la prueba
    test_accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Correct: {int(test_accuracy * len(predictions))}/{len(predictions)}")
    
    return model_data

def compare_weights(weight_files):
    """Comparar múltiples archivos de pesos"""
    
    print("📊 WEIGHT FILES COMPARISON")
    print("="*60)
    
    results = []
    
    for weight_file in weight_files:
        try:
            model_data = load_model_weights(weight_file.replace('.pkl', ''))
            info = model_data['model_info']
            
            result = {
                'filename': weight_file,
                'accuracy': info.get('accuracy', 0),
                'execution_time': info.get('execution_time', 0),
                'processes': info.get('processes', 0),
                'k': info.get('k', 0),
                'timestamp': model_data.get('timestamp', 'Unknown')
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {weight_file}: {e}")
    
    # Ordenar por accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'File':<25} {'Accuracy':<10} {'Time(s)':<8} {'Proc':<4} {'K':<3}")
    print("-" * 60)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['filename'][:24]:<25} {result['accuracy']:<10.4f} "
              f"{result['execution_time']:<8.2f} {result['processes']:<4} {result['k']:<3}")
    
    return results

if __name__ == "__main__":
    # Listar pesos disponibles
    available_weights = list_available_weights()
    
    if available_weights:
        print("\n" + "="*50)
        
        # Probar con el primer archivo disponible
        first_weight = available_weights[0].replace('.pkl', '')
        test_with_loaded_weights(first_weight, test_samples=20)
        
        print("\n" + "="*50)
        
        # Comparar todos los archivos
        if len(available_weights) > 1:
            compare_weights(available_weights)
    else:
        print("No weight files found. Run a KNN training script first.")