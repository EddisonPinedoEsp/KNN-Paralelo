import matplotlib.pyplot as plt
import numpy as np

def multi_data_performance_analysis():
    # Datos experimentales para diferentes tamaños de datasets
    datasets = {
        'Small (1437 samples)': {
            'processes': [1, 2, 4, 8, 16, 32, 48],
            'total_time': [1.5400, 0.7524, 0.3860, 0.2590, 0.2345, 0.6532, 2.6356],
            'computation_time': [1.5280, 0.7384, 0.3675, 0.2214, 0.1229, 0.0874, 0.0616],
            'communication_time': [0.0120, 0.0150, 0.0195, 0.0376, 0.1239, 0.8794, 2.7808],
            'accuracy': [0.9833, 0.9833, 0.9833, 0.9833, 0.9833, 0.9833, 0.9833],
            'gflops_per_second': [0.065, 0.068, 0.066, 0.056, 0.050, 0.035, 0.033],
            'samples': 1437
        },
        'Medium (5K samples)': {
            'processes': [1, 2, 4, 8, 16, 32, 48],
            'total_time': [23.7846, 11.6023, 5.8109, 3.2285, 2.8308, 6.4247, 22.9056],
            'computation_time': [23.6914, 11.4223, 5.7056, 3.0092, 2.0640, 1.0484, 0.7668],
            'communication_time': [0.1112, 0.1995, 0.1279, 0.2453, 0.8050, 5.4452, 22.2946],
            'accuracy': [0.9456, 0.9456, 0.9448, 0.9448, 0.9448, 0.9448, 0.9448],
            'gflops_per_second': [0.622, 0.645, 0.645, 0.611, 0.445, 0.438, 0.399],
            'samples': 5000
        },
        'Large (10K samples)': {
            'processes': [1, 2, 4, 8, 16, 32, 48],
            'total_time': [63.0599, 30.8106, 15.2900, 8.4851, 7.8372, 19.8398, 56.8132],
            'computation_time': [62.9042, 30.4992, 15.0939, 8.0376, 5.7747, 2.4182, 1.6420],
            'communication_time': [0.1782, 0.3471, 0.2271, 0.4875, 2.1077, 17.4996, 55.3638],
            'accuracy': [0.9562, 0.9562, 0.9562, 0.9562, 0.9562, 0.9562, 0.9562],
            'gflops_per_second': [0.617, 0.636, 0.642, 0.603, 0.419, 0.500, 0.492],
            'samples': 10000
        },
        'XLarge (20K samples)': {
            'processes': [1, 2, 4, 8, 16, 32],
            'total_time': [327.6870, 161.1778, 78.3861, 43.4060, 37.3959, 133.5939],
            'computation_time': [327.3308, 160.7783, 77.8176, 42.3485, 28.9190, 12.9366],
            'communication_time': [0.4125, 0.4675, 0.6444, 1.1491, 8.5783, 120.8159],
            'accuracy': [0.9630, 0.9630, 0.9630, 0.9630, 0.9630, 0.9630],
            'gflops_per_second': [0.614, 0.625, 0.645, 0.593, 0.434, 0.485],
            'samples': 20000
        }
    }
    
    # Configuración de colores y estilos para cada dataset
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('Análisis Comparativo de Rendimiento KNN Paralelo - Múltiples Tamaños de Datos', 
                 fontsize=18, fontweight='bold')
    
    # 1. Tiempo de Ejecución vs Procesos
    plt.subplot(3, 3, 1)
    for i, (dataset_name, data) in enumerate(datasets.items()):
        processes = np.array(data['processes'])
        times = np.array(data['total_time'])
        plt.plot(processes, times, marker=markers[i], color=colors[i], 
                linestyle=linestyles[i], linewidth=2, markersize=8, 
                label=dataset_name)
    
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Tiempo Total (s)', fontsize=12)
    plt.title('Tiempo de Ejecución vs Procesos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    
    # 2. Speedup vs Procesos
    plt.subplot(3, 3, 2)
    for i, (dataset_name, data) in enumerate(datasets.items()):
        processes = np.array(data['processes'])
        times = np.array(data['total_time'])
        sequential_time = times[0]
        speedups = sequential_time / times
        plt.plot(processes, speedups, marker=markers[i], color=colors[i], 
                linestyle=linestyles[i], linewidth=2, markersize=8, 
                label=dataset_name)
    
    # Línea de speedup ideal
    plt.plot(processes, processes, 'k--', linewidth=2, alpha=0.7, label='Speedup Ideal')
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Procesos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 3. Eficiencia vs Procesos
    plt.subplot(3, 3, 3)
    for i, (dataset_name, data) in enumerate(datasets.items()):
        processes = np.array(data['processes'])
        times = np.array(data['total_time'])
        sequential_time = times[0]
        speedups = sequential_time / times
        efficiencies = speedups / processes
        plt.plot(processes, efficiencies, marker=markers[i], color=colors[i], 
                linestyle=linestyles[i], linewidth=2, markersize=8, 
                label=dataset_name)
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Eficiencia Ideal')
    plt.axhline(y=0.5, color='r', linestyle=':', alpha=0.5, label='Umbral 50%')
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Eficiencia', fontsize=12)
    plt.title('Eficiencia vs Procesos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # 4. Tiempo de Cómputo vs Procesos
    plt.subplot(3, 3, 4)
    for i, (dataset_name, data) in enumerate(datasets.items()):
        processes = np.array(data['processes'])
        comp_times = np.array(data['computation_time'])
        plt.plot(processes, comp_times, marker=markers[i], color=colors[i], 
                linestyle=linestyles[i], linewidth=2, markersize=8, 
                label=dataset_name)
    
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Tiempo de Cómputo (s)', fontsize=12)
    plt.title('Tiempo de Cómputo vs Procesos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 5. Tiempo de Comunicación vs Procesos
    plt.subplot(3, 3, 5)
    for i, (dataset_name, data) in enumerate(datasets.items()):
        processes = np.array(data['processes'])
        comm_times = np.array(data['communication_time'])
        plt.plot(processes, comm_times, marker=markers[i], color=colors[i], 
                linestyle=linestyles[i], linewidth=2, markersize=8, 
                label=dataset_name)
    
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Tiempo de Comunicación (s)', fontsize=12)
    plt.title('Tiempo de Comunicación vs Procesos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 6. GFLOPS vs Procesos
    plt.subplot(3, 3, 6)
    for i, (dataset_name, data) in enumerate(datasets.items()):
        processes = np.array(data['processes'])
        gflops = np.array(data['gflops_per_second'])
        plt.plot(processes, gflops, marker=markers[i], color=colors[i], 
                linestyle=linestyles[i], linewidth=2, markersize=8, 
                label=dataset_name)
    
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('GFLOPS/segundo', fontsize=12)
    plt.title('Rendimiento Computacional vs Procesos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 7. Accuracy vs Procesos
    plt.subplot(3, 3, 7)
    for i, (dataset_name, data) in enumerate(datasets.items()):
        processes = np.array(data['processes'])
        accuracy = np.array(data['accuracy'])
        plt.plot(processes, accuracy, marker=markers[i], color=colors[i], 
                linestyle=linestyles[i], linewidth=2, markersize=8, 
                label=dataset_name)
    
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Procesos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.95, 1.01)
    
    # 8. Escalabilidad: Tiempo vs Tamaño de Dataset (para 4 procesos)
    plt.subplot(3, 3, 8)
    samples_sizes = []
    times_4_proc = []
    comp_times_4_proc = []
    comm_times_4_proc = []
    
    for dataset_name, data in datasets.items():
        samples_sizes.append(data['samples'])
        # Buscar índice para 4 procesos
        proc_index = data['processes'].index(4)
        times_4_proc.append(data['total_time'][proc_index])
        comp_times_4_proc.append(data['computation_time'][proc_index])
        comm_times_4_proc.append(data['communication_time'][proc_index])
    
    plt.plot(samples_sizes, times_4_proc, 'bo-', linewidth=3, markersize=8, label='Tiempo Total')
    plt.plot(samples_sizes, comp_times_4_proc, 'go-', linewidth=3, markersize=8, label='Tiempo Cómputo')
    plt.plot(samples_sizes, comm_times_4_proc, 'ro-', linewidth=3, markersize=8, label='Tiempo Comunicación')
    
    plt.xlabel('Tamaño del Dataset (muestras)', fontsize=12)
    plt.ylabel('Tiempo (s)', fontsize=12)
    plt.title('Escalabilidad de Datos (4 procesos)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Ajustar el layout y guardar la figura
    plt.tight_layout()
    plt.savefig('knn_multi_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return datasets

def print_performance_summary(datasets):
    """Imprimir resumen de rendimiento para todos los datasets"""
    
    print("="*80)
    print("RESUMEN DE RENDIMIENTO - MÚLTIPLES TAMAÑOS DE DATOS")
    print("="*80)
    
    for dataset_name, data in datasets.items():
        print(f"\n{dataset_name} ({data['samples']:,} muestras):")
        print("-" * 50)
        
        processes = np.array(data['processes'])
        times = np.array(data['total_time'])
        sequential_time = times[0]
        speedups = sequential_time / times
        efficiencies = speedups / processes
        
        print(f"{'Procesos':<10} {'Tiempo(s)':<12} {'Speedup':<10} {'Eficiencia':<12} {'Accuracy':<10}")
        print("-" * 60)
        
        for i, p in enumerate(processes):
            print(f"{p:<10} {times[i]:<12.4f} {speedups[i]:<10.2f} {efficiencies[i]:<12.3f} {data['accuracy'][i]:<10.4f}")
        
        # Mejor rendimiento
        best_speedup_idx = np.argmax(speedups[1:]) + 1  # Excluir secuencial
        best_efficiency_idx = np.argmax(efficiencies[1:]) + 1
        
        print(f"\nMejor Speedup: {speedups[best_speedup_idx]:.2f}x con {processes[best_speedup_idx]} procesos")
        print(f"Mejor Eficiencia: {efficiencies[best_efficiency_idx]:.3f} con {processes[best_efficiency_idx]} procesos")

def main():
    print("Iniciando análisis comparativo de rendimiento KNN Paralelo...")
    
    # Cargar datos predefinidos
    datasets = multi_data_performance_analysis()
    
    # Mostrar resumen
    print_performance_summary(datasets)
    
    print(f"\nAnálisis completado. Gráficos guardados en 'knn_multi_data_analysis.png'")

if __name__ == "__main__":
    main()
