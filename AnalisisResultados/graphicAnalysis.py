import matplotlib.pyplot as plt
import numpy as np

def performance_analysis():
    data = {
        'processes': [1, 2, 4, 8, 16, 32, 48],
        'total_time': [1.5400, 0.7524, 0.3860, 0.2590, 0.2345, 0.6532, 2.6356],
        'computation_time': [1.5280, 0.7384, 0.3675, 0.2214, 0.1229, 0.0874, 0.0616],
        'communication_time': [0.0120, 0.0150, 0.0195, 0.0376, 0.1239, 0.8794, 2.7808],
        'accuracy': [0.9833, 0.9833, 0.9833, 0.9833, 0.9833, 0.9833, 0.9833],
        'gflops_per_second': [0.065, 0.068, 0.066, 0.056, 0.050, 0.035, 0.033],
    }
    
    processes = np.array(data['processes'])
    parallel_times = np.array(data['total_time'])
    comp_times = np.array(data['computation_time'])
    comm_times = np.array(data['communication_time'])
    
    sequencial_time = parallel_times[0]
    speedups = sequencial_time / parallel_times
    efficiencies = speedups / processes
    
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Análisis de Rendimiento KNN Paralelo', 
                 fontsize=16, fontweight='bold')
    
    plt.subplot(3, 3, 1)
    plt.plot(processes, parallel_times, 'bo-', linewidth=3, markersize=10, label='Tiempo Real')
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Tiempo Total (s)', fontsize=12)
    plt.title('Tiempo de Ejecución vs Procesos', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    for i, (p, t) in enumerate(zip(processes, parallel_times)):
        plt.annotate(f'{t:.3f}s', (p, t), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.subplot(3, 3, 2)
    plt.plot(processes, speedups, 'go-', linewidth=3, markersize=10, label='Speedup Real')
    plt.plot(processes, processes, 'k--', linewidth=2, alpha=0.7, label='Speedup Ideal')
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Procesos', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    for i, (p, s) in enumerate(zip(processes, speedups)):
        plt.annotate(f'{s:.2f}x', (p, s), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.subplot(3, 3, 3)
    plt.plot(processes, efficiencies, 'ro-', linewidth=3, markersize=10)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Eficiencia Ideal')
    plt.axhline(y=0.5, color='r', linestyle=':', alpha=0.5, label='Umbral 50%')
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Eficiencia', fontsize=12)
    plt.title('Eficiencia vs Procesos', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, (p, e) in enumerate(zip(processes, efficiencies)):
        plt.annotate(f'{e:.2f}', (p, e), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)

    plt.subplot(3, 3, 4)
    width = 0.6
    x_pos = np.arange(len(processes))
    
    bars1 = plt.bar(x_pos, comp_times, width, label='Cómputo', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x_pos, comm_times, width, bottom=comp_times, label='Comunicación', alpha=0.8, color='orange')
    
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Tiempo (s)', fontsize=12)
    plt.title('Distribución: Cómputo vs Comunicación', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, processes)
    plt.legend()
    
    for i, (comp, comm) in enumerate(zip(comp_times, comm_times)):
        plt.text(i, comp/2, f'{comp:.3f}s', ha='center', va='center', fontweight='bold')
        plt.text(i, comp + comm/2, f'{comm:.3f}s', ha='center', va='center', fontweight='bold')
    
    plt.subplot(3, 3, 5)
    plt.plot(processes, data['accuracy'], 'mo-', linewidth=3, markersize=10)
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Procesos', fontsize=14, fontweight='bold')
    plt.ylim(0.95, 1.01)
    plt.grid(True, alpha=0.3)
    
    for i, (p, a) in enumerate(zip(processes, data['accuracy'])):
        plt.annotate(f'{a:.3f}', (p, a), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.subplot(3, 3, 6)
    plt.plot(processes, comp_times, 'co-', linewidth=3, markersize=10)
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Tiempo de Cómputo (s)', fontsize=12)
    plt.title('Tiempo de Cómputo vs Procesos', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (p, t) in enumerate(zip(processes, comp_times)):
        plt.annotate(f'{t:.3f}s', (p, t), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.subplot(3, 3, 7)
    plt.plot(processes, comm_times, 'ro-', linewidth=3, markersize=10)
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Tiempo de Comunicación (s)', fontsize=12)
    plt.title('Tiempo de Comunicación vs Procesos', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (p, t) in enumerate(zip(processes, comm_times)):
        plt.annotate(f'{t:.4f}s', (p, t), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.subplot(3, 3, 8)
    plt.plot(processes, data['gflops_per_second'], 'yo-', linewidth=3, markersize=10)
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('GFLOPS/segundo', fontsize=12)
    plt.title('Rendimiento Computacional (GFLOPS/s) vs Procesos', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (p, g) in enumerate(zip(processes, data['gflops_per_second'])):
        plt.annotate(f'{g:.3f}', (p, g), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('knn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return data, speedups, efficiencies


def main():
    print("Creando análisis de rendimiento KNN Paralelo...")
    
    data, speedups, efficiencies = performance_analysis()
    
    print(f"\\nGráfico guardado en 'knn_analysis.png'")

if __name__ == "__main__":
    main()
