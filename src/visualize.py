"""
visualize.py — Visualización de resultados del Digital Twin
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_training_history(data, save_path=None):
    """Grafica los datos de entrenamiento del sistema físico."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Datos del Sistema Físico: Smart Building Zone', fontsize=14, fontweight='bold')

    t = np.arange(len(data))
    labels = ['Temperatura (°C)', 'Humedad Relativa (%)', 'CO2 (ppm)', 'Ocupación (personas)']

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, data[:, i], color=plt.cm.tab10(i), linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Paso de tiempo (5 min)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [✓] {save_path}")
    plt.close()


def plot_prediction_vs_actual(y_true, y_pred, y_std=None, save_path=None):
    """Compara predicciones del DT con valores reales."""
    n_vars = y_true.shape[1]
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3 * n_vars))
    fig.suptitle('Digital Twin: Predicción vs Realidad', fontsize=14, fontweight='bold')

    names = ['Temperatura (°C)', 'Humedad Relativa (%)', 'CO2 (ppm)']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    for i, (ax, name, color) in enumerate(zip(axes, names, colors)):
        ax.plot(y_true[:, i], color=color, alpha=0.7, label='Real', linewidth=0.8)
        ax.plot(y_pred[:, i], color='black', linestyle='--', label='DT Predicción', linewidth=0.8)

        if y_std is not None:
            ax.fill_between(
                range(len(y_pred)),
                y_pred[:, i] - 2 * y_std[:, i],
                y_pred[:, i] + 2 * y_std[:, i],
                color='gray', alpha=0.2, label='95% CI'
            )

        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Muestra')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [✓] {save_path}")
    plt.close()


def plot_anomaly_detection(anomaly_results, save_path=None):
    """Visualiza resultados de detección de anomalías."""
    n_scenarios = len(anomaly_results)
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(16, 4 * n_scenarios))
    fig.suptitle('Detección de Anomalías con Digital Twin GP', fontsize=14, fontweight='bold')

    if n_scenarios == 1:
        axes = [axes]

    for idx, result in enumerate(anomaly_results):
        ax = axes[idx]
        scores = result['scores']
        mask = result['anomaly_mask']
        threshold = result.get('threshold', 2.0)

        # Puntuación de anomalía
        ax.plot(scores, color='#8E44AD', linewidth=0.8, label='Anomaly Score')
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Umbral={threshold}σ')

        # Marcar regiones anómalas
        if mask.any():
            ax.fill_between(
                range(len(mask)),
                0, np.max(scores) * 1.1,
                where=mask,
                color='red', alpha=0.1,
                label='Anomalía real'
            )

        # Marcar detecciones
        detected = scores > threshold
        if detected.any():
            ax.scatter(
                np.where(detected)[0],
                scores[detected],
                color='red', s=30, marker='x', zorder=5,
                label=f'Detectadas ({detected.sum()})'
            )

        ax.set_title(f'Escenario: {result["name"]}')
        ax.set_ylabel('Anomaly Score (σ)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Muestra')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [✓] {save_path}")
    plt.close()


def plot_uncertainty_evolution(trajectory, uncertainty, save_path=None):
    """Muestra cómo evoluciona la incertidumbre en la simulación."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Simulación con el Digital Twin: Propagación de Incertidumbre',
                 fontsize=14, fontweight='bold')

    names = ['Temperatura (°C)', 'Humedad Relativa (%)', 'CO2 (ppm)']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    for i, (ax, name, color) in enumerate(zip(axes, names, colors)):
        ax.plot(trajectory[:, i], color=color, linewidth=1.5, label='DT Simulación')
        ax.fill_between(
            range(len(trajectory)),
            trajectory[:, i] - 2 * uncertainty[:, i],
            trajectory[:, i] + 2 * uncertainty[:, i],
            color=color, alpha=0.15,
            label='95% CI (incertidumbre GP)'
        )
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Paso de simulación')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [✓] {save_path}")
    plt.close()


def plot_kernel_alignment(digital_twin, X_val, y_val, save_path=None):
    """
    Analiza la alineación del kernel aprendido vs datos.
    Conceptualmente conecta con NKA (Normalized Kernel Alignment) del TFM.
    """
    from sklearn.gaussian_process.kernels import RBF

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Análisis del Kernel GP — Conexión con NNGP', fontsize=14, fontweight='bold')

    for i, (ax, name) in enumerate(zip(axes, digital_twin.target_names)):
        gp = digital_twin.models[name]

        # Extraer kernel aprendido
        K = gp.kernel_(digital_twin.scaler_X.transform(X_val[:50]))

        # Visualizar matriz de kernel
        im = ax.imshow(K, cmap='viridis', aspect='auto')
        ax.set_title(f'Kernel "{name}" aprendido')
        ax.set_xlabel('Muestra')
        ax.set_ylabel('Muestra')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [✓] {save_path}")
    plt.close()


def plot_summary_dashboard(y_true, y_pred, y_std, scores_dict, save_path=None):
    """Dashboard resumen de rendimiento del Digital Twin."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Digital Twin GP — Dashboard de Rendimiento', fontsize=16, fontweight='bold')

    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1-3: Predicciones vs reales
    names = ['Temperatura (°C)', 'Humedad Relativa (%)', 'CO2 (ppm)']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(y_true[:200, i], color=colors[i], alpha=0.6, label='Real', linewidth=0.7)
        ax.plot(y_pred[:200, i], 'k--', label='DT Pred', linewidth=0.7)
        if y_std is not None:
            ax.fill_between(range(200),
                          y_pred[:200, i] - 1.96 * y_std[:200, i],
                          y_pred[:200, i] + 1.96 * y_std[:200, i],
                          color='gray', alpha=0.15)
        ax.set_ylabel(names[i].split('(')[0])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('Predicción del DT (primeros 200 pasos)')

    # 4: Error de predicción
    ax = fig.add_subplot(gs[0, 1])
    error = np.abs(y_true - y_pred)
    for i, (name, color) in enumerate(zip(['T', 'H', 'CO2'], colors)):
        ax.plot(error[:500, i], color=color, alpha=0.6, label=name, linewidth=0.5)
    ax.set_title('Error Absoluto de Predicción')
    ax.set_xlabel('Muestra')
    ax.set_ylabel('Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5: Incertidumbre
    ax = fig.add_subplot(gs[1, 1])
    if y_std is not None:
        for i, (name, color) in enumerate(zip(['T', 'H', 'CO2'], colors)):
            ax.plot(y_std[:500, i], color=color, alpha=0.6, label=name, linewidth=0.5)
    ax.set_title('Incertidumbre del GP (σ)')
    ax.set_xlabel('Muestra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6: Scatter real vs predicho
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(y_true[:, 0], y_pred[:, 0], c=colors[0], alpha=0.3, s=2, label='T')
    ax.scatter(y_true[:, 2], y_pred[:, 2], c=colors[2], alpha=0.3, s=2, label='CO2')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.5)
    ax.set_title('Real vs Predicho')
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicho')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7-9: Scores de anomalías
    for i, (key, result) in enumerate(scores_dict.items()):
        ax = fig.add_subplot(gs[i, 2])
        scores = result['scores']
        mask = result['anomaly_mask']
        ax.plot(scores, color='#8E44AD', linewidth=0.7)
        ax.axhline(result.get('threshold', 2.0), color='red', linestyle='--', alpha=0.5)
        if mask.any():
            ax.fill_between(range(len(mask)), 0, np.max(scores) * 1.1,
                          where=mask, color='red', alpha=0.08)
        ax.set_title(f'Anomalías: {key}')
        ax.set_xlabel('Muestra')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('Score (σ)')

    plt.savefig(save_path, dpi=150, bbox_inches='tight') if save_path else plt.show()
    print(f"  [✓] Dashboard guardado" if save_path else "")
    plt.close()
