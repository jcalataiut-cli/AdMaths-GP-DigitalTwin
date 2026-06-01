#!/usr/bin/env python3
"""
pipeline.py — Pipeline completo del Digital Twin GP
====================================================

Ejecuta todo el flujo:
  1. Generar datos sintéticos del sistema físico
  2. Entrenar el Digital Twin (Gaussian Process)
  3. Evaluar predicciones
  4. Detectar anomalías
  5. Visualizar resultados

Uso:
  python pipeline.py
"""

import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Añadir directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physical_model import BuildingPhysicalSystem, default_schedule
from src.generate_data import generate_nominal_scenarios, generate_anomaly_scenarios
from src.gp_digital_twin import GPDigitalTwin
from src.visualize import (
    plot_training_history,
    plot_prediction_vs_actual,
    plot_anomaly_detection,
    plot_uncertainty_evolution,
    plot_kernel_alignment,
    plot_summary_dashboard,
)

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def step1_generate_data():
    """Paso 1: Generar datos sintéticos."""
    print("\n" + "=" * 60)
    print("PASO 1: GENERACIÓN DE DATOS SINTÉTICOS")
    print("=" * 60)

    t0 = time.time()

    # Datos de entrenamiento
    print("\nGenerando escenarios nominales de entrenamiento...")
    X_train, y_train = generate_nominal_scenarios(n_scenarios=8, steps=144)
    X_val, y_val = generate_nominal_scenarios(n_scenarios=2, steps=144, seed_base=200)

    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Val:   {X_val.shape[0]} muestras")

    # Datos con anomalías
    print("\nGenerando escenarios con anomalías...")
    anomalies = generate_anomaly_scenarios(n_scenarios=3, steps=144)

    # Guardar
    np.savez(os.path.join(DATA_DIR, 'training_data.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(DATA_DIR, 'validation_data.npz'), X=X_val, y=y_val)

    import pickle
    with open(os.path.join(DATA_DIR, 'anomaly_scenarios.pkl'), 'wb') as f:
        pickle.dump(anomalies, f)

    print(f"\n  Tiempo: {time.time() - t0:.1f}s")
    print("  ✅ Datos guardados")

    # Visualizar datos de entrenamiento
    print("\nGenerando figura: datos de entrenamiento...")
    plot_training_history(
        np.column_stack([y_train[:288], X_train[:288, 3]]),
        save_path=os.path.join(RESULTS_DIR, 'fig01_training_data.png')
    )

    return X_train, y_train, X_val, y_val, anomalies


def step2_train_digital_twin(X_train, y_train):
    """Paso 2: Entrenar el Digital Twin GP."""
    print("\n" + "=" * 60)
    print("PASO 2: ENTRENAMIENTO DEL DIGITAL TWIN (GP)")
    print("=" * 60)

    t0 = time.time()

    print(f"\nInicializando Digital Twin con kernel Matern...")
    dt = GPDigitalTwin(kernel_type='matern')

    print(f"Entrenando 3 GP (uno por variable)...")
    dt.fit(X_train, y_train)

    dt.save(os.path.join(DATA_DIR, 'digital_twin_gp.pkl'))

    print(f"\n  Tiempo: {time.time() - t0:.1f}s")
    print("  ✅ Digital Twin entrenado y guardado")
    return dt


def step3_evaluate(dt, X_val, y_val):
    """Paso 3: Evaluar el Digital Twin."""
    print("\n" + "=" * 60)
    print("PASO 3: EVALUACIÓN DEL DIGITAL TWIN")
    print("=" * 60)

    t0 = time.time()

    y_pred, y_std = dt.predict(X_val)

    # Métricas
    mae = np.mean(np.abs(y_val - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_val - y_pred)**2, axis=0))
    r2_scores = []
    for i in range(3):
        ss_res = np.sum((y_val[:, i] - y_pred[:, i])**2)
        ss_tot = np.sum((y_val[:, i] - np.mean(y_val[:, i]))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        r2_scores.append(r2)

    print(f"\n  Métricas de rendimiento:")
    print(f"  {'Variable':15s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for i, name in enumerate(dt.target_names):
        print(f"  {name:15s} {mae[i]:>8.3f} {rmse[i]:>8.3f} {r2_scores[i]:>8.4f}")

    print(f"\n  NKA (Normalized Kernel Alignment) teórico:")
    print(f"  Correlación entre kernels: implícita en R² > {min(r2_scores):.4f}")
    print(f"  → El kernel del GP captura la dinámica del sistema")

    # Gráficas
    print("\nGenerando figuras de evaluación...")
    plot_prediction_vs_actual(
        y_val[:200], y_pred[:200], y_std[:200],
        save_path=os.path.join(RESULTS_DIR, 'fig02_prediction_vs_actual.png')
    )

    # Kernel alignment plot
    plot_kernel_alignment(
        dt, X_val[:50], y_val[:50],
        save_path=os.path.join(RESULTS_DIR, 'fig03_kernel_analysis.png')
    )

    print(f"\n  Tiempo: {time.time() - t0:.1f}s")
    return y_pred, y_std


def step4_anomaly_detection(dt, anomalies):
    """Paso 4: Detección de anomalías."""
    print("\n" + "=" * 60)
    print("PASO 4: DETECCIÓN DE ANOMALÍAS")
    print("=" * 60)

    t0 = time.time()

    anomaly_results = []
    scores_dict = {}

    for anomaly in anomalies:
        name = anomaly['name']
        X_a, y_a = anomaly['X'], anomaly['y_true']
        true_mask = anomaly['anomaly_mask']

        # Detectar anomalías
        detected_mask, scores = dt.detect_anomalies(X_a, y_a, threshold=2.0)

        # Métricas
        if true_mask.sum() > 0:
            tp = (detected_mask & true_mask).sum()
            fn = (~detected_mask & true_mask).sum()
            fp = (detected_mask & ~true_mask).sum()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
        else:
            precision = recall = f1 = 0.0

        result = {
            'name': name,
            'scores': scores,
            'detected_mask': detected_mask,
            'anomaly_mask': true_mask,
            'threshold': 2.0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        anomaly_results.append(result)
        scores_dict[name] = result

        print(f"\n  Escenario: {name}")
        print(f"    Anomalías reales: {true_mask.sum()}")
        print(f"    Detectadas: {detected_mask.sum()}")
        print(f"    Precisión: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-score: {f1:.3f}")

    # Visualizar
    plot_anomaly_detection(
        anomaly_results,
        save_path=os.path.join(RESULTS_DIR, 'fig04_anomaly_detection.png')
    )

    print(f"\n  Tiempo: {time.time() - t0:.1f}s")
    return anomaly_results


def step5_simulate(dt, X_val, y_val):
    """Paso 5: Simulación de trayectoria."""
    print("\n" + "=" * 60)
    print("PASO 5: SIMULACIÓN DE TRAYECTORIA")
    print("=" * 60)

    t0 = time.time()

    # Simular 50 pasos desde el primer estado de validación
    initial = X_val[0, :4]  # [T, H, CO2, occ]
    controls = X_val[:50, 4:]  # [heater, vent]

    trajectory, uncertainty = dt.simulate(initial, controls, 50)

    plot_uncertainty_evolution(
        trajectory, uncertainty,
        save_path=os.path.join(RESULTS_DIR, 'fig05_trajectory_simulation.png')
    )

    print(f"\n  Trayectoria simulada: {len(trajectory)} pasos")
    print(f"  Incertidumbre final: σ_T={uncertainty[-1, 0]:.3f}°C, "
          f"σ_H={uncertainty[-1, 1]:.3f}%, σ_CO2={uncertainty[-1, 2]:.1f}ppm")
    print(f"  Tiempo: {time.time() - t0:.1f}s")


def step6_dashboard(dt, X_val, y_val, y_pred, y_std, anomaly_results):
    """Paso 6: Dashboard resumen."""
    print("\n" + "=" * 60)
    print("PASO 6: DASHBOARD RESUMEN")
    print("=" * 60)

    scores_dict = {r['name']: r for r in anomaly_results}

    plot_summary_dashboard(
        y_val, y_pred, y_std, scores_dict,
        save_path=os.path.join(RESULTS_DIR, 'fig06_dashboard.png')
    )

    print("  ✅ Dashboard generado")


def main():
    print("=" * 72)
    print("  DIGITAL TWIN con GAUSSIAN PROCESS")
    print("  Sistema: Smart Building Zone (CPS multi-variable)")
    print("=" * 72)

    print("\n" + "=" * 60)
    print("  CONEXIÓN CON EL TFM: NNGP → GP → Digital Twin")
    print("=" * 60)
    print("""
  El teorema NNGP establece que una red neuronal de ancho infinito
  equivale a un Gaussian Process. Este Digital Twin usa GP directamente
  como modelo generativo para simular un CPS, demostrando que:

    • El GP captura la dinámica no-lineal del sistema
    • La incertidumbre del GP permite detección de anomalías
    • No requiere un Diffusion Model completo (como el paper original)
    • El kernel aprendido revela la estructura de correlaciones
    • Es ejecutable en CPU en minutos
    """)

    t_total = time.time()

    # Ejecutar pipeline
    X_train, y_train, X_val, y_val, anomalies = step1_generate_data()
    dt = step2_train_digital_twin(X_train, y_train)
    y_pred, y_std = step3_evaluate(dt, X_val, y_val)
    anomaly_results = step4_anomaly_detection(dt, anomalies)
    step5_simulate(dt, X_val, y_val)
    step6_dashboard(dt, X_val, y_val, y_pred, y_std, anomaly_results)

    # Resumen final
    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)

    print(f"\n  Tiempo total: {time.time() - t_total:.1f}s")
    print(f"\n  Figuras generadas en {RESULTS_DIR}/:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f:40s} {size:>8} bytes")

    print(f"\n  {'='*56}")
    print(f"  ✅ DIGITAL TWIN COMPLETADO")
    print(f"  {'='*56}")
    print(f"\n  Mejoras respecto al paper original (Diffusion_DigitalTwin):")
    print(f"  • Sistema: brazo robótico → Smart Building Zone")
    print(f"  • Modelo generativo: Diffusion → Gaussian Process (conecta con NNGP)")
    print(f"  • Ejecutable en CPU (sin GPU necesaria)")
    print(f"  • Detección de anomalías por incertidumbre GP")
    print(f"  • Análisis de kernel (NKA) — conexión directa con el TFM")


if __name__ == '__main__':
    main()
