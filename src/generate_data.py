"""
generate_data.py — Generación de datos sintéticos para el Digital Twin
=====================================================================

Ejecuta múltiples escenarios del sistema físico para generar:
1. Datos de entrenamiento (condiciones nominales)
2. Datos de validación (condiciones nominales)
3. Datos de prueba con anomalías
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physical_model import BuildingPhysicalSystem, default_schedule
import warnings
warnings.filterwarnings('ignore')


def generate_nominal_scenarios(n_scenarios: int = 10, steps: int = 144,
                               seed_base: int = 42) -> tuple:
    """
    Genera múltiples escenarios bajo condiciones nominales.

    Cada escenario varía ligeramente los parámetros para crear
    diversidad en los datos de entrenamiento.

    Returns:
        X: Input features (N, context_dim)
        y: Target variables (N, output_dim)
    """
    all_X, all_y = [], []

    for i in range(n_scenarios):
        # Variar ligeramente parámetros del sistema
        from src.physical_model import BuildingParams
        params = BuildingParams(
            C_thermal=100.0 + np.random.randn() * 10,
            C_humidity=50.0 + np.random.randn() * 5,
            T_out=15.0 + np.random.randn() * 2,
        )
        bld = BuildingPhysicalSystem(params=params, dt=5.0, seed=seed_base + i * 100)
        data = bld.run_scenario(steps, default_schedule)

        # Construir pares (estado_actual + control → estado_siguiente)
        for t in range(len(data) - 1):
            # Features: [T, H, CO2, occ, heater, vent] en t
            X_t = data[t]
            # Target: [T, H, CO2] en t+1 (sin occ)
            y_t = data[t + 1][:3]
            all_X.append(X_t)
            all_y.append(y_t)

    X = np.array(all_X)
    y = np.array(all_y)

    # Añadir ruido de medición sensorial
    X_noisy = X.copy()
    X_noisy[:, 0] += np.random.randn(X.shape[0]) * 0.1  # ruido T
    X_noisy[:, 1] += np.random.randn(X.shape[0]) * 0.5  # ruido H
    X_noisy[:, 2] += np.random.randn(X.shape[0]) * 10    # ruido CO2

    return X_noisy, y


def generate_anomaly_scenarios(n_scenarios: int = 3, steps: int = 144,
                                seed_base: int = 999) -> list:
    """
    Genera escenarios con anomalías para probar detección.

    Tipos de anomalías:
      - Heater stuck: el calefactor no responde a control
      - Occupancy spike: ocupación anormal
      - Sensor drift: deriva en sensor de temperatura
      - Ventilation failure: ventilación no funciona

    Returns:
        Lista de dicts con {name, data, anomaly_mask}
    """
    anomalies = []

    for i, anomaly_type in enumerate(['heater_stuck', 'ventilation_failure', 'occupancy_spike']):
        bld = BuildingPhysicalSystem(dt=5.0, seed=seed_base + i * 100)

        def make_anomaly_schedule(base_schedule, anomaly_type):
            def schedule(t, state):
                heater, vent, occ_change = base_schedule(t, state)
                anomaly_active = 80 < t < 160  # Anomalía entre t=80 y t=160 (aprox. 6-12h)

                if anomaly_active:
                    if anomaly_type == 'heater_stuck':
                        heater = 0.9  # Calefactor encendido siempre
                    elif anomaly_type == 'ventilation_failure':
                        vent = 0.02  # Ventilación mínima
                    elif anomaly_type == 'occupancy_spike':
                        if 100 < t < 120:
                            occ_change = 5  # Entrada masiva de personas
                        else:
                            occ_change = 0

                return heater, vent, occ_change
            return schedule

        schedule = make_anomaly_schedule(default_schedule, anomaly_type)
        data = bld.run_scenario(steps, schedule)

        # Crear máscara de anomalía
        anomaly_mask = np.zeros(len(data) - 1, dtype=bool)

        X_anom, y_anom = [], []
        for t in range(len(data) - 1):
            X_anom.append(data[t])
            y_anom.append(data[t + 1][:3])  # [T, H, CO2]
            # La anomalía afecta de t=80 a t=160 (en pasos de 5 min)
            anomaly_mask[t] = 80 <= t < 160  # Pasos de simulación

        anomalies.append({
            'name': anomaly_type,
            'X': np.array(X_anom),
            'y_true': np.array(y_anom),
            'anomaly_mask': anomaly_mask,
        })

    return anomalies


if __name__ == '__main__':
    print("=" * 60)
    print("GENERACIÓN DE DATOS PARA DIGITAL TWIN")
    print("=" * 60)

    print("\nGenerando escenarios nominales...")
    X_train, y_train = generate_nominal_scenarios(5, 144)
    X_val, y_val = generate_nominal_scenarios(2, 144, seed_base=200)
    print(f"  Train: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} muestras, {X_val.shape[1]} features")

    print("\nGenerando escenarios con anomalías...")
    anomalies = generate_anomaly_scenarios(3, 144)
    for a in anomalies:
        n_anom = a['anomaly_mask'].sum()
        print(f"  {a['name']:20s}: {len(a['X'])} muestras ({n_anom} anómalas)")

    # Guardar datos
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)

    np.savez(os.path.join(data_dir, 'training_data.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(data_dir, 'validation_data.npz'), X=X_val, y=y_val)
    import pickle
    with open(os.path.join(data_dir, 'anomaly_scenarios.pkl'), 'wb') as f:
        pickle.dump(anomalies, f)

    print(f"\n✅ Datos guardados en {data_dir}/")
