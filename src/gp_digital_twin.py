"""
gp_digital_twin.py — Digital Twin basado en Gaussian Processes
================================================================

En lugar de entrenar un Diffusion Model (costoso), usamos un Gaussian
Process como modelo generativo del sistema CPS. Esto está conceptualmente
alineado con el teorema NNGP del TFM: una red neuronal de ancho infinito
equivale a un GP.

Arquitectura del Digital Twin:
  - Un GP por variable de salida (Temperature, Humidity, CO2)
  - Features: [T(t), H(t), CO2(t), occ(t), heater(t), vent(t)]
  - Targets: [T(t+1), H(t+1), CO2(t+1)]
  - La incertidumbre del GP permite detección de anomalías
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel, DotProduct
)
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class GPDigitalTwin:
    """
    Digital Twin basado en Gaussian Process para un sistema CPS.

    El DT aprende la dinámica del sistema a partir de datos y puede:
    1. Predecir la evolución del sistema (simulación)
    2. Cuantificar incertidumbre en las predicciones
    3. Detectar anomalías (cuando el error > 2σ de incertidumbre)
    4. Generar trayectorias realistas muestreando de la posterior
    """

    def __init__(self, kernel_type='matern'):
        """
        Args:
            kernel_type: Tipo de kernel ('matern', 'rbf', 'linear')
        """
        self.kernel_type = kernel_type
        self.models = {}      # Un GP por variable de salida
        self.scaler_X = StandardScaler()
        self.scaler_y = {}    # Un scaler por variable
        self.feature_names = ['T(t)', 'H(t)', 'CO2(t)', 'occ(t)', 'heater(t)', 'vent(t)']
        self.target_names = ['T(t+1)', 'H(t+1)', 'CO2(t+1)']

    def _build_kernel(self):
        """Construye el kernel del GP según el tipo."""
        base_kernels = {
            'matern': Matern(length_scale=1.0, nu=1.5),
            'rbf': RBF(length_scale=1.0),
            'linear': DotProduct() + WhiteKernel(),
        }
        base = base_kernels.get(self.kernel_type, Matern(nu=1.5))
        return ConstantKernel(1.0) * base + WhiteKernel(0.1)

    def fit(self, X, y, verbose=True):
        """
        Entrena el Digital Twin.

        Args:
            X: Features (N, 6) — [T, H, CO2, occ, heater, vent]
            y: Targets (N, 3) — [T(t+1), H(t+1), CO2(t+1)]
        """
        # Estandarizar features
        X_scaled = self.scaler_X.fit_transform(X)

        for i, name in enumerate(self.target_names):
            target = y[:, i]

            # Estandarizar target
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()

            self.scaler_y[i] = scaler

            # Crear y entrenar GP
            kernel = self._build_kernel()
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                alpha=1e-6,
                normalize_y=False,
                random_state=42 + i,
            )
            gp.fit(X_scaled, y_scaled)

            self.models[name] = gp

            if verbose:
                print(f"  GP para {name:10s} → log-marginal-likelihood: {gp.log_marginal_likelihood():.1f}")

    def predict(self, X, return_std=True):
        """
        Predice el siguiente estado del sistema.

        Args:
            X: Input features (N, 6)
            return_std: Si incluir desviación estándar

        Returns:
            y_mean: Media de la predicción (N, 3)
            y_std: Desviación estándar (N, 3) si return_std=True
        """
        X_scaled = self.scaler_X.transform(X)
        means, stds = [], []

        for i, name in enumerate(self.target_names):
            mean, std = self.models[name].predict(X_scaled, return_std=True)

            # Desescalar
            scaler = self.scaler_y[i]
            mean = scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
            std = std * scaler.scale_[0]

            means.append(mean)
            stds.append(std)

        y_mean = np.column_stack(means)
        y_std = np.column_stack(stds) if return_std else None

        return (y_mean, y_std) if return_std else y_mean

    def simulate(self, initial_state, controls, n_steps):
        """
        Simula una trayectoria completa con el Digital Twin.

        Args:
            initial_state: Estado inicial [T, H, CO2, occ]
            controls: Array de controles (n_steps, 2) — [heater, vent]
            n_steps: Número de pasos

        Returns:
            trajectory: (n_steps, 3) — [T, H, CO2] predichos
            uncertainty: (n_steps, 3) — desviación estándar
        """
        state = np.array(initial_state, dtype=float)
        trajectory, uncertainty = [], []

        for t in range(min(n_steps, len(controls))):
            # Construir feature vector
            X_t = np.concatenate([state, controls[t]]).reshape(1, -1)

            # Predecir
            y_pred, y_std = self.predict(X_t)
            trajectory.append(y_pred[0])
            uncertainty.append(y_std[0])

            # Actualizar estado (solo T, H, CO2; occ se mantiene)
            state[:3] = y_pred[0]
            state[3] = controls[t, 1] * 5 if t > 0 else state[3]  # occ aprox

        return np.array(trajectory), np.array(uncertainty)

    def detect_anomalies(self, X, y_observed, threshold=2.0):
        """
        Detecta anomalías: cuando el error de predicción excede
        el umbral × desviación estándar del GP.

        Args:
            X: Input features (N, 6)
            y_observed: Estado observado (N, 3)
            threshold: Umbral de desviaciones estándar

        Returns:
            anomalies: Máscara booleana (N,)
            scores: Puntuación de anomalía por muestra (N,)
        """
        y_pred, y_std = self.predict(X)

        # Error normalizado (número de desviaciones estándar)
        error = np.abs(y_observed - y_pred)
        z_scores = error / (y_std + 1e-10)

        # Puntuación agregada (media de z-scores sobre variables)
        scores = np.mean(z_scores, axis=1)

        return scores > threshold, scores

    def save(self, path):
        """Guarda el modelo."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Carga el modelo."""
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    print("=" * 60)
    print("DIGITAL TWIN: Gaussian Process")
    print("=" * 60)

    # Cargar datos de entrenamiento
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    train = np.load(os.path.join(data_dir, 'training_data.npz'))
    X_train, y_train = train['X'], train['y']

    print(f"\nDatos de entrenamiento: {X_train.shape}")
    print(f"Features: {GPDigitalTwin().feature_names}")
    print(f"Targets:  {GPDigitalTwin().target_names}")

    # Entrenar Digital Twin
    print("\nEntrenando Digital Twin...")
    dt = GPDigitalTwin(kernel_type='matern')
    dt.fit(X_train, y_train)

    # Guardar modelo
    model_path = os.path.join(data_dir, 'digital_twin_gp.pkl')
    dt.save(model_path)
    print(f"\n✅ Digital Twin guardado en {model_path}")

    # Evaluación rápida
    val = np.load(os.path.join(data_dir, 'validation_data.npz'))
    X_val, y_val = val['X'], val['y']
    y_pred, y_std = dt.predict(X_val)

    mae = np.mean(np.abs(y_val - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_val - y_pred)**2, axis=0))

    print(f"\nEvaluación en validación:")
    for i, name in enumerate(dt.target_names):
        print(f"  {name:10s}: MAE={mae[i]:.3f}, RMSE={rmse[i]:.3f}")
