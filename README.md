# AdMaths-GP-DigitalTwin

**Digital Twin para Cyber-Physical Systems usando Gaussian Processes**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descripción

Este proyecto implementa un **Digital Twin** (gemelo digital) para sistemas ciber-físicos (CPS) usando **Gaussian Processes (GP)** como modelo generativo, en lugar del costoso Diffusion Model del paper original.

### Diferencias con el paper original

| Aspecto | Paper original (Diffusion_DigitalTwin) | Este proyecto |
|---------|--------------------------------------|---------------|
| **Sistema** | Brazo robótico antropomórfico | **Smart Building Zone** (temperatura, humedad, CO2, ocupación) |
| **Modelo generativo** | Diffusion Model (Rectified Flows) | **Gaussian Process** (conceptualmente alineado con NNGP) |
| **Entrenamiento** | GPU, horas | **CPU, minutos** |
| **Incertidumbre** | No cuantifica | **Sí** — el GP da intervalos de confianza |
| **Detección anomalías** | Por umbral en residual | **Por incertidumbre del GP** (score en desviaciones estándar) |
| **Conexión TFM** | Indirecta | **Directa**: NNGP ↔ GP ↔ Digital Twin |

## 🔬 Conexión con el TFM (NNGP/NTK)

El teorema **NNGP** (Neural Network Gaussian Process) establece que una red neuronal de ancho infinito equivale a un **Gaussian Process** con un kernel determinado por la arquitectura. Este proyecto aplica esa idea:

1. **GP como modelo generativo** del sistema CPS
2. **El kernel aprendido** revela la estructura de correlaciones entre variables
3. **NKA (Normalized Kernel Alignment)** como métrica de calidad
4. La **incertidumbre del GP** permite detección de anomalías

## 🚀 Instalación y Ejecución

```bash
# Clonar
git clone https://github.com/jcalataiut-cli/AdMaths-GP-DigitalTwin.git
cd AdMaths-GP-DigitalTwin

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
python pipeline.py
```

### Dependencias

- `numpy`, `scipy` — Computación numérica
- `scikit-learn` — Gaussian Process Regressor
- `matplotlib`, `seaborn` — Visualización

## 📁 Estructura del proyecto

```
AdMaths-GP-DigitalTwin/
├── pipeline.py              # Pipeline completo (ejecutar todo)
├── README.md
├── requirements.txt
├── src/
│   ├── physical_model.py    # Simulación del sistema físico (ground truth)
│   ├── generate_data.py     # Generación de datos sintéticos
│   ├── gp_digital_twin.py   # Digital Twin basado en GP
│   └── visualize.py         # Visualización de resultados
├── notebooks/
│   └── (próximamente)
├── data/                    # Datos generados (no versionados)
└── results/                 # Figuras generadas
```

## 🧪 Sistema: Smart Building Zone

Modelamos una **zona de edificio inteligente** como CPS con 4 variables:

| Variable | Rango típico | Descripción |
|----------|-------------|-------------|
| **Temperatura** | 15–35 °C | Controlada por calefactor |
| **Humedad relativa** | 10–90 % | Afectada por ocupación y ventilación |
| **CO2** | 400–5000 ppm | Indicador de calidad del aire |
| **Ocupación** | 0–5 personas | Variable discreta que afecta a las demás |

### Dinámica del sistema

```
dT/dt = f(heater, T_out, occupancy, ventilation)
dH/dt = f(occupancy, ventilation, materials)
dCO2/dt = f(occupancy, ventilation)
```

## 📊 Resultados esperados

Al ejecutar `pipeline.py` se generan estas figuras en `results/`:

| Figura | Descripción |
|--------|-------------|
| `fig01_training_data.png` | Datos de entrenamiento del sistema físico |
| `fig02_prediction_vs_actual.png` | Predicción del DT vs valores reales |
| `fig03_kernel_analysis.png` | Análisis de kernels (conexión NNGP) |
| `fig04_anomaly_detection.png` | Detección de anomalías con GP |
| `fig05_trajectory_simulation.png` | Simulación de trayectoria |
| `fig06_dashboard.png` | Dashboard resumen de rendimiento |

## 🔍 Mejoras sobre el paper original

1. **Sistema diferente**: Smart Building Zone reemplaza al brazo robótico
2. **Modelo ejecutable localmente**: GP entrena en CPU en minutos (vs horas GPU)
3. **Incertidumbre calibrada**: El GP da intervalos de confianza (95% CI)
4. **Detección de anomalías**: Por desviación estándar del GP (interpretable)
5. **Conexión teórica**: Alineado con NNGP/NTK del TFM
6. **Originalidad**: Aporta el uso de GP como modelo generativo para DT de CPS

## 📚 Referencias

- Lee et al. (2018) — Deep Neural Networks as Gaussian Processes
- Jacot et al. (2018) — Neural Tangent Kernel
- Yang (2020) — Tensor Programs I
- Diffusion_DigitalTwin (2026) — Paper original analizado
- Rasmussen & Williams (2006) — Gaussian Processes for Machine Learning

## 🧑‍🎓 Contexto

Proyecto desarrollado como parte del TFM **"Generalización del comportamiento NNGP/NTK a redes neuronales arbitrarias"**.
