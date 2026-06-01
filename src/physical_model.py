"""
physical_model.py — Simulación del sistema físico real (ground truth)
=====================================================================

Modela un sistema de control ambiental (Smart Building Zone) como un
CPS (Cyber-Physical System) con múltiples variables inter-dependientes.

A diferencia del paper original (brazo robótico), este sistema representa
una *zona de edificio inteligente* con sensores de temperatura, humedad,
CO2 y ocupación — un CPS igualmente válido pero más accesible.

Ecuaciones del modelo (simplificadas pero realistas):
  dT/dt = (α·P_heater - β·(T - T_out) - γ·occ·(T - T_body) - δ·vent·(T - T_in)) / C_thermal
  dH/dt = (η·occ·H_body - θ·vent·(H - H_out) - λ·(H - H_ideal)) / C_humidity
  dC/dt = (μ·occ·C_body - ν·vent·(C - C_out)) / C_co2
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ============================================================
# Parámetros del sistema físico
# ============================================================
@dataclass
class BuildingParams:
    """Parámetros físicos de la zona del edificio."""
    # Capacidades térmicas
    C_thermal: float = 100.0   # Capacidad calorífica (kJ/K)
    C_humidity: float = 50.0   # Capacidad de humedad
    C_co2: float = 80.0        # Capacidad de CO2

    # Coeficientes de transferencia
    alpha: float = 0.8    # Eficiencia del calefactor
    beta: float = 0.05    # Pérdida térmica al exterior
    gamma: float = 0.02   # Transferencia térmica por ocupante
    delta: float = 0.03   # Pérdida por ventilación

    # Coeficientes de humedad
    eta: float = 0.04     # Producción de humedad por ocupante
    theta: float = 0.06   # Renovación por ventilación
    lambda_: float = 0.01 # Absorción de humedad (materiales)

    # Coeficientes de CO2
    mu: float = 0.005     # Producción de CO2 por ocupante (L/min)
    nu: float = 0.05      # Renovación de CO2 por ventilación

    # Referencias
    T_body: float = 33.0  # Temperatura corporal (°C)
    H_body: float = 50.0  # Humedad relativa exhalada (%)
    C_body: float = 40000.0  # CO2 exhalado (ppm)
    T_out: float = 15.0   # Temperatura exterior (°C)
    H_out: float = 60.0   # Humedad exterior (%)
    C_out: float = 420.0  # CO2 exterior (ppm)
    H_ideal: float = 50.0 # Humedad objetivo (%)


class BuildingPhysicalSystem:
    """
    Simulación del sistema físico real de una zona de edificio.
    Sirve como "ground truth" — el sistema real del cual tomamos datos.
    """

    def __init__(self, params: Optional[BuildingParams] = None, dt: float = 1.0, seed: int = 42):
        self.params = params or BuildingParams()
        self.dt = dt  # Paso de tiempo (minutos)
        self.rng = np.random.RandomState(seed)

        # Estado inicial
        self.state = {
            'temperature': 21.0,   # °C
            'humidity': 45.0,      # % HR
            'co2': 500.0,          # ppm
            'occupancy': 0,        # número de personas
        }
        self.t = 0.0
        self.history = []

    def step(self, heater_power: float, ventilation: float,
             occupancy_change: Optional[int] = None) -> dict:
        """
        Avanza un paso de simulación.

        Args:
            heater_power: Potencia del calefactor [0, 1]
            ventilation: Tasa de ventilación [0, 1]
            occupancy_change: Cambio en ocupación (+1 entra, -1 sale, None = sin cambio)

        Returns:
            dict: Nuevo estado del sistema
        """
        p = self.params
        s = self.state

        # Actualizar ocupación
        if occupancy_change is not None:
            s['occupancy'] = max(0, s['occupancy'] + occupancy_change)

        occ = s['occupancy']
        T, H, C = s['temperature'], s['humidity'], s['co2']

        # Ruido del sistema (incertidumbre física real)
        noise_T = self.rng.randn() * 0.05
        noise_H = self.rng.randn() * 0.3
        noise_C = self.rng.randn() * 5.0

        # --- Temperatura ---
        dT = (p.alpha * heater_power * 5.0          # Aporte del calefactor
              - p.beta * (T - p.T_out)               # Pérdida al exterior
              - p.gamma * occ * (T - p.T_body)       # Transferencia con personas
              - p.delta * ventilation * (T - 18.0))  # Pérdida por ventilación
        dT = dT / p.C_thermal + noise_T

        # --- Humedad ---
        dH = (p.eta * occ * (p.H_body - H)            # Aporte de personas
              - p.theta * ventilation * (H - p.H_out)  # Renovación
              - p.lambda_ * (H - p.H_ideal))           # Absorción materiales
        dH = dH / p.C_humidity + noise_H

        # --- CO2 ---
        dC = (p.mu * occ * p.C_body                    # Producción por personas
              - p.nu * ventilation * (C - p.C_out))    # Renovación
        dC = dC / p.C_co2 + noise_C

        # Actualizar estado (con límites realistas)
        s['temperature'] = np.clip(T + dT * self.dt, 15, 35)
        s['humidity'] = np.clip(H + dH * self.dt, 10, 90)
        s['co2'] = np.clip(C + dC * self.dt, 400, 5000)

        self.t += self.dt

        # Registrar en historial
        entry = {
            't': self.t,
            'temperature': s['temperature'],
            'humidity': s['humidity'],
            'co2': s['co2'],
            'occupancy': occ,
            'heater_power': heater_power,
            'ventilation': ventilation,
        }
        self.history.append(entry)
        return entry

    def run_scenario(self, steps: int, schedule_fn) -> np.ndarray:
        """
        Ejecuta un escenario completo.

        Args:
            steps: Número de pasos de simulación
            schedule_fn: Función que recibe (t, state) y devuelve
                        (heater_power, ventilation, occupancy_change)

        Returns:
            Array de historial (steps × 6): [T, H, CO2, occ, heater, vent]
        """
        self.history = []
        self.t = 0.0
        self.state = {'temperature': 21.0, 'humidity': 45.0,
                      'co2': 500.0, 'occupancy': 0}

        for _ in range(steps):
            heater, vent, occ_change = schedule_fn(self.t, self.state)
            self.step(heater, vent, occ_change)

        return np.array([[e['temperature'], e['humidity'], e['co2'],
                          e['occupancy'], e['heater_power'], e['ventilation']]
                         for e in self.history])

    def get_observation(self, state_entry: dict) -> np.ndarray:
        """Devuelve el vector de observación sensorial."""
        return np.array([
            state_entry['temperature'],
            state_entry['humidity'],
            state_entry['co2'],
            state_entry['occupancy'],
        ])


def default_schedule(t: float, state: dict) -> tuple:
    """Escenario por defecto: actividad diaria en una oficina."""
    hour = (t / 60) % 24  # t en minutos, convertimos a hora del día

    # Ocupación: gente entra a las 8am, pausa a las 12, salen a las 6pm
    if 8 <= hour < 12:
        occ_change = 1 if state['occupancy'] == 0 else 0
        occ_target = 4
    elif 12 <= hour < 13:
        occ_change = -1 if np.random.rand() < 0.25 else 0
        occ_target = 2
    elif 14 <= hour < 18:
        occ_change = 1 if state['occupancy'] < 2 and np.random.rand() < 0.1 else 0
        occ_target = 3
    elif 18 <= hour < 19:
        occ_change = -1 if state['occupancy'] > 0 and np.random.rand() < 0.3 else 0
        occ_target = 1
    elif hour >= 19 or hour < 7:
        occ_change = -1 if state['occupancy'] > 0 and np.random.rand() < 0.1 else 0
        occ_target = 0
    else:
        occ_change = 0
        occ_target = state.get('occupancy', 0)

    # Calefactor: se activa si T < 21°, se apaga si T > 23°
    T = state['temperature']
    heater = 0.7 if T < 21.0 else (0.3 if T < 22.0 else 0.0)
    if occ_target > 0:
        heater = max(heater, 0.3)  # Mínimo para confort

    # Ventilación: más cuando hay gente
    vent = 0.1 + 0.2 * state['occupancy'] / 4.0
    if hour < 7 or hour >= 22:
        vent = 0.05  # Mínima por la noche

    # Añadir variabilidad realista
    vent *= 0.9 + 0.2 * np.random.rand()

    return heater, vent, occ_change


if __name__ == '__main__':
    # Demo: ejecutar escenario de 24 horas
    print("=" * 60)
    print("SIMULACIÓN FÍSICA: Zona de edificio inteligente")
    print("=" * 60)

    bld = BuildingPhysicalSystem(dt=5.0)  # pasos de 5 minutos
    data = bld.run_scenario(288, default_schedule)  # 24h × 12 pasos/hora

    print(f"\nGenerados {len(data)} pasos de simulación")
    print(f"\nPrimeras 5 muestras:")
    headers = ['T(°C)', 'H(%)', 'CO2(ppm)', 'Occ', 'Heater', 'Vent']
    print(f"{'':>4}  {'  '.join(f'{h:>8}' for h in headers)}")
    for i in range(5):
        print(f"{i:>4}  {'  '.join(f'{v:>8.2f}' for v in data[i])}")

    print(f"\nResumen estadístico:")
    print(f"  Temperatura: {data[:,0].min():.1f}°C - {data[:,0].max():.1f}°C  (media {data[:,0].mean():.1f})")
    print(f"  Humedad:     {data[:,1].min():.1f}% - {data[:,1].max():.1f}%  (media {data[:,1].mean():.1f})")
    print(f"  CO2:         {data[:,2].min():.0f}ppm - {data[:,2].max():.0f}ppm  (media {data[:,2].mean():.0f})")
    print(f"  Ocupación:   {data[:,3].min():.0f} - {data[:,3].max():.0f} personas")
