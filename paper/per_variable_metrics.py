#!/usr/bin/env python3
"""Obtiene métricas por variable para la tabla comparativa del paper."""
import numpy as np, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from src.generate_data import generate_nominal_scenarios
from src.gp_digital_twin import GPDigitalTwin

X_train, y_train = generate_nominal_scenarios(8, 144)
X_val, y_val = generate_nominal_scenarios(2, 144, seed_base=200)

methods = {
    'Linear Regression': lambda: LinearRegression(),
    'Random Forest': lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'MLP (NN)': lambda: MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42, early_stopping=True),
}

print(f"{'Variable':12s} {'Métrica':10s}", end='')
for m in methods: print(f'{m:>20s}', end='')
print(f'{"GP-CPS":>20s}')
print('-' * 82)

for var_i, var_name in enumerate(['Temperatura', 'Humedad', 'CO2']):
    for metric_name, metric_fn in [('R²', r2_score), ('MAE', mean_absolute_error)]:
        print(f'{var_name:12s} {metric_name:10s}', end='')
        
        for m_name, m_fn in methods.items():
            model = m_fn()
            sx = StandardScaler(); sy = StandardScaler()
            model.fit(sx.fit_transform(X_train), sy.fit_transform(y_train[:, var_i].reshape(-1,1)).ravel())
            yp = sy.inverse_transform(model.predict(sx.transform(X_val)).reshape(-1,1)).ravel()
            val = metric_fn(y_val[:, var_i], yp)
            print(f'{val:>20.4f}', end='')
        
        # GP
        dt = GPDigitalTwin('matern')
        dt.fit(X_train, y_train, verbose=False)
        yp, _ = dt.predict(X_val)
        val = metric_fn(y_val[:, var_i], yp[:, var_i])
        print(f'{val:>20.4f}')
    
    print()

# Average
print(f'{"Media":12s} {"R²":10s}', end='')
for m_name, m_fn in methods.items():
    r2s = []
    for i in range(3):
        model = m_fn()
        sx = StandardScaler(); sy = StandardScaler()
        model.fit(sx.fit_transform(X_train), sy.fit_transform(y_train[:, i].reshape(-1,1)).ravel())
        yp = sy.inverse_transform(model.predict(sx.transform(X_val)).reshape(-1,1)).ravel()
        r2s.append(r2_score(y_val[:, i], yp))
    print(f'{np.mean(r2s):>20.4f}', end='')

dt = GPDigitalTwin('matern')
dt.fit(X_train, y_train, verbose=False)
yp, _ = dt.predict(X_val)
gp_r2s = [r2_score(y_val[:, i], yp[:, i]) for i in range(3)]
print(f'{np.mean(gp_r2s):>20.4f}')
