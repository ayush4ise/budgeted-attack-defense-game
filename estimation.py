"""
Module to develop estimator functions for the defender's losses 
and attacker's gain using the simulated datasets
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# filepath = "data/aGain_simulation_random100.csv"
# savepath = "models/aGain_model_random100.json"
filepath = "data/aGain_simulation_lhs100.csv"
savepath = "models/aGain_model_lhs100.json"
data = pd.read_csv(filepath)

# T - Attacker allocations, Z_T - Attacker Gains
# G - Defender allocations, Z_G - Defender Losses

# Defining X and y for regression
# Degree 1 features
# X_1 = np.array(data[['G1','G2','G3','G4','G5']])
# y = data['Z_G']
X_1 = np.array(data[['T1','T2','T3','T4','T5']])
y = data['Z_T']

# Get quadratic features
X_2 = X_1**2

# Final input with quadratic features
X = np.hstack([X_1, X_2])

# Fitting a regression model
model = LinearRegression()
print("Fitting the model!")
model.fit(X,y)

print('Intercept:', round(model.intercept_,2))
print('Coefficients:', [round(coef,2) for coef in model.coef_])

print("Saving the model!")
with open(savepath, "w", encoding="utf-8") as fp:
    json.dump({
    'intercept' : round(model.intercept_,2),
    'coefficients' : [round(coef,2) for coef in model.coef_]
    } , fp)
