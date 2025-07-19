"""
Module to solve the problem as a sequential game
Algorithm details in README
"""
import logging
from functools import partial
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utility_calculation import utility_function, optimize_utility
from utils import attacker_lingo_model, prob_success

# Setting up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    handlers=[logging.FileHandler('logs/algorithm1.log'), logging.StreamHandler()])

# Parameters
NUM_TARGETS = 5
A_BUDGET = 5 # Attacker's budget
D_BUDGET = 30 # Defender's budget
alpha, beta, A = 1, 1, 0.1

B = np.array([20, 100, 50, 2, 20]) # Attacker's valuations
D = np.array([70, 1000, 50, 75, 150]) # Defender's valuations

MAX_ITERS = 30
ITERATION = 1

# Initialize attacker's and defender's allocations
# Intialized as equal allocatins to all targets
A0 = np.ones(NUM_TARGETS) * A_BUDGET/NUM_TARGETS
D0 = np.ones(NUM_TARGETS) * D_BUDGET/NUM_TARGETS

# Initial defender's utility function
DMODEL_PATH = 'models/dLoss_model_lhs100.json'
EXISTING_DATA = 'data/dLoss_simulation_lhs100.csv'
UPDATED_DATA = 'data/dLoss_updated_lhs100.csv'

# Copying data to a separate CSV file (which we can update)
loss_data = pd.read_csv(EXISTING_DATA)
loss_data.to_csv(UPDATED_DATA)

while ITERATION < MAX_ITERS:
    logging.info('==================== Iteration: %s ====================', ITERATION)

    # Solve the estimated utility for the defender
    defender_utility = partial(utility_function, model_path=DMODEL_PATH)
    D_star = optimize_utility(N=NUM_TARGETS, utility=defender_utility,
                              budget=D_BUDGET, goal='MINIMIZE')
    logging.info('D*: %s', D_star)

    # Solve the actual model for attacker
    # Assuming the attacker knows the defender's allocations
    A_star = attacker_lingo_model(n_targets=NUM_TARGETS, alpha=alpha, beta=beta, A=A,
                                t_budget=A_BUDGET, B_list=B, G_list=D_star)
    logging.info('A*: %s', A_star)

    # Get y_new by getting the defender's loss using D_star
    y_new = np.dot(D, prob_success(Ti=A_star, Gi=D_star))
    logging.info('y_new: %s', y_new)

    # Update dataset and save it to a CSV file
    loss_data.loc[len(loss_data)] = np.hstack((D_star, A_star, [y_new])).ravel()
    logging.info("Dataset Updated! Observation count: %s", len(loss_data))

    # Train new model
    X_1 = np.array(loss_data[['G1','G2','G3','G4','G5']])
    y = loss_data['Z_G']
    # Get quadratic features
    X_2 = X_1**2
    # Final input with quadratic features
    X = np.hstack([X_1, X_2])
    model = LinearRegression()
    model.fit(X,y)

    # Save new model
    DMODEL_PATH = f'models/itermodels/dLoss_lhs_100_iter{ITERATION}.json'
    with open(DMODEL_PATH, "w", encoding="utf-8") as fp:
        json.dump({
        'intercept' : round(model.intercept_,2),
        'coefficients' : [round(coef,2) for coef in model.coef_]
        } , fp)
    logging.info("Model updated and saved!")

    ITERATION+=1 # Update iteration

loss_data.to_csv(UPDATED_DATA)
logging.info("Updated data saved!")
logging.info("Terminated after %s iterations.", ITERATION)
