"""
Module to carry out simulations to generate attacker's allocations 
based on defender's allocations (attacker's optimization problem)
"""
import numpy as np
import pandas as pd
from utils import attacker_lingo_model, prob_success

np.random.seed(0) # Initial seed to generate 20 further seeds
seeds = np.random.randint(1,100,20) # List of 20 seeds,
# each for a data point, to reproduce same results everytime

# Defining parameters
N_TARGETS = 5 # For simulation purposes
alpha, beta, A = 1, 1, 0.1

# Values taken in examples in the paper
T_BUDGET = 5
G_BUDGET = 30

B = np.array([20, 100, 50, 2, 20]) # Infrastructure 6-10 valuations from multiple
# infrastructure model from the paper
D = np.array([70, 1000, 50, 75, 150])

# To store calculated values
combined_values = []

for seed in seeds:
    np.random.seed(seed)
    G = np.random.rand(N_TARGETS)
    # Getting a random array of defender allocations, G, with sum = G_BUDGET
    G = G_BUDGET * G / sum(G) 

    T = attacker_lingo_model(
        n_targets=N_TARGETS,
        alpha=alpha,
        beta=beta,
        A=A,
        t_budget=T_BUDGET,
        B_list=B,
        G_list=G
    )

    utility = np.dot(D, [prob_success(alpha,beta,Ti,Gi,A) for Ti,Gi in zip(T,G)])
    print('Z_g:', utility)

    combined_values.append(list(G) + list(T) + [utility])

# Create column names
column_names = [f'G{i}' for i in range(1,N_TARGETS+1)] + [f'T_best{i}' for i in range(1,N_TARGETS+1)] + ['Z_G']

# Create a dataframe
df = pd.DataFrame(combined_values, columns=column_names)
df.to_csv('results/attacker_best_simulation.csv', index=False)
