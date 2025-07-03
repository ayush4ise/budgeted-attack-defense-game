"""
Module to carry out simulations to optimmize attacker's allocations 
and observe defener's losses
based on simulated defender's allocations (attacker's optimization problem)
"""
import numpy as np
import pandas as pd
from utils import attacker_lingo_model, prob_success

np.random.seed(0) # Initial seed to generate 20 further seeds

# Defining parameters
N_TARGETS = 5 # For simulation purposes
N_SAMPLES = 100

# Values taken in examples in the paper
T_BUDGET = 5
G_BUDGET = 30
alpha, beta, A = 1, 1, 0.1

# Attacker's valuations
B = np.array([20, 100, 50, 2, 20]) # Infrastructure 6-10 valuations from multiple
# infrastructure model from the paper
D = np.array([70, 1000, 50, 75, 150]) # Defender's valuations

# To store calculated values
combined_values = []

# Generate a (N_SAMPLES * N_TARGETS) size array with random numbers
allocations = np.random.rand(N_SAMPLES, N_TARGETS)

for allocation in allocations:
    G = allocation
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
df.to_csv(f'results/dLoss_simulation_random{N_SAMPLES}.csv', index=False)
