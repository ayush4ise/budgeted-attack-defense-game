"""
Module to carry out simulations to optimize defender's allocations 
and observe attacker's gains
based on simulated attacker's allocations (defender's optimization problem)
"""
import numpy as np
import pandas as pd
from pyDOE3 import lhs
from utils import defender_lingo_model, prob_success

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

# # Generate a (N_SAMPLES * N_TARGETS) size array with random numbers
# allocations = np.random.rand(N_SAMPLES, N_TARGETS)

# Latin Hypercube Sampling (LHS) to generate a (N_SAMPLES * N_TARGETS) size array
# Inspired from: https://pydoe3.readthedocs.io/en/latest/randomized.html#randomized
allocations = lhs(N_TARGETS, samples=N_SAMPLES, criterion='maximin')

for allocation in allocations:
    T = allocation
    # Getting a random array of attacker allocations, T, with sum = T_BUDGET
    T = T_BUDGET * T / sum(T)

    G = defender_lingo_model(
        n_targets=N_TARGETS,
        alpha=alpha,
        beta=beta,
        A=A,
        g_budget=G_BUDGET,
        D_list=D,
        T_list=T
    )

    utility = np.dot(B, [prob_success(alpha,beta,Ti,Gi,A) for Ti,Gi in zip(T,G)])
    print('Z_t:', utility)

    combined_values.append(list(T) + list(G) + [utility])

# Create column names
column_names = [f'T{i}' for i in range(1,N_TARGETS+1)] + [f'G_best{i}' for i in range(1,N_TARGETS+1)] + ['Z_T']

# Create a dataframe
df = pd.DataFrame(combined_values, columns=column_names)
# df.to_csv(f'results/aGain_simulation_random{N_SAMPLES}.csv', index=False)
df.to_csv(f'results/aGain_simulation_lhs{N_SAMPLES}.csv', index=False)
