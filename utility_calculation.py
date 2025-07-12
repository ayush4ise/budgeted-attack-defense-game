"""
Module to solve the estimated utility functions (using Gurobi)
and obtain results for the simultaneous game
"""
import json
from functools import partial
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from utils import prob_success

def utility_function(allocations, model_path):
    """
    Reading the quadratic model of the format 
        coeff + X coeffs + X^2 coeffs
    and returning the utility value

    Parameters:
    -----------
    allocations : np.array
        Entity's resource allocation to each target. Length must be n.

    model_path : Str
        Path to the utility model JSON file
    """
    with open(model_path, "r", encoding="utf-8") as file:
        coeffs = json.load(file)

    utility = coeffs['intercept']

    for i,_ in enumerate(allocations):
        utility += coeffs['coefficients'][i] * allocations[i] # Adding the Xi term
        # Adding the Xi^2 term
        utility += coeffs['coefficients'][i+len(allocations)] * allocations[i] * allocations[i] 

    return utility

def optimize_utility(N, utility, budget, goal):
    """
    Function to access Gurobi to optimize allocations with a budget

    Parameters:
    -----------
    N : int
        Number of targets

    utility : Function
        Python function which is to be optimized

    budget : int or float
        Entity budget for the allocations

    goal : str
        ['MINIMIZE', 'MAXIMIZE']

    Returns:
    --------
    allocations : array
        Optimal allocations for the entity
    """
    targets = list(range(N))

    # Create a new model
    m = gp.Model("estimated_utility")

    # Variables
    allocations = m.addVars(targets, vtype=GRB.CONTINUOUS, name="allocations")

    # Objective function
    Z = utility(allocations)

    if goal == 'MAXIMIZE':
        m.setObjective(Z, GRB.MAXIMIZE)
    elif goal == 'MINIMIZE':
        m.setObjective(Z, GRB.MINIMIZE)

    # Constraints
    # Budget constraint of entity
    m.addConstr(allocations.sum() - budget == 0, name="budget")

    # Optimize the model
    m.optimize()

    # Output the solver status
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found.")
    else:
        print(f"Optimization ended with status {m.status}.")

    return [v.x for v in m.getVars()]


if __name__ == '__main__':
    # Sets
    num_targets = 5 # Number of targets

    # Parameters
    alpha, beta, A = 1, 1, 0.1
    T_BUDGET = 5 # Attacker's investment (Million USD)
    G_BUDGET = 30 # Defender's Investment (Million USD)

    # Attacker's valuations
    B = np.array([20, 100, 50, 2, 20]) # Infrastructure 6-10 valuations from multiple
    # infrastructure model from the paper
    D = np.array([70, 1000, 50, 75, 150]) # Defender's valuations

    # Defining utility functions for both attacker and defender
    # attacker_utility = partial(utility_function, model_path='models/aGain_model_random100.json')
    # defender_utility = partial(utility_function, model_path='models/dLoss_model_random100.json')
    attacker_utility = partial(utility_function, model_path='models/aGain_model_lhs100.json')
    defender_utility = partial(utility_function, model_path='models/dLoss_model_lhs100.json')

    # Attacker allocations
    T_list = optimize_utility(N=num_targets, utility=attacker_utility, budget=T_BUDGET, goal='MAXIMIZE')
    G_list = optimize_utility(N=num_targets, utility=defender_utility, budget=G_BUDGET, goal='MINIMIZE')

    print("===================================")
    print("Results for Simultaneous Game")
    print("===================================")

    print("Attacker allocations, T:", T_list)
    print("Attacker's Gains (Estimated):", attacker_utility(T_list))
    print("Attacker's Gains (CSF):", np.dot(B, [prob_success(Ti=Ti,Gi=Gi) for Ti,Gi in zip(T_list, G_list)]))

    print("\nDefender allocations, G:", G_list)
    print("Defender's Losses (Estimated):", defender_utility(G_list))
    print("Defender's Losses (CSF):", np.dot(D, [prob_success(Ti=Ti,Gi=Gi) for Ti,Gi in zip(T_list, G_list)]))
