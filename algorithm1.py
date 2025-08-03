"""
Module to solve the problem as a sequential (or simultaneous) game
Algorithm details in README
"""

import logging
import numpy as np
import pandas as pd
from utility_function import UtilityFunction
from utils import attacker_lingo_model, prob_success

# Setting up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    handlers=[logging.FileHandler('logs/algorithm1.log'), logging.StreamHandler()])

def algorithm1(**kwargs):
    """
    Function to run the algorithm 1 (Details in README)

    Parameters
    ----------
    n_targets : int
        Number of targets.
    max_iters : int
        Number of maximum iterations.
    defender_ftype : str
        Defender's utility function type
    attacker_budget : float or int
        Total budget available to the attacker.
    defender_budget : float or int
        Total budget available to the defender.
    attacker_valuation : np.array
        Attacker's valuation (importance) of each target. Length must be n_targets.
    defender_valuation : np.array
        Defender's valuation (importance) of each target. Length must be n_targets.
    alpha : float
        Defender's influence parameter in the success probability function.
    beta : float
        Attacker's influence parameter in the success probability function.
    A : float
        Inherent defense level of the targets.
    datapath :str
        Path to the data used to estimate defender utility function.
    updatepath :str
        Path to save the updated data after all the iterations.    

    Returns
    -------
    dict
        Dictionary containing:
        - "Defender's allocations" : np.array
        - "Attacker's allocations" : np.array
        - "Defender's Losses"      : float,
        - "Attacker's Gains"       : float
    """
    n_targets = kwargs.get("n_targets")
    max_iters = kwargs.get("max_iters")
    defender_ftype = kwargs.get("defender_ftype")

    # Parameters
    defender_budget = kwargs.get("defender_budget")
    attacker_budget = kwargs.get("attacker_budget")
    defender_valuation = kwargs.get("defender_valuation")
    attacker_valuation = kwargs.get("attacker_valuation")
    alpha, beta, A = kwargs.get("alpha"), kwargs.get("beta"), kwargs.get("A")

    loss_data = pd.read_csv(kwargs.get("datapath")) # Reading simulation data for defender's loss

    # Define defender's utility instance
    defender_utility = UtilityFunction(function_type=defender_ftype, entity="defender",
                                        n_targets=n_targets, g_budget=defender_budget)
    defender_utility.estimate_function(data=loss_data)
    # Do the same for attacker, if solving simultaneous problem with estimated attacker utility

    iteration = 1
    while iteration <= max_iters:
        logging.info('==================== Iteration: %s ====================', iteration)

        # Solve for optimal defender's allocation
        d_star = defender_utility.optimize()
        logging.info('D*: %s', d_star)

        # Solve for optimal attacker's allocation (using actual model for sequential game)
        a_star = attacker_lingo_model(n_targets=n_targets, alpha=alpha, beta=beta, A=A,
                                    t_budget=attacker_budget, B_list=attacker_valuation,
                                    G_list=d_star)
        logging.info('A*: %s', a_star)

        # Get y_new by getting the defender's loss using D_star, A_star
        y_new = np.dot(defender_valuation, prob_success(Ti=a_star, Gi=d_star,
                                                        alpha=alpha, beta=beta, Ai=A))
        logging.info('y_new: %s', y_new)

        # Also, calculate attacker's gains using D_star, A_star
        gains = np.dot(attacker_valuation, prob_success(Ti=a_star, Gi=d_star,
                                      alpha=alpha, beta=beta, Ai=A))
        logging.info('Gains: %s', gains)

        # Update dataset
        loss_data.loc[len(loss_data)] = np.hstack((d_star, a_star, [y_new])).ravel()
        logging.info("Dataset Updated! Observation count: %s", len(loss_data))

        # Retrain the model
        defender_utility.estimate_function(data=loss_data, 
                    savepath=f"models/itermodels/dLoss_iter{iteration}_({alpha},{beta},{A}).json")
        logging.info("Model updated and saved!")
        iteration+=1 # Update iteration
    loss_data.to_csv(kwargs.get("updatepath"))
    logging.info("Updated data saved!")
    logging.info("Terminated after %s iterations.", iteration-1)

    return {
        "Defender's allocations" : d_star,
        "Attacker's allocations" : a_star,
        "Defender's Losses"      : y_new,
        "Attacker's Gains"       : gains
    }


if __name__ == "__main__":
    # Parameters
    NUM_TARGETS = 5
    A_BUDGET = 5 # Attacker's budget
    D_BUDGET = 30 # Defender's budget
    ALPHA, BETA, A = 1, 1, 0.1

    B = np.array([20, 100, 50, 2, 20]) # Attacker's valuations
    D = np.array([70, 1000, 50, 75, 150]) # Defender's valuations

    MAX_ITERS = 30

    DEFENDER_FTYPE = "quadratic"

    EXISTING_DATA = 'data/dLoss_simulation_lhs100_(1,1,0.1).csv'
    UPDATED_DATA = 'data/dLoss_updated_lhs100_(1,1,0.1).csv'

    results = algorithm1(n_targets=NUM_TARGETS,
                         max_iters=MAX_ITERS,
                         defender_ftype=DEFENDER_FTYPE,
                         attacker_budget=A_BUDGET,defender_budget=D_BUDGET,
                         attacker_valuation=B,defender_valuation=D,
                         alpha=ALPHA,beta=BETA,A=A,
                         datapath=EXISTING_DATA,updatepath=UPDATED_DATA)
    print(results)
