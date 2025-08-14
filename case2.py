"""
Case 2:
Defender - Does not know the utility function
Attacker - Does not know the utility function, but knows the defender's allocations
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
                    handlers=[logging.FileHandler('logs/case2.log'), logging.StreamHandler()])

def case2(**kwargs):
    """ 
    Case details in README 

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
    defender_ftype = kwargs.get("defender_ftype", "quadratic")
    attacker_ftype = kwargs.get("attacker_ftype", "quadratic_sees")

    # Parameters
    defender_budget = kwargs.get("defender_budget")
    attacker_budget = kwargs.get("attacker_budget")
    defender_valuation = kwargs.get("defender_valuation")
    attacker_valuation = kwargs.get("attacker_valuation")
    alpha, beta, A = kwargs.get("alpha"), kwargs.get("beta"), kwargs.get("A")

    # Reading simulation data for defender's loss
    loss_data = pd.read_csv(f"data/simulations/dLoss_simulation_lhs100_({alpha},{beta},{A}).csv")

    # Define defender's utility instance
    defender_utility = UtilityFunction(function_type=defender_ftype, entity="defender",
                                        n_targets=n_targets, g_budget=defender_budget)
    defender_utility.estimate_function(data=loss_data)

    # Solve for optimal defender's allocation
    d_star = defender_utility.optimize()
    logging.info('D*: %s', d_star)

    if attacker_ftype == "actual":
        # Solve for optimal attacker's allocation (using actual model for sequential game)
        a_star = attacker_lingo_model(n_targets=n_targets, alpha=alpha, beta=beta, A=A,
                                    t_budget=attacker_budget, B_list=attacker_valuation,
                                    G_list=d_star)

    else:
        # Reading simulation data for attacker's loss
        gain_data = pd.read_csv(f"data/simulations/aGain_simulation_lhs100_({alpha},{beta},{A}).csv")

        # Define attacker's utility instance
        attacker_utility = UtilityFunction(function_type=attacker_ftype, entity="attacker",
                                            n_targets=n_targets, t_budget=attacker_budget)
        attacker_utility.estimate_function(data=gain_data,
                                           savepath=f"models/aGain_qnisees_lhs100_({alpha},{beta},{A}).json")

        # Solve for optimal defender's allocation
        a_star = attacker_utility.optimize()

    logging.info('A*: %s', a_star)

    # Get defender's loss using D_star, A_star
    losses = np.dot(defender_valuation, prob_success(Ti=a_star, Gi=d_star,
                                                    alpha=alpha, beta=beta, Ai=A))
    logging.info('Losses: %s', losses)

    # Also, calculate attacker's gains using D_star, A_star
    gains = np.dot(attacker_valuation, prob_success(Ti=a_star, Gi=d_star,
                                    alpha=alpha, beta=beta, Ai=A))
    logging.info('Gains: %s', gains)

    return {
        "Defender's allocations" : d_star,
        "Attacker's allocations" : a_star,
        "Defender's Losses"      : losses,
        "Attacker's Gains"       : gains
    }

if __name__ == '__main__':
    # Parameters
    NUM_TARGETS = 5
    A_BUDGET = 5 # Attacker's budget
    D_BUDGET = 30 # Defender's budget
    ALPHA, BETA, A = 1, 1, 0.1

    B = np.array([20, 100, 50, 2, 20]) # Attacker's valuations
    D = np.array([70, 1000, 50, 75, 150]) # Defender's valuations

    MAX_ITERS = 30

    DEFENDER_FTYPE = "quadratic"
    ATTACKER_FTYPE = "quadratic_sees"
    results = case2(n_targets=NUM_TARGETS,
                         defender_ftype=DEFENDER_FTYPE,attacker_ftype=ATTACKER_FTYPE,
                         attacker_budget=A_BUDGET,defender_budget=D_BUDGET,
                         attacker_valuation=B,defender_valuation=D,
                         alpha=ALPHA,beta=BETA,A=A)
    print(results)
