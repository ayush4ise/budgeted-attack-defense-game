"""
Module to generate plots for different algorithms by varying parameters (alpha, beta, A)
"""
import logging
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import game_lingo_model, greedy_allocation
from algorithm1 import algorithm1
from case1 import case1
from case2 import case2
from case4 import case4

# Setting up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    handlers=[logging.FileHandler('logs/plot_generator.log'),
                              logging.StreamHandler()])

def algorithm_str(algorithm):
    """
    Function to take string as input and return respective algorithm

    Available algorithms
    --------------------
    "actual_sequential" : Sequential Game LINGO model

    "actual_simultaneous" : Simultaneous Game LINGO model
    """

    if algorithm == "actual_sequential":
        return partial(game_lingo_model, game_type="sequential")

    if algorithm == "actual_simultaneous":
        return partial(game_lingo_model, game_type="simultaneous")

    if algorithm == "greedy":
        return greedy_allocation

    if algorithm == "algorithm1_quad":
        return partial(algorithm1, defender_ftype="quadratic")

    if algorithm == "algorithm1_quad_int":
        return partial(algorithm1, defender_ftype="quadratic_int")

    if algorithm == "case1":
        return case1

    if algorithm == "case2":
        return case2

    if algorithm == "case4":
        return case4

    return logging.error("Algorithm not available.")

def data_collector(algorithm, varying_param = "A",
                   varying_range = (0.01,1,0.01), **algorithm_params):
    """
    Collects data for a given algorithm by varying a parameters

    Parameters
    ----------
    algorithm : str
        Algorithm for which we are collecting the data

    varying_param: str
        Parameter to vary - ["alpha", "beta", "A"]

    varying_range: tuple
        Range to vary on - [start, end, step]
    
    algorithm_params : dict
        Additional parameters to pass to the algorithm

    Returns
    -------
    collected_data : pd.DataFrame
    """
    logging.info("Collecting data for %s", algorithm)
    logging.info("Varying parameter: %s", varying_param)

    algo_func = algorithm_str(algorithm=algorithm)
    losses, gains, values = [], [], []

    for val in np.round(np.arange(*varying_range),2):
        params = {varying_param: val, **algorithm_params}
        try:
            results = algo_func(**params)
        except:
            logging.info("Infeasible problem")
            results = {
                "Defender's Losses" : 0,
                "Attacker's Gains"  : 0
            }

        losses.append(results["Defender's Losses"])
        gains.append(results["Attacker's Gains"])
        values.append(val)

        logging.info("%s = %.4f, losses = %s, gains = %s",
                     varying_param, val, results["Defender's Losses"], results["Attacker's Gains"])

    df = pd.DataFrame({varying_param: values,
        "Defender's Losses": losses,
        "Attacker's Gains": gains})
    df.to_csv(f"data/{algorithm}_{varying_param}_{varying_range}.csv")
    return df

def plotter_function(datasets, names, varying_param, save_path):
    """
    Function to plot losses and gains for different algorithms, 
    for a give varying parameter.

    Parameters
    ----------

    datasets: list
        List of datasets of respective algorithms to be plotted.

    names: list
        List of names of the respective algorithms to be plotted.

    varying_param: str
        Parameter to vary - ["alpha", "beta", "A"]

    save_name : str
        The filename for saving the plot.
    """

    # Initialize the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))

    # Plot losses
    for data, name in zip(datasets, names):
        ax1.plot(data[varying_param], data["Defender's Losses"], label=name)
        ax1.set_title("Defender's Losses")
        ax1.set_xlabel(varying_param)
        ax1.set_ylabel("Losses")
        ax1.legend()
        ax1.grid(True)

    # Plot gains
    for data, name in zip(datasets, names):
        ax2.plot(data[varying_param], data["Attacker's Gains"], label=name)
        ax2.set_title("Attacker's Gains")
        ax2.set_xlabel(varying_param)
        ax2.set_ylabel("Gains")
        ax2.legend()
        ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)

    logging.info("Plot saved to %s", save_path)

if __name__ == '__main__':
    case_params = {
        'n_targets' : 5,
        'attacker_budget' : 5,
        'defender_budget' : 30,
        'attacker_valuation' : np.array([20, 100, 50, 2, 20]),
        'defender_valuation' : np.array([70, 1000, 50, 75, 150]),
        'varying_param' : 'A',
        'varying_range' : (0.01, 1, 0.01),
        'beta' : 1,
        'alpha' : 1
    }

    data_case1 = data_collector(algorithm='case1', **case_params)
    data_case2 = data_collector(algorithm='case2', **case_params)
    data_case4 = data_collector(algorithm='case4', **case_params)

    # data_case1 = pd.read_csv(f"data/case1_{case_params['varying_param']}_(0.01, 1, 0.01).csv")
    # data_case4 = pd.read_csv(f"data/case4_{case_params['varying_param']}_(0.01, 1, 0.01).csv")
    # data_case2 = pd.read_csv(f"data/case2_{case_params['varying_param']}_(0.01, 1, 0.01).csv")

    data_actual_seq = pd.read_csv(f"data/actual_sequential_{case_params['varying_param']}_(0.01, 1, 0.01).csv")
    data_actual_sim = pd.read_csv(f"data/actual_simultaneous_{case_params['varying_param']}_(0.01, 1, 0.01).csv")
    data_greedy = pd.read_csv(f"data/greedy_{case_params['varying_param']}_(0.01, 1, 0.01).csv")

    plotter_function(
        datasets=[data_actual_seq, data_actual_sim, data_greedy, data_case1, data_case2, data_case4],
        names=['actual_sequential', 'actual_simultaneous', 'greedy', 'case1', 'case2', 'case4'],
        varying_param=case_params['varying_param'],
        save_path=f"plots/actual+cases124_{case_params['varying_param']}.png")

    # # Parameters
    # NUM_TARGETS = 5
    # A_BUDGET = 5 # Attacker's budget
    # D_BUDGET = 30 # Defender's budget

    # B = np.array([20, 100, 50, 2, 20]) # Attacker's valuations
    # D = np.array([70, 1000, 50, 75, 150]) # Defender's valuations

    # MAX_ITERS = 30 # For Algorithm 1

    # VARYING_PARAM = "A"
    # VARYING_RANGE = (0.01,1, 0.01)
    # ALPHA = 1
    # BETA = 1
    # # A = 0.1

    # data_actual_sim = data_collector(algorithm="actual_simultaneous",
    #                       varying_param=VARYING_PARAM,
    #                       varying_range=VARYING_RANGE,
    #                       n_targets = NUM_TARGETS,
    #                       b_list = B,
    #                       d_list = D,
    #                       t_budget = A_BUDGET, g_budget = D_BUDGET,
    #                       beta = BETA, alpha = ALPHA)
