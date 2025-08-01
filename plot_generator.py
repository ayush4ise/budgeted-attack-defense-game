"""
Module to generate plots for different algorithms by varying parameters (alpha, beta, A)
"""
import logging
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lingo_api
from utils import game_lingo_model

# Setting up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    handlers=[logging.FileHandler('logs/plot_generator.log'), logging.StreamHandler()])

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

    return logging.error("Algorithm not available.")

def data_collector(algorithm, varying_param = "A", varying_range = (0,1,0.01), **algorithm_params):
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

    for val in np.arange(*varying_range):
        params = {varying_param: val, **algorithm_params}
        try:
            results = algo_func(**params)
        except lingo_api.LingoError:
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    data = data_collector(algorithm="actual_sequential",
                          varying_param="A",
                          varying_range=(0.01,1,0.01),
                          n_targets = 5,
                          alpha = 1, beta = 1,
                          b_list = np.array([20, 100, 50, 2, 20]),
                          d_list = np.array([70, 1000, 50, 75, 150]),
                          t_budget = 5, g_budget = 30)

    plotter_function(datasets=[data], names=['actual_sequential'],
                     varying_param="A", save_path="plots/actual_seq_A.png")
