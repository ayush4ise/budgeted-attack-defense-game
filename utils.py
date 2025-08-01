"Utility functions"

import lingo_api as lingo
import numpy as np

def prob_success(Ti, Gi, alpha=1, beta=1, Ai=0.1):
    "Probability of successful attack on target i"
    return (beta * Ti) / (beta * Ti + alpha * Gi + Ai)

def attacker_lingo_model(n_targets, alpha, beta, A, t_budget, B_list, G_list):
    """
    Solve the attacker's optimization model in LINGO and return attacker's allocations.

    Inspired by:
    https://github.com/lindosystems/lingoapi-python/blob/main/examples/CHESS/chess.py

    Parameters
    ----------
    n_targets : int
        Number of targets.
    alpha : float
        Defender's influence parameter in the success probability function.
    beta : float
        Attacker's influence parameter in the success probability function.
    A : float
        Inherent defense level of the targets.
    t_budget : float or int
        Total budget available to the attacker.
    B_list : np.array
        Attacker's valuation (importance) of each target. Length must be n_targets.
    G_list : np.array
        Defender's resource allocation to each target. Length must be n_targets.

    Returns
    -------
    np.array
        A NumPy array containing the attacker's resource allocations to each target.
    """

    # Defines uData and an error callback function for each model to call
    # This will rasie an exception if there are any errors from Lingo
    uData = {}
    def cb_error(pEnv, uData, nErrorCode, errorText):
        raise lingo.CallBackError(nErrorCode, errorText)

    # Using a model in LINGO to maintain uniformity 
    LINGO_SCRIPT = "LINGO models/attacker_model.lng"
    # LOG_PATH = "logs/attacker.log"

    # Defining variables to capture the results
    T = np.zeros(n_targets)

    # To check if the data was solved
    STATUS = -1

    # Create a model object
    model = lingo.Model(LINGO_SCRIPT, logFile=None)

    # model.set_logFile(LOG_PATH)

    # Set all pointers in the order that they appear in attacker_model.lng
    model.set_pointer("Pointer1",n_targets,lingo.PARAM)
    model.set_pointer("Pointer2",alpha,lingo.PARAM)
    model.set_pointer("Pointer3",beta,lingo.PARAM)
    model.set_pointer("Pointer4",A,lingo.PARAM)
    model.set_pointer("Pointer5", t_budget,lingo.PARAM)
    model.set_pointer("Pointer6",B_list,lingo.PARAM)
    model.set_pointer("Pointer7",G_list,lingo.PARAM)
    model.set_pointer("Pointer8",T,lingo.VAR)
    model.set_pointer("Pointer9",STATUS,lingo.VAR)

    # Set the call back function and user data
    model.set_cbError(cb_error)
    model.set_uData(uData)

    # Call solve(model)
    lingo.solve(model)

    # Get STATUS since it is not an NumPy array
    # it needs to retrived from model
    STATUS, ptrType  = model.get_pointer("Pointer9")

    # Check that the model has ben solved
    if STATUS == lingo.LS_STATUS_GLOBAL_LNG:
        print("\nGlobal optimum found!")
    elif STATUS == lingo.LS_STATUS_LOCAL_LNG:
        print("\nLocal optimum found!")
    else:
        print("\nSolution is non-optimal\n")

    # # Display the results
    # print("T", T)

    return T

def defender_lingo_model(n_targets, alpha, beta, A, g_budget, D_list, T_list):
    """
    Solve the defender's optimization model in LINGO and return defender's allocations.

    Parameters
    ----------
    n_targets : int
        Number of targets.
    alpha : float
        Defender's influence parameter in the success probability function.
    beta : float
        Attacker's influence parameter in the success probability function.
    A : float
        Inherent defense level of the targets.
    g_budget : float or int
        Total budget available to the defender.
    D_list : np.array
        Defender's valuation (importance) of each target. Length must be n_targets.
    T_list : np.array
        Attacker's resource allocation to each target. Length must be n_targets.

    Returns
    -------
    np.array
        A NumPy array containing the defender's resource allocations to each target.
    """

    # Defines uData and an error callback function for each model to call
    # This will rasie an exception if there are any errors from Lingo
    uData = {}
    def cb_error(pEnv, uData, nErrorCode, errorText):
        raise lingo.CallBackError(nErrorCode, errorText)

    # Using a model in LINGO to maintain uniformity 
    LINGO_SCRIPT = "LINGO models/defender_model.lng"
    LOG_PATH = "logs/defender.log"

    # Defining variables to capture the results
    G = np.zeros(n_targets)

    # To check if the data was solved
    STATUS = -1

    # Create a model object
    model = lingo.Model(LINGO_SCRIPT)

    model.set_logFile(LOG_PATH)

    # Set all pointers in the order that they appear in attacker_model.lng
    model.set_pointer("Pointer1",n_targets,lingo.PARAM)
    model.set_pointer("Pointer2",alpha,lingo.PARAM)
    model.set_pointer("Pointer3",beta,lingo.PARAM)
    model.set_pointer("Pointer4",A,lingo.PARAM)
    model.set_pointer("Pointer5", g_budget,lingo.PARAM)
    model.set_pointer("Pointer6",D_list,lingo.PARAM)
    model.set_pointer("Pointer7",T_list,lingo.PARAM)
    model.set_pointer("Pointer8",G,lingo.VAR)
    model.set_pointer("Pointer9",STATUS,lingo.VAR)

    # Set the call back function and user data
    model.set_cbError(cb_error)
    model.set_uData(uData)

    # Call solve(model)
    lingo.solve(model)

    # Get STATUS since it is not an NumPy array
    # it needs to retrived from model
    STATUS, ptrType  = model.get_pointer("Pointer9")

    # Check that the model has ben solved
    if STATUS == lingo.LS_STATUS_GLOBAL_LNG:
        print("\nGlobal optimum found!")
    elif STATUS == lingo.LS_STATUS_LOCAL_LNG:
        print("\nLocal optimum found!")
    else:
        print("\nSolution is non-optimal\n")

    return G

def game_lingo_model(game_type, n_targets, alpha, beta, A, b_list, d_list, t_budget, g_budget):
    """
    Solve the sequential or simultaneous game in LINGO and return attacker's and defender's allocations.

    Parameters
    ----------
    game_type : str
        Type of the game. Either 'sequential' or 'simultaneous'.
    n_targets : int
        Number of targets.
    alpha : float
        Defender's influence parameter in the success probability function.
    beta : float
        Attacker's influence parameter in the success probability function.
    A : float
        Inherent defense level of the targets.
    b_list : np.array
        Attacker's valuation (importance) of each target. Length must be n_targets.
    d_list : np.array
        Defender's valuation (importance) of each target. Length must be n_targets.
    t_budget : float or int
        Total budget available to the attacker.
    g_budget : float or int
        Total budget available to the defender.

    Returns
    -------
    dict
        Dictionary containing:
        - "Defender's allocations" : np.array
        - "Attacker's allocations" : np.array
    """

    # Defines uData and an error callback function for each model to call
    # This will rasie an exception if there are any errors from Lingo
    uData = {}
    def cb_error(pEnv, uData, nErrorCode, errorText):
        raise lingo.CallBackError(nErrorCode, errorText)

    # Using a model in LINGO to maintain uniformity
    LINGO_SCRIPT = f"LINGO models/{game_type}_model.lng"
    LOG_PATH = f"logs/{game_type}.log"

    # Defining variables to capture the results
    G = np.zeros(n_targets)
    T = np.zeros(n_targets)

    # To check if the data was solved
    STATUS = -1

    # Create a model object
    model = lingo.Model(LINGO_SCRIPT)

    model.set_logFile(LOG_PATH)

    # Set all pointers in the order that they appear in attacker_model.lng
    model.set_pointer("Pointer1",n_targets,lingo.PARAM)
    model.set_pointer("Pointer2",alpha,lingo.PARAM)
    model.set_pointer("Pointer3",beta,lingo.PARAM)
    model.set_pointer("Pointer4",A,lingo.PARAM)
    model.set_pointer("Pointer5", b_list,lingo.PARAM)
    model.set_pointer("Pointer6",d_list,lingo.PARAM)
    model.set_pointer("Pointer7",t_budget,lingo.PARAM)
    model.set_pointer("Pointer8",g_budget,lingo.VAR)
    model.set_pointer("Pointer9",T,lingo.VAR)
    model.set_pointer("Pointer10",G,lingo.VAR)
    model.set_pointer("Pointer11",STATUS,lingo.VAR)

    # Set the call back function and user data
    model.set_cbError(cb_error)
    model.set_uData(uData)

    # Call solve(model)
    lingo.solve(model)

    # Get STATUS since it is not an NumPy array
    # it needs to retrived from model
    STATUS, ptrType  = model.get_pointer("Pointer11")

    # Check that the model has ben solved
    if STATUS == lingo.LS_STATUS_GLOBAL_LNG:
        print("\nGlobal optimum found!")
    elif STATUS == lingo.LS_STATUS_LOCAL_LNG:
        print("\nLocal optimum found!")
    else:
        print("\nSolution is non-optimal\n")

    # Calculate losses and gains
    losses = np.dot(d_list, prob_success(Ti=T, Gi=G, alpha=alpha, beta=beta, Ai=A))
    gains = np.dot(b_list, prob_success(Ti=T, Gi=G, alpha=alpha, beta=beta, Ai=A))

    return {
        "Defender's allocations" : G,
        "Attacker's allocations" : T,
        "Defender's Losses"      : losses,
        "Attacker's Gains"       : gains
    }
