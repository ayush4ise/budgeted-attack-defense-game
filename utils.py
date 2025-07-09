"Utility functions"

import lingo_api as lingo
import numpy as np

def prob_success(Ti, Gi, alpha=1, beta=1, Ai=0.1):
    "Probability of successful attack on target i"
    return (beta * Ti) / (beta * Ti + alpha * Gi + Ai)

def attacker_lingo_model(n_targets, alpha, beta, A, t_budget, B_list, G_list):
    """
    Function to solve the attacker's optimization model in LINGO 
    and return attacker's allocations, T
    Inspired from: https://github.com/lindosystems/lingoapi-python/blob/main/examples/CHESS/chess.py

    Parameters:
    -----------
    n_targets : int
        Number of targets.

    alpha : float
        Defender's influence parameter in the success probability function.

    beta : float
        Attacker's influence parameter in the success probability function.

    A : float
        Inherent defense level of the targets
        
    t_budget : float or int
        Total budget available to the attacker.

    B_list : np.array
        Attacker's valuation (importance) of each target. Length must be n.

    G_list : np.array
        Defender's resource allocation to each target. Length must be n.

    Returns:
    --------
    np.array
        A numpy array object, with attacker's resource allocations to targets
    """

    # Defines uData and an error callback function for each model to call
    # This will rasie an exception if there are any errors from Lingo
    uData = {}
    def cb_error(pEnv, uData, nErrorCode, errorText):
        raise lingo.CallBackError(nErrorCode, errorText)

    # Using a model in LINGO to maintain uniformity 
    LINGO_SCRIPT = "LINGO models/attacker_model.lng"
    LOG_PATH = "logs/attacker.log"

    # Defining variables to capture the results
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
    Function to solve the defender's optimization model in LINGO 
    and return defender's allocations, T

    Parameters:
    -----------
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
        Defender's valuation (importance) of each target. Length must be n.

    T_list : np.array
        Attacker's resource allocation to each target. Length must be n.

    Returns:
    --------
    np.array
        A numpy array object, with defender's resource allocations to targets
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
