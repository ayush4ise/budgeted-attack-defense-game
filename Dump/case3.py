"""
Case 3:
Defender - Does not know the utility function
Attacker - Knows the utility function (but not defender's allocations)

Since the attacker knows the actual utility function (CSF),
we have to model this as a simultaneous game.
"""
import lingo_api as lingo
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import prob_success

def case3(**kwargs):
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

    # Parameters
    g_budget = kwargs.get("defender_budget")
    t_budget = kwargs.get("attacker_budget")
    d_list = kwargs.get("defender_valuation")
    b_list = kwargs.get("attacker_valuation")
    alpha, beta, A = kwargs.get("alpha"), kwargs.get("beta"), kwargs.get("A")

    # Calculating model coefficients for the quadratic defender utility function
    data = pd.read_csv(f"data/simulations/dLoss_simulation_lhs100_({alpha},{beta},{A}).csv")

    # Extracting X and y for regression
    X_1 = np.array(data[data.columns[:n_targets]])   # Degree 1 features
    X_2 = X_1**2                                # Get quadratic features
    y = data[data.columns[-1]]                  # Gains/Losses
    X = np.hstack([X_1, X_2])                   # Final input with quadratic features

    model = LinearRegression()                  # Fitting a regression model
    model.fit(X,y)
    coefficients = np.array(model.coef_)
    n_coeffs = len(coefficients)

    # Defines uData and an error callback function for each model to call
    # This will rasie an exception if there are any errors from Lingo
    uData = {}
    def cb_error(pEnv, uData, nErrorCode, errorText):
        # raise lingo.CallBackError(nErrorCode, errorText)
        pass

    # Using a model in LINGO to maintain uniformity
    LINGO_SCRIPT = "LINGO models/case3_model.lng"
    LOG_PATH = "logs/case3.log"

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
    model.set_pointer("Pointer2",n_coeffs, lingo.PARAM)
    model.set_pointer("Pointer3",alpha,lingo.PARAM)
    model.set_pointer("Pointer4",beta,lingo.PARAM)
    model.set_pointer("Pointer5",A,lingo.PARAM)
    model.set_pointer("Pointer6", b_list,lingo.PARAM)
    model.set_pointer("Pointer7",d_list,lingo.PARAM)
    model.set_pointer("Pointer8",t_budget,lingo.PARAM)
    model.set_pointer("Pointer9",g_budget,lingo.PARAM)
    model.set_pointer("Pointer10",coefficients,lingo.PARAM)
    model.set_pointer("Pointer11",T,lingo.VAR)
    model.set_pointer("Pointer12",G,lingo.VAR)
    model.set_pointer("Pointer13",STATUS,lingo.VAR)

    # Set the call back function and user data
    model.set_cbError(cb_error)
    model.set_uData(uData)

    # Call solve(model)
    lingo.solve(model)

    # Get STATUS since it is not an NumPy array
    # it needs to retrived from model
    STATUS, ptrType  = model.get_pointer("Pointer13")

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

if __name__ == '__main__':
    # Parameters
    NUM_TARGETS = 5
    A_BUDGET = 5 # Attacker's budget
    D_BUDGET = 30 # Defender's budget
    ALPHA, BETA, A = 1, 1, 0.1

    B = np.array([20, 100, 50, 2, 20]) # Attacker's valuations
    D = np.array([70, 1000, 50, 75, 150]) # Defender's valuations

    results = case3(n_targets=NUM_TARGETS,
                    attacker_budget=A_BUDGET,defender_budget=D_BUDGET,
                    attacker_valuation=B,defender_valuation=D,
                    alpha=ALPHA,beta=BETA,A=A)
    print(results)
