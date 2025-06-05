"""
Attacker's optimization probem, solved using KKT conditions
"""
import sympy as sp

def prob_success(alpha, beta, Ti, Gi, Ai):
    "Probability of successful attack on target i"
    return (beta * Ti) / (beta * Ti + alpha * Gi + Ai)

def attacker_model(n, T_budget, alpha, beta, A, B, G_list):
    """
    Function to solve attacker's optimization model and return allocation values, T
    
    Parameters:
    -----------
    n : int
        Number of targets.

    T_budget : float or int
        Total budget available to the attacker.

    alpha : float
        Defender's influence parameter in the success probability function.

    beta : float
        Attacker's influence parameter in the success probability function.

    A : float
        Inherent defense level of the targets

    B : list of floats
        Attacker's valuation (importance) of each target. Length must be n.

    G_list : list of floats
        Defender's resource allocation to each target. Length must be n.

    Returns:
    --------
    dict
        A dictionary mapping SymPy symbols to their optimal values (attacker's allocations T_i,
        dual variables mu_i, and the Lagrange multiplier lambda_a), representing a KKT solution.
    """
    # Symbols
    T_list = sp.symbols(f'T1:{n+1}', real=True, nonnegative=True)
    mu_t = sp.symbols(f'mu1:{n+1}', real=True, nonnegative=True)
    lambda_t = sp.Symbol('lambda_a', real=True)

    # Objective function
    P_list = [prob_success(alpha, beta, Ti, Gi, A) for Ti, Gi in zip(T_list, G_list)]
    Z_t = sum(Bi * Pi for Bi, Pi in zip(B, P_list))

    # KKT Conditions

    # 1. Stationarity
    stationarity = [sp.diff(Z_t, T_i) - lambda_t - mu_i for T_i, mu_i in zip(T_list, mu_t)]

    # 2. Primal feasibility (budget constraint)
    budget_constraint = [sum(T_list) - T_budget]

    # 3. Dual feasibility (mu_i >= 0) -- Already handled by symbol definition

    # 4. Complementary slackness (mu_i * T_i = 0)
    comp_slackness = [mu_i * T_i for mu_i, T_i in zip(mu_t, T_list)]

    # Combine all KKT conditions
    kkt_system = stationarity + budget_constraint + comp_slackness

    # Solve
    solution = sp.solve(kkt_system, list(T_list) + list(mu_t) + [lambda_t], dict=True)
    # The given conditions solve the model globally, hence one solution is obtained

    return solution[0]


# Parameters
n = 2
T_budget = 5
alpha, beta, A = 1, 1, 0.1
G_list = [15, 15]
B = [50, 100]

sol = attacker_model(n=n,
                    T_budget=T_budget,
                    alpha=alpha,
                    beta=beta,
                    A=A,
                    G_list=G_list,
                    B=B)

# Output
print("Solution:", sol)
