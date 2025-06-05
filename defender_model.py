"""
Defender's optimization probem, solved using KKT conditions
"""
import sympy as sp

def prob_success(alpha, beta, Ti, Gi, Ai):
    "Probability of successful attack on target i"
    return (beta * Ti) / (beta * Ti + alpha * Gi + Ai)

def defender_model(n, G_budget, alpha, beta, A, D, T_list):
    """
    Function to solve defender's optimization model and return allocation values, G

    Parameters:
    -----------
    n : int
        Number of targets.

    G_budget : float or int
        Total budget available to the defender.

    alpha : float
        Defender's influence parameter in the success probability function.

    beta : float
        Attacker's influence parameter in the success probability function.

    A : float
        Inherent defense level of the targets

    D : list of floats
        Defender's valuation (importance) of each target. Length must be n.

    T_list : list of floats
        Attacker's resource allocation to each target. Length must be n.

    Returns:
    --------
    dict
        A dictionary mapping SymPy symbols to their optimal values (defender's allocations G_i,
        dual variables mu_i, and the Lagrange multiplier lambda_a), representing a KKT solution.
    """
    # Symbols
    G_list = sp.symbols(f'T1:{n+1}', real=True, nonnegative=True)
    mu_g = sp.symbols(f'mu1:{n+1}', real=True, nonnegative=True)
    lambda_g = sp.Symbol('lambda_a', real=True)

    # Objective function
    P_list = [prob_success(alpha, beta, Ti, Gi, A) for Ti, Gi in zip(T_list, G_list)]
    Z_g = sum(Di * Pi for Di, Pi in zip(D, P_list))

    # KKT Conditions

    # 1. Stationarity
    stationarity = [-sp.diff(Z_g, G_i) - lambda_g - mu_i for G_i, mu_i in zip(G_list, mu_g)]

    # 2. Primal feasibility (budget constraint)
    budget_constraint = [sum(G_list) - G_budget]

    # 3. Dual feasibility (mu_i >= 0) -- Already handled by symbol definition

    # 4. Complementary slackness (mu_i * G_i = 0)
    comp_slackness = [- mu_i * G_i for mu_i, G_i in zip(mu_g, G_list)]

    # Combine all KKT conditions
    kkt_system = stationarity + budget_constraint + comp_slackness

    # Solve
    solution = sp.solve(kkt_system, list(G_list) + list(mu_g) + [lambda_g], dict=True)
    # The given conditions solve the model globally, hence one solution is obtained

    return solution[0]


# Parameters
n = 2
G_budget = 30
alpha, beta, A = 1, 1, 0.1
T_list = [3,2]
D = [150, 30]

sol = defender_model(n=n,
                    G_budget=G_budget,
                    alpha=alpha,
                    beta=beta,
                    A=A,
                    T_list=T_list,
                    D=D)

# Output
print("Solution:", sol)
