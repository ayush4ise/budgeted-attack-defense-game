"""
Simultaneous game, where both attacker and defender make a move simultaneously, 
without knowing each other's move
"""

from fractions import Fraction
import sympy as sp

def prob_success(alpha, beta, Ti, Gi, Ai):
    "Probability of successful attack on target i"
    return (beta * Ti) / (beta * Ti + alpha * Gi + Ai)

def decimal_to_rational(x, max_denominator=1000000):
    if isinstance(x, str):
        return sp.Rational(x)
    elif isinstance(x, float):
        frac = Fraction(x).limit_denominator(max_denominator)
        return sp.Rational(frac.numerator, frac.denominator)
    else:
        return sp.Rational(x)
  
def simultaneous_game(n, G_budget, T_budget, alpha, beta, A, B, D):
    """
    Function to solve simultaneous game model and return allocation values, G and T

    Parameters:
    -----------
    n : int
        Number of targets.

    G_budget : float or int
        Total budget available to the defender.

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

    D : list of floats
        Defender's valuation (importance) of each target. Length must be n.

    Returns:
    --------
    dict
        A dictionary mapping SymPy symbols to their optimal values (defender's allocations G_i, T_i
        dual variables mu_i, and the Lagrange multiplier lambda_a), representing a KKT solution.
    """
    A = decimal_to_rational(A)
    # G_budget = decimal_to_rational(G_budget)
    # T_budget = decimal_to_rational(T_budget)
    # alpha, beta = decimal_to_rational(alpha), decimal_to_rational(beta)
    # B = [decimal_to_rational(b) for b in B]
    # D = [decimal_to_rational(d) for d in D]

    # Symbols
    T_list = sp.symbols(f'T1:{n+1}', real=True, nonnegative=True)
    G_list = sp.symbols(f'G1:{n+1}', real=True, nonnegative=True)
    mu_t = sp.symbols(f'mut1:{n+1}', real=True, nonnegative=True)
    mu_g = sp.symbols(f'mug1:{n+1}', real=True, nonnegative=True)
    lambda_t = sp.Symbol('lambda_t', real=True)
    lambda_g = sp.Symbol('lambda_g', real=True)

    # Objective function
    P_list = [prob_success(alpha, beta, Ti, Gi, A) for Ti, Gi in zip(T_list, G_list)]
    Z_t = sum(Bi * Pi for Bi, Pi in zip(B, P_list))
    Z_g = sum(Di * Pi for Di, Pi in zip(D, P_list))

    # KKT Conditions (Attacker)

    # 1. Stationarity
    stationarity_t = [sp.simplify(sp.diff(Z_t, T_i)) - lambda_t - mu_i for T_i, mu_i in zip(T_list, mu_t)]

    # 2. Primal feasibility (budget constraint)
    budget_constraint_t = [sum(T_list) - T_budget]

    # 3. Dual feasibility (mu_i >= 0) -- Already handled by symbol definition

    # 4. Complementary slackness (mu_i * T_i = 0)
    comp_slackness_t = [mu_i * T_i for mu_i, T_i in zip(mu_t, T_list)]

    # Combine all KKT conditions
    kkt_system_t = stationarity_t + budget_constraint_t + comp_slackness_t

    # KKT Conditions (Defender)

    # 1. Stationarity
    stationarity_g = [sp.simplify(-sp.diff(Z_g, G_i)) - lambda_g - mu_i for G_i, mu_i in zip(G_list, mu_g)]

    # 2. Primal feasibility (budget constraint)
    budget_constraint_g = [sum(G_list) - G_budget]

    # 3. Dual feasibility (mu_i >= 0) -- Already handled by symbol definition

    # 4. Complementary slackness (mu_i * G_i = 0)
    comp_slackness_g = [- mu_i * G_i for mu_i, G_i in zip(mu_g, G_list)]

    # Combine all KKT condition
    kkt_system_g = stationarity_g + budget_constraint_g + comp_slackness_g

    kkt_final = kkt_system_g + kkt_system_t

    # Solve
    solution = sp.solve(kkt_final,
                        list(T_list) + list(G_list) + list(mu_t) + list(mu_g) + [lambda_t] + [lambda_g],
                        dict=True)
    # The given conditions solve the model globally, hence one solution is obtained

    # return {k: v.evalf() for k, v in solution[-1].items()}
    return solution

# Parameters
n = 2
T_budget = 5
G_budget = 30
alpha, beta, A = 1, 1, 0.1
B = [50, 100]
D = [150, 30]

sol = simultaneous_game(n=n,
                    G_budget=G_budget,
                    T_budget=T_budget,
                    alpha=alpha,
                    beta=beta,
                    A=A,
                    B=B,
                    D=D)

# Output
print("Solution:", sol)
