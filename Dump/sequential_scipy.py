"""
Attempt at sequential game, as a bilevel optimization problem, using Scipy
"""
import numpy as np
from scipy.optimize import minimize

# parameters
N = 2
B = np.array([50, 100])      # attacker benefits
D = np.array([150, 30])      # defender valuations
alpha, beta, A = 1.0, 1.0, 0.1
T_budget = 5.0
G_budget = 30.0

def attacker_obj(T, G):
    # maximize sum B_i * P_i -> minimize negative
    P = (beta * T) / (beta * T + alpha * G + A)
    return -np.dot(B, P)

def defender_obj(G):
    # solve inner attacker problem for this G
    cons_att = ({'type': 'eq',   'fun': lambda T: np.sum(T) - T_budget},
                {'type': 'ineq', 'fun': lambda T: T})
    x0 = np.full(N, T_budget/N)
    sol = minimize(attacker_obj, x0, args=(G,),
                   constraints=cons_att,
                   bounds=[(0, T_budget)]*N,
                   method='SLSQP')
    T_star = sol.x
    # defender wants to minimize expected damage = sum D_i * P_i
    P_star = (beta * T_star) / (beta * T_star + alpha * G + A)
    return np.dot(D, P_star)

# outer (defender) optimization
cons_def = ({'type': 'eq',   'fun': lambda G: np.sum(G) - G_budget},
            {'type': 'ineq', 'fun': lambda G: G})
G0 = np.full(N, G_budget/N)

res_def = minimize(defender_obj, G0,
                   constraints=cons_def,
                   bounds=[(0, G_budget)]*N,
                   method='SLSQP')

print("Defender allocation G* =", res_def.x)
print("Defender Losses:", res_def.fun)

# get final attacker response
att_res = minimize(attacker_obj, np.full(N, T_budget/N),
                   args=(res_def.x,),
                   constraints=({'type':'eq','fun':lambda T: np.sum(T)-T_budget},
                                {'type':'ineq','fun':lambda T: T}),
                   bounds=[(0, T_budget)]*N,
                   method='SLSQP')
print("Attacker allocation T* =", att_res.x)
