"""
Attempt at sequential game, as a bilevel optimization problem, using Cvxpy
"""
import cvxpy as cp

def prob_success(alpha, beta, Ti, Gi, Ai):
    "Probability of successful attack on target i"
    return (beta * Ti) / (beta * Ti + alpha * Gi + Ai)

# Sets
N = 2 # Number of targets

targets = list(range(1,N+1))
B = [50, 100] # Attacker's benefit from a successful attack on target 1 (Million USD)
D = [150, 30] # Defender's target evaluation for target 1 (Million USD)

# Parameters
alpha, beta, A = 1, 1, 0.1
T_budget = 5 # Attacker's investment (Million USD)
G_budget = 30 # Defender's Investment (Million USD)


############# CVXPY APPROACH #################
# Construct the problem.
T_list = cp.Variable(N, nonneg=True)
G_list = cp.Variable(N, nonneg=True)
lambda_t = cp.Variable(1)
mu_t_list = cp.Variable(N, nonneg=True)

# Objective
# P_list = [prob_success(alpha=alpha, beta=beta, Ti=Ti[i], Gi=Gi[i],Ai=A) for i in targets]
P_list = [(beta * Ti) / (beta * Ti + alpha * Gi + A) for Ti,Gi in zip(T_list,G_list)]
Z_g = sum(Di * Pi for Di, Pi in zip(B, P_list)) # Defender's objective (Upper level)

objective = cp.Minimize(Z_g)

stationarity = sum([B[i] * (beta*(alpha*G_list[i] + A))/(beta*T_list[i] + alpha*G_list[i] + A)**2 for i in range(N)]) - lambda_t - sum(mu_t_list)
comp_slackness = [mu_t_list[i] * T_list[i] == 0 for i in range(N)]

constraints = [
               sum(G_list) == G_budget,
               stationarity == 0] +  comp_slackness

prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(G_list.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)
