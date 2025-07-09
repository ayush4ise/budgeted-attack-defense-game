"""
Attempt at sequential game, as a bilevel optimization problem, using Gurobipy
"""
import gurobipy as gp
from gurobipy import GRB

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

############ GUROBIPY APPROACH #################
# Create a new model
m = gp.Model("smaller_instances")

# Variables
Ti = m.addVars(targets, vtype=GRB.CONTINUOUS, name="Ti")
Gi = m.addVars(targets, vtype=GRB.CONTINUOUS, name="Gi")

subnumi = m.addVars(targets, vtype=GRB.CONTINUOUS, name="subnumi")
subdivi = m.addVars(targets, vtype=GRB.CONTINUOUS, name="subdivi")

lambda_t = m.addVar(name='lambda_t')
mu_ti = m.addVars(targets, vtype=GRB.CONTINUOUS, name="mu_ti")


# Objective Function
# P_list = [prob_success(alpha=alpha, beta=beta, Ti=Ti[i], Gi=Gi[i],Ai=A) for i in targets]
P_list = [(beta * T) / (beta * T + alpha * G + A) for T,G in zip(Ti,Gi)]
Z_g = sum(Di * Pi for Di, Pi in zip(D, P_list)) # Defender's objective (Upper level)

m.setObjective(Z_g)

# Constraints

# Budget constraint of defender (Upper level)
m.addConstr(Gi.sum() - G_budget == 0, name="budget_upper")


# KKT Conditions of attacker (Lower level)

# Stationarity
# m.addConstr(
#     [B[i-1] * (beta*(alpha*Gi[i] + A))/(beta*Ti[i] + alpha*Gi[i] + A)**2 for i in targets].sum() 
#     - lambda_t
#     - mu_ti.sum() == 0, name="stationarity"
#     )


# sub (betaT * betaT + 2 * betaT*alphaG + 2*a*betaT  + alphaG * alphG + 2*a*alphaG + A*A)


m.addConstr(
    sum([B[i-1] * subnumi[i] * subdivi[i] for i in targets])
    - lambda_t
    - mu_ti.sum() == 0, name="stationarity"
    )

for i in targets:
    m.addConstr(subnumi[i] == beta*(alpha*Gi[i] + A))

    # # Formulating an expression
    # expr = gp.QuadExpr()
    # expr.addTerms(coeffs=beta*beta, vars= subdivi[i], vars2=Ti[i]*Ti[i])
    # m.addConstr(expr == 1, name=f'quad{i}')


    ############ COULDNT FORMULATE THE EXACT EXPRESSION ################   
    m.addConstr(beta*beta*subdivi[i]*Ti[i] == 1)

# Budget constraint of attacker (Lower level)
m.addConstr(Ti.sum() - T_budget == 0, name="budget_lower")

# Complimentary slackness
for i in targets:
    m.addConstr(mu_ti[i] * Ti[i] == 0, name=f'comp_slackness{i}')

m.params.FuncNonlinear = 0

# Optimize the model
m.optimize()

# Output the results
if m.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    # for var in m.getVars():
    #     print(f"{var.varName}: {var.x}")

    varInfo = {}
    for v in m.getVars():
        if v.x>0:
            varInfo[v.varName] = v.x

    print(varInfo)

    for c in m.getConstrs():
        if c.Slack > 1e-6:
            print('Constraint %s is not active at solution point' % (c.ConstrName))

else:
    print(f"Optimization ended with status {m.status}.")
