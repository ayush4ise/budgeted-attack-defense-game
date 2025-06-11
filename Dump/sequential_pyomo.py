"""
Attempt at sequential game, as a bilevel optimization problem, using Pyomo

Inspired from: https://github.com/mehdinejjar86/pyomo-bilevel-optimization
"""
import pao
from pao.pyomo import *
from pyomo.environ import *

def prob_success(alpha, beta, Ti, Gi, Ai):
    "Probability of successful attack on target i"
    return (beta * Ti) / (beta * Ti + alpha * Gi + Ai)

model = ConcreteModel()

N_TARGETS = 2
model.target_range = RangeSet(1, N_TARGETS)

# Data
B = {1:50,2:100} # Attacker's valuations
D = {1:30, 2:150} # Defender's valuations
alpha, beta, A = 1.0, 1.0, 0.1
T_BUDGET = 5.0
G_BUDGET = 30.0

model.B = Param(model.target_range, initialize=B)
model.D = Param(model.target_range, initialize=D)
model.alpha = Param(initialize=alpha)
model.beta = Param(initialize=beta)
model.A = Param(initialize=A)
model.t_budget = Param(initialize=T_BUDGET)
model.g_budget = Param(initialize=G_BUDGET)

# Decision variables (Upper Level)
# model.Ti = Var(model.target_range, within=NonNegativeReals, initialize=0) # Attacker
model.Gi = Var(model.target_range, within=NonNegativeReals, initialize=0) # Defender

# Defining the sub-model (lower level problem)
model.submodel = SubModel(fixed=[model.Gi])

# Decision Variables (Lower Level)
model.submodel.Ti = Var(model.target_range, within=NonNegativeReals, initialize=0) # Attacker

# Upper level objective function to minimize the defender's losses sum(Di*Pi)
def upper_level_obj(model):
    obj = sum(model.D[i] * prob_success(model.alpha, model.beta, model.submodel.Ti[i], model.Gi[i], model.A) for i in model.target_range)
    return obj

model.UpperLevelObjective = Objective(rule=upper_level_obj, sense=minimize)

# Lower level objective function to maximize the attacker's gains sum(Bi * Pi)
def lower_level_obj(submodel):
    obj = sum(model.B[i] * prob_success(model.alpha, model.beta, submodel.Ti[i], model.Gi[i], model.A) for i in model.target_range)
    return obj

model.submodel.LowerLevelObjective = Objective(rule=lower_level_obj, sense=maximize)

# Set the constraints
model.ResourceConstraint = Constraint(expr = (sum(model.Gi[i] for i in model.target_range) - model.g_budget <= 0))

model.submodel.ResourceConstraint = Constraint(expr = (sum(model.submodel.Ti[i] for i in model.target_range) - model.t_budget <= 0))

# Try solving the model
solver = pao.Solver("pao.pyomo.FA")
solution = solver.solve(model, tee=True)
