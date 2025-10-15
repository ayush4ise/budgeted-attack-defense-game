"""
Using EGO model on the given problem using Surrogate-Modeling Toolbox library

Inspired by- https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html#usage
"""

# Calling necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from smt.applications import EGO
from smt.design_space import DesignSpace
from smt.surrogate_models import KRG

from utils import attacker_lingo_model, prob_success

###############################################################################################
# Parameters
###############################################################################################

n_targets = 2
alpha, beta, A = 1, 1, 0.1
T_BUDGET = 5 # Attacker's investment (Million USD)
G_BUDGET = 30 # Defender's Investment (Million USD)

# Attacker's valuations
B = np.array([50, 20]) # Infrastructure 6-10 valuations from multiple
# infrastructure model from the paper
D = np.array([50, 150]) # Defender's valuations

###############################################################################################
# Function to Minimize
###############################################################################################

def def_loss_function(x):
    """Input target 1 allocation x, get defender losses (y)"""
    x = np.reshape(x, (-1,))
    y = np.array([])
    for alloc1 in x:
        def_allocation = np.array([alloc1, G_BUDGET - alloc1])
        att_allocation = attacker_lingo_model(
                n_targets=n_targets,
                alpha=alpha,
                beta=beta,
                A=A,
                t_budget=T_BUDGET,
                B_list=B,
                G_list=def_allocation
            )
        # Loss (y)
        loss = np.dot(D, [prob_success(Ti=Ti,Gi=Gi, alpha=alpha, beta=beta, Ai=A) for Ti,Gi in zip(att_allocation,def_allocation)])
        y = np.append(y, loss)
    return y.reshape((-1,1))

###############################################################################################
# EGO Method
###############################################################################################

n_iter = 6
xlimits = np.array([[0.0, G_BUDGET]])

random_state = 42  # for reproducibility
design_space = DesignSpace(xlimits, random_state=random_state)
xdoe = np.atleast_2d([0, G_BUDGET//2, G_BUDGET]).T
n_doe = xdoe.size

criterion = "EI"  #'EI' or 'SBO' or 'LCB'

ego = EGO(
    n_iter=n_iter,
    criterion=criterion,
    xdoe=xdoe,
    surrogate=KRG(design_space=design_space, print_global=False),
    random_state=random_state,
)

x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=def_loss_function)
print("EGO-derived Minimum in x={:.1f} with f(x)={:.1f}".format(x_opt.item(), y_opt.item()))

###############################################################################################
# Plotting
###############################################################################################
x_plot = np.atleast_2d(np.linspace(0, G_BUDGET, 100)).T
y_plot = def_loss_function(x_plot)

fig = plt.figure(figsize=[10, 10])
for i in range(n_iter):
    k = n_doe + i
    x_data_k = x_data[0:k]
    y_data_k = y_data[0:k]
    ego.gpr.set_training_values(x_data_k, y_data_k)
    ego.gpr.train()

    y_gp_plot = ego.gpr.predict_values(x_plot)
    y_gp_plot_var = ego.gpr.predict_variances(x_plot)
    y_ei_plot = -ego.EI(x_plot)

    ax = fig.add_subplot((n_iter + 1) // 2, 2, i + 1)
    ax1 = ax.twinx()
    (ei,) = ax1.plot(x_plot, y_ei_plot, color="red")

    (true_fun,) = ax.plot(x_plot, y_plot)
    (data,) = ax.plot(
        x_data_k, y_data_k, linestyle="", marker="o", color="orange"
    )
    if i < n_iter - 1:
        (opt,) = ax.plot(
            x_data[k], y_data[k], linestyle="", marker="*", color="r"
        )
    (gp,) = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
    sig_plus = y_gp_plot + 3 * np.sqrt(y_gp_plot_var)
    sig_moins = y_gp_plot - 3 * np.sqrt(y_gp_plot_var)
    un_gp = ax.fill_between(
        x_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
    )
    lines = [true_fun, data, gp, un_gp, opt, ei]
    fig.suptitle("EGO optimization of $f(x) = def_loss(x)$")
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
    ax.set_title("iteration {}".format(i + 1))
    fig.legend(
        lines,
        [
            "f(x)=def_loss(x)",
            "Given data points",
            "Kriging prediction",
            "Kriging 99% confidence interval",
            "Next point to evaluate",
            "Expected improvement function",
        ],
    )
# plt.show()
plt.savefig("plots/ego_2targets.png")
