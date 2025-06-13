# Attacker-Defender Game with Budget Constraints

## Notes

### Date - 11 May 2025

- We try to replicate the results of paper [[1]](#1) using Python. The results are in the notebook `implementation.ipynb`.

- Python files `defender_model.py` and `attacker_model.py` contain the implementation of the defender's and attacker's model respectively, solved using KKT conditions, as described in the paper.

- For the individual attacker's and defenser's problem, the KKT conditions provide inverted results for the edge cases.

    Example: Allocations [0,5], [5,0]

    The behaviour is unexpected, and isn't observed when the models are solved as a standard constrained optimization problem.

- For implementing the game models, we face several challenges with standard Python libraries, as listed below:

  - Sympy, which we used for the individual models, could not solve the simultaneous game model in a reasonable time frame (1 day).

  - Cvxpy and Pyomo do not support the ratio function in the objective function.

  - Gurobi does not support non-linear constraints in version 11.0, which is the version we are using due to license restrictions. To use non-linear constraints, we would need to update to version 12.0, which requires an academic network for the license.

  - Cvxpy does not support the constraints required for the problem.

  - Scipy.optimize uses the SLSQP method which requires a good initial guess, which is not available in this case. Hence, the results are inaccurate.

- The paper [[1]](#1) uses the software LINGO to solve the models.
  We decided to use LINGO to solve the models as well for consistency, and to avoid the issues with Python libraries mentioned above.

- For the simulatenous game, we solve the KKT conditions of both the attacker and defender models simultaneously, which gives us the optimal allocation for both the attacker and defender.

- For the sequential game, we model the defender's problem as a constrained optimization problem, and add the attacker's KKT conditions as constraints to the defender's problem. This gives us the optimal allocation for the defender, given the attacker's KKT conditions.

- The results of the simultaneous and sequential games are consistent with the results of the paper [[1]](#1).

### Date - 13 May 2025

- An API library for the LINGO solver is available as `lingo_api` and can be used to interact with the LINGO solver from Python. The library is available at <https://pypi.org/project/lingo-api/>

- We create an attacker's optimization model using the LINGO solver, stored in `LINGO models/attacker_model.lng`. we use it for simulations using `simulation.py` to find the attacker's optimal allocation with respect to the defender's allocation.
- We also obtain the defender's total losses for the given allocation.
. The results of the simulation are stored in `results/attacker_best_simulation.csv`.

## To Do

- Use the results to make an estimator function for the defender's total losses.
- Defender's total losses (given optimal attacker allocation) are a function of the defender's allocation, attacker's allocation, and the defender's valuation of the targets.

## References

<a id="1">[1]</a>
Peiqiu Guan, Meilin He, Jun Zhuang, Stephen C. Hora (2017) Modeling a Multitarget Attacker–Defender Game with Budget Constraints. Decision Analysis 14(2):87-107. <https://doi.org/10.1287/deca.2017.0346>
