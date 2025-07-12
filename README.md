# Attacker-Defender Game with Budget Constraints

## Notes

### Date - 11 June 2025

- We try to replicate the results of paper [[1]](#1) using Python. The results are in the notebook `notebooks/implementation.ipynb`.

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

### Date - 13 June 2025

- An API library for the LINGO solver is available as `lingo_api` and can be used to interact with the LINGO solver from Python. The library is available at <https://pypi.org/project/lingo-api/>

- We create an attacker's optimization model using the LINGO solver, stored in `LINGO models/attacker_model.lng`. We use it for simulations using `dLoss_simulation.py` to find the attacker's optimal allocations with respect to the defender's allocations.
- We also obtain the defender's total losses for the given allocation.
. The results of the simulation are stored in `results/dLoss_simulation_random20.csv`.

### Date - 25 June 2025

- Python library `PyMC` is used to implement Bayesian Linear Regression, which is used to estimate the defender's total losses based on the defender's allocation.
- The model is implemented in `bayesian_estimation.ipynb` using the data from `results/attacker_best_simulation.csv`.
- Since, we don't know the priors for the model, we cannot use the Bayesian model to estimate the defender's total losses.

### Date - 04 July 2025

- We implement the defender's model using the LINGO solver, stored in `LINGO models/defender_model.lng`. We use it for simulations using `aGain_simulation.py` to find attacker's total gains with respect to the attacker's allocations, assuming the defender chooses the optimal allocation using the defender's model.
- The results of the simulation are stored in `results/aGain_simulation_random100.csv`.
- We fit a quadratic regression model to the simulated data using `estimation.py` which will be used as a substitute to utility functions for both the attacker and defender in the iterative algorithm.
- The model coefficients are saved in `models` folder.

### Date - 09 July 2025

- We use the estimated utility functions to find the optimal allocations for the attacker and defender using Gurobi, which is implemented in `utility_calculation.py`.
- We also find the utility values using the contest success function formula (as used in the paper) for the optimal allocations.
- The results are stored in `results/utility_comparison_random100.xlsx`.

### Date - 12 July 2025

- Python library `PyDOE3` is used to generate a Latin Hypercube Sample (LHS) for the simulations, which is implemented and updated in `aGain_simulation.py` and `dLoss_simulation.py`. The results are stored in `results/aGain_simulation_lhs100.csv` and `results/dLoss_simulation_lhs100.csv` respectively.

- We fit quadratic regression models to this simulated data using `estimation.py`. The models coefficients are saved in `models` folder as `aGain_model_lhs100.json` and `dLoss_model_lhs100.json`.

- We use these estimated utility functions to carry out utility calculation using `utility_calculation.py`. The results are stored in `results/utility_comparison_lhs100.xlsx`.

## To Do

- Try to use the estimator functions in the iterative algorithm suggested.
- Try space filling designs/ factorial designs for the simulations.
- Figure out what went wrong with LHS samples and the results obtained from the simulations.

## Results

- Simultaneous game solution from LINGO model:
  - Attacker Allocations, T = [0.614, 1.349, 2.685, 0.008, 0.345]
  - Defender Allocations, G = [2.99, 19.303, 3.762, 0.328, 3.618]
  - Attacker's Total Gains = 32.056
  - Defender's Total Losses = 111.228
- Sequential game solution from LINGO model:
  - Attacker Allocations, T = [1.094, 0.385, 2.91, 0, 0.611]
  - Defender Allocations, G = [1.71, 22.426, 2.638, 0, 3.226]
  - Attacker's Total Gains = 38.08
  - Defender's Total Losses = 92.215

## Doubts/Suggestions

- The iterative algorithm suggested doesn't work for this estimated utility approach, because there's no requirement for any previous iteration values to be used for updation.

- Metaheuristic algorithms like Hill Climbing, Simulated Annealing, Genetic Algorithms, etc. can be used to find the optimal allocation for the attacker and defender.

- We can use the defender's allocations as inputs for estimating the attacker's utility function, for the sequential game.

- Effect of valuations on estimated utility functions (and results) can be studied.

## References

<a id="1">[1]</a>
Peiqiu Guan, Meilin He, Jun Zhuang, Stephen C. Hora (2017) Modeling a Multitarget Attacker–Defender Game with Budget Constraints. Decision Analysis 14(2):87-107. <https://doi.org/10.1287/deca.2017.0346>
