# Attacker-Defender Game with Budget Constraints

## Notes

### Date - 11 June 2025

- We try to replicate the results of paper [[1]](#1) using Python. The results are in the notebook [`notebooks/implementation.ipynb`](notebooks/implementation.ipynb).

- Python files [`defender_model.py`](defender_model.py) and [`attacker_model.py`](attacker_model.py) contain the implementation of the defender's and attacker's model respectively, solved using KKT conditions, as described in the paper.

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
- We create an attacker's optimization model using the LINGO solver, stored in [`LINGO models/attacker_model.lng`](LINGO%20models/attacker_model.lng). We use it for simulations using [`dLoss_simulation.py`](dLoss_simulation.py) to find the attacker's optimal allocations with respect to the defender's allocations.
- We also obtain the defender's total losses for the given allocation.
- The results of the simulation are stored in [`data/dLoss_simulation_random20.csv`](data/dLoss_simulation_random20.csv).

### Date - 25 June 2025

- Python library [`PyMC`](https://www.pymc.io/welcome.html) is used to implement Bayesian Linear Regression, which is used to estimate the defender's total losses based on the defender's allocation.
- The model is implemented in [`bayesian_estimation.ipynb`](bayesian_estimation.ipynb) using the data from [`data/dLoss_simulation_random20.csv`](data/dLoss_simulation_random20.csv).
- Since we don't know the priors for the model, we cannot use the Bayesian model to estimate the defender's total losses.

### Date - 04 July 2025

- We implement the defender's model using the LINGO solver, stored in [`LINGO models/defender_model.lng`](LINGO%20models/defender_model.lng). We use it for simulations using [`aGain_simulation.py`](aGain_simulation.py) to find the attacker's total gains with respect to the attacker's allocations, assuming the defender chooses the optimal allocation using the defender's model.
- The results of the simulation are stored in [`data/aGain_simulation_random100.csv`](data/aGain_simulation_random100.csv).
- We fit a quadratic regression model to the simulated data using [`estimation.py`](estimation.py), which will be used as a substitute for utility functions for both the attacker and defender in the iterative algorithm.
- The model coefficients are saved in the [`models`](models) folder.

### Date - 09 July 2025

- We use the estimated utility functions to find the optimal allocations for the attacker and defender using Gurobi, which is implemented in [`utility_calculation.py`](utility_calculation.py).
- We also find the utility values using the contest success function formula (as used in the paper) for the optimal allocations.
- The results are stored in [`results/utility_comparison.xlsx`](results/utility_comparison.xlsx).

### Date - 12 July 2025

- Python library [`PyDOE3`](https://pydoe3.readthedocs.io/en/latest/) is used to generate a Latin Hypercube Sample (LHS) for the simulations, which is implemented and updated in [`aGain_simulation.py`](aGain_simulation.py) and [`dLoss_simulation.py`](dLoss_simulation.py). The results are stored in [`data/aGain_simulation_lhs100_(1,1,0.1).csv`](data/aGain_simulation_lhs100_(1,1,0.1).csv) and [`data/dLoss_simulation_lhs100_(1,1,0.1).csv`](data/dLoss_simulation_lhs100_(1,1,0.1).csv), respectively.

- We fit quadratic regression models to this simulated data using [`estimation.py`](estimation.py). The model coefficients are saved in the [`models`](models) folder as [`aGain_model_lhs100.json`](models/aGain_qni_lhs100_(1,1,0.1).json) and [`dLoss_model_lhs100.json`](models/dLoss_qni_lhs100_(1,1,0.1).json).

- We use these estimated utility functions to carry out utility calculation using [`utility_calculation.py`](utility_calculation.py). The results are updated in [`results/utility_comparison.xlsx`](results/utility_comparison.xlsx).

### Date - 19 July 2025

- We implement an iterative algorithm to find optimal allocations for the attacker and defender. The algorithm is implemented in [`algorithm.py`](algorithm1.py). Explanation of the algorithn is provided [below](#algorithm-1---iterative-model-updation).

- The updated dataset is stored in [`data/dLoss_updated_lhs100.csv`](data/dLoss_updated_lhs100_(1,1,0.1).csv).

- The model coefficients calculated at each iteration are saved in the [`models/itermodels`](models/itermodels) folder. Logs of the iterations are saved in [`logs/algorithm1.log`](logs/algorithm1.log).

### Date - 01 August 2025

- We implement both simultaneous and sequential game models using the LINGO solver. The models are stored in [`LINGO models/simultaneous_game.lng`](LINGO%20models/simultaneous_game.lng) and [`LINGO models/sequential_game.lng`](LINGO%20models/sequential_game.lng).

- The function `game_lingo_model` in ['utils.py'](utils.py) is used to run the LINGO models and get the results.

- We implement a plotting function in [`plot_generator.py`](plot_generator.py) to generate plots for different algorithms by varying the parameters alpha, beta, and A. The plots are saved in the [`plots`](plots) folder.

## Date - 02 August 2025

- We implement a utility function class in [`utility_function.py`](utility_function.py) for all functions related to utility functions, including estimation and optimization for both the attacker and defender.

- The utility funciton class can be used in the iterative algorithm to optimize the utility functions. It currently supports **quadratic** utility functions, and can be extended to support other types of utility functions.

## Algorithms

### Algorithm 1 - Iterative Model Updation

```markdown
1. [Redundant] Initialize attacker and defender allocations a, d
2. While not converged:
3.         d* = argmin defender's estimated utility function
4.         a* = argmax attacker's actual utility function
5.         Calculate actual loss: y_new = dLoss(d*, a*)
6.         Add y_new to dLoss simulation dataset
7.         Update defender's model with new data
8. Return (a*, d*)
```

## To Do

- Try different estimators, preferably differentiable ones, so that the solver can optimize the model. [SVM, ANN maybe, etc.]
- Make plots, by varying A, B, alpha values in the probability success function. Pay attention to files where these values were fixed and change that.
- Also make greedy and random allocation methods, and compare results with the iterative algorithm.
- Do train-test splits for the regression models.
- Better convergence criteria for the iterative algorithm.

## Estimators

The following estimators are currently supported in the `UtilityFunction` class:

- **Quadratic**: Quadratic utility function, with no interactions between allocations.

## Abbreviations

The following abbreviations are used in defining some of the models and data files:

- lhs100: Latin Hypercube Sampling with 100 samples
- dLoss: Defender's Loss
- aGain: Attacker's Gain
- (1,1,0.1): Parameters for the utility functions, where 1 is the alpha value, 1 is the beta value, and 0.1 is the A value
- qni: Quadratic Non-Interactive
- random20: Random sampling of 20 data points

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

- Effect of valuations on estimated utility functions (and results) can be studied.
- Effect of budget too can be studied.

## References

<a id="1">[1]</a>
Peiqiu Guan, Meilin He, Jun Zhuang, Stephen C. Hora (2017) Modeling a Multitarget Attacker–Defender Game with Budget Constraints. Decision Analysis 14(2):87-107. <https://doi.org/10.1287/deca.2017.0346>
