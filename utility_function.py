"""
Module that defines the UtilityFunction class
used for utility functions and estimations using different estimators
"""
import json
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import LinearRegression

class UtilityFunction():
    """Class for everything related to utility function and estimation"""
    def __init__(self, function_type, entity, **kwargs):
        """
        All the usual parameters required to solve the optimization problem
        Such as - n_targets, alpha, beta, A, b_list, d_list, t_budget, g_budget

        function_type : str
            The type of utility function we're using - ['quadratic','quadratic_int','actual']

        entity : str
            The type of entity - ['attacker', 'defender']
        """
        self.function_type = function_type
        self.entity = entity
        self.model = {} # Stores coefficients for the utility function
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.n_targets = getattr(self,"n_targets")

    def estimate_function(self, filepath, savepath=None):
        """
        Function to fit a regressor to the data and estimate the utility function

        Parameters
        ----------
        filepath: str
            Path to the data file used for fitting the model

        savepath: str (Optional)
            Path to save the fitted model coefficients
        """
        if self.function_type == 'quadratic':
            data = pd.read_csv(filepath)
            # Defining X and y for regression
            X_1 = np.array(data[data.columns[:self.n_targets]])   # Degree 1 features
            X_2 = X_1**2                                # Get quadratic features
            y = data[data.columns[-1]]                  # Gains/Losses
            X = np.hstack([X_1, X_2])                   # Final input with quadratic features

            model = LinearRegression()                  # Fitting a regression model
            model.fit(X,y)

            coefficients = {
                'intercept' : round(model.intercept_,2),
                'coefficients' : [round(coef,2) for coef in model.coef_]
                }

            self.model = coefficients
            if savepath:
                self.save_model(savepath)

    def utility_function(self, allocations, modelpath):
        """
        Using the model coefficients to return the utility value

        Parameters
        -----------
        allocations : np.array
            Entity's resource allocation to each target. Length must be n.

        model_path : str (Optional)
            Path to the utility model JSON file
        """
        if modelpath: # Path to the utility model JSON file
            self.open_model(modelpath)

        if self.function_type == 'quadratic':
            utility = self.model['intercept']
            for i,_ in enumerate(allocations):
                utility += self.model['coefficients'][i] * allocations[i] # Adding the Xi term
                # Adding the Xi^2 term
                utility += self.model['coefficients'][i+len(allocations)] * allocations[i] * allocations[i]
            return utility
        return None

    def optimize(self, modelpath=None):
        """
        Use Gurobi to optimize the utility function with budget constraints
        """
        targets = list(range(self.n_targets))
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:
                allocations = m.addVars(targets, vtype=GRB.CONTINUOUS, name="allocations") # Vars

                Z = self.utility_function(allocations, modelpath) # Objective function

                if self.entity == 'attacker':
                    m.setObjective(Z, GRB.MAXIMIZE) # Maximizing Gains for Attacker
                    budget = getattr(self,"t_budget")
                else:
                    m.setObjective(Z, GRB.MINIMIZE) # Minimizing Losses for Defender
                    budget = getattr(self,"g_budget")

                m.addConstr(allocations.sum() - budget == 0, name="budget") # Constraint
                m.optimize() # Optimize the model

                if m.status == GRB.OPTIMAL: # Output the solver status
                    print("Optimal solution found.")
                else:
                    print(f"Optimization ended with status {m.status}.")
                return np.array([v.x for v in m.getVars()])

    def save_model(self, savepath):
        """Save estimated utility function coefficients"""
        with open(savepath, "w", encoding="utf-8") as fp:
            json.dump(self.model, fp)

    def open_model(self, modelpath):
        """Read utility function coefficients, if not estimating"""
        with open(modelpath, "r", encoding="utf-8") as file:
            self.model = json.load(file)

if __name__ == "__main__":
    ENTITY = 'defender'
    F_TYPE = 'quadratic'
    FILEPATH = "data/dLoss_simulation_lhs100_(1,1,0.1).csv"
    SAVEPATH = "models/classtester.json"
    MODELPATH = "models/classtester.json"

    defender_utility = UtilityFunction(
        entity=ENTITY, 
        function_type=F_TYPE,
        n_targets = 5,
        t_budget = 5, g_budget = 30)
    # alpha = 1, beta = 1, A = 0.1,
    # b_list = np.array([20, 100, 50, 2, 20]),
    # d_list = np.array([70, 1000, 50, 75, 150]),

    print("Defined class instance!")
    defender_utility.estimate_function(filepath=FILEPATH, savepath=SAVEPATH)
    print("Fitted a model and saved it!")
    print("Model coefficients:", defender_utility.model)

    print("Model used:",defender_utility.model)
    d_star = defender_utility.optimize(modelpath=MODELPATH)
    print("Optimized allocations:", d_star)
