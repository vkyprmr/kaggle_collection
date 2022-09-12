"""
Training script
"""
"""
Author: vickyparmar
File: victuner.py
Created on: 09-09-2022, Fri, 18:43:08
"""
"""
Last modified by: vickyparmar
Last modified on: 12-9-2022, Mon, 16:46:31
"""

# Imports
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from math import sqrt
import optuna


# Class VicTuner
class VicTuner:
    """
            Description of the class.

            ...

            Attributes
            ----------
            attr1 : type
                Description of attribute 1.

            Methods
            -------
            method1(param='abcd')
                Description of method 1.
            method2(param=xyz)
                Description of method 2.

            Returns
            -------
            return type
                Description of the return value.

            Raises
            ------
            type of Exception raised
                Description of raised exception.

            See Also
            --------
            othermodule : Other module to see.

            Notes
            -----
            The FFT is a fast implementation of the discrete Fourier transform:

            .. deprecated:: version
              `ndobj_old` will be removed in NumPy 2.0.0, it is replaced by
              `ndobj_new` because the latter works also with array subclasses.

            Example
            -------
            >>> provided an example
            """
    def __init__(self, X, y, save_loc, random_state=42):
        self.X, self.y = X, y
        self.save_loc = save_loc
        self.random_state = random_state
        self.scoring = [mean_absolute_error, mean_squared_error, make_scorer(self.root_mse), make_scorer(self.SMAPE),
                        r2_score]

    # Root-Mean-Squared-Error
    @staticmethod
    def root_mse(y_true, y_pred):
        return sqrt(mean_squared_error(y_true, y_pred))

    # SMAPE
    @staticmethod
    def SMAPE(y_true, y_pred):
        smape = np.mean(
            np.abs(y_pred - y_true) /
            ((np.abs(y_pred) + np.abs(y_true)) / 2)
        ) * 100
        return smape

    # Objective functions
    def xgb_objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 10, 1),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 1.0),
            "booster": trial.suggest_categorical("booster", ["gblinear", "dart", "gbtree"]),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-7, 1.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-7, 1.0),
        }
        xgb = XGBRegressor(random_state=self.random_state, eval_metric=mean_absolute_error, n_jobs=-1, **params)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(estimator=xgb, X=self.X, y=self.y, cv=cv, scoring=self.scoring)
        mae = cv_results["mean_absolute_error"]
        mse = cv_results["mean_squared_error"]
        rmse = cv_results["root_mse"]
        smape = cv_results["SMAPE"]
        r2 = cv_results["r2_score"]
        return mae, mse, rmse, smape, r2


