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
Last modified on: 13-9-2022, Tue, 17:35:15
"""

# Imports
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import optuna


# Logging
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


# Class RegTuner
class RegTuner:
    """
            Hyperparameter tuner for Regression tasks. It uses Optuna as a base.

            Parameter
            ---------
            X : pd.DataFrame or np.ndarray
                input DataFrame or array.
            y: pd.DataFrame or np.ndarray
                target DataFrame or array.
            save_loc: str
                location to save the results of the hyperparameter search.
            objective_functions: list
                a list of models / objective functions to run. See notes for currently available functions.
            n_trials: int
                number of trials (possibilities) to run the objective for.
            random_state: int
                an integer to make the results reproducible
            n_jobs: int
                number of parallel jobs to run

            Method
            ------
            tune()
                Runs all the objective functions specified aboved.

            Notes
            -----
            Currently available objective functions:
                - XGBRegressor: xgb
                - RandomForestRegressor: rf
                - DecisionTreeRegressor: dt
                - LightGBMRegressor: lgbm
                - LogisticRegression: logistic
                - KernelRidge: krr
                - SupportVectorRegressor: svr

            Example
            -------
            >>> df = pd.read_csv("any/data/you/need.csv")
            >>> X = df.drop("target", axis=1, inplace=False)
            >>> y = df["target"]
            >>> objective_functions = ["rf", "xgb"]
            >>> reg_tuner = RegTuner(X, y, objective_functions, n_trials=100, random_state=42, n_jobs=1)
            >>> reg_tuner.tune()
            """

    def __init__(self, X, y, save_loc, objective_functions, n_trials=100, random_state=42, n_jobs=1):
        """
        Initializing a class instance.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            input DataFrame or array.
        y: pd.DataFrame or np.ndarray
            target DataFrame or array.
        save_loc: str
            location to save the results of the hyperparameter search.
        objective_functions: list
            a list of models / objective functions to run. See notes for currently available functions.
        n_trials: int
            number of trials (possibilities) to run the objective for.
        random_state: int
            an integer to make the results reproducible
        n_jobs: int
            number of parallel jobs to run
        """
        self.X, self.y = X, y
        self.save_loc = Path(save_loc)
        self.random_state = random_state
        self.scoring = {
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "RMSE": "neg_root_mean_squared_error",
            "SMAPE": make_scorer(self.SMAPE, greater_is_better=False),
            "R2": "r2"
        }
        self.cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.objective_functions = objective_functions
        self.n_jobs = n_jobs
        self.n_trials = n_trials

    # SMAPE
    @staticmethod
    def SMAPE(y_true, y_pred):
        """
        Calculating SMAPE: Symmetric Mean Absolute Percentage Error.

        Parameters
        ----------
        y_true: np.ndarray or pd.DataFrame
            ground truth
        y_pred: np.ndarray or pd.DataFrame
            predictions

        Returns
        -------
        smape: float
            SMAPE

        """
        smape = np.mean(
            np.abs(y_pred - y_true) /
            ((np.abs(y_pred) + np.abs(y_true)) / 2)
        ) * 100
        return smape

    # CV-Results
    def cv_results(self, estimator):
        """
        Performs cross-validation with sklearn.model_selection.cross_validate

        Parameters
        ----------
        estimator:
            estimator to use for cross_validate

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        cv_results = cross_validate(estimator=estimator, X=self.X, y=self.y, cv=self.cv, scoring=self.scoring,
                                    n_jobs=self.n_jobs)
        mae = -np.mean(cv_results["test_MAE"])
        mse = -np.mean(cv_results["test_MSE"])
        rmse = -np.mean(cv_results["test_RMSE"])
        smape = -np.mean(cv_results["test_SMAPE"])
        r2 = np.mean(cv_results["test_R2"])
        return mae, mse, rmse, smape, r2

    # Objective functions
    # XGBRegressor
    def xgb_objective(self, trial):
        """
        Objective Function for XGBRegressor

        Parameters
        ----------
        trial: optuna.trial.Trial
            an optuna study trial

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 10, 1),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0),
            "booster": trial.suggest_categorical("booster", ["gblinear", "dart", "gbtree"]),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 1.0),
        }
        xgb = XGBRegressor(random_state=self.random_state, eval_metric=mean_absolute_error, n_jobs=self.n_jobs,
                           **params)
        mae, mse, rmse, smape, r2 = self.cv_results(estimator=xgb)
        return mae, mse, rmse, smape, r2

    # RandomForestRegressor
    def rf_objective(self, trial):
        """
        Objective Function for RandomForestRegressor

        Parameters
        ----------
        trial: optuna.trial.Trial
            an optuna study trial

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 10, 1),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "poisson"]),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.005, 0.1, step=0.05),
            "min_samples_leaf": trial.suggest_float("min_samples_split", 0.001, 0.05, step=0.05),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0, step=0.1),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0, step=0.1),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.1, step=0.05)
        }
        rf = RandomForestRegressor(oob_score=True, n_jobs=self.n_jobs, random_state=self.random_state, warm_start=False,
                                   **params)
        mae, mse, rmse, smape, r2 = self.cv_results(estimator=rf)
        return mae, mse, rmse, smape, r2

    # DecisionTreeRegressor
    def dt_objective(self, trial):
        """
        Objective Function for DecisionTreeRegressor

        Parameters
        ----------
        trial: optuna.trial.Trial
            an optuna study trial

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        params = {
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error",
                                                                 "poisson"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 2, 10, 1),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.005, 0.1, step=0.05),
            "min_samples_leaf": trial.suggest_float("min_samples_split", 0.001, 0.05, step=0.05),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0, step=0.1),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.1, step=0.05)
        }
        dt = DecisionTreeRegressor(random_state=self.random_state, **params)
        mae, mse, rmse, smape, r2 = self.cv_results(estimator=dt)
        return mae, mse, rmse, smape, r2

    # LightGBMRegressor
    def lgbm_objective(self, trial):
        """
        Objective Function for LightGBMRegressor

        Parameters
        ----------
        trial: optuna.trial.Trial
            an optuna study trial

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        params = {
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
            "num_leaves": trial.suggest_int("num_leaves", 15, 75, 10),
            "max_depth": trial.suggest_int("max_depth", 2, 10, 1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
        }
        lgbm = LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs, silent='warn', importance_type='gain',
                             **params)
        mae, mse, rmse, smape, r2 = self.cv_results(estimator=lgbm)
        return mae, mse, rmse, smape, r2

    # LogisticRegression
    def log_objective(self, trial):
        """
        Objective Function for LogisticRegression

        Parameters
        ----------
        trial: optuna.trial.Trial
            an optuna study trial

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        params = {
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"]),
            "C": trial.suggest_float("C", 1.0, 100.0),
            "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
        }
        log = LogisticRegression(random_state=self.random_state, n_jobs=self.n_jobs, **params)
        mae, mse, rmse, smape, r2 = self.cv_results(estimator=log)
        return mae, mse, rmse, smape, r2

    # LinearRegression
    def krr_objective(self, trial):
        """
        Objective Function for KernelRidgeRegressor

        Parameters
        ----------
        trial: optuna.trial.Trial
            an optuna study trial

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        params = {
            "alpha": trial.suggest_float("alpha", 0.001, 1.0, step=0.5),
            "kernel": trial.suggest_categorical("kernel", ["linear", "polynomial", "sigmoid",
                                                           "rbf", "laplacian", "chi2"]),
            "degree": trial.suggest_int("degree", 2, 5, 1),
        }
        krr = KernelRidge(**params)
        mae, mse, rmse, smape, r2 = self.cv_results(estimator=krr)
        return mae, mse, rmse, smape, r2

    # SupportVectorRegressor
    def svr_objective(self, trial):
        """
        Objective Function for SupportVectorRegressor

        Parameters
        ----------
        trial: optuna.trial.Trial
            an optuna study trial

        Returns
        -------
        mae, mse, rmse, smape, r2: (float, float, float, float, float)
            MAE, MSE, RMSE; SMAPE, R2

        """
        params = {
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"]),
            "degree": trial.suggest_int("degree", 2, 5, 1),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "C": trial.suggest_int("C", 1, 100)
        }
        svr = SVR(coef0=1.0, tol=0.001, epsilon=0.1, shrinking=True, cache_size=200,
                  verbose=False, max_iter=-1, **params)
        mae, mse, rmse, smape, r2 = self.cv_results(estimator=svr)
        return mae, mse, rmse, smape, r2

    # Optuna Study
    def create_and_run(self, objective_function):
        """
        Creating an Optuna study and optimizing the objective function.

        Parameters
        ----------
        objective_function:
            the objective function to be optimized.

        Returns
        -------
        study_df, tuner_object: (pd.DataFrame, dict)
            a DataFrame containing all trials, and
            a dictionary containing the following keys: "best_trial", "best_params", and "best_value"

        """
        study_name = objective_function.__name__
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=1)
        logger.info(f"Optimizing {study_name}...")
        study = optuna.create_study(study_name=study_name, pruner=pruner, load_if_exists=True,
                                    directions=["minimize", "minimize", "minimize", "minimize", "maximize"])
        study.optimize(objective_function, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True)
        study_df = study.trials_dataframe()
        best_trials = study.best_trials
        best = min(study.best_trials, key=lambda t: t.values[3])
        tuner_object = {
            "best_trial": best,
            "best_params": best.params,
            "best_value": best.values
        }
        return study_df, tuner_object

    # Finally running the study for all objective functions defined by the user
    def tune(self):
        """
        Tune the given objective functions.

        Returns
        -------
        None

        """
        func_map = {
            "xgb": self.xgb_objective,
            "rf": self.rf_objective,
            "dt": self.dt_objective,
            "lgbm": self.lgbm_objective,
            "logistic": self.log_objective,
            "krr": self.krr_objective,
            "svr": self.svr_objective
        }
        obj_funcs = [func_map[f] for f in self.objective_functions]
        for fnc in tqdm(obj_funcs, desc="Finding the best parameters"):
            fnc_name = fnc.__name__.split("_")[0]
            try:
                study_df, tuner_object = self.create_and_run(objective_function=fnc)
                logger.info(f"{tuner_object}")
                df_loc = self.save_loc / "csvs"
                params_loc = self.save_loc / "params"
                df_loc.mkdir(parents=True, exist_ok=True)
                params_loc.mkdir(parents=True, exist_ok=True)
                study_df.to_csv(df_loc / f"{fnc_name}.csv", index=False)
                logger.info(f"Best score for {fnc_name}: tuner_object['best_value']")
                logger.info(f"Best parameters:\ntuner_object['best_params']")
                with open(params_loc / f"{fnc_name}.pkl", "wb") as pf:
                    pickle.dump(tuner_object, pf)
                logger.info(f"Successfully optimized {fnc_name}!")
            except Exception as e:
                logger.warning(f"Unsuccessful in optimizing {fnc_name}...")
                logger.exception(f"\n{fnc_name}:\n{e}\n", exc_info=True)
