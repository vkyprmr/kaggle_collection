"""
The main script which works like an amazing pipeline.
"""
"""
Author: vickyparmar
File: main.py
Created on: 09-09-2022, Fri, 17:37:54
"""
"""
Last modified by: vickyparmar
Last modified on: 09-9-2022, Fri, 17:37:57
"""

# Imports
import yaml
from pathlib import Path
import pandas as pd
from prepare_data import AddFeatures, EncodeOneHot
from victuner import RegTuner
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

# Loading the config
config = yaml.safe_load(open("config.yml", "r"))
af_config = config["adding_features"]
enc_config = config["encode"]
tune_config = config["tuner"]


# Adding features
def adding_features():
    logger.info("Adding features...")
    save_train = Path(af_config["save_train"])
    save_test = Path(af_config["save_test"])
    if not save_train.exists():
        logger.info(f"Train:\n")
        train_af = AddFeatures(path_to_df=af_config["train"])
        train_df = train_af.add_features()
        train_df = train_df[["row_id", "store", "product", "date", "country", "day", "date_day", "month", "week", "year",
                             "quarter", "is_holiday", "is_weekend", "season", "covid_19", "num_sold"]]
        logger.info(train_df.head())
        train_df.to_csv(af_config["save_train"], index=False)

    if not save_test.exists():
        logger.info(f"\nTest:\n")
        test_af = AddFeatures(path_to_df=af_config["test"])
        test_df = test_af.add_features()
        test_df = test_df[
            ["row_id", "store", "product", "date", "country", "day", "date_day", "month", "week", "year", "quarter",
             "is_holiday", "is_weekend", "season", "covid_19"]]
        test_df.to_csv(af_config["save_test"], index=False)
        logger.info(test_df.head())


# Encoding
def encode():
    logger.info("Encoding...")
    save_train = Path(enc_config["save_train"])
    save_test = Path(enc_config["save_test"])
    if not save_train.exists():
        logger.info(f"Train:\n")
        train_enc = EncodeOneHot(path_to_df=enc_config["train"],
                                 cols_to_encode=enc_config["cols_to_encode"],
                                 save_loc=enc_config["save_enc"])
        train_encoded = train_enc.encode()
        train_encoded.columns = [c.replace(" ", "_") for c in train_encoded.columns]
        train_encoded.columns = [c.lower() for c in train_encoded.columns]
        train_encoded.to_csv(save_train, index=False)
        logger.info(train_encoded.head())

    if not save_test.exists():
        logger.info(f"\nTest:\n")
        test_enc = EncodeOneHot(path_to_df=enc_config["test"],
                                cols_to_encode=enc_config["cols_to_encode"],
                                save_loc=enc_config["save_enc"])
        test_encoded = test_enc.encode()
        test_encoded.columns = [c.replace(" ", "_") for c in test_encoded.columns]
        test_encoded.columns = [c.lower() for c in test_encoded.columns]
        test_encoded.to_csv(save_test, index=False)
        logger.info(test_encoded.head())


# Tuning
def tune():
    logger.info("Tuning hyper-parameters...")
    df = pd.read_csv(tune_config["train"], index_col="row_id")
    X = df.drop(["num_sold", "date"], axis=1, inplace=False)
    X.columns = [c.replace(":", "_") for c in X.columns]
    y = df["num_sold"]
    tuner = RegTuner(X=X, y=y, save_loc=tune_config["save_loc"],
                     objective_functions=tune_config["objective_functions"],
                     n_trials=tune_config["n_trials"],
                     random_state=tune_config["random_state"],
                     n_jobs=tune_config["n_jobs"])
    tuner.tune()
    logger.info(f"Tuning complete. Results saved @:\n{tune_config['save_loc']}")


# Main
if __name__ == '__main__':
    # ToDo: Streamline the process
    blocks = [adding_features, encode, tune]
    for block in blocks:
        block()
