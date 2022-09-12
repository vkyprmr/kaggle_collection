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
from prepare_data import AddFeatures, EncodeOneHot

# Loading the config
config = yaml.safe_load(open("config.yml", "r"))
af_config = config["adding_features"]
enc_config = config["encode"]


# Adding features
def adding_features():
    print("Adding features...")
    save_train = Path(af_config["save_train"])
    save_test = Path(af_config["save_test"])
    if not save_train.exists():
        print(f"Train:\n")
        train_af = AddFeatures(path_to_df=af_config["train"])
        train_df = train_af.add_features()
        train_df = train_df[["row_id", "store", "product", "date", "country", "day", "date_day", "month", "week", "year",
                             "quarter", "is_holiday", "is_weekend", "season", "covid_19", "num_sold"]]
        print(train_df.head())
        train_df.to_csv(af_config["save_train"], index=False)

    if not save_test.exists():
        print(f"\nTest:\n")
        test_af = AddFeatures(path_to_df=af_config["test"])
        test_df = test_af.add_features()
        test_df = test_df[
            ["row_id", "store", "product", "date", "country", "day", "date_day", "month", "week", "year", "quarter",
             "is_holiday", "is_weekend", "season", "covid_19"]]
        test_df.to_csv(af_config["save_test"], index=False)
        print(test_df.head())


# Encoding
def encode():
    print("Encoding...")
    save_train = Path(enc_config["save_train"])
    save_test = Path(enc_config["save_test"])
    if not save_train.exists():
        print(f"Train:\n")
        train_enc = EncodeOneHot(path_to_df=enc_config["train"],
                                 cols_to_encode=enc_config["cols_to_encode"],
                                 save_loc=enc_config["save_enc"])
        train_encoded = train_enc.encode()
        train_encoded.columns = [c.replace(" ", "_") for c in train_encoded.columns]
        train_encoded.columns = [c.lower() for c in train_encoded.columns]
        train_encoded.to_csv(save_train, index=False)
        print(train_encoded.head())

    if not save_test.exists():
        print(f"\nTest:\n")
        test_enc = EncodeOneHot(path_to_df=enc_config["test"],
                                cols_to_encode=enc_config["cols_to_encode"],
                                save_loc=enc_config["save_enc"])
        test_encoded = test_enc.encode()
        test_encoded.columns = [c.replace(" ", "_") for c in test_encoded.columns]
        test_encoded.columns = [c.lower() for c in test_encoded.columns]
        test_encoded.to_csv(save_test, index=False)
        print(test_encoded.head())


# Main
if __name__ == '__main__':
    # ToDo: Streamline the process
    blocks = [adding_features, encode]
    for block in blocks:
        block()
