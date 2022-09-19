"""
Preparing train and test set with feature engineering
"""
"""
Author: vickyparmar
File: prepare_data.py
Created on: 09-09-2022, Fri, 16:22:27
"""
"""
Last modified by: vickyparmar
Last modified on: 13-9-2022, Tue, 17:35:25
"""

# Imports
import pickle
from pathlib import Path
import pandas as pd
import holidays
from category_encoders.one_hot import OneHotEncoder


# Class AddFeatures
class AddFeatures:
    """
            Adding Time-Series Features to the given DataFrame. Following features will be added:
                - day
                - is_weekend
                - is_holiday
                - season

            Attribute
            ---------
            path_to_df : str or Path
                A path to a .csv file containing at least two columns: date and country.

            Method
            ------
            add_features()
                Adds the above-mentioned features to the dataframe

            Notes
            -----
            This class is designed specifically for a kaggle competition. Please modify it to adapt it for other projects.

            ToDos
            ----
            ToDo: Fine-Tune season according to different countries

            Example
            -------
            >>> af = AddFeatures(path_to_df="/path/to/df.csv")
            >>> df = af.add_features()
            """
    def __init__(self, path_to_df):
        """
        Initializing a class instance

        Parameters
        ----------
        path_to_df: str or Path
            A path to a .csv file containing minimum two columns: date & country
        """
        self.df = pd.read_csv(path_to_df)
        self.df["date"] = pd.to_datetime(self.df["date"])

    # Adding the day of the week
    @staticmethod
    def get_day(date):
        """
        Getting the day of the week.

        Parameters
        ----------
        date: pd.Timestamp
            input date

        Returns
        -------
        day: str
            the day of the week
        """
        days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        day_num = date.weekday()
        return days[day_num]

    # Adding Weekend feature
    @staticmethod
    def is_weekend(day):
        """
        Checking whether the day is a weekend.

        Parameters
        ----------
        day: str
            Day of the week

        Returns
        -------
        is_weekend: int
            whether it is a weekend
        """
        if day.lower() in ["saturday", "sunday"]:
            return 1
        return 0

    # Adding Holiday feature
    @staticmethod
    def is_holiday(row):
        """
        Checking whether the day is a holiday.

        Parameters
        ----------
        row: pd.Series
            series containing at least 2 elements: date and country

        Returns
        -------
        is_holiday: int
            whether it is a holiday
        """
        if row["date"] in holidays.CountryHoliday(row["country"]):
            return 1
        return 0

    # Adding the season of the year
    @staticmethod
    def get_season(date):
        """
        Getting the season of the year

        Parameters
        ----------
        date: pd.Timestamp
            input date

        Returns
        -------
        season: str
            season of the year
        """
        if date.month in range(3, 6):
            return "Spring"
        elif date.month in range(6, 9):
            return "Summer"
        elif date.month in range(9, 12):
            return "Autumn"
        else:
            return "Winter"

    # Adding COVID-19 year
    @staticmethod
    def add_covid(date):
        """
        Adding if the year was COVID-19 Pandemic year

        Parameters
        ----------
        date: pd.Timestamp
            input date

        Returns
        -------
        covid_year: int
            covid-year or not

        """
        if date.year >= 2020:
            return 1
        return 0

    # Adding all the features
    def add_features(self):
        """
        Adding day, holiday, weekend, and seasonal features.

        Returns
        -------
        DataFrame: pd.DataFrame
            dataframe with added features.
        """
        self.df["day"] = self.df["date"].apply(lambda dt: self.get_day(dt))
        self.df["date_day"] = self.df["date"].apply(lambda dt: dt.strftime("%d"))
        self.df["month"] = self.df["date"].apply(lambda dt: dt.month)
        self.df["week"] = self.df["date"].apply(lambda dt: dt.week)
        self.df["year"] = self.df["date"].apply(lambda dt: dt.year)
        self.df["quarter"] = self.df["date"].apply(lambda dt: dt.quarter)
        self.df["is_holiday"] = self.df.apply(lambda row: self.is_holiday(row), axis=1)
        self.df["is_weekend"] = self.df["day"].apply(lambda d: self.is_weekend(d))
        self.df["season"] = self.df["date"].apply(lambda dt: self.get_season(dt))
        self.df["covid_19"] = self.df["date"].apply(lambda dt: self.add_covid(dt))
        return self.df


# Class EncodeOneHot
class EncodeOneHot:
    """
            Encoding the categorical variables as One Hot variables

            Attribute
            ----------
            path_to_df : str or Path
                A path to a .csv file to encode.
            cols_to_encode: list
                A list of columns to be encoded
            save_loc: str or Path
                A path where to save a pickle file containing the encoder.

            Method
            -------
            encode()
                Encodes the given dataframe and returns the encoded one

            Notes
            -----
            This class is designed specifically for a kaggle competition. Please modify it to adapt it for other projects.

            Example
            -------
            >>> encoder = EncodeOneHot(path_to_df="path/to/df", cols_to_encode=["col1", "col2"], save_loc="save/path")
            >>> encoded_df = encoder.encode()
            """
    def __init__(self, path_to_df, cols_to_encode, save_loc):
        """
        Initializing a class instance.

        Parameters
        ----------
        path_to_df: str or Path
            A path to a .csv file to encode.
        cols_to_encode: list
            A list of columns to be encoded
        save_loc: str
            A path where to save a pickle file containing the encoder.
        """
        self.df = pd.read_csv(path_to_df)
        self.concat = False
        if "train" in str(path_to_df):
            self.y = self.df["num_sold"]
            self.df.drop(["num_sold"], axis=1, inplace=True)
            self.concat = True
        self.cols = cols_to_encode
        self.save_loc = Path(save_loc)

    # Encoding
    def encode(self):
        """
        Encodes the dataframe.

        Returns
        -------
        DataFrame: pd.DataFrame
            Encoded dataframe
        """
        if self.save_loc.exists():
            print("Loading existing encoder...")
            with open(self.save_loc, 'rb') as e:
                ohe = pickle.load(e)
            encoded_df = ohe.transform(self.df)
            return encoded_df

        ohe = OneHotEncoder(cols=self.cols, use_cat_names=True)
        ohe.fit(self.df)
        with open(self.save_loc, 'wb') as e:
            pickle.dump(ohe, e)
        print(f"Encoder saved @ {self.save_loc}")
        encoded_df = ohe.transform(self.df)
        if self.concat:
            encoded_df = pd.concat([encoded_df, self.y], axis=1)
        return encoded_df
