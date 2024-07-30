import pandas as pd
import tarfile
from pathlib import Path
import urllib.request


class DataLoader:
    def __init__(self):
        pass  # Constructor doesn't need to do anything here

    def download_data(self):
        """
        Downloads data from the url
        :return: DataFrame
        """
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        file_name = Path(url).name
        urllib.request.urlretrieve(url, file_name)
        with tarfile.open(file_name, "r") as f:
            f.extractall()
        return pd.read_csv("housing/housing.csv")

    def wrangle(self, df):
        """ Takes the downloaded data and wrangles it
        :param df: DataFrame
        :return: pd.DataFrame (wrangled)
        """
        # Create new attributes
        df["rooms_per_household"] = df["total_rooms"] / df["households"]
        df["people_per_household"] = df["population"] / df["households"]
        df["bedrooms_ratio"] = df["total_bedrooms"] / df["total_rooms"]

        # Drop multicollinearity columns
        drop_columns = ["total_rooms", "total_bedrooms", "population"]
        df.drop(columns=drop_columns, inplace=True)
        return df
