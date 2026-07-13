

import pandas as pd
import numpy as np


class DataPreprocessor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Load CSV Dataset
        """
        print("Loading dataset...")
        self.df = pd.read_csv(self.file_path)

        print("\nDataset Loaded Successfully")
        print("Shape:", self.df.shape)

        return self.df

    def display_basic_info(self):

        print("\n-----------------------------")
        print("Dataset Information")
        print("-----------------------------")

        print(self.df.info())

        print("\nFirst Five Records")
        print(self.df.head())

        print("\nLast Five Records")
        print(self.df.tail())

    def remove_duplicates(self):

        before = self.df.shape[0]

        self.df.drop_duplicates(inplace=True)

        after = self.df.shape[0]

        print(f"\nRemoved {before-after} duplicate rows")

    def missing_values(self):

        print("\nMissing Values")
        print("---------------------------")
        print(self.df.isnull().sum())

    def fill_missing_values(self):

        print("\nHandling Missing Values...")

        for column in self.df.columns:

            series = self.df[column]

            if pd.api.types.is_numeric_dtype(series):

                self.df[column] = series.fillna(series.median())

            else:

                self.df[column] = series.fillna("Unknown")

        print("Missing Values Filled")

    def standardize_columns(self):

        print("\nStandardizing Text Columns")

        object_columns = self.df.select_dtypes(include=["object", "string"]).columns

        for column in object_columns:

            self.df[column] = (
                self.df[column]
                .astype(str)
                .str.strip()
                .str.title()
            )

    def convert_datatypes(self):

        if "Aggregate_Rating" in self.df.columns:

            self.df["Aggregate_Rating"] = self.df["Aggregate_Rating"].astype(float)

        if "Average_Cost_for_Two" in self.df.columns:

            self.df["Average_Cost_for_Two"] = self.df["Average_Cost_for_Two"].astype(int)

        if "Votes" in self.df.columns:

            self.df["Votes"] = self.df["Votes"].astype(int)

    def create_features(self):

        print("\nCreating Extra Features...")

        if "Aggregate_Rating" in self.df.columns:

            self.df["Popularity_Score"] = (
                self.df["Aggregate_Rating"] *
                np.log1p(self.df["Votes"])
            )

        if "Average_Cost_for_Two" in self.df.columns:

            self.df["Cost_Category"] = pd.cut(

                self.df["Average_Cost_for_Two"],

                bins=[0,500,1000,2000,5000],

                labels=[
                    "Low",
                    "Medium",
                    "High",
                    "Premium"
                ]
            )

    def restaurant_statistics(self):

        print("\nRestaurant Statistics")
        print("-----------------------------")

        print(self.df.describe())

    def save_cleaned_data(self, output_path):

        self.df.to_csv(output_path, index=False)

        print(f"\nCleaned Dataset Saved to {output_path}")


def main():

    processor = DataPreprocessor("data/zomato.csv")

    processor.load_data()

    processor.display_basic_info()

    processor.remove_duplicates()

    processor.missing_values()

    processor.fill_missing_values()

    processor.standardize_columns()

    processor.convert_datatypes()

    processor.create_features()

    processor.restaurant_statistics()

    processor.save_cleaned_data("data/restaurants_cleaned.csv")


if __name__ == "__main__":

    main()