
import os

import joblib
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split


class SVDRecommendationModel:

    def __init__(self, ratings_file):

        self.ratings_file = ratings_file
        self.trainset = None
        self.testset = None
        self.model = None

    def load_dataset(self):

        print("\nLoading User Ratings Dataset...")

        ratings_df = pd.read_csv(self.ratings_file)

        temp_ratings_path = "data/surprise_ratings.csv"

        ratings_df[["User_ID", "Restaurant_ID", "Rating"]].to_csv(
            temp_ratings_path,
            index=False,
            header=False
        )

        reader = Reader(
            rating_scale=(1, 5),
            line_format="user item rating",
            sep="," 
        )

        data = Dataset.load_from_file(
            temp_ratings_path,
            reader=reader
        )

        self.trainset, self.testset = train_test_split(
            data,
            test_size=0.20,
            random_state=42
        )

        print("Dataset Loaded Successfully")

    def train_model(self):

        print("\nTraining SVD Model...")

        self.model = SVD(
            n_factors=100,
            n_epochs=30,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )

        self.model.fit(self.trainset)

        print("Training Completed")

    def evaluate(self):

        print("\nEvaluating Model...")

        predictions = self.model.test(self.testset)

        rmse = accuracy.rmse(predictions)

        mae = accuracy.mae(predictions)

        print("\nEvaluation Completed")

        return predictions, rmse, mae

    def save_model(self):

        print("\nSaving Model...")

        joblib.dump(
            self.model,
            "models/svd_model.pkl"
        )

        print("Model Saved Successfully")


def main():

    model = SVDRecommendationModel(
        "data/user_ratings.csv"
    )

    model.load_dataset()

    model.train_model()

    predictions, rmse, mae = model.evaluate()

    model.save_model()


if __name__ == "__main__":

    main()