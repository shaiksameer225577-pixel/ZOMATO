
import pandas as pd
import numpy as np
import random

NUM_USERS = 1000
MIN_VISITS = 10
MAX_VISITS = 30
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class UserRatingGenerator:

    def __init__(self, restaurant_file):
        self.restaurant_file = restaurant_file
        self.restaurants = None
        self.ratings = []

    def load_restaurants(self):
        print("Loading restaurant dataset...")
        self.restaurants = pd.read_csv(self.restaurant_file)

        if "Restaurant_ID" not in self.restaurants.columns:
            for candidate in ["restaurant_id", "id", "Unnamed: 0", "Unnamed: 0.1"]:
                if candidate in self.restaurants.columns:
                    self.restaurants = self.restaurants.rename(columns={candidate: "Restaurant_ID"})
                    break

            if "Restaurant_ID" not in self.restaurants.columns:
                self.restaurants.insert(0, "Restaurant_ID", range(1, len(self.restaurants) + 1))

        print("Dataset Loaded Successfully")
        print("Total Restaurants:", len(self.restaurants))

    def generate_ratings(self):

        restaurant_ids = self.restaurants["Restaurant_ID"].tolist()

        print("\nGenerating User Ratings...\n")

        for user in range(1, NUM_USERS + 1):

            visits = random.randint(MIN_VISITS, MAX_VISITS)

            visited_restaurants = random.sample(
                restaurant_ids,
                min(visits, len(restaurant_ids))
            )

            for restaurant in visited_restaurants:

                rating = self.generate_rating()

                self.ratings.append({
                    "User_ID": user,
                    "Restaurant_ID": restaurant,
                    "Rating": rating
                })

        print("Ratings Generated Successfully")

    def generate_rating(self):

        probability = random.random()

        if probability < 0.05:
            return 1

        elif probability < 0.15:
            return 2

        elif probability < 0.35:
            return 3

        elif probability < 0.70:
            return 4

        else:
            return 5

    def create_dataframe(self):

        ratings_df = pd.DataFrame(self.ratings)

        print("\nGenerated Ratings")
        print(ratings_df.head())

        print("\nDataset Shape:", ratings_df.shape)

        return ratings_df

    def statistics(self, ratings_df):

        print("\n===========================")
        print("Ratings Statistics")
        print("===========================")

        print(ratings_df["Rating"].value_counts().sort_index())

        print("\nAverage Rating")

        print(ratings_df["Rating"].mean())

        print("\nTotal Users")

        print(ratings_df["User_ID"].nunique())

        print("\nTotal Restaurants")

        print(ratings_df["Restaurant_ID"].nunique())

    def save(self, ratings_df):

        ratings_df.to_csv(
            "data/user_ratings.csv",
            index=False
        )

        print("\nuser_ratings.csv Saved Successfully")


def main():

    generator = UserRatingGenerator(
        "data/restaurants_cleaned.csv"
    )

    generator.load_restaurants()

    generator.generate_ratings()

    ratings_df = generator.create_dataframe()

    generator.statistics(ratings_df)

    generator.save(ratings_df)


if __name__ == "__main__":

    main()