

import joblib
import pandas as pd


class RestaurantRecommender:

    def __init__(self):

        self.model = joblib.load("models/svd_model.pkl")

        self.restaurants = pd.read_csv(
            "data/restaurants_cleaned.csv"
        )

        if "Restaurant_ID" not in self.restaurants.columns:
            for candidate in ["restaurant_id", "id", "Unnamed: 0", "Unnamed: 0.1"]:
                if candidate in self.restaurants.columns:
                    self.restaurants = self.restaurants.rename(columns={candidate: "Restaurant_ID"})
                    break

            if "Restaurant_ID" not in self.restaurants.columns:
                self.restaurants.insert(0, "Restaurant_ID", range(1, len(self.restaurants) + 1))

        if "Restaurant_Name" not in self.restaurants.columns:
            if "restaurant name" in self.restaurants.columns:
                self.restaurants = self.restaurants.rename(columns={"restaurant name": "Restaurant_Name"})

        if "City" not in self.restaurants.columns:
            if "area" in self.restaurants.columns:
                self.restaurants = self.restaurants.rename(columns={"area": "City"})

        if "Cuisine" not in self.restaurants.columns:
            if "cuisines type" in self.restaurants.columns:
                self.restaurants = self.restaurants.rename(columns={"cuisines type": "Cuisine"})

        if "Average_Cost_for_Two" not in self.restaurants.columns:
            if "avg cost (two people)" in self.restaurants.columns:
                self.restaurants = self.restaurants.rename(columns={"avg cost (two people)": "Average_Cost_for_Two"})

        if "Aggregate_Rating" not in self.restaurants.columns:
            if "rate (out of 5)" in self.restaurants.columns:
                self.restaurants = self.restaurants.rename(columns={"rate (out of 5)": "Aggregate_Rating"})

        self.ratings = pd.read_csv(
            "data/user_ratings.csv"
        )

    def get_unrated_restaurants(self, user_id):

        rated = self.ratings[
            self.ratings["User_ID"] == user_id
        ]["Restaurant_ID"].tolist()

        all_restaurants = self.restaurants[
            "Restaurant_ID"
        ].tolist()

        unrated = list(
            set(all_restaurants) - set(rated)
        )

        return unrated

    def recommend(self, user_id, top_n=10):

        if user_id not in self.ratings["User_ID"].tolist():
            raise ValueError(f"User ID {user_id} does not exist in the ratings data.")

        unrated = self.get_unrated_restaurants(user_id)

        predictions = []

        for restaurant in unrated:

            predicted = self.model.predict(
                user_id,
                restaurant
            )

            predictions.append(
                (
                    restaurant,
                    predicted.est
                )
            )

        predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = predictions[:top_n]

        result = []

        for restaurant_id, predicted_rating in recommendations:

            restaurant = self.restaurants[
                self.restaurants["Restaurant_ID"] == restaurant_id
            ].iloc[0]

            result.append({

                "Restaurant_ID": restaurant_id,

                "Restaurant_Name":
                restaurant["Restaurant_Name"],

                "City":
                restaurant["City"],

                "Cuisine":
                restaurant["Cuisine"],

                "Average_Cost_for_Two":
                restaurant["Average_Cost_for_Two"],

                "Aggregate_Rating":
                restaurant["Aggregate_Rating"],

                "Predicted_Rating":
                round(predicted_rating, 2)

            })

        return pd.DataFrame(result)

    def save(self, dataframe):

        dataframe.to_csv(
            "outputs/top_10_recommendations.csv",
            index=False
        )

        print("\nRecommendations Saved Successfully")


def main():

    raw_user_id = input("Enter User ID : ").strip()

    try:
        user_id = int(raw_user_id)
    except ValueError:
        user_id = 1
        print(f"Invalid user ID '{raw_user_id}'. Using default user ID {user_id}.")

    recommender = RestaurantRecommender()

    recommendations = recommender.recommend(
        user_id=user_id,
        top_n=10
    )

    print("\nTop Recommended Restaurants\n")

    print(recommendations)

    recommender.save(recommendations)


if __name__ == "__main__":

    main()