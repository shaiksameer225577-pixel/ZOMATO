
import os
import subprocess


class ZomatoRecommendationSystem:

    def __init__(self):

        print("=" * 60)
        print("     ZOMATO RESTAURANT RECOMMENDATION SYSTEM")
        print("=" * 60)

    # Run Data Preprocessing
    def preprocess_data(self):

        print("\nRunning Data Preprocessing...\n")

        os.system("python3 data_preprocessing.py")

    # Generate User Ratings
    def generate_user_ratings(self):

        print("\nGenerating User Ratings...\n")

        os.system("python3 generate_user_ratings.py")

    # Train SVD Model
    def train_model(self):

        print("\nTraining Recommendation Model...\n")

        os.system("python3 svd_model.py")

    # Recommend Restaurants
    def recommend_restaurants(self):

        print("\nGenerating Recommendations...\n")

        os.system("python3 recommend.py")

    # Evaluate Model
    def evaluate_model(self):

        print("\nEvaluating Model...\n")

        os.system("python3 evaluation.py")

    # Exit
    def exit_program(self):

        print("\nThank You for Using This Project")
        print("Good Bye 😊")

        exit()

    # Menu
    def menu(self):

        while True:

            print("\n")

            print("1. Data Preprocessing")

            print("2. Generate User Ratings")

            print("3. Train SVD Model")

            print("4. Recommend Restaurants")

            print("5. Evaluate Model")

            print("6. Exit")

            choice = input("\nEnter Your Choice : ")

            if choice == "1":

                self.preprocess_data()

            elif choice == "2":

                self.generate_user_ratings()

            elif choice == "3":

                self.train_model()

            elif choice == "4":

                self.recommend_restaurants()

            elif choice == "5":

                self.evaluate_model()

            elif choice == "6":

                self.exit_program()

            else:

                print("\nInvalid Choice")


# Main Function

def main():

    system = ZomatoRecommendationSystem()

    system.menu()


if __name__ == "__main__":

    main()