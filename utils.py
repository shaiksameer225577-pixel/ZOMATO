
import os
import joblib
import pandas as pd


# Check if a file exists
def check_file_exists(file_path):

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nERROR: {file_path} not found."
        )

    return True


# Load CSV File
def load_csv(file_path):

    check_file_exists(file_path)

    df = pd.read_csv(file_path)

    return df


# Save CSV File
def save_csv(df, output_path):

    df.to_csv(output_path, index=False)

    print(f"\nFile saved successfully at:\n{output_path}")


# Create Folder if not Exists
def create_directory(folder_name):

    if not os.path.exists(folder_name):

        os.makedirs(folder_name)

        print(f"\nFolder Created: {folder_name}")

    else:

        print(f"\nFolder Already Exists: {folder_name}")


# Load Saved Model
def load_model(model_path):

    check_file_exists(model_path)

    model = joblib.load(model_path)

    return model


# Validate User ID
def validate_user(user_id, ratings_df):

    users = ratings_df["User_ID"].unique()

    if user_id not in users:

        print("\nInvalid User ID")

        return False

    return True


# Display Dataset Information
def dataset_information(df):

    print("\n========== DATASET INFO ==========")

    print(df.info())

    print("\nRows :", df.shape[0])

    print("Columns :", df.shape[1])

    print("\nMissing Values")

    print(df.isnull().sum())


# Restaurant Statistics
def restaurant_statistics(df):

    print("\n========== RESTAURANT STATISTICS ==========")

    print(df.describe())


# Print Recommendations
def display_recommendations(df):

    print("\n========== TOP RESTAURANTS ==========\n")

    print(df.to_string(index=False))


# Display User History
def display_user_history(user_id, ratings_df):

    history = ratings_df[
        ratings_df["User_ID"] == user_id
    ]

    print("\n========== USER HISTORY ==========\n")

    print(history)


# Model Information
def model_information(model):

    print("\n========== MODEL INFORMATION ==========\n")

    print(model)


# Save Evaluation Report
def save_report(report, file_name):

    with open(file_name, "w") as file:

        file.write(report)

    print("\nEvaluation Report Saved Successfully")


# Print Header

def header(title):

    print("\n")

    print("=" * 60)

    print(title.center(60))

    print("=" * 60)


# Print Footer
def footer():

    print("\n")

    print("=" * 60)

    print("End of Program".center(60))

    print("=" * 60)