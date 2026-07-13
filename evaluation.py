
import os
import pandas as pd
import joblib

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


class ModelEvaluation:

    def __init__(self):

        self.model = joblib.load("models/svd_model.pkl")

        self.ratings = pd.read_csv(
            "data/user_ratings.csv"
        )

        self.trainset = None
        self.testset = None

    def prepare_dataset(self):

        print("Preparing Dataset...")

        reader = Reader(rating_scale=(1, 5))

        data = Dataset.load_from_df(
            self.ratings[
                ["User_ID", "Restaurant_ID", "Rating"]
            ],
            reader
        )

        self.trainset, self.testset = train_test_split(
            data,
            test_size=0.20,
            random_state=42
        )

    def evaluate(self):

        print("\nEvaluating Model...\n")

        predictions = self.model.test(self.testset)

        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)

        return predictions, rmse, mae

    def precision_recall_at_k(
        self,
        predictions,
        k=10,
        threshold=4
    ):

        from collections import defaultdict

        user_est_true = defaultdict(list)

        for uid, iid, true_r, est, _ in predictions:

            user_est_true[uid].append((est, true_r))

        precisions = {}
        recalls = {}

        for uid, user_ratings in user_est_true.items():

            user_ratings.sort(
                key=lambda x: x[0],
                reverse=True
            )

            n_rel = sum(
                (true_r >= threshold)
                for (_, true_r) in user_ratings
            )

            n_rec_k = sum(
                (est >= threshold)
                for (est, _) in user_ratings[:k]
            )

            n_rel_and_rec_k = sum(
                (
                    (true_r >= threshold)
                    and
                    (est >= threshold)
                )
                for (est, true_r)
                in user_ratings[:k]
            )

            precisions[uid] = (
                n_rel_and_rec_k / n_rec_k
                if n_rec_k != 0
                else 0
            )

            recalls[uid] = (
                n_rel_and_rec_k / n_rel
                if n_rel != 0
                else 0
            )

        precision = sum(
            precisions.values()
        ) / len(precisions)

        recall = sum(
            recalls.values()
        ) / len(recalls)

        f1 = (
            2 * precision * recall
            /
            (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print("\nPrecision@10 :", round(precision, 4))
        print("Recall@10    :", round(recall, 4))
        print("F1 Score     :", round(f1, 4))

        return precision, recall, f1

    def save_results(
        self,
        rmse,
        mae,
        precision,
        recall,
        f1
    ):

        os.makedirs("outputs", exist_ok=True)

        with open(
            "outputs/evaluation_metrics.txt",
            "w"
        ) as file:

            file.write("===== MODEL EVALUATION =====\n\n")

            file.write(f"RMSE : {rmse}\n")
            file.write(f"MAE : {mae}\n")
            file.write(f"Precision@10 : {precision}\n")
            file.write(f"Recall@10 : {recall}\n")
            file.write(f"F1 Score : {f1}\n")

        print("\nEvaluation report saved.")


def main():

    evaluation = ModelEvaluation()

    evaluation.prepare_dataset()

    predictions, rmse, mae = evaluation.evaluate()

    precision, recall, f1 = evaluation.precision_recall_at_k(
        predictions
    )

    evaluation.save_results(
        rmse,
        mae,
        precision,
        recall,
        f1
    )


if __name__ == "__main__":

    main()