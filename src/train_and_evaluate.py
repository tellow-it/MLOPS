import os
from dataclasses import asdict

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core.config import Settings
from src import params


def get_predictions_and_metrics(y_true, y_pred) -> dict:
    metrics_ = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
        zero_division=0.0
    )["macro avg"]
    return metrics_


if __name__ == "__main__":
    load_dotenv()

    mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

    with mlflow.start_run() as run:
        with open(Settings.TRACKING_COMMIT_FILE, "w") as f:
            f.write(f"{run.info.run_id}")

        model_params = {f"Model.{k}": v for k, v in asdict(params.Model()).items()}
        tfidf_params = {f"TfIdf.{k}": v for k, v in asdict(params.TfIdf()).items()}
        mlflow.log_params(model_params | tfidf_params)

        df = pd.read_csv(params.PreparedDataset.save_path)

        X = df["total_text"]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["category"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=params.Model.random_state)

        mlflow.log_metrics(
            {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "total_size": len(df)
            }
        )

        tfidf = TfidfVectorizer(
            max_features=params.TfIdf.max_features,
            stop_words=params.TfIdf.stop_words
        )
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=params.Model.n_estimators,
            max_depth=params.Model.max_depth,
            max_features=params.Model.max_features,
            random_state=params.Model.random_state
        )
        rf.fit(X_train_tfidf, y_train)

        train_metrics = get_predictions_and_metrics(y_train, rf.predict(X_train_tfidf))
        test_metrics = get_predictions_and_metrics(y_test, rf.predict(X_test_tfidf))

        metrics = {f"train_{k}": v for k, v in train_metrics.items()} | {
            f"test_{k}": v for k, v in test_metrics.items()
        }

        mlflow.log_metrics(metrics)

        input_sample = X_train_tfidf[:5]
        output_sample = rf.predict(X_train_tfidf[:5])
        signature = infer_signature(input_sample, output_sample)

        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="model",
            input_example=input_sample,
            signature=signature,
        )
