import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core.logger import logger
from src.params import Model, PreparedDataset, TfIdf

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

df = pd.read_csv(PreparedDataset.save_path)

X = df["total_text"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["category"])

X_train, X_test, y_train, y_test = train_test_split(X, y)
tfidf = TfidfVectorizer(
    max_features=TfIdf.max_features,
    stop_words=TfIdf.stop_words
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

mlflow.autolog()

logger.info("Starting run example...")

with mlflow.start_run():
    rf = RandomForestClassifier(
        n_estimators=Model.n_estimators,
        max_depth=Model.max_depth,
        max_features=Model.max_features
    )
    rf.fit(X_train_tfidf, y_train)
    prediction = rf.predict(X_test_tfidf)

logger.info("Example was successfully completed")
