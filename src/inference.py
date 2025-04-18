import argparse
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

from core.logger import logger
from ml_product_categorizer.scripts.preprocessing import (
    clear_product_text,
    pipeline_extract_data_4_url,
)
from src.prepare_dataset import json_collection_2_structured_data

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(prog="inference")
    parser.add_argument("--input-path", "-i", required=True)
    parser.add_argument("--output-path", "-o", required=True)

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labeled_structured_data = json_collection_2_structured_data(
        path_to_dir=input_path
    )
    df = pd.DataFrame(labeled_structured_data)
    df = df.dropna()
    df = df.drop_duplicates(subset="url")
    df = df.reset_index(drop=True)

    df["url_keywords_cleaned"] = df["url"].apply(
        pipeline_extract_data_4_url
    )
    df["picture_url_keywords_cleaned"] = df["picture_url"].apply(
        pipeline_extract_data_4_url
    )
    df["product_text_cleaned"] = df["product_text"].apply(
        clear_product_text
    )
    df["total_text"] = (
            df["product_text_cleaned"] +
            df["url_keywords_cleaned"] +
            df["picture_url_keywords_cleaned"]
    )
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model_uri = os.getenv("MLFLOW_INFERENCE_MODEL_NAME")
    logger.info(f"Start loading model from mlflow: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Start prediction")
    predictions = model.predict(df)
    result_df = pd.DataFrame({"category_id": predictions})
    result_df.to_csv(output_path, index=False)
    logger.info(f"Success save result prediction to {output_path}")
