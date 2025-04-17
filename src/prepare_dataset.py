import json
import os
import pathlib

import pandas as pd

from ml_product_categorizer.scripts.preprocessing import (
    clear_product_text,
    pipeline_extract_data_4_url,
)
from src.params import Dataset, PreparedDataset


def json_collection_2_structured_data(path_to_dir: str):
    extracting_data = []

    for labeled_task_file in os.listdir(path_to_dir):
        if labeled_task_file.endswith("json"):
            path_to_file = os.path.join(
                path_to_dir,
                labeled_task_file
            )
            with open(path_to_file, encoding="utf-8") as file:
                d = json.load(file)
                task_data = {"url": d["task"]["data"]["url"]}
                for field in d["result"]:
                    if field["from_name"] == "taxonomy":
                        taxonomy_key_name = list(field["value"].keys())[0]
                        taxonamy_value = field["value"][taxonomy_key_name][0]
                        if isinstance(taxonamy_value, list):
                            task_data["category"] = " ".join(taxonamy_value)
                        elif isinstance(taxonamy_value, str):
                            task_data["category"] = taxonamy_value
                    if field["from_name"] == "picture_url":
                        task_data["picture_url"] = field["value"]["text"][-1]
                    if field["from_name"] == "product_text":
                        task_data["product_text"] = " ".join(field["value"]["text"])
            extracting_data.append(task_data)

    return extracting_data


if __name__ == "__main__":
    labeled_structured_data = json_collection_2_structured_data(
        path_to_dir=Dataset.save_unzip_dataset_dir
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
    pathlib.Path(PreparedDataset.save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PreparedDataset.save_path, index=False)
