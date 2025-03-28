import csv
import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv()


session = boto3.Session()
s3_session = session.client(
service_name="s3",
endpoint_url="http://127.0.0.1:9000",
aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

CSV_FILE_PATH = r"data\mlops_url_dataset.csv"


def prepare_and_upload_files_s3() -> None:
    # Чтение CSV файла
    with open(CSV_FILE_PATH, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        k = 0
        for row in reader:
            url = row["url"]  # Предполагается, что в CSV есть колонка 'url'
            print(url)
            if not url:
                print("Пропущен пустой URL")
                continue

            # Создание JSON данных
            data = {"url": url}
            json_data = json.dumps(data, ensure_ascii=False)

            # Создание временного JSON файла
            temp_json_file = f"data/url_dataset/{k}.json"
            with open(temp_json_file, "w", encoding="utf-8") as json_file:
                json_file.write(json_data)

            # Загрузка JSON файла на S3
            s3_object_name = os.path.basename(temp_json_file)

            s3_session.upload_file(temp_json_file, "products-markup", s3_object_name)
            # # Удаление временного файла
            os.remove(temp_json_file)

            k+=1
