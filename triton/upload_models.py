import os

import boto3
from dotenv import load_dotenv

load_dotenv()


session = boto3.Session()
s3_session = session.client(
    service_name="s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
)

CLIP_MODEL_PATH = "triton/model_repository/clip_model/1/model.onnx"
E5_MODEL_PATH = "triton/model_repository/e5_model_small/1/model.onnx"


def upload_file_s3(path_to_file, s3_object_name) -> None:
    s3_session.upload_file(path_to_file, "iva.tikhonov.gitlab", s3_object_name)


upload_file_s3(CLIP_MODEL_PATH, CLIP_MODEL_PATH)
upload_file_s3(E5_MODEL_PATH, E5_MODEL_PATH)
