import os

import boto3
from dotenv import load_dotenv

load_dotenv()

def test_s3_connection() -> None:
    session = boto3.Session()
    s3 = session.client(
        service_name="s3",
        endpoint_url="http://127.0.0.1:9000",
        aws_access_key_id=os.environ["LOCAL_AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["LOCAL_AWS_SECRET_ACCESS_KEY"]
    )

    buckets = s3.list_buckets()["Buckets"]
    for bucket in buckets:
        print("Bucket name: ", bucket["Name"])
