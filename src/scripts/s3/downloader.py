import os
import boto3

from core.config import Settings
from core.logger import logger


def download_data_from_s3():
    cat_dir = "data"
    models_dir = "models"
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    category_filename = "ru_ecomm_tree_category.json"
    cat_model_l1 = "catboot_product_cat_l1.cbm"
    cat_model_l2 = "catboot_product_cat_l2.cbm"
    cat_model_l3 = "catboot_product_cat_l3.cbm"

    session = boto3.Session()
    s3 = session.client(
        service_name="s3",
        endpoint_url=Settings.S3_ENDPOINT_URL,
        aws_access_key_id=Settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Settings.AWS_SECRET_ACCESS_KEY
    )

    if not os.path.exists(os.path.join(cat_dir, category_filename)):
        try:
            logger.info("Start loading categories...")
            with open(os.path.join(cat_dir, category_filename), "wb") as f:
                s3.download_fileobj(
                    Settings.BUCKET_NAME,
                    f"project_defence/{category_filename}",
                    f
                )
            logger.info("Success loading categories")
        except Exception as e:
            logger.error(f"Problems with downloading : {e}")

    for cat_model in [cat_model_l1, cat_model_l2, cat_model_l3]:
        local_path = os.path.join(models_dir, cat_model)

        if os.path.exists(local_path):
            logger.info(f"File already downloaded: {local_path}")
            continue

        try:
            logger.info(f"Start loading {cat_model}...")
            with open(os.path.join(models_dir, cat_model), "wb") as f:
                s3.download_fileobj(
                    Settings.BUCKET_NAME,
                    f"project_defence/{cat_model}",
                    f
                )
            logger.info(f"Success loading {cat_model}")
        except Exception as e:
            logger.error(f"Problems with downloading : {e}")


def upload_file_to_s3():
    cat_dir = "data"
    models_dir = "models"

    category_filename = "ru_ecomm_tree_category.json"
    cat_model_l1 = "catboot_product_cat_l1.cbm"
    cat_model_l2 = "catboot_product_cat_l2.cbm"
    cat_model_l3 = "catboot_product_cat_l3.cbm"

    session = boto3.Session()
    s3 = session.client(
        service_name="s3",
        endpoint_url=Settings.S3_ENDPOINT_URL,
        aws_access_key_id=Settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Settings.AWS_SECRET_ACCESS_KEY
    )

    if os.path.exists(os.path.join(cat_dir, category_filename)):
        try:
            logger.info("Start upload categories...")
            s3.upload_file(
                os.path.join(cat_dir, category_filename),
                Settings.BUCKET_NAME,
                f"project_defence/{category_filename}"
            )
            logger.info("Success upload categories")
        except Exception as e:
            logger.error(f"Problems with upload : {e}")

    for cat_model in [cat_model_l1, cat_model_l2, cat_model_l3]:
        local_path = os.path.join(models_dir, cat_model)

        if not os.path.exists(local_path):
            logger.info(f"File already downloaded: {local_path}")
            continue

        try:
            logger.info(f"Start upload {cat_model}...")
            s3.upload_file(
                os.path.join(models_dir, cat_model),
                Settings.BUCKET_NAME,
                f"project_defence/{cat_model}"
            )
            logger.info(f"Success upload {cat_model}")
        except Exception as e:
            logger.error(f"Problems with uploading : {e}")
