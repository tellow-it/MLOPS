from catboost import CatBoostClassifier

import mlflow
from core.config import Settings
from core.logger import logger
from mlflow.tracking import MlflowClient

MODEL_PATHS = {
    "CLASSIFIER_MODEL_CAT_L1": "/app/models/catboot_product_cat_l1.cbm",
    "CLASSIFIER_MODEL_CAT_L2": "/app/models/catboot_product_cat_l2.cbm",
    "CLASSIFIER_MODEL_CAT_L3": "/app/models/catboot_product_cat_l3.cbm",
}


def create_model():
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        eval_metric="Accuracy",
        random_seed=42,
        task_type="GPU",
        gpu_ram_part=0.6
    )


def log_and_register_model(model_name: str, model_path: str, client: MlflowClient):
    logger.info(f"Uploading {model_name}...")

    model = create_model()
    model.load_model(fname=model_path)

    mlflow.log_params(model.get_params())

    mlflow.catboost.log_model(
        cb_model=model,
        name=model_name,
        registered_model_name=model_name
    )

    version = client.get_latest_versions(model_name, stages=["None"])[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

    logger.info(f"Model {model_name} version {version} is now in Production.")


def upload_model_to_mlflow():
    mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Product categorizer e5 clip catboost 3 lvs test")

    client = MlflowClient()

    with mlflow.start_run():
        for model_name, model_path in MODEL_PATHS.items():
            log_and_register_model(model_name, model_path, client)
