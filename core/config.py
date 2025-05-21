import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    API_TITLE = os.getenv("API_TITLE")
    API_DESCRIPTION = os.getenv("API_DESCRIPTION")
    API_VERSION = os.getenv("API_VERSION")
    PROJECT_VERSION = os.getenv("PROJECT_VERSION")

    MLFLOW_PORT = os.getenv("MLFLOW_PORT")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    TRACKING_COMMIT_FILE = "tracking_commit.lock"

    SERVICE_MODEL_API = os.getenv("SERVICE_MODEL_API")
