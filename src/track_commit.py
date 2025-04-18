import argparse

import mlflow

from core.config import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_and_evaluate")
    parser.add_argument("commit", nargs=1, type=str)
    commit = parser.parse_args().commit[0]

    with open(Settings.TRACKING_COMMIT_FILE) as f:
        run_id = f.read()

    mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)
    mlflow.start_run(run_id=run_id)
    mlflow.log_param("commit", commit)
