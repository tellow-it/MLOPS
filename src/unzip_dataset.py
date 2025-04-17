import os
import pathlib
import zipfile

from core.logger import logger
from src.params import Dataset


def unzip_dataset_file() -> None:
    logger.info("Extracting data")
    zip_obj = zipfile.ZipFile(file=Dataset.zipped_dataset_path)
    for file in zip_obj.filelist:
        if (
            ".ipynb_checkpoints" not in file.filename
            and "__MACOSX" not in file.filename
            and ".json" in file.filename
        ):
            data: bytes = zip_obj.read(file.filename)
            filename = file.filename.replace("labeled_dataset/", "")
            file_path = os.path.join(
                Dataset.save_unzip_dataset_dir,
                filename
            )
            logger.info(f"Saving data to {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            pathlib.Path(file_path).write_bytes(data)


if __name__ == "__main__":
    unzip_dataset_file()
