from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    zipped_dataset_path: str = "data/raw/labeled_dataset.zip"
    save_unzip_dataset_dir: str = "data/unzip/labeled_dataset"


@dataclass(frozen=True)
class PreparedDataset:
    save_path: str = "data/prepared/labeled_dataset.csv"


@dataclass(frozen=True)
class TfIdf:
    max_features: int = 50
    stop_words: str = "english"


@dataclass(frozen=True)
class Model:
    random_state: int = 777
    n_estimators: int = 200
    max_depth: int = 8
    max_features: int = 5
