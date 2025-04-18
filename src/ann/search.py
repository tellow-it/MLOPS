import numpy as np


def search_in_index(index, query_vector: np.ndarray, k: int = 5):
    distances, indexes = index.search(query_vector, k)
    indexes = indexes[0]
    return distances, indexes


def get_urls_by_indexes(urls: list[str], indexes: list[int]) -> list[str]:
    return [urls[index] for index in indexes]
