import faiss
import numpy as np
import pandas as pd

from core.logger import logger

urls_df = pd.read_parquet("data/urls_hw3_emb.parquet")

urls_in_index = urls_df["local_picture"].tolist()
embeddings = np.vstack(urls_df["embeddings"].values)

d = embeddings.shape[1]
np.random.seed(1234)

index_flat = faiss.IndexFlatL2(d)
index_flat.add(embeddings)
logger.info(f"Flat Index  num of rows: {index_flat.ntotal}")

M = 32

index_hnsw = faiss.IndexHNSWFlat(
    d, M, faiss.METRIC_L2
)
index_hnsw.hnsw.efConstruction = 40
index_hnsw.add(embeddings)
logger.info(f"HNSW Index  num of rows: {index_hnsw.ntotal}")
