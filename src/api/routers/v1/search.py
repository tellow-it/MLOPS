from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi_cache.decorator import cache
from starlette import status

from src.ann.search import get_urls_by_indexes, search_in_index
from src.ann.search_indexes import index_flat, index_hnsw, urls_in_index
from src.scripts.image_processing.embedding_image import get_embedding
from src.scripts.image_processing.get_image import download_image

router_search = APIRouter(prefix="/search", tags=["Search"])


@router_search.get("/flat-index", response_model=Optional[list[str]])
@cache(expire=30)
async def flat_index_search(url: str) -> Optional[list[str]]:
    """Semantic search in FlatIndex by url"""
    image_array = await download_image(url)
    if image_array is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    embedded_image = get_embedding(image_array)
    query_vector = np.expand_dims(embedded_image, axis=0)
    _, indexes = search_in_index(index=index_flat, query_vector=query_vector)
    urls = get_urls_by_indexes(urls=urls_in_index, indexes=indexes)
    return urls


@router_search.get("/hnsw-index", response_model=Optional[list[str]])
@cache(expire=30)
async def hnsw_index_search(url: str) -> Optional[list[str]]:
    """Semantic search in HnswIndex by url"""
    image_array = await download_image(url)
    if image_array is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    embedded_image = get_embedding(image_array)
    query_vector = np.expand_dims(embedded_image, axis=0)
    _, indexes = search_in_index(index=index_hnsw, query_vector=query_vector)
    urls = get_urls_by_indexes(urls=urls_in_index, indexes=indexes)
    return urls
