from fastapi import APIRouter

from core.config import Settings
from src.api.routers.v1.search import router_search

base_router = APIRouter(prefix=f"/api/{Settings.API_VERSION}")

base_router.include_router(router_search)
