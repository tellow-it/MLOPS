from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from starlette.middleware.cors import CORSMiddleware

from core.config import Settings
from core.logger import logger
from src.api.routers.base_router import base_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    redis = aioredis.from_url("redis://redis:6379")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield


app = FastAPI(
    description=Settings.API_DESCRIPTION,
    title=Settings.API_TITLE,
    version=Settings.PROJECT_VERSION,
    debug=False,
    lifespan=lifespan
)

app.include_router(base_router)


@app.exception_handler(Exception)
async def unexpected_exception_handler(
        request: Request,
        exc: Exception
) -> JSONResponse:
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred. Please try again later.",
            "detail": str(exc)
        }
    )


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
