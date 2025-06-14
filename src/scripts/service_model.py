import aiohttp
from fastapi import HTTPException

from core.config import Settings
from core.logger import logger


async def predict_service_model(text: str, image_base64: str = None) -> dict | None:
    async with aiohttp.ClientSession() as session:
        data = {"text": text}
        if image_base64:
            data["image_base64"] = image_base64
        async with session.post(
                url=f"{Settings.SERVICE_MODEL_API}/predict",
                json=data,
                timeout=10
        ) as response:
            text_resp = await response.text()
            if response.status == 200:
                return await response.json()
            else:
                logger.error(text_resp)
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Problems with getting predict from MaaS: {text_resp}"
                )
