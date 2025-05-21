import aiohttp

from core.config import Settings
from core.logger import logger


async def predict_service_model(text: str) -> dict | None:
    async with aiohttp.ClientSession() as session:
        data = {
            "text": text
        }
        async with session.post(
                url=f"{Settings.SERVICE_MODEL_API}/predict",
                json=data,
                timeout=10
        ) as response:
            text_resp = await response.text()
            logger.info(f"{text_resp}")
            if response.status == 200:
                return await response.json()
            logger.error(text_resp)
            return None
