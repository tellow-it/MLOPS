import io

import aiohttp
from PIL import Image
from PIL.ImageFile import ImageFile

from core.logger import logger


async def download_image(url: str, timeout_sec: int = 10) -> ImageFile | None:
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.get(url, timeout=timeout)
            if response.status == 200:
                image_data = await response.read()
                return Image.open(io.BytesIO(image_data)).convert("RGB")
            return None
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return None
