import io

import aiohttp
from bs4 import BeautifulSoup
from PIL import Image

from core.logger import logger


async def fetch_page_text(url: str) -> str | None:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            try:
                if response.status in [200, 201]:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    return soup.get_text(separator="\n", strip=True)
                return None
            except Exception as err:
                logger.error(str(err)[:100])
                return None


async def download_image(url, timeout=10):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return Image.open(io.BytesIO(image_data)).convert("RGB")
                return None
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return None
