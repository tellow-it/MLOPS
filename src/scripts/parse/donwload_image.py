import base64
from io import BytesIO

import requests
from PIL import Image


def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception:
        return None


def load_image_from_base64(b64_string: str):
    try:
        if not b64_string:
            return None
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception:
        return None
