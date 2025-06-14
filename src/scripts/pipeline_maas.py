import numpy as np

from src.ml.emeddings.emb_image import get_image_clip_emb
from src.ml.emeddings.emb_text import get_text_e5_emb
from src.scripts.parse.donwload_image import load_image_from_base64


def get_emb_by_data(text: str, image_base64: str = None):
    emb_text_e5 = get_text_e5_emb(text=text)
    emb_img_clip = np.zeros(512, dtype=np.float32)
    if image_base64:
        image_np = load_image_from_base64(image_base64)
        if image_np:
            emb_img_clip = get_image_clip_emb(image=image_np)

    return np.concatenate([emb_text_e5, emb_img_clip], axis=0)
