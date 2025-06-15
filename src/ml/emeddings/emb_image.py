import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# Модель и процессор
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(
    "cuda" if torch.cuda.is_available() else "cpu")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad()
def get_image_clip_emb(image) -> np.ndarray:
    device = model_clip.device
    inputs = processor_clip(
        images=[image], return_tensors="pt", padding=True
    ).to(device)
    features = model_clip.get_image_features(**inputs)
    features = torch.nn.functional.normalize(features, p=2, dim=1).cpu().numpy()
    return features.squeeze()
