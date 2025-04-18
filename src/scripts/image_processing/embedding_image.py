import numpy as np
import torch
from PIL.ImageFile import ImageFile
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_embedding(image: ImageFile) -> np.ndarray:
    try:
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(image)
        return embedding.squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
