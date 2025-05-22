import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import CLIPModel, CLIPProcessor


class ClipModel(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        clip = CLIPModel.from_pretrained(model_name, torchscript=True)
        self.vision = clip.vision_model.eval()
        self.proj = clip.visual_projection.eval()

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.vision(pixel_values)
        pooled = outputs.pooler_output
        proj = self.proj(pooled)
        return F.normalize(proj, p=2, dim=1)


if __name__ == "__main__":
    model_name = "openai/clip-vit-base-patch32"
    torch_model = ClipModel(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)

    image = np.random.randint(0, 255, (1024, 1024, 3))
    inputs = clip_processor(images=image, return_tensors="pt")["pixel_values"]

    torch.onnx.export(
        torch_model,
        inputs,
        "triton/model_repository/clip_model/1/model.onnx",
        input_names=["pixel_values"],
        output_names=["image_emb"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_emb": {0: "batch_size"}
        },
        opset_version=14,
    )

    session = ort.InferenceSession(
        "triton/model_repository/clip_model/1/model.onnx",
        providers=["CPUExecutionProvider"]
    )

    for inp in session.get_inputs():
        print(inp.name, inp.shape)

    for out in session.get_outputs():
        print(out.name, out.shape)
