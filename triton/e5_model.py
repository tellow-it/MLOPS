import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class E5Model(nn.Module):
    def __init__(self, model_name="intfloat/multilingual-e5-small"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            torchscript=True
        )

    def forward(self, input_ids, attention_mask):
        last_hidden_states, _ = self.model(input_ids, attention_mask)
        model_outputs = self.average_pool(last_hidden_states, attention_mask)
        return F.normalize(model_outputs, p=2, dim=1)

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


texts = [
    "Журнальный стол Лава 7 BMS состоит из двух столиков разной высоты, "
    "которые отлично смотрятся по отдельности и в комплекте. "
    "Формы и размеры столешниц удачно дополняют друг друга, ",
    "Стильный комплект из жилетки и брюк. Жилет по переду выполнен из эко-кожи",
    "Зубная паста, детская со вкусом клубника и вишня "
]

if __name__ == "__main__":
    torch_model = E5Model()
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    batch_dict = tokenizer(
        texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    example_inputs = (batch_dict["input_ids"], batch_dict["attention_mask"])

    torch.onnx.export(
        torch_model,
        example_inputs,
        "triton/model_repository/e5_model_small/1/model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["text_emb"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {
                0: "batch_size"
            },
            "attention_mask": {
                0: "batch_size"
            },
            "text_emb": {
                0: "batch_size"
            }
        }
    )

    session = ort.InferenceSession(
        "triton/model_repository/e5_model_small/1/model.onnx",
        providers=["CPUExecutionProvider"]
    )

    for inp in session.get_inputs():
        print(inp.name, inp.shape)

    for out in session.get_outputs():
        print(out.name, out.shape)
