from numpy import ndarray
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer_e5_text = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
model_e5_text = AutoModel.from_pretrained("intfloat/multilingual-e5-base")
model_e5_text.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_e5_text.to(device)


@torch.no_grad()
def get_text_e5_emb(text: str) -> ndarray:
    input_text = text.strip()
    encoded = tokenizer_e5_text(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    output = model_e5_text(**encoded)
    embeddings = output.last_hidden_state[:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.squeeze().cpu().numpy()
