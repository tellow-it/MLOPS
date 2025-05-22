import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

NAME_INPUT_TEXT = "raw_text"
NAME_OUTPUT_IDS = "input_ids"
NAME_OUTPUT_MASK = "attention_mask"

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.max_length = 512

    def execute(self, requests):
        total_size = 0
        requests_sizes = []
        raw_texts = []

        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, NAME_INPUT_TEXT)
            arr = inp.as_numpy().astype("object")
            flat = arr.flatten()
            size = flat.shape[0]
            requests_sizes.append(size)
            total_size += size
            for b in flat:
                if isinstance(b, (bytes, bytearray)):
                    raw_texts.append(b.decode("utf-8"))
                else:
                    raw_texts.append(str(b))

        enc = self.tokenizer(
            raw_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )

        responses = []
        offset = 0
        for size in requests_sizes:
            batch_ids   = enc["input_ids"][offset:offset+size]
            batch_mask  = enc["attention_mask"][offset:offset+size]
            offset += size

            out_ids  = pb_utils.Tensor(NAME_OUTPUT_IDS, batch_ids)
            out_mask = pb_utils.Tensor(NAME_OUTPUT_MASK, batch_mask)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_ids, out_mask]))

        return responses

    def finalize(self):
        pass
