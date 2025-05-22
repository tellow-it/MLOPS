import numpy as np
import triton_python_backend_utils as pb_utils

NAME_INPUT_TEXT_EMB = "text_emb"
NAME_INPUT_IMAGE_EMB = "image_emb"
NAME_OUTPUT_CONCATENATE = "concatenate_emb"

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            text_emb_tensor = pb_utils.get_input_tensor_by_name(request, NAME_INPUT_TEXT_EMB)
            image_emb_tensor = pb_utils.get_input_tensor_by_name(request, NAME_INPUT_IMAGE_EMB)

            # Получаем numpy-массивы из входных тензоров
            text_emb = text_emb_tensor.as_numpy()  # shape: [B, D1]
            image_emb = image_emb_tensor.as_numpy()  # shape: [B, D2]

            # Конкатенация вдоль последней оси (по признакам)
            concatenated = np.concatenate([text_emb, image_emb], axis=1)  # shape: [B, D1 + D2]

            # Создаём выходной тензор
            out_tensor = pb_utils.Tensor(NAME_OUTPUT_CONCATENATE, concatenated.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        pass
