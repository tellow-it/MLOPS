import base64
import binascii
import io
import re

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor

NAME_INPUT_IMAGE   = "raw_image"
NAME_OUTPUT_PIXELS = "pixel_values"

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def execute(self, requests):
        requests_sizes = []
        raws = []

        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, NAME_INPUT_IMAGE)
            arr = inp.as_numpy()
            self.logger.log_info(f"[img_preproc] Received input array dtype={arr.dtype}, shape={arr.shape}")

            flat = arr.flatten()
            requests_sizes.append(flat.shape[0])

            for idx, item in enumerate(flat):
                raw_bytes = None

                # CASE A: bytes/bytearray/memoryview
                if isinstance(item, (bytes, bytearray, memoryview)):
                    candidate = bytes(item)
                    self.logger.log_info(f"[img_preproc] Item#{idx} is bytes, len={len(candidate)}")
                    # Try as raw JPEG/PNG/etc.
                    try:
                        Image.open(io.BytesIO(candidate))
                        raw_bytes = candidate
                        self.logger.log_info(f"[img_preproc] Item#{idx} opened as raw image bytes")
                    except UnidentifiedImageError:
                        # Fallback: treat bytes as base64-encoded text
                        s = candidate.decode("utf-8", errors="ignore")
                        s = re.sub(r"\s+", "", s)
                        if s.startswith("data:") and "," in s:
                            s = s.split(",", 1)[1]
                        try:
                            raw_bytes = base64.b64decode(s, validate=False)
                        except binascii.Error:
                            pad = "=" * ((4 - len(s) % 4) % 4)
                            raw_bytes = base64.urlsafe_b64decode(s + pad)
                        self.logger.log_info(f"[img_preproc] Item#{idx} decoded from base64, len={len(raw_bytes)}")

                # CASE B: numpy.ndarray of uint8
                elif isinstance(item, np.ndarray) and item.dtype == np.uint8:
                    raw_bytes = item.tobytes()
                    self.logger.log_info(f"[img_preproc] Item#{idx} is ndarray(uint8), len={len(raw_bytes)}")

                # CASE C: str
                elif isinstance(item, str):
                    preview = item[:30] + ("…" if len(item) > 30 else "")
                    self.logger.log_info(f"[img_preproc] Item#{idx} is str, preview={preview!r}")
                    # Try raw bytes first
                    try:
                        candidate = item.encode("latin1", "ignore")
                        Image.open(io.BytesIO(candidate))
                        raw_bytes = candidate
                        self.logger.log_info(f"[img_preproc] Item#{idx} interpreted as raw bytes, len={len(raw_bytes)}")
                    except Exception:
                        s = re.sub(r"\s+", "", item)
                        if s.startswith("data:") and "," in s:
                            s = s.split(",", 1)[1]
                        try:
                            raw_bytes = base64.b64decode(s, validate=False)
                        except binascii.Error:
                            pad = "=" * ((4 - len(s) % 4) % 4)
                            raw_bytes = base64.urlsafe_b64decode(s + pad)
                        self.logger.log_info(f"[img_preproc] Item#{idx} decoded from base64, len={len(raw_bytes)}")

                else:
                    msg = f"[img_preproc] Unexpected input type: {type(item)}"
                    self.logger.log_warn(msg)
                    raise TypeError(msg)
                if not raw_bytes:
                    msg = f"[img_preproc] Decoded bytes empty for item#{idx}"
                    self.logger.log_error(msg)
                    raise UnidentifiedImageError(msg)

                raws.append(raw_bytes)

        # Convert to PIL and verify
        images = []
        for idx, b in enumerate(raws):
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                images.append(img)
                self.logger.log_info(f"[img_preproc] Successfully opened image#{idx}")
            except UnidentifiedImageError:
                msg = f"[img_preproc] ERROR: cannot identify image#{idx}, head={b[:10]!r}"
                self.logger.log_error(msg)
                raise

        # Process with CLIPProcessor
        proc = self.processor(images=images, return_tensors="np")
        pixels = proc["pixel_values"]

        # Build responses
        responses = []
        offset = 0
        for size in requests_sizes:
            batch = pixels[offset : offset + size]
            offset += size
            out_t = pb_utils.Tensor(NAME_OUTPUT_PIXELS, batch)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))

        return responses

    def finalize(self):
        pass
