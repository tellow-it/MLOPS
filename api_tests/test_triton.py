import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


def triton_infer(protocol_type: str, server_url: str, image_path: str, text_list: str):
    if protocol_type == "HTTPS":
        client = httpclient.InferenceServerClient(url=server_url)
    elif protocol_type == "GRPC":
        client = grpcclient.InferenceServerClient(url=server_url)
    else:
        raise ValueError("Not supported")

    imgs = []
    for i in range(len(image_path)):
        with open(image_path[i], "rb") as f:
            img_bytes = f.read()
            imgs.append(img_bytes)

    img_array = np.array(imgs, dtype=object).reshape(len(imgs), 1)

    if protocol_type == "HTTPS":
        in_img = httpclient.InferInput("raw_image", img_array.shape, "BYTES")
    elif protocol_type == "GRPC":
        in_img = grpcclient.InferInput("raw_image", img_array.shape, "BYTES")
    else:
        raise ValueError("Not supported")

    in_img.set_data_from_numpy(img_array)

    txt_bytes = [t.encode("utf-8") for t in text_list]
    txt_array = np.array(txt_bytes, dtype=object).reshape(len(txt_bytes), 1)
    if protocol_type == "HTTPS":
        in_txt = httpclient.InferInput("raw_text", txt_array.shape, "BYTES")
    elif protocol_type == "GRPC":
        in_txt = grpcclient.InferInput("raw_text", txt_array.shape, "BYTES")
    else:
        raise ValueError("Not supported")
    in_txt.set_data_from_numpy(txt_array)

    if protocol_type == "HTTPS":
        outputs = [
            httpclient.InferRequestedOutput("concatenate_emb")
        ]
    elif protocol_type == "GRPC":
        outputs = [
            grpcclient.InferRequestedOutput("concatenate_emb")
        ]
    else:
        raise ValueError("Not supported")

    response = client.infer(
        model_name="ensemble",
        inputs=[in_img, in_txt],
        outputs=outputs
    )
    concatenate_emb   = response.as_numpy("concatenate_emb")
    return concatenate_emb


if __name__ == "__main__":
    img_path1 = "img.png"
    img_path2 = "img.png"
    text1 = "Тестовый текст 1"
    text2 = "Тестовый текст 2"
    server_http = "localhost:9900"
    server_grpc = "localhost:9901"

    concatenate_emb_http = triton_infer(
        "HTTPS",
        server_http,
        [img_path1, img_path2],
        [text1, text2]
    )
    print("HTTP concatenate_emb:", concatenate_emb_http)

    concatenate_emb_grpc = triton_infer(
        "GRPC",
        server_grpc,
        [img_path1, img_path2],
        [text1, text2]
    )
    print("GRPC concatenate_emb:", concatenate_emb_grpc)
