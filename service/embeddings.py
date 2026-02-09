from typing import List, Literal

import torch
from PIL import Image


@torch.inference_mode()
def embed_images(
    images: List[Image.Image],
    processor,
    model,
    device,
    normalize: bool = True,
    pooling: Literal["cls", "mean"] = "cls",
) -> List[List[float]]:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # (batch, tokens, dim)

    if pooling == "mean":
        if hidden.shape[1] > 1:
            vecs = hidden[:, 1:, :].mean(dim=1)
        else:
            vecs = hidden[:, 0, :]
    else:
        vecs = hidden[:, 0, :]

    if normalize:
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

    return vecs.cpu().tolist()
