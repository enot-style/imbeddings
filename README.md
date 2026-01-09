# Imbeddings

**Imbeddings** (image + embeddings) is a lightweight FastAPI microservice that acts as a thin wrapper around **Hugging Face Transformers**, restricted to **image embedding extraction** from Vision Transformer (ViT)–based models such as **DINOv3, DINOv2, and standard ViT backbones**.

Imbeddings focuses on **image → embedding (vector)** as a primitive. It does not add model-specific logic, post-processing, or task-level abstractions. How the resulting vectors are stored, compared, or used is intentionally outside the scope of this service.

## Use cases

Image embeddings produced by Imbeddings can be used as a general-purpose visual representation for a wide range of downstream tasks. 

Imbeddings is a good fit if you need:

* a **dedicated image embedding service** for similarity search, clustering, or indexing
* clean integration with **other backend systems**
* a **stable, minimal API** you control

Common use cases include: 
- image similarity search
- nearest-neighbor retrieval
- clustering and grouping of visually similar images
- duplicate or near-duplicate detection
- visual indexing in vector databases
- anomaly or outlier detection

## What Imbeddings does

For each input image, Imbeddings produces **two embeddings**:

* **CLS embedding** — global image representation from the CLS token
* **MEAN embedding** — mean-pooled representation of image patch tokens

Both embeddings:

* come from the same model
* live in the same vector space
* are **L2-normalized** by default

## What Imbeddings does *not* do

* No similarity or “compare” endpoints
* No task-specific heads (classification, detection, segmentation)
* No fine-tuning or training
* No text embeddings
* No model-specific pooling logic

Imbeddings is a **pure image embedding microservice**.

## Supported models & embedding semantics

Imbeddings **only supports Vision Transformer (ViT)–based models** that expose **exactly the same token output structure**:

```
last_hidden_state = [ CLS | PATCH_1 | PATCH_2 | ... | PATCH_N ]
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                              
                       MEAN: mean over tokens 1..N (CLS excluded)
```

See `supported_models.txt` in the repo root for the full list of supported models.

You can add additional model if it strictly follows the same token layout and embedding semantics:
- `outputs.last_hidden_state` exists and has shape `(batch, tokens, dim)`
- token `0` is a single CLS token
- tokens `1..N` correspond to patch embeddings
- no additional special tokens are present

The model must be published on the [Hugging Face Hub](https://huggingface.co/models) and accessible using the provided Hugging Face access token.


### Design implications

* CLS and MEAN are **distinct and complementary**
* MEAN pooling is **patch-only** (CLS is never included)
* All supported models behave identically at the tensor level
* No conditional logic or token filtering is performed by design

### Explicitly unsupported model types

The following model families are **intentionally excluded**:

* DINOv2 models **with register tokens**
* DeiT **distilled** models
* ConvNeXt-based models
* SAM, DPT, or other task-specific heads
* Any model that does not expose a clean ViT token sequence

If a model violates the token contract, it is **out of scope by design**.

### Normalization

All embeddings are **L2-normalized by default**.

Normalization:

* is applied independently to CLS and MEAN vectors
* cannot be disabled in the current API

This ensures embeddings are immediately usable for cosine or dot-product similarity and most vector databases.

## Requirements & Deployment

> ⚠️ **Security & production readiness notice**
>
> Imbeddings is currently provided as a **development tool**. It is **not production-ready** out of the box.
>
> In particular:
> - No authentication or authorization is implemented
> - API documentation endpoints (`/docs`, `/redoc`, `/openapi.json`) are publicly exposed
> - No rate limiting or abuse protection is enabled
> - No explicit security hardening or penetration testing has been performed

### Hugging Face access token

Imbeddings loads models from the Hugging Face Hub at runtime. You must provide a **Hugging Face access token** via the environment variable:

```bash
export HF_TOKEN=your_token
```

If the token is missing or does not have sufficient access, model loading will fail at startup.

To create a token:
- Sign in at https://huggingface.co
- Go to Settings → Access Tokens
- Create a read-only token

The token is used **exclusively to download model weights**. This requirement is intentional and aligns with Hugging Face’s model access and licensing policies.

### Hugging Face model access approval

Some models on the Hugging Face Hub are **gated** and require explicit access approval from the model owner (for example, `facebook/dinov3` models).

If a model is gated:
- you must request access on the model’s Hugging Face page
- access is granted per Hugging Face account
- the same account must be used to generate the access token

After access is approved, no additional configuration is required — the service will be able to download the model using a read-only token.

### Python

This project depends on `PyTorch` and Hugging Face `Transformers`. As of early 2026, Python 3.8–3.12 are supported.

Recommended: **Python 3.12**

### Dependencies

Install dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
#### PyTorch

**PyTorch** is not included in the `requirements.txt` and must be installed from the **official PyTorch repository**.

* Required version: **PyTorch 2.9 or later**

1. Open the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
2. Select:

   * Your OS
   * `Pip`
   * `Python`
3. (Optional) Check CUDA version:

   ```bash
   nvidia-smi
   ```
4. Install PyTorch and Torchvision using link, provided by the guide.  

> ⚠️ **CUDA availability and model size**
>
> Imbeddings can run on CPU-only systems, and many models (including smaller DINOv3 / ViT variants) work reliably in CPU memory, including inside Docker containers.
>
> However, **larger models may fail to load or run without CUDA support** due to memory and performance constraints.
> This typically manifests as out-of-memory errors, extremely slow startup, or runtime failures during inference.
>
> Examples:
> - Small / base models (e.g. `dinov3-vits16`, `dinov3-vitb16`) usually work on CPU
> - Large models (e.g. `dinov3-vitl16`, `dinov3-vith16plus`) slow in CPU, much faster with CUDA
> - Very large models (e.g. `dinov3-vit7b16`, WebSSL DINO models) **require CUDA** and sufficient GPU memory
>
> If you plan to use large or research-scale models, a CUDA-enabled environment is strongly recommended.
> CPU-only deployments should explicitly select an appropriate model size.


## Running with Docker

Imbeddings can be run using Docker or Docker Compose. The provided Docker setup is CPU-only and suitable for small to medium models.

### Build and run locally (Docker)

From the repository root:

```bash
docker build -t imbeddings .
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  imbeddings
```

### Run using Docker Compose

Create a `.env` file with required environment variables (at minimum `HF_TOKEN`), then run:

```bash
docker compose up --build
```

### Run without cloning the repository

You can run Imbeddings using Docker Compose without checking out the repository by building directly from the Git URL.

Create a `docker-compose.yaml` file:

```yaml
services:
  imbeddings:
    build:
      context: https://github.com/kotylevskiy/imbeddings.git
    image: imbeddings
    environment:
      HF_TOKEN: ${HF_TOKEN:?HF_TOKEN is required}
    ports:
      - "${UVICORN_PORT:-8000}:${UVICORN_PORT:-8000}"
    volumes:
      - hf_cache:/root/.cache/huggingface

volumes:
  hf_cache:
```

## API documentation

FastAPI automatically exposes:

* `GET /docs` — Swagger UI
* `GET /redoc` — ReDoc
* `GET /openapi.json` — OpenAPI schema

### API compatibility and design goals

The Imbeddings API is intentionally designed to be **as close as possible to the OpenAI embeddings API**, while supporting image inputs and multiple embedding types.

Specifically:
- Request and response shapes follow OpenAI conventions (`model`, `data`, `usage`)
- Embeddings are returned as plain float arrays
- Responses are deterministic and stateless
- Batching semantics match OpenAI-style indexing and ordering

## Examples

See the `tests/` directory for:

* base64 image examples
* remote image URL examples
* batch requests
