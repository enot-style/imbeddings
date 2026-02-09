# Imbeddings

**Imbeddings** (image + embeddings) is a lightweight FastAPI microservice that acts as a thin wrapper around **[Hugging Face Transformers](https://huggingface.co/docs/transformers)**, restricted to **image embedding extraction** from Vision Transformer (ViT)–based models such as **DINOv3, DINOv2, and standard ViT backbones**.

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

For each input image, Imbeddings produces a **single embedding**:

* **CLS embedding** — global image representation from the CLS token

The embedding is:

* derived from the same model outputs
* **L2-normalized** by default

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
                       EMB: CLS token representation
```

See `supported_models.txt` in the repo root for the full list of supported models.

You can add additional model if it strictly follows the same token layout and embedding semantics:
- `outputs.last_hidden_state` exists and has shape `(batch, tokens, dim)`
- token `0` is a single CLS token
- tokens `1..N` correspond to patch embeddings
- no additional special tokens are present

The model must be published on the [Hugging Face Hub](https://huggingface.co/models) and accessible using the provided Hugging Face access token.


### Design implications

* Embeddings are derived from the CLS token
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

* is applied to the CLS vector
* cannot be disabled in the current API

This ensures embeddings are immediately usable for cosine or dot-product similarity and most vector databases.

## Requirements & Deployment

### Before Publishing

Make sure you have, at minimum:
- an API gateway or reverse proxy with TLS termination
- an external auth/authorization service
- rate limiting and abuse protection
- basic monitoring (logs + metrics + alerts)

### Authentication (optional)

Set `IMBEDDINGS_API_KEYS` to a comma/space-separated list of API keys to require
`Authorization: Bearer <key>` on `POST /v1/embeddings`. Leave it empty to disable auth.

### Hugging Face access token

Imbeddings loads models from the Hugging Face Hub at runtime. You must provide a **Hugging Face access token** via the environment variable:

```bash
export HF_TOKEN=your_token
```

If the token is missing or does not have sufficient access, model loading will fail at startup.

To create a token:
- Sign in at Hugging Face and open the [Access Tokens page](https://huggingface.co/settings/tokens)
- Create a read-only token

The token is used **exclusively to download model weights**. This requirement is intentional and aligns with Hugging Face’s model access and licensing policies.

### Hugging Face model access approval

Some models on the [Hugging Face Hub](https://huggingface.co/models) are **gated** and require explicit access approval from the model owner (for example, `facebook/dinov3` models).

If a model is gated:
- you must request access on the model’s Hugging Face page
- access is granted per Hugging Face account
- the same account must be used to generate the access token

After access is approved, no additional configuration is required — the service will be able to download the model using a read-only token.

> ⚠️ **CUDA availability and model size**
>
> Imbeddings can run on CPU-only systems, and many models (including smaller DINOv3 / ViT variants) work reliably in CPU memory, including inside Docker containers. However, **larger models may fail to load or run without CUDA support** due to memory and performance constraints.
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

The easiest way to run Imbeddings on your server is via `docker compose`. 

Two Dockerfiles are provided:

- `Dockerfile.cpu` for CPU-only builds (PyTorch CPU wheels)
- `Dockerfile.gpu` for GPU builds (based on `pytorch/pytorch` CUDA images)

### CPU ONLY processing via Docker Compose

Create a `.env` file with required environment variables (at minimum `HF_TOKEN`):

```dotenv
HF_TOKEN=your_token
IMBEDDINGS_PORT=8000
```
Then run:

```bash
git clone https://github.com/enot-style/imbeddings.git
cd imbeddings
docker compose up -d
```

### GPU processing via Docker Compose

> ⚠️ **Docker GPU passthrough via NVIDIA Container Toolkit is supported only on Linux hosts.**

To use the GPU image, install:

- NVIDIA GPU driver (host)
- NVIDIA Container Toolkit (nvidia-docker2) so Docker can pass GPUs into containers
- Docker Engine + Docker Compose

### Choosing a PyTorch GPU image

Set `PYTORCH_TAG` in your `.env` file to use `Dockerfile.gpu`.
Choose a tag with a CUDA version supported by your NVIDIA driver.
If you are unsure which CUDA version is supported, check:

```bash
nvidia-smi
```

Available tags are listed here: https://hub.docker.com/r/pytorch/pytorch/tags
Prefer `*-runtime` images.

Set `PYTORCH_TAG` to the tag only, so the Dockerfile resolves to
`pytorch/pytorch:TAG`:

```dotenv
HF_TOKEN=your_token
PYTORCH_TAG=2.10.0-cuda13.0-cudnn9-runtime
IMBEDDINGS_PORT=8000
```

To start the service, run:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build
```

## Native installation

### Python

This project depends on `PyTorch` and Hugging Face `Transformers`. As of early 2026, Python 3.8–3.12 are supported.

Recommended: **Python 3.12**

Install supported python version using your package manager.

### Dependencies

Clone the repo and create virtual environment:

```bash
git clone https://github.com/enot-style/imbeddings.git
cd imbeddings
python3.12 -m venv .venv
source .venv/bin/activate
```

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

### Run once with Uvicorn

Create a `.env` file with required environment variables (at minimum `HF_TOKEN`), then run:

```bash
uvicorn service.main:app
```

This will run imbeddings with local-only access on the default Uvicorn port (8000).

If you want a one-off local run on another interface (e.g. `0.0.0.0`) or port, you can add `IMBEDDINGS_HOST` and `IMBEDDINGS_PORT` to the `.env` file and pass the host/port explicitly:

```bash
source .env && uvicorn service.main:app --host=$IMBEDDINGS_HOST --port=$IMBEDDINGS_PORT
```

### Running as a Linux Service (systemd)

See `systemd_howto.md` to get detailed instructions on how to run imbeddings as a linux service.

## API schema

* `GET /schema.json` — OpenAPI schema

### API compatibility and design goals

The Imbeddings API is intentionally designed to be **as close as possible to the OpenAI embeddings API**, while supporting image inputs.

Specifically:
- Request and response shapes follow OpenAI conventions (`model`, `data`, `usage`)
- Embeddings are returned as plain float arrays
- Responses are deterministic and stateless
- Batching semantics match OpenAI-style indexing and ordering

## SDKs

* Go SDK: https://github.com/enot-style/imbeddings-go-sdk

## Examples

See the `tests/` directory for:

* base64 image examples
* remote image URL examples
* batch requests
