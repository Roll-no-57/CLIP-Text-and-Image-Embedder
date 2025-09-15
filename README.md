# CLIP & Multilingual Embedding Service

## Overview
This repository contains two FastAPI microservices that expose OpenCLIP-based encoders as simple HTTP APIs. The default service (`app.py`) serves the ViT-B/32 CLIP checkpoint for English text, while the multilingual variant (`m-clip.py`) pairs a multilingual text encoder with the CLIP ViT-B-16-plus image tower. Both services return L2-normalized embeddings and expose identical REST routes for text, image, or mixed requests.

## Key Features
- Text, image, and mixed embedding endpoints backed by OpenCLIP models
- Automatic L2 normalization for cosine-similarity ready vectors
- Robust image downloading with retries, redirects, and content-type validation
- Structured logging to surface slow downloads and inference failures
- Health probe suitable for container orchestration and Cloud Run
- Optional multilingual text encoding powered by `multilingual-clip`

## Repository Layout
| Path | Description |
| --- | --- |
| `app.py` | FastAPI service using the CLIP ViT-B/32 (laion400m) checkpoint. |
| `m-clip.py` | FastAPI service that uses Multilingual CLIP text encoder + CLIP image tower. |
| `test_service.py` | Smoke-test script that exercises every endpoint against a running instance. |
| `requirements.txt` | Dependencies for the single-language CLIP service. |
| `requirements_v2.txt` | Dependencies for the multilingual service (superset of `requirements.txt`). |
| `Dockerfile` | Container recipe (Python 3.9 slim) prepared for Cloud Run. |
| `deploy.sh` | Helper script that deploys the container to Google Cloud Run with `gcloud`. |
| `.dockerignore`, `.gitignore` | Development hygiene helpers for Docker and Git. |

## Service Variants
- **Standard CLIP (`app.py`)** – loads `ViT-B/32` with laion400m weights and OpenCLIP tokenizer. Text embeddings are English-centric.
- **Multilingual CLIP (`m-clip.py`)** – loads `M-CLIP/XLM-Roberta-Large-Vit-B-16Plus` for text and `ViT-B-16-plus-240` for images. Requires packages from `requirements_v2.txt`.

Both variants share the same request/response models. Swap the import when launching the API to choose the encoder that matches your use case.

## Prerequisites
- Python 3.9+
- pip (or compatible package manager)
- Git (for cloning the repository)
- (Optional) CUDA-compatible GPU + PyTorch build with CUDA if you plan to move models to GPU manually

## Installation
```bash
git clone <repository-url>
cd Text_and_image_ermbeddings
python -m venv .venv
# Activate the virtual environment
# Windows
. .venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Install dependencies (choose one set)
# For the standard CLIP service
pip install -r requirements.txt
# For the multilingual service
pip install -r requirements_v2.txt
```

## Running the API Locally
### Standard CLIP Service
```bash
# Ensure requirements.txt is installed
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
# or use the convenience entry point
python app.py
```

### Multilingual Service
```bash
# Ensure requirements_v2.txt is installed
uvicorn m-clip:app --host 0.0.0.0 --port 8080 --reload
# or run directly
python m-clip.py
```

Both services bind to port `8080` by default. When running in managed environments (e.g., Cloud Run or Heroku) ensure the process listens on the port provided by the platform.

## API Reference
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Basic service banner with model metadata. |
| `GET` | `/health` | Readiness probe – returns 200 when models are loaded. |
| `POST` | `/embed/text` | Generate embeddings for a list of text strings. |
| `POST` | `/embed/image` | Generate embeddings for a list of image URLs. |
| `POST` | `/embed/mixed` | Process optional text and image batches in one call. |

### POST `/embed/text`
Request body:
```json
{
  "texts": ["Hello world", "A photo of a cat"]
}
```
Successful response (`200 OK`):
```json
{
  "embeddings": [[0.031, -0.127, ...], [...]]
}
```

### POST `/embed/image`
Request body:
```json
{
  "urls": [
    "https://raw.githubusercontent.com/pytorch/examples/main/cpp/dcgan/assets/real_samples.png"
  ]
}
```
Successful response (`200 OK`):
```json
{
  "embeddings": [[-0.021, 0.114, ...]]
}
```

### POST `/embed/mixed`
Request body:
```json
{
  "texts": ["A beautiful landscape", "Computer vision"],
  "urls": ["https://raw.githubusercontent.com/pytorch/examples/main/cpp/dcgan/assets/real_samples.png"]
}
```
Successful response (`200 OK`):
```json
{
  "text_embeddings": [[0.041, -0.018, ...], [...]],
  "image_embeddings": [[-0.012, 0.222, ...]]
}
```

### GET `/health`
Response (`200 OK`):
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
Multilingual variant additionally returns `text_model_loaded` and `image_model_loaded` flags.

### Response Format and Errors
- Embedding vectors contain floating-point values (512 dimensions for the supplied checkpoints).
- Vectors are already L2 normalized and safe for cosine similarity comparisons.
- HTTP `400` – validation failure (e.g., empty `texts` or `urls`).
- HTTP `503` – startup has not finished loading the model.
- HTTP `500` – unexpected encoding failure (full trace logged server-side).
- Failed image downloads are reported in logs and skipped from the returned list.

## Testing the Service
The repository includes `test_service.py`, which performs live smoke tests against a running instance.

```bash
# Start the API in a separate shell
uvicorn app:app --host 0.0.0.0 --port 8080

# In a new terminal (same virtual environment)
python test_service.py
```
The script waits for `/health` to become ready, calls each endpoint, prints dimensions, and exercises error handling. It fetches a public sample image from GitHub; ensure outbound network access is available.

## Docker Workflow
1. Build the image:
   ```bash
   docker build -t clip-embedding-service .
   ```
2. Run the container:
   ```bash
   docker run --rm -p 8080:8080 clip-embedding-service
   ```

The supplied `Dockerfile` installs `requirements_v2.txt`, copies `m-clip.py`, and defaults to running `uvicorn app:app`. Adjust the final `COPY`/`CMD` directives to match the service you want to serve:
- For the standard service, copy `app.py` into the image and keep the command.
- For the multilingual service, either rename `m-clip.py` to `app.py` before building or update `CMD` to use `m-clip:app`.

## Deploying to Google Cloud Run
`deploy.sh` automates a source-based Cloud Run deployment.
```bash
./deploy.sh <PROJECT_ID> <REGION>
```
The script prompts for confirmation, enables required APIs, builds with Cloud Build, and deploys with sensible defaults (`2 vCPU`, `2Gi` memory, 15-minute timeout). After completion it prints the service URL and handy `curl` commands for verification.

## Configuration & Environment
- `LOG_LEVEL` – optional override consumed by `deploy.sh` during deployment (logging defaults to INFO).
- `PORT` – Cloud Run sets this automatically; ensure the launched uvicorn process listens on that port.
- Network egress must be allowed for image downloads when using `/embed/image` or `/embed/mixed`.

## Logging & Observability
- Python `logging` is configured at INFO level, emitting download successes/failures and embedding counts.
- Health endpoints (`/` and `/health`) are lightweight and safe for readiness checks.
- Container healthcheck in the Dockerfile polls `/health` every 30 seconds.

## Troubleshooting
- **Model load failure** – verify that `open_clip` and `torch` versions satisfy the requirements, and that there is enough memory (>1 GiB) during startup.
- **Slow or failed image downloads** – ensure outbound HTTPS connectivity; problematic URLs are logged with retry attempts.
- **Empty embeddings array** – indicates the download failed; inspect logs for a matching warning.
- **GPU usage** – the services default to CPU; modify the startup logic to move models to `cuda` if available.

## License
This project is distributed under the MIT License.
