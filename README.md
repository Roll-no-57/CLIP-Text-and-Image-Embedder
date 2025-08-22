# CLIP Embedding Service

A lightweight FastAPI microservice for generating embeddings from images and text using the CLIP model (ViT-B/32, laion400m pretrained).

## Features

- **Text Embeddings**: Generate embeddings from text strings
- **Image Embeddings**: Generate embeddings from image URLs
- **Mixed Embeddings**: Process both text and images in a single request
- **Normalized Embeddings**: All embeddings are L2 normalized
- **Error Handling**: Graceful handling of failed image downloads
- **Health Checks**: Built-in health check endpoints

## API Endpoints

### 1. Text Embeddings
```http
POST /embed/text
Content-Type: application/json

{
  "texts": ["Hello world", "Another text example"]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ]
}
```

### 2. Image Embeddings
```http
POST /embed/image
Content-Type: application/json

{
  "urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.png"
  ]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ]
}
```

### 3. Mixed Embeddings
```http
POST /embed/mixed
Content-Type: application/json

{
  "texts": ["A photo of a cat"],
  "urls": ["https://example.com/cat.jpg"]
}
```

**Response:**
```json
{
  "text_embeddings": [[0.1, 0.2, 0.3, ...]],
  "image_embeddings": [[0.4, 0.5, 0.6, ...]]
}
```

### 4. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Text_and_image_ermbeddings

# Install dependencies
pip install -r requirements.txt

# Run the service
python app.py
```

The service will be available at `http://localhost:8080`.

### Using uvicorn directly
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## Docker Deployment

### Build the Docker image
```bash
docker build -t clip-embedding-service .
```

### Run the container
```bash
docker run -p 8080:8080 clip-embedding-service
```

## Google Cloud Run Deployment

### Prerequisites
- Google Cloud SDK installed and configured
- Docker installed
- Project with Cloud Run API enabled

### Deploy to Cloud Run

1. **Build and push to Google Container Registry:**
```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Build and tag the image
docker build -t gcr.io/$PROJECT_ID/clip-embedding-service .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/clip-embedding-service
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy clip-embedding-service \
  --image gcr.io/$PROJECT_ID/clip-embedding-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 10
```

### Alternative: Direct deployment from source
```bash
gcloud run deploy clip-embedding-service \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

## Usage Examples

### Python Client Example
```python
import requests
import json

# Service URL (replace with your deployed URL)
BASE_URL = "http://localhost:8080"

# Text embedding example
text_response = requests.post(
    f"{BASE_URL}/embed/text",
    json={"texts": ["Hello world", "Computer vision"]}
)
text_embeddings = text_response.json()["embeddings"]

# Image embedding example
image_response = requests.post(
    f"{BASE_URL}/embed/image",
    json={"urls": ["https://example.com/image.jpg"]}
)
image_embeddings = image_response.json()["embeddings"]

# Mixed embedding example
mixed_response = requests.post(
    f"{BASE_URL}/embed/mixed",
    json={
        "texts": ["A beautiful sunset"],
        "urls": ["https://example.com/sunset.jpg"]
    }
)
mixed_result = mixed_response.json()
```

### cURL Examples
```bash
# Text embeddings
curl -X POST "http://localhost:8080/embed/text" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Machine learning"]}'

# Image embeddings
curl -X POST "http://localhost:8080/embed/image" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/image.jpg"]}'

# Health check
curl "http://localhost:8080/health"
```

## Configuration

### Environment Variables
- `PORT`: Service port (default: 8080)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Configuration
The service uses the CLIP ViT-B/32 model with laion400m pretraining. The model is loaded on CPU by default for cost efficiency on Cloud Run.

### Resource Requirements
- **Memory**: 2GB recommended (1GB minimum)
- **CPU**: 2 vCPU recommended
- **Storage**: ~500MB for model weights

## Performance Considerations

- **Batch Processing**: Send multiple texts/images in a single request for better throughput
- **Image Size**: Large images are automatically resized during preprocessing
- **Timeout**: Default request timeout is 30 seconds for image downloads
- **Concurrency**: Service supports concurrent requests

## Error Handling

- Invalid image URLs return empty embeddings with logged warnings
- Network timeouts for image downloads are handled gracefully
- Model loading errors prevent service startup
- Input validation ensures proper request format

## Monitoring

The service includes built-in logging and health check endpoints for monitoring:
- `/health` - Service health status
- Structured logging for request processing
- Error tracking for failed image downloads

## License

This project is licensed under the MIT License.
