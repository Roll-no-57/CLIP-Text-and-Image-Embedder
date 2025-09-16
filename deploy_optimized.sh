#!/bin/bash

# Cloud Run deployment script with improved configuration for CLIP service

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"cultivated-snow-469008-p6"}
SERVICE_NAME=${SERVICE_NAME:-"clip-text-and-image-embedder"}
REGION=${REGION:-"us-central1"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Building and deploying CLIP Embedding Service..."
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Push to Google Container Registry
echo "Pushing image to GCR..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run with optimized settings
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --project ${PROJECT_ID} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --concurrency 10 \
  --max-instances 3 \
  --set-env-vars "PYTHONUNBUFFERED=1,HF_HOME=/app/.cache/huggingface" \
  --port 8080

echo "Deployment complete!"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --project ${PROJECT_ID} \
  --region ${REGION} \
  --format "value(status.url)")

echo "Service URL: ${SERVICE_URL}"
echo "Health check: ${SERVICE_URL}/health"
echo ""
echo "Testing deployment..."

# Wait a bit for the service to start
sleep 10

# Test the health endpoint
python3 test_health.py ${SERVICE_URL} || echo "Note: Health check failed, but service may still be starting..."