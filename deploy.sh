#!/bin/bash

# Deployment script for Google Cloud Run
# Usage: ./deploy.sh [PROJECT_ID] [REGION]

set -e

# Default values
DEFAULT_REGION="us-central1"
SERVICE_NAME="clip-embedding-service"

# Get project ID and region
PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-$DEFAULT_REGION}

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID not provided and no default project set"
    echo "Usage: $0 [PROJECT_ID] [REGION]"
    echo "Or set default project: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "🚀 Deploying CLIP Embedding Service to Google Cloud Run"
echo "📋 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Service Name: $SERVICE_NAME"
echo ""

# Confirm deployment
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com --project=$PROJECT_ID

# Build and deploy using Cloud Build
echo "🏗️ Building and deploying service..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 900 \
    --max-instances 10 \
    --set-env-vars "LOG_LEVEL=INFO"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format="value(status.url)")

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "🧪 Test the service:"
echo "   Health check: curl $SERVICE_URL/health"
echo "   Text embedding: curl -X POST $SERVICE_URL/embed/text -H 'Content-Type: application/json' -d '{\"texts\": [\"Hello world\"]}'"
echo ""
echo "📊 Monitor the service:"
echo "   Logs: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --project=$PROJECT_ID"
echo "   Metrics: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics?project=$PROJECT_ID"
