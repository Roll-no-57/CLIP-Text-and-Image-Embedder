"""
FastAPI microservice for generating embeddings from images and text using CLIP model.
"""

import io
import logging
import time
import asyncio
from typing import List, Optional, Union
import requests
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CLIP Embedding Service",
    description="A microservice for generating embeddings from images and text using CLIP model",
    version="1.0.0"
)

# Global variables for model and preprocessing
model = None
preprocess = None
tokenizer = None

class TextEmbeddingRequest(BaseModel):
    """Request model for text embedding endpoint."""
    texts: List[str]

class ImageEmbeddingRequest(BaseModel):
    """Request model for image embedding endpoint."""
    urls: List[HttpUrl]

class MixedEmbeddingRequest(BaseModel):
    """Request model for mixed embedding endpoint."""
    texts: Optional[List[str]] = None
    urls: Optional[List[HttpUrl]] = None

class EmbeddingResponse(BaseModel):
    """Response model for embedding endpoints."""
    embeddings: List[List[float]]

class MixedEmbeddingResponse(BaseModel):
    """Response model for mixed embedding endpoint."""
    text_embeddings: Optional[List[List[float]]] = None
    image_embeddings: Optional[List[List[float]]] = None

class ErrorEntry(BaseModel):
    """Error entry for failed processing."""
    error: str
    index: int

@app.on_event("startup")
async def startup_event():
    """Load CLIP model on startup with retry logic and fallback options."""
    global model, preprocess, tokenizer
    
    # Model configurations to try in order of preference
    model_configs = [
        {'model': 'ViT-B-32', 'pretrained': 'laion400m_e32', 'name': 'ViT-B/32 (laion400m)'},
        {'model': 'ViT-B-32', 'pretrained': 'openai', 'name': 'ViT-B/32 (OpenAI)'},
        {'model': 'ViT-B-32', 'pretrained': None, 'name': 'ViT-B/32 (no pretrained weights)'},
        {'model': 'ViT-L-14', 'pretrained': 'openai', 'name': 'ViT-L/14 (OpenAI)'},
    ]
    
    for config in model_configs:
        try:
            logger.info(f"Attempting to load CLIP model: {config['name']}...")
            model, _, preprocess = await load_model_with_retry(
                config['model'], 
                config['pretrained'],
                max_retries=3,
                base_delay=2.0
            )
            tokenizer = open_clip.get_tokenizer(config['model'])
            
            # Set model to evaluation mode
            model.eval()
            
            # Move to CPU (default)
            device = torch.device('cpu')
            model = model.to(device)
            
            logger.info(f"CLIP model loaded successfully: {config['name']} on {device}")
            return
            
        except Exception as e:
            logger.warning(f"Failed to load {config['name']}: {e}")
            continue
    
    # If all models failed to load
    raise RuntimeError("Failed to load any CLIP model configuration. Please check your internet connection and try again.")


async def load_model_with_retry(model_name: str, pretrained: Optional[str], max_retries: int = 3, base_delay: float = 2.0):
    """Load model with exponential backoff retry logic."""
    
    for attempt in range(max_retries):
        try:
            # Add a small random delay to avoid thundering herd
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + (torch.rand(1).item() * 0.5)
                logger.info(f"Retrying model load in {delay:.1f} seconds (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            
            # Try to load the model
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained
            )
            
            return model, _, preprocess
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed, re-raise the exception
                raise e
            
            # Check if it's a rate limiting error
            error_str = str(e).lower()
            if "429" in error_str or "too many requests" in error_str:
                logger.warning(f"Rate limited while loading model (attempt {attempt + 1}/{max_retries}): {e}")
            else:
                logger.warning(f"Error loading model (attempt {attempt + 1}/{max_retries}): {e}")
    
    raise RuntimeError(f"Failed to load model after {max_retries} attempts")

def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Normalize embeddings by L2 norm."""
    return embeddings / embeddings.norm(dim=-1, keepdim=True)

def download_image(url: str, timeout: int = 120) -> Optional[Image.Image]:
    """Download and open image from URL."""
    try:
        # Add headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Make request with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    str(url), 
                    timeout=timeout, 
                    headers=headers,
                    stream=True,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Check if content type is an image
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                    logger.warning(f"URL does not point to an image: {url} (content-type: {content_type})")
                    return None
                
                # Open image from bytes
                image = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                logger.info(f"Successfully downloaded image from {url} (size: {image.size})")
                return image
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying in {wait_time}s: {e}")
                    import time
                    time.sleep(wait_time)
                else:
                    raise e
        
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        return None

def process_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    try:
        with torch.no_grad():
            # Tokenize texts
            text_tokens = tokenizer(texts)
            
            # Generate embeddings
            text_features = model.encode_text(text_tokens)
            
            # Normalize embeddings
            text_features = normalize_embeddings(text_features)
            
            # Convert to list of lists
            embeddings = text_features.cpu().numpy().tolist()
            
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to process text embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Text embedding processing failed: {e}")

def process_image_embeddings(urls: List[str]) -> tuple[List[List[float]], List[ErrorEntry]]:
    """Generate embeddings for a list of image URLs."""
    embeddings = []
    errors = []
    
    try:
        with torch.no_grad():
            for i, url in enumerate(urls):
                try:
                    # Download and preprocess image
                    image = download_image(url)
                    if image is None:
                        errors.append(ErrorEntry(error=f"Failed to download image from {url}", index=i))
                        embeddings.append([])  # Placeholder for failed image
                        continue
                    
                    # Preprocess image
                    image_tensor = preprocess(image).unsqueeze(0)
                    
                    # Generate embedding
                    image_features = model.encode_image(image_tensor)
                    
                    # Normalize embedding
                    image_features = normalize_embeddings(image_features)
                    
                    # Convert to list
                    embedding = image_features.cpu().numpy().squeeze().tolist()
                    embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Failed to process image {i} from {url}: {e}")
                    errors.append(ErrorEntry(error=str(e), index=i))
                    embeddings.append([])  # Placeholder for failed image
        
        # Filter out empty embeddings (failed images)
        valid_embeddings = [emb for emb in embeddings if emb]
        
        return valid_embeddings, errors
        
    except Exception as e:
        logger.error(f"Failed to process image embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Image embedding processing failed: {e}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "CLIP Embedding Service is running",
        "model": "ViT-B/32 (laion400m)",
        "endpoints": ["/embed/text", "/embed/image", "/embed/mixed"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(request: TextEmbeddingRequest):
    """Generate embeddings for text inputs."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Processing {len(request.texts)} texts for embedding")
    
    embeddings = process_text_embeddings(request.texts)
    
    return EmbeddingResponse(embeddings=embeddings)

@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(request: ImageEmbeddingRequest):
    """Generate embeddings for image inputs."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Processing {len(request.urls)} images for embedding")
    
    embeddings, errors = process_image_embeddings([str(url) for url in request.urls])
    
    if errors:
        logger.warning(f"Encountered {len(errors)} errors while processing images")
        # For now, we'll return successful embeddings and log errors
        # In production, you might want to include error information in the response
    
    return EmbeddingResponse(embeddings=embeddings)

@app.post("/embed/mixed", response_model=MixedEmbeddingResponse)
async def embed_mixed(request: MixedEmbeddingRequest):
    """Generate embeddings for both text and image inputs."""
    if not request.texts and not request.urls:
        raise HTTPException(status_code=400, detail="No texts or image URLs provided")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    text_embeddings = None
    image_embeddings = None
    
    if request.texts:
        logger.info(f"Processing {len(request.texts)} texts for embedding")
        text_embeddings = process_text_embeddings(request.texts)
    
    if request.urls:
        logger.info(f"Processing {len(request.urls)} images for embedding")
        image_embeddings, errors = process_image_embeddings([str(url) for url in request.urls])
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors while processing images")
    
    return MixedEmbeddingResponse(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    if model is None:
        raise HTTPException(status_code=503, detail="Service not ready - model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "preprocess_loaded": preprocess is not None
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "CLIP Embedding Service",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "text_embeddings": "/embed/text",
            "image_embeddings": "/embed/image", 
            "mixed_embeddings": "/embed/mixed",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info",
        access_log=True
    )
