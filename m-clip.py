"""
FastAPI microservice for generating embeddings from images and text using Multilingual CLIP model.
"""

import io
import logging
from typing import List, Optional, Union
import requests
import torch
import open_clip
import transformers
from multilingual_clip import pt_multilingual_clip
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual CLIP Embedding Service",
    description="A microservice for generating embeddings from images and text using Multilingual CLIP model",
    version="1.0.0"
)

# Global variables for models and preprocessing
text_model = None
text_tokenizer = None
image_model = None
image_preprocess = None

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
    """Load Multilingual CLIP models on startup."""
    global text_model, text_tokenizer, image_model, image_preprocess
    
    try:
        # Load multilingual text model
        logger.info("Loading Multilingual CLIP text model (XLM-Roberta-Large-Vit-B-16Plus)...")
        text_model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
        text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(text_model_name)
        text_tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)
        
        # Set text model to evaluation mode
        text_model.eval()
        
        logger.info("Multilingual text model loaded successfully")
        
        # Load corresponding image model
        logger.info("Loading CLIP image model (ViT-B-16-plus-240)...")
        image_model, _, image_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16-plus-240', 
            pretrained='laion400m_e32'
        )
        
        # Set image model to evaluation mode
        image_model.eval()
        
        # Move models to CPU (default)
        device = torch.device('cpu')
        text_model = text_model.to(device)
        image_model = image_model.to(device)
        
        logger.info(f"All models loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")

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
    """Generate embeddings for a list of texts using multilingual model."""
    try:
        with torch.no_grad():
            # Generate embeddings using multilingual model
            text_features = text_model.forward(texts, text_tokenizer)
            
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
                    image_tensor = image_preprocess(image).unsqueeze(0)
                    
                    # Generate embedding using image model
                    image_features = image_model.encode_image(image_tensor)
                    
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
        "message": "Multilingual CLIP Embedding Service is running",
        "text_model": "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus",
        "image_model": "ViT-B-16-plus-240 (laion400m)",
        "endpoints": ["/embed/text", "/embed/image", "/embed/mixed"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if text_model is None or image_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "healthy", 
        "text_model_loaded": text_model is not None,
        "image_model_loaded": image_model is not None
    }

@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(request: TextEmbeddingRequest):
    """Generate embeddings for text inputs."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if text_model is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")
    
    logger.info(f"Processing {len(request.texts)} texts for embedding")
    
    embeddings = process_text_embeddings(request.texts)
    
    return EmbeddingResponse(embeddings=embeddings)

@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(request: ImageEmbeddingRequest):
    """Generate embeddings for image inputs."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")
    
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")
    
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
    
    if text_model is None or image_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)