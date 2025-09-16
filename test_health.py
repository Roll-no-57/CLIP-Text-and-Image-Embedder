#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to test the health endpoint and model loading.
"""

import requests
import time
import sys

def test_health_endpoint(url="http://localhost:8080", max_wait=300):
    """Test the health endpoint with retry logic."""
    print(f"Testing health endpoint at {url}/health")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("Success: Health check passed!")
                print(f"  Status: {data.get('status')}")
                print(f"  Model loaded: {data.get('model_loaded')}")
                print(f"  Tokenizer loaded: {data.get('tokenizer_loaded')}")
                print(f"  Preprocess loaded: {data.get('preprocess_loaded')}")
                return True
            elif response.status_code == 503:
                data = response.json()
                print(f"Waiting: Service not ready: {data.get('detail')}")
            else:
                print(f"Error: Health check failed with status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("Waiting: Service is starting...")
        except requests.exceptions.Timeout:
            print("Waiting: Health check timed out, retrying...")
        except Exception as e:
            print(f"Error: Unexpected error: {e}")
        
        time.sleep(5)
    
    print(f"Error: Health check failed after {max_wait} seconds")
    return False

def test_embedding_endpoints(url="http://localhost:8080"):
    """Test the embedding endpoints."""
    print(f"\nTesting embedding endpoints at {url}")
    
    # Test text embedding
    try:
        response = requests.post(
            f"{url}/embed/text",
            json={"texts": ["hello world", "test text"]},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Success: Text embedding: {len(data['embeddings'])} embeddings generated")
        else:
            print(f"Error: Text embedding failed: {response.status_code}")
    except Exception as e:
        print(f"Error: Text embedding error: {e}")
    
    # Test image embedding with a public image
    try:
        response = requests.post(
            f"{url}/embed/image",
            json={"urls": ["https://httpbin.org/image/png"]},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Success: Image embedding: {len(data['embeddings'])} embeddings generated")
        else:
            print(f"Error: Image embedding failed: {response.status_code}")
    except Exception as e:
        print(f"Error: Image embedding error: {e}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    if test_health_endpoint(url):
        test_embedding_endpoints(url)
    else:
        sys.exit(1)