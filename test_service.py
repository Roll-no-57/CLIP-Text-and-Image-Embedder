"""
Test script for the CLIP embedding service.
"""

import requests
import json
import time

# Service URL
BASE_URL = "http://localhost:8080"

def wait_for_service(max_retries=30):
    """Wait for the service to be ready."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except requests.exceptions.ConnectionError:
            print(f"⏳ Waiting for service... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print("❌ Service failed to start")
    return False

def test_text_embeddings():
    """Test text embedding endpoint."""
    print("\n🧪 Testing text embeddings...")
    
    payload = {
        "texts": [
            "Hello world",
            "A photo of a cat",
            "Machine learning and artificial intelligence"
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/embed/text", json=payload)
        response.raise_for_status()
        
        result = response.json()
        embeddings = result["embeddings"]
        
        print(f"✅ Text embeddings: {len(embeddings)} embeddings generated")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Sample embedding (first 5 values): {embeddings[0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text embedding test failed: {e}")
        return False

def test_image_embeddings():
    """Test image embedding endpoint."""
    print("\n🧪 Testing image embeddings...")
    
    # Using publicly available test images
    payload = {
        "urls": [
            "https://raw.githubusercontent.com/pytorch/examples/main/cpp/dcgan/assets/real_samples.png"
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/embed/image", json=payload)
        response.raise_for_status()
        
        result = response.json()
        embeddings = result["embeddings"]
        
        print(f"✅ Image embeddings: {len(embeddings)} embeddings generated")
        if embeddings:
            print(f"   Embedding dimension: {len(embeddings[0])}")
            print(f"   Sample embedding (first 5 values): {embeddings[0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image embedding test failed: {e}")
        return False

def test_mixed_embeddings():
    """Test mixed embedding endpoint."""
    print("\n🧪 Testing mixed embeddings...")
    
    payload = {
        "texts": ["A beautiful landscape", "Computer vision"],
        "urls": ["https://raw.githubusercontent.com/pytorch/examples/main/cpp/dcgan/assets/real_samples.png"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/embed/mixed", json=payload)
        response.raise_for_status()
        
        result = response.json()
        text_embeddings = result.get("text_embeddings", [])
        image_embeddings = result.get("image_embeddings", [])
        
        print(f"✅ Mixed embeddings:")
        print(f"   Text embeddings: {len(text_embeddings)} generated")
        print(f"   Image embeddings: {len(image_embeddings)} generated")
        
        if text_embeddings:
            print(f"   Text embedding dimension: {len(text_embeddings[0])}")
        if image_embeddings:
            print(f"   Image embedding dimension: {len(image_embeddings[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed embedding test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n🧪 Testing error handling...")
    
    try:
        # Test with invalid image URL
        payload = {"urls": ["https://invalid-url-that-does-not-exist.com/image.jpg"]}
        response = requests.post(f"{BASE_URL}/embed/image", json=payload)
        
        # Should still return 200 but with empty embeddings or handle gracefully
        if response.status_code in [200, 400]:
            print("✅ Error handling for invalid URLs works correctly")
        else:
            print(f"⚠️ Unexpected status code for invalid URL: {response.status_code}")
        
        # Test with empty request
        response = requests.post(f"{BASE_URL}/embed/text", json={"texts": []})
        if response.status_code == 400:
            print("✅ Error handling for empty input works correctly")
        else:
            print(f"⚠️ Unexpected status code for empty input: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting CLIP Embedding Service Tests")
    print("=" * 50)
    
    # Wait for service to be ready
    if not wait_for_service():
        return
    
    # Run tests
    tests = [
        test_text_embeddings,
        test_image_embeddings,
        test_mixed_embeddings,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {total - passed} tests failed")

if __name__ == "__main__":
    main()
