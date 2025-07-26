#!/usr/bin/env python3
"""
Test script for the direct Gemma3 vLLM server
Tests both chat completions and regular completions endpoints
"""

import requests
import json
import time
import sys

# Server configuration
BASE_URL = "http://localhost:8001"
TIMEOUT = 60

def test_health():
    """Test server health"""
    print("🏥 Testing server health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Backend: {data.get('backend')}")
            print(f"   Model path: {data.get('model_path')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models_endpoint():
    """Test models listing endpoint"""
    print("\n📚 Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint working")
            print(f"   Available models: {[model['id'] for model in data['data']]}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_chat_completion():
    """Test chat completions endpoint"""
    print("\n💬 Testing chat completions...")

    messages = [
        {
            "role": "system",
            "content": "You are a legal AI assistant specializing in contract analysis and legal document review."
        },
        {
            "role": "user",
            "content": "What are the key elements that should be included in a software licensing agreement?"
        }
    ]

    payload = {
        "model": "gemma3-legal",
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 512
    }

    try:
        print("🔄 Sending chat completion request...")
        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )

        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']

            print(f"✅ Chat completion successful!")
            print(f"   Response time: {response_time:.2f} seconds")
            print(f"   Model: {data.get('model')}")
            print(f"   Tokens used: {data.get('usage', {}).get('total_tokens', 'N/A')}")
            print(f"\n📝 Response preview:")
            print(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ Chat completion timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False

def test_completion():
    """Test regular completions endpoint"""
    print("\n📝 Testing completions...")

    payload = {
        "model": "gemma3-legal",
        "prompt": "In legal terms, a contract is defined as",
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 256
    }

    try:
        print("🔄 Sending completion request...")
        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/v1/completions",
            json=payload,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )

        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['text']

            print(f"✅ Completion successful!")
            print(f"   Response time: {response_time:.2f} seconds")
            print(f"   Model: {data.get('model')}")
            print(f"   Tokens used: {data.get('usage', {}).get('total_tokens', 'N/A')}")
            print(f"\n📝 Response preview:")
            print(f"   {payload['prompt']}{content[:150]}{'...' if len(content) > 150 else ''}")
            return True
        else:
            print(f"❌ Completion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ Completion timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"❌ Completion error: {e}")
        return False

def test_legal_specific():
    """Test legal-specific functionality"""
    print("\n⚖️  Testing legal-specific queries...")

    legal_queries = [
        "What is the statute of limitations for breach of contract in California?",
        "Explain the difference between trademark and copyright protection.",
        "What are the key components of a valid contract?"
    ]

    success_count = 0

    for i, query in enumerate(legal_queries, 1):
        print(f"\n🔍 Query {i}: {query[:50]}...")

        messages = [
            {
                "role": "system",
                "content": "You are an expert legal AI assistant. Provide accurate, professional legal information."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        payload = {
            "model": "gemma3-legal",
            "messages": messages,
            "temperature": 0.05,  # Lower temperature for legal accuracy
            "max_tokens": 300
        }

        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=TIMEOUT,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print(f"✅ Legal query {i} successful")
                print(f"   Preview: {content[:100]}...")
                success_count += 1
            else:
                print(f"❌ Legal query {i} failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Legal query {i} error: {e}")

    print(f"\n📊 Legal queries summary: {success_count}/{len(legal_queries)} successful")
    return success_count == len(legal_queries)

def main():
    """Run all tests"""
    print("🧪 Starting Gemma3 vLLM Server Tests")
    print("=" * 50)

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding at {BASE_URL}")
            print("Make sure to start the server first with:")
            print("python direct-gemma3-vllm-server.py")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print(f"Error: {e}")
        print("Make sure to start the server first with:")
        print("python direct-gemma3-vllm-server.py")
        sys.exit(1)

    print(f"🌐 Connected to server at {BASE_URL}")

    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Models Endpoint", test_models_endpoint),
        ("Chat Completions", test_chat_completion),
        ("Regular Completions", test_completion),
        ("Legal Specific Tests", test_legal_specific)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Gemma3 vLLM server is working correctly.")
        print("\n🚀 You can now use the server with these endpoints:")
        print(f"   • Chat: POST {BASE_URL}/v1/chat/completions")
        print(f"   • Completions: POST {BASE_URL}/v1/completions")
        print(f"   • Models: GET {BASE_URL}/v1/models")
        print(f"   • Health: GET {BASE_URL}/health")
    else:
        print("⚠️  Some tests failed. Check the server logs and configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
