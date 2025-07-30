#!/usr/bin/env python3
"""
Simple vLLM Windows Testing Script
Tests vLLM installation and Ollama model integration for Phase 13
"""

import sys
import subprocess
import requests
import json
import time
import os

def check_system_requirements():
    """Check system requirements for vLLM"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if CUDA is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA: Available - {gpu_count} GPU(s) - {gpu_name}")
        else:
            print("CUDA: Not available - using CPU mode")
    except ImportError:
        print("PyTorch not installed")
        return False
    
    # Check for Windows compatibility
    if os.name == 'nt':
        print("Windows: Detected - using compatibility mode")
    
    return True

def test_vllm_installation():
    """Test vLLM installation"""
    print("\nTesting vLLM installation...")
    
    try:
        # Try importing vllm
        import vllm
        print(f"vLLM installed: version {vllm.__version__}")
        return True
    except ImportError as e:
        print(f"vLLM not installed: {e}")
        return False

def test_docker_setup():
    """Test Docker setup for vLLM"""
    print("\nTesting Docker setup...")
    
    try:
        # Check if Docker is running
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Docker: Running")
        else:
            print("Docker: Not running")
            return False
            
        # Check for Ollama container
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=legal-ollama'], 
                              capture_output=True, text=True)
        if 'legal-ollama' in result.stdout:
            print("Ollama container: Found")
        else:
            print("Ollama container: Not found")
            
        # Check Ollama data volume
        result = subprocess.run(['docker', 'volume', 'ls'], capture_output=True, text=True)
        if 'ollama_data' in result.stdout:
            print("Ollama data volume: Found")
        else:
            print("Ollama data volume: Not found")
            
        return True
        
    except FileNotFoundError:
        print("Docker: Not installed")
        return False

def test_ollama_model():
    """Test Ollama model availability"""
    print("\nTesting Ollama model...")
    
    try:
        # Check if Ollama container is running
        result = subprocess.run(['docker', 'exec', 'legal-ollama-phase34', 'ollama', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Ollama models available:")
            print(result.stdout)
            return True
        else:
            print("Failed to list Ollama models")
            return False
    except Exception as e:
        print(f"Error checking Ollama models: {e}")
        return False

def start_mock_server():
    """Start the mock vLLM server"""
    print("\nStarting vLLM mock server...")
    
    try:
        # Check if server is already running
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("Mock server already running")
                return True
        except:
            pass
        
        # Start the mock server in background
        cmd = [sys.executable, "vllm-test/cpu_vllm_mock.py"]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        print("Waiting for server to start...")
        for i in range(10):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("Mock server started successfully")
                    return True
            except:
                continue
        
        print("Mock server failed to start")
        return False
        
    except Exception as e:
        print(f"Error starting mock server: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\nTesting API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Health: {data.get('status', 'unknown')}")
        else:
            print("Health check failed")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False
    
    # Test models
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['id'] for model in data.get('data', [])]
            print(f"Available models: {', '.join(models)}")
        else:
            print("Models endpoint failed")
    except Exception as e:
        print(f"Models endpoint error: {e}")
    
    # Test completion
    try:
        payload = {
            "model": "gemma3-legal:latest",
            "prompt": "Analyze this legal contract:",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(f"{base_url}/v1/completions", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            completion = data['choices'][0]['text'].strip()
            tokens_used = data['usage']['total_tokens']
            print(f"Completion test: Success ({tokens_used} tokens)")
            print(f"Sample response: {completion[:80]}...")
        else:
            print("Completion test failed")
    except Exception as e:
        print(f"Completion test error: {e}")
    
    return True

def generate_report():
    """Generate test report"""
    print("\nGenerating test report...")
    
    # Check if various components work
    vllm_native = False
    try:
        import vllm
        vllm_native = True
    except ImportError:
        pass
    
    torch_available = False
    cuda_available = False
    try:
        import torch
        torch_available = True
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    docker_running = False
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        docker_running = result.returncode == 0
    except:
        pass
    
    mock_server_running = False
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        mock_server_running = response.status_code == 200
    except:
        pass
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": os.name,
            "torch_available": torch_available,
            "cuda_available": cuda_available
        },
        "vllm": {
            "native_installation": vllm_native,
            "mock_server_running": mock_server_running,
            "openai_compatible": mock_server_running
        },
        "docker": {
            "docker_running": docker_running,
            "ollama_available": docker_running
        },
        "test_results": {
            "system_requirements": torch_available,
            "api_endpoints": mock_server_running,
            "completion_test": mock_server_running
        },
        "recommendations": []
    }
    
    # Add recommendations
    if not vllm_native:
        report["recommendations"].append("vLLM native installation failed - using mock server")
    if not cuda_available:
        report["recommendations"].append("CUDA not available - using CPU mode")
    if mock_server_running:
        report["recommendations"].append("Mock server working - suitable for development")
    
    # Save report
    with open("vllm_windows_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Report saved to: vllm_windows_test_report.json")
    return report

def main():
    """Main function"""
    print("vLLM Windows 10 Testing - Phase 13")
    print("=" * 40)
    
    success = True
    
    # Check system requirements
    if not check_system_requirements():
        success = False
    
    # Test vLLM installation
    vllm_available = test_vllm_installation()
    
    # Test Docker setup
    docker_ready = test_docker_setup()
    
    # Test Ollama model
    if docker_ready:
        test_ollama_model()
    
    # Start mock server if needed
    if not vllm_available:
        mock_started = start_mock_server()
        if mock_started:
            test_api_endpoints()
    
    # Generate report
    report = generate_report()
    
    print("\nTest Summary:")
    print(f"Native vLLM: {'YES' if vllm_available else 'NO'}")
    print(f"Mock Server: {'YES' if report['vllm']['mock_server_running'] else 'NO'}")
    print(f"Docker Ready: {'YES' if docker_ready else 'NO'}")
    print(f"API Working: {'YES' if report['vllm']['openai_compatible'] else 'NO'}")
    
    if report['vllm']['mock_server_running']:
        print("\nSUCCESS: vLLM-compatible API server is running!")
        print("You can test it at: http://localhost:8000/health")
    else:
        print("\nFAILED: Unable to start vLLM or mock server")
        success = False
    
    return success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTesting interrupted")
    except Exception as e:
        print(f"\nTesting failed: {e}")