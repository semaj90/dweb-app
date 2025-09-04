#!/usr/bin/env python3
"""
GPU availability checker for Windows systems
"""

import json
import sys

def check_gpu_availability():
    """Check for GPU availability and capabilities"""
    gpu_info = {
        "available": False,
        "device": "none",
        "devices": [],
        "cuda_available": False,
        "torch_available": False,
        "memory_gb": 0,
        "compute_capability": None
    }
    
    # Check PyTorch and CUDA
    try:
        import torch
        gpu_info["torch_available"] = True
        
        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["available"] = True
            gpu_info["device"] = "cuda"
            
            # Get device information
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": device_props.name,
                    "memory_gb": round(device_props.total_memory / 1024**3, 2),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "multiprocessor_count": device_props.multi_processor_count
                }
                gpu_info["devices"].append(device_info)
            
            # Use primary device info
            if gpu_info["devices"]:
                primary_device = gpu_info["devices"][0]
                gpu_info["memory_gb"] = primary_device["memory_gb"]
                gpu_info["compute_capability"] = primary_device["compute_capability"]
        
        # Check for Apple Silicon
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = "mps"
            gpu_info["devices"].append({
                "id": 0,
                "name": "Apple Silicon GPU",
                "memory_gb": 8,  # Approximate
                "compute_capability": "Apple M1/M2"
            })
            
    except ImportError:
        pass
    
    # Check for Intel GPU (Windows-specific)
    try:
        import subprocess
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_names = [line.strip() for line in result.stdout.split('\n') 
                        if line.strip() and 'Name' not in line]
            
            for name in gpu_names:
                if 'intel' in name.lower() and 'uhd' in name.lower():
                    gpu_info["devices"].append({
                        "id": len(gpu_info["devices"]),
                        "name": name,
                        "type": "integrated",
                        "memory_gb": 2  # Shared memory
                    })
                elif any(brand in name.lower() for brand in ['nvidia', 'amd', 'radeon']):
                    if not gpu_info["available"]:
                        gpu_info["available"] = True
                        gpu_info["device"] = "discrete"
                    gpu_info["devices"].append({
                        "id": len(gpu_info["devices"]),
                        "name": name,
                        "type": "discrete",
                        "memory_gb": 4  # Estimated
                    })
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Check DirectML (Windows ML)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            gpu_info["directml_available"] = True
            if not gpu_info["available"]:
                gpu_info["available"] = True
                gpu_info["device"] = "directml"
    except ImportError:
        gpu_info["directml_available"] = False
    
    # Performance estimation
    if gpu_info["available"]:
        if gpu_info["device"] == "cuda" and gpu_info["memory_gb"] >= 6:
            gpu_info["performance_tier"] = "high"
        elif gpu_info["device"] in ["cuda", "mps", "directml"]:
            gpu_info["performance_tier"] = "medium"
        else:
            gpu_info["performance_tier"] = "low"
    else:
        gpu_info["performance_tier"] = "cpu_only"
    
    return gpu_info

def main():
    try:
        gpu_info = check_gpu_availability()
        print(json.dumps(gpu_info, indent=2))
    except Exception as e:
        error_info = {
            "available": False,
            "device": "none",
            "error": str(e),
            "torch_available": False,
            "cuda_available": False
        }
        print(json.dumps(error_info, indent=2))

if __name__ == "__main__":
    main()
