#!/bin/bash
# vLLM Setup for Unsloth-trained GGUF Models

echo "🚀 Setting up vLLM for Unsloth GGUF models..."

# Create models directory structure
mkdir -p ./models
mkdir -p ./models/unsloth
mkdir -p ./models/gguf

# Download and install vLLM with GGUF support
echo "📦 Installing vLLM with GGUF support..."
pip install vllm[gguf]

# Create model configuration
cat > ./models/model_config.json << 'EOF'
{
  "model_type": "gguf",
  "model_path": "/models/unsloth-model.gguf",
  "trust_remote_code": true,
  "max_model_len": 4096,
  "tensor_parallel_size": 1,
  "quantization": "gguf",
  "dtype": "float16",
  "gpu_memory_utilization": 0.8,
  "max_num_seqs": 256,
  "max_num_batched_tokens": 4096
}
EOF

# Create vLLM server startup script
cat > ./start_vllm.py << 'EOF'
import asyncio
import argparse
from vllm import AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os

app = FastAPI(title="Legal AI vLLM Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
engine = None

class ChatCompletionRequest(BaseModel):
    model: str = "legal-ai"
    messages: List[dict]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

@app.on_event("startup")
async def startup_event():
    global engine
    
    # Find GGUF model file
    model_path = None
    for root, dirs, files in os.walk("./models"):
        for file in files:
            if file.endswith((".gguf", ".bin")):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break
    
    if not model_path:
        print("⚠️ No GGUF model found. Place your Unsloth model in ./models/")
        model_path = "microsoft/DialoGPT-medium"  # Fallback
    
    print(f"🚀 Loading model: {model_path}")
    
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        dtype="float16"
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✅ vLLM engine initialized successfully")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "vllm-gguf", "engine": "ready" if engine else "loading"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "legal-ai",
                "object": "model",
                "created": 1234567890,
                "owned_by": "legal-ai"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert messages to prompt
    prompt = ""
    for message in request.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"Human: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    
    prompt += "Assistant: "
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    
    request_id = random_uuid()
    
    try:
        results = []
        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)
        
        if results:
            generated_text = results[-1].outputs[0].text
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": 1234567890,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    
    request_id = random_uuid()
    
    try:
        results = []
        async for output in engine.generate(request.prompt, sampling_params, request_id):
            results.append(output)
        
        if results:
            generated_text = results[-1].outputs[0].text
            return {
                "id": request_id,
                "object": "text_completion",
                "created": 1234567890,
                "model": "legal-ai",
                "choices": [
                    {
                        "text": generated_text,
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Legal AI vLLM Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

echo "✅ vLLM setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Place your Unsloth GGUF model in ./models/ directory"
echo "2. Run: python start_vllm.py"
echo "3. Test: curl http://localhost:8000/health"
echo ""
echo "🔧 Model formats supported:"
echo "- .gguf (Unsloth/llama.cpp format)"
echo "- .bin (safetensors format)"
echo ""
echo "🌐 API endpoints:"
echo "- Health: GET /health"
echo "- Chat: POST /v1/chat/completions"
echo "- Completion: POST /v1/completions"
