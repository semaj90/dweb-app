#!/usr/bin/env python3
"""
Direct Gemma3 GGUF Model Server using vLLM
Provides OpenAI-compatible API endpoints for the custom Gemma3 model
"""

import os
import sys
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import json
import time
from contextlib import asynccontextmanager

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    print("⚠️  vLLM not available, falling back to llama-cpp-python")
    VLLM_AVAILABLE = False

# Fallback to llama-cpp-python
if not VLLM_AVAILABLE:
    try:
        from llama_cpp import Llama
        LLAMA_CPP_AVAILABLE = True
    except ImportError:
        print("❌ Neither vLLM nor llama-cpp-python available!")
        print("Install with: pip install vllm llama-cpp-python")
        sys.exit(1)

# Configuration
MODEL_PATH = r"C:\Users\james\Desktop\deeds-web\deeds-web-app\gemma3Q4_K_M\mohf16-Q4_K_M.gguf"
HOST = "0.0.0.0"
PORT = 8001
MAX_TOKENS = 2048
CONTEXT_LENGTH = 8192

# Global model instance
model_engine = None

# Request/Response Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemma3-legal"
    messages: List[ChatMessage]
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

class CompletionRequest(BaseModel):
    model: str = "gemma3-legal"
    prompt: str
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    stream: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup model"""
    global model_engine

    print("🚀 Starting Gemma3 Legal AI Server...")
    print(f"📁 Model path: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        sys.exit(1)

    try:
        if VLLM_AVAILABLE:
            print("🔧 Loading model with vLLM...")
            # vLLM setup for GGUF files
            model_engine = LLM(
                model=MODEL_PATH,
                tokenizer_mode="auto",
                trust_remote_code=True,
                max_model_len=CONTEXT_LENGTH,
                gpu_memory_utilization=0.8,
                enforce_eager=True
            )
        else:
            print("🔧 Loading model with llama-cpp-python...")
            # Fallback to llama-cpp-python
            model_engine = Llama(
                model_path=MODEL_PATH,
                n_ctx=CONTEXT_LENGTH,
                n_threads=8,
                n_gpu_layers=-1,  # Use GPU if available
                verbose=False
            )

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    yield

    # Cleanup
    print("🔄 Shutting down model...")
    model_engine = None

# Initialize FastAPI app
app = FastAPI(
    title="Gemma3 Legal AI Server",
    description="Direct GGUF model server with OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_gemma3_prompt(messages: List[ChatMessage]) -> str:
    """Format messages for Gemma3 chat template"""
    formatted = ""

    for message in messages:
        if message.role == "system":
            formatted += f"<start_of_turn>user\nSystem: {message.content}<end_of_turn>\n"
        elif message.role == "user":
            formatted += f"<start_of_turn>user\n{message.content}<end_of_turn>\n"
        elif message.role == "assistant":
            formatted += f"<start_of_turn>model\n{message.content}<end_of_turn>\n"

    formatted += "<start_of_turn>model\n"
    return formatted

async def generate_completion(prompt: str, params: dict) -> str:
    """Generate completion using the loaded model"""
    global model_engine

    if model_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if VLLM_AVAILABLE:
            # vLLM generation
            sampling_params = SamplingParams(
                temperature=params.get('temperature', 0.1),
                top_p=params.get('top_p', 0.9),
                max_tokens=params.get('max_tokens', 1024),
                stop=["<start_of_turn>", "<end_of_turn>"]
            )

            outputs = model_engine.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()

        else:
            # llama-cpp-python generation
            response = model_engine(
                prompt,
                max_tokens=params.get('max_tokens', 1024),
                temperature=params.get('temperature', 0.1),
                top_p=params.get('top_p', 0.9),
                stop=["<start_of_turn>", "<end_of_turn>"],
                echo=False
            )
            return response['choices'][0]['text'].strip()

    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "gemma3-legal",
        "version": "1.0.0",
        "backend": "vLLM" if VLLM_AVAILABLE else "llama-cpp-python"
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma3-legal",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""

    # Format prompt for Gemma3
    formatted_prompt = format_gemma3_prompt(request.messages)

    # Add legal system context if not present
    if not any(msg.role == "system" for msg in request.messages):
        system_context = "You are a specialized Legal AI Assistant. You provide expert legal analysis, contract review, case law research, and legal document assistance. Always maintain professional accuracy."
        formatted_prompt = f"<start_of_turn>user\nSystem: {system_context}<end_of_turn>\n" + formatted_prompt

    # Generation parameters
    params = {
        'temperature': request.temperature,
        'top_p': request.top_p,
        'max_tokens': request.max_tokens
    }

    # Generate response
    response_text = await generate_completion(formatted_prompt, params)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(formatted_prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(formatted_prompt.split()) + len(response_text.split())
        }
    }

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint"""

    params = {
        'temperature': request.temperature,
        'top_p': request.top_p,
        'max_tokens': request.max_tokens
    }

    response_text = await generate_completion(request.prompt, params)

    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(request.prompt.split()) + len(response_text.split())
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model_engine is not None,
        "model_path": MODEL_PATH,
        "backend": "vLLM" if VLLM_AVAILABLE else "llama-cpp-python",
        "context_length": CONTEXT_LENGTH,
        "max_tokens": MAX_TOKENS
    }

if __name__ == "__main__":
    print("🚀 Starting Gemma3 Legal AI Server...")
    print(f"🌐 Server will be available at: http://{HOST}:{PORT}")
    print(f"📚 API docs: http://{HOST}:{PORT}/docs")
    print(f"🏥 Health check: http://{HOST}:{PORT}/health")

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the GGUF model file is present at the specified path.")
        sys.exit(1)

    uvicorn.run(
        "direct-gemma3-vllm-server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )
