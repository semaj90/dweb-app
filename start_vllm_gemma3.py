#!/usr/bin/env python3
"""
vLLM Server for Unsloth/Gemma3 GGUF Model
Optimized for Legal AI Applications
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal AI vLLM Server", 
    description="High-performance GGUF model server for legal applications",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
engine = None
model_name = "gemma3-legal"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemma3-legal"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str = "gemma3-legal"
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

def find_gguf_model():
    """Find the GGUF model file"""
    possible_paths = [
        "./gemma3Q4_K_M/mo16.gguf",
        "./models/gemma3Q4_K_M/mo16.gguf",
        "C:/Users/james/Desktop/deeds-web/deeds-web-app/gemma3Q4_K_M/mo16.gguf"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found GGUF model at: {path}")
            return path
    
    # Search for any .gguf files
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".gguf"):
                full_path = os.path.join(root, file)
                logger.info(f"Found GGUF file: {full_path}")
                return full_path
    
    return None

@app.on_event("startup")
async def startup_event():
    global engine, model_name
    
    try:
        # Try to import vLLM
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.sampling_params import SamplingParams
        
        model_path = find_gguf_model()
        
        if not model_path:
            logger.error("No GGUF model found! Please place your Gemma3 model in ./gemma3Q4_K_M/mo16.gguf")
            # Fall back to a simple HTTP proxy to Ollama
            return
        
        logger.info(f"🚀 Initializing vLLM with model: {model_path}")
        
        # Configure vLLM for GGUF
        engine_args = AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,
            dtype="float16",
            quantization="gguf" if model_path.endswith('.gguf') else None
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("✅ vLLM engine initialized successfully!")
        
    except ImportError:
        logger.warning("vLLM not installed. Install with: pip install vllm")
        logger.info("Falling back to Ollama proxy mode")
        engine = None
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        logger.info("Falling back to Ollama proxy mode")
        engine = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": model_name,
        "engine": "vllm" if engine else "ollama-proxy",
        "timestamp": time.time()
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "legal-ai"
            }
        ]
    }

async def proxy_to_ollama(request_data: dict, endpoint: str):
    """Fallback to Ollama API when vLLM is not available"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:11434/api/{endpoint}",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(status_code=response.status, detail="Ollama API error")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama connection failed: {str(e)}")

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt"""
    prompt_parts = []
    
    for message in messages:
        role = message.role
        content = message.content
        
        if role == "system":
            prompt_parts.append(f"<start_of_turn>system\\n{content}<end_of_turn>")
        elif role == "user":
            prompt_parts.append(f"<start_of_turn>user\\n{content}<end_of_turn>")
        elif role == "assistant":
            prompt_parts.append(f"<start_of_turn>model\\n{content}<end_of_turn>")
    
    prompt_parts.append("<start_of_turn>model\\n")
    return "\\n".join(prompt_parts)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global engine
    
    if engine:
        # Use vLLM
        try:
            from vllm.sampling_params import SamplingParams
            from vllm.utils import random_uuid
            
            prompt = messages_to_prompt(request.messages)
            
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=["<start_of_turn>", "<end_of_turn>"]
            )
            
            request_id = random_uuid()
            
            results = []
            async for output in engine.generate(prompt, sampling_params, request_id):
                results.append(output)
            
            if results:
                generated_text = results[-1].outputs[0].text.strip()
                return {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
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
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(generated_text.split()),
                        "total_tokens": len(prompt.split()) + len(generated_text.split())
                    }
                }
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            # Fall back to Ollama
    
    # Fallback to Ollama
    logger.info("Using Ollama fallback")
    prompt = messages_to_prompt(request.messages)
    
    ollama_request = {
        "model": "llama3.2:1b",  # Use the working model
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens
        }
    }
    
    result = await proxy_to_ollama(ollama_request, "generate")
    
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
                    "content": result.get("response", "")
                },
                "finish_reason": "stop"
            }
        ]
    }

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    global engine
    
    if engine:
        # Use vLLM
        try:
            from vllm.sampling_params import SamplingParams
            from vllm.utils import random_uuid
            
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
            
            request_id = random_uuid()
            
            results = []
            async for output in engine.generate(request.prompt, sampling_params, request_id):
                results.append(output)
            
            if results:
                generated_text = results[-1].outputs[0].text
                return {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "text": generated_text,
                            "index": 0,
                            "finish_reason": "stop"
                        }
                    ]
                }
        except Exception as e:
            logger.error(f"vLLM completion error: {e}")
    
    # Fallback to Ollama
    ollama_request = {
        "model": "llama3.2:1b",
        "prompt": request.prompt,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens
        }
    }
    
    result = await proxy_to_ollama(ollama_request, "generate")
    
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion", 
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": result.get("response", ""),
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Legal AI vLLM Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🤖 Model: Gemma3 GGUF (with Ollama fallback)")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
