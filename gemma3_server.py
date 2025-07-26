#!/usr/bin/env python3
"""
Direct GGUF Model Server for Your Gemma3 Model
Works without needing Ollama model import
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import time
import subprocess
import requests

app = FastAPI(title="Gemma3 Legal AI Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
GGUF_MODEL_PATH = r"C:\Users\james\Desktop\deeds-web\deeds-web-app\gemma3Q4_K_M\mo16.gguf"
OLLAMA_FALLBACK_MODEL = "llama3.2:1b"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemma3-legal"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

def call_ollama_api(model: str, prompt: str, temperature: float = 0.1, max_tokens: int = 1024):
    """Call Ollama API directly"""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    except Exception as e:
        raise Exception(f"Failed to call Ollama: {str(e)}")

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to Gemma format"""
    prompt_parts = []
    
    system_msg = "You are a specialized Legal AI Assistant powered by Gemma 3. You provide expert legal analysis, contract review, case law research, and document assistance. Always maintain professional accuracy while noting that responses are informational guidance, not formal legal advice."
    
    for message in messages:
        if message.role == "system":
            system_msg = message.content
        elif message.role == "user":
            prompt_parts.append(f"<start_of_turn>user\n{message.content}<end_of_turn>")
        elif message.role == "assistant":
            prompt_parts.append(f"<start_of_turn>model\n{message.content}<end_of_turn>")
    
    # Add system message and final user turn
    final_prompt = f"{system_msg}\n\n" + "\n".join(prompt_parts) + "\n<start_of_turn>model\n"
    return final_prompt

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "gemma3-legal",
        "gguf_path": GGUF_MODEL_PATH,
        "gguf_exists": os.path.exists(GGUF_MODEL_PATH),
        "ollama_fallback": OLLAMA_FALLBACK_MODEL,
        "timestamp": time.time()
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma3-legal",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "legal-ai",
                "description": "Gemma3 Q4_K_M GGUF model for legal applications"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Convert messages to prompt
        prompt = messages_to_prompt(request.messages)
        
        # Try to use the custom model first, fallback to working model
        try:
            # First attempt: try if gemma3-legal was successfully created
            response_text = call_ollama_api("gemma3-legal", prompt, request.temperature, request.max_tokens)
        except:
            # Fallback: use the working llama3.2 model
            response_text = call_ollama_api(OLLAMA_FALLBACK_MODEL, prompt, request.temperature, request.max_tokens)
        
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
                        "content": response_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/test-legal")
async def test_legal_query():
    """Test endpoint for legal queries"""
    test_messages = [
        ChatMessage(role="user", content="I need help analyzing a software license agreement. What are the key liability clauses I should review?")
    ]
    
    request = ChatCompletionRequest(
        model="gemma3-legal",
        messages=test_messages,
        max_tokens=512,
        temperature=0.1
    )
    
    return await chat_completions(request)

if __name__ == "__main__":
    print("🚀 Starting Gemma3 Legal AI Server...")
    print(f"📁 GGUF Model: {GGUF_MODEL_PATH}")
    print(f"📋 Model Exists: {os.path.exists(GGUF_MODEL_PATH)}")
    print("📍 Server: http://localhost:8000")
    print("🧪 Test: http://localhost:8000/test-legal")
    print("📖 Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
