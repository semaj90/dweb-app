#!/usr/bin/env python3
"""
vLLM CPU Mock Server for Windows Testing
Simulates vLLM API using Ollama models when GPU vLLM installation fails
"""

import os
import json
import time
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="vLLM CPU Mock Server", version="0.6.3.post1+mock")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemma3-legal:latest"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str = "gemma3-legal:latest"
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: Optional[Dict] = None
    text: Optional[str] = None
    finish_reason: str = "stop"

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# Mock legal AI responses
LEGAL_MOCK_RESPONSES = [
    "Based on the contract analysis, the liability clause in Section 4.2 establishes clear boundaries for potential damages. The contractual framework suggests that the responsible party must demonstrate reasonable care standards.",
    "The legal precedent established in this case indicates that contractual obligations require adherence to industry standards. Evidence suggests potential breach of fiduciary duty under applicable state regulations.",
    "Document review reveals critical compliance gaps that may result in regulatory penalties. The legal framework requires immediate remediation to ensure adherence to established statutes and case law precedents.",
    "Analysis of the evidence indicates potential violations of statutory requirements. The contractual provisions establish clear performance standards that must be evaluated within the context of applicable legal frameworks.",
    "The case law precedent suggests that liability assessment requires comprehensive evaluation of all contractual provisions. Evidence presented supports the conclusion that reasonable care standards were not maintained.",
]

def generate_mock_response(prompt: str, max_tokens: int = 512) -> str:
    """Generate a mock legal AI response"""
    import random
    
    # Select response based on prompt content
    if "contract" in prompt.lower() or "liability" in prompt.lower():
        response = LEGAL_MOCK_RESPONSES[0]
    elif "evidence" in prompt.lower() or "precedent" in prompt.lower():
        response = LEGAL_MOCK_RESPONSES[1]
    elif "compliance" in prompt.lower() or "regulation" in prompt.lower():
        response = LEGAL_MOCK_RESPONSES[2]
    elif "statute" in prompt.lower() or "legal" in prompt.lower():
        response = LEGAL_MOCK_RESPONSES[3]
    else:
        response = random.choice(LEGAL_MOCK_RESPONSES)
    
    # Truncate to max_tokens (rough approximation)
    words = response.split()
    if len(words) > max_tokens // 4:  # Rough token estimation
        response = " ".join(words[:max_tokens // 4])
    
    return response

def calculate_tokens(text: str) -> int:
    """Rough token calculation"""
    return max(1, len(text.split()) * 1.3)  # Approximate tokenization

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.6.3.post1+mock",
        "ready": True,
        "model_loaded": "gemma3-legal:latest",
        "backend": "CPU Mock (Windows Compatible)",
        "ollama_integration": True
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma3-legal:latest",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
                "root": "gemma3-legal:latest",
                "parent": None
            }
        ]
    }

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create text completion"""
    start_time = time.time()
    
    try:
        # Generate mock response
        response_text = generate_mock_response(request.prompt, request.max_tokens or 512)
        
        # Calculate tokens
        prompt_tokens = int(calculate_tokens(request.prompt))
        completion_tokens = int(calculate_tokens(response_text))
        
        # Simulate processing time
        processing_time = len(response_text) * 0.01  # 10ms per character
        await_time = max(0.1, min(2.0, processing_time))  # 100ms to 2s
        time.sleep(await_time)
        
        response = CompletionResponse(
            id=f"cmpl_{uuid.uuid4().hex[:8]}",
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    text=response_text,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion"""
    start_time = time.time()
    
    try:
        # Convert messages to prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"
        
        # Generate mock response
        response_text = generate_mock_response(prompt, request.max_tokens or 512)
        
        # Calculate tokens
        prompt_tokens = int(calculate_tokens(prompt))
        completion_tokens = int(calculate_tokens(response_text))
        
        # Simulate processing time
        processing_time = len(response_text) * 0.01
        await_time = max(0.1, min(2.0, processing_time))
        time.sleep(await_time)
        
        response = CompletionResponse(
            id=f"chatcmpl_{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response_text
                    },
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model info"""
    if model_id not in ["gemma3-legal:latest", "gemma3-legal"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
        "permission": [],
        "root": "gemma3-legal:latest",
        "parent": None
    }

@app.get("/metrics")
async def get_metrics():
    """Get server metrics"""
    return {
        "requests_total": 0,
        "requests_per_second": 0,
        "tokens_per_second": 150,  # Mock value
        "memory_usage_mb": 512,
        "gpu_utilization": 0,  # CPU mode
        "model_loaded": True,
        "uptime_seconds": int(time.time())
    }

if __name__ == "__main__":
    print("🚀 Starting vLLM CPU Mock Server...")
    print("📦 Model: gemma3-legal:latest (Mock)")
    print("🔧 Backend: CPU Compatible (Windows)")
    print("🌐 OpenAI API Compatible")
    print("📍 Health: http://localhost:8000/health")
    print("📍 Models: http://localhost:8000/v1/models")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )