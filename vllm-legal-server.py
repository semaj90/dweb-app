#!/usr/bin/env python3
"""
vLLM High-Performance Legal AI Server
Ultra-efficient serving for Gemma3 Legal model with GPU acceleration
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.environ.get("GEMMA3_MODEL_PATH", "C:/Users/james/Desktop/deeds-web/deeds-web-app/gemma3Q4_K_M/mohf16-Q4_K_M.gguf")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))

# Request/Response Models
class LegalAnalysisRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="Legal analysis prompt")
    case_id: Optional[str] = Field(None, description="Associated case ID")
    evidence_ids: Optional[List[str]] = Field(default=[], description="Related evidence IDs")
    analysis_type: str = Field(default="general", description="Type of legal analysis")
    max_tokens: int = Field(default=2048, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=40, ge=1, le=100, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, ge=0.1, le=2.0, description="Repetition penalty")
    stream: bool = Field(default=False, description="Enable streaming response")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")

class LegalAnalysisResponse(BaseModel):
    request_id: str
    analysis: str
    case_id: Optional[str]
    analysis_type: str
    confidence: float
    processing_time: float
    token_count: int
    model_info: Dict[str, str]

class StreamingChunk(BaseModel):
    request_id: str
    chunk: str
    done: bool = False
    token_count: Optional[int] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, Union[str, float]]
    uptime: float

class ModelInfo(BaseModel):
    model_path: str
    max_tokens: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    quantization: str
    optimization_level: str

# Global engine instance
engine: Optional[AsyncLLMEngine] = None
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the vLLM engine"""
    global engine
    
    try:
        logger.info("Initializing vLLM Legal AI Engine...")
        
        # Configure engine arguments for optimal performance
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            load_format="auto",
            quantization=None,  # Model is already quantized
            seed=42,
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            disable_log_stats=False,
            enable_prefix_caching=True,
            use_v2_block_manager=True,
            swap_space=4,  # GB
            cpu_offload_gb=0,
            enforce_eager=False,
            disable_custom_all_reduce=False
        )
        
        # Create the engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM Legal AI Engine initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {e}")
        raise
    finally:
        if engine:
            logger.info("Shutting down vLLM engine...")
            # Engine cleanup is handled automatically by vLLM

# Create FastAPI app
app = FastAPI(
    title="vLLM Legal AI Server",
    description="High-performance legal AI inference server using vLLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_legal_prompt(request: LegalAnalysisRequest) -> str:
    """Build a comprehensive legal analysis prompt"""
    
    system_prompt = """You are a specialized legal AI assistant trained to help prosecutors with case analysis, evidence evaluation, and legal research. You provide thorough, accurate, and ethically sound legal guidance while maintaining objectivity and professional standards.

Your expertise includes:
- Criminal law and procedure
- Evidence analysis and admissibility
- Case law research and precedent analysis
- Trial strategy and preparation
- Constitutional law considerations
- Ethical prosecutorial standards

Always provide detailed reasoning, cite relevant legal principles, and maintain the highest professional standards."""

    context_info = ""
    if request.case_id:
        context_info += f"\nCase ID: {request.case_id}"
    if request.evidence_ids:
        context_info += f"\nRelated Evidence: {', '.join(request.evidence_ids)}"
    if request.analysis_type:
        context_info += f"\nAnalysis Type: {request.analysis_type}"

    full_prompt = f"""{system_prompt}

{context_info}

User Query: {request.prompt}

Legal Analysis:"""

    return full_prompt

async def calculate_confidence(generated_text: str, processing_time: float) -> float:
    """Calculate confidence score based on response quality and timing"""
    base_confidence = 0.7
    
    # Adjust based on response length and completeness
    if len(generated_text) > 500:
        base_confidence += 0.1
    if len(generated_text) > 1000:
        base_confidence += 0.05
    
    # Check for legal terminology and structure
    legal_indicators = [
        "pursuant to", "evidence", "statute", "precedent", "constitutional",
        "admissible", "jurisdiction", "procedure", "court", "analysis"
    ]
    found_indicators = sum(1 for indicator in legal_indicators if indicator.lower() in generated_text.lower())
    base_confidence += min(found_indicators * 0.02, 0.1)
    
    # Adjust based on processing time (faster is generally better for confidence)
    if processing_time < 10.0:
        base_confidence += 0.05
    elif processing_time > 30.0:
        base_confidence -= 0.05
    
    return min(max(base_confidence, 0.1), 0.95)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        memory_info = {}
        if gpu_available:
            for i in range(gpu_count):
                memory_info[f"gpu_{i}_memory_allocated"] = f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB"
                memory_info[f"gpu_{i}_memory_reserved"] = f"{torch.cuda.memory_reserved(i) / 1024**3:.2f} GB"
        
        return HealthResponse(
            status="healthy" if engine else "initializing",
            model_loaded=engine is not None,
            gpu_available=gpu_available,
            memory_usage=memory_info,
            uptime=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model configuration information"""
    return ModelInfo(
        model_path=MODEL_PATH,
        max_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        quantization="Q4_K_M",
        optimization_level="high_performance"
    )

@app.post("/legal-analysis", response_model=LegalAnalysisResponse)
async def legal_analysis(request: LegalAnalysisRequest):
    """Generate legal analysis using the fine-tuned model"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model engine not initialized")
    
    request_id = random_uuid()
    start_time_req = time.time()
    
    try:
        # Build the prompt
        prompt = build_legal_prompt(request)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop_sequences or ["Human:", "User:", "\n\n---"],
            skip_special_tokens=True,
            spaces_between_special_tokens=False
        )
        
        # Generate response
        results = engine.generate(prompt, sampling_params, request_id)
        final_output = None
        
        async for request_output in results:
            if request_output.finished:
                final_output = request_output.outputs[0].text
                break
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        processing_time = time.time() - start_time_req
        confidence = await calculate_confidence(final_output, processing_time)
        
        return LegalAnalysisResponse(
            request_id=request_id,
            analysis=final_output.strip(),
            case_id=request.case_id,
            analysis_type=request.analysis_type,
            confidence=confidence,
            processing_time=processing_time,
            token_count=len(final_output.split()),
            model_info={
                "model": "gemma3-legal",
                "version": "Q4_K_M",
                "optimization": "vllm_accelerated"
            }
        )
        
    except Exception as e:
        logger.error(f"Legal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/legal-analysis-stream")
async def legal_analysis_stream(request: LegalAnalysisRequest):
    """Stream legal analysis response"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model engine not initialized")
    
    request_id = random_uuid()
    start_time_req = time.time()
    
    async def generate_stream():
        try:
            # Build the prompt
            prompt = build_legal_prompt(request)
            
            # Configure sampling parameters
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop_sequences or ["Human:", "User:", "\n\n---"],
                skip_special_tokens=True
            )
            
            # Generate streaming response
            results = engine.generate(prompt, sampling_params, request_id)
            full_text = ""
            
            async for request_output in results:
                if request_output.outputs:
                    new_text = request_output.outputs[0].text
                    chunk = new_text[len(full_text):]
                    full_text = new_text
                    
                    if chunk:
                        chunk_response = StreamingChunk(
                            request_id=request_id,
                            chunk=chunk,
                            done=False
                        )
                        yield f"data: {chunk_response.model_dump_json()}\n\n"
                
                if request_output.finished:
                    processing_time = time.time() - start_time_req
                    final_chunk = StreamingChunk(
                        request_id=request_id,
                        chunk="",
                        done=True,
                        token_count=len(full_text.split()),
                        processing_time=processing_time
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    break
                    
        except Exception as e:
            error_chunk = StreamingChunk(
                request_id=request_id,
                chunk=f"Error: {str(e)}",
                done=True
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/stream-events",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/metrics")
async def get_metrics():
    """Get server performance metrics"""
    try:
        import torch
        import psutil
        
        metrics = {
            "uptime": time.time() - start_time,
            "model_loaded": engine is not None
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            metrics["gpu_count"] = gpu_count
            
            for i in range(gpu_count):
                metrics[f"gpu_{i}_memory_allocated"] = torch.cuda.memory_allocated(i)
                metrics[f"gpu_{i}_memory_reserved"] = torch.cuda.memory_reserved(i)
                metrics[f"gpu_{i}_utilization"] = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
        
        # System metrics
        metrics["cpu_percent"] = psutil.cpu_percent()
        metrics["memory_percent"] = psutil.virtual_memory().percent
        metrics["disk_usage"] = psutil.disk_usage('/').percent
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Configuration
    host = os.environ.get("VLLM_HOST", "0.0.0.0")
    port = int(os.environ.get("VLLM_PORT", "8000"))
    workers = int(os.environ.get("VLLM_WORKERS", "1"))
    
    logger.info(f"Starting vLLM Legal AI Server on {host}:{port}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Max model length: {MAX_MODEL_LEN}")
    logger.info(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    
    uvicorn.run(
        "vllm-legal-server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False
    )