#!/usr/bin/env python3
"""
Embedding Microservice - FastAPI with GPU batching
Supports sentence-transformers, nomic-embed, and gemma3-legal
Native Windows, Redis Streams, Qdrant integration
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import psycopg
from psycopg.rows import dict_row
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_embeddings")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
GPU_DEVICE = os.getenv("GPU_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Global state
embedding_model: Optional[SentenceTransformer] = None
redis_client: Optional[redis.Redis] = None
qdrant_client: Optional[QdrantClient] = None
thread_pool: Optional[ThreadPoolExecutor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup()
    yield
    # Shutdown
    await shutdown()

app = FastAPI(title="Legal AI Embedding Service", version="1.0.0", lifespan=lifespan)

@dataclass 
class EmbedRequest:
    texts: List[str]
    owner_type: str = "document"
    owner_id: str = ""
    metadata: Dict[str, Any] = None
    normalize: bool = True

class EmbedBatch(BaseModel):
    texts: List[str]
    owner_type: Optional[str] = "document"
    owner_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    normalize: bool = True
    store_qdrant: bool = True
    store_postgres: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    dimensions: int
    processing_time_ms: float
    device: str

async def startup():
    global embedding_model, redis_client, qdrant_client, thread_pool
    
    logger.info(f"🚀 Starting Embedding Service on {GPU_DEVICE}")
    logger.info(f"📊 Model: {MODEL_NAME}")
    logger.info(f"🔢 Batch size: {BATCH_SIZE}")
    
    # Initialize embedding model
    try:
        embedding_model = SentenceTransformer(MODEL_NAME, device=GPU_DEVICE, trust_remote_code=True)
        embedding_model.eval()
        logger.info(f"✅ Model loaded: {MODEL_NAME} on {GPU_DEVICE}")
        
        # Test embedding to warm up
        test_embed = embedding_model.encode(["test sentence"], normalize_embeddings=True)
        logger.info(f"🎯 Model dimensions: {test_embed.shape[1]}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Initialize Redis
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise
    
    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        # Ensure collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"🔧 Creating Qdrant collection: {QDRANT_COLLECTION}")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        logger.info("✅ Qdrant connected")
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        raise
    
    # Initialize thread pool for blocking operations
    thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed-worker")
    
    # Start Redis stream consumer
    asyncio.create_task(consume_redis_stream())
    logger.info("✅ Service startup complete")

async def shutdown():
    global redis_client, thread_pool
    if redis_client:
        await redis_client.close()
    if thread_pool:
        thread_pool.shutdown(wait=True)
    logger.info("🛑 Service shutdown complete")

@app.get("/health")
async def health():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
    
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": GPU_DEVICE,
        "gpu_available": gpu_available,
        "gpu_memory_gb": gpu_memory,
        "batch_size": BATCH_SIZE,
        "redis_connected": redis_client is not None,
        "qdrant_connected": qdrant_client is not None,
        "timestamp": time.time()
    }

def compute_embeddings_batch(texts: List[str], normalize: bool = True) -> np.ndarray:
    """Compute embeddings using the loaded model (CPU/GPU optimized)"""
    if not embedding_model:
        raise ValueError("Embedding model not loaded")
    
    # Process in chunks for memory efficiency
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_embeddings = embedding_model.encode(
            batch_texts, 
            normalize_embeddings=normalize,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)

@app.post("/embed/batch", response_model=EmbedResponse)
async def embed_batch(request: EmbedBatch, background_tasks: BackgroundTasks):
    """Batch embedding endpoint with optional storage"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    start_time = time.time()
    
    try:
        # Run embedding computation in thread pool
        embeddings = await asyncio.get_event_loop().run_in_executor(
            thread_pool, compute_embeddings_batch, request.texts, request.normalize
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        # Store in background if requested
        if request.store_qdrant or request.store_postgres:
            background_tasks.add_task(
                store_embeddings,
                embeddings_list,
                request.texts,
                request.owner_type or "document",
                request.owner_ids,
                request.metadata,
                request.store_qdrant,
                request.store_postgres
            )
        
        return EmbedResponse(
            embeddings=embeddings_list,
            model_name=MODEL_NAME,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            processing_time_ms=processing_time,
            device=GPU_DEVICE
        )
        
    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/single")
async def embed_single(text: str, normalize: bool = True):
    """Single text embedding - optimized for quick requests"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    start_time = time.time()
    
    try:
        embeddings = await asyncio.get_event_loop().run_in_executor(
            thread_pool, compute_embeddings_batch, [text], normalize
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "embedding": embeddings[0].tolist(),
            "model_name": MODEL_NAME,
            "dimensions": len(embeddings[0]),
            "processing_time_ms": processing_time,
            "device": GPU_DEVICE
        }
        
    except Exception as e:
        logger.error(f"❌ Single embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def store_embeddings(
    embeddings: List[List[float]], 
    texts: List[str],
    owner_type: str, 
    owner_ids: Optional[List[str]], 
    metadata: Optional[Dict[str, Any]],
    store_qdrant: bool,
    store_postgres: bool
):
    """Background task to store embeddings"""
    try:
        # Store in Qdrant
        if store_qdrant and qdrant_client:
            points = []
            for i, embedding in enumerate(embeddings):
                point_id = owner_ids[i] if owner_ids and i < len(owner_ids) else f"{owner_type}_{i}_{int(time.time())}"
                point_metadata = {
                    "text": texts[i][:500],  # Truncate text for storage
                    "owner_type": owner_type,
                    "timestamp": time.time(),
                    **(metadata or {})
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=point_metadata
                ))
            
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            logger.info(f"✅ Stored {len(points)} embeddings in Qdrant")
        
        # Store in PostgreSQL
        if store_postgres:
            await store_embeddings_postgres(embeddings, texts, owner_type, owner_ids, metadata)
            
    except Exception as e:
        logger.error(f"❌ Storage failed: {e}")

async def store_embeddings_postgres(
    embeddings: List[List[float]], 
    texts: List[str],
    owner_type: str, 
    owner_ids: Optional[List[str]], 
    metadata: Optional[Dict[str, Any]]
):
    """Store embeddings in PostgreSQL vectors table"""
    try:
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                for i, embedding in enumerate(embeddings):
                    owner_id = owner_ids[i] if owner_ids and i < len(owner_ids) else None
                    if not owner_id:
                        continue
                    
                    payload = {
                        "text_preview": texts[i][:200],
                        "source": "embedding-service",
                        "model": MODEL_NAME,
                        "timestamp": time.time(),
                        **(metadata or {})
                    }
                    
                    # Upsert vector
                    await cur.execute("""
                        INSERT INTO vectors (owner_type, owner_id, embedding, payload, updated_at)
                        VALUES (%s, %s, %s, %s, now())
                        ON CONFLICT (owner_type, owner_id) 
                        DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            payload = EXCLUDED.payload,
                            updated_at = now()
                    """, (owner_type, owner_id, embedding, json.dumps(payload)))
                
                await conn.commit()
                logger.info(f"✅ Stored {len(embeddings)} embeddings in PostgreSQL")
                
    except Exception as e:
        logger.error(f"❌ PostgreSQL storage failed: {e}")

async def consume_redis_stream():
    """Consumer for Redis Stream embedding jobs"""
    if not redis_client:
        return
        
    stream_name = "embed:requests"
    group_name = "embed-workers"
    consumer_name = "embed-worker-1"
    
    try:
        # Create consumer group
        try:
            await redis_client.xgroup_create(stream_name, group_name, id="$", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        logger.info(f"🔄 Started Redis consumer: {stream_name}:{group_name}")
        
        while True:
            try:
                # Read from stream
                messages = await redis_client.xreadgroup(
                    group_name, consumer_name, {stream_name: ">"}, count=1, block=5000
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        await process_stream_message(stream_name, group_name, msg_id, fields)
                        
            except Exception as e:
                logger.error(f"❌ Stream consumer error: {e}")
                await asyncio.sleep(1)
                
    except Exception as e:
        logger.error(f"❌ Failed to start stream consumer: {e}")

async def process_stream_message(stream_name: str, group_name: str, msg_id: str, fields: Dict):
    """Process individual Redis stream message"""
    try:
        # Parse message payload
        payload_raw = fields.get(b'payload') or fields.get('payload')
        if isinstance(payload_raw, bytes):
            payload_raw = payload_raw.decode('utf-8')
        
        payload = json.loads(payload_raw) if payload_raw else fields
        
        # Extract texts and metadata
        texts = payload.get('texts', [])
        if not texts:
            text_single = payload.get('text', '')
            if text_single:
                texts = [text_single]
        
        if not texts:
            logger.warning(f"⚠️ No texts in message {msg_id}")
            await redis_client.xack(stream_name, group_name, msg_id)
            return
        
        # Generate embeddings
        embeddings = await asyncio.get_event_loop().run_in_executor(
            thread_pool, compute_embeddings_batch, texts, True
        )
        
        # Store results
        await store_embeddings(
            embeddings.tolist(),
            texts,
            payload.get('owner_type', 'document'),
            payload.get('owner_ids'),
            payload.get('metadata'),
            payload.get('store_qdrant', True),
            payload.get('store_postgres', True)
        )
        
        # Acknowledge message
        await redis_client.xack(stream_name, group_name, msg_id)
        logger.info(f"✅ Processed stream message {msg_id}: {len(texts)} texts")
        
    except Exception as e:
        logger.error(f"❌ Failed to process stream message {msg_id}: {e}")
        # Don't acknowledge failed messages - they'll be retried

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "embedding-service:app",
        host="0.0.0.0",
        port=8096,
        reload=False,
        log_level="info"
    )