"""
Local Embedding Service with SentenceTransformer
Fully private, no external API calls
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")

# Global variables
model: Optional[SentenceTransformer] = None
qdrant_client: Optional[QdrantClient] = None
redis_client: Optional[redis.Redis] = None

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation"""
    texts: List[str] = Field(..., description="List of texts to embed")
    normalize: bool = Field(True, description="Whether to normalize embeddings")

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model_name: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Embedding dimensions")
    processing_time: float = Field(..., description="Processing time in seconds")

class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query")
    collection_name: str = Field("legal_documents", description="Qdrant collection name")
    limit: int = Field(10, description="Number of results to return")
    score_threshold: float = Field(0.7, description="Minimum similarity score")

class SearchResponse(BaseModel):
    """Response model for semantic search"""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    query_embedding: List[float] = Field(..., description="Query embedding")
    processing_time: float = Field(..., description="Processing time in seconds")

class DocumentRequest(BaseModel):
    """Request model for document indexing"""
    documents: List[Dict[str, Any]] = Field(..., description="Documents to index")
    collection_name: str = Field("legal_documents", description="Qdrant collection name")
    batch_size: int = Field(100, description="Batch size for indexing")

async def load_model():
    """Load the SentenceTransformer model"""
    global model
    try:
        logger.info(f"Loading model {MODEL_NAME} on {DEVICE}")
        
        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        # Load model with caching
        model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
            cache_folder=MODEL_CACHE_DIR
        )
        
        # Warm up the model
        _ = model.encode(["test"], show_progress_bar=False)
        
        logger.info(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

async def init_qdrant():
    """Initialize Qdrant client"""
    global qdrant_client
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # Test connection
        collections = qdrant_client.get_collections()
        logger.info(f"Connected to Qdrant. Collections: {len(collections.collections)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return False

async def init_redis():
    """Initialize Redis client"""
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL)
        
        # Test connection
        await redis_client.ping()
        logger.info("Connected to Redis successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Local Embedding Service...")
    
    # Initialize services
    model_loaded = await load_model()
    qdrant_connected = await init_qdrant()
    redis_connected = await init_redis()
    
    if not model_loaded:
        raise RuntimeError("Failed to load embedding model")
    
    logger.info("Local Embedding Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local Embedding Service...")
    if redis_client:
        await redis_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="Local Embedding Service",
    description="Fully private embedding service using SentenceTransformer",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "model_loaded": model is not None,
        "qdrant_connected": qdrant_client is not None,
        "redis_connected": redis_client is not None
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "device": str(model.device),
        "max_seq_length": model.max_seq_length,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

async def get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get cached embedding from Redis"""
    if not redis_client:
        return None
    
    try:
        cache_key = f"embedding:{hash(text)}"
        cached = await redis_client.get(cache_key)
        if cached:
            return eval(cached.decode())  # Convert back to list
    except Exception as e:
        logger.warning(f"Cache retrieval failed: {e}")
    
    return None

async def cache_embedding(text: str, embedding: List[float]):
    """Cache embedding in Redis"""
    if not redis_client:
        return
    
    try:
        cache_key = f"embedding:{hash(text)}"
        await redis_client.setex(cache_key, CACHE_TTL, str(embedding))
    except Exception as e:
        logger.warning(f"Cache storage failed: {e}")

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Generate embeddings for given texts"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        # Check cache for single text requests
        if len(request.texts) == 1:
            cached = await get_cached_embedding(request.texts[0])
            if cached:
                return EmbeddingResponse(
                    embeddings=[cached],
                    model_name=MODEL_NAME,
                    dimensions=len(cached),
                    processing_time=time.time() - start_time
                )
        
        # Generate embeddings
        embeddings = model.encode(
            request.texts,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
            batch_size=BATCH_SIZE
        )
        
        # Convert to list format
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Cache single embeddings
        if len(request.texts) == 1:
            background_tasks.add_task(cache_embedding, request.texts[0], embeddings_list[0])
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model_name=MODEL_NAME,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Perform semantic search using Qdrant"""
    if not model or not qdrant_client:
        raise HTTPException(status_code=503, detail="Services not available")
    
    import time
    start_time = time.time()
    
    try:
        # Generate query embedding
        query_embedding = model.encode([request.query], normalize_embeddings=True)[0]
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=request.collection_name,
            query_vector=query_embedding.tolist(),
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            query_embedding=query_embedding.tolist(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/index")
async def index_documents(request: DocumentRequest):
    """Index documents in Qdrant"""
    if not model or not qdrant_client:
        raise HTTPException(status_code=503, detail="Services not available")
    
    try:
        # Create collection if it doesn't exist
        try:
            qdrant_client.get_collection(request.collection_name)
        except:
            embedding_dim = model.get_sentence_embedding_dimension()
            qdrant_client.create_collection(
                collection_name=request.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
        
        # Process documents in batches
        total_indexed = 0
        
        for i in range(0, len(request.documents), request.batch_size):
            batch = request.documents[i:i + request.batch_size]
            
            # Extract texts for embedding
            texts = [doc.get("content", doc.get("text", "")) for doc in batch]
            
            # Generate embeddings
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            
            # Create points for Qdrant
            points = []
            for idx, (doc, embedding) in enumerate(zip(batch, embeddings)):
                point = PointStruct(
                    id=doc.get("id", f"{request.collection_name}_{total_indexed + idx}"),
                    vector=embedding.tolist(),
                    payload=doc
                )
                points.append(point)
            
            # Upload to Qdrant
            qdrant_client.upsert(
                collection_name=request.collection_name,
                points=points
            )
            
            total_indexed += len(batch)
            logger.info(f"Indexed {total_indexed}/{len(request.documents)} documents")
        
        return {
            "message": f"Successfully indexed {total_indexed} documents",
            "collection_name": request.collection_name,
            "total_documents": total_indexed
        }
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.get("/collections")
async def list_collections():
    """List all Qdrant collections"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")
    
    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "status": col.status,
                    "vectors_count": qdrant_client.count(col.name).count
                }
                for col in collections.collections
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)