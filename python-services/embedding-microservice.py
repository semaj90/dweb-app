# embedding-microservice.py
# Production FastAPI embedding service with GPU batching and Qdrant integration
# Install: pip install fastapi uvicorn sentence-transformers torch httpx psycopg2-binary qdrant-client

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import torch
from sentence_transformers import SentenceTransformer
import numpy as np

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Configuration
MODEL_NAME = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
MAX_BATCH_SIZE = int(os.getenv("MAX_EMBED_BATCH_SIZE", "128"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_documents")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    ids: Optional[List[str]] = None
    owner_type: str = Field(default="chunk")
    owner_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    store_in_qdrant: bool = Field(default=True)
    store_in_postgres: bool = Field(default=True)

class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    ids: List[str]
    processing_time_ms: int
    model_info: Dict[str, str]
    stored_count: int

class BatchEmbedRequest(BaseModel):
    requests: List[EmbedRequest] = Field(..., max_items=10)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    qdrant_connected: bool
    postgres_connected: bool
    
class EmbeddingService:
    def __init__(self):
        self.model = None
        self.qdrant_client = None
        self.app = FastAPI(
            title="Legal AI Embedding Service",
            description="High-performance GPU embedding service for legal documents",
            version="1.0.0"
        )
        self.setup_routes()
        self.setup_middleware()
        
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            await self.initialize()
            
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            return await self.get_health_status()
            
        @self.app.post("/embed", response_model=EmbedResponse)
        async def embed_texts(request: EmbedRequest, background_tasks: BackgroundTasks):
            return await self.process_embedding_request(request, background_tasks)
            
        @self.app.post("/embed/batch", response_model=List[EmbedResponse])
        async def embed_batch(request: BatchEmbedRequest, background_tasks: BackgroundTasks):
            return await self.process_batch_embedding(request, background_tasks)
            
        @self.app.get("/model/info")
        async def model_info():
            if not self.model:
                raise HTTPException(status_code=503, detail="Model not loaded")
            return {
                "model_name": MODEL_NAME,
                "max_sequence_length": self.model.get_max_seq_length(),
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "device": str(self.model.device),
                "batch_size": BATCH_SIZE
            }
            
        @self.app.post("/qdrant/search")
        async def search_vectors(query_text: str, limit: int = 10, score_threshold: float = 0.7):
            return await self.search_similar_vectors(query_text, limit, score_threshold)
            
    async def initialize(self):
        """Initialize the embedding service"""
        logger.info(f"Initializing embedding service with device: {DEVICE}")
        
        # Load embedding model
        try:
            logger.info(f"Loading model: {MODEL_NAME}")
            self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            logger.info(f"Model loaded successfully on {DEVICE}")
            logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
            
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL)
            await self.ensure_qdrant_collection()
            logger.info("Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            
    async def ensure_qdrant_collection(self):
        """Ensure the Qdrant collection exists with correct configuration"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if COLLECTION_NAME not in collection_names:
                logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {COLLECTION_NAME} created successfully")
            else:
                logger.info(f"Collection {COLLECTION_NAME} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            raise
            
    async def get_health_status(self) -> HealthResponse:
        """Get comprehensive health status"""
        qdrant_connected = False
        postgres_connected = False
        
        # Test Qdrant connection
        try:
            if self.qdrant_client:
                collections = self.qdrant_client.get_collections()
                qdrant_connected = True
        except:
            pass
            
        # Test PostgreSQL connection
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.close()
            postgres_connected = True
        except:
            pass
            
        return HealthResponse(
            status="healthy" if self.model and qdrant_connected and postgres_connected else "degraded",
            model_loaded=self.model is not None,
            device=DEVICE,
            qdrant_connected=qdrant_connected,
            postgres_connected=postgres_connected
        )
        
    async def process_embedding_request(self, request: EmbedRequest, background_tasks: BackgroundTasks) -> EmbedResponse:
        """Process a single embedding request with GPU acceleration"""
        if not self.model:
            raise HTTPException(status_code=503, detail="Embedding model not loaded")
            
        start_time = datetime.now()
        
        # Generate IDs if not provided
        if not request.ids:
            request.ids = [str(uuid.uuid4()) for _ in request.texts]
        elif len(request.ids) != len(request.texts):
            raise HTTPException(status_code=400, detail="IDs and texts length mismatch")
            
        # Generate owner IDs if not provided
        if not request.owner_ids:
            request.owner_ids = request.ids.copy()
        elif len(request.owner_ids) != len(request.texts):
            raise HTTPException(status_code=400, detail="Owner IDs and texts length mismatch")
            
        try:
            # Process embeddings in batches for GPU efficiency
            all_embeddings = []
            
            for i in range(0, len(request.texts), MAX_BATCH_SIZE):
                batch_texts = request.texts[i:i + MAX_BATCH_SIZE]
                
                # Generate embeddings with GPU acceleration
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=min(BATCH_SIZE, len(batch_texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=DEVICE,
                    normalize_embeddings=True
                )
                
                all_embeddings.extend(batch_embeddings.tolist())
                
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Store embeddings asynchronously
            stored_count = 0
            if request.store_in_qdrant or request.store_in_postgres:
                background_tasks.add_task(
                    self.store_embeddings_async,
                    request.ids,
                    request.owner_ids,
                    request.owner_type,
                    all_embeddings,
                    request.metadata,
                    request.store_in_qdrant,
                    request.store_in_postgres
                )
                stored_count = len(all_embeddings)
                
            return EmbedResponse(
                vectors=all_embeddings,
                ids=request.ids,
                processing_time_ms=processing_time,
                model_info={
                    "model_name": MODEL_NAME,
                    "device": DEVICE,
                    "dimension": str(len(all_embeddings[0]) if all_embeddings else 0)
                },
                stored_count=stored_count
            )
            
        except Exception as e:
            logger.error(f"Embedding processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
            
    async def process_batch_embedding(self, request: BatchEmbedRequest, background_tasks: BackgroundTasks) -> List[EmbedResponse]:
        """Process multiple embedding requests efficiently"""
        responses = []
        
        for embed_request in request.requests:
            try:
                response = await self.process_embedding_request(embed_request, background_tasks)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch processing error for request: {e}")
                # Continue with other requests
                continue
                
        return responses
        
    async def store_embeddings_async(self, ids: List[str], owner_ids: List[str], 
                                   owner_type: str, embeddings: List[List[float]], 
                                   metadata: Dict[str, Any], store_qdrant: bool, store_postgres: bool):
        """Store embeddings in Qdrant and PostgreSQL asynchronously"""
        try:
            # Store in Qdrant
            if store_qdrant and self.qdrant_client:
                await self.store_in_qdrant(ids, owner_ids, owner_type, embeddings, metadata)
                
            # Store in PostgreSQL
            if store_postgres:
                await self.store_in_postgres(ids, owner_ids, owner_type, embeddings, metadata)
                
        except Exception as e:
            logger.error(f"Async storage failed: {e}")
            
    async def store_in_qdrant(self, ids: List[str], owner_ids: List[str], 
                             owner_type: str, embeddings: List[List[float]], 
                             metadata: Dict[str, Any]):
        """Store embeddings in Qdrant vector database"""
        try:
            points = []
            
            for i, (embed_id, owner_id, embedding) in enumerate(zip(ids, owner_ids, embeddings)):
                payload = {
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                    "created_at": datetime.now().isoformat(),
                    **metadata
                }
                
                points.append(PointStruct(
                    id=embed_id,
                    vector=embedding,
                    payload=payload
                ))
                
            # Batch upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            
            logger.info(f"Stored {len(points)} vectors in Qdrant")
            
        except Exception as e:
            logger.error(f"Qdrant storage failed: {e}")
            raise
            
    async def store_in_postgres(self, ids: List[str], owner_ids: List[str], 
                               owner_type: str, embeddings: List[List[float]], 
                               metadata: Dict[str, Any]):
        """Store embeddings in PostgreSQL with pgvector"""
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            
            # Prepare batch insert
            insert_query = """
                INSERT INTO vectors (id, owner_type, owner_id, embedding, payload, created_at, updated_at)
                VALUES %s
                ON CONFLICT (owner_type, owner_id) 
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
            """
            
            # Prepare data tuples
            data_tuples = []
            for embed_id, owner_id, embedding in zip(ids, owner_ids, embeddings):
                payload = {
                    "model": MODEL_NAME,
                    "created_at": datetime.now().isoformat(),
                    **metadata
                }
                
                data_tuples.append((
                    embed_id,
                    owner_type,
                    owner_id,
                    embedding,
                    json.dumps(payload),
                    datetime.now(),
                    datetime.now()
                ))
                
            # Execute batch insert
            from psycopg2.extras import execute_values
            execute_values(cur, insert_query, data_tuples)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(data_tuples)} vectors in PostgreSQL")
            
        except Exception as e:
            logger.error(f"PostgreSQL storage failed: {e}")
            raise
            
    async def search_similar_vectors(self, query_text: str, limit: int = 10, score_threshold: float = 0.7):
        """Search for similar vectors in Qdrant"""
        if not self.model or not self.qdrant_client:
            raise HTTPException(status_code=503, detail="Service not available")
            
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query_text], normalize_embeddings=True)[0].tolist()
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
                
            return {
                "query": query_text,
                "results": results,
                "total_found": len(results)
            }
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Create service instance
service = EmbeddingService()
app = service.app

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal AI Embedding Microservice")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=9001, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"""
    🚀 Starting Legal AI Embedding Microservice
    
    Configuration:
    - Model: {MODEL_NAME}
    - Device: {DEVICE}
    - Batch Size: {BATCH_SIZE}
    - Qdrant URL: {QDRANT_URL}
    - Collection: {COLLECTION_NAME}
    
    Endpoints:
    - Health: http://{args.host}:{args.port}/health
    - Embed: http://{args.host}:{args.port}/embed
    - Search: http://{args.host}:{args.port}/qdrant/search
    - Docs: http://{args.host}:{args.port}/docs
    """)
    
    uvicorn.run(
        "embedding-microservice:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        access_log=True
    )