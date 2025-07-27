"""
Local RAG API Service
Coordinates between embedding service and vector database
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
import redis.asyncio as redis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "legal_documents")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Global clients
http_client: Optional[httpx.AsyncClient] = None
qdrant_client: Optional[QdrantClient] = None
redis_client: Optional[redis.Redis] = None

class RAGQuery(BaseModel):
    """RAG query request"""
    query: str = Field(..., description="Search query")
    collection_name: str = Field(RAG_COLLECTION_NAME, description="Collection to search")
    limit: int = Field(10, description="Number of results")
    score_threshold: float = Field(0.7, description="Minimum similarity score")
    include_metadata: bool = Field(True, description="Include document metadata")
    context_window: int = Field(2048, description="Context window for results")

class RAGResponse(BaseModel):
    """RAG query response"""
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    context: str = Field(..., description="Formatted context for LLM")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

class DocumentUpload(BaseModel):
    """Document upload request"""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    collection_name: str = Field(RAG_COLLECTION_NAME, description="Collection name")
    chunk_size: int = Field(1000, description="Chunk size for splitting")
    chunk_overlap: int = Field(200, description="Overlap between chunks")

class IndexStats(BaseModel):
    """Index statistics"""
    total_documents: int = Field(..., description="Total documents indexed")
    total_chunks: int = Field(..., description="Total chunks created")
    collections: List[Dict[str, Any]] = Field(..., description="Collection information")

async def init_clients():
    """Initialize HTTP and database clients"""
    global http_client, qdrant_client, redis_client
    
    # HTTP client for embedding service
    http_client = httpx.AsyncClient(timeout=30.0)
    
    # Qdrant client
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        collections = qdrant_client.get_collections()
        logger.info(f"Connected to Qdrant. Collections: {len(collections.collections)}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
    
    # Redis client
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

async def close_clients():
    """Close all clients"""
    if http_client:
        await http_client.aclose()
    if redis_client:
        await redis_client.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Local RAG API Service...")
    await init_clients()
    logger.info("Local RAG API Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local RAG API Service...")
    await close_clients()

# Initialize FastAPI app
app = FastAPI(
    title="Local RAG API",
    description="Fully private RAG API using local embedding service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check embedding service
    embedding_healthy = False
    try:
        response = await http_client.get(f"{EMBEDDING_SERVICE_URL}/health")
        embedding_healthy = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy",
        "embedding_service": embedding_healthy,
        "qdrant_connected": qdrant_client is not None,
        "redis_connected": redis_client is not None
    }

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from local embedding service"""
    try:
        response = await http_client.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json={"texts": texts, "normalize": True}
        )
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding service error: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_break = max(last_period, last_newline)
            
            if last_break > start + chunk_size // 2:
                chunk = text[start:start + last_break + 1]
                end = start + last_break + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if chunk.strip()]

@app.post("/query", response_model=RAGResponse)
async def rag_query(request: RAGQuery):
    """Perform RAG query"""
    try:
        # Get query embedding
        query_embeddings = await get_embeddings([request.query])
        query_embedding = query_embeddings[0]
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=request.collection_name,
            query_vector=query_embedding,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        # Format results
        results = []
        context_parts = []
        
        for result in search_results:
            result_data = {
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "metadata": result.payload.get("metadata", {}) if request.include_metadata else {}
            }
            results.append(result_data)
            
            # Add to context
            content = result.payload.get("content", "")
            if len(content) > request.context_window:
                content = content[:request.context_window] + "..."
            context_parts.append(f"Document {len(context_parts) + 1}:\n{content}")
        
        # Create formatted context
        context = "\n\n".join(context_parts)
        
        # Metadata
        metadata = {
            "total_results": len(results),
            "collection_name": request.collection_name,
            "query_embedding_dim": len(query_embedding),
            "avg_score": sum(r["score"] for r in results) / len(results) if results else 0
        }
        
        return RAGResponse(
            query=request.query,
            results=results,
            context=context,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    collection_name: str = Form(RAG_COLLECTION_NAME),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Upload and index a document"""
    try:
        # Read file content
        content = await file.read()
        
        # Basic file type handling
        if file.filename.endswith('.txt'):
            text_content = content.decode('utf-8')
        elif file.filename.endswith('.pdf'):
            # Would need PyPDF2 implementation
            text_content = content.decode('utf-8', errors='ignore')
        else:
            text_content = content.decode('utf-8', errors='ignore')
        
        # Parse metadata
        try:
            doc_metadata = json.loads(metadata)
        except:
            doc_metadata = {}
        
        doc_metadata.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(content)
        })
        
        # Chunk the text
        chunks = chunk_text(text_content, chunk_size, chunk_overlap)
        
        # Prepare documents for indexing
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = doc_metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            
            documents.append({
                "id": f"{file.filename}_{i}",
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        # Index documents using embedding service
        response = await http_client.post(
            f"{EMBEDDING_SERVICE_URL}/index",
            json={
                "documents": documents,
                "collection_name": collection_name,
                "batch_size": 50
            }
        )
        response.raise_for_status()
        
        return {
            "message": f"Successfully uploaded and indexed {file.filename}",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "collection_name": collection_name,
            "total_characters": len(text_content)
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/index")
async def index_documents(request: DocumentUpload):
    """Index documents directly from JSON"""
    try:
        # Chunk the content
        chunks = chunk_text(request.content, request.chunk_size, request.chunk_overlap)
        
        # Prepare documents
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = request.metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            
            documents.append({
                "id": f"doc_{hash(request.content)}_{i}",
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        # Index using embedding service
        response = await http_client.post(
            f"{EMBEDDING_SERVICE_URL}/index",
            json={
                "documents": documents,
                "collection_name": request.collection_name,
                "batch_size": 50
            }
        )
        response.raise_for_status()
        
        return {
            "message": "Documents indexed successfully",
            "chunks_created": len(chunks),
            "collection_name": request.collection_name
        }
        
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.get("/stats", response_model=IndexStats)
async def get_index_stats():
    """Get indexing statistics"""
    try:
        # Get collections from embedding service
        response = await http_client.get(f"{EMBEDDING_SERVICE_URL}/collections")
        response.raise_for_status()
        collections_data = response.json()
        
        total_documents = sum(col.get("vectors_count", 0) for col in collections_data.get("collections", []))
        
        return IndexStats(
            total_documents=total_documents,
            total_chunks=total_documents,  # Assuming 1:1 for now
            collections=collections_data.get("collections", [])
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.delete("/collection/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        qdrant_client.delete_collection(collection_name)
        return {"message": f"Collection {collection_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=f"Collection deletion failed: {str(e)}")

# Legal-specific endpoints

@app.post("/legal/case-search")
async def legal_case_search(
    query: str = Form(...),
    jurisdiction: str = Form("federal"),
    case_type: str = Form("all"),
    date_range: str = Form("all"),
    limit: int = Form(10)
):
    """Search legal cases with domain-specific filtering"""
    # Add legal-specific metadata filtering
    enhanced_query = f"Legal case: {query}"
    if jurisdiction != "all":
        enhanced_query += f" jurisdiction:{jurisdiction}"
    if case_type != "all":
        enhanced_query += f" type:{case_type}"
    
    request = RAGQuery(
        query=enhanced_query,
        limit=limit,
        score_threshold=0.6,
        collection_name="legal_cases"
    )
    
    return await rag_query(request)

@app.post("/legal/statute-search")
async def legal_statute_search(
    query: str = Form(...),
    code_section: str = Form("all"),
    limit: int = Form(10)
):
    """Search legal statutes and codes"""
    enhanced_query = f"Legal statute: {query}"
    if code_section != "all":
        enhanced_query += f" code:{code_section}"
    
    request = RAGQuery(
        query=enhanced_query,
        limit=limit,
        score_threshold=0.7,
        collection_name="legal_statutes"
    )
    
    return await rag_query(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)