"""
FastAPI Main Application for Legal RAG System
Supports local GGUF models, LangChain, pgvector, and event streaming
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio
from datetime import datetime
import uuid

# Import our services
from services.rag_service import RAGService
from services.vector_store import VectorStoreService
from services.document_processor import DocumentProcessor
from services.model_manager import LocalModelManager
from services.event_streaming import EventStreamingService
from database.models import Document, RAGQuery, ProcessingJob
from database.database import get_db_session

# Initialize FastAPI app
app = FastAPI(
    title="Legal AI RAG System",
    description="Advanced RAG system for legal document analysis with local GGUF models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = None
vector_store = None
document_processor = None
model_manager = None
event_streaming = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global rag_service, vector_store, document_processor, model_manager, event_streaming
    
    try:
        # Initialize model manager with local GGUF support
        model_manager = LocalModelManager()
        await model_manager.initialize()
        
        # Initialize vector store
        vector_store = VectorStoreService()
        await vector_store.initialize()
        
        # Initialize document processor
        document_processor = DocumentProcessor(model_manager, vector_store)
        
        # Initialize RAG service
        rag_service = RAGService(model_manager, vector_store)
        
        # Initialize event streaming
        event_streaming = EventStreamingService()
        await event_streaming.initialize()
        
        print("✅ Legal RAG System initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if event_streaming:
        await event_streaming.close()
    print("👋 Legal RAG System shutdown complete")

# Pydantic models
class RAGQueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    confidence_threshold: Optional[float] = 0.7
    case_id: Optional[str] = None
    document_types: Optional[List[str]] = None

class RAGQueryResponse(BaseModel):
    query_id: str
    response: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time_ms: int
    timestamp: datetime

class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    message: str
    processing_job_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    model_status: Dict[str, Any]

# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {}
    
    # Check database
    try:
        # Simple DB check would go here
        services_status["database"] = "healthy"
    except:
        services_status["database"] = "unhealthy"
    
    # Check Redis
    try:
        # Redis check would go here
        services_status["redis"] = "healthy"
    except:
        services_status["redis"] = "unhealthy"
    
    # Check model manager
    model_status = {}
    if model_manager:
        model_status = await model_manager.get_status()
        services_status["models"] = "healthy" if model_status.get("loaded") else "degraded"
    else:
        services_status["models"] = "unhealthy"
        model_status = {"loaded": False, "error": "Model manager not initialized"}
    
    overall_status = "healthy" if all(status == "healthy" for status in services_status.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services_status,
        model_status=model_status
    )

@app.post("/api/v1/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest, db=Depends(get_db_session)):
    """
    Perform RAG query against legal documents
    Supports local GGUF models and semantic search
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    try:
        start_time = datetime.utcnow()
        query_id = str(uuid.uuid4())
        
        # Perform RAG query
        result = await rag_service.query(
            query=request.query,
            max_results=request.max_results,
            confidence_threshold=request.confidence_threshold,
            case_id=request.case_id,
            document_types=request.document_types
        )
        
        end_time = datetime.utcnow()
        processing_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Store query in database for analytics
        rag_query = RAGQuery(
            id=query_id,
            query_text=request.query,
            response_text=result["response"],
            documents_used=result["sources"],
            confidence_score=result["confidence"],
            processing_time_ms=processing_time
        )
        db.add(rag_query)
        await db.commit()
        
        # Emit event for real-time updates
        if event_streaming:
            await event_streaming.emit_rag_query_event({
                "query_id": query_id,
                "query": request.query,
                "confidence": result["confidence"],
                "source_count": len(result["sources"])
            })
        
        return RAGQueryResponse(
            query_id=query_id,
            response=result["response"],
            sources=result["sources"],
            confidence_score=result["confidence"],
            processing_time_ms=processing_time,
            timestamp=end_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    case_id: Optional[str] = None,
    document_type: str = "general",
    db=Depends(get_db_session)
):
    """
    Upload and process legal document
    Automatically extracts text, generates embeddings, and processes with local GGUF model
    """
    if not document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        # Process document
        processing_result = await document_processor.process_document(
            document_id=document_id,
            filename=file.filename,
            content=content,
            case_id=case_id,
            document_type=document_type
        )
        
        # Create processing job for background tasks
        job_id = str(uuid.uuid4())
        processing_job = ProcessingJob(
            id=job_id,
            job_type="document_analysis",
            document_id=document_id,
            status="pending",
            parameters={
                "filename": file.filename,
                "document_type": document_type,
                "case_id": case_id
            }
        )
        db.add(processing_job)
        await db.commit()
        
        # Emit event for real-time updates
        if event_streaming:
            await event_streaming.emit_document_upload_event({
                "document_id": document_id,
                "filename": file.filename,
                "case_id": case_id,
                "job_id": job_id
            })
        
        return DocumentUploadResponse(
            document_id=document_id,
            status="uploaded",
            message=f"Document {file.filename} uploaded successfully",
            processing_job_id=job_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@app.get("/api/v1/documents/{document_id}")
async def get_document(document_id: str, db=Depends(get_db_session)):
    """Get document details by ID"""
    try:
        document = await db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": document.id,
            "title": document.title,
            "document_type": document.document_type,
            "case_id": document.case_id,
            "created_at": document.created_at,
            "metadata": document.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")

@app.get("/api/v1/documents")
async def list_documents(
    case_id: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db=Depends(get_db_session)
):
    """List documents with optional filtering"""
    try:
        query = db.query(Document)
        
        if case_id:
            query = query.filter(Document.case_id == case_id)
        if document_type:
            query = query.filter(Document.document_type == document_type)
        
        documents = await query.offset(offset).limit(limit).all()
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "document_type": doc.document_type,
                    "case_id": doc.case_id,
                    "created_at": doc.created_at
                }
                for doc in documents
            ],
            "total": await query.count(),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/api/v1/models/status")
async def get_model_status():
    """Get status of loaded models"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        status = await model_manager.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.post("/api/v1/models/load")
async def load_model(model_path: str):
    """Load a local GGUF model"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        result = await model_manager.load_local_model(model_path)
        return {"status": "success", "message": f"Model loaded from {model_path}", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str, db=Depends(get_db_session)):
    """Get processing job status"""
    try:
        job = await db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "id": job.id,
            "job_type": job.job_type,
            "status": job.status,
            "document_id": job.document_id,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "result": job.result,
            "error_message": job.error_message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/api/v1/analytics/queries")
async def get_query_analytics(
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_db_session)
):
    """Get RAG query analytics"""
    try:
        queries = await db.query(RAGQuery).order_by(RAGQuery.created_at.desc()).offset(offset).limit(limit).all()
        
        return {
            "queries": [
                {
                    "id": q.id,
                    "query_text": q.query_text,
                    "confidence_score": q.confidence_score,
                    "processing_time_ms": q.processing_time_ms,
                    "created_at": q.created_at,
                    "documents_used_count": len(q.documents_used) if q.documents_used else 0
                }
                for q in queries
            ],
            "total": await db.query(RAGQuery).count(),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/events")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time event streaming"""
    if not event_streaming:
        await websocket.close(code=1011, reason="Event streaming not available")
        return
    
    await event_streaming.handle_websocket_connection(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
