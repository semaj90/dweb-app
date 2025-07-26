"""
Vector Store Service with PostgreSQL + pgvector
Handles document embeddings and semantic search for legal RAG
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import asyncpg
from loguru import logger
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib

class VectorStoreService:
    """Advanced vector store with PostgreSQL + pgvector and Qdrant integration"""
    
    def __init__(self):
        self.pg_connection = None
        self.qdrant_client = None
        self.db_url = os.getenv("DATABASE_URL")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = "legal_documents"
        
    async def initialize(self):
        """Initialize vector store connections"""
        try:
            # Initialize PostgreSQL connection
            await self._initialize_postgres()
            
            # Initialize Qdrant client
            await self._initialize_qdrant()
            
            logger.info("✅ Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    async def _initialize_postgres(self):
        """Initialize PostgreSQL connection with pgvector"""
        try:
            if not self.db_url:
                raise ValueError("DATABASE_URL not configured")
            
            self.pg_connection = await asyncpg.connect(self.db_url)
            
            # Test pgvector extension
            await self.pg_connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            logger.info("✅ PostgreSQL + pgvector connection established")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL initialization failed: {e}")
            raise
    
    async def _initialize_qdrant(self):
        """Initialize Qdrant vector database"""
        try:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            
            # Create collection if it doesn't exist
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"✅ Qdrant collection '{self.collection_name}' exists")
            except:
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Sentence transformer embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created Qdrant collection '{self.collection_name}'")
            
        except Exception as e:
            logger.warning(f"⚠️  Qdrant initialization failed: {e}")
            self.qdrant_client = None
    
    async def store_document_embedding(
        self, 
        document_id: str,
        title: str,
        content: str,
        embedding: List[float],
        document_type: str,
        case_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store document embedding in both PostgreSQL and Qdrant"""
        try:
            # Calculate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Store in PostgreSQL
            await self._store_in_postgres(
                document_id, title, content, embedding, 
                document_type, case_id, metadata, content_hash
            )
            
            # Store in Qdrant if available
            if self.qdrant_client:
                await self._store_in_qdrant(
                    document_id, embedding, title, 
                    document_type, case_id, metadata
                )
            
            logger.info(f"✅ Stored document embedding: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store document embedding: {e}")
            return False
    
    async def _store_in_postgres(
        self, 
        document_id: str, 
        title: str, 
        content: str, 
        embedding: List[float],
        document_type: str, 
        case_id: Optional[str], 
        metadata: Optional[Dict],
        content_hash: str
    ):
        """Store document in PostgreSQL with pgvector"""
        try:
            query = """
            INSERT INTO legal_documents 
            (id, title, content, document_type, case_id, metadata, embedding, file_hash, indexed_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (file_hash) 
            DO UPDATE SET 
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                document_type = EXCLUDED.document_type,
                case_id = EXCLUDED.case_id,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = NOW(),
                indexed_at = NOW()
            """
            
            await self.pg_connection.execute(
                query,
                document_id,
                title,
                content,
                document_type,
                case_id,
                json.dumps(metadata) if metadata else None,
                embedding,
                content_hash
            )
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL storage failed: {e}")
            raise
    
    async def _store_in_qdrant(
        self, 
        document_id: str, 
        embedding: List[float], 
        title: str,
        document_type: str, 
        case_id: Optional[str], 
        metadata: Optional[Dict]
    ):
        """Store document in Qdrant vector database"""
        try:
            point = PointStruct(
                id=document_id,
                vector=embedding,
                payload={
                    "title": title,
                    "document_type": document_type,
                    "case_id": case_id,
                    "metadata": metadata or {}
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
        except Exception as e:
            logger.error(f"❌ Qdrant storage failed: {e}")
            # Don't raise - Qdrant is optional
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        confidence_threshold: float = 0.7,
        case_id: Optional[str] = None,
        document_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using pgvector"""
        try:
            # Build query conditions
            conditions = ["embedding <=> $1 < $2"]  # pgvector cosine distance
            params = [query_embedding, 1.0 - confidence_threshold]  # Convert similarity to distance
            param_count = 2
            
            if case_id:
                param_count += 1
                conditions.append(f"case_id = ${param_count}")
                params.append(case_id)
            
            if document_types:
                param_count += 1
                conditions.append(f"document_type = ANY(${param_count})")
                params.append(document_types)
            
            # Build the query
            where_clause = " AND ".join(conditions)
            query = f"""
            SELECT 
                id as document_id,
                title,
                content,
                document_type,
                case_id,
                metadata,
                (1 - (embedding <=> $1)) as similarity_score,
                created_at
            FROM legal_documents
            WHERE {where_clause}
            ORDER BY embedding <=> $1
            LIMIT ${param_count + 1}
            """
            
            params.append(limit)
            
            # Execute search
            results = await self.pg_connection.fetch(query, *params)
            
            # Format results
            formatted_results = []
            for row in results:
                result = {
                    "document_id": str(row["document_id"]),
                    "title": row["title"],
                    "content": row["content"],
                    "document_type": row["document_type"],
                    "case_id": row["case_id"],
                    "similarity_score": float(row["similarity_score"]),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat()
                }
                formatted_results.append(result)
            
            logger.info(f"🔍 Found {len(formatted_results)} documents with similarity > {confidence_threshold}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            
            # Fallback to Qdrant if PostgreSQL fails
            if self.qdrant_client:
                return await self._search_qdrant(
                    query_embedding, limit, confidence_threshold, 
                    case_id, document_types
                )
            
            return []
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        try:
            query = """
            SELECT id, title, content, document_type, case_id, metadata, 
                   embedding, created_at, updated_at
            FROM legal_documents
            WHERE id = $1
            """
            
            result = await self.pg_connection.fetchrow(query, document_id)
            
            if result:
                return {
                    "document_id": str(result["id"]),
                    "title": result["title"],
                    "content": result["content"],
                    "document_type": result["document_type"],
                    "case_id": result["case_id"],
                    "metadata": json.loads(result["metadata"]) if result["metadata"] else {},
                    "embedding": result["embedding"],
                    "created_at": result["created_at"].isoformat(),
                    "updated_at": result["updated_at"].isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get document {document_id}: {e}")
            return None
    
    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            # PostgreSQL stats
            pg_stats_query = """
            SELECT 
                COUNT(*) as total_documents,
                COUNT(DISTINCT case_id) as unique_cases,
                COUNT(DISTINCT document_type) as document_types,
                AVG(LENGTH(content)) as avg_content_length,
                MIN(created_at) as oldest_document,
                MAX(created_at) as newest_document
            FROM legal_documents
            WHERE case_id IS NOT NULL
            """
            
            pg_stats = await self.pg_connection.fetchrow(pg_stats_query)
            
            # Document type distribution
            type_dist_query = """
            SELECT document_type, COUNT(*) as count
            FROM legal_documents
            GROUP BY document_type
            ORDER BY count DESC
            """
            
            type_distribution = await self.pg_connection.fetch(type_dist_query)
            
            stats = {
                "total_documents": pg_stats["total_documents"],
                "unique_cases": pg_stats["unique_cases"],
                "document_types_count": pg_stats["document_types"],
                "avg_content_length": float(pg_stats["avg_content_length"]) if pg_stats["avg_content_length"] else 0,
                "oldest_document": pg_stats["oldest_document"].isoformat() if pg_stats["oldest_document"] else None,
                "newest_document": pg_stats["newest_document"].isoformat() if pg_stats["newest_document"] else None,
                "document_type_distribution": {
                    row["document_type"]: row["count"] for row in type_distribution
                }
            }
            
            # Add Qdrant stats if available
            if self.qdrant_client:
                try:
                    qdrant_info = self.qdrant_client.get_collection(self.collection_name)
                    stats["qdrant_points"] = qdrant_info.points_count
                    stats["qdrant_status"] = "connected"
                except:
                    stats["qdrant_status"] = "disconnected"
            else:
                stats["qdrant_status"] = "not_configured"
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close vector store connections"""
        try:
            if self.pg_connection:
                await self.pg_connection.close()
            
            if self.qdrant_client:
                self.qdrant_client.close()
            
            logger.info("👋 Vector store connections closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing vector store: {e}")
