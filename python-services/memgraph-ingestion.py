# memgraph-ingestion.py
# Production Memgraph ingestion service with FAISS similarity edges
# Install: pip install psycopg2-binary faiss-cpu mgclient numpy scikit-learn

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional
import json
import uuid
from datetime import datetime
import time

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import faiss
from mgclient import connect
from sklearn.preprocessing import normalize

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db")
MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "127.0.0.1")
MEMGRAPH_PORT = int(os.getenv("MEMGRAPH_PORT", "7687"))
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "")
MEMGRAPH_PASS = os.getenv("MEMGRAPH_PASS", "")

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
MAX_NEIGHBORS = int(os.getenv("MAX_NEIGHBORS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "FlatIP")  # FlatIP, IVFFlat, HNSW

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemgraphIngestionService:
    def __init__(self):
        self.pg_conn = None
        self.mg_conn = None
        self.faiss_index = None
        self.vector_ids = []
        self.metadata_map = {}
        
    def connect_postgres(self):
        """Connect to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
            
    def connect_memgraph(self):
        """Connect to Memgraph"""
        try:
            self.mg_conn = connect(
                host=MEMGRAPH_HOST, 
                port=MEMGRAPH_PORT,
                username=MEMGRAPH_USER,
                password=MEMGRAPH_PASS
            )
            logger.info("Connected to Memgraph")
        except Exception as e:
            logger.error(f"Memgraph connection failed: {e}")
            raise
            
    def disconnect(self):
        """Clean up connections"""
        if self.pg_conn:
            self.pg_conn.close()
        if self.mg_conn:
            self.mg_conn.close()
            
    def fetch_vectors_from_postgres(self, limit: Optional[int] = None, 
                                   owner_type: Optional[str] = None) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """Fetch vectors and metadata from PostgreSQL"""
        try:
            cursor = self.pg_conn.cursor()
            
            # Build query
            base_query = """
                SELECT v.id::text, v.owner_type, v.owner_id::text, 
                       v.embedding, v.payload, v.created_at,
                       CASE 
                           WHEN v.owner_type = 'chunk' THEN c.text_excerpt
                           WHEN v.owner_type = 'document' THEN d.title
                           ELSE NULL
                       END as content_preview
                FROM vectors v
                LEFT JOIN chunks c ON v.owner_type = 'chunk' AND v.owner_id::uuid = c.id
                LEFT JOIN documents d ON v.owner_type = 'document' AND v.owner_id::uuid = d.id
                WHERE v.embedding IS NOT NULL
            """
            
            params = []
            if owner_type:
                base_query += " AND v.owner_type = %s"
                params.append(owner_type)
                
            base_query += " ORDER BY v.created_at DESC"
            
            if limit:
                base_query += " LIMIT %s"
                params.append(limit)
                
            logger.info(f"Fetching vectors with query: {base_query}")
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning("No vectors found in PostgreSQL")
                return [], np.array([]), []
                
            # Extract data
            vector_ids = []
            embeddings = []
            metadata = []
            
            for row in rows:
                vector_ids.append(row['id'])
                
                # Convert pgvector embedding to numpy array
                if isinstance(row['embedding'], list):
                    embedding = np.array(row['embedding'], dtype=np.float32)
                else:
                    # Handle other pgvector formats
                    embedding_str = str(row['embedding']).strip('[]')
                    embedding = np.fromstring(embedding_str, dtype=np.float32, sep=',')
                    
                embeddings.append(embedding)
                
                # Build comprehensive metadata
                meta = {
                    'vector_id': row['id'],
                    'owner_type': row['owner_type'],
                    'owner_id': row['owner_id'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'content_preview': row['content_preview'][:200] if row['content_preview'] else None
                }
                
                # Add payload data if exists
                if row['payload']:
                    try:
                        if isinstance(row['payload'], str):
                            payload = json.loads(row['payload'])
                        else:
                            payload = row['payload']
                        meta.update(payload)
                    except:
                        pass
                        
                metadata.append(meta)
                
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            
            logger.info(f"Fetched {len(vector_ids)} vectors with {embeddings_array.shape[1]} dimensions")
            return vector_ids, embeddings_array, metadata
            
        except Exception as e:
            logger.error(f"Error fetching vectors from PostgreSQL: {e}")
            raise
            
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search"""
        try:
            dimension = embeddings.shape[1]
            logger.info(f"Building FAISS index with {len(embeddings)} vectors, dimension {dimension}")
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = normalize(embeddings, norm='l2', axis=1)
            
            # Choose index type
            if FAISS_INDEX_TYPE == "FlatIP":
                index = faiss.IndexFlatIP(dimension)
            elif FAISS_INDEX_TYPE == "IVFFlat":
                # Use IVF for large datasets
                nlist = min(100, max(1, len(embeddings) // 100))
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                # Train the index
                index.train(normalized_embeddings)
            elif FAISS_INDEX_TYPE == "HNSW":
                # HNSW for very fast search
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 16
            else:
                # Default to flat index
                index = faiss.IndexFlatIP(dimension)
                
            # Add vectors to index
            index.add(normalized_embeddings)
            
            logger.info(f"FAISS index built successfully with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise
            
    def find_similar_vectors(self, embeddings: np.ndarray, k: int = MAX_NEIGHBORS) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar vectors using FAISS"""
        try:
            # Normalize query vectors
            normalized_embeddings = normalize(embeddings, norm='l2', axis=1)
            
            # Search for similar vectors
            scores, indices = self.faiss_index.search(normalized_embeddings, k + 1)  # +1 to exclude self
            
            return scores, indices
            
        except Exception as e:
            logger.error(f"Error finding similar vectors: {e}")
            raise
            
    def clear_memgraph_data(self):
        """Clear existing data from Memgraph"""
        try:
            cursor = self.mg_conn.cursor()
            cursor.execute("MATCH (n) DETACH DELETE n")
            self.mg_conn.commit()
            logger.info("Cleared existing Memgraph data")
        except Exception as e:
            logger.error(f"Error clearing Memgraph data: {e}")
            raise
            
    def create_vector_nodes(self, vector_ids: List[str], metadata: List[Dict]):
        """Create vector nodes in Memgraph"""
        try:
            cursor = self.mg_conn.cursor()
            
            # Create nodes in batches
            for i in range(0, len(vector_ids), BATCH_SIZE):
                batch_ids = vector_ids[i:i + BATCH_SIZE]
                batch_meta = metadata[i:i + BATCH_SIZE]
                
                for vid, meta in zip(batch_ids, batch_meta):
                    # Create comprehensive node properties
                    node_props = {
                        'id': vid,
                        'vector_id': vid,
                        'owner_type': meta.get('owner_type', 'unknown'),
                        'owner_id': meta.get('owner_id', ''),
                        'created_at': meta.get('created_at', ''),
                        'content_preview': meta.get('content_preview', '')[:100] if meta.get('content_preview') else ''
                    }
                    
                    # Add quaternion properties if available
                    if 'pose' in meta:
                        pose = meta['pose']
                        if isinstance(pose, str):
                            try:
                                pose = json.loads(pose)
                            except:
                                pose = {}
                        
                        if isinstance(pose, dict) and 'q' in pose:
                            q = pose['q']
                            node_props.update({
                                'qw': float(q.get('w', 1.0)),
                                'qx': float(q.get('x', 0.0)),
                                'qy': float(q.get('y', 0.0)),
                                'qz': float(q.get('z', 0.0))
                            })
                            
                        if isinstance(pose, dict) and 't' in pose:
                            t = pose['t']
                            node_props.update({
                                'tx': float(t.get('x', 0.0)),
                                'ty': float(t.get('y', 0.0)),
                                'tz': float(t.get('z', 0.0))
                            })
                    
                    # Create node with label based on owner type
                    label = meta.get('owner_type', 'Vector').capitalize()
                    
                    query = f"CREATE (n:{label} $props)"
                    cursor.execute(query, {'props': node_props})
                    
                self.mg_conn.commit()
                logger.info(f"Created {len(batch_ids)} vector nodes (batch {i//BATCH_SIZE + 1})")
                
        except Exception as e:
            logger.error(f"Error creating vector nodes: {e}")
            raise
            
    def create_similarity_edges(self, vector_ids: List[str], scores: np.ndarray, indices: np.ndarray):
        """Create similarity edges in Memgraph"""
        try:
            cursor = self.mg_conn.cursor()
            edge_count = 0
            
            for i, vid in enumerate(vector_ids):
                # Skip self-similarity (first result)
                for j in range(1, len(indices[i])):
                    neighbor_idx = indices[i][j]
                    similarity_score = float(scores[i][j])
                    
                    # Skip if below threshold
                    if similarity_score < SIMILARITY_THRESHOLD:
                        continue
                        
                    neighbor_id = vector_ids[neighbor_idx]
                    
                    # Create bidirectional similarity edge
                    edge_query = """
                        MATCH (a {vector_id: $vid1}), (b {vector_id: $vid2})
                        CREATE (a)-[:SIMILAR_TO {
                            score: $score,
                            similarity_type: 'cosine',
                            created_at: $created_at,
                            threshold: $threshold
                        }]->(b)
                    """
                    
                    cursor.execute(edge_query, {
                        'vid1': vid,
                        'vid2': neighbor_id,
                        'score': similarity_score,
                        'created_at': datetime.now().isoformat(),
                        'threshold': SIMILARITY_THRESHOLD
                    })
                    
                    edge_count += 1
                    
                # Commit in batches
                if (i + 1) % BATCH_SIZE == 0:
                    self.mg_conn.commit()
                    logger.info(f"Created similarity edges for {i + 1} nodes ({edge_count} edges total)")
                    
            self.mg_conn.commit()
            logger.info(f"Created {edge_count} similarity edges total")
            
        except Exception as e:
            logger.error(f"Error creating similarity edges: {e}")
            raise
            
    def create_domain_relationships(self, metadata: List[Dict]):
        """Create domain-specific relationships (case->document->chunk)"""
        try:
            cursor = self.mg_conn.cursor()
            
            # Group by owner types to create hierarchical relationships
            chunks_by_doc = {}
            docs_by_case = {}
            
            for meta in metadata:
                if meta['owner_type'] == 'chunk':
                    # Try to find document relationship
                    doc_id = meta.get('doc_id') or meta.get('document_id')
                    if doc_id:
                        if doc_id not in chunks_by_doc:
                            chunks_by_doc[doc_id] = []
                        chunks_by_doc[doc_id].append(meta['owner_id'])
                        
                elif meta['owner_type'] == 'document':
                    # Try to find case relationship
                    case_id = meta.get('case_id')
                    if case_id:
                        if case_id not in docs_by_case:
                            docs_by_case[case_id] = []
                        docs_by_case[case_id].append(meta['owner_id'])
            
            # Create document->chunk relationships
            for doc_id, chunk_ids in chunks_by_doc.items():
                for chunk_id in chunk_ids:
                    try:
                        cursor.execute("""
                            MATCH (d:Document {owner_id: $doc_id}), (c:Chunk {owner_id: $chunk_id})
                            CREATE (d)-[:CONTAINS]->(c)
                        """, {'doc_id': doc_id, 'chunk_id': chunk_id})
                    except:
                        pass  # Skip if nodes don't exist
                        
            # Create case->document relationships  
            for case_id, doc_ids in docs_by_case.items():
                for doc_id in doc_ids:
                    try:
                        cursor.execute("""
                            MATCH (c:Case {id: $case_id}), (d:Document {owner_id: $doc_id})
                            CREATE (c)-[:INCLUDES]->(d)
                        """, {'case_id': case_id, 'doc_id': doc_id})
                    except:
                        pass  # Skip if nodes don't exist
                        
            self.mg_conn.commit()
            logger.info("Created domain-specific relationships")
            
        except Exception as e:
            logger.error(f"Error creating domain relationships: {e}")
            
    def run_full_ingestion(self, limit: Optional[int] = None, owner_type: Optional[str] = None, 
                          clear_existing: bool = True):
        """Run complete ingestion pipeline"""
        try:
            start_time = time.time()
            
            # Connect to databases
            self.connect_postgres()
            self.connect_memgraph()
            
            # Clear existing data if requested
            if clear_existing:
                logger.info("Clearing existing Memgraph data...")
                self.clear_memgraph_data()
            
            # Fetch vectors from PostgreSQL
            logger.info("Fetching vectors from PostgreSQL...")
            vector_ids, embeddings, metadata = self.fetch_vectors_from_postgres(limit, owner_type)
            
            if len(vector_ids) == 0:
                logger.warning("No vectors found, exiting")
                return
                
            # Build FAISS index
            logger.info("Building FAISS similarity index...")
            self.faiss_index = self.build_faiss_index(embeddings)
            
            # Find similar vectors
            logger.info("Computing similarity relationships...")
            scores, indices = self.find_similar_vectors(embeddings, MAX_NEIGHBORS)
            
            # Create nodes in Memgraph
            logger.info("Creating vector nodes in Memgraph...")
            self.create_vector_nodes(vector_ids, metadata)
            
            # Create similarity edges
            logger.info("Creating similarity edges...")
            self.create_similarity_edges(vector_ids, scores, indices)
            
            # Create domain relationships
            logger.info("Creating domain-specific relationships...")
            self.create_domain_relationships(metadata)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Full ingestion completed in {elapsed_time:.2f} seconds")
            logger.info(f"   - Processed {len(vector_ids)} vectors")
            logger.info(f"   - Created similarity network with threshold {SIMILARITY_THRESHOLD}")
            logger.info(f"   - Max neighbors per node: {MAX_NEIGHBORS}")
            
        except Exception as e:
            logger.error(f"Full ingestion failed: {e}")
            raise
        finally:
            self.disconnect()
            
    def query_similar_vectors(self, vector_id: str, limit: int = 5):
        """Query similar vectors for a given vector ID"""
        try:
            self.connect_memgraph()
            cursor = self.mg_conn.cursor()
            
            query = """
                MATCH (n {vector_id: $vector_id})-[r:SIMILAR_TO]->(similar)
                RETURN similar.vector_id as similar_id, 
                       similar.owner_type as owner_type,
                       similar.content_preview as content,
                       r.score as similarity_score
                ORDER BY r.score DESC
                LIMIT $limit
            """
            
            cursor.execute(query, {'vector_id': vector_id, 'limit': limit})
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error querying similar vectors: {e}")
            return []
        finally:
            self.disconnect()

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memgraph Vector Ingestion Service")
    parser.add_argument("--limit", type=int, help="Limit number of vectors to process")
    parser.add_argument("--owner-type", help="Filter by owner type (chunk, document)")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing data")
    parser.add_argument("--query", help="Query similar vectors for given vector ID")
    parser.add_argument("--config", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.config:
        print(f"""
Memgraph Ingestion Configuration:
- Database URL: {DATABASE_URL}
- Memgraph: {MEMGRAPH_HOST}:{MEMGRAPH_PORT}
- Similarity Threshold: {SIMILARITY_THRESHOLD}
- Max Neighbors: {MAX_NEIGHBORS}
- Batch Size: {BATCH_SIZE}
- FAISS Index: {FAISS_INDEX_TYPE}
        """)
        return
        
    service = MemgraphIngestionService()
    
    if args.query:
        logger.info(f"Querying similar vectors for: {args.query}")
        results = service.query_similar_vectors(args.query)
        
        print(f"\nSimilar vectors for {args.query}:")
        for result in results:
            print(f"  - {result['similar_id']} ({result['owner_type']}) - Score: {result['similarity_score']:.4f}")
            if result['content']:
                print(f"    Content: {result['content'][:100]}...")
    else:
        logger.info("Starting full vector ingestion...")
        service.run_full_ingestion(
            limit=args.limit,
            owner_type=args.owner_type,
            clear_existing=not args.no_clear
        )

if __name__ == "__main__":
    main()