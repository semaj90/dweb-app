#!/usr/bin/env python3
"""
Test script for integrated PostgreSQL + pgvector + Ollama RAG system
Run: python test-integrated-rag.py
"""

import time
import json
import subprocess
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import requests

print("🚀 Testing Integrated Legal RAG System")
print("=" * 50)

# Configuration
OLLAMA_URL = "http://localhost:11434"
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "legal_ai_db",
    "user": "legal_admin",
    "password": "123456"
}

def test_ollama_models():
    """Test that Ollama models are available"""
    print("\n📊 Testing Ollama Models...")
    
    try:
        # Check available models
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        models = response.json()["models"]
        
        required_models = ["gemma3-legal", "nomic-embed-text"]
        found_models = [m["name"].split(":")[0] for m in models]
        
        for model in required_models:
            if model in found_models:
                print(f"  ✅ {model} - Available")
            else:
                print(f"  ❌ {model} - Missing")
                return False
        
        return True
    except Exception as e:
        print(f"  ❌ Error checking models: {e}")
        return False

def test_embeddings():
    """Test embedding generation with nomic-embed-text"""
    print("\n🧮 Testing Embedding Generation...")
    
    try:
        test_text = "This is a legal contract between two parties."
        
        # Generate embedding using Ollama
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": "nomic-embed-text:latest",
                "prompt": test_text
            }
        )
        
        if response.status_code == 200:
            embedding = response.json()["embedding"]
            print(f"  ✅ Generated {len(embedding)}-dimensional embedding")
            return embedding
        else:
            print(f"  ❌ Failed to generate embedding: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error generating embedding: {e}")
        return None

def test_pgvector():
    """Test PostgreSQL with pgvector"""
    print("\n🗄️ Testing PostgreSQL + pgvector...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if pgvector extension exists
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        if not cur.fetchone():
            print("  ⚠️ pgvector extension not installed, installing...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            print("  ✅ pgvector extension installed")
        else:
            print("  ✅ pgvector extension already installed")
        
        # Create test table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS test_embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        print("  ✅ Test table ready")
        
        # Test inserting an embedding
        test_content = "Sample legal document for testing"
        embedding = test_embeddings()
        
        if embedding:
            # Convert to PostgreSQL array format
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            cur.execute("""
                INSERT INTO test_embeddings (content, embedding)
                VALUES (%s, %s::vector)
                RETURNING id
            """, (test_content, embedding_str))
            
            inserted_id = cur.fetchone()["id"]
            conn.commit()
            print(f"  ✅ Inserted test embedding with ID: {inserted_id}")
            
            # Test similarity search
            cur.execute("""
                SELECT id, content, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM test_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT 1
            """, (embedding_str, embedding_str))
            
            result = cur.fetchone()
            print(f"  ✅ Similarity search working (similarity: {result['similarity']:.4f})")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False

def test_legal_qa():
    """Test legal question answering with gemma3-legal"""
    print("\n⚖️ Testing Legal QA with gemma3-legal...")
    
    try:
        # Simple legal question
        prompt = "What is consideration in contract law? Answer briefly."
        
        print(f"  📝 Question: {prompt}")
        print("  ⏳ Generating response...")
        
        start_time = time.time()
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "gemma3-legal:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100
                }
            },
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            answer = response.json()["response"]
            print(f"  ✅ Response ({elapsed:.2f}s):")
            print(f"     {answer[:200]}...")
            return True
        else:
            print(f"  ❌ Failed to generate response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error in legal QA: {e}")
        return False

def test_rag_pipeline():
    """Test complete RAG pipeline"""
    print("\n🔄 Testing Complete RAG Pipeline...")
    
    try:
        # Sample legal documents
        documents = [
            "The contract requires payment within 30 days of invoice.",
            "Force majeure clauses protect parties from unforeseen events.",
            "Confidentiality agreements must specify the duration of obligations."
        ]
        
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Ingest documents
        print("  📥 Ingesting documents...")
        for doc in documents:
            # Generate embedding
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": "nomic-embed-text:latest", "prompt": doc}
            )
            embedding = response.json()["embedding"]
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            cur.execute("""
                INSERT INTO test_embeddings (content, embedding)
                VALUES (%s, %s::vector)
            """, (doc, embedding_str))
        
        conn.commit()
        print(f"  ✅ Ingested {len(documents)} documents")
        
        # Test retrieval
        query = "What are the payment terms?"
        print(f"  🔍 Query: {query}")
        
        # Generate query embedding
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text:latest", "prompt": query}
        )
        query_embedding = response.json()["embedding"]
        query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        # Similarity search
        cur.execute("""
            SELECT content, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM test_embeddings
            WHERE 1 - (embedding <=> %s::vector) > 0.3
            ORDER BY embedding <=> %s::vector
            LIMIT 3
        """, (query_embedding_str, query_embedding_str, query_embedding_str))
        
        results = cur.fetchall()
        print(f"  ✅ Retrieved {len(results)} relevant documents")
        
        if results:
            # Use context for answer generation
            context = "\n".join([r["content"] for r in results])
            
            prompt = f"""Based on the following information, answer the question.
            
Context:
{context}

Question: {query}

Answer:"""
            
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "gemma3-legal:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 50}
                },
                timeout=30
            )
            
            answer = response.json()["response"]
            print(f"  ✅ Generated answer: {answer[:100]}...")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ RAG pipeline error: {e}")
        return False

def check_services():
    """Check status of all services"""
    print("\n🌐 Checking Service Status...")
    
    services = [
        ("Ollama", "http://localhost:11434/api/tags"),
        ("GPU Orchestrator", "http://localhost:8095/api/status"),
        ("Enhanced RAG", "http://localhost:8094/health"),
        ("Frontend", "http://localhost:5173"),
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"  ✅ {name} - Running")
            else:
                print(f"  ⚠️ {name} - Responding but status {response.status_code}")
        except:
            print(f"  ❌ {name} - Not available")

# Run all tests
print("\n🧪 Starting Tests...")
print("-" * 50)

all_passed = True

# Test 1: Ollama models
if not test_ollama_models():
    all_passed = False

# Test 2: PostgreSQL + pgvector
if not test_pgvector():
    all_passed = False

# Test 3: Legal QA
if not test_legal_qa():
    all_passed = False

# Test 4: RAG Pipeline
if not test_rag_pipeline():
    all_passed = False

# Check services
check_services()

# Summary
print("\n" + "=" * 50)
print("📊 TEST SUMMARY")
print("=" * 50)

if all_passed:
    print("✅ All tests PASSED!")
    print("🎉 Your Legal RAG system is fully operational!")
else:
    print("⚠️ Some tests failed. Please check the errors above.")

print("\n📝 Next Steps:")
print("1. Install LangChain: npm install langchain @langchain/community")
print("2. Run the TypeScript integration: npx tsx quick-setup-legal-rag.ts")
print("3. Start your frontend: npm run dev")
print("4. Access at: http://localhost:5173")
