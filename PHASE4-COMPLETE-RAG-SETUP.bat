@echo off
setlocal enabledelayedexpansion

echo =========================================================
echo   PHASE 4: Legal AI RAG System - Complete Setup
echo   Local GGUF + Ollama + FastAPI + LangChain + pgvector
echo =========================================================
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%🚀 Phase 4: Data Management & Event Streaming Setup%NC%
echo %YELLOW%Components: Local GGUF, FastAPI, LangChain, pgvector, RabbitMQ, Neo4j%NC%
echo.

:: Check prerequisites
echo %BLUE%📋 Checking prerequisites...%NC%
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Docker not found%NC%
    pause
    exit /b 1
)

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Python not found - required for FastAPI backend%NC%
    echo %YELLOW%Installing Python is recommended for full RAG capabilities%NC%
) else (
    echo %GREEN%✅ Python available%NC%
)

:: Stop existing services
echo.
echo %BLUE%🛑 Stopping existing services...%NC%
docker-compose -f docker-compose-optimized.yml down >nul 2>&1

:: Create enhanced directory structure for Phase 4
echo %BLUE%📁 Creating Phase 4 directory structure...%NC%
if not exist "rag-backend" mkdir rag-backend
if not exist "rag-backend\api" mkdir rag-backend\api
if not exist "rag-backend\services" mkdir rag-backend\services
if not exist "rag-backend\models" mkdir rag-backend\models
if not exist "rag-backend\workers" mkdir rag-backend\workers
if not exist "local-models" mkdir local-models
if not exist "vector-store" mkdir vector-store
if not exist "event-logs" mkdir event-logs
echo %GREEN%✅ Directory structure created%NC%

:: Create FastAPI requirements
echo %BLUE%📦 Creating FastAPI backend requirements...%NC%
(
echo fastapi==0.104.1
echo uvicorn==0.24.0
echo langchain==0.1.0
echo langchain-community==0.0.10
echo psycopg2-binary==2.9.9
echo pgvector==0.2.4
echo redis==5.0.1
echo celery==5.3.4
echo pika==1.3.2
echo sentence-transformers==2.2.2
echo torch==2.1.1
echo transformers==4.36.2
echo llama-cpp-python==0.2.20
echo unsloth
echo pydantic==2.5.0
echo sqlalchemy==2.0.23
echo alembic==1.13.0
echo python-multipart==0.0.6
echo aiofiles==23.2.1
echo jinja2==3.1.2
echo neo4j==5.15.0
) > rag-backend\requirements.txt
echo %GREEN%✅ FastAPI requirements created%NC%

:: Create enhanced Docker Compose for Phase 4
echo %BLUE%🐳 Creating Phase 4 Docker Compose configuration...%NC%
(
echo # Phase 4: Complete Legal AI RAG System
echo # Local GGUF + Ollama + FastAPI + LangChain + pgvector + Event Streaming
echo version: '3.8'
echo.
echo services:
echo   # PostgreSQL with pgvector for RAG
echo   postgres:
echo     image: pgvector/pgvector:pg16
echo     container_name: legal-postgres
echo     environment:
echo       POSTGRES_DB: legal_rag_db
echo       POSTGRES_USER: legal_admin
echo       POSTGRES_PASSWORD: LegalRAG2024!
echo       POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
echo     ports:
echo       - "5432:5432"
echo     volumes:
echo       - postgres_data:/var/lib/postgresql/data
echo       - ./database/init-rag.sql:/docker-entrypoint-initdb.d/init-rag.sql
echo     healthcheck:
echo       test: ["CMD-SHELL", "pg_isready -U legal_admin -d legal_rag_db"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo     restart: unless-stopped
echo.    
echo   # Redis for caching and session management
echo   redis:
echo     image: redis:7-alpine
echo     container_name: legal-redis
echo     ports:
echo       - "6379:6379"
echo     command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
echo     volumes:
echo       - redis_data:/data
echo     healthcheck:
echo       test: ["CMD", "redis-cli", "ping"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo     restart: unless-stopped
echo.
echo   # RabbitMQ for event streaming
echo   rabbitmq:
echo     image: rabbitmq:3-management-alpine
echo     container_name: legal-rabbitmq
echo     ports:
echo       - "5672:5672"
echo       - "15672:15672"
echo     environment:
echo       RABBITMQ_DEFAULT_USER: legal_admin
echo       RABBITMQ_DEFAULT_PASS: LegalRAG2024!
echo       RABBITMQ_VM_MEMORY_HIGH_WATERMARK: 0.6
echo     volumes:
echo       - rabbitmq_data:/var/lib/rabbitmq
echo     healthcheck:
echo       test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo     restart: unless-stopped
echo.
echo   # Neo4j for knowledge graphs
echo   neo4j:
echo     image: neo4j:5.15-community
echo     container_name: legal-neo4j
echo     ports:
echo       - "7474:7474"
echo       - "7687:7687"
echo     environment:
echo       NEO4J_AUTH: neo4j/LegalRAG2024!
echo       NEO4J_PLUGINS: "[\"apoc\", \"graph-data-science\"]"
echo       NEO4J_dbms_security_procedures_unrestricted: "apoc.*,gds.*"
echo       NEO4J_dbms_memory_heap_initial__size: 1G
echo       NEO4J_dbms_memory_heap_max__size: 2G
echo     volumes:
echo       - neo4j_data:/data
echo       - neo4j_logs:/logs
echo     healthcheck:
echo       test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "LegalRAG2024!", "RETURN 1"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo     restart: unless-stopped
echo.
echo   # Qdrant for vector similarity search
echo   qdrant:
echo     image: qdrant/qdrant:v1.7.0
echo     container_name: legal-qdrant
echo     ports:
echo       - "6333:6333"
echo       - "6334:6334"
echo     volumes:
echo       - qdrant_data:/qdrant/storage
echo     environment:
echo       QDRANT__SERVICE__HTTP_PORT: 6333
echo       QDRANT__SERVICE__GRPC_PORT: 6334
echo       QDRANT__LOG_LEVEL: INFO
echo     healthcheck:
echo       test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo     restart: unless-stopped
echo.
echo   # Ollama with local GGUF model support
echo   ollama:
echo     image: ollama/ollama:latest
echo     container_name: legal-ollama
echo     ports:
echo       - "11434:11434"
echo     volumes:
echo       - ollama_data:/root/.ollama
echo       - ./local-models:/models
echo     environment:
echo       - OLLAMA_HOST=0.0.0.0
echo       - OLLAMA_ORIGINS=*
echo       - OLLAMA_MAX_LOADED_MODELS=2
echo       - OLLAMA_KEEP_ALIVE=10m
echo       - OLLAMA_NUM_PARALLEL=4
echo       - OLLAMA_FLASH_ATTENTION=1
echo       - CUDA_VISIBLE_DEVICES=0
echo     deploy:
echo       resources:
echo         limits:
echo           memory: 8G
echo           cpus: '6.0'
echo         reservations:
echo           memory: 6G
echo           cpus: '4.0'
echo         devices:
echo           - driver: nvidia
echo             count: all
echo             capabilities: [gpu]
echo     depends_on:
echo       postgres:
echo         condition: service_healthy
echo       redis:
echo         condition: service_healthy
echo     healthcheck:
echo       test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo     restart: unless-stopped
echo.
echo   # FastAPI RAG backend
echo   rag-api:
echo     build:
echo       context: ./rag-backend
echo       dockerfile: Dockerfile
echo     container_name: legal-rag-api
echo     ports:
echo       - "8000:8000"
echo     environment:
echo       - DATABASE_URL=postgresql://legal_admin:LegalRAG2024!@postgres:5432/legal_rag_db
echo       - REDIS_URL=redis://redis:6379
echo       - RABBITMQ_URL=amqp://legal_admin:LegalRAG2024!@rabbitmq:5672
echo       - NEO4J_URL=bolt://neo4j:7687
echo       - NEO4J_USER=neo4j
echo       - NEO4J_PASSWORD=LegalRAG2024!
echo       - QDRANT_URL=http://qdrant:6333
echo       - OLLAMA_URL=http://ollama:11434
echo       - ENVIRONMENT=production
echo     volumes:
echo       - ./local-models:/app/models
echo       - ./vector-store:/app/vector-store
echo       - ./event-logs:/app/logs
echo     depends_on:
echo       postgres:
echo         condition: service_healthy
echo       redis:
echo         condition: service_healthy
echo       rabbitmq:
echo         condition: service_healthy
echo       neo4j:
echo         condition: service_healthy
echo       qdrant:
echo         condition: service_healthy
echo       ollama:
echo         condition: service_healthy
echo     restart: unless-stopped
echo.
echo   # Celery worker for background processing
echo   celery-worker:
echo     build:
echo       context: ./rag-backend
echo       dockerfile: Dockerfile.worker
echo     container_name: legal-celery-worker
echo     environment:
echo       - DATABASE_URL=postgresql://legal_admin:LegalRAG2024!@postgres:5432/legal_rag_db
echo       - REDIS_URL=redis://redis:6379
echo       - RABBITMQ_URL=amqp://legal_admin:LegalRAG2024!@rabbitmq:5672
echo       - OLLAMA_URL=http://ollama:11434
echo     volumes:
echo       - ./local-models:/app/models
echo       - ./vector-store:/app/vector-store
echo       - ./event-logs:/app/logs
echo     depends_on:
echo       - rag-api
echo     restart: unless-stopped
echo.
echo volumes:
echo   postgres_data:
echo   redis_data:
echo   rabbitmq_data:
echo   neo4j_data:
echo   neo4j_logs:
echo   qdrant_data:
echo   ollama_data:
echo.
echo networks:
echo   default:
echo     name: legal-rag-network
) > docker-compose-phase4.yml
echo %GREEN%✅ Phase 4 Docker Compose created%NC%

:: Create database initialization script
echo %BLUE%🗄️  Creating RAG database initialization...%NC%
if not exist "database" mkdir database
(
echo -- Phase 4: Legal RAG Database Initialization
echo -- PostgreSQL with pgvector for legal document embeddings
echo.
echo -- Enable required extensions
echo CREATE EXTENSION IF NOT EXISTS vector;
echo CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
echo CREATE EXTENSION IF NOT EXISTS pg_trgm;
echo CREATE EXTENSION IF NOT EXISTS btree_gin;
echo.
echo -- Legal documents table with vector embeddings
echo CREATE TABLE IF NOT EXISTS legal_documents ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     title VARCHAR^(500^) NOT NULL,
echo     content TEXT NOT NULL,
echo     document_type VARCHAR^(100^) NOT NULL,
echo     case_id VARCHAR^(255^),
echo     file_path VARCHAR^(1000^),
echo     file_hash VARCHAR^(64^) UNIQUE,
echo     metadata JSONB,
echo     embedding vector^(384^),
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     indexed_at TIMESTAMP
echo ^);
echo.
echo -- Vector similarity index
echo CREATE INDEX IF NOT EXISTS legal_documents_embedding_idx 
echo ON legal_documents USING ivfflat ^(embedding vector_cosine_ops^) WITH ^(lists = 100^);
echo.
echo -- Full-text search index
echo CREATE INDEX IF NOT EXISTS legal_documents_content_idx 
echo ON legal_documents USING GIN ^(to_tsvector^('english', content^)^);
echo.
echo -- Legal cases with RAG support
echo CREATE TABLE IF NOT EXISTS legal_cases ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     case_number VARCHAR^(255^) UNIQUE NOT NULL,
echo     title VARCHAR^(500^) NOT NULL,
echo     description TEXT,
echo     status VARCHAR^(50^) DEFAULT 'active',
echo     priority VARCHAR^(50^) DEFAULT 'medium',
echo     jurisdiction VARCHAR^(200^),
echo     court_name VARCHAR^(300^),
echo     assigned_attorney VARCHAR^(255^),
echo     client_name VARCHAR^(255^),
echo     case_summary_embedding vector^(384^),
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- Legal entities extracted by AI
echo CREATE TABLE IF NOT EXISTS legal_entities ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     document_id UUID REFERENCES legal_documents^(id^) ON DELETE CASCADE,
echo     entity_type VARCHAR^(100^) NOT NULL,
echo     entity_value VARCHAR^(500^) NOT NULL,
echo     confidence DECIMAL^(3,2^),
echo     start_position INTEGER,
echo     end_position INTEGER,
echo     context TEXT,
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- RAG query history
echo CREATE TABLE IF NOT EXISTS rag_queries ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     user_id VARCHAR^(255^),
echo     query_text TEXT NOT NULL,
echo     query_embedding vector^(384^),
echo     response_text TEXT,
echo     documents_used JSONB,
echo     confidence_score DECIMAL^(3,2^),
echo     processing_time_ms INTEGER,
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- Document processing jobs
echo CREATE TABLE IF NOT EXISTS processing_jobs ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     job_type VARCHAR^(100^) NOT NULL,
echo     status VARCHAR^(50^) DEFAULT 'pending',
echo     document_id UUID REFERENCES legal_documents^(id^),
echo     parameters JSONB,
echo     result JSONB,
echo     error_message TEXT,
echo     started_at TIMESTAMP,
echo     completed_at TIMESTAMP,
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- Audit trail for legal compliance
echo CREATE TABLE IF NOT EXISTS audit_logs ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     user_id VARCHAR^(255^),
echo     action VARCHAR^(255^) NOT NULL,
echo     entity_type VARCHAR^(100^),
echo     entity_id UUID,
echo     ip_address INET,
echo     user_agent TEXT,
echo     details JSONB,
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- Sample legal cases
echo INSERT INTO legal_cases ^(case_number, title, description, status, priority^) VALUES
echo ^('RAG-2024-001', 'Contract Dispute - AI Technology Licensing', 'Complex AI technology licensing dispute involving patent rights', 'active', 'high'^),
echo ^('RAG-2024-002', 'Employment Law - Remote Work Compliance', 'Multi-state remote work compliance and labor law issues', 'active', 'medium'^),
echo ^('RAG-2024-003', 'Data Privacy - GDPR Compliance Audit', 'Comprehensive GDPR compliance review and remediation', 'pending', 'high'^)
echo ON CONFLICT ^(case_number^) DO NOTHING;
echo.
echo -- Create indexes for performance
echo CREATE INDEX IF NOT EXISTS idx_legal_documents_case_id ON legal_documents^(case_id^);
echo CREATE INDEX IF NOT EXISTS idx_legal_documents_type ON legal_documents^(document_type^);
echo CREATE INDEX IF NOT EXISTS idx_legal_entities_document_id ON legal_entities^(document_id^);
echo CREATE INDEX IF NOT EXISTS idx_legal_entities_type ON legal_entities^(entity_type^);
echo CREATE INDEX IF NOT EXISTS idx_rag_queries_user_id ON rag_queries^(user_id^);
echo CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs^(status^);
echo CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs^(user_id^);
echo CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs^(created_at^);
) > database\init-rag.sql
echo %GREEN%✅ RAG database initialization created%NC%

:: Start Phase 4 services
echo.
echo %BLUE%🚀 Starting Phase 4 RAG services...%NC%
docker-compose -f docker-compose-phase4.yml up -d postgres redis rabbitmq neo4j qdrant
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to start infrastructure services%NC%
    goto :error_exit
)

:: Wait for services
echo %BLUE%⏳ Waiting for services to be ready...%NC%
timeout /t 10 >nul

:: Start Ollama with local model support
echo %BLUE%🤖 Starting Ollama with local GGUF support...%NC%
docker-compose -f docker-compose-phase4.yml up -d ollama
timeout /t 15 >nul

:: Check for local GGUF models
echo %BLUE%📥 Checking for local GGUF models...%NC%
if exist "local-models\*.gguf" (
    echo %GREEN%✅ Local GGUF models found%NC%
    echo %YELLOW%Models will be loaded via Ollama Modelfile%NC%
) else (
    echo %YELLOW%⚠️  No local GGUF models found in local-models directory%NC%
    echo %YELLOW%Place your Unsloth-trained GGUF models in local-models/%NC%
    echo %YELLOW%Falling back to pulling standard models...%NC%
    docker exec legal-ollama ollama pull gemma2:9b
)

:: Final status
echo.
echo %GREEN%🎉 Phase 4 Legal RAG System is running!%NC%
echo.
echo %BLUE%📊 Service Status:%NC%
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr -E "(legal-|NAMES)"

echo.
echo %BLUE%🔗 Service URLs:%NC%
echo %YELLOW%• PostgreSQL + pgvector: localhost:5432%NC%
echo %YELLOW%• Redis Cache: localhost:6379%NC%
echo %YELLOW%• RabbitMQ Management: http://localhost:15672%NC%
echo %YELLOW%• Neo4j Browser: http://localhost:7474%NC%
echo %YELLOW%• Qdrant API: http://localhost:6333%NC%
echo %YELLOW%• Ollama API: http://localhost:11434%NC%
echo %YELLOW%• FastAPI (when built): http://localhost:8000%NC%

echo.
echo %BLUE%📋 Next Steps:%NC%
echo %YELLOW%1. Build FastAPI backend: docker-compose -f docker-compose-phase4.yml build%NC%
echo %YELLOW%2. Start RAG API: docker-compose -f docker-compose-phase4.yml up -d rag-api%NC%
echo %YELLOW%3. Start Celery workers: docker-compose -f docker-compose-phase4.yml up -d celery-worker%NC%
echo %YELLOW%4. Place GGUF models in local-models/ directory%NC%
echo %YELLOW%5. Test RAG: curl http://localhost:8000/api/v1/rag/query%NC%

echo.
echo %GREEN%✨ Phase 4: Complete RAG system with local GGUF models ready!%NC%
pause
exit /b 0

:error_exit
echo.
echo %RED%❌ Phase 4 setup failed%NC%
echo %YELLOW%Check logs: docker-compose -f docker-compose-phase4.yml logs%NC%
pause
exit /b 1
