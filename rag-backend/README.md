# Enhanced RAG Backend

Production-ready Retrieval Augmented Generation (RAG) backend with PostgreSQL + pgvector, Ollama local LLM integration, multi-agent orchestration, and comprehensive document processing capabilities.

## 🚀 Features

### Core RAG Capabilities
- **Vector Search**: Semantic document search with pgvector and Ollama embeddings
- **Document Processing**: Support for PDF, DOCX, TXT, HTML, and image files with OCR
- **Multi-Modal Search**: Hybrid search combining vector similarity and text matching
- **Real-time Indexing**: Automatic document chunking and embedding generation

### AI & Agent Orchestration
- **Local LLM Integration**: Ollama support with gemma2:9b and nomic-embed-text models
- **Multi-Agent Workflows**: Document analysis, legal research, case preparation, contract review
- **Streaming Responses**: Real-time AI chat and analysis capabilities
- **Intelligent Caching**: Redis-based caching for embeddings and AI responses

### Production Features
- **PostgreSQL + pgvector**: Scalable vector database with optimized indexes
- **Health Monitoring**: Comprehensive health checks and system metrics
- **Rate Limiting**: API protection with configurable limits
- **WebSocket Support**: Real-time updates for collaborative workflows
- **Docker Ready**: Complete containerization with multi-service orchestration

## 📋 Prerequisites

- Node.js 18+ 
- PostgreSQL 14+ with pgvector extension
- Redis 6+
- Ollama with required models
- Docker & Docker Compose (optional)

## 🔧 Installation

### 1. Clone and Install Dependencies

```bash
cd rag-backend
npm install
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Database Setup

```bash
# Install PostgreSQL and pgvector
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE EXTENSION vector;"

# Create database
createdb deeds_web_db
```

### 4. Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma2:9b
ollama pull nomic-embed-text
```

### 5. Redis Setup

```bash
# Install Redis
sudo apt install redis-server
sudo systemctl start redis-server
```

## 🚀 Quick Start

### Development Mode

```bash
npm run dev
```

### Production Mode

```bash
npm run build
npm start
```

### Docker Deployment

```bash
docker-compose up -d
```

## 📚 API Documentation

### Core Endpoints

#### Search & Retrieval
```bash
# Semantic search
POST /api/v1/rag/search
{
  "query": "contract liability clauses",
  "caseId": "case-123",
  "limit": 10,
  "searchType": "hybrid"
}

# Document upload
POST /api/v1/rag/upload
Content-Type: multipart/form-data
- document: file
- title: "Contract Agreement"
- documentType: "contract"
- caseId: "case-123"
```

#### AI Analysis
```bash
# Text analysis
POST /api/v1/rag/analyze
{
  "text": "Document content...",
  "analysisType": "contract",
  "options": { "maxTokens": 1024 }
}

# Document summarization
POST /api/v1/rag/summarize
{
  "text": "Long document content...",
  "length": "medium"
}
```

#### Multi-Agent Workflows
```bash
# Document analysis workflow
POST /api/v1/agents/workflow
{
  "workflowType": "document_analysis",
  "input": {
    "content": "Document text...",
    "title": "Contract",
    "documentType": "contract"
  }
}

# Legal research workflow
POST /api/v1/agents/workflow
{
  "workflowType": "legal_research",
  "input": {
    "text": "Research query",
    "context": "Contract law",
    "jurisdiction": "federal"
  }
}
```

#### Document Management
```bash
# List documents
GET /api/v1/documents?caseId=case-123&limit=20

# Get document details
GET /api/v1/documents/:id?includeContent=true&includeChunks=true

# Update document
PUT /api/v1/documents/:id
{
  "title": "Updated Title",
  "documentType": "evidence"
}

# Delete document
DELETE /api/v1/documents/:id
```

#### Health & Monitoring
```bash
# Basic health check
GET /health

# Detailed system metrics
GET /health/detailed

# Component-specific health
GET /health/database
GET /health/ollama
GET /health/cache

# System statistics
GET /api/v1/rag/stats
```

## 🏗️ Architecture

### Service Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Express.js    │    │   PostgreSQL    │    │     Redis       │
│   API Server    │◄──►│   + pgvector    │    │     Cache       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Ollama      │    │   Document      │    │     Agent       │
│   Local LLM     │    │   Processor     │    │ Orchestrator    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Document Upload** → Processing → Chunking → Embedding → Storage
2. **Search Query** → Embedding → Vector Search → Ranking → Response
3. **AI Analysis** → Agent Selection → LLM Processing → Caching → Response

### Key Components
- **DatabaseService**: PostgreSQL + pgvector operations
- **VectorService**: Embedding generation and similarity search
- **CacheService**: Redis-based intelligent caching
- **OllamaService**: Local LLM integration and management
- **DocumentProcessor**: Multi-format document processing with OCR
- **AgentOrchestrator**: Multi-agent workflow coordination
- **HealthService**: System monitoring and diagnostics

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `OLLAMA_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default LLM model | `gemma2:9b` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |

### Docker Configuration

```yaml
services:
  rag-backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/deeds_web_db
      - REDIS_URL=redis://redis:6379
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - postgres
      - redis
      - ollama

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_DB=deeds_web_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_models:/root/.ollama
```

## 📊 Performance

### Benchmarks
- **Document Processing**: 50-100 docs/minute (depending on size and type)
- **Vector Search**: <100ms for 10K+ documents
- **AI Analysis**: 2-5 seconds per document
- **Concurrent Requests**: 100+ simultaneous requests supported

### Optimization Features
- **Intelligent Caching**: Multi-layer caching for embeddings, search results, and AI responses
- **Database Indexing**: Optimized pgvector indexes for fast similarity search
- **Connection Pooling**: Efficient database connection management
- **Parallel Processing**: Multi-core document processing capabilities

## 🔍 Monitoring

### Health Endpoints
- `/health` - Basic service health
- `/health/detailed` - Comprehensive system metrics
- `/health/metrics` - Prometheus-compatible metrics
- `/health/components` - Individual component status

### Key Metrics
- Document processing rates
- Search response times
- AI model performance
- Cache hit rates
- Database query performance
- System resource usage

### Logging
- Structured JSON logging with Winston
- Configurable log levels
- Error tracking and alerting
- Performance metrics logging

## 🧪 Testing

### Run Tests
```bash
# Unit tests
npm test

# Integration tests
npm run test:integration

# Load tests
npm run test:load
```

### API Testing
```bash
# Test search functionality
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test search", "limit": 5}'

# Test document upload
curl -X POST http://localhost:8000/api/v1/rag/upload \
  -F "document=@test.pdf" \
  -F "title=Test Document" \
  -F "documentType=general"

# Test health
curl http://localhost:8000/health
```

## 🚀 Deployment

### Production Checklist
- [ ] Configure environment variables
- [ ] Set up PostgreSQL with pgvector
- [ ] Configure Redis for caching
- [ ] Install and configure Ollama with models
- [ ] Set up reverse proxy (nginx/Apache)
- [ ] Configure SSL certificates
- [ ] Set up monitoring and logging
- [ ] Configure backup strategies
- [ ] Test disaster recovery procedures

### Docker Production
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale rag-backend=3

# Monitor logs
docker-compose logs -f rag-backend
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-backend
  template:
    metadata:
      labels:
        app: rag-backend
    spec:
      containers:
      - name: rag-backend
        image: rag-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: database-url
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full API Documentation](docs/api.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/rag-backend/issues)
- **Discord**: [Community Discord](https://discord.gg/your-server)

## 🗺️ Roadmap

### Phase 1 (Current)
- [x] Core RAG functionality
- [x] Multi-agent orchestration
- [x] Document processing pipeline
- [x] Health monitoring

### Phase 2 (Next)
- [ ] Advanced analytics dashboard
- [ ] Custom model fine-tuning
- [ ] Enterprise SSO integration
- [ ] Advanced workflow automation

### Phase 3 (Future)
- [ ] Multi-tenant architecture
- [ ] Advanced AI model marketplace
- [ ] Federated search capabilities
- [ ] Advanced compliance features