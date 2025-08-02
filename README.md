# Legal AI System - Prosecutor's Digital Assistant

A comprehensive AI-powered legal case management system designed for prosecutors, featuring advanced document analysis, case scoring, evidence synthesis, and vector-based legal research.

## 🎯 Key Features

- **AI-Powered Case Scoring** (0-100) with multi-criteria analysis
- **384-Dimensional Vector Search** using nomic-embed-text
- **Document Processing** with OCR and intelligent analysis
- **Evidence Synthesis** with timeline generation
- **Knowledge Graph** for legal relationships (Neo4j)
- **Real-time Chat** with specialized legal AI models
- **Automated Report Generation** with export capabilities

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (SvelteKit)                      │
│  - Bits UI v2 Components    - Real-time Updates            │
│  - TypeScript              - Responsive Design             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (Node.js)                       │
│  - RESTful Endpoints       - WebSocket Support             │
│  - Authentication         - Rate Limiting                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌──────────────┬──────────────┬──────────────┬───────────────┐
│  PostgreSQL  │    Redis     │   Qdrant     │    Ollama     │
│  + pgvector  │   Cache &    │   Vector     │      AI       │
│   Database   │   Sessions   │   Search     │    Models     │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 with PowerShell 7+
- Docker Desktop
- Node.js 18+
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU (optional, for faster AI processing)

### Installation

```powershell
# Clone the repository
git clone [repository-url]
cd legal-ai-system

# Run the automated installer
.\install.ps1

# Start the development server
cd sveltekit-frontend && npm run dev
```

Access the application at: http://localhost:5173

### Verify Installation

```powershell
# Check system health
.\health-check.ps1

# Run API tests
.\test-api.ps1 -TestAll
```

## 📋 Core Components

### 1. Case Management
- Create, update, and track criminal cases
- AI-powered case scoring (0-100)
- Automated risk assessment
- Resource allocation recommendations

### 2. Document Processing
- Vector embeddings (384-dim) for all documents
- Semantic search across case files
- OCR for scanned documents
- Automatic categorization and tagging

### 3. Evidence Analysis
- Evidence chain tracking
- Automated synthesis reports
- Timeline generation
- Relationship mapping

### 4. AI Services
- **Embedding Model**: nomic-embed-text (384 dimensions)
- **LLM Model**: gemma3-legal (custom fine-tuned)
- **Vector Database**: Qdrant with 3 collections
- **Knowledge Graph**: Neo4j for legal relationships

## 🛠️ Development

### Project Structure
```
legal-ai-system/
├── sveltekit-frontend/       # Frontend application
│   ├── src/
│   │   ├── lib/
│   │   │   ├── server/      # Server-side services
│   │   │   └── components/  # UI components
│   │   └── routes/          # Page routes
│   └── package.json
├── database/                 # Database schemas and migrations
├── local-models/            # Custom AI model configurations
├── tests/                   # Test suites
├── docker-compose.yml       # Infrastructure definition
└── *.ps1                    # PowerShell scripts
```

### Key Services

#### QdrantService
- Manages vector storage and search
- Fixed 384-dimensional vectors
- Supports batch operations
- Collection optimization

#### CaseScoringService
- AI-powered case analysis
- Multi-criteria scoring
- Temperature-controlled generation
- Historical score tracking

#### OllamaService
- LLM integration
- Embedding generation
- Custom model support
- Streaming responses

### API Endpoints

```typescript
POST   /api/case-scoring      # Score a case (0-100)
POST   /api/documents/embed   # Generate embeddings
POST   /api/documents/search  # Vector similarity search
POST   /api/ai/chat          # AI chat completions
POST   /api/evidence/synthesize # Synthesize evidence
GET    /api/cases            # List cases
POST   /api/cases            # Create case
PATCH  /api/cases/:id        # Update case
```

## 🔧 Configuration

### Environment Variables
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/prosecutor_db
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
OLLAMA_URL=http://localhost:11434
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=legal-ai-2024
```

### Vector Configuration
- **Model**: nomic-embed-text
- **Dimensions**: 384
- **Distance Metric**: Cosine
- **Index Type**: HNSW (m=16, ef=200)

## 📊 Performance

### Benchmarks
- Document embedding: ~100ms per document
- Vector search: <50ms for 10k documents
- Case scoring: 2-5 seconds
- Evidence synthesis: 5-10 seconds

### Optimization
```powershell
# Run system optimization
.\update.ps1 -Optimize

# Update all components
.\update.ps1 -UpdateAll -Backup
```

## 🧪 Testing

### Run Tests
```powershell
# Unit tests
npm test

# Integration tests
node tests/integration.test.js

# API tests
.\test-api.ps1 -TestAll

# Service tests
node tests/services.test.js
```

### Health Monitoring
```powershell
# Quick health check
.\health-check.ps1

# Detailed diagnostics
.\health-check.ps1 -Detailed

# Auto-fix issues
.\health-check.ps1 -Fix
```

## 🚨 Troubleshooting

### Common Issues

1. **Vector Dimension Mismatch**
   ```powershell
   # Recreate collections with correct dimensions
   docker-compose down qdrant
   docker-compose up -d qdrant
   ```

2. **Model Not Found**
   ```powershell
   # Pull required models
   ollama pull nomic-embed-text
   ollama pull gemma:2b
   ```

3. **Database Connection Failed**
   ```powershell
   # Reset database
   docker-compose down postgres
   docker-compose up -d postgres
   npm run db:push
   ```

### Complete Reset
```powershell
# Nuclear option - complete cleanup
docker-compose down -v
docker system prune -a
.\install.ps1
```

## 📚 Documentation

- [API Documentation](./docs/api.md)
- [Architecture Guide](./docs/architecture.md)
- [Deployment Guide](./docs/deployment.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

[License Type] - See LICENSE file for details

## 🙏 Acknowledgments

- Anthropic for Claude AI assistance
- Qdrant team for vector database
- Ollama for local LLM support
- SvelteKit community

---

Built with ❤️ for the legal community
