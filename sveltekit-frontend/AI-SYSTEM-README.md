# AI System Configuration - gemma3:legal-latest with legal-bert fallback

## 🚀 Overview

This high-performance AI assistant system is configured with a **focused two-model legal fallback chain**:

- **Primary**: `gemma3:legal-latest` - Specialized legal AI model
- **Fallback**: `legal-bert` - Legal-specific BERT model
- **No general models** - Fully focused on legal document processing

## 📋 System Requirements

### Required Services
- **Ollama** - Local LLM runtime
- **PostgreSQL** - Primary database with pgvector
- **Redis** - High-speed caching
- **Node.js 18+** - Runtime environment

### Optional Services (for full architecture)
- **Neo4j** - Knowledge graph database
- **RabbitMQ/NATS** - Message queuing
- **Qdrant** - Vector database (alternative to pgvector)

## 🎯 Model Configuration

### Fallback Chain
```yaml
Legal Tasks:
  1. gemma3:legal-latest (8192 tokens, GPU-accelerated)
  2. legal-bert (512 tokens, precision-focused)

Embeddings:
  1. nomic-embed-text (768 dimensions)
  2. bge-large-en (1024 dimensions)
```

### Key Changes
- ❌ **Removed**: `llama3.2` from fallback chain
- ✅ **Focus**: Both models specialized for legal text
- ✅ **Consistency**: Uniform legal expertise across fallbacks

## 🔧 Installation & Setup

### 1. Start Ollama Service
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### 2. Check Available Models
```bash
# List models
ollama list

# The system will use whichever legal model is available:
# - gemma3:legal-latest (primary)
# - legal-bert (fallback)
```

### 3. Start AI System
```bash
# Windows
START-AI-SYSTEM.bat

# Or manually start services
ollama serve
npm run dev
```

### 4. Test the System
```bash
# Run test script
node scripts/test-ai-system.mjs

# Test will show:
# - Available models
# - Fallback chain status
# - Model selection behavior
```

## 📁 File Structure

```
src/lib/server/ai/
├── ollama-config.ts      # Model configurations (no llama3.2)
├── ollama-service.ts     # Core Ollama integration
├── types.ts              # TypeScript definitions
└── ...

src/routes/api/ai/
├── generate/+server.ts   # Text generation endpoint
├── embeddings/+server.ts # Embedding generation
├── analyze/+server.ts    # Document analysis
└── query/+server.ts      # Query processing

Configuration files:
├── .env.ai               # AI environment variables
├── START-AI-SYSTEM.bat   # Startup script
├── AI-FALLBACK-SYSTEM.md # Fallback documentation
└── ...
```

## 🌐 API Endpoints

### Text Generation
```typescript
POST /api/ai/generate
{
  "prompt": "Your legal question here"
  // Model automatically selected based on availability
}

// Response includes:
{
  "response": "Generated text...",
  "model": "gemma3:legal-latest", // or "legal-bert"
  "fallback_used": false,
  "models_tried": ["gemma3:legal-latest"]
}
```

### Document Analysis
```typescript
POST /api/ai/analyze
{
  "content": "Full legal document text",
  "title": "Document Title",
  "type": "contract"
}

// System automatically uses best available legal model
```

## 🔥 Key Features

### 1. Intelligent Legal Model Selection
- Automatically detects legal content
- Selects appropriate model based on availability
- Seamless fallback if primary model unavailable

### 2. Legal Keyword Detection
The system recognizes legal tasks when prompts contain:
- Contract, agreement, legal, law, court
- Statute, regulation, compliance, liability
- Patent, trademark, copyright
- Tort, negligence, breach, damages
- Deed, title, evidence, testimony
- Prosecutor, defense, attorney, counsel

### 3. GPU Acceleration
- 35 layers offloaded to GPU
- Optimized for NVIDIA GPUs
- WebGPU support for client-side

### 4. Performance Features
- Request queuing (max 4 parallel)
- Smart caching with 1-hour TTL
- Model availability monitoring
- Stream response support

## 🛠️ Configuration

### Environment Variables (.env.ai)
```env
# Model Configuration
OLLAMA_MODEL=gemma3:legal-latest
OLLAMA_FALLBACK_MODEL=legal-bert
AI_FALLBACK_CHAIN=gemma3:legal-latest,legal-bert

# Performance
OLLAMA_GPU_LAYERS=35
PARALLEL_WORKERS=32
CACHE_ENABLED=true

# Legal Model Settings
LEGAL_MODEL_TEMPERATURE=0.5
LEGAL_MODEL_TOP_P=0.85
LEGAL_CONTEXT_WINDOW=8192
LEGAL_BERT_CONTEXT=512
VITE_WS_FANOUT_URL=ws://localhost:8080 # points to the ws-fanout-service broadcasting pipeline events.
```

## 📊 System Status

```javascript
// Get system status
const status = await ollamaService.getSystemStatus();
console.log(status);
/*
{
  primaryModel: 'gemma3:legal-latest' | 'not available',
  legalFallback: 'legal-bert' | 'not available',
  fallbackChain: {
    legal: ['gemma3:legal-latest', 'legal-bert'],
    general: ['gemma3:legal-latest', 'legal-bert'],
    embedding: ['nomic-embed-text', 'bge-large-en']
  }
}
*/
```

## 🔍 Troubleshooting

### No Models Available
```bash
# Check Ollama
ollama list

# Ensure at least one legal model is available
# The system needs either:
# - gemma3:legal-latest
# - legal-bert
```

### Test Fallback Behavior
```javascript
// The system automatically handles fallback
const response = await ollamaService.generate("Legal query");
console.log(`Model used: ${response.model}`);
console.log(`Was fallback: ${response.fallback_used}`);
```

### Performance Issues
- Reduce GPU layers if memory constrained
- Check cache hit rate
- Monitor queue length

## 📝 Important Notes

1. **No llama3.2**: General-purpose model removed for focused legal processing
2. **Legal Focus**: Both models optimized for legal text analysis
3. **Automatic Fallback**: System seamlessly switches between models
4. **No Model Pulling**: Uses only locally available models
5. **GPU Optimized**: Configured for 35-layer GPU acceleration

## 🚦 Quick Commands

```bash
# Start everything
START-AI-SYSTEM.bat

# Test system
node scripts/test-ai-system.mjs

# Check models
ollama list

# Verify endpoints
curl http://localhost:5173/api/ai/generate
```

## 💡 Benefits of Simplified Chain

1. **Legal Expertise**: Both models trained on legal corpus
2. **Consistent Quality**: No degradation to general models
3. **Faster Selection**: Simpler fallback logic
4. **Better Accuracy**: Specialized models for legal tasks
5. **Reduced Complexity**: Easier to maintain and debug

## 📈 Performance Metrics

### Expected Performance
- **Generation Speed**: ~50-100 tokens/second (with GPU)
- **Fallback Latency**: <100ms model switch
- **Cache Hit Rate**: >60% for repeated queries
- **Queue Processing**: 4 parallel requests max

### Resource Usage
- **RAM**: 8-16GB recommended
- **VRAM**: 6-8GB for gemma3 models
- **CPU**: 8+ cores recommended
- **Storage**: 10-15GB for models

---

**Version**: 2.0.0
**Updated**: Removed llama3.2 from fallback chain
**Status**: Production Ready
**Focus**: Legal document processing only
