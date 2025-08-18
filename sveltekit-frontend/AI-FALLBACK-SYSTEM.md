# AI System Configuration - Intelligent Fallback

## 🎯 Updated Model Configuration

The AI system now uses a streamlined **two-model fallback chain** focused on legal document processing:

### Primary Model
```
gemma3:legal-latest
- Type: Local specialized legal AI
- Context: 8192 tokens
- Purpose: Legal document analysis, contract review, case law research
- GPU Layers: 35
```

### Fallback Model
```
legal-bert
- Type: Legal-specific BERT model
- Context: 512 tokens
- Purpose: Legal text understanding, terminology identification
- Optimized for: Precision in legal analysis
```

## 🔄 Intelligent Fallback Chain

### Legal Analysis Tasks
```
1. gemma3:legal-latest (Primary)
   ↓ (if unavailable)
2. legal-bert (Fallback)
```

### Text Generation Tasks
```
1. gemma3:legal-latest (Primary)
   ↓ (if unavailable)
2. legal-bert (Fallback)
```

### Embedding Generation
```
1. nomic-embed-text (Primary)
   ↓ (if unavailable)
2. bge-large-en (Fallback)
```

## ⚡ Key Features

### Smart Model Selection
- **Automatic Legal Detection**: System analyzes prompts for legal keywords
- **Context-Aware Selection**: Chooses appropriate model based on task type
- **Graceful Degradation**: Seamlessly falls back if primary model unavailable

### Legal Keyword Detection
The system automatically detects legal tasks when prompts contain:
- Contract, agreement, legal, law, court, case
- Statute, regulation, compliance, liability, clause
- Jurisdiction, plaintiff, defendant, litigation
- Patent, trademark, copyright, intellectual property
- Tort, negligence, breach, damages, remedy
- Arbitration, mediation, settlement, precedent
- Deed, title, evidence, testimony, witness
- Prosecutor, defense, attorney, counsel, judge

### Performance Optimizations
- **GPU Acceleration**: 35 layers offloaded to GPU
- **Request Queuing**: Max 4 parallel requests
- **Smart Caching**: 1-hour TTL with automatic cleanup
- **Stream Support**: Real-time response streaming
- **Model Monitoring**: Checks available models every 30 seconds

## 📊 System Status API

```javascript
// Get complete system status
const status = await ollamaService.getSystemStatus();
/*
Returns:
{
  ollamaAvailable: boolean,
  availableModels: string[],
  primaryModel: 'gemma3:legal-latest' | 'not available',
  legalFallback: 'legal-bert' | 'not available',
  cacheSize: number,
  queueLength: number,
  activeRequests: number,
  fallbackChain: {
    legal: ['gemma3:legal-latest', 'legal-bert'],
    general: ['gemma3:legal-latest', 'legal-bert'],
    embedding: ['nomic-embed-text', 'bge-large-en']
  }
}
*/
```

## 🚀 Quick Start

### 1. Check Available Models
```bash
ollama list
```

### 2. Ensure Models are Available
```bash
# If you have the models locally, they'll be used automatically
# The system will show which models are available
```

### 3. Start the System
```bash
START-AI-SYSTEM.bat
```

### 4. Test Fallback Chain
```javascript
// The system automatically handles fallback
const response = await ollamaService.generate(
  "What are the key elements of a valid contract?",
  { 
    // No need to specify model - system selects automatically
  }
);

console.log(`Model used: ${response.model}`);
console.log(`Fallback used: ${response.fallback_used}`);
console.log(`Models tried: ${response.models_tried}`);
```

## 🔍 Fallback Behavior

### When Primary Model Fails
1. System attempts `gemma3:legal-latest`
2. If unavailable/fails, automatically tries `legal-bert`
3. Response includes metadata about which model was used

### Response Metadata
Every response includes:
- `model`: The model that successfully generated the response
- `fallback_used`: Boolean indicating if fallback was used
- `models_tried`: Array of models attempted in order

## 📝 Example Usage

### Legal Document Analysis
```javascript
const document = {
  id: 'doc-001',
  title: 'Service Agreement',
  type: 'contract',
  content: 'Full contract text...'
};

// System automatically selects best available legal model
const analysis = await ollamaService.analyzeLegalDocument(document);
console.log(`Analysis by: ${analysis.metadata.modelUsed}`);
```

### Query Processing
```javascript
const query = {
  query: "What are the breach of contract remedies?",
  // ... other fields
};

// System detects legal query and uses appropriate model
const response = await ollamaService.processQuery(query, relevantDocs);
```

## 🛡️ Error Handling

The system provides robust error handling:
- Automatic retry with fallback models
- Detailed error messages when all models fail
- Cache preservation during failures
- Queue management for overload protection

## 📊 Monitoring

### Event Emissions
The service emits events for monitoring:
- `model-selected`: When a model is chosen
- `request-start`: Request initiated
- `request-complete`: Request successful
- `request-error`: Request failed
- `cache-hit`: Cache used
- `models-updated`: Available models changed

### Example Event Listener
```javascript
ollamaService.on('model-selected', (data) => {
  console.log(`Selected ${data.selected} for ${data.task}`);
});

ollamaService.on('request-complete', (data) => {
  if (data.fallback) {
    console.log(`Fallback model ${data.model} was used`);
  }
});
```

## ⚙️ Configuration

### Update Environment Variables
```env
# .env.ai
OLLAMA_MODEL=gemma3:legal-latest
OLLAMA_FALLBACK_MODEL=legal-bert
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_GPU_LAYERS=35
```

## 🔧 Troubleshooting

### If Both Models Unavailable
```bash
# Check Ollama status
ollama list

# Ensure Ollama is running
ollama serve

# The system will report which models are missing
node scripts/test-ai-system.mjs
```

### Performance Issues
- Reduce GPU layers if memory constrained
- Adjust cache TTL for better hit rates
- Monitor queue length for bottlenecks

## 📈 Benefits of Simplified Fallback

1. **Focused on Legal**: Both models optimized for legal text
2. **Reduced Complexity**: Simpler fallback chain
3. **Better Performance**: Less overhead in model selection
4. **Consistent Quality**: Both models trained on legal corpus
5. **Predictable Behavior**: Clear primary/fallback relationship

---

**Configuration Updated**: llama3.2 removed from fallback chain
**Current Chain**: gemma3:legal-latest → legal-bert
**Status**: Ready for legal document processing
