# AI Synthesis System - Complete Integration Guide

## 🚀 Overview

This is a production-ready AI synthesis system with advanced features for legal document processing, including:

- **Multi-strategy retrieval** with RAG, MMR diversification, and cross-encoder reranking
- **Real-time streaming** with Server-Sent Events (SSE)
- **Intelligent caching** with Redis/LRU fallback
- **Machine learning feedback loop** for continuous improvement
- **Comprehensive monitoring** with metrics and alerts
- **Local LLM support** via Ollama integration
- **LegalBERT middleware** for domain-specific understanding

## 📁 Project Structure

```
src/
├── lib/
│   ├── server/
│   │   ├── ai/
│   │   │   ├── ai-assistant-input-synthesizer.ts  # Main orchestrator
│   │   │   ├── legalbert-middleware.ts           # Legal domain analysis
│   │   │   ├── caching-layer.ts                  # Multi-tier caching
│   │   │   ├── feedback-loop.ts                  # ML feedback system
│   │   │   ├── monitoring-service.ts             # Observability
│   │   │   ├── streaming-service.ts              # Real-time updates
│   │   │   └── ollama-local-llm.ts              # Local LLM inference
│   │   └── sse.ts                               # SSE helper
│   └── components/
│       └── ai-synthesis-client.svelte            # Frontend client
└── routes/
    └── api/
        └── ai-synthesizer/
            ├── +server.ts                        # Main API endpoint
            ├── stream/
            │   └── [streamId]/
            │       └── +server.ts                # SSE streaming endpoint
            └── test/
                └── +server.ts                    # Integration tests
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
npm install ioredis lru-cache
```

### 2. Set Up Redis (Optional but Recommended)

```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu/Debian
brew install redis                  # macOS

# Start Redis
redis-server

# Set environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### 3. Install Ollama for Local LLM (Optional)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve local gemma3:legal-latest
```

### 4. Environment Variables

Create a `.env` file:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
OLLAMA_URL=http://localhost:11434
```

## 🎯 Quick Start

### Basic Usage

```typescript
import { aiAssistantSynthesizer } from '$lib/server/ai/ai-assistant-input-synthesizer';

// Simple query
const result = await aiAssistantSynthesizer.synthesizeInput({
  query: "What are the key elements of a valid contract?",
  context: {
    userId: "user123",
    caseId: "case456"
  },
  options: {
    enableMMR: true,
    enableCrossEncoder: true,
    maxSources: 10
  }
});

console.log(result.processedQuery);    // Enhanced query with legal concepts
console.log(result.retrievedContext);  // Ranked and diversified sources
console.log(result.enhancedPrompt);    // Ready-to-use AI prompt
console.log(result.metadata);          // Quality scores and metrics
```

### Streaming Example

```typescript
// Client-side code
const response = await fetch('/api/ai-synthesizer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "Analyze this contract for potential issues",
    stream: true  // Enable streaming
  })
});

const { streamId } = await response.json();

// Connect to SSE stream
const eventSource = new EventSource(`/api/ai-synthesizer/stream/${streamId}`);

eventSource.addEventListener('progress', (event) => {
  const data = JSON.parse(event.data);
  console.log(`${data.stage}: ${data.progress}%`);
});

eventSource.addEventListener('source', (event) => {
  const source = JSON.parse(event.data);
  console.log('Found source:', source.title);
});

eventSource.addEventListener('complete', (event) => {
  const result = JSON.parse(event.data);
  console.log('Synthesis complete:', result);
  eventSource.close();
});
```

### Local LLM with Ollama

```typescript
import { ollamaLLM } from '$lib/server/ai/ollama-local-llm';

// Process legal document locally
const analysis = await ollamaLLM.processLegalDocument(
  documentText,
  'analyze',  // or 'summarize', 'extract', 'classify'
  { format: 'json' }
);

// Generate with streaming
await ollamaLLM.generateStream(
  {
    model: 'gemma3:legal-latest',
    prompt: 'Explain the doctrine of consideration',
    options: { temperature: 0.3 }
  },
  (token) => console.log(token),        // On each token
  (response) => console.log(response)   // On complete
);
```

## 📊 API Endpoints

### POST `/api/ai-synthesizer`
Main synthesis endpoint with optional streaming.

```typescript
{
  query: string;
  context?: {
    caseId?: string;
    userId: string;
    conversationHistory?: Array<{role, content, timestamp}>;
    documents?: Array<{id, title, content, type}>;
    preferences?: {
      responseStyle: 'formal' | 'casual' | 'technical';
      maxLength: number;
      includeCitations: boolean;
      focusAreas: string[];
    };
  };
  options?: {
    enableMMR: boolean;
    enableCrossEncoder: boolean;
    enableLegalBERT: boolean;
    enableRAG: boolean;
    maxSources: number;
    similarityThreshold: number;
    diversityLambda: number;
    bypassCache: boolean;
  };
  stream?: boolean;
  feedbackData?: {
    requestId: string;
    rating: number;
    feedback?: string;
  };
}
```

### GET `/api/ai-synthesizer`
Health check endpoint.

### GET `/api/ai-synthesizer/stream/[streamId]`
Server-Sent Events endpoint for real-time updates.

### GET `/api/ai-synthesizer/test`
Run integration tests.

### POST `/api/ai-synthesizer/test`
Manual testing with custom queries.

## 🔥 Advanced Features

### 1. Caching Strategy

The system uses a multi-tier caching approach:
- **Hot Cache**: Ultra-fast in-memory for frequently accessed items
- **LRU Cache**: Fast local cache with automatic eviction
- **Redis**: Distributed cache for scaling across instances

```typescript
import { cachingLayer } from '$lib/server/ai/caching-layer';

// Warm up cache
await cachingLayer.warmUp([
  { key: 'common_query_1', value: result1 },
  { key: 'common_query_2', value: result2 }
]);

// Invalidate by tags
await cachingLayer.invalidateByTags(['user:123', 'case:456']);
```

### 2. Feedback Loop

The system learns from user interactions:

```typescript
import { feedbackLoop } from '$lib/server/ai/feedback-loop';

// Submit feedback
await feedbackLoop.processFeedback({
  requestId: 'req_123',
  userId: 'user_456',
  rating: 5,
  feedback: 'Very helpful and accurate'
});

// Get personalized recommendations
const recommendations = await feedbackLoop.getPersonalizedRecommendations('user_456');
```

### 3. Monitoring & Metrics

```typescript
import { monitoringService } from '$lib/server/ai/monitoring-service';

// Register custom health check
monitoringService.registerHealthCheck('database', async () => {
  return await checkDatabaseConnection();
});

// Get Prometheus metrics
const metrics = monitoringService.exportPrometheusMetrics();

// Subscribe to alerts
monitoringService.on('alert', (alert) => {
  if (alert.severity === 'critical') {
    sendNotification(alert);
  }
});
```

### 4. Frontend Integration

Use the provided Svelte component:

```svelte
<script>
  import AISynthesisClient from '$lib/components/ai-synthesis-client.svelte';
</script>

<AISynthesisClient
  query={userQuery}
  caseId={currentCase}
  userId={currentUser}
  on:complete={handleResult}
  on:error={handleError}
/>
```

## 🧪 Testing

Run integration tests:

```bash
# Test all components
curl http://localhost:5173/api/ai-synthesizer/test

# Test with custom query
curl -X POST http://localhost:5173/api/ai-synthesizer/test \
  -H "Content-Type: application/json" \
  -d '{"query": "What is negligence in tort law?"}'
```

## 📈 Performance Optimization

1. **Enable Redis** for distributed caching
2. **Install Ollama** for local LLM inference (reduces API costs)
3. **Use streaming** for better user experience
4. **Configure MMR lambda** (0.3-0.7) based on diversity needs
5. **Adjust cache TTL** based on data freshness requirements
6. **Monitor P95 latency** and scale resources accordingly

## 🔍 Troubleshooting

### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping

# Check connection
redis-cli -h localhost -p 6379
```

### Ollama Not Available
```bash
# Check Ollama service
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### High Latency
- Check cache hit rate (should be >30%)
- Reduce `maxSources` if too high
- Enable `bypassCache: false`
- Check network connectivity to services

## 📝 Architecture Decisions

1. **Multi-tier Caching**: Balances speed, cost, and scalability
2. **Streaming First**: Better UX for long-running operations
3. **Fallback Strategy**: Graceful degradation when services unavailable
4. **Modular Design**: Each component can be used independently
5. **Observability Built-in**: Monitoring and alerts from day one

## 🚦 Production Checklist

- [ ] Redis configured and connected
- [ ] Ollama installed with legal models
- [ ] Environment variables set
- [ ] SSL/TLS configured for streaming endpoints
- [ ] Monitoring dashboards set up
- [ ] Backup strategy for feedback data
- [ ] Rate limiting configured
- [ ] Error alerting enabled
- [ ] Cache warming implemented
- [ ] Load testing completed

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [SSE Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [LegalBERT Paper](https://arxiv.org/abs/2010.02559)

## 🤝 Contributing

To add new features:

1. Extend the synthesizer with new strategies
2. Add new monitoring metrics
3. Implement additional caching layers
4. Create new Ollama model variants
5. Enhance the feedback loop algorithms

## 📄 License

This system is designed for production use in legal tech applications.
