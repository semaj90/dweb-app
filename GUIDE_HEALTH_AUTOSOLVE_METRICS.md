# Legal AI Platform: Health, Autosolve, Metrics & Tokenizer Guide

## 1. Overview

This guide explains how to:

- Run health checks & inspect core services
- Execute autosolve pipeline and confirm fix job publication
- Start full system components (gateway, worker, embedder, orchestrator)
- Verify PostgreSQL + pgvector state
- Exercise RAG / POI / evidence endpoints (probe strategy)
- Use new high_score prompt enhancement
- Access /metrics endpoints (embedder & worker)
- Generate fallback tokenizer if missing

## 2. Health Checks

Run orchestrator health script:

```
npm run orchestrator:health
```

Checks: Redis, RabbitMQ, CUDA (or mock), master process, Redis Go service.

Quick port sanity:

```
npm run utils:port-check
```

## 3. Autosolve Pipeline → Fix Jobs

Start an autosolve iteration that also queues fix jobs:

```
npm run autosolve:queue
```

Observe:

- Redis list `autosolve_summaries` gets entries (consumed every ~20s by orchestrator worker when AUTO_SOLVE_ENABLED=true)
- RabbitMQ queue `fix_jobs` receives jobs (watch management UI or log `🧩 Published fix job`)

Enable orchestrator autosolve mode (spawns worker polling summaries):

```
AUTO_SOLVE_ENABLED=true npm run orchestrator:dev
```

## 4. Full System Startup (Minimal Dev Set)

In separate terminals or using tasks:

```
# Start embedder (gRPC + metrics)
ONNX_MODEL_PATH=models/code-embed.onnx TOKENIZER_PATH=tokenizer.json npm run embedder:server

# Start worker (Redis stream consumer + metrics)
WORKER_METRICS_PORT=9301 node worker.js

# Start orchestrator master + worker processes
npm run orchestrator:dev

# Start gateway / production RAG (if needed)
npm run go:run:rag
```

Optional: start frontend `npm run dev:frontend`.

## 5. PostgreSQL + pgvector Verification

```
# Basic connectivity
psql postgresql://postgres:password@localhost:5432/postgres -c "\dt"

# Ensure todos table exists
psql ... -c "\d+ todos"

# Verify embedding column (if using pgvector directly)
psql ... -c "SELECT vector_dims(embedding) FROM legal_documents LIMIT 1;"
```

If dimension mismatch with model (expected 768), recreate index/collection & re-embed.

## 6. RAG / POI / Evidence Endpoints (Probe)

Likely base: http://localhost:8094/api
Sample probes:

```
# Precedent search
npm run legal:precedent
# Compliance check
npm run legal:compliance
# Case analysis
npm run legal:case-analysis
```

Evidence / POI (examples to adapt):

```
curl -X GET http://localhost:8094/api/evidence/list
curl -X POST http://localhost:8094/api/evidence/metadata -H 'Content-Type: application/json' -d '{"doc_id":"sample","action":"inspect"}'
```

Adjust once concrete routes confirmed.

## 7. High Score Prompt Enhancement

Worker now injects detailed context lines:

```
CTXi:: high_score=H (semantic=S, recencyBoost=R, overlapBoost=O)
<snippet>
```

Guidance text instructs Gemma to weigh contexts proportionally; improves TODO relevance & prioritization.

## 8. Metrics Endpoints

Embedder: http://localhost:9300/metrics

```
{
  service: "embedder",
  singleCalls, batchCalls, fallbackCalls,
  avgModelLatencyMs, uptimeSec
}
```

Worker: http://localhost:9301/metrics

```
{
  service: "worker",
  processedErrors, embeddings, embeddingErrors,
  avgEmbeddingLatencyMs, gemmaCalls, gemmaFailures, avgHighScore
}
```

Use for dashboards / alert thresholds (e.g., fallbackCalls > 0 indicates missing model/tokenizer).

## 9. Tokenizer Fallback Generation

If no `tokenizer.json` present:

```
npm run generate:tokenizer
# Produces tokenizer.generated.json
TOKENIZER_PATH=tokenizer.generated.json npm run embedder:server
```

Replace with real Legal-BERT or Gemma tokenizer when available. Confirm embedder metrics `fallbackCalls=0` after swap.
## 10. Migration to Domain Models
1. Acquire ONNX + tokenizer for Legal-BERT (768 dims) or domain-tuned MiniLM.
2. Swap files, restart embedder.
3. Flush Qdrant (optional) & re-run `npm run vector:scan`.
4. Monitor /metrics for improved latency & zero fallbacks.

## 11. Troubleshooting

| Symptom                  | Action                                                         |
| ------------------------ | -------------------------------------------------------------- |
| fallbackCalls increasing | Check MODEL_PATH & TOKENIZER_PATH envs                         |
| gemmaFailures rising     | Validate GEMMA_API endpoint & network timeout                  |
| avgHighScore near 0      | Ensure embeddings not random (fallback), verify tokenizer load |
| embeddingErrors > 0      | Inspect gRPC embedder_server logs                              |

## 12. Next Enhancements

- Prometheus exposition format (exporter) for /metrics
- Histogram buckets for latency
- Error classification taxonomy feeding overlapBoost
- Adaptive weighting: learn coefficients via regression over resolved errors

End of Guide.
