# Embedding Service Skeleton

This service batches unembedded `passages` rows and populates the `embedding` column (vector(768)).

## Features
- Adaptive batch loop (placeholder heuristic)
- Model spec discovery via `embedding_metadata` (falls back to default if empty)
- Dimension guard aligned with migration 002
- Minimal HTTP endpoints: `/health`, `/stats`, `/enqueue` (stub)
- Mock embedding generator (deterministic) – replace with real GPU model integration

## Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| DATABASE_URL | Postgres DSN | postgres://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable |
| EMB_BATCH_MAX | Max batch size | 64 |
| EMBED_SERVICE_PORT | HTTP listen port | 8098 |

## Next Steps
1. Insert an active embedding metadata record:
```sql
INSERT INTO embedding_metadata (model_name, model_version, dim, method, quantization)
VALUES ('nomic-embed-text', 'v0', 768, 'nomic-embed-text', 'fp16');
```
2. Start service:
```powershell
cd go-microservice
go run embedding-service.go
```
3. Tail progress by inspecting count of non-null embeddings:
```sql
SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL;
```
4. Replace `mockEmbed` with real model binding (Python gRPC, cgo, or local ONNX runtime) and enforce latency metrics.

## Index Activation
After >50k passages embedded:
```sql
CREATE INDEX CONCURRENTLY idx_passages_embedding_cosine
ON passages USING ivfflat (embedding vector_cosine_ops) WITH (lists=200);
```

## Metrics (Planned)
- embedding_batch_seconds (histogram)
- embedding_passages_total (counter)
- embedding_errors_total (counter)
- embedding_active_batch_size (gauge)

Add once Prometheus registry integrated.
