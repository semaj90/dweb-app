# Node API Microservice (NATS + Drizzle + Metrics)

Features:
- NATS publish/subscribe with circuit breaker & retry (p-retry)
- Drizzle logging (JSONB payload)
- Redis recent message cache (optional)
- Prometheus metrics `/api/v1/nats/metrics`
- Health endpoint `/healthz`
- Subject whitelist & payload size guard
- Config validation (zod) & trace IDs

Env Vars:
```
NATS_URL=nats://127.0.0.1:4222
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_db
REDIS_URL=redis://localhost:6379
NODE_API_SUBJECT_WHITELIST=legal.case.created,legal.case.updated
NODE_API_MAX_PAYLOAD_KB=256
```

Metrics:
```
nats_publish_total
nats_publish_failures_total
nats_subscriptions
nats_queue_backlog
nats_message_latency_ms{quantile="0.5|0.9|0.99"}
```

Next:
- BullMQ GPU/WASM pipeline
- Batch DB insert optimization
- HMAC/API key security layer
