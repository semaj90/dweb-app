# mcpGPUOrchestrator (Security Validation Service)

High-performance security & risk scoring microservice powering advanced registration flows.

## Features
- Fastify HTTP API (`/validate/security`) with Zod validation
- Heuristic risk scoring pipeline (extensible to GPU/AI models)
- Structured signal attribution for explainability
- WebSocket progress channel (`/ws/security`) for future streaming model inference
- Strict TypeScript schemas for request/response

## Endpoints
### Health
GET `/health`

### Security Validation
POST `/validate/security`
```jsonc
{
  "task": "security_validation",
  "fingerprint": { "userAgent": "...", "hardwareConcurrency": 8, "webglRenderer": "NVIDIA RTX" },
  "user": { "email": "user@example.com", "username": "alice" },
  "context": { "ipReputation": 0.82, "velocity": 0.1 }
}
```
Response:
```jsonc
{
  "requestId": "uuid",
  "riskScore": 0.23,
  "securityScore": 0.79,
  "verification": { "emailFormatValid": true, "usernameValid": true },
  "signals": [ { "name": "ipReputation", "contribution": -0.04 } ],
  "status": "allow",
  "modelVersion": "heuristic-v0.1",
  "durationMs": 12,
  "timestamp": "2025-09-02T12:00:00.000Z"
}
```

## Run
```bash
pnpm i
pnpm dev
# or
npm install
npm run dev
```

## Extending to GPU / AI
1. Replace `computeRisk` internals with batched embedding + inference pipeline (e.g., vLLM / ONNX + CUDA).
2. Add model warmup step at service start.
3. Stream intermediate feature extraction over the WebSocket channel.

## Next Steps
- gRPC/QUIC transport
- External reputation provider integration
- Persistent feature store (Redis) for velocity & anomaly tracking
- Model-based classification (XGBoost / transformer)
- Signed response envelope (HMAC) for tamper prevention
