Go Ollama SIMD Service

- Dev: run from project root via `npm run dev` (zx will start it on :8081)
- Prod: build `go-ollama-simd.exe` in go-microservice and `npm run start` will include it
- Health: http://localhost:8081/health
- Analyze Evidence Canvas: POST http://localhost:8081/api/evidence-canvas/analyze

Env:

- OLLAMA_URL (default http://localhost:11434)
- GO_OLLAMA_PORT (default 8081)
