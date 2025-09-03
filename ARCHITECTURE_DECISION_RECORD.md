# Architecture Decision Record: WebAssembly vs Go Backend

## Status: ✅ DECIDED - Go Backend-First with WebAssembly Fallback

## Context

We evaluated implementing client-side AI inference via WebAssembly vs leveraging existing Go backend services.

## Decision

**Prioritize Go Backend with WebAssembly as Optional Enhancement**

## Rationale

### ✅ **Advantages of Go Backend Architecture:**

1. **Production Ready**
   - Enhanced RAG service already compiled and running (port 8094)
   - Redis integration working (port 6379)
   - RTX 3060 Ti GPU acceleration available

2. **Superior Performance**
   - Native Go performance vs browser WebAssembly constraints
   - Direct GPU access vs WebGL/WebGPU limitations
   - No browser memory limits (8GB+ vs ~2GB browser heap)

3. **Simplified Development**
   - No C++/Emscripten compilation complexity
   - Existing Ollama integration (`gemma3-legal` model ready)
   - Proven PostgreSQL + pgvector integration

4. **Enterprise Features**
   - Server-side model caching and optimization
   - Centralized security and access control
   - Scalable multi-user architecture

### ⚠️ **WebAssembly Challenges Avoided:**

1. **Compilation Complexity**
   - Emscripten toolchain setup
   - llama.cpp build configuration
   - Cross-platform compatibility issues

2. **Browser Limitations**
   - Memory constraints for large models
   - No direct GPU compute access
   - Network latency for model downloads

3. **Maintenance Overhead**
   - Keeping WASM builds in sync with upstream
   - Browser compatibility testing
   - Model quantization for client-side use

## Implementation Strategy

### **Phase 1: Optimize Existing Go Backend** ✅
- Enhanced RAG service operational
- WebAssembly service automatically falls back to Go
- SvelteKit frontend integration complete

### **Phase 2: Future WebAssembly Enhancement** (Optional)
- Use pre-compiled libraries like `@mlc-ai/web-llm`
- Focus on specific use cases (offline mode, privacy-sensitive queries)
- Implement when client-side inference provides clear value

## Current Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SvelteKit     │    │   Enhanced      │    │     Redis       │
│   Frontend      │◄──►│   RAG Service   │◄──►│     Cache       │
│   (Port 5173)   │    │   (Port 8094)   │    │   (Port 6379)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │     Ollama      │
         │              │  gemma3-legal   │
         │              │ RTX 3060 Ti GPU │
         │              └─────────────────┘
         │
         ▼
┌─────────────────┐
│   WebAssembly   │
│   Worker        │
│ (Auto-fallback) │
└─────────────────┘
```

## Outcome

✅ **Immediate Value**: Working AI-powered legal platform
✅ **Future Flexibility**: WebAssembly infrastructure ready when needed
✅ **Optimal Performance**: Native Go + GPU acceleration
✅ **Simplified Maintenance**: Focus on proven, stable technologies

## Status Update

**Current System Status**: 🟢 **OPERATIONAL**
- SvelteKit: ✅ Running (5173)
- Enhanced RAG: ✅ Running (8094)
- Redis: ✅ Running (6379)
- WebAssembly: ✅ Ready with Go fallback
- GPU: ✅ RTX 3060 Ti available for Ollama

**Ready for Production Use**

## Detailed Rationale (Expanded)
This decision prioritizes stability, performance, and development velocity by leveraging the most mature, GPU-enabled backend assets already in place.

### Immediate Velocity
Unblocks all UI and feature development now that the canonical inference path (Go RAG API) is stable. Frontend teams can integrate endpoints without waiting on complex WASM build pipelines.

### Superior Performance & Capability
Native Go services have unrestricted access to: RTX 3060 Ti GPU, full system RAM, fast local storage, and optimized concurrency. Browser environments face sandbox limits (heap size, threading semantics, emerging WebGPU variability). The backend path guarantees maximum model throughput and lower tail latency.

### Reliability & Consistency
Server-controlled execution removes broad variability from user hardware, browser version fragmentation, missing WebGPU implementations, WASM streaming failures, and memory pressure crashes. Result: uniform quality across all clients.

### Production Readiness
The Go + Ollama + pgvector stack is already exercised with real models (e.g. `gemma3-legal`). Operational metrics and logs exist; tuning & observability layers can be iterated faster than green‑field WASM infra.

### Centralized Management
Server-side rollout enables atomic model upgrades, hot patches, guarded A/B experiments, gating, and resource pooling—none of which require client invalidation or multi-MB downloads.

### Future Flexibility Retained
WASM path remains as a progressive enhancement (privacy/offline / low-latency micro‑tasks) without blocking the core delivery roadmap.

## Consequences & Implementation
Primary Path: `webasm-llama-service.ts` now acts chiefly as a thin client adapter to the Go Enhanced RAG API; the former fallback is the authoritative execution path.

Network Latency: Introduced for every inference, but amortized by sub‑second backend compute and can be further reduced using HTTP/2 multiplexing, gRPC, or QUIC.

Server Load: All inference centralized—intentional for easier scaling (horizontal pods, GPU scheduling, batching, caching tiers).

Engineering Focus: Effort shifts from toolchain debugging to feature delivery (search UX, security validation, embeddings lifecycle, audit pipeline).

Future Flexibility: WASM worker + service scaffolding remains intact and can be progressively enabled behind a feature flag when a narrow, high‑value client-side use case emerges.

## Verification Status
All components in the server-first pipeline have been exercised end-to-end. The WASM layer successfully detects environment and defers to Go.

| Component              | Address / Context            | Status | Notes                                   |
|------------------------|------------------------------|--------|-----------------------------------------|
| SvelteKit Frontend     | http://localhost:5173        | 🟢     | UI loads & connects to API              |
| Enhanced RAG Service   | http://localhost:8094        | 🟢     | Primary inference backend               |
| Redis Cache            | localhost:6379               | 🟢     | Caching + future velocity features      |
| WebAssembly Worker     | In-browser (deferred)        | 🟢     | Initializes; routes to Go pathway       |
| GPU (RTX 3060 Ti)      | Host GPU                     | 🟢     | Available; Ollama acceleration active   |
| Ollama Model (gemma3)  | Local runtime                | 🟢     | Responding; validated via test prompts  |

End-to-end scenario: Frontend → WASM adapter (environment check) → Go Enhanced RAG → GPU‑accelerated model → Response returned (validated success path). No blocking defects observed.

## Next Action Items (Post-Decision Roadmap)

1. Integrate Security Validation Orchestrator (mcpGPUOrchestrator) into registration flow via `/api/security/validate`.
2. Add Redis-backed velocity & attempt tracking feeding `context.velocity` and `previousFailures`.
3. Implement structured tracing (OpenTelemetry) across Go services and SvelteKit API boundary.
4. Expose health & readiness aggregate endpoint consolidating: RAG, Redis, Ollama, Security Orchestrator.
5. Gradual WebAssembly enhancement experiments (target small on-device models <200MB; use feature flag).

### KPIs to Monitor
- Median end-to-end registration latency (target < 1200ms)
- RAG query P95 latency (target < 800ms)
- GPU utilization (keep 40–70% sustained)
- Cache hit rate (Redis) for repeated embeddings (> 85%)

### Security Hardening Follow-Ups
- HMAC sign orchestrator responses.
- Add rate limiting (IP + user) at API gateway.
- Log structured risk decision payloads (PII-scrubbed) for model retraining.