# Logging, Profiling & Observability Strategy (2025-09-01)

## 1. Objectives
Provide a unified, low‑overhead, high‑fidelity observability layer spanning:
- High-performance native (C/C++) components (future OCR/vision kernels, SIMD parsers, potential wasm sources).
- Go microservices (metrics, RAG, graph, embedding).
- Python ML workers (OCR, passage splitting, embeddings, graph analytics, RL prototype).
- Browser (SvelteKit/WebGPU) client instrumentation.

Goals: Precise hotspot identification, cross-layer latency attribution, adaptive optimization (batch size, caching), and high signal / low noise logging.

## 2. Layered Observability Stack
| Layer | Tooling | Purpose |
|-------|---------|---------|
| Native (C/C++) | spdlog (JSON), ETW providers, optional Perfetto trace exports | Ultra-low overhead structured logs + kernel/user timing |
| Python ML | Standard logging (structlog), PyTorch Profiler, NVTX ranges | Batch latency, GPU op breakdown |
| Go Services | zap/log/slog (JSON), OpenTelemetry (traces/metrics), pprof | API latency, queue depths, memory, goroutine profiling |
| Browser | Performance API, Web Vitals, custom tracing (OTel JS) | UX & client-side preprocessing metrics |
| Central Aggregation | Loki / Elasticsearch (future), lightweight Go collector (phase 1) | Unified query & retention |
| Metrics | Prometheus + exporters (CUDA service, pipeline workers) | Time-series SLIs & alerting |
| Traces | OpenTelemetry (OTLP), optional Jaeger/Tempo | Cross-service latency path |
| Profiling | CUPTI (GPU), VTune / uProf (CPU microanalysis), pprof, PyTorch profiler | Hotspots and hardware efficiency |

## 3. C++ Logging (spdlog) Configuration
- Sink stack: async ring buffer (bounded) → stdout (JSON) + rotating file.
- Pattern (JSON) example: `{ "ts":"%Y-%m-%dT%H:%M:%S.%eZ", "lvl":"%l", "msg":"%v", "file":"%s", "line":%# , "trace_id":"%X{trace_id}", "span_id":"%X{span_id}", "comp":"cuda_ocr" }`.
- Severity mapping: trace, debug (dev only), info, warn, error, critical.
- Category macros: `LOG_IO`, `LOG_GPU`, `LOG_PIPE`, each adds `component` field for filtering.

### 3.1 Example Initialization
```cpp
auto logger = spdlog::create_async<spdlog::sinks::stdout_color_sink_mt>("cuda_ocr");
logger->set_pattern("{ \"ts\":\"%Y-%m-%dT%H:%M:%S.%eZ\", \"lvl\":\"%l\", \"msg\":\"%v\", \"trace_id\":\"%X{trace_id}\", \"comp\":\"cuda_ocr\" }");
spdlog::set_default_logger(logger);
spdlog::set_level(spdlog::level::info);
```

## 4. ETW Integration (Windows)
- Create custom provider GUID (e.g., `LegalAI-OCR-Provider`).
- Emit events for: tile_processed, ocr_batch_start/end, gpu_copy, embedding_batch.
- Use manifest or TraceLogging dynamic APIs.
- Export ETW session to JSON/Chromium trace format for Perfetto UI correlation.

## 5. Central Go Log Collector (Phase 1 Minimal)
- Endpoint: `POST /ingest/log` (gzip, JSON array of entries).
- Accepts: `[{ ts, lvl, service, component, msg, trace_id?, span_id?, fields:{...}}]`.
- Writes to: local file (rotating) + optional stdout; batches flush every N ms or size threshold.
- Later: swap to Loki or OpenSearch sink (structured indexing by `service, component, lvl`).

### 5.1 Sample Ingestion Contract
```json
{
  "service": "cuda-ocr",
  "component": "tiler",
  "lvl": "info",
  "ts": "2025-09-01T21:04:20.123Z",
  "msg": "tile batch processed",
  "trace_id": "a1b2...",
  "span_id": "c9d0...",
  "fields": { "tiles": 128, "duration_ms": 42.7, "gpu_util": 0.78 }
}
```

## 6. Structured Log Schema (Canonical Fields)
| Field | Type | Required | Notes |
|-------|------|----------|-------|
| ts | RFC3339 | yes | millisecond precision |
| lvl | enum | yes | trace→critical |
| service | string | yes | emitting logical service |
| component | string | yes | submodule (tiler, ocr, embed, graph, rag) |
| msg | string | yes | human-readable summary |
| trace_id | string | recommended | OTel trace correlation |
| span_id | string | optional | fine-grained span correlation |
| fields | object | optional | numeric/contextual data |
| version | string | optional | binary/git version |

## 7. Correlation & Trace Propagation
- Adopt W3C trace context: incoming HTTP / gRPC carries `traceparent`.
- On ingestion pipeline steps, generate child spans (OpenTelemetry SDK in Go/Python).
- Native C++ side: provide simple API to set current trace context (thread-local) to embed trace/span IDs into spdlog MDC.

## 8. Profiling Tool Matrix
| Concern | Primary Tool | Secondary | Trigger to Activate |
|---------|--------------|----------|---------------------|
| GPU kernel timing | CUPTI Activity API | Nsight Systems | Always sampling (light) / deep dive when > SLA |
| GPU memory bandwidth | CUPTI metrics | Nsight Compute | Embedding latency regression |
| CPU hotspots (C++/Go) | VTune / pprof | Perfetto | P95 > budget |
| Python op breakdown | PyTorch Profiler | NVTX + Nsight | New model baseline or throughput drop |
| Allocation churn | heap profiling (pprof) | jemalloc stats | GC pauses or memory growth >X% |
| End-to-end latency | OTel traces | Tempo/Jaeger UI | Continuous |

## 9. Logging vs Metrics vs Traces Guidelines
| Use Case | Use |
|----------|-----|
| High-cardinality numeric time-series | Metrics (Prometheus) |
| Single request path latency debugging | Traces |
| Discrete event with context | Structured Log |
| Continuous performance abnormality detection | Metric anomaly + occasional log summary |

## 10. Pipeline Instrumentation Mapping
| Stage | Key Metric(s) | Log Events | Trace Span Names |
|-------|---------------|-----------|------------------|
| OCR Tiling | tiles/sec, avg tile ms | tile_batch_processed | ocr.tile_batch |
| OCR Inference | tokens/sec, WER | ocr_batch_completed | ocr.infer_batch |
| Passage Split | passages/sec | passage_batch_split | passage.split |
| Embedding | embeddings/sec, batch_ms | embedding_batch_completed | embed.batch |
| Graph Edge Build | edges/sec, PR_ms | similarity_job_completed | graph.build |
| RAG Retrieval | query_ms, context_tokens | rag_query_served | rag.query |
| Visualization Export | projection_ms | umap_export_finished | viz.umap |

## 11. Browser Capture & Web Content Semantic Pipeline
1. Puppeteer navigate → network idle.
2. Capture: final HTML, full-page screenshot (PNG), PDF (optional), HAR.
3. OCR (Tesseract/Paddle) on screenshot → text components.
4. YOLO (object detection) on screenshot for non-text semantics (figures, diagrams).
5. NLP (spaCy / transformer) on extracted text for topics, entities, sentiment.
6. Persist artifacts: MinIO (raw), Postgres (text/entities), vector store (page embeddings), index for retrieval.

## 12. Cross-Layer Caching Strategy
| Layer | Cache Type | Contents | Invalidation |
|-------|------------|----------|--------------|
| Browser | IndexedDB + SW | Recent passages, neighbors | Version hash / TTL |
| API Edge | Redis | RAG query results, neighbor sets | `rag.cache.invalidate` subject |
| Embedding Worker | LRU in‑process | Model warm vectors, tokenization cache | Size cap |
| Graph Builder | Redis / file | Previous PR vector | Graph update delta |
| Driver/GPU | Shader & kernel cache | Binary kernels | Driver-managed |
| Native Module | Memoization | Preprocessed tile geometry | Code hash |

## 13. Example Go Ingestion Endpoint (Sketch)
```go
func (s *Server) IngestLogs(w http.ResponseWriter, r *http.Request) {
  ctx := r.Context()
  var entries []LogEntry
  if err := json.NewDecoder(r.Body).Decode(&entries); err != nil {
    http.Error(w, "bad request", 400); return
  }
  now := time.Now()
  for _, e := range entries {
    s.buf.Append(e) // ring buffer or channel → async writer
    s.metrics.LogIngested.Inc()
  }
  w.WriteHeader(202)
  _ = json.NewEncoder(w).Encode(map[string]any{"accepted": len(entries), "ts": now})
}
```

## 14. Adoption Phases
| Phase | Scope | Deliverables |
|-------|-------|--------------|
| P0 | Baseline structured logs (Go + Python), Prom metrics, trace IDs | JSON log format live |
| P1 | spdlog integration + ingestion endpoint | Native logs visible centrally |
| P2 | CUPTI real snapshots + NVTX spans | GPU utilization vs latency dashboard |
| P3 | ETW provider & Perfetto export | Cross kernel ↔ pipeline correlation |
| P4 | Advanced trace sampling logic | Reduced noise, adaptive sampling |
| P5 | Alert correlation (log anomaly ↔ metric spike) | Root-cause panel |

## 15. KPIs & Target Budgets
| Metric | Target |
|--------|--------|
| Logging overhead | <2% CPU per service |
| Average end-to-end trace coverage | >95% sampled (P0), 50% (adaptive P4) |
| Max ingestion latency (log entry → index) | <3s |
| GPU embedding batch P95 latency | <120ms |
| OCR tile throughput | >2k tiles/min (baseline) |
| RAG query P95 | <1.2s |

## 16. Risk & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Log volume explosion | Storage / cost | Rate limiting + sampling + severity filtering |
| ETW complexity | Slow adoption | Start optional; provide Perfetto export script |
| CUPTI instability | Service crashes | Feature flag + fallback synthetic snapshot |
| Trace cardinality | Storage bloat | Tail-based or adaptive sampling (error/slow full) |
| JSON parse overhead | CPU | Use structured log libs + optional binary (Protobuf) ingestion later |

## 17. Immediate Implementation Checklist
- [ ] Introduce shared log schema constants (Go & Python).
- [ ] Add spdlog async JSON logger to native module skeleton.
- [ ] Build minimal `/ingest/log` endpoint + batch writer (Go).
- [ ] Propagate `traceparent` header through services (middleware).
- [ ] Add NVTX ranges around embedding & OCR kernels (Python/C++ mixed).
- [ ] Replace synthetic profiling snapshot with real CUPTI Activity capture (flagged).
- [ ] Define Prometheus histograms for each pipeline stage (Section 10).
- [ ] Add Redis-backed recent RAG query cache entries with hit/miss metrics.

## 18. Decision Points
- Binary log pipeline (Protobuf) only if ingestion CPU >5% or JSON payload >1MB/s sustained.
- ETW adoption after CUPTI + OTel stable (avoid simultaneous complexity spikes).
- Perfetto trace export integrated once NVTX + CUPTI deliver stable GPU span mapping.

---
Prepared 2025-09-01 to extend the existing roadmap with deep observability & logging strategy.
