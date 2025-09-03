# GPU Orchestrator (Internal)

Lightweight GPU/task scheduling service extracted from prior monolithic prototype. Provides:

## Features
- HTTP + WebSocket API (health, task submit/status, tensor cache, GPU stats)
- UCB (Upper Confidence Bound) based scheduler with priority + age boosts
- In‑memory tensor (4D) cache with simple LRU eviction
- Simulated execution handlers (embedding / inference / CUDA kernel / tensor op)
- Prometheus metrics endpoint `/metrics/prom` + simple text metrics `/metrics`
- WebSocket push events: `task_started`, `task_completed`, `gpu_stats_update`

## Key Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Service + GPU status |
| POST | /gpu/task | Submit task (json: type, priority, input, metadata) |
| GET | /gpu/task/status?id=TASK_ID | Get single task status |
| GET | /gpu/tasks | Active + queue depth overview |
| POST | /gpu/task/cancel?id=TASK_ID | Cancel running task |
| GET | /gpu/stats | Latest cached GPU stats |
| POST | /tensor/cache/put | Store tensor (key, shape, meta) |
| GET | /tensor/cache/get?key=K | Retrieve tensor metadata |
| GET | /metrics | Plain text counters / latency |
| GET | /metrics/prom | Prometheus exposition format |
| GET | /ws | WebSocket for real‑time events |

## Environment Variables
| Var | Default | Purpose |
|-----|---------|---------|
| GPU_ORCHESTRATOR_PORT | 8095 | HTTP listen port |
| GPU_ENABLED | false | Enable real GPU detection / nvidia-smi usage |
| CUDA_PATH | platform default | CUDA toolkit path (for capability check) |
| GPU_MAX_CONCURRENT | 8 | Max concurrent simulated tasks |
| GPU_MEMORY_LIMIT | 6GB | Informational cap (not enforced) |
| WEBASSEMBLY_URL | http://localhost:8080 | External WASM service (future) |

## Scheduling (UCB)
Score = exploitation + exploration + priorityBoost + ageBoost
- exploitation: value/visits (reward from previous completions)
- exploration: C * sqrt(log(totalSelections)/visits)
- priorityBoost: Priority * 0.05
- ageBoost: min(ageSeconds/30, 0.5)

## Metrics (Prometheus)
Counters / Histograms defined in `metrics.go` (task count + latency labeled by task type & status).

## Tests
Current tests cover:
- UCB selection basics
- Execution lifecycle + metrics
- Queue dispatch & completion (added)
- Tensor cache operations (added)

Run:
```
go test ./internal/orchestrator -count=1 -v
```

## Future Enhancements
- Real GPU task integration hooks
- Back‑pressure & task admission control
- Pluggable persistence for task history
- Distributed scheduling / sharding
- Structured logging & tracing
