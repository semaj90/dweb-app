# Production Build Baseline

This document defines the minimal, stable production build baseline for the Legal AI platform and how optional/experimental code is isolated using Go build tags.

## Core Production Binaries (Supported / Shippable)
These four binaries constitute the lean deployment surface:

| Binary | Path | Purpose |
|--------|------|---------|
| multi-protocol-gateway | `go-microservice/cmd/multi-protocol-gateway` | Unified entry exposing HTTP / future protocol routing (baseline gateway) |
| gpu-orchestrator | `go-microservice/cmd/gpu-orchestrator` | Coordinates GPU/accelerated tasks (non-experimental core subset only) |
| enhanced-rag-som | `go-microservice/cmd/enhanced-rag-som` | RAG pipeline + SOM / vector orchestration (production-safe path) |
| health-server | `go-microservice/cmd/health-server` | Lightweight health & readiness endpoint aggregation |

Build all four:

```powershell
# From repo root
powershell -NoProfile -Command "cd go-microservice; go build ./cmd/multi-protocol-gateway ./cmd/gpu-orchestrator ./cmd/enhanced-rag-som ./cmd/health-server"
```

## Tag Policy
All non‑baseline / legacy / GPU-heavy / experimental mains and feature modules are guarded by composite build constraints.

Patterns in files:
```go
//go:build experimental || legacy
```
Or specialized (example):
```go
//go:build experimental || loadbalancer
```

### Rationale
- Keeps default `go build ./...` (without tags) lean & deterministic.
- Allows progressive reintroduction: `-tags experimental` to include all gated code.
- Consolidates prior duplicate tag lines into **one** canonical constraint to avoid "multiple //go:build" errors.

### Using Tags
Include experimental surface:
```powershell
cd go-microservice
go build -tags "experimental" ./...
```
Add selective domain tag (example if present):
```powershell
go build -tags "experimental,loadbalancer" ./cmd/load-balancer
```

### Verifying No Leakage
Quick check for untagged non-core mains:
```powershell
Get-ChildItem go-microservice -Recurse -Filter *.go | Select-String -Pattern "package main" | Select-String -NotMatch "multi-protocol-gateway|gpu-orchestrator|enhanced-rag-som|health-server" | ForEach-Object { $_.Path }
```
Each listed file should have exactly one `//go:build` line containing `experimental` (or be intentionally excluded).

## Dependency Posture
- Bytedance/sonic & other high-risk parsing libs remain behind experimental code paths; baseline does not require their stabilization.
- Cache / Redis / QUIC / advanced GPU indexing binaries excluded unless tags supplied.

## Frontend Alignment
Frontend dev scripts now de-duplicated (`dev:full:legacy` conflict resolved). A CommonJS shim (`src/lib/shims/commonjs-shim.js`) exists but is **opt-in**—import only if a package probes `module`/`exports` at runtime.

## Database Superuser (Local Development)
For local development the startup scripts now prefer the Postgres superuser connection:

```
postgresql://postgres:postgres@localhost:5432/legal_ai_db
```

If you previously relied on `legal_admin`, set an explicit `DATABASE_URL` env var to override. This avoids permission blockers during migrations, extension installs (e.g. pgvector), and rapid prototyping.

## CI Suggestions (Future)
1. Add a lint step ensuring only the four production binaries are tag-free mains.
2. Add `go vet` & `staticcheck` pass restricted to baseline.
3. Add a matrix job compiling with `-tags experimental` to keep gated code from silently rotting.

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Experimental binary unexpectedly builds without `-tags` | Missing or malformed build line | Add `//go:build experimental || legacy` (single line) |
| Duplicate build constraint error | Multiple `//go:build` lines | Merge into single expression |
| Sonic iterator / GoMapIterator errors | Building experimental parsers unintentionally | Rebuild without tags or finish stub implementation |
| Frontend dev warning about duplicate script | Reintroduced duplicate key in `package.json` | Remove second key or rename variant (follow baseline) |

## Minimal Release Artifact Checklist
- [ ] All four baseline binaries build clean (no tags).
- [ ] `go test ./...` passes for baseline packages (optional if tests gated appropriately).
- [ ] No stray untagged `main` packages outside the four allowed.
- [ ] Frontend `npm run build` succeeds without CJS define hacks.

---
_Last updated: 2025-09-02_
