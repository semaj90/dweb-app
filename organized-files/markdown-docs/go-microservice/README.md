# Go Microservice — Build and Legacy Code Guide

This microservice uses Go build tags to keep the active codebase clean while preserving experimental and legacy implementations in the repository.

## Default behavior

- Legacy files are marked with build constraints and are excluded from normal builds.
- Active service code lives under `go-microservice/service` and is launched via the entrypoint in `go-microservice/cmd`.

---

## Building without legacy code (default)

Run builds from the Go module folder (`go-microservice`):

```bash
cd go-microservice
go build ./...
```

This excludes files tagged with:

```go
//go:build legacy
// +build legacy
```

## Building with legacy code enabled

If you need to include the legacy implementations:

```bash
cd go-microservice
go build -tags=legacy ./...
```

---

## Retagging newly added legacy files

A helper script tags legacy files in one pass (skips `service/` and `cmd/`):

```bash
# from the repository root
node scripts/tag-legacy-go-files.mjs
```

There is also a VS Code task: “Tag Legacy Go Files” (Terminal → Run Task…)

> Script location: `./scripts/tag-legacy-go-files.mjs`

---

## Notes

- Keep active source under `go-microservice/service` with a clean public API.
- The main entrypoint for the Ollama SIMD service is under `go-microservice/cmd/go-ollama-simd`.
- See `README-GO-OLLAMA.md` for service endpoints, health, and metrics.

### Hardening option (allowlist)

If you want, we can tighten the tagger to use an explicit allowlist of safe directories instead of “tag everything except service/cmd.” This further reduces the chance of tagging active files by mistake.
