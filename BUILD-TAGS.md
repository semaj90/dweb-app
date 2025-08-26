Build Tags Reference
=====================

These Go build tags gate optional / experimental executables. Default `go build ./...` now compiles only the minimal health server and redis-service.

Available tags:

| Tag | Purpose |
|-----|---------|
| enhancedrag | Enhanced RAG + SOM intent clustering service |
| dblayer | Unified PostgreSQL/Neo4j/Redis integration layer |
| quiccoord | Full QUIC coordinator (low-latency streams) |
| quictensor | QUIC tensor transport experimental layer |
| realtime | gRPC/WebSocket hybrid real-time communication layer |
| simpleapi | Minimal REST API exposing /api/rag and /api/ai |
| orchestrator | MCP / multi-protocol GPU orchestrator |
| llamachat | LLaMA chat streaming service |
| legacy / legacygpu | Legacy GPU or deprecated prototypes |
| ignore | Files intentionally excluded (stubs / backups) |

Examples:

```powershell
# Build enhanced RAG + DB layer
go build -tags "enhancedrag dblayer" -o bin/enhanced-rag.exe .

# Build QUIC coordinator only
go build -tags quiccoord -o bin/quic-coordinator.exe .

# Run tests with specific feature code included
go test -tags "simpleapi" ./...
```

Add new feature binaries by placing `//go:build <tag>` + `// +build <tag>` lines at top of the file.
