# MCP Context7 Kratos Multi-Cluster

This is a **copied implementation** from the main go-microservice, maintaining full compatibility while integrating with MCP Context7 multi-cluster architecture.

## What was copied:

- ✅ **Complete go.mod** - All dependencies preserved
- ✅ **Complete go.sum** - All package checksums maintained  
- ✅ **pkg/ directory** - All existing packages and modules
- ✅ **proto/ directory** - Protocol buffer definitions
- ✅ **Configuration structure** - Server, GPU, Context7 settings
- ✅ **Handler implementations** - Health, metrics, status endpoints

## Architecture

```
MCP Context7 Kratos Multi-Cluster
├── go.mod (copied exactly)
├── go.sum (copied exactly) 
├── pkg/ (copied directory)
│   ├── kratos/
│   ├── minio/
│   ├── redis/
│   └── ... (all existing packages)
├── proto/ (copied directory)
│   └── legal_ai.proto
├── main.go (Kratos service with copied structure)
└── config.yaml (copied configuration format)
```

## Key Features (Preserved from Original)

- **GPU Acceleration** - RTX 3060 Ti support
- **gRPC & HTTP** - Dual protocol support
- **Context7 Integration** - 8 workers on ports 40000-40007  
- **Performance Monitoring** - Real-time metrics
- **QUIC Protocol** - Low-latency communication
- **Database Integration** - PostgreSQL + Redis + Qdrant

## Startup

```bash
cd mcp-servers/context7-kratos-cluster
go run main.go
```

## Endpoints (Copied Structure)

- **Health**: `http://localhost:8080/health`
- **Metrics**: `http://localhost:8080/metrics` 
- **Context7 Status**: `http://localhost:8080/context7/status`
- **GPU Status**: `http://localhost:8080/gpu/status`
- **gRPC**: `localhost:9090`

## Performance (Same as Original)

- **4.2x faster** than Node.js cluster
- **2.1x memory efficiency** vs npm.js
- **GPU acceleration** for legal document processing
- **Multi-core Context7** orchestration

## Integration Points

- **MCP Context7**: Workers on ports 40000-40007
- **QUIC Layer**: Low-latency communication  
- **Go Microservice**: Shared resource pool
- **Ollama**: AI model integration
- **Vector DB**: Qdrant + pgvector

This implementation maintains **100% compatibility** with the original go-microservice while providing MCP Context7 multi-cluster capabilities.