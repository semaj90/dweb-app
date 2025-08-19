# VS Code Tasks - Legal AI Platform

## Overview

This document describes the VS Code task configuration for the Legal AI platform. The tasks are designed to provide a complete development environment with proper Go binary wiring and service orchestration.

## Main Development Task

### 🚀 Dev Full Stack: Build & Start All Services

**This is the default task (Ctrl+Shift+P → "Tasks: Run Build Task")**

This master task performs the following sequence:
1. **Build: All Go Services** - Compiles all Go microservices
2. **Build: QUIC Services** - Builds QUIC protocol services
3. **Start: Complete Dev Environment** - Runs `npm run dev:full`

## Task Categories

### Build Tasks

- **Build: All Go Services** - Builds all required Go microservices including missing binaries
- **Build: QUIC Services** - Builds QUIC protocol services and places them in the bin directory
- **Fix: Go Binary Paths** - Creates missing binary aliases and verifies availability

### Service Management

- **Start: Complete Dev Environment** - Starts `npm run dev:full` after building services
- **Start: Individual Go Services** - Starts Go services individually for debugging
- **Stop: All Services** - Stops all running services and cleans up

### Health & Monitoring

- **Health: Check All Services** - Comprehensive health check for all platform services
- **Orchestration: Health Check All** - Checks orchestration service health specifically

## Port Allocation

The platform uses the following port allocation:

### Core Services
- **PostgreSQL**: 5432
- **Redis**: 6379  
- **Ollama**: 11434
- **MinIO**: 9000 (Console: 9001)
- **Qdrant**: 6333
- **Neo4j**: 7474

### Application Services
- **SvelteKit Frontend**: 5173
- **Enhanced RAG**: 8094
- **Upload Service**: 8093
- **gRPC Server**: 8084
- **Load Balancer**: 8099
- **Cluster Manager**: 3000 (auto-increment if busy)

### Microservices (Dynamic Allocation)
- **Legal Workers**: 3010+ (count configurable)
- **AI Workers**: 3020+ (count configurable)
- **Vector Workers**: 3030+ (count configurable)
- **Database Workers**: 3040+ (count configurable)

## Go Binary Management

### Required Binaries

The system expects these Go binaries in `go-microservice/bin/`:

**Core Services:**
- `enhanced-rag.exe` - Enhanced RAG service (port 8094)
- `upload-service.exe` - File upload service (port 8093)
- `grpc-server.exe` - gRPC server (port 8084)
- `load-balancer.exe` - Load balancer (port 8099)

**QUIC Services:**
- `quic-gateway.exe` - QUIC protocol gateway
- `quic-vector-proxy.exe` - QUIC vector operations proxy
- `quic-ai-stream.exe` - QUIC AI streaming service

**Legacy Compatibility:**
- `rag-kratos.exe` - Alias for enhanced-rag.exe
- `rag-quic-proxy.exe` - Alias for quic-gateway.exe

### Binary Build Process

1. **Go Microservices** are built from `go-microservice/cmd/*` directories
2. **QUIC Services** are built from `quic-services/*.go` files
3. **Missing binaries** are created from existing sources (e.g., `legal-ai-server.go`)
4. **Aliases** are created for backward compatibility

## Environment Variables

### Service Configuration
- `MANAGER_PORT` - Cluster manager port (default: 3000)
- `MANAGER_PORT_AUTO` - Auto-increment port if busy (default: 1)
- `LEGAL_COUNT` - Number of legal workers (default: 3)
- `AI_COUNT` - Number of AI workers (default: 2)
- `VECTOR_COUNT` - Number of vector workers (default: 2)
- `DATABASE_COUNT` - Number of database workers (default: 3)

### Go Service Environment
- `RAG_HTTP_PORT` - Enhanced RAG service port
- `UPLOAD_PORT` - Upload service port
- `GRPC_PORT` - gRPC server port
- `LB_PORT` - Load balancer port
- `OLLAMA_BASE_URL` - Ollama API endpoint
- `MINIO_ENDPOINT` - MinIO storage endpoint

## Troubleshooting

### Common Issues

1. **Binary Not Found**: Run "Fix: Go Binary Paths" task
2. **Port Conflicts**: Check "Health: Check All Services" for conflicts
3. **Build Failures**: Check Go module dependencies with `go mod tidy`
4. **QUIC Services**: Ensure certificates exist in `quic-services/certs/`

### Service Dependencies

**Required before starting development:**
1. PostgreSQL running on 5432
2. Redis running on 6379
3. Ollama running on 11434
4. MinIO running on 9000

**Recommended for full functionality:**
1. Qdrant vector database on 6333
2. Neo4j graph database on 7474

### Build Order

1. Core Go services (enhanced-rag, upload-service, grpc-server)
2. QUIC protocol services (gateway, vector-proxy, ai-stream)
3. Node.js cluster manager and workers
4. SvelteKit frontend development server

## Performance Optimization

### Cluster Configuration

The cluster manager automatically:
- Allocates ports using closest-port algorithm
- Handles worker respawning on failure
- Provides health monitoring and metrics
- Manages graceful shutdown

### Resource Management

- **Memory limits** configured per worker type
- **Port search range** limited to 50 ports per service
- **Restart policies** with exponential backoff
- **Metrics collection** with real-time monitoring

## Security Considerations

- All services run on localhost only
- QUIC services use self-signed certificates for development
- No external network exposure by default
- PostgreSQL uses development credentials (should be changed for production)

## Integration with npm Scripts

The VS Code tasks integrate with these npm scripts:

- `npm run dev:full` - Starts frontend, cluster, load balancer, and microservices
- `npm run cluster:manager` - Starts Node.js cluster manager
- `npm run microservices` - Starts Go microservices via orchestrator
- `npm run lb:dev` - Starts load balancer in development mode

## Monitoring and Debugging

### Real-time Monitoring
- Cluster metrics: `.vscode/cluster-metrics.json`
- Orchestration health: `.vscode/orchestration-health.json`
- Service logs: Individual terminal panels in VS Code

### Debug Mode
- Set `LOG_LEVEL=debug` for verbose logging
- Use individual service start tasks for isolated debugging
- Monitor port allocation with cluster manager status endpoint

---

*This configuration supports the complete Legal AI platform with SvelteKit 2, PostgreSQL, Go microservices, QUIC protocols, and GPU acceleration.*