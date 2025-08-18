# GPU Orchestrator Scaffold - Quick Start Guide
## Native Windows GPU Acceleration with Node.js + CUDA + RabbitMQ + Redis

🎯 **Complete native Windows scaffold for GPU orchestration - No Docker required!**

## Prerequisites

### Required Software
1. **Node.js 18+** - [Download](https://nodejs.org/)
2. **RabbitMQ for Windows** - [Download](https://www.rabbitmq.com/install-windows.html)
3. **Redis for Windows** - Use WSL2 Redis or [Memurai](https://www.memurai.com/)
4. **NVIDIA CUDA Toolkit** - [Download](https://developer.nvidia.com/cuda-downloads)
5. **Visual Studio Build Tools** - Required for NVCC
6. **Go 1.21+** - [Download](https://golang.org/dl/)

### Optional
- **Clang/LLVM** - [Download](https://llvm.org/builds/) for additional compiler testing

## Quick Setup (5 minutes)

### 1. Clone and Install Dependencies
```bash
cd C:\Users\james\Desktop\deeds-web\deeds-web-app
npm install
```

### 2. Check Environment
```bash
npm run orchestrator:check-env
```
This runs `check_cuda_clang.ps1` to verify:
- ✅ NVCC compiler availability
- ✅ CUDA runtime functionality  
- ✅ Visual Studio Build Tools
- ⚠️ Clang availability (optional)

### 3. Build CUDA Worker
```bash
npm run orchestrator:build-cuda
```
Compiles `cuda-worker/cuda-worker.cu` → `cuda-worker.exe`

### 4. Setup Redis Go Service
```bash
cd redis-service
go mod tidy
go mod download
```

## Start All Services

### Method 1: Individual Services (Recommended for Development)

#### Terminal 1: Start RabbitMQ
```bash
# Windows Service Manager
services.msc
# Start RabbitMQ service, or:
net start RabbitMQ
```

#### Terminal 2: Start Redis  
```bash
# WSL2 Redis (recommended)
wsl redis-server

# Or native Windows Redis
redis-server.exe
```

#### Terminal 3: Start Redis Go Service
```bash
npm run orchestrator:redis-service
# Starts HTTP API on http://localhost:8081
```

#### Terminal 4: Start GPU Orchestrator
```bash
npm run orchestrator:start
# Starts master + worker cluster
# Health endpoint: http://localhost:8099/health
```

### Method 2: Development Mode
```bash
npm run orchestrator:dev
# Starts with fewer workers + debug logging
```

## Test the System

### 1. Health Check
```bash
npm run orchestrator:health
```
Expected output:
```
🏥 Starting comprehensive health check...
✅ REDIS
✅ RABBITMQ  
✅ CUDA_WORKER
✅ ORCHESTRATOR_MASTER
✅ REDIS_GO_SERVICE
📊 Health Score: 100% (5/5)
```

### 2. Publish Test Jobs
```bash
# Single embedding job
npm run orchestrator:test-job single embedding

# Batch of 10 similarity jobs  
npm run orchestrator:test-job batch similarity 10

# Stress test for 30 seconds
npm run orchestrator:test-job stress 30

# All job types
npm run orchestrator:test-job all
```

### 3. Monitor Job Results
```bash
# Via Redis Go service
curl http://localhost:8081/jobs/recent

# Via Redis CLI
redis-cli
127.0.0.1:6379> LRANGE job_history 0 9
```

## Architecture Overview

```
[RabbitMQ Queue] → [Node Cluster Workers] → [spawn] → [cuda-worker.exe]
                                    ↓
[Redis Storage] ← [Results] ← [JSON stdout] ← [CUDA Kernels]
                                    ↓
[XState Machine] → [Auto-Index] → [Idle Detection] → [Trigger Jobs]
```

### Components

1. **Master Process** (`orchestrator/master.js`)
   - Spawns worker cluster
   - Health monitoring
   - Graceful shutdown
   - Metrics collection

2. **Worker Processes** (`orchestrator/worker_process.js`)  
   - RabbitMQ consumers
   - CUDA worker spawning
   - XState idle detection
   - Result storage

3. **CUDA Worker** (`cuda-worker/cuda-worker.cu`)
   - JSON stdin/stdout IPC
   - GPU kernel execution
   - Vector operations (embedding, similarity, SOM)

4. **Redis Go Service** (`redis-service/main.go`)
   - HTTP API for Redis
   - WebSocket pub/sub
   - Job status tracking

## Available Job Types

| Type | Description | Input | Output |
|------|-------------|-------|--------|
| `embedding` | Basic vector transformation | `[1,2,3,4]` | `[1.23,2.47,3.70,4.94]` |
| `similarity` | Vector similarity computation | `[vec1,vec2]` | `[sim_results]` |
| `autoindex` | Auto-indexing operation | `[random_data]` | `[processed + metadata]` |

## Monitoring & APIs

### Health Endpoints
- **Orchestrator**: http://localhost:8099/health
- **Redis Service**: http://localhost:8081/health
- **Redis Stats**: http://localhost:8081/stats

### Job Management
- **Recent Jobs**: `GET /jobs/recent?limit=10`
- **Job Status**: `GET /jobs/status/{jobId}`
- **Publish**: `POST /publish` (Redis pub/sub)

### Redis Operations via HTTP
```bash
# Set value
curl -X POST http://localhost:8081/set \
  -H "Content-Type: application/json" \
  -d '{"key":"test","value":"hello","ttl":60}'

# Get value  
curl http://localhost:8081/get/test

# List operations
curl -X POST http://localhost:8081/lpush \
  -H "Content-Type: application/json" \
  -d '{"key":"mylist","values":["item1","item2"]}'
```

## Configuration

### Environment Variables
```bash
# RabbitMQ
RABBIT_URL=amqp://localhost
QUEUE_NAME=gpu_jobs

# Redis
REDIS_URL=redis://127.0.0.1:6379

# CUDA
CUDA_WORKER_PATH=./cuda-worker/cuda-worker.exe

# Modes
WORKER_MODE=production
AUTO_SOLVE_ENABLED=true
```

### Scaling
- **Workers**: Automatically uses CPU cores - 1
- **CUDA Concurrency**: Managed by worker spawn limiting
- **Memory**: Monitor GPU VRAM usage for optimal job sizing

## Troubleshooting

### Common Issues

#### CUDA Worker Not Found
```bash
# Check build
npm run orchestrator:build-cuda

# Manual test
cd cuda-worker
echo '{"jobId":"test","type":"embedding","data":[1,2,3,4]}' | .\cuda-worker.exe
```

#### RabbitMQ Connection Failed
```bash
# Check service
net start RabbitMQ
# Or install: https://www.rabbitmq.com/install-windows.html
```

#### Redis Connection Failed  
```bash
# WSL2 Redis (recommended)
wsl redis-server

# Check connection
redis-cli ping
```

#### GPU Memory Issues
- Reduce vector dimensions in test jobs
- Monitor GPU memory: `nvidia-smi`
- Adjust worker concurrency

### Logs
- **Master**: Console output + http://localhost:8099/health
- **Workers**: Job processing logs
- **CUDA**: stderr output in worker logs
- **Redis Go**: HTTP access logs

## Next Steps

### Extension Points
1. **NATS Integration**: Replace RabbitMQ with NATS for lower latency
2. **Kratos Auth**: Add identity management
3. **ELK Stack**: Centralized logging
4. **WebGPU Client**: Real-time browser visualization (see `WEBGPU_INTEGRATION_NOTES.md`)
5. **Production Optimization**: Load balancing, monitoring, clustering

### Performance Tuning
1. **CUDA Kernels**: Optimize for your specific GPU architecture
2. **Memory Management**: Implement buffer pooling
3. **Batch Processing**: Group small vectors for better GPU utilization
4. **Pipeline Optimization**: Overlap CPU/GPU work

## Support

### Development Commands
```bash
# Environment check
npm run orchestrator:check-env

# Health monitoring  
npm run orchestrator:health

# Build CUDA worker
npm run orchestrator:build-cuda

# Start services
npm run orchestrator:start
npm run orchestrator:redis-service

# Test publishing
npm run orchestrator:test-job help
```

### Production Deployment
- Use PM2 or Windows Service for orchestrator
- Configure RabbitMQ clustering
- Setup Redis persistence
- Monitor GPU temperature/memory
- Implement backup/recovery

🎉 **You now have a complete native Windows GPU orchestration system!**

The scaffold provides the foundation for building production AI/ML workloads with CUDA acceleration, message queuing, and real-time processing capabilities.