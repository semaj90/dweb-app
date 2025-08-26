# Enterprise Vector Consumer Service v2.0 - Native Windows Deployment

## 🚀 **Complete Native Windows Setup (No Docker)**

### **📋 Prerequisites**

1. **Go 1.21+** - Native Windows installation
2. **PostgreSQL 15+** with pgvector extension
3. **Redis** - Native Windows installation  
4. **CUDA Toolkit 12.0+** (optional, for GPU acceleration)
5. **Protocol Buffers Compiler** (`protoc`)

### **🔧 Build Process**

#### **1. Quick Build & Start**
```cmd
REM Build the service
build-vector-consumer-v2.bat

REM Start the service
start-vector-consumer-v2.bat
```

#### **2. Manual Build Steps**
```cmd
REM Set environment for native Windows
set CGO_ENABLED=1
set GOOS=windows
set GOARCH=amd64

REM Generate protobuf code
cd proto
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative aiserver.proto
cd ..

REM Build with CUDA support
go build -o bin\vector-consumer-v2.exe .\cmd\vector-consumer-v2\
```

### **🎯 Service Architecture**

#### **Enterprise Components Integrated:**
- ✅ **gRPC/Protobuf** - High-performance inter-service communication
- ✅ **Enhanced CUDA Worker** - cuBLAS mathematical precision
- ✅ **sqlc Database Layer** - Type-safe PostgreSQL operations
- ✅ **Multi-layer Caching** - L1 (Memory) + L2 (Redis) + L3 (PostgreSQL JSONB)
- ✅ **Kratos Authentication** - Enterprise identity management
- ✅ **ELK Stack Observability** - Structured logging for production

### **📡 API Endpoints**

#### **Vector Operations**
```bash
# Process vector rotation with cuBLAS precision
grpc://localhost:8095/VectorService/ProcessRotation

# Compute similarity with CUDA acceleration
grpc://localhost:8095/VectorService/ProcessSimilarity

# Process legal documents with NLP
grpc://localhost:8095/VectorService/ProcessLegalDocument
```

#### **Health & Monitoring**
```bash
# Service health check
grpc://localhost:8095/grpc.health.v1.Health/Check

# Performance metrics via observability
grpc://localhost:8095/HealthService/GetMetrics
```

### **⚡ Performance Configuration**

#### **CUDA Optimization (RTX 3060 Ti)**
```cmd
REM Enable CUDA with optimized memory usage
bin\vector-consumer-v2.exe ^
    --cuda=true ^
    --port=8095 ^
    --max-concurrency=1000
```

#### **CPU-Only Mode**
```cmd
REM Fallback to CPU processing
bin\vector-consumer-v2.exe ^
    --cuda=false ^
    --port=8095 ^
    --max-concurrency=500
```

### **🔐 Security Integration**

#### **Kratos Authentication**
```yaml
# Kratos configuration for Windows
kratos_url: http://localhost:4433
identity_validation: enabled
rbac_enforcement: strict
session_caching: enabled
```

#### **Database Security**
```sql
-- PostgreSQL connection with SSL
postgres://legal_admin:123456@localhost:5432/legal_ai_db?sslmode=require
```

### **📊 Monitoring & Observability**

#### **ELK Stack Integration**
```json
{
  "service_name": "vector-consumer-v2",
  "environment": "production",
  "elasticsearch_endpoint": "http://localhost:9200",
  "structured_logging": true,
  "performance_metrics": true
}
```

#### **Real-time Metrics**
- Request throughput and latency
- CUDA GPU utilization and memory usage
- Cache hit ratios (L1/L2/L3)
- Database connection pool stats
- Authentication success/failure rates

### **🗄️ Database Schema**

#### **Vector Operations Tracking**
```sql
CREATE TABLE vector_operations (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    input_dimensions INTEGER,
    output_dimensions INTEGER,
    processing_time_ms BIGINT,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### **Multi-layer Cache Tables**
```sql
CREATE TABLE cache_entries (
    cache_key VARCHAR(512) PRIMARY KEY,
    cache_value JSONB NOT NULL,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cache_expires ON cache_entries(expires_at);
CREATE INDEX idx_cache_created ON cache_entries(created_at);
```

### **🚀 Production Deployment**

#### **Service Registration**
```cmd
REM Install as Windows service
sc create VectorConsumerV2 binPath= "C:\path\to\bin\vector-consumer-v2.exe --port=8095"
sc start VectorConsumerV2
```

#### **Load Balancing**
```yaml
# Multiple instances for high availability
instances:
  - port: 8095
    cuda_device: 0
    max_memory: 6GB
  - port: 8096  
    cuda_device: 0
    max_memory: 2GB
  - port: 8097
    cuda_enabled: false  # CPU-only backup
```

### **📈 Performance Benchmarks**

#### **Expected Performance (RTX 3060 Ti)**
- **Vector Rotation**: < 5ms (768-dimensional vectors)
- **Cosine Similarity**: < 2ms (cuBLAS precision)
- **Legal Document Processing**: < 50ms (average document)
- **Cache Hit Ratio**: > 85% (multi-layer)
- **Database Queries**: < 10ms (pgvector similarity)

#### **Throughput Targets**
- **Concurrent Requests**: 1,000+ (with CUDA)
- **Requests/Second**: 10,000+ (cached operations)
- **GPU Memory Usage**: 6GB/8GB (optimized allocation)

### **🛠️ Troubleshooting**

#### **Common Issues**

1. **CUDA Not Available**
   ```cmd
   REM Check NVIDIA drivers
   nvidia-smi
   
   REM Verify CUDA installation
   nvcc --version
   ```

2. **Database Connection Failed**
   ```cmd
   REM Test PostgreSQL connection
   psql postgres://legal_admin:123456@localhost:5432/legal_ai_db -c "SELECT 1;"
   ```

3. **Redis Connection Failed**
   ```cmd
   REM Test Redis connection
   redis-cli -h localhost -p 6379 ping
   ```

4. **Port Already in Use**
   ```cmd
   REM Find process using port
   netstat -ano | findstr :8095
   
   REM Kill process if needed
   taskkill /PID [PID_NUMBER] /F
   ```

### **📝 Service Logs**

#### **Log Locations**
- **Application Logs**: `logs\vector-consumer-v2.log`
- **Error Logs**: `logs\error.log`
- **Performance Logs**: `logs\performance.log`
- **Audit Logs**: `logs\audit.log`

#### **Log Levels**
```cmd
REM Debug mode for development
bin\vector-consumer-v2.exe --log-level=debug

REM Production logging
bin\vector-consumer-v2.exe --log-level=info
```

### **✅ Verification Steps**

#### **1. Service Health Check**
```cmd
REM Check service status
curl -X POST http://localhost:8095/grpc.health.v1.Health/Check
```

#### **2. CUDA Functionality**
```cmd
REM Test CUDA vector operations
grpcurl -plaintext -d '{"vector": [1,2,3,4], "rotation_matrix": [0.9,0.1,0.1,0.9]}' localhost:8095 VectorService/ProcessRotation
```

#### **3. Database Integration**
```cmd
REM Verify database operations
psql postgres://legal_admin:123456@localhost:5432/legal_ai_db -c "SELECT COUNT(*) FROM vector_operations;"
```

### **🎉 Deployment Complete**

Your **Enterprise Vector Consumer Service v2.0** is now running natively on Windows with:

- ✅ **High-Performance gRPC** communication
- ✅ **CUDA-accelerated** vector operations
- ✅ **Enterprise-grade security** with Kratos
- ✅ **Multi-layer caching** for optimal performance
- ✅ **Production observability** with ELK integration
- ✅ **Type-safe database** operations with sqlc
- ✅ **Native Windows** deployment (no Docker dependency)

**Service Endpoint**: `grpc://localhost:8095`  
**Health Check**: Available via gRPC health service  
**Performance**: Optimized for RTX 3060 Ti with 6GB VRAM allocation  
**Security**: Kratos-authenticated with RBAC enforcement