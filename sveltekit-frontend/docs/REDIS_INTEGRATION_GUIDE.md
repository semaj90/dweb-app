# Redis Integration Guide for Legal AI Platform

## 🎯 **Complete Redis Setup & Optimization**

This guide shows how to integrate Redis across your Legal AI platform with optimized configurations, concurrency patterns, and service worker integration.

---

## 📁 **File Structure Overview**

```
sveltekit-frontend/
├── redis.conf                           # Redis server configuration
├── scripts/start-redis-with-config.bat  # Windows Redis startup script
├── src/lib/config/redis-config.ts       # Centralized Redis configuration
├── src/lib/utils/redis-helper.ts        # Enhanced Redis connection utilities
├── src/lib/server/redisRateLimit.ts     # Rate limiting with Lua scripts
├── src/lib/server/cache/redis-service.ts # Main Redis service
├── src/lib/optimization/redis-som-cache.ts # SOM cache integration
├── src/lib/cache/loki-redis-integration.ts # Loki.js + Redis hybrid
└── src/lib/shims/ioredis-browser-shim.js # Browser compatibility layer
```

---

## 🚀 **Quick Start**

### 1. Start Redis with Optimized Configuration

```bash
# Using npm script
npm run redis:start

# Or directly
scripts\start-redis-with-config.bat

# Check health
npm run redis:health
```

### 2. Environment Configuration

```bash
# .env.local
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=            # Leave empty for development
REDIS_DB=0
REDIS_CONFIG_PATH=redis.conf
```

### 3. Import and Use in Your Code

```typescript
// For server-side code
import { getRedisClient, setupRedisFromConfig } from '$lib/utils/redis-helper';
import { redis } from '$lib/server/cache/redis-service';

// For browser-side code (automatically uses shim)
import RedisShim from '$lib/shims/ioredis-browser-shim.js';
```

---

## ⚙️ **Configuration Architecture**

### **Centralized Configuration (`src/lib/config/redis-config.ts`)**

```typescript
// Service-specific configurations
const configs = {
  MAIN_CACHE: { db: 0, keyPrefix: 'legal_ai:' },
  RATE_LIMIT: { db: 2, keyPrefix: 'rate:' },
  LOKI_CACHE: { db: 3, keyPrefix: 'loki:' },
  GPU_CACHE: { db: 4, keyPrefix: 'gpu:' },
  WORKER_QUEUE: { db: 6, keyPrefix: 'worker:' }
};

// Usage
import { createServiceConfig } from '$lib/config/redis-config';
const rateLimit = new Redis(createServiceConfig('RATE_LIMIT'));
```

### **Database Separation Strategy**

| Database | Purpose | Key Prefix | TTL Strategy |
|----------|---------|------------|--------------|
| DB 0 | General caching | `legal_ai:` | Variable (5min - 24hrs) |
| DB 1 | User sessions | `session:` | 24 hours |
| DB 2 | Rate limiting | `rate:` | 1 hour |
| DB 3 | Loki.js integration | `loki:` | 30 minutes |
| DB 4 | GPU cache orchestration | `gpu:` | 12 hours |
| DB 5 | NATS messaging cache | `nats:` | 10 minutes |
| DB 6 | Worker queues | `worker:` | 1 hour |

---

## 🔧 **Service Integration Examples**

### **1. Rate Limiting with Lua Scripts**

```typescript
// src/lib/server/redisRateLimit.ts
import { redisRateLimit } from '$lib/server/redisRateLimit';

export async function handleRateLimit(identifier: string) {
  const result = await redisRateLimit({
    key: identifier,
    limit: 100,        // 100 requests
    windowSec: 3600    // per hour
  });
  
  return {
    allowed: result.allowed,
    remaining: result.limit - result.count,
    resetTime: result.retryAfter
  };
}
```

### **2. Hybrid Loki.js + Redis Caching**

```typescript
// src/lib/cache/loki-redis-integration.ts
import { lokiRedisCache } from '$lib/cache/loki-redis-integration';

// Store legal document with multi-tier caching
await lokiRedisCache.storeDocument({
  id: 'doc-123',
  type: 'contract',
  content: documentText,
  metadata: { riskLevel: 'high', priority: 200 }
});

// Retrieve with intelligent cache promotion
const document = await lokiRedisCache.getDocument('doc-123');
```

### **3. GPU Cache Orchestration**

```typescript
// src/lib/optimization/redis-som-cache.ts
import { createDockerOptimizedCache } from '$lib/optimization/redis-som-cache';

const cache = createDockerOptimizedCache();

// Store with neural network clustering
await cache.set('gpu-tensor-123', tensorData, {
  metadata: { 
    ai_relevance: 0.9,
    access_pattern: 'frequent'
  }
});

// Analyze access patterns with ML
const analysis = await cache.analyzeAccessPatterns();
console.log('ML Recommendations:', analysis.recommendations);
```

---

## 🌐 **Browser Integration**

### **Service Worker Pattern**

```typescript
// src/lib/shims/ioredis-browser-shim.js - Usage
const redis = new RedisShim({
  keyPrefix: 'legal_ai:',
  enableOfflineMode: true,
  useServiceWorker: true
});

// Cross-tab communication
await redis.publish('case_update', { caseId: '123', status: 'updated' });

// Subscribe to updates
await redis.subscribe('case_update', (channel, message) => {
  console.log('Case updated:', JSON.parse(message));
});

// Performance monitoring
const stats = redis.getStats();
console.log(`Hit rate: ${stats.hitRate}%`);
```

### **SSR Integration**

```typescript
// src/hooks.server.ts
import { setupRedisFromConfig } from '$lib/utils/redis-helper';

export const handle = async ({ event, resolve }) => {
  // Ensure Redis is connected on server startup
  if (!globalThis.redisInitialized) {
    await setupRedisFromConfig();
    globalThis.redisInitialized = true;
  }
  
  return resolve(event);
};
```

---

## ⚡ **Performance Optimizations**

### **Redis Server Configuration (redis.conf)**

```ini
# Memory optimization for Legal AI workloads
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Performance tuning
tcp-keepalive 60
tcp-backlog 511
hz 10

# Persistence for legal document safety
appendonly yes
appendfsync everysec
save 300 1

# Client buffer limits for streaming
client-output-buffer-limit pubsub 32mb 8mb 60
```

### **Connection Pooling**

```typescript
// Production connection pool configuration
const poolConfig = {
  production: {
    min: 5,
    max: 50,
    acquireTimeoutMillis: 60000,
    idleTimeoutMillis: 60000,
    enableAutoPipelining: true
  }
};
```

### **Lua Script Optimization**

```typescript
// Atomic operations with Lua scripts
const scripts = {
  RATE_LIMIT: `
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
    redis.call('ZADD', key, now, now)
    local count = redis.call('ZCARD', key)
    redis.call('PEXPIRE', key, window)
    
    return { count <= limit and 1 or 0, count }
  `
};
```

---

## 🔄 **Concurrency Patterns**

### **Worker Queue Pattern**

```typescript
// Worker producer
async function enqueueJob(jobData: any) {
  const redis = getRedisClient();
  await redis.lpush('worker:jobs', JSON.stringify({
    ...jobData,
    id: crypto.randomUUID(),
    timestamp: Date.now()
  }));
}

// Worker consumer (with row-level locking pattern)
async function processJobs() {
  const redis = getRedisClient();
  
  while (true) {
    const job = await redis.brpop('worker:jobs', 10); // 10 second timeout
    
    if (job) {
      const [queueName, jobData] = job;
      const parsed = JSON.parse(jobData);
      
      try {
        await processJob(parsed);
        console.log(`✅ Job ${parsed.id} completed`);
      } catch (error) {
        // Dead letter queue
        await redis.lpush('worker:failed', jobData);
        console.error(`❌ Job ${parsed.id} failed:`, error);
      }
    }
  }
}
```

### **PostgreSQL + Redis Coordination**

```typescript
// Combined PostgreSQL + Redis pattern for legal documents
async function storeDocument(document: LegalDocument) {
  // 1. Store in PostgreSQL (source of truth)
  const result = await db.insert(documents).values(document).returning();
  
  // 2. Cache in Redis for fast access
  await redis.setex(
    `doc:${result.id}`, 
    3600, // 1 hour TTL
    JSON.stringify(result)
  );
  
  // 3. Publish event for workers
  await redis.publish('document:created', JSON.stringify({
    documentId: result.id,
    type: 'legal_document',
    timestamp: Date.now()
  }));
  
  return result;
}
```

---

## 📊 **Monitoring & Health Checks**

### **Health Check API**

```typescript
// src/routes/api/redis/health/+server.ts
import { checkRedisHealth, getRedisInfo } from '$lib/utils/redis-helper';

export async function GET() {
  const isHealthy = await checkRedisHealth();
  const info = await getRedisInfo();
  
  return json({
    status: isHealthy ? 'healthy' : 'unhealthy',
    info: info.connected ? info : null,
    timestamp: new Date().toISOString()
  });
}
```

### **Performance Metrics**

```bash
# Check Redis performance
curl http://localhost:5173/api/redis/health

# Monitor Redis in real-time
npm run redis:monitor

# Get Redis configuration
npm run redis:config
```

---

## 🎯 **Integration with Go Services**

### **Multi-Protocol Service Integration**

```typescript
// Service routing with Redis coordination
const serviceRouting = {
  'rag.query': { 
    tier: 'QUIC',           // < 5ms
    endpoint: 'rag-quic-proxy:8216',
    cache: { ttl: 300, db: 0 }
  },
  'legal.process': { 
    tier: 'gRPC',           // < 15ms  
    endpoint: 'kratos-server:50051',
    cache: { ttl: 1800, db: 0 }
  },
  'file.upload': { 
    tier: 'HTTP',           // < 50ms
    endpoint: 'upload-service:8093',
    cache: { ttl: 3600, db: 0 }
  }
};
```

### **Service Discovery with Redis**

```typescript
// Register Go service with Redis
async function registerService(serviceName: string, port: number) {
  const redis = getRedisClient();
  const serviceKey = `services:${serviceName}`;
  
  await redis.hset(serviceKey, {
    port,
    status: 'healthy',
    lastSeen: Date.now(),
    pid: process.pid
  });
  
  // TTL for automatic cleanup
  await redis.expire(serviceKey, 60); // 60 seconds
}
```

---

## 🛠 **Troubleshooting**

### **Common Issues & Solutions**

1. **Redis Connection Refused**
   ```bash
   # Start Redis with config
   npm run redis:start
   
   # Check if running
   npm run redis:health
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   redis-cli info memory
   
   # Clear specific database
   redis-cli -n 0 flushdb
   ```

3. **Script Errors**
   ```typescript
   // Reset Lua script cache
   if (error.message.includes('NOSCRIPT')) {
     sha = null; // Force script reload
   }
   ```

4. **Browser Shim Issues**
   ```typescript
   // Check localStorage quota
   const stats = redisShim.getStats();
   console.log('Storage usage:', stats.storage);
   
   // Cleanup if needed
   await redisShim.cleanup();
   ```

---

## 🚀 **Production Deployment**

### **Production Checklist**

- [ ] Configure Redis password authentication
- [ ] Set up Redis persistence (AOF + RDB)
- [ ] Configure memory limits and eviction policies
- [ ] Set up Redis monitoring and alerting
- [ ] Configure backup strategy
- [ ] Set up Redis Cluster for high availability
- [ ] Configure network security (bind, protected-mode)

### **Docker Alternative (if needed)**

```yaml
# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf
    volumes:
      - ./redis.conf:/usr/local/etc/redis/redis.conf
      - redis_data:/data
    ports:
      - "6379:6379"
    
volumes:
  redis_data:
```

---

## 📈 **Performance Benchmarks**

| Operation | Redis Server | Browser Shim | Notes |
|-----------|-------------|--------------|-------|
| GET | < 1ms | < 5ms | LocalStorage lookup |
| SET | < 1ms | < 10ms | JSON serialization |
| Pub/Sub | < 5ms | < 20ms | BroadcastChannel |
| Lua Script | < 2ms | N/A | Server-side only |
| TTL Check | < 1ms | < 5ms | Timestamp comparison |

---

## 🎯 **Best Practices Summary**

1. **Use centralized configuration** (`redis-config.ts`)
2. **Separate databases** by service/purpose
3. **Implement graceful degradation** in browser shim
4. **Use Lua scripts** for atomic operations
5. **Monitor performance** with built-in metrics
6. **Configure appropriate TTLs** for different data types
7. **Use connection pooling** in production
8. **Implement health checks** and monitoring
9. **Plan for offline mode** in browser environments
10. **Test Redis failover** scenarios

This integration provides a robust, performant Redis setup that scales from development through production deployment while maintaining compatibility across server and browser environments.