/**
 * Unified Service Registry with Redis Caching Layer
 * Handles service discovery, health monitoring, and hot query caching
 * Prevents bundling Neo4j locally - uses remote caching strategy
 */

import { browser } from '$app/environment';

export interface ServiceStatus {
  name: string;
  online: boolean;
  responseTime?: number;
  lastChecked: Date;
  version?: string;
  metadata?: Record<string, any>;
}

export interface SystemStatus {
  services: Record<string, ServiceStatus>;
  cached: boolean;
  lastUpdate: Date;
  healthScore: number;
}

export interface HotGraphQuery {
  query: string;
  result: any;
  timestamp: Date;
  ttl: number;
  hitCount: number;
}

class UnifiedServiceRegistry {
  private redis: any = null;
  private statusCache = new Map<string, SystemStatus>();
  private graphCache = new Map<string, HotGraphQuery>();

  constructor() {
    // Only initialize Redis connection in server environment
    if (!browser && typeof process !== 'undefined') {
      this.initializeRedis();
    }
  }

  private async initializeRedis() {
    try {
      // Dynamic import to avoid bundling Redis in browser
      const Redis = await import('ioredis').then(m => m.default);
      this.redis = new Redis({
        host: 'localhost',
        port: 6379,
        retryDelayOnFailover: 100,
        enableOfflineQueue: false,
        lazyConnect: true,
        maxRetriesPerRequest: 2
      });
      
      console.log('✅ Unified Service Registry: Redis connected');
    } catch (error) {
      console.warn('⚠️ Unified Service Registry: Redis not available, using memory cache');
    }
  }

  /**
   * Get comprehensive system status with Redis caching
   */
  async getSystemStatus(useCache = true): Promise<SystemStatus> {
    const cacheKey = 'system:unified:status';
    
    // Try cache first if enabled
    if (useCache) {
      const cached = await this.getCachedStatus(cacheKey);
      if (cached) {
        return cached;
      }
    }

    // Perform live health checks
    const status = await this.performHealthChecks();
    
    // Cache the result
    await this.cacheStatus(cacheKey, status, 10); // 10 second TTL
    
    return status;
  }

  private async getCachedStatus(cacheKey: string): Promise<SystemStatus | null> {
    try {
      if (this.redis) {
        const cached = await this.redis.get(cacheKey);
        if (cached) {
          const parsed = JSON.parse(cached);
          return {
            ...parsed,
            lastUpdate: new Date(parsed.lastUpdate),
            cached: true
          };
        }
      } else {
        // Fallback to memory cache
        const cached = this.statusCache.get(cacheKey);
        if (cached && Date.now() - cached.lastUpdate.getTime() < 10000) {
          return { ...cached, cached: true };
        }
      }
    } catch (error) {
      console.warn('UnifiedServiceRegistry: Cache read failed:', error);
    }
    return null;
  }

  private async cacheStatus(cacheKey: string, status: SystemStatus, ttl: number) {
    try {
      if (this.redis) {
        await this.redis.setex(cacheKey, ttl, JSON.stringify(status));
      } else {
        // Fallback to memory cache
        this.statusCache.set(cacheKey, status);
      }
    } catch (error) {
      console.warn('UnifiedServiceRegistry: Cache write failed:', error);
    }
  }

  private async performHealthChecks(): Promise<SystemStatus> {
    const services: Record<string, ServiceStatus> = {};

    // Core services to check
    const serviceChecks = [
      { name: 'redis', check: () => this.checkRedis() },
      { name: 'postgres', check: () => this.checkPostgreSQL() },
      { name: 'ollama', check: () => this.checkOllama() },
      { name: 'qdrant', check: () => this.checkQdrant() },
      { name: 'wasm_graph', check: () => this.checkWasmGraph() }
    ];

    // Run checks in parallel
    const results = await Promise.allSettled(
      serviceChecks.map(async ({ name, check }) => ({
        name,
        status: await check()
      }))
    );

    // Process results
    results.forEach((result, index) => {
      const serviceName = serviceChecks[index].name;
      if (result.status === 'fulfilled') {
        services[serviceName] = result.value.status;
      } else {
        services[serviceName] = {
          name: serviceName,
          online: false,
          lastChecked: new Date(),
          metadata: { error: result.reason?.message || 'Unknown error' }
        };
      }
    });

    // Calculate health score
    const totalServices = Object.keys(services).length;
    const onlineServices = Object.values(services).filter(s => s.online).length;
    const healthScore = Math.round((onlineServices / totalServices) * 100);

    return {
      services,
      cached: false,
      lastUpdate: new Date(),
      healthScore
    };
  }

  /**
   * Cache hot graph queries in Redis and Qdrant/PostgreSQL
   * Avoids bundling Neo4j locally
   */
  async cacheGraphQuery(query: string, result: any, ttl = 300): Promise<void> {
    const cacheKey = `graph:query:${this.hashQuery(query)}`;
    const hotQuery: HotGraphQuery = {
      query,
      result,
      timestamp: new Date(),
      ttl,
      hitCount: 1
    };

    try {
      if (this.redis) {
        await this.redis.setex(cacheKey, ttl, JSON.stringify(hotQuery));
      }
      
      // Also cache in memory for immediate access
      this.graphCache.set(cacheKey, hotQuery);
      
    } catch (error) {
      console.warn('UnifiedServiceRegistry: Graph cache failed:', error);
    }
  }

  async getGraphQuery(query: string): Promise<HotGraphQuery | null> {
    const cacheKey = `graph:query:${this.hashQuery(query)}`;
    
    try {
      // Check memory first
      const memoryResult = this.graphCache.get(cacheKey);
      if (memoryResult && Date.now() - memoryResult.timestamp.getTime() < memoryResult.ttl * 1000) {
        memoryResult.hitCount++;
        return memoryResult;
      }

      // Check Redis
      if (this.redis) {
        const cached = await this.redis.get(cacheKey);
        if (cached) {
          const parsed: HotGraphQuery = JSON.parse(cached);
          parsed.timestamp = new Date(parsed.timestamp);
          parsed.hitCount++;
          
          // Update memory cache
          this.graphCache.set(cacheKey, parsed);
          return parsed;
        }
      }
      
    } catch (error) {
      console.warn('UnifiedServiceRegistry: Graph query retrieval failed:', error);
    }
    
    return null;
  }

  private hashQuery(query: string): string {
    // Simple hash function for cache keys
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  private async checkRedis(): Promise<ServiceStatus> {
    const startTime = Date.now();
    try {
      if (this.redis) {
        await this.redis.ping();
        return {
          name: 'redis',
          online: true,
          responseTime: Date.now() - startTime,
          lastChecked: new Date()
        };
      }
      throw new Error('Redis not initialized');
    } catch (error) {
      return {
        name: 'redis',
        online: false,
        responseTime: Date.now() - startTime,
        lastChecked: new Date(),
        metadata: { error: (error as Error).message }
      };
    }
  }

  private async checkPostgreSQL(): Promise<ServiceStatus> {
    const startTime = Date.now();
    try {
      const response = await fetch('/api/sync-config');
      if (response.ok) {
        const data = await response.json();
        const pgStatus = data.services?.postgresql;
        
        return {
          name: 'postgres',
          online: pgStatus?.status === 'connected',
          responseTime: Date.now() - startTime,
          lastChecked: new Date(),
          metadata: pgStatus?.details || {}
        };
      }
      throw new Error(`HTTP ${response.status}`);
    } catch (error) {
      return {
        name: 'postgres',
        online: false,
        responseTime: Date.now() - startTime,
        lastChecked: new Date(),
        metadata: { error: (error as Error).message }
      };
    }
  }

  private async checkOllama(): Promise<ServiceStatus> {
    const startTime = Date.now();
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        return {
          name: 'ollama',
          online: true,
          responseTime: Date.now() - startTime,
          lastChecked: new Date()
        };
      }
      throw new Error(`HTTP ${response.status}`);
    } catch (error) {
      return {
        name: 'ollama',
        online: false,
        responseTime: Date.now() - startTime,
        lastChecked: new Date(),
        metadata: { error: (error as Error).message }
      };
    }
  }

  private async checkQdrant(): Promise<ServiceStatus> {
    const startTime = Date.now();
    try {
      const response = await fetch('http://localhost:6333/collections');
      if (response.ok) {
        return {
          name: 'qdrant',
          online: true,
          responseTime: Date.now() - startTime,
          lastChecked: new Date()
        };
      }
      throw new Error(`HTTP ${response.status}`);
    } catch (error) {
      return {
        name: 'qdrant',
        online: false,
        responseTime: Date.now() - startTime,
        lastChecked: new Date(),
        metadata: { error: (error as Error).message }
      };
    }
  }

  private async checkWasmGraph(): Promise<ServiceStatus> {
    const startTime = Date.now();
    try {
      const wasmLoaded = globalThis.__WASM_GRAPH_ENGINE__ !== undefined;
      
      return {
        name: 'wasm_graph',
        online: wasmLoaded,
        responseTime: Date.now() - startTime,
        lastChecked: new Date(),
        metadata: { 
          engine: wasmLoaded ? 'loaded' : 'not_loaded',
          cacheSize: this.graphCache.size
        }
      };
    } catch (error) {
      return {
        name: 'wasm_graph',
        online: false,
        responseTime: Date.now() - startTime,
        lastChecked: new Date(),
        metadata: { error: (error as Error).message }
      };
    }
  }

  /**
   * Get hot queries for WASM graph engine hydration
   */
  async getHotQueries(limit = 20): Promise<HotGraphQuery[]> {
    const queries = Array.from(this.graphCache.values())
      .sort((a, b) => b.hitCount - a.hitCount)
      .slice(0, limit);
    
    return queries;
  }

  getCacheStats() {
    return {
      statusCache: this.statusCache.size,
      graphCache: this.graphCache.size,
      redisConnected: !!this.redis,
      totalQueries: Array.from(this.graphCache.values()).reduce((sum, q) => sum + q.hitCount, 0)
    };
  }
}

// Export singleton instance
export const unifiedServiceRegistry = new UnifiedServiceRegistry();