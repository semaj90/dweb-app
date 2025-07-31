/**
 * Cache Service with Redis
 * Provides caching layer for RAG operations with intelligent TTL
 */

import { createClient } from 'redis';

export class CacheService {
  constructor(config) {
    this.config = config;
    this.client = null;
    this.connected = false;
  }

  async initialize() {
    try {
      this.client = createClient({
        url: this.config.url,
        retry_unfulfilled_commands: true,
        socket: {
          reconnectStrategy: (retries) => Math.min(retries * 50, 1000)
        }
      });

      this.client.on('error', (err) => {
        console.error('Redis Client Error:', err);
        this.connected = false;
      });

      this.client.on('connect', () => {
        console.log('✅ Redis client connected');
        this.connected = true;
      });

      this.client.on('disconnect', () => {
        console.log('⚠️ Redis client disconnected');
        this.connected = false;
      });

      await this.client.connect();
      return true;
    } catch (error) {
      console.error('Failed to connect to Redis:', error);
      return false;
    }
  }

  /**
   * Generate cache key for different types of data
   */
  generateKey(type, ...params) {
    const sanitized = params.map(p => 
      typeof p === 'object' ? JSON.stringify(p) : String(p)
    ).join(':');
    return `rag:${type}:${sanitized}`;
  }

  /**
   * Set cache with TTL
   */
  async set(key, value, ttlSeconds = 3600) {
    if (!this.connected || !this.client) {
      console.warn('Cache not available, skipping set');
      return false;
    }

    try {
      const serialized = typeof value === 'object' ? JSON.stringify(value) : String(value);
      await this.client.setEx(key, ttlSeconds, serialized);
      return true;
    } catch (error) {
      console.error('Cache set failed:', error);
      return false;
    }
  }

  /**
   * Get from cache
   */
  async get(key) {
    if (!this.connected || !this.client) {
      return null;
    }

    try {
      const value = await this.client.get(key);
      if (value === null) return null;

      // Try to parse as JSON, fallback to string
      try {
        return JSON.parse(value);
      } catch {
        return value;
      }
    } catch (error) {
      console.error('Cache get failed:', error);
      return null;
    }
  }

  /**
   * Delete from cache
   */
  async delete(key) {
    if (!this.connected || !this.client) {
      return false;
    }

    try {
      await this.client.del(key);
      return true;
    } catch (error) {
      console.error('Cache delete failed:', error);
      return false;
    }
  }

  /**
   * Cache search results
   */
  async cacheSearchResults(query, filters, results, ttl = 1800) {
    const key = this.generateKey('search', query, filters);
    return await this.set(key, {
      query,
      filters,
      results,
      timestamp: Date.now()
    }, ttl);
  }

  /**
   * Get cached search results
   */
  async getCachedSearchResults(query, filters) {
    const key = this.generateKey('search', query, filters);
    return await this.get(key);
  }

  /**
   * Cache document embedding
   */
  async cacheEmbedding(text, embedding, ttl = 86400) {
    const key = this.generateKey('embedding', text);
    return await this.set(key, {
      text,
      embedding,
      timestamp: Date.now()
    }, ttl);
  }

  /**
   * Get cached embedding
   */
  async getCachedEmbedding(text) {
    const key = this.generateKey('embedding', text);
    return await this.get(key);
  }

  /**
   * Cache document processing results
   */
  async cacheDocumentProcessing(documentId, processingResult, ttl = 7200) {
    const key = this.generateKey('processing', documentId);
    return await this.set(key, {
      documentId,
      result: processingResult,
      timestamp: Date.now()
    }, ttl);
  }

  /**
   * Get cached document processing results
   */
  async getCachedDocumentProcessing(documentId) {
    const key = this.generateKey('processing', documentId);
    return await this.get(key);
  }

  /**
   * Cache agent orchestration results
   */
  async cacheAgentResults(prompt, context, results, ttl = 3600) {
    const key = this.generateKey('agent', prompt, context);
    return await this.set(key, {
      prompt,
      context,
      results,
      timestamp: Date.now()
    }, ttl);
  }

  /**
   * Get cached agent results
   */
  async getCachedAgentResults(prompt, context) {
    const key = this.generateKey('agent', prompt, context);
    return await this.get(key);
  }

  /**
   * Cache health statistics
   */
  async cacheHealthStats(stats, ttl = 300) {
    const key = this.generateKey('health', 'stats');
    return await this.set(key, {
      stats,
      timestamp: Date.now()
    }, ttl);
  }

  /**
   * Get cached health statistics
   */
  async getCachedHealthStats() {
    const key = this.generateKey('health', 'stats');
    return await this.get(key);
  }

  /**
   * Increment counter with expiration
   */
  async incrementCounter(key, ttl = 3600) {
    if (!this.connected || !this.client) {
      return 0;
    }

    try {
      const fullKey = this.generateKey('counter', key);
      const count = await this.client.incr(fullKey);
      
      // Set expiration on first increment
      if (count === 1) {
        await this.client.expire(fullKey, ttl);
      }
      
      return count;
    } catch (error) {
      console.error('Counter increment failed:', error);
      return 0;
    }
  }

  /**
   * Get counter value
   */
  async getCounter(key) {
    if (!this.connected || !this.client) {
      return 0;
    }

    try {
      const fullKey = this.generateKey('counter', key);
      const value = await this.client.get(fullKey);
      return value ? parseInt(value) : 0;
    } catch (error) {
      console.error('Get counter failed:', error);
      return 0;
    }
  }

  /**
   * Store multiple values with pipeline
   */
  async setMultiple(entries, ttl = 3600) {
    if (!this.connected || !this.client) {
      return false;
    }

    try {
      const pipeline = this.client.multi();
      
      for (const [key, value] of entries) {
        const serialized = typeof value === 'object' ? JSON.stringify(value) : String(value);
        pipeline.setEx(key, ttl, serialized);
      }
      
      await pipeline.exec();
      return true;
    } catch (error) {
      console.error('Multiple set failed:', error);
      return false;
    }
  }

  /**
   * Get multiple values
   */
  async getMultiple(keys) {
    if (!this.connected || !this.client) {
      return {};
    }

    try {
      const values = await this.client.mGet(keys);
      const result = {};
      
      keys.forEach((key, index) => {
        const value = values[index];
        if (value !== null) {
          try {
            result[key] = JSON.parse(value);
          } catch {
            result[key] = value;
          }
        }
      });
      
      return result;
    } catch (error) {
      console.error('Multiple get failed:', error);
      return {};
    }
  }

  /**
   * Clear cache by pattern
   */
  async clearByPattern(pattern) {
    if (!this.connected || !this.client) {
      return 0;
    }

    try {
      const keys = await this.client.keys(pattern);
      if (keys.length === 0) return 0;
      
      await this.client.del(keys);
      return keys.length;
    } catch (error) {
      console.error('Clear by pattern failed:', error);
      return 0;
    }
  }

  /**
   * Get cache statistics
   */
  async getStats() {
    if (!this.connected || !this.client) {
      return null;
    }

    try {
      const info = await this.client.info('memory');
      const stats = await this.client.info('stats');
      
      return {
        connected: this.connected,
        memory: this.parseRedisInfo(info),
        stats: this.parseRedisInfo(stats)
      };
    } catch (error) {
      console.error('Get cache stats failed:', error);
      return null;
    }
  }

  /**
   * Parse Redis INFO command output
   */
  parseRedisInfo(info) {
    const lines = info.split('\r\n');
    const result = {};
    
    for (const line of lines) {
      if (line.includes(':')) {
        const [key, value] = line.split(':');
        result[key] = isNaN(value) ? value : Number(value);
      }
    }
    
    return result;
  }

  /**
   * Close Redis connection
   */
  async close() {
    if (this.client && this.connected) {
      try {
        await this.client.quit();
        console.log('✅ Redis connection closed');
      } catch (error) {
        console.error('Error closing Redis connection:', error);
      }
    }
  }
}