/**
 * Redis Caching Layer with Session Management
 * Production-ready Redis service for caching and session storage
 */

import { Redis, type RedisOptions } from 'ioredis';
import { dev } from '$app/environment';

// Redis configuration
const REDIS_CONFIG: RedisOptions = {
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD || undefined,
  db: parseInt(process.env.REDIS_DB || '0'),
  
  // Connection settings
  connectTimeout: 10000,
  commandTimeout: 5000,
  maxRetriesPerRequest: 3,
  
  // Connection pool
  lazyConnect: true,
  keepAlive: 30000,
  family: 4,
  
  // Reconnection strategy - stop after 3 attempts to avoid spam
  retryStrategy: (times: number) => {
    if (times > 3) {
      console.log('Redis: Max reconnection attempts reached, stopping retries');
      return null; // Stop retrying
    }
    const delay = Math.min(times * 1000, 3000);
    return delay;
  },
  
  // Development settings
  ...(dev && {
    maxRetriesPerRequest: 1,
    connectTimeout: 5000,
  })
};

// Cache key prefixes
const CACHE_KEYS = {
  SESSION: 'session:',
  USER: 'user:',
  CASE: 'case:',
  DOCUMENT: 'document:',
  EVIDENCE: 'evidence:',
  SEARCH: 'search:',
  RAG: 'rag:',
  VECTOR: 'vector:',
  API_RATE_LIMIT: 'rate:',
  JOB_QUEUE: 'job:',
  WEBSOCKET: 'ws:'
} as const;

// Cache TTL (Time To Live) in seconds
const CACHE_TTL = {
  SESSION: 24 * 60 * 60,      // 24 hours
  USER: 60 * 60,              // 1 hour
  SEARCH: 15 * 60,            // 15 minutes
  RAG: 30 * 60,              // 30 minutes
  VECTOR: 60 * 60,           // 1 hour
  API_RESPONSE: 5 * 60,      // 5 minutes
  RATE_LIMIT: 60 * 60,       // 1 hour
  TEMP: 5 * 60               // 5 minutes
} as const;

export class RedisService {
  private static instance: RedisService;
  private redis: Redis;
  private subscribers: Map<string, Redis> = new Map();
  private connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected';
  
  constructor() {
    this.redis = new Redis(REDIS_CONFIG);
    this.setupEventHandlers();
  }

  static getInstance(): RedisService {
    if (!RedisService.instance) {
      RedisService.instance = new RedisService();
    }
    return RedisService.instance;
  }

  private setupEventHandlers() {
    this.redis.on('connect', () => {
      this.connectionStatus = 'connected';
      console.log('✅ Redis connected');
    });

    this.redis.on('error', (error) => {
      this.connectionStatus = 'error';
      console.error('❌ Redis error:', error.message);
    });

    this.redis.on('close', () => {
      this.connectionStatus = 'disconnected';
      console.log('🔌 Redis connection closed');
    });

    this.redis.on('reconnecting', () => {
      this.connectionStatus = 'connecting';
      console.log('🔄 Redis reconnecting...');
    });
  }

  async connect(): Promise<boolean> {
    try {
      await this.redis.ping();
      this.connectionStatus = 'connected';
      return true;
    } catch (error) {
      this.connectionStatus = 'error';
      console.error('❌ Redis connection failed:', error);
      return false;
    }
  }

  getConnectionStatus() {
    return this.connectionStatus;
  }

  // ==================== SESSION MANAGEMENT ====================

  async createSession(sessionId: string, userId: number, metadata: Record<string, any> = {}) {
    const sessionData = {
      userId,
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
      ipAddress: metadata.ipAddress || null,
      userAgent: metadata.userAgent || null,
      ...metadata
    };

    await this.redis.setex(
      `${CACHE_KEYS.SESSION}${sessionId}`,
      CACHE_TTL.SESSION,
      JSON.stringify(sessionData)
    );

    // Also store user's active sessions
    await this.redis.sadd(`${CACHE_KEYS.USER}${userId}:sessions`, sessionId);
    
    return sessionData;
  }

  async getSession(sessionId: string): Promise<any | null> {
    try {
      const data = await this.redis.get(`${CACHE_KEYS.SESSION}${sessionId}`);
      if (!data) return null;

      const session = JSON.parse(data);
      
      // Update last accessed time
      session.lastAccessedAt = Date.now();
      await this.redis.setex(
        `${CACHE_KEYS.SESSION}${sessionId}`,
        CACHE_TTL.SESSION,
        JSON.stringify(session)
      );

      return session;
    } catch (error) {
      console.error('Error getting session:', error);
      return null;
    }
  }

  async updateSession(sessionId: string, updates: Record<string, any>) {
    const sessionData = await this.getSession(sessionId);
    if (!sessionData) return false;

    const updatedData = {
      ...sessionData,
      ...updates,
      updatedAt: Date.now()
    };

    await this.redis.setex(
      `${CACHE_KEYS.SESSION}${sessionId}`,
      CACHE_TTL.SESSION,
      JSON.stringify(updatedData)
    );

    return true;
  }

  async deleteSession(sessionId: string) {
    const sessionData = await this.getSession(sessionId);
    if (sessionData) {
      // Remove from user's active sessions
      await this.redis.srem(`${CACHE_KEYS.USER}${sessionData.userId}:sessions`, sessionId);
    }
    
    await this.redis.del(`${CACHE_KEYS.SESSION}${sessionId}`);
  }

  async getUserSessions(userId: number): Promise<string[]> {
    return await this.redis.smembers(`${CACHE_KEYS.USER}${userId}:sessions`);
  }

  async deleteAllUserSessions(userId: number) {
    const sessions = await this.getUserSessions(userId);
    if (sessions.length > 0) {
      // Delete all session data
      const sessionKeys = sessions.map(sessionId => `${CACHE_KEYS.SESSION}${sessionId}`);
      await this.redis.del(...sessionKeys);
      
      // Clear user's session set
      await this.redis.del(`${CACHE_KEYS.USER}${userId}:sessions`);
    }
  }

  // ==================== CACHING OPERATIONS ====================

  async set(key: string, value: any, ttl?: number): Promise<boolean> {
    try {
      const serialized = JSON.stringify(value);
      if (ttl) {
        await this.redis.setex(key, ttl, serialized);
      } else {
        await this.redis.set(key, serialized);
      }
      return true;
    } catch (error) {
      console.error('Redis set error:', error);
      return false;
    }
  }

  async get<T = any>(key: string): Promise<T | null> {
    try {
      const data = await this.redis.get(key);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Redis get error:', error);
      return null;
    }
  }

  async del(key: string): Promise<boolean> {
    try {
      await this.redis.del(key);
      return true;
    } catch (error) {
      console.error('Redis delete error:', error);
      return false;
    }
  }

  async exists(key: string): Promise<boolean> {
    try {
      const result = await this.redis.exists(key);
      return result === 1;
    } catch (error) {
      console.error('Redis exists error:', error);
      return false;
    }
  }

  async expire(key: string, ttl: number): Promise<boolean> {
    try {
      await this.redis.expire(key, ttl);
      return true;
    } catch (error) {
      console.error('Redis expire error:', error);
      return false;
    }
  }

  // ==================== SPECIALIZED CACHE METHODS ====================

  async cacheSearchResults(query: string, results: any[], ttl: number = CACHE_TTL.SEARCH) {
    const key = `${CACHE_KEYS.SEARCH}${Buffer.from(query).toString('base64')}`;
    await this.set(key, { query, results, timestamp: Date.now() }, ttl);
    return key;
  }

  async getCachedSearchResults(query: string): Promise<any[] | null> {
    const key = `${CACHE_KEYS.SEARCH}${Buffer.from(query).toString('base64')}`;
    const cached = await this.get(key);
    return cached ? cached.results : null;
  }

  async cacheRAGResponse(query: string, response: string, context: any[], ttl: number = CACHE_TTL.RAG) {
    const key = `${CACHE_KEYS.RAG}${Buffer.from(query).toString('base64')}`;
    await this.set(key, { query, response, context, timestamp: Date.now() }, ttl);
    return key;
  }

  async getCachedRAGResponse(query: string): Promise<any | null> {
    const key = `${CACHE_KEYS.RAG}${Buffer.from(query).toString('base64')}`;
    return await this.get(key);
  }

  async cacheVectorEmbedding(text: string, embedding: number[], ttl: number = CACHE_TTL.VECTOR) {
    const key = `${CACHE_KEYS.VECTOR}${Buffer.from(text).toString('base64')}`;
    await this.set(key, { text, embedding, timestamp: Date.now() }, ttl);
    return key;
  }

  async getCachedVectorEmbedding(text: string): Promise<number[] | null> {
    const key = `${CACHE_KEYS.VECTOR}${Buffer.from(text).toString('base64')}`;
    const cached = await this.get(key);
    return cached ? cached.embedding : null;
  }

  // ==================== RATE LIMITING ====================

  async checkRateLimit(identifier: string, limit: number, window: number = 3600): Promise<{allowed: boolean, remaining: number}> {
    const key = `${CACHE_KEYS.API_RATE_LIMIT}${identifier}`;
    
    try {
      const current = await this.redis.get(key);
      const count = current ? parseInt(current) : 0;
      
      if (count >= limit) {
        return { allowed: false, remaining: 0 };
      }
      
      // Increment counter
      const newCount = await this.redis.incr(key);
      
      // Set expiry on first increment
      if (newCount === 1) {
        await this.redis.expire(key, window);
      }
      
      return { allowed: true, remaining: limit - newCount };
      
    } catch (error) {
      console.error('Rate limit check error:', error);
      // Allow request if Redis is down
      return { allowed: true, remaining: limit };
    }
  }

  // ==================== PUB/SUB OPERATIONS ====================

  async publish(channel: string, message: any): Promise<boolean> {
    try {
      await this.redis.publish(channel, JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Redis publish error:', error);
      return false;
    }
  }

  async subscribe(channel: string, callback: (message: any) => void): Promise<void> {
    if (!this.subscribers.has(channel)) {
      const subscriber = new Redis(REDIS_CONFIG);
      this.subscribers.set(channel, subscriber);
      
      subscriber.subscribe(channel);
      subscriber.on('message', (receivedChannel, message) => {
        if (receivedChannel === channel) {
          try {
            const parsed = JSON.parse(message);
            callback(parsed);
          } catch (error) {
            console.error('Error parsing pubsub message:', error);
            callback(message);
          }
        }
      });
    }
  }

  async unsubscribe(channel: string): Promise<void> {
    const subscriber = this.subscribers.get(channel);
    if (subscriber) {
      await subscriber.unsubscribe(channel);
      await subscriber.disconnect();
      this.subscribers.delete(channel);
    }
  }

  // ==================== HEALTH CHECK ====================

  async healthCheck(): Promise<{status: string, details: any}> {
    try {
      const start = Date.now();
      await this.redis.ping();
      const responseTime = Date.now() - start;
      
      const info = await this.redis.info();
      const memory = await this.redis.memory('usage');
      
      return {
        status: 'healthy',
        details: {
          connectionStatus: this.connectionStatus,
          responseTime,
          memory,
          subscriberCount: this.subscribers.size,
          version: info.split('\r\n').find(line => line.startsWith('redis_version:'))?.split(':')[1]
        }
      };
      
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          connectionStatus: this.connectionStatus,
          error: error instanceof Error ? error.message : 'Unknown error'
        }
      };
    }
  }

  // ==================== CLEANUP ====================

  async clearAllCache(): Promise<void> {
    if (dev) {
      await this.redis.flushdb();
      console.log('🧹 Redis cache cleared (development mode)');
    } else {
      console.warn('⚠️ Cache clear is disabled in production');
    }
  }

  async disconnect(): Promise<void> {
    // Close all subscribers
    for (const [channel, subscriber] of this.subscribers) {
      await subscriber.disconnect();
    }
    this.subscribers.clear();
    
    // Close main connection
    await this.redis.disconnect();
    this.connectionStatus = 'disconnected';
    console.log('👋 Redis disconnected');
  }
}

// Singleton instance
export const redis = RedisService.getInstance();

// Utility functions
export const cacheKeys = CACHE_KEYS;
export const cacheTTL = CACHE_TTL;

// Session management helpers
export class SessionManager {
  static async createUserSession(userId: number, sessionId: string, metadata?: Record<string, any>) {
    return await redis.createSession(sessionId, userId, metadata);
  }
  
  static async validateSession(sessionId: string): Promise<any | null> {
    return await redis.getSession(sessionId);
  }
  
  static async invalidateSession(sessionId: string): Promise<void> {
    await redis.deleteSession(sessionId);
  }
  
  static async invalidateAllUserSessions(userId: number): Promise<void> {
    await redis.deleteAllUserSessions(userId);
  }
}

// Export types
export interface CacheOptions {
  ttl?: number;
  key?: string;
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
}

export interface SearchCacheData {
  query: string;
  results: any[];
  timestamp: number;
}

export interface RAGCacheData {
  query: string;
  response: string;
  context: any[];
  timestamp: number;
}

// Health check endpoint data
export async function getRedisHealth() {
  return await redis.healthCheck();
}