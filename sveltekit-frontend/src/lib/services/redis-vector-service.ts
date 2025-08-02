/**
 * Redis Vector Service for caching and vector operations
 */

import Redis from "ioredis";
import { logger } from "../server/logger";

export interface VectorSearchResult {
  id: string;
  score: number;
  payload: any;
  vector?: number[];
}

export interface DocumentVector {
  id: string;
  vector: number[];
  payload: any;
  metadata?: any;
}

export class RedisVectorService {
  private redis: Redis;
  private isConnected = false;

  constructor() {
    const redisOptions = {
      host: process.env.REDIS_HOST || "localhost",
      port: parseInt(process.env.REDIS_PORT || "6379"),
      password: process.env.REDIS_PASSWORD,
      db: parseInt(process.env.REDIS_DB || "0"),
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
    };

    this.redis = new Redis(redisOptions);

    this.redis.on("connect", () => {
      this.isConnected = true;
      logger.info("Redis Vector Service connected");
    });

    this.redis.on("error", (error) => {
      this.isConnected = false;
      logger.error("Redis Vector Service error", error);
    });
  }

  async isHealthy(): Promise<boolean> {
    try {
      await this.redis.ping();
      return this.isConnected;
    } catch (error) {
      logger.error("Redis health check failed", error);
      return false;
    }
  }

  async storeVector(id: string, vector: number[], payload: any): Promise<void> {
    try {
      const vectorData = {
        id,
        vector,
        payload,
        timestamp: Date.now(),
      };

      await this.redis.hset(`vector:${id}`, "data", JSON.stringify(vectorData));

      // Also store in a set for quick lookup
      await this.redis.sadd("vectors:all", id);

      logger.debug(`Stored vector for ID: ${id}`);
    } catch (error) {
      logger.error("Failed to store vector", { id, error });
      throw error;
    }
  }

  async getVector(id: string): Promise<DocumentVector | null> {
    try {
      const data = await this.redis.hget(`vector:${id}`, "data");
      if (!data) return null;

      const vectorData = JSON.parse(data);
      return {
        id: vectorData.id,
        vector: vectorData.vector,
        payload: vectorData.payload,
        metadata: vectorData.metadata,
      };
    } catch (error) {
      logger.error("Failed to get vector", { id, error });
      return null;
    }
  }

  async deleteVector(id: string): Promise<void> {
    try {
      await this.redis.del(`vector:${id}`);
      await this.redis.srem("vectors:all", id);
      logger.debug(`Deleted vector for ID: ${id}`);
    } catch (error) {
      logger.error("Failed to delete vector", { id, error });
      throw error;
    }
  }

  async searchVectors(
    queryVector: number[],
    options: {
      limit?: number;
      threshold?: number;
      collection?: string;
    } = {}
  ): Promise<VectorSearchResult[]> {
    try {
      // This is a simple implementation - in production you'd use Redis Search or RedisAI
      const allVectorIds = await this.redis.smembers("vectors:all");
      const results: VectorSearchResult[] = [];

      for (const id of allVectorIds) {
        const vectorData = await this.getVector(id);
        if (!vectorData) continue;

        const similarity = this.cosineSimilarity(
          queryVector,
          vectorData.vector
        );

        if (similarity >= (options.threshold || 0.7)) {
          results.push({
            id,
            score: similarity,
            payload: vectorData.payload,
            vector: vectorData.vector,
          });
        }
      }

      // Sort by score and limit
      results.sort((a, b) => b.score - a.score);
      return results.slice(0, options.limit || 10);
    } catch (error) {
      logger.error("Failed to search vectors", error);
      return [];
    }
  }

  async cacheEmbedding(
    text: string,
    embedding: number[],
    model: string
  ): Promise<void> {
    try {
      const key = `embedding:${this.hashText(text)}:${model}`;
      await this.redis.setex(key, 3600, JSON.stringify(embedding)); // Cache for 1 hour
    } catch (error) {
      logger.error("Failed to cache embedding", error);
    }
  }

  async getCachedEmbedding(
    text: string,
    model: string
  ): Promise<number[] | null> {
    try {
      const key = `embedding:${this.hashText(text)}:${model}`;
      const cached = await this.redis.get(key);
      return cached ? JSON.parse(cached) : null;
    } catch (error) {
      logger.error("Failed to get cached embedding", error);
      return null;
    }
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private hashText(text: string): string {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
  }

  async close(): Promise<void> {
    await this.redis.quit();
    this.isConnected = false;
  }
}

export const redisVectorService = new RedisVectorService();
export default redisVectorService;
