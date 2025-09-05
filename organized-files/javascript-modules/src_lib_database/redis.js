// Redis Database Service
// Manages Redis connections and operations

import { createClient } from 'redis';

class RedisService {
  constructor() {
    this.client = null;
    this.connected = false;
  }

  /**
   * Connect to Redis
   */
  async connect() {
    try {
      this.client = createClient({
        url: 'redis://localhost:6379'
      });

      this.client.on('error', (err) => {
        console.error('Redis Client Error:', err);
        this.connected = false;
      });

      await this.client.connect();
      this.connected = true;
      console.log('Connected to Redis');
      return true;
    } catch (error) {
      console.error('Failed to connect to Redis:', error);
      this.connected = false;
      return false;
    }
  }

  /**
   * Disconnect from Redis
   */
  async disconnect() {
    if (this.client) {
      await this.client.quit();
      this.connected = false;
      console.log('Disconnected from Redis');
    }
  }

  /**
   * Get value from Redis
   * @param {string} key - Key to retrieve
   * @returns {Promise<any>} Value or null
   */
  async get(key) {
    if (!this.connected) await this.connect();
    try {
      const value = await this.client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error(`Redis GET error for key ${key}:`, error);
      return null;
    }
  }

  /**
   * Set value in Redis
   * @param {string} key - Key to set
   * @param {any} value - Value to store
   * @param {number} ttl - Time to live in seconds
   */
  async set(key, value, ttl = 3600) {
    if (!this.connected) await this.connect();
    try {
      const serialized = JSON.stringify(value);
      if (ttl > 0) {
        await this.client.setEx(key, ttl, serialized);
      } else {
        await this.client.set(key, serialized);
      }
      return true;
    } catch (error) {
      console.error(`Redis SET error for key ${key}:`, error);
      return false;
    }
  }

  /**
   * Delete key from Redis
   * @param {string} key - Key to delete
   */
  async delete(key) {
    if (!this.connected) await this.connect();
    try {
      await this.client.del(key);
      return true;
    } catch (error) {
      console.error(`Redis DELETE error for key ${key}:`, error);
      return false;
    }
  }

  /**
   * Check if key exists
   * @param {string} key - Key to check
   */
  async exists(key) {
    if (!this.connected) await this.connect();
    try {
      return await this.client.exists(key);
    } catch (error) {
      console.error(`Redis EXISTS error for key ${key}:`, error);
      return false;
    }
  }

  /**
   * Get all keys matching pattern
   * @param {string} pattern - Pattern to match
   */
  async keys(pattern = '*') {
    if (!this.connected) await this.connect();
    try {
      return await this.client.keys(pattern);
    } catch (error) {
      console.error(`Redis KEYS error for pattern ${pattern}:`, error);
      return [];
    }
  }

  /**
   * Clear all data
   */
  async flushAll() {
    if (!this.connected) await this.connect();
    try {
      await this.client.flushAll();
      console.log('Redis cache cleared');
      return true;
    } catch (error) {
      console.error('Redis FLUSHALL error:', error);
      return false;
    }
  }

  /**
   * Get connection status
   */
  getStatus() {
    return {
      connected: this.connected,
      ready: this.client?.isReady || false
    };
  }
}

// Export singleton instance
export const cacheManager = new RedisService();
export const redisService = cacheManager;

// Also export class
export { RedisService };