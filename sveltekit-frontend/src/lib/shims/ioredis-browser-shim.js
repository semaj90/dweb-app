/**
 * IORedis Browser Shim
 * Provides browser-compatible Redis client interface
 */

// Mock Redis client for browser environment
export default class RedisShim {
  constructor(config) {
    this.config = config;
    this.connected = false;
    console.log('🔧 Redis Browser Shim initialized:', config);
  }

  async connect() {
    this.connected = true;
    return Promise.resolve();
  }

  async disconnect() {
    this.connected = false;
    return Promise.resolve();
  }

  async ping() {
    return Promise.resolve('PONG');
  }

  // Basic Redis operations (browser-compatible stubs)
  async get(key) {
    const value = localStorage.getItem(`redis:${key}`);
    return value ? JSON.parse(value) : null;
  }

  async set(key, value, ...args) {
    localStorage.setItem(`redis:${key}`, JSON.stringify(value));
    return 'OK';
  }

  async del(key) {
    localStorage.removeItem(`redis:${key}`);
    return 1;
  }

  async exists(key) {
    return localStorage.getItem(`redis:${key}`) ? 1 : 0;
  }

  async hget(hash, field) {
    const data = localStorage.getItem(`redis:${hash}`);
    if (data) {
      const obj = JSON.parse(data);
      return obj[field] || null;
    }
    return null;
  }

  async hset(hash, field, value) {
    const existing = localStorage.getItem(`redis:${hash}`);
    const obj = existing ? JSON.parse(existing) : {};
    obj[field] = value;
    localStorage.setItem(`redis:${hash}`, JSON.stringify(obj));
    return 1;
  }

  async lpush(key, ...values) {
    const existing = localStorage.getItem(`redis:${key}`);
    const array = existing ? JSON.parse(existing) : [];
    array.unshift(...values);
    localStorage.setItem(`redis:${key}`, JSON.stringify(array));
    return array.length;
  }

  async rpop(key) {
    const existing = localStorage.getItem(`redis:${key}`);
    if (existing) {
      const array = JSON.parse(existing);
      const value = array.pop();
      localStorage.setItem(`redis:${key}`, JSON.stringify(array));
      return value;
    }
    return null;
  }

  // Additional Redis methods that might be called
  async flushall() {
    const keys = Object.keys(localStorage).filter(key => key.startsWith('redis:'));
    keys.forEach(key => localStorage.removeItem(key));
    return 'OK';
  }

  async keys(pattern) {
    const keys = Object.keys(localStorage)
      .filter(key => key.startsWith('redis:'))
      .map(key => key.replace('redis:', ''));
    return keys;
  }

  async ttl(key) {
    return -1; // No expiration in localStorage
  }

  async expire(key, seconds) {
    // localStorage doesn't support expiration, so this is a no-op
    return 1;
  }

  // Event emitter stubs
  on(event, handler) {
    console.log(`🔧 Redis event registered: ${event}`);
    return this;
  }

  emit(event, ...args) {
    console.log(`🔧 Redis event emitted: ${event}`, args);
    return this;
  }

  // Status getters
  get status() {
    return this.connected ? 'ready' : 'connecting';
  }

  // Cluster stub
  static Cluster = class {
    constructor(config) {
      console.log('🔧 Redis Cluster Browser Shim initialized:', config);
      return new RedisShim(config);
    }
  };
}

// Named exports for compatibility
export const Redis = RedisShim;
export const Cluster = RedisShim.Cluster;