/**
 * Enhanced Environment Variables Type Definitions
 * Fixes "$env/dynamic/private" module resolution issues
 */

// SvelteKit environment modules
declare module '$env/dynamic/private' {
  export const env: {
    QDRANT_URL?: string;
    OLLAMA_URL?: string;
    ENHANCED_RAG_MAX_RESULTS?: string;
    ENHANCED_RAG_CONFIDENCE?: string;
    ENHANCED_RAG_CLUSTERING?: string;
    ENHANCED_RAG_CACHING?: string;
    ENHANCED_RAG_CACHE_THRESHOLD?: string;
    ENHANCED_RAG_WORKERS?: string;
    ENHANCED_RAG_MAX_CONCURRENT?: string;
    ENHANCED_RAG_PRECACHING?: string;
    DATABASE_URL?: string;
    POSTGRES_URL?: string;
    REDIS_URL?: string;
    NEO4J_URL?: string;
    NEO4J_USERNAME?: string;
    NEO4J_PASSWORD?: string;
    MINIO_URL?: string;
    MINIO_ACCESS_KEY?: string;
    MINIO_SECRET_KEY?: string;
    JWT_SECRET?: string;
    OPENAI_API_KEY?: string;
    ANTHROPIC_API_KEY?: string;
    GOOGLE_API_KEY?: string;
    NODE_ENV?: string;
    [key: string]: string | undefined;
  };
}

declare module '$env/dynamic/public' {
  export const env: Record<string, string | undefined>;
}

declare module '$env/static/private' {
  export const DATABASE_URL: string;
  export const POSTGRES_URL: string;
  export const REDIS_URL: string;
  export const OLLAMA_URL: string;
  export const QDRANT_URL: string;
  export const NEO4J_URL: string;
  export const NEO4J_USERNAME: string;
  export const NEO4J_PASSWORD: string;
  export const MINIO_URL: string;
  export const MINIO_ACCESS_KEY: string;
  export const MINIO_SECRET_KEY: string;
  export const JWT_SECRET: string;
  export const OPENAI_API_KEY: string;
  export const ANTHROPIC_API_KEY: string;
  export const GOOGLE_API_KEY: string;
  export const NODE_ENV: string;
}

declare module '$env/static/public' {
  export const PUBLIC_API_BASE_URL: string;
  export const PUBLIC_OLLAMA_URL: string;
  export const PUBLIC_QDRANT_URL: string;
  export const PUBLIC_MINIO_URL: string;
  export const PUBLIC_NEO4J_URL: string;
}

// Enhanced environment interface
export interface EnhancedEnv {
  // Database configuration
  DATABASE_URL: string;
  POSTGRES_URL: string;
  POSTGRES_HOST: string;
  POSTGRES_PORT: string;
  POSTGRES_DB: string;
  POSTGRES_USER: string;
  POSTGRES_PASSWORD: string;

  // Redis configuration
  REDIS_URL: string;
  REDIS_HOST: string;
  REDIS_PORT: string;
  REDIS_PASSWORD: string;

  // AI services
  OLLAMA_URL: string;
  OLLAMA_HOST: string;
  OLLAMA_PORT: string;
  OPENAI_API_KEY: string;
  ANTHROPIC_API_KEY: string;
  GOOGLE_API_KEY: string;

  // Vector database
  QDRANT_URL: string;
  QDRANT_HOST: string;
  QDRANT_PORT: string;
  QDRANT_API_KEY: string;

  // Graph database
  NEO4J_URL: string;
  NEO4J_HOST: string;
  NEO4J_PORT: string;
  NEO4J_USERNAME: string;
  NEO4J_PASSWORD: string;

  // Object storage
  MINIO_URL: string;
  MINIO_HOST: string;
  MINIO_PORT: string;
  MINIO_ACCESS_KEY: string;
  MINIO_SECRET_KEY: string;
  MINIO_BUCKET: string;

  // Security
  JWT_SECRET: string;
  ENCRYPTION_KEY: string;
  API_SECRET: string;

  // Application
  NODE_ENV: string;
  PORT: string;
  HOST: string;
  PUBLIC_API_BASE_URL: string;

  // Development
  DEBUG: string;
  LOG_LEVEL: string;
  ENABLE_LOGGING: string;
}

// Environment helper functions
export const envHelper = {
  // Get environment variable with fallback
  get: (key: string, defaultValue: string = ''): string => {
    if (typeof process !== 'undefined' && process.env) {
      return process.env[key] || defaultValue;
    }
    return defaultValue;
  },

  // Get boolean environment variable
  getBool: (key: string, defaultValue: boolean = false): boolean => {
    const value = envHelper.get(key);
    if (!value) return defaultValue;
    return value.toLowerCase() === 'true' || value === '1';
  },

  // Get number environment variable
  getNumber: (key: string, defaultValue: number = 0): number => {
    const value = envHelper.get(key);
    const parsed = parseInt(value, 10);
    return isNaN(parsed) ? defaultValue : parsed;
  },

  // Get required environment variable (throws if missing)
  getRequired: (key: string): string => {
    const value = envHelper.get(key);
    if (!value) {
      throw new Error(`Required environment variable ${key} is not set`);
    }
    return value;
  },

  // Get database URL with validation
  getDatabaseUrl: (): string => {
    return envHelper.get('DATABASE_URL') ||
      envHelper.get('POSTGRES_URL') ||
           `postgresql://postgres:postgres@localhost:5432/legal_ai_db`;
  },

  // Get Redis URL with validation
  getRedisUrl: (): string => {
    return envHelper.get('REDIS_URL') ||
           `redis://localhost:6379`;
  },

  // Get Ollama URL with validation
  getOllamaUrl: (): string => {
    return envHelper.get('OLLAMA_URL') ||
      envHelper.get('PUBLIC_OLLAMA_URL') ||
           `http://localhost:11434`;
  }
};

// Export types
export type { EnhancedEnv };