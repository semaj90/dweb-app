#!/usr/bin/env node

/**
 * 🌐 Legal AI Platform - Environment Configuration
 * Windows Native Setup with GPU Acceleration
 */

// Environment configuration
const envConfig = {
  // Database
  DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
  DATABASE_HOST: 'localhost',
  DATABASE_PORT: '5432',
  DATABASE_NAME: 'legal_ai_db',
  DATABASE_USER: 'legal_admin',
  DATABASE_PASSWORD: '123456',
  PGPASSWORD: '123456',

  // Frontend & API
  FRONTEND_PORT: '5173',
  PROXY_PORT: '5180',
  API_PORT: '8080',
  API_URL: 'http://localhost:8080',
  FRONTEND_URL: 'http://localhost:5173',
  PROXY_URL: 'http://localhost:5180',

  // Redis
  REDIS_URL: 'redis://localhost:6379',
  REDIS_HOST: 'localhost',
  REDIS_PORT: '6379',

  // AI & Ollama
  OLLAMA_ENDPOINT: 'http://localhost:11434',
  OLLAMA_API_URL: 'http://localhost:11434/api',
  GEMMA3_MODEL_PATH: 'C:\\Users\\james\\.ollama\\models\\gemma3-legal',

  // Microservices
  UPLOAD_SERVICE_PORT: '8093',
  RAG_SERVICE_PORT: '8094',
  QUIC_GATEWAY_PORT: '8447',
  CLUSTER_MANAGER_PORT: '3099',
  LOAD_BALANCER_PORT: '8099',

  // GPU & CUDA
  GPU_ENABLED: 'true',
  CUDA_PATH: 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0',
  CGO_ENABLED: '1',
  ENABLE_GPU: 'true',

  // File paths
  UPLOADS_DIR: './uploads',
  DOCUMENTS_DIR: './documents',
  EVIDENCE_DIR: './evidence',
  LOGS_DIR: './logs',
  GENERATED_REPORTS_DIR: './generated_reports',

  // Security
  JWT_SECRET: 'your-super-secret-jwt-key-here-change-in-production',
  SESSION_SECRET: 'your-session-secret-key-here',

  // Environment
  NODE_ENV: 'development',
  DEBUG: 'true',
  LOG_LEVEL: 'info'
};

// Set environment variables
Object.entries(envConfig).forEach(([key, value]) => {
  process.env[key] = value;
});

console.log('✅ Environment variables loaded');

// Export for use in other modules
export default envConfig;
