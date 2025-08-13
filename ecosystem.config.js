// PM2 Ecosystem Configuration for Legal AI System
module.exports = {
  apps: [
    // SvelteKit Frontend Cluster
    {
      name: 'sveltekit-frontend',
      script: './sveltekit-frontend/build/index.js',
      instances: 'max', // Use all CPU cores
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'development',
        PORT: 5173,
        REDIS_URL: 'redis://localhost:6379',
        DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
        QUIC_TENSOR_URL: 'https://localhost:4433',
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 3000,
        REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379',
        DATABASE_URL: process.env.DATABASE_URL,
      },
      max_memory_restart: '1G',
      error_file: './logs/sveltekit-error.log',
      out_file: './logs/sveltekit-out.log',
      log_file: './logs/sveltekit-combined.log',
      time: true,
      max_restarts: 10,
      min_uptime: '10s',
    },

    // Go QUIC Tensor Server
    {
      name: 'quic-tensor-server',
      cwd: './go-microservice',
      script: 'quic-tensor-server.exe',
      interpreter: 'none',
      instances: 2, // Limited instances for GPU sharing
      exec_mode: 'fork',
      env: {
        PORT: 4433,
        REDIS_URL: 'redis://localhost:6379',
        GPU_WORKERS: 4,
        TENSOR_CACHE_SIZE: '512MB',
        SOM_NODES: 400,
      },
      env_production: {
        GPU_WORKERS: process.env.GPU_WORKERS || 8,
        TENSOR_CACHE_SIZE: process.env.TENSOR_CACHE_SIZE || '1GB',
      },
      max_memory_restart: '2G',
      error_file: '../logs/quic-tensor-error.log',
      out_file: '../logs/quic-tensor-out.log',
      time: true,
      max_restarts: 5,
      min_uptime: '30s',
    },

    // Enhanced SIMD Server (existing)
    {
      name: 'simd-server',
      cwd: './go-microservice',
      script: 'simd-server-prod.exe',
      interpreter: 'none',
      instances: 1,
      exec_mode: 'fork',
      env: {
        REDIS_HOST: 'localhost',
        REDIS_PORT: 6379,
        SIMD_WORKERS: 32,
        GIN_MODE: 'release',
        PORT: 8080,
      },
      max_memory_restart: '2G',
      error_file: '../logs/simd-error.log',
      out_file: '../logs/simd-out.log'
    },

    // BullMQ Job Processor Workers
    {
      name: 'job-processor',
      script: './workers/job-processor.js',
      instances: 4,
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'development',
        REDIS_URL: 'redis://localhost:6379',
        DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
        WORKER_CONCURRENCY: 3,
        OLLAMA_URL: 'http://localhost:11434',
      },
      max_memory_restart: '1G',
      error_file: './logs/job-processor-error.log',
      out_file: './logs/job-processor-out.log',
      time: true,
      max_restarts: 10,
      min_uptime: '10s',
    },

    // WebSocket Server
    {
      name: 'websocket-server',
      script: './websocket/server.js',
      instances: 2,
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'development',
        WEBSOCKET_PORT: 8090,
        REDIS_URL: 'redis://localhost:6379',
        CORS_ORIGIN: 'http://localhost:5173',
      },
      max_memory_restart: '512M',
      error_file: './logs/websocket-error.log',
      out_file: './logs/websocket-out.log',
      time: true,
    },

    // Enhanced Vite Production (existing)
    {
      name: 'vite-prod',
      script: 'npm',
      args: 'run preview',
      env: { 
        NODE_ENV: 'production',
        PORT: 4173,
      },
      instances: 2,
      exec_mode: 'cluster'
    },

    // Redis Monitor (existing)
    {
      name: 'redis-monitor',
      script: './scripts/redis-monitor.js',
      env: {
        REDIS_URL: 'redis://localhost:6379',
        MONITORING_INTERVAL: 30000,
      },
      max_memory_restart: '256M',
      error_file: './logs/redis-monitor-error.log',
      out_file: './logs/redis-monitor-out.log',
    }
  ],

  // Deployment configuration
  deploy: {
    production: {
      user: 'deploy',
      host: 'production-server',
      ref: 'origin/main',
      repo: 'git@github.com:yourorg/legal-ai-system.git',
      path: '/var/www/legal-ai',
      'post-deploy': 'npm install && npm run build && pm2 reload ecosystem.config.js --env production',
      env: {
        NODE_ENV: 'production'
      }
    }
  }
};