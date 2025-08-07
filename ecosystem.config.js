// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'simd-server',
    cwd: './go-microservice',
    script: 'simd-server-prod.exe',
    interpreter: 'none',
    env: {
      REDIS_HOST: 'localhost',
      REDIS_PORT: 6379,
      SIMD_WORKERS: 32,
      GIN_MODE: 'release'
    },
    max_memory_restart: '2G',
    error_file: '../logs/simd-error.log',
    out_file: '../logs/simd-out.log'
  }, {
    name: 'vite-prod',
    script: 'npm',
    args: 'run preview',
    env: { NODE_ENV: 'production' },
    instances: 2,
    exec_mode: 'cluster'
  }, {
    name: 'redis-monitor',
    script: './scripts/redis-monitor.js'
  }]
};