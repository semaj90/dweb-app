/**
 * PM2 Ecosystem Configuration for Legal AI System
 * Manages Node.js processes: SvelteKit server and BullMQ workers
 */

module.exports = {
  apps: [
    {
      name: 'legal-ai-sveltekit',
      script: 'build/index.js',
      cwd: './',
      env: {
        NODE_ENV: 'production',
        PORT: 3000,
        GO_SERVER_URL: 'http://localhost:8081',
        DATABASE_URL: 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db',
        REDIS_URL: 'redis://localhost:6379',
        OLLAMA_BASE_URL: 'http://localhost:11434',
        QDRANT_URL: 'http://localhost:6333'
      },
      env_development: {
        NODE_ENV: 'development',
        PORT: 5173,
        GO_SERVER_URL: 'http://localhost:8081'
      },
      instances: 2,
      exec_mode: 'cluster',
      watch: false,
      max_memory_restart: '2G',
      log_file: './logs/legal-ai-sveltekit.log',
      error_file: './logs/legal-ai-sveltekit-error.log',
      out_file: './logs/legal-ai-sveltekit-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      restart_delay: 3000,
      max_restarts: 5,
      min_uptime: '10s'
    },
    {
      name: 'legal-ai-bullmq-worker',
      script: 'src/lib/workers/bullmq-worker.js',
      cwd: './',
      env: {
        NODE_ENV: 'production',
        WORKER_CONCURRENCY: 3,
        GO_SERVER_URL: 'http://localhost:8081',
        DATABASE_URL: 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db',
        REDIS_URL: 'redis://localhost:6379'
      },
      instances: 2, // Run 2 worker instances for load balancing
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '1G',
      log_file: './logs/legal-ai-worker.log',
      error_file: './logs/legal-ai-worker-error.log',
      out_file: './logs/legal-ai-worker-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      restart_delay: 5000,
      max_restarts: 10,
      min_uptime: '5s'
    },
    {
      name: 'legal-ai-scheduler',
      script: './scripts/scheduler.mjs',
      cwd: './',
      env: {
        NODE_ENV: 'production',
        GO_SERVER_URL: 'http://localhost:8081',
        DATABASE_URL: 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db',
        REDIS_URL: 'redis://localhost:6379'
      },
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      cron_restart: '0 2 * * *', // Restart daily at 2 AM
      max_memory_restart: '256M',
      log_file: './logs/legal-ai-scheduler.log',
      error_file: './logs/legal-ai-scheduler-error.log',
      out_file: './logs/legal-ai-scheduler-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      restart_delay: 5000,
      max_restarts: 3,
      min_uptime: '60s'
    }
  ],

  deploy: {
    production: {
      user: 'legal-ai',
      host: 'localhost',
      ref: 'origin/main',
      repo: 'git@github.com:your-org/legal-ai-system.git',
      path: '/var/www/legal-ai',
      'pre-setup': 'npm install pm2 -g',
      'post-setup': 'npm install && npm run build',
      'pre-deploy-local': '',
      'post-deploy': 'npm install && npm run build && pm2 reload ecosystem.config.js --env production',
      'pre-deploy': 'git add -A && git commit -m "Deploy update" || true && git push || true',
      env: {
        NODE_ENV: 'production'
      }
    }
  }
};