module.exports = {
  apps: [
    // MCP Context7 Server Cluster
    {
      name: 'mcp-context7-cluster',
      script: 'mcp-servers/mcp-context7-wrapper.js',
      instances: 4, // Multi-core utilization
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'production',
        PROJECT_ROOT: 'C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app',
        OLLAMA_ENDPOINT: 'http://localhost:11434',
        DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
        MCP_SERVER_PORT: '4000'
      },
      max_memory_restart: '500M',
      watch: false,
      autorestart: true,
      max_restarts: 5,
      min_uptime: '10s'
    },

    // Context7 Multi-Workers (8 workers on different ports)
    ...Array.from({ length: 8 }, (_, i) => ({
      name: `context7-worker-${i + 1}`,
      script: 'scripts/context7-worker.js',
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: 'production',
        WORKER_ID: `worker_${i + 1}`,
        WORKER_PORT: 4100 + i,
        OLLAMA_ENDPOINT: 'http://localhost:11434',
        DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
        GPU_ACCELERATION: 'true',
        LEGAL_BERT_MODEL: 'nlpaueb/legal-bert-base-uncased',
        GOLLAMA_ENABLED: 'true'
      },
      max_memory_restart: '800M',
      watch: false,
      autorestart: true
    })),

    // Go Microservice Load Balancer
    {
      name: 'go-load-balancer',
      script: 'go-microservice/bin/load-balancer.exe',
      instances: 1,
      exec_mode: 'fork',
      env: {
        PORT: '8099',
        UPSTREAM_SERVICES: 'http://localhost:8094,http://localhost:8095,http://localhost:8096',
        LOAD_BALANCER_STRATEGY: 'round_robin',
        HEALTH_CHECK_INTERVAL: '30s',
        CUDA_ACCELERATION: 'true',
        GPU_MEMORY_LIMIT: '6GB'
      },
      max_memory_restart: '200M',
      autorestart: true,
      watch: false
    },

    // Enhanced RAG Cluster
    {
      name: 'enhanced-rag-cluster', 
      script: 'go-microservice/bin/enhanced-rag.exe',
      instances: 2,
      exec_mode: 'cluster',
      env: {
        PORT: '8094',
        DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
        REDIS_URL: 'redis://localhost:6379',
        OLLAMA_ENDPOINT: 'http://localhost:11434',
        VECTOR_DIMENSION: '384',
        LEGAL_BERT_ENABLED: 'true',
        SEMANTIC_ANALYSIS: 'advanced'
      },
      max_memory_restart: '1GB',
      autorestart: true
    },

    // Recommendation Service Cluster
    {
      name: 'recommendation-cluster',
      script: 'go-microservice/bin/recommendation-service.exe', 
      instances: 3,
      exec_mode: 'cluster',
      env: {
        PORT: '8096',
        OLLAMA_ENDPOINT: 'http://localhost:11434',
        CONTEXT7_WORKERS: 'http://localhost:4100,http://localhost:4101,http://localhost:4102,http://localhost:4103,http://localhost:4104,http://localhost:4105,http://localhost:4106,http://localhost:4107',
        TOKENIZER_MODEL: 'legal-bert',
        CUDA_ENABLED: 'true',
        BATCH_SIZE: '32',
        SEMANTIC_SIMILARITY_THRESHOLD: '0.8'
      },
      max_memory_restart: '800M',
      autorestart: true
    },

    // SvelteKit Frontend (Development)
    {
      name: 'sveltekit-frontend',
      script: 'npm',
      args: 'run dev',
      cwd: './sveltekit-frontend',
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: 'development',
        VITE_API_BASE: 'http://localhost:8099', // Points to load balancer
        VITE_OLLAMA_ENDPOINT: 'http://localhost:11434',
        VITE_CONTEXT7_WORKERS: '8'
      },
      watch: ['src'],
      ignore_watch: ['node_modules', 'build'],
      autorestart: true
    },

    // Error Processing Automation
    {
      name: 'error-processor',
      script: 'scripts/error-processor-daemon.js',
      instances: 1,
      exec_mode: 'fork',
      cron_restart: '*/5 * * * *', // Check every 5 minutes
      env: {
        CHECK_INTERVAL: '300000', // 5 minutes
        AUTO_FIX_ENABLED: 'true',
        RECOMMENDATION_ENDPOINT: 'http://localhost:8099/api/process-error-log',
        LOG_DIRECTORY: './error-logs',
        CONFIDENCE_THRESHOLD: '0.85'
      },
      autorestart: true
    }
  ]
};