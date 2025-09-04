// Environment variable type definitions

declare module '$env/static/private' {
  // AI Model Configuration
  export const OLLAMA_MODEL: string;
  export const OLLAMA_MODEL_LEGAL: string;
  export const OLLAMA_MODEL_CHAT: string;
  export const EMBEDDING_MODEL: string;
  export const ONNX_EMBEDDING_MODEL: string;
  export const EMBEDDING_DIMENSION: string;
  export const OLLAMA_MODEL_SUMMARY: string;
  export const OLLAMA_MODEL_ANALYSIS: string;

  // Database Configuration
  export const DATABASE_URL: string;
  export const PG_CONN_STRING: string;
  export const POSTGRES_PASSWORD: string;
  export const DB_HOST: string;
  export const DB_PORT: string;
  export const DB_NAME: string;
  export const DB_USER: string;

  // Service URLs and Ports
  export const POSTGRES_PORT: string;
  export const REDIS_PORT: string;
  export const QDRANT_PORT: string;
  export const MINIO_PORT: string;
  export const MINIO_CONSOLE_PORT: string;
  export const NEO4J_HTTP_PORT: string;
  export const NEO4J_BOLT_PORT: string;
  export const OLLAMA_PORT: string;

  // Application Service Ports
  export const SVELTEKIT_PORT: string;
  export const CLUSTER_MANAGER_API_PORT: string;
  export const NODE_API_SERVICE_PORT: string;
  export const GO_RAG_SERVICE_PORT: string;
  export const GO_UPLOAD_SERVICE_PORT: string;
  export const GO_SIMPLE_UPLOAD_PORT: string;
  export const GO_GRPC_SERVICE_PORT: string;
  export const GO_LOAD_BALANCER_PORT: string;
  export const WASM_WORKER_PORT: string;
  export const GPU_WORKER_PORT: string;

  // Cluster & Load Balancing
  export const CLUSTER_MANAGER_PORT: string;
  export const LOAD_BALANCER_PORT: string;
  export const LOAD_BALANCER_WS_PORT: string;

  // QUIC Protocol Services
  export const QUIC_HTTP_PORT: string;
  export const QUIC_QUIC_PORT: string;
  export const QUIC_LEGAL_GATEWAY_PORT: string;

  // Monitoring & Metrics
  export const METRICS_PORT: string;
  export const HEALTH_CHECK_PORT: string;
  export const LOG_SERVER_PORT: string;

  // Service URLs
  export const REDIS_URL: string;
  export const REDIS_HOST: string;
  export const REDIS_DB: string;
  export const REDIS_PASSWORD: string;
  export const QDRANT_URL: string;
  export const QDRANT_HOST: string;
  export const QDRANT_COLLECTION: string;
  export const OLLAMA_API_URL: string;
  export const OLLAMA_BASE_URL: string;
  export const OLLAMA_URL: string;

  // MinIO Configuration
  export const MINIO_ENDPOINT: string;
  export const MINIO_ACCESS_KEY: string;
  export const MINIO_SECRET_KEY: string;
  export const MINIO_BUCKET: string;
  export const MINIO_USE_SSL: string;

  // Security
  export const JWT_SECRET: string;
  export const API_KEY: string;
  export const CORS_ORIGIN: string;
  export const SECURE_COOKIES: string;
  export const RATE_LIMIT_PER_MINUTE: string;

  // RAG Configuration
  export const RAG_STREAMING_ENABLED: string;
  export const RAG_CHUNK_SIZE: string;
  export const RAG_CHUNK_OVERLAP: string;
  export const RAG_SIMILARITY_THRESHOLD: string;
  export const RAG_MAX_RESULTS: string;
  export const RAG_CACHE_TTL: string;
  export const CHUNK_SIZE: string;
  export const CHUNK_OVERLAP: string;
  export const MAX_CONTEXT_CHUNKS: string;
  export const SIMILARITY_THRESHOLD: string;

  // Neo4j Configuration
  export const NEO4J_URI: string;
  export const NEO4J_USERNAME: string;
  export const NEO4J_PASSWORD: string;
  export const NEO4J_DATABASE: string;

  // RabbitMQ Configuration
  export const RABBITMQ_URL: string;
  export const DOCUMENT_QUEUE_NAME: string;
  export const EMBEDDING_QUEUE_NAME: string;
  export const JOB_RETRY_ATTEMPTS: string;
  export const JOB_TIMEOUT: string;

  // Development
  export const NODE_ENV: string;
  export const PORT: string;
  export const HOST: string;
  export const LOG_LEVEL: string;
  export const METRICS_ENABLED: string;
  export const HEALTH_CHECK_INTERVAL: string;

  // YoRHa Theme
  export const YORHA_THEME_ENABLED: string;
  export const YORHA_TERMINAL_ENABLED: string;
  export const YORHA_ANIMATIONS_ENABLED: string;

  // Evidence Processing
  export const EVIDENCE_STORAGE_PATH: string;
  export const EVIDENCE_PROCESSING_QUEUE: string;
  export const EVIDENCE_METADATA_CACHE_TTL: string;
  export const MAX_EVIDENCE_FILE_SIZE: string;
  export const SUPPORTED_EVIDENCE_FORMATS: string;

  // Auto-Solver Configuration
  export const AUTO_SOLVER_ENABLED: string;
  export const AUTO_SOLVER_MAX_WORKERS: string;
  export const AUTO_SOLVER_GPU_ENABLED: string;

  // Cluster Configuration
  export const CLUSTER_LEGAL_WORKERS: string;
  export const CLUSTER_AI_WORKERS: string;
  export const CLUSTER_VECTOR_WORKERS: string;
  export const CLUSTER_DATABASE_WORKERS: string;
  export const CLUSTER_LEGAL_BASE_PORT: string;
  export const CLUSTER_AI_BASE_PORT: string;
  export const CLUSTER_VECTOR_BASE_PORT: string;
  export const CLUSTER_DATABASE_BASE_PORT: string;
}

declare module '$env/dynamic/private' {
  export const env: Record<string, string | undefined>;
}

declare module '$env/static/public' {
  export const PUBLIC_ENVIRONMENT: string;
}

declare module '$env/dynamic/public' {
  export const env: Record<string, string | undefined>;
}
