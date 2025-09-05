// PM2 Configuration for Legal AI Native Windows Stack
module.exports = {
  apps: [
    // Legal BERT ONNX Service (Go)
    {
      name: 'legal-bert-onnx',
      script: './go-services/legal-bert-onnx/legal-bert-service.exe',
      env: {
        PORT: 8081,
        MODEL_PATH: './models/legal-bert.onnx',
        PROTOCOL: 'grpc+http',
        ONNX_THREADS: 4,
        TENSOR_CACHE_SIZE: '2GB'
      },
      instances: 1,
      exec_mode: 'fork',
      restart_delay: 5000,
      max_restarts: 5
    },
    
    // Qdrant Vector Service (Rust)
    {
      name: 'qdrant-vector', 
      script: './rust-services/qdrant-vector/target/release/qdrant-service.exe',
      env: {
        QDRANT_PORT: 6334,
        VECTOR_DIM: 768,
        WEBASM_BRIDGE: 'true',
        COLLECTION_NAME: 'legal_documents',
        RUST_LOG: 'info'
      },
      instances: 1,
      exec_mode: 'fork',
      restart_delay: 3000
    },
    
    // SIMD Processing Service (Go)
    {
      name: 'go-simd',
      script: './go-services/simd-operations/simd-service.exe',
      env: {
        SIMD_PORT: 8082,
        ENABLE_AVX512: 'true',
        TENSOR_ACCELERATION: 'true',
        BATCH_SIZE: 64,
        WORKER_THREADS: 8
      },
      instances: 1,
      exec_mode: 'fork'
    },
    
    // QUIC Coordination Service (Go)
    {
      name: 'quic-coordinator',
      script: './go-services/quic-server/quic-service.exe',
      env: {
        QUIC_PORT: 8083,
        IPC_BRIDGE: '\\\\.\\pipe\\legal-ai-coord',
        TLS_CERT_PATH: './certs/server.crt',
        TLS_KEY_PATH: './certs/server.key'
      },
      instances: 1,
      exec_mode: 'fork'
    },
    
    // Enhanced Context7 MCP Server
    {
      name: 'context7-mcp',
      script: './mcp/custom-context7-server.js',
      env: {
        PORT: 3000,
        OLLAMA_URL: 'http://localhost:11434',
        OLLAMA_EMBED_MODEL: 'nomic-embed-text',
        WATCH_FILES: 'true',
        CONTENT_DIR: './documents'
      },
      instances: 1,
      exec_mode: 'fork'
    },
    
    // SvelteKit Frontend
    {
      name: 'sveltekit-frontend',
      script: 'npm',
      args: 'run dev',
      cwd: './sveltekit-frontend',
      env: {
        PORT: 5173,
        WEBGPU_ENABLED: 'true',
        XSTATE_DEVTOOLS: 'true',
        VITE_API_BASE: 'http://localhost:3000'
      },
      instances: 1,
      exec_mode: 'fork'
    }
  ]
};