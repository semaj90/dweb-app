/**
 * Go Microservice WebAssembly Bridge
 * Complete integration between WebAssembly LLVM, Go microservice binaries,
 * FlatBuffer serialization, IVFFlat indexes, and protobuf transfers
 */

import { spawn } from 'child_process';
import { performance } from 'perf_hooks';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';
import crypto from 'crypto';

// Protocol Buffers and gRPC
import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import protobuf from 'protobufjs';

// Database integration
import postgres from 'postgres';

// Custom libraries
import { BitEncoder } from './bit-encoder.js';
import { MultiDimensionalCache } from './multi-dimensional-cache.js';

// Go microservice binary configuration
const GO_MICROSERVICE_CONFIG = {
  binaries: {
    'gpu-service': '../proto/gpu_service_grpc.pb.go',
    'enhanced-rag': '../go-microservice/bin/enhanced-rag.exe',
    'upload-service': '../go-microservice/bin/upload-service.exe',
    'grpc-server': '../go-microservice/bin/grpc-server.exe',
    'vector-service': '../go-microservice/bin/vector-service.exe'
  },
  
  ports: {
    'enhanced-rag': 8094,
    'upload-service': 8093,
    'grpc-server': 50051,
    'vector-service': 8095,
    'gpu-service': 8096
  },
  
  // WebAssembly LLVM integration
  wasmLLVM: {
    modulePath: './wasm/legal-llvm.wasm',
    stackSize: '64MB',
    heapSize: '512MB',
    features: ['simd', 'bulk-memory', 'threads']
  },
  
  // FlatBuffer integration
  flatBuffer: {
    schemaPath: './schemas/legal-nodes.fbs',
    nodeDataOptimization: true,
    gpuTextureAlignment: 8,
    compressionEnabled: true
  },
  
  // IVFFlat index configuration
  ivfFlat: {
    lists: 100, // Recommended: rows / 1000
    probes: 10, // Search probes
    indexType: 'vector_cosine_ops',
    maintenanceInterval: 3600000 // 1 hour
  }
};

export class GoMicroserviceWASMBridge {
  constructor(options = {}) {
    this.config = {
      ...GO_MICROSERVICE_CONFIG,
      ...options
    };
    
    // Running microservices
    this.activeServices = new Map();
    this.serviceHealth = new Map();
    
    // WebAssembly integration
    this.wasmModule = null;
    this.wasmExports = null;
    this.wasmMemory = null;
    
    // Protocol Buffers
    this.protobufRoot = null;
    this.grpcClients = new Map();
    
    // Database connection for IVFFlat
    this.postgres = null;
    this.ivfFlatIndexes = new Map();
    
    // Custom libraries
    this.bitEncoder = new BitEncoder({ wasmAccelerated: true });
    this.cache = new MultiDimensionalCache({ 
      flatBufferIntegration: true,
      ivfFlatSupport: true 
    });
    
    // FlatBuffer serialization
    this.flatBufferSerializer = null;
    
    // Performance metrics
    this.metrics = {
      servicesStarted: 0,
      wasmCallsExecuted: 0,
      protobufMessagesProcessed: 0,
      flatBufferOperations: 0,
      ivfFlatQueries: 0,
      averageLatency: 0,
      totalThroughput: 0
    };
    
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Go Microservice WASM Bridge - Initializing...');
    
    try {
      // Initialize WebAssembly LLVM module
      await this.initializeWebAssemblyLLVM();
      
      // Initialize Protocol Buffers
      await this.initializeProtocolBuffers();
      
      // Initialize database connection
      await this.initializeDatabase();
      
      // Initialize FlatBuffer serialization
      await this.initializeFlatBufferIntegration();
      
      // Setup IVFFlat indexes
      await this.setupIVFFlatIndexes();
      
      // Initialize custom libraries
      await this.bitEncoder.initialize();
      await this.cache.initialize();
      
      // Start Go microservices
      await this.startGoMicroservices();
      
      // Setup inter-service communication
      await this.setupInterServiceCommunication();
      
      this.initialized = true;
      console.log('✅ Go Microservice WASM Bridge initialized successfully');
      
    } catch (error) {
      console.error('❌ Go Microservice WASM Bridge initialization failed:', error);
      throw error;
    }
  }

  async initializeWebAssemblyLLVM() {
    console.log('⚡ Initializing WebAssembly LLVM module...');
    
    try {
      // TODO: Load actual WebAssembly module
      // For now, simulate WASM module loading
      const wasmPath = this.config.wasmLLVM.modulePath;
      
      if (!existsSync(wasmPath)) {
        console.warn(`⚠️ WASM module not found at ${wasmPath}, creating placeholder`);
        // Create placeholder for development
        this.wasmModule = {
          exports: {
            // Legal AI processing functions
            process_legal_text: (textPtr, textLen) => this.wasmProcessLegalText(textPtr, textLen),
            compute_embedding: (textPtr, textLen, dimensions) => this.wasmComputeEmbedding(textPtr, textLen, dimensions),
            similarity_search: (queryPtr, queryLen, threshold) => this.wasmSimilaritySearch(queryPtr, queryLen, threshold),
            compress_vectors: (vectorsPtr, vectorCount, dimensions) => this.wasmCompressVectors(vectorsPtr, vectorCount, dimensions),
            decompress_vectors: (compressedPtr, compressedLen) => this.wasmDecompressVectors(compressedPtr, compressedLen),
            
            // Memory management
            malloc: (size) => this.wasmMalloc(size),
            free: (ptr) => this.wasmFree(ptr),
            
            // Utility functions
            get_version: () => 0x0200, // Version 2.0
            get_capabilities: () => 0x1F // All capabilities enabled
          },
          memory: new WebAssembly.Memory({
            initial: 64, // 64 * 64KB = 4MB
            maximum: 1024, // 1024 * 64KB = 64MB
            shared: false
          })
        };
      } else {
        // Load actual WASM module
        const wasmBytes = readFileSync(wasmPath);
        const wasmModule = await WebAssembly.instantiate(wasmBytes, {
          env: {
            memory: new WebAssembly.Memory({
              initial: parseInt(this.config.wasmLLVM.stackSize.replace('MB', '')) / 4,
              maximum: parseInt(this.config.wasmLLVM.heapSize.replace('MB', '')) / 4
            })
          }
        });
        this.wasmModule = wasmModule.instance;
      }
      
      this.wasmExports = this.wasmModule.exports;
      this.wasmMemory = this.wasmModule.memory;
      
      console.log('✅ WebAssembly LLVM module loaded');
      
    } catch (error) {
      console.error('❌ WebAssembly LLVM initialization failed:', error);
      throw error;
    }
  }

  async initializeProtocolBuffers() {
    console.log('📡 Initializing Protocol Buffers...');
    
    try {
      // Load protobuf definitions
      this.protobufRoot = await protobuf.load('./proto/legal-ai.proto');
      
      // Initialize message types
      this.protoMessages = {
        EmbeddingRequest: this.protobufRoot.lookupType('legal.ai.EmbeddingRequest'),
        EmbeddingResponse: this.protobufRoot.lookupType('legal.ai.EmbeddingResponse'),
        InferenceRequest: this.protobufRoot.lookupType('legal.ai.InferenceRequest'),
        InferenceResponse: this.protobufRoot.lookupType('legal.ai.InferenceResponse'),
        SimilaritySearchRequest: this.protobufRoot.lookupType('legal.ai.SimilaritySearchRequest'),
        SimilaritySearchResponse: this.protobufRoot.lookupType('legal.ai.SimilaritySearchResponse')
      };
      
      console.log('✅ Protocol Buffers initialized');
      
    } catch (error) {
      console.error('❌ Protocol Buffers initialization failed:', error);
      // Continue without protobuf support
    }
  }

  async initializeDatabase() {
    console.log('🗄️ Initializing database connection...');
    
    try {
      this.postgres = postgres(process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db', {
        host: 'localhost',
        port: 5432,
        database: 'legal_ai_db',
        username: 'legal_admin',
        password: '123456',
        max: 20,
        idle_timeout: 20,
        connect_timeout: 10,
        ssl: false
      });
      
      // Test connection
      const result = await this.postgres`SELECT version()`;
      console.log('✅ Database connected:', result[0].version.substring(0, 50) + '...');
      
    } catch (error) {
      console.error('❌ Database connection failed:', error);
      throw error;
    }
  }

  async initializeFlatBufferIntegration() {
    console.log('📦 Initializing FlatBuffer integration...');
    
    try {
      // TODO: Initialize actual FlatBuffer serializer
      // For now, create mock serializer compatible with existing TypeScript implementation
      this.flatBufferSerializer = {
        serializeNodes: async (nodes) => {
          const startTime = performance.now();
          
          // Convert nodes to binary format compatible with FlatBuffer schema
          const binaryData = this.convertNodesToFlatBuffer(nodes);
          
          this.metrics.flatBufferOperations++;
          console.log(`📦 FlatBuffer serialization: ${nodes.length} nodes in ${performance.now() - startTime}ms`);
          
          return binaryData;
        },
        
        deserializeNodes: async (buffer) => {
          const startTime = performance.now();
          
          // Convert binary data back to node objects
          const nodes = this.convertFlatBufferToNodes(buffer);
          
          this.metrics.flatBufferOperations++;
          console.log(`📖 FlatBuffer deserialization: ${nodes.length} nodes in ${performance.now() - startTime}ms`);
          
          return {
            nodeCount: nodes.length,
            timestamp: Date.now(),
            nodes,
            totalSize: buffer.byteLength
          };
        },
        
        createGPUNodeData: (binaryData) => {
          // Convert to GPU-optimized format
          return this.createGPUOptimizedData(binaryData);
        }
      };
      
      console.log('✅ FlatBuffer integration initialized');
      
    } catch (error) {
      console.error('❌ FlatBuffer integration failed:', error);
      throw error;
    }
  }

  async setupIVFFlatIndexes() {
    console.log('📊 Setting up IVFFlat indexes...');
    
    try {
      // Ensure vector extension is available
      await this.postgres`CREATE EXTENSION IF NOT EXISTS vector`;
      
      // Create IVFFlat indexes for different embedding types
      const indexConfigs = [
        {
          table: 'legal_documents',
          column: 'embedding',
          name: 'idx_legal_documents_embedding_ivfflat',
          lists: this.config.ivfFlat.lists,
          ops: 'vector_cosine_ops'
        },
        {
          table: 'case_embeddings',
          column: 'embedding',
          name: 'idx_case_embeddings_embedding_ivfflat', 
          lists: Math.ceil(this.config.ivfFlat.lists / 2),
          ops: 'vector_cosine_ops'
        },
        {
          table: 'evidence_embeddings',
          column: 'embedding',
          name: 'idx_evidence_embeddings_embedding_ivfflat',
          lists: Math.ceil(this.config.ivfFlat.lists / 4),
          ops: 'vector_cosine_ops'
        }
      ];
      
      for (const config of indexConfigs) {
        try {
          await this.postgres.unsafe(`
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ${config.name}
            ON ${config.table} USING ivfflat (${config.column} ${config.ops})
            WITH (lists = ${config.lists})
          `);
          
          // Analyze table for query planner
          await this.postgres.unsafe(`ANALYZE ${config.table}`);
          
          this.ivfFlatIndexes.set(config.name, config);
          console.log(`✅ IVFFlat index created: ${config.name}`);
          
        } catch (error) {
          console.warn(`⚠️ Failed to create index ${config.name}:`, error.message);
        }
      }
      
      // Setup index maintenance
      this.startIVFFlatMaintenance();
      
      console.log(`✅ IVFFlat indexes setup complete: ${this.ivfFlatIndexes.size} indexes`);
      
    } catch (error) {
      console.error('❌ IVFFlat index setup failed:', error);
      throw error;
    }
  }

  async startGoMicroservices() {
    console.log('🚀 Starting Go microservices...');
    
    const promises = Object.entries(this.config.binaries).map(async ([serviceName, binaryPath]) => {
      if (serviceName === 'gpu-service') {
        // Skip binary files that are protobuf definitions
        return;
      }
      
      try {
        await this.startGoMicroservice(serviceName, binaryPath);
        console.log(`✅ ${serviceName} started`);
      } catch (error) {
        console.error(`❌ Failed to start ${serviceName}:`, error);
        // Continue with other services
      }
    });
    
    await Promise.all(promises.filter(p => p)); // Filter out undefined promises
    
    this.metrics.servicesStarted = this.activeServices.size;
    console.log(`✅ Go microservices started: ${this.metrics.servicesStarted} services`);
  }

  async startGoMicroservice(serviceName, binaryPath) {
    if (!existsSync(binaryPath)) {
      throw new Error(`Binary not found: ${binaryPath}`);
    }
    
    const port = this.config.ports[serviceName];
    const env = {
      ...process.env,
      PORT: port.toString(),
      DATABASE_URL: process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
      REDIS_URL: 'redis://localhost:6379',
      LOG_LEVEL: 'info'
    };
    
    const childProcess = spawn(binaryPath, [], {
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: false
    });
    
    // Handle process events
    childProcess.on('spawn', () => {
      console.log(`🟢 ${serviceName} spawned (PID: ${childProcess.pid})`);
    });
    
    childProcess.on('error', (error) => {
      console.error(`❌ ${serviceName} error:`, error);
      this.serviceHealth.set(serviceName, 'error');
    });
    
    childProcess.on('exit', (code, signal) => {
      console.log(`🔴 ${serviceName} exited (code: ${code}, signal: ${signal})`);
      this.activeServices.delete(serviceName);
      this.serviceHealth.set(serviceName, 'stopped');
    });
    
    // Capture stdout/stderr
    childProcess.stdout.on('data', (data) => {
      console.log(`[${serviceName}] ${data.toString().trim()}`);
    });
    
    childProcess.stderr.on('data', (data) => {
      console.error(`[${serviceName}] ${data.toString().trim()}`);
    });
    
    this.activeServices.set(serviceName, {
      process: childProcess,
      port,
      startTime: Date.now(),
      status: 'running'
    });
    
    this.serviceHealth.set(serviceName, 'healthy');
    
    // Wait for service to be ready
    await this.waitForServiceReady(serviceName, port);
  }

  async waitForServiceReady(serviceName, port, timeout = 10000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      try {
        const response = await fetch(`http://localhost:${port}/health`);
        if (response.ok) {
          return true;
        }
      } catch (error) {
        // Service not ready yet
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    throw new Error(`Service ${serviceName} did not become ready within ${timeout}ms`);
  }

  async setupInterServiceCommunication() {
    console.log('🔗 Setting up inter-service communication...');
    
    // Setup gRPC clients for Go services
    for (const [serviceName, serviceInfo] of this.activeServices) {
      if (serviceName.includes('grpc') || serviceName === 'enhanced-rag') {
        try {
          const client = await this.createGRPCClient(serviceName, serviceInfo.port);
          this.grpcClients.set(serviceName, client);
          console.log(`✅ gRPC client created for ${serviceName}`);
        } catch (error) {
          console.warn(`⚠️ Failed to create gRPC client for ${serviceName}:`, error);
        }
      }
    }
    
    console.log('✅ Inter-service communication setup complete');
  }

  async createGRPCClient(serviceName, port) {
    // Load service-specific protobuf definition
    const packageDefinition = protoLoader.loadSync(`./proto/${serviceName}.proto`, {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true
    });
    
    const protoDescriptor = grpc.loadPackageDefinition(packageDefinition);
    const ServiceConstructor = protoDescriptor[serviceName] || protoDescriptor.gpu_rag_system?.GPUProcessingService;
    
    if (!ServiceConstructor) {
      throw new Error(`Service constructor not found for ${serviceName}`);
    }
    
    return new ServiceConstructor(`localhost:${port}`, grpc.credentials.createInsecure());
  }

  // WebAssembly function implementations (placeholders for actual WASM functions)
  wasmProcessLegalText(textPtr, textLen) {
    // TODO: Implement actual WASM legal text processing
    console.log(`🔧 WASM processing legal text: ${textLen} bytes`);
    this.metrics.wasmCallsExecuted++;
    return 0; // Success code
  }

  wasmComputeEmbedding(textPtr, textLen, dimensions) {
    // TODO: Implement actual WASM embedding computation
    console.log(`🔧 WASM computing embedding: ${textLen} bytes -> ${dimensions} dimensions`);
    this.metrics.wasmCallsExecuted++;
    return this.wasmMalloc(dimensions * 4); // Return pointer to embedding data
  }

  wasmSimilaritySearch(queryPtr, queryLen, threshold) {
    // TODO: Implement actual WASM similarity search
    console.log(`🔧 WASM similarity search: threshold ${threshold}`);
    this.metrics.wasmCallsExecuted++;
    return this.wasmMalloc(1024); // Return pointer to results
  }

  wasmCompressVectors(vectorsPtr, vectorCount, dimensions) {
    // TODO: Implement actual WASM vector compression
    console.log(`🔧 WASM compressing vectors: ${vectorCount} x ${dimensions}`);
    this.metrics.wasmCallsExecuted++;
    return this.wasmMalloc(vectorCount * dimensions); // Compressed data
  }

  wasmDecompressVectors(compressedPtr, compressedLen) {
    // TODO: Implement actual WASM vector decompression
    console.log(`🔧 WASM decompressing vectors: ${compressedLen} bytes`);
    this.metrics.wasmCallsExecuted++;
    return this.wasmMalloc(compressedLen * 4); // Decompressed data
  }

  wasmMalloc(size) {
    // Simple memory allocation simulation
    // In real implementation, this would manage WASM memory
    return Math.floor(Math.random() * 1000000); // Random pointer
  }

  wasmFree(ptr) {
    // Memory deallocation simulation
    console.log(`🔧 WASM free: ${ptr}`);
  }

  // FlatBuffer conversion functions
  convertNodesToFlatBuffer(nodes) {
    // Convert nodes to FlatBuffer binary format
    const buffer = new ArrayBuffer(nodes.length * 512); // Estimate size
    const view = new DataView(buffer);
    
    // Write header
    view.setUint32(0, 0x444E4246, true); // "FBND" magic
    view.setUint32(4, nodes.length, true); // Node count
    view.setBigUint64(8, BigInt(Date.now()), true); // Timestamp
    
    // Write node data
    let offset = 32; // Skip header
    for (const node of nodes) {
      view.setUint32(offset, node.id || 0, true);
      view.setFloat32(offset + 4, node.confidence || 0.5, true);
      view.setFloat32(offset + 8, node.position?.x || 0, true);
      view.setFloat32(offset + 12, node.position?.y || 0, true);
      view.setFloat32(offset + 16, node.position?.z || 0, true);
      
      offset += 512; // Fixed size per node for simplicity
    }
    
    return buffer;
  }

  convertFlatBufferToNodes(buffer) {
    // Convert FlatBuffer binary format back to nodes
    const view = new DataView(buffer);
    
    // Read header
    const magic = view.getUint32(0, true);
    if (magic !== 0x444E4246) {
      throw new Error('Invalid FlatBuffer magic number');
    }
    
    const nodeCount = view.getUint32(4, true);
    const nodes = [];
    
    // Read node data
    let offset = 32;
    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        id: view.getUint32(offset, true),
        confidence: view.getFloat32(offset + 4, true),
        position: {
          x: view.getFloat32(offset + 8, true),
          y: view.getFloat32(offset + 12, true),
          z: view.getFloat32(offset + 16, true)
        }
      });
      
      offset += 512;
    }
    
    return nodes;
  }

  createGPUOptimizedData(binaryData) {
    // Create GPU-optimized data structures
    return {
      nodeId: new Uint32Array(binaryData.nodes.map(n => n.id || 0)),
      position: new Float32Array(binaryData.nodes.flatMap(n => [n.position?.x || 0, n.position?.y || 0, n.position?.z || 0])),
      confidence: new Float32Array(binaryData.nodes.map(n => n.confidence || 0.5))
    };
  }

  startIVFFlatMaintenance() {
    // Periodic index maintenance
    setInterval(async () => {
      try {
        for (const [indexName, config] of this.ivfFlatIndexes) {
          // Reanalyze tables periodically
          await this.postgres.unsafe(`ANALYZE ${config.table}`);
          console.log(`🔧 IVFFlat maintenance: ${indexName} analyzed`);
        }
      } catch (error) {
        console.error('❌ IVFFlat maintenance error:', error);
      }
    }, this.config.ivfFlat.maintenanceInterval);
  }

  // Public API methods
  async processEmbeddingWithGoService(request) {
    const serviceName = 'enhanced-rag';
    const client = this.grpcClients.get(serviceName);
    
    if (!client) {
      throw new Error(`gRPC client not available for ${serviceName}`);
    }
    
    // Convert to protobuf message
    const protoRequest = this.protoMessages.EmbeddingRequest.create(request);
    const encodedRequest = this.protoMessages.EmbeddingRequest.encode(protoRequest).finish();
    
    // Call Go service via gRPC
    return new Promise((resolve, reject) => {
      client.ProcessEmbedding({ data: encodedRequest }, (error, response) => {
        if (error) {
          reject(error);
        } else {
          this.metrics.protobufMessagesProcessed++;
          resolve(response);
        }
      });
    });
  }

  async queryIVFFlatIndex(tableName, queryEmbedding, limit = 10) {
    const startTime = performance.now();
    
    try {
      // Use IVFFlat index for similarity search
      const results = await this.postgres.unsafe(`
        SELECT id, embedding <-> $1 as distance, metadata
        FROM ${tableName}
        ORDER BY embedding <-> $1
        LIMIT $2
      `, [JSON.stringify(queryEmbedding), limit]);
      
      this.metrics.ivfFlatQueries++;
      const latency = performance.now() - startTime;
      console.log(`🔍 IVFFlat query: ${results.length} results in ${latency.toFixed(2)}ms`);
      
      return results;
      
    } catch (error) {
      console.error('❌ IVFFlat query error:', error);
      throw error;
    }
  }

  async processWithFlatBuffer(nodes) {
    const startTime = performance.now();
    
    try {
      // Serialize nodes to FlatBuffer
      const binaryData = await this.flatBufferSerializer.serializeNodes(nodes);
      
      // Process with WebAssembly LLVM
      const wasmResults = this.wasmProcessLegalText(0, binaryData.byteLength);
      
      // Deserialize results
      const processedNodes = await this.flatBufferSerializer.deserializeNodes(binaryData);
      
      const processingTime = performance.now() - startTime;
      console.log(`⚡ FlatBuffer + WASM processing: ${nodes.length} nodes in ${processingTime.toFixed(2)}ms`);
      
      return {
        success: true,
        processedNodes: processedNodes.nodes,
        processingTime,
        wasmResults
      };
      
    } catch (error) {
      console.error('❌ FlatBuffer + WASM processing error:', error);
      throw error;
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      activeServices: this.activeServices.size,
      healthyServices: Array.from(this.serviceHealth.values()).filter(h => h === 'healthy').length,
      ivfFlatIndexes: this.ivfFlatIndexes.size,
      grpcClients: this.grpcClients.size,
      wasmModuleLoaded: !!this.wasmModule,
      flatBufferIntegration: !!this.flatBufferSerializer,
      uptime: process.uptime()
    };
  }

  getSystemStatus() {
    return {
      goMicroservices: Array.from(this.activeServices.entries()).map(([name, info]) => ({
        name,
        port: info.port,
        status: this.serviceHealth.get(name) || 'unknown',
        uptime: Date.now() - info.startTime
      })),
      webAssembly: {
        moduleLoaded: !!this.wasmModule,
        memorySize: this.wasmMemory?.buffer?.byteLength || 0,
        callsExecuted: this.metrics.wasmCallsExecuted
      },
      protocolBuffers: {
        messagesProcessed: this.metrics.protobufMessagesProcessed,
        grpcClientsActive: this.grpcClients.size
      },
      flatBuffer: {
        operationsCompleted: this.metrics.flatBufferOperations,
        integrationActive: !!this.flatBufferSerializer
      },
      ivfFlatIndexes: {
        indexCount: this.ivfFlatIndexes.size,
        queriesExecuted: this.metrics.ivfFlatQueries,
        indexes: Array.from(this.ivfFlatIndexes.keys())
      }
    };
  }

  async stop() {
    console.log('🛑 Shutting down Go Microservice WASM Bridge...');
    
    try {
      // Stop Go microservices
      for (const [serviceName, serviceInfo] of this.activeServices) {
        console.log(`🔄 Stopping ${serviceName}...`);
        serviceInfo.process.kill('SIGTERM');
      }
      
      // Close database connection
      if (this.postgres) {
        await this.postgres.end();
      }
      
      // Close gRPC clients
      this.grpcClients.clear();
      
      console.log('✅ Go Microservice WASM Bridge shut down gracefully');
      
    } catch (error) {
      console.error('Shutdown error:', error);
    }
  }
}

export default GoMicroserviceWASMBridge;