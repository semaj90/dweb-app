/**
 * Node.js GPU Microservice with WebGPU (kmamal/gpu)
 * High-performance embedding processing using Dawn WebGPU bindings
 */

import { Server, ServerCredentials } from '@grpc/grpc-js';
import { GPUServiceService } from '../proto/gpu_service_grpc_pb';
import { GPUProcessor } from './gpu-processor';
import { MetricsCollector } from './metrics-collector';
import { ShaderManager } from './shader-manager';
import { Logger } from './logger';
import cluster from 'cluster';
import os from 'os';

// Import WebGPU bindings
const gpu = require('kmamal/gpu');

interface ServiceConfig {
  port: number;
  gpuDevice: string;
  maxBatchSize: number;
  shaderCacheSize: number;
  enableMetrics: boolean;
  logLevel: string;
  clusterMode: boolean;
}

class NodeGPUService {
  private server: Server;
  private gpuProcessor: GPUProcessor;
  private metricsCollector: MetricsCollector;
  private shaderManager: ShaderManager;
  private logger: Logger;
  private config: ServiceConfig;

  constructor(config: ServiceConfig) {
    this.config = config;
    this.logger = new Logger(config.logLevel);
    this.server = new Server({
      'grpc.max_receive_message_length': 64 * 1024 * 1024, // 64MB
      'grpc.max_send_message_length': 64 * 1024 * 1024,
    });
  }

  async initialize(): Promise<void> {
    this.logger.info('🚀 Initializing Node.js GPU Service');

    try {
      // Initialize WebGPU
      await this.initializeWebGPU();

      // Initialize components
      this.shaderManager = new ShaderManager(this.logger);
      await this.shaderManager.initialize();

      this.gpuProcessor = new GPUProcessor({
        gpu: this.getGPUDevice(),
        shaderManager: this.shaderManager,
        maxBatchSize: this.config.maxBatchSize,
        logger: this.logger
      });
      await this.gpuProcessor.initialize();

      if (this.config.enableMetrics) {
        this.metricsCollector = new MetricsCollector(this.logger);
        this.metricsCollector.start();
      }

      // Register gRPC services
      this.registerServices();

      this.logger.info('✅ Service initialization completed');
    } catch (error) {
      this.logger.error('❌ Service initialization failed:', error);
      throw error;
    }
  }

  private async initializeWebGPU(): Promise<void> {
    this.logger.info('🎮 Initializing WebGPU...');

    try {
      // Check GPU availability
      const gpuDevices = gpu.getDevices();
      this.logger.info(`Found ${gpuDevices.length} GPU devices:`, gpuDevices);

      if (gpuDevices.length === 0) {
        throw new Error('No GPU devices found');
      }

      // Initialize default adapter
      const adapter = await gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!adapter) {
        throw new Error('Failed to get WebGPU adapter');
      }

      this.logger.info('✅ WebGPU initialized successfully');
      this.logger.info(`GPU Info: ${adapter.info.vendor} ${adapter.info.device}`);
    } catch (error) {
      this.logger.error('❌ WebGPU initialization failed:', error);
      throw error;
    }
  }

  private getGPUDevice(): any {
    const devices = gpu.getDevices();
    return devices[0]; // Use first available device
  }

  private registerServices(): void {
    this.logger.info('📝 Registering gRPC services...');

    this.server.addService(GPUServiceService, {
      processEmbeddings: this.gpuProcessor.processEmbeddings.bind(this.gpuProcessor),
      performClustering: this.gpuProcessor.performClustering.bind(this.gpuProcessor),
      computeSimilarity: this.gpuProcessor.computeSimilarity.bind(this.gpuProcessor),
      applyBoostTransform: this.gpuProcessor.applyBoostTransform.bind(this.gpuProcessor),
      processDocument: this.gpuProcessor.processDocument.bind(this.gpuProcessor),
      streamDocuments: this.gpuProcessor.streamDocuments.bind(this.gpuProcessor),
      getHealthStatus: this.getHealthStatus.bind(this)
    });

    this.logger.info('✅ gRPC services registered');
  }

  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server.bindAsync(
        `0.0.0.0:${this.config.port}`,
        ServerCredentials.createInsecure(),
        (error: Error | null, port: number) => {
          if (error) {
            this.logger.error('❌ Failed to start server:', error);
            reject(error);
            return;
          }

          this.server.start();
          this.logger.info(`🎯 Node.js GPU Service started on port ${port}`);
          this.logger.info(`🔧 GPU Device: ${this.config.gpuDevice}`);
          this.logger.info(`📦 Max Batch Size: ${this.config.maxBatchSize}`);
          this.logger.info(`🧠 Shader Cache Size: ${this.config.shaderCacheSize}`);

          if (this.config.enableMetrics) {
            this.startMetricsReporting();
          }

          resolve();
        }
      );
    });
  }

  private startMetricsReporting(): void {
    setInterval(() => {
      const metrics = this.metricsCollector.getMetrics();
      this.logger.info('📊 Performance Metrics:', {
        embeddingsProcessed: metrics.embeddingsProcessed,
        avgProcessingTime: `${metrics.avgProcessingTime.toFixed(2)}ms`,
        gpuUtilization: `${metrics.gpuUtilization.toFixed(1)}%`,
        memoryUsage: `${metrics.memoryUsage.toFixed(1)}MB`,
        cacheHitRate: `${metrics.cacheHitRate.toFixed(1)}%`
      });
    }, 30000); // Report every 30 seconds
  }

  private async getHealthStatus(call: any, callback: any): Promise<void> {
    try {
      const metrics = this.config.enableMetrics ? this.metricsCollector.getMetrics() : null;
      
      const response = {
        status: 'healthy',
        uptime: process.uptime(),
        metrics: metrics ? {
          embeddings_processed: metrics.embeddingsProcessed.toString(),
          avg_processing_time: metrics.avgProcessingTime.toString(),
          gpu_utilization: metrics.gpuUtilization.toString(),
          memory_usage: metrics.memoryUsage.toString(),
          cache_hit_rate: metrics.cacheHitRate.toString()
        } : {}
      };

      callback(null, response);
    } catch (error) {
      callback(error, null);
    }
  }

  async shutdown(): Promise<void> {
    this.logger.info('🛑 Shutting down Node.js GPU Service...');

    if (this.metricsCollector) {
      this.metricsCollector.stop();
    }

    if (this.gpuProcessor) {
      await this.gpuProcessor.cleanup();
    }

    if (this.shaderManager) {
      this.shaderManager.cleanup();
    }

    return new Promise((resolve) => {
      this.server.tryShutdown(() => {
        this.logger.info('✅ Service shutdown completed');
        resolve();
      });
    });
  }
}

// Cluster mode support
function startClusterMode(config: ServiceConfig): void {
  const numCPUs = os.cpus().length;

  if (cluster.isMaster) {
    console.log(`🚀 Starting cluster with ${numCPUs} workers`);

    // Fork workers
    for (let i = 0; i < numCPUs; i++) {
      cluster.fork();
    }

    cluster.on('exit', (worker, code, signal) => {
      console.log(`Worker ${worker.process.pid} died with code ${code} and signal ${signal}`);
      cluster.fork(); // Respawn worker
    });
  } else {
    startSingleInstance(config);
  }
}

async function startSingleInstance(config: ServiceConfig): Promise<void> {
  const service = new NodeGPUService(config);

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\n🛑 Received SIGINT, shutting down gracefully...');
    await service.shutdown();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
    await service.shutdown();
    process.exit(0);
  });

  try {
    await service.initialize();
    await service.start();
  } catch (error) {
    console.error('❌ Failed to start service:', error);
    process.exit(1);
  }
}

// Main entry point
async function main(): Promise<void> {
  const config: ServiceConfig = {
    port: parseInt(process.env.PORT || '50052'),
    gpuDevice: process.env.GPU_DEVICE || 'auto',
    maxBatchSize: parseInt(process.env.MAX_BATCH_SIZE || '32'),
    shaderCacheSize: parseInt(process.env.SHADER_CACHE_SIZE || '100'),
    enableMetrics: process.env.ENABLE_METRICS !== 'false',
    logLevel: process.env.LOG_LEVEL || 'info',
    clusterMode: process.env.CLUSTER_MODE === 'true'
  };

  console.log('🎮 Node.js GPU Service Configuration:');
  console.log(`   Port: ${config.port}`);
  console.log(`   GPU Device: ${config.gpuDevice}`);
  console.log(`   Max Batch Size: ${config.maxBatchSize}`);
  console.log(`   Shader Cache Size: ${config.shaderCacheSize}`);
  console.log(`   Enable Metrics: ${config.enableMetrics}`);
  console.log(`   Log Level: ${config.logLevel}`);
  console.log(`   Cluster Mode: ${config.clusterMode}`);
  console.log('');

  if (config.clusterMode) {
    startClusterMode(config);
  } else {
    await startSingleInstance(config);
  }
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

if (require.main === module) {
  main().catch((error) => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

export { NodeGPUService, ServiceConfig };