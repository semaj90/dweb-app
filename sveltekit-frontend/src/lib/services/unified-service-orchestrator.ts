// Unified Service Orchestrator - Complete Integration Layer
// Manages all GPU/WASM services, QUIC communication, and system coordination

import { UnifiedWASMGPUOrchestrator } from './unified-wasm-gpu-orchestrator';
import { QUICGatewayClient } from './quic-gateway-client';
import { LlamaCppOllamaService } from './llamacpp-ollama-integration';
import { NESStyleGPUBridge, type CanvasState } from './nes-gpu-bridge';
import { dev } from '$app/environment';

// System health status
export interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'critical';
  services: {
    wasmGPU: 'online' | 'degraded' | 'offline';
    quicGateway: 'online' | 'degraded' | 'offline';
    llamaOllama: 'online' | 'degraded' | 'offline';
    nesGPUBridge: 'online' | 'degraded' | 'offline';
    postgres: 'online' | 'degraded' | 'offline';
    redis: 'online' | 'degraded' | 'offline';
  };
  performance: {
    averageLatency: number;
    throughput: number;
    gpuUtilization: number;
    memoryUsage: number;
  };
  lastChecked: Date;
}

// Task priority levels
export enum TaskPriority {
  CRITICAL = 0,
  HIGH = 1,
  NORMAL = 2,
  LOW = 3,
  BACKGROUND = 4
}

// Service task definition
export interface ServiceTask {
  id: string;
  type: 'document' | 'inference' | 'canvas' | 'gpu' | 'analysis';
  priority: TaskPriority;
  data: any;
  options?: any;
  retries: number;
  maxRetries: number;
  timeout: number;
  startTime?: Date;
  endTime?: Date;
  error?: string;
}

// Orchestration result
export interface OrchestrationResult {
  taskId: string;
  success: boolean;
  data?: any;
  error?: string;
  processingTime: number;
  servicesUsed: string[];
  fallbacksTriggered: string[];
  performance: {
    latency: number;
    throughput: number;
    resourceUsage: number;
  };
}

// Service configuration
export interface ServiceConfig {
  enabledServices: string[];
  fallbackPriority: string[];
  performanceThresholds: {
    maxLatency: number;
    minThroughput: number;
    maxCpuUsage: number;
    maxMemoryUsage: number;
  };
  retryConfiguration: {
    maxRetries: number;
    backoffMultiplier: number;
    initialDelay: number;
  };
  monitoring: {
    healthCheckInterval: number;
    metricsRetentionPeriod: number;
    alertThresholds: Record<string, number>;
  };
}

export class UnifiedServiceOrchestrator {
  private wasmGPUOrchestrator: UnifiedWASMGPUOrchestrator;
  private quicClient: QUICGatewayClient;
  private llamaService: LlamaCppOllamaService;
  private nesGPUBridge: NESStyleGPUBridge;
  private taskQueue: ServiceTask[] = [];
  private activeTasks: Map<string, ServiceTask> = new Map();
  private systemHealth: SystemHealth;
  private config: ServiceConfig;
  private healthCheckInterval?: NodeJS.Timeout;
  private performanceMetrics: Array<{
    timestamp: Date;
    latency: number;
    throughput: number;
    resourceUsage: number;
  }> = [];

  constructor(config?: Partial<ServiceConfig>) {
    this.config = this.mergeConfig(config || {});
    this.initializeServices();
    this.initializeSystemHealth();
    this.startHealthMonitoring();
  }

  private mergeConfig(userConfig: Partial<ServiceConfig>): ServiceConfig {
    const defaultConfig: ServiceConfig = {
      enabledServices: ['wasmGPU', 'quicGateway', 'llamaOllama', 'nesGPUBridge'],
      fallbackPriority: ['wasmGPU', 'quicGateway', 'llamaOllama', 'cpu'],
      performanceThresholds: {
        maxLatency: 1000,
        minThroughput: 10,
        maxCpuUsage: 80,
        maxMemoryUsage: 70
      },
      retryConfiguration: {
        maxRetries: 3,
        backoffMultiplier: 2,
        initialDelay: 100
      },
      monitoring: {
        healthCheckInterval: 30000,
        metricsRetentionPeriod: 3600000,
        alertThresholds: {
          latency: 2000,
          errorRate: 5,
          throughput: 5
        }
      }
    };

    return {
      ...defaultConfig,
      ...userConfig,
      performanceThresholds: {
        ...defaultConfig.performanceThresholds,
        ...(userConfig.performanceThresholds || {})
      },
      retryConfiguration: {
        ...defaultConfig.retryConfiguration,
        ...(userConfig.retryConfiguration || {})
      },
      monitoring: {
        ...defaultConfig.monitoring,
        ...(userConfig.monitoring || {})
      }
    };
  }

  private initializeServices(): void {
    try {
      this.wasmGPUOrchestrator = new UnifiedWASMGPUOrchestrator();
      this.quicClient = new QUICGatewayClient();
      this.llamaService = new LlamaCppOllamaService();
      this.nesGPUBridge = new NESStyleGPUBridge();

      if (dev) {
        console.log('[Orchestrator] All services initialized successfully');
      }
    } catch (error) {
      console.error('[Orchestrator] Service initialization failed:', error);
      throw new Error(`Service initialization failed: ${error}`);
    }
  }

  private initializeSystemHealth(): void {
    this.systemHealth = {
      overall: 'healthy',
      services: {
        wasmGPU: 'online',
        quicGateway: 'online',
        llamaOllama: 'online',
        nesGPUBridge: 'online',
        postgres: 'online',
        redis: 'online'
      },
      performance: {
        averageLatency: 0,
        throughput: 0,
        gpuUtilization: 0,
        memoryUsage: 0
      },
      lastChecked: new Date()
    };
  }

  private startHealthMonitoring(): void {
    this.healthCheckInterval = setInterval(
      () => this.performHealthCheck(),
      this.config.monitoring.healthCheckInterval
    );
  }

  private async performHealthCheck(): Promise<void> {
    try {
      const startTime = Date.now();
      const healthChecks = await Promise.allSettled([
        this.checkWASMGPUHealth(),
        this.checkQUICGatewayHealth(),
        this.checkLlamaOllamaHealth(),
        this.checkNESGPUBridgeHealth(),
        this.checkPostgresHealth(),
        this.checkRedisHealth()
      ]);

      // Update service statuses
      this.systemHealth.services.wasmGPU = this.getHealthStatus(healthChecks[0]);
      this.systemHealth.services.quicGateway = this.getHealthStatus(healthChecks[1]);
      this.systemHealth.services.llamaOllama = this.getHealthStatus(healthChecks[2]);
      this.systemHealth.services.nesGPUBridge = this.getHealthStatus(healthChecks[3]);
      this.systemHealth.services.postgres = this.getHealthStatus(healthChecks[4]);
      this.systemHealth.services.redis = this.getHealthStatus(healthChecks[5]);

      // Calculate overall health
      const serviceStatuses = Object.values(this.systemHealth.services);
      const offlineCount = serviceStatuses.filter(s => s === 'offline').length;
      const degradedCount = serviceStatuses.filter(s => s === 'degraded').length;

      if (offlineCount > 2) {
        this.systemHealth.overall = 'critical';
      } else if (offlineCount > 0 || degradedCount > 1) {
        this.systemHealth.overall = 'degraded';
      } else {
        this.systemHealth.overall = 'healthy';
      }

      this.systemHealth.lastChecked = new Date();
      
      const healthCheckTime = Date.now() - startTime;
      if (dev) {
        console.log(`[Orchestrator] Health check completed in ${healthCheckTime}ms - Status: ${this.systemHealth.overall}`);
      }
    } catch (error) {
      console.error('[Orchestrator] Health check failed:', error);
      this.systemHealth.overall = 'critical';
    }
  }

  private getHealthStatus(result: PromiseSettledResult<boolean>): 'online' | 'degraded' | 'offline' {
    if (result.status === 'fulfilled') {
      return result.value ? 'online' : 'degraded';
    }
    return 'offline';
  }

  private async checkWASMGPUHealth(): Promise<boolean> {
    try {
      const testData = new Float32Array([1, 2, 3, 4]);
      const result = await this.wasmGPUOrchestrator.performNeuralInference(testData, {
        timeout: 5000,
        priority: 'low'
      });
      return result.success;
    } catch {
      return false;
    }
  }

  private async checkQUICGatewayHealth(): Promise<boolean> {
    try {
      const response = await this.quicClient.request({
        method: 'GET',
        path: '/health',
        timeout: 5000
      });
      return response.success && response.statusCode === 200;
    } catch {
      return false;
    }
  }

  private async checkLlamaOllamaHealth(): Promise<boolean> {
    try {
      const response = await this.llamaService.generateCompletion({
        prompt: 'health check',
        max_tokens: 1,
        temperature: 0.1,
        stream: false
      });
      return response.success;
    } catch {
      return false;
    }
  }

  private async checkNESGPUBridgeHealth(): Promise<boolean> {
    try {
      const testCanvas: CanvasState = {
        width: 32,
        height: 32,
        data: new Uint8ClampedArray(32 * 32 * 4),
        format: 'RGBA'
      };
      const result = await this.nesGPUBridge.canvasStateToTensor(testCanvas);
      return result.data.length > 0;
    } catch {
      return false;
    }
  }

  private async checkPostgresHealth(): Promise<boolean> {
    try {
      // Simple connection test - would use actual DB connection
      return new Promise(resolve => {
        setTimeout(() => resolve(true), 100);
      });
    } catch {
      return false;
    }
  }

  private async checkRedisHealth(): Promise<boolean> {
    try {
      // Simple connection test - would use actual Redis connection
      return new Promise(resolve => {
        setTimeout(() => resolve(true), 100);
      });
    } catch {
      return false;
    }
  }

  // Main orchestration methods
  public async processLegalDocument(
    document: string,
    options: any = {}
  ): Promise<OrchestrationResult> {
    const task: ServiceTask = {
      id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'document',
      priority: options.priority || TaskPriority.NORMAL,
      data: { document, options },
      options,
      retries: 0,
      maxRetries: this.config.retryConfiguration.maxRetries,
      timeout: options.timeout || 30000,
      startTime: new Date()
    };

    return await this.executeTask(task);
  }

  public async performNeuralInference(
    input: Float32Array,
    options: any = {}
  ): Promise<OrchestrationResult> {
    const task: ServiceTask = {
      id: `inference_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'inference',
      priority: options.priority || TaskPriority.HIGH,
      data: { input, options },
      options,
      retries: 0,
      maxRetries: this.config.retryConfiguration.maxRetries,
      timeout: options.timeout || 15000,
      startTime: new Date()
    };

    return await this.executeTask(task);
  }

  public async processCanvasState(
    canvasState: CanvasState,
    options: any = {}
  ): Promise<OrchestrationResult> {
    const task: ServiceTask = {
      id: `canvas_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'canvas',
      priority: options.priority || TaskPriority.NORMAL,
      data: { canvasState, options },
      options,
      retries: 0,
      maxRetries: this.config.retryConfiguration.maxRetries,
      timeout: options.timeout || 20000,
      startTime: new Date()
    };

    return await this.executeTask(task);
  }

  public async executeGPUComputation(
    operation: string,
    data: any,
    options: any = {}
  ): Promise<OrchestrationResult> {
    const task: ServiceTask = {
      id: `gpu_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'gpu',
      priority: options.priority || TaskPriority.HIGH,
      data: { operation, data, options },
      options,
      retries: 0,
      maxRetries: this.config.retryConfiguration.maxRetries,
      timeout: options.timeout || 25000,
      startTime: new Date()
    };

    return await this.executeTask(task);
  }

  private async executeTask(task: ServiceTask): Promise<OrchestrationResult> {
    const startTime = Date.now();
    const servicesUsed: string[] = [];
    const fallbacksTriggered: string[] = [];

    try {
      this.activeTasks.set(task.id, task);

      let result: any;
      let success = false;

      // Execute based on task type and available services
      switch (task.type) {
        case 'document':
          ({ result, success, servicesUsed: servicesUsed, fallbacksTriggered: fallbacksTriggered } = 
            await this.executeDocumentTask(task, servicesUsed, fallbacksTriggered));
          break;

        case 'inference':
          ({ result, success, servicesUsed: servicesUsed, fallbacksTriggered: fallbacksTriggered } = 
            await this.executeInferenceTask(task, servicesUsed, fallbacksTriggered));
          break;

        case 'canvas':
          ({ result, success, servicesUsed: servicesUsed, fallbacksTriggered: fallbacksTriggered } = 
            await this.executeCanvasTask(task, servicesUsed, fallbacksTriggered));
          break;

        case 'gpu':
          ({ result, success, servicesUsed: servicesUsed, fallbacksTriggered: fallbacksTriggered } = 
            await this.executeGPUTask(task, servicesUsed, fallbacksTriggered));
          break;

        default:
          throw new Error(`Unknown task type: ${task.type}`);
      }

      const processingTime = Date.now() - startTime;
      this.updatePerformanceMetrics(processingTime, 1, success ? 0 : 1);

      task.endTime = new Date();
      this.activeTasks.delete(task.id);

      return {
        taskId: task.id,
        success,
        data: result,
        processingTime,
        servicesUsed,
        fallbacksTriggered,
        performance: {
          latency: processingTime,
          throughput: 1000 / processingTime,
          resourceUsage: this.calculateResourceUsage(servicesUsed)
        }
      };

    } catch (error) {
      const processingTime = Date.now() - startTime;
      task.error = String(error);
      task.endTime = new Date();
      this.activeTasks.delete(task.id);

      this.updatePerformanceMetrics(processingTime, 0, 1);

      return {
        taskId: task.id,
        success: false,
        error: String(error),
        processingTime,
        servicesUsed,
        fallbacksTriggered,
        performance: {
          latency: processingTime,
          throughput: 0,
          resourceUsage: 0
        }
      };
    }
  }

  private async executeDocumentTask(
    task: ServiceTask,
    servicesUsed: string[],
    fallbacksTriggered: string[]
  ): Promise<{ result: any; success: boolean; servicesUsed: string[]; fallbacksTriggered: string[] }> {
    const { document, options } = task.data;

    // Primary: WASM GPU Orchestrator
    if (this.isServiceHealthy('wasmGPU')) {
      try {
        servicesUsed.push('wasmGPU');
        const result = await this.wasmGPUOrchestrator.processLegalDocument(document, options);
        if (result.success) {
          return { result, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        fallbacksTriggered.push('wasmGPU -> quicGateway');
        if (dev) console.warn('[Orchestrator] WASM GPU failed, falling back to QUIC:', error);
      }
    }

    // Fallback: QUIC Gateway
    if (this.isServiceHealthy('quicGateway')) {
      try {
        servicesUsed.push('quicGateway');
        const result = await this.quicClient.analyzeLegalDocument(document, options.analysisType || 'comprehensive');
        if (result.success) {
          return { result: result.data, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        fallbacksTriggered.push('quicGateway -> llamaOllama');
        if (dev) console.warn('[Orchestrator] QUIC Gateway failed, falling back to Llama:', error);
      }
    }

    // Final Fallback: Direct Llama/Ollama
    if (this.isServiceHealthy('llamaOllama')) {
      try {
        servicesUsed.push('llamaOllama');
        const result = await this.llamaService.generateCompletion({
          prompt: `Analyze this legal document:\n\n${document}`,
          max_tokens: options.maxTokens || 2048,
          temperature: options.temperature || 0.7,
          stream: false
        });
        
        if (result.success) {
          return { result: result.data, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        if (dev) console.warn('[Orchestrator] All services failed for document task:', error);
      }
    }

    throw new Error('All document processing services failed');
  }

  private async executeInferenceTask(
    task: ServiceTask,
    servicesUsed: string[],
    fallbacksTriggered: string[]
  ): Promise<{ result: any; success: boolean; servicesUsed: string[]; fallbacksTriggered: string[] }> {
    const { input, options } = task.data;

    // Primary: WASM GPU Orchestrator
    if (this.isServiceHealthy('wasmGPU')) {
      try {
        servicesUsed.push('wasmGPU');
        const result = await this.wasmGPUOrchestrator.performNeuralInference(input, options);
        if (result.success) {
          return { result, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        fallbacksTriggered.push('wasmGPU -> llamaOllama');
        if (dev) console.warn('[Orchestrator] WASM GPU inference failed, falling back to Llama:', error);
      }
    }

    // Fallback: Llama/Ollama Service
    if (this.isServiceHealthy('llamaOllama')) {
      try {
        servicesUsed.push('llamaOllama');
        // Convert Float32Array to prompt for LLM inference
        const inputArray = Array.from(input).slice(0, 100); // Limit for prompt
        const result = await this.llamaService.generateCompletion({
          prompt: `Perform neural inference on this data: [${inputArray.join(', ')}]`,
          max_tokens: options.maxTokens || 512,
          temperature: options.temperature || 0.1,
          stream: false
        });
        
        if (result.success) {
          return { result: result.data, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        if (dev) console.warn('[Orchestrator] All services failed for inference task:', error);
      }
    }

    throw new Error('All inference services failed');
  }

  private async executeCanvasTask(
    task: ServiceTask,
    servicesUsed: string[],
    fallbacksTriggered: string[]
  ): Promise<{ result: any; success: boolean; servicesUsed: string[]; fallbacksTriggered: string[] }> {
    const { canvasState, options } = task.data;

    // Primary: NES GPU Bridge
    if (this.isServiceHealthy('nesGPUBridge')) {
      try {
        servicesUsed.push('nesGPUBridge');
        const result = await this.nesGPUBridge.canvasStateToTensor(canvasState);
        return { result, success: true, servicesUsed, fallbacksTriggered };
      } catch (error) {
        fallbacksTriggered.push('nesGPUBridge -> wasmGPU');
        if (dev) console.warn('[Orchestrator] NES GPU Bridge failed, falling back to WASM:', error);
      }
    }

    // Fallback: WASM GPU Orchestrator
    if (this.isServiceHealthy('wasmGPU')) {
      try {
        servicesUsed.push('wasmGPU');
        const result = await this.wasmGPUOrchestrator.processCanvasState(canvasState);
        if (result.success) {
          return { result, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        if (dev) console.warn('[Orchestrator] All services failed for canvas task:', error);
      }
    }

    throw new Error('All canvas processing services failed');
  }

  private async executeGPUTask(
    task: ServiceTask,
    servicesUsed: string[],
    fallbacksTriggered: string[]
  ): Promise<{ result: any; success: boolean; servicesUsed: string[]; fallbacksTriggered: string[] }> {
    const { operation, data, options } = task.data;

    // Primary: WASM GPU Orchestrator
    if (this.isServiceHealthy('wasmGPU')) {
      try {
        servicesUsed.push('wasmGPU');
        const result = await this.wasmGPUOrchestrator.executeGPUComputation(operation, data);
        if (result.success) {
          return { result, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        fallbacksTriggered.push('wasmGPU -> quicGateway');
        if (dev) console.warn('[Orchestrator] WASM GPU computation failed, falling back to QUIC:', error);
      }
    }

    // Fallback: QUIC Gateway for GPU-accelerated operations
    if (this.isServiceHealthy('quicGateway')) {
      try {
        servicesUsed.push('quicGateway');
        const result = await this.quicClient.request({
          method: 'POST',
          path: '/gpu/compute',
          body: { operation, data, options },
          timeout: task.timeout
        });
        
        if (result.success) {
          return { result: result.data, success: true, servicesUsed, fallbacksTriggered };
        }
      } catch (error) {
        if (dev) console.warn('[Orchestrator] All services failed for GPU task:', error);
      }
    }

    throw new Error('All GPU computation services failed');
  }

  private isServiceHealthy(service: keyof SystemHealth['services']): boolean {
    return this.systemHealth.services[service] === 'online';
  }

  private calculateResourceUsage(servicesUsed: string[]): number {
    // Simple resource usage calculation based on services used
    const weights: Record<string, number> = {
      wasmGPU: 0.8,
      quicGateway: 0.3,
      llamaOllama: 0.6,
      nesGPUBridge: 0.5
    };

    return servicesUsed.reduce((total, service) => {
      return total + (weights[service] || 0.1);
    }, 0);
  }

  private updatePerformanceMetrics(latency: number, success: number, error: number): void {
    const metric = {
      timestamp: new Date(),
      latency,
      throughput: success,
      resourceUsage: error
    };

    this.performanceMetrics.push(metric);

    // Keep only recent metrics
    const cutoff = Date.now() - this.config.monitoring.metricsRetentionPeriod;
    this.performanceMetrics = this.performanceMetrics.filter(
      m => m.timestamp.getTime() > cutoff
    );

    // Update system health performance metrics
    if (this.performanceMetrics.length > 0) {
      this.systemHealth.performance.averageLatency = 
        this.performanceMetrics.reduce((sum, m) => sum + m.latency, 0) / this.performanceMetrics.length;
      
      this.systemHealth.performance.throughput = 
        this.performanceMetrics.reduce((sum, m) => sum + m.throughput, 0) / this.performanceMetrics.length;
    }
  }

  // Public API methods
  public getSystemHealth(): SystemHealth {
    return { ...this.systemHealth };
  }

  public getActiveTasks(): ServiceTask[] {
    return Array.from(this.activeTasks.values());
  }

  public getTaskQueue(): ServiceTask[] {
    return [...this.taskQueue];
  }

  public getPerformanceMetrics(): typeof this.performanceMetrics {
    return [...this.performanceMetrics];
  }

  public async shutdown(): Promise<void> {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    // Wait for active tasks to complete (with timeout)
    const maxWaitTime = 30000; // 30 seconds
    const startTime = Date.now();
    
    while (this.activeTasks.size > 0 && (Date.now() - startTime) < maxWaitTime) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Force cleanup remaining tasks
    this.activeTasks.clear();
    this.taskQueue.length = 0;

    if (dev) {
      console.log('[Orchestrator] Shutdown complete');
    }
  }
}

// Singleton instance for global use
let orchestratorInstance: UnifiedServiceOrchestrator | null = null;

export function getOrchestrator(config?: Partial<ServiceConfig>): UnifiedServiceOrchestrator {
  if (!orchestratorInstance) {
    orchestratorInstance = new UnifiedServiceOrchestrator(config);
  }
  return orchestratorInstance;
}

export function resetOrchestrator(): void {
  if (orchestratorInstance) {
    orchestratorInstance.shutdown();
    orchestratorInstance = null;
  }
}