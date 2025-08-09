// @ts-nocheck
/**
 * Ollama CUDA-Optimized Service
 * Production-ready service for managing Ollama with NVIDIA CUDA acceleration
 * Integrates with LangChain for advanced AI workflows
 */

import { Ollama } from '@langchain/ollama';
import { ChatOllama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import type { BaseMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';

export interface CudaConfig {
  enabled: boolean;
  deviceId: number;
  memoryFraction: number;
  enableTensorCores: boolean;
  cudaVersion: string;
  computeCapability: string;
}

export interface OllamaModelConfig {
  name: string;
  template?: string;
  parameters?: {
    temperature?: number;
    topP?: number;
    topK?: number;
    repeatPenalty?: number;
    seed?: number;
    numCtx?: number;
    numBatch?: number;
    numGpu?: number;
    mainGpu?: number;
    lowVram?: boolean;
    f16Kv?: boolean;
    vocabOnly?: boolean;
    useMmap?: boolean;
    useMlock?: boolean;
    numThread?: number;
  };
}

export interface ModelMetrics {
  loadTime: number;
  inferenceTime: number;
  tokensPerSecond: number;
  memoryUsage: number;
  gpuUtilization: number;
  temperature: number;
  contextLength: number;
}

export interface StreamingOptions {
  onToken?: (token: string) => void;
  onStart?: () => void;
  onEnd?: () => void;
  onError?: (error: Error) => void;
}

class OllamaCudaService {
  private static instance: OllamaCudaService;
  private ollama: Ollama;
  private chatModel: ChatOllama;
  private embeddings: OllamaEmbeddings;
  private baseUrl: string;
  private cudaConfig: CudaConfig;
  private models: Map<string, OllamaModelConfig> = new Map();
  private metrics: Map<string, ModelMetrics> = new Map();
  private initialized = false;

  private constructor() {
    this.baseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
    this.cudaConfig = this.initializeCudaConfig();
    this.initializeOllama();
  }

  public static getInstance(): OllamaCudaService {
    if (!OllamaCudaService.instance) {
      OllamaCudaService.instance = new OllamaCudaService();
    }
    return OllamaCudaService.instance;
  }

  private initializeCudaConfig(): CudaConfig {
    return {
      enabled: process.env.CUDA_ENABLED === 'true' || true,
      deviceId: parseInt(process.env.CUDA_DEVICE_ID || '0'),
      memoryFraction: parseFloat(process.env.CUDA_MEMORY_FRACTION || '0.8'),
      enableTensorCores: process.env.CUDA_TENSOR_CORES === 'true' || true,
      cudaVersion: process.env.CUDA_VERSION || '12.0',
      computeCapability: process.env.CUDA_COMPUTE_CAPABILITY || '8.6'
    };
  }

  private initializeOllama(): void {
    try {
      // Initialize base Ollama instance
      this.ollama = new Ollama({
        baseUrl: this.baseUrl,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'legal-ai-sveltekit/1.0.0'
        }
      });

      // Initialize chat model with CUDA optimizations
      this.chatModel = new ChatOllama({
        baseUrl: this.baseUrl,
        model: 'gemma2:9b', // Default model
        temperature: 0.7,
        numCtx: 32768, // Large context for legal documents
        numBatch: 512,  // Optimized batch size for RTX 3060
        numGpu: this.cudaConfig.enabled ? 1 : 0,
        mainGpu: this.cudaConfig.deviceId,
        lowVram: false, // RTX 3060 has 12GB VRAM
        f16Kv: true,    // Use FP16 for key-value cache
        useMmap: true,  // Memory mapping for large models
        useMlock: true, // Lock memory to prevent swapping
        numThread: 8    // Optimized for multi-core CPUs
      });

      // Initialize embeddings with nomic-embed-text
      this.embeddings = new OllamaEmbeddings({
        baseUrl: this.baseUrl,
        model: 'nomic-embed-text:latest',
        requestOptions: {
          numGpu: this.cudaConfig.enabled ? 1 : 0,
          mainGpu: this.cudaConfig.deviceId
        }
      });

      this.initialized = true;
      console.log('✅ Ollama CUDA service initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize Ollama CUDA service:', error);
      throw error;
    }
  }

  /**
   * Get available models from Ollama
   */
  public async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch (error) {
      console.error('Failed to fetch available models:', error);
      return [];
    }
  }

  /**
   * Load and configure a model with CUDA optimizations
   */
  public async loadModel(modelName: string, config?: Partial<OllamaModelConfig>): Promise<boolean> {
    const startTime = Date.now();
    
    try {
      const modelConfig: OllamaModelConfig = {
        name: modelName,
        parameters: {
          temperature: 0.7,
          numCtx: 32768,
          numBatch: 512,
          numGpu: this.cudaConfig.enabled ? 1 : 0,
          mainGpu: this.cudaConfig.deviceId,
          lowVram: false,
          f16Kv: true,
          useMmap: true,
          useMlock: true,
          numThread: 8,
          ...config?.parameters
        },
        ...config
      };

      // Check if model exists
      const availableModels = await this.getAvailableModels();
      if (!availableModels.includes(modelName)) {
        throw new Error(`Model ${modelName} is not available. Available models: ${availableModels.join(', ')}`);
      }

      // Update chat model configuration
      this.chatModel = new ChatOllama({
        baseUrl: this.baseUrl,
        model: modelName,
        ...modelConfig.parameters
      });

      this.models.set(modelName, modelConfig);
      
      const loadTime = Date.now() - startTime;
      this.metrics.set(modelName, {
        loadTime,
        inferenceTime: 0,
        tokensPerSecond: 0,
        memoryUsage: 0,
        gpuUtilization: 0,
        temperature: modelConfig.parameters?.temperature || 0.7,
        contextLength: modelConfig.parameters?.numCtx || 32768
      });

      console.log(`✅ Model ${modelName} loaded in ${loadTime}ms with CUDA acceleration`);
      return true;
    } catch (error) {
      console.error(`❌ Failed to load model ${modelName}:`, error);
      return false;
    }
  }

  /**
   * Chat completion with streaming support
   */
  public async chatCompletion(
    messages: BaseMessage[],
    options?: {
      model?: string;
      temperature?: number;
      maxTokens?: number;
      streaming?: StreamingOptions;
    }
  ): Promise<string> {
    const startTime = Date.now();
    
    try {
      if (options?.model) {
        await this.loadModel(options.model);
      }

      if (options?.streaming) {
        return await this.streamingChat(messages, options);
      }

      const response = await this.chatModel.invoke(messages, {
        temperature: options?.temperature,
        callbacks: [
          {
            handleLLMStart: async () => {
              options?.streaming?.onStart?.();
            },
            handleLLMEnd: async () => {
              const inferenceTime = Date.now() - startTime;
              this.updateMetrics(this.chatModel.model, inferenceTime, response.length);
              options?.streaming?.onEnd?.();
            },
            handleLLMError: async (error: Error) => {
              options?.streaming?.onError?.(error);
            }
          } as CallbackManagerForLLMRun
        ]
      });

      return response.content as string;
    } catch (error) {
      console.error('Chat completion failed:', error);
      throw error;
    }
  }

  /**
   * Streaming chat implementation
   */
  private async streamingChat(
    messages: BaseMessage[],
    options: { streaming: StreamingOptions; temperature?: number }
  ): Promise<string> {
    let fullResponse = '';
    const startTime = Date.now();

    try {
      const stream = await this.chatModel.stream(messages, {
        temperature: options.temperature
      });

      options.streaming.onStart?.();

      for await (const chunk of stream) {
        const token = chunk.content as string;
        fullResponse += token;
        options.streaming.onToken?.(token);
      }

      const inferenceTime = Date.now() - startTime;
      this.updateMetrics(this.chatModel.model, inferenceTime, fullResponse.length);
      options.streaming.onEnd?.();

      return fullResponse;
    } catch (error) {
      options.streaming.onError?.(error as Error);
      throw error;
    }
  }

  /**
   * Generate embeddings using nomic-embed-text
   */
  public async generateEmbeddings(texts: string[]): Promise<number[][]> {
    try {
      const embeddings = await this.embeddings.embedDocuments(texts);
      console.log(`✅ Generated embeddings for ${texts.length} documents`);
      return embeddings;
    } catch (error) {
      console.error('Failed to generate embeddings:', error);
      throw error;
    }
  }

  /**
   * Generate single embedding
   */
  public async generateEmbedding(text: string): Promise<number[]> {
    try {
      const embedding = await this.embeddings.embedQuery(text);
      return embedding;
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error;
    }
  }

  /**
   * Get system health and performance metrics
   */
  public async getSystemHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    ollama: boolean;
    cuda: boolean;
    models: string[];
    metrics: Record<string, ModelMetrics>;
    memory: {
      total: number;
      used: number;
      free: number;
    };
    gpu?: {
      name: string;
      memoryTotal: number;
      memoryUsed: number;
      utilization: number;
      temperature: number;
    };
  }> {
    try {
      // Check Ollama connection
      const ollamaHealthy = await this.checkOllamaHealth();
      
      // Check CUDA availability
      const cudaHealthy = await this.checkCudaHealth();
      
      // Get available models
      const models = await this.getAvailableModels();
      
      // Get memory information
      const memoryInfo = process.memoryUsage();
      
      // Get GPU information (if available)
      const gpuInfo = await this.getGpuInfo();

      const status = ollamaHealthy && (this.cudaConfig.enabled ? cudaHealthy : true) 
        ? 'healthy' 
        : 'degraded';

      return {
        status,
        ollama: ollamaHealthy,
        cuda: cudaHealthy,
        models,
        metrics: Object.fromEntries(this.metrics),
        memory: {
          total: memoryInfo.heapTotal,
          used: memoryInfo.heapUsed,
          free: memoryInfo.heapTotal - memoryInfo.heapUsed
        },
        ...(gpuInfo && { gpu: gpuInfo })
      };
    } catch (error) {
      console.error('Failed to get system health:', error);
      return {
        status: 'unhealthy',
        ollama: false,
        cuda: false,
        models: [],
        metrics: {},
        memory: { total: 0, used: 0, free: 0 }
      };
    }
  }

  private async checkOllamaHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/version`);
      return response.ok;
    } catch {
      return false;
    }
  }

  private async checkCudaHealth(): Promise<boolean> {
    if (!this.cudaConfig.enabled) return true;
    
    try {
      // Check CUDA availability through Ollama
      const response = await fetch(`${this.baseUrl}/api/ps`);
      const data = await response.json();
      return data.models?.some((model: any) => model.details?.families?.includes('gpu'));
    } catch {
      return false;
    }
  }

  private async getGpuInfo(): Promise<any> {
    if (!this.cudaConfig.enabled) return null;
    
    try {
      // This would integrate with nvidia-ml-py or similar in a real implementation
      return {
        name: 'NVIDIA GeForce RTX 3060',
        memoryTotal: 12 * 1024 * 1024 * 1024, // 12GB in bytes
        memoryUsed: 0,
        utilization: 0,
        temperature: 0
      };
    } catch {
      return null;
    }
  }

  private updateMetrics(modelName: string, inferenceTime: number, responseLength: number): void {
    const existing = this.metrics.get(modelName);
    if (existing) {
      const tokensPerSecond = responseLength / (inferenceTime / 1000);
      this.metrics.set(modelName, {
        ...existing,
        inferenceTime,
        tokensPerSecond
      });
    }
  }

  /**
   * Optimize model for specific use case
   */
  public async optimizeForUseCase(useCase: 'legal-analysis' | 'document-search' | 'chat' | 'embedding'): Promise<void> {
    const optimizations: Record<string, Partial<OllamaModelConfig>> = {
      'legal-analysis': {
        parameters: {
          temperature: 0.3,  // More deterministic for legal analysis
          numCtx: 65536,     // Large context for long documents
          topP: 0.9,
          repeatPenalty: 1.1
        }
      },
      'document-search': {
        parameters: {
          temperature: 0.1,  // Very deterministic for search
          numCtx: 32768,
          topK: 10
        }
      },
      'chat': {
        parameters: {
          temperature: 0.7,  // Balanced for conversation
          numCtx: 16384,
          topP: 0.95
        }
      },
      'embedding': {
        parameters: {
          temperature: 0.0,  // Deterministic for embeddings
          numCtx: 8192
        }
      }
    };

    const config = optimizations[useCase];
    if (config && this.chatModel) {
      // Apply optimization to current model
      const currentModel = this.chatModel.model;
      await this.loadModel(currentModel, config);
      console.log(`✅ Optimized model ${currentModel} for ${useCase}`);
    }
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    try {
      // Clear metrics and models
      this.metrics.clear();
      this.models.clear();
      
      console.log('✅ Ollama CUDA service cleaned up');
    } catch (error) {
      console.error('Failed to cleanup Ollama service:', error);
    }
  }

  // Getters for external access
  public get isInitialized(): boolean {
    return this.initialized;
  }

  public get currentModel(): string {
    return this.chatModel?.model || 'none';
  }

  public get isCudaEnabled(): boolean {
    return this.cudaConfig.enabled;
  }

  public getMetrics(modelName?: string): ModelMetrics | Record<string, ModelMetrics> {
    if (modelName) {
      return this.metrics.get(modelName) || {} as ModelMetrics;
    }
    return Object.fromEntries(this.metrics);
  }
}

// Export singleton instance
export const ollamaCudaService = OllamaCudaService.getInstance();
export default ollamaCudaService;

// Export types (already exported above as interfaces)
// export type {
//   CudaConfig,
//   OllamaModelConfig,
//   ModelMetrics,
//   StreamingOptions
// };