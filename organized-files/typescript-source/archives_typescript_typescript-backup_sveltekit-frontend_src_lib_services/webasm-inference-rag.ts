/**
 * WebAssembly Inference RAG Integration
 * Integrates WebAssembly inference capabilities with the Enhanced RAG pipeline
 * 
 * Architecture Integration:
 * - XState orchestration for workflow management
 * - RabbitMQ messaging for async processing
 * - PostgreSQL-Qdrant sync for vector storage
 * - Auto-tagging worker coordination
 * - Vertex buffer image analysis integration
 */

import { createMachine, assign, fromPromise } from 'xstate';
import { rabbitMQIntegration, type LegalAIMessage } from '../messaging/rabbitmq-xstate-integration';
import { postgresqlQdrantSync } from './postgresql-qdrant-sync';
import { vertexBufferImageAnalyzer } from './vertex-buffer-image-analyzer';

// WebAssembly inference types
export interface WASMInferenceConfig {
  modelPath: string;
  threads: number;
  contextLength: number;
  enableGPU: boolean;
  batchSize: number;
  quantization: 'q4_0' | 'q4_1' | 'q8_0' | 'f16' | 'f32';
}

export interface WASMInferenceRequest {
  id: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  stopSequences?: string[];
  systemMessage?: string;
  contextDocuments?: string[];
  enableRAG: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface WASMInferenceResult {
  id: string;
  text: string;
  tokens: number;
  processingTime: number;
  memoryUsage: number;
  cacheHit: boolean;
  ragContext?: {
    documentsUsed: number;
    relevanceScores: number[];
    sources: string[];
  };
  metadata: {
    model: string;
    quantization: string;
    threads: number;
    wasmVersion: string;
  };
}

export interface WASMRAGContext {
  wasmModule: WebAssembly.Module | null;
  wasmInstance: WebAssembly.Instance | null;
  isInitialized: boolean;
  config: WASMInferenceConfig;
  activeRequests: Map<string, WASMInferenceRequest>;
  results: Map<string, WASMInferenceResult>;
  performanceMetrics: {
    totalInferences: number;
    averageLatency: number;
    cacheHitRate: number;
    memoryPeak: number;
  };
  error: string | null;
}

// XState machine for WebAssembly RAG inference
export const wasmInferenceMachine = createMachine({
  id: 'wasmInferenceRAG',
  initial: 'initializing',
  context: {
    wasmModule: null,
    wasmInstance: null,
    isInitialized: false,
    config: {
      modelPath: '/models/gemma3-legal-q4.wasm',
      threads: 8,
      contextLength: 4096,
      enableGPU: true,
      batchSize: 4,
      quantization: 'q4_0'
    },
    activeRequests: new Map(),
    results: new Map(),
    performanceMetrics: {
      totalInferences: 0,
      averageLatency: 0,
      cacheHitRate: 0,
      memoryPeak: 0
    },
    error: null
  } as WASMRAGContext,

  states: {
    initializing: {
      invoke: {
        id: 'initializeWASM',
        src: fromPromise(async ({ input }) => {
          return await WASMInferenceRAGService.initialize(input.config);
        }),
        input: ({ context }) => ({ config: context.config }),
        onDone: {
          target: 'ready',
          actions: assign({
            wasmModule: ({ event }) => event.output.module,
            wasmInstance: ({ event }) => event.output.instance,
            isInitialized: true,
            error: null
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => (event.error as Error)?.message || 'Initialization failed',
            isInitialized: false
          })
        }
      }
    },

    ready: {
      initial: 'idle',
      states: {
        idle: {
          on: {
            INFERENCE_REQUEST: {
              target: 'processing',
              actions: assign({
                activeRequests: ({ context, event }) => {
                  const newMap = new Map(context.activeRequests);
                  newMap.set(event.request.id, event.request);
                  return newMap;
                }
              })
            }
          }
        },

        processing: {
          invoke: {
            id: 'processInference',
            src: fromPromise(async ({ input }) => {
              const { request, context } = input;
              return await WASMInferenceRAGService.processInferenceWithRAG(request, context);
            }),
            input: ({ context, event }) => ({
              request: Array.from(context.activeRequests.values())[0],
              context
            }),
            onDone: {
              target: 'idle',
              actions: [
                assign({
                  results: ({ context, event }) => {
                    const newMap = new Map(context.results);
                    newMap.set(event.output.id, event.output);
                    return newMap;
                  },
                  activeRequests: ({ context }) => {
                    const newMap = new Map(context.activeRequests);
                    const firstKey = Array.from(newMap.keys())[0];
                    if (firstKey) newMap.delete(firstKey);
                    return newMap;
                  },
                  performanceMetrics: ({ context, event }) => ({
                    ...context.performanceMetrics,
                    totalInferences: context.performanceMetrics.totalInferences + 1,
                    averageLatency: (context.performanceMetrics.averageLatency + event.output.processingTime) / 2
                  })
                }),
                'publishResult'
              ]
            },
            onError: {
              target: 'idle',
              actions: [
                assign({
                  error: ({ event }) => (event.error as Error)?.message || 'Processing failed',
                  activeRequests: ({ context }) => {
                    const newMap = new Map(context.activeRequests);
                    const firstKey = Array.from(newMap.keys())[0];
                    if (firstKey) newMap.delete(firstKey);
                    return newMap;
                  }
                }),
                'publishError'
              ]
            }
          }
        }
      }
    },

    error: {
      on: {
        RETRY_INIT: {
          target: 'initializing',
          actions: assign({
            error: null
          })
        }
      }
    }
  }
}, {
  actions: {
    publishResult: ({ context, event }) => {
      // Publish result via RabbitMQ
      rabbitMQIntegration.publishMessage({
        type: 'ai_analysis',
        payload: {
          result: event.output,
          success: true
        },
        priority: 7
      }).catch(console.error);
    },
    
    publishError: ({ context, event }) => {
      // Publish error via RabbitMQ
      rabbitMQIntegration.publishMessage({
        type: 'error_recovery',
        payload: {
          error: event.error.message,
          service: 'wasm_inference_rag'
        },
        priority: 9
      }).catch(console.error);
    }
  }
});

// Main WebAssembly Inference RAG Service
export class WASMInferenceRAGService {
  private static wasmModule: WebAssembly.Module | null = null;
  private static wasmInstance: WebAssembly.Instance | null = null;
  private static isInitialized = false;
  private static config: WASMInferenceConfig;

  /**
   * Initialize WebAssembly inference with enhanced RAG integration
   */
  static async initialize(config: WASMInferenceConfig): Promise<{ module: WebAssembly.Module; instance: WebAssembly.Instance }> {
    console.log('🚀 Initializing WebAssembly Inference RAG Service');
    
    try {
      // Initialize vertex buffer image analyzer for multimodal RAG
      await vertexBufferImageAnalyzer.initialize();
      
      // Initialize PostgreSQL-Qdrant sync for vector retrieval
      await postgresqlQdrantSync.ensureCollection();
      
      // Load WebAssembly module
      const wasmBuffer = await this.loadWASMModule(config.modelPath);
      this.wasmModule = await WebAssembly.compile(wasmBuffer);
      
      // Create instance with memory and imports
      const memory = new WebAssembly.Memory({ 
        initial: 256, // 16MB
        maximum: 1024, // 64MB
        shared: true 
      });
      
      const imports = this.createWASMImports(memory, config);
      this.wasmInstance = await WebAssembly.instantiate(this.wasmModule, imports);
      
      this.config = config;
      this.isInitialized = true;
      
      console.log('✅ WebAssembly Inference RAG Service initialized');
      console.log(`📊 Config: ${config.quantization}, ${config.threads} threads, GPU: ${config.enableGPU}`);
      
      return {
        module: this.wasmModule,
        instance: this.wasmInstance
      };
      
    } catch (error: any) {
      console.error('❌ WebAssembly initialization failed:', error);
      throw error;
    }
  }

  /**
   * Process inference request with enhanced RAG capabilities
   */
  static async processInferenceWithRAG(
    request: WASMInferenceRequest, 
    context: WASMRAGContext
  ): Promise<WASMInferenceResult> {
    const startTime = performance.now();
    
    try {
      console.log(`🧠 Processing WASM inference: ${request.id}`);
      
      // Step 1: RAG Document Retrieval (if enabled)
      let ragContext: WASMInferenceResult['ragContext'] | undefined;
      let enhancedPrompt = request.prompt;
      
      if (request.enableRAG) {
        console.log('📚 Retrieving RAG context from PostgreSQL-Qdrant');
        
        // Use PostgreSQL-Qdrant sync for vector search
        const similarDocs = await this.retrieveRelevantDocuments(request.prompt, 5);
        
        if (similarDocs.length > 0) {
          const contextText = similarDocs.map(doc => doc.content).join('\n\n');
          enhancedPrompt = `Context:\n${contextText}\n\nQuestion: ${request.prompt}`;
          
          ragContext = {
            documentsUsed: similarDocs.length,
            relevanceScores: similarDocs.map(doc => doc.score),
            sources: similarDocs.map(doc => doc.id)
          };
          
          console.log(`📖 Enhanced prompt with ${similarDocs.length} context documents`);
        }
      }
      
      // Step 2: WebAssembly Inference
      const inferenceResult = await this.runWASMInference(
        enhancedPrompt,
        request.maxTokens,
        request.temperature,
        request.stopSequences
      );
      
      // Step 3: Post-processing and tagging
      await this.triggerAutoTagging(request.id, inferenceResult.text, request.priority);
      
      const processingTime = performance.now() - startTime;
      const memoryUsage = this.getMemoryUsage();
      
      const result: WASMInferenceResult = {
        id: request.id,
        text: inferenceResult.text,
        tokens: inferenceResult.tokens,
        processingTime,
        memoryUsage,
        cacheHit: false, // TODO: Implement caching
        ragContext,
        metadata: {
          model: 'gemma3-legal-wasm',
          quantization: this.config.quantization,
          threads: this.config.threads,
          wasmVersion: '1.0.0'
        }
      };
      
      // Store inference result for future RAG improvements
      if (ragContext && ragContext.documentsUsed > 0) {
        try {
          const { postgresqlQdrantSync } = await import('./postgresql-qdrant-sync.js');
          const queryEmbedding = await this.generateQueryEmbedding(request.prompt);
          
          await postgresqlQdrantSync.storeWASMInferenceResult(
            queryEmbedding,
            ragContext.sources,
            inferenceResult.text,
            {
              inferenceId: request.id,
              model: 'gemma3-legal-wasm',
              processingTime,
              ragContext
            }
          );
          
          console.log(`📝 Stored WASM inference result for future RAG improvements`);
        } catch (storageError) {
          console.warn('⚠️ Failed to store WASM inference result:', storageError);
        }
      }

      console.log(`✅ WASM inference completed: ${request.id} (${processingTime.toFixed(2)}ms)`);
      return result;
      
    } catch (error: any) {
      console.error(`❌ WASM inference failed for ${request.id}:`, error);
      throw error;
    }
  }

  /**
   * Retrieve relevant documents using WebAssembly-optimized PostgreSQL-Qdrant search
   */
  private static async retrieveRelevantDocuments(query: string, limit: number = 5): Promise<Array<{
    id: string;
    content: string;
    score: number;
    metadata: any;
  }>> {
    try {
      console.log(`🔍 Retrieving relevant documents for WASM inference: "${query.slice(0, 50)}..."`);
      
      // First, we need to get the query embedding
      // For now, we'll use a simple approach - in production this would use your embedding service
      const queryEmbedding = await this.generateQueryEmbedding(query);
      
      // Use WebAssembly-optimized PostgreSQL-Qdrant search
      const { postgresqlQdrantSync } = await import('./postgresql-qdrant-sync.js');
      
      const results = await postgresqlQdrantSync.searchForWASMInference(
        queryEmbedding,
        limit,
        0.7, // Score threshold for legal documents
        {
          type: ['evidence', 'document'], // Filter to relevant document types
          // Add more filters as needed
        }
      );
      
      console.log(`📚 Retrieved ${results.length} relevant documents for WASM inference`);
      return results;
      
    } catch (error: any) {
      console.warn('⚠️ WASM RAG retrieval error:', error);
      
      // Fallback to the existing enhanced RAG service if available
      try {
        console.log('🔄 Falling back to enhanced RAG service...');
        const response = await fetch('http://localhost:8094/api/rag/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query,
            limit,
            enableHybrid: true,
            includeMetadata: true
          })
        });
        
        if (response.ok) {
          const results = await response.json();
          console.log(`📚 Fallback retrieval successful: ${results.documents?.length || 0} documents`);
          return results.documents || [];
        }
      } catch (fallbackError) {
        console.warn('⚠️ Fallback RAG search also failed:', fallbackError);
      }
      
      return [];
    }
  }

  /**
   * Generate query embedding for vector search
   */
  private static async generateQueryEmbedding(query: string): Promise<number[]> {
    try {
      // Try to use the sentence transformer service first
      const { legalNLP } = await import('./sentence-transformer.js');
      const embedding = await legalNLP.embedText(query);
      return Array.isArray(embedding) ? embedding : Array.from(embedding as any);
      
    } catch (error: any) {
      console.warn('⚠️ Sentence transformer not available, using mock embedding:', error);
      
      // Generate a mock embedding for development (384 dimensions for nomic-embed-text)
      const dimensions = 384;
      const mockEmbedding = Array(dimensions).fill(0).map(() => Math.random() * 2 - 1);
      
      // Add some deterministic elements based on query for consistent results
      const queryHash = query.split('').reduce((hash, char) => 
        ((hash << 5) - hash + char.charCodeAt(0)) & 0xffffffff, 0
      );
      
      for (let i = 0; i < Math.min(10, dimensions); i++) {
        mockEmbedding[i] = (queryHash % 1000 + i) / 1000 - 0.5;
      }
      
      return mockEmbedding;
    }
  }

  /**
   * Execute WebAssembly inference
   */
  private static async runWASMInference(
    prompt: string,
    maxTokens: number,
    temperature: number,
    stopSequences?: string[]
  ): Promise<{ text: string; tokens: number }> {
    if (!this.wasmInstance) {
      throw new Error('WebAssembly instance not initialized');
    }
    
    try {
      // Create prompt buffer in WASM memory
      const promptBuffer = new TextEncoder().encode(prompt);
      const inputPtr = this.allocateWASMMemory(promptBuffer.length);
      this.writeToWASMMemory(inputPtr, promptBuffer);
      
      // Call WASM inference function
      // Note: This assumes a specific WASM interface - adjust based on your actual WASM module
      const wasmExports = this.wasmInstance.exports as any;
      const resultPtr = wasmExports.infer(inputPtr, promptBuffer.length, maxTokens, temperature);
      
      // Read result from WASM memory
      const resultText = this.readStringFromWASMMemory(resultPtr);
      const tokens = this.countTokens(resultText);
      
      // Free WASM memory
      wasmExports.free(inputPtr);
      wasmExports.free(resultPtr);
      
      return {
        text: resultText,
        tokens
      };
      
    } catch (error: any) {
      console.error('WASM inference execution error:', error);
      throw error;
    }
  }

  /**
   * Trigger auto-tagging via RabbitMQ for generated content
   */
  private static async triggerAutoTagging(inferenceId: string, generatedText: string, priority: string) {
    try {
      await rabbitMQIntegration.publishMessage({
        type: 'ai_analysis',
        payload: {
          inferenceId,
          generatedText,
          requiresTagging: true,
          source: 'wasm_inference'
        },
        priority: priority === 'critical' ? 10 : priority === 'high' ? 8 : 5,
        correlationId: inferenceId
      });
      console.log(`🏷️ Triggered auto-tagging for inference: ${inferenceId}`);
      
    } catch (error: any) {
      console.warn('⚠️ Auto-tagging trigger failed:', error);
    }
  }

  // Helper methods for WebAssembly memory management
  private static async loadWASMModule(modelPath: string): Promise<ArrayBuffer> {
    // In production, load from actual file system or network
    // For now, return a mock buffer
    console.log(`📥 Loading WASM module from: ${modelPath}`);
    return new ArrayBuffer(1024 * 1024); // 1MB mock buffer
  }

  private static createWASMImports(memory: WebAssembly.Memory, config: WASMInferenceConfig) {
    return {
      env: {
        memory,
        abort: () => { throw new Error('WASM execution aborted'); },
        // Add other WASM import functions as needed
      }
    };
  }

  private static allocateWASMMemory(size: number): number {
    // Mock memory allocation
    return Math.floor(Math.random() * 1000000);
  }

  private static writeToWASMMemory(ptr: number, data: Uint8Array): void {
    // Mock memory write
    console.log(`💾 Writing ${data.length} bytes to WASM memory at ${ptr}`);
  }

  private static readStringFromWASMMemory(ptr: number): string {
    // Mock memory read
    return "Generated response from WebAssembly inference";
  }

  private static countTokens(text: string): number {
    // Simple token counting approximation
    return text.split(/\s+/).length;
  }

  private static getMemoryUsage(): number {
    // Return mock memory usage
    return Math.floor(Math.random() * 100 * 1024 * 1024); // Random MB
  }

  /**
   * Cleanup WebAssembly resources
   */
  static async cleanup(): Promise<void> {
    if (this.wasmInstance) {
      // Cleanup WASM instance
      this.wasmInstance = null;
    }
    
    if (this.wasmModule) {
      this.wasmModule = null;
    }
    
    this.isInitialized = false;
    console.log('🧹 WebAssembly Inference RAG Service cleaned up');
  }

  /**
   * Health check
   */
  static getHealthStatus(): {
    status: 'healthy' | 'degraded' | 'unhealthy';
    wasm: boolean;
    rag: boolean;
    messaging: boolean;
  } {
    return {
      status: this.isInitialized ? 'healthy' : 'unhealthy',
      wasm: this.isInitialized && !!this.wasmInstance,
      rag: true, // Assume RAG is healthy
      messaging: true // Assume messaging is healthy
    };
  }
}

// Export singleton for integration
export const wasmInferenceRAGService = WASMInferenceRAGService;
export default WASMInferenceRAGService;