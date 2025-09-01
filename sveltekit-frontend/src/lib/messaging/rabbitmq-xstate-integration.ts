/**
 * RabbitMQ + XState Integration for Self-Prompting Legal AI
 * Free, high-performance message queuing with state machine coordination
 */

import { createMachine, assign, fromPromise } from 'xstate';
import { browser } from '$app/environment';

// RabbitMQ Web STOMP configuration (free tier)
export interface RabbitMQConfig {
  host: string;
  port: number;
  vhost: string;
  username: string;
  password: string;
  ssl: boolean;
  heartbeat: number;
}

// Legal AI message types (enhanced for WebAssembly inference)
export type LegalAIMessageType =
  | 'document_ingestion'
  | 'vector_search'
  | 'ai_analysis'
  | 'self_prompt'
  | 'user_history_update'
  | 'cache_invalidation'
  | 'gpu_task'
  | 'wasm_compilation'
  | 'wasm_inference'           // NEW: WebAssembly inference requests
  | 'wasm_inference_result'    // NEW: WebAssembly inference results
  | 'wasm_model_load'          // NEW: WebAssembly model loading
  | 'wasm_model_unload'        // NEW: WebAssembly model cleanup
  | 'wasm_batch_inference'     // NEW: Batch WebAssembly inference
  | 'wasm_stream_inference'    // NEW: Streaming WebAssembly inference
  | 'wasm_health_check'        // NEW: WebAssembly service health
  | 'error_recovery';

export interface LegalAIMessage {
  id: string;
  type: LegalAIMessageType;
  payload: any;
  priority: number; // 1-10, 10 being highest
  timestamp: number;
  userId?: string;
  sessionId?: string;
  correlationId?: string;
  replyTo?: string;
}

// Self-prompting context for legal AI
export interface SelfPromptingContext {
  userHistory: any[];
  activeSession: string | null;
  pendingTasks: LegalAIMessage[];
  completedTasks: LegalAIMessage[];
  errorTasks: LegalAIMessage[];
  performanceMetrics: {
    averageResponseTime: number;
    successRate: number;
    cacheHitRate: number;
    gpuUtilization: number;
  };
  rabbitMQConnection: any;
  isConnected: boolean;
  reconnectAttempts: number;
  lastHeartbeat: number;
}

// XState machine for self-prompting legal AI
export const selfPromptingMachine = createMachine({
  id: 'legalAISelfPrompting',
  initial: 'initializing',
  context: {
    userHistory: [],
    activeSession: null,
    pendingTasks: [],
    completedTasks: [],
    errorTasks: [],
    performanceMetrics: {
      averageResponseTime: 0,
      successRate: 0.95,
      cacheHitRate: 0.8,
      gpuUtilization: 0
    },
    rabbitMQConnection: null,
    isConnected: false,
    reconnectAttempts: 0,
    lastHeartbeat: 0
  } as SelfPromptingContext,

  states: {
    initializing: {
      invoke: {
        id: 'initializeRabbitMQ',
        src: fromPromise(async () => {
          return await RabbitMQXStateIntegration.initialize();
        }),
        onDone: {
          target: 'connected',
          actions: assign({
            rabbitMQConnection: ({ event }) => event.output.connection,
            isConnected: true,
            reconnectAttempts: 0
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            reconnectAttempts: ({ context }) => context.reconnectAttempts + 1
          })
        }
      }
    },

    connected: {
      initial: 'idle',
      entry: ['setupMessageHandlers', 'startHeartbeat'],

      states: {
        idle: {
          on: {
            NEW_MESSAGE: {
              target: 'processing',
              actions: assign({
                pendingTasks: ({ context, event }) => [
                  ...context.pendingTasks,
                  event.message
                ]
              })
            },
            SELF_PROMPT_TRIGGER: {
              target: 'selfPrompting',
              actions: 'triggerSelfAnalysis'
            },
            USER_HISTORY_UPDATE: {
              actions: assign({
                userHistory: ({ context, event }) => [
                  ...context.userHistory.slice(-100), // Keep last 100 entries
                  {
                    action: event.action,
                    timestamp: Date.now(),
                    data: event.data,
                    sessionId: context.activeSession
                  }
                ]
              })
            }
          }
        },

        processing: {
          invoke: {
            id: 'processMessage',
            src: fromPromise(async ({ input }) => {
              const { message } = input;
              return await RabbitMQXStateIntegration.processLegalAIMessage(message);
            }),
            input: ({ context }) => ({
              message: context.pendingTasks[0]
            }),
            onDone: {
              target: 'idle',
              actions: [
                assign({
                  completedTasks: ({ context, event }) => [
                    ...context.completedTasks.slice(-50), // Keep last 50
                    {
                      ...context.pendingTasks[0],
                      result: event.output,
                      completedAt: Date.now()
                    }
                  ],
                  pendingTasks: ({ context }) => context.pendingTasks.slice(1)
                }),
                'updatePerformanceMetrics'
              ]
            },
            onError: {
              target: 'idle',
              actions: [
                assign({
                  errorTasks: ({ context, event }) => [
                    ...context.errorTasks.slice(-20), // Keep last 20 errors
                    {
                      ...context.pendingTasks[0],
                      error: event.error,
                      errorAt: Date.now()
                    }
                  ],
                  pendingTasks: ({ context }) => context.pendingTasks.slice(1)
                }),
                'logError'
              ]
            }
          }
        },

        selfPrompting: {
          invoke: {
            id: 'performSelfAnalysis',
            src: fromPromise(async ({ input }) => {
              const { context, userHistory } = input;
              return await RabbitMQXStateIntegration.performSelfPromptingAnalysis(context, userHistory);
            }),
            input: ({ context }) => ({
              context,
              userHistory: context.userHistory
            }),
            onDone: {
              target: 'idle',
              actions: [
                assign({
                  pendingTasks: ({ context, event }) => [
                    ...context.pendingTasks,
                    ...event.output.recommendedActions
                  ]
                }),
                'publishSelfPromptResults'
              ]
            },
            onError: {
              target: 'idle',
              actions: 'logSelfPromptError'
            }
          }
        }
      },

      on: {
        CONNECTION_LOST: {
          target: 'reconnecting',
          actions: assign({
            isConnected: false
          })
        },
        HEARTBEAT_TIMEOUT: {
          target: 'reconnecting'
        }
      }
    },

    reconnecting: {
      after: {
        5000: {
          target: 'initializing',
          guard: ({ context }) => context.reconnectAttempts < 10
        },
        30000: {
          target: 'error',
          guard: ({ context }) => context.reconnectAttempts >= 10
        }
      }
    },

    error: {
      entry: 'logConnectionError',
      after: {
        60000: 'initializing' // Retry after 1 minute
      }
    }
  }
}, {
  actions: {
    setupMessageHandlers: ({ context }) => {
      console.log('🔗 Setting up RabbitMQ message handlers');
    },

    startHeartbeat: assign({
      lastHeartbeat: Date.now()
    }),

    triggerSelfAnalysis: ({ context }) => {
      console.log('🧠 Triggering self-prompting analysis based on user history');
    },

    updatePerformanceMetrics: assign({
      performanceMetrics: ({ context }) => {
        const completed = context.completedTasks;
        const errors = context.errorTasks;
        const total = completed.length + errors.length;

        return {
          ...context.performanceMetrics,
          successRate: total > 0 ? completed.length / total : 1.0,
          averageResponseTime: completed.length > 0
            ? completed.reduce((sum, task) => sum + ((task as any).completedAt - task.timestamp), 0) / completed.length
            : context.performanceMetrics.averageResponseTime
        };
      }
    }),

    publishSelfPromptResults: ({ context, event }) => {
      if (context.rabbitMQConnection) {
        RabbitMQXStateIntegration.publishMessage({
          type: 'self_prompt',
          payload: event.output,
          priority: 8
        });
      }
    },

    logError: ({ context, event }) => {
      console.error('❌ Legal AI task error:', event.error);
    },

    logSelfPromptError: ({ context, event }) => {
      console.error('❌ Self-prompting error:', event.error);
    },

    logConnectionError: ({ context }) => {
      console.error('❌ RabbitMQ connection error, attempt:', context.reconnectAttempts);
    }
  }
});

// RabbitMQ integration class
export class RabbitMQXStateIntegration {
  private static connection: any = null;
  private static channel: any = null;
  private static isInitialized = false;

  // Free RabbitMQ configuration (CloudAMQP free tier)
  private static config: RabbitMQConfig = {
    host: import.meta.env.RABBITMQ_HOST || 'localhost',
    port: parseInt(import.meta.env.RABBITMQ_PORT || '15674'), // Web STOMP port
    vhost: import.meta.env.RABBITMQ_VHOST || '/',
    username: import.meta.env.RABBITMQ_USERNAME || 'guest',
    password: import.meta.env.RABBITMQ_PASSWORD || 'guest',
    ssl: import.meta.env.RABBITMQ_SSL === 'true',
    heartbeat: 60
  };

  // Legal AI queues (enhanced for WebAssembly inference)
  private static queues = {
    HIGH_PRIORITY: 'legal_ai_high_priority',
    NORMAL_PRIORITY: 'legal_ai_normal',
    LOW_PRIORITY: 'legal_ai_low',
    SELF_PROMPTING: 'legal_ai_self_prompting',
    USER_HISTORY: 'legal_ai_user_history',
    GPU_TASKS: 'legal_ai_gpu_tasks',
    CACHE_UPDATES: 'legal_ai_cache_updates',
    WASM_INFERENCE: 'legal_ai_wasm_inference',           // NEW: WebAssembly inference queue
    WASM_BATCH: 'legal_ai_wasm_batch',                   // NEW: WebAssembly batch processing
    WASM_STREAMING: 'legal_ai_wasm_streaming',           // NEW: WebAssembly streaming queue
    WASM_MODEL_MANAGEMENT: 'legal_ai_wasm_models'        // NEW: WebAssembly model operations
  };

  /**
   * Initialize RabbitMQ connection (free tier compatible)
   */
  static async initialize(): Promise<{ connection: any; isConnected: boolean }> {
    try {
      if (browser) {
        // Browser environment - use WebSocket STOMP client
        const StompJS = await import('@stomp/stompjs');

        this.connection = new StompJS.Client({
          brokerURL: `${this.config.ssl ? 'wss' : 'ws'}://${this.config.host}:${this.config.port}/ws`,
          connectHeaders: {
            login: this.config.username,
            passcode: this.config.password,
            'heart-beat': `${this.config.heartbeat * 1000},${this.config.heartbeat * 1000}`
          },
          debug: (str) => console.log('RabbitMQ STOMP:', str),
          onConnect: (frame) => {
            console.log('✅ Connected to RabbitMQ via WebSocket STOMP');
            this.setupQueues();
          },
          onStompError: (frame) => {
            console.error('❌ RabbitMQ STOMP error:', frame);
          },
          onWebSocketClose: (event: any) => {
            console.log('🔌 RabbitMQ WebSocket closed:', event);
          }
        });

        this.connection.activate();

      } else {
        // Server environment - use amqplib
        const amqp = await import('amqplib');

        const connectionString = `amqp${this.config.ssl ? 's' : ''}://${this.config.username}:${this.config.password}@${this.config.host}/${this.config.vhost}`;
        this.connection = await amqp.connect(connectionString);
        this.channel = await this.connection.createChannel();

        await this.setupQueues();
        console.log('✅ Connected to RabbitMQ via AMQP');
      }

      this.isInitialized = true;
      return { connection: this.connection, isConnected: true };

    } catch (error: any) {
      console.error('❌ Failed to initialize RabbitMQ:', error);
      throw error;
    }
  }

  /**
   * Setup legal AI message queues
   */
  private static async setupQueues(): Promise<void> {
    if (browser && this.connection) {
      // Browser STOMP setup
      for (const queueName of Object.values(this.queues)) {
        this.connection.subscribe(`/queue/${queueName}`, (message) => {
          this.handleMessage(JSON.parse(message.body), queueName);
        });
      }
    } else if (this.channel) {
      // Server AMQP setup
      for (const queueName of Object.values(this.queues)) {
        await this.channel.assertQueue(queueName, {
          durable: true,
          arguments: {
            'x-max-priority': 10, // Priority queue support
            'x-message-ttl': 600000, // 10 minutes TTL
          }
        });

        await this.channel.consume(queueName, (msg) => {
          if (msg) {
            const message = JSON.parse(msg.content.toString());
            this.handleMessage(message, queueName);
            this.channel.ack(msg);
          }
        });
      }
    }
  }

  /**
   * Publish legal AI message
   */
  static async publishMessage(message: Omit<LegalAIMessage, 'id' | 'timestamp'>): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('RabbitMQ not initialized');
    }

    const fullMessage: LegalAIMessage = {
      id: this.generateId(),
      timestamp: Date.now(),
      ...message
    };

    const queueName = this.selectQueue(message.priority || 5, message.type);

    if (browser && this.connection) {
      // Browser STOMP publish
      this.connection.publish({
        destination: `/queue/${queueName}`,
        body: JSON.stringify(fullMessage),
        headers: {
          priority: message.priority?.toString() || '5',
          'content-type': 'application/json'
        }
      });
    } else if (this.channel) {
      // Server AMQP publish
      await this.channel.sendToQueue(
        queueName,
        Buffer.from(JSON.stringify(fullMessage)),
        {
          priority: message.priority || 5,
          persistent: true,
          contentType: 'application/json'
        }
      );
    }
  }

  /**
   * Process legal AI message based on type
   */
  static async processLegalAIMessage(message: LegalAIMessage): Promise<any> {
    const startTime = Date.now();

    try {
      switch (message.type) {
        case 'document_ingestion':
          return await this.processDocumentIngestion(message.payload);

        case 'vector_search':
          return await this.processVectorSearch(message.payload);

        case 'ai_analysis':
          return await this.processAIAnalysis(message.payload);

        case 'self_prompt':
          return await this.processSelfPrompt(message.payload);

        case 'user_history_update':
          return await this.processUserHistoryUpdate(message.payload);

        case 'gpu_task':
          return await this.processGPUTask(message.payload);

        case 'wasm_compilation':
          return await this.processWASMCompilation(message.payload);

        case 'wasm_inference':
          return await this.processWASMInference(message.payload);

        case 'wasm_inference_result':
          return await this.processWASMInferenceResult(message.payload);

        case 'wasm_model_load':
          return await this.processWASMModelLoad(message.payload);

        case 'wasm_model_unload':
          return await this.processWASMModelUnload(message.payload);

        case 'wasm_batch_inference':
          return await this.processWASMBatchInference(message.payload);

        case 'wasm_stream_inference':
          return await this.processWASMStreamInference(message.payload);

        case 'wasm_health_check':
          return await this.processWASMHealthCheck(message.payload);

        case 'cache_invalidation':
          return await this.processCacheInvalidation(message.payload);

        default:
          throw new Error(`Unknown message type: ${message.type}`);
      }
    } catch (error: any) {
      console.error(`❌ Failed to process ${message.type}:`, error);
      throw error;
    } finally {
      const processingTime = Date.now() - startTime;
      console.log(`⚡ Processed ${message.type} in ${processingTime}ms`);
    }
  }

  /**
   * Perform self-prompting analysis based on user history
   */
  static async performSelfPromptingAnalysis(
    context: SelfPromptingContext,
    userHistory: any[]
  ): Promise<{ recommendedActions: LegalAIMessage[]; analysis: any }> {

    // Analyze user behavior patterns
    const patterns = this.analyzeUserPatterns(userHistory);

    // Generate self-prompting recommendations
    const recommendations = [];

    // Pattern: Frequent document searches
    if (patterns.searchFrequency > 10) {
      recommendations.push({
        type: 'cache_invalidation' as LegalAIMessageType,
        payload: {
          action: 'preload_popular_searches',
          searches: patterns.popularSearches
        },
        priority: 7,
        correlationId: context.activeSession
      });
    }

    // Pattern: GPU underutilization
    if (context.performanceMetrics.gpuUtilization < 0.3) {
      recommendations.push({
        type: 'gpu_task' as LegalAIMessageType,
        payload: {
          action: 'batch_vector_processing',
          documents: patterns.recentDocuments
        },
        priority: 6,
        correlationId: context.activeSession
      });
    }

    // Pattern: Low cache hit rate
    if (context.performanceMetrics.cacheHitRate < 0.7) {
      recommendations.push({
        type: 'cache_invalidation' as LegalAIMessageType,
        payload: {
          action: 'rebuild_cache',
          strategy: 'user_behavior_based'
        },
        priority: 8,
        correlationId: context.activeSession
      });
    }

    // Pattern: WebAssembly optimization opportunities
    if (patterns.wasmInferenceFrequency > 5 && patterns.averageWasmLatency > 1000) {
      recommendations.push({
        type: 'wasm_model_load' as LegalAIMessageType,
        payload: {
          action: 'preload_model',
          modelPath: '/models/gemma3-legal-q4.wasm',
          optimization: 'latency_focused',
          reason: 'frequent_usage_detected'
        },
        priority: 7,
        correlationId: context.activeSession
      });
    }

    // Pattern: WebAssembly batch opportunities
    if (patterns.concurrentWasmRequests > 3) {
      recommendations.push({
        type: 'wasm_batch_inference' as LegalAIMessageType,
        payload: {
          action: 'suggest_batching',
          batchSize: Math.min(patterns.concurrentWasmRequests, 8),
          reason: 'concurrent_requests_detected'
        },
        priority: 6,
        correlationId: context.activeSession
      });
    }

    // Pattern: WebAssembly health monitoring
    if (patterns.wasmErrors > 2) {
      recommendations.push({
        type: 'wasm_health_check' as LegalAIMessageType,
        payload: {
          action: 'health_check',
          focus: 'error_investigation',
          reason: 'error_threshold_exceeded'
        },
        priority: 8,
        correlationId: context.activeSession
      });
    }

    return {
      recommendedActions: recommendations.map(rec => ({
        ...rec,
        id: this.generateId(),
        timestamp: Date.now()
      })),
      analysis: patterns
    };
  }

  /**
   * Analyze user behavior patterns for self-prompting (enhanced for WebAssembly)
   */
  private static analyzeUserPatterns(userHistory: any[]): any {
    const recentHistory = userHistory.slice(-50); // Last 50 actions

    return {
      searchFrequency: recentHistory.filter(h => h.action === 'search').length,
      popularSearches: this.extractPopularSearches(recentHistory),
      recentDocuments: this.extractRecentDocuments(recentHistory),
      sessionDuration: this.calculateSessionDuration(recentHistory),
      mostUsedFeatures: this.extractMostUsedFeatures(recentHistory),
      timePatterns: this.analyzeTimePatterns(recentHistory),
      // WebAssembly-specific patterns
      wasmInferenceFrequency: recentHistory.filter(h => h.action === 'wasm_inference').length,
      averageWasmLatency: this.calculateAverageWasmLatency(recentHistory),
      concurrentWasmRequests: this.countConcurrentWasmRequests(recentHistory),
      wasmErrors: recentHistory.filter(h => h.action === 'wasm_error').length,
      wasmModelUsage: this.analyzeWasmModelUsage(recentHistory),
      wasmBatchOpportunities: this.identifyWasmBatchOpportunities(recentHistory)
    };
  }

  // Message processing methods
  private static async processDocumentIngestion(payload: any): Promise<any> {
    // Implementation would handle document ingestion with NES memory + GPU
    return { status: 'ingested', documents: payload.documents?.length || 0 };
  }

  private static async processVectorSearch(payload: any): Promise<any> {
    // Implementation would use GPU-accelerated vector search
    return { results: [], processingTime: Date.now() };
  }

  private static async processAIAnalysis(payload: any): Promise<any> {
    // Implementation would perform AI analysis with WASM acceleration
    return { analysis: 'completed', confidence: 0.95 };
  }

  private static async processSelfPrompt(payload: any): Promise<any> {
    // Implementation would handle self-prompting logic
    return { prompt: 'generated', actions: [] };
  }

  private static async processUserHistoryUpdate(payload: any): Promise<any> {
    // Implementation would update user history in NES memory
    return { updated: true, historySize: payload.actions?.length || 0 };
  }

  private static async processGPUTask(payload: any): Promise<any> {
    // Implementation would queue GPU tasks
    return { queued: true, estimatedTime: '2ms' };
  }

  private static async processWASMCompilation(payload: any): Promise<any> {
    // Implementation would handle WASM compilation
    return { compiled: true, moduleSize: payload.sourceSize };
  }

  private static async processCacheInvalidation(payload: any): Promise<any> {
    // Implementation would handle cache operations
    return { invalidated: true, cacheKeys: payload.keys?.length || 0 };
  }

  /**
   * Process WebAssembly inference request
   */
  private static async processWASMInference(payload: any): Promise<any> {
    try {
      console.log('🧠 Processing WASM inference request:', payload.id);

      // Import WebAssembly inference service dynamically
      const { WASMInferenceRAGService } = await import('../services/webasm-inference-rag.js');

      // Validate payload
      if (!payload.prompt) {
        throw new Error('Missing prompt in WASM inference request');
      }

      // Create inference request
      const request = {
        id: payload.id || this.generateId(),
        prompt: payload.prompt,
        maxTokens: payload.maxTokens || 2048,
        temperature: payload.temperature || 0.7,
        enableRAG: payload.enableRAG !== false,
        priority: payload.priority || 'medium',
        systemMessage: payload.systemMessage,
        contextDocuments: payload.contextDocuments,
        stopSequences: payload.stopSequences
      };

      // Process inference with RAG context
      const result = await WASMInferenceRAGService.processInferenceWithRAG(request, {
        wasmModule: null,
        wasmInstance: null,
        isInitialized: false,
        config: {
          modelPath: payload.modelPath || '/models/gemma3-legal-q4.wasm',
          threads: payload.threads || 8,
          contextLength: payload.contextLength || 4096,
          enableGPU: payload.enableGPU !== false,
          batchSize: payload.batchSize || 4,
          quantization: payload.quantization || 'q4_0'
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
      });

      // Publish result back to RabbitMQ
      await this.publishMessage({
        type: 'wasm_inference_result',
        payload: {
          originalRequestId: payload.id,
          result,
          success: true,
          processingTime: Date.now() - (payload.startTime || Date.now())
        },
        priority: payload.priority === 'critical' ? 9 : 7,
        correlationId: payload.correlationId,
        replyTo: payload.replyTo
      });

      return {
        status: 'completed',
        inferenceId: result.id,
        text: result.text,
        tokens: result.tokens,
        processingTime: result.processingTime,
        ragContext: result.ragContext
      };

    } catch (error: any) {
      console.error('❌ WASM inference processing failed:', error);

      // Publish error result
      await this.publishMessage({
        type: 'wasm_inference_result',
        payload: {
          originalRequestId: payload.id,
          error: error.message,
          success: false
        },
        priority: 8,
        correlationId: payload.correlationId
      });

      throw error;
    }
  }

  /**
   * Process WebAssembly inference result
   */
  private static async processWASMInferenceResult(payload: any): Promise<any> {
    console.log('📤 Processing WASM inference result:', payload.originalRequestId);

    // Store result for client retrieval or trigger callbacks
    if (payload.success) {
      console.log(`✅ WASM inference completed: ${payload.result?.text?.slice(0, 100)}...`);
    } else {
      console.error(`❌ WASM inference failed: ${payload.error}`);
    }

    return {
      processed: true,
      success: payload.success,
      originalRequestId: payload.originalRequestId
    };
  }

  /**
   * Process WebAssembly model loading
   */
  private static async processWASMModelLoad(payload: any): Promise<any> {
    try {
      console.log('📥 Loading WASM model:', payload.modelPath);

      const { WASMInferenceRAGService } = await import('../services/webasm-inference-rag.js');

      const config = {
        modelPath: payload.modelPath,
        threads: payload.threads || 8,
        contextLength: payload.contextLength || 4096,
        enableGPU: payload.enableGPU !== false,
        batchSize: payload.batchSize || 4,
        quantization: payload.quantization || 'q4_0'
      };

      const result = await WASMInferenceRAGService.initialize(config);

      return {
        status: 'loaded',
        modelPath: payload.modelPath,
        moduleSize: result.module ? 'loaded' : 'mock',
        instanceCreated: !!result.instance,
        config
      };

    } catch (error: any) {
      console.error('❌ WASM model loading failed:', error);
      throw error;
    }
  }

  /**
   * Process WebAssembly model unloading
   */
  private static async processWASMModelUnload(payload: any): Promise<any> {
    try {
      console.log('📤 Unloading WASM model:', payload.modelPath);

      const { WASMInferenceRAGService } = await import('../services/webasm-inference-rag.js');
      await WASMInferenceRAGService.cleanup();

      return {
        status: 'unloaded',
        modelPath: payload.modelPath,
        cleanupCompleted: true
      };

    } catch (error: any) {
      console.error('❌ WASM model unloading failed:', error);
      throw error;
    }
  }

  /**
   * Process WebAssembly batch inference
   */
  private static async processWASMBatchInference(payload: any): Promise<any> {
    try {
      console.log('🔄 Processing WASM batch inference:', payload.requests?.length || 0, 'requests');

      const { WASMInferenceRAGService } = await import('../services/webasm-inference-rag.js');
      const results = [];

      // Process each request in the batch
      for (const request of payload.requests || []) {
        try {
          const result = await WASMInferenceRAGService.processInferenceWithRAG(request, payload.context);
          results.push({
            requestId: request.id,
            result,
            success: true
          });
        } catch (error: any) {
          results.push({
            requestId: request.id,
            error: error.message,
            success: false
          });
        }
      }

      return {
        status: 'batch_completed',
        batchId: payload.batchId,
        totalRequests: payload.requests?.length || 0,
        successfulResults: results.filter(r => r.success).length,
        failedResults: results.filter(r => !r.success).length,
        results
      };

    } catch (error: any) {
      console.error('❌ WASM batch inference failed:', error);
      throw error;
    }
  }

  /**
   * Process WebAssembly streaming inference
   */
  private static async processWASMStreamInference(payload: any): Promise<any> {
    try {
      console.log('🌊 Processing WASM streaming inference:', payload.id);

      const { WASMInferenceRAGService } = await import('../services/webasm-inference-rag.js');

      // For streaming, we would set up a series of partial results
      // This is a simplified implementation
      const request = {
        ...payload.request,
        maxTokens: Math.min(payload.request.maxTokens || 2048, 512), // Smaller chunks for streaming
      };

      const result = await WASMInferenceRAGService.processInferenceWithRAG(request, payload.context);

      // In a real streaming implementation, we would send multiple messages
      // with partial results. For now, we simulate streaming behavior.
      const chunks = this.chunkText(result.text, 50); // Split into ~50 char chunks

      for (let i = 0; i < chunks.length; i++) {
        await this.publishMessage({
          type: 'wasm_inference_result',
          payload: {
            originalRequestId: payload.id,
            chunk: chunks[i],
            chunkIndex: i,
            totalChunks: chunks.length,
            isComplete: i === chunks.length - 1,
            success: true
          },
          priority: 7,
          correlationId: payload.correlationId
        });

        // Small delay to simulate streaming
        await new Promise(resolve => setTimeout(resolve, 50));
      }

      return {
        status: 'streaming_completed',
        streamId: payload.id,
        totalChunks: chunks.length
      };

    } catch (error: any) {
      console.error('❌ WASM streaming inference failed:', error);
      throw error;
    }
  }

  /**
   * Process WebAssembly health check
   */
  private static async processWASMHealthCheck(payload: any): Promise<any> {
    try {
      console.log('🏥 Performing WASM health check');

      const { WASMInferenceRAGService } = await import('../services/webasm-inference-rag.js');
      const healthStatus = WASMInferenceRAGService.getHealthStatus();

      return {
        status: 'health_check_completed',
        timestamp: Date.now(),
        health: healthStatus,
        uptime: Date.now() - (payload.startTime || Date.now()),
        version: '1.0.0'
      };

    } catch (error: any) {
      console.error('❌ WASM health check failed:', error);
      return {
        status: 'health_check_failed',
        timestamp: Date.now(),
        error: error.message,
        health: {
          status: 'unhealthy',
          wasm: false,
          rag: false,
          messaging: false
        }
      };
    }
  }

  /**
   * Helper method to chunk text for streaming
   */
  private static chunkText(text: string, chunkSize: number = 50): string[] {
    const chunks: string[] = [];
    for (let i = 0; i < text.length; i += chunkSize) {
      chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
  }

  // Utility methods
  private static selectQueue(priority: number, messageType?: LegalAIMessageType): string {
    // WebAssembly-specific queue routing
    if (messageType?.startsWith('wasm_')) {
      switch (messageType) {
        case 'wasm_inference':
        case 'wasm_inference_result':
          return this.queues.WASM_INFERENCE;
        case 'wasm_batch_inference':
          return this.queues.WASM_BATCH;
        case 'wasm_stream_inference':
          return this.queues.WASM_STREAMING;
        case 'wasm_model_load':
        case 'wasm_model_unload':
        case 'wasm_health_check':
          return this.queues.WASM_MODEL_MANAGEMENT;
        default:
          // Fall through to priority-based routing
          break;
      }
    }

    // Priority-based queue selection for non-WASM messages
    if (priority >= 8) return this.queues.HIGH_PRIORITY;
    if (priority >= 5) return this.queues.NORMAL_PRIORITY;
    return this.queues.LOW_PRIORITY;
  }

  private static generateId(): string {
    return `legal-ai-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private static handleMessage(message: LegalAIMessage, queueName: string): void {
    console.log(`📨 Received message from ${queueName}:`, message.type);
    // Message handling would be coordinated with XState machine
  }

  private static extractPopularSearches(history: any[]): string[] {
    return history
      .filter(h => h.action === 'search')
      .map(h => h.data?.query)
      .filter(Boolean)
      .slice(0, 10);
  }

  private static extractRecentDocuments(history: any[]): string[] {
    return history
      .filter(h => h.action === 'view_document')
      .map(h => h.data?.documentId)
      .filter(Boolean)
      .slice(0, 20);
  }

  private static calculateSessionDuration(history: any[]): number {
    if (history.length === 0) return 0;
    return history[history.length - 1].timestamp - history[0].timestamp;
  }

  private static extractMostUsedFeatures(history: any[]): Record<string, number> {
    const features: Record<string, number> = {};
    history.forEach(h => {
      features[h.action] = (features[h.action] || 0) + 1;
    });
    return features;
  }

  private static analyzeTimePatterns(history: any[]): any {
    const hours = history.map(h => new Date(h.timestamp).getHours());
    const hourCounts: Record<number, number> = {};
    hours.forEach(h => hourCounts[h] = (hourCounts[h] || 0) + 1);

    return {
      mostActiveHour: Object.keys(hourCounts).reduce((a, b) =>
        hourCounts[a] > hourCounts[b] ? a : b
      ),
      activityDistribution: hourCounts
    };
  }

  /**
   * Calculate average WebAssembly inference latency
   */
  private static calculateAverageWasmLatency(history: any[]): number {
    const wasmInferences = history.filter(h => h.action === 'wasm_inference' && h.data?.latency);
    if (wasmInferences.length === 0) return 0;

    const totalLatency = wasmInferences.reduce((sum, h) => sum + (h.data?.latency || 0), 0);
    return totalLatency / wasmInferences.length;
  }

  /**
   * Count concurrent WebAssembly requests
   */
  private static countConcurrentWasmRequests(history: any[]): number {
    const wasmRequests = history.filter(h => h.action === 'wasm_inference');
    if (wasmRequests.length <= 1) return 0;

    // Find overlapping time windows (simplified heuristic)
    let maxConcurrent = 0;
    const timeWindow = 5000; // 5 seconds

    wasmRequests.forEach((request, i) => {
      const concurrent = wasmRequests.filter(other =>
        Math.abs(other.timestamp - request.timestamp) < timeWindow
      ).length;
      maxConcurrent = Math.max(maxConcurrent, concurrent);
    });

    return maxConcurrent;
  }

  /**
   * Analyze WebAssembly model usage patterns
   */
  private static analyzeWasmModelUsage(history: any[]): any {
    const modelActions = history.filter(h => h.action?.includes('wasm_model') || h.action === 'wasm_inference');

    const modelUsage: Record<string, number> = {};
    modelActions.forEach(h => {
      const modelPath = h.data?.modelPath || h.data?.config?.modelPath || 'unknown';
      modelUsage[modelPath] = (modelUsage[modelPath] || 0) + 1;
    });

    return {
      totalModelActions: modelActions.length,
      modelUsageBreakdown: modelUsage,
      mostUsedModel: Object.keys(modelUsage).reduce((a, b) =>
        modelUsage[a] > modelUsage[b] ? a : b, 'none'
      )
    };
  }

  /**
   * Identify WebAssembly batch processing opportunities
   */
  private static identifyWasmBatchOpportunities(history: any[]): any {
    const wasmInferences = history.filter(h => h.action === 'wasm_inference');
    const timeWindow = 10000; // 10 seconds

    const batches = [];
    let currentBatch = [];
    let lastTimestamp = 0;

    wasmInferences.forEach(inference => {
      if (inference.timestamp - lastTimestamp < timeWindow && currentBatch.length > 0) {
        currentBatch.push(inference);
      } else {
        if (currentBatch.length > 1) {
          batches.push(currentBatch);
        }
        currentBatch = [inference];
      }
      lastTimestamp = inference.timestamp;
    });

    // Don't forget the last batch
    if (currentBatch.length > 1) {
      batches.push(currentBatch);
    }

    return {
      totalBatchOpportunities: batches.length,
      averageBatchSize: batches.length > 0
        ? batches.reduce((sum, batch) => sum + batch.length, 0) / batches.length
        : 0,
      largestBatchSize: batches.length > 0
        ? Math.max(...batches.map(batch => batch.length))
        : 0,
      potentialLatencySavings: batches.length > 0
        ? batches.reduce((sum, batch) => sum + (batch.length - 1) * 200, 0) // Est. 200ms saved per batched request
        : 0
    };
  }

  /**
   * Cleanup and close connections
   */
  static async cleanup(): Promise<void> {
    if (browser && this.connection) {
      this.connection.deactivate();
    } else if (this.connection) {
      await this.connection.close();
    }

    this.isInitialized = false;
    console.log('🧹 RabbitMQ connections cleaned up');
  }
}

// Export singleton for global use
export const rabbitMQIntegration = RabbitMQXStateIntegration;