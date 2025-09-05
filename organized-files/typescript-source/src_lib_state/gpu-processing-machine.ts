/**
 * XState Machine for GPU Processing Orchestration
 * Manages concurrent document processing with Go SIMD and Node.js GPU services
 */

import { createMachine, assign, createActor, setup } from 'xstate';

// Event types
type GPUProcessingEvents =
  | { type: 'PROCESS_DOCUMENT'; documentId: string; content: string; options?: ProcessingOptions }
  | { type: 'BATCH_PROCESS'; documents: DocumentInput[] }
  | { type: 'PROCESSING_COMPLETE'; documentId: string; result: ProcessingResult }
  | { type: 'PROCESSING_ERROR'; documentId: string; error: string }
  | { type: 'RETRY'; documentId: string }
  | { type: 'CANCEL'; documentId: string }
  | { type: 'PAUSE_PROCESSING' }
  | { type: 'RESUME_PROCESSING' }
  | { type: 'CLEAR_QUEUE' }
  | { type: 'SERVICE_HEALTH_CHECK' }
  | { type: 'SERVICE_STATUS'; service: 'go-simd' | 'node-gpu'; status: 'healthy' | 'unhealthy' };

interface ProcessingOptions {
  processType: 'embeddings' | 'clustering' | 'similarity' | 'boost' | 'full';
  priority: number;
  timeout: number;
  retries: number;
  batchSize: number;
}

interface DocumentInput {
  documentId: string;
  content: string;
  title?: string;
  metadata?: Record<string, any>;
  options?: ProcessingOptions;
}

interface ProcessingResult {
  documentId: string;
  embeddings?: Float32Array[];
  clusters?: number[];
  similarities?: number[];
  boostTransforms?: Float32Array[];
  processingTime: number;
  metadata: Record<string, any>;
}

interface QueuedDocument {
  document: DocumentInput;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  retryCount: number;
  startTime?: number;
  endTime?: number;
  error?: string;
  result?: ProcessingResult;
}

// Context
interface GPUProcessingContext {
  processingQueue: QueuedDocument[];
  activeProcessing: Map<string, QueuedDocument>;
  completedDocuments: Map<string, ProcessingResult>;
  errorDocuments: Map<string, { error: string; document: DocumentInput }>;
  serviceHealth: {
    goSimd: 'healthy' | 'unhealthy' | 'unknown';
    nodeGpu: 'healthy' | 'unhealthy' | 'unknown';
  };
  maxConcurrentProcessing: number;
  totalProcessed: number;
  totalErrors: number;
  metrics: {
    avgProcessingTime: number;
    throughputPerMinute: number;
    errorRate: number;
  };
}

// Machine setup
const gpuProcessingMachine = setup({
  types: {
    context: {} as GPUProcessingContext,
    events: {} as GPUProcessingEvents
  },
  actions: {
    addToQueue: assign({
      processingQueue: ({ context, event }) => {
        if (event.type === 'PROCESS_DOCUMENT') {
          const queuedDoc: QueuedDocument = {
            document: {
              documentId: event.documentId,
              content: event.content,
              options: event.options
            },
            status: 'queued',
            retryCount: 0
          };
          return [...context.processingQueue, queuedDoc];
        }
        if (event.type === 'BATCH_PROCESS') {
          const queuedDocs = event.documents.map(doc => ({
            document: doc,
            status: 'queued' as const,
            retryCount: 0
          }));
          return [...context.processingQueue, ...queuedDocs];
        }
        return context.processingQueue;
      }
    }),

    startProcessing: assign({
      activeProcessing: ({ context }) => {
        const newActive = new Map(context.activeProcessing);
        const availableSlots = context.maxConcurrentProcessing - newActive.size;
        
        for (let i = 0; i < availableSlots && i < context.processingQueue.length; i++) {
          const queuedDoc = context.processingQueue[i];
          if (queuedDoc.status === 'queued') {
            queuedDoc.status = 'processing';
            queuedDoc.startTime = Date.now();
            newActive.set(queuedDoc.document.documentId, queuedDoc);
          }
        }
        
        return newActive;
      },
      processingQueue: ({ context }) => {
        return context.processingQueue.filter(doc => 
          doc.status !== 'processing' || !context.activeProcessing.has(doc.document.documentId)
        );
      }
    }),

    completeProcessing: assign({
      activeProcessing: ({ context, event }) => {
        if (event.type !== 'PROCESSING_COMPLETE') return context.activeProcessing;
        
        const newActive = new Map(context.activeProcessing);
        newActive.delete(event.documentId);
        return newActive;
      },
      completedDocuments: ({ context, event }) => {
        if (event.type !== 'PROCESSING_COMPLETE') return context.completedDocuments;
        
        const newCompleted = new Map(context.completedDocuments);
        newCompleted.set(event.documentId, event.result);
        return newCompleted;
      },
      totalProcessed: ({ context, event }) => {
        if (event.type !== 'PROCESSING_COMPLETE') return context.totalProcessed;
        return context.totalProcessed + 1;
      }
    }),

    handleProcessingError: assign({
      activeProcessing: ({ context, event }) => {
        if (event.type !== 'PROCESSING_ERROR') return context.activeProcessing;
        
        const newActive = new Map(context.activeProcessing);
        const doc = newActive.get(event.documentId);
        
        if (doc) {
          doc.retryCount++;
          doc.error = event.error;
          doc.status = 'failed';
          
          // Check if we should retry
          const maxRetries = doc.document.options?.retries ?? 3;
          if (doc.retryCount < maxRetries) {
            doc.status = 'queued';
          } else {
            newActive.delete(event.documentId);
          }
        }
        
        return newActive;
      },
      errorDocuments: ({ context, event }) => {
        if (event.type !== 'PROCESSING_ERROR') return context.errorDocuments;
        
        const doc = context.activeProcessing.get(event.documentId);
        if (doc && doc.retryCount >= (doc.document.options?.retries ?? 3)) {
          const newErrors = new Map(context.errorDocuments);
          newErrors.set(event.documentId, { error: event.error, document: doc.document });
          return newErrors;
        }
        
        return context.errorDocuments;
      },
      totalErrors: ({ context, event }) => {
        if (event.type !== 'PROCESSING_ERROR') return context.totalErrors;
        return context.totalErrors + 1;
      }
    }),

    updateServiceHealth: assign({
      serviceHealth: ({ context, event }) => {
        if (event.type !== 'SERVICE_STATUS') return context.serviceHealth;
        
        return {
          ...context.serviceHealth,
          [event.service === 'go-simd' ? 'goSimd' : 'nodeGpu']: event.status
        };
      }
    }),

    clearQueue: assign({
      processingQueue: () => [],
      activeProcessing: () => new Map(),
      errorDocuments: () => new Map()
    }),

    cancelDocument: assign({
      processingQueue: ({ context, event }) => {
        if (event.type !== 'CANCEL') return context.processingQueue;
        
        return context.processingQueue.map(doc => 
          doc.document.documentId === event.documentId 
            ? { ...doc, status: 'cancelled' as const }
            : doc
        ).filter(doc => doc.status !== 'cancelled');
      },
      activeProcessing: ({ context, event }) => {
        if (event.type !== 'CANCEL') return context.activeProcessing;
        
        const newActive = new Map(context.activeProcessing);
        newActive.delete(event.documentId);
        return newActive;
      }
    }),

    updateMetrics: assign({
      metrics: ({ context }) => {
        const totalDocs = context.totalProcessed + context.totalErrors;
        const errorRate = totalDocs > 0 ? (context.totalErrors / totalDocs) * 100 : 0;
        
        // Calculate average processing time from completed documents
        let avgProcessingTime = 0;
        let totalTime = 0;
        let count = 0;
        
        for (const result of context.completedDocuments.values()) {
          totalTime += result.processingTime;
          count++;
        }
        
        if (count > 0) {
          avgProcessingTime = totalTime / count;
        }
        
        // Calculate throughput (simplified)
        const throughputPerMinute = context.totalProcessed / (Date.now() / 60000);
        
        return {
          avgProcessingTime,
          throughputPerMinute,
          errorRate
        };
      }
    })
  },

  actors: {
    processDocument: async ({ input }: { input: QueuedDocument }) => {
      // This would call the actual Go SIMD and Node.js GPU services
      try {
        const response = await fetch('/api/gpu-processing/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            documentId: input.document.documentId,
            content: input.document.content,
            options: input.document.options
          })
        });
        
        if (!response.ok) {
          throw new Error(`Processing failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        return result;
      } catch (error) {
        throw error;
      }
    },

    healthChecker: async () => {
      const healthChecks = await Promise.allSettled([
        fetch('/api/gpu-processing/health/go-simd'),
        fetch('/api/gpu-processing/health/node-gpu')
      ]);
      
      return {
        goSimd: healthChecks[0].status === 'fulfilled' && healthChecks[0].value.ok ? 'healthy' : 'unhealthy',
        nodeGpu: healthChecks[1].status === 'fulfilled' && healthChecks[1].value.ok ? 'healthy' : 'unhealthy'
      };
    }
  },

  guards: {
    canProcessMore: ({ context }) => {
      return context.activeProcessing.size < context.maxConcurrentProcessing &&
             context.processingQueue.some(doc => doc.status === 'queued');
    },

    servicesHealthy: ({ context }) => {
      return context.serviceHealth.goSimd === 'healthy' &&
             context.serviceHealth.nodeGpu === 'healthy';
    },

    hasQueuedDocuments: ({ context }) => {
      return context.processingQueue.some(doc => doc.status === 'queued');
    },

    isProcessingActive: ({ context }) => {
      return context.activeProcessing.size > 0;
    }
  }
}).createMachine({
  id: 'gpuProcessing',
  
  initial: 'idle',
  
  context: {
    processingQueue: [],
    activeProcessing: new Map(),
    completedDocuments: new Map(),
    errorDocuments: new Map(),
    serviceHealth: {
      goSimd: 'unknown',
      nodeGpu: 'unknown'
    },
    maxConcurrentProcessing: 5,
    totalProcessed: 0,
    totalErrors: 0,
    metrics: {
      avgProcessingTime: 0,
      throughputPerMinute: 0,
      errorRate: 0
    }
  },

  states: {
    idle: {
      entry: ['updateMetrics'],
      on: {
        PROCESS_DOCUMENT: {
          target: 'processing',
          actions: ['addToQueue']
        },
        BATCH_PROCESS: {
          target: 'processing',
          actions: ['addToQueue']
        },
        SERVICE_HEALTH_CHECK: {
          target: 'checkingHealth'
        }
      }
    },

    checkingHealth: {
      invoke: {
        src: 'healthChecker',
        onDone: {
          target: 'idle',
          actions: [
            ({ event }) => {
              // Dispatch service status events
              return [
                { type: 'SERVICE_STATUS', service: 'go-simd', status: event.output.goSimd },
                { type: 'SERVICE_STATUS', service: 'node-gpu', status: event.output.nodeGpu }
              ];
            },
            'updateServiceHealth'
          ]
        },
        onError: {
          target: 'idle',
          actions: [
            'updateServiceHealth'
          ]
        }
      }
    },

    processing: {
      entry: ['startProcessing'],
      always: [
        {
          target: 'idle',
          guard: ({ context }) => !context.hasQueuedDocuments && context.activeProcessing.size === 0
        }
      ],
      on: {
        PROCESS_DOCUMENT: {
          actions: ['addToQueue', 'startProcessing']
        },
        BATCH_PROCESS: {
          actions: ['addToQueue', 'startProcessing']
        },
        PROCESSING_COMPLETE: {
          actions: ['completeProcessing', 'updateMetrics'],
          always: {
            guard: 'canProcessMore',
            actions: ['startProcessing']
          }
        },
        PROCESSING_ERROR: {
          actions: ['handleProcessingError', 'updateMetrics'],
          always: {
            guard: 'canProcessMore',
            actions: ['startProcessing']
          }
        },
        CANCEL: {
          actions: ['cancelDocument']
        },
        PAUSE_PROCESSING: {
          target: 'paused'
        },
        CLEAR_QUEUE: {
          target: 'idle',
          actions: ['clearQueue']
        }
      }
    },

    paused: {
      on: {
        RESUME_PROCESSING: {
          target: 'processing',
          guard: 'hasQueuedDocuments'
        },
        CLEAR_QUEUE: {
          target: 'idle',
          actions: ['clearQueue']
        }
      }
    },

    error: {
      on: {
        RETRY: {
          target: 'processing'
        },
        CLEAR_QUEUE: {
          target: 'idle',
          actions: ['clearQueue']
        }
      }
    }
  }
});

// Factory function to create actor
export function createGPUProcessingActor() {
  return createActor(gpuProcessingMachine);
}

// Export types
export type { GPUProcessingEvents, ProcessingOptions, DocumentInput, ProcessingResult, GPUProcessingContext };
export { gpuProcessingMachine };