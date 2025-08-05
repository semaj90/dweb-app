/**
 * Go Microservice XState Machine
 * Manages connections and requests to the Go AI microservice
 */

import { createMachine, assign, fromPromise } from 'xstate';
import type { 
  GoMicroserviceContext, 
  GoMicroserviceEvents, 
  GoServiceRequest,
  GoServiceResponse 
} from './types';

const DEFAULT_TIMEOUT = 30000; // 30 seconds
const HEALTH_CHECK_INTERVAL = 60000; // 1 minute

export const goMicroserviceMachine = createMachine({
  id: 'goMicroservice',
  types: {} as {
    context: GoMicroserviceContext;
    events: GoMicroserviceEvents[keyof GoMicroserviceEvents];
  },
  
  context: {
    userId: undefined,
    sessionId: '',
    retryCount: 0,
    timestamp: Date.now(),
    endpoint: 'http://localhost:8080',
    connectionStatus: 'disconnected',
    healthCheck: {
      lastCheck: 0,
      status: 'unhealthy'
    }
  },

  initial: 'disconnected',

  states: {
    disconnected: {
      entry: assign({
        connectionStatus: 'disconnected',
        retryCount: 0
      }),
      
      on: {
        CONNECT: {
          target: 'connecting',
          actions: assign({
            endpoint: ({ event }) => event.endpoint
          })
        }
      }
    },

    connecting: {
      entry: assign({
        connectionStatus: 'connecting',
        timestamp: Date.now()
      }),

      invoke: {
        id: 'connectToService',
        src: fromPromise(async ({ input }: { input: { endpoint: string } }) => {
          const response = await fetch(`${input.endpoint}/health`, {
            method: 'GET',
            timeout: 5000
          });
          
          if (!response.ok) {
            throw new Error(`Connection failed: ${response.statusText}`);
          }
          
          return await response.json();
        }),
        input: ({ context }) => ({ endpoint: context.endpoint }),
        onDone: {
          target: 'connected',
          actions: [
            assign({
              connectionStatus: 'connected',
              healthCheck: ({ event }) => ({
                lastCheck: Date.now(),
                status: 'healthy',
                responseTime: Date.now() - event.data.timestamp
              })
            }),
            'logConnection'
          ]
        },
        onError: {
          target: 'error',
          actions: [
            assign({
              connectionStatus: 'error',
              error: ({ event }) => event.error.message
            }),
            'logConnectionError'
          ]
        }
      }
    },

    connected: {
      entry: ['startHealthCheckTimer'],
      exit: ['stopHealthCheckTimer'],

      initial: 'idle',

      states: {
        idle: {
          on: {
            MAKE_REQUEST: {
              target: 'requesting'
            },
            HEALTH_CHECK: {
              target: 'health_checking'
            }
          }
        },

        requesting: {
          invoke: {
            id: 'makeRequest',
            src: fromPromise(async ({ input }: { input: { request: GoServiceRequest; endpoint: string } }) => {
              const { request, endpoint } = input;
              const startTime = Date.now();
              
              const response = await fetch(`${endpoint}${request.path}`, {
                method: request.method,
                headers: {
                  'Content-Type': 'application/json',
                  ...request.headers
                },
                body: request.body ? JSON.stringify(request.body) : undefined,
                timeout: DEFAULT_TIMEOUT
              });

              if (!response.ok) {
                throw new Error(`Request failed: ${response.status} ${response.statusText}`);
              }

              const data = await response.json();
              const duration = Date.now() - startTime;

              return {
                status: response.status,
                data,
                headers: Object.fromEntries(response.headers.entries()),
                duration
              } as GoServiceResponse;
            }),
            input: ({ context, event }) => ({ 
              request: event.request, 
              endpoint: context.endpoint 
            }),
            onDone: {
              target: 'idle',
              actions: [
                assign({
                  response: ({ event }) => event.output,
                  retryCount: 0
                }),
                'logRequestSuccess'
              ]
            },
            onError: {
              target: 'idle',
              actions: [
                assign({
                  error: ({ event }) => event.error.message,
                  retryCount: ({ context }) => context.retryCount + 1
                }),
                'logRequestError'
              ]
            }
          }
        },

        health_checking: {
          invoke: {
            id: 'healthCheck',
            src: fromPromise(async ({ input }: { input: { endpoint: string } }) => {
              const startTime = Date.now();
              const response = await fetch(`${input.endpoint}/health`, {
                method: 'GET',
                timeout: 5000
              });
              
              if (!response.ok) {
                throw new Error(`Health check failed: ${response.statusText}`);
              }
              
              const data = await response.json();
              return {
                ...data,
                responseTime: Date.now() - startTime
              };
            }),
            input: ({ context }) => ({ endpoint: context.endpoint }),
            onDone: {
              target: 'idle',
              actions: assign({
                healthCheck: ({ event }) => ({
                  lastCheck: Date.now(),
                  status: 'healthy',
                  responseTime: event.output.responseTime
                })
              })
            },
            onError: {
              target: 'idle',
              actions: assign({
                healthCheck: {
                  lastCheck: Date.now(),
                  status: 'unhealthy'
                }
              })
            }
          }
        }
      },

      on: {
        DISCONNECT: {
          target: 'disconnected'
        },
        CONNECTION_ERROR: {
          target: 'error'
        }
      }
    },

    error: {
      entry: assign({
        connectionStatus: 'error',
        timestamp: Date.now()
      }),

      after: {
        5000: [
          {
            target: 'connecting',
            guard: 'shouldRetry'
          },
          {
            target: 'disconnected'
          }
        ]
      },

      on: {
        CONNECT: {
          target: 'connecting',
          actions: assign({
            retryCount: 0,
            error: undefined
          })
        }
      }
    }
  }
}, {
  actions: {
    logConnection: ({ context }) => {
      console.log(`✅ Connected to Go microservice at ${context.endpoint}`);
    },
    
    logConnectionError: ({ context }) => {
      console.error(`❌ Failed to connect to Go microservice: ${context.error}`);
    },
    
    logRequestSuccess: ({ context }) => {
      console.log(`🚀 Request successful in ${context.response?.duration}ms`);
    },
    
    logRequestError: ({ context }) => {
      console.error(`❌ Request failed: ${context.error}`);
    },

    startHealthCheckTimer: () => {
      // In a real implementation, start periodic health checks
      console.log('🔄 Starting health check timer');
    },

    stopHealthCheckTimer: () => {
      // Stop the health check timer
      console.log('⏹️ Stopping health check timer');
    }
  },

  guards: {
    shouldRetry: ({ context }) => {
      return context.retryCount < 3;
    }
  }
});

// Service functions for common Go microservice operations
export const goMicroserviceServices = {
  // High-performance JSON parsing
  parseJSON: (data: any, options?: { parallel?: boolean; chunkSize?: number }) => ({
    type: 'MAKE_REQUEST' as const,
    request: {
      method: 'POST' as const,
      path: '/parse',
      body: {
        data,
        format: 'json',
        options: {
          parallel: options?.parallel || false,
          chunk_size: options?.chunkSize || 1024,
          compression: true
        }
      }
    }
  }),

  // Train Self-Organizing Map
  trainSOM: (vectors: number[][], labels: string[], options?: {
    width?: number;
    height?: number;
    iterations?: number;
    learningRate?: number;
  }) => ({
    type: 'MAKE_REQUEST' as const,
    request: {
      method: 'POST' as const,
      path: '/train-som',
      body: {
        vectors,
        labels,
        dimensions: {
          width: options?.width || 10,
          height: options?.height || 10
        },
        iterations: options?.iterations || 1000,
        learning_rate: options?.learningRate || 0.1
      }
    }
  }),

  // CUDA inference
  cudaInfer: (model: string, input: any, options?: {
    batchSize?: number;
    precision?: 'fp32' | 'fp16' | 'int8';
    streaming?: boolean;
  }) => ({
    type: 'MAKE_REQUEST' as const,
    request: {
      method: 'POST' as const,
      path: '/cuda-infer',
      body: {
        model,
        input,
        batch_size: options?.batchSize || 1,
        precision: options?.precision || 'fp32',
        streaming: options?.streaming || false
      }
    }
  }),

  // Get system metrics
  getMetrics: () => ({
    type: 'MAKE_REQUEST' as const,
    request: {
      method: 'GET' as const,
      path: '/metrics'
    }
  }),

  // Health check
  healthCheck: () => ({
    type: 'HEALTH_CHECK' as const
  })
};

// Utility functions for working with the machine
export const createGoMicroserviceActor = (endpoint = 'http://localhost:8080') => {
  const actor = goMicroserviceMachine.provide({
    // Add any custom implementations here
  });
  
  return actor;
};

// Helper to check if service is ready
export const isServiceReady = (state: any) => {
  return state.matches('connected.idle') && state.context.healthCheck.status === 'healthy';
};

// Helper to get last response
export const getLastResponse = (state: any): GoServiceResponse | undefined => {
  return state.context.response;
};

// Helper to get connection status
export const getConnectionStatus = (state: any) => {
  return {
    status: state.context.connectionStatus,
    endpoint: state.context.endpoint,
    lastHealthCheck: state.context.healthCheck.lastCheck,
    healthStatus: state.context.healthCheck.status,
    responseTime: state.context.healthCheck.responseTime,
    error: state.context.error
  };
};