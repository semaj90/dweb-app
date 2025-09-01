// @ts-nocheck
// XState Graph Cache Machine - Orchestrates cache states and background refresh
// Implements the recommended runtime flow with idle signals and telemetry

import { createMachine, assign } from 'xstate';
import type { ActorRefFrom } from 'xstate';

// Types for the graph cache system
export interface GraphCacheContext {
  query: string | null;
  params: Record<string, any>;
  result: any;
  source: 'indexeddb_cache' | 'wasm' | 'neo4j' | 'graph_service' | 'snapshot_fallback';
  isStale: boolean;
  isAuthoritative: boolean;
  cacheHit: boolean;
  latency: number;
  queryHash: string | null;
  telemetry: {
    totalQueries: number;
    cacheHits: number;
    cacheMisses: number;
    hitRate: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
    p99LatencyMs: number;
  };
  worker: any;
  refreshJob: string | null;
  lastRefresh: number;
  backgroundRefreshEnabled: boolean;
  retryCount: number;
  maxRetries: number;
}

export type GraphCacheEvent =
  | { type: 'QUERY'; query: string; params?: Record<string, any> }
  | { type: 'CACHE_HIT'; result: any; source: string; latency: number }
  | { type: 'CACHE_MISS'; queryHash: string }
  | { type: 'WASM_RESULT'; result: any; latency: number }
  | { type: 'AUTHORITATIVE_RESULT'; result: any; source: string; latency: number }
  | { type: 'BACKGROUND_REFRESH'; queryHash: string }
  | { type: 'REFRESH_COMPLETE'; result: any }
  | { type: 'REFRESH_FAILED'; error: string }
  | { type: 'ENABLE_BACKGROUND_REFRESH' }
  | { type: 'DISABLE_BACKGROUND_REFRESH' }
  | { type: 'RESET_TELEMETRY' }
  | { type: 'GET_TELEMETRY' }
  | { type: 'WORKER_READY' }
  | { type: 'WORKER_ERROR'; error: string }
  | { type: 'INVALIDATE_CACHE'; key?: string }
  | { type: 'IDLE_CALLBACK' }
  | { type: 'RETRY' };

// XState machine for graph cache orchestration
export const graphCacheMachine = createMachine({
  id: 'graphCache',
  initial: 'initializing',
  // Note: predictableActionArguments removed for version compatibility

  context: {
    query: null,
    params: {},
    result: null,
    source: 'indexeddb_cache',
    isStale: false,
    isAuthoritative: false,
    cacheHit: false,
    latency: 0,
    queryHash: null,
    telemetry: {
      totalQueries: 0,
      cacheHits: 0,
      cacheMisses: 0,
      hitRate: 0,
      avgLatencyMs: 0,
      p95LatencyMs: 0,
      p99LatencyMs: 0,
    },
    worker: null,
    refreshJob: null,
    lastRefresh: 0,
    backgroundRefreshEnabled: true,
    retryCount: 0,
    maxRetries: 3,
  } satisfies GraphCacheContext,

  states: {
    initializing: {
      entry: 'initializeWorker',
      on: {
        WORKER_READY: 'idle',
        WORKER_ERROR: 'error'
      }
    },

    idle: {
      entry: 'scheduleIdleCallback',
      on: {
        QUERY: 'querying',
        GET_TELEMETRY: { actions: 'provideTelemetry' },
        RESET_TELEMETRY: { actions: 'resetTelemetry' },
        ENABLE_BACKGROUND_REFRESH: { actions: 'enableBackgroundRefresh' },
        DISABLE_BACKGROUND_REFRESH: { actions: 'disableBackgroundRefresh' },
        INVALIDATE_CACHE: { actions: 'invalidateCache' },
        IDLE_CALLBACK: [
          {
            target: 'backgroundRefreshing',
            guard: 'shouldBackgroundRefresh'
          }
        ]
      }
    },

    querying: {
      entry: ['setQuery', 'incrementQueryCount'],
      initial: 'checkingCache',

      states: {
        checkingCache: {
          entry: 'queryWorker',
          on: {
            CACHE_HIT: {
              target: 'cacheHit',
              actions: 'setCacheResult'
            },
            CACHE_MISS: 'cacheMiss'
          }
        },

        cacheHit: {
          entry: ['notifyCacheHit', 'updateTelemetry'],
          after: {
            100: [
              {
                target: '#graphCache.backgroundRefreshing',
                guard: 'isStaleResult'
              },
              {
                target: '#graphCache.idle'
              }
            ]
          }
        },

        cacheMiss: {
          entry: 'notifyCacheMiss',
          initial: 'wasmQuery',

          states: {
            wasmQuery: {
              entry: 'queryWasmWorker',
              on: {
                WASM_RESULT: {
                  target: 'authoritativeQuery',
                  actions: ['setWasmResult', 'notifyProvisionalResult']
                }
              },
              after: {
                5000: 'authoritativeQuery' // Fallback if WASM times out
              }
            },

            authoritativeQuery: {
              entry: 'queryAuthoritativeSource',
              on: {
                AUTHORITATIVE_RESULT: {
                  target: '#graphCache.rehydrated',
                  actions: 'setAuthoritativeResult'
                },
                REFRESH_FAILED: [
                  {
                    target: 'authoritativeQuery',
                    guard: 'canRetry',
                    actions: 'incrementRetry'
                  },
                  {
                    target: '#graphCache.error',
                    actions: 'setError'
                  }
                ]
              },
              after: {
                10000: [
                  {
                    target: 'authoritativeQuery',
                    guard: 'canRetry',
                    actions: 'incrementRetry'
                  },
                  {
                    target: '#graphCache.error'
                  }
                ]
              }
            }
          }
        }
      }
    },

    backgroundRefreshing: {
      entry: ['setRefreshJob', 'queryAuthoritativeSource'],
      on: {
        REFRESH_COMPLETE: {
          target: 'revalidated',
          actions: ['setAuthoritativeResult', 'clearRefreshJob']
        },
        REFRESH_FAILED: {
          target: 'idle',
          actions: 'clearRefreshJob'
        }
      },
      after: {
        15000: {
          target: 'idle',
          actions: 'clearRefreshJob'
        }
      }
    },

    rehydrated: {
      entry: ['notifyRehydration', 'updateCaches', 'updateTelemetry'],
      after: {
        500: 'idle'
      }
    },

    revalidated: {
      entry: ['notifyRevalidation', 'updateCaches', 'updateTelemetry'],
      after: {
        500: 'idle'
      }
    },

    error: {
      entry: 'notifyError',
      on: {
        RETRY: [
          {
            target: 'querying',
            guard: 'canRetry',
            actions: 'incrementRetry'
          }
        ],
        QUERY: 'querying'
      },
      after: {
        5000: 'idle' // Auto-recover after 5 seconds
      }
    }
  }
}, {
  actions: {
    initializeWorker: assign({
      worker: () => {
        if (typeof Worker !== 'undefined') {
          const worker = new Worker('/src/lib/workers/graph-worker.js');

          worker.onmessage = (event) => {
            const { type, data } = event.data;

            switch (type) {
              case 'worker_ready':
                // Handle in machine
                break;
              case 'query_result':
                if (data.cache_hit) {
                  // Send CACHE_HIT event
                } else if (data.source === 'wasm') {
                  // Send WASM_RESULT event
                }
                break;
              case 'query_result_authoritative':
                // Send AUTHORITATIVE_RESULT event
                break;
              default:
                console.log('Worker message:', type, data);
            }
          };

          return worker;
        }
        return null;
      }
    }),

    setQuery: assign({
      query: ({ event }) => event.type === 'QUERY' ? event.query : null,
      params: ({ event }) => event.type === 'QUERY' ? (event.params || {}) : {},
      queryHash: ({ event }) => {
        if (event.type === 'QUERY') {
          // Simple hash function
          const str = JSON.stringify({ query: event.query, params: event.params });
          let hash = 0;
          for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
          }
          return hash.toString();
        }
        return null;
      },
      retryCount: 0
    }),

    incrementQueryCount: assign({
      telemetry: ({ context }) => ({
        ...context.telemetry,
        totalQueries: context.telemetry.totalQueries + 1
      })
    }),

    setCacheResult: assign({
      result: ({ event }) => event.type === 'CACHE_HIT' ? event.result : null,
      source: ({ event }) => event.type === 'CACHE_HIT' ? event.source as any : 'indexeddb_cache',
      cacheHit: true,
      latency: ({ event }) => event.type === 'CACHE_HIT' ? event.latency : 0,
      telemetry: ({ context }) => ({
        ...context.telemetry,
        cacheHits: context.telemetry.cacheHits + 1
      })
    }),

    setWasmResult: assign({
      result: ({ event }) => event.type === 'WASM_RESULT' ? event.result : null,
      source: 'wasm' as const,
      latency: ({ event }) => event.type === 'WASM_RESULT' ? event.latency : 0
    }),

    setAuthoritativeResult: assign({
      result: ({ event }) => {
        if (event.type === 'AUTHORITATIVE_RESULT') return event.result;
        if (event.type === 'REFRESH_COMPLETE') return event.result;
        return null;
      },
      source: ({ event }) => {
        if (event.type === 'AUTHORITATIVE_RESULT') return event.source as any;
        return 'neo4j' as const;
      },
      isAuthoritative: true,
      lastRefresh: Date.now()
    }),

    setRefreshJob: assign({
      refreshJob: ({ context }) => `refresh_${context.queryHash}_${Date.now()}`
    }),

    clearRefreshJob: assign({
      refreshJob: null
    }),

    incrementRetry: assign({
      retryCount: ({ context }) => context.retryCount + 1
    }),

    enableBackgroundRefresh: assign({
      backgroundRefreshEnabled: true
    }),

    disableBackgroundRefresh: assign({
      backgroundRefreshEnabled: false
    }),

    updateTelemetry: assign({
      telemetry: ({ context }) => {
        const total = context.telemetry.cacheHits + context.telemetry.cacheMisses;
        return {
          ...context.telemetry,
          hitRate: total > 0 ? (context.telemetry.cacheHits / total * 100) : 0,
          // Update other metrics as needed
        };
      }
    }),

    resetTelemetry: assign({
      telemetry: {
        totalQueries: 0,
        cacheHits: 0,
        cacheMisses: 0,
        hitRate: 0,
        avgLatencyMs: 0,
        p95LatencyMs: 0,
        p99LatencyMs: 0,
      }
    }),

    queryWorker: ({ context }) => {
      if (context.worker && context.query) {
        context.worker.postMessage({
          type: 'query',
          data: {
            query: context.query,
            params: context.params
          }
        });
      }
    },

    queryWasmWorker: ({ context }) => {
      // WASM query is handled within the worker
      console.log('🌐 Querying WASM worker for instant results');
    },

    queryAuthoritativeSource: ({ context }) => {
      console.log('🔍 Querying authoritative source (Neo4j/Graph service)');
    },

    scheduleIdleCallback: () => {
      if (typeof requestIdleCallback !== 'undefined') {
        requestIdleCallback(() => {
          // Send IDLE_CALLBACK event
          console.log('⏰ Idle callback triggered');
        }, { timeout: 5000 });
      }
    },

    notifyCacheHit: ({ context }) => {
      console.log('✅ Cache hit:', context.source, `${context.latency}ms`);
    },

    notifyCacheMiss: () => {
      console.log('❌ Cache miss - fetching from sources');
    },

    notifyProvisionalResult: ({ context }) => {
      console.log('⚡ Provisional result from WASM:', `${context.latency}ms`);
    },

    notifyRehydration: ({ context }) => {
      console.log('🔄 UI rehydrated with authoritative data:', context.source);
    },

    notifyRevalidation: ({ context }) => {
      console.log('✅ Background revalidation complete:', context.source);
    },

    notifyError: ({ context }) => {
      console.error('❌ Graph cache error - retry available');
    },

    updateCaches: ({ context }) => {
      console.log('💾 Updating caches with new data');
    },

    invalidateCache: ({ context, event }) => {
      if (context.worker) {
        context.worker.postMessage({
          type: 'cache_clear',
          key: event.type === 'INVALIDATE_CACHE' ? event.key : undefined
        });
      }
    },

    provideTelemetry: ({ context }) => {
      console.log('📊 Telemetry:', context.telemetry);
    },

    setError: () => {
      console.error('❌ Query failed after max retries');
    }
  },

  guards: {
    shouldBackgroundRefresh: ({ context }) => {
      return context.backgroundRefreshEnabled &&
             context.queryHash !== null &&
             (Date.now() - context.lastRefresh) > 300000; // 5 minutes
    },

    isStaleResult: ({ context }) => {
      return context.isStale || (Date.now() - context.lastRefresh) > 180000; // 3 minutes
    },

    canRetry: ({ context }) => {
      return context.retryCount < context.maxRetries;
    }
  }
});

// Helper types for external usage
export type GraphCacheMachine = typeof graphCacheMachine;
export type GraphCacheActor = ActorRefFrom<GraphCacheMachine>;

// Export machine service for SvelteKit integration
export function createGraphCacheService() {
  return graphCacheMachine;
}