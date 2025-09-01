/**
 * XState Idle Detection Machine with Background Processing
 * Triggers WASM graph cache hydration and hot query preloading
 * Optimizes for user experience during idle periods
 */

import { createMachine, interpret, assign } from 'xstate';
import { unifiedServiceRegistry } from '$lib/services/unifiedServiceRegistry';
import { EventEmitter } from "events";

export interface IdleContext {
  lastActivity: Date;
  idleCount: number;
  backgroundTasks: string[];
  wasmEngineLoaded: boolean;
  hotQueriesCached: number;
}

export type IdleEvent = 
  | { type: 'ACTIVITY'; timestamp: number }
  | { type: 'MOUSEMOVE' }
  | { type: 'KEYDOWN' }
  | { type: 'SCROLL' }
  | { type: 'CLICK' }
  | { type: 'IDLE_DETECTED' }
  | { type: 'BACKGROUND_COMPLETE'; taskName: string }
  | { type: 'WASM_ENGINE_LOADED' }
  | { type: 'CACHE_HYDRATED'; queryCount: number };

export const idleDetectionMachine = createMachine<IdleContext, IdleEvent>({
  id: 'idleDetection',
  initial: 'active',
  context: {
    lastActivity: new Date(),
    idleCount: 0,
    backgroundTasks: [],
    wasmEngineLoaded: false,
    hotQueriesCached: 0
  },
  states: {
    active: {
      entry: assign({
        lastActivity: () => new Date()
      }),
      on: {
        ACTIVITY: {
          target: 'active',
          actions: assign({
            lastActivity: ({ event }) => new Date(event.timestamp)
          })
        },
        MOUSEMOVE: 'active',
        KEYDOWN: 'active',
        SCROLL: 'active',
        CLICK: 'active'
      },
      after: {
        // 5 minutes of inactivity triggers idle state
        300000: 'idle'
      }
    },
    idle: {
      entry: [
        assign({
          idleCount: ({ context }) => context.idleCount + 1
        }),
        'startBackgroundProcessing'
      ],
      on: {
        ACTIVITY: 'active',
        MOUSEMOVE: 'active',
        KEYDOWN: 'active',
        SCROLL: 'active',
        CLICK: 'active',
        BACKGROUND_COMPLETE: {
          actions: assign({
            backgroundTasks: ({ context, event }) => 
              context.backgroundTasks.filter(task => task !== event.taskName)
          })
        },
        WASM_ENGINE_LOADED: {
          actions: assign({
            wasmEngineLoaded: true
          })
        },
        CACHE_HYDRATED: {
          actions: assign({
            hotQueriesCached: ({ event }) => event.queryCount
          })
        }
      },
      states: {
        initial: 'checkingServices',
        checkingServices: {
          invoke: {
            src: 'checkServices',
            onDone: 'loadingWasmEngine',
            onError: 'error'
          }
        },
        loadingWasmEngine: {
          invoke: {
            src: 'loadWasmEngine',
            onDone: {
              target: 'hydratingCache',
              actions: 'wasmEngineLoaded'
            },
            onError: 'hydratingCache' // Continue even if WASM fails
          }
        },
        hydratingCache: {
          invoke: {
            src: 'hydrateGraphCache',
            onDone: 'backgroundComplete',
            onError: 'backgroundComplete' // Don't block on cache errors
          }
        },
        backgroundComplete: {
          type: 'final'
        },
        error: {
          // Graceful degradation - don't block user experience
          after: {
            5000: 'backgroundComplete'
          }
        }
      }
    }
  }
}, {
  actions: {
    startBackgroundProcessing: (context) => {
      console.log('🔄 Starting background processing (idle detected)');
      context.backgroundTasks = ['checkServices', 'loadWasmEngine', 'hydrateCache'];
    },
    wasmEngineLoaded: () => {
      console.log('✅ WASM Graph Engine loaded during idle time');
    }
  },
  services: {
    checkServices: async () => {
      console.log('🔍 Checking service health during idle...');
      const status = await unifiedServiceRegistry.getSystemStatus(false); // Force fresh check
      return status;
    },
    
    loadWasmEngine: async () => {
      console.log('📦 Loading WASM Graph Engine...');
      
      // Check if already loaded
      if (globalThis.__WASM_GRAPH_ENGINE__) {
        return { already_loaded: true };
      }
      
      try {
        // This will be implemented with the TinyGo WASM module
        // For now, simulate loading
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Mock WASM engine initialization
        globalThis.__WASM_GRAPH_ENGINE__ = {
          loaded: true,
          version: '1.0.0',
          queryCache: new Map(),
          
          // Mock methods for graph operations
          executeQuery: async (query: string) => {
            return { result: 'mocked', query };
          },
          
          cacheQuery: async (query: string, result: any) => {
            globalThis.__WASM_GRAPH_ENGINE__.queryCache.set(query, result);
          },
          
          getStats: () => ({
            queriesCached: globalThis.__WASM_GRAPH_ENGINE__.queryCache.size,
            memoryUsage: '2MB',
            uptime: Date.now() - (globalThis.__WASM_ENGINE_START_TIME__ || Date.now())
          })
        };
        
        globalThis.__WASM_ENGINE_START_TIME__ = Date.now();
        
        console.log('✅ WASM Graph Engine initialized');
        return { loaded: true };
      } catch (error) {
        console.warn('⚠️ WASM Graph Engine loading failed:', error);
        throw error;
      }
    },
    
    hydrateGraphCache: async () => {
      console.log('💧 Hydrating graph cache with hot queries...');
      
      try {
        // Get hot queries from service registry
        const hotQueries = await unifiedServiceRegistry.getHotQueries(10);
        
        // If WASM engine is available, pre-cache common queries
        if (globalThis.__WASM_GRAPH_ENGINE__) {
          const commonQueries = [
            'MATCH (case:Case) RETURN case LIMIT 10',
            'MATCH (evidence:Evidence)-[:BELONGS_TO]->(case:Case) RETURN evidence, case LIMIT 5',
            'MATCH (person:Person)-[:INVOLVED_IN]->(case:Case) RETURN person, case LIMIT 5'
          ];
          
          for (const query of commonQueries) {
            try {
              // Check if already cached in service registry
              const cached = await unifiedServiceRegistry.getGraphQuery(query);
              if (!cached) {
                // Mock query execution - in production this would hit remote Neo4j
                const mockResult = {
                  nodes: Math.floor(Math.random() * 10) + 1,
                  relationships: Math.floor(Math.random() * 5),
                  query,
                  timestamp: new Date()
                };
                
                // Cache in both service registry and WASM engine
                await unifiedServiceRegistry.cacheGraphQuery(query, mockResult);
                await globalThis.__WASM_GRAPH_ENGINE__.cacheQuery(query, mockResult);
              }
            } catch (error) {
              console.warn(`Failed to cache query: ${query}`, error);
            }
          }
        }
        
        const totalCached = hotQueries.length + (globalThis.__WASM_GRAPH_ENGINE__?.queryCache?.size || 0);
        console.log(`✅ Graph cache hydrated with ${totalCached} queries`);
        
        return { queriesCached: totalCached };
      } catch (error) {
        console.warn('⚠️ Cache hydration failed:', error);
        throw error;
      }
    }
  }
});

// Create service for browser usage
export class IdleDetectionService {
  private service: any;
  private listeners: (() => void)[] = [];

  constructor() {
    this.service = interpret(idleDetectionMachine);
  }

  start() {
    if (typeof window === 'undefined') return;
    
    this.service.start();
    console.log('🎯 Idle detection service started');

    // Add event listeners for activity detection
    const events = ['mousemove', 'keydown', 'scroll', 'click', 'touchstart'];
    const activityHandler = (event: Event) => {
      this.service.send({
        type: 'ACTIVITY',
        timestamp: Date.now()
      });
    };

    events.forEach(eventType => {
      window.addEventListener(eventType, activityHandler, { passive: true });
      this.listeners.push(() => window.removeEventListener(eventType, activityHandler));
    });

    // Listen for state changes
    this.service.onTransition((state: any) => {
      if (state.changed) {
        console.log(`🔄 Idle state: ${state.value}`, state.context);
        
        // Emit custom events for other parts of the app to listen to
        window.dispatchEvent(new CustomEvent('idle-state-change', {
          detail: {
            state: state.value,
            context: state.context,
            isIdle: state.matches('idle')
          }
        }));
      }
    });

    return this;
  }

  stop() {
    this.service.stop();
    this.listeners.forEach(cleanup => cleanup());
    this.listeners = [];
    console.log('🛑 Idle detection service stopped');
  }

  getState() {
    return this.service.state;
  }

  isIdle(): boolean {
    return this.service.state.matches('idle');
  }

  getContext(): IdleContext {
    return this.service.state.context;
  }

  // Manual trigger for testing
  triggerIdle() {
    this.service.send('IDLE_DETECTED');
  }

  triggerActivity() {
    this.service.send('ACTIVITY', { timestamp: Date.now() });
  }
}

// Export singleton for easy usage
export const idleDetectionService = new IdleDetectionService();
;
// Browser-only auto-start (with cleanup)
if (typeof window !== 'undefined') {
  // Auto-start when module loads
  idleDetectionService.start();
  
  // Auto-cleanup on page unload
  window.addEventListener('beforeunload', () => {
    idleDetectionService.stop();
  });
}