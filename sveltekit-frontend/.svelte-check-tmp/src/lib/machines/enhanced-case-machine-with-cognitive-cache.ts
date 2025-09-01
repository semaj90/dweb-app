/**
 * Enhanced Case Management Machine with Cognitive Cache Integration
 * Demonstrates complete integration of XState, Database, and Cognitive Caching
 */

import { createMachine, assign, fromPromise, type StateFrom } from 'xstate';
import type { CaseData, EvidenceData } from '../mcp/cases.mcp';
import { loadCase, createCase, updateCase } from '../mcp/cases.mcp';
import { cognitiveCacheManager } from '../services/cognitive-cache-integration';
import { initializeDatabase, getDatabaseHealth } from '../database';

// Enhanced Context with Cognitive Cache Integration
export interface EnhancedCaseManagementContext {
  // Core case data
  currentCase: CaseData | null;
  cases: CaseData[];
  evidence: EvidenceData[];
  
  // Search and filtering
  searchQuery: string;
  searchResults: CaseData[];
  filters: {
    status?: string;
    priority?: string;
    dateRange?: { from: Date; to: Date };
  };
  
  // UI and interaction state
  selectedCaseId: string | null;
  isLoading: boolean;
  error: string | null;
  
  // Pagination
  pagination: {
    page: number;
    limit: number;
    totalCount: number;
  };
  
  // User context for cognitive caching
  userId: string;
  workflowStep: string;
  
  // Cognitive cache metrics
  cacheMetrics: {
    hitRate: number;
    averageLatency: number;
    cognitiveAccuracy: number;
  };
  
  // Database health
  databaseHealth: {
    postgres: boolean;
    qdrant: boolean;
    overall: 'healthy' | 'degraded' | 'unhealthy';
  };
}

// Enhanced Events with Cache Integration
type EnhancedCaseManagementEvent =
  | { type: 'INITIALIZE_SYSTEM'; userId: string }
  | { type: 'LOAD_CASE'; caseId: string; enableCache?: boolean }
  | { type: 'LOAD_CASE_WITH_PREDICTION'; caseId: string; predictNext?: boolean }
  | { type: 'CREATE_CASE'; caseData: Omit<CaseData, 'id' | 'createdAt' | 'updatedAt'> }
  | { type: 'UPDATE_CASE'; caseId: string; updates: Partial<CaseData> }
  | { type: 'DELETE_CASE'; caseId: string }
  | { type: 'SEARCH_CASES_COGNITIVE'; query: string; useML?: boolean }
  | { type: 'PRELOAD_RELATED_CASES'; caseId: string }
  | { type: 'REFRESH_CACHE_METRICS' }
  | { type: 'CHECK_DATABASE_HEALTH' }
  | { type: 'OPTIMIZE_WORKFLOW'; workflowData: any };

/**
 * Smart Cache Service - Integrates Database with Cognitive Cache
 */
const smartCacheService = {
  async loadCaseWithCache(caseId: string, context: EnhancedCaseManagementContext) {
    const cacheRequest = {
      key: `case_${caseId}`,
      type: 'legal-data' as const,
      context: {
        userId: context.userId,
        workflowStep: context.workflowStep,
        documentType: 'case-data',
        priority: 'medium' as const,
        semanticTags: ['case-management', 'legal-ai']
      },
      options: {
        enablePredictive: true,
        enablePhysics: true
      }
    };

    // Try cognitive cache first
    const cachedResult = await cognitiveCacheManager.get(cacheRequest);
    if (cachedResult && cachedResult.confidence > 0.8) {
      return {
        case: cachedResult.data,
        source: 'cache',
        processingTime: cachedResult.processingTime,
        predictions: cachedResult.predictions
      };
    }

    // Fallback to database
    const caseData = await loadCase(caseId);
    if (caseData) {
      // Cache the result for future use
      await cognitiveCacheManager.set(cacheRequest, caseData, {
        distributeAcrossCaches: true,
        cognitiveValue: 0.7
      });
    }

    return {
      case: caseData,
      source: 'database',
      processingTime: 0,
      predictions: null
    };
  },

  async searchCasesWithML(query: string, context: EnhancedCaseManagementContext) {
    const cacheKey = `search_${query.replace(/\s+/g, '_')}`;
    const cacheRequest = {
      key: cacheKey,
      type: 'legal-data' as const,
      context: {
        userId: context.userId,
        workflowStep: 'search',
        priority: 'high' as const,
        semanticTags: ['search', 'legal-cases', ...query.split(' ').slice(0, 3)]
      }
    };

    // Check if we have cached search results
    const cachedSearch = await cognitiveCacheManager.get(cacheRequest);
    if (cachedSearch && cachedSearch.confidence > 0.6) {
      return cachedSearch.data;
    }

    // Perform new search (this would integrate with your search service)
    const searchResults = []; // Placeholder for actual search implementation
    
    // Cache the results
    await cognitiveCacheManager.set(cacheRequest, searchResults);
    return searchResults;
  }
};

/**
 * Database Health Service
 */
const databaseHealthService = fromPromise(async () => {
  const health = await getDatabaseHealth();
  return health;
});

/**
 * Cache Metrics Service
 */
const cacheMetricsService = fromPromise(async () => {
  const metrics = cognitiveCacheManager.getMetrics();
  return {
    hitRate: metrics.totalRequests > 0 ? (metrics.cacheHits / metrics.totalRequests) : 0,
    averageLatency: metrics.averageLatency,
    cognitiveAccuracy: metrics.cognitiveAccuracy
  };
});

/**
 * Enhanced Case Management Machine with Cognitive Cache
 */
export const enhancedCaseManagementMachine = createMachine({
  id: 'enhancedCaseManagement',
  initial: 'initializing',
  context: {
    currentCase: null,
    cases: [],
    evidence: [],
    searchQuery: '',
    searchResults: [],
    filters: {},
    selectedCaseId: null,
    isLoading: false,
    error: null,
    pagination: {
      page: 1,
      limit: 20,
      totalCount: 0
    },
    userId: '',
    workflowStep: 'init',
    cacheMetrics: {
      hitRate: 0,
      averageLatency: 0,
      cognitiveAccuracy: 0
    },
    databaseHealth: {
      postgres: false,
      qdrant: false,
      overall: 'unhealthy' as const
    }
  },
  types: {} as {
    context: EnhancedCaseManagementContext;
    events: EnhancedCaseManagementEvent;
  },
  states: {
    initializing: {
      invoke: [
        {
          src: fromPromise(async () => {
            const health = await initializeDatabase();
            return health;
          }),
          onDone: {
            target: 'idle',
            actions: assign({
              databaseHealth: ({ event }) => ({
                postgres: event.output.postgres,
                qdrant: event.output.qdrant,
                overall: event.output.postgres && event.output.qdrant ? 'healthy' as const : 'degraded' as const
              })
            })
          },
          onError: {
            target: 'error',
            actions: assign({
              error: ({ event }) => `Database initialization failed: ${event.error}`
            })
          }
        }
      ],
      on: {
        INITIALIZE_SYSTEM: {
          actions: assign({
            userId: ({ event }) => event.userId,
            workflowStep: () => 'initializing'
          })
        }
      }
    },

    idle: {
      entry: assign({ isLoading: () => false }),
      on: {
        LOAD_CASE: {
          target: 'loadingCase',
          actions: assign({
            selectedCaseId: ({ event }) => event.caseId,
            workflowStep: () => 'loading-case'
          })
        },
        LOAD_CASE_WITH_PREDICTION: {
          target: 'loadingCaseWithPrediction',
          actions: assign({
            selectedCaseId: ({ event }) => event.caseId,
            workflowStep: () => 'predictive-loading'
          })
        },
        CREATE_CASE: {
          target: 'creatingCase',
          actions: assign({
            workflowStep: () => 'creating-case'
          })
        },
        SEARCH_CASES_COGNITIVE: {
          target: 'searchingWithCognition',
          actions: assign({
            searchQuery: ({ event }) => event.query,
            workflowStep: () => 'cognitive-search'
          })
        },
        CHECK_DATABASE_HEALTH: {
          target: 'checkingHealth'
        },
        REFRESH_CACHE_METRICS: {
          target: 'refreshingMetrics'
        }
      }
    },

    loadingCase: {
      entry: assign({ isLoading: () => true }),
      invoke: {
        src: fromPromise(async ({ input }) => {
          const { caseId, context } = input as { caseId: string; context: EnhancedCaseManagementContext };
          return await smartCacheService.loadCaseWithCache(caseId, context);
        }),
        input: ({ context, event }) => ({
          caseId: (event as any).caseId,
          context
        }),
        onDone: {
          target: 'idle',
          actions: assign({
            currentCase: ({ event }) => event.output.case,
            isLoading: () => false,
            error: () => null
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => `Failed to load case: ${event.error}`,
            isLoading: () => false
          })
        }
      }
    },

    loadingCaseWithPrediction: {
      entry: assign({ isLoading: () => true }),
      invoke: {
        src: fromPromise(async ({ input }) => {
          const { caseId, context } = input as { caseId: string; context: EnhancedCaseManagementContext };
          const result = await smartCacheService.loadCaseWithCache(caseId, context);
          
          // If we have predictions, preload related cases
          if (result.predictions?.relatedKeys) {
            const preloadPromises = result.predictions.relatedKeys.slice(0, 3).map(async (key: string) => {
              const relatedCaseId = key.replace('case_', '');
              return await smartCacheService.loadCaseWithCache(relatedCaseId, context);
            });
            
            await Promise.all(preloadPromises);
          }
          
          return result;
        }),
        input: ({ context, event }) => ({
          caseId: (event as any).caseId,
          context
        }),
        onDone: {
          target: 'idle',
          actions: assign({
            currentCase: ({ event }) => event.output.case,
            isLoading: () => false,
            error: () => null
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => `Failed to load case with prediction: ${event.error}`,
            isLoading: () => false
          })
        }
      }
    },

    searchingWithCognition: {
      entry: assign({ isLoading: () => true }),
      invoke: {
        src: fromPromise(async ({ input }) => {
          const { query, context } = input as { query: string; context: EnhancedCaseManagementContext };
          return await smartCacheService.searchCasesWithML(query, context);
        }),
        input: ({ context, event }) => ({
          query: (event as any).query,
          context
        }),
        onDone: {
          target: 'idle',
          actions: assign({
            searchResults: ({ event }) => event.output,
            isLoading: () => false,
            error: () => null
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => `Cognitive search failed: ${event.error}`,
            isLoading: () => false
          })
        }
      }
    },

    creatingCase: {
      entry: assign({ isLoading: () => true }),
      invoke: {
        src: fromPromise(async ({ input }) => {
          const { caseData, context } = input as { caseData: any; context: EnhancedCaseManagementContext };
          
          // Create case in database
          const newCase = await createCase({
            ...caseData,
            createdBy: context.userId
          });
          
          // Cache the newly created case
          await cognitiveCacheManager.set({
            key: `case_${newCase.caseId}`,
            type: 'legal-data',
            context: {
              userId: context.userId,
              workflowStep: 'case-created',
              documentType: 'case-data',
              priority: 'high',
              semanticTags: ['new-case', 'legal-data']
            }
          }, newCase);
          
          return newCase;
        }),
        input: ({ context, event }) => ({
          caseData: (event as any).caseData,
          context
        }),
        onDone: {
          target: 'idle',
          actions: assign({
            currentCase: ({ event }) => event.output,
            cases: ({ context, event }) => [event.output, ...context.cases],
            isLoading: () => false,
            error: () => null
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => `Failed to create case: ${event.error}`,
            isLoading: () => false
          })
        }
      }
    },

    checkingHealth: {
      invoke: {
        src: databaseHealthService,
        onDone: {
          target: 'idle',
          actions: assign({
            databaseHealth: ({ event }) => event.output
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            databaseHealth: () => ({
              postgres: false,
              qdrant: false,
              overall: 'unhealthy' as const
            })
          })
        }
      }
    },

    refreshingMetrics: {
      invoke: {
        src: cacheMetricsService,
        onDone: {
          target: 'idle',
          actions: assign({
            cacheMetrics: ({ event }) => event.output
          })
        },
        onError: {
          target: 'idle'
        }
      }
    },

    error: {
      entry: assign({ isLoading: () => false }),
      on: {
        '*': {
          target: 'idle',
          actions: assign({ error: () => null })
        }
      }
    }
  }
});

// Type for the machine
export type EnhancedCaseManagementMachine = StateFrom<typeof enhancedCaseManagementMachine>;

