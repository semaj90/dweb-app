// Search State Machine - XState v5 compatible
// Manages legal document and case search with history and analytics

import { createMachine, assign, fromPromise } from 'xstate';

export interface SearchContext {
  query: string;
  filters: {
    caseStatus?: string[];
    evidenceType?: string[];
    dateRange?: {
      from?: string;
      to?: string;
    };
    priority?: string[];
    tags?: string[];
  };
  results: any[];
  searchHistory: string[];
  analytics: {
    totalResults: number;
    searchTime: number;
    relevanceScore: number;
  };
  validationErrors: Record<string, string[]>;
  error: string | null;
  isLoading: boolean;
}

export const searchMachine = createMachine({
  id: 'search',
  initial: 'idle',
  types: {
    context: {} as SearchContext,
    events: {} as 
      | { type: 'SEARCH'; data: any }
      | { type: 'UPDATE_QUERY'; query: string }
      | { type: 'UPDATE_FILTERS'; filters: any }
      | { type: 'CLEAR_RESULTS' }
      | { type: 'LOAD_HISTORY' }
      | { type: 'SAVE_SEARCH' }
      | { type: 'RETRY' }
  },
  context: {
    query: '',
    filters: {},
    results: [],
    searchHistory: [],
    analytics: {
      totalResults: 0,
      searchTime: 0,
      relevanceScore: 0
    },
    validationErrors: {},
    error: null,
    isLoading: false
  },
  states: {
    idle: {
      entry: assign({ isLoading: false }),
      on: {
        UPDATE_QUERY: {
          actions: assign({
            query: ({ event }) => event.query,
            error: null
          })
        },
        UPDATE_FILTERS: {
          actions: assign({
            filters: ({ context, event }) => ({
              ...context.filters,
              ...event.filters
            })
          })
        },
        SEARCH: 'validating',
        LOAD_HISTORY: 'loadingHistory'
      }
    },
    loadingHistory: {
      invoke: {
        id: 'loadSearchHistory',
        src: fromPromise(async () => {
          // Load search history from localStorage or API
          try {
            const stored = localStorage.getItem('legal-ai:search-history');
            return stored ? JSON.parse(stored) : [];
          } catch {
            return [];
          }
        }),
        onDone: {
          target: 'idle',
          actions: assign({
            searchHistory: ({ event }) => event.output || []
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            error: 'Failed to load search history'
          })
        }
      }
    },
    validating: {
      invoke: {
        id: 'validateSearch',
        src: fromPromise(async ({ input }: { input: SearchContext }) => {
          const errors: Record<string, string[]> = {};
          
          if (!input.query?.trim() && !Object.keys(input.filters).length) {
            errors.query = ['Please enter a search query or select filters'];
          }
          
          if (input.query && input.query.length > 200) {
            errors.query = ['Search query too long (max 200 characters)'];
          }
          
          // Validate date range
          if (input.filters.dateRange?.from && input.filters.dateRange?.to) {
            const fromDate = new Date(input.filters.dateRange.from);
            const toDate = new Date(input.filters.dateRange.to);
            
            if (fromDate > toDate) {
              errors.dateRange = ['Start date must be before end date'];
            }
          }
          
          if (Object.keys(errors).length > 0) {
            throw { validationErrors: errors };
          }
          
          return { query: input.query, filters: input.filters };
        }),
        input: ({ context }) => context,
        onDone: {
          target: 'searching',
          actions: assign({
            validationErrors: {},
            error: null
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            validationErrors: ({ event }) => (event as any).error?.validationErrors || {},
            error: 'Search validation failed'
          })
        }
      }
    },
    searching: {
      entry: assign({ isLoading: true }),
      invoke: {
        id: 'performSearch',
        src: fromPromise(async ({ input }: { input: SearchContext }) => {
          const startTime = Date.now();
          
          // Build search parameters
          const searchParams = new URLSearchParams();
          if (input.query) searchParams.append('query', input.query);
          
          // Add filters
          Object.entries(input.filters).forEach(([key, value]) => {
            if (Array.isArray(value)) {
              searchParams.append(key, value.join(','));
            } else if (typeof value === 'object' && value !== null) {
              // Handle date range
              if (key === 'dateRange') {
                if ((value as any).from) searchParams.append('dateStart', (value as any).from);
                if ((value as any).to) searchParams.append('dateEnd', (value as any).to);
              }
            } else if (value) {
              searchParams.append(key, String(value));
            }
          });
          
          // Perform search API call
          const response = await fetch(`/api/search/legal?${searchParams}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json'
            }
          });
          
          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Search failed: HTTP ${response.status}`);
          }
          
          const data = await response.json();
          const searchTime = Date.now() - startTime;
          
          return {
            results: data.results || [],
            analytics: {
              totalResults: data.total || 0,
              searchTime,
              relevanceScore: data.averageRelevance || 0
            }
          };
        }),
        input: ({ context }) => context,
        onDone: {
          target: 'results',
          actions: [
            assign({
              results: ({ event }) => event.output.results,
              analytics: ({ event }) => event.output.analytics,
              error: null,
              isLoading: false
            }),
            // Save to search history
            ({ context }) => {
              if (context.query) {
                const history = [...new Set([context.query, ...context.searchHistory])].slice(0, 10);
                try {
                  localStorage.setItem('legal-ai:search-history', JSON.stringify(history));
                } catch (e: any) {
                  console.warn('Failed to save search history:', e);
                }
              }
            }
          ]
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => (event as any).error?.message || 'Search failed',
            isLoading: false
          })
        }
      }
    },
    results: {
      entry: assign({ isLoading: false }),
      on: {
        SEARCH: 'validating',
        UPDATE_QUERY: {
          target: 'idle',
          actions: assign({
            query: ({ event }) => event.query
          })
        },
        UPDATE_FILTERS: {
          target: 'idle',
          actions: assign({
            filters: ({ context, event }) => ({
              ...context.filters,
              ...event.filters
            })
          })
        },
        CLEAR_RESULTS: {
          target: 'idle',
          actions: assign({
            results: [],
            query: '',
            filters: {},
            analytics: {
              totalResults: 0,
              searchTime: 0,
              relevanceScore: 0
            }
          })
        }
      }
    },
    error: {
      on: {
        RETRY: 'searching',
        SEARCH: 'validating',
        CLEAR_RESULTS: {
          target: 'idle',
          actions: assign({
            error: null,
            results: []
          })
        }
      }
    }
  }
});

export default searchMachine;