// Search State Machine with XState
// Manages document search, filtering, and semantic search capabilities

import { createMachine, assign, type InterpreterFrom } from 'xstate';
import { multiProtocolRouter, routerHelpers } from '../services/multi-protocol-router';

// Types for search management
export interface SearchResult {
  id: string;
  title: string;
  content: string;
  snippet: string;
  score: number;
  type: 'document' | 'case' | 'evidence' | 'legal_reference';
  metadata: {
    caseId?: string;
    documentType?: string;
    createdAt?: number;
    tags?: string[];
    source?: string;
    citations?: number;
    relevance?: number;
  };
  highlights?: string[];
}

export interface SearchFilter {
  dateRange?: {
    start: number;
    end: number;
  };
  documentTypes?: string[];
  caseIds?: string[];
  tags?: string[];
  scoreThreshold?: number;
  contentLength?: {
    min?: number;
    max?: number;
  };
}

export interface SearchContext {
  query: string;
  results: SearchResult[];
  totalResults: number;
  currentPage: number;
  pageSize: number;
  isSearching: boolean;
  searchType: 'text' | 'semantic' | 'hybrid' | 'legal';
  filters: SearchFilter;
  suggestions: string[];
  recentQueries: string[];
  selectedResults: string[];
  sortBy: 'relevance' | 'date' | 'score' | 'title';
  sortOrder: 'asc' | 'desc';
  searchHistory: Array<{
    query: string;
    timestamp: number;
    resultCount: number;
    searchType: string;
  }>;
  performance: {
    lastSearchTime: number;
    averageSearchTime: number;
    totalSearches: number;
    cacheHits: number;
    protocolUsage: Record<string, number>;
  };
  settings: {
    maxResults: number;
    enableSuggestions: boolean;
    enableHighlighting: boolean;
    enableCaching: boolean;
    preferredProtocol: 'auto' | 'quic' | 'grpc' | 'rest';
    semanticThreshold: number;
  };
}

// Search events
export type SearchEvent =
  | { type: 'UPDATE_QUERY'; query: string }
  | { type: 'SEARCH' }
  | { type: 'CLEAR_SEARCH' }
  | { type: 'APPLY_FILTERS'; filters: Partial<SearchFilter> }
  | { type: 'CLEAR_FILTERS' }
  | { type: 'CHANGE_SEARCH_TYPE'; searchType: SearchContext['searchType'] }
  | { type: 'LOAD_MORE' }
  | { type: 'SORT_RESULTS'; sortBy: SearchContext['sortBy']; order?: SearchContext['sortOrder'] }
  | { type: 'SELECT_RESULT'; resultId: string }
  | { type: 'DESELECT_RESULT'; resultId: string }
  | { type: 'SELECT_ALL_RESULTS' }
  | { type: 'CLEAR_SELECTION' }
  | { type: 'EXPORT_RESULTS'; format: 'json' | 'csv' | 'pdf' }
  | { type: 'SAVE_QUERY'; name?: string }
  | { type: 'LOAD_SAVED_QUERY'; query: string }
  | { type: 'UPDATE_SETTINGS'; settings: Partial<SearchContext['settings']> }
  | { type: 'SEARCH_COMPLETE'; results: unknown }
  | { type: 'SEARCH_ERROR'; error: string };

// Default context
const defaultContext: SearchContext = {
  query: '',
  results: [],
  totalResults: 0,
  currentPage: 1,
  pageSize: 20,
  isSearching: false,
  searchType: 'hybrid',
  filters: {},
  suggestions: [],
  recentQueries: [],
  selectedResults: [],
  sortBy: 'relevance',
  sortOrder: 'desc',
  searchHistory: [],
  performance: {
    lastSearchTime: 0,
    averageSearchTime: 0,
    totalSearches: 0,
    cacheHits: 0,
    protocolUsage: { quic: 0, grpc: 0, rest: 0 }
  },
  settings: {
    maxResults: 100,
    enableSuggestions: true,
    enableHighlighting: true,
    enableCaching: true,
    preferredProtocol: 'auto',
    semanticThreshold: 0.7
  }
};

// Services for search machine
const searchServices = {
  performSearch: async (context: SearchContext, event: unknown) => {
    const startTime = performance.now();
    
    try {
      const searchOptions = {
        query: context.query,
        searchType: context.searchType,
        filters: context.filters,
        page: context.currentPage,
        pageSize: context.pageSize,
        sortBy: context.sortBy,
        sortOrder: context.sortOrder,
        protocol: context.settings.preferredProtocol === 'auto' ? undefined : context.settings.preferredProtocol,
        enableHighlighting: context.settings.enableHighlighting,
        semanticThreshold: context.settings.semanticThreshold
      };

      // Route search through multi-protocol system
      const searchResult = await routerHelpers.search(searchOptions);
      
      const processingTime = performance.now() - startTime;

      return {
        results: searchResult.results || [],
        totalResults: searchResult.total || 0,
        suggestions: searchResult.suggestions || [],
        processingTime,
        protocol: searchResult.metadata?.protocol || 'unknown',
        cached: searchResult.cached || false
      };

    } catch (error) {
      // Fallback to local search if multi-protocol fails
      console.warn('Multi-protocol search failed, falling back to local search:', error);
      return await searchServices.performLocalSearch(context);
    }
  },

  performLocalSearch: async (context: SearchContext) => {
    const startTime = performance.now();
    
    try {
      // Implement local search fallback
      const results = await performLocalDocumentSearch(context.query, {
        filters: context.filters,
        searchType: context.searchType,
        pageSize: context.pageSize,
        page: context.currentPage
      });

      const processingTime = performance.now() - startTime;

      return {
        results: results.documents || [],
        totalResults: results.total || 0,
        suggestions: [],
        processingTime,
        protocol: 'local',
        cached: false
      };

    } catch (error) {
      throw new Error(`Local search failed: ${error.message}`);
    }
  },

  generateSuggestions: async (context: SearchContext) => {
    if (!context.settings.enableSuggestions || context.query.length < 3) {
      return [];
    }

    try {
      // Generate search suggestions based on query
      const suggestions = await routerHelpers.getSuggestions(context.query, {
        limit: 5,
        includeRecent: true,
        includePopular: true
      });

      return suggestions || [];
    } catch (error) {
      console.warn('Suggestion generation failed:', error);
      return generateLocalSuggestions(context.query, context.recentQueries);
    }
  },

  exportResults: async (context: SearchContext, event: unknown) => {
    const { format } = event;
    const selectedResults = context.results.filter(r => 
      context.selectedResults.includes(r.id)
    );

    const exportData = {
      query: context.query,
      searchType: context.searchType,
      filters: context.filters,
      totalResults: context.totalResults,
      results: selectedResults.length > 0 ? selectedResults : context.results,
      exportedAt: new Date().toISOString(),
      format
    };

    switch (format) {
      case 'json':
        return downloadJSON(exportData, `search_results_${Date.now()}.json`);
      case 'csv':
        return downloadCSV(exportData.results, `search_results_${Date.now()}.csv`);
      case 'pdf':
        return downloadPDF(exportData, `search_results_${Date.now()}.pdf`);
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }
};

// Helper functions
async function performLocalDocumentSearch(query: string, options: unknown) {
  // Simulate local search - in practice this would query your local database
  return {
    documents: [],
    total: 0
  };
}

function generateLocalSuggestions(query: string, recentQueries: string[]) {
  // Generate suggestions from recent queries
  return recentQueries
    .filter(q => q.toLowerCase().includes(query.toLowerCase()) && q !== query)
    .slice(0, 5);
}

function downloadJSON(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json'
  });
  downloadBlob(blob, filename);
}

function downloadCSV(results: SearchResult[], filename: string) {
  const headers = ['ID', 'Title', 'Type', 'Score', 'Created At', 'Case ID'];
  const rows = results.map(r => [
    r.id,
    r.title,
    r.type,
    r.score,
    r.metadata.createdAt ? new Date(r.metadata.createdAt).toISOString() : '',
    r.metadata.caseId || ''
  ]);

  const csvContent = [headers, ...rows]
    .map(row => row.map(cell => `"${cell}"`).join(','))
    .join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv' });
  downloadBlob(blob, filename);
}

function downloadPDF(data: unknown, filename: string) {
  // Simplified PDF generation - in practice you'd use a proper PDF library
  const textContent = `
Search Results Export

Query: ${data.query}
Search Type: ${data.searchType}
Total Results: ${data.totalResults}
Exported: ${data.exportedAt}

Results:
${data.results.map((r: SearchResult, i: number) => `
${i + 1}. ${r.title}
   Type: ${r.type}
   Score: ${r.score}
   Content: ${r.snippet}
`).join('\n')}
  `;

  const blob = new Blob([textContent], { type: 'text/plain' });
  downloadBlob(blob, filename.replace('.pdf', '.txt'));
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Search state machine
export const searchMachine = createMachine({
  id: 'search',
  initial: 'idle',
  context: defaultContext,
  states: {
    idle: {
      entry: 'loadSearchHistory',
      on: {
        UPDATE_QUERY: {
          actions: ['updateQuery', 'generateSuggestions']
        },
        SEARCH: [
          {
            target: 'searching',
            guard: 'hasQuery'
          },
          {
            actions: 'showEmptyQueryError'
          }
        ],
        LOAD_SAVED_QUERY: {
          actions: 'loadSavedQuery'
        }
      }
    },

    searching: {
      entry: ['recordSearchStart', 'addToRecentQueries'],
      invoke: {
        id: 'performSearch',
        src: 'performSearch',
        onDone: {
          target: 'results',
          actions: [
            'setSearchResults',
            'updatePerformanceMetrics',
            'recordSearchHistory'
          ]
        },
        onError: {
          target: 'error',
          actions: 'setSearchError'
        }
      },
      on: {
        CLEAR_SEARCH: {
          target: 'idle',
          actions: 'clearSearch'
        }
      }
    },

    results: {
      entry: 'setResultsState',
      on: {
        UPDATE_QUERY: {
          actions: ['updateQuery', 'generateSuggestions']
        },
        SEARCH: {
          target: 'searching',
          guard: 'hasQuery'
        },
        CLEAR_SEARCH: {
          target: 'idle',
          actions: 'clearSearch'
        },
        APPLY_FILTERS: {
          target: 'searching',
          actions: 'applyFilters'
        },
        CLEAR_FILTERS: {
          target: 'searching',
          actions: 'clearFilters'
        },
        CHANGE_SEARCH_TYPE: {
          target: 'searching',
          actions: 'changeSearchType'
        },
        LOAD_MORE: {
          target: 'loading_more',
          guard: 'hasMoreResults'
        },
        SORT_RESULTS: {
          actions: 'sortResults'
        },
        SELECT_RESULT: {
          actions: 'selectResult'
        },
        DESELECT_RESULT: {
          actions: 'deselectResult'
        },
        SELECT_ALL_RESULTS: {
          actions: 'selectAllResults'
        },
        CLEAR_SELECTION: {
          actions: 'clearSelection'
        },
        EXPORT_RESULTS: {
          target: 'exporting'
        },
        SAVE_QUERY: {
          actions: 'saveQuery'
        }
      }
    },

    loading_more: {
      entry: 'incrementPage',
      invoke: {
        id: 'loadMore',
        src: 'performSearch',
        onDone: {
          target: 'results',
          actions: 'appendResults'
        },
        onError: {
          target: 'results',
          actions: 'handleLoadMoreError'
        }
      }
    },

    exporting: {
      invoke: {
        id: 'exportResults',
        src: 'exportResults',
        onDone: {
          target: 'results',
          actions: 'handleExportComplete'
        },
        onError: {
          target: 'results',
          actions: 'handleExportError'
        }
      }
    },

    error: {
      on: {
        SEARCH: {
          target: 'searching',
          guard: 'hasQuery'
        },
        CLEAR_SEARCH: {
          target: 'idle',
          actions: 'clearSearch'
        },
        UPDATE_QUERY: {
          target: 'idle',
          actions: 'updateQuery'
        }
      }
    }
  },

  on: {
    UPDATE_SETTINGS: {
      actions: 'updateSettings'
    }
  }
}, {
  services: searchServices,
  guards: {
    hasQuery: (context) => context.query.trim().length > 0,
    hasMoreResults: (context) => {
      const totalPages = Math.ceil(context.totalResults / context.pageSize);
      return context.currentPage < totalPages;
    }
  },
  actions: {
    loadSearchHistory: assign({
      searchHistory: () => {
        try {
          const saved = localStorage.getItem('search_history');
          return saved ? JSON.parse(saved) : [];
        } catch {
          return [];
        }
      },
      recentQueries: () => {
        try {
          const saved = localStorage.getItem('recent_queries');
          return saved ? JSON.parse(saved) : [];
        } catch {
          return [];
        }
      }
    }),

    updateQuery: assign({
      query: (_, event) => event.query,
      currentPage: 1
    }),

    generateSuggestions: assign({
      suggestions: async (context) => {
        return await searchServices.generateSuggestions(context);
      }
    }),

    recordSearchStart: assign({
      isSearching: true,
      performance: (context) => ({
        ...context.performance,
        lastSearchTime: performance.now()
      })
    }),

    addToRecentQueries: assign({
      recentQueries: (context) => {
        const updated = [
          context.query,
          ...context.recentQueries.filter(q => q !== context.query)
        ].slice(0, 10);
        
        localStorage.setItem('recent_queries', JSON.stringify(updated));
        return updated;
      }
    }),

    setSearchResults: assign({
      results: (_, event) => event.data.results,
      totalResults: (_, event) => event.data.totalResults,
      suggestions: (_, event) => event.data.suggestions,
      isSearching: false
    }),

    updatePerformanceMetrics: assign({
      performance: (context, event) => {
        const { processingTime, protocol, cached } = event.data;
        const perf = context.performance;
        
        return {
          lastSearchTime: processingTime,
          averageSearchTime: ((perf.averageSearchTime * perf.totalSearches) + processingTime) / (perf.totalSearches + 1),
          totalSearches: perf.totalSearches + 1,
          cacheHits: perf.cacheHits + (cached ? 1 : 0),
          protocolUsage: {
            ...perf.protocolUsage,
            [protocol]: (perf.protocolUsage[protocol] || 0) + 1
          }
        };
      }
    }),

    recordSearchHistory: assign({
      searchHistory: (context, event) => {
        const entry = {
          query: context.query,
          timestamp: Date.now(),
          resultCount: event.data.totalResults,
          searchType: context.searchType
        };

        const updated = [entry, ...context.searchHistory.slice(0, 49)];
        localStorage.setItem('search_history', JSON.stringify(updated));
        return updated;
      }
    }),

    clearSearch: assign({
      query: '',
      results: [],
      totalResults: 0,
      currentPage: 1,
      selectedResults: [],
      isSearching: false
    }),

    applyFilters: assign({
      filters: (context, event) => ({
        ...context.filters,
        ...event.filters
      }),
      currentPage: 1
    }),

    clearFilters: assign({
      filters: {},
      currentPage: 1
    }),

    changeSearchType: assign({
      searchType: (_, event) => event.searchType,
      currentPage: 1
    }),

    incrementPage: assign({
      currentPage: (context) => context.currentPage + 1
    }),

    appendResults: assign({
      results: (context, event) => [...context.results, ...event.data.results],
      isSearching: false
    }),

    sortResults: assign({
      results: (context, event) => {
        const { sortBy, order = 'desc' } = event;
        const sorted = [...context.results].sort((a, b) => {
          let aVal: unknown, bVal: unknown;
          
          switch (sortBy) {
            case 'relevance':
              aVal = a.score;
              bVal = b.score;
              break;
            case 'date':
              aVal = a.metadata.createdAt || 0;
              bVal = b.metadata.createdAt || 0;
              break;
            case 'title':
              aVal = a.title.toLowerCase();
              bVal = b.title.toLowerCase();
              break;
            default:
              return 0;
          }
          
          if (order === 'asc') {
            return aVal > bVal ? 1 : -1;
          } else {
            return aVal < bVal ? 1 : -1;
          }
        });

        return sorted;
      },
      sortBy: (_, event) => event.sortBy,
      sortOrder: (_, event) => event.order || 'desc'
    }),

    selectResult: assign({
      selectedResults: (context, event) => {
        if (context.selectedResults.includes(event.resultId)) {
          return context.selectedResults;
        }
        return [...context.selectedResults, event.resultId];
      }
    }),

    deselectResult: assign({
      selectedResults: (context, event) => 
        context.selectedResults.filter(id => id !== event.resultId)
    }),

    selectAllResults: assign({
      selectedResults: (context) => context.results.map(r => r.id)
    }),

    clearSelection: assign({
      selectedResults: []
    }),

    saveQuery: (context, event) => {
      const savedQueries = JSON.parse(localStorage.getItem('saved_queries') || '[]');
      const query = {
        name: event.name || context.query,
        query: context.query,
        filters: context.filters,
        searchType: context.searchType,
        savedAt: Date.now()
      };
      
      savedQueries.push(query);
      localStorage.setItem('saved_queries', JSON.stringify(savedQueries));
    },

    loadSavedQuery: assign({
      query: (_, event) => event.query,
      currentPage: 1
    }),

    updateSettings: assign({
      settings: (context, event) => ({
        ...context.settings,
        ...event.settings
      })
    }),

    setSearchError: assign({
      isSearching: false
    }),

    setResultsState: assign({
      isSearching: false
    }),

    showEmptyQueryError: () => {
      console.warn('Cannot search with empty query');
    },

    handleLoadMoreError: (context, event) => {
      console.error('Load more failed:', event.data);
    },

    handleExportComplete: () => {
      console.log('Export completed successfully');
    },

    handleExportError: (context, event) => {
      console.error('Export failed:', event.data);
    }
  }
});

// Type for the search service
export type SearchService = InterpreterFrom<typeof searchMachine>;

// Helper functions for common operations
export const searchActions = {
  updateQuery: (query: string) => ({
    type: 'UPDATE_QUERY' as const,
    query
  }),

  search: () => ({
    type: 'SEARCH' as const
  }),

  clearSearch: () => ({
    type: 'CLEAR_SEARCH' as const
  }),

  applyFilters: (filters: Partial<SearchFilter>) => ({
    type: 'APPLY_FILTERS' as const,
    filters
  }),

  changeSearchType: (searchType: SearchContext['searchType']) => ({
    type: 'CHANGE_SEARCH_TYPE' as const,
    searchType
  }),

  loadMore: () => ({
    type: 'LOAD_MORE' as const
  }),

  sortResults: (sortBy: SearchContext['sortBy'], order?: SearchContext['sortOrder']) => ({
    type: 'SORT_RESULTS' as const,
    sortBy,
    order
  }),

  selectResult: (resultId: string) => ({
    type: 'SELECT_RESULT' as const,
    resultId
  }),

  exportResults: (format: 'json' | 'csv' | 'pdf') => ({
    type: 'EXPORT_RESULTS' as const,
    format
  })
};

// Selectors for derived state
export const searchSelectors = {
  isIdle: (state: unknown) => state.matches('idle'),
  isSearching: (state: unknown) => state.matches('searching') || state.matches('loading_more'),
  hasResults: (context: SearchContext) => context.results.length > 0,
  hasSelection: (context: SearchContext) => context.selectedResults.length > 0,
  canLoadMore: (context: SearchContext) => {
    const totalPages = Math.ceil(context.totalResults / context.pageSize);
    return context.currentPage < totalPages;
  },
  searchStats: (context: SearchContext) => ({
    totalResults: context.totalResults,
    currentPage: context.currentPage,
    totalPages: Math.ceil(context.totalResults / context.pageSize),
    selectedCount: context.selectedResults.length,
    averageScore: context.results.length > 0 
      ? context.results.reduce((sum, r) => sum + r.score, 0) / context.results.length 
      : 0
  })
};