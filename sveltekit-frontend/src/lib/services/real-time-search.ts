// Real-time Legal Search with WebSocket/NATS Integration
// Optimized for Svelte 5 and SvelteKit 2 with bits-ui components

import { writable, derived, type Writable } from 'svelte/store';

// Enhanced RAG Service Configuration
const ENHANCED_RAG_URL = 'http://localhost:8094';
const UPLOAD_SERVICE_URL = 'http://localhost:8093';
const NATS_WS_URL = 'ws://localhost:4222';

// Real-time search state management
interface RealTimeSearchState {
  isConnected: boolean;
  isSearching: boolean;
  currentQuery: string;
  results: SearchResult[];
  suggestions: string[];
  error: string | null;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  searchMetrics: {
    totalQueries: number;
    averageResponseTime: number;
    lastQueryTime: number;
  };
}

interface SearchResult {
  id: string;
  title: string;
  type: 'case' | 'evidence' | 'precedent' | 'statute' | 'criminal' | 'document';
  content: string;
  score: number;
  similarity?: number;
  metadata: {
    date?: string;
    jurisdiction?: string;
    status?: string;
    confidentiality?: string;
    caseId?: string;
    tags?: string[];
  };
  highlights?: string[];
  createdAt?: string;
  realTime?: boolean;
}

// Enhanced Real-time Search Service
export class RealTimeSearchService {
  private ws: WebSocket | null = null;
  private natsConnection: any = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 2000;
  private searchQueue: Array<{ query: string; callback: (results: SearchResult[]) => void }> = [];
  private isProcessingQueue = false;

  // Reactive stores for Svelte 5 compatibility
  public state: Writable<RealTimeSearchState> = writable({
    isConnected: false,
    isSearching: false,
    currentQuery: '',
    results: [],
    suggestions: [],
    error: null,
    connectionStatus: 'disconnected',
    searchMetrics: {
      totalQueries: 0,
      averageResponseTime: 0,
      lastQueryTime: 0
    }
  });

  // Derived stores for enhanced UX
  public isReady = derived(this.state, $state => $state.isConnected && !$state.isSearching);
  public hasResults = derived(this.state, $state => $state.results.length > 0);
  public searchStatus = derived(this.state, $state => {
    if ($state.isSearching) return 'searching';
    if ($state.error) return 'error';
    if ($state.results.length > 0) return 'results';
    return 'idle';
  });

  constructor() {
    this.initializeConnections();
  }

  // Initialize WebSocket and NATS connections
  private async initializeConnections(): Promise<void> {
    try {
      this.state.update(s => ({ ...s, connectionStatus: 'connecting' }));

      // Initialize WebSocket connection to Enhanced RAG service
      await this.connectWebSocket();

      // Initialize NATS connection for real-time messaging
      await this.connectNATS();

      this.state.update(s => ({ 
        ...s, 
        isConnected: true, 
        connectionStatus: 'connected',
        error: null 
      }));

      console.log('🔗 Real-time search connections established');
    } catch (error) {
      console.error('❌ Failed to initialize real-time search:', error);
      this.state.update(s => ({ 
        ...s, 
        error: `Connection failed: ${error}`,
        connectionStatus: 'error'
      }));
      
      // Attempt reconnection
      this.scheduleReconnection();
    }
  }

  // WebSocket connection to Enhanced RAG service
  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Use the WebSocket endpoint from the Enhanced RAG service
        this.ws = new WebSocket(`ws://localhost:8094/ws/legal-search-client`);

        this.ws.onopen = () => {
          console.log('🚀 WebSocket connected to Enhanced RAG service');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleWebSocketMessage(event);
        };

        this.ws.onclose = () => {
          console.log('🔌 WebSocket disconnected');
          this.state.update(s => ({ ...s, isConnected: false, connectionStatus: 'disconnected' }));
          this.scheduleReconnection();
        };

        this.ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          reject(error);
        };

        // Timeout after 5 seconds
        setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket connection timeout'));
          }
        }, 5000);

      } catch (error) {
        reject(error);
      }
    });
  }

  // NATS connection for distributed messaging
  private async connectNATS(): Promise<void> {
    try {
      // Use a lightweight NATS client for browser
      // Since we're in a browser environment, we'll use WebSocket-based NATS
      console.log('📡 Connecting to NATS via WebSocket...');
      
      // For now, we'll focus on WebSocket and add full NATS later
      // This allows immediate testing with the existing Enhanced RAG WebSocket
    } catch (error) {
      console.warn('⚠️ NATS connection failed, using WebSocket only:', error);
    }
  }

  // Handle WebSocket messages from Enhanced RAG service
  private handleWebSocketMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data);

      switch (message.type) {
        case 'search_results':
          this.handleSearchResults(message.data);
          break;

        case 'search_suggestions':
          this.handleSearchSuggestions(message.data);
          break;

        case 'search_progress':
          this.handleSearchProgress(message.data);
          break;

        case 'error':
          this.handleSearchError(message.data);
          break;

        default:
          console.log('📨 Received WebSocket message:', message);
      }
    } catch (error) {
      console.error('❌ Failed to parse WebSocket message:', error);
    }
  }

  // Real-time search with streaming results
  public async performRealTimeSearch(query: string, options: {
    categories?: string[];
    vectorSearch?: boolean;
    streamResults?: boolean;
    includeAI?: boolean;
  } = {}): Promise<SearchResult[]> {
    const startTime = Date.now();

    this.state.update(s => ({ 
      ...s, 
      isSearching: true, 
      currentQuery: query,
      error: null 
    }));

    try {
      // If WebSocket is available, use real-time streaming
      if (this.ws?.readyState === WebSocket.OPEN && options.streamResults !== false) {
        return await this.performStreamingSearch(query, options);
      }

      // Fallback to HTTP API with enhanced error handling
      return await this.performHTTPSearch(query, options);

    } catch (error) {
      console.error('❌ Real-time search failed:', error);
      this.state.update(s => ({ 
        ...s, 
        error: `Search failed: ${error}`,
        isSearching: false 
      }));

      // Return fallback results
      return await this.getFallbackResults(query);
    } finally {
      const responseTime = Date.now() - startTime;
      this.updateSearchMetrics(responseTime);
    }
  }

  // Streaming search via WebSocket
  private async performStreamingSearch(query: string, options: any): Promise<SearchResult[]> {
    return new Promise((resolve, reject) => {
      const searchId = `search_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const results: SearchResult[] = [];
      let searchCompleted = false;

      // Set up message handler for this specific search
      const messageHandler = (event: MessageEvent) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.searchId !== searchId) return;

          switch (message.type) {
            case 'search_result_chunk':
              // Add streaming results as they arrive
              const chunkResults = message.data.results || [];
              results.push(...chunkResults.map((r: any) => ({ ...r, realTime: true })));
              
              // Update store with partial results
              this.state.update(s => ({ ...s, results: [...results] }));
              break;

            case 'search_completed':
              searchCompleted = true;
              this.state.update(s => ({ ...s, isSearching: false, results }));
              resolve(results);
              break;

            case 'search_error':
              reject(new Error(message.data.error));
              break;
          }
        } catch (error) {
          console.error('❌ Error processing streaming search response:', error);
        }
      };

      // Add temporary message handler
      this.ws!.addEventListener('message', messageHandler);

      // Send search request
      this.ws!.send(JSON.stringify({
        type: 'real_time_search',
        searchId,
        query,
        options: {
          categories: options.categories || ['cases', 'evidence', 'documents'],
          vectorSearch: options.vectorSearch !== false,
          includeAI: options.includeAI !== false,
          streamResults: true,
          legalContext: {
            jurisdiction: 'federal',
            practiceAreas: 'all'
          }
        }
      }));

      // Cleanup and timeout handling
      const timeout = setTimeout(() => {
        if (!searchCompleted) {
          this.ws!.removeEventListener('message', messageHandler);
          reject(new Error('Streaming search timeout'));
        }
      }, 30000); // 30 second timeout

      // Clean up on completion
      const cleanup = () => {
        clearTimeout(timeout);
        this.ws!.removeEventListener('message', messageHandler);
      };

      Promise.resolve().then(() => {
        setTimeout(cleanup, 1000); // Clean up after 1 second delay
      });
    });
  }

  // HTTP fallback search
  private async performHTTPSearch(query: string, options: any): Promise<SearchResult[]> {
    const searchParams = new URLSearchParams({
      q: query,
      categories: (options.categories || ['cases', 'evidence', 'documents']).join(','),
      vectorSearch: String(options.vectorSearch !== false),
      aiSuggestions: String(options.includeAI !== false),
      limit: '20'
    });

    // Try multiple endpoints with fallback
    const endpoints = [
      `/api/search/legal?${searchParams}`,
      `http://localhost:8094/search?${searchParams}`,
      `http://localhost:8093/search?${searchParams}`
    ];

    let lastError: Error | null = null;

    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const results = data.results || [];

        this.state.update(s => ({ ...s, results, isSearching: false }));
        return results;

      } catch (error) {
        console.warn(`❌ Endpoint ${endpoint} failed:`, error);
        lastError = error as Error;
        continue;
      }
    }

    throw lastError || new Error('All search endpoints failed');
  }

  // Handle streaming search results
  private handleSearchResults(data: any): void {
    const results = Array.isArray(data) ? data : data.results || [];
    this.state.update(s => ({ ...s, results, isSearching: false }));
  }

  // Handle search suggestions
  private handleSearchSuggestions(data: any): void {
    const suggestions = Array.isArray(data) ? data : data.suggestions || [];
    this.state.update(s => ({ ...s, suggestions }));
  }

  // Handle search progress updates
  private handleSearchProgress(data: any): void {
    console.log('🔄 Search progress:', data);
  }

  // Handle search errors
  private handleSearchError(data: any): void {
    const error = data.error || data.message || 'Unknown search error';
    this.state.update(s => ({ ...s, error, isSearching: false }));
  }

  // Update search metrics
  private updateSearchMetrics(responseTime: number): void {
    this.state.update(s => {
      const newTotalQueries = s.searchMetrics.totalQueries + 1;
      const newAverageTime = ((s.searchMetrics.averageResponseTime * s.searchMetrics.totalQueries) + responseTime) / newTotalQueries;
      
      return {
        ...s,
        searchMetrics: {
          totalQueries: newTotalQueries,
          averageResponseTime: Math.round(newAverageTime),
          lastQueryTime: responseTime
        }
      };
    });
  }

  // Reconnection logic
  private scheduleReconnection(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      this.state.update(s => ({ 
        ...s, 
        error: 'Connection lost. Please refresh the page.',
        connectionStatus: 'error'
      }));
      return;
    }

    this.reconnectAttempts++;
    console.log(`🔄 Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);

    setTimeout(() => {
      this.initializeConnections();
    }, this.reconnectInterval * this.reconnectAttempts);
  }

  // Get fallback results when all else fails
  private async getFallbackResults(query: string): Promise<SearchResult[]> {
    return [
      {
        id: `fallback-${Date.now()}`,
        title: `Fallback search: ${query}`,
        type: 'document' as const,
        content: `Fallback search result for "${query}". Real-time services are currently unavailable.`,
        score: 0.5,
        metadata: {
          date: new Date().toISOString(),
          status: 'fallback',
          tags: ['fallback', query.toLowerCase()]
        },
        createdAt: new Date().toISOString(),
        realTime: false
      }
    ];
  }

  // Graceful cleanup
  public disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.state.update(s => ({ 
      ...s, 
      isConnected: false, 
      connectionStatus: 'disconnected' 
    }));
  }
}

// Global real-time search service instance
export const realTimeSearchService = new RealTimeSearchService();

// Enhanced search hooks for Svelte 5 components
export function useRealTimeSearch() {
  return {
    state: realTimeSearchService.state,
    isReady: realTimeSearchService.isReady,
    hasResults: realTimeSearchService.hasResults,
    searchStatus: realTimeSearchService.searchStatus,
    search: (query: string, options?: any) => realTimeSearchService.performRealTimeSearch(query, options),
    disconnect: () => realTimeSearchService.disconnect()
  };
}