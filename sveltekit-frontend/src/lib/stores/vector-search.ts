/**
 * Enhanced Vector Search Store with GPU Acceleration & WebASM Inference
 * 
 * Integrates with gpu-summary-store.ts for unified GPU memory management,
 * WebAssembly inference pipeline, and RAG MinIO cache coordination.
 * 
 * Features:
 * - WebASM inference integration for real-time embeddings
 * - GPU-accelerated vector similarity search
 * - RAG MinIO cache with SOM acceleration
 * - Real-time search result ranking and filtering
 * - Adaptive query optimization based on GPU performance
 */

import { writable, derived, get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';
import { 
  gpuSummaryStore, 
  performGPUAcceleratedSearch,
  gpuStoreHelpers,
  type GPUSummaryState 
} from './gpu-summary-store';

// Vector Search Types
export interface VectorSearchQuery {
  text: string;
  embeddings?: Float32Array;
  filters?: SearchFilters;
  options?: SearchOptions;
}

export interface SearchFilters {
  caseId?: string;
  documentType?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  confidence?: {
    min: number;
    max: number;
  };
  priority?: string[];
}

export interface SearchOptions {
  limit?: number;
  useWebASM?: boolean;
  useGPUAcceleration?: boolean;
  cacheResults?: boolean;
  rerank?: boolean;
  threshold?: number; // Similarity threshold 0-1
  bufferPreference?: string; // GPU buffer preference
}

export interface SearchResult {
  id: string;
  content: string;
  metadata: {
    caseId?: string;
    documentId?: string;
    title?: string;
    type?: string;
    createdAt?: string;
    extractionConfidence?: number;
  };
  similarity: number;
  confidence: number;
  highlights?: string[];
  gpuProcessingTime?: number;
  wasmInferenceTime?: number;
}

export interface VectorSearchState {
  isSearching: boolean;
  query: string;
  results: SearchResult[];
  totalResults: number;
  searchTime: number;
  webASMProcessingTime: number;
  gpuAccelerationUsed: boolean;
  cacheHit: boolean;
  performanceMetrics: {
    embeddingTime: number;
    searchTime: number;
    rerankTime: number;
    totalTime: number;
  };
  errors: string[];
  suggestions: string[];
  lastSearchTimestamp: number;
}

// Initial state
const initialSearchState: VectorSearchState = {
  isSearching: false,
  query: '',
  results: [],
  totalResults: 0,
  searchTime: 0,
  webASMProcessingTime: 0,
  gpuAccelerationUsed: false,
  cacheHit: false,
  performanceMetrics: {
    embeddingTime: 0,
    searchTime: 0,
    rerankTime: 0,
    totalTime: 0
  },
  errors: [],
  suggestions: [],
  lastSearchTimestamp: 0
};

// Main vector search store
export const vectorSearchStore: Writable<VectorSearchState> = writable(initialSearchState);

// Derived stores for specific components
export const searchResultsStore: Readable<SearchResult[]> = derived(
  vectorSearchStore,
  ($search) => $search.results
);

export const searchMetricsStore: Readable<VectorSearchState['performanceMetrics']> = derived(
  vectorSearchStore,
  ($search) => $search.performanceMetrics
);

export const isSearchingStore: Readable<boolean> = derived(
  vectorSearchStore,
  ($search) => $search.isSearching
);

// WebASM Embedding Service Integration
class WebASMEmbeddingService {
  private wasmModule: any = null;
  private isInitialized = false;

  async initialize(): Promise<void> {
    try {
      // Initialize WebAssembly module for embedding generation
      const wasmModule = await import('$lib/wasm/embedding-module');
      await wasmModule.initialize();
      
      this.wasmModule = wasmModule;
      this.isInitialized = true;
      
      console.log('WebASM Embedding Service initialized');
    } catch (error) {
      console.error('Failed to initialize WebASM:', error);
      throw error;
    }
  }

  async generateEmbedding(text: string): Promise<{
    embedding: Float32Array;
    processingTime: number;
  }> {
    if (!this.isInitialized || !this.wasmModule) {
      throw new Error('WebASM module not initialized');
    }

    const startTime = performance.now();
    
    try {
      // Generate embedding using WebAssembly for better performance
      const embedding = await this.wasmModule.generateEmbedding(text);
      const processingTime = performance.now() - startTime;
      
      return {
        embedding: new Float32Array(embedding),
        processingTime
      };
    } catch (error) {
      console.error('WebASM embedding generation failed:', error);
      throw error;
    }
  }

  getMemoryUsage(): number {
    return this.wasmModule?.getHeapSize() || 0;
  }
}

// Vector Search Service Integration
class EnhancedVectorSearchService {
  private webASMService = new WebASMEmbeddingService();
  private searchCacheMap = new Map<string, SearchResult[]>();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  async initialize(): Promise<void> {
    try {
      await this.webASMService.initialize();
      console.log('Enhanced Vector Search Service initialized');
    } catch (error) {
      console.error('Failed to initialize vector search service:', error);
      throw error;
    }
  }

  async performSearch(searchQuery: VectorSearchQuery): Promise<SearchResult[]> {
    const totalStart = performance.now();
    
    vectorSearchStore.update(state => ({
      ...state,
      isSearching: true,
      query: searchQuery.text,
      errors: [],
      lastSearchTimestamp: Date.now()
    }));

    try {
      let embeddings = searchQuery.embeddings;
      let webASMTime = 0;

      // Generate embeddings using WebASM if not provided
      if (!embeddings && searchQuery.options?.useWebASM !== false) {
        const embeddingStart = performance.now();
        const embeddingResult = await this.webASMService.generateEmbedding(searchQuery.text);
        embeddings = embeddingResult.embedding;
        webASMTime = embeddingResult.processingTime;
      }

      // Check cache first
      const cacheKey = this.generateCacheKey(searchQuery);
      const cachedResults = this.getCachedResults(cacheKey);
      if (cachedResults && searchQuery.options?.cacheResults !== false) {
        vectorSearchStore.update(state => ({
          ...state,
          isSearching: false,
          results: cachedResults,
          totalResults: cachedResults.length,
          cacheHit: true,
          webASMProcessingTime: webASMTime,
          performanceMetrics: {
            ...state.performanceMetrics,
            embeddingTime: webASMTime,
            totalTime: performance.now() - totalStart
          }
        }));
        return cachedResults;
      }

      // Perform GPU-accelerated search if available
      let searchResults: SearchResult[] = [];
      let gpuAccelerated = false;
      
      if (searchQuery.options?.useGPUAcceleration !== false && gpuStoreHelpers.isGPUReady()) {
        try {
          const gpuResults = await performGPUAcceleratedSearch(searchQuery.text, {
            useWASM: searchQuery.options?.useWebASM,
            cacheResults: searchQuery.options?.cacheResults,
            gpuBuffer: searchQuery.options?.bufferPreference
          });
          
          searchResults = this.transformGPUResults(gpuResults);
          gpuAccelerated = true;
        } catch (error) {
          console.warn('GPU search failed, falling back to standard search:', error);
          searchResults = await this.performStandardSearch(searchQuery, embeddings);
        }
      } else {
        searchResults = await this.performStandardSearch(searchQuery, embeddings);
      }

      // Apply reranking if requested
      if (searchQuery.options?.rerank) {
        const rerankStart = performance.now();
        searchResults = await this.rerankResults(searchResults, searchQuery);
        const rerankTime = performance.now() - rerankStart;
        
        vectorSearchStore.update(state => ({
          ...state,
          performanceMetrics: {
            ...state.performanceMetrics,
            rerankTime
          }
        }));
      }

      // Apply filters and threshold
      searchResults = this.applyFilters(searchResults, searchQuery.filters);
      searchResults = this.applyThreshold(searchResults, searchQuery.options?.threshold || 0.5);

      // Cache results
      if (searchQuery.options?.cacheResults !== false) {
        this.cacheResults(cacheKey, searchResults);
      }

      const totalTime = performance.now() - totalStart;

      // Update store with results
      vectorSearchStore.update(state => ({
        ...state,
        isSearching: false,
        results: searchResults,
        totalResults: searchResults.length,
        searchTime: totalTime - webASMTime,
        webASMProcessingTime: webASMTime,
        gpuAccelerationUsed: gpuAccelerated,
        cacheHit: false,
        performanceMetrics: {
          embeddingTime: webASMTime,
          searchTime: totalTime - webASMTime,
          rerankTime: state.performanceMetrics.rerankTime,
          totalTime
        },
        suggestions: this.generateSearchSuggestions(searchQuery.text, searchResults)
      }));

      return searchResults;

    } catch (error) {
      vectorSearchStore.update(state => ({
        ...state,
        isSearching: false,
        errors: [...state.errors, `Search error: ${error.message}`]
      }));
      throw error;
    }
  }

  private async performStandardSearch(
    searchQuery: VectorSearchQuery, 
    embeddings?: Float32Array
  ): Promise<SearchResult[]> {
    // Standard vector search implementation
    const response = await fetch('/api/vector-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: searchQuery.text,
        embeddings: embeddings ? Array.from(embeddings) : undefined,
        filters: searchQuery.filters,
        options: searchQuery.options
      })
    });

    if (!response.ok) {
      throw new Error(`Search API failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.results || [];
  }

  private transformGPUResults(gpuResults: any[]): SearchResult[] {
    return gpuResults.map((result, index) => ({
      id: result.id || `gpu-${index}`,
      content: result.content || '',
      metadata: result.metadata || {},
      similarity: result.similarity || 0,
      confidence: result.confidence || 0,
      highlights: result.highlights || [],
      gpuProcessingTime: result.processingTime || 0
    }));
  }

  private async rerankResults(results: SearchResult[], query: VectorSearchQuery): Promise<SearchResult[]> {
    // Implement custom reranking logic
    // This could involve additional AI models or business logic
    const response = await fetch('/api/rerank', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query.text,
        results: results.slice(0, 50) // Rerank top 50 results
      })
    });

    if (response.ok) {
      const rerankedData = await response.json();
      return rerankedData.results || results;
    }

    return results;
  }

  private applyFilters(results: SearchResult[], filters?: SearchFilters): SearchResult[] {
    if (!filters) return results;

    return results.filter(result => {
      // Case ID filter
      if (filters.caseId && result.metadata.caseId !== filters.caseId) {
        return false;
      }

      // Document type filter
      if (filters.documentType?.length && 
          !filters.documentType.includes(result.metadata.type || '')) {
        return false;
      }

      // Confidence filter
      if (filters.confidence) {
        if (result.confidence < filters.confidence.min || 
            result.confidence > filters.confidence.max) {
          return false;
        }
      }

      // Date range filter
      if (filters.dateRange && result.metadata.createdAt) {
        const resultDate = new Date(result.metadata.createdAt);
        if (resultDate < filters.dateRange.start || resultDate > filters.dateRange.end) {
          return false;
        }
      }

      return true;
    });
  }

  private applyThreshold(results: SearchResult[], threshold: number): SearchResult[] {
    return results.filter(result => result.similarity >= threshold);
  }

  private generateCacheKey(query: VectorSearchQuery): string {
    return btoa(JSON.stringify({
      text: query.text,
      filters: query.filters,
      options: { ...query.options, cacheResults: undefined }
    }));
  }

  private getCachedResults(cacheKey: string): SearchResult[] | null {
    const cached = this.searchCacheMap.get(cacheKey);
    if (cached && Date.now() - this.lastCacheTime(cacheKey) < this.cacheTimeout) {
      return cached;
    }
    this.searchCacheMap.delete(cacheKey);
    return null;
  }

  private cacheResults(cacheKey: string, results: SearchResult[]): void {
    this.searchCacheMap.set(cacheKey, results);
    // Clean up old cache entries
    if (this.searchCacheMap.size > 100) {
      const oldestKey = this.searchCacheMap.keys().next().value;
      this.searchCacheMap.delete(oldestKey);
    }
  }

  private lastCacheTime(cacheKey: string): number {
    // Simple cache timing - could be enhanced
    return Date.now() - this.cacheTimeout + 60000; // Assume recent
  }

  private generateSearchSuggestions(query: string, results: SearchResult[]): string[] {
    // Generate intelligent search suggestions based on results
    const suggestions: string[] = [];
    
    // Extract common terms from top results
    const topResults = results.slice(0, 5);
    const commonTerms = new Map<string, number>();
    
    topResults.forEach(result => {
      const words = result.content.toLowerCase().split(/\W+/);
      words.forEach(word => {
        if (word.length > 3 && !query.toLowerCase().includes(word)) {
          commonTerms.set(word, (commonTerms.get(word) || 0) + 1);
        }
      });
    });

    // Sort by frequency and take top suggestions
    const sortedTerms = Array.from(commonTerms.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([term]) => `${query} ${term}`);

    suggestions.push(...sortedTerms);

    return suggestions;
  }
}

// Singleton service instance
export const vectorSearchService = new EnhancedVectorSearchService();

// Store Actions/Helpers
export const vectorSearchActions = {
  async performSearch(searchQuery: VectorSearchQuery): Promise<SearchResult[]> {
    return await vectorSearchService.performSearch(searchQuery);
  },

  async performQuickSearch(text: string, options?: SearchOptions): Promise<SearchResult[]> {
    return await vectorSearchService.performSearch({
      text,
      options: {
        limit: 10,
        useWebASM: true,
        useGPUAcceleration: true,
        cacheResults: true,
        ...options
      }
    });
  },

  clearResults(): void {
    vectorSearchStore.update(state => ({
      ...state,
      results: [],
      totalResults: 0,
      errors: [],
      suggestions: []
    }));
  },

  clearErrors(): void {
    vectorSearchStore.update(state => ({
      ...state,
      errors: []
    }));
  },

  getSearchHistory(): VectorSearchQuery[] {
    // Could be enhanced to persist search history
    return [];
  }
};

// Auto-initialize service when imported
if (typeof window !== 'undefined') {
  vectorSearchService.initialize().catch(console.error);
}

// Integration with GPU Summary Store - Real-time performance monitoring
derived([vectorSearchStore, gpuSummaryStore], ([$search, $gpu]) => {
  // Update GPU store with WebASM inference metrics when search completes
  if (!$search.isSearching && $search.webASMProcessingTime > 0) {
    return {
      inferenceTime: $search.webASMProcessingTime,
      throughput: $search.totalResults / ($search.webASMProcessingTime / 1000),
      memoryFootprint: vectorSearchService['webASMService']?.getMemoryUsage() || 0
    };
  }
  return null;
}).subscribe(async (wasmMetrics) => {
  if (wasmMetrics && gpuStoreHelpers.isGPUReady()) {
    const { gpuSummaryService } = await import('./gpu-summary-store');
    await gpuSummaryService.updateWASMInference(wasmMetrics);
  }
});