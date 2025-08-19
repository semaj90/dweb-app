/**
 * Vector Search Store - PostgreSQL pgvector + Ollama Integration
 * Enhanced RAG with semantic search capabilities
 */

import { writable, derived, readable } from 'svelte/store';
// Orphaned content: import type { VectorSearchResult, RAGContext
import {
productionServiceClient } from "../services/productionServiceClient.js";

export interface VectorSearchState {
  // Search State
  query: string;
  results: VectorSearchResult[];
  isSearching: boolean;
  lastSearchTime: number | null;
  
  // RAG Context
  ragContext: VectorSearchResult[];
  ragResponse: string | null;
  isGeneratingResponse: boolean;
  
  // Configuration
  searchThreshold: number;
  searchLimit: number;
  embeddingModel: 'nomic-embed-text' | 'nvidia-llama';
  
  // Performance Metrics
  searchLatency: number;
  ragLatency: number;
  vectorDbConnected: boolean;
  
  // History
  searchHistory: Array<{
    query: string;
    timestamp: number;
    resultCount: number;
    latency: number;
  }>;
  
  error: string | null;
}

const initialState: VectorSearchState = {
  query: '',
  results: [],
  isSearching: false,
  lastSearchTime: null,
  ragContext: [],
  ragResponse: null,
  isGeneratingResponse: false,
  searchThreshold: 0.7,
  searchLimit: 10,
  embeddingModel: 'nomic-embed-text',
  searchLatency: 0,
  ragLatency: 0,
  vectorDbConnected: false,
  searchHistory: [],
  error: null
};

// Core store
export const vectorSearchStore = writable<VectorSearchState>(initialState);

// Derived stores
export const isVectorSearchActive = derived(
  vectorSearchStore,
  $store => $store.isSearching || $store.isGeneratingResponse
);

export const hasSearchResults = derived(
  vectorSearchStore,
  $store => $store.results.length > 0
);

export const averageSearchLatency = derived(
  vectorSearchStore,
  $store => {
    if ($store.searchHistory.length === 0) return 0;
    const total = $store.searchHistory.reduce((sum, item) => sum + item.latency, 0);
    return total / $store.searchHistory.length;
  }
);

// Actions
export const vectorSearchActions = {
  /**
   * Perform semantic vector search
   */
  async search(query: string, userId: string, caseId?: string): Promise<void> {
    if (!query.trim()) return;

    vectorSearchStore.update(state => ({
      ...state,
      query,
      isSearching: true,
      error: null
    }));

    const startTime = Date.now();

    try {
      // Call vector search API
      const response = await fetch('/api/v1/vector/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          userId,
          caseId,
          limit: initialState.searchLimit,
          threshold: initialState.searchThreshold
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      const latency = Date.now() - startTime;

      vectorSearchStore.update(state => ({
        ...state,
        results: data.results || [],
        searchLatency: latency,
        lastSearchTime: Date.now(),
        isSearching: false,
        searchHistory: [
          ...state.searchHistory.slice(-9), // Keep last 10
          {
            query,
            timestamp: Date.now(),
            resultCount: data.results?.length || 0,
            latency
          }
        ]
      }));

    } catch (error) {
      console.error('Vector search failed:', error);
      vectorSearchStore.update(state => ({
        ...state,
        isSearching: false,
        error: error instanceof Error ? error.message : 'Search failed'
      }));
    }
  },

  /**
   * Perform enhanced RAG query with context
   */
  async performRAG(query: string, userId: string, caseId?: string): Promise<void> {
    if (!query.trim()) return;

    vectorSearchStore.update(state => ({
      ...state,
      isGeneratingResponse: true,
      ragResponse: null,
      error: null
    }));

    const startTime = Date.now();

    try {
      // First perform vector search to get context
      await vectorSearchActions.search(query, userId, caseId);

      // Then generate RAG response
      const response = await fetch('/api/v1/rag/enhanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          userId,
          caseId,
          useContext: true,
          model: 'gemma3-legal'
        })
      });

      if (!response.ok) {
        throw new Error(`RAG query failed: ${response.statusText}`);
      }

      const data = await response.json();
      const latency = Date.now() - startTime;

      vectorSearchStore.update(state => ({
        ...state,
        ragResponse: data.response,
        ragContext: data.context || state.results,
        ragLatency: latency,
        isGeneratingResponse: false
      }));

    } catch (error) {
      console.error('RAG query failed:', error);
      vectorSearchStore.update(state => ({
        ...state,
        isGeneratingResponse: false,
        error: error instanceof Error ? error.message : 'RAG query failed'
      }));
    }
  },

  /**
   * Find similar cases using vector similarity
   */
  async findSimilarCases(caseId: string, userId: string, limit: number = 5): Promise<void> {
    vectorSearchStore.update(state => ({
      ...state,
      isSearching: true,
      error: null
    }));

    try {
      const response = await fetch('/api/v1/vector/similar-cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caseId, userId, limit })
      });

      if (!response.ok) {
        throw new Error(`Similar cases search failed: ${response.statusText}`);
      }

      const data = await response.json();

      vectorSearchStore.update(state => ({
        ...state,
        results: data.results || [],
        isSearching: false
      }));

    } catch (error) {
      console.error('Similar cases search failed:', error);
      vectorSearchStore.update(state => ({
        ...state,
        isSearching: false,
        error: error instanceof Error ? error.message : 'Similar cases search failed'
      }));
    }
  },

  /**
   * Update search configuration
   */
  updateConfig(config: Partial<{
    searchThreshold: number;
    searchLimit: number;
    embeddingModel: 'nomic-embed-text' | 'nvidia-llama';
  }>): void {
    vectorSearchStore.update(state => ({
      ...state,
      ...config
    }));
  },

  /**
   * Clear search results and state
   */
  clear(): void {
    vectorSearchStore.update(state => ({
      ...state,
      query: '',
      results: [],
      ragContext: [],
      ragResponse: null,
      error: null
    }));
  },

  /**
   * Check vector database connection
   */
  async checkConnection(): Promise<void> {
    try {
      const response = await fetch('/api/v1/vector/health');
      const data = await response.json();
      
      vectorSearchStore.update(state => ({
        ...state,
        vectorDbConnected: response.ok && data.status === 'healthy'
      }));
    } catch (error) {
      vectorSearchStore.update(state => ({
        ...state,
        vectorDbConnected: false
      }));
    }
  }
};

// Initialize connection check
if (typeof window !== 'undefined') {
  vectorSearchActions.checkConnection();
}