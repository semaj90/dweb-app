/**
 * XState + Loki.js Client-Side Inference Caching
 * Optimized for legal AI workflows with prefetching and persistence
 */

import { writable, derived, get } from 'svelte/store';
import { createActor, assign, setup, fromPromise } from 'xstate';
import Loki from 'lokijs';
import Fuse from 'fuse.js';
import { browser } from '$app/environment';

// Types
interface InferenceRequest {
  id: string;
  prompt: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  metadata?: Record<string, any>;
}

interface InferenceResult {
  id: string;
  requestId: string;
  text: string;
  success: boolean;
  timestamp: number;
  cached: boolean;
  metadata?: Record<string, any>;
}

interface CachedQuery {
  id: string;
  query: string;
  results: InferenceResult[];
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
}

interface PrefetchCandidate {
  prompt: string;
  priority: number;
  reason: string;
}

// Loki.js Database Manager
class InferenceCacheDB {
  private db: Loki;
  private results: Collection<InferenceResult>;
  private queries: Collection<CachedQuery>;
  private requests: Collection<InferenceRequest>;
  private initialized = false;

  constructor() {
    if (!browser) return;
    
    this.db = new Loki('inference-cache.db', {
      persistenceMethod: 'indexeddb',
      autosave: true,
      autosaveInterval: 30000, // 30 seconds
    });
  }

  async initialize(): Promise<void> {
    if (!browser || this.initialized) return;
    
    return new Promise((resolve) => {
      this.db.loadDatabase({}, () => {
        // Initialize collections
        this.results = this.db.getCollection('results') || 
          this.db.addCollection('results', { 
            indices: ['requestId', 'timestamp'],
            ttl: { age: 24 * 60 * 60 * 1000 } // 24 hours
          });
        
        this.queries = this.db.getCollection('queries') || 
          this.db.addCollection('queries', { 
            indices: ['query', 'timestamp'],
            ttl: { age: 6 * 60 * 60 * 1000 } // 6 hours
          });
        
        this.requests = this.db.getCollection('requests') || 
          this.db.addCollection('requests', { 
            indices: ['id', 'prompt'],
            ttl: { age: 2 * 60 * 60 * 1000 } // 2 hours
          });
        
        this.initialized = true;
        resolve();
      });
    });
  }

  cacheResult(result: InferenceResult): void {
    if (!this.initialized) return;
    this.results.insert({ ...result, $loki: undefined });
  }

  getCachedResult(requestHash: string): InferenceResult | null {
    if (!this.initialized) return null;
    return this.results.findOne({ id: requestHash });
  }

  cacheQuery(query: string, results: InferenceResult[]): void {
    if (!this.initialized) return;
    
    const existing = this.queries.findOne({ query });
    if (existing) {
      existing.results = results;
      existing.lastAccessed = Date.now();
      existing.accessCount++;
      this.queries.update(existing);
    } else {
      this.queries.insert({
        id: this.generateId(),
        query,
        results,
        timestamp: Date.now(),
        accessCount: 1,
        lastAccessed: Date.now()
      });
    }
  }

  getCachedQuery(query: string): CachedQuery | null {
    if (!this.initialized) return null;
    
    const result = this.queries.findOne({ query });
    if (result) {
      result.lastAccessed = Date.now();
      result.accessCount++;
      this.queries.update(result);
    }
    return result;
  }

  searchSimilarQueries(query: string, limit: number = 5): CachedQuery[] {
    if (!this.initialized) return [];
    
    const allQueries = this.queries.find();
    const fuse = new Fuse(allQueries, {
      keys: ['query'],
      threshold: 0.6,
      includeScore: true
    });
    
    const results = fuse.search(query).slice(0, 20);
    return results.map(r => r.item);
  }

  getPopularQueries(limit: number = 10): CachedQuery[] {
    if (!this.initialized) return [];
    
    return this.queries
      .chain()
      .simplesort('accessCount', true)
      .limit(limit)
      .data();
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

// XState Machine for Inference Management
const inferenceMachine = setup({
  types: {
    context: {} as {
      currentRequest: InferenceRequest | null;
      currentResult: InferenceResult | null;
      error: string | null;
      loading: boolean;
      prefetchQueue: PrefetchCandidate[];
      recentResults: InferenceResult[];
      cacheStats: {
        hits: number;
        misses: number;
        size: number;
      };
    },
    events: {} as
      | { type: 'SUBMIT_REQUEST'; request: InferenceRequest }
      | { type: 'REQUEST_SUCCESS'; result: InferenceResult }
      | { type: 'REQUEST_ERROR'; error: string }
      | { type: 'CACHE_HIT'; result: InferenceResult }
      | { type: 'PREFETCH_PROMPT'; prompt: string }
      | { type: 'CLEAR_CACHE' }
      | { type: 'RETRY' }
  },
  actors: {
    submitInference: fromPromise(async ({ input }: { input: InferenceRequest }) => {
      // Check cache first
      const db = get(cacheDB);
      const requestHash = await hashRequest(input);
      const cached = db.getCachedResult(requestHash);
      
      if (cached && (Date.now() - cached.timestamp) < 30 * 60 * 1000) { // 30 min cache
        return { ...cached, cached: true };
      }
      
      // Make actual request
      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      });
      
      if (!response.ok) {
        throw new Error(`Request failed: ${response.statusText}`);
      }
      
      const result: InferenceResult = await response.json();
      result.cached = false;
      result.timestamp = Date.now();
      
      // Cache the result
      db.cacheResult(result);
      
      return result;
    })
  }
}).createMachine({
  id: 'inference',
  initial: 'idle',
  context: {
    currentRequest: null,
    currentResult: null,
    error: null,
    loading: false,
    prefetchQueue: [],
    recentResults: [],
    cacheStats: { hits: 0, misses: 0, size: 0 }
  },
  states: {
    idle: {
      on: {
        SUBMIT_REQUEST: {
          target: 'processing',
          actions: assign({
            currentRequest: ({ event }) => event.request,
            error: null,
            loading: true
          })
        },
        PREFETCH_PROMPT: {
          actions: assign({
            prefetchQueue: ({ context, event }) => [
              ...context.prefetchQueue,
              {
                prompt: event.prompt,
                priority: Date.now(),
                reason: 'user_typing'
              }
            ]
          })
        },
        CLEAR_CACHE: {
          actions: assign({
            recentResults: [],
            cacheStats: { hits: 0, misses: 0, size: 0 }
          })
        }
      }
    },
    processing: {
      invoke: {
        src: 'submitInference',
        input: ({ context }) => context.currentRequest!,
        onDone: {
          target: 'success',
          actions: [
            assign({
              currentResult: ({ event }) => event.output,
              loading: false,
              recentResults: ({ context, event }) => [
                event.output,
                ...context.recentResults.slice(0, 19) // Keep last 20
              ],
              cacheStats: ({ context, event }) => ({
                ...context.cacheStats,
                [event.output.cached ? 'hits' : 'misses']: 
                  context.cacheStats[event.output.cached ? 'hits' : 'misses'] + 1
              })
            }),
            ({ event }) => {
              if (event.output.cached) {
                // Emit cache hit event for analytics
                window.dispatchEvent(new CustomEvent('cache-hit', { 
                  detail: event.output 
                }));
              }
            }
          ]
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => event.error.message,
            loading: false
          })
        }
      }
    },
    success: {
      on: {
        SUBMIT_REQUEST: {
          target: 'processing',
          actions: assign({
            currentRequest: ({ event }) => event.request,
            error: null,
            loading: true
          })
        }
      }
    },
    error: {
      on: {
        RETRY: {
          target: 'processing',
          guard: ({ context }) => !!context.currentRequest
        },
        SUBMIT_REQUEST: {
          target: 'processing',
          actions: assign({
            currentRequest: ({ event }) => event.request,
            error: null,
            loading: true
          })
        }
      }
    }
  }
});

// Stores and Services
const cacheDB = writable(new InferenceCacheDB());
const inferenceActor = createActor(inferenceMachine).start();

// Reactive stores
export const inferenceState = writable(inferenceActor.getSnapshot());
export const isLoading = derived(inferenceState, ($state) => $state.context.loading);
export const currentResult = derived(inferenceState, ($state) => $state.context.currentResult);
export const error = derived(inferenceState, ($state) => $state.context.error);
export const recentResults = derived(inferenceState, ($state) => $state.context.recentResults);
export const cacheStats = derived(inferenceState, ($state) => $state.context.cacheStats);

// Update store when actor state changes
inferenceActor.subscribe((state) => {
  inferenceState.set(state);
});

// Utility functions
async function hashRequest(request: InferenceRequest): Promise<string> {
  const str = JSON.stringify({
    prompt: request.prompt,
    model: request.model,
    temperature: request.temperature,
    maxTokens: request.maxTokens
  });
  
  if (!browser || !window.crypto?.subtle) {
    // Fallback hash for SSR or older browsers
    return btoa(str).slice(0, 16);
  }
  
  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  const hashBuffer = await window.crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

// Public API
export class InferenceCacheService {
  private static instance: InferenceCacheService;
  private prefetchTimeout: NodeJS.Timeout | null = null;

  static getInstance(): InferenceCacheService {
    if (!InferenceCacheService.instance) {
      InferenceCacheService.instance = new InferenceCacheService();
    }
    return InferenceCacheService.instance;
  }

  async initialize(): Promise<void> {
    const db = get(cacheDB);
    await db.initialize();
  }

  async submitRequest(request: Omit<InferenceRequest, 'id'>): Promise<void> {
    const fullRequest: InferenceRequest = {
      id: await hashRequest(request as InferenceRequest),
      ...request
    };
    
    inferenceActor.send({ type: 'SUBMIT_REQUEST', request: fullRequest });
  }

  async prefetchPrompt(prompt: string, delay: number = 500): Promise<void> {
    // Debounce prefetching
    if (this.prefetchTimeout) {
      clearTimeout(this.prefetchTimeout);
    }
    
    this.prefetchTimeout = setTimeout(() => {
      // Only prefetch if not currently processing
      const state = inferenceActor.getSnapshot();
      if (state.value === 'idle' && prompt.length > 10) {
        inferenceActor.send({ type: 'PREFETCH_PROMPT', prompt });
        
        // Trigger background prefetch request
        this.backgroundPrefetch(prompt);
      }
    }, delay);
  }

  private async backgroundPrefetch(prompt: string): Promise<void> {
    try {
      // Submit low-priority prefetch request
      await fetch('/api/ai/prefetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt, 
          priority: 'low',
          maxTokens: 50 // Shorter for prefetch
        })
      });
    } catch (error) {
      // Silently fail prefetch
      console.debug('Prefetch failed:', error);
    }
  }

  getSimilarResults(query: string): InferenceResult[] {
    const db = get(cacheDB);
    const similarQueries = db.searchSimilarQueries(query);
    return similarQueries.flatMap(q => q.results).slice(0, 5);
  }

  getPopularQueries(limit: number = 10): string[] {
    const db = get(cacheDB);
    return db.getPopularQueries(limit).map(q => q.query);
  }

  clearCache(): void {
    inferenceActor.send({ type: 'CLEAR_CACHE' });
  }

  retry(): void {
    inferenceActor.send({ type: 'RETRY' });
  }

  // Legal-specific helpers
  async analyzeLegalDocument(documentText: string, analysisType: string = 'summary'): Promise<void> {
    const prompt = this.buildLegalPrompt(documentText, analysisType);
    await this.submitRequest({
      prompt,
      model: 'legal-specialist',
      temperature: 0.3, // More deterministic for legal analysis
      maxTokens: 500,
      metadata: { type: 'legal_analysis', analysisType }
    });
  }

  private buildLegalPrompt(documentText: string, analysisType: string): string {
    const prompts = {
      summary: `Provide a concise legal summary of the following document:\n\n${documentText}`,
      risks: `Identify potential legal risks and issues in this document:\n\n${documentText}`,
      compliance: `Analyze compliance requirements and obligations:\n\n${documentText}`,
      clauses: `Extract and analyze key clauses and provisions:\n\n${documentText}`
    };
    
    return prompts[analysisType as keyof typeof prompts] || prompts.summary;
  }
}

// Initialize service if in browser
if (browser) {
  InferenceCacheService.getInstance().initialize();
}

// Export singleton
export const inferenceCacheService = InferenceCacheService.getInstance();