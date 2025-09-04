/**
 * Embedding Service for Vector Operations
 * Integrates with Ollama nomic-embed-text model for generating 384D embeddings
 * Enhanced with caching and fallback support
 */

import { redis } from '$lib/server/redis';

const OLLAMA_BASE_URL = 'http://localhost:11434';
const EMBEDDING_MODEL = 'nomic-embed-text';
const EMBEDDING_DIMENSIONS = 384;
const CACHE_TTL = 3600; // 1 hour cache for embeddings

export interface EmbeddingResult {
  model: string;
  dimensions: number;
  vector: number[];
  latencyMs: number;
  source: 'ollama' | 'rag-service';
  truncated?: boolean;
}

interface GenerateOptions {
  model?: string;
  maxInputChars?: number;
  preferRagService?: boolean;
  signal?: AbortSignal;
  useCache?: boolean;
}

interface EmbeddingResponse {
  embedding: number[];
}

interface EmbeddingCacheEntry {
  embedding: number[];
  model: string;
  dimensions: number;
  created_at: string;
}

const DEFAULT_MODEL = (import.meta.env?.VITE_EMBEDDING_MODEL as string) || EMBEDDING_MODEL;
const OLLAMA_URL = (import.meta.env?.VITE_OLLAMA_EMBED_URL as string) || `${OLLAMA_BASE_URL}/api/embeddings`;
const RAG_EMBED_URL = (import.meta.env?.VITE_RAG_EMBED_URL as string) || '/api/ai/embeddings';

/**
 * Generate embedding for a single text using Ollama (primary interface for API compatibility)
 */
export async function createEmbedding(text: string, useCache: boolean = true): Promise<number[]> {
  if (!text || text.trim().length === 0) {
    throw new Error('Text cannot be empty');
  }
  
  // Normalize text
  const normalizedText = text.trim().replace(/\s+/g, ' ');
  
  // Try cache first
  if (useCache) {
    const cacheKey = `embedding:${Buffer.from(normalizedText).toString('base64')}`;
    
    try {
      const cached = await redis.get(cacheKey);
      if (cached) {
        const cachedEntry: EmbeddingCacheEntry = JSON.parse(cached);
        
        // Validate cached embedding
        if (cachedEntry.embedding && 
            Array.isArray(cachedEntry.embedding) && 
            cachedEntry.embedding.length === EMBEDDING_DIMENSIONS &&
            cachedEntry.model === EMBEDDING_MODEL) {
          return cachedEntry.embedding;
        }
      }
    } catch (cacheError) {
      console.warn('Cache retrieval failed, continuing without cache:', cacheError);
    }
  }
  
  // Generate embedding via Ollama API
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        prompt: normalizedText
      }),
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });
    
    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }
    
    const result: EmbeddingResponse = await response.json();
    
    if (!result.embedding || !Array.isArray(result.embedding)) {
      throw new Error('Invalid embedding response from Ollama');
    }
    
    // Validate embedding dimensions
    if (result.embedding.length !== EMBEDDING_DIMENSIONS) {
      throw new Error(`Expected ${EMBEDDING_DIMENSIONS} dimensions, got ${result.embedding.length}`);
    }
    
    // Cache the result
    if (useCache) {
      try {
        const cacheKey = `embedding:${Buffer.from(normalizedText).toString('base64')}`;
        const cacheEntry: EmbeddingCacheEntry = {
          embedding: result.embedding,
          model: EMBEDDING_MODEL,
          dimensions: EMBEDDING_DIMENSIONS,
          created_at: new Date().toISOString()
        };
        
        await redis.setex(cacheKey, CACHE_TTL, JSON.stringify(cacheEntry));
      } catch (cacheError) {
        console.warn('Failed to cache embedding:', cacheError);
        // Continue without caching
      }
    }
    
    return result.embedding;
    
  } catch (error) {
    console.error('Embedding generation failed:', error);
    
    // Enhanced error handling
    if (error instanceof Error) {
      if (error.message.includes('ECONNREFUSED')) {
        throw new Error('Ollama service is not running. Please start Ollama server.');
      } else if (error.message.includes('timeout')) {
        throw new Error('Embedding generation timed out. Please try again.');
      } else if (error.message.includes('404')) {
        throw new Error(`Model '${EMBEDDING_MODEL}' not found. Please pull the model first.`);
      }
    }
    
    throw new Error(`Failed to generate embedding: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Enhanced embedding generation with fallback support (backward compatible interface)
 */
async function generateEmbedding(input: string, opts: GenerateOptions = {}): Promise<EmbeddingResult> {
  const start = performance.now();
  const model = opts.model || DEFAULT_MODEL;
  const maxChars = opts.maxInputChars || 8192;
  const useCache = opts.useCache !== false;
  let truncated = false;
  let text = input;
  
  if (text.length > maxChars) { 
    text = text.slice(0, maxChars); 
    truncated = true; 
  }

  // Try preferred route first
  const routes: { url: string; body: any; source: 'ollama' | 'rag-service' }[] = [];
  if (opts.preferRagService) {
    routes.push({ url: RAG_EMBED_URL, body: { model, input: text }, source: 'rag-service' });
    routes.push({ url: OLLAMA_URL, body: { model, prompt: text }, source: 'ollama' });
  } else {
    routes.push({ url: OLLAMA_URL, body: { model, prompt: text }, source: 'ollama' });
    routes.push({ url: RAG_EMBED_URL, body: { model, input: text }, source: 'rag-service' });
  }

  let lastError: any;
  for (const route of routes) {
    try {
      const res = await fetch(route.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(route.body),
        signal: opts.signal
      });
      if (!res.ok) throw new Error(`Embedding request failed (${res.status})`);
      const json = await res.json();
      const vector: number[] = json?.data?.[0]?.embedding || json?.embedding || json?.vector || [];
      if (!Array.isArray(vector) || vector.length === 0) throw new Error('No embedding vector in response');
      
      // Cache successful result if using Ollama
      if (route.source === 'ollama' && useCache) {
        try {
          const cacheKey = `embedding:${Buffer.from(text).toString('base64')}`;
          const cacheEntry: EmbeddingCacheEntry = {
            embedding: vector,
            model,
            dimensions: vector.length,
            created_at: new Date().toISOString()
          };
          await redis.setex(cacheKey, CACHE_TTL, JSON.stringify(cacheEntry));
        } catch (cacheError) {
          console.warn('Failed to cache embedding:', cacheError);
        }
      }
      
      return {
        model,
        dimensions: vector.length,
        vector,
        latencyMs: performance.now() - start,
        source: route.source,
        truncated
      };
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error('Embedding generation failed');
}

/**
 * Generate embeddings for multiple texts in batch
 */
export async function createEmbeddings(
  texts: string[], 
  concurrency: number = 5,
  useCache: boolean = true
): Promise<number[][]> {
  if (!texts || texts.length === 0) {
    return [];
  }
  
  const validTexts = texts.filter(text => text && text.trim().length > 0);
  
  if (validTexts.length === 0) {
    throw new Error('No valid texts provided');
  }
  
  const results: number[][] = [];
  
  for (let i = 0; i < validTexts.length; i += concurrency) {
    const batch = validTexts.slice(i, i + concurrency);
    const batchPromises = batch.map(text => createEmbedding(text, useCache));
    
    try {
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
    } catch (error) {
      console.error(`Batch ${Math.floor(i / concurrency) + 1} failed:`, error);
      throw error;
    }
  }
  
  return results;
}

/**
 * Calculate cosine similarity between two embeddings
 */
export function calculateCosineSimilarity(embedding1: number[], embedding2: number[]): number {
  if (embedding1.length !== embedding2.length) {
    throw new Error('Embeddings must have the same dimensions');
  }
  
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    norm1 += embedding1[i] * embedding1[i];
    norm2 += embedding2[i] * embedding2[i];
  }
  
  if (norm1 === 0 || norm2 === 0) {
    return 0;
  }
  
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * Health check for embedding service
 */
export async function checkEmbeddingServiceHealth(): Promise<{
  status: 'healthy' | 'unhealthy';
  ollama_available: boolean;
  model_available: boolean;
  redis_available: boolean;
  error?: string;
}> {
  let ollamaAvailable = false;
  let modelAvailable = false;
  let redisAvailable = false;
  let error = '';
  
  try {
    const ollamaResponse = await fetch(`${OLLAMA_BASE_URL}/api/version`, {
      signal: AbortSignal.timeout(5000)
    });
    ollamaAvailable = ollamaResponse.ok;
    
    if (ollamaAvailable) {
      const modelsResponse = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
      if (modelsResponse.ok) {
        const models = await modelsResponse.json();
        modelAvailable = models.models?.some((model: any) => 
          model.name?.includes(EMBEDDING_MODEL)
        ) || false;
      }
    }
  } catch (ollamaError) {
    error += `Ollama: ${ollamaError instanceof Error ? ollamaError.message : String(ollamaError)}; `;
  }
  
  try {
    await redis.ping();
    redisAvailable = true;
  } catch (redisError) {
    error += `Redis: ${redisError instanceof Error ? redisError.message : String(redisError)}; `;
  }
  
  const isHealthy = ollamaAvailable && modelAvailable && redisAvailable;
  
  return {
    status: isHealthy ? 'healthy' : 'unhealthy',
    ollama_available: ollamaAvailable,
    model_available: modelAvailable,
    redis_available: redisAvailable,
    error: error || undefined
  };
}

// Backward compatibility exports
export const embeddingService = { generateEmbedding };
export default embeddingService;
