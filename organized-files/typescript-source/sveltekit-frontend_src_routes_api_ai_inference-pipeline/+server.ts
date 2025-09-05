/**
 * Unified AI Inference Pipeline API
 * Integrates Redis caching with Go gateway, Python GPU worker, and client-side XState
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { redisIntegration } from '$lib/server/services/redis-integration';
import { createHash } from 'crypto';

// Request types
interface PipelineRequest {
  type: 'tokenize' | 'embed' | 'generate' | 'similarity' | 'legal_analysis';
  data: any;
  options?: {
    useCache?: boolean;
    userId?: string;
    caseId?: string;
    priority?: 'low' | 'medium' | 'high';
  };
}

interface TokenizeRequest {
  text: string;
  model?: string;
}

interface EmbedRequest {
  texts: string[];
  model?: string;
  normalize?: boolean;
}

interface GenerateRequest {
  prompt: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

interface SimilarityRequest {
  queryVector: number[];
  limit?: number;
  threshold?: number;
  filters?: Record<string, any>;
}

interface LegalAnalysisRequest {
  documentId: string;
  analysisType: 'summary' | 'risks' | 'clauses' | 'compliance';
  content?: string;
  documentType?: string;
}

// Service endpoints configuration
const SERVICES = {
  GO_GATEWAY: process.env.GO_GATEWAY_URL || 'http://localhost:8090',
  PYTHON_GPU: process.env.PYTHON_GPU_URL || 'http://localhost:8091',
  POSTGRES_SIMILARITY: process.env.POSTGRES_URL || 'postgresql://localhost:5432/legal_ai_db'
};

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const body: PipelineRequest = await request.json();
    const { type, data, options = {} } = body;
    const useCache = options.useCache !== false; // Default to true
    const requestId = generateRequestId();

    // Initialize Redis if needed
    if (useCache && !(await redisIntegration.initialize())) {
      console.warn('Redis not available, proceeding without cache');
    }

    switch (type) {
      case 'tokenize':
        return await handleTokenization(data as TokenizeRequest, options, useCache, requestId);
      
      case 'embed':
        return await handleEmbedding(data as EmbedRequest, options, useCache, requestId);
      
      case 'generate':
        return await handleGeneration(data as GenerateRequest, options, useCache, requestId);
      
      case 'similarity':
        return await handleSimilaritySearch(data as SimilarityRequest, options, useCache, requestId);
      
      case 'legal_analysis':
        return await handleLegalAnalysis(data as LegalAnalysisRequest, options, useCache, requestId);
      
      default:
        throw error(400, `Unknown request type: ${type}`);
    }
  } catch (err) {
    console.error('Pipeline error:', err);
    throw error(500, err instanceof Error ? err.message : 'Pipeline processing failed');
  }
};

// Tokenization handler with Redis caching
async function handleTokenization(
  data: TokenizeRequest,
  options: any,
  useCache: boolean,
  requestId: string
) {
  const { text, model = 'default' } = data;
  const cacheRequest = { text, model, userId: options.userId };

  // Check cache first
  if (useCache) {
    const cached = await redisIntegration.getCachedTokenization(cacheRequest);
    if (cached) {
      return json({
        requestId,
        tokens: cached,
        cached: true,
        processingTime: 0
      });
    }
  }

  // Call Go gateway tokenization service
  const startTime = Date.now();
  try {
    const response = await fetch(`${SERVICES.GO_GATEWAY}/api/tokenize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model })
    });

    if (!response.ok) {
      throw new Error(`Tokenization failed: ${response.statusText}`);
    }

    const result = await response.json();
    const processingTime = Date.now() - startTime;

    // Cache the result
    if (useCache && result.tokens) {
      await redisIntegration.cacheTokenization(cacheRequest, result.tokens);
    }

    return json({
      requestId,
      tokens: result.tokens,
      cached: false,
      processingTime
    });
  } catch (err) {
    throw error(500, `Tokenization service error: ${err.message}`);
  }
}

// Embedding handler with batch Redis caching
async function handleEmbedding(
  data: EmbedRequest,
  options: any,
  useCache: boolean,
  requestId: string
) {
  const { texts, model = 'nomic-embed-text', normalize = true } = data;
  const cacheRequest = { texts, model, userId: options.userId };

  // Check cache for all texts
  if (useCache) {
    const cached = await redisIntegration.getBatchCachedEmbeddings(cacheRequest);
    const allCached = cached.every(item => item.embedding !== null);
    
    if (allCached) {
      return json({
        requestId,
        embeddings: cached.map(item => ({
          text: item.text,
          vector: item.embedding,
          dimensions: item.embedding?.length || 0
        })),
        cached: true,
        processingTime: 0
      });
    }
  }

  // Call Python GPU worker for embedding generation
  const startTime = Date.now();
  try {
    const response = await fetch(`${SERVICES.PYTHON_GPU}/embed/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, model, normalize })
    });

    if (!response.ok) {
      throw new Error(`Embedding failed: ${response.statusText}`);
    }

    const result = await response.json();
    const processingTime = Date.now() - startTime;

    // Cache the embeddings
    if (useCache && result.embeddings) {
      const embeddings = result.embeddings.map((emb: any) => emb.vector);
      await redisIntegration.batchCacheEmbeddings(cacheRequest, embeddings);
    }

    return json({
      requestId,
      embeddings: result.embeddings,
      cached: false,
      processingTime
    });
  } catch (err) {
    throw error(500, `Embedding service error: ${err.message}`);
  }
}

// Generation handler with inference caching
async function handleGeneration(
  data: GenerateRequest,
  options: any,
  useCache: boolean,
  requestId: string
) {
  const {
    prompt,
    model = 'gemma3-legal',
    temperature = 0.7,
    maxTokens = 500,
    stream = false
  } = data;

  const cacheRequest = { prompt, model, temperature, maxTokens, userId: options.userId };

  // Check cache (only for non-streaming requests)
  if (useCache && !stream) {
    const cached = await redisIntegration.getCachedInferenceResult(cacheRequest);
    if (cached) {
      return json({
        requestId,
        text: cached.text,
        tokensUsed: cached.tokensUsed,
        cached: true,
        processingTime: 0
      });
    }
  }

  // Call Go gateway for generation
  const startTime = Date.now();
  try {
    const endpoint = stream ? '/api/generate/stream' : '/api/generate';
    const response = await fetch(`${SERVICES.GO_GATEWAY}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        model,
        temperature,
        maxTokens,
        userId: options.userId,
        caseId: options.caseId
      })
    });

    if (!response.ok) {
      throw new Error(`Generation failed: ${response.statusText}`);
    }

    if (stream) {
      // Return streaming response
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/stream',
          'Cache-Control': 'no-cache'
        }
      });
    }

    const result = await response.json();
    const processingTime = Date.now() - startTime;

    // Cache the result
    if (useCache) {
      await redisIntegration.cacheInferenceResult(cacheRequest, {
        ...result,
        processingTime
      });
    }

    return json({
      requestId,
      ...result,
      cached: false,
      processingTime
    });
  } catch (err) {
    throw error(500, `Generation service error: ${err.message}`);
  }
}

// Similarity search with vector caching
async function handleSimilaritySearch(
  data: SimilarityRequest,
  options: any,
  useCache: boolean,
  requestId: string
) {
  const {
    queryVector,
    limit = 10,
    threshold = 0.7,
    filters = {}
  } = data;

  const searchOptions = {
    caseId: options.caseId,
    documentType: filters.documentType,
    threshold,
    limit
  };

  // Check cache
  if (useCache) {
    const cached = await redisIntegration.getCachedSimilaritySearch(queryVector, searchOptions);
    if (cached) {
      return json({
        requestId,
        results: cached,
        cached: true,
        processingTime: 0
      });
    }
  }

  // Call pgvector similarity service
  const startTime = Date.now();
  try {
    // Use our existing pgvector similarity service
    const { searchByEmbedding } = await import('$lib/server/ai/pgvector-similarity');
    
    const results = await searchByEmbedding(queryVector, {
      limit,
      threshold,
      caseId: options.caseId,
      documentType: filters.documentType
    });

    const processingTime = Date.now() - startTime;

    // Cache the results
    if (useCache) {
      await redisIntegration.cacheSimilaritySearch(queryVector, results, searchOptions);
    }

    return json({
      requestId,
      results,
      cached: false,
      processingTime
    });
  } catch (err) {
    throw error(500, `Similarity search error: ${err.message}`);
  }
}

// Legal document analysis with specialized caching
async function handleLegalAnalysis(
  data: LegalAnalysisRequest,
  options: any,
  useCache: boolean,
  requestId: string
) {
  const {
    documentId,
    analysisType,
    content,
    documentType = 'document'
  } = data;

  // Check cache
  if (useCache) {
    const cached = await redisIntegration.getCachedLegalDocumentAnalysis(
      documentId,
      analysisType,
      options.caseId
    );
    if (cached) {
      return json({
        requestId,
        analysis: cached,
        cached: true,
        processingTime: 0
      });
    }
  }

  // Generate legal analysis prompt
  const prompt = buildLegalAnalysisPrompt(content || '', analysisType);

  // Use generation pipeline for analysis
  const generationRequest = {
    prompt,
    model: 'gemma3-legal',
    temperature: 0.3, // More deterministic for legal analysis
    maxTokens: 800
  };

  const startTime = Date.now();
  try {
    const analysisResult = await handleGeneration(
      generationRequest,
      { ...options, useCache: false }, // Generate fresh analysis
      false,
      requestId
    );

    const result = await analysisResult.json();
    const processingTime = Date.now() - startTime;

    const analysis = {
      documentId,
      analysisType,
      documentType,
      content: result.text,
      confidence: calculateAnalysisConfidence(result.text, analysisType),
      extractedEntities: extractLegalEntities(result.text),
      timestamp: new Date().toISOString()
    };

    // Cache the analysis
    if (useCache) {
      const document = {
        id: documentId,
        caseId: options.caseId,
        content: content || '',
        documentType: documentType as any,
        metadata: {}
      };
      await redisIntegration.cacheLegalDocumentAnalysis(document, analysisType, analysis);
    }

    return json({
      requestId,
      analysis,
      cached: false,
      processingTime
    });
  } catch (err) {
    throw error(500, `Legal analysis error: ${err.message}`);
  }
}

// Health endpoint
export const GET: RequestHandler = async () => {
  try {
    const health = await redisIntegration.getPipelineHealth();
    
    // Test service connectivity
    const services = await Promise.allSettled([
      fetch(`${SERVICES.GO_GATEWAY}/health`).then(r => r.ok),
      fetch(`${SERVICES.PYTHON_GPU}/health`).then(r => r.ok)
    ]);

    return json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        redis: health.redis.connected,
        goGateway: services[0].status === 'fulfilled' ? services[0].value : false,
        pythonGpu: services[1].status === 'fulfilled' ? services[1].value : false
      },
      cache: health.pipeline,
      redis: health.redis
    });
  } catch (err) {
    return json({
      status: 'unhealthy',
      error: err.message,
      timestamp: new Date().toISOString()
    });
  }
};

// Utility functions
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function buildLegalAnalysisPrompt(content: string, analysisType: string): string {
  const prompts = {
    summary: `Provide a comprehensive legal summary of this document:\n\n${content}\n\nFocus on key legal points, obligations, and important clauses.`,
    risks: `Analyze potential legal risks and liabilities in this document:\n\n${content}\n\nIdentify specific risk factors, potential issues, and recommended mitigations.`,
    clauses: `Extract and analyze key legal clauses from this document:\n\n${content}\n\nProvide detailed analysis of each important clause and its implications.`,
    compliance: `Review this document for compliance requirements and obligations:\n\n${content}\n\nIdentify all compliance requirements, deadlines, and regulatory obligations.`
  };

  return prompts[analysisType as keyof typeof prompts] || prompts.summary;
}

function calculateAnalysisConfidence(text: string, analysisType: string): number {
  // Simple confidence scoring based on content completeness
  let confidence = 0.5;
  
  if (text.length > 100) confidence += 0.2;
  if (text.includes('legal') || text.includes('obligation')) confidence += 0.1;
  if (text.includes('analysis') || text.includes('review')) confidence += 0.1;
  if (analysisType === 'summary' && text.includes('summary')) confidence += 0.1;
  
  return Math.min(confidence, 1.0);
}

function extractLegalEntities(text: string): string[] {
  // Simple entity extraction for legal terms
  const legalTerms = [
    'contract', 'agreement', 'liability', 'obligation', 'clause',
    'defendant', 'plaintiff', 'jurisdiction', 'statute', 'precedent',
    'evidence', 'testimony', 'witness', 'damages', 'settlement'
  ];
  
  const found = legalTerms.filter(term => 
    text.toLowerCase().includes(term.toLowerCase())
  );
  
  return [...new Set(found)]; // Remove duplicates
}