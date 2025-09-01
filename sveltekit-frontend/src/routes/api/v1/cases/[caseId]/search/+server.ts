import type { RequestHandler } from './$types.js';

/**
 * Case-specific Vector Search API - PostgreSQL pgvector + RAG Integration
 * Semantic search within case evidence using stored embeddings
 */

import { json } from '@sveltejs/kit';
import { db } from '$lib/server/db/client';
import { cases, evidence, documents, chatMessages } from '$lib/db/schema';
import { eq, and, sql, desc } from 'drizzle-orm';
import { redis } from '$lib/server/cache/redis-service';
import { URL } from "url";

export interface SearchRequest {
  query: string;
  limit?: number;
  threshold?: number;
  includeRAG?: boolean;
  evidenceTypes?: string[];
}

export interface VectorSearchResult {
  id: string;
  title: string;
  content: string;
  evidenceType: string;
  similarity: number;
  metadata: any;
  filePath?: string;
  mimeType?: string;
  createdAt: string;
  chunk?: {
    text: string;
    chunkIndex: number;
    totalChunks: number;
  };
}

export interface RAGResponse {
  answer: string;
  confidence: number;
  sources: VectorSearchResult[];
  model: string;
  tokensUsed?: number;
}

// POST /api/v1/cases/[caseId]/search - Semantic search within case evidence
export const POST: RequestHandler = async ({ params, request, getClientAddress }) => {
  try {
    const { caseId } = params;

    if (!caseId) {
      return json({
        success: false,
        error: 'caseId parameter is required'
      }, { status: 400 });
    }

    const searchRequest: SearchRequest = await request.json();
    
    if (!searchRequest.query?.trim()) {
      return json({
        success: false,
        error: 'query parameter is required'
      }, { status: 400 });
    }

    const query = searchRequest.query.trim();
    const limit = Math.min(searchRequest.limit || 10, 50);
    const threshold = searchRequest.threshold || 0.7;
    const includeRAG = searchRequest.includeRAG || false;

    console.log(`🔍 POST /api/v1/cases/${caseId}/search - Query: "${query}"`);

    // Check if case exists
    const [caseExists] = await db
      .select({ id: cases.id, title: cases.title })
      .from(cases)
      .where(eq(cases.id, caseId));

    if (!caseExists) {
      return json({
        success: false,
        error: 'Case not found'
      }, { status: 404 });
    }

    // Try cache first for exact query
    const cacheKey = `search:${caseId}:${Buffer.from(query).toString('base64')}:${limit}:${threshold}`;
    const cachedResults = await redis.get(cacheKey);
    
    if (cachedResults) {
      console.log(`✅ Search results retrieved from cache`);
      const parsed = JSON.parse(cachedResults);
      
      // If RAG was requested but not in cache, we'll need to generate it
      if (includeRAG && !parsed.rag) {
        const ragResponse = await generateRAGResponse(query, parsed.results, caseId);
        parsed.rag = ragResponse;
        
        // Update cache with RAG response
        await redis.setex(cacheKey, 1800, JSON.stringify(parsed)); // 30 minutes
      }
      
      return json({
        success: true,
        data: parsed,
        metadata: {
          timestamp: Date.now(),
          clientAddress: getClientAddress(),
          cacheStatus: 'hit'
        }
      });
    }

    // Step 1: Generate embedding for the query using our sentence transformer service
    const queryEmbedding = await generateQueryEmbedding(query);

    if (!queryEmbedding) {
      return json({
        success: false,
        error: 'Failed to generate query embedding'
      }, { status: 500 });
    }

    // Step 2: Perform vector similarity search across case evidence
    const vectorSearchResults = await performVectorSearch(
      caseId,
      queryEmbedding,
      limit,
      threshold,
      searchRequest.evidenceTypes
    );

    // Step 3: Generate RAG response if requested
    let ragResponse: RAGResponse | null = null;
    if (includeRAG && vectorSearchResults.length > 0) {
      ragResponse = await generateRAGResponse(query, vectorSearchResults, caseId);
    }

    const searchResults = {
      query,
      caseId,
      caseTitle: caseExists.title,
      results: vectorSearchResults,
      resultCount: vectorSearchResults.length,
      threshold,
      rag: ragResponse,
      searchMetadata: {
        embeddingModel: 'nomic-embed-text',
        vectorDimensions: 384,
        searchAlgorithm: 'cosine_similarity'
      }
    };

    // Cache the results for 30 minutes
    await redis.setex(cacheKey, 1800, JSON.stringify(searchResults));

    // Log search for analytics
    await redis.lpush(`search_log:${caseId}`, JSON.stringify({
      query,
      resultCount: vectorSearchResults.length,
      timestamp: Date.now(),
      clientAddress: getClientAddress()
    }));

    const response = {
      success: true,
      data: searchResults,
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        cacheStatus: 'miss',
        endpoint: `/api/v1/cases/${caseId}/search`
      }
    };

    console.log(`✅ Vector search completed: ${vectorSearchResults.length} results found`);
    return json(response);

  } catch (error: any) {
    console.error(`❌ POST /api/v1/cases/${params.caseId}/search error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Search failed',
      metadata: { timestamp: Date.now() }
    }, { status: 500 });
  }
};

// GET /api/v1/cases/[caseId]/search/history - Get search history for case
export const GET: RequestHandler = async ({ params, url, getClientAddress }) => {
  try {
    const { caseId } = params;
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '20'), 100);

    console.log(`📋 GET /api/v1/cases/${caseId}/search/history`);

    // Get search history from Redis
    const searchHistory = await redis.lrange(`search_log:${caseId}`, 0, limit - 1);
    
    const historyData = searchHistory.map(entry => {
      try {
        return JSON.parse(entry);
      } catch (e: any) {
        return null;
      }
    }).filter(Boolean);

    const response = {
      success: true,
      data: {
        caseId,
        history: historyData,
        count: historyData.length
      },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress()
      }
    };

    return json(response);

  } catch (error: any) {
    console.error(`❌ GET /api/v1/cases/${params.caseId}/search/history error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get search history'
    }, { status: 500 });
  }
};

/**
 * Generate embedding for search query using sentence transformer
 */
async function generateQueryEmbedding(query: string): Promise<number[] | null> {
  try {
    // Call our sentence transformer service
    const response = await fetch('http://localhost:8094/api/embed', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: query,
        model: 'nomic-embed-text'
      })
    });

    if (!response.ok) {
      throw new Error(`Embedding service error: ${response.statusText}`);
    }

    const result = await response.json();
    return result.embedding || null;

  } catch (error: any) {
    console.error('Failed to generate query embedding:', error);
    return null;
  }
}

/**
 * Perform vector similarity search using pgvector
 */
async function performVectorSearch(
  caseId: string,
  queryEmbedding: number[],
  limit: number,
  threshold: number,
  evidenceTypes?: string[]
): Promise<VectorSearchResult[]> {
  
  // Convert embedding array to pgvector format
  const embeddingStr = `[${queryEmbedding.join(',')}]`;

  // Build the base query with pgvector cosine similarity
  let query = db
    .select({
      evidenceId: evidence.id,
      evidenceTitle: evidence.title,
      evidenceContent: evidence.content,
      evidenceType: evidence.type,
      evidenceMetadata: evidence.metadata,
      evidenceFilePath: evidence.filePath,
      evidenceMimeType: evidence.mimeType,
      evidenceCreatedAt: evidence.createdAt,
      // Calculate cosine similarity using pgvector
      similarity: sql<number>`1 - (${evidence.embedding} <=> ${embeddingStr}::vector)`,
      // Join document data if available
      documentId: documents.id,
      documentContent: documents.extractedText,
      documentAnalysis: documents.analysis
    })
    .from(evidence)
    .leftJoin(documents, eq(evidence.id, documents.evidenceId))
    .where(
      and(
        eq(evidence.caseId, caseId),
        // Only include evidence with embeddings
        sql`${evidence.embedding} IS NOT NULL`,
        // Similarity threshold
        sql`1 - (${evidence.embedding} <=> ${embeddingStr}::vector) >= ${threshold}`
      )
    )
    .orderBy(desc(sql`1 - (${evidence.embedding} <=> ${embeddingStr}::vector)`))
    .limit(limit);

  // Add evidence type filter if specified
  if (evidenceTypes && evidenceTypes.length > 0) {
    query = query.where(
      and(
        eq(evidence.caseId, caseId),
        sql`${evidence.embedding} IS NOT NULL`,
        sql`1 - (${evidence.embedding} <=> ${embeddingStr}::vector) >= ${threshold}`,
        sql`${evidence.type} = ANY(${evidenceTypes})`
      )
    );
  }

  const results = await query;

  return results.map(row => ({
    id: row.evidenceId,
    title: row.evidenceTitle,
    content: row.documentContent || row.evidenceContent || '',
    evidenceType: row.evidenceType,
    similarity: row.similarity,
    metadata: row.evidenceMetadata,
    filePath: row.evidenceFilePath,
    mimeType: row.evidenceMimeType,
    createdAt: row.evidenceCreatedAt.toISOString(),
    // Include document analysis if available
    ...(row.documentAnalysis && { analysis: row.documentAnalysis })
  }));
}

/**
 * Generate RAG response using search results as context
 */
async function generateRAGResponse(
  query: string,
  searchResults: VectorSearchResult[],
  caseId: string
): Promise<RAGResponse | null> {
  try {
    // Prepare context from search results
    const context = searchResults
      .slice(0, 5) // Use top 5 results for context
      .map((result, index) => 
        `[Source ${index + 1}: ${result.title}]\n${result.content.substring(0, 500)}...`
      )
      .join('\n\n');

    const prompt = `Based on the following evidence from the case, please answer the question.

Evidence Context:
${context}

Question: ${query}

Please provide a comprehensive answer based only on the evidence provided. If the evidence doesn't contain enough information to answer the question, please state that clearly.`;

    // Call RAG service
    const response = await fetch('http://localhost:8094/api/rag', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: prompt,
        model: 'gemma3-legal',
        context: {
          caseId,
          evidenceCount: searchResults.length
        }
      })
    });

    if (!response.ok) {
      throw new Error(`RAG service error: ${response.statusText}`);
    }

    const result = await response.json();

    return {
      answer: result.answer || result.response || 'No response generated',
      confidence: result.confidence || 0.8,
      sources: searchResults.slice(0, 5),
      model: 'gemma3-legal',
      tokensUsed: result.tokensUsed
    };

  } catch (error: any) {
    console.error('Failed to generate RAG response:', error);
    return null;
  }
}