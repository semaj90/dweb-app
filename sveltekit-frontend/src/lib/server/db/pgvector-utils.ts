/**
 * PostgreSQL pgvector utilities for vector operations
 * Provides proper vector similarity search and embedding operations
 */

import { db, sql } from './index.js';
import type { 
  chatMessages, 
  chatRecommendations, 
  evidence, 
  documentEmbeddings 
} from './schema-unified.js';

export interface VectorSearchResult {
  id: string;
  content: string;
  similarity: number;
  metadata?: any;
  documentType?: string;
}

export interface VectorSearchOptions {
  limit?: number;
  threshold?: number;
  includeMetadata?: boolean;
}

/**
 * Initialize pgvector extension and create necessary functions
 */
export async function initializePgVector(): Promise<boolean> {
  try {
    // Enable pgvector extension
    await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vector`);
    
    // Create cosine distance function if it doesn't exist
    await db.execute(sql`
      CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
      RETURNS float AS $$
      SELECT 1 - (a <=> b)
      $$ LANGUAGE SQL IMMUTABLE STRICT;
    `);

    // Create vector search function for chat messages
    await db.execute(sql`
      CREATE OR REPLACE FUNCTION search_similar_messages(
        query_embedding vector(384),
        similarity_threshold float DEFAULT 0.7,
        result_limit int DEFAULT 10
      )
      RETURNS TABLE (
        id uuid,
        content text,
        similarity float,
        metadata jsonb,
        created_at timestamp
      ) AS $$
      SELECT 
        chat_messages.id,
        chat_messages.content,
        cosine_similarity(chat_messages.embedding, query_embedding) as similarity,
        chat_messages.metadata,
        chat_messages.created_at
      FROM chat_messages
      WHERE chat_messages.embedding IS NOT NULL
        AND cosine_similarity(chat_messages.embedding, query_embedding) > similarity_threshold
      ORDER BY similarity DESC
      LIMIT result_limit;
      $$ LANGUAGE SQL STABLE;
    `);

    // Create vector search function for evidence
    await db.execute(sql`
      CREATE OR REPLACE FUNCTION search_similar_evidence(
        query_embedding vector(384),
        case_id_filter uuid DEFAULT NULL,
        similarity_threshold float DEFAULT 0.7,
        result_limit int DEFAULT 10
      )
      RETURNS TABLE (
        id uuid,
        title varchar,
        description text,
        similarity float,
        case_id uuid,
        evidence_type varchar,
        metadata jsonb
      ) AS $$
      SELECT 
        e.id,
        e.title,
        e.description,
        GREATEST(
          cosine_similarity(e.title_embedding, query_embedding),
          cosine_similarity(e.content_embedding, query_embedding)
        ) as similarity,
        e.case_id,
        e.evidence_type,
        e.ai_analysis
      FROM evidence e
      WHERE (e.title_embedding IS NOT NULL OR e.content_embedding IS NOT NULL)
        AND (case_id_filter IS NULL OR e.case_id = case_id_filter)
        AND GREATEST(
          COALESCE(cosine_similarity(e.title_embedding, query_embedding), 0),
          COALESCE(cosine_similarity(e.content_embedding, query_embedding), 0)
        ) > similarity_threshold
      ORDER BY similarity DESC
      LIMIT result_limit;
      $$ LANGUAGE SQL STABLE;
    `);

    console.log('✅ pgvector utilities initialized successfully');
    return true;
  } catch (error: any) {
    console.error('❌ Failed to initialize pgvector:', error);
    return false;
  }
}

/**
 * Convert JavaScript array to PostgreSQL vector format
 */
export function arrayToVector(embedding: number[]): string {
  if (!Array.isArray(embedding) || embedding.length === 0) {
    throw new Error('Invalid embedding: must be a non-empty array');
  }
  
  // Ensure all values are finite numbers
  const validEmbedding = embedding.map(val => {
    if (!isFinite(val)) return 0;
    return val;
  });
  
  return `[${validEmbedding.join(',')}]`;
}

/**
 * Convert PostgreSQL vector to JavaScript array
 */
export function vectorToArray(vectorString: string): number[] {
  if (!vectorString || typeof vectorString !== 'string') {
    return [];
  }
  
  try {
    // Remove brackets and split by comma
    const cleaned = vectorString.replace(/^\[|\]$/g, '');
    return cleaned.split(',').map(val => parseFloat(val.trim()));
  } catch (error: any) {
    console.warn('Failed to parse vector string:', vectorString);
    return [];
  }
}

/**
 * Search for similar chat messages using vector similarity
 */
export async function searchSimilarMessages(
  queryEmbedding: number[],
  options: VectorSearchOptions = {}
): Promise<VectorSearchResult[]> {
  const {
    limit = 10,
    threshold = 0.7,
    includeMetadata = true
  } = options;

  try {
    const vectorString = arrayToVector(queryEmbedding);
    
    const results = await db.execute(sql`
      SELECT * FROM search_similar_messages(
        ${vectorString}::vector,
        ${threshold}::float,
        ${limit}::int
      )
    `);

    return results.map((row: any) => ({
      id: row.id,
      content: row.content,
      similarity: row.similarity,
      metadata: includeMetadata ? row.metadata : undefined,
      documentType: 'chat_message'
    }));
  } catch (error: any) {
    console.error('Vector search for messages failed:', error);
    return [];
  }
}

/**
 * Search for similar evidence using vector similarity
 */
export async function searchSimilarEvidence(
  queryEmbedding: number[],
  caseId?: string,
  options: VectorSearchOptions = {}
): Promise<VectorSearchResult[]> {
  const {
    limit = 10,
    threshold = 0.7,
    includeMetadata = true
  } = options;

  try {
    const vectorString = arrayToVector(queryEmbedding);
    const caseIdParam = caseId ? `'${caseId}'::uuid` : 'NULL::uuid';
    
    const results = await db.execute(sql`
      SELECT * FROM search_similar_evidence(
        ${vectorString}::vector,
        ${sql.raw(caseIdParam)},
        ${threshold}::float,
        ${limit}::int
      )
    `);

    return results.map((row: any) => ({
      id: row.id,
      content: row.description || row.title || '',
      similarity: row.similarity,
      metadata: includeMetadata ? {
        title: row.title,
        evidenceType: row.evidence_type,
        caseId: row.case_id,
        ...row.metadata
      } : undefined,
      documentType: 'evidence'
    }));
  } catch (error: any) {
    console.error('Vector search for evidence failed:', error);
    return [];
  }
}

/**
 * Insert chat message with vector embedding
 */
export async function insertChatMessageWithEmbedding(
  messageData: {
    id: string;
    sessionId: string;
    role: string;
    content: string;
    embedding: number[];
    metadata?: any;
  }
): Promise<boolean> {
  try {
    const vectorString = arrayToVector(messageData.embedding);
    
    await db.execute(sql`
      INSERT INTO chat_messages (
        id, session_id, role, content, embedding, metadata
      ) VALUES (
        ${messageData.id}::uuid,
        ${messageData.sessionId}::uuid,
        ${messageData.role},
        ${messageData.content},
        ${vectorString}::vector,
        ${JSON.stringify(messageData.metadata || {})}::jsonb
      )
    `);
    
    return true;
  } catch (error: any) {
    console.error('Failed to insert chat message with embedding:', error);
    return false;
  }
}

/**
 * Update evidence with embeddings
 */
export async function updateEvidenceEmbeddings(
  evidenceId: string,
  titleEmbedding?: number[],
  contentEmbedding?: number[]
): Promise<boolean> {
  try {
    const updates: string[] = [];
    const params: any[] = [];

    if (titleEmbedding && titleEmbedding.length > 0) {
      updates.push('title_embedding = $' + (params.length + 2));
      params.push(`${arrayToVector(titleEmbedding)}::vector`);
    }

    if (contentEmbedding && contentEmbedding.length > 0) {
      updates.push('content_embedding = $' + (params.length + 2));
      params.push(`${arrayToVector(contentEmbedding)}::vector`);
    }

    if (updates.length === 0) {
      return false;
    }

    await db.execute(sql`
      UPDATE evidence 
      SET ${sql.raw(updates.join(', '))}
      WHERE id = ${evidenceId}::uuid
    `);
    
    return true;
  } catch (error: any) {
    console.error('Failed to update evidence embeddings:', error);
    return false;
  }
}

/**
 * Batch search across multiple vector tables
 */
export async function searchAcrossAllVectors(
  queryEmbedding: number[],
  options: VectorSearchOptions & { 
    includeMessages?: boolean;
    includeEvidence?: boolean;
    caseId?: string;
  } = {}
): Promise<VectorSearchResult[]> {
  const {
    limit = 20,
    threshold = 0.6,
    includeMessages = true,
    includeEvidence = true,
    caseId
  } = options;

  const searchPromises: Promise<VectorSearchResult[]>[] = [];

  // Search messages
  if (includeMessages) {
    searchPromises.push(
      searchSimilarMessages(queryEmbedding, { 
        limit: Math.ceil(limit / 2), 
        threshold 
      })
    );
  }

  // Search evidence
  if (includeEvidence) {
    searchPromises.push(
      searchSimilarEvidence(queryEmbedding, caseId, { 
        limit: Math.ceil(limit / 2), 
        threshold 
      })
    );
  }

  try {
    const results = await Promise.all(searchPromises);
    const combined = results.flat();
    
    // Sort by similarity and limit results
    return combined
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  } catch (error: any) {
    console.error('Batch vector search failed:', error);
    return [];
  }
}

/**
 * Calculate cosine similarity between two vectors
 */
export function calculateCosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimension');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Health check for pgvector functionality
 */
export async function pgvectorHealthCheck(): Promise<{
  available: boolean;
  version?: string;
  functions: string[];
  error?: string;
}> {
  try {
    // Check if pgvector extension exists
    const extensionCheck = await db.execute(sql`
      SELECT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
      ) as has_vector, 
      (SELECT extversion FROM pg_extension WHERE extname = 'vector') as version
    `);

    if (!extensionCheck[0]?.has_vector) {
      return {
        available: false,
        functions: [],
        error: 'pgvector extension not installed'
      };
    }

    // Check if our custom functions exist
    const functionsCheck = await db.execute(sql`
      SELECT routine_name 
      FROM information_schema.routines 
      WHERE routine_schema = 'public' 
        AND routine_name IN (
          'cosine_similarity', 
          'search_similar_messages', 
          'search_similar_evidence'
        )
    `);

    const availableFunctions = functionsCheck.map((row: any) => row.routine_name);

    return {
      available: true,
      version: (extensionCheck[0]?.version as string) || 'unknown',
      functions: availableFunctions
    };
  } catch (error: any) {
    return {
      available: false,
      functions: [],
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

// Initialize on import (only in non-production)
if (typeof process !== 'undefined' && import.meta.env.NODE_ENV !== 'production') {
  initializePgVector().catch(error => {
    console.warn('pgvector initialization failed:', error);
  });
}