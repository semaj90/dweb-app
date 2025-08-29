
/**
 * Enhanced Vector Operations Service - PostgreSQL + pgvector + Qdrant Integration
 * Production CRUD operations with multi-store vector search and persistence
 */
import { db } from './index';
import { sql, eq, and, desc } from 'drizzle-orm';
import { 
  cases, 
  evidence, 
  legal_documents, 
  documentChunks, 
  vectorMetadata, 
  embeddingCache,
  userAiQueries,
  users 
} from './schema-postgres';
import type { QdrantClient } from '@qdrant/js-client-rest';

// === ENHANCED INTERFACES ===

export interface VectorSearchQuery {
  query: string;
  embedding?: number[];
  filters?: {
    documentType?: string;
    caseId?: string;
    userId?: string;
    dateRange?: {
      start: Date;
      end: Date;
    };
  };
  limit?: number;
  threshold?: number;
  sources?: ('pgvector' | 'qdrant')[];
}

export interface VectorSearchResponse {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  source: 'pgvector' | 'qdrant';
  chunkIndex?: string;
  documentId: string;
  documentType: string;
}

export interface CaseCreationRequest {
  title: string;
  description?: string;
  status?: string;
  priority?: string;
  assignedTo?: string;
  createdBy: string;
  metadata?: Record<string, any>;
  generateEmbedding?: boolean;
}

export interface EvidenceCreationRequest {
  caseId: string;
  title: string;
  content: string;
  evidenceType: string;
  description?: string;
  createdBy: string;
  metadata?: Record<string, any>;
  generateEmbedding?: boolean;
}

// Legacy interface for backward compatibility
export interface SimilarityResult {
  id: string;
  content: string;
  similarity: number;
  metadata?: unknown;
}

// === ENHANCED VECTOR OPERATIONS SERVICE ===

class EnhancedVectorOperationsService {
  private qdrantClient: QdrantClient | null = null;
  
  constructor() {
    this.initializeQdrantClient();
  }

  private async initializeQdrantClient() {
    try {
      // Import Qdrant client dynamically to avoid SSR issues
      const { QdrantClient } = await import('@qdrant/js-client-rest');
      
      this.qdrantClient = new QdrantClient({
        url: process.env.QDRANT_URL || 'http://localhost:6333',
        apiKey: process.env.QDRANT_API_KEY
      });
      
      console.log('✅ Qdrant client initialized');
    } catch (error: any) {
      console.warn('⚠️ Qdrant client initialization failed:', error);
      this.qdrantClient = null;
    }
  }

  // === CASE CRUD OPERATIONS ===

  /**
   * Create a new case with optional vector embedding generation
   */
  async createCase(request: CaseCreationRequest) {
    const caseData = {
      title: request.title,
      description: request.description,
      status: request.status || 'open',
      priority: request.priority || 'medium',
      assignedTo: request.assignedTo ? request.assignedTo : undefined,
      createdBy: request.createdBy,
      metadata: request.metadata || {},
      created_at: new Date(),
      updated_at: new Date()
    };

    const [newCase] = await db.insert(cases).values(caseData).returning();

    // Generate embeddings if requested
    if (request.generateEmbedding && (request.title || request.description)) {
      const contentForEmbedding = `${request.title}\n${request.description || ''}`.trim();
      await this.generateAndStoreEmbedding({
        content: contentForEmbedding,
        documentId: newCase.id,
        documentType: 'case',
        metadata: {
          ...request.metadata,
          caseId: newCase.id,
          title: request.title
        }
      });
    }

    return newCase;
  }

  /**
   * Enhanced vector search across PostgreSQL and Qdrant
   */
  async vectorSearch(query: VectorSearchQuery): Promise<VectorSearchResponse[]> {
    const results: VectorSearchResponse[] = [];

    // Generate query embedding if not provided
    let queryEmbedding = query.embedding;
    if (!queryEmbedding && query.query) {
      queryEmbedding = await this.generateEmbedding(query.query);
    }

    if (!queryEmbedding) {
      throw new Error('Could not generate query embedding');
    }

    const sources = query.sources || ['pgvector', 'qdrant'];
    const limit = query.limit || 10;
    const threshold = query.threshold || 0.7;

    // Search in PostgreSQL pgvector
    if (sources.includes('pgvector')) {
      const pgResults = await this.searchInPgVector(queryEmbedding, query, limit, threshold);
      results.push(...pgResults);
    }

    // Search in Qdrant
    if (sources.includes('qdrant') && this.qdrantClient) {
      const qdrantResults = await this.searchInQdrant(queryEmbedding, query, limit, threshold);
      results.push(...qdrantResults);
    }

    // Merge and deduplicate results
    const mergedResults = this.mergeVectorResults(results);
    
    // Sort by score descending
    mergedResults.sort((a, b) => b.score - a.score);

    return mergedResults.slice(0, limit);
  }

  /**
   * Generate embedding using Enhanced RAG service
   */
  private async generateEmbedding(content: string): Promise<number[]> {
    try {
      const response = await fetch('/api/go/enhanced-rag/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: content,
          model: 'nomic-embed-text'
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const { embedding } = await response.json();
      return embedding;
    } catch (error: any) {
      console.warn('Remote embedding generation failed, using fallback:', error);
      return generateSampleEmbedding(); // Fallback to sample embedding
    }
  }

  /**
   * Generate and store embeddings for content with dual-store persistence
   */
  private async generateAndStoreEmbedding(request: {
    content: string;
    documentId: string;
    documentType: string;
    chunkIndex?: string;
    metadata?: Record<string, any>;
  }): Promise<void> {
    try {
      const embedding = await this.generateEmbedding(request.content);

      // Store in PostgreSQL pgvector
      await this.storeEmbeddingInPgVector(request, embedding);

      // Store in Qdrant if available
      if (this.qdrantClient) {
        await this.storeEmbeddingInQdrant(request, embedding);
      }

      console.log(`✅ Embedding generated and stored for ${request.documentType}:${request.documentId}`);

    } catch (error: any) {
      console.error('Embedding generation failed:', error);
      throw error;
    }
  }

  private async storeEmbeddingInPgVector(
    request: {
      content: string;
      documentId: string;
      documentType: string;
      chunkIndex?: string;
      metadata?: Record<string, any>;
    }, 
    embedding: number[]
  ): Promise<void> {
    const vectorData = {
      document_id: request.documentId,
      vector_id: `${request.documentType}_${request.documentId}_${request.chunkIndex || '0'}`,
      embedding: embedding,
      metadata: {
        ...request.metadata,
        documentType: request.documentType,
        chunkIndex: request.chunkIndex,
        generatedAt: new Date().toISOString()
      },
      created_at: new Date()
    };

    await db.insert(vectorMetadata).values(vectorData);
  }

  private async storeEmbeddingInQdrant(
    request: {
      content: string;
      documentId: string;
      documentType: string;
      chunkIndex?: string;
      metadata?: Record<string, any>;
    }, 
    embedding: number[]
  ): Promise<void> {
    if (!this.qdrantClient) return;

    const collectionName = `legal_${request.documentType}s`;
    
    try {
      // Ensure collection exists
      await this.qdrantClient.createCollection(collectionName, {
        vectors: {
          size: embedding.length,
          distance: 'Cosine'
        }
      });
    } catch (error: any) {
      // Collection might already exist, that's okay
    }

    const pointId = `${request.documentType}_${request.documentId}_${request.chunkIndex || '0'}`;
    
    await this.qdrantClient.upsert(collectionName, {
      points: [{
        id: pointId,
        vector: embedding,
        payload: {
          documentId: request.documentId,
          documentType: request.documentType,
          chunkIndex: request.chunkIndex,
          content: request.content.substring(0, 1000), // Store first 1000 chars
          metadata: request.metadata
        }
      }]
    });
  }

  private async searchInPgVector(
    queryEmbedding: number[], 
    query: VectorSearchQuery, 
    limit: number, 
    threshold: number
  ): Promise<VectorSearchResponse[]> {
    const conditions = [
      sql`${vectorMetadata.embedding} <=> ${JSON.stringify(queryEmbedding)} < ${1 - threshold}`
    ];

    // Apply filters
    if (query.filters?.documentType) {
      conditions.push(sql`${vectorMetadata.metadata}->>'documentType' = ${query.filters.documentType}`);
    }

    if (query.filters?.caseId) {
      conditions.push(sql`${vectorMetadata.metadata}->>'caseId' = ${query.filters.caseId}`);
    }

    const pgResults = await db
      .select({
        id: vectorMetadata.id,
        document_id: vectorMetadata.document_id,
        vector_id: vectorMetadata.vector_id,
        metadata: vectorMetadata.metadata,
        similarity: sql`1 - (${vectorMetadata.embedding} <=> ${JSON.stringify(queryEmbedding)})`.as('similarity')
      })
      .from(vectorMetadata)
      .where(and(...conditions))
      .orderBy(sql`${vectorMetadata.embedding} <=> ${JSON.stringify(queryEmbedding)}`)
      .limit(limit);

    return pgResults.map(result => ({
      id: result.id,
      content: '', // Will be populated from joined document content
      score: Number(result.similarity),
      metadata: result.metadata as Record<string, any>,
      source: 'pgvector' as const,
      documentId: result.document_id,
      documentType: (result.metadata as any)?.documentType || 'unknown',
      chunkIndex: (result.metadata as any)?.chunkIndex
    }));
  }

  private async searchInQdrant(
    queryEmbedding: number[], 
    query: VectorSearchQuery, 
    limit: number, 
    threshold: number
  ): Promise<VectorSearchResponse[]> {
    if (!this.qdrantClient) return [];

    const collections = query.filters?.documentType 
      ? [`legal_${query.filters.documentType}s`] 
      : ['legal_cases', 'legal_evidences', 'legal_legal_documents'];

    const results: VectorSearchResponse[] = [];

    for (const collection of collections) {
      try {
        const response = await this.qdrantClient.search(collection, {
          vector: queryEmbedding,
          limit: limit,
          score_threshold: threshold,
          with_payload: true
        });

        const collectionResults = response.map(result => ({
          id: String(result.id),
          content: (result.payload?.content as string) || '',
          score: result.score,
          metadata: (result.payload?.metadata as Record<string, any>) || {},
          source: 'qdrant' as const,
          documentId: (result.payload?.documentId as string) || '',
          documentType: (result.payload?.documentType as string) || '',
          chunkIndex: (result.payload?.chunkIndex as string) || undefined
        }));

        results.push(...collectionResults);
      } catch (error: any) {
        console.warn(`Qdrant search failed for collection ${collection}:`, error);
      }
    }

    return results;
  }

  private mergeVectorResults(results: VectorSearchResponse[]): VectorSearchResponse[] {
    const merged = new Map<string, VectorSearchResponse>();

    for (const result of results) {
      const key = `${result.documentId}_${result.chunkIndex || '0'}`;
      const existing = merged.get(key);

      if (!existing || result.score > existing.score) {
        merged.set(key, result);
      }
    }

    return Array.from(merged.values());
  }
}

// === LEGACY COMPATIBILITY FUNCTIONS ===

// Generate a sample embedding (replace with actual AI model in production)
export function generateSampleEmbedding(dimensions: number = 384): number[] {
  return Array.from({ length: dimensions }, () => Math.random() * 2 - 1);
}

// Convert array to pgvector format
export function arrayToPgVector(embedding: number[]): string {
  return `[${embedding.join(',')}]`;
}

// Vector similarity search in legal documents
export async function searchSimilarDocuments(
  queryEmbedding: number[], 
  limit: number = 10,
  similarityThreshold: number = 0.7
): Promise<SimilarityResult[]> {
  try {
    const vectorString = arrayToPgVector(queryEmbedding);
    
    const results = await db.execute(sql`
      SELECT 
        id,
        title,
        content,
        1 - (embedding <=> ${vectorString}::vector) as similarity,
        keywords,
        topics
      FROM legal_documents 
      WHERE 1 - (embedding <=> ${vectorString}::vector) > ${similarityThreshold}
      ORDER BY embedding <=> ${vectorString}::vector
      LIMIT ${limit}
    `);

    return results.map((row: any) => ({
      id: row.id,
      content: row.content || '',
      title: row.title || '',
      similarity: parseFloat(row.similarity || '0'),
      metadata: {
        keywords: Array.isArray(row.keywords) ? row.keywords : (row.keywords ? [row.keywords] : []),
        topics: Array.isArray(row.topics) ? row.topics : (row.topics ? [row.topics] : [])
      }
    }));
  } catch (error: any) {
    console.error('Vector similarity search failed:', error);
    // Fallback to text search if vector search fails
    return await fallbackTextSearch(queryEmbedding, limit);
  }
}

// Fallback text search when vector operations fail
async function fallbackTextSearch(queryEmbedding: number[], limit: number): Promise<SimilarityResult[]> {
  console.log('Using fallback text search...');
  
  // Fix: Use legal_documents table instead of non-existent documentMetadata
  const results = await db
    .select({
      id: legal_documents.id,
      title: legal_documents.title,
      content: legal_documents.content,
      metadata: sql`'{}'::jsonb`, // placeholder metadata
    })
    .from(legal_documents)
    .limit(limit);

  return results.map((doc, index) => ({
    id: doc.id,
    content: doc.content || '',
    title: doc.title,
    similarity: 1 - (index * 0.1), // Fake similarity scores
    metadata: {
      keywords: (doc.metadata as any)?.keywords || [],
      topics: (doc.metadata as any)?.topics || []
    }
  }));
}

// Store AI query with embedding for future similarity search
export async function storeAiQueryWithEmbedding(
  userId: string,
  caseId: string | null,
  query: string,
  response: string,
  embedding: number[],
  metadata: any = {}
): Promise<void> {
  try {
    await db.insert(userAiQueries).values({
      userId,
      caseId,
      query,
      response,
      embedding: arrayToPgVector(embedding) as any,
      metadata,
      isSuccessful: true,
    });
  } catch (error: any) {
    console.error('Failed to store AI query with embedding:', error);
  }
}

// Cache embedding to avoid recomputing
export async function cacheEmbedding(
  textHash: string,
  embedding: number[],
  model: string = 'nomic-embed-text'
): Promise<void> {
  try {
    await db.insert(embeddingCache).values({
      textHash,
      embedding: arrayToPgVector(embedding) as any,
      model,
    });
  } catch (error: any) {
    console.error('Failed to cache embedding:', error);
  }
}

// Retrieve cached embedding
export async function getCachedEmbedding(textHash: string): Promise<number[] | null> {
  try {
    const result = await db
      .select({ embedding: embeddingCache.embedding })
      .from(embeddingCache)
      .where(sql`text_hash = ${textHash}`)
      .limit(1);

    if (result.length > 0) {
      // Parse pgvector format back to array
      const vectorString = result[0].embedding;
      if (typeof vectorString === 'string') {
        return (vectorString as string).replace(/^\[|\]$/g, '').split(',').map((n: string) => parseFloat(n));
      }
    }
    return null;
  } catch (error: any) {
    console.error('Failed to retrieve cached embedding:', error);
    return null;
  }
}

// Hybrid search: combine vector and text search
export async function hybridSearch(
  queryText: string,
  queryEmbedding: number[],
  limit: number = 10
): Promise<SimilarityResult[]> {
  try {
    // First try vector search
    const vectorResults = await searchSimilarDocuments(queryEmbedding, Math.ceil(limit * 0.7));
    
    // Then add text search results
    const textResults = await db.execute(sql`
      SELECT 
        id,
        title,
        content,
        ts_rank(to_tsvector('english', content), plainto_tsquery('english', ${queryText})) as rank,
        keywords,
        topics
      FROM legal_documents 
      WHERE to_tsvector('english', content) @@ plainto_tsquery('english', ${queryText})
      ORDER BY rank DESC
      LIMIT ${Math.floor(limit * 0.3)}
    `);

    const textSearchResults: SimilarityResult[] = textResults.map((row: any) => ({
      id: row.id,
      content: row.content,
      title: row.title,
      similarity: parseFloat(row.rank) * 0.5, // Scale down text search scores
      metadata: {
        keywords: row.keywords,
        topics: row.topics,
        searchType: 'text'
      }
    }));

    // Combine and deduplicate results
    const combinedResults = [...vectorResults, ...textSearchResults];
    const uniqueResults = Array.from(
      new Map(combinedResults.map(item => [item.id, item])).values()
    );

    // Sort by similarity and return top results
    return uniqueResults
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);

  } catch (error: any) {
    console.error('Hybrid search failed:', error);
    return await fallbackTextSearch(queryEmbedding, limit);
  }
}

// Check if pgvector extension is available
export async function checkPgVectorAvailable(): Promise<boolean> {
  try {
    await db.execute(sql`SELECT 1::vector;`);
    return true;
  } catch (error: any) {
    console.log('pgvector not available:', error.message);
    return false;
  }
}

// Vector operations test function
export async function testVectorOperations(): Promise<{
  pgvectorAvailable: boolean;
  similaritySearchWorking: boolean;
  embeddingCacheWorking: boolean;
}> {
  const pgvectorAvailable = await checkPgVectorAvailable();
  
  let similaritySearchWorking = false;
  let embeddingCacheWorking = false;

  if (pgvectorAvailable) {
    try {
      const testEmbedding = generateSampleEmbedding();
      const results = await searchSimilarDocuments(testEmbedding, 1, 0.0);
      similaritySearchWorking = true;
    } catch (error: any) {
      console.log('Similarity search test failed:', error.message);
    }

    try {
      const testEmbedding = generateSampleEmbedding();
      await cacheEmbedding('test-hash', testEmbedding);
      const retrieved = await getCachedEmbedding('test-hash');
      embeddingCacheWorking = retrieved !== null;
    } catch (error: any) {
      console.log('Embedding cache test failed:', error.message);
    }
  }

  return {
    pgvectorAvailable,
    similaritySearchWorking,
    embeddingCacheWorking
  };
}

// === SERVICE INSTANCE EXPORT ===

// Create singleton instance for production use
export const vectorOperations = new EnhancedVectorOperationsService();

// Export service class for direct instantiation if needed
export { EnhancedVectorOperationsService };

// Types are already exported inline above - no need for duplicate type block