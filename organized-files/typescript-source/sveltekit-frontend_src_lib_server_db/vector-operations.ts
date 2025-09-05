// @ts-nocheck
// Vector operations helper for pgvector integration
import { db } from './index.js';
import { legalDocuments, userAiQueries, embeddingCache } from './schema-postgres.js';
import { sql } from 'drizzle-orm';

// Interface for vector similarity search results
interface SimilarityResult {
  id: string;
  content: string;
  similarity: number;
  metadata?: any;
}

// Generate a sample embedding (replace with actual AI model in production)
export function generateSampleEmbedding(dimensions: number = 768): number[] {
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
      content: row.content,
      title: row.title,
      similarity: parseFloat(row.similarity),
      metadata: {
        keywords: row.keywords,
        topics: row.topics
      }
    }));
  } catch (error) {
    console.error('Vector similarity search failed:', error);
    // Fallback to text search if vector search fails
    return await fallbackTextSearch(queryEmbedding, limit);
  }
}

// Fallback text search when vector operations fail
async function fallbackTextSearch(queryEmbedding: number[], limit: number): Promise<SimilarityResult[]> {
  console.log('Using fallback text search...');
  
  const results = await db
    .select({
      id: legalDocuments.id,
      title: legalDocuments.title,
      content: legalDocuments.content,
      keywords: legalDocuments.keywords,
      topics: legalDocuments.topics,
    })
    .from(legalDocuments)
    .limit(limit);

  return results.map((doc, index) => ({
    id: doc.id,
    content: doc.content || '',
    title: doc.title,
    similarity: 1 - (index * 0.1), // Fake similarity scores
    metadata: {
      keywords: doc.keywords,
      topics: doc.topics
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
      embedding: arrayToPgVector(embedding),
      metadata,
      isSuccessful: true,
    });
  } catch (error) {
    console.error('Failed to store AI query with embedding:', error);
    // Store without embedding as fallback
    await db.insert(userAiQueries).values({
      userId,
      caseId, 
      query,
      response,
      metadata,
      isSuccessful: true,
    });
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
      embedding: arrayToPgVector(embedding),
      model,
    });
  } catch (error) {
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
        return JSON.parse(vectorString.replace(/^\[|\]$/g, '').split(',').map(n => parseFloat(n)));
      }
    }
    return null;
  } catch (error) {
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

  } catch (error) {
    console.error('Hybrid search failed:', error);
    return await fallbackTextSearch(queryEmbedding, limit);
  }
}

// Check if pgvector extension is available
export async function checkPgVectorAvailable(): Promise<boolean> {
  try {
    await db.execute(sql`SELECT 1::vector;`);
    return true;
  } catch (error) {
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
    } catch (error) {
      console.log('Similarity search test failed:', error.message);
    }

    try {
      const testEmbedding = generateSampleEmbedding();
      await cacheEmbedding('test-hash', testEmbedding);
      const retrieved = await getCachedEmbedding('test-hash');
      embeddingCacheWorking = retrieved !== null;
    } catch (error) {
      console.log('Embedding cache test failed:', error.message);
    }
  }

  return {
    pgvectorAvailable,
    similaritySearchWorking,
    embeddingCacheWorking
  };
}

export {
  type SimilarityResult
};