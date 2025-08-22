// Enhanced Vector Service with Nomic Embeddings
import { QdrantClient } from '@qdrant/js-client-rest';
import { db } from '../db';
import { units, documents, documentChunks } from '../db/schema';
import { eq, sql } from 'drizzle-orm';
import fetch from 'node-fetch';
import { env } from '$env/dynamic/private';

const QDRANT_URL = env.QDRANT_URL || 'http://localhost:6333';
const NOMIC_API_KEY = env.NOMIC_API_KEY || '';
const NOMIC_API_URL = 'https://api-atlas.nomic.ai/v1/embedding';

// Nomic embedding dimensions
const EMBEDDING_DIMENSION = 768;
const NOMIC_MODEL = 'nomic-embed-text-v1.5';

export class EnhancedVectorService {
  private qdrantClient: QdrantClient;

  // Collection names with updated dimensions
  private collections = {
    users: 'yorha_users_v2',
    activities: 'yorha_activities_v2',
    missions: 'yorha_missions_v2',
    knowledge: 'yorha_knowledge_v2',
    documents: 'yorha_documents_v2',
    conversations: 'yorha_conversations_v2'
  };

  // Chunk settings for RAG
  private chunkSettings = {
    maxChunkSize: 512,
    chunkOverlap: 128,
    minChunkSize: 100
  };

  constructor() {
    this.qdrantClient = new QdrantClient({
      url: QDRANT_URL,
      apiKey: env.QDRANT_API_KEY
    });

    this.initializeCollections();
  }

  // Initialize Qdrant collections with Nomic dimensions
  private async initializeCollections(): Promise<void> {
    try {
      for (const [name, collectionName] of Object.entries(this.collections)) {
        await this.createCollectionIfNotExists(collectionName, EMBEDDING_DIMENSION);
      }
      console.log('✅ Qdrant collections initialized with Nomic embeddings');
    } catch (error) {
      console.error('Failed to initialize Qdrant collections:', error);
    }
  }

  // Create collection with hybrid search capabilities
  private async createCollectionIfNotExists(name: string, vectorSize: number): Promise<void> {
    try {
      const collections = await this.qdrantClient.getCollections();
      const exists = collections.collections.some(c => c.name === name);

      if (!exists) {
        await this.qdrantClient.createCollection(name, {
          vectors: {
            size: vectorSize,
            distance: 'Cosine'
          },
          optimizers_config: {
            default_segment_number: 2,
            indexing_threshold: 20000
          },
          hnsw_config: {
            m: 16,
            ef_construct: 100,
            full_scan_threshold: 10000
          },
          quantization_config: {
            scalar: {
              type: 'int8',
              quantile: 0.99,
              always_ram: true
            }
          }
        });

        // Create payload indexes for hybrid search
        await this.createPayloadIndexes(name);

        console.log(`Created collection: ${name} with dimension ${vectorSize}`);
      }
    } catch (error) {
      console.error(`Failed to create collection ${name}:`, error);
    }
  }

  // Create payload indexes for better filtering
  private async createPayloadIndexes(collectionName: string): Promise<void> {
    const indexFields = ['userId', 'type', 'category', 'timestamp', 'source', 'metadata.keywords'];

    for (const field of indexFields) {
      try {
        // Use createPayloadIndex instead of createFieldIndex
        await this.qdrantClient.createPayloadIndex(collectionName, {
          field_name: field,
          field_schema: { type: 'keyword' }
        });
      } catch (error) {
        // Index might already exist, skip silently
        console.log(`Index for field ${field} may already exist`);
      }
    }
  }

  // Generate embedding using Nomic API
  async generateEmbedding(text: string, taskType: 'search_document' | 'search_query' = 'search_document'): Promise<number[]> {
    try {
      const response = await fetch(NOMIC_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${NOMIC_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: NOMIC_MODEL,
          texts: [text],
          task_type: taskType
        })
      });

      if (!response.ok) {
        throw new Error(`Nomic API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.embeddings[0];
    } catch (error) {
      console.error('Failed to generate Nomic embedding:', error);
      throw error;
    }
  }

  // Batch generate embeddings
  async generateBatchEmbeddings(texts: string[], taskType: 'search_document' | 'search_query' = 'search_document'): Promise<number[][]> {
    try {
      // Nomic supports batch processing
      const response = await fetch(NOMIC_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${NOMIC_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: NOMIC_MODEL,
          texts,
          task_type: taskType
        })
      });

      if (!response.ok) {
        throw new Error(`Nomic API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.embeddings;
    } catch (error) {
      console.error('Failed to generate batch embeddings:', error);
      throw error;
    }
  }

  // Enhanced text chunking for RAG
  chunkText(text: string, metadata: unknown = {}): Array<{ text: string, metadata: unknown }> {
    const chunks: Array<{ text: string, metadata: unknown }> = [];
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];

    let currentChunk = '';
    let chunkIndex = 0;

    for (const sentence of sentences) {
      if ((currentChunk + sentence).length > this.chunkSettings.maxChunkSize) {
        if (currentChunk.length >= this.chunkSettings.minChunkSize) {
          chunks.push({
            text: currentChunk.trim(),
            metadata: {
              ...metadata,
              chunkIndex,
              totalChunks: -1, // Will be updated later
              startChar: text.indexOf(currentChunk),
              endChar: text.indexOf(currentChunk) + currentChunk.length
            }
          });
          chunkIndex++;

          // Add overlap
          const words = currentChunk.split(' ');
          const overlapWords = Math.floor(words.length * 0.2);
          currentChunk = words.slice(-overlapWords).join(' ') + ' ' + sentence;
        } else {
          currentChunk += sentence;
        }
      } else {
        currentChunk += sentence;
      }
    }

    // Add the last chunk
    if (currentChunk.length >= this.chunkSettings.minChunkSize) {
      chunks.push({
        text: currentChunk.trim(),
        metadata: {
          ...metadata,
          chunkIndex,
          totalChunks: chunks.length + 1,
          startChar: text.indexOf(currentChunk),
          endChar: text.indexOf(currentChunk) + currentChunk.length
        }
      });
    }

    // Update total chunks count
    chunks.forEach(chunk => {
      chunk.metadata.totalChunks = chunks.length;
    });

    return chunks;
  }

  // Store document with chunking for RAG
  async storeDocument(
    documentId: string,
    content: string,
    metadata: unknown = {},
    collectionName?: string
  ): Promise<void> {
    try {
      const collection = collectionName || this.collections.documents;
      const chunks = this.chunkText(content, { documentId, ...metadata });

      // Generate embeddings for all chunks
      const texts = chunks.map(c => c.text);
      const embeddings = await this.generateBatchEmbeddings(texts, 'search_document');

      // Prepare points for Qdrant
      const points = chunks.map((chunk, index) => ({
        id: `${documentId}_chunk_${index}`,
        vector: embeddings[index],
        payload: {
          documentId,
          chunkText: chunk.text,
          chunkMetadata: chunk.metadata,
          ...metadata,
          timestamp: new Date().toISOString()
        }
      }));

      // Store in Qdrant
      await this.qdrantClient.upsert(collection, {
        wait: true,
        points
      });

      // Also store in PostgreSQL for hybrid search
      await this.storeDocumentInPostgres(documentId, content, chunks, embeddings, metadata);

      console.log(`Stored document ${documentId} with ${chunks.length} chunks`);
    } catch (error) {
      console.error('Failed to store document:', error);
      throw error;
    }
  }

  // Store document chunks in PostgreSQL with pgvector
  private async storeDocumentInPostgres(
    documentId: string,
    fullContent: string,
    chunks: unknown[],
    embeddings: number[][],
    metadata: unknown
  ): Promise<void> {
    try {
      // Store main document
      await db.insert(documents).values({
        id: documentId,
        title: metadata.title || 'Untitled',
        content: fullContent,
        source: metadata.source || 'manual',
        metadata
        // TODO: Re-enable when pgvector extension is properly configured
        // embedding: sql`${embeddings[0]}::vector(${EMBEDDING_DIMENSION})`
      }).onConflictDoUpdate({
        target: documents.id,
        set: {
          content: fullContent,
          metadata,
          // embedding: sql`${embeddings[0]}::vector(${EMBEDDING_DIMENSION})`,
          updatedAt: new Date()
        }
      });

      // Store chunks (embedding disabled in schema for now)
      for (let i = 0; i < chunks.length; i++) {
        await db.insert(documentChunks).values({
          documentId,
          chunkIndex: i,
          content: chunks[i].text,
          metadata: chunks[i].metadata
          // TODO: Re-enable when pgvector extension is properly configured
          // embedding: sql`${embeddings[i]}::vector(${EMBEDDING_DIMENSION})`
        });
      }
    } catch (error) {
      console.error('Failed to store document in PostgreSQL:', error);
      throw error;
    }
  }

  // Enhanced hybrid search combining vector and keyword search
  async hybridSearch(
    query: string,
    options: {
      collection?: string;
      limit?: number;
      filter?: unknown;
      scoreThreshold?: number;
      rerank?: boolean;
      includeMetadata?: boolean;
      searchType?: 'vector' | 'keyword' | 'hybrid';
    } = {}
  ): Promise<unknown[]> {
    const {
      collection = this.collections.documents,
      limit = 10,
      filter = {},
      scoreThreshold = 0.7,
      rerank = true,
      includeMetadata = true,
      searchType = 'hybrid'
    } = options;

    try {
      const results: unknown[] = [];

      // Vector search
      if (searchType === 'vector' || searchType === 'hybrid') {
        const queryEmbedding = await this.generateEmbedding(query, 'search_query');
        const vectorResults = await this.qdrantClient.search(collection, {
          vector: queryEmbedding,
          limit: limit * 2, // Get more results for reranking
          filter,
          with_payload: includeMetadata,
          score_threshold: scoreThreshold
        });
        results.push(...vectorResults);
      }

      // Keyword search in PostgreSQL
      if (searchType === 'keyword' || searchType === 'hybrid') {
        const keywordResults = await this.keywordSearch(query, filter, limit * 2);
        results.push(...keywordResults);
      }

      // Combine and deduplicate results
      const combinedResults = this.combineSearchResults(results);

      // Rerank if requested
      if (rerank && combinedResults.length > 0) {
        return await this.rerankResults(query, combinedResults, limit);
      }

      return combinedResults.slice(0, limit);
    } catch (error) {
      console.error('Hybrid search failed:', error);
      throw error;
    }
  }

  // Keyword search using PostgreSQL full-text search
  private async keywordSearch(query: string, filter: unknown, limit: number): Promise<unknown[]> {
    try {
      // Use PostgreSQL's full-text search
      const results = await db.execute(sql`
        SELECT
          dc.*,
          ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', ${query})) as rank,
          dc.embedding <=> (
            SELECT embedding FROM document_chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', ${query})
            LIMIT 1
          ) as vector_distance
        FROM document_chunks dc
        WHERE to_tsvector('english', dc.content) @@ plainto_tsquery('english', ${query})
        ${filter.documentId ? sql`AND dc.document_id = ${filter.documentId}` : sql``}
        ORDER BY rank DESC, vector_distance ASC
        LIMIT ${limit}
      `);

      return (results as unknown[]).map(row => ({
        id: row.id,
        score: row.rank,
        payload: {
          chunkText: row.content,
          metadata: row.metadata,
          documentId: row.document_id
        }
      }));
    } catch (error) {
      console.error('Keyword search failed:', error);
      return [];
    }
  }

  // Combine and deduplicate search results
  private combineSearchResults(results: unknown[]): unknown[] {
    const seen = new Set();
    const combined = [];

    for (const result of results) {
      const id = result.id || result.payload?.documentId;
      if (!seen.has(id)) {
        seen.add(id);
        combined.push(result);
      }
    }

    // Sort by score
    return combined.sort((a, b) => (b.score || 0) - (a.score || 0));
  }

  // Rerank results using cross-encoder or additional scoring
  private async rerankResults(query: string, results: unknown[], limit: number): Promise<unknown[]> {
    try {
      // Calculate additional relevance scores
      const rerankedResults = await Promise.all(results.map(async (result) => {
        const text = result.payload?.chunkText || result.payload?.content || '';

        // Calculate various relevance scores
        const scores = {
          vectorScore: result.score || 0,
          keywordScore: this.calculateKeywordScore(query, text),
          lengthScore: this.calculateLengthScore(text),
          recencyScore: this.calculateRecencyScore(result.payload?.timestamp),
          contextScore: await this.calculateContextScore(query, text)
        };

        // Weighted combination of scores
        const finalScore =
          scores.vectorScore * 0.4 +
          scores.keywordScore * 0.2 +
          scores.contextScore * 0.2 +
          scores.lengthScore * 0.1 +
          scores.recencyScore * 0.1;

        return {
          ...result,
          score: finalScore,
          scores
        };
      }));

      // Sort by final score and return top results
      return rerankedResults
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    } catch (error) {
      console.error('Reranking failed:', error);
      return results.slice(0, limit);
    }
  }

  // Calculate keyword overlap score
  private calculateKeywordScore(query: string, text: string): number {
    const queryTokens = query.toLowerCase().split(/\s+/);
    const textTokens = text.toLowerCase().split(/\s+/);
    const textSet = new Set(textTokens);

    let matches = 0;
    for (const token of queryTokens) {
      if (textSet.has(token)) matches++;
    }

    return matches / queryTokens.length;
  }

  // Calculate length score (prefer moderate length chunks)
  private calculateLengthScore(text: string): number {
    const length = text.length;
    const ideal = 300;

    if (length < ideal / 2) return length / (ideal / 2);
    if (length > ideal * 2) return Math.max(0, 1 - (length - ideal * 2) / ideal);
    return 1;
  }

  // Calculate recency score
  private calculateRecencyScore(timestamp?: string): number {
    if (!timestamp) return 0.5;

    const age = Date.now() - new Date(timestamp).getTime();
    const dayInMs = 24 * 60 * 60 * 1000;

    if (age < dayInMs) return 1;
    if (age < 7 * dayInMs) return 0.8;
    if (age < 30 * dayInMs) return 0.6;
    return 0.4;
  }

  // Calculate context score using semantic similarity
  private async calculateContextScore(query: string, text: string): Promise<number> {
    try {
      // Extract key concepts from query
      const concepts = this.extractConcepts(query);
      const textConcepts = this.extractConcepts(text);

      // Calculate concept overlap
      const overlap = concepts.filter(c => textConcepts.includes(c)).length;
      return overlap / Math.max(concepts.length, 1);
    } catch (error) {
      return 0.5;
    }
  }

  // Extract key concepts from text
  private extractConcepts(text: string): string[] {
    // Simple concept extraction - can be enhanced with NLP
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']);
    return text.toLowerCase()
      .split(/\s+/)
      .filter(word => word.length > 3 && !stopWords.has(word))
      .slice(0, 10);
  }

  // Generate and store user embedding with Nomic
  async generateUserEmbedding(userId: string): Promise<void> {
    try {
      const user = await db.query.units.findFirst({
        where: eq(units.id, userId)
      });

      if (!user) {
        throw new Error('User not found');
      }

      // Create comprehensive user profile text
      const userText = `
        Unit: ${user.name}
        Type: ${user.unitType}
        Level: ${user.level}
        Rank: ${user.rank}
        Bio: ${user.bio}
        Specialization: ${user.unitType}
        Experience: Level ${user.level} with ${user.xp} XP
        Combat Performance: ${user.combatRating}% efficiency
        Mission History: ${user.missionsCompleted} successful operations
        Active Time: ${user.hoursActive} hours in field
        Achievements: ${user.achievementsUnlocked} unlocked
      `.trim();

      // Generate embedding with Nomic
      const embedding = await this.generateEmbedding(userText, 'search_document');

      // Store in pgvector (disabled until schema enables embedding field)
      // await db.update(units)
      //   .set({
      //     embedding: sql`${embedding}::vector(${EMBEDDING_DIMENSION})`
      //   })
      //   .where(eq(units.id, userId));

      // Store in Qdrant with metadata
      await this.qdrantClient.upsert(this.collections.users, {
        wait: true,
        points: [
          {
            id: userId,
            vector: embedding,
            payload: {
              userId,
              name: user.name,
              unitType: user.unitType,
              level: user.level,
              rank: user.rank,
              combatRating: parseFloat(user.combatRating),
              missionsCompleted: user.missionsCompleted,
              keywords: [user.unitType, `level_${user.level}`, `rank_${user.rank}`]
            }
          }
        ]
      });

      console.log(`User embedding generated for ${user.name} using Nomic`);
    } catch (error) {
      console.error('Failed to generate user embedding:', error);
      throw error;
    }
  }

  // Find similar users with enhanced matching
  async findSimilarUsers(userId: string, options: {
    limit?: number;
    minSimilarity?: number;
    filters?: unknown;
  } = {}): Promise<unknown[]> {
    const { limit = 10, minSimilarity = 0.7, filters = {} } = options;

    try {
      const user = await db.query.units.findFirst({
        where: eq(units.id, userId)
      });

      if (!user) {
        throw new Error('User not found');
      }

      // TODO: Re-enable when embedding field is available in schema
      // if (!user.embedding) {
      //   throw new Error('User embedding not found');
      // }

      // Search in Qdrant (using dummy vector for now)
      const qdrantResults = await this.qdrantClient.search(this.collections.users, {
        vector: new Array(EMBEDDING_DIMENSION).fill(0), // user.embedding as any,
        limit: limit * 2,
        filter: {
          must_not: [
            { key: 'userId', match: { value: userId } }
          ],
          ...filters
        },
        score_threshold: minSimilarity
      });

      // Search in pgvector for comparison (disabled until embedding field available)
      // const pgResults = await db.execute(sql`
      //   SELECT
      //     u.*,
      //     1 - (u.embedding <=> ${user.embedding}::vector(${EMBEDDING_DIMENSION})) as similarity
      //   FROM units u
      //   WHERE u.id != ${userId}
      //     AND u.embedding IS NOT NULL
      //     AND 1 - (u.embedding <=> ${user.embedding}::vector(${EMBEDDING_DIMENSION})) > ${minSimilarity}
      //   ORDER BY similarity DESC
      //   LIMIT ${limit}
      // `);

      // Combine and rank results (using only Qdrant for now)
      const combined = this.combineUserResults(qdrantResults, []);

      return combined;
    } catch (error) {
      console.error('Failed to find similar users:', error);
      throw error;
    }
  }

  // Combine user search results from multiple sources
  private combineUserResults(qdrantResults: unknown[], pgResults: unknown[]): unknown[] {
    const resultMap = new Map();

    // Add Qdrant results
    for (const result of qdrantResults) {
      resultMap.set(result.payload.userId, {
        userId: result.payload.userId,
        similarity: result.score,
        source: 'qdrant',
        metadata: result.payload
      });
    }

    // Add or merge PostgreSQL results
    for (const result of pgResults) {
      const existing = resultMap.get(result.id);
      if (existing) {
        existing.similarity = (existing.similarity + result.similarity) / 2;
        existing.source = 'both';
      } else {
        resultMap.set(result.id, {
          userId: result.id,
          similarity: result.similarity,
          source: 'postgres',
          metadata: {
            name: result.name,
            unitType: result.unit_type,
            level: result.level,
            rank: result.rank
          }
        });
      }
    }

    return Array.from(resultMap.values())
      .sort((a, b) => b.similarity - a.similarity);
  }

  // Semantic search with query expansion
  async semanticSearch(
    query: string,
    collection: string,
    options: {
      limit?: number;
      expandQuery?: boolean;
      filters?: unknown;
    } = {}
  ): Promise<unknown[]> {
    const { limit = 20, expandQuery = true, filters = {} } = options;

    try {
      let searchQueries = [query];

      // Query expansion for better recall
      if (expandQuery) {
        searchQueries = await this.expandQuery(query);
      }

      // Generate embeddings for all queries
      const embeddings = await this.generateBatchEmbeddings(searchQueries, 'search_query');

      // Search with each query embedding
      const allResults = [];
      for (const embedding of embeddings) {
        const results = await this.qdrantClient.search(collection, {
          vector: embedding,
          limit,
          filter: filters,
          with_payload: true
        });
        allResults.push(...results);
      }

      // Deduplicate and sort
      return this.combineSearchResults(allResults).slice(0, limit);
    } catch (error) {
      console.error('Semantic search failed:', error);
      throw error;
    }
  }

  // Query expansion for better search
  private async expandQuery(query: string): Promise<string[]> {
    const expanded = [query];

    // Add variations
    if (query.length > 10) {
      // Add question form
      if (!query.includes('?')) {
        expanded.push(`What is ${query}?`);
      }

      // Add context
      expanded.push(`Information about ${query}`);

      // Add action form
      if (!query.startsWith('How')) {
        expanded.push(`How to ${query}`);
      }
    }

    return expanded.slice(0, 3); // Limit to 3 variations
  }

  // Store conversation for conversational AI
  async storeConversation(
    conversationId: string,
    messages: Array<{ role: string; content: string }>,
    metadata: unknown = {}
  ): Promise<void> {
    try {
      // Create conversation summary
      const summary = messages
        .map(m => `${m.role}: ${m.content}`)
        .join('\n');

      // Generate embedding for the conversation
      const embedding = await this.generateEmbedding(summary, 'search_document');

      // Store in Qdrant
      await this.qdrantClient.upsert(this.collections.conversations, {
        wait: true,
        points: [
          {
            id: conversationId,
            vector: embedding,
            payload: {
              conversationId,
              messages,
              messageCount: messages.length,
              timestamp: new Date().toISOString(),
              ...metadata
            }
          }
        ]
      });
    } catch (error) {
      console.error('Failed to store conversation:', error);
      throw error;
    }
  }

  // Delete embeddings
  async deleteEmbedding(collection: keyof typeof this.collections, ids: string | string[]): Promise<void> {
    try {
      const idsArray = Array.isArray(ids) ? ids : [ids];
      await this.qdrantClient.delete(this.collections[collection], {
        wait: true,
        points: idsArray
      });
    } catch (error) {
      console.error('Failed to delete embeddings:', error);
      throw error;
    }
  }

  // Get collection statistics
  async getCollectionStats(collection: keyof typeof this.collections): Promise<any> {
    try {
      const info = await this.qdrantClient.getCollection(this.collections[collection]);
      return {
        name: this.collections[collection],
        vectorCount: info.vectors_count,
        pointsCount: info.points_count,
        indexedVectorsCount: info.indexed_vectors_count,
        status: info.status,
        config: info.config
      };
    } catch (error) {
      console.error('Failed to get collection stats:', error);
      return null;
    }
  }
}