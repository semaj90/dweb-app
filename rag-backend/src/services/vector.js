/**
 * Vector Service for Enhanced RAG
 * Handles embedding generation and vector operations with pgvector
 */

import { OpenAIEmbeddings } from '@langchain/openai';

export class VectorService {
  constructor(databaseService) {
    this.db = databaseService;
    this.embeddings = null;
  }

  async initialize() {
    // Initialize embeddings with Ollama local model
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: 'not-needed',
      openAIApiBase: process.env.OLLAMA_URL || 'http://localhost:11434',
      modelName: process.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text',
      dimensions: 384
    });

    console.log('✅ Vector service initialized with local embeddings');
    return true;
  }

  /**
   * Generate embedding for text
   */
  async generateEmbedding(text) {
    if (!this.embeddings) {
      throw new Error('Vector service not initialized');
    }

    try {
      const embedding = await this.embeddings.embedQuery(text);
      return embedding;
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings for multiple texts
   */
  async generateEmbeddings(texts) {
    if (!this.embeddings) {
      throw new Error('Vector service not initialized');
    }

    try {
      const embeddings = await this.embeddings.embedDocuments(texts);
      return embeddings;
    } catch (error) {
      console.error('Failed to generate embeddings:', error);
      throw error;
    }
  }

  /**
   * Perform vector similarity search on documents
   */
  async searchDocuments(queryText, options = {}) {
    const {
      limit = 10,
      threshold = 0.7,
      caseId,
      documentTypes = [],
      includeContent = true
    } = options;

    try {
      // Generate embedding for query
      const queryEmbedding = await this.generateEmbedding(queryText);

      // Search using database service
      const results = await this.db.vectorSearch(queryEmbedding, {
        limit,
        threshold,
        caseId,
        documentTypes,
        includeContent
      });

      return results;
    } catch (error) {
      console.error('Vector search failed:', error);
      throw error;
    }
  }

  /**
   * Perform vector similarity search on chunks
   */
  async searchChunks(queryText, options = {}) {
    const {
      limit = 20,
      threshold = 0.7,
      documentIds = []
    } = options;

    try {
      // Generate embedding for query
      const queryEmbedding = await this.generateEmbedding(queryText);

      // Search using database service
      const results = await this.db.vectorSearchChunks(queryEmbedding, {
        limit,
        threshold,
        documentIds
      });

      return results;
    } catch (error) {
      console.error('Chunk search failed:', error);
      throw error;
    }
  }

  /**
   * Add document with embedding
   */
  async addDocument(documentData) {
    try {
      // Generate embedding for document content
      const embedding = await this.generateEmbedding(documentData.content);

      // Insert document with embedding
      const document = await this.db.insertDocument({
        ...documentData,
        embedding
      });

      console.log(`✅ Document added with embedding: ${document.id}`);
      return document;
    } catch (error) {
      console.error('Failed to add document:', error);
      throw error;
    }
  }

  /**
   * Add document chunks with embeddings
   */
  async addDocumentChunks(documentId, chunks) {
    try {
      const results = [];

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        
        // Generate embedding for chunk
        const embedding = await this.generateEmbedding(chunk.content);

        // Insert chunk with embedding
        const chunkResult = await this.db.insertChunk({
          documentId,
          chunkIndex: i,
          content: chunk.content,
          embedding,
          metadata: chunk.metadata || {}
        });

        results.push(chunkResult);
      }

      console.log(`✅ Added ${results.length} chunks for document ${documentId}`);
      return results;
    } catch (error) {
      console.error('Failed to add document chunks:', error);
      throw error;
    }
  }

  /**
   * Update document embedding
   */
  async updateDocumentEmbedding(documentId, content) {
    try {
      // Generate new embedding
      const embedding = await this.generateEmbedding(content);

      // Update document with new embedding
      const document = await this.db.updateDocumentEmbedding(documentId, embedding);

      console.log(`✅ Updated embedding for document: ${documentId}`);
      return document;
    } catch (error) {
      console.error('Failed to update document embedding:', error);
      throw error;
    }
  }

  /**
   * Hybrid search combining vector and text search
   */
  async hybridSearch(queryText, options = {}) {
    const {
      limit = 10,
      vectorWeight = 0.7,
      textWeight = 0.3,
      caseId,
      documentTypes = []
    } = options;

    try {
      // Perform vector search
      const vectorResults = await this.searchDocuments(queryText, {
        limit: limit * 2, // Get more results for reranking
        caseId,
        documentTypes
      });

      // Perform text search (simple keyword matching)
      const textResults = await this.db.getDocuments({
        caseId,
        documentTypes,
        limit: limit * 2,
        search: queryText
      });

      // Combine and rerank results
      const combinedResults = this.combineSearchResults(
        vectorResults,
        textResults,
        vectorWeight,
        textWeight
      );

      return combinedResults.slice(0, limit);
    } catch (error) {
      console.error('Hybrid search failed:', error);
      throw error;
    }
  }

  /**
   * Combine vector and text search results
   */
  combineSearchResults(vectorResults, textResults, vectorWeight, textWeight) {
    const resultMap = new Map();

    // Add vector results with weighted scores
    vectorResults.forEach(result => {
      const score = (result.similarity_score || 0) * vectorWeight;
      resultMap.set(result.id, {
        ...result,
        combined_score: score,
        sources: ['vector']
      });
    });

    // Add text results with weighted scores
    textResults.forEach(result => {
      const textScore = 0.8; // Default text relevance score
      const score = textScore * textWeight;
      
      if (resultMap.has(result.id)) {
        // Combine scores if document appears in both results
        const existing = resultMap.get(result.id);
        existing.combined_score += score;
        existing.sources.push('text');
      } else {
        resultMap.set(result.id, {
          ...result,
          combined_score: score,
          sources: ['text']
        });
      }
    });

    // Sort by combined score and return
    return Array.from(resultMap.values())
      .sort((a, b) => b.combined_score - a.combined_score);
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  cosineSimilarity(vectorA, vectorB) {
    if (vectorA.length !== vectorB.length) {
      throw new Error('Vectors must have the same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += vectorA[i] * vectorA[i];
      normB += vectorB[i] * vectorB[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Find similar documents based on document embedding
   */
  async findSimilarDocuments(documentId, options = {}) {
    const { limit = 5, threshold = 0.8 } = options;

    try {
      // Get the source document
      const sourceDoc = await this.db.getDocument(documentId);
      if (!sourceDoc || !sourceDoc.embedding) {
        throw new Error('Document not found or has no embedding');
      }

      // Search for similar documents
      const results = await this.db.vectorSearch(sourceDoc.embedding, {
        limit: limit + 1, // +1 to exclude the source document
        threshold
      });

      // Filter out the source document
      return results.filter(result => result.id !== documentId).slice(0, limit);
    } catch (error) {
      console.error('Failed to find similar documents:', error);
      throw error;
    }
  }
}