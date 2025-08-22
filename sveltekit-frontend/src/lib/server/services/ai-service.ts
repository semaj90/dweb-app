import { OllamaService } from "$lib/services/ollamaService.js";
import { userAiQueries, autoTags, documentChunks, embeddingCache } from "../db/schema.js";
import { eq, desc, sql } from "drizzle-orm";
import type { NewUserAiQuery, NewAutoTag, NewDocumentChunk } from "../db/schema.js";
import { generateIdFromEntropySize } from "lucia";
import crypto from 'crypto';
import { db } from "../database/index.js";

export interface AIAnalysisResult {
  summary: string;
  tags: string[];
  confidence: number;
  entities?: string[];
  keywords?: string[];
  recommendations?: string[];
}

export interface AIQueryOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  includeContext?: boolean;
  saveQuery?: boolean;
}

export interface VectorSearchResult {
  content: string;
  similarity: number;
  metadata: Record<string, any>;
  documentId: string;
}

export class AIService {
  private ollama: OllamaService;

  constructor() {
    this.ollama = new OllamaService();
  }

  /**
   * Process AI query with context and logging
   */
  async processQuery(
    query: string,
    userId: string,
    caseId?: string,
    options: AIQueryOptions = {}
  ): Promise<{
    response: string;
    confidence: number;
    contextUsed: string[];
    queryId: string;
  }> {
    const startTime = Date.now();
    const {
      model = 'gemma3-legal',
      temperature = 0.7,
      maxTokens = 2000,
      includeContext = true,
      saveQuery = true
    } = options;

    try {
      // Get relevant context if requested
      let contextDocuments: VectorSearchResult[] = [];
      let systemPrompt = 'You are a legal AI assistant specialized in prosecutor and detective workflows. Provide accurate, detailed, and actionable legal analysis.';

      if (includeContext && caseId) {
        const queryEmbedding = await this.ollama.generateEmbedding(query);
        contextDocuments = await this.findSimilarDocuments(queryEmbedding, 5);
        
        if (contextDocuments.length > 0) {
          const contextText = contextDocuments
            .map((doc: any) => `[Context] ${doc.content}`)
            .join('\n\n');
          
          systemPrompt += `\n\nRelevant case context:\n${contextText}`;
        }
      }

      // Generate AI response
      const response = await this.ollama.generateCompletion(query, {
        systemPrompt,
        temperature,
        maxTokens
      });

      const processingTime = Date.now() - startTime;
      const confidence = this.calculateConfidence(response, contextDocuments.length);
      const contextUsed = contextDocuments.map((doc: any) => doc.documentId);

      // Save query log if requested
      let queryId = '';
      if (saveQuery) {
        queryId = await this.logQuery({
          userId,
          caseId,
          query,
          response,
          model,
          confidence,
          processingTime,
          contextUsed,
          embedding: await this.ollama.generateEmbedding(query)
        });
      }

      return {
        response,
        confidence,
        contextUsed,
        queryId
      };

    } catch (error) {
      console.error('AI query processing failed:', error);
      
      // Log failed query
      if (saveQuery) {
        await this.logQuery({
          userId,
          caseId,
          query,
          response: '',
          model,
          confidence: 0,
          processingTime: Date.now() - startTime,
          contextUsed: [],
          isSuccessful: false,
          errorMessage: error instanceof Error ? error.message : 'Unknown error'
        });
      }

      throw error;
    }
  }

  /**
   * Analyze evidence and generate auto-tags
   */
  async analyzeEvidence(
    evidenceId: string,
    content: string,
    evidenceType: string
  ): Promise<AIAnalysisResult> {
    try {
      const systemPrompt = `You are a legal AI assistant specialized in evidence analysis. 
      Analyze the following ${evidenceType} evidence and provide:
      1. A concise summary (2-3 sentences)
      2. Relevant tags for legal categorization
      3. Key entities mentioned
      4. Important keywords
      5. Recommendations for investigation

      Format your response as JSON with the following structure:
      {
        "summary": "Brief summary here",
        "tags": ["tag1", "tag2", "tag3"],
        "confidence": 0.85,
        "entities": ["entity1", "entity2"],
        "keywords": ["keyword1", "keyword2"],
        "recommendations": ["recommendation1", "recommendation2"]
      }`;

      const response = await this.ollama.generateCompletion(content, {
        systemPrompt,
        temperature: 0.3, // Lower temperature for more consistent structured output
        maxTokens: 1000
      });

      // Parse AI response
      let analysis: AIAnalysisResult;
      try {
        analysis = JSON.parse(response);
      } catch {
        // Fallback to manual parsing if JSON fails
        analysis = this.parseAnalysisResponse(response);
      }

      // Generate and store auto-tags
      if (analysis.tags && analysis.tags.length > 0) {
        await this.generateAutoTags(evidenceId, 'evidence', analysis.tags, analysis.confidence);
      }

      // Store document chunks for vector search
      await this.storeDocumentChunk(evidenceId, 'evidence', content, analysis);

      return analysis;

    } catch (error) {
      console.error('Evidence analysis failed:', error);
      throw error;
    }
  }

  /**
   * Find similar documents using vector search
   */
  async findSimilarDocuments(
    queryEmbedding: number[],
    limit: number = 10,
    threshold: number = 0.7
  ): Promise<VectorSearchResult[]> {
    try {
      // Convert embedding to pgvector format
      const embeddingVector = `[${queryEmbedding.join(',')}]`;

      const results = await db.execute(sql`
        SELECT 
          id,
          document_id,
          content,
          document_type,
          metadata,
          1 - (embedding <=> ${embeddingVector}::vector) as similarity
        FROM document_chunks
        WHERE 1 - (embedding <=> ${embeddingVector}::vector) > ${threshold}
        ORDER BY embedding <=> ${embeddingVector}::vector
        LIMIT ${limit}
      `);

      return results.rows.map((row: any) => ({
        content: row.content as string,
        similarity: row.similarity as number,
        metadata: row.metadata as Record<string, any>,
        documentId: row.document_id as string
      }));

    } catch (error) {
      console.error('Vector search failed:', error);
      return [];
    }
  }

  /**
   * Find similar queries for smart suggestions
   */
  async findSimilarQueries(
    queryEmbedding: number[],
    userId?: string,
    limit: number = 5
  ): Promise<Array<{ query: string; response: string; similarity: number }>> {
    try {
      const embeddingVector = `[${queryEmbedding.join(',')}]`;
      
      let whereClause = sql`1 - (embedding <=> ${embeddingVector}::vector) > 0.8`;
      if (userId) {
        whereClause = sql`${whereClause} AND user_id = ${userId}`;
      }

      const results = await db.execute(sql`
        SELECT 
          query,
          response,
          1 - (embedding <=> ${embeddingVector}::vector) as similarity
        FROM user_ai_queries
        WHERE ${whereClause}
        AND embedding IS NOT NULL
        ORDER BY embedding <=> ${embeddingVector}::vector
        LIMIT ${limit}
      `);

      return results.rows.map((row: any) => ({
        query: row.query as string,
        response: row.response as string,
        similarity: row.similarity as number
      }));

    } catch (error) {
      console.error('Similar query search failed:', error);
      return [];
    }
  }

  /**
   * Get cached embedding or generate new one
   */
  async getOrCreateEmbedding(text: string): Promise<number[]> {
    const textHash = crypto.createHash('sha256').update(text).digest('hex');

    try {
      // Check cache first
      const cached = await db.query.embeddingCache.findFirst({
        where: eq(embeddingCache.textHash, textHash)
      });

      if (cached && cached.embedding) {
        return cached.embedding;
      }

      // Generate new embedding
      const embedding = await this.ollama.generateEmbedding(text);

      // Cache the embedding
      await db.insert(embeddingCache).values({
        textHash,
        embedding,
        model: 'nomic-embed-text'
      });

      return embedding;

    } catch (error) {
      console.error('Embedding generation failed:', error);
      throw error;
    }
  }

  /**
   * Log AI query to database
   */
  private async logQuery(data: {
    userId: string;
    caseId?: string;
    query: string;
    response: string;
    model: string;
    confidence: number;
    processingTime: number;
    contextUsed: string[];
    embedding?: number[];
    isSuccessful?: boolean;
    errorMessage?: string;
  }): Promise<string> {
    try {
      const queryData: NewUserAiQuery = {
        id: generateIdFromEntropySize(10),
        userId: data.userId,
        caseId: data.caseId,
        query: data.query,
        response: data.response,
        model: data.model,
        confidence: data.confidence.toString(),
        processingTime: data.processingTime,
        contextUsed: data.contextUsed,
        embedding: data.embedding,
        isSuccessful: data.isSuccessful !== false,
        errorMessage: data.errorMessage
      };

      const [inserted] = await db.insert(userAiQueries).values(queryData).returning();
      return inserted.id;

    } catch (error) {
      console.error('Query logging failed:', error);
      throw error;
    }
  }

  /**
   * Generate and store auto-tags
   */
  private async generateAutoTags(
    entityId: string,
    entityType: string,
    tags: string[],
    confidence: number
  ): Promise<void> {
    try {
      const tagData: NewAutoTag[] = tags.map((tag: any) => ({
        id: generateIdFromEntropySize(10),
        entityId,
        entityType,
        tag,
        confidence: confidence.toString(),
        source: 'ai_analysis',
        model: 'gemma3-legal'
      }));

      await db.insert(autoTags).values(tagData);

    } catch (error) {
      console.error('Auto-tag generation failed:', error);
      throw error;
    }
  }

  /**
   * Store document chunk for vector search
   */
  private async storeDocumentChunk(
    documentId: string,
    documentType: string,
    content: string,
    analysis: AIAnalysisResult
  ): Promise<void> {
    try {
      const embedding = await this.ollama.generateEmbedding(content);

      const chunkData: NewDocumentChunk = {
        id: generateIdFromEntropySize(10),
        documentId,
        documentType,
        chunkIndex: 0,
        content: content.slice(0, 2000), // Limit content size
        embedding: JSON.stringify(embedding),
        metadata: {
          analysis,
          contentLength: content.length,
          generatedAt: new Date().toISOString()
        }
      };

      await db.insert(documentChunks).values(chunkData);

    } catch (error) {
      console.error('Document chunk storage failed:', error);
      throw error;
    }
  }

  /**
   * Calculate confidence score based on response and context
   */
  private calculateConfidence(response: string, contextCount: number): number {
    let confidence = 0.7; // Base confidence

    // Boost confidence based on response length and detail
    if (response.length > 500) confidence += 0.1;
    if (response.includes('evidence') || response.includes('statute')) confidence += 0.05;
    if (response.includes('recommend') || response.includes('suggest')) confidence += 0.05;

    // Boost confidence based on available context
    confidence += Math.min(contextCount * 0.02, 0.15);

    return Math.min(confidence, 0.99);
  }

  /**
   * Parse AI analysis response if JSON parsing fails
   */
  private parseAnalysisResponse(response: string): AIAnalysisResult {
    // Fallback parsing logic for non-JSON responses
    const lines = response.split('\n');
    
    return {
      summary: response.split('\n')[0] || 'Analysis completed',
      tags: this.extractTags(response),
      confidence: 0.75,
      entities: this.extractEntities(response),
      keywords: this.extractKeywords(response),
      recommendations: this.extractRecommendations(response)
    };
  }

  private extractTags(text: string): string[] {
    // Simple tag extraction logic
    const tagPatterns = /(?:tag|category|classification)s?:?\s*([^\n]+)/gi;
    const matches = text.match(tagPatterns);
    return matches ? matches.flatMap((m: any) => m.split(/[,;]/).map((t: any) => t.trim().toLowerCase())) : [];
  }

  private extractEntities(text: string): string[] {
    // Simple entity extraction
    const entityPattern = /(?:entity|entities|person|organization)s?:?\s*([^\n]+)/gi;
    const matches = text.match(entityPattern);
    return matches ? matches.flatMap((m: any) => m.split(/[,;]/).map((t: any) => t.trim())) : [];
  }

  private extractKeywords(text: string): string[] {
    // Simple keyword extraction
    const keywordPattern = /(?:keyword|key\s+word)s?:?\s*([^\n]+)/gi;
    const matches = text.match(keywordPattern);
    return matches ? matches.flatMap((m: any) => m.split(/[,;]/).map((t: any) => t.trim())) : [];
  }

  private extractRecommendations(text: string): string[] {
    // Simple recommendation extraction
    const recPattern = /(?:recommend|suggestion|advice)s?:?\s*([^\n]+)/gi;
    const matches = text.match(recPattern);
    return matches ? matches.map((m: any) => m.trim()) : [];
  }
}

// Export singleton instance
export const aiService = new AIService();