
import { db } from "$lib/server/db";
import { 
  evidence, 
  documentVectors, 
  evidenceVectors, 
  caseSummaryVectors,
  knowledgeNodes,
  knowledgeEdges
} from "$lib/server/db/schema";
import { ollamaService } from "./ollamaService";
import { eq, sql } from "drizzle-orm";
import type { DocumentProcessingOptions } from '$lib/schemas/upload';

export interface PipelineResult {
  success: boolean;
  documentId: string;
  summary?: string;
  entities?: string[];
  sentiment?: string;
  classification?: string;
  embeddings?: {
    count: number;
    dimension: number;
  };
  error?: string;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
}

export class AIPipeline {
  /**
   * Process a document through the full AI pipeline
   */
  async processDocument(
    documentId: string,
    content: string,
    options: DocumentProcessingOptions = {
      extractText: true,
      generateEmbeddings: true,
      generateSummary: true,
      extractEntities: true,
      analyzeSentiment: true,
      classifyDocument: true,
      chunkSize: 1000,
      chunkOverlap: 200
    }
  ): Promise<PipelineResult> {
    const result: PipelineResult = {
      success: false,
      documentId
    };

    try {
      // 1. Extract text if needed (already done in this case)
      
      // 2. Generate embeddings if requested
      if (options.generateEmbeddings) {
        const embeddingResult = await this.generateAndStoreEmbeddings(
          documentId,
          content,
          options.chunkSize || 1000,
          options.chunkOverlap || 200
        );
        result.embeddings = embeddingResult;
      }

      // 3. Generate summary if requested
      if (options.generateSummary) {
        result.summary = await ollamaService.analyzeDocument(content, 'summary');
        
        // Store summary embedding for case-level search
        if (result.summary) {
          await this.storeSummaryVector(documentId, result.summary);
        }
      }

      // 4. Extract entities if requested
      if (options.extractEntities) {
        const entitiesText = await ollamaService.analyzeDocument(content, 'entities');
        result.entities = this.parseEntities(entitiesText);
        
        // Create knowledge graph nodes for entities
        if (result.entities.length > 0) {
          await this.createEntityNodes(documentId, result.entities);
        }
      }

      // 5. Analyze sentiment if requested
      if (options.analyzeSentiment) {
        result.sentiment = await ollamaService.analyzeDocument(content, 'sentiment');
      }

      // 6. Classify document if requested
      if (options.classifyDocument) {
        result.classification = await ollamaService.analyzeDocument(content, 'classification');
      }

      result.success = true;
    } catch (error) {
      console.error('Pipeline processing error:', error);
      result.error = error instanceof Error ? error.message : 'Unknown error';
    }

    return result;
  }

  /**
   * Generate and store embeddings for document chunks
   */
  private async generateAndStoreEmbeddings(
    documentId: string,
    content: string,
    chunkSize: number,
    chunkOverlap: number
  ): Promise<{ count: number; dimension: number }> {
    const { chunks } = await ollamaService.embedDocument(content, { documentId });
    
    // Store each chunk with its embedding
    for (const chunk of chunks) {
      await db.insert(documentVectors).values({
        documentId,
        chunkIndex: chunk.metadata.chunkIndex,
        content: chunk.content,
        embedding: chunk.embedding,
        metadata: chunk.metadata
      });
    }

    return {
      count: chunks.length,
      dimension: chunks[0]?.embedding.length || 384
    };
  }

  /**
   * Store summary vector for case-level search
   */
  private async storeSummaryVector(documentId: string, summary: string) {
    // Get the associated case ID
    const [doc] = await db
      .select({ caseId: evidence.caseId })
      .from(evidence)
      .where(eq(evidence.id, documentId))
      .limit(1);

    if (doc?.caseId) {
      const embedding = await ollamaService.generateEmbedding(summary);
      
      // Update or insert case summary vector
      await db.insert(caseSummaryVectors)
        .values({
          caseId: doc.caseId,
          summary,
          embedding,
          confidence: 0.9 // High confidence for AI-generated summaries
        })
        .onConflictDoUpdate({
          target: caseSummaryVectors.caseId,
          set: {
            summary,
            embedding,
            lastUpdated: sql`NOW()`
          }
        });
    }
  }

  /**
   * Parse entities from LLM response
   */
  private parseEntities(entitiesText: string): string[] {
    // Simple parsing - in production, you'd want more robust parsing
    const entities: string[] = [];
    const lines = entitiesText.split('\n');
    
    for (const line of lines) {
      // Look for patterns like "- Person: John Doe" or "Person: John Doe"
      const match = line.match(/[-•*]?\s*(?:Person|Organization|Location|Date):\s*(.+)/i);
      if (match) {
        entities.push(match[1].trim());
      }
    }
    
    return entities;
  }

  /**
   * Create knowledge graph nodes for entities
   */
  private async createEntityNodes(documentId: string, entities: string[]) {
    for (const entity of entities) {
      const embedding = await ollamaService.generateEmbedding(entity);
      
      const [node] = await db.insert(knowledgeNodes).values({
        nodeType: 'entity',
        nodeId: documentId,
        label: entity,
        embedding,
        properties: {
          source: 'document',
          extractedAt: new Date()
        }
      }).returning();

      // Create edge linking entity to document
      if (node) {
        const [docNode] = await db.insert(knowledgeNodes).values({
          nodeType: 'document',
          nodeId: documentId,
          label: `Document ${documentId}`,
          embedding: await ollamaService.generateEmbedding(`Document ${documentId}`),
          properties: {}
        }).returning();

        if (docNode) {
          await db.insert(knowledgeEdges).values({
            sourceId: node.id,
            targetId: docNode.id,
            relationship: 'extracted_from',
            weight: 1.0
          });
        }
      }
    }
  }

  /**
   * Semantic search across documents
   */
  async semanticSearch(
    query: string,
    options: {
      limit?: number;
      threshold?: number;
      caseId?: string;
      type?: 'document' | 'evidence' | 'case';
    } = {}
  ): Promise<SearchResult[]> {
    const { limit = 10, threshold = 0.7, caseId, type = 'document' } = options;
    
    // Generate query embedding
    const queryEmbedding = await ollamaService.generateEmbedding(query);
    
    // Search based on type
    let results: SearchResult[] = [];
    
    if (type === 'document' || type === 'evidence') {
      // Search document vectors
      const searchQuery = db
        .select({
          id: documentVectors.id,
          content: documentVectors.content,
          score: sql<number>`1 - (${documentVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`,
          metadata: documentVectors.metadata
        })
        .from(documentVectors)
        .where(sql`1 - (${documentVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector) > ${threshold}`)
        .orderBy(sql`${documentVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector`)
        .limit(limit);

      const docResults = await searchQuery;
      results = docResults.map((r: any) => ({
        id: r.id,
        content: r.content,
        score: r.score,
        metadata: r.metadata || {}
      }));
    } else if (type === 'case') {
      // Search case summaries
      const caseQuery = db
        .select({
          id: caseSummaryVectors.caseId,
          content: caseSummaryVectors.summary,
          score: sql<number>`1 - (${caseSummaryVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`,
          confidence: caseSummaryVectors.confidence
        })
        .from(caseSummaryVectors)
        .where(sql`1 - (${caseSummaryVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector) > ${threshold}`)
        .orderBy(sql`${caseSummaryVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector`)
        .limit(limit);

      const caseResults = await caseQuery;
      results = caseResults.map((r: any) => ({
        id: r.id,
        content: r.content,
        score: r.score * (r.confidence || 1),
        metadata: { type: 'case_summary' }
      }));
    }

    return results;
  }

  /**
   * Find similar documents based on content
   */
  async findSimilarDocuments(
    documentId: string,
    limit: number = 5
  ): Promise<SearchResult[]> {
    // Get average embedding for the document
    const [docEmbedding] = await db
      .select({
        avgEmbedding: sql<number[]>`AVG(${documentVectors.embedding})::vector`
      })
      .from(documentVectors)
      .where(eq(documentVectors.documentId, documentId));

    if (!docEmbedding?.avgEmbedding) {
      return [];
    }

    // Find similar documents
    const results = await db
      .select({
        documentId: documentVectors.documentId,
        avgScore: sql<number>`AVG(1 - (${documentVectors.embedding} <=> ${JSON.stringify(docEmbedding.avgEmbedding)}::vector))`,
        content: sql<string>`STRING_AGG(${documentVectors.content}, ' ' ORDER BY ${documentVectors.chunkIndex})`
      })
      .from(documentVectors)
      .where(sql`${documentVectors.documentId} != ${documentId}`)
      .groupBy(documentVectors.documentId)
      .orderBy(sql`AVG(${documentVectors.embedding} <=> ${JSON.stringify(docEmbedding.avgEmbedding)}::vector)`)
      .limit(limit);

    return results.map((r: any) => ({
      id: r.documentId,
      content: r.content || '',
      score: r.avgScore || 0,
      metadata: { type: 'similar_document' }
    }));
  }

  /**
   * Generate recommendations based on user activity
   */
  async generateRecommendations(
    userId: string,
    type: 'case' | 'evidence' | 'document' = 'document'
  ): Promise<SearchResult[]> {
    // This would be implemented with the recommendation engine
    // For now, return empty array
    return [];
  }
}

// Export singleton instance
export const aiPipeline = new AIPipeline();