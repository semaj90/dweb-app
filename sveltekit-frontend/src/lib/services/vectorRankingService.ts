
import { db } from "$lib/db";
import {
  documentVectors, 
  caseSummaryVectors, 
  evidenceVectors,
  queryVectors,
  knowledgeNodes,
  knowledgeEdges,
  recommendationCache
} from "$lib/server/db/schema";
import { ollamaService } from "./ollamaService";
import {
  sql, eq, and, desc
} from "drizzle-orm";

export interface RankedSearchResult {
  id: string;
  content: string;
  score: number;
  rankingFactors: {
    vectorSimilarity: number;
    documentRecency: number;
    userPreference: number;
    contextRelevance: number;
    entityOverlap: number;
  };
  metadata: Record<string, any>;
  explanation?: string;
}

export interface SearchOptions {
  limit?: number;
  threshold?: number;
  caseId?: string;
  documentType?: 'document' | 'evidence' | 'case';
  includeExplanation?: boolean;
  personalized?: boolean;
  userId?: string;
}

export class VectorRankingService {
  /**
   * Perform ranked semantic search with multi-factor scoring
   */
  async rankedSearch(
    query: string,
    options: SearchOptions = {}
  ): Promise<RankedSearchResult[]> {
    const {
      limit = 10,
      threshold = 0.6,
      caseId,
      documentType = 'document',
      includeExplanation = false,
      personalized = false,
      userId
    } = options;

    // Generate query embedding
    const queryEmbedding = await ollamaService.generateEmbedding(query);
    
    // Store query for future recommendations
    if (userId) {
      await this.storeQueryVector(userId, query, queryEmbedding);
    }

    // Perform vector search based on document type
    let vectorResults: any[] = [];
    
    if (documentType === 'document') {
      vectorResults = await this.searchDocumentVectors(queryEmbedding, caseId, limit * 2);
    } else if (documentType === 'evidence') {
      vectorResults = await this.searchEvidenceVectors(queryEmbedding, caseId, limit * 2);
    } else if (documentType === 'case') {
      vectorResults = await this.searchCaseVectors(queryEmbedding, limit * 2);
    }

    // Extract entities from query
    const queryEntities = await this.extractQueryEntities(query);

    // Calculate multi-factor scores
    const rankedResults = await Promise.all(
      vectorResults.map(async (result) => {
        const rankingFactors = await this.calculateRankingFactors(
          result,
          queryEmbedding,
          queryEntities,
          { personalized, userId }
        );

        // Calculate final score (weighted combination)
        const finalScore = this.calculateFinalScore(rankingFactors);

        return {
          id: result.id,
          content: result.content,
          score: finalScore,
          rankingFactors,
          metadata: result.metadata || {},
          explanation: includeExplanation 
            ? this.generateExplanation(rankingFactors)
            : undefined
        };
      })
    );

    // Sort by final score and apply threshold
    const filteredResults = rankedResults
      .filter((r: any) => r.score >= threshold)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    // Update user preferences if personalized
    if (personalized && userId && filteredResults.length > 0) {
      await this.updateUserPreferences(userId, query, filteredResults);
    }

    return filteredResults;
  }

  /**
   * Search document vectors
   */
  private async searchDocumentVectors(
    queryEmbedding: number[],
    caseId: string | undefined,
    limit: number
  ) {
    const query = db
      .select({
        id: documentVectors.id,
        documentId: documentVectors.documentId,
        content: documentVectors.content,
        metadata: documentVectors.metadata,
        similarity: sql<number>`1 - (${documentVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`
      })
      .from(documentVectors)
      .orderBy(sql`${documentVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector`)
      .limit(limit);

    return await query;
  }

  /**
   * Search evidence vectors
   */
  private async searchEvidenceVectors(
    queryEmbedding: number[],
    caseId: string | undefined,
    limit: number
  ) {
    const query = db
      .select({
        id: evidenceVectors.id,
        evidenceId: evidenceVectors.evidenceId,
        content: evidenceVectors.content,
        metadata: evidenceVectors.metadata,
        similarity: sql<number>`1 - (${evidenceVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`
      })
      .from(evidenceVectors)
      .orderBy(sql`${evidenceVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector`)
      .limit(limit);

    return await query;
  }

  /**
   * Search case summary vectors
   */
  private async searchCaseVectors(
    queryEmbedding: number[],
    limit: number
  ) {
    const query = db
      .select({
        id: caseSummaryVectors.id,
        caseId: caseSummaryVectors.caseId,
        content: caseSummaryVectors.summary,
        confidence: caseSummaryVectors.confidence,
        similarity: sql<number>`1 - (${caseSummaryVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`
      })
      .from(caseSummaryVectors)
      .orderBy(sql`${caseSummaryVectors.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector`)
      .limit(limit);

    return await query;
  }

  /**
   * Calculate multi-factor ranking scores
   */
  private async calculateRankingFactors(
    result: any,
    queryEmbedding: number[],
    queryEntities: string[],
    options: { personalized: boolean; userId?: string }
  ): Promise<RankedSearchResult['rankingFactors']> {
    // 1. Vector similarity (already calculated)
    const vectorSimilarity = result.similarity || 0;

    // 2. Document recency score
    const createdAt = result.metadata?.createdAt || new Date();
    const ageInDays = (Date.now() - new Date(createdAt).getTime()) / (1000 * 60 * 60 * 24);
    const documentRecency = Math.exp(-ageInDays / 30); // Decay over 30 days

    // 3. User preference score
    let userPreference = 0.5; // Default neutral
    if (options.personalized && options.userId) {
      userPreference = await this.calculateUserPreference(
        options.userId,
        result.id,
        result.metadata
      );
    }

    // 4. Context relevance (based on metadata)
    const contextRelevance = this.calculateContextRelevance(result.metadata);

    // 5. Entity overlap score
    const entityOverlap = await this.calculateEntityOverlap(
      result.id,
      queryEntities
    );

    return {
      vectorSimilarity,
      documentRecency,
      userPreference,
      contextRelevance,
      entityOverlap
    };
  }

  /**
   * Calculate final score from ranking factors
   */
  private calculateFinalScore(factors: RankedSearchResult['rankingFactors']): number {
    const weights = {
      vectorSimilarity: 0.4,
      documentRecency: 0.1,
      userPreference: 0.2,
      contextRelevance: 0.2,
      entityOverlap: 0.1
    };

    return Object.entries(weights).reduce((score, [factor, weight]) => {
      return score + (factors[factor as keyof typeof factors] * weight);
    }, 0);
  }

  /**
   * Extract entities from query
   */
  private async extractQueryEntities(query: string): Promise<string[]> {
    try {
      const entitiesText = await ollamaService.analyzeDocument(query, 'entities');
      // Simple entity extraction - could be enhanced
      return entitiesText
        .split('\n')
        .filter((line: any) => line.includes(':'))
        .map((line: any) => line.split(':')[1]?.trim())
        .filter(Boolean);
    } catch (error) {
      console.error('Entity extraction failed:', error);
      return [];
    }
  }

  /**
   * Calculate user preference score
   */
  private async calculateUserPreference(
    userId: string,
    documentId: string,
    metadata: any
  ): Promise<number> {
    // Check if user has interacted with similar documents
    const userQueries = await db
      .select()
      .from(queryVectors)
      .where(eq(queryVectors.userId, userId))
      .orderBy(desc(queryVectors.createdAt))
      .limit(10);

    if (userQueries.length === 0) {
      return 0.5; // Neutral preference
    }

    // Calculate preference based on clicked results history
    let preferenceScore = 0.5;
    for (const query of userQueries) {
      const clickedResults = query.clickedResults as unknown[] || [];
      if (clickedResults.includes(documentId)) {
        preferenceScore += 0.1;
      }
    }

    return Math.min(preferenceScore, 1.0);
  }

  /**
   * Calculate context relevance score
   */
  private calculateContextRelevance(metadata: any): number {
    let score = 0.5;

    // Boost score based on metadata quality
    if (metadata?.aiSummary) score += 0.1;
    if (metadata?.entities?.length > 0) score += 0.1;
    if (metadata?.classification) score += 0.1;
    if (metadata?.confidence > 0.8) score += 0.2;

    return Math.min(score, 1.0);
  }

  /**
   * Calculate entity overlap between query and document
   */
  private async calculateEntityOverlap(
    documentId: string,
    queryEntities: string[]
  ): Promise<number> {
    if (queryEntities.length === 0) return 0.5;

    // Get document entities from knowledge graph
    const documentNodes = await db
      .select()
      .from(knowledgeNodes)
      .where(and(
        eq(knowledgeNodes.nodeType, 'entity'),
        eq(knowledgeNodes.nodeId, documentId)
      ))
      .limit(20);

    const documentEntities = documentNodes.map((n: any) => n.label.toLowerCase());
    const queryEntitiesLower = queryEntities.map((e: any) => e.toLowerCase());

    // Calculate Jaccard similarity
    const intersection = queryEntitiesLower.filter((e: any) => documentEntities.some((de: any) => de.includes(e) || e.includes(de))
    );
    
    const union = new Set([...queryEntitiesLower, ...documentEntities]);
    
    return intersection.length / union.size;
  }

  /**
   * Generate explanation for ranking
   */
  private generateExplanation(factors: RankedSearchResult['rankingFactors']): string {
    const explanations: string[] = [];

    if (factors.vectorSimilarity > 0.8) {
      explanations.push('High semantic similarity to query');
    }
    if (factors.documentRecency > 0.8) {
      explanations.push('Recently created or updated');
    }
    if (factors.userPreference > 0.7) {
      explanations.push('Matches your preferences');
    }
    if (factors.contextRelevance > 0.7) {
      explanations.push('Rich metadata and AI analysis');
    }
    if (factors.entityOverlap > 0.6) {
      explanations.push('Contains relevant entities');
    }

    return explanations.join('. ');
  }

  /**
   * Store query vector for recommendations
   */
  private async storeQueryVector(
    userId: string,
    query: string,
    embedding: number[]
  ) {
    await db.insert(queryVectors).values({
      userId,
      query,
      embedding,
      resultCount: 0,
      clickedResults: []
    });
  }

  /**
   * Update user preferences based on interactions
   */
  private async updateUserPreferences(
    userId: string,
    query: string,
    results: RankedSearchResult[]
  ) {
    // This would typically track which results users click on
    // For now, we'll update the recommendation cache
    const recommendations = results.slice(0, 5).map((r: any) => ({
      id: r.id,
      score: r.score,
      metadata: r.metadata
    }));

    await db.insert(recommendationCache).values({
      userId,
      recommendationType: 'search_results',
      recommendations,
      score: results[0]?.score || 0,
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
    });
  }

  /**
   * Get personalized recommendations
   */
  async getRecommendations(
    userId: string,
    type: 'case' | 'evidence' | 'document' = 'document'
  ): Promise<RankedSearchResult[]> {
    // Check cache first
    const cached = await db
      .select()
      .from(recommendationCache)
      .where(and(
        eq(recommendationCache.userId, userId),
        eq(recommendationCache.recommendationType, type),
        sql`${recommendationCache.expiresAt} > NOW()`
      ))
      .orderBy(desc(recommendationCache.createdAt))
      .limit(1);

    if (cached.length > 0) {
      return cached[0].recommendations as any;
    }

    // Generate new recommendations based on user history
    const userQueries = await db
      .select()
      .from(queryVectors)
      .where(eq(queryVectors.userId, userId))
      .orderBy(desc(queryVectors.createdAt))
      .limit(5);

    if (userQueries.length === 0) {
      return [];
    }

    // Use the average of recent query vectors
    const avgEmbedding = this.averageEmbeddings(
      userQueries.map((q: any) => q.embedding)
    );

    // Search with the average embedding
    return this.rankedSearch('', {
      limit: 10,
      documentType: type,
      personalized: true,
      userId
    });
  }

  /**
   * Calculate average of multiple embeddings
   */
  private averageEmbeddings(embeddings: number[][]): number[] {
    if (embeddings.length === 0) return [];
    
    const dimension = embeddings[0].length;
    const avg = new Array(dimension).fill(0);
    
    for (const embedding of embeddings) {
      for (let i = 0; i < dimension; i++) {
        avg[i] += embedding[i];
      }
    }
    
    return avg.map((v: any) => v / embeddings.length);
  }
}

// Export singleton instance
export const vectorRankingService = new VectorRankingService();