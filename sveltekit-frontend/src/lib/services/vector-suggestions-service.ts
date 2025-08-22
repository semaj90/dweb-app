import { db } from '$lib/server/db/index.js';
import { chatMessages, chatRecommendations, cases, evidence } from '$lib/server/db/schema-unified.js';
import { generateEnhancedEmbedding } from '$lib/server/ai/embeddings-enhanced.js';
import { eq, desc, and, or, sql } from 'drizzle-orm';

// Import pgvector utilities
import { 
  searchSimilarMessages, 
  searchSimilarEvidence,
  searchAcrossAllVectors,
  type VectorSearchResult,
  type VectorSearchOptions
} from '$lib/server/db/pgvector-utils.js';

export interface VectorSearchContext {
  documentId: string;
  content: string;
  similarityScore: number;
  documentType: string;
  metadata: Record<string, string>;
}

export interface GraphContext {
  relatedNodes: GraphNode[];
  relationships: GraphRelationship[];
}

export interface GraphNode {
  id: string;
  type: 'case' | 'evidence' | 'precedent' | 'person';
  properties: Record<string, string>;
}

export interface GraphRelationship {
  fromNode: string;
  toNode: string;
  relationshipType: string;
  weight: number;
}

export interface ContextualSuggestion {
  content: string;
  type: string;
  confidence: number;
  reasoning: string;
  metadata: {
    source: 'vector_search' | 'graph_context' | 'similar_cases' | 'evidence_analysis';
    sourceDocumentId?: string;
    similarityScore?: number;
    contextNodes?: string[];
    keywords?: string[];
    category: string;
  };
}

/**
 * Vector-based suggestions service using PostgreSQL pgvector
 * Provides contextual AI suggestions based on semantic similarity search
 */
export class VectorSuggestionsService {
  private readonly similarityThreshold: number = 0.7;
  private readonly maxResults: number = 10;

  constructor() {}

  /**
   * Generate contextual suggestions using vector similarity search
   */
  async generateVectorContextualSuggestions(
    content: string,
    reportType: string,
    userId?: string,
    caseId?: string
  ): Promise<ContextualSuggestion[]> {
    try {
      const suggestions: ContextualSuggestion[] = [];

      // Generate embedding for the input content
      const contentEmbedding = await this.generateEmbedding(content);

      // Get vector-based context from similar messages
      const vectorContext = await this.getVectorContext(contentEmbedding, reportType);
      
      // Get graph context if case ID is provided
      const graphContext = caseId ? await this.getGraphContext(caseId) : null;

      // Generate suggestions from vector context
      const vectorSuggestions = this.extractSuggestionsFromVectorContext(vectorContext, content);
      suggestions.push(...vectorSuggestions);

      // Generate suggestions from graph context
      if (graphContext) {
        const graphSuggestions = await this.extractSuggestionsFromGraphContext(graphContext, content);
        suggestions.push(...graphSuggestions);
      }

      // Get suggestions from similar cases
      if (caseId) {
        const caseSuggestions = await this.getSimilarCaseSuggestions(contentEmbedding, caseId);
        suggestions.push(...caseSuggestions);
      }

      // Get evidence-based suggestions
      const evidenceSuggestions = await this.getEvidenceBasedSuggestions(contentEmbedding, content);
      suggestions.push(...evidenceSuggestions);

      // Deduplicate and rank suggestions
      return this.deduplicateAndRankSuggestions(suggestions);
    } catch (error) {
      console.error('Vector contextual suggestions failed:', error);
      throw error;
    }
  }

  /**
   * Get vector context from similar chat messages and recommendations using pgvector
   */
  private async getVectorContext(
    embedding: number[],
    reportType: string
  ): Promise<VectorSearchContext[]> {
    try {
      // Use pgvector for proper similarity search
      const similarMessages = await searchSimilarMessages(embedding, {
        limit: this.maxResults,
        threshold: this.similarityThreshold,
        includeMetadata: true
      });

      const vectorContext: VectorSearchContext[] = [];

      // For each similar message, get its recommendations
      for (const result of similarMessages) {
        try {
          const recommendations = await db
            .select({
              content: chatRecommendations.content,
              recommendationType: chatRecommendations.recommendationType,
              confidence: chatRecommendations.confidence
            })
            .from(chatRecommendations)
            .where(eq(chatRecommendations.messageId, result.id))
            .limit(3);

          vectorContext.push({
            documentId: result.id,
            content: result.content,
            similarityScore: result.similarity,
            documentType: 'chat_message',
            metadata: {
              ...result.metadata,
              recommendations: recommendations.map(rec => ({
                content: rec.content,
                type: rec.recommendationType,
                confidence: rec.confidence
              }))
            }
          });
        } catch (error) {
          console.warn('Failed to fetch recommendations for message:', result.id, error);
        }
      }

      return vectorContext.sort((a, b) => b.similarityScore - a.similarityScore);
    } catch (error) {
      console.error('Failed to get vector context:', error);
      return [];
    }
  }

  /**
   * Get graph context for a specific case
   */
  private async getGraphContext(caseId: string): Promise<GraphContext | null> {
    try {
      // Get case information
      const caseInfo = await db
        .select()
        .from(cases)
        .where(eq(cases.id, caseId))
        .limit(1);

      if (caseInfo.length === 0) {
        return null;
      }

      const caseData = caseInfo[0];

      // Get related evidence
      const relatedEvidence = await db
        .select()
        .from(evidence)
        .where(eq(evidence.caseId, caseId))
        .limit(20);

      // Build graph nodes and relationships
      const nodes: GraphNode[] = [
        {
          id: caseData.id,
          type: 'case',
          properties: {
            title: caseData.title || '',
            description: caseData.description || '',
            status: caseData.status || '',
            caseNumber: caseData.caseNumber || ''
          }
        }
      ];

      const relationships: GraphRelationship[] = [];

      relatedEvidence.forEach(evidenceItem => {
        nodes.push({
          id: evidenceItem.id,
          type: 'evidence',
          properties: {
            description: evidenceItem.description || '',
            type: evidenceItem.type || '',
            source: evidenceItem.source || '',
            significance: evidenceItem.significance || ''
          }
        });

        relationships.push({
          fromNode: caseData.id,
          toNode: evidenceItem.id,
          relationshipType: 'HAS_EVIDENCE',
          weight: 1.0
        });
      });

      return { relatedNodes: nodes, relationships };
    } catch (error) {
      console.error('Failed to get graph context:', error);
      return null;
    }
  }

  /**
   * Extract suggestions from vector search results
   */
  private extractSuggestionsFromVectorContext(
    vectorContext: VectorSearchContext[],
    currentContent: string
  ): ContextualSuggestion[] {
    const suggestions: ContextualSuggestion[] = [];

    vectorContext.forEach(context => {
      if (context.metadata.recommendationContent) {
        suggestions.push({
          content: context.metadata.recommendationContent,
          type: context.metadata.recommendationType || 'vector_based',
          confidence: parseFloat(context.metadata.confidence || '0.7') * context.similarityScore,
          reasoning: `Based on similar content with ${(context.similarityScore * 100).toFixed(1)}% similarity`,
          metadata: {
            source: 'vector_search',
            sourceDocumentId: context.documentId,
            similarityScore: context.similarityScore,
            category: 'similarity_based'
          }
        });
      }

      // Generate contextual suggestions based on similar content patterns
      if (this.hasCommonLegalPatterns(context.content, currentContent)) {
        suggestions.push({
          content: `Consider incorporating similar analysis patterns from related documents that achieved high confidence scores.`,
          type: 'pattern_analysis',
          confidence: context.similarityScore * 0.8,
          reasoning: 'Similar documents used effective legal analysis patterns',
          metadata: {
            source: 'vector_search',
            sourceDocumentId: context.documentId,
            similarityScore: context.similarityScore,
            category: 'pattern_matching'
          }
        });
      }
    });

    return suggestions;
  }

  /**
   * Extract suggestions from graph context
   */
  private async extractSuggestionsFromGraphContext(
    graphContext: GraphContext,
    currentContent: string
  ): Promise<ContextualSuggestion[]> {
    const suggestions: ContextualSuggestion[] = [];

    // Analyze case relationships
    const caseNode = graphContext.relatedNodes.find(node => node.type === 'case');
    const evidenceNodes = graphContext.relatedNodes.filter(node => node.type === 'evidence');

    if (caseNode && evidenceNodes.length > 0) {
      suggestions.push({
        content: `Consider referencing the ${evidenceNodes.length} pieces of evidence associated with this case to strengthen your analysis.`,
        type: 'evidence_integration',
        confidence: 0.8,
        reasoning: 'Multiple evidence items available in case context',
        metadata: {
          source: 'graph_context',
          contextNodes: evidenceNodes.map(node => node.id),
          category: 'case_integration'
        }
      });

      // Analyze evidence patterns
      const evidenceTypes = new Set(evidenceNodes.map(node => node.properties.type).filter(Boolean));
      if (evidenceTypes.size > 1) {
        suggestions.push({
          content: `Cross-reference different types of evidence (${Array.from(evidenceTypes).join(', ')}) to build a comprehensive argument.`,
          type: 'evidence_correlation',
          confidence: 0.75,
          reasoning: 'Multiple evidence types available for cross-validation',
          metadata: {
            source: 'graph_context',
            contextNodes: evidenceNodes.map(node => node.id),
            keywords: Array.from(evidenceTypes),
            category: 'evidence_analysis'
          }
        });
      }
    }

    return suggestions;
  }

  /**
   * Get suggestions based on similar cases using vector search
   */
  private async getSimilarCaseSuggestions(
    embedding: number[],
    currentCaseId: string
  ): Promise<ContextualSuggestion[]> {
    try {
      // Use comprehensive vector search across evidence and case content
      const similarResults = await searchAcrossAllVectors(embedding, {
        limit: 10,
        threshold: 0.4,
        includeEvidence: true,
        includeMessages: false // Focus on evidence for case precedents
      });

      const suggestions: ContextualSuggestion[] = [];
      const seenCases = new Set<string>();

      // Group results by case and create suggestions
      for (const result of similarResults) {
        if (result.documentType === 'evidence' && 
            result.metadata?.caseId && 
            result.metadata.caseId !== currentCaseId &&
            !seenCases.has(result.metadata.caseId)) {
          
          seenCases.add(result.metadata.caseId);

          // Get case information
          const caseInfo = await db
            .select({
              title: cases.title,
              status: cases.status,
              caseType: cases.caseType
            })
            .from(cases)
            .where(eq(cases.id, result.metadata.caseId))
            .limit(1);

          if (caseInfo.length > 0) {
            const case_data = caseInfo[0];
            
            suggestions.push({
              content: `Review similar case "${case_data.title}" which has ${result.metadata.evidenceType} evidence with ${(result.similarity * 100).toFixed(1)}% similarity to your content.`,
              type: 'case_precedent',
              confidence: result.similarity * 0.8,
              reasoning: `Similar evidence found in related case with high vector similarity`,
              metadata: {
                source: 'similar_cases',
                sourceDocumentId: result.metadata.caseId,
                similarityScore: result.similarity,
                keywords: [case_data.status || '', case_data.caseType || '', result.metadata.evidenceType].filter(Boolean),
                category: 'precedent_analysis',
                evidenceType: result.metadata.evidenceType
              }
            });
          }
        }
      }

      return suggestions.slice(0, 3); // Limit similar case suggestions
    } catch (error) {
      console.error('Failed to get similar case suggestions:', error);
      return [];
    }
  }

  /**
   * Get evidence-based contextual suggestions using vector search
   */
  private async getEvidenceBasedSuggestions(
    embedding: number[],
    content: string
  ): Promise<ContextualSuggestion[]> {
    try {
      // Use pgvector for similar evidence search
      const similarEvidence = await searchSimilarEvidence(embedding, undefined, {
        limit: 15,
        threshold: 0.5,
        includeMetadata: true
      });

      const suggestions: ContextualSuggestion[] = [];
      const contentLower = content.toLowerCase();
      const evidenceTypesSeen = new Set<string>();

      for (const result of similarEvidence) {
        const evidenceType = result.metadata?.evidenceType || 'evidence';
        
        // Type-specific authentication suggestion
        if (result.similarity > 0.6) {
          suggestions.push({
            content: `Based on similar ${evidenceType} cases, ensure proper authentication and chain of custody documentation. Consider the admissibility requirements specific to this evidence type.`,
            type: 'evidence_authentication',
            confidence: result.similarity * 0.8,
            reasoning: `High similarity (${(result.similarity * 100).toFixed(1)}%) with ${evidenceType} evidence requiring authentication`,
            metadata: {
              source: 'evidence_analysis',
              sourceDocumentId: result.id,
              similarityScore: result.similarity,
              keywords: [evidenceType, 'authentication', 'chain of custody'],
              category: 'procedural_compliance',
              evidenceType
            }
          });
        }

        // Evidence handling procedures
        if (!evidenceTypesSeen.has(evidenceType) && result.similarity > 0.4) {
          evidenceTypesSeen.add(evidenceType);
          
          suggestions.push({
            content: `For ${evidenceType} evidence similar to your case, review collection procedures and ensure compliance with evidence handling protocols.`,
            type: 'evidence_procedure',
            confidence: result.similarity * 0.7,
            reasoning: `Similar ${evidenceType} evidence found requiring specific handling procedures`,
            metadata: {
              source: 'evidence_analysis',
              sourceDocumentId: result.id,
              similarityScore: result.similarity,
              keywords: [evidenceType, 'procedures', 'handling'],
              category: 'evidence_handling',
              evidenceType
            }
          });
        }
      }

      // Check for direct evidence type mentions in content
      const commonEvidenceTypes = ['digital', 'physical', 'documentary', 'testimonial', 'forensic'];
      commonEvidenceTypes.forEach(type => {
        if (contentLower.includes(type) && !evidenceTypesSeen.has(type)) {
          suggestions.push({
            content: `Given the ${type} evidence mentioned, ensure compliance with specific ${type} evidence standards and admissibility requirements.`,
            type: 'evidence_standards',
            confidence: 0.75,
            reasoning: `Content mentions ${type} evidence which has specific legal requirements`,
            metadata: {
              source: 'evidence_analysis',
              keywords: [type, 'standards', 'admissibility'],
              category: 'legal_compliance',
              evidenceType: type
            }
          });
        }
      });

      return suggestions.slice(0, 5); // Limit evidence suggestions
    } catch (error) {
      console.error('Failed to get evidence-based suggestions:', error);
      return [];
    }
  }

  /**
   * Generate embedding using the enhanced embedding service
   */
  private async generateEmbedding(content: string): Promise<number[]> {
    return await generateEnhancedEmbedding(content, {
      provider: 'nomic-embed',
      legalDomain: true,
      cache: true
    }) as number[];
  }

  /**
   * Calculate cosine similarity between two embeddings
   */
  private calculateCosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      return 0;
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
   * Simple text similarity calculation (placeholder for vector similarity)
   */
  private calculateTextSimilarity(text1: string, text2: string): number {
    const words1 = new Set(text1.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    const words2 = new Set(text2.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    
    const intersection = new Set([...words1].filter(word => words2.has(word)));
    const union = new Set([...words1, ...words2]);
    
    return union.size === 0 ? 0 : intersection.size / union.size;
  }

  /**
   * Check if two pieces of content have common legal patterns
   */
  private hasCommonLegalPatterns(content1: string, content2: string): boolean {
    const legalPatterns = [
      'evidence', 'witness', 'testimony', 'defendant', 'plaintiff', 'statute',
      'precedent', 'ruling', 'motion', 'brief', 'discovery', 'deposition'
    ];

    const patterns1 = legalPatterns.filter(pattern => 
      content1.toLowerCase().includes(pattern)
    );
    const patterns2 = legalPatterns.filter(pattern => 
      content2.toLowerCase().includes(pattern)
    );

    const commonPatterns = patterns1.filter(pattern => patterns2.includes(pattern));
    return commonPatterns.length >= 2;
  }

  /**
   * Deduplicate and rank suggestions by confidence
   */
  private deduplicateAndRankSuggestions(suggestions: ContextualSuggestion[]): ContextualSuggestion[] {
    const seen = new Set<string>();
    const uniqueSuggestions: ContextualSuggestion[] = [];

    suggestions.forEach(suggestion => {
      const key = suggestion.content.toLowerCase().replace(/\s+/g, ' ').trim();
      if (!seen.has(key)) {
        seen.add(key);
        uniqueSuggestions.push(suggestion);
      }
    });

    return uniqueSuggestions
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 8); // Limit total suggestions
  }
}

// Singleton instance
export const vectorSuggestionsService = new VectorSuggestionsService();

/**
 * Convenience function for generating vector contextual suggestions
 */
export async function generateVectorContextualSuggestions(
  content: string,
  reportType: string,
  userId?: string,
  caseId?: string
): Promise<ContextualSuggestion[]> {
  return await vectorSuggestionsService.generateVectorContextualSuggestions(
    content,
    reportType,
    userId,
    caseId
  );
}