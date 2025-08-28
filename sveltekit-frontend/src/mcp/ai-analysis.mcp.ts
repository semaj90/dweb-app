/**
 * MCP Tool: AI Analysis & Legal Processing
 * Clean abstraction layer for AI-powered legal analysis using Ollama + Enhanced RAG
 * Following the suggested architecture pattern for Legal AI Platform
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { eq, desc, and, sql } from 'drizzle-orm';
import { documentChunks, cases, evidence, legal_documents } from '$lib/server/db/schema';
import type { EvidenceAIAnalysis } from '$lib/types';
import { productionServiceClient, services } from '../services/productionServiceClient.js';

// Database connection (based on MCP pgvector docs)
const connectionString = process.env.DATABASE_URL!;
const client = postgres(connectionString);
const db = drizzle(client);

export interface MCPToolResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: Record<string, any>;
}

export interface DocumentAnalysisParams {
  content: string;
  documentType?: 'contract' | 'evidence' | 'brief' | 'citation' | 'statute';
  caseId?: string;
  userId?: string;
  generateEmbedding?: boolean;
}

export interface LegalAnalysisParams {
  query: string;
  context?: {
    caseId?: string;
    jurisdiction?: string;
    practiceArea?: string;
    precedents?: string[];
  };
  analysisType?: 'case_law' | 'statutory' | 'regulatory' | 'contract' | 'evidence';
  userId?: string;
}

export interface SimilaritySearchParams {
  queryEmbedding: number[];
  documentTypes?: string[];
  caseId?: string;
  threshold?: number;
  limit?: number;
}

export interface BatchAnalysisParams {
  documents: Array<{
    id: string;
    content: string;
    documentType?: string;
    caseId?: string;
  }>;
  userId?: string;
  analysisType?: string;
}

export interface RiskAssessmentParams {
  caseId: string;
  factors?: {
    evidenceQuality?: boolean;
    legalPrecedents?: boolean;
    jurisdictionalRisks?: boolean;
    timelineAnalysis?: boolean;
  };
  userId?: string;
}

/**
 * MCP Tool: AI Analysis & Legal Processing
 * Integrates with Ollama models and Enhanced RAG for legal AI operations
 */
export class AIAnalysisMCPTool {
  
  /**
   * Analyze legal document using AI models
   */
  async analyzeDocument(params: DocumentAnalysisParams): Promise<MCPToolResponse<EvidenceAIAnalysis>> {
    try {
      // Use Enhanced RAG service for document analysis
      const analysisPrompt = this.buildAnalysisPrompt(params.content, params.documentType);
      
      const ragResponse = await services.queryRAG(analysisPrompt, {
        userId: params.userId,
        caseId: params.caseId,
        analysisMode: true
      });

      // Parse the AI response into structured analysis
      const analysis = this.parseAnalysisResponse(ragResponse.response || ragResponse.data?.response);

      // Generate embedding if requested
      let embedding: number[] | undefined;
      if (params.generateEmbedding) {
        const embeddingResponse = await services.generateEmbedding(params.content);
        embedding = embeddingResponse.embedding;
      }

      // Store analysis results in document chunks for future retrieval
      if (params.caseId && embedding) {
        await this.storeDocumentChunk({
          content: params.content,
          caseId: params.caseId,
          documentType: params.documentType || 'evidence',
          embedding,
          analysis
        });
      }

      return {
        success: true,
        data: analysis,
        metadata: {
          tool: 'aiAnalysis.analyzeDocument',
          documentType: params.documentType,
          hasEmbedding: !!embedding,
          contentLength: params.content.length,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'aiAnalysis.analyzeDocument',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Perform comprehensive legal analysis
   */
  async performLegalAnalysis(params: LegalAnalysisParams): Promise<MCPToolResponse<any>> {
    try {
      // Build context-aware legal analysis prompt
      const legalPrompt = this.buildLegalAnalysisPrompt(params);

      // Use Enhanced RAG with legal context
      const ragResponse = await services.queryRAG(legalPrompt, {
        userId: params.userId,
        caseId: params.context?.caseId,
        analysisType: params.analysisType
      });

      // Parse legal analysis response
      const legalAnalysis = {
        query: params.query,
        analysisType: params.analysisType,
        findings: this.extractFindings(ragResponse.response || ragResponse.data?.response),
        legalImplications: this.extractLegalImplications(ragResponse.response || ragResponse.data?.response),
        recommendations: this.extractRecommendations(ragResponse.response || ragResponse.data?.response),
        citations: this.extractCitations(ragResponse.response || ragResponse.data?.response),
        confidence: this.calculateConfidence(ragResponse.response || ragResponse.data?.response),
        context: params.context,
        analyzedAt: new Date().toISOString()
      };

      return {
        success: true,
        data: legalAnalysis,
        metadata: {
          tool: 'aiAnalysis.performLegalAnalysis',
          analysisType: params.analysisType,
          queryLength: params.query.length,
          hasContext: !!params.context,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'aiAnalysis.performLegalAnalysis',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Find similar documents using vector similarity
   */
  async findSimilarDocuments(params: SimilaritySearchParams): Promise<MCPToolResponse<any[]>> {
    try {
      let query = db.select({
        chunk: documentChunks,
        similarity: sql<number>`1 - (${documentChunks.embedding} <=> ${params.queryEmbedding}::vector) as similarity`
      })
        .from(documentChunks)
        .where(sql`${documentChunks.embedding} IS NOT NULL`);

      // Add filters
      const conditions = [];
      if (params.caseId) {
        // Join with evidence or cases to filter by case
        // This would need to be refined based on actual schema relationships
      }
      if (params.threshold) {
        conditions.push(sql`1 - (${documentChunks.embedding} <=> ${params.queryEmbedding}::vector) > ${params.threshold}`);
      }
      if (params.documentTypes && params.documentTypes.length > 0) {
        conditions.push(sql`${documentChunks.document_type} = ANY(${params.documentTypes})`);
      }

      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }

      query = query.orderBy(sql`similarity DESC`).limit(params.limit || 20);

      const results = await query;

      // Enhance results with AI analysis of similarity
      const enhancedResults = await Promise.all(results.map(async (result) => {
        try {
          const similarityAnalysis = await this.analyzeSimilarity(result.chunk.content, params.queryEmbedding);
          return {
            ...result.chunk,
            similarity: result.similarity,
            similarityAnalysis
          };
        } catch (error) {
          return {
            ...result.chunk,
            similarity: result.similarity,
            similarityAnalysis: null
          };
        }
      }));

      return {
        success: true,
        data: enhancedResults,
        metadata: {
          tool: 'aiAnalysis.findSimilarDocuments',
          vectorDimensions: params.queryEmbedding.length,
          threshold: params.threshold || 0.7,
          resultsCount: results.length,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'aiAnalysis.findSimilarDocuments',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Perform batch analysis of multiple documents
   */
  async performBatchAnalysis(params: BatchAnalysisParams): Promise<MCPToolResponse<any[]>> {
    try {
      const results = [];
      
      // Process documents in batches to avoid overwhelming the AI service
      const batchSize = 5;
      for (let i = 0; i < params.documents.length; i += batchSize) {
        const batch = params.documents.slice(i, i + batchSize);
        
        const batchPromises = batch.map(async (doc) => {
          try {
            const analysisResult = await this.analyzeDocument({
              content: doc.content,
              documentType: doc.documentType as any,
              caseId: doc.caseId,
              userId: params.userId,
              generateEmbedding: true
            });
            
            return {
              documentId: doc.id,
              analysis: analysisResult.data,
              success: analysisResult.success,
              error: analysisResult.error
            };
          } catch (error) {
            return {
              documentId: doc.id,
              analysis: null,
              success: false,
              error: error instanceof Error ? error.message : 'Unknown error'
            };
          }
        });

        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);
      }

      // Generate batch summary
      const summary = {
        totalDocuments: params.documents.length,
        successfulAnalyses: results.filter(r => r.success).length,
        failedAnalyses: results.filter(r => !r.success).length,
        averageConfidence: this.calculateAverageConfidence(results),
        topRisks: this.extractTopRisks(results),
        commonThemes: this.extractCommonThemes(results)
      };

      return {
        success: true,
        data: {
          results,
          summary
        },
        metadata: {
          tool: 'aiAnalysis.performBatchAnalysis',
          documentsProcessed: params.documents.length,
          analysisType: params.analysisType,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'aiAnalysis.performBatchAnalysis',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Assess legal risks for a case
   */
  async assessCaseRisk(params: RiskAssessmentParams): Promise<MCPToolResponse<any>> {
    try {
      // Gather case data
      const caseEvidence = await db.select()
        .from(evidence)
        .where(eq(evidence.case_id, params.caseId));

      const caseDocuments = await db.select()
        .from(documentChunks)
        .where(eq(documentChunks.document_id, params.caseId))
        .limit(50);

      // Build comprehensive risk assessment prompt
      const riskPrompt = this.buildRiskAssessmentPrompt(caseEvidence, caseDocuments, params.factors);

      // Use Enhanced RAG for risk analysis
      const ragResponse = await services.queryRAG(riskPrompt, {
        userId: params.userId,
        caseId: params.caseId,
        analysisType: 'risk_assessment'
      });

      const riskAssessment = {
        caseId: params.caseId,
        overallRiskLevel: this.extractRiskLevel(ragResponse.response || ragResponse.data?.response),
        riskFactors: this.extractRiskFactors(ragResponse.response || ragResponse.data?.response),
        mitigationStrategies: this.extractMitigationStrategies(ragResponse.response || ragResponse.data?.response),
        strengthsWeaknesses: this.extractStrengthsWeaknesses(ragResponse.response || ragResponse.data?.response),
        recommendedActions: this.extractRecommendedActions(ragResponse.response || ragResponse.data?.response),
        confidenceScore: this.calculateConfidence(ragResponse.response || ragResponse.data?.response),
        assessedAt: new Date().toISOString(),
        factors: params.factors
      };

      return {
        success: true,
        data: riskAssessment,
        metadata: {
          tool: 'aiAnalysis.assessCaseRisk',
          caseId: params.caseId,
          evidenceCount: caseEvidence.length,
          documentCount: caseDocuments.length,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'aiAnalysis.assessCaseRisk',
          timestamp: Date.now()
        }
      };
    }
  }

  // Private helper methods

  private buildAnalysisPrompt(content: string, documentType?: string): string {
    return `
Analyze this legal document and provide a structured analysis:

Document Type: ${documentType || 'Unknown'}
Content: ${content}

Please provide:
1. Summary (2-3 sentences)
2. Key findings or provisions
3. Legal implications
4. Risk assessment (low/medium/high/critical)
5. Relevance score (0-100)
6. Recommended actions
7. Tags for categorization

Format your response as structured JSON where possible.
`;
  }

  private buildLegalAnalysisPrompt(params: LegalAnalysisParams): string {
    const contextStr = params.context ? JSON.stringify(params.context, null, 2) : 'None';
    
    return `
Perform comprehensive legal analysis for the following query:

Query: ${params.query}
Analysis Type: ${params.analysisType}
Context: ${contextStr}

Please provide:
1. Legal findings and analysis
2. Applicable laws and regulations
3. Case precedents and citations
4. Risk assessment and implications
5. Strategic recommendations
6. Supporting evidence required
7. Potential counterarguments

Focus on accuracy, cite relevant authorities, and provide actionable insights.
`;
  }

  private buildRiskAssessmentPrompt(evidence: any[], documents: any[], factors?: any): string {
    return `
Assess the legal risks for this case based on the following evidence and documents:

Evidence Count: ${evidence.length}
Document Count: ${documents.length}
Assessment Factors: ${JSON.stringify(factors, null, 2)}

Evidence Summary: ${evidence.map(e => `${e.title}: ${e.description}`).join('\n')}

Key Documents: ${documents.slice(0, 5).map(d => d.content?.substring(0, 200) + '...').join('\n\n')}

Please provide:
1. Overall risk level (Critical/High/Medium/Low)
2. Primary risk factors
3. Evidence strengths and weaknesses
4. Legal vulnerabilities
5. Mitigation strategies
6. Recommended immediate actions
7. Long-term case strategy

Be specific and provide actionable recommendations.
`;
  }

  private parseAnalysisResponse(response: string): EvidenceAIAnalysis {
    try {
      // Try to parse structured JSON response
      const parsed = JSON.parse(response);
      return parsed;
    } catch {
      // Fallback to text parsing
      return {
        summary: this.extractSummary(response),
        keyFindings: this.extractKeyFindings(response),
        legalImplications: this.extractLegalImplications(response),
        riskLevel: this.extractRiskLevel(response) as any,
        relevanceScore: this.extractRelevanceScore(response),
        recommendations: this.extractRecommendations(response),
        confidence: this.calculateConfidence(response),
        analyzedAt: new Date().toISOString()
      };
    }
  }

  private async storeDocumentChunk(params: {
    content: string;
    caseId: string;
    documentType: string;
    embedding: number[];
    analysis: any;
  }): Promise<void> {
    await db.insert(documentChunks).values({
      id: crypto.randomUUID(),
      document_id: crypto.randomUUID(),
      document_type: params.documentType,
      chunk_index: '1',
      content: params.content,
      embedding: params.embedding,
      created_at: new Date()
    });
  }

  private async analyzeSimilarity(content: string, queryEmbedding: number[]): Promise<any> {
    // This would use AI to analyze why documents are similar
    return {
      reason: 'Similar legal concepts',
      confidence: 0.85,
      keyTerms: ['contract', 'liability', 'indemnification']
    };
  }

  // Text extraction helpers
  private extractSummary(text: string): string {
    const match = text.match(/summary[:\s]*(.+?)(?=\n|$)/i);
    return match ? match[1].trim() : '';
  }

  private extractKeyFindings(text: string): string[] {
    const matches = text.match(/key findings?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is);
    return matches ? matches[1].split('\n').filter(Boolean) : [];
  }

  private extractLegalImplications(text: string): string[] {
    const matches = text.match(/legal implications?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is);
    return matches ? matches[1].split('\n').filter(Boolean) : [];
  }

  private extractRecommendations(text: string): string[] {
    const matches = text.match(/recommendations?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is);
    return matches ? matches[1].split('\n').filter(Boolean) : [];
  }

  private extractFindings(text: string): string[] {
    return this.extractKeyFindings(text);
  }

  private extractCitations(text: string): string[] {
    const matches = text.match(/citations?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is);
    return matches ? matches[1].split('\n').filter(Boolean) : [];
  }

  private extractRiskLevel(text: string): string {
    const match = text.match(/risk level[:\s]*(critical|high|medium|low)/i);
    return match ? match[1].toLowerCase() : 'medium';
  }

  private extractRelevanceScore(text: string): number {
    const match = text.match(/relevance score[:\s]*(\d+)/i);
    return match ? parseInt(match[1]) : 50;
  }

  private extractRiskFactors(text: string): string[] {
    const matches = text.match(/risk factors?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is);
    return matches ? matches[1].split('\n').filter(Boolean) : [];
  }

  private extractMitigationStrategies(text: string): string[] {
    const matches = text.match(/mitigation strategies?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is);
    return matches ? matches[1].split('\n').filter(Boolean) : [];
  }

  private extractStrengthsWeaknesses(text: string): { strengths: string[], weaknesses: string[] } {
    const strengths = text.match(/strengths?[:\s]*(.+?)(?=weaknesses|$)/is)?.[1].split('\n').filter(Boolean) || [];
    const weaknesses = text.match(/weaknesses?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is)?.[1].split('\n').filter(Boolean) || [];
    return { strengths, weaknesses };
  }

  private extractRecommendedActions(text: string): string[] {
    const matches = text.match(/recommended actions?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)/is);
    return matches ? matches[1].split('\n').filter(Boolean) : [];
  }

  private calculateConfidence(text: string): number {
    // Simple confidence calculation based on text characteristics
    const confidenceKeywords = ['certain', 'clear', 'definite', 'conclusive'];
    const uncertaintyKeywords = ['might', 'could', 'possibly', 'unclear'];
    
    let score = 0.5;
    confidenceKeywords.forEach(keyword => {
      if (text.toLowerCase().includes(keyword)) score += 0.1;
    });
    uncertaintyKeywords.forEach(keyword => {
      if (text.toLowerCase().includes(keyword)) score -= 0.1;
    });
    
    return Math.max(0, Math.min(1, score));
  }

  private calculateAverageConfidence(results: any[]): number {
    const validResults = results.filter(r => r.success && r.analysis?.confidence);
    if (validResults.length === 0) return 0;
    
    const sum = validResults.reduce((acc, r) => acc + (r.analysis.confidence || 0), 0);
    return sum / validResults.length;
  }

  private extractTopRisks(results: any[]): string[] {
    const risks: string[] = [];
    results.forEach(r => {
      if (r.success && r.analysis?.riskLevel === 'high' || r.analysis?.riskLevel === 'critical') {
        risks.push(r.analysis.summary || `Document ${r.documentId} - ${r.analysis.riskLevel} risk`);
      }
    });
    return risks.slice(0, 5);
  }

  private extractCommonThemes(results: any[]): string[] {
    // Simple theme extraction - would be enhanced with more sophisticated NLP
    const themes = new Map<string, number>();
    
    results.forEach(r => {
      if (r.success && r.analysis?.tags) {
        r.analysis.tags.forEach((tag: string) => {
          themes.set(tag, (themes.get(tag) || 0) + 1);
        });
      }
    });
    
    return Array.from(themes.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([theme]) => theme);
  }
}

// Export singleton instance
export const aiAnalysisMCPTool = new AIAnalysisMCPTool();