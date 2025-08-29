/**
 * LLVM-Quality WebAssembly Gemma3 Integration with Loki Database
 * High-performance legal AI inference with intelligent caching and evidence processing
 */

import { gemma3Service } from './gemma3-local-service';
import { cacheManager } from '../server/cache/loki-cache';
import { lokiEvidenceService } from '../utils/loki-evidence';
import { fuseLazySearch } from './fuse-lazy-search-indexeddb';
import type { Evidence } from '../stores/evidenceStore';

export interface LegalAnalysisRequest {
  content: string;
  title: string;
  caseId?: string;
  analysisType?: 'comprehensive' | 'quick' | 'risk-focused' | 'legal-precedent';
  useCache?: boolean;
  storeResults?: boolean;
  userId?: string;
}

export interface LegalAnalysisResult {
  analysis: {
    summary: string;
    keyFindings: string[];
    legalRisks: Array<{
      type: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      recommendation: string;
    }>;
    entities: string[];
    keyTerms: string[];
    precedents: string[];
    confidence: number;
    method: 'webassembly' | 'ollama' | 'hybrid';
    processingTime: number;
  };
  embeddings: {
    vector: Float32Array;
    dimensions: number;
    model: string;
    processingTime: number;
  };
  caching: {
    cached: boolean;
    cacheKey?: string;
    storageResults?: any;
  };
  evidence?: {
    evidenceId: string;
    indexed: boolean;
    searchable: boolean;
  };
}

export class Gemma3LokiIntegration {
  private initialized = false;
  private performanceMetrics = {
    totalRequests: 0,
    cacheHits: 0,
    webassemblyInferences: 0,
    ollamaFallbacks: 0,
    averageProcessingTime: 0,
    totalProcessingTime: 0
  };

  constructor() {
    this.initialize();
  }

  private async initialize(): Promise<any> {
    if (this.initialized) return;

    try {
      // Initialize all services
      await Promise.all([
        gemma3Service.initialize(),
        cacheManager.db ? Promise.resolve() : new Promise(resolve => {
          cacheManager.on('ready', resolve);
        }),
        lokiEvidenceService.isReady() ? Promise.resolve() : new Promise(resolve => {
          setTimeout(resolve, 1000); // Allow Loki to initialize
        }),
        fuseLazySearch.initialize()
      ]);

      this.initialized = true;
      console.log('🚀 Gemma3-Loki Integration initialized successfully');

    } catch (error: any) {
      console.error('❌ Failed to initialize Gemma3-Loki Integration:', error);
      throw error;
    }
  }

  /**
   * High-performance legal document analysis with intelligent caching
   */
  async analyzeLegalDocument(request: LegalAnalysisRequest): Promise<LegalAnalysisResult> {
    await this.initialize();
    
    const startTime = performance.now();
    this.performanceMetrics.totalRequests++;

    // Generate cache key
    const cacheKey = this.generateCacheKey(request.content, request.analysisType || 'comprehensive');
    
    // Check Loki cache first
    if (request.useCache !== false) {
      const cached = await cacheManager.get(cacheKey);
      if (cached) {
        this.performanceMetrics.cacheHits++;
        console.log(`🎯 Cache hit for analysis: ${cacheKey.substring(0, 16)}...`);
        
        return {
          ...cached,
          caching: {
            cached: true,
            cacheKey
          }
        };
      }
    }

    try {
      // Perform analysis with Gemma3 service (WebAssembly preferred)
      const analysisPromise = this.performLegalAnalysis(request);
      const embeddingsPromise = this.generateEmbeddings(request.content);

      const [analysis, embeddings] = await Promise.all([analysisPromise, embeddingsPromise]);

      const totalTime = performance.now() - startTime;
      this.updateMetrics(totalTime, analysis.method);

      const result: LegalAnalysisResult = {
        analysis,
        embeddings,
        caching: {
          cached: false,
          cacheKey
        }
      };

      // Store in cache with legal-specific metadata
      if (request.useCache !== false) {
        await cacheManager.set(cacheKey, result, {
          contentType: 'legal-analysis',
          confidence: analysis.confidence,
          processingTime: totalTime,
          tags: [
            `analysis-${request.analysisType || 'comprehensive'}`,
            'legal-ai',
            'gemma3-integration',
            ...analysis.entities.slice(0, 5) // Top 5 entities as tags
          ],
          ttl: 3600000, // 1 hour cache
          userId: request.userId,
          sessionId: request.caseId
        });
      }

      // Store as evidence if requested
      if (request.storeResults && request.userId) {
        const evidenceResult = await this.storeAsEvidence(request, result);
        result.evidence = evidenceResult;
      }

      return result;

    } catch (error: any) {
      console.error('❌ Legal analysis failed:', error);
      throw error;
    }
  }

  /**
   * Perform legal analysis using WebAssembly-first approach
   */
  private async performLegalAnalysis(request: LegalAnalysisRequest): Promise<LegalAnalysisResult['analysis']> {
    const startTime = performance.now();

    try {
      // Try WebAssembly first for LLVM-quality performance
      const result = await gemma3Service.analyzeDocument(
        request.title,
        request.content,
        request.analysisType || 'comprehensive'
      );

      if (result.success) {
        this.performanceMetrics.webassemblyInferences++;
        
        // Enhance with legal-specific analysis
        const legalAnalysis = await this.enhanceLegalAnalysis(result, request.content);
        
        return {
          ...legalAnalysis,
          method: 'webassembly',
          processingTime: performance.now() - startTime
        };
      }
    } catch (wasmError) {
      console.warn('⚠️ WebAssembly inference failed, falling back to Ollama:', wasmError);
    }

    // Fallback to Ollama
    this.performanceMetrics.ollamaFallbacks++;
    
    try {
      const ollamaResult = await this.performOllamaAnalysis(request);
      return {
        ...ollamaResult,
        method: 'ollama',
        processingTime: performance.now() - startTime
      };
    } catch (ollamaError) {
      console.error('❌ Both WebAssembly and Ollama failed:', ollamaError);
      throw ollamaError;
    }
  }

  /**
   * Enhance analysis with legal-specific processing
   */
  private async enhanceLegalAnalysis(basicResult: any, content: string): Promise<any> {
    // Extract legal entities using pattern matching
    const legalPatterns = {
      cases: /([A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+)/g,
      citations: /\b\d+\s+[A-Z][\w.]*\s*\d+\b/g,
      statutes: /\b\d+\s+U\.?S\.?C\.?\s*§?\s*\d+/g,
      courts: /\b(Supreme Court|Court of Appeals|District Court|Circuit Court)\b/gi,
      parties: /\b(plaintiff|defendant|appellant|appellee|petitioner|respondent)\b/gi
    };

    const entities = [];
    for (const [type, pattern] of Object.entries(legalPatterns)) {
      const matches = content.match(pattern);
      if (matches) {
        entities.push(...matches.map(match => `${type}:${match.trim()}`));
      }
    }

    // Risk assessment based on legal keywords
    const riskKeywords = {
      high: ['criminal', 'felony', 'breach', 'violation', 'damages', 'liable'],
      medium: ['contract', 'dispute', 'claim', 'negligence', 'injury'],
      low: ['agreement', 'terms', 'conditions', 'compliance', 'policy']
    };

    const risks = [];
    for (const [severity, keywords] of Object.entries(riskKeywords)) {
      for (const keyword of keywords) {
        if (content.toLowerCase().includes(keyword)) {
          risks.push({
            type: 'legal-risk',
            severity: severity as 'low' | 'medium' | 'high',
            description: `Document contains ${severity}-risk legal term: ${keyword}`,
            recommendation: `Review ${keyword} implications with legal counsel`
          });
        }
      }
    }

    return {
      summary: basicResult.summary || 'Legal document analysis completed',
      keyFindings: basicResult.keyPoints || [],
      legalRisks: risks,
      entities: [...new Set(entities)],
      keyTerms: basicResult.keyTerms || [],
      precedents: entities.filter(e => e.startsWith('cases:')),
      confidence: Math.min(basicResult.confidence || 0.8, 0.95) // Cap confidence for legal analysis
    };
  }

  /**
   * Fallback Ollama analysis
   */
  private async performOllamaAnalysis(request: LegalAnalysisRequest): Promise<any> {
    const prompt = this.buildLegalAnalysisPrompt(request);
    
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal',
        prompt,
        stream: false,
        options: {
          temperature: 0.1,
          top_p: 0.9,
          max_tokens: 2048
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama analysis failed: ${response.status}`);
    }

    const result = await response.json();
    
    // Parse structured response
    return this.parseOllamaResponse(result.response);
  }

  /**
   * Generate embeddings using optimized service
   */
  private async generateEmbeddings(content: string): Promise<LegalAnalysisResult['embeddings']> {
    const startTime = performance.now();
    
    try {
      const result = await gemma3Service.generateEmbeddings(content);
      
      return {
        vector: new Float32Array(result.embedding),
        dimensions: result.dimensions,
        model: result.model || 'nomic-embed-text',
        processingTime: performance.now() - startTime
      };
    } catch (error: any) {
      console.warn('⚠️ Embedding generation failed, using fallback');
      
      // Fallback to direct nomic-embed-text
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: content.substring(0, 1000) // Limit for embedding
        })
      });

      const embeddingData = await response.json();
      
      return {
        vector: new Float32Array(embeddingData.embedding),
        dimensions: embeddingData.embedding.length,
        model: 'nomic-embed-text',
        processingTime: performance.now() - startTime
      };
    }
  }

  /**
   * Store analysis results as evidence
   */
  private async storeAsEvidence(
    request: LegalAnalysisRequest, 
    result: LegalAnalysisResult
  ): Promise<{ evidenceId: string; indexed: boolean; searchable: boolean }> {
    const evidenceId = crypto.randomUUID();
    
    const evidence: Evidence = {
      id: evidenceId,
      fileName: request.title,
      description: result.analysis.summary,
      type: 'ai-analysis',
      caseId: request.caseId || 'default',
      filePath: '', // Generated document
      fileSize: request.content.length,
      mimeType: 'application/json',
      uploadedBy: request.userId || 'system',
      tags: result.analysis.keyTerms,
      timeline: {
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      aiAnalysis: {
        confidence: result.analysis.confidence,
        riskLevel: this.calculateRiskLevel(result.analysis.legalRisks),
        entities: result.analysis.entities,
        processed: true,
        model: result.analysis.method,
        processingTime: result.analysis.processingTime
      }
    };

    // Store in Loki evidence database
    await lokiEvidenceService.createEvidence(evidence);
    
    // Index in search system
    await fuseLazySearch.addItem({
      id: evidenceId,
      title: request.title,
      content: request.content,
      keywords: result.analysis.keyTerms,
      embedding: result.embeddings.vector,
      metadata: {
        type: 'ai-analysis',
        caseId: request.caseId,
        confidence: result.analysis.confidence,
        riskLevel: evidence.aiAnalysis?.riskLevel,
        entities: result.analysis.entities
      }
    });

    return {
      evidenceId,
      indexed: true,
      searchable: true
    };
  }

  /**
   * Search legal documents and evidence
   */
  async searchLegalContent(
    query: string, 
    options: {
      caseId?: string;
      analysisType?: string;
      includeEmbeddings?: boolean;
      maxResults?: number;
    } = {}
  ): Promise<any[]> {
    await this.initialize();

    // Search cached analyses
    const cacheResults = await cacheManager.semanticSearch(query, options.maxResults || 10);
    
    // Search evidence database
    const evidenceResults = lokiEvidenceService.searchEvidence(query);
    
    // Search with embeddings if requested
    let semanticResults = [];
    if (options.includeEmbeddings) {
      semanticResults = await fuseLazySearch.search(query, {
        useEmbeddings: true,
        maxResults: options.maxResults || 10,
        threshold: 0.7
      });
    }

    // Combine and deduplicate results
    const allResults = [
      ...cacheResults.map(r => ({ ...r, source: 'cache' })),
      ...evidenceResults.map(r => ({ ...r, source: 'evidence' })),
      ...semanticResults.map(r => ({ ...r.item, source: 'semantic', similarity: r.similarity }))
    ];

    // Filter by caseId if specified
    if (options.caseId) {
      return allResults.filter(r => 
        r.caseId === options.caseId || 
        r.metadata?.caseId === options.caseId
      );
    }

    return allResults
      .slice(0, options.maxResults || 20)
      .sort((a, b) => (b.similarity || b.confidence || 0) - (a.similarity || a.confidence || 0));
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats() {
    const hitRate = this.performanceMetrics.totalRequests > 0 
      ? (this.performanceMetrics.cacheHits / this.performanceMetrics.totalRequests) * 100 
      : 0;

    return {
      ...this.performanceMetrics,
      averageProcessingTime: this.performanceMetrics.totalRequests > 0
        ? this.performanceMetrics.totalProcessingTime / this.performanceMetrics.totalRequests
        : 0,
      cacheHitRate: hitRate,
      webassemblySuccessRate: this.performanceMetrics.totalRequests > 0
        ? (this.performanceMetrics.webassemblyInferences / this.performanceMetrics.totalRequests) * 100
        : 0,
      cacheStats: cacheManager.getStats(),
      searchStats: fuseLazySearch.getStats(),
      evidenceStats: lokiEvidenceService.getEvidenceStats()
    };
  }

  // Helper methods
  private generateCacheKey(content: string, analysisType: string): string {
    const contentHash = this.simpleHash(content);
    return `legal-analysis:${analysisType}:${contentHash}`;
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
  }

  private updateMetrics(processingTime: number, method: string): void {
    this.performanceMetrics.totalProcessingTime += processingTime;
    this.performanceMetrics.averageProcessingTime = 
      this.performanceMetrics.totalProcessingTime / this.performanceMetrics.totalRequests;

    if (method === 'webassembly') {
      this.performanceMetrics.webassemblyInferences++;
    } else if (method === 'ollama') {
      this.performanceMetrics.ollamaFallbacks++;
    }
  }

  private calculateRiskLevel(risks: any[]): 'low' | 'medium' | 'high' | 'critical' {
    if (risks.some(r => r.severity === 'critical')) return 'critical';
    if (risks.some(r => r.severity === 'high')) return 'high';
    if (risks.some(r => r.severity === 'medium')) return 'medium';
    return 'low';
  }

  private buildLegalAnalysisPrompt(request: LegalAnalysisRequest): string {
    return `
Analyze this legal document and provide a structured response:

Title: ${request.title}
Analysis Type: ${request.analysisType || 'comprehensive'}

Document Content:
${request.content}

Provide analysis in the following format:
SUMMARY: [Brief overview]
KEY_FINDINGS: [Important points]
LEGAL_RISKS: [Potential risks and their severity]
ENTITIES: [Legal entities, cases, statutes mentioned]
KEY_TERMS: [Important legal terms]
PRECEDENTS: [Relevant case law]
CONFIDENCE: [Analysis confidence 0-1]

Focus on legal implications, risks, and actionable insights.
    `.trim();
  }

  private parseOllamaResponse(response: string): any {
    // Simple parsing - in production, use more sophisticated NLP
    const sections = response.split(/(?:SUMMARY|KEY_FINDINGS|LEGAL_RISKS|ENTITIES|KEY_TERMS|PRECEDENTS|CONFIDENCE):/);
    
    return {
      summary: sections[1]?.trim() || 'Analysis completed',
      keyFindings: sections[2]?.split('\n').filter(s => s.trim()) || [],
      legalRisks: sections[3]?.split('\n').map(risk => ({
        type: 'legal-risk',
        severity: 'medium' as const,
        description: risk.trim(),
        recommendation: 'Review with legal counsel'
      })).filter(r => r.description) || [],
      entities: sections[4]?.split('\n').filter(s => s.trim()) || [],
      keyTerms: sections[5]?.split('\n').filter(s => s.trim()) || [],
      precedents: sections[6]?.split('\n').filter(s => s.trim()) || [],
      confidence: parseFloat(sections[7]?.trim()) || 0.8
    };
  }
}

// Export singleton instance
export const gemma3LokiIntegration = new Gemma3LokiIntegration();

// Export for API use
export default gemma3LokiIntegration;