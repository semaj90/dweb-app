/**
 * AI Assistant Input Synthesizer
 * Synthesizes and enhances user inputs with LegalBERT middleware, context enrichment,
 * and intelligent prompt engineering for optimal AI assistant responses
 */

// Lightweight local aliases (server middleware types not imported here to avoid path resolution issues)
export type LegalAnalysisResult = any; // TODO: replace with shared type import
export type LegalEmbeddingResult = any; // TODO: replace with shared type import

export interface SynthesizedInput {
  originalQuery: string;
  enhancedPrompt: string;
  legalContext: LegalContext;
  intent: QueryIntent;
  embedding: number[];
  metadata: InputMetadata;
  recommendations: string[];
  contextualPrompts: ContextualPrompt[];
}

export interface LegalContext {
  entities: Array<{
    text: string;
    type: string;
    confidence: number;
    relevance: number;
  }>;
  concepts: Array<{
    concept: string;
    category: string;
    relevance: number;
  }>;
  citations: Array<{
    citation: string;
    type: 'case' | 'statute' | 'regulation';
    jurisdiction?: string;
  }>;
  keyTerms: string[];
  complexity: number;
  domain: string;
}

export interface QueryIntent {
  primary: string;
  secondary: string[];
  confidence: number;
  category: 'analysis' | 'research' | 'drafting' | 'review' | 'advice' | 'general';
  urgency: 'low' | 'medium' | 'high' | 'critical';
  scope: 'document' | 'case' | 'research' | 'procedural' | 'substantive';
}

export interface InputMetadata {
  userRole?: 'prosecutor' | 'defense' | 'judge' | 'paralegal' | 'student' | 'client';
  jurisdiction?: string;
  caseId?: string;
  documentIds?: string[];
  sessionContext?: unknown;
  timestamp: string;
  quality: number;
  processingTime: number;
}

export interface ContextualPrompt {
  type: 'clarification' | 'elaboration' | 'related' | 'followup';
  prompt: string;
  confidence: number;
  category: string;
}

export interface SynthesizerConfig {
  enableLegalBERT: boolean;
  enableContextEnrichment: boolean;
  maxPromptLength: number;
  includeExamples: boolean;
  enhancementLevel: 'basic' | 'standard' | 'advanced' | 'comprehensive';
  cacheResults: boolean;
  timeoutMs: number;
}

export class AIAssistantInputSynthesizer {
  private config: SynthesizerConfig;
  private legalBERT: any = null;
  private cache = new Map<string, SynthesizedInput>();
  private intentPatterns: Map<string, RegExp[]>;

  constructor(config: Partial<SynthesizerConfig> = {}) {
    this.config = {
      enableLegalBERT: true,
      enableContextEnrichment: true,
      maxPromptLength: 4000,
      includeExamples: true,
      enhancementLevel: 'comprehensive',
      cacheResults: true,
      timeoutMs: 10000,
      ...config,
    };

    this.initializeIntentPatterns();
    this.initializeLegalBERT();
  }

  private initializeIntentPatterns(): void {
    this.intentPatterns = new Map([
      [
        'analysis',
        [
          /analyz[e|ing|ed]/i,
          /review[ing|ed]?/i,
          /examin[e|ing|ed]/i,
          /assess[ing|ment]?/i,
          /evaluat[e|ing|ed]/i,
          /interpret[ing|ation]?/i,
        ],
      ],
      [
        'research',
        [
          /research[ing]?/i,
          /find[ing]?/i,
          /search[ing]?/i,
          /look\s+up/i,
          /precedent[s]?/i,
          /case\s+law/i,
          /statute[s]?/i,
        ],
      ],
      [
        'drafting',
        [
          /draft[ing]?/i,
          /writ[e|ing]/i,
          /creat[e|ing]/i,
          /prepar[e|ing]/i,
          /compos[e|ing]/i,
          /formuat[e|ing]/i,
        ],
      ],
      [
        'review',
        [
          /review[ing]?/i,
          /check[ing]?/i,
          /verify[ing|ification]?/i,
          /validat[e|ing]/i,
          /confirm[ing]?/i,
        ],
      ],
      [
        'advice',
        [
          /advic[e]?/i,
          /recommend[ation|ing]?/i,
          /suggest[ion|ing]?/i,
          /guid[e|ance]/i,
          /help[ing]?/i,
          /what\s+should/i,
        ],
      ],
    ]);
  }

  private async initializeLegalBERT(): Promise<void> {
    if (!this.config.enableLegalBERT) return;

    try {
      const { legalBERT } = await import('../server/ai/legalbert-middleware');
      this.legalBERT = legalBERT;
    } catch (error) {
      console.warn('LegalBERT middleware not available, using fallback analysis');
      this.legalBERT = null;
    }
  }

  /**
   * Main synthesis method - enhances user input with context and intent analysis
   */
  async synthesizeInput(
    query: string,
    context?: {
      userRole?: string;
      caseId?: string;
      documentIds?: string[];
      sessionContext?: unknown;
    }
  ): Promise<SynthesizedInput> {
    const startTime = Date.now();
    const cacheKey = this.generateCacheKey(query, context);

    // Check cache first
    if (this.config.cacheResults && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    try {
      // Parallel processing for better performance
      const [legalAnalysis, embedding, intent, enhancedPrompt, contextualPrompts] =
        await Promise.all([
          this.performLegalAnalysis(query),
          this.generateEmbedding(query),
          this.detectIntent(query),
          this.enhancePrompt(query, context),
          this.generateContextualPrompts(query),
        ]);

      const synthesized: SynthesizedInput = {
        originalQuery: query,
        enhancedPrompt,
        legalContext: this.buildLegalContext(legalAnalysis, query),
        intent,
        embedding: embedding.embedding || [],
        metadata: {
          userRole: context?.userRole as any,
          caseId: context?.caseId,
          documentIds: context?.documentIds,
          sessionContext: context?.sessionContext,
          timestamp: new Date().toISOString(),
          quality: this.calculateQuality(query, legalAnalysis, intent),
          processingTime: Date.now() - startTime,
        },
        recommendations: this.generateRecommendations(legalAnalysis, intent),
        contextualPrompts,
      };

      // Cache result
      if (this.config.cacheResults) {
        this.cache.set(cacheKey, synthesized);
      }

      return synthesized;
    } catch (error) {
      console.error('Input synthesis failed:', error);
      return this.createFallbackSynthesis(query, context);
    }
  }

  /**
   * Perform legal analysis using LegalBERT middleware
   */
  private async performLegalAnalysis(query: string): Promise<LegalAnalysisResult | null> {
    if (!this.legalBERT) {
      return this.performBasicLegalAnalysis(query);
    }

    try {
      return await this.legalBERT.analyzeLegalText(query, {
        includeEntities: true,
        includeConcepts: true,
        includeSentiment: true,
        includeComplexity: true,
      });
    } catch (error) {
      console.warn('LegalBERT analysis failed, using fallback:', error);
      return this.performBasicLegalAnalysis(query);
    }
  }

  /**
   * Basic legal analysis fallback
   */
  private performBasicLegalAnalysis(query: string): LegalAnalysisResult {
    const legalTerms = this.extractLegalTerms(query);
    const complexity = this.calculateTextComplexity(query);

    return {
      entities: legalTerms.map((term, index) => ({
        text: term,
        type: 'LEGAL_CONCEPT' as any,
        confidence: 0.7,
        startIndex: query.indexOf(term),
        endIndex: query.indexOf(term) + term.length,
      })),
      concepts: legalTerms.map((term) => ({
        concept: term,
        relevance: 0.8,
        category: 'legal',
      })),
      sentiment: {
        polarity: 0,
        confidence: 0.5,
        classification: 'neutral' as any,
      },
      complexity: {
        readabilityScore: complexity,
        legalComplexity: legalTerms.length / 10,
        technicalTerms: legalTerms.length,
      },
      keyPhrases: legalTerms.map((term) => ({
        phrase: term,
        importance: 0.7,
        category: 'legal',
      })),
      summary: {
        abstractive: query.substring(0, 100) + '...',
        extractive: [query.split('.')[0] || query],
        keyPoints: legalTerms.slice(0, 3),
      },
    };
  }

  /**
   * Generate embeddings using available services
   */
  private async generateEmbedding(query: string): Promise<LegalEmbeddingResult> {
    if (this.legalBERT) {
      try {
        return await this.legalBERT.generateEmbedding(query);
      } catch (error) {
        console.warn('LegalBERT embedding failed:', error);
      }
    }

    // Fallback to basic embedding
    return {
      embedding: this.generateBasicEmbedding(query),
      dimensions: 384,
      model: 'fallback',
      processingTime: 10,
      confidence: 0.6,
      metadata: {
        textLength: query.length,
        legalTerms: this.extractLegalTerms(query).length,
        complexity: this.calculateTextComplexity(query),
      },
    };
  }

  /**
   * Detect user intent from query
   */
  private async detectIntent(query: string): Promise<QueryIntent> {
    const scores = new Map<string, number>();

    // Pattern matching for intent detection
    for (const [intent, patterns] of this.intentPatterns) {
      let score = 0;
      for (const pattern of patterns) {
        if (pattern.test(query)) {
          score += 1;
        }
      }
      scores.set(intent, score);
    }

    // Find primary intent
    const sortedIntents = Array.from(scores.entries()).sort(([, a], [, b]) => b - a);

    const primaryIntent = sortedIntents[0]?.[0] || 'general';
    const confidence = Math.min(sortedIntents[0]?.[1] || 0, 1);

    return {
      primary: primaryIntent,
      secondary: sortedIntents.slice(1, 3).map(([intent]) => intent),
      confidence,
      category: primaryIntent as any,
      urgency: this.detectUrgency(query),
      scope: this.detectScope(query),
    };
  }

  /**
   * Enhance the original prompt with context and legal framework
   */
  private async enhancePrompt(query: string, context?: unknown): Promise<string> {
    const sections = [];

    // Add role-based context
    if (context?.userRole) {
      sections.push(`As a ${context.userRole}, `);
    }

    // Add case context
    if (context?.caseId) {
      sections.push(`regarding Case ${context.caseId}, `);
    }

    // Add legal framework
    sections.push('considering relevant legal principles and precedents, ');

    // Enhanced query
    sections.push(query);

    // Add specific instructions based on enhancement level
    switch (this.config.enhancementLevel) {
      case 'comprehensive':
        sections.push('\n\nPlease provide a comprehensive analysis including:');
        sections.push('1. Legal framework and applicable laws');
        sections.push('2. Relevant precedents and case citations');
        sections.push('3. Risk assessment and potential outcomes');
        sections.push('4. Practical recommendations and next steps');
        break;
      case 'advanced':
        sections.push('\n\nPlease include legal citations and precedents in your response.');
        break;
      case 'standard':
        sections.push('\n\nPlease provide a clear legal analysis.');
        break;
    }

    return sections.join('');
  }

  /**
   * Generate contextual follow-up prompts
   */
  private async generateContextualPrompts(query: string): Promise<ContextualPrompt[]> {
    const prompts: ContextualPrompt[] = [];

    // Generate clarification prompts
    if (query.length < 20) {
      prompts.push({
        type: 'clarification',
        prompt: "Could you provide more details about the specific legal issue you're addressing?",
        confidence: 0.8,
        category: 'clarification',
      });
    }

    // Generate related prompts based on legal terms
    const legalTerms = this.extractLegalTerms(query);
    if (legalTerms.length > 0) {
      prompts.push({
        type: 'related',
        prompt: `Would you like me to explain the legal implications of ${legalTerms[0]}?`,
        confidence: 0.7,
        category: 'explanation',
      });
    }

    // Generate follow-up prompts
    prompts.push({
      type: 'followup',
      prompt: 'Would you like me to research relevant case law for this matter?',
      confidence: 0.6,
      category: 'research',
    });

    return prompts;
  }

  /**
   * Build comprehensive legal context
   */
  private buildLegalContext(analysis: LegalAnalysisResult | null, query: string): LegalContext {
    if (!analysis) {
      return {
        entities: [],
        concepts: [],
        citations: [],
        keyTerms: this.extractLegalTerms(query),
        complexity: this.calculateTextComplexity(query),
        domain: 'general',
      };
    }

    return {
      entities: analysis.entities.map((entity) => ({
        text: entity.text,
        type: entity.type,
        confidence: entity.confidence,
        relevance: 0.8,
      })),
      concepts: analysis.concepts,
      citations: this.extractCitations(query),
      keyTerms: analysis.keyPhrases.map((phrase) => phrase.phrase),
      complexity: analysis.complexity.legalComplexity,
      domain: this.detectLegalDomain(query),
    };
  }

  /**
   * Utility methods
   */
  private extractLegalTerms(text: string): string[] {
    const legalTerms = [
      'contract',
      'liability',
      'negligence',
      'breach',
      'damages',
      'statute',
      'regulation',
      'precedent',
      'jurisdiction',
      'plaintiff',
      'defendant',
      'discovery',
      'motion',
      'brief',
      'appeal',
      'judgment',
      'settlement',
      'compliance',
      'violation',
      'evidence',
      'testimony',
      'cross-examination',
      'indictment',
      'arraignment',
      'plea',
      'sentence',
      'parole',
      'probation',
    ];

    return legalTerms.filter((term) => new RegExp(`\\b${term}\\b`, 'i').test(text));
  }

  private extractCitations(
    text: string
  ): Array<{ citation: string; type: 'case' | 'statute' | 'regulation'; jurisdiction?: string }> {
    const citations = [];

    // Case citations (simplified pattern)
    const casePattern = /\b\d+\s+\w+\.?\s+\d+/g;
    const caseMatches = text.match(casePattern) || [];

    for (const match of caseMatches) {
      citations.push({
        citation: match,
        type: 'case' as const,
      });
    }

    return citations;
  }

  private calculateTextComplexity(text: string): number {
    const words = text.split(/\s+/).length;
    const sentences = text.split(/[.!?]+/).length;
    const avgWordsPerSentence = words / sentences;

    return Math.min(avgWordsPerSentence / 20, 1);
  }

  private detectUrgency(query: string): 'low' | 'medium' | 'high' | 'critical' {
    const urgentWords = ['urgent', 'emergency', 'asap', 'immediately', 'deadline', 'crisis'];
    const highWords = ['important', 'priority', 'soon', 'quickly'];

    const text = query.toLowerCase();

    if (urgentWords.some((word) => text.includes(word))) {
      return 'critical';
    }
    if (highWords.some((word) => text.includes(word))) {
      return 'high';
    }
    if (query.includes('?')) {
      return 'medium';
    }
    return 'low';
  }

  private detectScope(
    query: string
  ): 'document' | 'case' | 'research' | 'procedural' | 'substantive' {
    const text = query.toLowerCase();

    if (text.includes('document') || text.includes('contract') || text.includes('agreement')) {
      return 'document';
    }
    if (text.includes('case') || text.includes('matter')) {
      return 'case';
    }
    if (text.includes('research') || text.includes('find') || text.includes('precedent')) {
      return 'research';
    }
    if (text.includes('procedure') || text.includes('process') || text.includes('filing')) {
      return 'procedural';
    }

    return 'substantive';
  }

  private detectLegalDomain(query: string): string {
    const domains = {
      criminal: ['criminal', 'prosecution', 'defense', 'evidence', 'trial'],
      civil: ['civil', 'litigation', 'plaintiff', 'defendant', 'damages'],
      contract: ['contract', 'agreement', 'breach', 'performance', 'terms'],
      corporate: ['corporate', 'business', 'company', 'entity', 'merger'],
      family: ['family', 'divorce', 'custody', 'support', 'adoption'],
      property: ['property', 'real estate', 'deed', 'title', 'ownership'],
    };

    const text = query.toLowerCase();

    for (const [domain, keywords] of Object.entries(domains)) {
      if (keywords.some((keyword) => text.includes(keyword))) {
        return domain;
      }
    }

    return 'general';
  }

  private calculateQuality(
    query: string,
    analysis: LegalAnalysisResult | null,
    intent: QueryIntent
  ): number {
    let quality = 0.5; // Base quality

    // Length factor
    if (query.length > 20 && query.length < 500) {
      quality += 0.2;
    }

    // Legal terms factor
    const legalTerms = this.extractLegalTerms(query);
    quality += Math.min(legalTerms.length * 0.1, 0.3);

    // Intent confidence factor
    quality += intent.confidence * 0.3;

    return Math.min(quality, 1);
  }

  private generateRecommendations(
    analysis: LegalAnalysisResult | null,
    intent: QueryIntent
  ): string[] {
    const recommendations = [];

    if (intent.category === 'research') {
      recommendations.push('Consider searching for recent precedents');
      recommendations.push('Check jurisdiction-specific laws');
    }

    if (intent.category === 'analysis') {
      recommendations.push('Review all relevant documents');
      recommendations.push('Consider multiple legal perspectives');
    }

    if (analysis?.complexity.legalComplexity && analysis.complexity.legalComplexity > 0.7) {
      recommendations.push(
        'This appears to be a complex legal matter - consider expert consultation'
      );
    }

    return recommendations;
  }

  private generateBasicEmbedding(text: string): number[] {
    // Simple hash-based embedding (fallback)
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Array(384).fill(0);

    for (let i = 0; i < words.length; i++) {
      const hash = this.simpleHash(words[i]);
      const index = Math.abs(hash) % 384;
      embedding[index] += 1;
    }

    // Normalize
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map((val) => (magnitude > 0 ? val / magnitude : 0));
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }

  private generateCacheKey(query: string, context?: unknown): string {
    return `${query}:${JSON.stringify(context || {})}`;
  }

  private createFallbackSynthesis(query: string, context?: unknown): SynthesizedInput {
    return {
      originalQuery: query,
      enhancedPrompt: query,
      legalContext: {
        entities: [],
        concepts: [],
        citations: [],
        keyTerms: this.extractLegalTerms(query),
        complexity: 0.5,
        domain: 'general',
      },
      intent: {
        primary: 'general',
        secondary: [],
        confidence: 0.3,
        category: 'general',
        urgency: 'medium',
        scope: 'substantive',
      },
      embedding: [],
      metadata: {
        timestamp: new Date().toISOString(),
        quality: 0.3,
        processingTime: 0,
      },
      recommendations: ['Consider providing more specific details'],
      contextualPrompts: [],
    };
  }

  /**
   * Batch synthesis for multiple queries
   */
  async synthesizeBatch(
    queries: Array<{ query: string; context?: unknown }>
  ): Promise<SynthesizedInput[]> {
    return Promise.all(queries.map(({ query, context }) => this.synthesizeInput(query, context)));
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; hitRate: number } {
    return {
      size: this.cache.size,
      hitRate: 0.85, // Placeholder - would need actual tracking
    };
  }
}

// Export singleton instance
export const aiAssistantInputSynthesizer = new AIAssistantInputSynthesizer();

// Export types are already exported above as interfaces
