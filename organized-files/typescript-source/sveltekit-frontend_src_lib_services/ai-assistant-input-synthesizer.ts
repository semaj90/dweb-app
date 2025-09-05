/**
 * AI Assistant Input Synthesizer
 * Synthesizes and enhances user inputs with LegalBERT middleware, context enrichment,
 * and intelligent prompt engineering for optimal AI assistant responses.
 *
 * NOTE: Cleaned up from previous merge-conflicted / duplicated stub + implementation.
 */

export interface LegalEntity {
  text: string;
  type: string;
  confidence: number;
  startIndex: number;
  endIndex: number;
}

export interface LegalConcept {
  concept: string;
  relevance: number;
  category: string;
}

export interface SentimentAnalysis {
  polarity: number;
  confidence: number;
  classification: string;
}

export interface ComplexityAnalysis {
  readabilityScore: number;
  legalComplexity: number;
  technicalTerms: number;
}

export interface KeyPhrase {
  phrase: string;
  importance: number;
  category: string;
}

export interface SummaryInfo {
  abstractive: string;
  extractive: string[];
  keyPoints: string[];
}

export interface LegalAnalysisResult {
  entities: LegalEntity[];
  concepts: LegalConcept[];
  sentiment: SentimentAnalysis;
  complexity: ComplexityAnalysis;
  keyPhrases: KeyPhrase[];
  summary: SummaryInfo;
  [key: string]: any; // allow extra fields from external analyzers
}

export interface LegalEmbeddingResult {
  embedding: number[];
  dimensions: number;
  model: string;
  processingTime: number;
  confidence: number;
  metadata: {
    textLength: number;
    legalTerms: number;
    complexity: number;
    [key: string]: any;
  };
  [key: string]: any;
}

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

export interface SynthesizerContext {
  userRole?: string;
  caseId?: string;
  documentIds?: string[];
  sessionContext?: unknown;
}

export class AIAssistantInputSynthesizer {
  private config: SynthesizerConfig;
  private legalBERT: any = null;
  private cache = new Map<string, SynthesizedInput>();
  private intentPatterns!: Map<string, RegExp[]>;
  private cacheHits = 0;
  private cacheMisses = 0;

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
    void this.initializeLegalBERT();
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
          /formulat[e|ing]/i,
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
    } catch (error: any) {
      console.warn('LegalBERT middleware not available, using fallback analysis', error);
    }
  }

  async synthesizeInput(query: string, context?: SynthesizerContext): Promise<SynthesizedInput> {
    const startTime = Date.now();
    const cacheKey = this.generateCacheKey(query, context);

    if (this.config.cacheResults && this.cache.has(cacheKey)) {
      this.cacheHits++;
      return this.cache.get(cacheKey)!;
    }
    this.cacheMisses++;

    try {
      const [legalAnalysis, embeddingResult, intent, enhancedPrompt, contextualPrompts] =
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
        embedding: (embeddingResult as any)?.embedding || [],
        metadata: {
          userRole: context?.userRole as any,
          caseId: context?.caseId,
          documentIds: context?.documentIds,
          sessionContext: context?.sessionContext,
            // jurisdiction could be inferred in future
          timestamp: new Date().toISOString(),
          quality: this.calculateQuality(query, legalAnalysis, intent),
          processingTime: Date.now() - startTime,
        },
        recommendations: this.generateRecommendations(legalAnalysis, intent),
        contextualPrompts,
      };

      if (this.config.cacheResults) {
        this.cache.set(cacheKey, synthesized);
      }

      return synthesized;
    } catch (error: any) {
      console.error('Input synthesis failed:', error);
      return this.createFallbackSynthesis(query, context);
    }
  }

  async synthesizeBatch(
    queries: Array<{ query: string; context?: SynthesizerContext }>
  ): Promise<SynthesizedInput[]> {
    return Promise.all(queries.map(({ query, context }) => this.synthesizeInput(query, context)));
  }

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
    } catch (error: any) {
      console.warn('LegalBERT analysis failed, using fallback:', error);
      return this.performBasicLegalAnalysis(query);
    }
  }

  private performBasicLegalAnalysis(query: string): LegalAnalysisResult {
    const legalTerms = this.extractLegalTerms(query);
    const complexity = this.calculateTextComplexity(query);

    return {
      entities: legalTerms.map((term) => ({
        text: term,
        type: 'LEGAL_CONCEPT',
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
        classification: 'neutral',
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
        abstractive: query.substring(0, 100) + (query.length > 100 ? '...' : ''),
        extractive: [query.split('.')[0] || query],
        keyPoints: legalTerms.slice(0, 3),
      },
    };
  }

  private async generateEmbedding(query: string): Promise<LegalEmbeddingResult> {
    if (this.legalBERT) {
      try {
        return await this.legalBERT.generateEmbedding(query);
      } catch (error: any) {
        console.warn('LegalBERT embedding failed, using fallback:', error);
      }
    }
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

  private async detectIntent(query: string): Promise<QueryIntent> {
    const scores = new Map<string, number>();

    for (const [intent, patterns] of this.intentPatterns) {
      let score = 0;
      for (const pattern of patterns) {
        if (pattern.test(query)) score += 1;
      }
      scores.set(intent, score);
    }

    const sortedIntents = Array.from(scores.entries()).sort(([, a], [, b]) => b - a);
    const primaryIntent = sortedIntents[0]?.[0] || 'general';
    const confidence = Math.min(sortedIntents[0]?.[1] || 0, 1);

    return {
      primary: primaryIntent,
      secondary: sortedIntents.slice(1, 3).map(([i]) => i),
      confidence,
      category: (['analysis', 'research', 'drafting', 'review', 'advice'].includes(primaryIntent)
        ? primaryIntent
        : 'general') as QueryIntent['category'],
      urgency: this.detectUrgency(query),
      scope: this.detectScope(query),
    };
  }

  private async enhancePrompt(query: string, context?: SynthesizerContext): Promise<string> {
    const sections: string[] = [];

    if (context?.userRole) {
      sections.push(`As a ${context.userRole}, `);
    }
    if (context?.caseId) {
      sections.push(`regarding Case ${context.caseId}, `);
    }

    sections.push('considering relevant legal principles and precedents, ');
    sections.push(query);

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
      case 'basic':
      default:
        break;
    }

    const result = sections.join('');
    return result.length > this.config.maxPromptLength
      ? result.slice(0, this.config.maxPromptLength)
      : result;
  }

  private async generateContextualPrompts(query: string): Promise<ContextualPrompt[]> {
    const prompts: ContextualPrompt[] = [];

    if (query.length < 20) {
      prompts.push({
        type: 'clarification',
        prompt: "Could you provide more details about the specific legal issue you're addressing?",
        confidence: 0.8,
        category: 'clarification',
      });
    }

    const legalTerms = this.extractLegalTerms(query);
    if (legalTerms.length > 0) {
      prompts.push({
        type: 'related',
        prompt: `Would you like me to explain the legal implications of ${legalTerms[0]}?`,
        confidence: 0.7,
        category: 'explanation',
      });
    }

    prompts.push({
      type: 'followup',
      prompt: 'Would you like me to research relevant case law for this matter?',
      confidence: 0.6,
      category: 'research',
    });

    return prompts;
  }

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
      entities: (analysis.entities || []).map((entity: any) => ({
        text: entity.text,
        type: entity.type,
        confidence: entity.confidence,
        relevance: 0.8,
      })),
      concepts: analysis.concepts || [],
      citations: this.extractCitations(query),
      keyTerms: (analysis.keyPhrases || []).map((p: any) => p.phrase),
      complexity: analysis.complexity?.legalComplexity ?? 0.5,
      domain: this.detectLegalDomain(query),
    };
  }

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
      'deed',
      'title',
      'agreement',
    ];
    return legalTerms.filter((term) => new RegExp(`\\b${term}\\b`, 'i').test(text));
  }

  private extractCitations(
    text: string
  ): Array<{ citation: string; type: 'case' | 'statute' | 'regulation'; jurisdiction?: string }> {
    const citations: Array<{ citation: string; type: 'case' | 'statute' | 'regulation'; jurisdiction?: string }> = [];
    const casePattern = /\b\d+\s+\w+\.?\s+\d+/g;
    const caseMatches = text.match(casePattern) || [];
    for (const match of caseMatches) {
      citations.push({ citation: match, type: 'case' });
    }
    return citations;
  }

  private calculateTextComplexity(text: string): number {
    const words = text.trim().split(/\s+/).filter(Boolean).length || 1;
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0).length || 1;
    const avgWordsPerSentence = words / sentences;
    return Math.min(avgWordsPerSentence / 20, 1);
  }

  private detectUrgency(query: string): 'low' | 'medium' | 'high' | 'critical' {
    const urgentWords = ['urgent', 'emergency', 'asap', 'immediately', 'deadline', 'crisis'];
    const highWords = ['important', 'priority', 'soon', 'quickly'];
    const text = query.toLowerCase();
    if (urgentWords.some((w) => text.includes(w))) return 'critical';
    if (highWords.some((w) => text.includes(w))) return 'high';
    if (query.includes('?')) return 'medium';
    return 'low';
  }

  private detectScope(query: string): 'document' | 'case' | 'research' | 'procedural' | 'substantive' {
    const text = query.toLowerCase();
    if (/(document|contract|agreement)/.test(text)) return 'document';
    if (/(case|matter)/.test(text)) return 'case';
    if (/(research|find|precedent)/.test(text)) return 'research';
    if (/(procedure|process|filing)/.test(text)) return 'procedural';
    return 'substantive';
  }

  private detectLegalDomain(query: string): string {
    const domains: Record<string, string[]> = {
      criminal: ['criminal', 'prosecution', 'defense', 'evidence', 'trial'],
      civil: ['civil', 'litigation', 'plaintiff', 'defendant', 'damages'],
      contract: ['contract', 'agreement', 'breach', 'performance', 'terms'],
      corporate: ['corporate', 'business', 'company', 'entity', 'merger'],
      family: ['family', 'divorce', 'custody', 'support', 'adoption'],
      property: ['property', 'real estate', 'deed', 'title', 'ownership'],
    };
    const text = query.toLowerCase();
    for (const [domain, keywords] of Object.entries(domains)) {
      if (keywords.some((k) => text.includes(k))) return domain;
    }
    return 'general';
  }

  private calculateQuality(
    query: string,
    analysis: LegalAnalysisResult | null,
    intent: QueryIntent
  ): number {
    let quality = 0.5;
    if (query.length > 20 && query.length < 500) quality += 0.2;
    const legalTerms = this.extractLegalTerms(query);
    quality += Math.min(legalTerms.length * 0.1, 0.3);
    quality += intent.confidence * 0.3;
    if (analysis?.complexity?.legalComplexity) {
      quality += Math.min(analysis.complexity.legalComplexity * 0.1, 0.1);
    }
    return Math.min(quality, 1);
  }

  private generateRecommendations(
    analysis: LegalAnalysisResult | null,
    intent: QueryIntent
  ): string[] {
    const rec: string[] = [];
    if (intent.category === 'research') {
      rec.push('Consider searching for recent precedents', 'Check jurisdiction-specific laws');
    }
    if (intent.category === 'analysis') {
      rec.push('Review all relevant documents', 'Consider multiple legal perspectives');
    }
    if (analysis?.complexity?.legalComplexity && analysis.complexity.legalComplexity > 0.7) {
      rec.push('This appears complex - consider expert consultation');
    }
    return rec;
  }

  private generateBasicEmbedding(text: string): number[] {
    const words = text.toLowerCase().split(/\s+/).filter(Boolean);
    const embedding = new Array(384).fill(0);
    for (const w of words) {
      const hash = this.simpleHash(w);
      const index = Math.abs(hash) % 384;
      embedding[index] += 1;
    }
    const magnitude = Math.sqrt(embedding.reduce((s, v) => s + v * v, 0));
    return embedding.map((v) => (magnitude ? v / magnitude : 0));
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash |= 0;
    }
    return hash;
  }

  private generateCacheKey(query: string, context?: SynthesizerContext): string {
    try {
      return `${query}:${JSON.stringify(context || {})}`;
    } catch {
      return `${query}:nocontent`;
    }
  }

  private createFallbackSynthesis(query: string, context?: SynthesizerContext): SynthesizedInput {
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

  clearCache(): void {
    this.cache.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
  }

  getCacheStats(): { size: number; hitRate: number } {
    const total = this.cacheHits + this.cacheMisses;
    return {
      size: this.cache.size,
      hitRate: total === 0 ? 0 : this.cacheHits / total,
    };
  }
}

export const aiAssistantInputSynthesizer = new AIAssistantInputSynthesizer();
export const aiAssistantSynthesizer = aiAssistantInputSynthesizer;

// Removed duplicate interfaces, class, and exports that caused redeclaration and syntax errors.

// Export types are already exported above as interfaces
