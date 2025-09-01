import { logger } from './logger.js';
import crypto from "crypto";

// lib/server/ai/legalbert-middleware.ts
// LegalBERT middleware for specialized legal embeddings and analysis

// Type interfaces will be defined below
import { generateEmbedding } from './embeddings-simple.js';

// Type definitions will be defined later in file

export interface LegalEmbeddingResult {
  embedding: number[];
  model: string;
  dimensions: number;
  processingTime: number;
}

// Metrics stub (replace with proper metrics service later)
const metrics = {
  increment: (name: string, value: number = 1) => console.log(`[METRIC] ${name}: +${value}`),
  gauge: (name: string, value: number) => console.log(`[METRIC] ${name}: ${value}`),
  histogram: (name: string, value: number) => console.log(`[METRIC] ${name}: ${value}ms`),
};

// Utility functions
async function withRetry<T>(fn: () => Promise<T>, retries: number = 3): Promise<T> {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
    }
  }
  throw new Error('Max retries exceeded');
}

// LegalBERT model configurations
const LEGALBERT_MODELS = {
  // Local Ollama models
  local: {
    embedding: 'nomic-embed-text:latest',
    analysis: 'gemma3-legal:latest',
    baseUrl: import.meta.env.OLLAMA_URL || 'http://localhost:11434',
  },
  // Remote API endpoints
  huggingface: {
    embedding: 'nlpaueb/legal-bert-base-uncased',
    analysis: 'nlpaueb/legal-bert-small-uncased',
    apiKey: import.meta.env.HUGGINGFACE_API_KEY,
    baseUrl: 'https://api-inference.huggingface.co/models',
  },
  openai: {
    embedding: 'text-embedding-3-small',
    analysis: 'gpt-4',
    apiKey: import.meta.env.OPENAI_API_KEY,
    baseUrl: 'https://api.openai.com/v1',
  },
};

// Legal domain-specific entity types
const LEGAL_ENTITY_TYPES = {
  CASE_CITATION: 'case_citation',
  STATUTE: 'statute_reference',
  REGULATION: 'regulation',
  COURT: 'court',
  JUDGE: 'judge',
  PARTY: 'party',
  LEGAL_CONCEPT: 'legal_concept',
  MONETARY_AMOUNT: 'monetary_amount',
  DATE: 'legal_date',
  JURISDICTION: 'jurisdiction',
  CONTRACT_TERM: 'contract_term',
  LEGAL_PRINCIPLE: 'legal_principle',
} as const;

// Legal text analysis results
export interface LegalBertAnalysisResult {
  entities: Array<{
    text: string;
    type: keyof typeof LEGAL_ENTITY_TYPES;
    confidence: number;
    startIndex: number;
    endIndex: number;
    context?: string;
  }>;
  concepts: Array<{
    concept: string;
    relevance: number;
    category: string;
  }>;
  sentiment: {
    polarity: number; // -1 to 1
    confidence: number;
    classification: 'positive' | 'neutral' | 'negative';
  };
  complexity: {
    readabilityScore: number;
    legalComplexity: number;
    technicalTerms: number;
  };
  keyPhrases: Array<{
    phrase: string;
    importance: number;
    category: string;
  }>;
  summary: {
    abstractive: string;
    extractive: string[];
    keyPoints: string[];
  };
}

// LegalBERT embedding result
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
  };
}

// Legal document classification
export interface LegalClassificationResult {
  documentType: string;
  confidence: number;
  subCategories: Array<{
    category: string;
    confidence: number;
  }>;
  jurisdiction: string;
  practiceArea: string;
  urgency: 'low' | 'medium' | 'high';
  recommendations: string[];
}

// LegalBERT middleware class
export class LegalBERTMiddleware {
  private modelConfig: any;
  private cache = new Map<string, any>();
  private requestCount = 0;

  constructor(preferredModel: 'local' | 'huggingface' | 'openai' = 'local') {
    this.modelConfig = LEGALBERT_MODELS[preferredModel];
    this.initializeModel();
  }

  private async initializeModel(): Promise<void> {
    try {
      logger.info('[LegalBERT] Initializing middleware...');

      // Test model connectivity
      const testResult = await this.generateLegalEmbedding('test legal document');
      if (testResult.embedding.length > 0) {
        logger.info('[LegalBERT] Model initialized successfully');
        metrics.increment('legalbert_initializations');
      } else {
        throw new Error('Model test failed');
      }
    } catch (error: any) {
      logger.error('[LegalBERT] Initialization failed:', error);
      // Fallback to basic embedding model
      this.modelConfig = LEGALBERT_MODELS.local;
    }
  }

  /**
   * Generate legal-domain specialized embeddings
   */
  async generateLegalEmbedding(text: string): Promise<LegalEmbeddingResult> {
    const startTime = Date.now();
    const textHash = this.hashText(text);

    // Check cache first
    const cached = this.cache.get(`embedding_${textHash}`);
    if (cached) {
      metrics.increment('legalbert_cache_hits');
      return cached;
    }

    try {
      // Preprocess legal text
      const preprocessedText = this.preprocessLegalText(text);

      // Generate embedding using the configured model
      let embedding: number[];

      if (this.modelConfig === LEGALBERT_MODELS.local) {
        embedding = await this.generateLocalEmbedding(preprocessedText);
      } else if (this.modelConfig === LEGALBERT_MODELS.huggingface) {
        embedding = await this.generateHuggingFaceEmbedding(preprocessedText);
      } else {
        embedding = await this.generateOpenAIEmbedding(preprocessedText);
      }

      // Analyze legal content for metadata
      const legalTerms = this.countLegalTerms(text);
      const complexity = this.calculateLegalComplexity(text);

      const result: LegalEmbeddingResult = {
        embedding,
        dimensions: embedding.length,
        model: this.modelConfig.embedding,
        processingTime: Date.now() - startTime,
        confidence: this.calculateEmbeddingConfidence(text, embedding),
        metadata: {
          textLength: text.length,
          legalTerms,
          complexity,
        },
      };

      // Cache result
      this.cache.set(`embedding_${textHash}`, result);
      metrics.increment('legalbert_embeddings_generated');
      metrics.histogram('legalbert_embedding_time', result.processingTime);

      return result;
    } catch (error: any) {
      logger.error('[LegalBERT] Embedding generation failed:', error);
      metrics.increment('legalbert_embedding_errors');

      // Fallback to basic embedding
      const embedding = (await generateEmbedding(text)) || [];
      return {
        embedding,
        dimensions: embedding.length,
        model: 'fallback',
        processingTime: Date.now() - startTime,
        confidence: 0.7,
        metadata: {
          textLength: text.length,
          legalTerms: 0,
          complexity: 0.5,
        },
      };
    }
  }

  /**
   * Comprehensive legal text analysis
   */
  async analyzeLegalText(text: string): Promise<LegalBertAnalysisResult> {
    const startTime = Date.now();
    const textHash = this.hashText(text);

    // Check cache
    const cached = this.cache.get(`analysis_${textHash}`);
    if (cached) {
      return cached;
    }

    try {
      // Run all analysis components in parallel
      const [entities, concepts, sentiment, complexity, keyPhrases, summary] = await Promise.all([
        this.extractLegalEntities(text),
        this.extractLegalConcepts(text),
        this.analyzeLegalSentiment(text),
        this.calculateTextComplexity(text),
        this.extractKeyPhrases(text),
        this.generateLegalSummary(text),
      ]);

      const result = {
        entities,
        concepts,
        sentiment,
        complexity,
        keyPhrases,
        summary,
      };

      // Cache result
      this.cache.set(`analysis_${textHash}`, result);
      metrics.histogram('legalbert_analysis_time', Date.now() - startTime);

      return result;
    } catch (error: any) {
      logger.error('[LegalBERT] Text analysis failed:', error);
      return this.generateFallbackAnalysis(text);
    }
  }

  /**
   * Classify legal documents
   */
  async classifyLegalDocument(text: string): Promise<LegalClassificationResult> {
    try {
      const classification = await this.performDocumentClassification(text);
      metrics.increment('legalbert_classifications');
      return classification;
    } catch (error: any) {
      logger.error('[LegalBERT] Document classification failed:', error);
      return this.generateFallbackClassification(text);
    }
  }

  /**
   * Legal semantic similarity comparison
   */
  async calculateLegalSimilarity(
    text1: string,
    text2: string
  ): Promise<{
    similarity: number;
    confidence: number;
    factors: {
      semantic: number;
      structural: number;
      legal_concepts: number;
    };
  }> {
    try {
      const [emb1, emb2] = await Promise.all([
        this.generateLegalEmbedding(text1),
        this.generateLegalEmbedding(text2),
      ]);

      const semantic = this.cosineSimilarity(emb1.embedding, emb2.embedding);
      const structural = this.calculateStructuralSimilarity(text1, text2);
      const legal_concepts = this.calculateConceptSimilarity(text1, text2);

      const similarity = semantic * 0.5 + structural * 0.2 + legal_concepts * 0.3;

      return {
        similarity,
        confidence: Math.min(emb1.confidence, emb2.confidence),
        factors: {
          semantic,
          structural,
          legal_concepts,
        },
      };
    } catch (error: any) {
      logger.error('[LegalBERT] Similarity calculation failed:', error);
      return {
        similarity: 0.5,
        confidence: 0.3,
        factors: { semantic: 0.5, structural: 0.5, legal_concepts: 0.5 },
      };
    }
  }

  // === PRIVATE HELPER METHODS ===

  private async generateLocalEmbedding(text: string): Promise<number[]> {
    return await withRetry(async () => {
      const embedding = await generateEmbedding(text, { model: 'local' });
      if (!embedding || embedding.length === 0) {
        throw new Error('Local embedding generation failed');
      }
      return embedding;
    });
  }

  private async generateHuggingFaceEmbedding(text: string): Promise<number[]> {
    return await withRetry(async () => {
      const response = await fetch(`${this.modelConfig.baseUrl}/${this.modelConfig.embedding}`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.modelConfig.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: text,
          options: { wait_for_model: true },
        }),
      });

      if (!response.ok) {
        throw new Error(`HuggingFace API error: ${response.statusText}`);
      }

      const embedding = await response.json();
      return Array.isArray(embedding) ? embedding : embedding.embeddings || [];
    });
  }

  private async generateOpenAIEmbedding(text: string): Promise<number[]> {
    return await withRetry(async () => {
      const response = await fetch(`${this.modelConfig.baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.modelConfig.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.modelConfig.embedding,
          input: text,
        }),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.statusText}`);
      }

      const result = await response.json();
      return result.data[0]?.embedding || [];
    });
  }

  private preprocessLegalText(text: string): string {
    // Remove excessive whitespace
    let processed = text.replace(/\s+/g, ' ').trim();

    // Normalize legal citations
    processed = processed.replace(/(\d+)\s+U\.?S\.?C\.?\s+§?\s*(\d+)/g, '$1 U.S.C. § $2');

    // Normalize case citations
    processed = processed.replace(/(\d+)\s+([A-Z][a-z]+)\s+(\d+)/g, '$1 $2 $3');

    // Preserve legal abbreviations
    const legalAbbrevs = ['Inc.', 'Corp.', 'LLC', 'L.P.', 'Co.', 'Ltd.', 'P.C.'];
    legalAbbrevs.forEach((abbrev) => {
      const regex = new RegExp(abbrev.replace('.', '\\.'), 'g');
      processed = processed.replace(regex, abbrev);
    });

    return processed;
  }

  private async extractLegalEntities(text: string): Promise<LegalBertAnalysisResult['entities']> {
    const entities: LegalBertAnalysisResult['entities'] = [];

    // Case citations: 123 F.3d 456, 789 U.S. 123
    const casePattern = /\b\d{1,4}\s+[A-Z][a-z]*\.?\s*\d*[a-z]*\s+\d{1,4}\b/g;
    let match;
    while ((match = casePattern.exec(text)) !== null) {
      entities.push({
        text: match[0],
        type: 'CASE_CITATION',
        confidence: 0.9,
        startIndex: match.index,
        endIndex: match.index + match[0].length,
        context: this.getContext(text, match.index, 50),
      });
    }

    // Statutes: 42 U.S.C. § 1983
    const statutePattern = /\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+[a-z]*/g;
    while ((match = statutePattern.exec(text)) !== null) {
      entities.push({
        text: match[0],
        type: 'STATUTE',
        confidence: 0.95,
        startIndex: match.index,
        endIndex: match.index + match[0].length,
        context: this.getContext(text, match.index, 50),
      });
    }

    // Monetary amounts: $1,000, $50,000.00
    const moneyPattern = /\$[\d,]+(?:\.\d{2})?/g;
    while ((match = moneyPattern.exec(text)) !== null) {
      entities.push({
        text: match[0],
        type: 'MONETARY_AMOUNT',
        confidence: 0.85,
        startIndex: match.index,
        endIndex: match.index + match[0].length,
      });
    }

    // Courts
    const courtPattern =
      /(?:Supreme Court|Court of Appeals|District Court|Circuit Court|Bankruptcy Court|Tax Court)/gi;
    while ((match = courtPattern.exec(text)) !== null) {
      entities.push({
        text: match[0],
        type: 'COURT',
        confidence: 0.8,
        startIndex: match.index,
        endIndex: match.index + match[0].length,
      });
    }

    return entities;
  }

  private async extractLegalConcepts(text: string): Promise<LegalBertAnalysisResult['concepts']> {
    const concepts: LegalBertAnalysisResult['concepts'] = [];

    const legalConcepts = {
      contract: ['contract', 'agreement', 'terms', 'conditions', 'breach', 'performance'],
      tort: ['negligence', 'liability', 'damages', 'duty', 'standard of care'],
      criminal: ['criminal', 'felony', 'misdemeanor', 'sentence', 'conviction'],
      constitutional: ['constitution', 'amendment', 'rights', 'due process', 'equal protection'],
      corporate: ['corporation', 'shareholder', 'director', 'fiduciary', 'merger'],
      property: ['property', 'real estate', 'ownership', 'title', 'deed'],
      employment: ['employment', 'discrimination', 'wrongful termination', 'wages', 'benefits'],
    };

    const textLower = text.toLowerCase();

    for (const [category, terms] of Object.entries(legalConcepts)) {
      let relevance = 0;
      for (const term of terms) {
        const regex = new RegExp(`\\b${term}\\b`, 'gi');
        const matches = textLower.match(regex);
        if (matches) {
          relevance += matches.length * 0.1;
        }
      }

      if (relevance > 0.1) {
        concepts.push({
          concept: category,
          relevance: Math.min(relevance, 1.0),
          category: 'legal_domain',
        });
      }
    }

    return concepts.sort((a, b) => b.relevance - a.relevance);
  }

  private async analyzeLegalSentiment(text: string): Promise<LegalBertAnalysisResult['sentiment']> {
    // Simple rule-based sentiment for legal text
    const positiveWords = ['granted', 'approved', 'affirmed', 'successful', 'favorable', 'upheld'];
    const negativeWords = [
      'denied',
      'rejected',
      'reversed',
      'failed',
      'breach',
      'violation',
      'guilty',
    ];

    const textLower = text.toLowerCase();
    let positiveScore = 0;
    let negativeScore = 0;

    positiveWords.forEach((word) => {
      const matches = textLower.match(new RegExp(`\\b${word}\\b`, 'g'));
      if (matches) positiveScore += matches.length;
    });

    negativeWords.forEach((word) => {
      const matches = textLower.match(new RegExp(`\\b${word}\\b`, 'g'));
      if (matches) negativeScore += matches.length;
    });

    const totalScore = positiveScore + negativeScore;
    const polarity = totalScore > 0 ? (positiveScore - negativeScore) / totalScore : 0;

    return {
      polarity,
      confidence: Math.min(totalScore / 10, 1.0),
      classification: polarity > 0.1 ? 'positive' : polarity < -0.1 ? 'negative' : 'neutral',
    };
  }

  private calculateTextComplexity(text: string): LegalBertAnalysisResult['complexity'] {
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);
    const words = text.split(/\s+/).filter((w) => w.length > 0);
    const avgWordsPerSentence = words.length / sentences.length;

    // Count complex legal terms
    const complexTerms = [
      'notwithstanding',
      'heretofore',
      'whereas',
      'pursuant',
      'aforementioned',
      'jurisdiction',
      'precedent',
      'jurisprudence',
      'adjudicate',
      'indemnify',
    ];

    let technicalTerms = 0;
    const textLower = text.toLowerCase();
    complexTerms.forEach((term) => {
      const matches = textLower.match(new RegExp(`\\b${term}\\b`, 'g'));
      if (matches) technicalTerms += matches.length;
    });

    const readabilityScore = Math.max(
      0,
      Math.min(100, 206.835 - 1.015 * avgWordsPerSentence - 84.6 * (technicalTerms / words.length))
    );
    const legalComplexity = Math.min(1, (technicalTerms / words.length) * 100);

    return {
      readabilityScore,
      legalComplexity,
      technicalTerms,
    };
  }

  private async extractKeyPhrases(text: string): Promise<LegalBertAnalysisResult['keyPhrases']> {
    const phrases: LegalBertAnalysisResult['keyPhrases'] = [];

    // Extract noun phrases and legal terms
    const legalPhrases = [
      /\b(?:due process|equal protection|reasonable doubt|burden of proof|statute of limitations)\b/gi,
      /\b(?:breach of contract|material breach|anticipatory breach|fundamental breach)\b/gi,
      /\b(?:punitive damages|compensatory damages|liquidated damages|consequential damages)\b/gi,
      /\b(?:motion for summary judgment|motion to dismiss|motion in limine)\b/gi,
    ];

    legalPhrases.forEach((pattern, index) => {
      let match;
      while ((match = pattern.exec(text)) !== null) {
        phrases.push({
          phrase: match[0],
          importance: 0.8 + index * 0.05,
          category: 'legal_term',
        });
      }
    });

    return phrases.sort((a, b) => b.importance - a.importance).slice(0, 10);
  }

  private async generateLegalSummary(text: string): Promise<LegalBertAnalysisResult['summary']> {
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);

    // Simple extractive summary - take first and most important sentences
    const extractive = sentences.slice(0, 3).map((s) => s.trim());

    // Key points extraction
    const keyPoints = [];
    if (text.includes('HOLDING') || text.includes('holding')) {
      keyPoints.push('Case contains holding or legal ruling');
    }
    if (text.includes('contract') || text.includes('agreement')) {
      keyPoints.push('Document relates to contractual matters');
    }
    if (text.includes('damages') || text.includes('liability')) {
      keyPoints.push('Involves potential damages or liability');
    }

    return {
      abstractive: `Legal document summary: ${extractive[0]?.substring(0, 200) || 'No summary available'}...`,
      extractive,
      keyPoints,
    };
  }

  private async performDocumentClassification(text: string): Promise<LegalClassificationResult> {
    const textLower = text.toLowerCase();

    // Rule-based classification
    let documentType = 'general';
    let confidence = 0.5;

    if (textLower.includes('contract') || textLower.includes('agreement')) {
      documentType = 'contract';
      confidence = 0.8;
    } else if (textLower.includes('motion') || textLower.includes('petition')) {
      documentType = 'motion';
      confidence = 0.9;
    } else if (textLower.includes('opinion') || textLower.includes('judgment')) {
      documentType = 'court_opinion';
      confidence = 0.85;
    } else if (textLower.includes('statute') || textLower.includes('code')) {
      documentType = 'statute';
      confidence = 0.9;
    }

    return {
      documentType,
      confidence,
      subCategories: [
        { category: 'primary', confidence: confidence },
        { category: 'secondary', confidence: confidence * 0.7 },
      ],
      jurisdiction: this.extractJurisdiction(text),
      practiceArea: this.extractPracticeArea(text),
      urgency: this.assessUrgency(text),
      recommendations: ['Review for accuracy', 'Check citations', 'Verify jurisdiction'],
    };
  }

  // === UTILITY METHODS ===

  private hashText(text: string): string {
    return crypto.createHash('sha256').update(text.trim()).digest('hex');
  }

  private getContext(text: string, index: number, length: number): string {
    const start = Math.max(0, index - length);
    const end = Math.min(text.length, index + length);
    return text.substring(start, end);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private calculateStructuralSimilarity(text1: string, text2: string): number {
    const sentences1 = text1.split(/[.!?]+/).length;
    const sentences2 = text2.split(/[.!?]+/).length;
    const words1 = text1.split(/\s+/).length;
    const words2 = text2.split(/\s+/).length;

    const sentenceRatio = Math.min(sentences1, sentences2) / Math.max(sentences1, sentences2);
    const wordRatio = Math.min(words1, words2) / Math.max(words1, words2);

    return (sentenceRatio + wordRatio) / 2;
  }

  private calculateConceptSimilarity(text1: string, text2: string): number {
    const concepts1 = this.extractBasicConcepts(text1);
    const concepts2 = this.extractBasicConcepts(text2);

    const intersection = concepts1.filter((c) => concepts2.includes(c));
    const union = [...new Set([...concepts1, ...concepts2])];

    return union.length > 0 ? intersection.length / union.length : 0;
  }

  private extractBasicConcepts(text: string): string[] {
    const legalTerms = [
      'contract',
      'tort',
      'criminal',
      'civil',
      'damages',
      'liability',
      'negligence',
      'breach',
      'jurisdiction',
      'court',
      'statute',
      'regulation',
    ];

    const textLower = text.toLowerCase();
    return legalTerms.filter((term) => textLower.includes(term));
  }

  private countLegalTerms(text: string): number {
    const legalTerms = [
      'plaintiff',
      'defendant',
      'appellant',
      'appellee',
      'petitioner',
      'respondent',
      'jurisdiction',
      'venue',
      'standing',
      'mootness',
      'ripeness',
      'justiciability',
      'negligence',
      'liability',
      'damages',
      'injunction',
      'mandamus',
      'certiorari',
    ];

    const textLower = text.toLowerCase();
    return legalTerms.reduce((count, term) => {
      const matches = textLower.match(new RegExp(`\\b${term}\\b`, 'g'));
      return count + (matches ? matches.length : 0);
    }, 0);
  }

  private calculateLegalComplexity(text: string): number {
    const indicators = [
      /\b(?:whereas|heretofore|aforementioned|notwithstanding|pursuant)\b/gi,
      /\b(?:inter alia|prima facie|res judicata|stare decisis|habeas corpus)\b/gi,
      /\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+/g,
      /\b\d+\s+[A-Z][a-z]+\s+\d+\b/g,
    ];

    let complexity = 0;
    indicators.forEach((pattern) => {
      const matches = text.match(pattern);
      if (matches) complexity += matches.length * 0.1;
    });

    return Math.min(complexity, 1.0);
  }

  private calculateEmbeddingConfidence(text: string, embedding: number[]): number {
    // Base confidence on text quality and embedding properties
    const textQuality = text.length > 50 && text.length < 10000 ? 0.8 : 0.6;
    const embeddingQuality = embedding.length > 0 ? 0.9 : 0.3;
    const legalContent = this.countLegalTerms(text) > 0 ? 0.9 : 0.7;

    return (textQuality + embeddingQuality + legalContent) / 3;
  }

  private extractJurisdiction(text: string): string {
    const jurisdictions = ['federal', 'state', 'local', 'international'];
    const textLower = text.toLowerCase();

    for (const jurisdiction of jurisdictions) {
      if (textLower.includes(jurisdiction)) {
        return jurisdiction;
      }
    }

    return 'unknown';
  }

  private extractPracticeArea(text: string): string {
    const practiceAreas = {
      contract: ['contract', 'agreement', 'breach'],
      tort: ['negligence', 'liability', 'damages'],
      criminal: ['criminal', 'felony', 'misdemeanor'],
      corporate: ['corporation', 'merger', 'securities'],
      employment: ['employment', 'discrimination', 'wrongful termination'],
    };

    const textLower = text.toLowerCase();

    for (const [area, terms] of Object.entries(practiceAreas)) {
      if (terms.some((term) => textLower.includes(term))) {
        return area;
      }
    }

    return 'general';
  }

  private assessUrgency(text: string): 'low' | 'medium' | 'high' {
    const urgentTerms = ['emergency', 'urgent', 'immediate', 'expedited', 'deadline'];
    const textLower = text.toLowerCase();

    const urgentCount = urgentTerms.reduce((count, term) => {
      return count + (textLower.includes(term) ? 1 : 0);
    }, 0);

    if (urgentCount >= 2) return 'high';
    if (urgentCount >= 1) return 'medium';
    return 'low';
  }

  private generateFallbackAnalysis(text: string): LegalBertAnalysisResult {
    return {
      entities: [],
      concepts: [{ concept: 'general', relevance: 0.5, category: 'unknown' }],
      sentiment: { polarity: 0, confidence: 0.3, classification: 'neutral' },
      complexity: { readabilityScore: 50, legalComplexity: 0.5, technicalTerms: 0 },
      keyPhrases: [{ phrase: 'legal document', importance: 0.5, category: 'general' }],
      summary: {
        abstractive: 'Legal document analysis unavailable',
        extractive: [text.substring(0, 100) + '...'],
        keyPoints: ['Document requires manual review'],
      },
    };
  }

  private generateFallbackClassification(text: string): LegalClassificationResult {
    return {
      documentType: 'general',
      confidence: 0.3,
      subCategories: [{ category: 'unknown', confidence: 0.3 }],
      jurisdiction: 'unknown',
      practiceArea: 'general',
      urgency: 'medium',
      recommendations: ['Manual classification required', 'Review document type'],
    };
  }

  // === PUBLIC API METHODS ===

  /**
   * Get middleware statistics
   */
  getStatistics(): Record<string, any> {
    return {
      requestCount: this.requestCount,
      cacheSize: this.cache.size,
      model: this.modelConfig,
      // Basic metrics structure (no getAllMetrics available)
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
    logger.info('[LegalBERT] Cache cleared');
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; details: Record<string, any> }> {
    try {
      const testResult = await this.generateLegalEmbedding('health check');
      return {
        status: 'healthy',
        details: {
          model: this.modelConfig.embedding,
          embeddingDimensions: testResult.dimensions,
          confidence: testResult.confidence,
          cacheSize: this.cache.size,
        },
      };
    } catch (error: any) {
      return {
        status: 'unhealthy',
        details: {
          error: error.message,
          model: this.modelConfig.embedding,
        },
      };
    }
  }
}

// Export singleton instance
export const legalBERT = new LegalBERTMiddleware();
;
// Types are exported above where they're defined
