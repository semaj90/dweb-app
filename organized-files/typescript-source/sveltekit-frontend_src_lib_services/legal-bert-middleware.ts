// Legal-BERT Middleware with TensorFlow.js
// Advanced NLP preprocessing and analysis for legal documents

import * as tf from '@tensorflow/tfjs';
import type { Tensor } from '@tensorflow/tfjs';
import { browser } from '$app/environment';

export interface LegalBERTConfig {
  modelUrl: string;
  vocabUrl: string;
  maxSequenceLength: number;
  enableGPU: boolean;
  batchSize: number;
  confidenceThreshold: number;
}

export interface LegalEntity {
  text: string;
  label: string;
  confidence: number;
  startIndex: number;
  endIndex: number;
  category: 'PERSON' | 'ORGANIZATION' | 'CONTRACT' | 'DATE' | 'MONEY' | 'LEGAL_CONCEPT' | 'JURISDICTION' | 'OTHER';
}

export interface LegalClassification {
  documentType: string;
  confidence: number;
  practiceArea: string[];
  jurisdiction: string[];
  complexity: 'LOW' | 'MEDIUM' | 'HIGH';
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export interface LegalBERTAnalysis {
  entities: LegalEntity[];
  classification: LegalClassification;
  keyTerms: string[];
  sentiment: {
    score: number;
    label: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
    confidence: number;
  };
  embeddings: Float32Array;
  processingTime: number;
  modelVersion: string;
}

export interface LanguageExtractionResult {
  language: string;
  confidence: number;
  supportedForLegal: boolean;
  dialectVariant?: string;
}

/**
 * Legal-BERT Middleware Service
 * Provides advanced NLP preprocessing for legal document analysis
 */
export class LegalBERTMiddleware {
  private model: tf.GraphModel | null = null;
  private tokenizer: any = null;
  private vocabulary: Map<string, number> = new Map();
  private config: LegalBERTConfig;
  private initialized = false;
  private modelVersion = 'legal-bert-v2.1';

  // Legal domain-specific patterns
  private readonly LEGAL_PATTERNS = {
    MONEY: /\$[\d,]+(?:\.\d{2})?|USD\s*[\d,]+|\d+\s*dollars?/gi,
    DATE: /\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/gi,
    CONTRACT_TYPE: /\b(?:employment|service|lease|license|purchase|sale|merger|acquisition|confidentiality|non-disclosure|partnership|joint venture)\s+(?:agreement|contract)\b/gi,
    LEGAL_CONCEPTS: /\b(?:indemnification|liability|breach|termination|force majeure|intellectual property|confidentiality|warranty|representation|covenant|jurisdiction|governing law|dispute resolution|arbitration|mediation)\b/gi,
    JURISDICTION: /\b(?:federal|state|local|California|New York|Delaware|Texas|Florida|Illinois|Pennsylvania|Ohio|Georgia|North Carolina|Michigan|New Jersey)\s+(?:law|jurisdiction|court)\b/gi,
    PERSON: /\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b/g,
    ORGANIZATION: /\b[A-Z][a-zA-Z\s&]+(Inc\.?|LLC|Corp\.?|Corporation|Company|Ltd\.?|Limited|Partnership|LP|LLP)\b/g
  };

  private readonly LEGAL_VOCABULARY = [
    'contract', 'agreement', 'party', 'parties', 'obligation', 'right', 'duty',
    'liability', 'indemnification', 'breach', 'termination', 'force majeure',
    'intellectual property', 'confidentiality', 'warranty', 'representation',
    'covenant', 'jurisdiction', 'governing law', 'dispute resolution',
    'arbitration', 'mediation', 'damages', 'remedy', 'compliance',
    'regulation', 'statute', 'precedent', 'case law', 'common law',
    'civil law', 'criminal law', 'tort', 'negligence', 'strict liability',
    'constitutional', 'statutory', 'regulatory', 'contractual', 'fiduciary'
  ];

  constructor(config: Partial<LegalBERTConfig> = {}) {
    this.config = {
      modelUrl: '/models/legal-bert/model.json',
      vocabUrl: '/models/legal-bert/vocab.json',
      maxSequenceLength: 512,
      enableGPU: browser && !!navigator.gpu,
      batchSize: 32,
      confidenceThreshold: 0.7,
      ...config
    };
  }

  /**
   * Initialize Legal-BERT middleware
   */
  async initialize(): Promise<boolean> {
    if (!browser) {
      console.warn('[Legal-BERT] Not running in browser environment');
      return false;
    }

    if (this.initialized) {
      return true;
    }

    try {
      console.log('[Legal-BERT] Initializing TensorFlow.js backend...');
      
      // Initialize TensorFlow backend
      if (this.config.enableGPU) {
        try {
          await tf.setBackend('webgl');
          console.log('[Legal-BERT] WebGL backend enabled');
        } catch (error) {
          console.warn('[Legal-BERT] WebGL backend failed, falling back to CPU:', error);
          await tf.setBackend('cpu');
        }
      } else {
        await tf.setBackend('cpu');
        console.log('[Legal-BERT] CPU backend enabled');
      }

      await tf.ready();
      
      // Load Legal-BERT model (simulated for now)
      console.log('[Legal-BERT] Loading Legal-BERT model...');
      
      try {
        // In production, this would load the actual Legal-BERT model
        // this.model = await tf.loadGraphModel(this.config.modelUrl);
        
        // For now, create a mock model architecture
        this.model = this.createMockLegalBERTModel();
        console.log('[Legal-BERT] Mock Legal-BERT model created');
        
        // Load vocabulary
        await this.loadVocabulary();
        
        // Initialize tokenizer
        this.tokenizer = this.createMockTokenizer();
        
        this.initialized = true;
        console.log('[Legal-BERT] Initialization complete');
        return true;
        
      } catch (error: any) {
        console.error('[Legal-BERT] Model loading failed:', error);
        return false;
      }

    } catch (error: any) {
      console.error('[Legal-BERT] Initialization failed:', error);
      return false;
    }
  }

  /**
   * Analyze legal text with Legal-BERT
   */
  async analyzeLegalText(text: string): Promise<LegalBERTAnalysis> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized || !this.model) {
      throw new Error('Legal-BERT middleware not initialized');
    }

    const startTime = performance.now();

    try {
      // Step 1: Language detection and extraction
      const language = await this.detectLanguage(text);
      if (!language.supportedForLegal) {
        throw new Error(`Unsupported language for legal analysis: ${language.language}`);
      }

      // Step 2: Tokenization
      const tokens = await this.tokenizeText(text);
      const inputTensor = this.createInputTensor(tokens);

      // Step 3: Legal entity extraction
      const entities = await this.extractLegalEntities(text, inputTensor);

      // Step 4: Document classification
      const classification = await this.classifyLegalDocument(text, inputTensor);

      // Step 5: Key term extraction
      const keyTerms = this.extractKeyTerms(text);

      // Step 6: Sentiment analysis (legal-specific)
      const sentiment = await this.analyzeLegalSentiment(text, inputTensor);

      // Step 7: Generate embeddings
      const embeddings = await this.generateEmbeddings(inputTensor);

      const processingTime = performance.now() - startTime;

      return {
        entities,
        classification,
        keyTerms,
        sentiment,
        embeddings,
        processingTime,
        modelVersion: this.modelVersion
      };

    } catch (error: any) {
      console.error('[Legal-BERT] Analysis failed:', error);
      throw error;
    }
  }

  /**
   * Extract legal entities using BERT-based NER
   */
  private async extractLegalEntities(text: string, inputTensor: tf.Tensor): Promise<LegalEntity[]> {
    const entities: LegalEntity[] = [];

    // Pattern-based entity extraction (enhanced with BERT confidence scoring)
    for (const [category, pattern] of Object.entries(this.LEGAL_PATTERNS)) {
      const matches = Array.from(text.matchAll(pattern));
      
      for (const match of matches) {
        if (match.index !== undefined) {
          // Use BERT to score confidence (simulated for now)
          const confidence = await this.scoreBERTConfidence(match[0], category, inputTensor);
          
          entities.push({
            text: match[0],
            label: category,
            confidence,
            startIndex: match.index,
            endIndex: match.index + match[0].length,
            category: this.mapToEntityCategory(category)
          });
        }
      }
    }

    // Sort by confidence and remove low-confidence entities
    return entities
      .filter(entity => entity.confidence >= this.config.confidenceThreshold)
      .sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Classify legal document type and characteristics
   */
  private async classifyLegalDocument(text: string, inputTensor: tf.Tensor): Promise<LegalClassification> {
    // Simulate BERT-based classification
    const documentTypes = ['contract', 'agreement', 'policy', 'regulation', 'statute', 'case_law'];
    const practiceAreas = ['corporate', 'employment', 'intellectual_property', 'real_estate', 'litigation'];
    const jurisdictions = ['federal', 'state', 'california', 'new_york', 'delaware'];

    // Use BERT embeddings to classify (simulated)
    const documentType = await this.predictDocumentType(text, inputTensor);
    const practiceArea = await this.predictPracticeAreas(text, inputTensor);
    const jurisdiction = await this.predictJurisdictions(text, inputTensor);
    const complexity = this.assessComplexity(text);
    const riskLevel = this.assessRiskLevel(text);

    return {
      documentType: documentType.type,
      confidence: documentType.confidence,
      practiceArea,
      jurisdiction,
      complexity,
      riskLevel
    };
  }

  /**
   * Extract key legal terms with BERT-based importance scoring
   */
  private extractKeyTerms(text: string): string[] {
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const legalTerms = words.filter(word => 
      this.LEGAL_VOCABULARY.includes(word) && word.length > 3
    );

    // Remove duplicates and sort by importance (simplified)
    const uniqueTerms = Array.from(new Set(legalTerms));
    
    // In production, use BERT embeddings to score importance
    return uniqueTerms.slice(0, 20); // Top 20 terms
  }

  /**
   * Analyze legal sentiment (different from general sentiment)
   */
  private async analyzeLegalSentiment(text: string, inputTensor: tf.Tensor): Promise<{
    score: number;
    label: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
    confidence: number;
  }> {
    // Legal sentiment focuses on risk, obligation, and enforceability
    const riskIndicators = ['breach', 'violation', 'penalty', 'damages', 'liability', 'terminate'];
    const positiveIndicators = ['benefit', 'right', 'protection', 'warranty', 'indemnify'];
    const neutralIndicators = ['shall', 'will', 'agree', 'party', 'section', 'clause'];

    const textLower = text.toLowerCase();
    let riskScore = 0;
    let positiveScore = 0;
    let neutralScore = 0;

    riskIndicators.forEach(indicator => {
      const matches = (textLower.match(new RegExp(indicator, 'g')) || []).length;
      riskScore += matches;
    });

    positiveIndicators.forEach(indicator => {
      const matches = (textLower.match(new RegExp(indicator, 'g')) || []).length;
      positiveScore += matches;
    });

    neutralIndicators.forEach(indicator => {
      const matches = (textLower.match(new RegExp(indicator, 'g')) || []).length;
      neutralScore += matches;
    });

    const total = riskScore + positiveScore + neutralScore;
    if (total === 0) {
      return { score: 0, label: 'NEUTRAL', confidence: 0.5 };
    }

    const normalizedRisk = riskScore / total;
    const normalizedPositive = positiveScore / total;

    let label: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
    let score: number;
    let confidence: number;

    if (normalizedRisk > normalizedPositive && normalizedRisk > 0.3) {
      label = 'NEGATIVE';
      score = -normalizedRisk;
      confidence = normalizedRisk;
    } else if (normalizedPositive > normalizedRisk && normalizedPositive > 0.3) {
      label = 'POSITIVE';
      score = normalizedPositive;
      confidence = normalizedPositive;
    } else {
      label = 'NEUTRAL';
      score = 0;
      confidence = neutralScore / total;
    }

    return { score, label, confidence };
  }

  /**
   * Detect language and assess legal support
   */
  private async detectLanguage(text: string): Promise<LanguageExtractionResult> {
    // Simplified language detection - in production would use proper language detection
    const englishPatterns = /\b(?:the|and|of|to|a|in|is|it|you|that|he|was|for|on|are|as|with|his|they|at|be|this|have|from|or|one|had|by|words|but|not|what|all|were|we|when|your|can|said|there|use|an|each|which|do|how|their|if|up|about|get|who|oil|its|now)\b/gi;
    const englishMatches = (text.match(englishPatterns) || []).length;
    const totalWords = (text.match(/\b\w+\b/g) || []).length;
    const englishRatio = englishMatches / totalWords;

    if (englishRatio > 0.3) {
      return {
        language: 'en',
        confidence: Math.min(englishRatio * 2, 1.0),
        supportedForLegal: true,
        dialectVariant: 'us' // Assume US legal English
      };
    }

    // For non-English text, check if it's a supported legal language
    return {
      language: 'unknown',
      confidence: 0.5,
      supportedForLegal: false
    };
  }

  /**
   * Generate BERT embeddings for the input
   */
  private async generateEmbeddings(inputTensor: tf.Tensor): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    try {
      // In production, this would use the actual BERT model
      // const embeddings = this.model.predict(inputTensor) as tf.Tensor;
      
      // For now, generate mock embeddings
      const embeddingSize = 768; // BERT-base embedding size
      const mockEmbeddings = new Float32Array(embeddingSize);
      
      // Generate deterministic mock embeddings based on input
      const inputArray = await inputTensor.data();
      for (let i = 0; i < embeddingSize; i++) {
        mockEmbeddings[i] = Math.sin(i * 0.1 + (inputArray[i % inputArray.length] || 0)) * 0.1;
      }

      return mockEmbeddings;

    } catch (error: any) {
      console.error('[Legal-BERT] Embedding generation failed:', error);
      throw error;
    }
  }

  // Helper methods

  private createMockLegalBERTModel(): tf.GraphModel {
    // Create a simplified mock model for demonstration
    // In production, this would load the actual Legal-BERT weights
    return {} as tf.GraphModel;
  }

  private async loadVocabulary(): Promise<void> {
    // Load BERT vocabulary (simplified)
    const commonTokens = [
      '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
      ...this.LEGAL_VOCABULARY,
      'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that'
    ];

    commonTokens.forEach((token, index) => {
      this.vocabulary.set(token, index);
    });

    console.log(`[Legal-BERT] Loaded vocabulary with ${this.vocabulary.size} tokens`);
  }

  private createMockTokenizer(): any {
    return {
      tokenize: (text: string) => {
        // Simple whitespace tokenization (BERT uses WordPiece)
        return text.toLowerCase().match(/\b\w+\b/g) || [];
      },
      encode: (tokens: string[]) => {
        return tokens.map(token => this.vocabulary.get(token) || this.vocabulary.get('[UNK]') || 0);
      }
    };
  }

  private async tokenizeText(text: string): Promise<number[]> {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized');
    }

    const tokens = this.tokenizer.tokenize(text);
    const encoded = this.tokenizer.encode(tokens);
    
    // Add special tokens and pad/truncate to max length
    const withSpecialTokens = [
      this.vocabulary.get('[CLS]') || 0,
      ...encoded.slice(0, this.config.maxSequenceLength - 2),
      this.vocabulary.get('[SEP]') || 0
    ];

    // Pad to max sequence length
    while (withSpecialTokens.length < this.config.maxSequenceLength) {
      withSpecialTokens.push(this.vocabulary.get('[PAD]') || 0);
    }

    return withSpecialTokens;
  }

  private createInputTensor(tokens: number[]): tf.Tensor {
    return tf.tensor2d([tokens], [1, this.config.maxSequenceLength]);
  }

  private async scoreBERTConfidence(text: string, category: string, inputTensor: tf.Tensor): Promise<number> {
    // Simulate BERT-based confidence scoring
    const baseConfidence = 0.8;
    const textLength = text.length;
    const lengthBonus = Math.min(textLength / 50, 0.2);
    
    return Math.min(baseConfidence + lengthBonus + Math.random() * 0.1, 1.0);
  }

  private mapToEntityCategory(label: string): LegalEntity['category'] {
    const categoryMap: Record<string, LegalEntity['category']> = {
      'MONEY': 'MONEY',
      'DATE': 'DATE',
      'PERSON': 'PERSON',
      'ORGANIZATION': 'ORGANIZATION',
      'CONTRACT_TYPE': 'CONTRACT',
      'LEGAL_CONCEPTS': 'LEGAL_CONCEPT',
      'JURISDICTION': 'JURISDICTION'
    };
    
    return categoryMap[label] || 'OTHER';
  }

  private async predictDocumentType(text: string, inputTensor: tf.Tensor): Promise<{type: string, confidence: number}> {
    // Simulate document type prediction
    const contractIndicators = ['agreement', 'contract', 'party', 'obligation'];
    const policyIndicators = ['policy', 'procedure', 'guideline', 'standard'];
    const regulationIndicators = ['regulation', 'rule', 'requirement', 'compliance'];

    const textLower = text.toLowerCase();
    let contractScore = 0;
    let policyScore = 0;
    let regulationScore = 0;

    contractIndicators.forEach(indicator => {
      contractScore += (textLower.match(new RegExp(indicator, 'g')) || []).length;
    });

    policyIndicators.forEach(indicator => {
      policyScore += (textLower.match(new RegExp(indicator, 'g')) || []).length;
    });

    regulationIndicators.forEach(indicator => {
      regulationScore += (textLower.match(new RegExp(indicator, 'g')) || []).length;
    });

    if (contractScore >= policyScore && contractScore >= regulationScore) {
      return { type: 'contract', confidence: Math.min(contractScore / 10, 1.0) };
    } else if (policyScore >= regulationScore) {
      return { type: 'policy', confidence: Math.min(policyScore / 10, 1.0) };
    } else {
      return { type: 'regulation', confidence: Math.min(regulationScore / 10, 1.0) };
    }
  }

  private async predictPracticeAreas(text: string, inputTensor: tf.Tensor): Promise<string[]> {
    const practiceAreaKeywords = {
      'corporate': ['corporation', 'merger', 'acquisition', 'securities', 'governance'],
      'employment': ['employee', 'employment', 'workplace', 'benefits', 'termination'],
      'intellectual_property': ['patent', 'trademark', 'copyright', 'trade secret', 'intellectual property'],
      'real_estate': ['property', 'real estate', 'lease', 'mortgage', 'title'],
      'litigation': ['lawsuit', 'litigation', 'court', 'dispute', 'trial']
    };

    const textLower = text.toLowerCase();
    const practiceAreas: string[] = [];

    for (const [area, keywords] of Object.entries(practiceAreaKeywords)) {
      const score = keywords.reduce((sum, keyword) => {
        return sum + (textLower.match(new RegExp(keyword, 'g')) || []).length;
      }, 0);

      if (score > 0) {
        practiceAreas.push(area);
      }
    }

    return practiceAreas.length > 0 ? practiceAreas : ['general'];
  }

  private async predictJurisdictions(text: string, inputTensor: tf.Tensor): Promise<string[]> {
    const jurisdictionPatterns = {
      'federal': /\b(?:federal|united states|u\.?s\.?)\b/gi,
      'california': /\b(?:california|ca|calif\.?)\b/gi,
      'new_york': /\b(?:new york|ny|n\.?y\.?)\b/gi,
      'delaware': /\b(?:delaware|de|del\.?)\b/gi,
      'texas': /\b(?:texas|tx|tex\.?)\b/gi
    };

    const jurisdictions: string[] = [];
    
    for (const [jurisdiction, pattern] of Object.entries(jurisdictionPatterns)) {
      if (pattern.test(text)) {
        jurisdictions.push(jurisdiction);
      }
    }

    return jurisdictions.length > 0 ? jurisdictions : ['general'];
  }

  private assessComplexity(text: string): 'LOW' | 'MEDIUM' | 'HIGH' {
    const wordCount = (text.match(/\b\w+\b/g) || []).length;
    const sentenceCount = (text.match(/[.!?]+/g) || []).length;
    const avgWordsPerSentence = sentenceCount > 0 ? wordCount / sentenceCount : wordCount;
    
    const legalTermCount = this.LEGAL_VOCABULARY.filter(term => 
      text.toLowerCase().includes(term)
    ).length;

    const complexityScore = (avgWordsPerSentence / 20) + (legalTermCount / 10) + (wordCount / 1000);

    if (complexityScore < 1) return 'LOW';
    if (complexityScore < 2) return 'MEDIUM';
    return 'HIGH';
  }

  private assessRiskLevel(text: string): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    const highRiskTerms = ['terminate', 'breach', 'penalty', 'damages', 'liability', 'indemnify'];
    const mediumRiskTerms = ['obligation', 'requirement', 'compliance', 'warranty', 'representation'];
    
    const textLower = text.toLowerCase();
    
    const highRiskScore = highRiskTerms.reduce((sum, term) => 
      sum + (textLower.match(new RegExp(term, 'g')) || []).length, 0
    );
    
    const mediumRiskScore = mediumRiskTerms.reduce((sum, term) => 
      sum + (textLower.match(new RegExp(term, 'g')) || []).length, 0
    );

    const totalRisk = highRiskScore * 2 + mediumRiskScore;
    
    if (totalRisk > 10) return 'CRITICAL';
    if (totalRisk > 5) return 'HIGH';
    if (totalRisk > 2) return 'MEDIUM';
    return 'LOW';
  }

  /**
   * Get health status
   */
  getHealthStatus(): {
    initialized: boolean;
    modelLoaded: boolean;
    tokenizerReady: boolean;
    vocabularySize: number;
    tfBackend: string;
    memoryUsage: { numTensors: number; numBytes: number };
  } {
    return {
      initialized: this.initialized,
      modelLoaded: !!this.model,
      tokenizerReady: !!this.tokenizer,
      vocabularySize: this.vocabulary.size,
      tfBackend: tf.getBackend(),
      memoryUsage: tf.memory()
    };
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.model) {
      // In production, dispose of the actual model
      this.model = null;
    }
    
    // Clean up TensorFlow tensors
    tf.disposeVariables();
    
    this.initialized = false;
    console.log('[Legal-BERT] Resources disposed');
  }
}

// Export singleton instance
export const legalBERTMiddleware = new LegalBERTMiddleware();