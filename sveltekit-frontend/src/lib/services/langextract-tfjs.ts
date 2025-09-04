// Advanced Language Extraction with TensorFlow.js
// Enhanced NLP for legal document processing and multi-language support

import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { browser } from '$app/environment';
import type { LegalEntity, LegalBERTAnalysis } from './legal-bert-middleware.js';

export interface LanguageExtractionConfig {
  enableUniversalEncoder: boolean;
  enableMultiLanguageSupport: boolean;
  enableSemanticSimilarity: boolean;
  confidenceThreshold: number;
  maxTextLength: number;
  cachingEnabled: boolean;
}

export interface ExtractedConcept {
  concept: string;
  category: 'legal' | 'business' | 'technical' | 'procedural';
  importance: number;
  context: string;
  semanticCluster: string;
  relatedTerms: string[];
}

export interface SemanticRelationship {
  source: string;
  target: string;
  relationship: 'synonyms' | 'antonyms' | 'implies' | 'requires' | 'conflicts';
  confidence: number;
  contextual: boolean;
}

export interface LanguageStructure {
  sentences: SentenceAnalysis[];
  paragraphs: ParagraphAnalysis[];
  documentStructure: DocumentStructure;
  writingStyle: WritingStyleAnalysis;
}

export interface SentenceAnalysis {
  text: string;
  embedding: Float32Array;
  complexity: number;
  legalWeight: number;
  entities: LegalEntity[];
  sentiment: number;
  syntacticRole: 'definition' | 'obligation' | 'condition' | 'exception' | 'general';
}

export interface ParagraphAnalysis {
  sentences: SentenceAnalysis[];
  mainTopic: string;
  legalFunction: 'preamble' | 'definitions' | 'obligations' | 'conditions' | 'remedies' | 'general';
  coherenceScore: number;
  embedding: Float32Array;
}

export interface DocumentStructure {
  sections: string[];
  hierarchy: string[];
  crossReferences: string[];
  definitionTerms: string[];
  obligations: string[];
  conditions: string[];
}

export interface WritingStyleAnalysis {
  formalityScore: number;
  complexityScore: number;
  passiveVoiceRatio: number;
  averageSentenceLength: number;
  legalJargonDensity: number;
  readabilityScore: number;
}

export interface AdvancedExtractionResult {
  extractedConcepts: ExtractedConcept[];
  semanticRelationships: SemanticRelationship[];
  languageStructure: LanguageStructure;
  documentEmbedding: Float32Array;
  keyPhrases: string[];
  abstractiveSummary: string;
  processingMetrics: {
    processingTime: number;
    tensorsUsed: number;
    memoryUsage: number;
    modelAccuracy: number;
  };
}

/**
 * Advanced Language Extraction Service
 * Powered by TensorFlow.js and Universal Sentence Encoder
 */
export class LangExtractTensorFlow {
  private useModel: use.UniversalSentenceEncoder | null = null;
  private config: LanguageExtractionConfig;
  private initialized = false;
  private conceptCache = new Map<string, ExtractedConcept[]>();
  private embeddingCache = new Map<string, Float32Array>();

  // Legal concept taxonomy
  private readonly LEGAL_TAXONOMY = {
    contractual: {
      formation: ['offer', 'acceptance', 'consideration', 'capacity', 'legality'],
      performance: ['obligations', 'duties', 'conditions', 'warranties', 'representations'],
      breach: ['breach', 'damages', 'remedies', 'specific performance', 'restitution'],
      termination: ['termination', 'expiration', 'dissolution', 'cancellation', 'rescission']
    },
    property: {
      ownership: ['title', 'possession', 'ownership', 'property rights', 'estate'],
      transfer: ['sale', 'assignment', 'conveyance', 'transfer', 'delivery'],
      security: ['lien', 'mortgage', 'pledge', 'security interest', 'collateral']
    },
    corporate: {
      governance: ['board', 'directors', 'shareholders', 'officers', 'fiduciary'],
      compliance: ['securities', 'disclosure', 'reporting', 'audit', 'internal controls'],
      transactions: ['merger', 'acquisition', 'reorganization', 'spin-off', 'joint venture']
    },
    litigation: {
      procedure: ['jurisdiction', 'venue', 'service', 'pleadings', 'discovery'],
      evidence: ['admissible', 'hearsay', 'relevance', 'prejudicial', 'expert testimony'],
      remedies: ['injunction', 'damages', 'restitution', 'declaratory', 'mandamus']
    }
  };

  // Semantic relationship patterns
  private readonly SEMANTIC_PATTERNS = {
    synonyms: [
      ['contract', 'agreement'], ['party', 'contracting party'], ['breach', 'violation'],
      ['terminate', 'end'], ['indemnify', 'hold harmless'], ['warranty', 'guarantee']
    ],
    implications: [
      ['breach', 'damages'], ['termination', 'cessation of obligations'], ['force majeure', 'excuse'],
      ['confidentiality', 'non-disclosure'], ['intellectual property', 'ownership rights']
    ],
    conflicts: [
      ['warranty', 'disclaimer'], ['liability', 'limitation of liability'], ['obligation', 'excuse']
    ]
  };

  constructor(config: Partial<LanguageExtractionConfig> = {}) {
    this.config = {
      enableUniversalEncoder: true,
      enableMultiLanguageSupport: false, // Start with English only
      enableSemanticSimilarity: true,
      confidenceThreshold: 0.75,
      maxTextLength: 10000,
      cachingEnabled: true,
      ...config
    };
  }

  /**
   * Initialize TensorFlow.js language extraction service
   */
  async initialize(): Promise<boolean> {
    if (!browser) {
      console.warn('[LangExtract] Not running in browser environment');
      return false;
    }

    if (this.initialized) {
      return true;
    }

    try {
      console.log('[LangExtract] Initializing TensorFlow.js language extraction...');

      // Initialize TensorFlow backend
      await tf.ready();
      console.log('[LangExtract] TensorFlow.js ready, backend:', tf.getBackend());

      if (this.config.enableUniversalEncoder) {
        try {
          console.log('[LangExtract] Loading Universal Sentence Encoder...');
          this.useModel = await use.load();
          console.log('[LangExtract] Universal Sentence Encoder loaded successfully');
        } catch (error) {
          console.error('[LangExtract] Failed to load Universal Sentence Encoder:', error);
          // Continue without USE model
        }
      }

      this.initialized = true;
      console.log('[LangExtract] Initialization complete');
      return true;

    } catch (error: any) {
      console.error('[LangExtract] Initialization failed:', error);
      return false;
    }
  }

  /**
   * Extract advanced language features from legal text
   */
  async extractAdvancedFeatures(text: string): Promise<AdvancedExtractionResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    const startTime = performance.now();
    const initialMemory = tf.memory();

    try {
      // Step 1: Preprocess text
      const preprocessedText = this.preprocessText(text);
      
      // Step 2: Extract concepts using semantic understanding
      const extractedConcepts = await this.extractSemanticConcepts(preprocessedText);
      
      // Step 3: Identify semantic relationships
      const semanticRelationships = await this.identifySemanticRelationships(extractedConcepts, preprocessedText);
      
      // Step 4: Analyze language structure
      const languageStructure = await this.analyzeLanguageStructure(preprocessedText);
      
      // Step 5: Generate document-level embeddings
      const documentEmbedding = await this.generateDocumentEmbedding(preprocessedText);
      
      // Step 6: Extract key phrases
      const keyPhrases = this.extractKeyPhrases(preprocessedText, extractedConcepts);
      
      // Step 7: Generate abstractive summary
      const abstractiveSummary = await this.generateAbstractiveSummary(preprocessedText, extractedConcepts);

      const endTime = performance.now();
      const finalMemory = tf.memory();

      return {
        extractedConcepts,
        semanticRelationships,
        languageStructure,
        documentEmbedding,
        keyPhrases,
        abstractiveSummary,
        processingMetrics: {
          processingTime: endTime - startTime,
          tensorsUsed: finalMemory.numTensors - initialMemory.numTensors,
          memoryUsage: finalMemory.numBytes - initialMemory.numBytes,
          modelAccuracy: this.estimateAccuracy(extractedConcepts)
        }
      };

    } catch (error: any) {
      console.error('[LangExtract] Advanced extraction failed:', error);
      throw error;
    }
  }

  /**
   * Extract semantic concepts using Universal Sentence Encoder
   */
  private async extractSemanticConcepts(text: string): Promise<ExtractedConcept[]> {
    const concepts: ExtractedConcept[] = [];
    
    // Check cache first
    if (this.config.cachingEnabled && this.conceptCache.has(text)) {
      return this.conceptCache.get(text)!;
    }

    try {
      // Split into sentences for concept extraction
      const sentences = this.splitIntoSentences(text);
      
      // Get embeddings for each sentence if USE is available
      let sentenceEmbeddings: tf.Tensor2D | null = null;
      if (this.useModel) {
        sentenceEmbeddings = await this.useModel.embed(sentences);
      }

      // Extract concepts from each sentence
      for (let i = 0; i < sentences.length; i++) {
        const sentence = sentences[i];
        const sentenceConcepts = await this.extractConceptsFromSentence(
          sentence, 
          sentenceEmbeddings ? sentenceEmbeddings.slice([i, 0], [1, -1]).squeeze() : null
        );
        concepts.push(...sentenceConcepts);
      }

      // Clean up tensors
      if (sentenceEmbeddings) {
        sentenceEmbeddings.dispose();
      }

      // Cluster and deduplicate concepts
      const clusteredConcepts = this.clusterConcepts(concepts);

      // Cache results
      if (this.config.cachingEnabled) {
        this.conceptCache.set(text, clusteredConcepts);
      }

      return clusteredConcepts;

    } catch (error: any) {
      console.error('[LangExtract] Concept extraction failed:', error);
      return [];
    }
  }

  /**
   * Extract concepts from individual sentence
   */
  private async extractConceptsFromSentence(
    sentence: string, 
    embedding: tf.Tensor | null
  ): Promise<ExtractedConcept[]> {
    const concepts: ExtractedConcept[] = [];
    
    // Extract legal concepts using taxonomy matching
    for (const [domain, categories] of Object.entries(this.LEGAL_TAXONOMY)) {
      for (const [category, terms] of Object.entries(categories)) {
        for (const term of terms) {
          const regex = new RegExp(`\\b${term}\\b`, 'gi');
          const matches = sentence.match(regex);
          
          if (matches) {
            const importance = this.calculateConceptImportance(term, sentence, embedding);
            
            concepts.push({
              concept: term,
              category: 'legal',
              importance,
              context: sentence,
              semanticCluster: `${domain}.${category}`,
              relatedTerms: terms.filter(t => t !== term)
            });
          }
        }
      }
    }

    return concepts;
  }

  /**
   * Identify semantic relationships between concepts
   */
  private async identifySemanticRelationships(
    concepts: ExtractedConcept[],
    text: string
  ): Promise<SemanticRelationship[]> {
    const relationships: SemanticRelationship[] = [];

    // Pattern-based relationship detection
    for (const [relationshipType, patterns] of Object.entries(this.SEMANTIC_PATTERNS)) {
      for (const pattern of patterns) {
        const [source, target] = pattern;
        
        // Check if both concepts exist in extracted concepts
        const sourceConcept = concepts.find(c => c.concept.toLowerCase().includes(source.toLowerCase()));
        const targetConcept = concepts.find(c => c.concept.toLowerCase().includes(target.toLowerCase()));
        
        if (sourceConcept && targetConcept) {
          const confidence = this.calculateRelationshipConfidence(source, target, text);
          
          if (confidence >= this.config.confidenceThreshold) {
            relationships.push({
              source,
              target,
              relationship: relationshipType as SemanticRelationship['relationship'],
              confidence,
              contextual: this.isContextualRelationship(source, target, text)
            });
          }
        }
      }
    }

    // Semantic similarity-based relationships if USE is available
    if (this.useModel && concepts.length > 1) {
      const conceptTexts = concepts.map(c => c.concept);
      const embeddings = await this.useModel.embed(conceptTexts);
      
      // Calculate pairwise similarities
      for (let i = 0; i < concepts.length; i++) {
        for (let j = i + 1; j < concepts.length; j++) {
          const similarity = await this.calculateCosineSimilarity(
            embeddings.slice([i, 0], [1, -1]),
            embeddings.slice([j, 0], [1, -1])
          );
          
          if (similarity > 0.8) {
            relationships.push({
              source: concepts[i].concept,
              target: concepts[j].concept,
              relationship: 'synonyms',
              confidence: similarity,
              contextual: true
            });
          }
        }
      }
      
      embeddings.dispose();
    }

    return relationships;
  }

  /**
   * Analyze language structure comprehensively
   */
  private async analyzeLanguageStructure(text: string): Promise<LanguageStructure> {
    const sentences = this.splitIntoSentences(text);
    const paragraphs = this.splitIntoParagraphs(text);

    // Analyze sentences
    const sentenceAnalyses: SentenceAnalysis[] = [];
    for (const sentence of sentences) {
      const analysis = await this.analyzeSentence(sentence);
      sentenceAnalyses.push(analysis);
    }

    // Analyze paragraphs
    const paragraphAnalyses: ParagraphAnalysis[] = [];
    for (const paragraph of paragraphs) {
      const analysis = await this.analyzeParagraph(paragraph);
      paragraphAnalyses.push(analysis);
    }

    // Analyze document structure
    const documentStructure = this.analyzeDocumentStructure(text);

    // Analyze writing style
    const writingStyle = this.analyzeWritingStyle(text, sentenceAnalyses);

    return {
      sentences: sentenceAnalyses,
      paragraphs: paragraphAnalyses,
      documentStructure,
      writingStyle
    };
  }

  /**
   * Generate document-level embeddings
   */
  private async generateDocumentEmbedding(text: string): Promise<Float32Array> {
    const cacheKey = `doc_${text.substring(0, 100)}`;
    
    if (this.config.cachingEnabled && this.embeddingCache.has(cacheKey)) {
      return this.embeddingCache.get(cacheKey)!;
    }

    try {
      if (this.useModel) {
        const embedding = await this.useModel.embed([text]);
        const embeddingArray = await embedding.data();
        embedding.dispose();
        
        const result = new Float32Array(embeddingArray);
        
        if (this.config.cachingEnabled) {
          this.embeddingCache.set(cacheKey, result);
        }
        
        return result;
      } else {
        // Fallback: create simple embedding based on concept frequency
        const concepts = await this.extractSemanticConcepts(text);
        const embedding = new Float32Array(512); // Fixed size
        
        concepts.forEach((concept, index) => {
          if (index < 512) {
            embedding[index] = concept.importance;
          }
        });
        
        return embedding;
      }
    } catch (error: any) {
      console.error('[LangExtract] Document embedding failed:', error);
      return new Float32Array(512);
    }
  }

  /**
   * Generate abstractive summary using concept extraction
   */
  private async generateAbstractiveSummary(
    text: string, 
    concepts: ExtractedConcept[]
  ): Promise<string> {
    // Identify most important concepts
    const topConcepts = concepts
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10);

    // Extract key obligations and conditions
    const obligations = topConcepts.filter(c => 
      c.semanticCluster.includes('obligations') || c.concept.includes('shall') || c.concept.includes('must')
    );

    const conditions = topConcepts.filter(c => 
      c.semanticCluster.includes('conditions') || c.concept.includes('if') || c.concept.includes('unless')
    );

    // Build abstractive summary
    let summary = 'This legal document ';

    if (obligations.length > 0) {
      summary += `establishes obligations related to ${obligations.map(o => o.concept).join(', ')}. `;
    }

    if (conditions.length > 0) {
      summary += `Key conditions include ${conditions.map(c => c.concept).join(', ')}. `;
    }

    const mainConcepts = topConcepts.slice(0, 5);
    if (mainConcepts.length > 0) {
      summary += `Primary legal concepts covered: ${mainConcepts.map(c => c.concept).join(', ')}.`;
    }

    return summary || 'Document summary not available.';
  }

  // Helper methods

  private preprocessText(text: string): string {
    // Truncate if too long
    if (text.length > this.config.maxTextLength) {
      text = text.substring(0, this.config.maxTextLength);
    }
    
    // Clean up whitespace and normalize
    return text.replace(/\s+/g, ' ').trim();
  }

  private splitIntoSentences(text: string): string[] {
    return text.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 10);
  }

  private splitIntoParagraphs(text: string): string[] {
    return text.split(/\n\s*\n/).map(p => p.trim()).filter(p => p.length > 20);
  }

  private calculateConceptImportance(term: string, context: string, embedding: tf.Tensor | null): number {
    // Base importance on term frequency and position
    const termCount = (context.toLowerCase().match(new RegExp(term.toLowerCase(), 'g')) || []).length;
    const positionWeight = context.toLowerCase().indexOf(term.toLowerCase()) < context.length / 2 ? 1.2 : 1.0;
    
    // Legal terms get higher importance
    const legalWeight = this.LEGAL_VOCABULARY.includes(term.toLowerCase()) ? 1.5 : 1.0;
    
    const baseImportance = Math.min(termCount * positionWeight * legalWeight * 0.2, 1.0);
    
    // If embedding available, boost importance based on semantic density
    if (embedding) {
      // Simplified: use embedding magnitude as a proxy for semantic richness
      try {
        const magnitude = embedding.norm().dataSync()[0];
        return Math.min(baseImportance * (1 + magnitude * 0.1), 1.0);
      } catch (error) {
        // If embedding calculation fails, use base importance
      }
    }
    
    return baseImportance;
  }

  private calculateRelationshipConfidence(source: string, target: string, text: string): number {
    const textLower = text.toLowerCase();
    const sourceCount = (textLower.match(new RegExp(source.toLowerCase(), 'g')) || []).length;
    const targetCount = (textLower.match(new RegExp(target.toLowerCase(), 'g')) || []).length;
    
    // Co-occurrence boost
    const cooccurrencePattern = new RegExp(`${source.toLowerCase()}.*?${target.toLowerCase()}|${target.toLowerCase()}.*?${source.toLowerCase()}`, 'gi');
    const cooccurrences = (textLower.match(cooccurrencePattern) || []).length;
    
    const baseConfidence = Math.min((sourceCount + targetCount) * 0.1, 0.8);
    const cooccurrenceBoost = cooccurrences * 0.2;
    
    return Math.min(baseConfidence + cooccurrenceBoost, 1.0);
  }

  private isContextualRelationship(source: string, target: string, text: string): boolean {
    const proximityThreshold = 100; // characters
    const textLower = text.toLowerCase();
    
    const sourceIndices = [];
    const targetIndices = [];
    
    let sourceIndex = textLower.indexOf(source.toLowerCase());
    while (sourceIndex !== -1) {
      sourceIndices.push(sourceIndex);
      sourceIndex = textLower.indexOf(source.toLowerCase(), sourceIndex + 1);
    }
    
    let targetIndex = textLower.indexOf(target.toLowerCase());
    while (targetIndex !== -1) {
      targetIndices.push(targetIndex);
      targetIndex = textLower.indexOf(target.toLowerCase(), targetIndex + 1);
    }
    
    // Check if any source and target are within proximity threshold
    for (const sIdx of sourceIndices) {
      for (const tIdx of targetIndices) {
        if (Math.abs(sIdx - tIdx) < proximityThreshold) {
          return true;
        }
      }
    }
    
    return false;
  }

  private async calculateCosineSimilarity(tensor1: tf.Tensor, tensor2: tf.Tensor): Promise<number> {
    try {
      const dotProduct = tf.sum(tf.mul(tensor1, tensor2));
      const norm1 = tf.norm(tensor1);
      const norm2 = tf.norm(tensor2);
      
      const similarity = tf.div(dotProduct, tf.mul(norm1, norm2));
      const result = await similarity.data();
      
      // Clean up tensors
      dotProduct.dispose();
      norm1.dispose();
      norm2.dispose();
      similarity.dispose();
      
      return result[0];
    } catch (error) {
      console.error('[LangExtract] Cosine similarity calculation failed:', error);
      return 0;
    }
  }

  private clusterConcepts(concepts: ExtractedConcept[]): ExtractedConcept[] {
    // Simple clustering based on semantic clusters and importance
    const clustered = new Map<string, ExtractedConcept>();
    
    for (const concept of concepts) {
      const key = concept.semanticCluster + concept.concept;
      const existing = clustered.get(key);
      
      if (!existing || concept.importance > existing.importance) {
        clustered.set(key, concept);
      }
    }
    
    return Array.from(clustered.values())
      .sort((a, b) => b.importance - a.importance);
  }

  private async analyzeSentence(sentence: string): Promise<SentenceAnalysis> {
    const embedding = this.useModel ? 
      await this.useModel.embed([sentence]).then(e => e.data().then(d => new Float32Array(d))) :
      new Float32Array(512);

    return {
      text: sentence,
      embedding,
      complexity: this.calculateSentenceComplexity(sentence),
      legalWeight: this.calculateLegalWeight(sentence),
      entities: [], // Would integrate with Legal-BERT here
      sentiment: this.calculateSentiment(sentence),
      syntacticRole: this.determineSyntacticRole(sentence)
    };
  }

  private async analyzeParagraph(paragraph: string): Promise<ParagraphAnalysis> {
    const sentences = this.splitIntoSentences(paragraph);
    const sentenceAnalyses: SentenceAnalysis[] = [];
    
    for (const sentence of sentences) {
      sentenceAnalyses.push(await this.analyzeSentence(sentence));
    }

    const embedding = this.useModel ?
      await this.useModel.embed([paragraph]).then(e => e.data().then(d => new Float32Array(d))) :
      new Float32Array(512);

    return {
      sentences: sentenceAnalyses,
      mainTopic: this.extractMainTopic(paragraph),
      legalFunction: this.determineLegalFunction(paragraph),
      coherenceScore: this.calculateCoherence(sentenceAnalyses),
      embedding
    };
  }

  private analyzeDocumentStructure(text: string): DocumentStructure {
    // Extract sections based on common patterns
    const sectionPattern = /(?:^|\n)(?:SECTION|Article|Part|Chapter)\s+[\dIVX]+/gm;
    const sections = (text.match(sectionPattern) || []).map(s => s.trim());

    // Extract definitions
    const definitionPattern = /"([^"]+)"|'([^']+)'|\b(\w+)\s+means?\b/gi;
    const definitionTerms: string[] = [];
    let match;
    while ((match = definitionPattern.exec(text)) !== null) {
      const term = match[1] || match[2] || match[3];
      if (term && term.length > 2) {
        definitionTerms.push(term);
      }
    }

    // Extract obligations (shall, must, will)
    const obligationPattern = /\b(?:shall|must|will)\s+([^.!?]+)/gi;
    const obligations: string[] = [];
    while ((match = obligationPattern.exec(text)) !== null) {
      obligations.push(match[1].trim());
    }

    // Extract conditions (if, unless, provided that)
    const conditionPattern = /\b(?:if|unless|provided that)\s+([^.!?]+)/gi;
    const conditions: string[] = [];
    while ((match = conditionPattern.exec(text)) !== null) {
      conditions.push(match[1].trim());
    }

    return {
      sections,
      hierarchy: sections, // Simplified
      crossReferences: this.extractCrossReferences(text),
      definitionTerms: Array.from(new Set(definitionTerms)),
      obligations: Array.from(new Set(obligations)),
      conditions: Array.from(new Set(conditions))
    };
  }

  private analyzeWritingStyle(text: string, sentences: SentenceAnalysis[]): WritingStyleAnalysis {
    const words = text.match(/\b\w+\b/g) || [];
    const totalWords = words.length;
    
    // Calculate metrics
    const averageSentenceLength = totalWords / sentences.length;
    const passiveVoiceCount = sentences.filter(s => 
      /\b(?:was|were|is|are|been|being)\s+\w+ed\b/.test(s.text)
    ).length;
    const passiveVoiceRatio = passiveVoiceCount / sentences.length;
    
    const legalTermCount = words.filter(word => 
      this.LEGAL_VOCABULARY.includes(word.toLowerCase())
    ).length;
    const legalJargonDensity = legalTermCount / totalWords;
    
    const complexityScore = sentences.reduce((sum, s) => sum + s.complexity, 0) / sentences.length;
    
    return {
      formalityScore: Math.min(legalJargonDensity * 5, 1.0),
      complexityScore,
      passiveVoiceRatio,
      averageSentenceLength,
      legalJargonDensity,
      readabilityScore: Math.max(0, 1 - (complexityScore * 0.5) - (averageSentenceLength / 50))
    };
  }

  private extractKeyPhrases(text: string, concepts: ExtractedConcept[]): string[] {
    // Combine high-importance concepts with n-gram extraction
    const conceptPhrases = concepts
      .filter(c => c.importance > 0.7)
      .map(c => c.concept);

    // Extract 2-3 gram phrases
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const bigrams: string[] = [];
    const trigrams: string[] = [];

    for (let i = 0; i < words.length - 1; i++) {
      bigrams.push(`${words[i]} ${words[i + 1]}`);
    }

    for (let i = 0; i < words.length - 2; i++) {
      trigrams.push(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
    }

    // Score and filter n-grams
    const scoredPhrases = [...bigrams, ...trigrams]
      .map(phrase => ({
        phrase,
        score: this.scorePhraseImportance(phrase, text)
      }))
      .filter(p => p.score > 0.3)
      .sort((a, b) => b.score - a.score)
      .map(p => p.phrase);

    // Combine and deduplicate
    const allPhrases = [...conceptPhrases, ...scoredPhrases.slice(0, 20)];
    return Array.from(new Set(allPhrases)).slice(0, 30);
  }

  private estimateAccuracy(concepts: ExtractedConcept[]): number {
    // Estimate model accuracy based on concept confidence distribution
    if (concepts.length === 0) return 0.5;
    
    const averageImportance = concepts.reduce((sum, c) => sum + c.importance, 0) / concepts.length;
    const conceptVariance = concepts.reduce((sum, c) => 
      sum + Math.pow(c.importance - averageImportance, 2), 0
    ) / concepts.length;
    
    // Higher average importance and lower variance indicate better accuracy
    return Math.min(averageImportance * (1 - conceptVariance), 1.0);
  }

  // Additional helper methods for completeness
  private LEGAL_VOCABULARY = [
    'contract', 'agreement', 'party', 'obligation', 'liability', 'indemnification',
    'breach', 'termination', 'warranty', 'representation', 'covenant', 'jurisdiction',
    'governing law', 'dispute resolution', 'arbitration', 'mediation', 'damages',
    'remedy', 'compliance', 'confidentiality', 'intellectual property'
  ];

  private calculateSentenceComplexity(sentence: string): number {
    const words = sentence.split(/\s+/).length;
    const characters = sentence.length;
    const avgWordLength = characters / words;
    
    return Math.min((words / 20) + (avgWordLength / 10), 1.0);
  }

  private calculateLegalWeight(sentence: string): number {
    const legalTermCount = this.LEGAL_VOCABULARY.filter(term =>
      sentence.toLowerCase().includes(term)
    ).length;
    
    return Math.min(legalTermCount / 5, 1.0);
  }

  private calculateSentiment(sentence: string): number {
    // Simplified legal sentiment - focus on obligation vs benefit
    const obligationTerms = ['shall', 'must', 'required', 'obligated', 'liable'];
    const benefitTerms = ['may', 'entitled', 'benefit', 'right', 'privilege'];
    
    let obligationCount = 0;
    let benefitCount = 0;
    
    obligationTerms.forEach(term => {
      if (sentence.toLowerCase().includes(term)) obligationCount++;
    });
    
    benefitTerms.forEach(term => {
      if (sentence.toLowerCase().includes(term)) benefitCount++;
    });
    
    if (obligationCount > benefitCount) return -0.5;
    if (benefitCount > obligationCount) return 0.5;
    return 0;
  }

  private determineSyntacticRole(sentence: string): SentenceAnalysis['syntacticRole'] {
    const lower = sentence.toLowerCase();
    
    if (lower.includes('means') || lower.includes('defined as') || lower.includes('shall mean')) {
      return 'definition';
    }
    if (lower.includes('shall') || lower.includes('must') || lower.includes('required')) {
      return 'obligation';
    }
    if (lower.includes('if') || lower.includes('unless') || lower.includes('provided')) {
      return 'condition';
    }
    if (lower.includes('except') || lower.includes('excluding') || lower.includes('other than')) {
      return 'exception';
    }
    
    return 'general';
  }

  private extractMainTopic(paragraph: string): string {
    // Extract most frequent meaningful words
    const words = paragraph.toLowerCase().match(/\b\w{4,}\b/g) || [];
    const frequency: Record<string, number> = {};
    
    words.forEach(word => {
      if (!['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were'].includes(word)) {
        frequency[word] = (frequency[word] || 0) + 1;
      }
    });
    
    const topWord = Object.entries(frequency)
      .sort(([, a], [, b]) => b - a)[0];
    
    return topWord ? topWord[0] : 'general';
  }

  private determineLegalFunction(paragraph: string): ParagraphAnalysis['legalFunction'] {
    const lower = paragraph.toLowerCase();
    
    if (lower.includes('whereas') || lower.includes('recital')) return 'preamble';
    if (lower.includes('means') || lower.includes('definition')) return 'definitions';
    if (lower.includes('shall') || lower.includes('obligation')) return 'obligations';
    if (lower.includes('condition') || lower.includes('provided')) return 'conditions';
    if (lower.includes('remedy') || lower.includes('damages')) return 'remedies';
    
    return 'general';
  }

  private calculateCoherence(sentences: SentenceAnalysis[]): number {
    if (sentences.length < 2) return 1.0;
    
    // Calculate average semantic similarity between adjacent sentences
    // Simplified version using shared vocabulary
    let totalSimilarity = 0;
    
    for (let i = 0; i < sentences.length - 1; i++) {
      const words1 = new Set(sentences[i].text.toLowerCase().match(/\b\w+\b/g) || []);
      const words2 = new Set(sentences[i + 1].text.toLowerCase().match(/\b\w+\b/g) || []);
      
      const intersection = new Set([...words1].filter(x => words2.has(x)));
      const union = new Set([...words1, ...words2]);
      
      const similarity = intersection.size / union.size;
      totalSimilarity += similarity;
    }
    
    return totalSimilarity / (sentences.length - 1);
  }

  private extractCrossReferences(text: string): string[] {
    const refPattern = /(?:Section|Article|Part|Chapter|Clause)\s+[\dIVX]+(?:\.\d+)?/gi;
    const matches = text.match(refPattern) || [];
    return Array.from(new Set(matches));
  }

  private scorePhraseImportance(phrase: string, text: string): number {
    const frequency = (text.toLowerCase().match(new RegExp(phrase, 'g')) || []).length;
    const isLegalTerm = this.LEGAL_VOCABULARY.some(term => phrase.includes(term));
    
    let score = frequency * 0.1;
    if (isLegalTerm) score *= 2;
    if (phrase.length > 15) score *= 1.2; // Longer phrases may be more specific
    
    return Math.min(score, 1.0);
  }

  /**
   * Get health status
   */
  getHealthStatus(): {
    initialized: boolean;
    useModelLoaded: boolean;
    tfBackend: string;
    memoryUsage: { numTensors: number; numBytes: number };
    cacheSize: number;
  } {
    return {
      initialized: this.initialized,
      useModelLoaded: !!this.useModel,
      tfBackend: tf.getBackend(),
      memoryUsage: tf.memory(),
      cacheSize: this.conceptCache.size + this.embeddingCache.size
    };
  }

  /**
   * Clear caches
   */
  clearCache(): void {
    this.conceptCache.clear();
    this.embeddingCache.clear();
    console.log('[LangExtract] Cache cleared');
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.clearCache();
    
    if (this.useModel) {
      // Universal Sentence Encoder cleanup is handled automatically
      this.useModel = null;
    }
    
    tf.disposeVariables();
    this.initialized = false;
    console.log('[LangExtract] Resources disposed');
  }
}

// Export singleton instance
export const langExtractTensorFlow = new LangExtractTensorFlow();