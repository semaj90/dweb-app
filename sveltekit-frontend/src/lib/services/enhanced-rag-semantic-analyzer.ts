/**
 * Enhanced RAG Semantic Analyzer with Context7 Integration
 * Provides advanced semantic analysis, entity extraction, and concept mapping for legal documents
 */

import { writable, type Writable } from 'svelte/store';

export interface SemanticEntity {
  text: string;
  type:
    | 'PERSON'
    | 'ORGANIZATION'
    | 'LOCATION'
    | 'DATE'
    | 'MONEY'
    | 'LEGAL_CONCEPT'
    | 'CASE_REF'
    | 'STATUTE'
    | 'CONTRACT_TERM';
  confidence: number;
  position: { start: number; end: number };
  metadata?: Record<string, any>;
}

export interface ConceptMapping {
  concept: string;
  relatedConcepts: string[];
  confidenceScore: number;
  semanticCluster: string;
  legalCategory: 'CONTRACT' | 'TORT' | 'CRIMINAL' | 'CONSTITUTIONAL' | 'CORPORATE' | 'PROPERTY';
}

export interface SemanticAnalysisResult {
  id: string;
  documentId: string;
  entities: SemanticEntity[];
  concepts: ConceptMapping[];
  summaryEmbedding: number[];
  sentimentScore: number;
  complexityIndex: number;
  legalRelevanceScore: number;
  timestamp: Date;
  processingTime: number;
}

export interface RAGQuery {
  query: string;
  context?: string;
  filters?: {
    entityTypes?: string[];
    legalCategories?: string[];
    dateRange?: { start: Date; end: Date };
    confidenceThreshold?: number;
  };
  semantic?: {
    useEmbeddings?: boolean;
    expandConcepts?: boolean;
    includeRelated?: boolean;
  };
}

export interface RAGResponse {
  query: string;
  results: Array<{
    documentId: string;
    title: string;
    relevanceScore: number;
    excerpt: string;
    entities: SemanticEntity[];
    concepts: ConceptMapping[];
    metadata: Record<string, any>;
  }>;
  totalFound: number;
  semanticExpansions?: string[];
  processingTime: number;
  timestamp: Date;
}

class EnhancedRAGSemanticAnalyzer {
  private baseUrl: string = 'http://localhost:8094';
  private context7Url: string = 'http://localhost:40000';
  private qdrantUrl: string = 'http://localhost:6333';
  private ollamaUrl: string = 'http://localhost:11434';

  // Legal concept mappings for enhanced semantic understanding
  private legalConceptMap = new Map<string, ConceptMapping>([
    [
      'breach of contract',
      {
        concept: 'breach of contract',
        relatedConcepts: [
          'damages',
          'specific performance',
          'remedies',
          'consideration',
          'offer and acceptance',
        ],
        confidenceScore: 0.95,
        semanticCluster: 'contract_law',
        legalCategory: 'CONTRACT',
      },
    ],
    [
      'negligence',
      {
        concept: 'negligence',
        relatedConcepts: ['duty of care', 'causation', 'damages', 'reasonable person standard'],
        confidenceScore: 0.92,
        semanticCluster: 'tort_law',
        legalCategory: 'TORT',
      },
    ],
    [
      'liability',
      {
        concept: 'liability',
        relatedConcepts: ['negligence', 'damages', 'causation', 'duty', 'standard of care'],
        confidenceScore: 0.88,
        semanticCluster: 'tort_law',
        legalCategory: 'TORT',
      },
    ],
  ]);

  // Named Entity Recognition patterns for legal documents
  private entityPatterns = new Map<SemanticEntity['type'], RegExp[]>([
    [
      'PERSON',
      [
        /\b([A-Z][a-z]+ [A-Z][a-z]+(?:,? (?:Jr|Sr|III?|Esq))?)\b/g,
        /\bDefendant (\w+)\b/gi,
        /\bPlaintiff (\w+)\b/gi,
      ],
    ],
    ['ORGANIZATION', [/\b([A-Z][A-Za-z\s]+ (?:Inc|Corp|LLC|Ltd|Co)\b)/g, /\b(The .+ Company)\b/g]],
    [
      'CASE_REF',
      [
        /\b(\w+\s+v\.?\s+\w+),?\s+(\d+\s+\w+\.?\s+\d+)/g,
        /\b(\d+\s+U\.S\.?\s+\d+)/g,
        /\b(\d+\s+F\.\d?d\s+\d+)/g,
      ],
    ],
    ['STATUTE', [/\b(\d+\s+U\.S\.C\.?\s+§?\s*\d+)/g, /\bSection\s+(\d+(?:\.\d+)*)/gi]],
    ['MONEY', [/\$[\d,]+(?:\.\d{2})?/g, /\b(\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?)\b/gi]],
    [
      'DATE',
      [
        /\b(\d{1,2}\/\d{1,2}\/\d{4})\b/g,
        /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/gi,
      ],
    ],
  ]);

  /**
   * Perform comprehensive semantic analysis on legal document text
   */
  async analyzeDocument(text: string, documentId: string): Promise<SemanticAnalysisResult> {
    const startTime = performance.now();

    try {
      // Extract entities using NER
      const entities = await this.extractEntities(text);

      // Map legal concepts
      const concepts = await this.mapConcepts(text);

      // Generate embeddings
      const summaryEmbedding = await this.generateEmbeddings(text);

      // Calculate scores
      const sentimentScore = await this.analyzeSentiment(text);
      const complexityIndex = this.calculateComplexity(text);
      const legalRelevanceScore = this.calculateLegalRelevance(entities, concepts);

      const processingTime = performance.now() - startTime;

      const result: SemanticAnalysisResult = {
        id: `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        documentId,
        entities,
        concepts,
        summaryEmbedding,
        sentimentScore,
        complexityIndex,
        legalRelevanceScore,
        timestamp: new Date(),
        processingTime,
      };

      // Store in Qdrant for vector similarity search
      await this.storeInVectorDB(result);

      return result;
    } catch (error) {
      console.error('Semantic analysis failed:', error);
      throw new Error(
        `Semantic analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Perform enhanced RAG query with semantic expansion
   */
  async enhancedQuery(query: RAGQuery): Promise<RAGResponse> {
    const startTime = performance.now();

    try {
      // Expand query semantically if requested
      let expandedQuery = query.query;
      let semanticExpansions: string[] = [];

      if (query.semantic?.expandConcepts) {
        const expansionResult = await this.expandQueryConcepts(query.query);
        expandedQuery = expansionResult.expandedQuery;
        semanticExpansions = expansionResult.expansions;
      }

      // Generate query embedding
      const queryEmbedding = await this.generateEmbeddings(expandedQuery);

      // Perform vector similarity search
      const vectorResults = await this.searchVectorDB(queryEmbedding, query.filters);

      // Perform traditional keyword search
      const keywordResults = await this.searchKeywords(expandedQuery, query.filters);

      // Merge and rank results
      const mergedResults = await this.mergeAndRankResults(vectorResults, keywordResults, query);

      const processingTime = performance.now() - startTime;

      return {
        query: query.query,
        results: mergedResults,
        totalFound: mergedResults.length,
        semanticExpansions,
        processingTime,
        timestamp: new Date(),
      };
    } catch (error) {
      console.error('Enhanced RAG query failed:', error);
      throw new Error(
        `Enhanced RAG query failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Extract named entities from legal text using pattern matching and ML
   */
  private async extractEntities(text: string): Promise<SemanticEntity[]> {
    const entities: SemanticEntity[] = [];

    // Pattern-based extraction
    for (const [entityType, patterns] of this.entityPatterns.entries()) {
      for (const pattern of patterns) {
        let match;
        while ((match = pattern.exec(text)) !== null) {
          entities.push({
            text: match[0],
            type: entityType,
            confidence: 0.85, // Pattern-based confidence
            position: {
              start: match.index,
              end: match.index + match[0].length,
            },
          });
        }
      }
    }

    // Enhanced ML-based entity extraction via Ollama
    try {
      const mlEntities = await this.extractEntitiesML(text);
      entities.push(...mlEntities);
    } catch (error) {
      console.warn('ML entity extraction failed:', error);
    }

    // Deduplicate and sort by position
    return this.deduplicateEntities(entities);
  }

  /**
   * ML-based entity extraction using Ollama
   */
  private async extractEntitiesML(text: string): Promise<SemanticEntity[]> {
    const prompt = `Extract named entities from this legal text. Return JSON array with format: [{"text": "entity", "type": "PERSON|ORGANIZATION|LOCATION|LEGAL_CONCEPT", "confidence": 0.9}]\n\nText: ${text.substring(0, 1000)}`;

    const response = await fetch(`${this.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma2:2b',
        prompt,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama request failed: ${response.status}`);
    }

    const result = await response.json();

    try {
      const entities = JSON.parse(result.response);
      return entities.map((entity: any, index: number) => ({
        ...entity,
        position: { start: 0, end: entity.text.length }, // Simplified positioning
        metadata: { source: 'ml' },
      }));
    } catch {
      return []; // Return empty array if JSON parsing fails
    }
  }

  /**
   * Map legal concepts using semantic understanding
   */
  private async mapConcepts(text: string): Promise<ConceptMapping[]> {
    const concepts: ConceptMapping[] = [];
    const textLower = text.toLowerCase();

    // Direct concept mapping
    for (const [conceptText, mapping] of this.legalConceptMap.entries()) {
      if (textLower.includes(conceptText)) {
        concepts.push({
          ...mapping,
          confidenceScore:
            (mapping.confidenceScore * (textLower.split(conceptText).length - 1)) / 10, // Adjust based on frequency
        });
      }
    }

    // Semantic concept extraction via Context7
    try {
      const context7Concepts = await this.extractContext7Concepts(text);
      concepts.push(...context7Concepts);
    } catch (error) {
      console.warn('Context7 concept extraction failed:', error);
    }

    return concepts;
  }

  /**
   * Context7-powered concept extraction
   */
  private async extractContext7Concepts(text: string): Promise<ConceptMapping[]> {
    const response = await fetch(`${this.context7Url}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error('Context7 not available');
    }

    // Placeholder for Context7 integration - would make actual calls to analyze concepts
    return [];
  }

  /**
   * Generate embeddings using local embedding model
   */
  private async generateEmbeddings(text: string): Promise<number[]> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: text.substring(0, 2000), // Limit text length
        }),
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.status}`);
      }

      const result = await response.json();
      return result.embedding || [];
    } catch (error) {
      console.warn('Embedding generation failed, using fallback:', error);
      // Return a mock embedding for fallback
      return new Array(384).fill(0).map(() => Math.random() * 0.1);
    }
  }

  /**
   * Analyze sentiment of legal text
   */
  private async analyzeSentiment(text: string): Promise<number> {
    // Simplified sentiment analysis for legal documents
    // In production, this would use a legal-domain-specific sentiment model

    const positiveWords = ['agree', 'successful', 'valid', 'approved', 'granted', 'favor'];
    const negativeWords = ['breach', 'violation', 'denied', 'failed', 'liable', 'damages'];

    const words = text.toLowerCase().split(/\W+/);
    const positiveCount = words.filter((word) => positiveWords.includes(word)).length;
    const negativeCount = words.filter((word) => negativeWords.includes(word)).length;

    const totalSentimentWords = positiveCount + negativeCount;
    if (totalSentimentWords === 0) return 0; // Neutral

    return (positiveCount - negativeCount) / totalSentimentWords;
  }

  /**
   * Calculate document complexity index
   */
  private calculateComplexity(text: string): number {
    const sentences = text.split(/[.!?]+/).length;
    const words = text.split(/\W+/).length;
    const avgWordsPerSentence = words / sentences;

    // Legal complexity factors
    const legalTerms = ['whereas', 'heretofore', 'aforementioned', 'pursuant', 'notwithstanding'];
    const legalTermCount = legalTerms.reduce(
      (count, term) => count + (text.toLowerCase().split(term).length - 1),
      0
    );

    // Normalize to 0-10 scale
    const complexityScore = Math.min(
      10,
      (avgWordsPerSentence / 15) * 5 + (legalTermCount / words) * 1000 * 5
    );

    return Math.round(complexityScore * 100) / 100;
  }

  /**
   * Calculate legal relevance score based on entities and concepts
   */
  private calculateLegalRelevance(entities: SemanticEntity[], concepts: ConceptMapping[]): number {
    const legalEntityTypes = ['LEGAL_CONCEPT', 'CASE_REF', 'STATUTE'];
    const legalEntitiesCount = entities.filter((e) => legalEntityTypes.includes(e.type)).length;

    const conceptScore = concepts.reduce((sum, concept) => sum + concept.confidenceScore, 0);

    return Math.min(1.0, (legalEntitiesCount * 0.1 + conceptScore * 0.3) / (entities.length + 1));
  }

  /**
   * Store semantic analysis results in Qdrant vector database
   */
  private async storeInVectorDB(analysis: SemanticAnalysisResult): Promise<void> {
    try {
      const payload = {
        points: [
          {
            id: analysis.id,
            vector: analysis.summaryEmbedding,
            payload: {
              documentId: analysis.documentId,
              entities: analysis.entities,
              concepts: analysis.concepts,
              sentimentScore: analysis.sentimentScore,
              complexityIndex: analysis.complexityIndex,
              legalRelevanceScore: analysis.legalRelevanceScore,
              timestamp: analysis.timestamp.toISOString(),
            },
          },
        ],
      };

      const response = await fetch(`${this.qdrantUrl}/collections/legal_documents/points`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Qdrant storage failed: ${response.status}`);
      }
    } catch (error) {
      console.warn('Vector DB storage failed:', error);
      // Continue without failing - vector storage is optional
    }
  }

  /**
   * Search vector database for similar documents
   */
  private async searchVectorDB(
    queryEmbedding: number[],
    filters?: RAGQuery['filters']
  ): Promise<any[]> {
    try {
      const searchPayload = {
        vector: queryEmbedding,
        limit: 20,
        with_payload: true,
        score_threshold: filters?.confidenceThreshold || 0.7,
      };

      const response = await fetch(`${this.qdrantUrl}/collections/legal_documents/points/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(searchPayload),
      });

      if (!response.ok) {
        throw new Error(`Qdrant search failed: ${response.status}`);
      }

      const result = await response.json();
      return result.result || [];
    } catch (error) {
      console.warn('Vector search failed:', error);
      return [];
    }
  }

  /**
   * Expand query with related legal concepts
   */
  private async expandQueryConcepts(
    query: string
  ): Promise<{ expandedQuery: string; expansions: string[] }> {
    const queryLower = query.toLowerCase();
    const expansions: string[] = [];
    let expandedQuery = query;

    // Find related concepts
    for (const [concept, mapping] of this.legalConceptMap.entries()) {
      if (queryLower.includes(concept)) {
        expansions.push(...mapping.relatedConcepts);
      }
    }

    // Add expansions to query
    if (expansions.length > 0) {
      expandedQuery = `${query} ${expansions.join(' ')}`;
    }

    return { expandedQuery, expansions: [...new Set(expansions)] }; // Remove duplicates
  }

  /**
   * Deduplicate entities by position and text similarity
   */
  private deduplicateEntities(entities: SemanticEntity[]): SemanticEntity[] {
    const unique = new Map<string, SemanticEntity>();

    for (const entity of entities) {
      const key = `${entity.text.toLowerCase()}_${entity.type}`;
      const existing = unique.get(key);

      if (!existing || entity.confidence > existing.confidence) {
        unique.set(key, entity);
      }
    }

    return Array.from(unique.values()).sort((a, b) => a.position.start - b.position.start);
  }

  /**
   * Placeholder methods for traditional keyword search and result merging
   */
  private async searchKeywords(query: string, filters?: RAGQuery['filters']): Promise<any[]> {
    // Implementation would perform traditional keyword search
    return [];
  }

  private async mergeAndRankResults(
    vectorResults: any[],
    keywordResults: any[],
    query: RAGQuery
  ): Promise<any[]> {
    // Implementation would merge and rank results from different sources
    return [];
  }
}

// Export singleton instance
export const semanticAnalyzer = new EnhancedRAGSemanticAnalyzer();

// Svelte stores for reactive semantic analysis state
export const semanticAnalysisStore: Writable<SemanticAnalysisResult | null> = writable(null);
export const ragQueryStore: Writable<RAGQuery | null> = writable(null);
export const ragResponseStore: Writable<RAGResponse | null> = writable(null);
export const isAnalyzingStore: Writable<boolean> = writable(false);
