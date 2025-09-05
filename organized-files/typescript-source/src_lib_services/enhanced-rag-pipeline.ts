// @ts-nocheck
import { db } from './unified-database-service.js';
import { aiService } from './unified-ai-service.js';

/**
 * Enhanced RAG Pipeline with Self-Organizing Loop System
 * Implements advanced retrieval, ranking, and recommendation features
 */
export class EnhancedRAGPipeline {
  private initialized: boolean = false;
  private config: any;
  private cache: Map<string, any> = new Map();
  private performanceMetrics: any = {};

  constructor(config: any = {}) {
    this.config = {
      // Embedding configuration
      embeddingModel: config.embeddingModel || 'nomic-embed-text',
      embeddingDimensions: config.embeddingDimensions || 384,
      
      // Retrieval configuration
      maxRetrievalResults: config.maxRetrievalResults || 20,
      defaultTopK: config.defaultTopK || 5,
      hybridWeight: config.hybridWeight || { text: 0.4, vector: 0.6 },
      
      // Chunking configuration
      chunkSize: config.chunkSize || 512,
      chunkOverlap: config.chunkOverlap || 50,
      
      // Performance configuration
      cacheEnabled: config.cacheEnabled ?? true,
      cacheTTL: config.cacheTTL || 3600,
      compressionEnabled: config.compressionEnabled ?? true,
      
      // Advanced features
      selfOrganizing: config.selfOrganizing ?? true,
      feedbackLoopEnabled: config.feedbackLoopEnabled ?? true,
      realTimeUpdates: config.realTimeUpdates ?? true,
      
      ...config
    };
  }

  async initialize(): Promise<boolean> {
    try {
      await db.initialize();
      await aiService.initialize();
      
      // Initialize vector collection if needed
      await this.initializeVectorCollection();
      
      // Start performance monitoring
      this.startPerformanceMonitoring();
      
      this.initialized = true;
      console.log('✓ Enhanced RAG Pipeline initialized');
      return true;
    } catch (error) {
      console.error('RAG Pipeline initialization failed:', error);
      return false;
    }
  }

  // ============ Document Ingestion ============
  async ingestDocuments(documents: any[]): Promise<any> {
    const results = {
      processed: 0,
      failed: 0,
      stored: 0,
      chunks: 0,
      embeddings: 0
    };

    for (const doc of documents) {
      try {
        // Process document with all operations
        const processed = await aiService.processDocument(doc, {
          operations: ['extract', 'chunk', 'embed', 'analyze', 'summarize', 'store']
        });

        // Create knowledge graph relationships
        if (this.config.selfOrganizing) {
          await this.createKnowledgeRelationships(processed);
        }

        results.processed++;
        results.stored += processed.stored ? 1 : 0;
        results.chunks += processed.chunks?.length || 0;
        results.embeddings += processed.embeddings?.length || 0;

      } catch (error) {
        console.error(`Failed to process document ${doc.id}:`, error);
        results.failed++;
      }
    }

    // Update performance metrics
    this.updateMetrics('ingestion', results);
    
    return results;
  }

  async createKnowledgeRelationships(document: any): Promise<void> {
    // Extract entities and create relationships in Neo4j
    if (document.analysis && document.analysis.entities) {
      // Create document node
      await db.createNode('Document', {
        id: document.id,
        title: document.title,
        type: document.analysis.documentType,
        createdAt: new Date().toISOString()
      });

      // Create entity nodes and relationships
      for (const entity of document.analysis.entities) {
        await db.createNode('Entity', {
          id: `entity_${entity.name}`,
          name: entity.name,
          type: entity.type,
          confidence: entity.confidence
        });

        await db.createRelationship(
          document.id,
          `entity_${entity.name}`,
          'MENTIONS',
          { confidence: entity.confidence }
        );
      }

      // Create case relationships
      if (document.caseId) {
        await db.createRelationship(document.id, document.caseId, 'BELONGS_TO');
      }
    }
  }

  // ============ Enhanced Query Processing ============
  async query(query: string, options: any = {}): Promise<any> {
    const startTime = performance.now();
    
    // Check cache first
    const cacheKey = this.generateCacheKey(query, options);
    if (this.config.cacheEnabled && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      this.updateMetrics('cache_hit');
      return { ...cached, cached: true };
    }

    // Phase 1: Multi-modal retrieval
    const retrievalResults = await this.performRetrievalPhase(query, options);
    
    // Phase 2: Intelligent ranking and fusion
    const rankedResults = await this.performRankingPhase(query, retrievalResults, options);
    
    // Phase 3: Context building and compression
    const contextData = await this.buildOptimizedContext(rankedResults, options);
    
    // Phase 4: Response generation with feedback loop
    const response = await this.generateEnhancedResponse(query, contextData, options);
    
    // Phase 5: Self-organizing feedback
    if (this.config.feedbackLoopEnabled) {
      await this.processFeedbackLoop(query, response, options);
    }

    const processingTime = performance.now() - startTime;
    
    const result = {
      ...response,
      metadata: {
        ...response.metadata,
        processingTime,
        phases: ['retrieval', 'ranking', 'context', 'generation', 'feedback'],
        cacheKey
      }
    };

    // Cache result
    if (this.config.cacheEnabled) {
      this.cache.set(cacheKey, result);
      setTimeout(() => this.cache.delete(cacheKey), this.config.cacheTTL * 1000);
    }

    this.updateMetrics('query_processed', { processingTime });
    
    return result;
  }

  private async performRetrievalPhase(query: string, options: any): Promise<any> {
    const retrievalMethods = [];

    // 1. Semantic vector search
    const queryEmbedding = await aiService.embedSingle(query);
    const vectorResults = db.vectorSearch(
      queryEmbedding,
      this.config.maxRetrievalResults,
      options.filters || {}
    );

    // 2. Full-text search
    const textResults = db.searchLegalDocuments(query, options.caseId);

    // 3. Knowledge graph traversal
    const graphResults = this.performGraphTraversal(query, options);

    // 4. Hybrid fuzzy matching
    const fuzzyResults = this.performFuzzySearch(query, options);

    const [vector, text, graph, fuzzy] = await Promise.all([
      vectorResults,
      textResults,
      graphResults,
      fuzzyResults
    ]);

    return {
      vector,
      text, 
      graph,
      fuzzy,
      queryEmbedding
    };
  }

  private async performGraphTraversal(query: string, options: any): Promise<any[]> {
    try {
      // Extract key terms from query
      const keyTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 3);
      
      // Traverse knowledge graph for related documents
      const graphQuery = `
        MATCH (d:Document)-[:MENTIONS]->(e:Entity)
        WHERE ANY(term IN $terms WHERE toLower(e.name) CONTAINS term)
        WITH d, e, count(*) as relevance
        ORDER BY relevance DESC
        LIMIT 10
        MATCH (d)-[:BELONGS_TO]->(c:Case)
        RETURN d, e, c, relevance
      `;
      
      const results = await db.runCypher(graphQuery, { terms: keyTerms });
      
      return results.map(record => ({
        document: record.get('d').properties,
        entity: record.get('e').properties,
        case: record.get('c')?.properties,
        relevance: record.get('relevance').toNumber(),
        source: 'graph'
      }));
    } catch (error) {
      console.error('Graph traversal error:', error);
      return [];
    }
  }

  private async performFuzzySearch(query: string, options: any): Promise<any[]> {
    // Implement fuzzy string matching for legal terms
    // This would integrate with a legal terminology database
    return [];
  }

  private async performRankingPhase(query: string, retrievalResults: any, options: any): Promise<any[]> {
    const allResults = new Map<string, any>();
    
    // Combine results from all retrieval methods
    this.combineRetrievalResults(allResults, retrievalResults.vector, 'vector');
    this.combineRetrievalResults(allResults, retrievalResults.text, 'text');
    this.combineRetrievalResults(allResults, retrievalResults.graph, 'graph');
    this.combineRetrievalResults(allResults, retrievalResults.fuzzy, 'fuzzy');

    // Apply advanced ranking algorithms
    const rankedResults = Array.from(allResults.values())
      .map(result => this.calculateAdvancedScore(result, query, retrievalResults.queryEmbedding))
      .sort((a, b) => b.finalScore - a.finalScore);

    // Apply diversity and novelty filters
    const diversifiedResults = this.applyDiversityFiltering(rankedResults, options);
    
    return diversifiedResults.slice(0, options.topK || this.config.defaultTopK);
  }

  private combineRetrievalResults(map: Map<string, any>, results: any[], source: string): void {
    results.forEach((result, index) => {
      const id = result.id || result.document?.id || `${source}_${index}`;
      
      if (map.has(id)) {
        const existing = map.get(id);
        existing.sources.push(source);
        existing.scores[source] = this.normalizeScore(result, source);
      } else {
        map.set(id, {
          ...result,
          id,
          sources: [source],
          scores: { [source]: this.normalizeScore(result, source) }
        });
      }
    });
  }

  private normalizeScore(result: any, source: string): number {
    switch (source) {
      case 'vector':
        return result.score || 0;
      case 'text':
        return result.rank || 0;
      case 'graph':
        return (result.relevance || 0) / 10; // Normalize graph relevance
      case 'fuzzy':
        return result.similarity || 0;
      default:
        return 0;
    }
  }

  private calculateAdvancedScore(result: any, query: string, queryEmbedding: number[]): any {
    const scores = result.scores;
    const weights = {
      vector: this.config.hybridWeight.vector,
      text: this.config.hybridWeight.text,
      graph: 0.2,
      fuzzy: 0.1
    };

    // Calculate weighted score
    let weightedScore = 0;
    let totalWeight = 0;

    for (const [source, score] of Object.entries(scores)) {
      const weight = weights[source as keyof typeof weights] || 0.1;
      weightedScore += score * weight;
      totalWeight += weight;
    }

    const baseScore = totalWeight > 0 ? weightedScore / totalWeight : 0;

    // Apply boosting factors
    const boostFactors = this.calculateBoostFactors(result, query);
    const finalScore = baseScore * boostFactors.total;

    return {
      ...result,
      baseScore,
      boostFactors,
      finalScore
    };
  }

  private calculateBoostFactors(result: any, query: string): any {
    const factors = {
      recency: 1.0,
      authority: 1.0,
      completeness: 1.0,
      relevance: 1.0,
      total: 1.0
    };

    // Recency boost - newer documents get higher scores
    if (result.createdAt || result.document?.createdAt) {
      const date = new Date(result.createdAt || result.document.createdAt);
      const daysSinceCreation = (Date.now() - date.getTime()) / (1000 * 60 * 60 * 24);
      factors.recency = Math.max(0.5, 1.0 - (daysSinceCreation / 365)); // Decay over year
    }

    // Authority boost - documents with more citations/references
    if (result.metadata?.citations) {
      factors.authority = 1.0 + (Math.log(result.metadata.citations + 1) * 0.1);
    }

    // Completeness boost - longer, more detailed documents
    if (result.content || result.document?.content) {
      const contentLength = (result.content || result.document.content).length;
      factors.completeness = 1.0 + Math.min(0.3, contentLength / 10000);
    }

    // Calculate total multiplier
    factors.total = factors.recency * factors.authority * factors.completeness * factors.relevance;

    return factors;
  }

  private applyDiversityFiltering(results: any[], options: any): any[] {
    const diversityThreshold = options.diversityThreshold || 0.8;
    const filtered = [];
    
    for (const candidate of results) {
      let tooSimilar = false;
      
      for (const existing of filtered) {
        const similarity = this.calculateContentSimilarity(candidate, existing);
        if (similarity > diversityThreshold) {
          tooSimilar = true;
          break;
        }
      }
      
      if (!tooSimilar) {
        filtered.push(candidate);
      }
    }
    
    return filtered;
  }

  private calculateContentSimilarity(doc1: any, doc2: any): number {
    // Implement content similarity calculation
    // This is a simplified version - in practice you'd use more sophisticated methods
    const content1 = doc1.content || doc1.document?.content || '';
    const content2 = doc2.content || doc2.document?.content || '';
    
    const words1 = new Set(content1.toLowerCase().split(/\s+/));
    const words2 = new Set(content2.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...words1].filter(word => words2.has(word)));
    const union = new Set([...words1, ...words2]);
    
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  private async buildOptimizedContext(results: any[], options: any): Promise<any> {
    const maxContextLength = options.maxContextLength || 4000;
    let currentLength = 0;
    const contextPieces = [];
    
    for (const result of results) {
      const content = result.content || result.document?.content || '';
      const piece = {
        id: result.id,
        content: content.substring(0, Math.min(content.length, maxContextLength - currentLength)),
        score: result.finalScore,
        source: result.sources,
        title: result.title || result.document?.title
      };
      
      contextPieces.push(piece);
      currentLength += piece.content.length;
      
      if (currentLength >= maxContextLength) break;
    }
    
    // Apply context compression if enabled
    if (this.config.compressionEnabled && contextPieces.length > 3) {
      return await this.compressContext(contextPieces, options);
    }
    
    return {
      pieces: contextPieces,
      totalLength: currentLength,
      compression: 'none'
    };
  }

  private async compressContext(contextPieces: any[], options: any): Promise<any> {
    // Implement intelligent context compression
    // This could use extractive summarization or key sentence extraction
    
    const compressedPieces = [];
    
    for (const piece of contextPieces) {
      // Extract key sentences using simple scoring
      const sentences = piece.content.match(/[^.!?]+[.!?]+/g) || [];
      const keySentences = sentences
        .map(sentence => ({
          sentence,
          score: this.scoreSentenceImportance(sentence, piece.score)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, Math.max(1, Math.floor(sentences.length / 3)))
        .map(item => item.sentence);
      
      compressedPieces.push({
        ...piece,
        content: keySentences.join(' '),
        compression: 'extractive'
      });
    }
    
    return {
      pieces: compressedPieces,
      totalLength: compressedPieces.reduce((sum, p) => sum + p.content.length, 0),
      compression: 'extractive',
      compressionRatio: compressedPieces.reduce((sum, p) => sum + p.content.length, 0) / 
                       contextPieces.reduce((sum, p) => sum + p.content.length, 0)
    };
  }

  private scoreSentenceImportance(sentence: string, baseScore: number): number {
    let score = baseScore;
    
    // Boost sentences with legal keywords
    const legalKeywords = ['defendant', 'plaintiff', 'court', 'evidence', 'witness', 'statute', 'precedent'];
    const keywordMatches = legalKeywords.filter(keyword => 
      sentence.toLowerCase().includes(keyword)
    ).length;
    score += keywordMatches * 0.1;
    
    // Boost sentences with numbers/dates (often important in legal contexts)
    const numberMatches = sentence.match(/\d+/g);
    if (numberMatches) {
      score += numberMatches.length * 0.05;
    }
    
    // Penalize very short sentences
    if (sentence.split(' ').length < 5) {
      score *= 0.8;
    }
    
    return score;
  }

  private async generateEnhancedResponse(query: string, contextData: any, options: any): Promise<any> {
    const enhancedPrompt = this.buildEnhancedPrompt(query, contextData, options);
    
    // Use streaming if requested
    if (options.stream) {
      return {
        stream: aiService.ragStream(query, {
          ...options,
          context: contextData
        }),
        sources: contextData.pieces,
        contextMetadata: contextData
      };
    }
    
    // Standard response generation
    const response = await aiService.ragQuery(query, {
      ...options,
      context: contextData
    });
    
    return {
      answer: response.answer,
      sources: contextData.pieces.map(piece => ({
        id: piece.id,
        title: piece.title,
        score: piece.score,
        snippet: piece.content.substring(0, 200) + '...',
        sources: piece.source
      })),
      contextMetadata: contextData,
      rawResponse: response
    };
  }

  private buildEnhancedPrompt(query: string, contextData: any, options: any): string {
    const context = contextData.pieces
      .map((piece, index) => `[${index + 1}] ${piece.title || 'Document'}: ${piece.content}`)
      .join('\n\n');

    return `
You are an expert legal AI assistant with access to a comprehensive legal knowledge base. 
Analyze the query and provide accurate, well-reasoned legal guidance based on the provided context.

Context Documents:
${context}

Query: ${query}

Instructions:
- Provide detailed, accurate legal analysis based on the context
- Cite specific sources using [1], [2], etc. format
- Include relevant legal precedents, statutes, or case law mentioned in the context
- If information is incomplete, clearly state what additional information would be needed
- Structure your response with clear headings and bullet points when appropriate
- Consider multiple perspectives and potential legal implications
- Maintain professional legal writing standards

${options.specialInstructions || ''}

Response:`;
  }

  private async processFeedbackLoop(query: string, response: any, options: any): Promise<void> {
    // Implement self-organizing feedback system
    try {
      // Store query-response pair for learning
      await db.setCached(`feedback:${this.generateCacheKey(query, options)}`, {
        query,
        response: response.answer,
        sources: response.sources,
        timestamp: Date.now(),
        performance: response.metadata?.processingTime
      }, 86400); // 24 hours

      // Update source quality scores based on usage
      for (const source of response.sources || []) {
        await this.updateSourceQualityScore(source.id, response.metadata?.processingTime);
      }

      // Learn from successful queries for future optimization
      if (this.config.selfOrganizing) {
        await this.updateQueryPatterns(query, response);
      }

    } catch (error) {
      console.error('Feedback loop error:', error);
    }
  }

  private async updateSourceQualityScore(sourceId: string, processingTime?: number): Promise<void> {
    const key = `source_quality:${sourceId}`;
    const existing = await db.getCached(key) || { score: 0.5, usageCount: 0 };
    
    // Update based on successful usage (simple approach)
    const newScore = existing.score * 0.9 + 0.1; // Slight boost for being used
    
    await db.setCached(key, {
      score: Math.min(1.0, newScore),
      usageCount: existing.usageCount + 1,
      lastUsed: Date.now(),
      avgProcessingTime: processingTime
    }, 86400 * 7); // Keep for a week
  }

  private async updateQueryPatterns(query: string, response: any): Promise<void> {
    // Implement query pattern learning for optimization
    const key = `query_patterns:${this.hashString(query)}`;
    const pattern = {
      query,
      responseTime: response.metadata?.processingTime,
      sourcesUsed: response.sources?.length || 0,
      success: true, // Could be determined by user feedback
      timestamp: Date.now()
    };
    
    await db.setCached(key, pattern, 86400 * 7);
  }

  // ============ Recommendation Engine ============
  async getRecommendations(userId: string, context: any = {}): Promise<any[]> {
    try {
      // Get user's query history
      const userHistory = await this.getUserQueryHistory(userId);
      
      // Build user profile from history
      const userProfile = await this.buildUserProfile(userHistory);
      
      // Generate recommendations based on profile and current context
      const recommendations = await this.generateRecommendations(userProfile, context);
      
      return recommendations;
    } catch (error) {
      console.error('Recommendation error:', error);
      return [];
    }
  }

  private async getUserQueryHistory(userId: string): Promise<any[]> {
    // Get recent queries for the user
    const pattern = `user_query:${userId}:*`;
    // This would need to be implemented based on your user tracking system
    return [];
  }

  private async buildUserProfile(history: any[]): Promise<any> {
    // Build user interest profile from query history
    return {
      interests: [],
      expertise_level: 'intermediate',
      frequent_topics: [],
      preferred_response_style: 'detailed'
    };
  }

  private async generateRecommendations(profile: any, context: any): Promise<any[]> {
    // Generate personalized recommendations
    return [];
  }

  // ============ Performance Monitoring ============
  private startPerformanceMonitoring(): void {
    setInterval(() => {
      this.logPerformanceMetrics();
    }, 60000); // Log every minute
  }

  private updateMetrics(operation: string, data?: any): void {
    if (!this.performanceMetrics[operation]) {
      this.performanceMetrics[operation] = {
        count: 0,
        totalTime: 0,
        avgTime: 0,
        lastOperation: null
      };
    }
    
    const metric = this.performanceMetrics[operation];
    metric.count++;
    metric.lastOperation = Date.now();
    
    if (data?.processingTime) {
      metric.totalTime += data.processingTime;
      metric.avgTime = metric.totalTime / metric.count;
    }
  }

  private logPerformanceMetrics(): void {
    console.log('RAG Pipeline Performance Metrics:', {
      timestamp: new Date().toISOString(),
      cacheSize: this.cache.size,
      operations: this.performanceMetrics
    });
  }

  // ============ Utility Methods ============
  private async initializeVectorCollection(): Promise<void> {
    try {
      const response = await fetch(`${process.env.QDRANT_URL || 'http://localhost:6333'}/collections/legal_documents`, {
        method: 'GET'
      });
      
      if (!response.ok) {
        // Create collection if it doesn't exist
        await fetch(`${process.env.QDRANT_URL || 'http://localhost:6333'}/collections/legal_documents`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            vectors: {
              size: this.config.embeddingDimensions,
              distance: 'Cosine'
            }
          })
        });
        console.log('✓ Qdrant collection created');
      }
    } catch (error) {
      console.warn('Qdrant initialization warning:', error.message);
    }
  }

  private generateCacheKey(query: string, options: any): string {
    const keyData = {
      query: query.toLowerCase().trim(),
      caseId: options.caseId,
      topK: options.topK || this.config.defaultTopK,
      filters: options.filters
    };
    return this.hashString(JSON.stringify(keyData));
  }

  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  async getHealthStatus(): Promise<any> {
    return {
      initialized: this.initialized,
      cacheSize: this.cache.size,
      performance: this.performanceMetrics,
      config: {
        embeddingModel: this.config.embeddingModel,
        defaultTopK: this.config.defaultTopK,
        cacheEnabled: this.config.cacheEnabled,
        selfOrganizing: this.config.selfOrganizing
      }
    };
  }
}

// Export singleton instance
export const ragPipeline = new EnhancedRAGPipeline();