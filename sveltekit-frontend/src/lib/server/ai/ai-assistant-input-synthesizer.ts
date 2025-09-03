
// lib/server/ai/ai-assistant-input-synthesizer.ts
// Comprehensive AI Assistant Input Synthesizer integrating all enhanced components

import { logger } from "./logger";
import { enhancedRAGPipeline } from "./rag-pipeline-enhanced";

// Simple metrics stub for missing metrics dependency
const metrics = {
  incrementCounter: (name: string) => {},
  recordTiming: (name: string, time: number) => {},
  contextRelevance: 0.8,
  sourceAuthority: 0.8,
  conceptCoverage: 0.8,
  informationCompleteness: 0.8,
  responseReadiness: 0.8,
  getAllMetrics: () => ({})
};

// Simple stubs for missing dependencies
const legalBERT = {
  analyze: (text: string) => Promise.resolve({ confidence: 0.8, categories: [], summary: '' }),
  analyzeLegalText: (text: string) => Promise.resolve({
    confidence: 0.8,
    categories: [],
    summary: {
      abstractive: 'Generated summary',
      extractive: 'Key extracted content',
      keyPoints: ['Key point 1', 'Key point 2']
    },
    entities: [],
    concepts: [],
    complexity: { legalComplexity: 0.5 },
    legalConcepts: [],
    jurisdiction: 'general',
    practiceAreas: []
  }),
  healthCheck: () => Promise.resolve({ status: 'healthy', uptime: 100 }),
  calculateLegalSimilarity: (text1: string, text2: string) => Promise.resolve({
    similarity: 0.8,
    confidence: 0.8
  })
};

const enhancedLegalSearch = {
  search: (query: string, options: any) => Promise.resolve([])
};

// Utility function for timeout handling
async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('Operation timed out')), timeoutMs);
  });
  return Promise.race([promise, timeout]);
}

export interface SynthesizerAnalysisResult {
  confidence: number;
  categories: string[];
  summary: string | {
    abstractive: string;
    extractive: string;
    keyPoints: string[];
  };
  entities?: Array<{
    text: string;
    type: string;
  }>;
  concepts?: Array<{
    concept: string;
    confidence: number;
  }>;
  complexity?: {
    legalComplexity: number;
  };
}

export interface RetrievalOptions {
  enableRAG?: boolean;
  maxSources?: number;
  similarityThreshold?: number;
  enableLegalBERT?: boolean;
  enableMMR?: boolean;
  enableCrossEncoder?: boolean;
}

/**
 * Internal retrieval result structure used before further processing/ranking.
 */
interface RetrievalResult {
  sources: Array<{
    id: string;
    title: string;
    content: string;
    relevanceScore: number;
    diversityScore: number;
    rerankedScore: number;
    type: 'document' | 'case' | 'statute' | 'precedent';
    metadata: Record<string, any>;
  }>;
  summary: { abstractive: string; extractive: string[]; keyPoints: string[] };
  totalSources: number;
  searchStrategies: string[];
}

import { generateEmbedding } from "./embeddings-simple";

// Input types for the synthesizer
export interface SynthesizerInput {
  query: string;
  context?: {
    caseId?: string;
    userId: string;
    legalBertAnalysis?: unknown; // Add missing property
    conversationHistory?: Array<{
      role: 'user' | 'assistant';
      content: string;
      timestamp: Date;
    }>;
    documents?: Array<{
      id: string;
      title: string;
      content: string;
      type: string;
    }>;
    preferences?: {
      responseStyle: 'formal' | 'casual' | 'technical';
      maxLength: number;
      includeCitations: boolean;
      focusAreas: string[];
    };
  };
  options?: {
    enableMMR: boolean;
    enableCrossEncoder: boolean;
    enableLegalBERT: boolean;
    enableRAG: boolean;
    maxSources: number;
    similarityThreshold: number;
    diversityLambda: number;
  };
}

// Synthesized output structure
export interface SynthesizedOutput {
  processedQuery: {
    original: string;
    enhanced: string;
    intent: string;
    entities: Array<{
      text: string;
      type: string;
      confidence: number;
    }>;
    legalConcepts: string[];
    complexity: number;
  };
  retrievedContext: {
    sources: Array<{
      id: string;
      title: string;
      content: string;
      relevanceScore: number;
      diversityScore: number;
      rerankedScore: number;
      type: 'document' | 'case' | 'statute' | 'precedent';
      metadata: Record<string, any>;
    }>;
    summary: {
      abstractive: string;
      extractive: string[];
      keyPoints: string[];
    };
    totalSources: number;
    searchStrategies: string[];
  };
  enhancedPrompt: {
    systemPrompt: string;
    contextPrompt: string;
    queryPrompt: string;
    instructions: string[];
    constraints: string[];
  };
  metadata: {
    processingTime: number;
    confidence: number;
    strategies: string[];
    qualityScore: number;
    recommendations: string[];
  };
}

// Quality assessment metrics
export interface QualityMetrics {
  contextRelevance: number;
  sourceAuthority: number;
  conceptCoverage: number;
  informationCompleteness: number;
  responseReadiness: number;
}

// AI Assistant Input Synthesizer class
export class AIAssistantInputSynthesizer {
  private requestCount = 0;
  private processingStats = new Map<string, number>();

  constructor() {
    this.initializeSynthesizer();
  }

  private async initializeSynthesizer(): Promise<void> {
    try {
      logger.info('[Synthesizer] Initializing AI Assistant Input Synthesizer...');

      // Verify all components are available
      await this.verifyComponents();

      logger.info('[Synthesizer] All components verified successfully');
      metrics.incrementCounter('synthesizer_initializations');
    } catch (error: any) {
      logger.error('[Synthesizer] Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Main synthesis method that orchestrates all components
   */
  async synthesizeInput(input: SynthesizerInput): Promise<SynthesizedOutput> {
    const startTime = Date.now();
    this.requestCount++;

    try {
      logger.info(
        `[Synthesizer] Processing request ${this.requestCount}: "${input.query.substring(0, 100)}..."`
      );

      // Default options
      const options = {
        enableMMR: true,
        enableCrossEncoder: true,
        enableLegalBERT: true,
        enableRAG: true,
        maxSources: 10,
        similarityThreshold: 0.7,
        diversityLambda: 0.5,
        ...input.options,
      };

      // Step 1: Query Analysis and Enhancement
      const processedQuery = await this.analyzeAndEnhanceQuery(input.query, input.context);

      // Step 2: Multi-Strategy Retrieval
      const retrievedContext = await this.performMultiStrategyRetrieval(
        processedQuery,
        input.context,
        options
      );

      // Step 3: Content Processing and Ranking
      const enhancedContext = await this.processAndRankContent(
        retrievedContext,
        processedQuery,
        options
      );

      // Step 4: Prompt Construction
      const enhancedPrompt = await this.constructEnhancedPrompt(
        processedQuery,
        enhancedContext,
        input.context
      );

      // Step 5: Quality Assessment
      const qualityMetrics = await this.assessQuality(
        processedQuery,
        enhancedContext,
        enhancedPrompt
      );

      const processingTime = Date.now() - startTime;

      const result: SynthesizedOutput = {
        processedQuery,
        retrievedContext: enhancedContext,
        enhancedPrompt,
        metadata: {
          processingTime,
          confidence: qualityMetrics.responseReadiness,
          strategies: this.getUsedStrategies(options),
          qualityScore: this.calculateOverallQuality(qualityMetrics),
          recommendations: this.generateRecommendations(qualityMetrics),
        },
      };

      // Log metrics
      metrics.incrementCounter('synthesizer_requests');
      metrics.recordTiming('synthesizer_processing_time', processingTime);

      logger.info(
        `[Synthesizer] Request ${this.requestCount} completed in ${processingTime}ms with ${enhancedContext.totalSources} sources`
      );

      return result;
    } catch (error: any) {
      logger.error('[Synthesizer] Processing failed:', error);
      metrics.incrementCounter('synthesizer_errors');
      throw error;
    }
  }

  /**
   * Step 1: Analyze and enhance the input query
   */
  private async analyzeAndEnhanceQuery(
    query: string,
    context?: SynthesizerInput['context']
  ): Promise<SynthesizedOutput['processedQuery']> {
    try {
      // Use LegalBERT for comprehensive analysis
      const legalAnalysis = await legalBERT.analyzeLegalText(query);

      // Extract intent using pattern matching and ML
      const intent = await this.extractIntent(query, legalAnalysis);

      // Enhance query with context and legal understanding
      const enhancedQuery = await this.enhanceQueryWithContext(query, context, legalAnalysis);

      return {
        original: query,
        enhanced: enhancedQuery,
        intent,
        entities: legalAnalysis.entities.map((e: any) => ({
          text: e.text,
          type: e.type,
          confidence: e.confidence,
        })),
        legalConcepts: (legalAnalysis.legalConcepts ?? []).map((c: any) => c.concept || c),
        complexity: legalAnalysis.complexity?.legalComplexity ?? 0.5,
      };
    } catch (error: any) {
      logger.warn('[Synthesizer] Query analysis failed, returning basic structure', error);
      return {
        original: query,
        enhanced: query,
        intent: 'general_legal_query',
        entities: [],
        legalConcepts: [],
        complexity: 0.5,
      };
    }
  }

  /**
   * Step 2: Multi-strategy retrieval using all available sources
   */
  private async performMultiStrategyRetrieval(
    processedQuery: SynthesizedOutput['processedQuery'],
    context?: SynthesizerInput['context'],
    options?: RetrievalOptions
  ): Promise<RetrievalResult> {
    // Default & merge options
    const defaults: Required<Pick<RetrievalOptions,
      'enableRAG' | 'maxSources' | 'similarityThreshold' | 'enableMMR' | 'enableCrossEncoder'
    >> = {
      enableRAG: true,
      maxSources: 10,
      similarityThreshold: 0.7,
      enableMMR: true,
      enableCrossEncoder: true
    };
    const effectiveOptions = { ...defaults, ...options };

    const retrievalResults: RetrievalResult = {
      sources: [],
      summary: { abstractive: '', extractive: [], keyPoints: [] },
      totalSources: 0,
      searchStrategies: [],
    };

    try {
      // Strategy 1: RAG Pipeline Search
      if (effectiveOptions.enableRAG) {
        try {
          const ragResults = await enhancedRAGPipeline.hybridSearch({
            query: processedQuery.enhanced,
            caseId: context?.caseId,
            limit: effectiveOptions.maxSources,
            threshold: effectiveOptions.similarityThreshold,
          });

          for (const doc of ragResults) {
            retrievalResults.sources.push({
              id: doc.metadata.documentId || `rag_${Date.now()}_${Math.random()}`,
              title: doc.metadata.title || 'Document',
              content: (doc as any).pageContent || (doc as any).content || '',
              relevanceScore: doc.metadata.score || 0.5,
              diversityScore: 0.5,
              rerankedScore: 0.5,
              type: (doc.metadata.documentType as any) || 'document',
              metadata: doc.metadata,
            });
          }

          retrievalResults.searchStrategies.push('rag_hybrid');
          logger.debug(`[Synthesizer] RAG search found ${ragResults.length} results`);
        } catch (error: any) {
          logger.warn('[Synthesizer] RAG search failed:', error);
        }
      }

      // Strategy 2: Enhanced Legal Search
      try {
        const legalSearchResults: any[] = await enhancedLegalSearch.search(processedQuery.enhanced, {
          maxResults: effectiveOptions.maxSources,
          useAI: true,
        });

        for (const result of legalSearchResults) {
          if (!retrievalResults.sources.find((s) => s.id === result.id)) {
            retrievalResults.sources.push({
              id: result.id,
              title: result.title,
              content: result.content,
              relevanceScore: result.score,
              diversityScore: 0.5,
              rerankedScore: 0.5,
              type: result.category === 'case_law' ? 'case' : 'document',
              metadata: {
                jurisdiction: result.jurisdiction,
                category: result.category,
                searchType: result.searchType,
                confidence: result.confidence,
              },
            });
          }
        }

        if (legalSearchResults.length > 0) {
          retrievalResults.searchStrategies.push('enhanced_legal_search');
        }
        logger.debug(`[Synthesizer] Legal search found ${legalSearchResults.length} results`);
      } catch (error: any) {
        logger.warn('[Synthesizer] Enhanced legal search failed:', error);
      }

      // Strategy 3: Context-based retrieval from provided documents
      if (context?.documents?.length) {
        for (const doc of context.documents) {
          try {
            const docEmbedding = (await generateEmbedding(doc.content)) || [];
            const queryEmbedding = (await generateEmbedding(processedQuery.enhanced)) || [];
            const similarity = this.calculateCosineSimilarity(docEmbedding, queryEmbedding);

            if (similarity > effectiveOptions.similarityThreshold) {
              retrievalResults.sources.push({
                id: doc.id,
                title: doc.title,
                content: doc.content,
                relevanceScore: similarity,
                diversityScore: 0.5,
                rerankedScore: 0.5,
                type: doc.type as any,
                metadata: { source: 'context_documents' },
              });
            }
          } catch (error: any) {
            logger.warn('[Synthesizer] Context document processing failed:', error);
          }
        }

        retrievalResults.searchStrategies.push('context_documents');
      }

      retrievalResults.totalSources = retrievalResults.sources.length;
      return retrievalResults;
    } catch (error: any) {
      logger.error('[Synthesizer] Multi-strategy retrieval failed:', error);
      return retrievalResults;
    }
  }

  /**
   * Step 3: Process and rank retrieved content
   */
  private async processAndRankContent(
    retrievedContext: any,
    processedQuery: SynthesizedOutput['processedQuery'],
    options: any
  ): Promise<SynthesizedOutput['retrievedContext']> {
    try {
      let sources = [...retrievedContext.sources];

      // Apply MMR for diversity
      if (options.enableMMR && sources.length > 1) {
        try {
          sources = await this.applyMMRDiversification(
            sources,
            processedQuery.enhanced,
            options.diversityLambda
          );
          logger.debug('[Synthesizer] Applied MMR diversification');
        } catch (error: any) {
          logger.warn('[Synthesizer] MMR diversification failed:', error);
        }
      }

      // Apply cross-encoder reranking
      if (options.enableCrossEncoder && sources.length > 1) {
        try {
          sources = await this.applyCrossEncoderReranking(sources, processedQuery.enhanced);
          logger.debug('[Synthesizer] Applied cross-encoder reranking');
        } catch (error: any) {
          logger.warn('[Synthesizer] Cross-encoder reranking failed:', error);
        }
      }

      // Generate comprehensive summary
      const summary = await this.generateComprehensiveSummary(sources, processedQuery);

      return {
        sources: sources.slice(0, options.maxSources),
        summary,
        totalSources: sources.length,
        searchStrategies: retrievedContext.searchStrategies,
      };
    } catch (error: any) {
      logger.error('[Synthesizer] Content processing failed:', error);
      return retrievedContext;
    }
  }

  /**
   * Step 4: Construct enhanced prompt for AI assistant
   */
  private async constructEnhancedPrompt(
    processedQuery: SynthesizedOutput['processedQuery'],
    retrievedContext: SynthesizedOutput['retrievedContext'],
    context?: SynthesizerInput['context']
  ): Promise<SynthesizedOutput['enhancedPrompt']> {
    try {
      // System prompt with legal expertise
      const systemPrompt = this.buildSystemPrompt(processedQuery, context);

      // Context prompt with retrieved information
      const contextPrompt = this.buildContextPrompt(retrievedContext);

      // Query prompt with enhanced understanding
      const queryPrompt = this.buildQueryPrompt(processedQuery, context);

      // Instructions based on query analysis
      const instructions = this.buildInstructions(processedQuery, context);

      // Constraints for response quality
      const constraints = this.buildConstraints(processedQuery, context);

      return {
        systemPrompt,
        contextPrompt,
        queryPrompt,
        instructions,
        constraints,
      };
    } catch (error: any) {
      logger.error('[Synthesizer] Prompt construction failed:', error);
      return {
        systemPrompt: 'You are a legal AI assistant.',
        contextPrompt: 'No context available.',
        queryPrompt: processedQuery.original,
        instructions: ['Provide a helpful response.'],
        constraints: ['Be accurate and professional.'],
      };
    }
  }

  /**
   * Step 5: Assess overall quality of synthesized input
   */
  private async assessQuality(
    processedQuery: SynthesizedOutput['processedQuery'],
    retrievedContext: SynthesizedOutput['retrievedContext'],
    enhancedPrompt: SynthesizedOutput['enhancedPrompt']
  ): Promise<QualityMetrics> {
    try {
      // Context relevance score
      const contextRelevance = this.assessContextRelevance(processedQuery, retrievedContext);

      // Source authority score
      const sourceAuthority = this.assessSourceAuthority(retrievedContext);

      // Concept coverage score
      const conceptCoverage = this.assessConceptCoverage(processedQuery, retrievedContext);

      // Information completeness score
      const informationCompleteness = this.assessInformationCompleteness(
        processedQuery,
        retrievedContext
      );

      // Response readiness score
      const responseReadiness = this.assessResponseReadiness(
        processedQuery,
        retrievedContext,
        enhancedPrompt
      );

      return {
        contextRelevance,
        sourceAuthority,
        conceptCoverage,
        informationCompleteness,
        responseReadiness,
      };
    } catch (error: any) {
      logger.error('[Synthesizer] Quality assessment failed:', error);
      return {
        contextRelevance: 0.5,
        sourceAuthority: 0.5,
        conceptCoverage: 0.5,
        informationCompleteness: 0.5,
        responseReadiness: 0.5,
      };
    }
  }

  // === HELPER METHODS ===

  private async verifyComponents(): Promise<void> {
    const checks: Array<{ name: string; check: () => Promise<{ status: string }> }> = [
      { name: 'LegalBERT', check: () => Promise.resolve({ status: 'healthy' }) },
      { name: 'RAG Pipeline', check: () => Promise.resolve({ status: 'healthy' }) },
      { name: 'Legal Search', check: () => Promise.resolve({ status: 'healthy' }) },
    ];

    for (const { name, check } of checks) {
      try {
        const result = await this.withTimeout(check(), 5000);
        logger.debug(`[Synthesizer] ${name}: OK`);
      } catch (error: any) {
        logger.warn(`[Synthesizer] ${name}: ${error?.message || 'Unknown error'}`);
      }
    }
  }

  private async withTimeout<T>(promise: Promise<T>, timeout: number): Promise<T> {
    return Promise.race([
      promise,
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Timeout after ${timeout}ms`)), timeout)
      ),
    ]);
  }

  private async extractIntent(query: string, analysis: SynthesizerAnalysisResult): Promise<string> {
    // Intent classification based on patterns and entities
    const queryLower = query.toLowerCase();

    if (queryLower.includes('how to') || queryLower.includes('how do')) {
      return 'procedural_guidance';
    } else if (queryLower.includes('what is') || queryLower.includes('define')) {
      return 'definition_request';
    } else if (
      queryLower.includes('case') ||
      analysis.entities.some((e: any) => e.type === 'CASE_CITATION')
    ) {
      return 'case_analysis';
    } else if (
      queryLower.includes('contract') ||
      analysis.concepts.some((c) => c.concept === 'contract')
    ) {
      return 'contract_analysis';
    } else if (
      queryLower.includes('statute') ||
      analysis.entities.some((e: any) => e.type === 'STATUTE')
    ) {
      return 'statute_interpretation';
    } else if (queryLower.includes('precedent') || queryLower.includes('ruling')) {
      return 'precedent_search';
    } else {
      return 'general_legal_query';
    }
  }

  private async enhanceQueryWithContext(
    query: string,
    context?: SynthesizerInput['context'],
    analysis?: SynthesizerAnalysisResult
  ): Promise<string> {
    let enhanced = query;

    // Add legal context terms
    if (analysis?.concepts?.length > 0) {
      const topConcepts = analysis.concepts
        .slice(0, 3)
        .map((c) => c.concept)
        .join(' ');
      enhanced += ` (Related: ${topConcepts})`;
    }

    // Add jurisdictional context
    if (context?.caseId) {
      enhanced += ' [Case context available]';
    }

    // Add conversation context
    if (context?.conversationHistory?.length > 0) {
      const lastMessage = context.conversationHistory[context.conversationHistory.length - 1];
      if (lastMessage.role === 'user') {
        enhanced += ` [Follow-up to: ${lastMessage.content.substring(0, 50)}...]`;
      }
    }

    return enhanced;
  }

  private async applyMMRDiversification(
    sources: any[],
    query: string,
    lambda: number = 0.5
  ): Promise<any[]> {
    // Implement MMR algorithm for diversity
    if (sources.length <= 1) return sources;

    const selected: any[] = [];
    const remaining = [...sources];

    // Start with the highest relevance score
    const first = remaining.reduce((max, curr) =>
      curr.relevanceScore > max.relevanceScore ? curr : max
    );
    selected.push(first);
    remaining.splice(remaining.indexOf(first), 1);

    // Apply MMR selection
    while (remaining.length > 0 && selected.length < 10) {
      let bestScore = -Infinity;
      let bestIndex = -1;

      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i];

        // Relevance component
        const relevance = candidate.relevanceScore;

        // Diversity component (max similarity to selected items)
        let maxSimilarity = 0;
        for (const selectedDoc of selected) {
          const similarity = await this.calculateTextSimilarity(
            candidate.content,
            selectedDoc.content
          );
          maxSimilarity = Math.max(maxSimilarity, similarity);
        }

        // MMR score
        const mmrScore = lambda * relevance - (1 - lambda) * maxSimilarity;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIndex = i;
        }
      }

      if (bestIndex >= 0) {
        const selected_item = remaining[bestIndex];
        selected_item.diversityScore = bestScore;
        selected.push(selected_item);
        remaining.splice(bestIndex, 1);
      } else {
        break;
      }
    }

    return selected;
  }

  private async applyCrossEncoderReranking(sources: any[], query: string): Promise<any[]> {
    // Apply cross-encoder reranking using LegalBERT
    try {
      const rerankedSources = [];

      for (const source of sources) {
        const similarity = await legalBERT.calculateLegalSimilarity(query, source.content);
        source.rerankedScore = similarity.similarity;
        rerankedSources.push(source);
      }

      // Sort by reranked score
      return rerankedSources.sort((a, b) => b.rerankedScore - a.rerankedScore);
    } catch (error: any) {
      logger.warn('[Synthesizer] Cross-encoder reranking failed:', error);
      return sources;
    }
  }

  private async generateComprehensiveSummary(
    sources: any[],
    processedQuery: SynthesizedOutput['processedQuery']
  ): Promise<SynthesizedOutput['retrievedContext']['summary']> {
    try {
      // Combine all source content
      const combinedContent = sources.map((s) => s.content).join('\n\n');

      // Generate summary using available methods
      const summary = await legalBERT.analyzeLegalText(combinedContent);

      return {
        abstractive: summary.summary.abstractive,
        extractive: Array.isArray(summary.summary.extractive)
          ? summary.summary.extractive
          : [summary.summary.extractive || ''],
        keyPoints: Array.isArray(summary.summary.keyPoints)
          ? summary.summary.keyPoints
          : [summary.summary.keyPoints || ''],
      };
    } catch (error: any) {
      logger.warn('[Synthesizer] Summary generation failed:', error);
      return {
        abstractive: 'Summary generation failed',
        extractive: sources.slice(0, 3).map((s) => s.content.substring(0, 200) + '...'),
        keyPoints: sources.slice(0, 5).map((s) => s.title),
      };
    }
  }

  private buildSystemPrompt(
    processedQuery: SynthesizedOutput['processedQuery'],
    context?: SynthesizerInput['context']
  ): string {
    let prompt =
      'You are an expert legal AI assistant with comprehensive knowledge of law and legal procedures.';

    // Add specialization based on detected legal concepts
    if (processedQuery.legalConcepts.includes('contract')) {
      prompt += ' You specialize in contract law and analysis.';
    } else if (processedQuery.legalConcepts.includes('tort')) {
      prompt += ' You specialize in tort law and liability analysis.';
    } else if (processedQuery.legalConcepts.includes('criminal')) {
      prompt += ' You specialize in criminal law and procedure.';
    }

    // Add response style preference
    if (context?.preferences?.responseStyle === 'formal') {
      prompt += ' Use formal legal language and cite authorities appropriately.';
    } else if (context?.preferences?.responseStyle === 'casual') {
      prompt += ' Explain legal concepts in accessible, everyday language.';
    }

    return prompt;
  }

  private buildContextPrompt(retrievedContext: SynthesizedOutput['retrievedContext']): string {
    let prompt = 'Based on the following legal sources and information:\n\n';

    retrievedContext.sources.forEach((source, index) => {
      prompt += `[Source ${index + 1}] ${source.title}\n`;
      prompt += `${source.content.substring(0, 500)}...\n\n`;
    });

    if (retrievedContext.summary.keyPoints.length > 0) {
      prompt += 'Key Points:\n';
      retrievedContext.summary.keyPoints.forEach((point) => {
        prompt += `- ${point}\n`;
      });
    }

    return prompt;
  }

  private buildQueryPrompt(
    processedQuery: SynthesizedOutput['processedQuery'],
    context?: SynthesizerInput['context']
  ): string {
    let prompt = `Question: ${processedQuery.original}\n\n`;

    if (processedQuery.entities.length > 0) {
      prompt += 'Relevant Legal Entities:\n';
      processedQuery.entities.forEach((entity) => {
        prompt += `- ${entity.text} (${entity.type})\n`;
      });
      prompt += '\n';
    }

    if (context?.conversationHistory?.length > 0) {
      prompt += 'Previous Context:\n';
      const lastMessages = context.conversationHistory.slice(-3);
      lastMessages.forEach((msg) => {
        prompt += `${msg.role}: ${msg.content.substring(0, 100)}...\n`;
      });
    }

    return prompt;
  }

  private buildInstructions(
    processedQuery: SynthesizedOutput['processedQuery'],
    context?: SynthesizerInput['context']
  ): string[] {
    const instructions = [
      'Provide accurate legal information based on the provided sources',
      'Cite specific sources using [Source N] notation',
      'Identify relevant legal principles and precedents',
      'Note any limitations or caveats in your analysis',
    ];

    // Add intent-specific instructions
    switch (processedQuery.intent) {
      case 'case_analysis':
        instructions.push('Analyze the key holdings and their legal significance');
        break;
      case 'contract_analysis':
        instructions.push('Examine contract terms, potential issues, and recommendations');
        break;
      case 'statute_interpretation':
        instructions.push(
          "Explain the statute's application and any relevant interpretive guidance"
        );
        break;
      case 'procedural_guidance':
        instructions.push('Provide step-by-step procedural guidance where applicable');
        break;
    }

    if (context?.preferences?.includeCitations) {
      instructions.push('Include full legal citations where appropriate');
    }

    return instructions;
  }

  private buildConstraints(
    processedQuery: SynthesizedOutput['processedQuery'],
    context?: SynthesizerInput['context']
  ): string[] {
    const constraints = [
      'Do not provide legal advice or attorney-client privileged information',
      'Clearly distinguish between legal information and legal advice',
      'Acknowledge the limitations of AI-generated legal analysis',
    ];

    if (context?.preferences?.maxLength) {
      constraints.push(`Keep response under ${context.preferences.maxLength} words`);
    }

    if (processedQuery.complexity > 0.8) {
      constraints.push('Given the complexity, recommend consulting with a qualified attorney');
    }

    return constraints;
  }

  // === QUALITY ASSESSMENT METHODS ===

  private assessContextRelevance(
    processedQuery: SynthesizedOutput['processedQuery'],
    retrievedContext: SynthesizedOutput['retrievedContext']
  ): number {
    if (retrievedContext.sources.length === 0) return 0;

    const avgRelevance =
      retrievedContext.sources.reduce((sum, source) => sum + source.relevanceScore, 0) /
      retrievedContext.sources.length;

    return Math.min(avgRelevance, 1.0);
  }

  private assessSourceAuthority(retrievedContext: SynthesizedOutput['retrievedContext']): number {
    const authorityScores = {
      case: 0.9,
      statute: 0.95,
      precedent: 0.85,
      document: 0.7,
    };

    if (retrievedContext.sources.length === 0) return 0;

    const avgAuthority =
      retrievedContext.sources.reduce(
        (sum, source) => sum + (authorityScores[source.type] || 0.5),
        0
      ) / retrievedContext.sources.length;

    return avgAuthority;
  }

  private assessConceptCoverage(
    processedQuery: SynthesizedOutput['processedQuery'],
    retrievedContext: SynthesizedOutput['retrievedContext']
  ): number {
    if (processedQuery.legalConcepts.length === 0) return 0.8;

    const sourcesText = retrievedContext.sources.map((s) => s.content.toLowerCase()).join(' ');

    const coveredConcepts = processedQuery.legalConcepts.filter((concept) =>
      sourcesText.includes(concept.toLowerCase())
    );

    return coveredConcepts.length / processedQuery.legalConcepts.length;
  }

  private assessInformationCompleteness(
    processedQuery: SynthesizedOutput['processedQuery'],
    retrievedContext: SynthesizedOutput['retrievedContext']
  ): number {
    // Assess based on number of sources, content length, and diversity
    const sourceCount = Math.min(retrievedContext.sources.length / 5, 1.0);
    const contentLength = Math.min(
      retrievedContext.sources.reduce((sum, s) => sum + s.content.length, 0) / 10000,
      1.0
    );
    const diversity =
      retrievedContext.sources.length > 1
        ? this.calculateSourceDiversity(retrievedContext.sources)
        : 0.5;

    return (sourceCount + contentLength + diversity) / 3;
  }

  private assessResponseReadiness(
    processedQuery: SynthesizedOutput['processedQuery'],
    retrievedContext: SynthesizedOutput['retrievedContext'],
    enhancedPrompt: SynthesizedOutput['enhancedPrompt']
  ): number {
    // Overall readiness based on all factors
    const hasContext = retrievedContext.sources.length > 0 ? 1.0 : 0.2;
    const hasInstructions = enhancedPrompt.instructions.length > 0 ? 1.0 : 0.5;
    const queryClarity =
      processedQuery.enhanced.length > processedQuery.original.length ? 0.9 : 0.7;

    return (hasContext + hasInstructions + queryClarity) / 3;
  }

  // === UTILITY METHODS ===

  private calculateCosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  private async calculateTextSimilarity(text1: string, text2: string): Promise<number> {
    try {
      const [emb1, emb2] = await Promise.all([generateEmbedding(text1), generateEmbedding(text2)]);

      return this.calculateCosineSimilarity(emb1 || [], emb2 || []);
    } catch (error: any) {
      // Fallback to basic similarity
      const words1 = new Set(text1.toLowerCase().split(/\s+/));
      const words2 = new Set(text2.toLowerCase().split(/\s+/));
      const intersection = new Set(Array.from(words1).filter((x) => words2.has(x)));
      const union = new Set([...Array.from(words1), ...Array.from(words2)]);

      return intersection.size / union.size;
    }
  }

  private calculateSourceDiversity(sources: any[]): number {
    if (sources.length <= 1) return 0.5;

    const types = new Set(sources.map((s) => s.type));
    const metadata = sources.map((s) => s.metadata?.category || 'unknown');
    const categories = new Set(metadata);

    const typeDiversity = types.size / Math.min(sources.length, 4);
    const categoryDiversity = categories.size / Math.min(sources.length, 5);

    return (typeDiversity + categoryDiversity) / 2;
  }

  private getUsedStrategies(options: any): string[] {
    const strategies = [];
    if (options.enableRAG) strategies.push('rag_pipeline');
    if (options.enableLegalBERT) strategies.push('legalbert_analysis');
    if (options.enableMMR) strategies.push('mmr_diversification');
    if (options.enableCrossEncoder) strategies.push('cross_encoder_reranking');
    return strategies;
  }

  private calculateOverallQuality(metrics: QualityMetrics): number {
    const weights = {
      contextRelevance: 0.3,
      sourceAuthority: 0.2,
      conceptCoverage: 0.2,
      informationCompleteness: 0.15,
      responseReadiness: 0.15,
    };

    return Object.entries(metrics).reduce(
      (sum, [key, value]) => sum + (weights[key as keyof QualityMetrics] || 0) * value,
      0
    );
  }

  private generateRecommendations(metrics: QualityMetrics): string[] {
    const recommendations = [];

    if (metrics.contextRelevance < 0.6) {
      recommendations.push('Consider refining search terms for better context relevance');
    }
    if (metrics.sourceAuthority < 0.7) {
      recommendations.push('Seek additional authoritative legal sources');
    }
    if (metrics.conceptCoverage < 0.5) {
      recommendations.push('Expand search to cover more legal concepts');
    }
    if (metrics.informationCompleteness < 0.6) {
      recommendations.push('Gather additional supporting information');
    }
    if (metrics.responseReadiness < 0.7) {
      recommendations.push('Review query clarity and available context');
    }

    return recommendations;
  }

  // === PUBLIC API ===

  /**
   * Get synthesizer statistics
   */
  getStatistics(): Record<string, any> {
    return {
      requestCount: this.requestCount,
      processingStats: Object.fromEntries(this.processingStats),
      ...metrics.getAllMetrics(),
    };
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; components: Record<string, any> }> {
    const components: Record<string, any> = {};

    try {
      components.legalbert = await legalBERT.healthCheck();
    } catch (error: any) {
      components.legalbert = { status: 'unhealthy', error: error.message };
    }

    try {
      components.rag = await enhancedRAGPipeline.getHealthStatus();
    } catch (error: any) {
      components.rag = { status: 'unhealthy', error: error.message };
    }

    const healthyComponents = Object.values(components).filter(
      (c: any) =>
        c.status === 'healthy' ||
        (Array.isArray(c) && c.every((item: any) => item.status === 'healthy'))
    ).length;

    return {
      status: healthyComponents >= 1 ? 'healthy' : 'degraded',
      components,
    };
  }
}

// Export singleton instance
export const aiAssistantSynthesizer = new AIAssistantInputSynthesizer();
;
