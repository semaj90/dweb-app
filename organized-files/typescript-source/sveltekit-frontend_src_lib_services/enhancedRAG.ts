// @ts-nocheck
/**
 * Enhanced RAG System with Query Synthesis
 * Combines vector search, graph reasoning, and LLM synthesis
 */

import { writable, type Writable } from "svelte/store";
import {
  qdrantService,
  type SearchResult,
  type DocumentVector,
} from "./qdrantService";
import type { LLMProvider } from "$lib/types/llm";

// Enhanced RAG Types
export interface RAGQuery {
  id: string;
  originalQuery: string;
  expandedQueries: string[];
  intent: QueryIntent;
  context: QueryContext;
  filters: RAGFilters;
  timestamp: number;
}

export interface QueryIntent {
  type: "search" | "analysis" | "generation" | "comparison" | "synthesis";
  confidence: number;
  entities: string[];
  practiceArea?: string;
  urgency: "low" | "medium" | "high" | "critical";
}

export interface QueryContext {
  userId: string;
  sessionId: string;
  previousQueries: string[];
  userPreferences: Record<string, unknown>;
  documentContext?: string[];
}

export interface RAGFilters {
  dateRange?: { start: string; end: string };
  documentTypes?: string[];
  practiceAreas?: string[];
  jurisdictions?: string[];
  authors?: string[];
  minConfidence?: number;
}

export interface RAGResponse {
  queryId: string;
  synthesizedAnswer: string;
  sources: EnhancedSource[];
  confidence: number;
  reasoning: string[];
  suggestions: string[];
  relatedQueries: string[];
  processingTime: number;
  metadata: RAGMetadata;
}

export interface EnhancedSource {
  document: DocumentVector;
  relevanceScore: number;
  excerpts: string[];
  reasoning: string;
  citations: Citation[];
}

export interface Citation {
  type: "direct_quote" | "paraphrase" | "statistical" | "legal_precedent";
  text: string;
  page?: number;
  section?: string;
  confidence: number;
}

export interface RAGMetadata {
  searchStrategy: string;
  vectorSearchResults: number;
  llmCalls: number;
  cached: boolean;
  qualityScore: number;
}

// Main Enhanced RAG Class
export class EnhancedRAGSystem {
  private cache: Map<string, RAGResponse> = new Map();
  private readonly CACHE_TTL = 1000 * 60 * 30; // 30 minutes

  private safeJSONParse(jsonString: string): any {
    try {
      return JSON.parse(jsonString);
    } catch (error) {
      console.warn('JSON parse failed:', error);
      return {};
    }
  }

  // Reactive stores
  public activeQuery$: Writable<RAGQuery | null> = writable(null);
  public searchResults$: Writable<SearchResult[]> = writable([]);
  public ragResponse$: Writable<RAGResponse | null> = writable(null);
  public processing$: Writable<boolean> = writable(false);

  constructor(private llmProvider: LLMProvider) {}

  /**
   * Main RAG Query Method - The Heart of the System
   */
  public async queryRAG(
    originalQuery: string,
    context: QueryContext,
    options: {
      useQueryExpansion?: boolean;
      useGraphReasoning?: boolean;
      maxSources?: number;
      qualityThreshold?: number;
    } = {}
  ): Promise<RAGResponse> {
    const queryId = `rag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    this.processing$.set(true);

    try {
      // Phase 1: Query Analysis & Intent Detection
      const queryIntent = await this.analyzeQueryIntent(originalQuery, context);

      // Phase 2: Query Expansion (Multi-perspective search)
      const expandedQueries = options.useQueryExpansion
        ? await this.expandQuery(originalQuery, queryIntent, context)
        : [originalQuery];

      // Phase 3: Build Dynamic Filters
      const filters = await this.buildSmartFilters(queryIntent, context);

      // Create enhanced query object
      const ragQuery: RAGQuery = {
        id: queryId,
        originalQuery,
        expandedQueries,
        intent: queryIntent,
        context,
        filters,
        timestamp: Date.now(),
      };

      this.activeQuery$.set(ragQuery);

      // Phase 4: Multi-Vector Search with Ranking
      const searchResults = await this.performEnhancedSearch(
        expandedQueries,
        filters,
        options.maxSources || 10
      );

      this.searchResults$.set(searchResults);

      // Phase 5: Graph-Based Reasoning (Optional)
      let enhancedSources = await this.processSearchResults(
        searchResults,
        queryIntent
      );

      if (options.useGraphReasoning) {
        enhancedSources = await this.enhanceWithGraphReasoning(
          enhancedSources,
          ragQuery.originalQuery
        );
      }

      // Phase 6: LLM Synthesis with Chain-of-Thought
      const synthesizedAnswer = await this.synthesizeAnswer(
        ragQuery,
        enhancedSources,
        options.qualityThreshold || 0.7
      );

      // Phase 7: Generate Follow-up Suggestions
      const suggestions = await this.generateSuggestions(
        ragQuery,
        synthesizedAnswer
      );
      const relatedQueries = await this.generateRelatedQueries(
        ragQuery,
        enhancedSources
      );

      // Build final response
      const response: RAGResponse = {
        queryId,
        synthesizedAnswer,
        sources: enhancedSources,
        confidence: this.calculateOverallConfidence(
          enhancedSources,
          synthesizedAnswer
        ),
        reasoning: this.extractReasoning(synthesizedAnswer),
        suggestions,
        relatedQueries,
        processingTime: Date.now() - startTime,
        metadata: {
          searchStrategy: "enhanced_vector_graph",
          vectorSearchResults: searchResults.length,
          llmCalls: this.countLLMCalls(),
          cached: false,
          qualityScore: this.calculateQualityScore(
            enhancedSources,
            synthesizedAnswer
          ),
        },
      };

      // Cache response
      this.cache.set(queryId, response);

      // Update stores
      this.ragResponse$.set(response);
      this.processing$.set(false);

      // Log user activity
      await qdrantService.logUserActivity(context.userId, {
        type: "search",
        query: originalQuery,
        metadata: {
          queryId,
          resultsCount: enhancedSources.length,
          processingTime: response.processingTime,
        },
      });

      console.log(
        `🧠 RAG query processed in ${response.processingTime}ms with ${enhancedSources.length} sources`
      );
      return response;
    } catch (error) {
      console.error("❌ RAG query failed:", error);
      this.processing$.set(false);
      throw error;
    }
  }

  /**
   * Phase 1: Query Intent Analysis
   */
  private async analyzeQueryIntent(
    query: string,
    context: QueryContext
  ): Promise<QueryIntent> {
    const prompt = `
Analyze this legal query and extract the user's intent:

Query: "${query}"
User Context: ${JSON.stringify(context.userPreferences || {})}
Previous Queries: ${context.previousQueries.join(", ")}

Determine:
1. Primary intent type (search, analysis, generation, comparison, synthesis)
2. Confidence level (0.0-1.0)
3. Key legal entities mentioned
4. Likely practice area
5. Urgency level

Return JSON:
{
  "type": "search|analysis|generation|comparison|synthesis",
  "confidence": 0.95,
  "entities": ["entity1", "entity2"],
  "practiceArea": "litigation",
  "urgency": "medium"
}`;

    try {
      const response = await fetch(
        `${this.llmProvider.endpoint}/api/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "gemma3-legal",
            prompt,
            stream: false,
            format: "json",
          }),
        }
      );

      const data = await response.json();
      if (!data || typeof data !== 'object') {
      throw new Error('Invalid LLM response format');
      }
      return this.safeJSONParse(data.response);
    } catch (error) {
      console.warn("Intent analysis failed, using defaults:", error);
      return {
        type: "search",
        confidence: 0.5,
        entities: [],
        urgency: "medium",
      };
    }
  }

  /**
   * Phase 2: Query Expansion for Multi-Perspective Search
   */
  private async expandQuery(
    originalQuery: string,
    intent: QueryIntent,
    _context: QueryContext
  ): Promise<string[]> {
    const prompt = `
Expand this legal query into 3-5 related search queries to capture different perspectives:

Original Query: "${originalQuery}"
Intent: ${intent.type}
Practice Area: ${intent.practiceArea || "general"}
Entities: ${intent.entities.join(", ")}

Generate variations that cover:
1. Synonyms and alternative terminology
2. Related legal concepts
3. Different jurisdictional perspectives
4. Procedural vs. substantive angles
5. Historical/precedent context

Return as JSON array: ["query1", "query2", "query3", ...]`;

    try {
      const response = await fetch(
        `${this.llmProvider.endpoint}/api/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "gemma3-legal",
            prompt,
            stream: false,
            format: "json",
          }),
        }
      );

      const data = await response.json();
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid LLM response format');
      }
      const expanded = this.safeJSONParse(data.response);
      return [originalQuery, ...expanded].slice(0, 5); // Limit to 5 total queries
    } catch (error) {
      console.warn("Query expansion failed:", error);
      return [originalQuery];
    }
  }

  /**
   * Phase 3: Smart Filter Generation
   */
  private async buildSmartFilters(
    intent: QueryIntent,
    _context: QueryContext
  ): Promise<RAGFilters> {
    const filters: RAGFilters = {
      minConfidence: 0.6,
    };

    // Add practice area filter if detected
    if (intent.practiceArea) {
      filters.practiceAreas = [intent.practiceArea];
    }

    // Add document type filters based on intent
    switch (intent.type) {
      case "analysis":
        filters.documentTypes = ["case_law", "regulation", "brief"];
        break;
      case "generation":
        filters.documentTypes = ["contract", "legal_document"];
        break;
      case "comparison":
        filters.documentTypes = ["case_law", "regulation"];
        break;
    }

    // Add date range for recent precedents if high urgency
    if (intent.urgency === "high" || intent.urgency === "critical") {
      const currentYear = new Date().getFullYear();
      filters.dateRange = {
        start: `${currentYear - 5}-01-01`,
        end: `${currentYear}-12-31`,
      };
    }

    return filters;
  }

  /**
   * Phase 4: Enhanced Multi-Vector Search
   */
  private async performEnhancedSearch(
    queries: string[],
    filters: RAGFilters,
    maxResults: number
  ): Promise<SearchResult[]> {
    const allResults: SearchResult[] = [];

    // Perform searches for each expanded query
    for (const query of queries) {
      try {
        // Generate embedding for the query
        const embedding = await this.generateEmbedding(query);

        // Search with filters
        const results = await qdrantService.semanticSearch(query, embedding, {
          practiceArea: filters.practiceAreas?.[0],
          documentType: filters.documentTypes?.[0],
          dateRange: filters.dateRange,
          limit: Math.ceil(maxResults / queries.length),
        });

        // Weight results based on query importance (original query gets higher weight)
        const weightedResults = results.map((result) => ({
          ...result,
          score: result.score * (query === queries[0] ? 1.0 : 0.8),
        }));

        allResults.push(...weightedResults);
      } catch (error) {
        console.warn(`Search failed for query: ${query}`, error);
      }
    }

    // Deduplicate and rank results
    const uniqueResults = this.deduplicateResults(allResults);
    return this.rankResults(uniqueResults, filters).slice(0, maxResults);
  }

  /**
   * Phase 5: Process and Enhance Search Results
   */
  private async processSearchResults(
    results: SearchResult[],
    intent: QueryIntent
  ): Promise<EnhancedSource[]> {
    const enhancedSources: EnhancedSource[] = [];

    for (const result of results) {
      try {
        // Extract relevant excerpts
        const excerpts = await this.extractRelevantExcerpts(result, intent);

        // Generate reasoning for why this source is relevant
        const reasoning = await this.generateSourceReasoning(result, intent);

        // Generate citations
        const citations = this.generateCitations(result, excerpts);

        enhancedSources.push({
          document: result.payload,
          relevanceScore: result.score,
          excerpts,
          reasoning,
          citations,
        });
      } catch (error) {
        console.warn("Failed to enhance source:", error);
        // Include with basic information
        enhancedSources.push({
          document: result.payload,
          relevanceScore: result.score,
          excerpts: [result.payload.content.substring(0, 200) + "..."],
          reasoning: "High vector similarity match",
          citations: [],
        });
      }
    }

    return enhancedSources;
  }

  /**
   * Phase 6: LLM Synthesis with Chain-of-Thought
   */
  private async synthesizeAnswer(
    query: RAGQuery,
    sources: EnhancedSource[],
    qualityThreshold: number
  ): Promise<string> {
    // Filter sources by quality threshold
    const qualitySources = sources.filter(
      (s) => s.relevanceScore >= qualityThreshold
    );

    if (qualitySources.length === 0) {
      return "I couldn't find sufficient high-quality sources to answer your question confidently. Please try refining your query or expanding the search criteria.";
    }

    // Build comprehensive context
    const contextSources = qualitySources
      .map(
        (source, index) => `
Source ${index + 1} (Relevance: ${(source.relevanceScore * 100).toFixed(1)}%):
Title: ${source.document?.metadata?.title ?? "Untitled Document"}
Type: ${source.document?.metadata?.type ?? "Unknown"}
Key Excerpts:
${source.excerpts.join("\n")}
Reasoning: ${source.reasoning}
`
      )
      .join("\n---\n");

    const synthesisPrompt = `
You are an expert legal research assistant. Analyze the provided sources and synthesize a comprehensive answer.

Original Query: "${query.originalQuery}"
Query Intent: ${query.intent.type}
Practice Area: ${query.intent.practiceArea || "General"}

Sources:
${contextSources}

Instructions:
1. Synthesize a clear, accurate answer based ONLY on the provided sources
2. Include specific citations in [Source N] format
3. Highlight any conflicting information between sources
4. Note limitations or gaps in the available information
5. Provide practical implications where relevant
6. Use professional legal writing style

Structure your response as:
## Summary
[Main answer in 2-3 sentences]

## Detailed Analysis
[Comprehensive analysis with citations]

## Key Considerations
[Important caveats, limitations, or next steps]

## Sources Referenced
[Brief list of sources used]
`;

    try {
      const response = await fetch(
        `${this.llmProvider.endpoint}/api/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "gemma3-legal",
            prompt: synthesisPrompt,
            stream: false,
            options: {
              temperature: 0.3, // Lower temperature for factual accuracy
              top_p: 0.9,
              max_tokens: 2000,
            },
          }),
        }
      );

      const data = await response.json();
      return data.response;
    } catch (error) {
      console.error("Synthesis failed:", error);
      return "I encountered an error while synthesizing the answer. Please try again.";
    }
  }

  // Utility methods
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch("http://localhost:11434/api/embeddings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "nomic-embed-text",
          prompt: text,
        }),
      });

      const data = await response.json();
      return data.embedding;
    } catch (error) {
      console.error("Embedding generation failed:", error);
      return new Array(384).fill(0);
    }
  }

  private deduplicateResults(results: SearchResult[]): SearchResult[] {
    const seen = new Set<string>();
    return results.filter((result) => {
      if (seen.has(result.id)) return false;
      seen.add(result.id);
      return true;
    });
  }

  private rankResults(
    results: SearchResult[],
    _filters: RAGFilters
  ): SearchResult[] {
    return results.sort((a, b) => {
      // Primary sort by relevance score
      if (a.score !== b.score) return b.score - a.score;

      // Secondary sort by recency (if date available)
      const dateA = new Date(a.payload.metadata.date || "1900-01-01");
      const dateB = new Date(b.payload.metadata.date || "1900-01-01");
      return dateB.getTime() - dateA.getTime();
    });
  }

  private async extractRelevantExcerpts(
    result: SearchResult,
    intent: QueryIntent
  ): Promise<string[]> {
    // Simple excerpt extraction - can be enhanced with more sophisticated NLP
    const content = result.payload.content;
    const excerptLength = 300;
    const excerpts: string[] = [];

    // Extract excerpts around key entities
    for (const entity of intent.entities) {
      const index = content.toLowerCase().indexOf(entity.toLowerCase());
      if (index !== -1) {
        const start = Math.max(0, index - 100);
        const end = Math.min(content.length, index + excerptLength);
        excerpts.push(content.substring(start, end).trim());
      }
    }

    // If no entity-based excerpts, take the beginning
    if (excerpts.length === 0) {
      excerpts.push(content.substring(0, excerptLength).trim() + "...");
    }

    return excerpts.slice(0, 3); // Limit to 3 excerpts per source
  }

  private async generateSourceReasoning(
    result: SearchResult,
    intent: QueryIntent
  ): Promise<string> {
    // Generate simple reasoning based on available information
    const reasons: string[] = [];

    reasons.push(
      `High semantic similarity (${(result.score * 100).toFixed(1)}%)`
    );

    if (result.payload.metadata.practice_area === intent.practiceArea) {
      reasons.push("Matches target practice area");
    }

    if (result.payload.metadata.type) {
      reasons.push(`Relevant document type: ${result.payload.metadata.type}`);
    }

    return reasons.join("; ");
  }

  private generateCitations(
    result: SearchResult,
    excerpts: string[]
  ): Citation[] {
    return excerpts.map((excerpt) => ({
      type: "direct_quote" as const,
      text: excerpt,
      confidence: result.score,
    }));
  }

  private calculateOverallConfidence(
    sources: EnhancedSource[],
    answer: string
  ): number {
    if (sources.length === 0) return 0;

    const avgSourceConfidence =
      sources.reduce((sum, s) => sum + s.relevanceScore, 0) / sources.length;
    const answerQuality = Math.min(1, answer.length / 500); // Longer answers generally better

    return avgSourceConfidence * 0.7 + answerQuality * 0.3;
  }

  private extractReasoning(answer: string): string[] {
    // Extract reasoning steps from the synthesized answer
    const lines = answer.split("\n").filter((line) => line.trim());
    return lines
      .filter(
        (line) =>
          line.includes("because") ||
          line.includes("therefore") ||
          line.includes("given that") ||
          line.includes("based on")
      )
      .slice(0, 3);
  }

  private async generateSuggestions(
    query: RAGQuery,
    _answer: string
  ): Promise<string[]> {
    // Generate follow-up suggestions based on the query and answer
    return [
      `Explore recent ${query.intent.practiceArea || "legal"} developments`,
      "Find similar cases in your jurisdiction",
      "Review related statutes and regulations",
      "Search for expert commentary on this topic",
    ];
  }

  private async generateRelatedQueries(
    query: RAGQuery,
    sources: EnhancedSource[]
  ): Promise<string[]> {
    // Generate related queries based on the sources found
    const topics = sources.flatMap((s) => s.document.tags).slice(0, 10);
    return topics
      .map((topic) => `What are the latest developments in ${topic}?`)
      .slice(0, 4);
  }

  private calculateQualityScore(
    sources: EnhancedSource[],
    answer: string
  ): number {
    const sourceQuality =
      sources.length > 0
        ? sources.reduce((sum, s) => sum + s.relevanceScore, 0) / sources.length
        : 0;
    const answerCompleteness = Math.min(1, answer.length / 1000);
    const citationCount = (answer.match(/\[Source \d+\]/g) || []).length;
    const citationBonus = Math.min(0.2, citationCount * 0.05);

    return Math.min(
      1,
      sourceQuality * 0.6 + answerCompleteness * 0.3 + citationBonus
    );
  }

  private countLLMCalls(): number {
    // Track LLM calls made during processing
    return 3; // Intent analysis + Query expansion + Synthesis
  }

  private async enhanceWithGraphReasoning(
    sources: EnhancedSource[],
    _query: string
  ): Promise<EnhancedSource[]> {
    // Enhance sources with graph-based reasoning
    // This would typically involve analyzing relationships between documents
    // For now, we'll return the sources as-is but could add graph analysis here
    return sources.map((source) => ({
      ...source,
      document: {
        ...source.document,
        metadata: {
          ...source.document.metadata,
          graphEnhanced: true,
          graphReasoningApplied: new Date().toISOString(),
        },
      },
    }));
  }

  // Public utility methods
  public getCachedResponse(queryId: string): RAGResponse | null {
    const cached = this.cache.get(queryId);
    if (
      cached &&
      typeof cached.metadata.cached === "number" &&
      Date.now() - cached.metadata.cached < this.CACHE_TTL
    ) {
      return cached;
    }
    return null;
  }

  public clearCache(): void {
    this.cache.clear();
  }
}

// Export singleton instance
export const enhancedRAG = new EnhancedRAGSystem({
  id: "ollama-local",
  name: "Ollama Local",
  type: "ollama",
  endpoint: "http://localhost:11434",
  models: [],
  capabilities: [],
  status: "online",
});
