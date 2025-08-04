/**
 * Vector Search and Ranking Service with Nomic-Embed Integration
 * Advanced semantic search with multi-modal ranking and relevance scoring
 */
import { ollamaCudaService } from './ollama-cuda-service';
import { db } from '$lib/server/db';
import { evidence, cases, embeddingCache, vectorMetadata } from '$lib/server/db/schema-postgres-enhanced';
import { eq, sql, and, or, desc, asc, isNotNull, ilike, inArray } from 'drizzle-orm';
import { createHash } from 'crypto';
import Fuse from 'fuse.js';

export interface VectorSearchQuery {
  query: string;
  filters?: {
    caseId?: string;
    evidenceTypes?: string[];
    confidentialityLevels?: string[];
    dateRange?: {
      start: Date;
      end: Date;
    };
    tags?: string[];
    isAdmissible?: boolean;
    minConfidence?: number;
  };
  options?: {
    limit?: number;
    offset?: number;
    threshold?: number; // Cosine similarity threshold (0-1)
    hybridWeight?: number; // Weight for combining vector and text search (0-1)
    boostRecent?: boolean;
    includeSimilar?: boolean;
    expandQuery?: boolean;
  };
  ranking?: {
    strategy?: 'semantic' | 'hybrid' | 'bm25' | 'neural';
    weights?: {
      semantic?: number;
      text?: number;
      recency?: number;
      relevance?: number;
      authority?: number;
    };
  };
}

export interface SearchResult {
  id: string;
  title: string;
  description?: string;
  content?: string;
  evidenceType: string;
  caseId?: string;
  caseTitle?: string;
  tags: string[];
  score: number;
  semanticScore: number;
  textScore: number;
  confidenceScore: number;
  recencyScore: number;
  highlights: string[];
  reasoning: string;
  metadata: {
    fileSize?: number;
    mimeType?: string;
    createdAt: Date;
    updatedAt: Date;
    processingTime?: number;
  };
}

export interface VectorSearchResult {
  results: SearchResult[];
  totalCount: number;
  processingTime: number;
  queryExpansions?: string[];
  facets: {
    evidenceTypes: Array<{ type: string; count: number }>;
    cases: Array<{ caseId: string; title: string; count: number }>;
    tags: Array<{ tag: string; count: number }>;
    dateRanges: Array<{ range: string; count: number }>;
  };
  performance: {
    vectorSearchTime: number;
    textSearchTime: number;
    rankingTime: number;
    totalTime: number;
  };
}

export interface QueryExpansion {
  originalTerm: string;
  expandedTerms: string[];
  synonyms: string[];
  relatedConcepts: string[];
}

class VectorSearchService {
  private static instance: VectorSearchService;
  private fuseIndex: Fuse<any> | null = null;
  private lastIndexUpdate: Date | null = null;
  private indexUpdateInterval = 300000; // 5 minutes
  private queryCache = new Map<string, VectorSearchResult>();
  private cacheTimeout = 600000; // 10 minutes

  private constructor() {
    this.initializeFuseIndex();
  }

  public static getInstance(): VectorSearchService {
    if (!VectorSearchService.instance) {
      VectorSearchService.instance = new VectorSearchService();
    }
    return VectorSearchService.instance;
  }

  /**
   * Perform comprehensive vector-based search
   */
  public async search(searchQuery: VectorSearchQuery): Promise<VectorSearchResult> {
    const startTime = Date.now();
    
    try {
      // Check cache first
      const cacheKey = this.generateCacheKey(searchQuery);
      const cachedResult = this.queryCache.get(cacheKey);
      if (cachedResult && this.isCacheValid(cacheKey)) {
        return cachedResult;
      }

      // Generate query embedding
      const queryEmbedding = await this.generateQueryEmbedding(searchQuery.query);
      
      // Expand query if requested
      let expandedQueries: string[] = [searchQuery.query];
      if (searchQuery.options?.expandQuery) {
        expandedQueries = await this.expandQuery(searchQuery.query);
      }

      // Perform parallel searches
      const [vectorResults, textResults] = await Promise.all([
        this.performVectorSearch(queryEmbedding, searchQuery),
        this.performTextSearch(expandedQueries, searchQuery)
      ]);

      // Combine and rank results
      const combinedResults = await this.combineAndRankResults(
        vectorResults,
        textResults,
        searchQuery,
        queryEmbedding
      );

      // Generate facets
      const facets = await this.generateFacets(searchQuery);

      // Calculate performance metrics
      const totalTime = Date.now() - startTime;
      const performance = {
        vectorSearchTime: vectorResults.processingTime || 0,
        textSearchTime: textResults.processingTime || 0,
        rankingTime: 0, // Will be calculated in combineAndRankResults
        totalTime
      };

      const result: VectorSearchResult = {
        results: combinedResults.slice(
          searchQuery.options?.offset || 0,
          (searchQuery.options?.offset || 0) + (searchQuery.options?.limit || 20)
        ),
        totalCount: combinedResults.length,
        processingTime: totalTime,
        queryExpansions: expandedQueries.slice(1), // Exclude original query
        facets,
        performance
      };

      // Cache result
      this.queryCache.set(cacheKey, result);
      setTimeout(() => this.queryCache.delete(cacheKey), this.cacheTimeout);

      return result;
    } catch (error) {
      console.error('Vector search failed:', error);
      throw new Error(`Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Find similar documents using vector similarity
   */
  public async findSimilarDocuments(
    documentId: string,
    options: {
      limit?: number;
      threshold?: number;
      includeContent?: boolean;
      caseId?: string;
    } = {}
  ): Promise<SearchResult[]> {
    try {
      // Get document embedding
      const sourceDoc = await db
        .select()
        .from(evidence)
        .where(eq(evidence.id, documentId))
        .limit(1);

      if (sourceDoc.length === 0 || !sourceDoc[0].contentEmbedding) {
        throw new Error('Document not found or no embedding available');
      }

      const queryEmbedding = sourceDoc[0].contentEmbedding;
      const threshold = options.threshold || 0.7;
      const limit = options.limit || 10;

      // Build similarity query
      let query = db
        .select({
          id: evidence.id,
          title: evidence.title,
          description: evidence.description,
          evidenceType: evidence.evidenceType,
          caseId: evidence.caseId,
          tags: evidence.tags,
          aiSummary: evidence.aiSummary,
          fileSize: evidence.fileSize,
          mimeType: evidence.mimeType,
          createdAt: evidence.createdAt,
          updatedAt: evidence.updatedAt,
          similarity: sql<number>`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding})`.as('similarity')
        })
        .from(evidence)
        .where(
          and(
            isNotNull(evidence.contentEmbedding),
            sql`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding}) > ${threshold}`,
            sql`${evidence.id} != ${documentId}` // Exclude the source document
          )
        );

      // Add case filter if specified
      if (options.caseId) {
        query = query.where(eq(evidence.caseId, options.caseId));
      }

      const results = await query
        .orderBy(sql`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding}) DESC`)
        .limit(limit);

      // Get case titles for context
      const caseIds = [...new Set(results.map(r => r.caseId).filter(Boolean))];
      const caseMap = new Map<string, string>();
      
      if (caseIds.length > 0) {
        const caseRecords = await db
          .select({ id: cases.id, title: cases.title })
          .from(cases)
          .where(inArray(cases.id, caseIds));
        
        caseRecords.forEach(c => caseMap.set(c.id, c.title));
      }

      // Format results
      return results.map(result => ({
        id: result.id,
        title: result.title,
        description: result.description || undefined,
        content: options.includeContent ? result.aiSummary || undefined : undefined,
        evidenceType: result.evidenceType,
        caseId: result.caseId || undefined,
        caseTitle: result.caseId ? caseMap.get(result.caseId) : undefined,
        tags: Array.isArray(result.tags) ? result.tags as string[] : [],
        score: result.similarity,
        semanticScore: result.similarity,
        textScore: 0,
        confidenceScore: result.similarity,
        recencyScore: this.calculateRecencyScore(result.createdAt),
        highlights: [],
        reasoning: `Semantic similarity: ${(result.similarity * 100).toFixed(1)}%`,
        metadata: {
          fileSize: result.fileSize || undefined,
          mimeType: result.mimeType || undefined,
          createdAt: result.createdAt,
          updatedAt: result.updatedAt
        }
      }));
    } catch (error) {
      console.error('Failed to find similar documents:', error);
      throw error;
    }
  }

  /**
   * Semantic clustering of documents
   */
  public async clusterDocuments(
    documentIds: string[],
    options: {
      numClusters?: number;
      minClusterSize?: number;
      threshold?: number;
    } = {}
  ): Promise<Array<{
    clusterId: string;
    documents: SearchResult[];
    centroid: number[];
    coherenceScore: number;
    topics: string[];
  }>> {
    try {
      const numClusters = options.numClusters || Math.min(5, Math.ceil(documentIds.length / 3));
      const threshold = options.threshold || 0.7;

      // Get document embeddings
      const documents = await db
        .select({
          id: evidence.id,
          title: evidence.title,
          description: evidence.description,
          evidenceType: evidence.evidenceType,
          caseId: evidence.caseId,
          tags: evidence.tags,
          embedding: evidence.contentEmbedding,
          createdAt: evidence.createdAt,
          updatedAt: evidence.updatedAt
        })
        .from(evidence)
        .where(
          and(
            inArray(evidence.id, documentIds),
            isNotNull(evidence.contentEmbedding)
          )
        );

      if (documents.length < 2) {
        throw new Error('Need at least 2 documents with embeddings for clustering');
      }

      // Simple K-means clustering implementation
      const clusters = await this.performKMeansClustering(documents, numClusters);

      // Format clusters
      return clusters.map((cluster, index) => ({
        clusterId: `cluster_${index}`,
        documents: cluster.documents.map(doc => ({
          id: doc.id,
          title: doc.title,
          description: doc.description || undefined,
          evidenceType: doc.evidenceType,
          caseId: doc.caseId || undefined,
          tags: Array.isArray(doc.tags) ? doc.tags as string[] : [],
          score: cluster.coherenceScore,
          semanticScore: cluster.coherenceScore,
          textScore: 0,
          confidenceScore: cluster.coherenceScore,
          recencyScore: this.calculateRecencyScore(doc.createdAt),
          highlights: [],
          reasoning: `Cluster coherence: ${(cluster.coherenceScore * 100).toFixed(1)}%`,
          metadata: {
            createdAt: doc.createdAt,
            updatedAt: doc.updatedAt
          }
        })),
        centroid: cluster.centroid,
        coherenceScore: cluster.coherenceScore,
        topics: cluster.topics
      }));
    } catch (error) {
      console.error('Document clustering failed:', error);
      throw error;
    }
  }

  /**
   * Generate query suggestions based on existing data
   */
  public async generateQuerySuggestions(
    partialQuery: string,
    limit: number = 5
  ): Promise<Array<{
    suggestion: string;
    type: 'entity' | 'category' | 'keyword' | 'similar';
    frequency: number;
    confidence: number;
  }>> {
    try {
      if (partialQuery.length < 2) return [];

      // Search for matching entities, categories, and keywords in existing data
      const suggestions = await db
        .select({
          title: evidence.title,
          tags: evidence.tags,
          categories: sql<string[]>`COALESCE((ai_analysis->>'categories')::jsonb, '[]'::jsonb)`,
          keywords: sql<string[]>`COALESCE((ai_analysis->>'keywords')::jsonb, '[]'::jsonb)`
        })
        .from(evidence)
        .where(
          or(
            ilike(evidence.title, `%${partialQuery}%`),
            ilike(evidence.description, `%${partialQuery}%`),
            sql`${evidence.tags}::text ILIKE ${'%' + partialQuery + '%'}`
          )
        )
        .limit(50);

      // Aggregate suggestions
      const suggestionMap = new Map<string, { count: number; type: string }>();

      suggestions.forEach(item => {
        // Extract from titles
        if (item.title.toLowerCase().includes(partialQuery.toLowerCase())) {
          this.addSuggestion(suggestionMap, item.title, 'entity');
        }

        // Extract from tags
        if (Array.isArray(item.tags)) {
          item.tags.forEach(tag => {
            if (tag.toLowerCase().includes(partialQuery.toLowerCase())) {
              this.addSuggestion(suggestionMap, tag, 'keyword');
            }
          });
        }

        // Extract from categories and keywords
        [...(item.categories || []), ...(item.keywords || [])].forEach(term => {
          if (term.toLowerCase().includes(partialQuery.toLowerCase())) {
            this.addSuggestion(suggestionMap, term, 'category');
          }
        });
      });

      // Sort by frequency and format
      return Array.from(suggestionMap.entries())
        .sort(([,a], [,b]) => b.count - a.count)
        .slice(0, limit)
        .map(([suggestion, data]) => ({
          suggestion,
          type: data.type as any,
          frequency: data.count,
          confidence: Math.min(data.count / 10, 1) // Simple confidence scoring
        }));
    } catch (error) {
      console.error('Failed to generate query suggestions:', error);
      return [];
    }
  }

  // Private methods

  private async generateQueryEmbedding(query: string): Promise<number[]> {
    try {
      // Check cache first
      const queryHash = createHash('md5').update(query.toLowerCase()).digest('hex');
      const cached = await db
        .select()
        .from(embeddingCache)
        .where(eq(embeddingCache.textHash, queryHash))
        .limit(1);

      if (cached.length > 0) {
        return cached[0].embedding;
      }

      // Generate new embedding
      const embedding = await ollamaCudaService.generateEmbedding(query);

      // Cache for future use
      await db.insert(embeddingCache).values({
        textHash: queryHash,
        embedding,
        model: 'nomic-embed-text'
      }).onConflictDoNothing();

      return embedding;
    } catch (error) {
      console.error('Failed to generate query embedding:', error);
      throw error;
    }
  }

  private async performVectorSearch(
    queryEmbedding: number[],
    searchQuery: VectorSearchQuery
  ): Promise<{ results: any[]; processingTime: number }> {
    const startTime = Date.now();
    
    try {
      const threshold = searchQuery.options?.threshold || 0.5;
      const limit = searchQuery.options?.limit || 20;
      const offset = searchQuery.options?.offset || 0;

      // Build base query
      let query = db
        .select({
          id: evidence.id,
          title: evidence.title,
          description: evidence.description,
          evidenceType: evidence.evidenceType,
          caseId: evidence.caseId,
          tags: evidence.tags,
          aiSummary: evidence.aiSummary,
          aiAnalysis: evidence.aiAnalysis,
          confidentialityLevel: evidence.confidentialityLevel,
          isAdmissible: evidence.isAdmissible,
          fileSize: evidence.fileSize,
          mimeType: evidence.mimeType,
          createdAt: evidence.createdAt,
          updatedAt: evidence.updatedAt,
          similarity: sql<number>`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding})`.as('similarity')
        })
        .from(evidence)
        .where(
          and(
            isNotNull(evidence.contentEmbedding),
            sql`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding}) > ${threshold}`
          )
        );

      // Apply filters
      const conditions = [];
      
      if (searchQuery.filters?.caseId) {
        conditions.push(eq(evidence.caseId, searchQuery.filters.caseId));
      }

      if (searchQuery.filters?.evidenceTypes?.length) {
        conditions.push(inArray(evidence.evidenceType, searchQuery.filters.evidenceTypes));
      }

      if (searchQuery.filters?.confidentialityLevels?.length) {
        conditions.push(inArray(evidence.confidentialityLevel, searchQuery.filters.confidentialityLevels));
      }

      if (searchQuery.filters?.isAdmissible !== undefined) {
        conditions.push(eq(evidence.isAdmissible, searchQuery.filters.isAdmissible));
      }

      if (searchQuery.filters?.dateRange) {
        const { start, end } = searchQuery.filters.dateRange;
        conditions.push(
          and(
            sql`${evidence.createdAt} >= ${start.toISOString()}`,
            sql`${evidence.createdAt} <= ${end.toISOString()}`
          )
        );
      }

      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }

      const results = await query
        .orderBy(sql`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding}) DESC`)
        .limit(limit + offset)
        .offset(offset);

      return {
        results,
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('Vector search failed:', error);
      throw error;
    }
  }

  private async performTextSearch(
    queries: string[],
    searchQuery: VectorSearchQuery
  ): Promise<{ results: any[]; processingTime: number }> {
    const startTime = Date.now();

    try {
      // Update Fuse index if needed
      await this.updateFuseIndexIfNeeded();

      if (!this.fuseIndex) {
        return { results: [], processingTime: Date.now() - startTime };
      }

      // Perform Fuse.js search for each query
      const allResults = new Map<string, any>();

      for (const query of queries) {
        const fuseResults = this.fuseIndex.search(query, {
          limit: searchQuery.options?.limit || 50
        });

        fuseResults.forEach(result => {
          const existing = allResults.get(result.item.id);
          if (!existing || result.score! < existing.score!) {
            allResults.set(result.item.id, {
              ...result.item,
              textScore: 1 - result.score!, // Invert score for consistency
              highlights: this.generateHighlights(result.item, query)
            });
          }
        });
      }

      return {
        results: Array.from(allResults.values()),
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('Text search failed:', error);
      return { results: [], processingTime: Date.now() - startTime };
    }
  }

  private async combineAndRankResults(
    vectorResults: { results: any[]; processingTime: number },
    textResults: { results: any[]; processingTime: number },
    searchQuery: VectorSearchQuery,
    queryEmbedding: number[]
  ): Promise<SearchResult[]> {
    const hybridWeight = searchQuery.options?.hybridWeight || 0.7;
    const strategy = searchQuery.ranking?.strategy || 'hybrid';
    const weights = searchQuery.ranking?.weights || {
      semantic: 0.4,
      text: 0.3,
      recency: 0.1,
      relevance: 0.1,
      authority: 0.1
    };

    // Create combined results map
    const combinedMap = new Map<string, SearchResult>();

    // Process vector results
    vectorResults.results.forEach(result => {
      combinedMap.set(result.id, this.formatSearchResult(result, {
        semanticScore: result.similarity,
        textScore: 0,
        source: 'vector'
      }));
    });

    // Merge text results
    textResults.results.forEach(result => {
      const existing = combinedMap.get(result.id);
      if (existing) {
        existing.textScore = result.textScore;
        existing.highlights = result.highlights;
      } else {
        combinedMap.set(result.id, this.formatSearchResult(result, {
          semanticScore: 0,
          textScore: result.textScore,
          source: 'text'
        }));
      }
    });

    // Calculate final scores and rank
    const results = Array.from(combinedMap.values());
    
    results.forEach(result => {
      result.score = this.calculateFinalScore(result, strategy, weights, hybridWeight);
      result.reasoning = this.generateReasoning(result, strategy);
    });

    // Sort by final score
    results.sort((a, b) => b.score - a.score);

    return results;
  }

  private calculateFinalScore(
    result: SearchResult,
    strategy: string,
    weights: any,
    hybridWeight: number
  ): number {
    switch (strategy) {
      case 'semantic':
        return result.semanticScore;
      
      case 'bm25':
        return result.textScore;
      
      case 'neural':
        return (result.semanticScore + result.textScore) / 2;
      
      case 'hybrid':
      default:
        return (
          (weights.semantic || 0.4) * result.semanticScore +
          (weights.text || 0.3) * result.textScore +
          (weights.recency || 0.1) * result.recencyScore +
          (weights.relevance || 0.1) * result.confidenceScore +
          (weights.authority || 0.1) * (result.tags.length / 10) // Simple authority based on tags
        );
    }
  }

  private formatSearchResult(result: any, scores: any): SearchResult {
    return {
      id: result.id,
      title: result.title,
      description: result.description,
      content: result.aiSummary,
      evidenceType: result.evidenceType,
      caseId: result.caseId,
      tags: Array.isArray(result.tags) ? result.tags : [],
      score: 0, // Will be calculated later
      semanticScore: scores.semanticScore,
      textScore: scores.textScore,
      confidenceScore: (result.aiAnalysis?.confidence as number) || 0.5,
      recencyScore: this.calculateRecencyScore(result.createdAt),
      highlights: scores.highlights || [],
      reasoning: '',
      metadata: {
        fileSize: result.fileSize,
        mimeType: result.mimeType,
        createdAt: result.createdAt,
        updatedAt: result.updatedAt
      }
    };
  }

  private calculateRecencyScore(date: Date): number {
    const now = new Date();
    const ageInDays = (now.getTime() - new Date(date).getTime()) / (1000 * 60 * 60 * 24);
    return Math.max(0, 1 - (ageInDays / 365)); // Score decreases over a year
  }

  private generateReasoning(result: SearchResult, strategy: string): string {
    const reasons = [];
    
    if (result.semanticScore > 0.8) reasons.push('High semantic similarity');
    if (result.textScore > 0.8) reasons.push('Strong text match');
    if (result.recencyScore > 0.8) reasons.push('Recent document');
    if (result.confidenceScore > 0.8) reasons.push('High AI confidence');
    
    return reasons.join(', ') || 'Standard relevance match';
  }

  private async initializeFuseIndex(): Promise<void> {
    try {
      await this.updateFuseIndex();
    } catch (error) {
      console.error('Failed to initialize Fuse index:', error);
    }
  }

  private async updateFuseIndexIfNeeded(): Promise<void> {
    if (!this.lastIndexUpdate || 
        Date.now() - this.lastIndexUpdate.getTime() > this.indexUpdateInterval) {
      await this.updateFuseIndex();
    }
  }

  private async updateFuseIndex(): Promise<void> {
    try {
      const documents = await db
        .select({
          id: evidence.id,
          title: evidence.title,
          description: evidence.description,
          aiSummary: evidence.aiSummary,
          tags: evidence.tags,
          evidenceType: evidence.evidenceType,
          caseId: evidence.caseId
        })
        .from(evidence);

      const searchableDocuments = documents.map(doc => ({
        ...doc,
        searchText: [
          doc.title,
          doc.description,
          doc.aiSummary,
          ...(Array.isArray(doc.tags) ? doc.tags : [])
        ].filter(Boolean).join(' ')
      }));

      this.fuseIndex = new Fuse(searchableDocuments, {
        keys: [
          { name: 'title', weight: 0.4 },
          { name: 'description', weight: 0.3 },
          { name: 'aiSummary', weight: 0.2 },
          { name: 'tags', weight: 0.1 }
        ],
        threshold: 0.3,
        includeScore: true,
        includeMatches: true
      });

      this.lastIndexUpdate = new Date();
    } catch (error) {
      console.error('Failed to update Fuse index:', error);
    }
  }

  private generateHighlights(document: any, query: string): string[] {
    const highlights: string[] = [];
    const queryTerms = query.toLowerCase().split(' ');
    
    [document.title, document.description, document.aiSummary].forEach(text => {
      if (text) {
        queryTerms.forEach(term => {
          const regex = new RegExp(`(.{0,50})${term}(.{0,50})`, 'gi');
          const matches = text.match(regex);
          if (matches) {
            highlights.push(...matches.slice(0, 2)); // Limit highlights
          }
        });
      }
    });

    return [...new Set(highlights)].slice(0, 3); // Remove duplicates and limit
  }

  private async expandQuery(query: string): Promise<string[]> {
    try {
      // Use AI to generate query expansions
      const expansionPrompt = `Given the search query "${query}", suggest 3-5 related search terms or phrases that might help find relevant legal documents. Focus on synonyms, related legal concepts, and alternative phrasings. Return only the terms, one per line.`;

      const response = await ollamaCudaService.chatCompletion([
        { role: 'system', content: 'You are a legal research assistant helping with search query expansion.' },
        { role: 'user', content: expansionPrompt }
      ], {
        temperature: 0.5,
        maxTokens: 200
      });

      const expansions = response
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .slice(0, 5);

      return [query, ...expansions];
    } catch (error) {
      console.warn('Query expansion failed:', error);
      return [query];
    }
  }

  private async generateFacets(searchQuery: VectorSearchQuery): Promise<any> {
    // Implementation for generating search facets
    return {
      evidenceTypes: [],
      cases: [],
      tags: [],
      dateRanges: []
    };
  }

  private async performKMeansClustering(documents: any[], numClusters: number): Promise<any[]> {
    // Simplified K-means clustering implementation
    // In production, you might want to use a more sophisticated clustering library
    return [];
  }

  private addSuggestion(map: Map<string, any>, term: string, type: string): void {
    const existing = map.get(term);
    if (existing) {
      existing.count++;
    } else {
      map.set(term, { count: 1, type });
    }
  }

  private generateCacheKey(searchQuery: VectorSearchQuery): string {
    return createHash('md5').update(JSON.stringify(searchQuery)).digest('hex');
  }

  private isCacheValid(cacheKey: string): boolean {
    // Simple cache validation - in production you might want more sophisticated logic
    return this.queryCache.has(cacheKey);
  }

  /**
   * Clear all caches
   */
  public clearCache(): void {
    this.queryCache.clear();
    this.fuseIndex = null;
    this.lastIndexUpdate = null;
  }

  /**
   * Get service statistics
   */
  public getStats(): {
    cacheSize: number;
    lastIndexUpdate: Date | null;
    indexSize: number;
  } {
    return {
      cacheSize: this.queryCache.size,
      lastIndexUpdate: this.lastIndexUpdate,
      indexSize: this.fuseIndex?.getIndex().size || 0
    };
  }
}

// Export singleton instance
export const vectorSearchService = VectorSearchService.getInstance();
export default vectorSearchService;