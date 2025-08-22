
/**
 * Enhanced Semantic Search Engine with Fuse.js
 * Provides fuzzy search, semantic matching, and intelligent ranking
 */

import Fuse from 'fuse.js';
import { EventEmitter } from 'events';
import { CacheManager } from '../server/cache/loki-cache';

interface SearchDocument {
  id: string;
  title: string;
  content: string;
  metadata: {
    documentType: 'contract' | 'legal-brief' | 'case-law' | 'regulation' | 'memo' | 'evidence' | 'other';
    caseId?: string;
    userId?: string;
    confidence: number;
    processingDate: number;
    tags: string[];
    language: string;
    jurisdiction?: string;
    practiceArea?: string;
    
    // Legal-specific metadata
    legalConcepts?: string[];
    entities?: Array<{
      type: 'PERSON' | 'ORGANIZATION' | 'DATE' | 'MONEY' | 'LOCATION' | 'STATUTE';
      value: string;
      confidence: number;
    }>;
    citations?: Array<{
      type: 'case' | 'statute' | 'regulation';
      citation: string;
      relevance: number;
    }>;
    
    // Semantic embeddings
    embedding?: number[];
    semanticClusters?: string[];
  };
  
  // Search optimization
  searchableText: string; // Preprocessed text for faster searching
  wordCount: number;
  readability: number;
  importance: number; // Calculated importance score
}

interface SearchQuery {
  text: string;
  filters?: {
    documentType?: string[];
    dateRange?: { start: number; end: number };
    caseId?: string;
    userId?: string;
    tags?: string[];
    jurisdiction?: string;
    practiceArea?: string;
    minConfidence?: number;
  };
  options?: {
    fuzzyThreshold?: number;
    maxResults?: number;
    includeScore?: boolean;
    semanticSearch?: boolean;
    highlightMatches?: boolean;
    sortBy?: 'relevance' | 'date' | 'confidence' | 'importance';
    sortDirection?: 'asc' | 'desc';
  };
}

interface SearchResult {
  document: SearchDocument;
  score: number;
  matches: Array<{
    field: string;
    value: string;
    indices: Array<[number, number]>;
    highlighted?: string;
  }>;
  relevanceFactors: {
    textMatch: number;
    semanticSimilarity: number;
    metadataMatch: number;
    userContext: number;
    recency: number;
    importance: number;
    totalScore: number;
  };
  explanation: string;
}

interface SearchIndex {
  documents: Map<string, SearchDocument>;
  fuse: Fuse<SearchDocument>;
  lastUpdated: number;
  statistics: {
    totalDocuments: number;
    averageConfidence: number;
    documentTypes: Record<string, number>;
    topTags: Array<{ tag: string; count: number }>;
    practiceAreas: Record<string, number>;
  };
}

export class EnhancedSemanticSearch extends EventEmitter {
  private indices: Map<string, SearchIndex> = new Map(); // Multiple indices for different contexts
  private cache: CacheManager;
  private defaultIndex: SearchIndex;
  private config: {
    fuseOptions: Fuse.IFuseOptions<SearchDocument>;
    semanticThreshold: number;
    cacheTimeout: number;
    maxIndexSize: number;
  };

  constructor(options: {
    cacheManager?: CacheManager;
    semanticThreshold?: number;
    cacheTimeout?: number;
    maxIndexSize?: number;
  } = {}) {
    super();

    this.cache = options.cacheManager || new CacheManager();
    
    this.config = {
      fuseOptions: {
        keys: [
          { name: 'title', weight: 0.3 },
          { name: 'searchableText', weight: 0.4 },
          { name: 'metadata.tags', weight: 0.1 },
          { name: 'metadata.legalConcepts', weight: 0.15 },
          { name: 'metadata.practiceArea', weight: 0.05 }
        ],
        threshold: 0.3, // 0 = perfect match, 1 = match anything
        distance: 1000,
        minMatchCharLength: 3,
        includeScore: true,
        includeMatches: true,
        ignoreLocation: false,
        ignoreFieldNorm: false,
        findAllMatches: true
      },
      semanticThreshold: options.semanticThreshold || 0.7,
      cacheTimeout: options.cacheTimeout || 5 * 60 * 1000, // 5 minutes
      maxIndexSize: options.maxIndexSize || 10000
    };

    this.initializeDefaultIndex();
    
    console.log('🔍 Enhanced Semantic Search Engine initialized');
  }

  private initializeDefaultIndex(): void {
    this.defaultIndex = {
      documents: new Map(),
      fuse: new Fuse([], this.config.fuseOptions),
      lastUpdated: Date.now(),
      statistics: {
        totalDocuments: 0,
        averageConfidence: 0,
        documentTypes: {},
        topTags: [],
        practiceAreas: {}
      }
    };
    
    this.indices.set('default', this.defaultIndex);
  }

  // Document management
  public async addDocument(document: Omit<SearchDocument, 'searchableText' | 'wordCount' | 'readability' | 'importance'>): Promise<void> {
    const enrichedDocument: SearchDocument = {
      ...document,
      searchableText: this.preprocessText(document.title + ' ' + document.content),
      wordCount: this.countWords(document.content),
      readability: this.calculateReadability(document.content),
      importance: this.calculateImportance(document)
    };

    const indexName = this.getIndexForDocument(enrichedDocument);
    const index = this.indices.get(indexName) || this.defaultIndex;

    // Add to document map
    index.documents.set(document.id, enrichedDocument);

    // Update Fuse index
    const documentsArray = Array.from(index.documents.values());
    index.fuse.setCollection(documentsArray);
    
    // Update statistics
    this.updateIndexStatistics(index);
    
    // Cache for faster retrieval
    await this.cache.set(`doc_${document.id}`, enrichedDocument, {
      contentType: 'search-document',
      confidence: document.metadata.confidence,
      tags: ['search-index', document.metadata.documentType],
      ttl: this.config.cacheTimeout
    });

    console.log(`📄 Added document ${document.id} to search index (${indexName})`);
    this.emit('document-added', { documentId: document.id, index: indexName });
  }

  public async addDocuments(documents: Array<Omit<SearchDocument, 'searchableText' | 'wordCount' | 'readability' | 'importance'>>): Promise<void> {
    const enrichedDocuments = documents.map(doc => ({
      ...doc,
      searchableText: this.preprocessText(doc.title + ' ' + doc.content),
      wordCount: this.countWords(doc.content),
      readability: this.calculateReadability(doc.content),
      importance: this.calculateImportance(doc)
    }));

    // Group documents by index
    const documentsByIndex = new Map<string, SearchDocument[]>();
    
    enrichedDocuments.forEach(doc => {
      const indexName = this.getIndexForDocument(doc);
      if (!documentsByIndex.has(indexName)) {
        documentsByIndex.set(indexName, []);
      }
      documentsByIndex.get(indexName)!.push(doc);
    });

    // Update each index
    for (const [indexName, docs] of documentsByIndex) {
      const index = this.indices.get(indexName) || this.defaultIndex;
      
      // Add documents
      docs.forEach(doc => {
        index.documents.set(doc.id, doc);
      });

      // Update Fuse index
      const documentsArray = Array.from(index.documents.values());
      index.fuse.setCollection(documentsArray);
      
      // Update statistics
      this.updateIndexStatistics(index);
    }

    // Cache documents
    const cachePromises = enrichedDocuments.map(doc =>
      this.cache.set(`doc_${doc.id}`, doc, {
        contentType: 'search-document',
        confidence: doc.metadata.confidence,
        tags: ['search-index', doc.metadata.documentType],
        ttl: this.config.cacheTimeout
      })
    );

    await Promise.all(cachePromises);

    console.log(`📚 Added ${documents.length} documents to search indices`);
    this.emit('documents-added', { count: documents.length, indices: Array.from(documentsByIndex.keys()) });
  }

  public async removeDocument(documentId: string): Promise<boolean> {
    let removed = false;
    
    for (const [indexName, index] of this.indices) {
      if (index.documents.has(documentId)) {
        index.documents.delete(documentId);
        
        // Update Fuse index
        const documentsArray = Array.from(index.documents.values());
        index.fuse.setCollection(documentsArray);
        
        // Update statistics
        this.updateIndexStatistics(index);
        
        removed = true;
        console.log(`🗑️ Removed document ${documentId} from index ${indexName}`);
      }
    }

    // Remove from cache
    await this.cache.clear(`doc_${documentId}`);
    
    if (removed) {
      this.emit('document-removed', { documentId });
    }
    
    return removed;
  }

  // Core search functionality
  public async search(query: SearchQuery): Promise<SearchResult[]> {
    const startTime = Date.now();
    
    console.log(`🔍 Searching for: "${query.text}"`);
    
    // Check cache first
    const cacheKey = this.generateCacheKey(query);
    const cached = await this.cache.get(cacheKey);
    
    if (cached) {
      console.log(`📦 Using cached search results`);
      return cached;
    }

    // Determine which indices to search
    const indicesToSearch = this.getIndicesForQuery(query);
    
    // Perform search across relevant indices
    const allResults: SearchResult[] = [];
    
    for (const indexName of indicesToSearch) {
      const index = this.indices.get(indexName);
      if (!index) continue;

      const indexResults = await this.searchInIndex(index, query);
      allResults.push(...indexResults);
    }

    // Apply filters
    const filteredResults = this.applyFilters(allResults, query.filters);
    
    // Enhance results with semantic similarity if requested
    let enhancedResults = filteredResults;
    if (query.options?.semanticSearch) {
      enhancedResults = await this.enhanceWithSemanticSimilarity(filteredResults, query);
    }

    // Sort results
    const sortedResults = this.sortResults(enhancedResults, query.options);
    
    // Limit results
    const maxResults = query.options?.maxResults || 20;
    const finalResults = sortedResults.slice(0, maxResults);

    // Add highlighting if requested
    if (query.options?.highlightMatches) {
      finalResults.forEach(result => {
        this.addHighlighting(result, query.text);
      });
    }

    // Cache results
    await this.cache.set(cacheKey, finalResults, {
      contentType: 'search-results',
      confidence: finalResults.length > 0 ? finalResults[0].score : 0,
      tags: ['search-results', 'semantic-search'],
      ttl: this.config.cacheTimeout
    });

    const processingTime = Date.now() - startTime;
    console.log(`✅ Search completed in ${processingTime}ms, found ${finalResults.length} results`);
    
    this.emit('search-completed', {
      query: query.text,
      resultCount: finalResults.length,
      processingTime,
      cacheHit: false
    });

    return finalResults;
  }

  private async searchInIndex(index: SearchIndex, query: SearchQuery): Promise<SearchResult[]> {
    // Configure Fuse options based on query
    const searchOptions: Fuse.IFuseOptions<SearchDocument> = {
      ...this.config.fuseOptions,
      threshold: query.options?.fuzzyThreshold || this.config.fuseOptions.threshold
    };

    // Create temporary Fuse instance with custom options
    const fuseInstance = new Fuse(Array.from(index.documents.values()), searchOptions);
    
    // Perform search
    const fuseResults = fuseInstance.search(query.text);
    
    // Convert to SearchResult format
    const results: SearchResult[] = fuseResults.map(fuseResult => {
      const document = fuseResult.item;
      const score = 1 - (fuseResult.score || 0); // Invert Fuse score (lower is better -> higher is better)
      
      // Calculate relevance factors
      const relevanceFactors = this.calculateRelevanceFactors(document, query, score);
      
      return {
        document,
        score: relevanceFactors.totalScore,
        matches: fuseResult.matches?.map(match => ({
          field: match.key || '',
          value: match.value || '',
          indices: match.indices || [],
          highlighted: undefined // Will be added later if requested
        })) || [],
        relevanceFactors,
        explanation: this.generateExplanation(relevanceFactors, document, query)
      };
    });

    return results;
  }

  // Filtering and ranking
  private applyFilters(results: SearchResult[], filters?: SearchQuery['filters']): SearchResult[] {
    if (!filters) return results;

    return results.filter(result => {
      const doc = result.document;
      const meta = doc.metadata;

      // Document type filter
      if (filters.documentType && filters.documentType.length > 0) {
        if (!filters.documentType.includes(meta.documentType)) return false;
      }

      // Date range filter
      if (filters.dateRange) {
        const docDate = meta.processingDate;
        if (docDate < filters.dateRange.start || docDate > filters.dateRange.end) {
          return false;
        }
      }

      // Case ID filter
      if (filters.caseId && meta.caseId !== filters.caseId) {
        return false;
      }

      // User ID filter
      if (filters.userId && meta.userId !== filters.userId) {
        return false;
      }

      // Tags filter (any tag matches)
      if (filters.tags && filters.tags.length > 0) {
        const hasMatchingTag = filters.tags.some(tag => meta.tags.includes(tag));
        if (!hasMatchingTag) return false;
      }

      // Jurisdiction filter
      if (filters.jurisdiction && meta.jurisdiction !== filters.jurisdiction) {
        return false;
      }

      // Practice area filter
      if (filters.practiceArea && meta.practiceArea !== filters.practiceArea) {
        return false;
      }

      // Minimum confidence filter
      if (filters.minConfidence && meta.confidence < filters.minConfidence) {
        return false;
      }

      return true;
    });
  }

  private async enhanceWithSemanticSimilarity(results: SearchResult[], query: SearchQuery): Promise<SearchResult[]> {
    // This would integrate with actual embedding models in production
    // For now, we'll use a simplified approach based on content similarity
    
    return results.map(result => {
      const semanticScore = this.calculateSemanticSimilarity(result.document, query);
      
      // Blend semantic score with existing score
      const blendedScore = (result.score * 0.7) + (semanticScore * 0.3);
      
      return {
        ...result,
        score: blendedScore,
        relevanceFactors: {
          ...result.relevanceFactors,
          semanticSimilarity: semanticScore,
          totalScore: blendedScore
        }
      };
    });
  }

  private sortResults(results: SearchResult[], options?: SearchQuery['options']): SearchResult[] {
    const sortBy = options?.sortBy || 'relevance';
    const direction = options?.sortDirection || 'desc';
    const multiplier = direction === 'asc' ? 1 : -1;

    return results.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'relevance':
          comparison = a.score - b.score;
          break;
        case 'date':
          comparison = a.document.metadata.processingDate - b.document.metadata.processingDate;
          break;
        case 'confidence':
          comparison = a.document.metadata.confidence - b.document.metadata.confidence;
          break;
        case 'importance':
          comparison = a.document.importance - b.document.importance;
          break;
      }

      return comparison * multiplier;
    });
  }

  // Relevance calculation
  private calculateRelevanceFactors(
    document: SearchDocument,
    query: SearchQuery,
    fuseScore: number
  ): SearchResult['relevanceFactors'] {
    const factors = {
      textMatch: fuseScore,
      semanticSimilarity: 0.5, // Placeholder - would use embeddings in production
      metadataMatch: 0,
      userContext: 0,
      recency: 0,
      importance: 0,
      totalScore: 0
    };

    // Metadata match scoring
    if (query.filters) {
      let metadataScore = 0;
      let metadataFactors = 0;

      if (query.filters.documentType?.includes(document.metadata.documentType)) {
        metadataScore += 0.2;
        metadataFactors++;
      }

      if (query.filters.practiceArea === document.metadata.practiceArea) {
        metadataScore += 0.15;
        metadataFactors++;
      }

      if (query.filters.tags?.some(tag => document.metadata.tags.includes(tag))) {
        metadataScore += 0.1;
        metadataFactors++;
      }

      factors.metadataMatch = metadataFactors > 0 ? metadataScore / metadataFactors : 0;
    }

    // Recency scoring (newer documents get higher scores)
    const daysSinceProcessing = (Date.now() - document.metadata.processingDate) / (24 * 60 * 60 * 1000);
    factors.recency = Math.max(0, 1 - (daysSinceProcessing / 365)); // Decay over a year

    // Importance scoring
    factors.importance = document.importance;

    // User context scoring (placeholder - would use user history in production)
    factors.userContext = 0.5;

    // Calculate total score with weights
    factors.totalScore = (
      factors.textMatch * 0.4 +
      factors.semanticSimilarity * 0.25 +
      factors.metadataMatch * 0.15 +
      factors.recency * 0.1 +
      factors.importance * 0.05 +
      factors.userContext * 0.05
    );

    return factors;
  }

  private calculateSemanticSimilarity(document: SearchDocument, query: SearchQuery): number {
    // Simplified semantic similarity - would use embeddings in production
    const queryTerms = query.text.toLowerCase().split(/\s+/);
    const docTerms = document.searchableText.toLowerCase().split(/\s+/);
    
    // Calculate Jaccard similarity
    const querySet = new Set(queryTerms);
    const docSet = new Set(docTerms);
    
    const intersection = new Set([...querySet].filter(term => docSet.has(term)));
    const union = new Set([...querySet, ...docSet]);
    
    return intersection.size / union.size;
  }

  // Utility methods
  private preprocessText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();
  }

  private countWords(text: string): number {
    return text.split(/\s+/).filter(word => word.length > 0).length;
  }

  private calculateReadability(text: string): number {
    // Simplified Flesch Reading Ease calculation
    const words = this.countWords(text);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    const syllables = this.countSyllables(text);
    
    if (words === 0 || sentences === 0) return 0;
    
    const averageWordsPerSentence = words / sentences;
    const averageSyllablesPerWord = syllables / words;
    
    return 206.835 - (1.015 * averageWordsPerSentence) - (84.6 * averageSyllablesPerWord);
  }

  private countSyllables(text: string): number {
    // Simplified syllable counting
    return text.toLowerCase().replace(/[^a-z]/g, '').replace(/[aeiouy]+/g, 'a').length;
  }

  private calculateImportance(document: Omit<SearchDocument, 'searchableText' | 'wordCount' | 'readability' | 'importance'>): number {
    let importance = 0.5; // Base importance

    // Document type importance
    const typeWeights = {
      'case-law': 0.9,
      'regulation': 0.8,
      'contract': 0.7,
      'legal-brief': 0.6,
      'memo': 0.4,
      'evidence': 0.5,
      'other': 0.3
    };
    importance += (typeWeights[document.metadata.documentType] || 0.3) * 0.3;

    // Confidence importance
    importance += document.metadata.confidence * 0.2;

    // Entity count importance
    const entityCount = document.metadata.entities?.length || 0;
    importance += Math.min(entityCount / 20, 0.1); // Cap at 0.1 for 20+ entities

    // Citation count importance
    const citationCount = document.metadata.citations?.length || 0;
    importance += Math.min(citationCount / 10, 0.1); // Cap at 0.1 for 10+ citations

    return Math.min(importance, 1.0);
  }

  private getIndexForDocument(document: SearchDocument): string {
    // Route documents to specific indices based on characteristics
    if (document.metadata.caseId) {
      return `case_${document.metadata.caseId}`;
    }
    
    if (document.metadata.practiceArea) {
      return `practice_${document.metadata.practiceArea}`;
    }
    
    return 'default';
  }

  private getIndicesForQuery(query: SearchQuery): string[] {
    const indices = ['default'];
    
    if (query.filters?.caseId) {
      indices.push(`case_${query.filters.caseId}`);
    }
    
    if (query.filters?.practiceArea) {
      indices.push(`practice_${query.filters.practiceArea}`);
    }
    
    // Remove duplicates
    return Array.from(new Set(indices));
  }

  private updateIndexStatistics(index: SearchIndex): void {
    const documents = Array.from(index.documents.values());
    
    index.statistics.totalDocuments = documents.length;
    
    if (documents.length > 0) {
      index.statistics.averageConfidence = 
        documents.reduce((sum, doc) => sum + doc.metadata.confidence, 0) / documents.length;
    }

    // Document types
    index.statistics.documentTypes = {};
    documents.forEach(doc => {
      const type = doc.metadata.documentType;
      index.statistics.documentTypes[type] = (index.statistics.documentTypes[type] || 0) + 1;
    });

    // Top tags
    const tagCounts = new Map<string, number>();
    documents.forEach(doc => {
      doc.metadata.tags.forEach(tag => {
        tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
      });
    });

    index.statistics.topTags = Array.from(tagCounts.entries())
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 20);

    // Practice areas
    index.statistics.practiceAreas = {};
    documents.forEach(doc => {
      if (doc.metadata.practiceArea) {
        const area = doc.metadata.practiceArea;
        index.statistics.practiceAreas[area] = (index.statistics.practiceAreas[area] || 0) + 1;
      }
    });

    index.lastUpdated = Date.now();
  }

  private generateCacheKey(query: SearchQuery): string {
    const filterKey = query.filters ? JSON.stringify(query.filters) : '';
    const optionsKey = query.options ? JSON.stringify(query.options) : '';
    
    return `search_${btoa(query.text)}_${btoa(filterKey)}_${btoa(optionsKey)}`;
  }

  private addHighlighting(result: SearchResult, queryText: string): void {
    const terms = queryText.toLowerCase().split(/\s+/);
    
    result.matches.forEach(match => {
      if (match.value) {
        let highlighted = match.value;
        
        terms.forEach(term => {
          const regex = new RegExp(`(${term})`, 'gi');
          highlighted = highlighted.replace(regex, '<mark>$1</mark>');
        });
        
        match.highlighted = highlighted;
      }
    });
  }

  private generateExplanation(
    factors: SearchResult['relevanceFactors'],
    document: SearchDocument,
    query: SearchQuery
  ): string {
    const explanations = [];

    if (factors.textMatch > 0.8) {
      explanations.push('Strong text match with query terms');
    } else if (factors.textMatch > 0.5) {
      explanations.push('Good text match with query terms');
    }

    if (factors.metadataMatch > 0.5) {
      explanations.push('Matches specified filters and metadata');
    }

    if (factors.recency > 0.8) {
      explanations.push('Recently processed document');
    }

    if (document.importance > 0.8) {
      explanations.push('High-importance document');
    }

    if (document.metadata.confidence > 0.9) {
      explanations.push('High-confidence AI analysis');
    }

    return explanations.length > 0 
      ? explanations.join('; ')
      : 'Basic relevance match';
  }

  // Public API methods
  public async getDocumentById(documentId: string): Promise<SearchDocument | null> {
    // Check cache first
    const cached = await this.cache.get(`doc_${documentId}`);
    if (cached) return cached;

    // Search in all indices
    for (const index of this.indices.values()) {
      if (index.documents.has(documentId)) {
        return index.documents.get(documentId)!;
      }
    }

    return null;
  }

  public async suggestQueries(partialQuery: string, limit: number = 5): Promise<string[]> {
    // Simple query suggestion based on existing document content
    const suggestions = new Set<string>();
    
    for (const index of this.indices.values()) {
      const documents = Array.from(index.documents.values());
      
      documents.forEach(doc => {
        // Extract potential query suggestions from titles and tags
        const title = doc.title.toLowerCase();
        const tags = doc.metadata.tags.map(tag => tag.toLowerCase());
        
        if (title.includes(partialQuery.toLowerCase())) {
          suggestions.add(doc.title);
        }
        
        tags.forEach(tag => {
          if (tag.includes(partialQuery.toLowerCase())) {
            suggestions.add(tag);
          }
        });
      });
      
      if (suggestions.size >= limit) break;
    }

    return Array.from(suggestions).slice(0, limit);
  }

  public getStatistics(): unknown {
    const overallStats = {
      totalIndices: this.indices.size,
      totalDocuments: 0,
      averageConfidence: 0,
      documentTypes: {} as Record<string, number>,
      topTags: [] as Array<{ tag: string; count: number }>,
      practiceAreas: {} as Record<string, number>
    };

    // Aggregate statistics from all indices
    for (const [indexName, index] of this.indices) {
      overallStats.totalDocuments += index.statistics.totalDocuments;
      
      // Merge document types
      for (const [type, count] of Object.entries(index.statistics.documentTypes)) {
        overallStats.documentTypes[type] = (overallStats.documentTypes[type] || 0) + count;
      }
      
      // Merge practice areas
      for (const [area, count] of Object.entries(index.statistics.practiceAreas)) {
        overallStats.practiceAreas[area] = (overallStats.practiceAreas[area] || 0) + count;
      }
    }

    // Calculate overall average confidence
    let totalConfidence = 0;
    let documentCount = 0;
    
    for (const index of this.indices.values()) {
      totalConfidence += index.statistics.averageConfidence * index.statistics.totalDocuments;
      documentCount += index.statistics.totalDocuments;
    }
    
    overallStats.averageConfidence = documentCount > 0 ? totalConfidence / documentCount : 0;

    return overallStats;
  }

  public async clearIndex(indexName?: string): Promise<number> {
    if (indexName) {
      const index = this.indices.get(indexName);
      if (index) {
        const count = index.documents.size;
        index.documents.clear();
        index.fuse.setCollection([]);
        this.updateIndexStatistics(index);
        console.log(`🗑️ Cleared ${count} documents from index ${indexName}`);
        return count;
      }
      return 0;
    } else {
      // Clear all indices
      let totalCount = 0;
      for (const [name, index] of this.indices) {
        totalCount += index.documents.size;
        index.documents.clear();
        index.fuse.setCollection([]);
        this.updateIndexStatistics(index);
      }
      console.log(`🗑️ Cleared ${totalCount} documents from all indices`);
      return totalCount;
    }
  }

  public async shutdown(): Promise<void> {
    console.log('🔄 Shutting down Enhanced Semantic Search...');
    
    // Clear all indices
    this.indices.clear();
    
    console.log('✅ Enhanced Semantic Search shutdown complete');
  }
}