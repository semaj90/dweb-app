/**
 * Unified Loki.js Fuzzy Search Integration
 * Connects all Loki services with Fuse.js-style fuzzy search capabilities
 */

import { fuseLazySearch, type SearchableItem, type SearchResult, type SearchOptions } from './fuse-lazy-search-indexeddb.js';
import { cacheManager } from '../server/cache/loki-cache.js';
import { lokiEvidenceService } from '../utils/loki-evidence.js';
import { loki } from '../stores/lokiStore.js';
import { lokiRedisCache, type CachedDocument } from '../cache/loki-redis-integration.js';
import { EventEmitter } from 'events';

interface UnifiedSearchOptions extends SearchOptions {
  sources?: Array<'cache' | 'evidence' | 'store' | 'redis' | 'all'>;
  includeMetadata?: boolean;
  legalContextOnly?: boolean;
  prioritizeRecent?: boolean;
}

interface UnifiedSearchResult extends SearchResult {
  source: 'cache' | 'evidence' | 'store' | 'redis';
  lokiId?: number;
  cacheLocation?: 'loki' | 'redis' | 'nes';
  confidence?: number;
  riskLevel?: string;
}

/**
 * Unified fuzzy search across all Loki.js services
 */
export class UnifiedLokiFuzzySearch extends EventEmitter {
  private isInitialized = false;
  private indexingInProgress = false;
  private lastIndexTime = 0;
  private searchStats = {
    totalQueries: 0,
    averageResponseTime: 0,
    sourcesQueried: { cache: 0, evidence: 0, store: 0, redis: 0 },
    totalResults: 0
  };

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🔍 Initializing Unified Loki Fuzzy Search...');

    try {
      // Initialize all underlying services
      await Promise.all([
        fuseLazySearch.initialize(),
        lokiRedisCache.initialize().catch(() => console.warn('Redis cache unavailable')),
        loki.init().catch(() => console.warn('Loki store unavailable'))
      ]);

      // Perform initial indexing
      await this.reindexAllSources();

      this.isInitialized = true;
      this.emit('initialized');

      console.log('✅ Unified Loki Fuzzy Search initialized');
    } catch (error) {
      console.error('❌ Failed to initialize unified search:', error);
      throw error;
    }
  }

  /**
   * Unified search across all Loki sources with fuzzy matching
   */
  async search(query: string, options: UnifiedSearchOptions = {}): Promise<UnifiedSearchResult[]> {
    const startTime = Date.now();
    await this.initialize();

    const searchOptions: UnifiedSearchOptions = {
      sources: ['all'],
      threshold: 0.4,
      maxResults: 50,
      useEmbeddings: true,
      includeMetadata: true,
      legalContextOnly: false,
      prioritizeRecent: true,
      ...options
    };

    console.log(`🔍 Unified search: "${query}" across ${JSON.stringify(searchOptions.sources)}`);

    try {
      // Determine which sources to search
      const sourcesToSearch = this.determineSources(searchOptions.sources);

      // Search all sources in parallel
      const searchPromises = sourcesToSearch.map(source => 
        this.searchSource(source, query, searchOptions)
      );

      const sourceResults = await Promise.all(searchPromises);
      
      // Combine and deduplicate results
      const allResults = sourceResults.flat();
      const unifiedResults = this.combineAndRankResults(allResults, query, searchOptions);

      // Update statistics
      const responseTime = Date.now() - startTime;
      this.updateSearchStats(query, unifiedResults, sourceResults, responseTime);

      console.log(`📊 Search complete: ${unifiedResults.length} results in ${responseTime}ms`);
      this.emit('searchComplete', { query, results: unifiedResults, responseTime });

      return unifiedResults;

    } catch (error) {
      console.error('❌ Unified search failed:', error);
      return [];
    }
  }

  /**
   * Search specific source with fuzzy matching
   */
  private async searchSource(
    source: 'cache' | 'evidence' | 'store' | 'redis', 
    query: string, 
    options: UnifiedSearchOptions
  ): Promise<UnifiedSearchResult[]> {
    this.searchStats.sourcesQueried[source]++;

    try {
      switch (source) {
        case 'cache':
          return await this.searchCacheManager(query, options);
        
        case 'evidence':
          return await this.searchEvidenceService(query, options);
        
        case 'store':
          return await this.searchLokiStore(query, options);
        
        case 'redis':
          return await this.searchRedisCache(query, options);
        
        default:
          return [];
      }
    } catch (error) {
      console.warn(`⚠️ Search failed for source ${source}:`, error);
      return [];
    }
  }

  private async searchCacheManager(query: string, options: UnifiedSearchOptions): Promise<UnifiedSearchResult[]> {
    try {
      // Search through cache manager's semantic search
      const cacheResults = await cacheManager.semanticSearch(query, options.maxResults || 50);
      
      return cacheResults.map(entry => ({
        item: {
          id: entry.key,
          title: entry.metadata?.contentType || 'Cached Result',
          content: typeof entry.data === 'string' ? entry.data : JSON.stringify(entry.data),
          keywords: entry.metadata?.tags || [],
          metadata: entry.metadata,
          timestamp: entry.metadata?.created
        },
        source: 'cache' as const,
        score: 1 - (entry.metadata?.confidence || 0.5),
        confidence: entry.metadata?.confidence,
        refIndex: 0
      }));
    } catch (error) {
      console.warn('Cache manager search failed:', error);
      return [];
    }
  }

  private async searchEvidenceService(query: string, options: UnifiedSearchOptions): Promise<UnifiedSearchResult[]> {
    try {
      if (!lokiEvidenceService.isReady()) {
        console.warn('Evidence service not ready');
        return [];
      }

      const evidenceResults = lokiEvidenceService.searchEvidence(query);
      
      return evidenceResults.map((evidence, index) => ({
        item: {
          id: evidence.id,
          title: evidence.fileName || evidence.id,
          content: evidence.description || '',
          keywords: evidence.tags || [],
          metadata: {
            type: evidence.type,
            caseId: evidence.caseId,
            timeline: evidence.timeline
          },
          timestamp: evidence.timeline?.createdAt ? new Date(evidence.timeline.createdAt).getTime() : Date.now()
        },
        source: 'evidence' as const,
        lokiId: (evidence as any).$loki,
        score: this.calculateEvidenceRelevance(evidence, query),
        refIndex: index
      }));
    } catch (error) {
      console.warn('Evidence service search failed:', error);
      return [];
    }
  }

  private async searchLokiStore(query: string, options: UnifiedSearchOptions): Promise<UnifiedSearchResult[]> {
    try {
      const results: UnifiedSearchResult[] = [];

      // Search evidence in store
      const evidenceResults = loki.evidence.search(query);
      evidenceResults.forEach((evidence, index) => {
        results.push({
          item: {
            id: evidence.id,
            title: evidence.fileName || evidence.id,
            content: evidence.description || '',
            keywords: evidence.tags || [],
            metadata: { source: 'loki-store-evidence', type: evidence.type },
            timestamp: evidence.timeline?.createdAt ? new Date(evidence.timeline.createdAt).getTime() : Date.now()
          },
          source: 'store' as const,
          lokiId: (evidence as any).$loki,
          score: this.calculateStoreRelevance(evidence, query),
          refIndex: index
        });
      });

      // Search notes in store
      const noteResults = loki.notes.search(query);
      noteResults.forEach((note: any, index) => {
        results.push({
          item: {
            id: note.id,
            title: note.title || 'Note',
            content: note.content || '',
            keywords: note.tags || [],
            metadata: { source: 'loki-store-notes', reportId: note.reportId },
            timestamp: note.createdAt ? new Date(note.createdAt).getTime() : Date.now()
          },
          source: 'store' as const,
          lokiId: (note as any).$loki,
          score: 0.5, // Default relevance for notes
          refIndex: index + evidenceResults.length
        });
      });

      return results;
    } catch (error) {
      console.warn('Loki store search failed:', error);
      return [];
    }
  }

  private async searchRedisCache(query: string, options: UnifiedSearchOptions): Promise<UnifiedSearchResult[]> {
    try {
      const searchResults = await lokiRedisCache.searchDocuments(query, {
        type: options.legalContextOnly ? ['contract', 'evidence', 'brief'] : undefined
      }, {
        limit: options.maxResults || 50,
        useSemanticSearch: options.useEmbeddings,
        cacheResults: true
      });

      return searchResults.map((result, index) => ({
        item: {
          id: result.document.id,
          title: result.document.id,
          content: result.document.metadata?.content || '',
          keywords: [],
          metadata: result.document.metadata,
          timestamp: (result.document as CachedDocument).cacheTimestamp
        },
        source: 'redis' as const,
        cacheLocation: (result.document as CachedDocument).cacheLocation,
        confidence: result.document.confidenceLevel,
        riskLevel: result.document.riskLevel,
        score: result.score,
        refIndex: index
      }));
    } catch (error) {
      console.warn('Redis cache search failed:', error);
      return [];
    }
  }

  /**
   * Combine and rank results from multiple sources
   */
  private combineAndRankResults(
    results: UnifiedSearchResult[], 
    query: string, 
    options: UnifiedSearchOptions
  ): UnifiedSearchResult[] {
    // Deduplicate by ID
    const uniqueResults = new Map<string, UnifiedSearchResult>();

    for (const result of results) {
      const existing = uniqueResults.get(result.item.id);
      if (!existing || (result.score || 1) < (existing.score || 1)) {
        uniqueResults.set(result.item.id, result);
      }
    }

    let finalResults = Array.from(uniqueResults.values());

    // Apply enhanced ranking
    finalResults = finalResults.map(result => ({
      ...result,
      enhancedScore: this.calculateEnhancedScore(result, query, options)
    }));

    // Sort by enhanced score
    finalResults.sort((a, b) => (a.enhancedScore || 1) - (b.enhancedScore || 1));

    // Apply filters and limits
    if (options.legalContextOnly) {
      finalResults = finalResults.filter(result => 
        this.isLegalContext(result.item.content, result.item.keywords)
      );
    }

    return finalResults.slice(0, options.maxResults || 50);
  }

  private calculateEnhancedScore(result: UnifiedSearchResult, query: string, options: UnifiedSearchOptions): number {
    let score = result.score || 1;

    // Source priority weighting
    const sourceWeights = { redis: 0.9, cache: 0.95, evidence: 0.85, store: 1.0 };
    score *= sourceWeights[result.source];

    // Confidence boost
    if (result.confidence) {
      score *= (1 - result.confidence * 0.2);
    }

    // Risk level consideration
    if (result.riskLevel) {
      const riskWeights = { critical: 0.7, high: 0.8, medium: 0.9, low: 1.0 };
      score *= riskWeights[result.riskLevel as keyof typeof riskWeights] || 1.0;
    }

    // Recent content boost
    if (options.prioritizeRecent && result.item.timestamp) {
      const age = Date.now() - result.item.timestamp;
      const daysSince = age / (1000 * 60 * 60 * 24);
      const recencyBoost = Math.max(0.8, 1 - daysSince * 0.01);
      score *= recencyBoost;
    }

    // Cache location optimization
    if (result.cacheLocation) {
      const locationWeights = { loki: 0.9, redis: 0.95, nes: 1.0 };
      score *= locationWeights[result.cacheLocation];
    }

    return score;
  }

  private calculateEvidenceRelevance(evidence: any, query: string): number {
    const queryLower = query.toLowerCase();
    let score = 1.0;

    // Title/filename match
    if (evidence.fileName && evidence.fileName.toLowerCase().includes(queryLower)) {
      score *= 0.7;
    }

    // Description match
    if (evidence.description && evidence.description.toLowerCase().includes(queryLower)) {
      score *= 0.8;
    }

    // Tag matches
    if (evidence.tags && evidence.tags.some((tag: string) => tag.toLowerCase().includes(queryLower))) {
      score *= 0.75;
    }

    // Type relevance
    if (evidence.type && evidence.type.toLowerCase().includes(queryLower)) {
      score *= 0.85;
    }

    return Math.max(0.1, Math.min(1.0, score));
  }

  private calculateStoreRelevance(item: any, query: string): number {
    // Similar to evidence relevance but with different weights
    return this.calculateEvidenceRelevance(item, query);
  }

  private isLegalContext(content: string, keywords: string[]): boolean {
    const legalTerms = [
      'legal', 'court', 'case', 'evidence', 'contract', 'liability',
      'plaintiff', 'defendant', 'attorney', 'law', 'statute', 'regulation',
      'jurisdiction', 'precedent', 'trial', 'hearing', 'motion'
    ];

    const contentLower = content.toLowerCase();
    const keywordLower = keywords.map(k => k.toLowerCase());

    return legalTerms.some(term => 
      contentLower.includes(term) || keywordLower.includes(term)
    );
  }

  private determineSources(sources: Array<'cache' | 'evidence' | 'store' | 'redis' | 'all'> | undefined): Array<'cache' | 'evidence' | 'store' | 'redis'> {
    if (!sources || sources.includes('all')) {
      return ['cache', 'evidence', 'store', 'redis'];
    }
    return sources.filter(s => s !== 'all') as Array<'cache' | 'evidence' | 'store' | 'redis'>;
  }

  private updateSearchStats(query: string, results: UnifiedSearchResult[], sourceResults: UnifiedSearchResult[][], responseTime: number): void {
    this.searchStats.totalQueries++;
    this.searchStats.totalResults += results.length;
    
    // Update average response time
    const totalTime = this.searchStats.averageResponseTime * (this.searchStats.totalQueries - 1) + responseTime;
    this.searchStats.averageResponseTime = totalTime / this.searchStats.totalQueries;
  }

  /**
   * Reindex all sources for improved search performance
   */
  async reindexAllSources(): Promise<void> {
    if (this.indexingInProgress) {
      console.log('📝 Indexing already in progress, skipping...');
      return;
    }

    this.indexingInProgress = true;
    console.log('📝 Starting unified reindexing...');

    try {
      const startTime = Date.now();
      let totalIndexed = 0;

      // Index cache manager data
      try {
        const cacheEntries = await cacheManager.getByContentType('general', 1000);
        for (const entry of cacheEntries) {
          await this.indexCacheEntry(entry);
          totalIndexed++;
        }
      } catch (error) {
        console.warn('Cache indexing failed:', error);
      }

      // Index evidence service data
      try {
        if (lokiEvidenceService.isReady()) {
          const evidenceItems = lokiEvidenceService.getAllEvidence();
          for (const evidence of evidenceItems) {
            await this.indexEvidenceItem(evidence);
            totalIndexed++;
          }
        }
      } catch (error) {
        console.warn('Evidence indexing failed:', error);
      }

      // Index Loki store data
      try {
        const storeEvidence = loki.evidence.getAll();
        const storeNotes = loki.notes.getAll();
        
        for (const item of [...storeEvidence, ...storeNotes]) {
          await this.indexStoreItem(item);
          totalIndexed++;
        }
      } catch (error) {
        console.warn('Store indexing failed:', error);
      }

      const indexTime = Date.now() - startTime;
      this.lastIndexTime = Date.now();

      console.log(`✅ Reindexing complete: ${totalIndexed} items in ${indexTime}ms`);
      this.emit('reindexComplete', { totalIndexed, indexTime });

    } finally {
      this.indexingInProgress = false;
    }
  }

  private async indexCacheEntry(entry: any): Promise<void> {
    const searchableItem: SearchableItem = {
      id: `cache_${entry.key}`,
      title: entry.metadata?.contentType || 'Cached Item',
      content: typeof entry.data === 'string' ? entry.data : JSON.stringify(entry.data),
      keywords: entry.metadata?.tags || [],
      metadata: { source: 'cache', ...entry.metadata },
      timestamp: entry.metadata?.created || Date.now()
    };

    await fuseLazySearch.addItem(searchableItem);
  }

  private async indexEvidenceItem(evidence: any): Promise<void> {
    const searchableItem: SearchableItem = {
      id: `evidence_${evidence.id}`,
      title: evidence.fileName || evidence.id,
      content: evidence.description || '',
      keywords: evidence.tags || [],
      metadata: { source: 'evidence', type: evidence.type, caseId: evidence.caseId },
      timestamp: evidence.timeline?.createdAt ? new Date(evidence.timeline.createdAt).getTime() : Date.now()
    };

    await fuseLazySearch.addItem(searchableItem);
  }

  private async indexStoreItem(item: any): Promise<void> {
    const searchableItem: SearchableItem = {
      id: `store_${item.id}`,
      title: item.fileName || item.title || item.id,
      content: item.description || item.content || '',
      keywords: item.tags || [],
      metadata: { source: 'store', type: item.type || 'store-item' },
      timestamp: item.createdAt ? new Date(item.createdAt).getTime() : Date.now()
    };

    await fuseLazySearch.addItem(searchableItem);
  }

  /**
   * Get comprehensive search statistics
   */
  getSearchStats() {
    return {
      ...this.searchStats,
      lastIndexTime: this.lastIndexTime,
      indexingInProgress: this.indexingInProgress,
      isInitialized: this.isInitialized,
      fuseStats: fuseLazySearch.getStats()
    };
  }

  /**
   * Advanced legal document search
   */
  async searchLegalDocuments(query: string, options: UnifiedSearchOptions = {}): Promise<UnifiedSearchResult[]> {
    return this.search(query, {
      ...options,
      legalContextOnly: true,
      sources: ['cache', 'evidence', 'redis'],
      useEmbeddings: true,
      threshold: 0.3
    });
  }

  /**
   * Evidence-specific search with case context
   */
  async searchEvidence(caseId?: string, query?: string): Promise<UnifiedSearchResult[]> {
    const searchQuery = query || (caseId ? `caseId:${caseId}` : '*');
    
    return this.search(searchQuery, {
      sources: ['evidence', 'store'],
      legalContextOnly: true,
      maxResults: 100,
      threshold: 0.2
    });
  }

  /**
   * Clear all indexed data and reinitialize
   */
  async reset(): Promise<void> {
    console.log('🔄 Resetting unified search...');
    
    await fuseLazySearch.clearAll();
    this.searchStats = {
      totalQueries: 0,
      averageResponseTime: 0,
      sourcesQueried: { cache: 0, evidence: 0, store: 0, redis: 0 },
      totalResults: 0
    };
    
    await this.reindexAllSources();
    console.log('✅ Unified search reset complete');
  }
}

// Export singleton instance
export const unifiedLokiFuzzySearch = new UnifiedLokiFuzzySearch();

// Auto-initialize in browser
if (typeof window !== 'undefined') {
  unifiedLokiFuzzySearch.initialize().catch(console.warn);
}

// Export types
export type { UnifiedSearchOptions, UnifiedSearchResult };