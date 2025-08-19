/**
 * Fuse.js Lazy Search with IndexedDB Integration
 * High-performance keyword and embedding search with persistence
 */

import Fuse from '$lib/utils/fuse-import';

export interface SearchableItem {
  id: string;
  title: string;
  content: string;
  keywords: string[];
  embedding?: Float32Array;
  metadata?: Record<string, any>;
  timestamp?: number;
}

export interface SearchOptions {
  threshold?: number;
  includeScore?: boolean;
  includeMatches?: boolean;
  useEmbeddings?: boolean;
  maxResults?: number;
  cached?: boolean;
}

export interface SearchResult {
  item: SearchableItem;
  score?: number;
  matches?: any[];
  similarity?: number;
  refIndex: number;
}

/**
 * Enhanced search service with Fuse.js, IndexedDB, and vector embeddings
 */
export class FuseLazySearchService {
  private db: IDBDatabase | null = null;
  private fuse: any = null;
  private items: SearchableItem[] = [];
  private isInitialized = false;
  private dbName = 'legal-ai-search';
  private dbVersion = 2;
  private storeName = 'searchable-items';

  /**
   * Initialize IndexedDB and Fuse.js search
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🔍 Initializing Fuse.js lazy search with IndexedDB...');

    try {
      // Initialize IndexedDB
      await this.initializeIndexedDB();
      
      // Load existing data
      await this.loadFromIndexedDB();
      
      // Initialize Fuse.js
      this.initializeFuse();
      
      this.isInitialized = true;
      console.log(`✅ Fuse lazy search initialized with ${this.items.length} items`);
    } catch (error) {
      console.error('❌ Failed to initialize Fuse lazy search:', error);
      throw error;
    }
  }

  /**
   * Initialize IndexedDB for persistent storage
   */
  private async initializeIndexedDB(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create object store with auto-incrementing key
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'id' });
          
          // Create indexes for efficient searching
          store.createIndex('title', 'title', { unique: false });
          store.createIndex('keywords', 'keywords', { unique: false, multiEntry: true });
          store.createIndex('timestamp', 'timestamp', { unique: false });
          store.createIndex('content', 'content', { unique: false });
        }
      };
    });
  }

  /**
   * Load existing items from IndexedDB
   */
  private async loadFromIndexedDB(): Promise<void> {
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.getAll();

      request.onsuccess = () => {
        this.items = request.result || [];
        console.log(`📊 Loaded ${this.items.length} items from IndexedDB`);
        resolve();
      };

      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Initialize Fuse.js with optimized configuration
   */
  private initializeFuse(): void {
    const fuseOptions = {
      keys: [
        { name: 'title', weight: 0.3 },
        { name: 'content', weight: 0.4 },
        { name: 'keywords', weight: 0.3 }
      ],
      threshold: 0.4,
      includeScore: true,
      includeMatches: true,
      minMatchCharLength: 2,
      findAllMatches: true,
      ignoreLocation: true,
      useExtendedSearch: true
    };

    this.fuse = new Fuse(this.items, fuseOptions);
    console.log('⚡ Fuse.js search engine initialized');
  }

  /**
   * Add or update searchable item
   */
  async addItem(item: SearchableItem): Promise<void> {
    await this.initialize();

    if (!this.db) throw new Error('IndexedDB not initialized');

    // Add timestamp if not provided
    if (!item.timestamp) {
      item.timestamp = Date.now();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.put(item);

      request.onsuccess = () => {
        // Update in-memory array
        const existingIndex = this.items.findIndex(i => i.id === item.id);
        if (existingIndex >= 0) {
          this.items[existingIndex] = item;
        } else {
          this.items.push(item);
        }

        // Reinitialize Fuse with updated data
        this.initializeFuse();
        
        console.log(`📝 Added/updated item: ${item.title}`);
        resolve();
      };

      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Batch add multiple items
   */
  async addItems(items: SearchableItem[]): Promise<void> {
    await this.initialize();

    if (!this.db) throw new Error('IndexedDB not initialized');

    console.log(`📦 Batch adding ${items.length} items...`);

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      
      let completed = 0;
      const total = items.length;

      for (const item of items) {
        if (!item.timestamp) {
          item.timestamp = Date.now();
        }

        const request = store.put(item);
        
        request.onsuccess = () => {
          completed++;
          if (completed === total) {
            // Reload all items and reinitialize Fuse
            this.loadFromIndexedDB().then(() => {
              this.initializeFuse();
              console.log(`✅ Batch add complete: ${total} items`);
              resolve();
            });
          }
        };

        request.onerror = () => reject(request.error);
      }
    });
  }

  /**
   * Search with Fuse.js and optional vector similarity
   */
  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    await this.initialize();

    const searchOptions = {
      threshold: 0.4,
      includeScore: true,
      includeMatches: true,
      useEmbeddings: false,
      maxResults: 50,
      cached: true,
      ...options
    };

    console.log(`🔍 Searching for: "${query}" (${this.items.length} items)`);

    try {
      // Fuse.js text search
      const fuseResults = this.fuse.search(query, {
        limit: searchOptions.maxResults
      });

      let results: SearchResult[] = fuseResults.map((result: any) => ({
        item: result.item,
        score: result.score,
        matches: result.matches,
        refIndex: result.refIndex
      }));

      // Add vector similarity if embeddings are available and requested
      if (searchOptions.useEmbeddings) {
        results = await this.enhanceWithVectorSimilarity(query, results);
      }

      // Sort by combined score (text relevance + vector similarity)
      results.sort((a, b) => {
        const scoreA = (a.score || 1) * 0.7 + (1 - (a.similarity || 0)) * 0.3;
        const scoreB = (b.score || 1) * 0.7 + (1 - (b.similarity || 0)) * 0.3;
        return scoreA - scoreB;
      });

      console.log(`📊 Search complete: ${results.length} results`);
      return results.slice(0, searchOptions.maxResults);

    } catch (error) {
      console.error('❌ Search failed:', error);
      return [];
    }
  }

  /**
   * Enhanced search with vector similarity (requires embeddings)
   */
  private async enhanceWithVectorSimilarity(
    query: string, 
    textResults: SearchResult[]
  ): Promise<SearchResult[]> {
    try {
      // Generate embedding for query (mock implementation)
      const queryEmbedding = await this.generateEmbedding(query);
      
      // Calculate cosine similarity for items with embeddings
      for (const result of textResults) {
        if (result.item.embedding) {
          result.similarity = this.cosineSimilarity(queryEmbedding, result.item.embedding);
        }
      }

      // Also search items that didn't match text but have high vector similarity
      const vectorOnlyResults = this.items
        .filter(item => 
          item.embedding && 
          !textResults.some(r => r.item.id === item.id)
        )
        .map(item => {
          const similarity = this.cosineSimilarity(queryEmbedding, item.embedding!);
          return {
            item,
            similarity,
            score: 1 - similarity, // Convert similarity to score format
            refIndex: this.items.indexOf(item)
          };
        })
        .filter(result => result.similarity > 0.7)
        .slice(0, 10);

      return [...textResults, ...vectorOnlyResults];

    } catch (error) {
      console.warn('⚠️ Vector similarity enhancement failed:', error);
      return textResults;
    }
  }

  /**
   * Generate embedding for text (integration point with nomic-embed-text)
   */
  private async generateEmbedding(text: string): Promise<Float32Array> {
    try {
      // Call Ollama nomic-embed-text model
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding API error: ${response.status}`);
      }

      const data = await response.json();
      return new Float32Array(data.embedding);

    } catch (error) {
      console.warn('⚠️ Embedding generation failed, using fallback:', error);
      // Fallback: simple hash-based pseudo-embedding
      return this.generateFallbackEmbedding(text);
    }
  }

  /**
   * Fallback embedding generation using hash-based approach
   */
  private generateFallbackEmbedding(text: string): Float32Array {
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Float32Array(384); // nomic-embed-text dimension

    for (let i = 0; i < words.length && i < 100; i++) {
      const word = words[i];
      for (let j = 0; j < word.length; j++) {
        const charCode = word.charCodeAt(j);
        const index = (charCode + i * 7 + j * 13) % embedding.length;
        embedding[index] += 0.1;
      }
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }

    return embedding;
  }

  /**
   * Calculate cosine similarity between two embeddings
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Search by keywords with lazy loading
   */
  async searchKeywords(keywords: string[], options: SearchOptions = {}): Promise<SearchResult[]> {
    await this.initialize();

    console.log(`🏷️ Searching by keywords: ${keywords.join(', ')}`);

    const keywordResults: SearchResult[] = [];

    // Search each keyword separately and combine results
    for (const keyword of keywords) {
      const results = await this.search(keyword, { ...options, maxResults: 20 });
      keywordResults.push(...results);
    }

    // Deduplicate and rank by keyword relevance
    const uniqueResults = new Map<string, SearchResult>();
    
    for (const result of keywordResults) {
      const existing = uniqueResults.get(result.item.id);
      if (!existing || (result.score || 1) < (existing.score || 1)) {
        uniqueResults.set(result.item.id, result);
      }
    }

    const finalResults = Array.from(uniqueResults.values())
      .sort((a, b) => (a.score || 1) - (b.score || 1))
      .slice(0, options.maxResults || 50);

    console.log(`🎯 Keyword search complete: ${finalResults.length} results`);
    return finalResults;
  }

  /**
   * Vector-only search using embeddings
   */
  async searchVectors(queryEmbedding: Float32Array, threshold = 0.7): Promise<SearchResult[]> {
    await this.initialize();

    console.log('🔢 Performing vector similarity search...');

    const vectorResults: SearchResult[] = [];

    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];
      if (item.embedding) {
        const similarity = this.cosineSimilarity(queryEmbedding, item.embedding);
        if (similarity > threshold) {
          vectorResults.push({
            item,
            similarity,
            score: 1 - similarity,
            refIndex: i
          });
        }
      }
    }

    const results = vectorResults
      .sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
      .slice(0, 50);

    console.log(`📊 Vector search complete: ${results.length} results above ${threshold} threshold`);
    return results;
  }

  /**
   * Hybrid search combining text and vector similarity
   */
  async hybridSearch(
    query: string, 
    queryEmbedding?: Float32Array, 
    options: SearchOptions = {}
  ): Promise<SearchResult[]> {
    await this.initialize();

    console.log(`🚀 Hybrid search for: "${query}"`);

    // Get text search results
    const textResults = await this.search(query, { ...options, useEmbeddings: false });
    
    // Get vector search results if embedding provided
    let vectorResults: SearchResult[] = [];
    if (queryEmbedding && options.useEmbeddings) {
      vectorResults = await this.searchVectors(queryEmbedding, 0.6);
    }

    // Combine and deduplicate results
    const combinedResults = new Map<string, SearchResult>();

    // Add text results with text score weight
    for (const result of textResults) {
      combinedResults.set(result.item.id, {
        ...result,
        combinedScore: (result.score || 1) * 0.6 // Text search weight
      });
    }

    // Add or enhance with vector results
    for (const result of vectorResults) {
      const existing = combinedResults.get(result.item.id);
      if (existing) {
        // Combine scores: text (0.6) + vector (0.4)
        existing.combinedScore = (existing.score || 1) * 0.6 + (1 - (result.similarity || 0)) * 0.4;
        existing.similarity = result.similarity;
      } else {
        // Vector-only result
        combinedResults.set(result.item.id, {
          ...result,
          combinedScore: (1 - (result.similarity || 0)) * 0.8 // Vector-only penalty
        });
      }
    }

    // Sort by combined score and return top results
    const finalResults = Array.from(combinedResults.values())
      .sort((a, b) => (a.combinedScore || 1) - (b.combinedScore || 1))
      .slice(0, options.maxResults || 50);

    console.log(`🎯 Hybrid search complete: ${finalResults.length} results (${textResults.length} text + ${vectorResults.length} vector)`);
    return finalResults;
  }

  /**
   * Clear all stored data
   */
  async clearAll(): Promise<void> {
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.clear();

      request.onsuccess = () => {
        this.items = [];
        this.initializeFuse();
        console.log('🧹 All search data cleared');
        resolve();
      };

      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Get search statistics
   */
  getStats() {
    return {
      totalItems: this.items.length,
      itemsWithEmbeddings: this.items.filter(item => item.embedding).length,
      dbSize: this.db ? 'connected' : 'disconnected',
      fuseInitialized: !!this.fuse,
      isReady: this.isInitialized
    };
  }

  /**
   * Export all data (for backup or migration)
   */
  async exportData(): Promise<SearchableItem[]> {
    await this.initialize();
    return [...this.items];
  }

  /**
   * Import data (for restore or migration)
   */
  async importData(items: SearchableItem[]): Promise<void> {
    await this.clearAll();
    await this.addItems(items);
    console.log(`📥 Imported ${items.length} items`);
  }
}

// Global service instance
export const fuseLazySearch = new FuseLazySearchService();

// Auto-initialize on import (browser only)
if (typeof window !== 'undefined') {
  fuseLazySearch.initialize().catch(console.warn);
}

// Legal AI specific search utilities
export class LegalSearchUtils {
  static async indexLegalDocument(
    id: string,
    title: string,
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<void> {
    // Extract legal keywords
    const keywords = this.extractLegalKeywords(content);
    
    // Generate embedding
    const embedding = await fuseLazySearch['generateEmbedding'](content);

    const item: SearchableItem = {
      id,
      title,
      content,
      keywords,
      embedding,
      metadata: {
        ...metadata,
        type: 'legal-document',
        indexed: new Date().toISOString()
      },
      timestamp: Date.now()
    };

    await fuseLazySearch.addItem(item);
    console.log(`⚖️ Indexed legal document: ${title}`);
  }

  static extractLegalKeywords(content: string): string[] {
    const legalTerms = [
      'plaintiff', 'defendant', 'evidence', 'testimony', 'witness',
      'contract', 'liability', 'damages', 'breach', 'indemnification',
      'jurisdiction', 'precedent', 'statute', 'regulation', 'compliance',
      'negligence', 'malpractice', 'tort', 'criminal', 'civil',
      'court', 'judge', 'jury', 'attorney', 'counsel', 'legal',
      'case', 'trial', 'hearing', 'motion', 'appeal', 'verdict'
    ];

    const words = content.toLowerCase().split(/\W+/);
    const foundTerms = words.filter(word => legalTerms.includes(word));
    
    // Add extracted entities (simplified)
    const entities = content.match(/[A-Z][a-z]+ v\. [A-Z][a-z]+/g) || [];
    const citations = content.match(/\d+ [A-Z]\w*\.?\s*\d+/g) || [];

    return [...new Set([...foundTerms, ...entities, ...citations])];
  }

  static async searchCases(query: string): Promise<SearchResult[]> {
    return fuseLazySearch.search(query, {
      threshold: 0.3,
      useEmbeddings: true,
      maxResults: 20
    });
  }

  static async searchEvidence(query: string): Promise<SearchResult[]> {
    return fuseLazySearch.search(query, {
      threshold: 0.4,
      useEmbeddings: true,
      maxResults: 30
    });
  }
}

// Export utilities
export { LegalSearchUtils };