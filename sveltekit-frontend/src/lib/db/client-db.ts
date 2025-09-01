/**
 * Client-Side Legal AI Database - IndexedDB with Dexie.js
 * 
 * Advanced client-side persistence layer for legal document analysis:
 * - Chat history and user interactions
 * - Cached vector search results  
 * - Legal document annotations
 * - Graph visualization data
 * - AI analysis cache
 * 
 * Complements your server-side "tricubic tensor" PostgreSQL system
 */

import Dexie from 'dexie';
import type { Table } from 'dexie';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface ChatMessage {
  id?: number;
  sessionId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    model?: string;
    processingTime?: number;
    tokenCount?: number;
    confidence?: number;
    sources?: string[];
    ragContext?: boolean;
  };
}

export interface DocumentCache {
  id?: number;
  documentId: string;
  hash: string;
  title: string;
  content: string;
  contentType: 'text' | 'pdf' | 'docx' | 'md';
  fileSize: number;
  lastAccessed: Date;
  metadata: {
    aiSummary?: string;
    keyTerms?: string[];
    legalEntities?: any;
    riskLevel?: string;
    jurisdiction?: string;
    documentType?: string;
  };
}

export interface SearchHistory {
  id?: number;
  query: string;
  timestamp: Date;
  resultCount: number;
  processingTime: number;
  searchType: 'vector' | 'hybrid' | 'text' | 'legal';
  userId?: string;
  filters?: {
    evidenceType?: string[];
    priority?: string[];
    dateRange?: { start: Date; end: Date };
    jurisdiction?: string[];
  };
}

export interface VectorSearchCache {
  id?: number;
  queryHash: string; // MD5 of query + filters for deduplication
  query: string;
  results: Array<{
    id: string;
    content: string;
    similarity: number;
    metadata: any;
    sourceType: 'case' | 'evidence' | 'document';
    rankingMatrix: number[][];
  }>;
  timestamp: Date;
  expiresAt: Date;
  lodLevel: number;
  hitCount: number; // Track cache usage
}

export interface UserAnnotation {
  id?: number;
  documentId: string;
  chunkId?: string;
  text: string;
  note: string;
  position?: {
    start: number;
    end: number;
    page?: number;
  };
  tags: string[];
  importance: 'low' | 'medium' | 'high' | 'critical';
  userId?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface LegalEntity {
  id?: number;
  name: string;
  type: 'person' | 'organization' | 'court' | 'statute' | 'case' | 'concept';
  aliases: string[];
  description?: string;
  relatedDocuments: string[];
  confidence: number;
  extractedFrom: {
    documentId: string;
    chunkId?: string;
    context: string;
  }[];
  metadata: {
    jurisdiction?: string;
    dates?: string[];
    role?: string;
    importance?: number;
  };
  lastUpdated: Date;
}

export interface GraphVisualizationData {
  id?: number;
  graphId: string;
  graphType: 'document-similarity' | 'legal-entities' | 'case-relationships' | 'citation-network';
  nodes: Array<{
    id: string;
    label: string;
    type: string;
    position: { x: number; y: number; z?: number };
    size: number;
    color: string;
    metadata: any;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    weight: number;
    type: string;
    color: string;
    metadata: any;
  }>;
  layout: {
    algorithm: string;
    parameters: any;
    dimensions: 2 | 3;
  };
  cameraPosition?: { x: number; y: number; z: number };
  createdAt: Date;
  lastAccessed: Date;
  computationTime: number;
}

export interface AIAnalysisCache {
  id?: number;
  contentHash: string;
  analysisType: 'summary' | 'entities' | 'risk' | 'classification' | 'similarity';
  input: string;
  result: any;
  model: string;
  confidence: number;
  processingTime: number;
  timestamp: Date;
  expiresAt: Date;
}

export interface UserPreferences {
  id?: number;
  userId?: string;
  preferences: {
    theme: 'light' | 'dark' | 'yorha';
    layout: 'grid' | 'list' | 'graph';
    defaultSearchType: 'vector' | 'hybrid' | 'text';
    cacheSettings: {
      maxDocuments: number;
      maxSearchResults: number;
      cacheExpiry: number; // hours
    };
    visualization: {
      defaultGraphType: string;
      showLabels: boolean;
      enablePhysics: boolean;
      colorScheme: string;
    };
    ai: {
      preferredModel: string;
      temperature: number;
      includeAnalysis: boolean;
      autoSummarize: boolean;
    };
  };
  lastUpdated: Date;
}

// ============================================================================
// DATABASE CLASS
// ============================================================================

export class LegalAIClientDB extends Dexie {
  // Table declarations with proper Dexie typing
  chatHistory!: Dexie.Table<ChatMessage, number>;
  documentCache!: Dexie.Table<DocumentCache, number>;
  searchHistory!: Dexie.Table<SearchHistory, number>;
  vectorSearchCache!: Dexie.Table<VectorSearchCache, number>;
  userAnnotations!: Dexie.Table<UserAnnotation, number>;
  legalEntities!: Dexie.Table<LegalEntity, number>;
  graphVisualizationData!: Dexie.Table<GraphVisualizationData, number>;
  aiAnalysisCache!: Dexie.Table<AIAnalysisCache, number>;
  userPreferences!: Dexie.Table<UserPreferences, number>;

  constructor() {
    super('LegalAIClientDB');
    
    // Database schema with optimized indexes
    this.version(1).stores({
      chatHistory: '++id, sessionId, timestamp, role',
      documentCache: '++id, documentId, hash, lastAccessed, title',
      searchHistory: '++id, timestamp, query, searchType, userId',
      vectorSearchCache: '++id, queryHash, timestamp, expiresAt, hitCount',
      userAnnotations: '++id, documentId, chunkId, userId, createdAt, importance',
      legalEntities: '++id, name, type, lastUpdated, confidence',
      graphVisualizationData: '++id, graphId, graphType, lastAccessed, createdAt',
      aiAnalysisCache: '++id, contentHash, analysisType, timestamp, expiresAt',
      userPreferences: '++id, userId, lastUpdated'
    });

    // Hooks for data management
    this.chatHistory.hook('creating', (primaryKey, obj, trans) => {
      obj.timestamp = new Date();
    });

    this.userAnnotations.hook('creating', (primaryKey, obj, trans) => {
      (obj as any).createdAt = new Date();
      (obj as any).updatedAt = new Date();
    });

    this.userAnnotations.hook('updating', (modifications, primaryKey, obj, trans) => {
      (modifications as any).updatedAt = new Date();
    });
  }
}

// ============================================================================
// DATABASE INSTANCE & UTILITIES
// ============================================================================

export const legalDB = new LegalAIClientDB();
;
// Database utility functions
export class LegalDBUtils {
  /**
   * Clean up expired cache entries
   */
  static async cleanupExpiredCache(): Promise<void> {
    const now = new Date();
    
    // Remove expired vector search cache
    await legalDB.vectorSearchCache
      .where('expiresAt')
      .below(now)
      .delete();

    // Remove expired AI analysis cache
    await legalDB.aiAnalysisCache
      .where('expiresAt')
      .below(now)
      .delete();

    console.log('[ClientDB] Cleaned up expired cache entries');
  }

  /**
   * Manage document cache size (LRU eviction)
   */
  static async manageDocumentCacheSize(maxDocuments = 1000): Promise<void> {
    const count = await legalDB.documentCache.count();
    
    if (count > maxDocuments) {
      // Remove least recently accessed documents
      const oldDocuments = await legalDB.documentCache
        .orderBy('lastAccessed')
        .limit(count - maxDocuments)
        .toArray();
      
      const idsToDelete = oldDocuments.map(doc => doc.id!);
      await legalDB.documentCache.bulkDelete(idsToDelete);
      
      console.log(`[ClientDB] Evicted ${idsToDelete.length} old cached documents`);
    }
  }

  /**
   * Get database statistics
   */
  static async getStorageStats(): Promise<{
    totalRecords: number;
    storageUsed: string;
    tables: Array<{ name: string; count: number }>;
  }> {
    const stats = {
      totalRecords: 0,
      storageUsed: 'Unknown',
      tables: [] as Array<{ name: string; count: number }>
    };

    // Count records in each table
    const tables = [
      'chatHistory',
      'documentCache', 
      'searchHistory',
      'vectorSearchCache',
      'userAnnotations',
      'legalEntities',
      'graphVisualizationData',
      'aiAnalysisCache',
      'userPreferences'
    ];

    for (const tableName of tables) {
      const count = await (legalDB as any)[tableName].count();
      stats.tables.push({ name: tableName, count });
      stats.totalRecords += count;
    }

    // Estimate storage usage (if available)
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      if (estimate.usage) {
        stats.storageUsed = `${(estimate.usage / 1024 / 1024).toFixed(2)} MB`;
      }
    }

    return stats;
  }

  /**
   * Create content hash for caching
   */
  static createHash(content: string): string {
    let hash = 0;
    if (content.length === 0) return hash.toString();
    
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return Math.abs(hash).toString(36);
  }

  /**
   * Intelligent cache cleanup based on usage patterns
   */
  static async intelligentCleanup(): Promise<void> {
    console.log('[ClientDB] Starting intelligent cleanup...');
    
    // 1. Remove expired entries
    await this.cleanupExpiredCache();
    
    // 2. Clean low-hit vector search cache
    await legalDB.vectorSearchCache
      .where('hitCount')
      .below(2) // Remove rarely used cached searches
      .and(item => {
        const daysSinceCreated = (Date.now() - item.timestamp.getTime()) / (1000 * 60 * 60 * 24);
        return daysSinceCreated > 7; // Older than a week
      })
      .delete();
    
    // 3. Manage document cache size
    await this.manageDocumentCacheSize(1000);
    
    // 4. Remove old graph visualization data (keep only recent)
    const oneMonthAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    await legalDB.graphVisualizationData
      .where('lastAccessed')
      .below(oneMonthAgo)
      .delete();
    
    console.log('[ClientDB] Intelligent cleanup completed');
  }
}

// ============================================================================
// REACTIVE STORES FOR SVELTE
// ============================================================================

import { writable, derived } from 'svelte/store';
import { liveQuery } from 'dexie';

/**
 * Reactive store for recent chat messages
 */
export const recentChatMessages = liveQuery(async () => {
  return await legalDB.chatHistory
    .orderBy('timestamp')
    .reverse()
    .limit(50)
    .toArray();
});

/**
 * Reactive store for search history
 */
export const searchHistory = liveQuery(async () => {
  return await legalDB.searchHistory
    .orderBy('timestamp')
    .reverse()
    .limit(20)
    .toArray();
});

/**
 * Reactive store for cached documents count
 */
export const cachedDocumentsCount = liveQuery(async () => {
  return await legalDB.documentCache.count();
});

/**
 * Reactive store for user annotations count
 */
export const annotationsCount = liveQuery(async () => {
  return await legalDB.userAnnotations.count();
});

/**
 * Storage usage monitor
 */
export const storageStats = writable({ totalRecords: 0, storageUsed: 'Unknown', tables: [] });
;
// Update storage stats periodically
if (typeof window !== 'undefined') {
  setInterval(async () => {
    const stats = await LegalDBUtils.getStorageStats();
    storageStats.set(stats);
  }, 30000); // Update every 30 seconds
}

// ============================================================================
// INITIALIZATION & CLEANUP
// ============================================================================

// Initialize database and set up periodic cleanup
if (typeof window !== 'undefined') {
  // Set up automatic cleanup every hour
  setInterval(() => {
    LegalDBUtils.intelligentCleanup().catch(console.error);
  }, 60 * 60 * 1000);

  // Initial cleanup on load
  LegalDBUtils.cleanupExpiredCache().catch(console.error);
  
  console.log('[ClientDB] Legal AI Client Database initialized');
}