// @ts-nocheck
/**
 * Advanced LokiJS Caching Layer with Multi-Level Storage
 * Supports real-time caching, background persistence, and intelligent eviction
 */

import Loki from 'lokijs';
import { EventEmitter } from 'events';
import path from 'path';
import fs from 'fs/promises';

interface CacheEntry {
  id: string;
  key: string;
  data: any;
  metadata: {
    sessionId?: string;
    userId?: string;
    contentType: string;
    confidence: number;
    processingTime: number;
    accessCount: number;
    lastAccessed: number;
    created: number;
    expires?: number;
    tags: string[];
  };
}

interface CacheStats {
  totalEntries: number;
  memoryUsage: number;
  hitRate: number;
  totalHits: number;
  totalMisses: number;
  averageProcessingTime: number;
  topTags: { tag: string; count: number; }[];
}

export class CacheManager extends EventEmitter {
  private db: Loki;
  private collections: {
    aiResults: Collection<CacheEntry>;
    userSessions: Collection<any>;
    recommendations: Collection<any>;
    semanticTokens: Collection<any>;
    analytics: Collection<any>;
  };
  private dbPath: string;
  private saveInterval: NodeJS.Timeout;
  private cleanupInterval: NodeJS.Timeout;
  private stats: CacheStats;

  constructor(options: {
    dbPath?: string;
    autoSaveInterval?: number;
    cleanupInterval?: number;
    maxMemoryUsage?: number;
  } = {}) {
    super();

    this.dbPath = options.dbPath || path.join(process.cwd(), 'cache', 'ai-cache.db');
    this.stats = {
      totalEntries: 0,
      memoryUsage: 0,
      hitRate: 0,
      totalHits: 0,
      totalMisses: 0,
      averageProcessingTime: 0,
      topTags: []
    };

    this.initializeDatabase();
    this.setupAutoSave(options.autoSaveInterval || 30000); // 30 seconds
    this.setupCleanup(options.cleanupInterval || 300000); // 5 minutes
    
    console.log('📦 Advanced LokiJS Cache Manager initialized');
  }

  private async initializeDatabase() {
    // Ensure cache directory exists
    await fs.mkdir(path.dirname(this.dbPath), { recursive: true });

    this.db = new Loki(this.dbPath, {
      adapter: new Loki.LokiFsAdapter(),
      autoload: true,
      autoloadCallback: () => {
        this.setupCollections();
        this.emit('ready');
      },
      autosave: true,
      autosaveInterval: 30000
    });
  }

  private setupCollections() {
    // AI Results cache - primary cache for processed results
    this.collections = {
      aiResults: this.db.getCollection('aiResults') || 
                 this.db.addCollection('aiResults', {
                   indices: ['key', 'metadata.sessionId', 'metadata.userId', 'metadata.contentType'],
                   unique: ['id']
                 }),

      // User sessions for continuity
      userSessions: this.db.getCollection('userSessions') || 
                   this.db.addCollection('userSessions', {
                     indices: ['userId', 'sessionId', 'lastActivity']
                   }),

      // Cached recommendations
      recommendations: this.db.getCollection('recommendations') || 
                      this.db.addCollection('recommendations', {
                        indices: ['userId', 'contentType', 'confidence']
                      }),

      // Semantic tokens cache
      semanticTokens: this.db.getCollection('semanticTokens') || 
                     this.db.addCollection('semanticTokens', {
                       indices: ['content', 'tokenizer', 'language']
                     }),

      // Analytics data cache
      analytics: this.db.getCollection('analytics') || 
                this.db.addCollection('analytics', {
                  indices: ['userId', 'eventType', 'timestamp']
                })
    };

    this.updateStats();
    console.log('🔧 LokiJS collections initialized');
  }

  // Core caching methods
  public async set(key: string, data: any, options: {
    sessionId?: string;
    userId?: string;
    contentType?: string;
    confidence?: number;
    processingTime?: number;
    ttl?: number; // Time to live in milliseconds
    tags?: string[];
  } = {}): Promise<void> {
    const entry: CacheEntry = {
      id: this.generateId(),
      key,
      data,
      metadata: {
        sessionId: options.sessionId,
        userId: options.userId,
        contentType: options.contentType || 'general',
        confidence: options.confidence || 1.0,
        processingTime: options.processingTime || 0,
        accessCount: 0,
        lastAccessed: Date.now(),
        created: Date.now(),
        expires: options.ttl ? Date.now() + options.ttl : undefined,
        tags: options.tags || []
      }
    };

    // Remove existing entry with same key
    this.collections.aiResults.findAndRemove({ key });

    // Add new entry
    this.collections.aiResults.insert(entry);
    
    this.updateStats();
    this.emit('cache-set', { key, contentType: entry.metadata.contentType });

    console.log(`💾 Cached result for key: ${key} (${entry.metadata.contentType})`);
  }

  public async get(key: string): Promise<any> {
    const entry = this.collections.aiResults.findOne({ key });
    
    if (!entry) {
      this.stats.totalMisses++;
      this.updateStats();
      return null;
    }

    // Check if expired
    if (entry.metadata.expires && Date.now() > entry.metadata.expires) {
      this.collections.aiResults.remove(entry);
      this.stats.totalMisses++;
      this.updateStats();
      return null;
    }

    // Update access statistics
    entry.metadata.accessCount++;
    entry.metadata.lastAccessed = Date.now();
    this.collections.aiResults.update(entry);

    this.stats.totalHits++;
    this.updateStats();

    console.log(`📖 Cache hit for key: ${key} (accessed ${entry.metadata.accessCount} times)`);
    return entry.data;
  }

  // Advanced query methods
  public async getByUser(userId: string, limit: number = 50): Promise<CacheEntry[]> {
    return this.collections.aiResults
      .chain()
      .find({ 'metadata.userId': userId })
      .simplesort('metadata.lastAccessed', true)
      .limit(limit)
      .data();
  }

  public async getBySession(sessionId: string): Promise<CacheEntry[]> {
    return this.collections.aiResults
      .find({ 'metadata.sessionId': sessionId });
  }

  public async getByContentType(contentType: string, limit: number = 100): Promise<CacheEntry[]> {
    return this.collections.aiResults
      .chain()
      .find({ 'metadata.contentType': contentType })
      .simplesort('metadata.confidence', true)
      .limit(limit)
      .data();
  }

  public async getByTags(tags: string[], matchAll: boolean = false): Promise<CacheEntry[]> {
    return this.collections.aiResults
      .chain()
      .where((entry) => {
        if (matchAll) {
          return tags.every(tag => entry.metadata.tags.includes(tag));
        } else {
          return tags.some(tag => entry.metadata.tags.includes(tag));
        }
      })
      .data();
  }

  // Semantic search within cache
  public async semanticSearch(query: string, limit: number = 10): Promise<CacheEntry[]> {
    // Simple text-based search - in production you'd use embeddings
    const searchTerms = query.toLowerCase().split(' ');
    
    return this.collections.aiResults
      .chain()
      .where((entry) => {
        const searchableText = JSON.stringify(entry.data).toLowerCase();
        return searchTerms.some(term => searchableText.includes(term));
      })
      .simplesort('metadata.confidence', true)
      .limit(limit)
      .data();
  }

  // User session management
  public async setUserSession(userId: string, sessionData: any): Promise<void> {
    const existingSession = this.collections.userSessions.findOne({ userId });
    
    if (existingSession) {
      existingSession.data = sessionData;
      existingSession.lastActivity = Date.now();
      this.collections.userSessions.update(existingSession);
    } else {
      this.collections.userSessions.insert({
        id: this.generateId(),
        userId,
        data: sessionData,
        created: Date.now(),
        lastActivity: Date.now()
      });
    }
  }

  public async getUserSession(userId: string): Promise<any> {
    const session = this.collections.userSessions.findOne({ userId });
    return session ? session.data : null;
  }

  // Recommendation caching
  public async setRecommendations(
    userId: string, 
    recommendations: any[], 
    contentType: string,
    confidence: number = 0.8
  ): Promise<void> {
    // Remove old recommendations for this user and content type
    this.collections.recommendations.findAndRemove({
      userId,
      contentType
    });

    this.collections.recommendations.insert({
      id: this.generateId(),
      userId,
      contentType,
      recommendations,
      confidence,
      created: Date.now()
    });
  }

  public async getRecommendations(userId: string, contentType?: string): Promise<any[]> {
    const query: any = { userId };
    if (contentType) query.contentType = contentType;

    const results = this.collections.recommendations.find(query);
    return results.length > 0 ? results[0].recommendations : [];
  }

  // Analytics caching
  public async logAnalytics(
    userId: string, 
    eventType: string, 
    data: any
  ): Promise<void> {
    this.collections.analytics.insert({
      id: this.generateId(),
      userId,
      eventType,
      data,
      timestamp: Date.now()
    });
  }

  public async getAnalytics(
    userId?: string, 
    eventType?: string, 
    timeRange?: { start: number; end: number }
  ): Promise<any[]> {
    let query: any = {};
    
    if (userId) query.userId = userId;
    if (eventType) query.eventType = eventType;
    
    let chain = this.collections.analytics.chain().find(query);
    
    if (timeRange) {
      chain = chain.where((entry) => 
        entry.timestamp >= timeRange.start && entry.timestamp <= timeRange.end
      );
    }
    
    return chain.simplesort('timestamp', true).data();
  }

  // Cleanup and maintenance
  private setupAutoSave(interval: number) {
    this.saveInterval = setInterval(() => {
      this.db.saveDatabase();
      console.log('💾 Auto-saved cache database');
    }, interval);
  }

  private setupCleanup(interval: number) {
    this.cleanupInterval = setInterval(async () => {
      await this.performCleanup();
    }, interval);
  }

  public async performCleanup(): Promise<{
    expired: number;
    lowAccess: number;
    oldEntries: number;
    totalRemoved: number;
  }> {
    const now = Date.now();
    let removed = { expired: 0, lowAccess: 0, oldEntries: 0, totalRemoved: 0 };

    // Remove expired entries
    const expired = this.collections.aiResults.findAndRemove(
      (entry: CacheEntry) => entry.metadata.expires && now > entry.metadata.expires
    );
    removed.expired = expired.length;

    // Remove low-access old entries (older than 7 days, accessed < 3 times)
    const sevenDaysAgo = now - (7 * 24 * 60 * 60 * 1000);
    const lowAccess = this.collections.aiResults.findAndRemove(
      (entry: CacheEntry) => 
        entry.metadata.created < sevenDaysAgo && entry.metadata.accessCount < 3
    );
    removed.lowAccess = lowAccess.length;

    // Remove very old entries (older than 30 days)
    const thirtyDaysAgo = now - (30 * 24 * 60 * 60 * 1000);
    const oldEntries = this.collections.aiResults.findAndRemove(
      (entry: CacheEntry) => entry.metadata.created < thirtyDaysAgo
    );
    removed.oldEntries = oldEntries.length;

    removed.totalRemoved = removed.expired + removed.lowAccess + removed.oldEntries;

    if (removed.totalRemoved > 0) {
      this.updateStats();
      console.log(`🧹 Cache cleanup: removed ${removed.totalRemoved} entries`);
      this.emit('cache-cleanup', removed);
    }

    return removed;
  }

  // Statistics and monitoring
  private updateStats() {
    const entries = this.collections.aiResults.data;
    this.stats.totalEntries = entries.length;
    this.stats.hitRate = this.stats.totalHits / (this.stats.totalHits + this.stats.totalMisses) * 100;
    
    if (entries.length > 0) {
      this.stats.averageProcessingTime = entries.reduce(
        (sum, entry) => sum + entry.metadata.processingTime, 0
      ) / entries.length;
    }

    // Calculate top tags
    const tagCounts = new Map<string, number>();
    entries.forEach(entry => {
      entry.metadata.tags.forEach(tag => {
        tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
      });
    });

    this.stats.topTags = Array.from(tagCounts.entries())
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }

  public getStats(): CacheStats {
    this.updateStats();
    return { ...this.stats };
  }

  public async clear(pattern?: string): Promise<number> {
    let removed = 0;
    
    if (pattern) {
      // Clear entries matching pattern
      const toRemove = this.collections.aiResults.find(
        (entry: CacheEntry) => entry.key.includes(pattern)
      );
      removed = toRemove.length;
      this.collections.aiResults.remove(toRemove);
    } else {
      // Clear all
      removed = this.collections.aiResults.count();
      this.collections.aiResults.clear();
    }

    this.updateStats();
    console.log(`🗑️ Cleared ${removed} cache entries`);
    return removed;
  }

  // Utility methods
  private generateId(): string {
    return `cache_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  public async export(format: 'json' | 'csv' = 'json'): Promise<string> {
    const data = {
      aiResults: this.collections.aiResults.data,
      userSessions: this.collections.userSessions.data,
      recommendations: this.collections.recommendations.data,
      semanticTokens: this.collections.semanticTokens.data,
      analytics: this.collections.analytics.data,
      stats: this.getStats()
    };

    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else {
      // Simple CSV export for main cache entries
      const entries = this.collections.aiResults.data;
      const csv = [
        'id,key,contentType,confidence,accessCount,created,lastAccessed',
        ...entries.map(entry => 
          `${entry.id},${entry.key},${entry.metadata.contentType},${entry.metadata.confidence},${entry.metadata.accessCount},${entry.metadata.created},${entry.metadata.lastAccessed}`
        )
      ].join('\n');
      
      return csv;
    }
  }

  public async shutdown(): Promise<void> {
    console.log('🔄 Shutting down Cache Manager...');
    
    // Clear intervals
    if (this.saveInterval) clearInterval(this.saveInterval);
    if (this.cleanupInterval) clearInterval(this.cleanupInterval);
    
    // Final save
    await new Promise<void>((resolve) => {
      this.db.saveDatabase(() => {
        console.log('✅ Final cache save completed');
        resolve();
      });
    });
    
    console.log('✅ Cache Manager shutdown complete');
  }
}

// Export singleton instance
export const cacheManager = new CacheManager();