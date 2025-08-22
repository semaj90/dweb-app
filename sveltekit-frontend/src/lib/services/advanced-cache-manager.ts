/**
 * Advanced Cache Manager - Legal AI Platform
 * Enterprise-grade caching with intelligent eviction, legal document optimization,
 * and multi-tier storage for enhanced performance and data security
 */

import { writable, derived, get } from "svelte/store";
import { browser } from "$app/environment";

export interface CacheItem<T = any> {
  data: T;
  timestamp: number;
  ttl: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  access_count: number;
  last_accessed: number;
  tags: string[];
  size: number;
  encrypted?: boolean;
  legal_sensitive?: boolean;
  document_type?: 'evidence' | 'contract' | 'case_file' | 'general';
  confidentiality_level?: 'public' | 'confidential' | 'privileged';
  checksum?: string;
}

export interface LazyLoadConfig {
  threshold: number;
  rootMargin: string;
  enabled: boolean;
  legal_priority?: boolean;
  max_concurrent?: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  total_size: number;
  items_count: number;
  legal_items_count: number;
  privileged_items_count: number;
  encryption_overhead: number;
  cache_efficiency: number;
}

export interface CachePerformanceMetrics {
  hitRate: number;
  averageItemSize: number;
  memoryEfficiency: number;
  totalItems: number;
  legalItemsRatio: number;
  privilegedItemsRatio: number;
  encryptionOverhead: number;
  averageAccessTime: number;
  evictionRate: number;
}

export interface SecurityConfig {
  enableEncryption: boolean;
  encryptPrivileged: boolean;
  maxPrivilegedCacheTime: number;
  auditLogging: boolean;
  accessControlValidation: boolean;
}

export interface CacheOptions {
  ttl?: number;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  tags?: string[];
  encrypt?: boolean;
  legal_sensitive?: boolean;
  document_type?: 'evidence' | 'contract' | 'case_file' | 'general';
  confidentiality_level?: 'public' | 'confidential' | 'privileged';
}

class AdvancedCacheManager {
  private cache = new Map<string, CacheItem>();
  private indexDB: IDBDatabase | null = null;
  private encryptionKey: CryptoKey | null = null;
  
  private stats = writable<CacheStats>({
    hits: 0,
    misses: 0,
    evictions: 0,
    total_size: 0,
    items_count: 0,
    legal_items_count: 0,
    privileged_items_count: 0,
    encryption_overhead: 0,
    cache_efficiency: 0
  });
  
  private maxSize = 100 * 1024 * 1024; // 100MB default
  private maxItems = 2000; // Increased for legal documents
  private maxPrivilegedItems = 100; // Strict limit for privileged content
  private lazyLoadObserver: IntersectionObserver | null = null;
  private pendingLoads = new Map<string, Promise<any>>();
  private accessLog: Array<{ key: string; timestamp: number; action: string }> = [];
  
  private securityConfig: SecurityConfig = {
    enableEncryption: true,
    encryptPrivileged: true,
    maxPrivilegedCacheTime: 30 * 60 * 1000, // 30 minutes for privileged content
    auditLogging: true,
    accessControlValidation: true
  };

  constructor(config?: Partial<SecurityConfig>) {
    if (config) {
      this.securityConfig = { ...this.securityConfig, ...config };
    }

    if (browser) {
      this.initializeStorage();
      this.initializeLazyLoading();
      this.setupPeriodicCleanup();
      this.setupPrivilegedContentMonitoring();
      this.loadFromPersistentStorage();
    }
  }

  /**
   * Initialize secure storage and encryption
   */
  private async initializeStorage(): Promise<void> {
    try {
      if (this.securityConfig.enableEncryption && 'crypto' in window && 'subtle' in window.crypto) {
        // Generate or retrieve encryption key for sensitive data
        this.encryptionKey = await this.getOrCreateEncryptionKey();
      }

      // Initialize IndexedDB for large legal documents
      if ('indexedDB' in window) {
        const request = indexedDB.open('LegalAICacheDB', 1);
        
        request.onupgradeneeded = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;
          if (!db.objectStoreNames.contains('cache')) {
            const store = db.createObjectStore('cache', { keyPath: 'key' });
            store.createIndex('priority', 'priority', { unique: false });
            store.createIndex('timestamp', 'timestamp', { unique: false });
            store.createIndex('document_type', 'document_type', { unique: false });
          }
        };
        
        request.onsuccess = (event) => {
          this.indexDB = (event.target as IDBOpenDBRequest).result;
        };
      }
    } catch (error) {
      console.error('Failed to initialize advanced cache storage:', error);
    }
  }

  /**
   * Initialize intersection observer for lazy loading with legal document priority
   */
  private initializeLazyLoading(): void {
    if ('IntersectionObserver' in window) {
      this.lazyLoadObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              const element = entry.target as HTMLElement;
              const cacheKey = element.dataset.cacheKey;
              const loader = element.dataset.loader;
              const isLegal = element.dataset.legalSensitive === 'true';
              
              if (cacheKey && loader) {
                this.lazyLoad(cacheKey, loader, {
                  priority: isLegal ? 'high' : 'medium',
                  legal_sensitive: isLegal
                });
                this.lazyLoadObserver?.unobserve(element);
              }
            }
          });
        },
        {
          threshold: 0.1,
          rootMargin: '100px' // Increased margin for legal documents
        }
      );
    }
  }

  /**
   * Advanced caching with encryption and legal document handling
   */
  async set<T>(
    key: string, 
    data: T, 
    options: CacheOptions = {}
  ): Promise<void> {
    const {
      ttl = this.getDefaultTTL(options.confidentiality_level),
      priority = 'medium',
      tags = [],
      encrypt = this.shouldEncrypt(options),
      legal_sensitive = false,
      document_type = 'general',
      confidentiality_level = 'public'
    } = options;

    // Security validation for privileged content
    if (confidentiality_level === 'privileged') {
      await this.validatePrivilegedAccess(key);
      if (this.getPrivilegedItemsCount() >= this.maxPrivilegedItems) {
        await this.evictOldestPrivileged();
      }
    }

    let processedData = data;
    let encryptionOverhead = 0;
    let checksum: string | undefined;

    // Encrypt sensitive data
    if (encrypt && this.encryptionKey) {
      try {
        const encrypted = await this.encryptData(JSON.stringify(data));
        processedData = encrypted as T;
        encryptionOverhead = this.calculateEncryptionOverhead(data, encrypted);
        checksum = await this.generateChecksum(JSON.stringify(data));
      } catch (error) {
        console.error('Encryption failed, storing unencrypted:', error);
      }
    }

    const serialized = JSON.stringify(processedData);
    const size = new Blob([serialized]).size + encryptionOverhead;

    // Ensure capacity before adding
    await this.ensureCapacity(size);

    const item: CacheItem<T> = {
      data: processedData,
      timestamp: Date.now(),
      ttl,
      priority,
      access_count: 0,
      last_accessed: Date.now(),
      tags,
      size,
      encrypted: encrypt,
      legal_sensitive,
      document_type,
      confidentiality_level,
      checksum
    };

    this.cache.set(key, item);
    this.updateStats({ 
      items_count: 1, 
      total_size: size,
      legal_items_count: legal_sensitive ? 1 : 0,
      privileged_items_count: confidentiality_level === 'privileged' ? 1 : 0,
      encryption_overhead: encryptionOverhead
    });

    // Audit logging for legal documents
    if (this.securityConfig.auditLogging && legal_sensitive) {
      this.logAccess(key, 'SET');
    }

    // Persist critical and privileged items
    if (priority === 'critical' || confidentiality_level === 'privileged') {
      await this.persistToStorage(key, item);
    }
  }

  /**
   * Advanced get with decryption and access validation
   */
  async get<T>(key: string): Promise<T | null> {
    let item = this.cache.get(key) as CacheItem<T> | undefined;
    
    if (!item) {
      this.updateStats({ misses: 1 });
      
      // Try to load from persistent storage
      item = await this.loadFromStorage<T>(key);
      if (item && this.isValid(item)) {
        this.cache.set(key, item);
      } else {
        return null;
      }
    }

    // Validate item
    if (!this.isValid(item!)) {
      await this.removeItem(key);
      this.updateStats({ misses: 1 });
      return null;
    }

    // Access control validation for legal documents
    if (item!.legal_sensitive && this.securityConfig.accessControlValidation) {
      const hasAccess = await this.validateLegalAccess(key, item!);
      if (!hasAccess) {
        this.logAccess(key, 'ACCESS_DENIED');
        return null;
      }
    }

    // Update access statistics
    item!.access_count++;
    item!.last_accessed = Date.now();
    
    // Decrypt if necessary
    let result = item!.data;
    if (item!.encrypted && this.encryptionKey) {
      try {
        const decrypted = await this.decryptData(result as any);
        result = JSON.parse(decrypted) as T;
      } catch (error) {
        console.error('Decryption failed:', error);
        await this.removeItem(key);
        return null;
      }
    }

    // Audit logging
    if (this.securityConfig.auditLogging && item!.legal_sensitive) {
      this.logAccess(key, 'GET');
    }
    
    this.updateStats({ hits: 1 });
    return result;
  }

  /**
   * Intelligent lazy loading with legal document prioritization
   */
  async lazyLoad<T>(
    key: string, 
    loader: string | (() => Promise<T>),
    options: CacheOptions & {
      prefetch?: boolean;
      legal_priority?: boolean;
    } = {}
  ): Promise<T | null> {
    const { 
      priority = 'medium', 
      prefetch = false,
      legal_priority = false,
      ...cacheOptions 
    } = options;

    // Check cache first
    const cached = await this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    // Avoid duplicate requests
    if (this.pendingLoads.has(key)) {
      return this.pendingLoads.get(key);
    }

    const loadPromise = (async () => {
      try {
        let data: T;
        
        if (typeof loader === 'string') {
          // Load from API endpoint with legal document headers
          const headers: Record<string, string> = {
            'Content-Type': 'application/json'
          };
          
          if (legal_priority) {
            headers['X-Legal-Priority'] = 'true';
            headers['X-Document-Type'] = cacheOptions.document_type || 'general';
          }
          
          const response = await fetch(loader, { headers });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          data = await response.json();
        } else {
          // Execute loader function
          data = await loader();
        }

        // Cache the result with appropriate settings
        await this.set(key, data, { 
          priority: legal_priority ? 'high' : priority,
          ttl: this.getDefaultTTL(cacheOptions.confidentiality_level),
          ...cacheOptions
        });

        return data;
      } catch (error) {
        console.error(`Failed to lazy load ${key}:`, error);
        return null;
      } finally {
        this.pendingLoads.delete(key);
      }
    })();

    this.pendingLoads.set(key, loadPromise);
    return loadPromise;
  }

  /**
   * Intelligent prefetching based on legal document access patterns
   */
  async prefetchByPattern(patterns: string[], options?: { legal_only?: boolean }): Promise<void> {
    const { legal_only = false } = options || {};
    
    const prefetchPromises = patterns.map(async (pattern) => {
      const keys = Array.from(this.cache.keys()).filter((key) => {
        const item = this.cache.get(key);
        const matchesPattern = key.includes(pattern) || this.getRelatedKeys(key, pattern).length > 0;
        const isLegalMatch = !legal_only || (item?.legal_sensitive === true);
        
        return matchesPattern && isLegalMatch;
      });

      // Prioritize frequently accessed legal documents
      const sortedKeys = keys
        .map(key => ({ key, item: this.cache.get(key)! }))
        .sort((a, b) => {
          if (a.item.legal_sensitive && !b.item.legal_sensitive) return -1;
          if (!a.item.legal_sensitive && b.item.legal_sensitive) return 1;
          return b.item.access_count - a.item.access_count;
        })
        .slice(0, 5); // Limit prefetching

      for (const { key, item } of sortedKeys) {
        if (item.access_count > 2) {
          // Increase priority for frequently accessed items
          item.priority = item.legal_sensitive ? 'critical' : 'high';
        }
      }
    });

    await Promise.all(prefetchPromises);
  }

  /**
   * Tag-based cache invalidation with legal document protection
   */
  async invalidateByTags(tags: string[], options?: { preserve_privileged?: boolean }): Promise<void> {
    const { preserve_privileged = true } = options || {};
    const toDelete: string[] = [];
    
    for (const [key, item] of this.cache.entries()) {
      // Protect privileged content unless explicitly requested
      if (preserve_privileged && item.confidentiality_level === 'privileged') {
        continue;
      }
      
      if (item.tags.some(tag => tags.includes(tag))) {
        toDelete.push(key);
      }
    }

    for (const key of toDelete) {
      await this.removeItem(key);
    }
  }

  /**
   * Legal document search within cache
   */
  async searchLegalDocuments(query: {
    document_type?: string;
    confidentiality_level?: string;
    tags?: string[];
    content_search?: string;
  }): Promise<Array<{ key: string; item: CacheItem }>> {
    const results: Array<{ key: string; item: CacheItem }> = [];
    
    for (const [key, item] of this.cache.entries()) {
      if (!item.legal_sensitive) continue;
      
      let matches = true;
      
      if (query.document_type && item.document_type !== query.document_type) {
        matches = false;
      }
      
      if (query.confidentiality_level && item.confidentiality_level !== query.confidentiality_level) {
        matches = false;
      }
      
      if (query.tags && !query.tags.some(tag => item.tags.includes(tag))) {
        matches = false;
      }
      
      if (query.content_search) {
        // Simple content search (would be enhanced with proper indexing)
        const dataStr = JSON.stringify(item.data).toLowerCase();
        if (!dataStr.includes(query.content_search.toLowerCase())) {
          matches = false;
        }
      }
      
      if (matches) {
        results.push({ key, item });
      }
    }
    
    return results.sort((a, b) => b.item.last_accessed - a.item.last_accessed);
  }

  /**
   * Observer for lazy loading elements with legal document support
   */
  observeElement(
    element: HTMLElement, 
    cacheKey: string, 
    loader: string,
    options?: { legal_sensitive?: boolean; document_type?: string }
  ): void {
    if (this.lazyLoadObserver) {
      element.dataset.cacheKey = cacheKey;
      element.dataset.loader = loader;
      element.dataset.legalSensitive = String(options?.legal_sensitive || false);
      element.dataset.documentType = options?.document_type || 'general';
      this.lazyLoadObserver.observe(element);
    }
  }

  /**
   * Get comprehensive cache statistics
   */
  getStats() {
    return this.stats;
  }

  /**
   * Get detailed performance metrics
   */
  getPerformanceMetrics(): CachePerformanceMetrics {
    const stats = get(this.stats);
    const hitRate = stats.hits / (stats.hits + stats.misses) || 0;
    
    return {
      hitRate,
      averageItemSize: stats.total_size / stats.items_count || 0,
      memoryEfficiency: (this.maxSize - stats.total_size) / this.maxSize,
      totalItems: stats.items_count,
      legalItemsRatio: stats.legal_items_count / stats.items_count || 0,
      privilegedItemsRatio: stats.privileged_items_count / stats.items_count || 0,
      encryptionOverhead: stats.encryption_overhead / stats.total_size || 0,
      averageAccessTime: this.calculateAverageAccessTime(),
      evictionRate: stats.evictions / (stats.items_count + stats.evictions) || 0
    };
  }

  /**
   * Get access audit log for legal compliance
   */
  getAccessAuditLog(limit = 100): Array<{ key: string; timestamp: number; action: string }> {
    return this.accessLog.slice(-limit);
  }

  /**
   * Export cache data for legal discovery/compliance
   */
  async exportLegalData(options?: { 
    include_privileged?: boolean; 
    document_types?: string[];
    date_range?: { start: number; end: number };
  }): Promise<{
    items: Array<{ key: string; metadata: Partial<CacheItem>; data?: unknown }>;
    audit_log: Array<{ key: string; timestamp: number; action: string }>;
  }> {
    const {
      include_privileged = false,
      document_types = [],
      date_range
    } = options || {};
    
    const items: Array<{ key: string; metadata: Partial<CacheItem>; data?: unknown }> = [];
    
    for (const [key, item] of this.cache.entries()) {
      if (!item.legal_sensitive) continue;
      
      // Filter by privilege level
      if (!include_privileged && item.confidentiality_level === 'privileged') {
        continue;
      }
      
      // Filter by document type
      if (document_types.length > 0 && !document_types.includes(item.document_type || 'general')) {
        continue;
      }
      
      // Filter by date range
      if (date_range && (item.timestamp < date_range.start || item.timestamp > date_range.end)) {
        continue;
      }
      
      const metadata: Partial<CacheItem> = {
        timestamp: item.timestamp,
        ttl: item.ttl,
        priority: item.priority,
        access_count: item.access_count,
        last_accessed: item.last_accessed,
        tags: item.tags,
        size: item.size,
        legal_sensitive: item.legal_sensitive,
        document_type: item.document_type,
        confidentiality_level: item.confidentiality_level
      };
      
      const exportItem: { key: string; metadata: Partial<CacheItem>; data?: unknown } = {
        key,
        metadata
      };
      
      // Include data for non-privileged items or when explicitly requested
      if (include_privileged || item.confidentiality_level !== 'privileged') {
        exportItem.data = item.encrypted ? '[ENCRYPTED]' : item.data;
      }
      
      items.push(exportItem);
    }
    
    return {
      items,
      audit_log: this.getAccessAuditLog(1000)
    };
  }

  // Private helper methods

  private async removeItem(key: string): Promise<void> {
    const item = this.cache.get(key);
    if (item) {
      this.cache.delete(key);
      this.updateStats({ 
        items_count: -1, 
        total_size: -item.size,
        legal_items_count: item.legal_sensitive ? -1 : 0,
        privileged_items_count: item.confidentiality_level === 'privileged' ? -1 : 0,
        encryption_overhead: item.encrypted ? -this.calculateEncryptionOverhead(item.data, item.data) : 0
      });
      
      // Remove from persistent storage
      await this.removeFromStorage(key);
    }
  }

  private isValid(item: CacheItem): boolean {
    const isNotExpired = Date.now() - item.timestamp < item.ttl;
    
    // Stricter validation for privileged content
    if (item.confidentiality_level === 'privileged') {
      const privilegedTTL = this.securityConfig.maxPrivilegedCacheTime;
      return Date.now() - item.timestamp < privilegedTTL;
    }
    
    return isNotExpired;
  }

  private async ensureCapacity(newItemSize: number): Promise<void> {
    const stats = get(this.stats);
    
    while (
      stats.total_size + newItemSize > this.maxSize ||
      stats.items_count >= this.maxItems
    ) {
      await this.evictLeastValuable();
    }
  }

  private async evictLeastValuable(): Promise<void> {
    let leastValuable: { key: string; score: number } | null = null;
    
    for (const [key, item] of this.cache.entries()) {
      // Never evict privileged content automatically
      if (item.confidentiality_level === 'privileged') continue;
      
      // Skip critical items unless absolutely necessary
      if (item.priority === 'critical' && this.cache.size > this.maxItems * 0.9) continue;
      
      const score = this.calculateEvictionScore(item);
      
      if (!leastValuable || score > leastValuable.score) {
        leastValuable = { key, score };
      }
    }

    if (leastValuable) {
      await this.removeItem(leastValuable.key);
      this.updateStats({ evictions: 1 });
    }
  }

  private async evictOldestPrivileged(): Promise<void> {
    let oldest: { key: string; timestamp: number } | null = null;
    
    for (const [key, item] of this.cache.entries()) {
      if (item.confidentiality_level === 'privileged') {
        if (!oldest || item.timestamp < oldest.timestamp) {
          oldest = { key, timestamp: item.timestamp };
        }
      }
    }
    
    if (oldest) {
      await this.removeItem(oldest.key);
    }
  }

  private calculateEvictionScore(item: CacheItem): number {
    const ageScore = (Date.now() - item.last_accessed) / item.ttl;
    const accessScore = 1 / (item.access_count + 1);
    const priorityScore = {
      low: 4,
      medium: 2,
      high: 1,
      critical: 0
    }[item.priority];
    
    // Legal documents get lower eviction scores (harder to evict)
    const legalBonus = item.legal_sensitive ? -0.5 : 0;
    
    return ageScore + accessScore + priorityScore + legalBonus;
  }

  private getRelatedKeys(key: string, pattern: string): string[] {
    return Array.from(this.cache.keys()).filter((k) => 
      k !== key && (
        k.startsWith(pattern) || 
        key.startsWith(pattern) ||
        this.levenshteinDistance(k, key) < 3
      )
    );
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => 
      Array(str1.length + 1).fill(null)
    );
    
    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
    
    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        );
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  private updateStats(delta: Partial<CacheStats>): void {
    this.stats.update((current) => {
      const updated = {
        hits: current.hits + (delta.hits || 0),
        misses: current.misses + (delta.misses || 0),
        evictions: current.evictions + (delta.evictions || 0),
        total_size: current.total_size + (delta.total_size || 0),
        items_count: current.items_count + (delta.items_count || 0),
        legal_items_count: current.legal_items_count + (delta.legal_items_count || 0),
        privileged_items_count: current.privileged_items_count + (delta.privileged_items_count || 0),
        encryption_overhead: current.encryption_overhead + (delta.encryption_overhead || 0),
        cache_efficiency: 0
      };
      
      // Calculate cache efficiency
      updated.cache_efficiency = updated.hits / (updated.hits + updated.misses) || 0;
      
      return updated;
    });
  }

  private setupPeriodicCleanup(): void {
    if (browser) {
      setInterval(async () => {
        const now = Date.now();
        const toDelete: string[] = [];
        
        for (const [key, item] of this.cache.entries()) {
          if (!this.isValid(item)) {
            toDelete.push(key);
          }
        }
        
        for (const key of toDelete) {
          await this.removeItem(key);
        }
      }, 60000); // Clean up every minute
    }
  }

  private setupPrivilegedContentMonitoring(): void {
    if (browser) {
      setInterval(() => {
        const privilegedItems = Array.from(this.cache.entries())
          .filter(([_, item]) => item.confidentiality_level === 'privileged');
        
        // Force cleanup of expired privileged content
        for (const [key, item] of privilegedItems) {
          if (Date.now() - item.timestamp > this.securityConfig.maxPrivilegedCacheTime) {
            this.removeItem(key);
            this.logAccess(key, 'PRIVILEGED_EXPIRED');
          }
        }
      }, 30000); // Check every 30 seconds
    }
  }

  private async loadFromPersistentStorage(): Promise<void> {
    if (browser) {
      // Load from localStorage
      try {
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key?.startsWith('legal_cache_')) {
            const cacheKey = key.substring(12);
            const stored = localStorage.getItem(key);
            if (stored) {
              const item = JSON.parse(stored) as CacheItem;
              if (this.isValid(item)) {
                this.cache.set(cacheKey, item);
                this.updateStats({ 
                  items_count: 1, 
                  total_size: item.size,
                  legal_items_count: item.legal_sensitive ? 1 : 0,
                  privileged_items_count: item.confidentiality_level === 'privileged' ? 1 : 0
                });
              } else {
                localStorage.removeItem(key);
              }
            }
          }
        }
      } catch (e) {
        console.warn('Failed to load cache from localStorage:', e);
      }
    }
  }

  // Security and encryption methods

  private shouldEncrypt(options: CacheOptions): boolean {
    if (!this.securityConfig.enableEncryption) return false;
    if (options.encrypt !== undefined) return options.encrypt;
    if (options.confidentiality_level === 'privileged') return this.securityConfig.encryptPrivileged;
    return options.legal_sensitive || false;
  }

  private getDefaultTTL(confidentiality_level?: string): number {
    switch (confidentiality_level) {
      case 'privileged':
        return this.securityConfig.maxPrivilegedCacheTime;
      case 'confidential':
        return 60 * 60 * 1000; // 1 hour
      default:
        return 5 * 60 * 1000; // 5 minutes
    }
  }

  private async getOrCreateEncryptionKey(): Promise<CryptoKey> {
    // In a real implementation, this would use a secure key management system
    const keyData = new Uint8Array(32);
    crypto.getRandomValues(keyData);
    
    return await crypto.subtle.importKey(
      'raw',
      keyData,
      { name: 'AES-GCM' },
      false,
      ['encrypt', 'decrypt']
    );
  }

  private async encryptData(data: string): Promise<string> {
    if (!this.encryptionKey) throw new Error('Encryption key not available');
    
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(data);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    
    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      this.encryptionKey,
      dataBuffer
    );
    
    const combined = new Uint8Array(iv.length + encrypted.byteLength);
    combined.set(iv);
    combined.set(new Uint8Array(encrypted), iv.length);
    
    return btoa(String.fromCharCode(...combined));
  }

  private async decryptData(encryptedData: string): Promise<string> {
    if (!this.encryptionKey) throw new Error('Encryption key not available');
    
    const combined = new Uint8Array(atob(encryptedData).split('').map(c => c.charCodeAt(0)));
    const iv = combined.slice(0, 12);
    const encrypted = combined.slice(12);
    
    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv },
      this.encryptionKey,
      encrypted
    );
    
    const decoder = new TextDecoder();
    return decoder.decode(decrypted);
  }

  private async generateChecksum(data: string): Promise<string> {
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(data);
    const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  private calculateEncryptionOverhead(original: any, encrypted: any): number {
    const originalSize = new Blob([JSON.stringify(original)]).size;
    const encryptedSize = new Blob([JSON.stringify(encrypted)]).size;
    return encryptedSize - originalSize;
  }

  private async validatePrivilegedAccess(key: string): Promise<void> {
    // In a real implementation, this would validate user permissions
    // For now, we'll just log the access attempt
    this.logAccess(key, 'PRIVILEGED_ACCESS_ATTEMPT');
  }

  private async validateLegalAccess(key: string, item: CacheItem): Promise<boolean> {
    // In a real implementation, this would check user permissions against the document
    // For now, we'll assume access is granted but log it
    this.logAccess(key, 'LEGAL_ACCESS');
    return true;
  }

  private logAccess(key: string, action: string): void {
    if (this.securityConfig.auditLogging) {
      this.accessLog.push({
        key,
        timestamp: Date.now(),
        action
      });
      
      // Keep only the last 1000 log entries
      if (this.accessLog.length > 1000) {
        this.accessLog = this.accessLog.slice(-1000);
      }
    }
  }

  private getPrivilegedItemsCount(): number {
    return Array.from(this.cache.values())
      .filter(item => item.confidentiality_level === 'privileged').length;
  }

  private calculateAverageAccessTime(): number {
    const items = Array.from(this.cache.values());
    if (items.length === 0) return 0;
    
    const totalTime = items.reduce((sum, item) => {
      return sum + (Date.now() - item.timestamp) / item.access_count || 0;
    }, 0);
    
    return totalTime / items.length;
  }

  private async persistToStorage(key: string, item: CacheItem): Promise<void> {
    if (browser) {
      try {
        // Use IndexedDB for large items, localStorage for small ones
        if (item.size > 1024 * 1024 && this.indexDB) { // > 1MB
          const transaction = this.indexDB.transaction(['cache'], 'readwrite');
          const store = transaction.objectStore('cache');
          await store.put({ key, ...item });
        } else {
          localStorage.setItem(`legal_cache_${key}`, JSON.stringify(item));
        }
      } catch (error) {
        console.warn('Failed to persist to storage:', error);
      }
    }
  }

  private async loadFromStorage<T>(key: string): Promise<CacheItem<T> | null> {
    if (!browser) return null;
    
    try {
      // Try localStorage first
      const stored = localStorage.getItem(`legal_cache_${key}`);
      if (stored) {
        return JSON.parse(stored) as CacheItem<T>;
      }
      
      // Try IndexedDB if available
      if (this.indexDB) {
        return new Promise((resolve) => {
          const transaction = this.indexDB!.transaction(['cache'], 'readonly');
          const store = transaction.objectStore('cache');
          const request = store.get(key);
          
          request.onsuccess = () => {
            resolve(request.result || null);
          };
          
          request.onerror = () => {
            resolve(null);
          };
        });
      }
    } catch (error) {
      console.warn('Failed to load from storage:', error);
    }
    
    return null;
  }

  private async removeFromStorage(key: string): Promise<void> {
    if (browser) {
      try {
        localStorage.removeItem(`legal_cache_${key}`);
        
        if (this.indexDB) {
          const transaction = this.indexDB.transaction(['cache'], 'readwrite');
          const store = transaction.objectStore('cache');
          store.delete(key);
        }
      } catch (error) {
        console.warn('Failed to remove from storage:', error);
      }
    }
  }
}

// Export singleton instance
export const advancedCache = new AdvancedCacheManager();

// Export factory function for custom instances
export function createAdvancedCacheManager(config?: Partial<SecurityConfig>): AdvancedCacheManager {
  return new AdvancedCacheManager(config);
}

// Export utility functions for legal document caching
export const legalCacheUtils = {
  /**
   * Cache legal document with appropriate security settings
   */
  cacheLegalDocument: async <T>(
    key: string,
    document: T,
    options: {
      document_type: 'evidence' | 'contract' | 'case_file';
      confidentiality_level: 'public' | 'confidential' | 'privileged';
      tags?: string[];
    }
  ) => {
    return advancedCache.set(key, document, {
      legal_sensitive: true,
      encrypt: options.confidentiality_level !== 'public',
      priority: options.confidentiality_level === 'privileged' ? 'critical' : 'high',
      ttl: options.confidentiality_level === 'privileged' ? 30 * 60 * 1000 : 60 * 60 * 1000,
      ...options
    });
  },

  /**
   * Search cached legal documents
   */
  searchLegalDocuments: (query: {
    document_type?: string;
    confidentiality_level?: string;
    tags?: string[];
    content_search?: string;
  }) => {
    return advancedCache.searchLegalDocuments(query);
  },

  /**
   * Export legal data for compliance
   */
  exportForCompliance: (options?: {
    include_privileged?: boolean;
    document_types?: string[];
    date_range?: { start: number; end: number };
  }) => {
    return advancedCache.exportLegalData(options);
  },

  /**
   * Get audit trail
   */
  getAuditTrail: (limit?: number) => {
    return advancedCache.getAccessAuditLog(limit);
  }
};

export default AdvancedCacheManager;