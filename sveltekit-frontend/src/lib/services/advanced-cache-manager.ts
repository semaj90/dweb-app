// @ts-nocheck
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';

export interface CacheItem<T = any> {
  data: T;
  timestamp: number;
  ttl: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  access_count: number;
  last_accessed: number;
  tags: string[];
  size: number;
}

export interface LazyLoadConfig {
  threshold: number;
  rootMargin: string;
  enabled: boolean;
}

export interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  total_size: number;
  items_count: number;
}

class AdvancedCacheManager {
  private cache = new Map<string, CacheItem>();
  private stats = writable<CacheStats>({
    hits: 0,
    misses: 0,
    evictions: 0,
    total_size: 0,
    items_count: 0
  });
  
  private maxSize = 50 * 1024 * 1024; // 50MB
  private maxItems = 1000;
  private lazyLoadObserver: IntersectionObserver | null = null;
  private pendingLoads = new Map<string, Promise<any>>();

  constructor() {
    if (browser) {
      this.initializeLazyLoading();
      this.setupPeriodicCleanup();
      this.loadFromStorage();
    }
  }

  private initializeLazyLoading() {
    if ('IntersectionObserver' in window) {
      this.lazyLoadObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry: any) => {
            if (entry.isIntersecting) {
              const element = entry.target as HTMLElement;
              const cacheKey = element.dataset.cacheKey;
              const loader = element.dataset.loader;
              
              if (cacheKey && loader) {
                this.lazyLoad(cacheKey, loader);
                this.lazyLoadObserver?.unobserve(element);
              }
            }
          });
        },
        {
          threshold: 0.1,
          rootMargin: '50px'
        }
      );
    }
  }

  // Advanced caching with priority-based eviction
  async set<T>(
    key: string, 
    data: T, 
    options: {
      ttl?: number;
      priority?: 'low' | 'medium' | 'high' | 'critical';
      tags?: string[];
    } = {}
  ): Promise<void> {
    const {
      ttl = 5 * 60 * 1000, // 5 minutes default
      priority = 'medium',
      tags = []
    } = options;

    const serialized = JSON.stringify(data);
    const size = new Blob([serialized]).size;

    // Check if we need to evict items
    await this.ensureCapacity(size);

    const item: CacheItem<T> = {
      data,
      timestamp: Date.now(),
      ttl,
      priority,
      access_count: 0,
      last_accessed: Date.now(),
      tags,
      size
    };

    this.cache.set(key, item);
    this.updateStats({ items_count: 1, total_size: size });
    
    // Persist critical items to localStorage
    if (priority === 'critical' && browser) {
      try {
        localStorage.setItem(`cache_${key}`, JSON.stringify(item));
      } catch (e) {
        console.warn('Failed to persist to localStorage:', e);
      }
    }
  }

  async get<T>(key: string): Promise<T | null> {
    const item = this.cache.get(key) as CacheItem<T> | undefined;
    
    if (!item) {
      this.updateStats({ misses: 1 });
      
      // Try to load from localStorage for critical items
      if (browser) {
        try {
          const stored = localStorage.getItem(`cache_${key}`);
          if (stored) {
            const parsedItem = JSON.parse(stored) as CacheItem<T>;
            if (this.isValid(parsedItem)) {
              this.cache.set(key, parsedItem);
              this.updateStats({ hits: 1 });
              return parsedItem.data;
            }
          }
        } catch (e) {
          console.warn('Failed to load from localStorage:', e);
        }
      }
      
      return null;
    }

    // Check if item is expired
    if (!this.isValid(item)) {
      this.cache.delete(key);
      this.updateStats({ misses: 1, items_count: -1, total_size: -item.size });
      return null;
    }

    // Update access statistics
    item.access_count++;
    item.last_accessed = Date.now();
    
    this.updateStats({ hits: 1 });
    return item.data;
  }

  // Lazy loading with intelligent prefetching
  async lazyLoad<T>(
    key: string, 
    loader: string | (() => Promise<T>),
    options: {
      priority?: 'low' | 'medium' | 'high' | 'critical';
      prefetch?: boolean;
    } = {}
  ): Promise<T | null> {
    const { priority = 'medium', prefetch = false } = options;

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
          // Load from API endpoint
          const response = await fetch(loader);
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          data = await response.json();
        } else {
          // Execute loader function
          data = await loader();
        }

        // Cache the result
        await this.set(key, data, { 
          priority,
          ttl: priority === 'critical' ? 24 * 60 * 60 * 1000 : 5 * 60 * 1000 // 24h for critical, 5min for others
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

  // Smart prefetching based on user patterns
  async prefetchByPattern(patterns: string[]): Promise<void> {
    const prefetchPromises = patterns.map(async (pattern) => {
      // Simulate intelligent prefetching logic
      const keys = Array.from(this.cache.keys()).filter((key: any) => key.includes(pattern) || this.getRelatedKeys(key, pattern).length > 0
      );

      for (const key of keys.slice(0, 3)) { // Limit to 3 prefetches per pattern
        const item = this.cache.get(key);
        if (item && item.access_count > 2) {
          // This item is frequently accessed, prioritize it
          item.priority = 'high';
        }
      }
    });

    await Promise.all(prefetchPromises);
  }

  // Tag-based cache invalidation
  async invalidateByTags(tags: string[]): Promise<void> {
    const toDelete: string[] = [];
    
    for (const [key, item] of this.cache.entries()) {
      if (item.tags.some((tag: any) => tags.includes(tag))) {
        toDelete.push(key);
      }
    }

    for (const key of toDelete) {
      const item = this.cache.get(key);
      if (item) {
        this.cache.delete(key);
        this.updateStats({ items_count: -1, total_size: -item.size });
        
        // Remove from localStorage if it exists
        if (browser) {
          localStorage.removeItem(`cache_${key}`);
        }
      }
    }
  }

  // Observer for lazy loading elements
  observeElement(element: HTMLElement, cacheKey: string, loader: string): void {
    if (this.lazyLoadObserver) {
      element.dataset.cacheKey = cacheKey;
      element.dataset.loader = loader;
      this.lazyLoadObserver.observe(element);
    }
  }

  // Get cache statistics
  getStats() {
    return this.stats;
  }

  // Get performance metrics
  getPerformanceMetrics() {
    const stats = get(this.stats);
    const hitRate = stats.hits / (stats.hits + stats.misses) || 0;
    
    return {
      hitRate,
      averageItemSize: stats.total_size / stats.items_count || 0,
      memoryEfficiency: (this.maxSize - stats.total_size) / this.maxSize,
      totalItems: stats.items_count
    };
  }

  private isValid(item: CacheItem): boolean {
    return Date.now() - item.timestamp < item.ttl;
  }

  private async ensureCapacity(newItemSize: number): Promise<void> {
    const stats = get(this.stats);
    
    // Check if we need to evict items
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
      // Skip critical items unless we're really out of space
      if (item.priority === 'critical') continue;
      
      // Calculate value score (lower is less valuable)
      const ageScore = (Date.now() - item.last_accessed) / item.ttl;
      const accessScore = 1 / (item.access_count + 1);
      const priorityScore = {
        low: 4,
        medium: 2,
        high: 1,
        critical: 0
      }[item.priority];
      
      const totalScore = ageScore + accessScore + priorityScore;
      
      if (!leastValuable || totalScore > leastValuable.score) {
        leastValuable = { key, score: totalScore };
      }
    }

    if (leastValuable) {
      const item = this.cache.get(leastValuable.key);
      if (item) {
        this.cache.delete(leastValuable.key);
        this.updateStats({ 
          evictions: 1, 
          items_count: -1, 
          total_size: -item.size 
        });
      }
    }
  }

  private getRelatedKeys(key: string, pattern: string): string[] {
    // Simple relationship detection - can be made more sophisticated
    return Array.from(this.cache.keys()).filter((k: any) => k !== key && (
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
    this.stats.update((current: any) => ({
      hits: current.hits + (delta.hits || 0),
      misses: current.misses + (delta.misses || 0),
      evictions: current.evictions + (delta.evictions || 0),
      total_size: current.total_size + (delta.total_size || 0),
      items_count: current.items_count + (delta.items_count || 0)
    }));
  }

  private setupPeriodicCleanup(): void {
    if (browser) {
      setInterval(() => {
        const now = Date.now();
        const toDelete: string[] = [];
        
        for (const [key, item] of this.cache.entries()) {
          if (now - item.timestamp > item.ttl) {
            toDelete.push(key);
          }
        }
        
        for (const key of toDelete) {
          const item = this.cache.get(key);
          if (item) {
            this.cache.delete(key);
            this.updateStats({ items_count: -1, total_size: -item.size });
          }
        }
      }, 60000); // Clean up every minute
    }
  }

  private loadFromStorage(): void {
    if (browser) {
      try {
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key?.startsWith('cache_')) {
            const cacheKey = key.substring(6);
            const stored = localStorage.getItem(key);
            if (stored) {
              const item = JSON.parse(stored) as CacheItem;
              if (this.isValid(item)) {
                this.cache.set(cacheKey, item);
                this.updateStats({ items_count: 1, total_size: item.size });
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
}

export const advancedCache = new AdvancedCacheManager();