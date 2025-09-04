/**
 * GPU Cache Invalidation & Cleanup System
 * Provides automated cache management with LRU eviction, memory pressure handling,
 * and predictive cache warming for enhanced performance
 */

import type { 
  EnhancedGPUCacheEntry, 
  TextureCacheEntry, 
  CompiledShaderCache,
  GPUCacheMetadata,
  CachePerformanceTracker
} from '$lib/types/gpu-cache-integration';

// Cache invalidation strategies
export type InvalidationStrategy = 
  | 'lru'           // Least Recently Used
  | 'lfu'           // Least Frequently Used  
  | 'ttl'           // Time To Live
  | 'memory-pressure' // Based on memory usage
  | 'adaptive'      // Combines multiple strategies
  | 'predictive';   // Based on usage patterns

export interface CacheInvalidationConfig {
  strategy: InvalidationStrategy;
  maxMemoryMB: number;
  maxEntries: number;
  defaultTTL: number; // milliseconds
  memoryPressureThreshold: number; // 0-1 (percentage)
  cleanupInterval: number; // milliseconds
  predictiveWindow: number; // milliseconds for pattern analysis
  enableWarmup: boolean;
  warmupBatchSize: number;
}

export interface CacheEntry extends EnhancedGPUCacheEntry {
  lastAccessed: number;
  accessCount: number;
  priority: number; // Higher = more important to keep
  dependencies: string[]; // Cache keys this entry depends on
  warmupScore: number; // Likelihood of being needed soon (0-1)
}

export interface MemoryPressureMetrics {
  totalMemoryMB: number;
  usedMemoryMB: number;
  utilizationPercentage: number;
  fragmentationIndex: number; // 0-1, higher = more fragmented
  pressureLevel: 'low' | 'medium' | 'high' | 'critical';
  recommendedCleanupMB: number;
}

export interface PredictivePattern {
  pattern: string; // Pattern identifier
  frequency: number;
  lastOccurrence: number;
  nextPrediction: number;
  confidence: number; // 0-1
  associatedKeys: string[];
}

export interface InvalidationEvent {
  timestamp: number;
  type: 'evicted' | 'expired' | 'dependency-invalidated' | 'manual' | 'memory-pressure';
  key: string;
  reason: string;
  memoryFreedMB: number;
  strategy: InvalidationStrategy;
}

export class GPUCacheInvalidationSystem {
  private config: CacheInvalidationConfig;
  private cacheEntries = new Map<string, CacheEntry>();
  private accessHistory: { key: string; timestamp: number }[] = [];
  private patterns = new Map<string, PredictivePattern>();
  private invalidationEvents: InvalidationEvent[] = [];
  private cleanupTimer: number | null = null;
  private memoryMonitorTimer: number | null = null;

  constructor(config: Partial<CacheInvalidationConfig> = {}) {
    this.config = {
      strategy: 'adaptive',
      maxMemoryMB: 512,
      maxEntries: 1000,
      defaultTTL: 30 * 60 * 1000, // 30 minutes
      memoryPressureThreshold: 0.85,
      cleanupInterval: 60 * 1000, // 1 minute
      predictiveWindow: 5 * 60 * 1000, // 5 minutes
      enableWarmup: true,
      warmupBatchSize: 10,
      ...config
    };

    this.startAutomaticCleanup();
    this.startMemoryMonitoring();
  }

  /**
   * Register a cache entry for management
   */
  registerEntry(key: string, entry: EnhancedGPUCacheEntry): void {
    const cacheEntry: CacheEntry = {
      ...entry,
      lastAccessed: Date.now(),
      accessCount: 1,
      priority: this.calculatePriority(entry),
      dependencies: this.extractDependencies(entry),
      warmupScore: 0
    };

    this.cacheEntries.set(key, cacheEntry);
    this.recordAccess(key);
    
    // Trigger cleanup if we exceed limits
    if (this.cacheEntries.size > this.config.maxEntries) {
      this.performCleanup('entry-limit-exceeded');
    }
  }

  /**
   * Record cache access for LRU/LFU tracking
   */
  recordAccess(key: string): void {
    const entry = this.cacheEntries.get(key);
    if (entry) {
      entry.lastAccessed = Date.now();
      entry.accessCount++;
      
      // Record for pattern analysis
      this.accessHistory.push({ key, timestamp: Date.now() });
      
      // Keep access history manageable
      if (this.accessHistory.length > 10000) {
        this.accessHistory = this.accessHistory.slice(-5000);
      }
      
      // Update predictive patterns
      this.updatePredictivePatterns(key);
    }
  }

  /**
   * Check if entry should be invalidated
   */
  shouldInvalidate(key: string, entry: CacheEntry): boolean {
    const now = Date.now();
    
    // TTL check
    if (entry.creationTime + this.config.defaultTTL < now) {
      return true;
    }
    
    // Dependency invalidation check
    if (entry.dependencies.some(dep => !this.cacheEntries.has(dep))) {
      return true;
    }
    
    // Memory pressure check
    const memoryMetrics = this.getMemoryPressureMetrics();
    if (memoryMetrics.pressureLevel === 'critical' && entry.priority < 0.5) {
      return true;
    }
    
    return false;
  }

  /**
   * Perform cache cleanup based on configured strategy
   */
  async performCleanup(reason: string): Promise<number> {
    const startTime = Date.now();
    let cleanedCount = 0;
    let memoryFreed = 0;

    const memoryMetrics = this.getMemoryPressureMetrics();
    const targetCleanupMB = memoryMetrics.recommendedCleanupMB;

    console.log(`[GPU Cache Cleanup] Starting cleanup: ${reason}, target: ${targetCleanupMB}MB`);

    // Get entries to clean based on strategy
    const entriesToClean = this.getEntriesForCleanup(targetCleanupMB);

    for (const [key, entry] of entriesToClean) {
      try {
        // Calculate memory freed (approximate)
        const entrySize = this.estimateEntrySize(entry);
        
        // Remove from cache
        this.cacheEntries.delete(key);
        
        // Record invalidation event
        this.recordInvalidationEvent({
          timestamp: Date.now(),
          type: 'evicted',
          key,
          reason: `${this.config.strategy} cleanup: ${reason}`,
          memoryFreedMB: entrySize,
          strategy: this.config.strategy
        });

        memoryFreed += entrySize;
        cleanedCount++;

        // Stop if we've freed enough memory
        if (memoryFreed >= targetCleanupMB) {
          break;
        }
      } catch (error) {
        console.error(`[GPU Cache Cleanup] Error cleaning entry ${key}:`, error);
      }
    }

    const duration = Date.now() - startTime;
    console.log(`[GPU Cache Cleanup] Completed in ${duration}ms: ${cleanedCount} entries, ${memoryFreed.toFixed(2)}MB freed`);

    // Trigger predictive warmup after cleanup
    if (this.config.enableWarmup && cleanedCount > 0) {
      setTimeout(() => this.performPredictiveWarmup(), 1000);
    }

    return cleanedCount;
  }

  /**
   * Get entries for cleanup based on configured strategy
   */
  private getEntriesForCleanup(targetCleanupMB: number): [string, CacheEntry][] {
    const entries = Array.from(this.cacheEntries.entries());
    const now = Date.now();

    switch (this.config.strategy) {
      case 'lru':
        return entries
          .sort(([, a], [, b]) => a.lastAccessed - b.lastAccessed)
          .filter(([key, entry]) => this.shouldInvalidate(key, entry));

      case 'lfu':
        return entries
          .sort(([, a], [, b]) => a.accessCount - b.accessCount)
          .filter(([key, entry]) => this.shouldInvalidate(key, entry));

      case 'ttl':
        return entries
          .filter(([key, entry]) => 
            entry.creationTime + this.config.defaultTTL < now)
          .sort(([, a], [, b]) => a.creationTime - b.creationTime);

      case 'memory-pressure':
        return entries
          .sort(([, a], [, b]) => {
            const sizeA = this.estimateEntrySize(a);
            const sizeB = this.estimateEntrySize(b);
            return sizeB / Math.max(a.priority, 0.1) - sizeA / Math.max(b.priority, 0.1);
          })
          .filter(([, entry]) => entry.priority < 0.7);

      case 'adaptive':
        return entries
          .sort(([, a], [, b]) => {
            // Combined score: recency, frequency, priority, memory impact
            const scoreA = this.calculateAdaptiveScore(a);
            const scoreB = this.calculateAdaptiveScore(b);
            return scoreA - scoreB;
          })
          .filter(([key, entry]) => 
            this.shouldInvalidate(key, entry) || entry.priority < 0.5);

      case 'predictive':
        return entries
          .filter(([, entry]) => entry.warmupScore < 0.3)
          .sort(([, a], [, b]) => a.warmupScore - b.warmupScore);

      default:
        return entries.filter(([key, entry]) => this.shouldInvalidate(key, entry));
    }
  }

  /**
   * Calculate adaptive cleanup score (lower = clean first)
   */
  private calculateAdaptiveScore(entry: CacheEntry): number {
    const now = Date.now();
    const age = now - entry.creationTime;
    const timeSinceAccess = now - entry.lastAccessed;
    const size = this.estimateEntrySize(entry);

    // Normalize factors (0-1)
    const recencyScore = Math.min(timeSinceAccess / (24 * 60 * 60 * 1000), 1); // Days since access
    const frequencyScore = 1 / Math.max(entry.accessCount, 1); // Inverse frequency
    const sizeScore = size / 100; // Size impact (MB)
    const priorityScore = 1 - entry.priority; // Inverse priority
    const ageScore = Math.min(age / (7 * 24 * 60 * 60 * 1000), 1); // Age in weeks

    // Weighted combination
    return (
      recencyScore * 0.3 +
      frequencyScore * 0.2 + 
      sizeScore * 0.2 +
      priorityScore * 0.2 +
      ageScore * 0.1
    );
  }

  /**
   * Perform predictive cache warmup based on usage patterns
   */
  async performPredictiveWarmup(): Promise<void> {
    if (!this.config.enableWarmup) return;

    const now = Date.now();
    const predictions = Array.from(this.patterns.values())
      .filter(pattern => 
        pattern.nextPrediction <= now + this.config.predictiveWindow &&
        pattern.confidence > 0.5
      )
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, this.config.warmupBatchSize);

    console.log(`[GPU Cache Warmup] Starting predictive warmup for ${predictions.length} patterns`);

    for (const pattern of predictions) {
      try {
        // This would trigger pre-loading of associated cache entries
        await this.warmupPattern(pattern);
      } catch (error) {
        console.error(`[GPU Cache Warmup] Error warming pattern ${pattern.pattern}:`, error);
      }
    }
  }

  /**
   * Warmup cache entries for a specific pattern
   */
  private async warmupPattern(pattern: PredictivePattern): Promise<void> {
    // This would be implemented to pre-load cache entries
    // For now, just update the warmup scores
    for (const key of pattern.associatedKeys) {
      const entry = this.cacheEntries.get(key);
      if (entry) {
        entry.warmupScore = Math.min(entry.warmupScore + pattern.confidence * 0.3, 1.0);
      }
    }
  }

  /**
   * Update predictive patterns based on access history
   */
  private updatePredictivePatterns(key: string): void {
    const now = Date.now();
    const recentAccesses = this.accessHistory
      .filter(access => now - access.timestamp <= this.config.predictiveWindow)
      .map(access => access.key);

    // Simple pattern detection: sequences of 2-3 cache keys
    for (let i = 1; i < Math.min(recentAccesses.length, 4); i++) {
      const sequence = recentAccesses.slice(-i - 1, -1).join('→') + '→' + key;
      
      let pattern = this.patterns.get(sequence);
      if (!pattern) {
        pattern = {
          pattern: sequence,
          frequency: 0,
          lastOccurrence: now,
          nextPrediction: now + (this.config.predictiveWindow / 2),
          confidence: 0.1,
          associatedKeys: [key]
        };
        this.patterns.set(sequence, pattern);
      }

      pattern.frequency++;
      pattern.lastOccurrence = now;
      pattern.confidence = Math.min(pattern.confidence + 0.05, 1.0);
      
      // Update next prediction based on historical intervals
      if (pattern.frequency > 1) {
        const avgInterval = this.config.predictiveWindow / pattern.frequency;
        pattern.nextPrediction = now + avgInterval;
      }
    }

    // Clean old patterns
    const cutoff = now - (this.config.predictiveWindow * 3);
    for (const [patternKey, pattern] of this.patterns.entries()) {
      if (pattern.lastOccurrence < cutoff) {
        this.patterns.delete(patternKey);
      }
    }
  }

  /**
   * Get current memory pressure metrics
   */
  getMemoryPressureMetrics(): MemoryPressureMetrics {
    const totalMemory = this.config.maxMemoryMB;
    const usedMemory = Array.from(this.cacheEntries.values())
      .reduce((sum, entry) => sum + this.estimateEntrySize(entry), 0);
    
    const utilization = usedMemory / totalMemory;
    const fragmentationIndex = this.calculateFragmentationIndex();

    let pressureLevel: MemoryPressureMetrics['pressureLevel'] = 'low';
    let recommendedCleanupMB = 0;

    if (utilization > 0.95) {
      pressureLevel = 'critical';
      recommendedCleanupMB = totalMemory * 0.3; // Clean 30%
    } else if (utilization > this.config.memoryPressureThreshold) {
      pressureLevel = 'high';
      recommendedCleanupMB = totalMemory * 0.2; // Clean 20%
    } else if (utilization > 0.7) {
      pressureLevel = 'medium';
      recommendedCleanupMB = totalMemory * 0.1; // Clean 10%
    }

    return {
      totalMemoryMB: totalMemory,
      usedMemoryMB: usedMemory,
      utilizationPercentage: utilization * 100,
      fragmentationIndex,
      pressureLevel,
      recommendedCleanupMB
    };
  }

  /**
   * Calculate cache fragmentation index
   */
  private calculateFragmentationIndex(): number {
    // Simplified fragmentation calculation based on size distribution
    const sizes = Array.from(this.cacheEntries.values()).map(entry => this.estimateEntrySize(entry));
    if (sizes.length === 0) return 0;

    const avgSize = sizes.reduce((sum, size) => sum + size, 0) / sizes.length;
    const variance = sizes.reduce((sum, size) => sum + Math.pow(size - avgSize, 2), 0) / sizes.length;
    const stdDev = Math.sqrt(variance);
    
    // Higher standard deviation relative to mean indicates more fragmentation
    return Math.min(stdDev / Math.max(avgSize, 0.1), 1.0);
  }

  /**
   * Estimate memory size of cache entry (MB)
   */
  private estimateEntrySize(entry: CacheEntry): number {
    let size = 0.1; // Base overhead

    // Texture cache entries
    if ('textureData' in entry && entry.textureData) {
      const textureEntry = entry as TextureCacheEntry;
      // Approximate based on dimensions and format
      const width = textureEntry.dimensions?.width ?? 512;
      const height = textureEntry.dimensions?.height ?? 512;
      const bytesPerPixel = 4; // RGBA
      size += (width * height * bytesPerPixel) / (1024 * 1024); // Convert to MB
    }

    // Shader cache entries  
    if ('compiledShader' in entry) {
      size += 0.5; // Approximate shader size
    }

    // Metadata and additional data
    size += 0.01; // Metadata overhead

    return size;
  }

  /**
   * Calculate entry priority based on characteristics
   */
  private calculatePriority(entry: EnhancedGPUCacheEntry): number {
    let priority = 0.5; // Base priority

    // Higher priority for more recent entries
    const age = Date.now() - entry.creationTime;
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    priority += Math.max(0, (maxAge - age) / maxAge) * 0.2;

    // Higher priority for entries with performance impact
    if (entry.metadata?.performanceImpact === 'high') {
      priority += 0.3;
    } else if (entry.metadata?.performanceImpact === 'medium') {
      priority += 0.15;
    }

    return Math.min(Math.max(priority, 0), 1);
  }

  /**
   * Extract dependencies from cache entry
   */
  private extractDependencies(entry: EnhancedGPUCacheEntry): string[] {
    const dependencies: string[] = [];
    
    // Add dependencies based on metadata
    if (entry.metadata?.dependencies) {
      dependencies.push(...entry.metadata.dependencies);
    }
    
    return dependencies;
  }

  /**
   * Record invalidation event for analytics
   */
  private recordInvalidationEvent(event: InvalidationEvent): void {
    this.invalidationEvents.push(event);
    
    // Keep events manageable
    if (this.invalidationEvents.length > 1000) {
      this.invalidationEvents = this.invalidationEvents.slice(-500);
    }
  }

  /**
   * Start automatic cleanup timer
   */
  private startAutomaticCleanup(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
    }

    this.cleanupTimer = setInterval(() => {
      this.performCleanup('periodic-cleanup').catch(error => {
        console.error('[GPU Cache Cleanup] Automatic cleanup failed:', error);
      });
    }, this.config.cleanupInterval);
  }

  /**
   * Start memory monitoring
   */
  private startMemoryMonitoring(): void {
    if (this.memoryMonitorTimer) {
      clearInterval(this.memoryMonitorTimer);
    }

    this.memoryMonitorTimer = setInterval(() => {
      const metrics = this.getMemoryPressureMetrics();
      
      if (metrics.pressureLevel === 'critical') {
        console.warn('[GPU Cache] Critical memory pressure detected, forcing cleanup');
        this.performCleanup('memory-pressure-critical');
      } else if (metrics.pressureLevel === 'high') {
        console.log('[GPU Cache] High memory pressure, scheduling cleanup');
        setTimeout(() => this.performCleanup('memory-pressure-high'), 5000);
      }
    }, 30000); // Check every 30 seconds
  }

  /**
   * Get invalidation analytics
   */
  getAnalytics(): {
    totalInvalidations: number;
    invalidationsByType: Record<InvalidationEvent['type'], number>;
    invalidationsByStrategy: Record<InvalidationStrategy, number>;
    memoryFreedMB: number;
    avgInvalidationsPerHour: number;
  } {
    const now = Date.now();
    const hourMs = 60 * 60 * 1000;
    const recentEvents = this.invalidationEvents.filter(event => 
      now - event.timestamp <= hourMs
    );

    const byType = this.invalidationEvents.reduce((acc, event) => {
      acc[event.type] = (acc[event.type] || 0) + 1;
      return acc;
    }, {} as Record<InvalidationEvent['type'], number>);

    const byStrategy = this.invalidationEvents.reduce((acc, event) => {
      acc[event.strategy] = (acc[event.strategy] || 0) + 1;
      return acc;
    }, {} as Record<InvalidationStrategy, number>);

    return {
      totalInvalidations: this.invalidationEvents.length,
      invalidationsByType: byType,
      invalidationsByStrategy: byStrategy,
      memoryFreedMB: this.invalidationEvents.reduce((sum, event) => sum + event.memoryFreedMB, 0),
      avgInvalidationsPerHour: recentEvents.length
    };
  }

  /**
   * Manual cache invalidation
   */
  async invalidateKey(key: string, reason: string = 'manual'): Promise<boolean> {
    const entry = this.cacheEntries.get(key);
    if (!entry) return false;

    const entrySize = this.estimateEntrySize(entry);
    this.cacheEntries.delete(key);

    this.recordInvalidationEvent({
      timestamp: Date.now(),
      type: 'manual',
      key,
      reason,
      memoryFreedMB: entrySize,
      strategy: this.config.strategy
    });

    return true;
  }

  /**
   * Clear all cache entries
   */
  async clearAll(reason: string = 'manual-clear'): Promise<number> {
    const count = this.cacheEntries.size;
    const totalSize = Array.from(this.cacheEntries.values())
      .reduce((sum, entry) => sum + this.estimateEntrySize(entry), 0);

    for (const [key, entry] of this.cacheEntries.entries()) {
      this.recordInvalidationEvent({
        timestamp: Date.now(),
        type: 'manual',
        key,
        reason,
        memoryFreedMB: this.estimateEntrySize(entry),
        strategy: this.config.strategy
      });
    }

    this.cacheEntries.clear();
    this.accessHistory = [];
    this.patterns.clear();

    console.log(`[GPU Cache] Cleared all ${count} entries, ${totalSize.toFixed(2)}MB freed`);
    return count;
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<CacheInvalidationConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Restart timers with new intervals
    this.startAutomaticCleanup();
    this.startMemoryMonitoring();
  }

  /**
   * Get current statistics
   */
  getStats() {
    const memoryMetrics = this.getMemoryPressureMetrics();
    const analytics = this.getAnalytics();

    return {
      config: this.config,
      cacheEntries: this.cacheEntries.size,
      memoryMetrics,
      analytics,
      patterns: this.patterns.size,
      accessHistorySize: this.accessHistory.length,
      recentInvalidations: this.invalidationEvents.slice(-10)
    };
  }

  /**
   * Cleanup and dispose
   */
  dispose(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
    
    if (this.memoryMonitorTimer) {
      clearInterval(this.memoryMonitorTimer);
      this.memoryMonitorTimer = null;
    }

    this.cacheEntries.clear();
    this.accessHistory = [];
    this.patterns.clear();
    this.invalidationEvents = [];
  }
}

// Global cache invalidation system instance
export const gpuCacheInvalidationSystem = new GPUCacheInvalidationSystem({
  strategy: 'adaptive',
  maxMemoryMB: 512,
  maxEntries: 1000,
  defaultTTL: 30 * 60 * 1000, // 30 minutes
  memoryPressureThreshold: 0.85,
  cleanupInterval: 60 * 1000, // 1 minute
  predictiveWindow: 5 * 60 * 1000, // 5 minutes
  enableWarmup: true,
  warmupBatchSize: 10
});

// Export types for use in other modules
export type {
  InvalidationStrategy,
  CacheInvalidationConfig,
  CacheEntry,
  MemoryPressureMetrics,
  PredictivePattern,
  InvalidationEvent
};