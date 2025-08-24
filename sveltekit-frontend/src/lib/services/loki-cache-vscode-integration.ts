/**
 * Loki.js Caching Layer with VS Code Task Integration
 * High-performance in-memory database with VS Code automation
 */

import Loki from 'lokijs';
import type { Collection } from 'lokijs';

export interface CacheableItem {
  id: string;
  type: 'document' | 'search' | 'embedding' | 'analysis' | 'task' | 'config';
  key: string;
  data: any;
  metadata: {
    created: number;
    accessed: number;
    hits: number;
    size: number;
    ttl?: number;
    tags?: string[];
  };
  expiry?: number;
}

export interface VSCodeTask {
  label: string;
  type: string;
  command: string;
  args?: string[];
  group?: string;
  presentation?: {
    echo?: boolean;
    reveal?: string;
    focus?: boolean;
    panel?: string;
    showReuseMessage?: boolean;
    clear?: boolean;
  };
  problemMatcher?: string[];
  dependsOn?: string[];
  runOptions?: {
    runOn?: string;
  };
}

export interface CacheStats {
  totalItems: number;
  hitRate: number;
  memoryUsage: number;
  collections: Record<string, number>;
  recentActivity: Array<{
    operation: 'get' | 'set' | 'delete' | 'expire';
    key: string;
    timestamp: number;
  }>;
}

/**
 * Advanced Loki.js caching service with VS Code task automation
 */
export class LokiCacheVSCodeIntegration {
  private db: loki | null = null;
  private collections: Map<string, Collection<CacheableItem>> = new Map();
  private isInitialized = false;
  private dbName = 'legal-ai-cache.db';
  private activity: Array<{ operation: string; key: string; timestamp: number }> = [];
  private hitCount = 0;
  private missCount = 0;

  // VS Code task configuration
  private vsCodeTasksPath = '.vscode/tasks.json';
  private taskTemplates: Record<string, VSCodeTask> = {
    'ai-process': {
      label: 'AI: Process Document',
      type: 'shell',
      command: 'npm',
      args: ['run', 'ai:process', '${input:documentPath}'],
      group: 'build',
      presentation: {
        echo: true,
        reveal: 'always',
        focus: false,
        panel: 'shared'
      },
      problemMatcher: ['$tsc']
    },
    'vector-search': {
      label: 'Vector: Semantic Search',
      type: 'shell', 
      command: 'npm',
      args: ['run', 'vector:search', '${input:searchQuery}'],
      group: 'test',
      presentation: {
        echo: true,
        reveal: 'silent',
        focus: false
      }
    },
    'neo4j-sync': {
      label: 'Neo4j: Sync Graph Data',
      type: 'shell',
      command: 'npm',
      args: ['run', 'neo4j:sync'],
      group: 'build',
      dependsOn: ['ai-process']
    },
    'cache-clear': {
      label: 'Cache: Clear All Data',
      type: 'shell',
      command: 'npm',
      args: ['run', 'cache:clear'],
      group: 'build',
      presentation: {
        clear: true,
        reveal: 'always'
      }
    },
    'gpu-status': {
      label: 'GPU: Check Status',
      type: 'shell',
      command: 'npm',
      args: ['run', 'gpu:status'],
      group: 'test',
      presentation: {
        reveal: 'always',
        panel: 'new'
      }
    }
  };

  constructor() {
    this.initializeDB();
  }

  /**
   * Initialize Loki.js database with collections
   */
  private initializeDB(): void {
    console.log('🗄️ Initializing Loki.js in-memory database...');

    this.db = new Loki(this.dbName, {
      adapter: typeof window !== 'undefined' ? new Loki.LokiMemoryAdapter() : undefined,
      autoload: false,
      autoloadCallback: () => {
        console.log('📊 Loki.js database loaded');
      },
      autosave: true,
      autosaveInterval: 5000
    });

    // Create collections
    this.createCollections();
    this.isInitialized = true;
    
    console.log('✅ Loki.js cache initialized with collections');
  }

  /**
   * Create specialized collections for different data types
   */
  private createCollections(): void {
    const collectionConfigs = [
      {
        name: 'documents',
        indexes: ['type', 'key', 'metadata.created'],
        ttl: 3600000 // 1 hour
      },
      {
        name: 'searches',
        indexes: ['key', 'metadata.accessed'],
        ttl: 1800000 // 30 minutes
      },
      {
        name: 'embeddings',
        indexes: ['key', 'type'],
        ttl: 7200000 // 2 hours
      },
      {
        name: 'analyses',
        indexes: ['type', 'metadata.created'],
        ttl: 3600000 // 1 hour
      },
      {
        name: 'tasks',
        indexes: ['key', 'type'],
        ttl: null // No expiry for tasks
      },
      {
        name: 'configs',
        indexes: ['key'],
        ttl: null // No expiry for configs
      }
    ];

    for (const config of collectionConfigs) {
      if (!this.db!.getCollection(config.name)) {
        const collection = this.db!.addCollection(config.name, {
          indices: config.indexes,
          ttl: config.ttl,
          ttlInterval: 60000 // Check every minute
        });
        
        this.collections.set(config.name, collection);
        console.log(`📁 Created collection: ${config.name}`);
      }
    }
  }

  /**
   * Store item in cache with automatic collection selection
   */
  async set(
    key: string,
    data: any,
    options: {
      type?: CacheableItem['type'];
      ttl?: number;
      tags?: string[];
      collection?: string;
    } = {}
  ): Promise<void> {
    if (!this.isInitialized) this.initializeDB();

    const itemType = options.type || this.inferType(key, data);
    const collectionName = options.collection || this.getCollectionForType(itemType);
    const collection = this.collections.get(collectionName);

    if (!collection) {
      throw new Error(`Collection not found: ${collectionName}`);
    }

    const now = Date.now();
    const item: CacheableItem = {
      id: crypto.randomUUID(),
      type: itemType,
      key,
      data,
      metadata: {
        created: now,
        accessed: now,
        hits: 0,
        size: this.calculateSize(data),
        ttl: options.ttl,
        tags: options.tags || []
      },
      expiry: options.ttl ? now + options.ttl : undefined
    };

    // Remove existing item with same key
    const existing = collection.findOne({ key });
    if (existing) {
      collection.remove(existing);
    }

    // Insert new item
    collection.insert(item);
    
    this.recordActivity('set', key);
    console.log(`💾 Cached ${itemType}: ${key} (${item.metadata.size} bytes)`);

    // Auto-trigger VS Code task for certain cache operations
    if (itemType === 'analysis' || itemType === 'document') {
      await this.triggerVSCodeTask('cache-sync', { key, type: itemType });
    }
  }

  /**
   * Retrieve item from cache
   */
  async get<T = any>(key: string, type?: CacheableItem['type']): Promise<T | null> {
    if (!this.isInitialized) return null;

    const collectionName = type ? this.getCollectionForType(type) : null;
    
    // Search in specific collection or all collections
    const collectionsToSearch = collectionName ? 
      [this.collections.get(collectionName)].filter(Boolean) : 
      Array.from(this.collections.values());

    for (const collection of collectionsToSearch) {
      const item = collection!.findOne({ key });
      
      if (item) {
        // Check expiry
        if (item.expiry && Date.now() > item.expiry) {
          collection!.remove(item);
          this.recordActivity('expire', key);
          continue;
        }

        // Update access metadata
        item.metadata.accessed = Date.now();
        item.metadata.hits++;
        collection!.update(item);

        this.hitCount++;
        this.recordActivity('get', key);
        
        console.log(`🎯 Cache hit: ${key} (${item.metadata.hits} hits)`);
        return item.data as T;
      }
    }

    this.missCount++;
    console.log(`❌ Cache miss: ${key}`);
    return null;
  }

  /**
   * Delete item from cache
   */
  async delete(key: string, type?: CacheableItem['type']): Promise<boolean> {
    if (!this.isInitialized) return false;

    const collectionName = type ? this.getCollectionForType(type) : null;
    const collectionsToSearch = collectionName ? 
      [this.collections.get(collectionName)].filter(Boolean) : 
      Array.from(this.collections.values());

    for (const collection of collectionsToSearch) {
      const item = collection!.findOne({ key });
      if (item) {
        collection!.remove(item);
        this.recordActivity('delete', key);
        console.log(`🗑️ Deleted from cache: ${key}`);
        return true;
      }
    }

    return false;
  }

  /**
   * Cache search results with intelligent key generation
   */
  async cacheSearchResult(
    query: string,
    results: any[],
    searchType: 'text' | 'vector' | 'graph' | 'hybrid',
    ttl = 1800000 // 30 minutes
  ): Promise<void> {
    const cacheKey = `search:${searchType}:${this.hashString(query)}`;
    
    await this.set(cacheKey, {
      query,
      results,
      searchType,
      resultCount: results.length,
      cached: new Date().toISOString()
    }, {
      type: 'search',
      ttl,
      tags: ['search', searchType]
    });
  }

  /**
   * Get cached search results
   */
  async getCachedSearch(
    query: string,
    searchType: 'text' | 'vector' | 'graph' | 'hybrid'
  ): Promise<{ results: any[]; cached: string } | null> {
    const cacheKey = `search:${searchType}:${this.hashString(query)}`;
    return this.get(cacheKey, 'search');
  }

  /**
   * Cache document processing results
   */
  async cacheDocumentAnalysis(
    documentId: string,
    analysis: any,
    ttl = 7200000 // 2 hours
  ): Promise<void> {
    const cacheKey = `analysis:${documentId}`;
    
    await this.set(cacheKey, analysis, {
      type: 'analysis',
      ttl,
      tags: ['analysis', 'document']
    });
  }

  /**
   * Cache embeddings for reuse
   */
  async cacheEmbedding(
    text: string,
    embedding: number[],
    model = 'nomic-embed-text',
    ttl = 86400000 // 24 hours
  ): Promise<void> {
    const cacheKey = `embedding:${model}:${this.hashString(text)}`;
    
    await this.set(cacheKey, {
      text: text.slice(0, 200), // Store preview only
      embedding,
      model,
      dimensions: embedding.length
    }, {
      type: 'embedding',
      ttl,
      tags: ['embedding', model]
    });
  }

  /**
   * Get cached embedding
   */
  async getCachedEmbedding(
    text: string,
    model = 'nomic-embed-text'
  ): Promise<number[] | null> {
    const cacheKey = `embedding:${model}:${this.hashString(text)}`;
    const cached = await this.get(cacheKey, 'embedding');
    return cached?.embedding || null;
  }

  /**
   * VS Code task integration - create and manage tasks
   */
  async createVSCodeTask(
    taskId: string,
    taskDefinition: VSCodeTask,
    autoRun = false
  ): Promise<void> {
    console.log(`📋 Creating VS Code task: ${taskDefinition.label}`);

    // Cache task definition
    await this.set(`task:${taskId}`, taskDefinition, {
      type: 'task',
      tags: ['vscode', 'task', taskDefinition.group || 'general']
    });

    // Add to VS Code tasks.json if possible
    await this.updateVSCodeTasksFile(taskId, taskDefinition);

    if (autoRun) {
      await this.triggerVSCodeTask(taskId);
    }
  }

  /**
   * Update VS Code tasks.json file
   */
  private async updateVSCodeTasksFile(taskId: string, task: VSCodeTask): Promise<void> {
    try {
      // This would integrate with VS Code workspace tasks
      // For now, cache the task and log the configuration
      console.log(`📝 VS Code task cached: ${taskId}`);
      
      // Future: Write to .vscode/tasks.json
      const taskConfig = {
        version: "2.0.0",
        tasks: [task]
      };

      // Cache task configuration for VS Code integration
      await this.set(`vscode:task:${taskId}`, taskConfig, {
        type: 'config',
        tags: ['vscode', 'tasks', taskId]
      });

    } catch (error) {
      console.warn('⚠️ VS Code tasks.json update failed:', error);
    }
  }

  /**
   * Trigger VS Code task execution
   */
  async triggerVSCodeTask(
    taskId: string,
    args: Record<string, string> = {}
  ): Promise<{ success: boolean; output?: string; error?: string }> {
    console.log(`🚀 Triggering VS Code task: ${taskId}`);

    try {
      // Get cached task definition
      const taskDef = await this.get(`task:${taskId}`, 'task') as VSCodeTask;
      
      if (!taskDef) {
        throw new Error(`Task not found: ${taskId}`);
      }

      // Execute task (simulation for now)
      const result = await this.executeTask(taskDef, args);
      
      // Cache execution result
      await this.set(`task-result:${taskId}:${Date.now()}`, result, {
        type: 'task',
        ttl: 3600000, // 1 hour
        tags: ['task-result', taskId]
      });

      console.log(`✅ Task executed: ${taskId}`);
      return result;

    } catch (error) {
      console.error(`❌ Task execution failed: ${taskId}`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Execute task (mock implementation for VS Code integration)
   */
  private async executeTask(
    task: VSCodeTask,
    args: Record<string, string>
  ): Promise<{ success: boolean; output?: string; error?: string }> {
    // Mock task execution - in real implementation this would call VS Code API
    console.log(`⚡ Executing: ${task.command} ${task.args?.join(' ') || ''}`);
    
    // Simulate task execution based on type
    switch (task.label) {
      case 'AI: Process Document':
        return {
          success: true,
          output: `Document processed successfully: ${args.documentPath || 'unknown'}`
        };
      
      case 'Vector: Semantic Search':
        return {
          success: true,
          output: `Search completed for: ${args.searchQuery || 'unknown'}`
        };
      
      case 'Neo4j: Sync Graph Data':
        return {
          success: true,
          output: 'Graph data synchronized successfully'
        };
      
      case 'Cache: Clear All Data':
        await this.clearAll();
        return {
          success: true,
          output: 'Cache cleared successfully'
        };
      
      case 'GPU: Check Status':
        return {
          success: true,
          output: 'GPU Status: RTX 3060 Ti - Available, 35 layers configured'
        };
      
      default:
        return {
          success: true,
          output: `Task executed: ${task.label}`
        };
    }
  }

  /**
   * Create predefined VS Code tasks for legal AI workflow
   */
  async setupLegalAITasks(): Promise<void> {
    console.log('📋 Setting up Legal AI VS Code tasks...');

    // Create tasks from templates
    for (const [taskId, template] of Object.entries(this.taskTemplates)) {
      await this.createVSCodeTask(taskId, template);
    }

    // Create custom legal AI tasks
    const customTasks: Record<string, VSCodeTask> = {
      'legal-analysis': {
        label: 'Legal AI: Full Case Analysis',
        type: 'shell',
        command: 'npm',
        args: ['run', 'legal:analyze', '${input:caseId}'],
        group: 'build',
        presentation: {
          echo: true,
          reveal: 'always',
          focus: true,
          panel: 'dedicated'
        },
        dependsOn: ['ai-process', 'vector-search', 'neo4j-sync']
      },
      'evidence-process': {
        label: 'Evidence: Process and Index',
        type: 'shell',
        command: 'npm',
        args: ['run', 'evidence:process', '${input:evidenceFile}'],
        group: 'build',
        presentation: {
          echo: true,
          reveal: 'always'
        }
      },
      'cache-optimize': {
        label: 'Cache: Optimize and Cleanup',
        type: 'shell',
        command: 'npm',
        args: ['run', 'cache:optimize'],
        group: 'build',
        runOptions: {
          runOn: 'folderOpen'
        }
      }
    };

    for (const [taskId, taskDef] of Object.entries(customTasks)) {
      await this.createVSCodeTask(taskId, taskDef);
    }

    console.log('✅ Legal AI VS Code tasks configured');
  }

  /**
   * Intelligent cache with automatic optimization
   */
  async smartCache<T>(
    key: string,
    producer: () => Promise<T>,
    options: {
      type?: CacheableItem['type'];
      ttl?: number;
      tags?: string[];
      forceRefresh?: boolean;
    } = {}
  ): Promise<T> {
    // Check cache first unless force refresh
    if (!options.forceRefresh) {
      const cached = await this.get<T>(key, options.type);
      if (cached !== null) {
        return cached;
      }
    }

    // Execute producer and cache result
    console.log(`🔄 Generating and caching: ${key}`);
    const result = await producer();
    
    await this.set(key, result, options);
    return result;
  }

  /**
   * Cache with VS Code task automation
   */
  async cacheWithTask<T>(
    key: string,
    producer: () => Promise<T>,
    taskId?: string,
    options: {
      type?: CacheableItem['type'];
      ttl?: number;
      triggerTask?: boolean;
    } = {}
  ): Promise<T> {
    // Get or generate data
    const result = await this.smartCache(key, producer, options);

    // Trigger related VS Code task if specified
    if (options.triggerTask && taskId) {
      await this.triggerVSCodeTask(taskId, { cacheKey: key });
    }

    return result;
  }

  /**
   * Batch cache operations with task automation
   */
  async batchCache(
    operations: Array<{
      key: string;
      producer: () => Promise<any>;
      options?: { type?: CacheableItem['type']; ttl?: number };
    }>,
    triggerTaskAfter?: string
  ): Promise<unknown[]> {
    console.log(`📦 Batch caching ${operations.length} items...`);

    const results = await Promise.all(
      operations.map(op => this.smartCache(op.key, op.producer, op.options))
    );

    // Trigger VS Code task after batch completion
    if (triggerTaskAfter) {
      await this.triggerVSCodeTask(triggerTaskAfter, {
        batchSize: operations.length,
        keys: operations.map(op => op.key)
      });
    }

    console.log(`✅ Batch cache complete: ${results.length} items`);
    return results;
  }

  /**
   * Cache invalidation with VS Code task trigger
   */
  async invalidateCache(
    pattern: string | RegExp,
    triggerCleanupTask = true
  ): Promise<number> {
    let deletedCount = 0;

    for (const collection of this.collections.values()) {
      const items = collection.find({});
      
      for (const item of items) {
        const matches = pattern instanceof RegExp ? 
          pattern.test(item.key) : 
          item.key.includes(pattern);
          
        if (matches) {
          collection.remove(item);
          deletedCount++;
          this.recordActivity('delete', item.key);
        }
      }
    }

    if (triggerCleanupTask && deletedCount > 0) {
      await this.triggerVSCodeTask('cache-optimize', {
        deletedCount: deletedCount.toString(),
        pattern: pattern.toString()
      });
    }

    console.log(`🧹 Cache invalidated: ${deletedCount} items matching ${pattern}`);
    return deletedCount;
  }

  /**
   * Get comprehensive cache statistics
   */
  getCacheStats(): CacheStats {
    const stats: CacheStats = {
      totalItems: 0,
      hitRate: this.hitCount / (this.hitCount + this.missCount) || 0,
      memoryUsage: 0,
      collections: {},
      recentActivity: this.activity.slice(-20)
    };

    for (const [name, collection] of this.collections) {
      const items = collection.find({});
      stats.totalItems += items.length;
      stats.collections[name] = items.length;
      
      // Calculate memory usage
      stats.memoryUsage += items.reduce((sum, item) => sum + item.metadata.size, 0);
    }

    return stats;
  }

  /**
   * Optimize cache performance
   */
  async optimizeCache(): Promise<{
    itemsRemoved: number;
    memoryFreed: number;
    optimizations: string[];
  }> {
    console.log('🔧 Optimizing cache performance...');

    const optimizations: string[] = [];
    let itemsRemoved = 0;
    let memoryFreed = 0;

    // Remove expired items
    for (const [name, collection] of this.collections) {
      const expired = collection.find({
        expiry: { $lt: Date.now() }
      });

      for (const item of expired) {
        memoryFreed += item.metadata.size;
        collection.remove(item);
        itemsRemoved++;
      }

      if (expired.length > 0) {
        optimizations.push(`Removed ${expired.length} expired items from ${name}`);
      }
    }

    // Remove least recently used items if memory usage is high
    const stats = this.getCacheStats();
    if (stats.memoryUsage > 100 * 1024 * 1024) { // 100MB threshold
      optimizations.push('Memory threshold exceeded, removing LRU items');
      
      for (const collection of this.collections.values()) {
        const items = collection.find({});
        const sortedByAccess = items.sort((a, b) => a.metadata.accessed - b.metadata.accessed);
        
        // Remove oldest 20%
        const toRemove = sortedByAccess.slice(0, Math.floor(items.length * 0.2));
        for (const item of toRemove) {
          memoryFreed += item.metadata.size;
          collection.remove(item);
          itemsRemoved++;
        }
      }
    }

    // Trigger VS Code optimization task
    await this.triggerVSCodeTask('cache-optimize', {
      itemsRemoved: itemsRemoved.toString(),
      memoryFreed: (memoryFreed / 1024 / 1024).toFixed(2) + 'MB'
    });

    const result = { itemsRemoved, memoryFreed, optimizations };
    console.log(`✅ Cache optimization complete:`, result);
    return result;
  }

  /**
   * Clear all cached data
   */
  async clearAll(): Promise<void> {
    for (const collection of this.collections.values()) {
      collection.clear();
    }

    this.activity = [];
    this.hitCount = 0;
    this.missCount = 0;

    console.log('🧹 All cache data cleared');
  }

  /**
   * Export cache data for backup
   */
  exportCache(): Record<string, CacheableItem[]> {
    const exported: Record<string, CacheableItem[]> = {};
    
    for (const [name, collection] of this.collections) {
      exported[name] = collection.find({});
    }

    return exported;
  }

  /**
   * Import cache data from backup
   */
  async importCache(data: Record<string, CacheableItem[]>): Promise<void> {
    await this.clearAll();

    for (const [collectionName, items] of Object.entries(data)) {
      const collection = this.collections.get(collectionName);
      if (collection) {
        collection.insert(items);
        console.log(`📥 Imported ${items.length} items to ${collectionName}`);
      }
    }
  }

  // Helper methods
  private inferType(key: string, data: any): CacheableItem['type'] {
    if (key.startsWith('search:')) return 'search';
    if (key.startsWith('embedding:')) return 'embedding';
    if (key.startsWith('analysis:')) return 'analysis';
    if (key.startsWith('task:')) return 'task';
    if (key.startsWith('config:')) return 'config';
    return 'document';
  }

  private getCollectionForType(type: CacheableItem['type']): string {
    const mapping: Record<CacheableItem['type'], string> = {
      'document': 'documents',
      'search': 'searches',
      'embedding': 'embeddings',
      'analysis': 'analyses',
      'task': 'tasks',
      'config': 'configs'
    };
    return mapping[type] || 'documents';
  }

  private calculateSize(data: any): number {
    return JSON.stringify(data).length * 2; // Rough byte estimation
  }

  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(16);
  }

  private recordActivity(operation: string, key: string): void {
    this.activity.push({
      operation,
      key,
      timestamp: Date.now()
    });

    // Keep only last 100 activities
    if (this.activity.length > 100) {
      this.activity.shift();
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    
    this.collections.clear();
    this.isInitialized = false;
    
    console.log('🧹 Loki.js cache cleaned up');
  }
}

// Global cache service instance
export const lokiCache = new LokiCacheVSCodeIntegration();

// Auto-setup tasks on initialization
if (typeof window !== 'undefined') {
  lokiCache.setupLegalAITasks().catch(console.warn);
}

/**
 * High-level caching utilities for legal AI workflow
 */
export class LegalAICacheUtils {
  /**
   * Cache legal document analysis with VS Code task automation
   */
  static async cacheDocumentAnalysis(
    documentId: string,
    title: string,
    content: string
  ): Promise<any> {
    return lokiCache.cacheWithTask(
      `legal-analysis:${documentId}`,
      async () => {
        // Import Neo4j summarization service
        const { neo4jSummarization } = await import('./neo4j-transformers-summarization');
        return neo4jSummarization.processDocument(documentId, title, content);
      },
      'legal-analysis',
      {
        type: 'analysis',
        ttl: 7200000, // 2 hours
        triggerTask: true
      }
    );
  }

  /**
   * Cache vector search results with optimization
   */
  static async cacheVectorSearch(
    query: string,
    searchType: 'legal' | 'evidence' | 'case' = 'legal'
  ): Promise<unknown[]> {
    return lokiCache.smartCache(
      `vector-search:${searchType}:${query}`,
      async () => {
        // Import vector services
        const { vectorProxy } = await import('./grpc-quic-vector-proxy');
        const { langChainOllamaService } = await import('./langchain-ollama-llama-integration');
        
        const embedding = await langChainOllamaService.generateEmbedding(query);
        const result = await vectorProxy.search(embedding, {
          query,
          limit: 20,
          threshold: 0.7
        });
        
        return result.success ? result.data : [];
      },
      {
        type: 'search',
        ttl: 1800000, // 30 minutes
        tags: ['vector-search', searchType]
      }
    );
  }

  /**
   * Cache embedding with deduplication
   */
  static async cacheEmbedding(text: string, model = 'nomic-embed-text'): Promise<number[]> {
    return lokiCache.smartCache(
      `embedding:${model}:${text}`,
      async () => {
        const { langChainOllamaService } = await import('./langchain-ollama-llama-integration');
        return langChainOllamaService.generateEmbedding(text);
      },
      {
        type: 'embedding',
        ttl: 86400000, // 24 hours
        tags: ['embedding', model]
      }
    );
  }

  /**
   * Cache graph analysis results
   */
  static async cacheGraphAnalysis(
    query: string,
    documentIds: string[] = []
  ): Promise<any> {
    return lokiCache.cacheWithTask(
      `graph-analysis:${query}:${documentIds.join(',')}`,
      async () => {
        const { neo4jSummarization } = await import('./neo4j-transformers-summarization');
        return neo4jSummarization.generateGraphEnhancedAnalysis(query, documentIds);
      },
      'neo4j-sync',
      {
        type: 'analysis',
        ttl: 3600000, // 1 hour
        triggerTask: true
      }
    );
  }

  /**
   * Auto-optimize cache based on usage patterns
   */
  static async autoOptimize(): Promise<void> {
    const stats = lokiCache.getCacheStats();
    
    // Optimize if hit rate is low or memory usage is high
    if (stats.hitRate < 0.5 || stats.memoryUsage > 50 * 1024 * 1024) {
      await lokiCache.optimizeCache();
    }

    // Trigger VS Code task for periodic optimization
    await lokiCache.triggerVSCodeTask('cache-optimize');
  }
}

export { LegalAICacheUtils };