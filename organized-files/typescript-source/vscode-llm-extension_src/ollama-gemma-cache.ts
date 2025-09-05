/**
 * Ollama Gemma Embedding Cache System for VS Code Extension
 * Provides intelligent caching with semantic embeddings for Context7 MCP operations
 */

// @ts-ignore - VSCode types handled at runtime
const vscode = typeof require !== 'undefined' ? require('vscode') : null;
import * as crypto from 'crypto';
import * as fs from 'fs/promises';
import * as path from 'path';

export interface OllamaConfig {
  endpoint: string;
  model: string;
  embeddingModel: string;
  maxCacheSize: number;
  cacheTTL: number; // Time to live in milliseconds
  enablePersistence: boolean;
}

export interface EmbeddingCache {
  key: string;
  text: string;
  embedding: number[];
  timestamp: number;
  metadata: {
    model: string;
    context?: string;
    fileType?: string;
    similarity?: number;
  };
}

export interface CacheQuery {
  text: string;
  context?: string;
  similarityThreshold: number;
  maxResults: number;
}

export interface CacheResponse {
  found: boolean;
  exact?: EmbeddingCache;
  similar: EmbeddingCache[];
  confidence: number;
}

export class OllamaGemmaCacheManager {
  private config: OllamaConfig;
  private cache: Map<string, EmbeddingCache> = new Map();
  private cacheFile: string;
  private isReady = false;

  constructor(config?: Partial<OllamaConfig>) {
    this.config = {
      endpoint: config?.endpoint || 'http://localhost:11434',
      model: config?.model || 'gemma3-legal',
      embeddingModel: config?.embeddingModel || 'nomic-embed-text',
      maxCacheSize: config?.maxCacheSize || 10000,
      cacheTTL: config?.cacheTTL || 24 * 60 * 60 * 1000, // 24 hours
      enablePersistence: config?.enablePersistence ?? true
    };

    // Set up cache file path in extension storage
    const storageUri = vscode?.workspace?.getConfiguration()?.get('mcpContext7.cacheStoragePath') as string 
      || path.join(__dirname, '..', 'cache');
    this.cacheFile = path.join(storageUri, 'ollama-gemma-cache.json');
  }

  /**
   * Initialize the cache manager
   */
  async initialize(): Promise<void> {
    try {
      // Create cache directory if it doesn't exist
      await fs.mkdir(path.dirname(this.cacheFile), { recursive: true });

      // Load existing cache if persistence is enabled
      if (this.config.enablePersistence) {
        await this.loadCache();
      }

      // Test Ollama connection
      await this.testOllamaConnection();

      this.isReady = true;
      console.log('Ollama Gemma cache manager initialized successfully');

    } catch (error) {
      console.error('Failed to initialize Ollama Gemma cache:', error);
      throw error;
    }
  }

  /**
   * Get or create embedding for text with intelligent caching
   */
  async getEmbedding(text: string, context?: string): Promise<number[]> {
    if (!this.isReady) {
      throw new Error('Cache manager not initialized');
    }

    // Generate cache key
    const key = this.generateCacheKey(text, context);

    // Check exact match first
    const cached = this.cache.get(key);
    if (cached && this.isCacheValid(cached)) {
      console.log(`Cache hit for: ${text.substring(0, 50)}...`);
      return cached.embedding;
    }

    // Check for similar embeddings
    const similarCache = await this.findSimilarEmbedding(text, context);
    if (similarCache.found && similarCache.exact) {
      console.log(`Similar cache hit for: ${text.substring(0, 50)}...`);
      return similarCache.exact.embedding;
    }

    // Generate new embedding
    console.log(`Generating new embedding for: ${text.substring(0, 50)}...`);
    const embedding = await this.generateEmbedding(text);

    // Cache the result
    const cacheEntry: EmbeddingCache = {
      key,
      text,
      embedding,
      timestamp: Date.now(),
      metadata: {
        model: this.config.embeddingModel,
        context,
        fileType: this.detectFileType(text)
      }
    };

    await this.setCacheEntry(cacheEntry);
    return embedding;
  }

  /**
   * Query cache for similar embeddings
   */
  async querySimilar(query: CacheQuery): Promise<CacheResponse> {
    if (!this.isReady) {
      throw new Error('Cache manager not initialized');
    }

    // Get embedding for query text
    const queryEmbedding = await this.getEmbedding(query.text, query.context);

    // Find similar entries
    const similar: Array<EmbeddingCache & { similarity: number }> = [];

    for (const entry of this.cache.values()) {
      if (!this.isCacheValid(entry)) continue;

      // Calculate cosine similarity
      const similarity = this.cosineSimilarity(queryEmbedding, entry.embedding);
      
      if (similarity >= query.similarityThreshold) {
        similar.push({
          ...entry,
          similarity
        });
      }
    }

    // Sort by similarity (highest first) and limit results
    similar.sort((a, b) => b.similarity - a.similarity);
    const topResults = similar.slice(0, query.maxResults);

    // Check for exact match (very high similarity)
    const exactMatch = topResults.find(r => r.similarity > 0.95);

    return {
      found: topResults.length > 0,
      exact: exactMatch,
      similar: topResults,
      confidence: topResults.length > 0 ? topResults[0].similarity : 0
    };
  }

  /**
   * Enhanced Context7 integration - cache analysis results
   */
  async cacheContext7Analysis(component: string, context: string, analysis: any): Promise<void> {
    const cacheKey = `context7_${component}_${context}`;
    const text = `Context7 analysis for ${component} in ${context}: ${JSON.stringify(analysis)}`;
    
    await this.getEmbedding(text, `context7_${component}`);
  }

  /**
   * Intelligent pre-caching based on workspace analysis
   */
  async preCacheWorkspace(): Promise<{
    filesProcessed: number;
    embeddingsGenerated: number;
    cacheHits: number;
  }> {
    let filesProcessed = 0;
    let embeddingsGenerated = 0;
    let cacheHits = 0;

    try {
      // Get workspace files
      const workspaceFiles = await vscode.workspace.findFiles(
        '**/*.{ts,js,svelte,md,json}',
        '**/node_modules/**',
        1000 // Limit to 1000 files
      );

      for (const file of workspaceFiles) {
        try {
          const content = await vscode.workspace.fs.readFile(file);
          const text = Buffer.from(content).toString('utf8');

          // Skip very large files
          if (text.length > 50000) continue;

          // Extract meaningful chunks
          const chunks = this.extractMeaningfulChunks(text, file.fsPath);

          for (const chunk of chunks) {
            const key = this.generateCacheKey(chunk.text, chunk.context);
            
            if (this.cache.has(key)) {
              cacheHits++;
            } else {
              await this.getEmbedding(chunk.text, chunk.context);
              embeddingsGenerated++;
            }
          }

          filesProcessed++;

        } catch (error) {
          console.warn(`Failed to process file ${file.fsPath}:`, error);
        }
      }

      // Save cache after pre-caching
      if (this.config.enablePersistence) {
        await this.saveCache();
      }

    } catch (error) {
      console.error('Pre-caching failed:', error);
    }

    return { filesProcessed, embeddingsGenerated, cacheHits };
  }

  /**
   * Extract meaningful chunks from code/documentation
   */
  private extractMeaningfulChunks(text: string, filePath: string): Array<{text: string; context: string}> {
    const chunks: Array<{text: string; context: string}> = [];
    const fileExt = path.extname(filePath);
    const fileName = path.basename(filePath);

    // Split into logical chunks based on file type
    switch (fileExt) {
      case '.ts':
      case '.js':
        chunks.push(...this.extractCodeChunks(text, `typescript_${fileName}`));
        break;
      
      case '.svelte':
        chunks.push(...this.extractSvelteChunks(text, `svelte_${fileName}`));
        break;
      
      case '.md':
        chunks.push(...this.extractMarkdownChunks(text, `markdown_${fileName}`));
        break;
      
      case '.json':
        chunks.push({
          text: text.substring(0, 2000), // Limit JSON size
          context: `json_${fileName}`
        });
        break;
      
      default:
        // Generic text chunking
        chunks.push({
          text: text.substring(0, 1000),
          context: `generic_${fileName}`
        });
    }

    return chunks;
  }

  /**
   * Extract TypeScript/JavaScript code chunks
   */
  private extractCodeChunks(text: string, context: string): Array<{text: string; context: string}> {
    const chunks: Array<{text: string; context: string}> = [];
    
    // Extract functions, classes, interfaces
    const patterns = [
      /(?:export\s+)?(?:async\s+)?function\s+\w+[^{]*{[^}]*}/g,
      /(?:export\s+)?(?:interface|type)\s+\w+[^{]*{[^}]*}/g,
      /(?:export\s+)?class\s+\w+[^{]*{[^}]*}/g,
      /\/\*\*[\s\S]*?\*\//g // JSDoc comments
    ];

    for (const pattern of patterns) {
      const matches = text.match(pattern);
      if (matches) {
        for (const match of matches) {
          if (match.length > 50 && match.length < 2000) {
            chunks.push({
              text: match,
              context: `${context}_code`
            });
          }
        }
      }
    }

    return chunks;
  }

  /**
   * Extract Svelte component chunks
   */
  private extractSvelteChunks(text: string, context: string): Array<{text: string; context: string}> {
    const chunks: Array<{text: string; context: string}> = [];
    
    // Extract script, style, and markup sections
    const scriptMatch = text.match(/<script[^>]*>([\s\S]*?)<\/script>/);
    if (scriptMatch && scriptMatch[1].trim().length > 50) {
      chunks.push({
        text: scriptMatch[1],
        context: `${context}_script`
      });
    }

    const styleMatch = text.match(/<style[^>]*>([\s\S]*?)<\/style>/);
    if (styleMatch && styleMatch[1].trim().length > 50) {
      chunks.push({
        text: styleMatch[1],
        context: `${context}_style`
      });
    }

    // Extract markup (remove script and style)
    const markup = text
      .replace(/<script[^>]*>[\s\S]*?<\/script>/g, '')
      .replace(/<style[^>]*>[\s\S]*?<\/style>/g, '')
      .trim();
    
    if (markup.length > 50) {
      chunks.push({
        text: markup,
        context: `${context}_markup`
      });
    }

    return chunks;
  }

  /**
   * Extract Markdown chunks
   */
  private extractMarkdownChunks(text: string, context: string): Array<{text: string; context: string}> {
    const chunks: Array<{text: string; context: string}> = [];
    
    // Split by headers
    const sections = text.split(/^#+\s+/m);
    
    for (let i = 1; i < sections.length; i++) {
      const section = sections[i].trim();
      if (section.length > 100 && section.length < 3000) {
        chunks.push({
          text: section,
          context: `${context}_section_${i}`
        });
      }
    }

    return chunks;
  }

  /**
   * Generate embedding using Ollama
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch(`${this.config.endpoint}/api/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: this.config.embeddingModel,
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.embedding || [];

    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error;
    }
  }

  /**
   * Test Ollama connection
   */
  private async testOllamaConnection(): Promise<void> {
    try {
      const response = await fetch(`${this.config.endpoint}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });

      if (!response.ok) {
        throw new Error(`Ollama not available: ${response.status}`);
      }

      const data = await response.json();
      const hasEmbeddingModel = data.models?.some((m: any) => m.name === this.config.embeddingModel);
      
      if (!hasEmbeddingModel) {
        console.warn(`Embedding model ${this.config.embeddingModel} not found. Available models:`, 
          data.models?.map((m: any) => m.name));
      }

    } catch (error) {
      throw new Error(`Failed to connect to Ollama at ${this.config.endpoint}: ${error}`);
    }
  }

  /**
   * Calculate cosine similarity between two embeddings
   */
  private cosineSimilarity(a: number[], b: number[]): number {
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
   * Generate cache key
   */
  private generateCacheKey(text: string, context?: string): string {
    const content = `${text}${context || ''}`;
    return crypto.createHash('sha256').update(content).digest('hex');
  }

  /**
   * Check if cache entry is still valid
   */
  private isCacheValid(entry: EmbeddingCache): boolean {
    return Date.now() - entry.timestamp < this.config.cacheTTL;
  }

  /**
   * Detect file type from text content
   */
  private detectFileType(text: string): string {
    if (text.includes('export') && text.includes('function')) return 'typescript';
    if (text.includes('<script') && text.includes('svelte')) return 'svelte';
    if (text.includes('# ') || text.includes('## ')) return 'markdown';
    if (text.trim().startsWith('{') && text.trim().endsWith('}')) return 'json';
    return 'text';
  }

  /**
   * Find similar embedding in cache
   */
  private async findSimilarEmbedding(text: string, context?: string): Promise<CacheResponse> {
    return this.querySimilar({
      text,
      context,
      similarityThreshold: 0.85,
      maxResults: 5
    });
  }

  /**
   * Set cache entry with size management
   */
  private async setCacheEntry(entry: EmbeddingCache): Promise<void> {
    // Remove old entries if cache is full
    if (this.cache.size >= this.config.maxCacheSize) {
      await this.evictOldEntries();
    }

    this.cache.set(entry.key, entry);

    // Auto-save periodically
    if (this.config.enablePersistence && this.cache.size % 100 === 0) {
      await this.saveCache();
    }
  }

  /**
   * Evict old cache entries
   */
  private async evictOldEntries(): Promise<void> {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

    // Remove oldest 20% of entries
    const entriesToRemove = Math.floor(entries.length * 0.2);
    for (let i = 0; i < entriesToRemove; i++) {
      this.cache.delete(entries[i][0]);
    }

    console.log(`Evicted ${entriesToRemove} old cache entries`);
  }

  /**
   * Load cache from persistent storage
   */
  private async loadCache(): Promise<void> {
    try {
      const data = await fs.readFile(this.cacheFile, 'utf8');
      const cacheData = JSON.parse(data);

      for (const entry of cacheData) {
        if (this.isCacheValid(entry)) {
          this.cache.set(entry.key, entry);
        }
      }

      console.log(`Loaded ${this.cache.size} cache entries from storage`);

    } catch (error) {
      console.log('No existing cache file found, starting fresh');
    }
  }

  /**
   * Save cache to persistent storage
   */
  private async saveCache(): Promise<void> {
    try {
      const cacheData = Array.from(this.cache.values())
        .filter(entry => this.isCacheValid(entry));

      await fs.writeFile(this.cacheFile, JSON.stringify(cacheData, null, 2));
      console.log(`Saved ${cacheData.length} cache entries to storage`);

    } catch (error) {
      console.error('Failed to save cache:', error);
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    totalEntries: number;
    validEntries: number;
    totalSize: number;
    hitRate: number;
    modelInfo: {
      model: string;
      embeddingModel: string;
      endpoint: string;
    };
  } {
    const validEntries = Array.from(this.cache.values()).filter(e => this.isCacheValid(e));
    const totalSize = JSON.stringify(Array.from(this.cache.values())).length;

    return {
      totalEntries: this.cache.size,
      validEntries: validEntries.length,
      totalSize,
      hitRate: 0, // Would need hit/miss tracking
      modelInfo: {
        model: this.config.model,
        embeddingModel: this.config.embeddingModel,
        endpoint: this.config.endpoint
      }
    };
  }

  /**
   * Clear cache
   */
  async clearCache(): Promise<void> {
    this.cache.clear();
    
    if (this.config.enablePersistence) {
      try {
        await fs.unlink(this.cacheFile);
      } catch (error) {
        // File might not exist
      }
    }

    console.log('Cache cleared');
  }
}

// Export singleton instance
export const ollamaGemmaCache = new OllamaGemmaCacheManager();