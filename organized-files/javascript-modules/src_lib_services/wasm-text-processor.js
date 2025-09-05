/**
 * WebAssembly Text Processor for .txt file processing and tokenization
 * Optimized for fast text chunking, tokenization, and autocomplete suggestions
 */

class WasmTextProcessor {
  constructor() {
    this.wasmModule = null;
    this.tokenizer = null;
    this.textChunks = new Map();
    this.ngramIndex = new Map();
    this.autocompleteCache = new Map();
    this.isInitialized = false;
    
    // Configuration
    this.chunkSize = 2048;
    this.maxNgramOrder = 3;
    this.vocabularySize = 50000;
    this.cacheSize = 10000;
  }

  /**
   * Initialize the WebAssembly module and tokenizer
   */
  async initialize() {
    try {
      // Load WebAssembly module for fast text processing
      this.wasmModule = await this.loadWasmModule();
      
      // Initialize tokenizer (using HuggingFace tokenizers or custom BPE)
      await this.initializeTokenizer();
      
      this.isInitialized = true;
      console.log('🎮 WASM Text Processor initialized');
      
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize WASM Text Processor:', error);
      return false;
    }
  }

  /**
   * Load WebAssembly module for text processing
   */
  async loadWasmModule() {
    // In a real implementation, this would load a compiled WASM binary
    // For now, we'll simulate with a JavaScript implementation
    return {
      // Fast text chunking
      chunkText: (text, chunkSize) => {
        const chunks = [];
        for (let i = 0; i < text.length; i += chunkSize) {
          chunks.push({
            content: text.slice(i, i + chunkSize),
            index: Math.floor(i / chunkSize),
            offset: i,
            length: Math.min(chunkSize, text.length - i)
          });
        }
        return chunks;
      },

      // Fast n-gram extraction
      extractNgrams: (tokens, order) => {
        const ngrams = new Map();
        for (let i = 0; i <= tokens.length - order; i++) {
          const ngram = tokens.slice(i, i + order).join('_');
          ngrams.set(ngram, (ngrams.get(ngram) || 0) + 1);
        }
        return ngrams;
      },

      // Fast prefix matching
      findPrefixMatches: (prefix, ngramMap, maxResults = 10) => {
        const matches = [];
        for (const [ngram, count] of ngramMap.entries()) {
          if (ngram.startsWith(prefix)) {
            matches.push({ ngram, count });
          }
          if (matches.length >= maxResults) break;
        }
        return matches.sort((a, b) => b.count - a.count);
      },

      // Memory management
      getMemoryUsage: () => ({
        heapUsed: process.memoryUsage ? process.memoryUsage().heapUsed : 0,
        heapTotal: process.memoryUsage ? process.memoryUsage().heapTotal : 0
      })
    };
  }

  /**
   * Initialize tokenizer with BPE or WordPiece
   */
  async initializeTokenizer() {
    // Simulate tokenizer initialization
    this.tokenizer = {
      encode: (text) => {
        // Simple word-based tokenization for demo
        // In production, use proper BPE/WordPiece tokenization
        const tokens = text.toLowerCase()
          .replace(/[^\w\s]/g, ' ')
          .split(/\s+/)
          .filter(token => token.length > 0);
        
        return {
          tokens,
          ids: tokens.map((token, index) => this.getTokenId(token)),
          attention_mask: new Array(tokens.length).fill(1)
        };
      },

      decode: (ids) => {
        return ids.map(id => this.getTokenFromId(id)).join(' ');
      }
    };
  }

  /**
   * Process a .txt file and extract chunks, tokens, and n-grams
   */
  async processTextFile(file) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      // Read file content
      const text = await this.readFileContent(file);
      
      // Chunk the text using WebAssembly for speed
      const chunks = this.wasmModule.chunkText(text, this.chunkSize);
      
      // Process each chunk
      const processedChunks = await Promise.all(
        chunks.map(chunk => this.processTextChunk(chunk, file.name))
      );

      // Build global n-gram index for autocomplete
      await this.buildNgramIndex(processedChunks);

      // Cache results
      const fileId = this.generateFileId(file);
      this.textChunks.set(fileId, processedChunks);

      return {
        fileId,
        chunks: processedChunks,
        totalChunks: chunks.length,
        totalTokens: processedChunks.reduce((sum, chunk) => sum + chunk.tokens.length, 0),
        vocabularySize: this.ngramIndex.size,
        memoryUsage: this.wasmModule.getMemoryUsage()
      };
    } catch (error) {
      console.error('❌ Error processing text file:', error);
      throw error;
    }
  }

  /**
   * Process a single text chunk
   */
  async processTextChunk(chunk, fileName) {
    // Tokenize the chunk
    const tokenResult = this.tokenizer.encode(chunk.content);
    
    // Extract n-grams for autocomplete
    const ngrams = new Map();
    for (let order = 1; order <= this.maxNgramOrder; order++) {
      const orderNgrams = this.wasmModule.extractNgrams(tokenResult.tokens, order);
      for (const [ngram, count] of orderNgrams) {
        ngrams.set(ngram, count);
      }
    }

    // Generate hash for deduplication
    const hash = await this.generateTextHash(chunk.content);

    return {
      id: `${fileName}_chunk_${chunk.index}`,
      content: chunk.content,
      tokens: tokenResult.tokens,
      tokenIds: tokenResult.ids,
      ngrams,
      hash,
      metadata: {
        fileName,
        chunkIndex: chunk.index,
        offset: chunk.offset,
        length: chunk.length,
        tokenCount: tokenResult.tokens.length,
        ngramCount: ngrams.size,
        timestamp: Date.now()
      }
    };
  }

  /**
   * Build global n-gram index for fast autocomplete
   */
  async buildNgramIndex(processedChunks) {
    this.ngramIndex.clear();

    for (const chunk of processedChunks) {
      for (const [ngram, count] of chunk.ngrams) {
        const currentCount = this.ngramIndex.get(ngram) || 0;
        this.ngramIndex.set(ngram, currentCount + count);
      }
    }

    // Prune low-frequency n-grams to save memory
    const minFrequency = 2;
    for (const [ngram, count] of this.ngramIndex) {
      if (count < minFrequency) {
        this.ngramIndex.delete(ngram);
      }
    }

    console.log(`📊 Built n-gram index with ${this.ngramIndex.size} entries`);
  }

  /**
   * Get autocomplete suggestions for a prefix
   */
  async getAutocompleteSuggestions(prefix, maxResults = 10) {
    if (!this.isInitialized || !prefix.trim()) {
      return [];
    }

    // Check cache first
    const cacheKey = `${prefix}_${maxResults}`;
    if (this.autocompleteCache.has(cacheKey)) {
      return this.autocompleteCache.get(cacheKey);
    }

    try {
      // Normalize prefix
      const normalizedPrefix = prefix.toLowerCase().trim();
      
      // Find prefix matches using WebAssembly
      const matches = this.wasmModule.findPrefixMatches(
        normalizedPrefix, 
        this.ngramIndex, 
        maxResults * 2 // Get more to filter and rank
      );

      // Process and rank suggestions
      const suggestions = matches
        .map(match => ({
          text: match.ngram.replace(/_/g, ' '),
          confidence: this.calculateConfidence(match.count, normalizedPrefix),
          frequency: match.count,
          type: this.getCompletionType(match.ngram)
        }))
        .filter(suggestion => suggestion.confidence > 0.1)
        .slice(0, maxResults);

      // Cache the results
      this.autocompleteCache.set(cacheKey, suggestions);
      
      // Limit cache size
      if (this.autocompleteCache.size > this.cacheSize) {
        const firstKey = this.autocompleteCache.keys().next().value;
        this.autocompleteCache.delete(firstKey);
      }

      return suggestions;
    } catch (error) {
      console.error('❌ Error getting autocomplete suggestions:', error);
      return [];
    }
  }

  /**
   * Get semantic embeddings for text chunks (integrates with GPU processing)
   */
  async getEmbeddingsForChunks(fileId, embeddingDimensions = 768) {
    const chunks = this.textChunks.get(fileId);
    if (!chunks) {
      throw new Error(`File not found: ${fileId}`);
    }

    // Prepare data for GPU processing
    const embeddingRequests = chunks.map(chunk => ({
      documentId: chunk.id,
      content: chunk.content,
      tokens: chunk.tokenIds,
      options: {
        processType: 'full',
        priority: 5,
        timeout: 30000,
        retries: 3,
        batchSize: 1
      }
    }));

    return embeddingRequests;
  }

  /**
   * Convert to bytes for GPU matrix calculations
   */
  convertToBytes(data) {
    if (typeof data === 'string') {
      return new TextEncoder().encode(data);
    } else if (Array.isArray(data)) {
      // Convert array of numbers to Float32Array bytes
      const float32Array = new Float32Array(data);
      return new Uint8Array(float32Array.buffer);
    } else if (data instanceof Object) {
      // Convert object to JSON bytes
      const jsonString = JSON.stringify(data);
      return new TextEncoder().encode(jsonString);
    }
    return new Uint8Array();
  }

  /**
   * Helper methods
   */
  async readFileContent(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = e => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsText(file, 'UTF-8');
    });
  }

  generateFileId(file) {
    return `file_${Date.now()}_${file.name.replace(/[^a-zA-Z0-9]/g, '_')}`;
  }

  async generateTextHash(text) {
    // Simple hash for demo - use crypto.subtle.digest in production
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  getTokenId(token) {
    // Simple hash-based token ID
    let id = 0;
    for (let i = 0; i < token.length; i++) {
      id = ((id << 5) - id) + token.charCodeAt(i);
      id = id & id;
    }
    return Math.abs(id) % this.vocabularySize;
  }

  getTokenFromId(id) {
    // This would be a proper reverse mapping in production
    return `token_${id}`;
  }

  calculateConfidence(frequency, prefix) {
    // Simple confidence calculation based on frequency and prefix length
    const frequencyScore = Math.log(frequency + 1) / Math.log(1000);
    const lengthPenalty = Math.max(0.1, 1 - (prefix.length * 0.1));
    return Math.min(1.0, frequencyScore * lengthPenalty);
  }

  getCompletionType(ngram) {
    const words = ngram.split('_');
    if (words.length === 1) return 'word';
    if (words.length === 2) return 'bigram';
    if (words.length === 3) return 'trigram';
    return 'phrase';
  }

  // Cleanup methods
  clearCache() {
    this.autocompleteCache.clear();
    console.log('🧹 Autocomplete cache cleared');
  }

  getStats() {
    return {
      initialized: this.isInitialized,
      chunksStored: this.textChunks.size,
      ngramIndexSize: this.ngramIndex.size,
      cacheSize: this.autocompleteCache.size,
      memoryUsage: this.wasmModule?.getMemoryUsage() || null
    };
  }
}

// Export singleton instance
export const wasmTextProcessor = new WasmTextProcessor();
export default WasmTextProcessor;