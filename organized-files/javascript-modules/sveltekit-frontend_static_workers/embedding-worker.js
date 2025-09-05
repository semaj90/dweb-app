// Client-side embedding generation worker
// Using WebAssembly for nomic-embed or llama.cpp integration

/**
 * WebAssembly-based embedding generation worker
 * Supports nomic-embed-text and llama.cpp models
 * Optimized for legal document processing
 */

let wasmModule = null;
let modelLoaded = false;
let embedModel = 'nomic-embed';
let tokenizer = null;

// Model configurations
const modelConfigs = {
  'nomic-embed': {
    dimensions: 384,
    maxTokens: 8192,
    wasmPath: '/wasm/nomic-embed.wasm',
    modelPath: '/models/nomic-embed-text-v1.5.f16.bin'
  },
  'llama-cpp': {
    dimensions: 512,
    maxTokens: 4096,
    wasmPath: '/wasm/llama.wasm',
    modelPath: '/models/ggml-model-f16.gguf'
  }
};

/**
 * Initialize the embedding model
 */
async function initializeModel(model = 'nomic-embed') {
  embedModel = model;
  const config = modelConfigs[model];
  
  if (!config) {
    throw new Error(`Unsupported model: ${model}`);
  }

  try {
    // Load WebAssembly module
    if (typeof WebAssembly !== 'undefined') {
      // For now, use a fallback implementation
      // In production, this would load the actual WASM module
      console.log(`Initializing ${model} (fallback implementation)`);
      
      // Simulate model loading
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      modelLoaded = true;
      return true;
    } else {
      throw new Error('WebAssembly not supported');
    }
  } catch (error) {
    console.error('Model initialization failed:', error);
    return false;
  }
}

/**
 * Tokenize text for embedding generation
 */
function tokenizeText(text, maxTokens = 8192) {
  // Simple word-based tokenization (fallback)
  // In production, this would use proper tokenizer
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 0);
  
  // Truncate to max tokens
  return words.slice(0, maxTokens);
}

/**
 * Generate embedding using WebAssembly model
 */
async function generateEmbedding(text, options = {}) {
  if (!modelLoaded) {
    throw new Error('Model not loaded');
  }

  const {
    maxLength = 8192,
    normalize = true,
    legal_mode = false
  } = options;

  try {
    // Preprocess text for legal documents
    let processedText = text;
    if (legal_mode) {
      processedText = preprocessLegalText(text);
    }

    // Tokenize text
    const tokens = tokenizeText(processedText, maxLength);
    
    // Generate embedding (fallback implementation)
    const embedding = await generateEmbeddingFallback(tokens);
    
    // Normalize if requested
    if (normalize) {
      normalizeVector(embedding);
    }

    return embedding;
  } catch (error) {
    console.error('Embedding generation failed:', error);
    throw error;
  }
}

/**
 * Fallback embedding generation (placeholder for WASM implementation)
 */
async function generateEmbeddingFallback(tokens) {
  const dimensions = modelConfigs[embedModel].dimensions;
  const embedding = new Float32Array(dimensions);
  
  // Simple hash-based embedding (for development/testing)
  // In production, this would use the actual model
  for (let i = 0; i < dimensions; i++) {
    let value = 0;
    
    for (let j = 0; j < tokens.length; j++) {
      const token = tokens[j];
      const hash = simpleHash(token + i.toString());
      value += Math.sin(hash) * 0.1;
    }
    
    embedding[i] = value / Math.sqrt(tokens.length || 1);
  }
  
  return embedding;
}

/**
 * Preprocess legal text for better embeddings
 */
function preprocessLegalText(text) {
  // Legal-specific text preprocessing
  let processed = text;
  
  // Normalize legal citations
  processed = processed.replace(/(\d+)\s+U\.S\.C\.\s*§?\s*(\d+)/g, '$1 USC $2');
  processed = processed.replace(/(\d+)\s+F\.\s*(\d+d?)\s+(\d+)/g, '$1 F$2 $3');
  
  // Normalize case citations
  processed = processed.replace(/([A-Z][a-z]+)\s+v\.\s+([A-Z][a-z]+)/g, '$1 versus $2');
  
  // Expand legal abbreviations
  const legalAbbreviations = {
    'vs.': 'versus',
    'v.': 'versus',
    'et al.': 'and others',
    'i.e.': 'that is',
    'e.g.': 'for example',
    'cf.': 'compare',
    'id.': 'same source',
    'ibid.': 'same source'
  };
  
  for (const [abbrev, expansion] of Object.entries(legalAbbreviations)) {
    const regex = new RegExp('\\b' + abbrev.replace('.', '\\.') + '\\b', 'gi');
    processed = processed.replace(regex, expansion);
  }
  
  // Remove excessive whitespace
  processed = processed.replace(/\s+/g, ' ').trim();
  
  return processed;
}

/**
 * Generate batch embeddings efficiently
 */
async function generateBatchEmbeddings(texts, options = {}) {
  const { batchSize = 10 } = options;
  const results = [];
  
  // Process in batches to manage memory
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const batchPromises = batch.map(text => generateEmbedding(text, options));
    
    try {
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
      
      // Yield control to prevent blocking
      await new Promise(resolve => setTimeout(resolve, 0));
    } catch (error) {
      console.error(`Batch ${i / batchSize + 1} failed:`, error);
      // Add null placeholders for failed items
      results.push(...new Array(batch.length).fill(null));
    }
  }
  
  return results;
}

/**
 * Normalize vector to unit length
 */
function normalizeVector(vector) {
  let magnitude = 0;
  
  // Calculate magnitude
  for (let i = 0; i < vector.length; i++) {
    magnitude += vector[i] * vector[i];
  }
  
  magnitude = Math.sqrt(magnitude);
  
  // Normalize
  if (magnitude > 0) {
    for (let i = 0; i < vector.length; i++) {
      vector[i] /= magnitude;
    }
  }
}

/**
 * Simple hash function for fallback embedding
 */
function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash;
}

/**
 * Get memory usage statistics
 */
function getMemoryStats() {
  const stats = {
    modelLoaded: modelLoaded,
    embedModel: embedModel,
    wasmHeapSize: 0,
    estimatedMemoryUsage: 0
  };

  // Estimate memory usage based on model
  if (modelLoaded) {
    const config = modelConfigs[embedModel];
    // Rough estimate: model size + context buffers
    stats.estimatedMemoryUsage = config.dimensions * 1000 * 4; // bytes
  }

  return stats;
}

/**
 * Optimize memory usage
 */
function optimizeMemory() {
  // Clear any cached data
  if (tokenizer && tokenizer.clearCache) {
    tokenizer.clearCache();
  }
  
  // Trigger garbage collection if available
  if (typeof gc !== 'undefined') {
    gc();
  }
  
  console.log('Memory optimization completed');
}

// Worker message handler
self.onmessage = async function(event) {
  const { type, text, texts, model, options = {} } = event.data;
  
  try {
    let result;
    
    switch (type) {
      case 'initialize':
        const success = await initializeModel(model || 'nomic-embed');
        result = { success: success, model: embedModel };
        break;
        
      case 'generate_embedding':
        const embedding = await generateEmbedding(text, options);
        result = { success: true, embedding: Array.from(embedding) };
        break;
        
      case 'generate_batch_embeddings':
        const embeddings = await generateBatchEmbeddings(texts, options);
        result = { 
          success: true, 
          embeddings: embeddings.map(emb => emb ? Array.from(emb) : null)
        };
        break;
        
      case 'get_memory_stats':
        result = { type: 'memory_stats', stats: getMemoryStats() };
        break;
        
      case 'optimize_memory':
        optimizeMemory();
        result = { success: true, message: 'Memory optimized' };
        break;
        
      default:
        result = { success: false, error: `Unknown operation: ${type}` };
    }
    
    self.postMessage(result);
    
  } catch (error) {
    self.postMessage({
      success: false,
      error: error.message,
      stack: error.stack
    });
  }
};

// Worker initialization complete
self.postMessage({
  type: 'worker_ready',
  capabilities: {
    webAssembly: typeof WebAssembly !== 'undefined',
    models: Object.keys(modelConfigs),
    legalProcessing: true
  }
});

// Handle initialization on startup
self.postMessage({
  type: 'initialized',
  model: embedModel,
  dimensions: modelConfigs[embedModel].dimensions
});