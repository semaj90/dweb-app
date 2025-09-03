/**
 * Memory-Optimized LLaMA WebAssembly Service Worker
 * Client-Side RL with NES Architecture + Nomic Embeddings
 */

// Memory management configuration
const MEMORY_CONFIG = {
  maxModelSize: 4 * 1024 * 1024 * 1024, // 4GB max model
  chunkSize: 64 * 1024, // 64KB chunks
  gcThreshold: 0.8, // Trigger GC at 80% memory usage
  embedDimensions: 384, // nomic-embed-text dimensions
  maxContextLength: 4096,
  batchSize: 8 // RL batch processing
};

// Global state management
let wasmModule = null;
let modelLoaded = false;
let memoryBuffer = null;
let rlAgent = null;
let embeddings = [];

// Import FlatBuffers for binary serialization
importScripts('/workers/flatbuffers.js');
importScripts('/workers/nes-rl.js');

// Service Worker message handling
self.addEventListener('message', async (event) => {
  const { type, data, id } = event.data;
  
  try {
    switch (type) {
      case 'INIT_WASM':
        await initializeWasm();
        postMessage({ type: 'WASM_READY', id });
        break;
        
      case 'LOAD_MODEL':
        await loadModel(data.modelUrl, data.config);
        postMessage({ type: 'MODEL_LOADED', id });
        break;
        
      case 'RL_INFERENCE':
        const result = await performRLInference(data.prompt, data.context);
        postMessage({ type: 'RL_RESULT', data: result, id });
        break;
        
      case 'EMBED_TEXT':
        const embedding = await generateEmbedding(data.text);
        postMessage({ type: 'EMBEDDING_RESULT', data: embedding, id });
        break;
        
      case 'TRAIN_RL':
        await trainRLAgent(data.episodes);
        postMessage({ type: 'TRAINING_COMPLETE', id });
        break;
        
      case 'MEMORY_STATS':
        const stats = getMemoryStats();
        postMessage({ type: 'MEMORY_STATS', data: stats, id });
        break;
        
      default:
        console.warn('Unknown message type:', type);
    }
  } catch (error) {
    postMessage({ 
      type: 'ERROR', 
      data: { message: error.message, stack: error.stack },
      id 
    });
  }
});

/**
 * Initialize WebAssembly module with memory optimization
 */
async function initializeWasm() {
  try {
    console.log('🚀 Initializing LLaMA WebAssembly...');
    
    // Load WebAssembly module
    const wasmResponse = await fetch('/wasm/llama.wasm');
    const wasmBytes = await wasmResponse.arrayBuffer();
    
    // Memory-optimized instantiation
    const importObject = {
      env: {
        memory: new WebAssembly.Memory({ 
          initial: 1024, // 64MB initial
          maximum: 65536, // 4GB maximum
          shared: true
        }),
        emscripten_resize_heap: (size) => {
          console.log(`📏 Resizing heap to ${size / 1024 / 1024}MB`);
          return true;
        }
      }
    };
    
    wasmModule = await WebAssembly.instantiate(wasmBytes, importObject);
    memoryBuffer = new Uint8Array(importObject.env.memory.buffer);
    
    // Initialize RL agent
    rlAgent = new NESRLAgent({
      stateSize: MEMORY_CONFIG.embedDimensions,
      actionSize: 256, // Token vocab subset
      learningRate: 0.001,
      gamma: 0.99
    });
    
    console.log('✅ WebAssembly initialized successfully');
    return true;
  } catch (error) {
    console.error('❌ WASM initialization failed:', error);
    throw error;
  }
}

/**
 * Load model with chunked loading for memory efficiency
 */
async function loadModel(modelUrl, config = {}) {
  if (!wasmModule) {
    throw new Error('WASM not initialized');
  }
  
  console.log('📥 Loading model:', modelUrl);
  
  try {
    // Chunked model loading
    const response = await fetch(modelUrl);
    const contentLength = parseInt(response.headers.get('Content-Length'), 10);
    const reader = response.body.getReader();
    
    let loadedBytes = 0;
    let chunks = [];
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      chunks.push(value);
      loadedBytes += value.length;
      
      // Memory management during loading
      if (loadedBytes > MEMORY_CONFIG.maxModelSize * MEMORY_CONFIG.gcThreshold) {
        await triggerGarbageCollection();
      }
      
      // Progress reporting
      const progress = (loadedBytes / contentLength) * 100;
      postMessage({
        type: 'LOADING_PROGRESS',
        data: { progress, loadedBytes, totalBytes: contentLength }
      });
    }
    
    // Combine chunks efficiently
    const modelData = new Uint8Array(loadedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      modelData.set(chunk, offset);
      offset += chunk.length;
    }
    
    // Load into WASM
    const modelPtr = wasmModule.instance.exports.malloc(modelData.length);
    memoryBuffer.set(modelData, modelPtr);
    
    const success = wasmModule.instance.exports.llama_load_model(
      modelPtr,
      modelData.length,
      config.contextLength || MEMORY_CONFIG.maxContextLength
    );
    
    if (success) {
      modelLoaded = true;
      console.log('✅ Model loaded successfully');
    } else {
      throw new Error('Failed to load model in WASM');
    }
    
  } catch (error) {
    console.error('❌ Model loading failed:', error);
    throw error;
  }
}

/**
 * Perform RL-enhanced inference with NES architecture
 */
async function performRLInference(prompt, context = []) {
  if (!modelLoaded || !rlAgent) {
    throw new Error('Model or RL agent not ready');
  }
  
  console.log('🧠 Performing RL inference...');
  
  // Generate embedding for current state
  const stateEmbedding = await generateEmbedding(prompt);
  
  // RL agent action selection
  const action = rlAgent.selectAction(stateEmbedding);
  
  // Prepare FlatBuffers-encoded prompt
  const encodedPrompt = encodeFlatBuffer({
    text: prompt,
    context: context,
    rlAction: action,
    timestamp: Date.now()
  });
  
  // WASM inference call
  const responsePtr = wasmModule.instance.exports.llama_generate(
    encodedPrompt.buffer,
    encodedPrompt.length,
    action.temperature || 0.7,
    action.maxTokens || 256
  );
  
  // Decode response
  const responseLength = wasmModule.instance.exports.get_response_length();
  const responseData = memoryBuffer.slice(responsePtr, responsePtr + responseLength);
  const response = new TextDecoder().decode(responseData);
  
  // RL feedback loop
  const reward = calculateReward(response, prompt);
  rlAgent.updatePolicy(stateEmbedding, action, reward);
  
  // Memory cleanup
  wasmModule.instance.exports.free(responsePtr);
  await optimizeMemoryUsage();
  
  return {
    text: response,
    embedding: stateEmbedding,
    rlMetrics: {
      action: action,
      reward: reward,
      epsilon: rlAgent.epsilon
    },
    memoryUsage: getMemoryStats()
  };
}

/**
 * Generate embeddings using nomic-embed-text (client-side)
 */
async function generateEmbedding(text) {
  // Use either WASM-based embedding or fallback to server
  try {
    if (wasmModule && wasmModule.instance.exports.generate_embedding) {
      // Client-side embedding generation
      const textPtr = wasmModule.instance.exports.malloc(text.length);
      const textBytes = new TextEncoder().encode(text);
      memoryBuffer.set(textBytes, textPtr);
      
      const embeddingPtr = wasmModule.instance.exports.generate_embedding(
        textPtr,
        text.length,
        MEMORY_CONFIG.embedDimensions
      );
      
      const embedding = new Float32Array(
        memoryBuffer.buffer,
        embeddingPtr,
        MEMORY_CONFIG.embedDimensions
      );
      
      wasmModule.instance.exports.free(textPtr);
      wasmModule.instance.exports.free(embeddingPtr);
      
      return Array.from(embedding);
    }
  } catch (error) {
    console.warn('Client-side embedding failed, using server fallback:', error);
  }
  
  // Fallback to server-side nomic-embed-text
  const response = await fetch('/api/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model: 'nomic-embed-text' })
  });
  
  const result = await response.json();
  return result.embedding;
}

/**
 * Train RL agent with experience replay
 */
async function trainRLAgent(episodes) {
  if (!rlAgent) {
    throw new Error('RL agent not initialized');
  }
  
  console.log(`🎯 Training RL agent with ${episodes.length} episodes...`);
  
  // Batch training for memory efficiency
  for (let i = 0; i < episodes.length; i += MEMORY_CONFIG.batchSize) {
    const batch = episodes.slice(i, i + MEMORY_CONFIG.batchSize);
    
    await rlAgent.trainBatch(batch);
    
    // Memory management during training
    if (i % 100 === 0) {
      await optimizeMemoryUsage();
      postMessage({
        type: 'TRAINING_PROGRESS',
        data: {
          progress: (i / episodes.length) * 100,
          episode: i,
          totalEpisodes: episodes.length
        }
      });
    }
  }
  
  console.log('✅ RL training completed');
}

/**
 * FlatBuffers encoding for binary performance
 */
function encodeFlatBuffer(data) {
  const builder = new flatbuffers.Builder(1024);
  
  // Create string offsets
  const textOffset = builder.createString(data.text);
  const contextOffset = builder.createString(JSON.stringify(data.context));
  
  // Build FlatBuffer
  builder.startObject(4);
  builder.addFieldOffset(0, textOffset, 0);
  builder.addFieldOffset(1, contextOffset, 0);
  builder.addFieldFloat32(2, data.rlAction.temperature || 0.7, 0);
  builder.addFieldInt64(3, BigInt(data.timestamp), 0n);
  const offset = builder.endObject();
  
  builder.finish(offset);
  return builder.asUint8Array();
}

/**
 * Calculate reward for RL training
 */
function calculateReward(response, prompt) {
  // Simple reward function (can be enhanced)
  let reward = 0;
  
  // Length reward
  if (response.length > 10 && response.length < 500) {
    reward += 0.1;
  }
  
  // Quality heuristics
  if (!response.includes('sorry') && !response.includes('cannot')) {
    reward += 0.2;
  }
  
  // Relevance (basic keyword matching)
  const promptWords = prompt.toLowerCase().split(' ');
  const responseWords = response.toLowerCase().split(' ');
  const overlap = promptWords.filter(word => responseWords.includes(word));
  reward += (overlap.length / promptWords.length) * 0.3;
  
  return Math.max(0, Math.min(1, reward));
}

/**
 * Memory optimization and garbage collection
 */
async function optimizeMemoryUsage() {
  // Force garbage collection
  if (self.gc) {
    self.gc();
  }
  
  // Clear old embeddings
  if (embeddings.length > 1000) {
    embeddings = embeddings.slice(-500);
  }
  
  // WASM memory compaction
  if (wasmModule?.instance.exports.compact_memory) {
    wasmModule.instance.exports.compact_memory();
  }
}

/**
 * Get memory usage statistics
 */
function getMemoryStats() {
  const stats = {
    wasmMemory: wasmModule ? wasmModule.instance.exports.memory.buffer.byteLength : 0,
    jsHeapUsed: 0,
    jsHeapTotal: 0,
    modelLoaded: modelLoaded,
    embeddingsCount: embeddings.length
  };
  
  // Performance memory API if available
  if (performance.memory) {
    stats.jsHeapUsed = performance.memory.usedJSHeapSize;
    stats.jsHeapTotal = performance.memory.totalJSHeapSize;
  }
  
  return stats;
}

/**
 * Trigger garbage collection when memory usage is high
 */
async function triggerGarbageCollection() {
  await optimizeMemoryUsage();
  
  // Wait for GC to complete
  await new Promise(resolve => setTimeout(resolve, 100));
}

console.log('🔧 LLaMA RL Service Worker initialized');