/**
 * WebAssembly Llama Worker
 * High-performance Web Worker for running llama.cpp inference
 * Bridges between SvelteKit frontend and Go/WASM backends
 */

// Worker state
let isInitialized = false;
let isModelLoaded = false;
let wasmModule = null;
let goBackendUrl = 'http://localhost:8081/api/infer';

// Model and inference state
let currentModel = null;
let isGenerating = false;
let stopRequested = false;

// Performance tracking
let tokensGenerated = 0;
let inferenceStartTime = 0;

/**
 * Message handler for worker communication
 */
self.addEventListener('message', async (event) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case 'init':
        await handleInit();
        break;

      case 'load':
        await handleLoadModel(payload);
        break;

      case 'infer':
        await handleInference(payload);
        break;

      case 'infer_stream':
        await handleStreamingInference(payload);
        break;

      case 'stop':
        handleStop();
        break;

      default:
        console.warn('Unknown message type:', type);
    }
  } catch (error) {
    console.error('Worker error:', error);
    postMessage({
      type: 'error',
      payload: { error: error.message }
    });
  }
});

/**
 * Initialize the worker with both WASM and Go backend detection
 */
async function handleInit() {
  try {
    console.log('🔄 Initializing Llama Worker...');

    // Check if WebAssembly is supported
    if (typeof WebAssembly !== 'undefined') {
      console.log('✅ WebAssembly supported, attempting WASM initialization...');
      
      try {
        // Try to load WASM module first (faster, client-side)
        wasmModule = await initializeWasm();
        console.log('✅ WASM module loaded successfully');
      } catch (wasmError) {
        console.log('⚠️ WASM loading failed, falling back to Go backend:', wasmError.message);
        wasmModule = null;
      }
    } else {
      console.log('⚠️ WebAssembly not supported, using Go backend');
    }

    // Test Go backend availability
    try {
      const response = await fetch(goBackendUrl.replace('/infer', '/health'), {
        method: 'GET',
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      if (response.ok) {
        console.log('✅ Go backend available');
      } else {
        console.log('⚠️ Go backend not responding');
      }
    } catch (goError) {
      console.log('⚠️ Go backend connection failed:', goError.message);
    }

    isInitialized = true;
    postMessage({ type: 'initialized' });

  } catch (error) {
    console.error('❌ Worker initialization failed:', error);
    postMessage({
      type: 'error',
      payload: { error: `Initialization failed: ${error.message}` }
    });
  }
}

/**
 * Initialize WebAssembly module
 */
async function initializeWasm() {
  try {
    // Load the compiled llama.cpp WASM module
    const wasmResponse = await fetch('/wasm/llama.wasm');
    if (!wasmResponse.ok) {
      throw new Error('Failed to fetch WASM module');
    }

    const wasmBytes = await wasmResponse.arrayBuffer();
    
    // Create memory for the WASM module (128MB initial, 512MB max)
    const memory = new WebAssembly.Memory({ 
      initial: 2048, // 128MB
      maximum: 8192  // 512MB
    });

    // Import object for WASM module
    const importObject = {
      env: {
        memory: memory,
        abort: (msg, file, line, column) => {
          console.error('WASM abort:', { msg, file, line, column });
        },
        console_log: (ptr) => {
          // Convert pointer to string if needed
          console.log('WASM log:', ptr);
        },
        // Memory management functions
        malloc: (size) => {
          // Simple bump allocator - in production, use a proper allocator
          return memory.buffer.byteLength;
        },
        free: (ptr) => {
          // Memory deallocation
        }
      },
      wasi_snapshot_preview1: {
        // WASI functions if needed
        proc_exit: (code) => {
          console.log('WASM process exit:', code);
        },
        fd_write: (fd, iovs, iovs_len, nwritten) => {
          // Handle file descriptor writes
          return 0;
        }
      }
    };

    // Instantiate the WASM module
    const wasmModule = await WebAssembly.instantiate(wasmBytes, importObject);
    
    // Initialize the WASM runtime
    if (wasmModule.instance.exports.initialize) {
      const initResult = wasmModule.instance.exports.initialize();
      if (initResult !== 0) {
        throw new Error(`WASM initialization failed with code: ${initResult}`);
      }
    }

    return wasmModule;

  } catch (error) {
    console.error('WASM initialization error:', error);
    throw error;
  }
}

/**
 * Load model using the best available method
 */
async function handleLoadModel(payload) {
  const { modelUrl, contextSize = 4096, enableGPU = true } = payload;

  try {
    console.log(`🔄 Loading model: ${modelUrl}`);
    postMessage({ type: 'progress', payload: { progress: 0 } });

    if (wasmModule) {
      // Use WASM for client-side inference
      await loadWasmModel(modelUrl, contextSize, enableGPU);
    } else {
      // Use Go backend
      await loadGoModel(modelUrl);
    }

    isModelLoaded = true;
    postMessage({ type: 'loaded' });
    console.log('✅ Model loaded successfully');

  } catch (error) {
    console.error('❌ Model loading failed:', error);
    postMessage({
      type: 'error',
      payload: { error: `Model loading failed: ${error.message}` }
    });
  }
}

/**
 * Load model via WebAssembly
 */
async function loadWasmModel(modelUrl, contextSize, enableGPU) {
  // Fetch model file
  const modelResponse = await fetch(modelUrl);
  if (!modelResponse.ok) {
    throw new Error(`Failed to fetch model: ${modelResponse.status}`);
  }

  const modelBytes = await modelResponse.arrayBuffer();
  postMessage({ type: 'progress', payload: { progress: 0.5 } });

  // Load model into WASM memory
  if (wasmModule.instance.exports.load_model) {
    const modelPtr = wasmModule.instance.exports.load_model(
      modelBytes.byteLength,
      contextSize,
      enableGPU ? 1 : 0
    );

    if (modelPtr === 0) {
      throw new Error('WASM model loading failed');
    }

    currentModel = { type: 'wasm', ptr: modelPtr };
  } else {
    throw new Error('WASM module does not support model loading');
  }

  postMessage({ type: 'progress', payload: { progress: 1.0 } });
}

/**
 * Load model via Go backend
 */
async function loadGoModel(modelUrl) {
  // Test if Go backend can access the model
  const testResponse = await fetch(`${goBackendUrl}/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelPath: modelUrl })
  });

  if (!testResponse.ok) {
    throw new Error('Go backend model test failed');
  }

  currentModel = { type: 'go', url: modelUrl };
  postMessage({ type: 'progress', payload: { progress: 1.0 } });
}

/**
 * Handle single inference request
 */
async function handleInference(payload) {
  if (!isModelLoaded) {
    throw new Error('No model loaded');
  }

  const { prompt, temperature = 0.7, topK = 40, topP = 0.9, maxTokens = 512 } = payload;
  
  isGenerating = true;
  stopRequested = false;
  tokensGenerated = 0;
  inferenceStartTime = performance.now();

  try {
    let result;

    if (currentModel.type === 'wasm') {
      result = await generateWithWasm(prompt, { temperature, topK, topP, maxTokens });
    } else {
      result = await generateWithGo(prompt, { temperature, maxTokens });
    }

    const inferenceTime = performance.now() - inferenceStartTime;
    const tokensPerSecond = tokensGenerated / (inferenceTime / 1000);

    postMessage({
      type: 'result',
      payload: { result }
    });

    postMessage({
      type: 'complete',
      payload: { 
        totalTokens: tokensGenerated,
        tokensPerSecond: tokensPerSecond,
        inferenceTime: inferenceTime
      }
    });

  } catch (error) {
    console.error('Inference error:', error);
    postMessage({
      type: 'error',
      payload: { error: error.message }
    });
  } finally {
    isGenerating = false;
  }
}

/**
 * Handle streaming inference
 */
async function handleStreamingInference(payload) {
  if (!isModelLoaded) {
    throw new Error('No model loaded');
  }

  const { prompt, temperature = 0.7, maxTokens = 512 } = payload;
  
  isGenerating = true;
  stopRequested = false;
  tokensGenerated = 0;
  inferenceStartTime = performance.now();

  try {
    if (currentModel.type === 'wasm') {
      await streamWithWasm(prompt, { temperature, maxTokens });
    } else {
      await streamWithGo(prompt, { temperature, maxTokens });
    }

    const inferenceTime = performance.now() - inferenceStartTime;
    const tokensPerSecond = tokensGenerated / (inferenceTime / 1000);

    postMessage({
      type: 'complete',
      payload: { 
        totalTokens: tokensGenerated,
        tokensPerSecond: tokensPerSecond
      }
    });

  } catch (error) {
    console.error('Streaming error:', error);
    postMessage({
      type: 'error',
      payload: { error: error.message }
    });
  } finally {
    isGenerating = false;
  }
}

/**
 * Generate text using WASM
 */
async function generateWithWasm(prompt, options) {
  // This would call the actual WASM functions
  // For now, simulate the process
  const chunks = ['This ', 'is ', 'a ', 'simulated ', 'WASM ', 'response.'];
  let result = '';

  for (const chunk of chunks) {
    if (stopRequested) break;
    
    result += chunk;
    tokensGenerated++;
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  return result;
}

/**
 * Stream tokens using WASM
 */
async function streamWithWasm(prompt, options) {
  const chunks = ['Legal ', 'analysis: ', 'This ', 'contract ', 'contains ', 'standard ', 'clauses...'];
  
  for (const chunk of chunks) {
    if (stopRequested) break;
    
    postMessage({
      type: 'token',
      payload: { token: chunk, tokensPerSecond: tokensGenerated / ((performance.now() - inferenceStartTime) / 1000) }
    });
    
    tokensGenerated++;
    await new Promise(resolve => setTimeout(resolve, 150));
  }
}

/**
 * Generate text using Go backend
 */
async function generateWithGo(prompt, options) {
  const response = await fetch(goBackendUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt,
      temperature: options.temperature,
      tokens: options.maxTokens
    })
  });

  if (!response.ok) {
    throw new Error(`Go backend error: ${response.status}`);
  }

  const data = await response.json();
  
  if (data.error) {
    throw new Error(data.error);
  }

  tokensGenerated = data.text.split(' ').length; // Approximate token count
  return data.text;
}

/**
 * Stream tokens using Go backend
 */
async function streamWithGo(prompt, options) {
  // For streaming, we'd need to modify the Go backend to support streaming
  // For now, simulate streaming by breaking up a regular response
  const fullResponse = await generateWithGo(prompt, options);
  const words = fullResponse.split(' ');
  
  for (const word of words) {
    if (stopRequested) break;
    
    postMessage({
      type: 'token',
      payload: { 
        token: word + ' ',
        tokensPerSecond: tokensGenerated / ((performance.now() - inferenceStartTime) / 1000)
      }
    });
    
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

/**
 * Stop generation
 */
function handleStop() {
  stopRequested = true;
  isGenerating = false;
  console.log('🛑 Generation stopped by user');
}

// Error handling
self.addEventListener('error', (error) => {
  console.error('Worker error:', error);
  postMessage({
    type: 'error',
    payload: { error: error.message }
  });
});

self.addEventListener('unhandledrejection', (event) => {
  console.error('Worker unhandled rejection:', event.reason);
  postMessage({
    type: 'error',
    payload: { error: event.reason.message || 'Unhandled promise rejection' }
  });
});

console.log('🚀 Llama Worker initialized and ready');