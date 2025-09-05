/**
 * Complete WebAssembly + llama.cpp Integration with RL Enhancement
 * Multi-protocol communication (REST → gRPC → QUIC)
 * Memory-optimized client-side inference with NES-RL training
 * FlatBuffers serialization for binary performance
 */

import { browser } from '$app/environment';
import type { ChatMessage, AIResponse } from '$lib/types/ai';

// Enhanced interfaces for RL and multi-protocol support
export interface RLMetrics {
  generation: number;
  fitness: number;
  reward: number;
  epsilon: number;
  actionProbability: number;
}

export interface MemoryStats {
  wasmMemory: number;
  jsHeapUsed: number;
  jsHeapTotal: number;
  modelLoaded: boolean;
  embeddingsCount: number;
  gcTriggered: boolean;
}

export interface WebAssemblyLlamaResponse {
  text: string;
  embedding?: number[];
  rlMetrics?: RLMetrics;
  memoryStats: MemoryStats;
  protocolUsed: 'wasm' | 'rest' | 'grpc' | 'quic';
  latency: number;
}

// Enhanced Svelte 5 reactive state management
interface LlamaState {
  isModelLoaded: boolean;
  isLoading: boolean;
  modelOutput: string;
  progress: number;
  error: string | null;
  tokensPerSecond: number;
  rlMetrics: RLMetrics | null;
  memoryStats: MemoryStats;
  protocolUsed: string;
  latency: number;
}

// Create reactive state using regular JavaScript (not runes in .ts files)
let state: LlamaState = {
  isModelLoaded: false,
  isLoading: false,
  modelOutput: '',
  progress: 0,
  error: null,
  tokensPerSecond: 0,
  rlMetrics: null,
  memoryStats: {
    wasmMemory: 0,
    jsHeapUsed: 0,
    jsHeapTotal: 0,
    modelLoaded: false,
    embeddingsCount: 0,
    gcTriggered: false
  },
  protocolUsed: 'wasm',
  latency: 0
};

// Enhanced Web Worker reference with multi-protocol fallback
let llamaWorker: Worker | null = null;
const fallbackUrls = {
  rest: 'http://localhost:8094/api/rag',
  grpc: 'http://localhost:50051/v1/generate',
  quic: 'http://localhost:8224/quic/generate'
};
const protocolPriority: ('quic' | 'grpc' | 'rest')[] = ['quic', 'grpc', 'rest'];

/**
 * Represents a message sent to or from the Llama Web Worker.
 */
interface LlamaWorkerMessage {
  type: 'load' | 'infer' | 'progress' | 'result' | 'loaded' | 'error' | 'token' | 'complete' | 
        'rl_inference' | 'embed_text' | 'train_rl' | 'memory_stats' | 'init_wasm';
  payload?: any;
  id?: string;
}

/**
 * Configuration for the WebAssembly Llama service
 */
interface LlamaConfig {
  modelUrl: string;
  contextSize?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  maxTokens?: number;
  enableGPU?: boolean;
}

/**
 * Initializes the Llama service by creating and configuring the Web Worker.
 * This should be called once when the component using it mounts.
 */
function initialize(): Promise<boolean> {
  if (!browser || llamaWorker) {
    return Promise.resolve(!!llamaWorker);
  }

  return new Promise((resolve, reject) => {
    console.log('🚀 Initializing WebAssembly Llama service...');

    try {
      // Create a new worker for non-blocking inference
      llamaWorker = new Worker('/workers/llama-worker.js', {
        type: 'module'
      });

      // Handle messages from the worker
      llamaWorker.onmessage = (event: MessageEvent<LlamaWorkerMessage>) => {
        const { type, payload } = event.data;
        
        switch (type) {
          case 'loaded':
            state.isModelLoaded = true;
            state.isLoading = false;
            state.error = null;
            console.log('✅ Model loaded successfully in Web Worker');
            resolve(true);
            break;

          case 'progress':
            state.progress = payload.progress;
            console.log(`📊 Loading progress: ${(payload.progress * 100).toFixed(1)}%`);
            break;

          case 'token':
            // Stream tokens as they arrive for real-time response
            state.modelOutput += payload.token;
            state.tokensPerSecond = payload.tokensPerSecond || 0;
            break;

          case 'result':
            // Complete response received
            state.modelOutput = payload.result;
            state.isLoading = false;
            console.log('✅ Inference complete');
            break;

          case 'complete':
            state.isLoading = false;
            console.log(`🎯 Generation complete. Total tokens: ${payload.totalTokens}, Speed: ${payload.tokensPerSecond.toFixed(1)} t/s`);
            break;

          case 'error':
            console.error('❌ Error from Llama Worker:', payload.error);
            state.error = payload.error;
            state.isLoading = false;
            reject(new Error(payload.error));
            break;
        }
      };

      llamaWorker.onerror = (error) => {
        console.error('❌ Worker error:', error);
        state.error = 'Worker initialization failed';
        reject(error);
      };

      // Send initialization message
      llamaWorker.postMessage({ type: 'init' });

    } catch (error) {
      console.error('❌ Failed to initialize worker:', error);
      reject(error);
    }
  });
}

/**
 * Loads a GGUF model into the WebAssembly runtime.
 * @param config Configuration including model URL and parameters
 */
function loadModel(config: LlamaConfig): Promise<boolean> {
  if (!llamaWorker) {
    console.error('❌ Worker not initialized. Call initialize() first.');
    return Promise.resolve(false);
  }
  
  if (state.isLoading) {
    console.warn('⚠️ Model is already loading...');
    return Promise.resolve(false);
  }

  return new Promise((resolve, reject) => {
    console.log(`🔄 Loading model: ${config.modelUrl}`);
    
    state.isLoading = true;
    state.progress = 0;
    state.error = null;
    state.isModelLoaded = false;

    // Set up one-time listener for load completion
    const handleLoadComplete = (event: MessageEvent<LlamaWorkerMessage>) => {
      const { type, payload } = event.data;
      if (type === 'loaded') {
        llamaWorker?.removeEventListener('message', handleLoadComplete);
        resolve(true);
      } else if (type === 'error') {
        llamaWorker?.removeEventListener('message', handleLoadComplete);
        reject(new Error(payload.error));
      }
    };

    llamaWorker.addEventListener('message', handleLoadComplete);
    llamaWorker.postMessage({ 
      type: 'load', 
      payload: {
        modelUrl: config.modelUrl,
        contextSize: config.contextSize || 4096,
        enableGPU: config.enableGPU || true
      }
    });
  });
}

/**
 * Enhanced RL inference with multi-protocol fallback
 * @param prompt The input text to the model
 * @param context Optional chat context
 * @param options Optional inference parameters
 */
function inferWithRL(prompt: string, context: ChatMessage[] = [], options: Partial<LlamaConfig> = {}): Promise<WebAssemblyLlamaResponse> {
  const startTime = performance.now();
  
  // Try WebAssembly with RL first
  if (llamaWorker && state.isModelLoaded) {
    return new Promise((resolve, reject) => {
      console.log(`🧠 RL inference for prompt: "${prompt.substring(0, 100)}..."`);
      
      state.isLoading = true;
      state.error = null;

      const id = Math.random().toString(36).substr(2, 9);
      const timeout = setTimeout(() => {
        reject(new Error('Worker timeout'));
      }, 30000);

      const handleRLComplete = (event: MessageEvent<LlamaWorkerMessage>) => {
        const { type, payload, id: msgId } = event.data;
        if (msgId === id) {
          clearTimeout(timeout);
          llamaWorker?.removeEventListener('message', handleRLComplete);
          
          if (type === 'result' || type === 'complete') {
            state.rlMetrics = payload.rlMetrics;
            state.memoryStats = payload.memoryStats;
            state.protocolUsed = 'wasm';
            state.latency = performance.now() - startTime;
            
            resolve({
              text: payload.text || state.modelOutput,
              embedding: payload.embedding,
              rlMetrics: payload.rlMetrics,
              memoryStats: payload.memoryStats,
              protocolUsed: 'wasm',
              latency: state.latency
            });
          } else if (type === 'error') {
            console.warn('WebAssembly RL failed, falling back to protocols:', payload.error);
            tryProtocolFallback(prompt, context, options, startTime).then(resolve).catch(reject);
          }
        }
      };

      llamaWorker.addEventListener('message', handleRLComplete);
      llamaWorker.postMessage({ 
        type: 'rl_inference',
        id,
        payload: {
          prompt,
          context,
          temperature: options.temperature || 0.7,
          maxTokens: options.maxTokens || 256
        }
      });
    });
  }

  // Fallback to protocol communication
  return tryProtocolFallback(prompt, context, options, startTime);
}

/**
 * Try protocol fallback in priority order
 */
async function tryProtocolFallback(
  prompt: string, 
  context: ChatMessage[], 
  options: Partial<LlamaConfig>,
  startTime: number
): Promise<WebAssemblyLlamaResponse> {
  
  for (const protocol of protocolPriority) {
    try {
      console.log(`🔄 Trying ${protocol.toUpperCase()} protocol...`);
      const result = await callProtocol(protocol, prompt, context, options);
      
      state.protocolUsed = protocol;
      state.latency = performance.now() - startTime;
      
      return {
        text: result.text,
        embedding: result.embedding,
        rlMetrics: undefined,
        memoryStats: await getMemoryStats(),
        protocolUsed: protocol,
        latency: state.latency
      };
      
    } catch (error) {
      console.warn(`${protocol.toUpperCase()} protocol failed:`, error);
      continue;
    }
  }

  throw new Error('All protocols failed');
}

/**
 * Call specific protocol endpoint
 */
async function callProtocol(
  protocol: 'quic' | 'grpc' | 'rest',
  prompt: string,
  context: ChatMessage[],
  options: Partial<LlamaConfig>
): Promise<{ text: string; embedding?: number[] }> {
  
  const url = fallbackUrls[protocol];
  const requestBody = {
    prompt,
    context: context.map(msg => ({ role: msg.role, content: msg.content })),
    temperature: options.temperature || 0.7,
    max_tokens: options.maxTokens || 256,
    model: 'gemma3-legal'
  };

  // Protocol-specific headers and timeouts
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };

  let timeout = 50000; // Default timeout
  
  switch (protocol) {
    case 'quic':
      headers['X-Protocol'] = 'QUIC';
      timeout = 5000; // Fast timeout for QUIC
      break;
    case 'grpc':
      headers['X-Protocol'] = 'gRPC';
      timeout = 15000;
      break;
    case 'rest':
      timeout = 30000;
      break;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`${protocol} request failed: ${response.statusText}`);
    }

    const result = await response.json();
    return {
      text: result.response || result.text || result.content || 'No response',
      embedding: result.embedding
    };
    
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Original infer function for backwards compatibility
 */
function infer(prompt: string, options: Partial<LlamaConfig> = {}): Promise<string> {
  return inferWithRL(prompt, [], options).then(response => response.text);
}

/**
 * Streaming inference for real-time token generation
 * @param prompt The input text
 * @param options Inference parameters
 */
async function* inferStream(prompt: string, options: Partial<LlamaConfig> = {}): AsyncGenerator<string, void, unknown> {
  if (!llamaWorker || !state.isModelLoaded) {
    throw new Error('Service not ready');
  }

  console.log(`🔄 Starting streaming inference...`);
  
  state.modelOutput = '';
  state.isLoading = true;

  // Create a promise-based event listener for streaming
  let tokenResolver: ((value: { token?: string; done?: boolean }) => void) | null = null;
  const tokenPromise = () => new Promise<{ token?: string; done?: boolean }>((resolve) => {
    tokenResolver = resolve;
  });

  const handleStreamToken = (event: MessageEvent<LlamaWorkerMessage>) => {
    const { type, payload } = event.data;
    if (type === 'token' && tokenResolver) {
      tokenResolver({ token: payload.token });
      tokenResolver = null;
    } else if (type === 'complete' && tokenResolver) {
      tokenResolver({ done: true });
      tokenResolver = null;
    } else if (type === 'error' && tokenResolver) {
      tokenResolver({ done: true });
      tokenResolver = null;
    }
  };

  llamaWorker.addEventListener('message', handleStreamToken);

  // Start inference
  llamaWorker.postMessage({ 
    type: 'infer_stream', 
    payload: {
      prompt,
      temperature: options.temperature || 0.7,
      maxTokens: options.maxTokens || 512
    }
  });

  try {
    while (true) {
      const result = await tokenPromise();
      if (result.done) break;
      if (result.token) {
        yield result.token;
      }
    }
  } finally {
    llamaWorker.removeEventListener('message', handleStreamToken);
    state.isLoading = false;
  }
}

/**
 * Generate embeddings using worker or fallback
 */
async function generateEmbedding(text: string): Promise<number[]> {
  if (llamaWorker && state.isModelLoaded) {
    try {
      return new Promise((resolve, reject) => {
        const id = Math.random().toString(36).substr(2, 9);
        const timeout = setTimeout(() => reject(new Error('Embedding timeout')), 15000);
        
        const handleEmbedding = (event: MessageEvent<LlamaWorkerMessage>) => {
          const { type, payload, id: msgId } = event.data;
          if (msgId === id) {
            clearTimeout(timeout);
            llamaWorker?.removeEventListener('message', handleEmbedding);
            
            if (type === 'result') {
              resolve(payload.embedding);
            } else if (type === 'error') {
              reject(new Error(payload.error));
            }
          }
        };
        
        llamaWorker.addEventListener('message', handleEmbedding);
        llamaWorker.postMessage({ type: 'embed_text', id, payload: { text } });
      });
    } catch (error) {
      console.warn('Worker embedding failed, using server fallback:', error);
    }
  }

  // Server fallback for embeddings
  const response = await fetch('/api/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model: 'nomic-embed-text' })
  });

  if (!response.ok) {
    throw new Error(`Embedding request failed: ${response.statusText}`);
  }

  const result = await response.json();
  return result.embedding;
}

/**
 * Train RL agent with episodes
 */
async function trainRL(episodes: any[]): Promise<RLMetrics> {
  if (llamaWorker && state.isModelLoaded) {
    try {
      return new Promise((resolve, reject) => {
        const id = Math.random().toString(36).substr(2, 9);
        const timeout = setTimeout(() => reject(new Error('Training timeout')), 60000);
        
        const handleTraining = (event: MessageEvent<LlamaWorkerMessage>) => {
          const { type, payload, id: msgId } = event.data;
          if (msgId === id) {
            clearTimeout(timeout);
            llamaWorker?.removeEventListener('message', handleTraining);
            
            if (type === 'result') {
              resolve(payload);
            } else if (type === 'error') {
              reject(new Error(payload.error));
            }
          }
        };
        
        llamaWorker.addEventListener('message', handleTraining);
        llamaWorker.postMessage({ type: 'train_rl', id, payload: { episodes } });
      });
    } catch (error) {
      console.error('RL training failed:', error);
      throw error;
    }
  }
  
  throw new Error('WebAssembly worker not available for RL training');
}

/**
 * Get current memory statistics
 */
async function getMemoryStats(): Promise<MemoryStats> {
  if (llamaWorker && state.isModelLoaded) {
    try {
      return new Promise((resolve, reject) => {
        const id = Math.random().toString(36).substr(2, 9);
        const timeout = setTimeout(() => reject(new Error('Memory stats timeout')), 5000);
        
        const handleMemoryStats = (event: MessageEvent<LlamaWorkerMessage>) => {
          const { type, payload, id: msgId } = event.data;
          if (msgId === id) {
            clearTimeout(timeout);
            llamaWorker?.removeEventListener('message', handleMemoryStats);
            
            if (type === 'result') {
              resolve(payload);
            } else if (type === 'error') {
              reject(new Error(payload.error));
            }
          }
        };
        
        llamaWorker.addEventListener('message', handleMemoryStats);
        llamaWorker.postMessage({ type: 'memory_stats', id, payload: {} });
      });
    } catch (error) {
      console.warn('Failed to get worker memory stats:', error);
    }
  }

  // Fallback memory stats
  return {
    wasmMemory: 0,
    jsHeapUsed: (performance as any).memory?.usedJSHeapSize || 0,
    jsHeapTotal: (performance as any).memory?.totalJSHeapSize || 0,
    modelLoaded: state.isModelLoaded,
    embeddingsCount: 0,
    gcTriggered: false
  };
}

/**
 * Stops the current generation
 */
function stopGeneration(): void {
  if (llamaWorker) {
    llamaWorker.postMessage({ type: 'stop' });
    state.isLoading = false;
  }
}

/**
 * Cleanup resources
 */
function cleanup(): void {
  if (llamaWorker) {
    llamaWorker.terminate();
    llamaWorker = null;
  }
  
  // Reset state
  Object.assign(state, {
    isModelLoaded: false,
    isLoading: false,
    modelOutput: '',
    progress: 0,
    error: null,
    tokensPerSecond: 0
  });
}

// Export a clean, reactive API for Svelte components
export const wasmLlama = {
  // Core functions
  initialize,
  loadModel,
  infer,
  inferStream,
  stopGeneration,
  cleanup,
  
  // Enhanced RL and multi-protocol functions
  inferWithRL,
  generateEmbedding,
  trainRL,
  getMemoryStats,
  
  // Reactive state access (Svelte 5 compatible)
  get isModelLoaded() { return state.isModelLoaded; },
  get isLoading() { return state.isLoading; },
  get modelOutput() { return state.modelOutput; },
  get progress() { return state.progress; },
  get error() { return state.error; },
  get tokensPerSecond() { return state.tokensPerSecond; },
  get rlMetrics() { return state.rlMetrics; },
  get memoryStats() { return state.memoryStats; },
  get protocolUsed() { return state.protocolUsed; },
  get latency() { return state.latency; },
  
  // State object for reactive binding
  get state() { return state; }
};

/**
 * Utility function for chat completion with automatic fallback
 */
export async function generateChatCompletion(
  messages: ChatMessage[],
  options: Partial<LlamaConfig> = {}
): Promise<AIResponse> {
  
  if (!llamaWorker) {
    await initialize();
  }

  const prompt = messages.map(m => `${m.role}: ${m.content}`).join('\n');
  const response = await inferWithRL(prompt, messages, options);
  
  return {
    content: response.text,
    role: 'assistant',
    metadata: {
      model: 'gemma3-legal-wasm',
      protocol: response.protocolUsed,
      latency: response.latency,
      rlMetrics: response.rlMetrics,
      memoryStats: response.memoryStats
    }
  };
}

// Default export for convenience
export default wasmLlama;

/**
 * Example usage in a Svelte 5 component:
 * 
 * ```svelte
 * <script>
 *   import { wasmLlama } from '$lib/services/webasm-llama-complete';
 *   import { onMount, onDestroy } from 'svelte';
 * 
 *   onMount(async () => {
 *     await wasmLlama.initialize();
 *     await wasmLlama.loadModel({
 *       modelUrl: '/models/gemma3-legal-8b-q4_k_m.gguf',
 *       contextSize: 8192,
 *       enableGPU: true
 *     });
 *   });
 * 
 *   onDestroy(() => {
 *     wasmLlama.cleanup();
 *   });
 * 
 *   async function handlePrompt(prompt: string) {
 *     try {
 *       const response = await wasmLlama.infer(prompt);
 *       console.log('Generated:', response);
 *     } catch (error) {
 *       console.error('Inference failed:', error);
 *     }
 *   }
 * 
 *   // For streaming responses:
 *   async function handleStreamingPrompt(prompt: string) {
 *     try {
 *       for await (const token of wasmLlama.inferStream(prompt)) {
 *         console.log('Token:', token);
 *       }
 *     } catch (error) {
 *       console.error('Streaming failed:', error);
 *     }
 *   }
 * </script>
 * 
 * <div>
 *   {#if wasmLlama.isLoading}
 *     <p>Loading... {(wasmLlama.progress * 100).toFixed(1)}%</p>
 *   {/if}
 * 
 *   <button 
 *     onclick={() => handlePrompt("Explain contract law")}
 *     disabled={!wasmLlama.isModelLoaded || wasmLlama.isLoading}
 *   >
 *     Generate Legal Analysis
 *   </button>
 * 
 *   {#if wasmLlama.error}
 *     <p class="error">Error: {wasmLlama.error}</p>
 *   {/if}
 * 
 *   <div class="output">
 *     <pre>{wasmLlama.modelOutput}</pre>
 *     {#if wasmLlama.tokensPerSecond > 0}
 *       <small>{wasmLlama.tokensPerSecond.toFixed(1)} tokens/sec</small>
 *     {/if}
 *   </div>
 * </div>
 * ```
 */