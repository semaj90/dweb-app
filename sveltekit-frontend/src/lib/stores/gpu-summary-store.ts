/**
 * Unified GPU Summary Consumption Store
 * 
 * Centralizes all GPU summary operations, WebASM inference integration,
 * and MinIO cache coordination with GPU SOM acceleration.
 * 
 * Features:
 * - RTX 3060 Ti CUDA buffer management
 * - WebAssembly inference pipeline integration
 * - NES-GPU memory bridge coordination
 * - RAG MinIO cache with GPU acceleration
 * - Real-time performance metrics and adaptive scaling
 */

import { writable, derived, get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';

// GPU Buffer and Memory Management Types
export interface GPUMemoryStats {
  totalVRAM: number;
  usedVRAM: number;
  availableVRAM: number;
  bufferCount: number;
  activeTextures: number;
  memoryUtilization: number; // 0-100%
  temperatureCelsius: number;
  powerUsageWatts: number;
}

export interface GPUBufferInfo {
  id: string;
  size: number;
  type: 'vector' | 'texture' | 'summary' | 'inference';
  allocated: boolean;
  lastAccessed: number;
  priority: number; // 0-255 (NES-style)
  bankId?: string; // NES memory bank reference
}

export interface WebASMInferenceMetrics {
  modelLoaded: boolean;
  inferenceTime: number;
  accuracy: number;
  throughput: number; // tokens/second
  memoryFootprint: number;
  wasmHeapSize: number;
  jsToWasmCallTime: number;
}

export interface RAGCacheMetrics {
  minioConnected: boolean;
  cacheHitRate: number;
  cacheMissRate: number;
  totalCachedItems: number;
  compressionRatio: number;
  avgRetrievalTime: number;
  gpuAcceleratedOps: number;
}

export interface GPUSummaryState {
  isInitialized: boolean;
  cudaAvailable: boolean;
  deviceName: string;
  computeCapability: string;
  memory: GPUMemoryStats;
  buffers: Map<string, GPUBufferInfo>;
  wasmInference: WebASMInferenceMetrics;
  ragCache: RAGCacheMetrics;
  performanceMetrics: {
    framesPerSecond: number;
    inferenceLatency: number;
    cacheEfficiency: number;
    overallScore: number; // Composite performance score
  };
  errors: string[];
  lastUpdate: number;
}

// Initial state
const initialState: GPUSummaryState = {
  isInitialized: false,
  cudaAvailable: false,
  deviceName: 'Unknown',
  computeCapability: '0.0',
  memory: {
    totalVRAM: 0,
    usedVRAM: 0,
    availableVRAM: 0,
    bufferCount: 0,
    activeTextures: 0,
    memoryUtilization: 0,
    temperatureCelsius: 0,
    powerUsageWatts: 0
  },
  buffers: new Map(),
  wasmInference: {
    modelLoaded: false,
    inferenceTime: 0,
    accuracy: 0,
    throughput: 0,
    memoryFootprint: 0,
    wasmHeapSize: 0,
    jsToWasmCallTime: 0
  },
  ragCache: {
    minioConnected: false,
    cacheHitRate: 0,
    cacheMissRate: 0,
    totalCachedItems: 0,
    compressionRatio: 0,
    avgRetrievalTime: 0,
    gpuAcceleratedOps: 0
  },
  performanceMetrics: {
    framesPerSecond: 0,
    inferenceLatency: 0,
    cacheEfficiency: 0,
    overallScore: 0
  },
  errors: [],
  lastUpdate: Date.now()
};

// Main store
export const gpuSummaryStore: Writable<GPUSummaryState> = writable(initialState);

// Derived stores for specific components
export const gpuMemoryStore: Readable<GPUMemoryStats> = derived(
  gpuSummaryStore,
  ($gpu) => $gpu.memory
);

export const wasmInferenceStore: Readable<WebASMInferenceMetrics> = derived(
  gpuSummaryStore,
  ($gpu) => $gpu.wasmInference
);

export const ragCacheStore: Readable<RAGCacheMetrics> = derived(
  gpuSummaryStore,
  ($gpu) => $gpu.ragCache
);

export const performanceStore: Readable<GPUSummaryState['performanceMetrics']> = derived(
  gpuSummaryStore,
  ($gpu) => $gpu.performanceMetrics
);

// GPU Buffer Service Integration
class GPUSummaryService {
  private updateInterval: number | null = null;
  private bufferServerUrl = 'http://localhost:8095'; // Go GPU buffer server

  async initializeGPU(): Promise<void> {
    try {
      const response = await fetch(`${this.bufferServerUrl}/gpu/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`GPU initialization failed: ${response.statusText}`);
      }

      const gpuInfo = await response.json();
      
      gpuSummaryStore.update(state => ({
        ...state,
        isInitialized: true,
        cudaAvailable: gpuInfo.cudaAvailable,
        deviceName: gpuInfo.deviceName,
        computeCapability: gpuInfo.computeCapability,
        lastUpdate: Date.now()
      }));

      this.startMetricsUpdates();
    } catch (error) {
      gpuSummaryStore.update(state => ({
        ...state,
        errors: [...state.errors, `GPU init error: ${error.message}`],
        lastUpdate: Date.now()
      }));
    }
  }

  async updateMemoryStats(): Promise<void> {
    try {
      const response = await fetch(`${this.bufferServerUrl}/gpu/memory`);
      const memoryStats: GPUMemoryStats = await response.json();

      gpuSummaryStore.update(state => ({
        ...state,
        memory: memoryStats,
        lastUpdate: Date.now()
      }));
    } catch (error) {
      console.error('Failed to update GPU memory stats:', error);
    }
  }

  async allocateBuffer(id: string, size: number, type: GPUBufferInfo['type']): Promise<void> {
    try {
      const response = await fetch(`${this.bufferServerUrl}/gpu/buffer/allocate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, size, type })
      });

      if (!response.ok) {
        throw new Error(`Buffer allocation failed: ${response.statusText}`);
      }

      const bufferInfo: GPUBufferInfo = await response.json();
      
      gpuSummaryStore.update(state => {
        const newBuffers = new Map(state.buffers);
        newBuffers.set(id, bufferInfo);
        return {
          ...state,
          buffers: newBuffers,
          lastUpdate: Date.now()
        };
      });
    } catch (error) {
      gpuSummaryStore.update(state => ({
        ...state,
        errors: [...state.errors, `Buffer allocation error: ${error.message}`],
        lastUpdate: Date.now()
      }));
    }
  }

  async updateWASMInference(metrics: Partial<WebASMInferenceMetrics>): Promise<void> {
    gpuSummaryStore.update(state => ({
      ...state,
      wasmInference: {
        ...state.wasmInference,
        ...metrics
      },
      lastUpdate: Date.now()
    }));
  }

  async updateRAGCache(metrics: Partial<RAGCacheMetrics>): Promise<void> {
    gpuSummaryStore.update(state => ({
      ...state,
      ragCache: {
        ...state.ragCache,
        ...metrics
      },
      lastUpdate: Date.now()
    }));
  }

  private startMetricsUpdates(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }

    this.updateInterval = window.setInterval(async () => {
      await this.updateMemoryStats();
      await this.calculatePerformanceScore();
    }, 1000); // Update every second
  }

  private async calculatePerformanceScore(): Promise<void> {
    const state = get(gpuSummaryStore);
    
    // Calculate composite performance score (0-100)
    const memoryScore = Math.max(0, 100 - state.memory.memoryUtilization);
    const inferenceScore = state.wasmInference.accuracy * 100;
    const cacheScore = state.ragCache.cacheHitRate * 100;
    
    const overallScore = (memoryScore + inferenceScore + cacheScore) / 3;

    gpuSummaryStore.update(prevState => ({
      ...prevState,
      performanceMetrics: {
        ...prevState.performanceMetrics,
        overallScore: Math.round(overallScore)
      },
      lastUpdate: Date.now()
    }));
  }

  destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}

// Singleton service instance
export const gpuSummaryService = new GPUSummaryService();

// WebASM Inference Integration Functions
export async function initializeWASMInference(): Promise<void> {
  try {
    // Initialize WebAssembly inference module
    const wasmModule = await import('$lib/wasm/inference-module');
    await wasmModule.initialize();

    await gpuSummaryService.updateWASMInference({
      modelLoaded: true,
      wasmHeapSize: wasmModule.getHeapSize(),
      lastUpdate: Date.now()
    });
  } catch (error) {
    gpuSummaryStore.update(state => ({
      ...state,
      errors: [...state.errors, `WASM init error: ${error.message}`],
      lastUpdate: Date.now()
    }));
  }
}

// RAG MinIO Cache Integration Functions
export async function initializeRAGCache(): Promise<void> {
  try {
    const minioResponse = await fetch('/api/minio/status');
    const minioConnected = minioResponse.ok;

    await gpuSummaryService.updateRAGCache({
      minioConnected,
      lastUpdate: Date.now()
    });
  } catch (error) {
    gpuSummaryStore.update(state => ({
      ...state,
      errors: [...state.errors, `RAG cache error: ${error.message}`],
      lastUpdate: Date.now()
    }));
  }
}

// Vector Search with GPU Acceleration
export async function performGPUAcceleratedSearch(query: string, options: {
  useWASM?: boolean;
  cacheResults?: boolean;
  gpuBuffer?: string;
} = {}): Promise<any[]> {
  const state = get(gpuSummaryStore);
  
  if (!state.isInitialized || !state.cudaAvailable) {
    throw new Error('GPU not initialized or unavailable');
  }

  const searchStart = performance.now();

  try {
    // Prepare search with GPU acceleration
    const searchPayload = {
      query,
      useGPU: true,
      useWASM: options.useWASM || false,
      cacheResults: options.cacheResults || true,
      bufferIds: options.gpuBuffer ? [options.gpuBuffer] : []
    };

    const response = await fetch('/api/vector-search/gpu-accelerated', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(searchPayload)
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    const results = await response.json();
    const searchTime = performance.now() - searchStart;

    // Update performance metrics
    await gpuSummaryService.updateWASMInference({
      inferenceTime: searchTime,
      throughput: results.length / (searchTime / 1000)
    });

    return results;
  } catch (error) {
    gpuSummaryStore.update(state => ({
      ...state,
      errors: [...state.errors, `GPU search error: ${error.message}`],
      lastUpdate: Date.now()
    }));
    throw error;
  }
}

// Utility functions for component integration
export const gpuStoreHelpers = {
  isGPUReady: (): boolean => {
    const state = get(gpuSummaryStore);
    return state.isInitialized && state.cudaAvailable;
  },

  getMemoryUtilization: (): number => {
    const state = get(gpuSummaryStore);
    return state.memory.memoryUtilization;
  },

  getPerformanceScore: (): number => {
    const state = get(gpuSummaryStore);
    return state.performanceMetrics.overallScore;
  },

  clearErrors: (): void => {
    gpuSummaryStore.update(state => ({
      ...state,
      errors: [],
      lastUpdate: Date.now()
    }));
  }
};

// Auto-initialize on import
if (typeof window !== 'undefined') {
  // Initialize GPU and related services
  gpuSummaryService.initializeGPU();
  initializeWASMInference();
  initializeRAGCache();
}