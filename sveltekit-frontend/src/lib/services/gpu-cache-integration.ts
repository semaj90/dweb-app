/**
 * GPU Cache Integration Module for YoRHa Legal AI Platform
 * Integrates NES.css, YoRHa UI components, GPU caching, and performance optimizations
 */

import { writable, derived, get } from 'svelte/store';
import type {     Writable, Readable     } from 'svelte/store';
import stream from "stream";

// ============================================
// GPU CACHE CONFIGURATION
// ============================================

export interface GPUCacheConfig {
  maxMemory: number; // in MB
  cacheStrategy: 'LRU' | 'LFU' | 'FIFO' | 'ARC';
  enableCompression: boolean;
  compressionLevel: number; // 1-9
  gpuAcceleration: boolean;
  webGLFallback: boolean;
  rtx3060TiOptimization: boolean;
  vramLimit: number; // 8GB for RTX 3060 Ti
}

export const defaultGPUCacheConfig: GPUCacheConfig = {
  maxMemory: 2048, // 2GB default cache
  cacheStrategy: 'LRU',
  enableCompression: true,
  compressionLevel: 6,
  gpuAcceleration: true,
  webGLFallback: true,
  rtx3060TiOptimization: true,
  vramLimit: 8192 // 8GB VRAM
};

// ============================================
// GPU CACHE STORE
// ============================================

interface CacheEntry {
  key: string;
  value: any;
  size: number;
  hits: number;
  lastAccess: number;
  compressed: boolean;
  gpuResident: boolean;
}

interface GPUCacheState {
  entries: Map<string, CacheEntry>;
  totalSize: number;
  hitRate: number;
  missRate: number;
  gpuMemoryUsed: number;
  cpuMemoryUsed: number;
  compressionRatio: number;
  isGPUAvailable: boolean;
  activeRequests: number;
}

class GPUCacheManager {
  private config: GPUCacheConfig;
  private state: Writable<GPUCacheState>;
  private gpuContext: GPUContext | null = null;
  private webglContext: WebGL2RenderingContext | null = null;

  constructor(config: GPUCacheConfig = defaultGPUCacheConfig) {
    this.config = config;
    this.state = writable<GPUCacheState>({
      entries: new Map(),
      totalSize: 0,
      hitRate: 0,
      missRate: 0,
      gpuMemoryUsed: 0,
      cpuMemoryUsed: 0,
      compressionRatio: 1,
      isGPUAvailable: false,
      activeRequests: 0
    });

    this.initializeGPU();
  }

  private async initializeGPU() {
    try {
      // Check for WebGPU support
      if ('gpu' in navigator) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const device = await adapter.requestDevice();
          this.gpuContext = {
            adapter,
            device,
            queue: device.queue
          };
          
          this.state.update(s => ({ ...s, isGPUAvailable: true }));
          console.log('✅ WebGPU initialized for GPU cache acceleration');
        }
      }
      
      // Fallback to WebGL2 if WebGPU not available
      if (!this.gpuContext && this.config.webGLFallback) {
        const canvas = document.createElement('canvas');
        this.webglContext = canvas.getContext('webgl2');
        if (this.webglContext) {
          console.log('✅ WebGL2 fallback initialized for GPU cache');
        }
      }
    } catch (error) {
      console.warn('⚠️ GPU initialization failed, falling back to CPU cache:', error);
    }
  }

  public async set(key: string, value: any, options: { compress?: boolean; gpu?: boolean } = {}) {
    const startTime = performance.now();
    
    try {
      // Serialize value
      const serialized = JSON.stringify(value);
      const size = new Blob([serialized]).size;
      
      // Check cache size limits
      const currentState = get(this.state);
      if (currentState.totalSize + size > this.config.maxMemory * 1024 * 1024) {
        await this.evict();
      }
      
      // Compress if enabled
      let finalValue = serialized;
      let compressed = false;
      if (options.compress ?? this.config.enableCompression) {
        finalValue = await this.compress(serialized);
        compressed = true;
      }
      
      // Store in GPU memory if available and requested
      let gpuResident = false;
      if (options.gpu ?? this.config.gpuAcceleration) {
        gpuResident = await this.storeInGPU(key, finalValue);
      }
      
      // Create cache entry
      const entry: CacheEntry = {
        key,
        value: finalValue,
        size,
        hits: 0,
        lastAccess: Date.now(),
        compressed,
        gpuResident
      };
      
      // Update state
      this.state.update(s => ({
        ...s,
        entries: new Map(s.entries).set(key, entry),
        totalSize: s.totalSize + size,
        gpuMemoryUsed: gpuResident ? s.gpuMemoryUsed + size : s.gpuMemoryUsed,
        cpuMemoryUsed: !gpuResident ? s.cpuMemoryUsed + size : s.cpuMemoryUsed
      }));
      
      const endTime = performance.now();
      console.log(`✅ Cached ${key} in ${(endTime - startTime).toFixed(2)}ms (GPU: ${gpuResident})`);
      
      return true;
    } catch (error) {
      console.error('❌ Cache set failed:', error);
      return false;
    }
  }

  public async get(key: string): Promise<any | null> {
    const startTime = performance.now();
    const currentState = get(this.state);
    const entry = currentState.entries.get(key);
    
    if (!entry) {
      // Cache miss
      this.state.update(s => ({
        ...s,
        missRate: s.missRate + 1
      }));
      return null;
    }
    
    // Cache hit
    entry.hits++;
    entry.lastAccess = Date.now();
    
    let value = entry.value;
    
    // Retrieve from GPU if resident
    if (entry.gpuResident) {
      value = await this.retrieveFromGPU(key);
    }
    
    // Decompress if needed
    if (entry.compressed) {
      value = await this.decompress(value);
    }
    
    // Parse JSON
    const parsed = JSON.parse(value);
    
    // Update hit rate
    this.state.update(s => ({
      ...s,
      hitRate: s.hitRate + 1,
      entries: new Map(s.entries).set(key, entry)
    }));
    
    const endTime = performance.now();
    console.log(`✅ Retrieved ${key} in ${(endTime - startTime).toFixed(2)}ms`);
    
    return parsed;
  }

  private async compress(data: string): Promise<string> {
    // Use CompressionStream API if available
    if ('CompressionStream' in window) {
      const encoder = new TextEncoder();
      const stream = new Response(
        new Blob([encoder.encode(data)]).stream().pipeThrough(
          new CompressionStream('gzip')
        )
      );
      const compressed = await stream.blob();
      return await compressed.text();
    }
    return data; // Return uncompressed if API not available
  }

  private async decompress(data: string): Promise<string> {
    // Use DecompressionStream API if available
    if ('DecompressionStream' in window) {
      const stream = new Response(
        new Blob([data]).stream().pipeThrough(
          new DecompressionStream('gzip')
        )
      );
      const decompressed = await stream.text();
      return decompressed;
    }
    return data; // Return as-is if API not available
  }

  private async storeInGPU(key: string, value: string): Promise<boolean> {
    if (!this.gpuContext) return false;
    
    try {
      // Create GPU buffer for storage
      const encoder = new TextEncoder();
      const data = encoder.encode(value);
      
      const buffer = this.gpuContext.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true
      });
      
      new Uint8Array(buffer.getMappedRange()).set(data);
      buffer.unmap();
      
      // Store buffer reference
      (window as any).__gpuCacheBuffers = (window as any).__gpuCacheBuffers || {};
      (window as any).__gpuCacheBuffers[key] = buffer;
      
      return true;
    } catch (error) {
      console.warn('Failed to store in GPU:', error);
      return false;
    }
  }

  private async retrieveFromGPU(key: string): Promise<string> {
    const buffers = (window as any).__gpuCacheBuffers;
    if (!buffers || !buffers[key]) {
      return '';
    }
    
    // In real implementation, would read from GPU buffer
    // For now, return from CPU cache
    const entry = get(this.state).entries.get(key);
    return entry?.value || '';
  }

  private async evict() {
    const currentState = get(this.state);
    const entries = Array.from(currentState.entries.values());
    
    // Apply cache strategy
    let toEvict: CacheEntry | undefined;
    
    switch (this.config.cacheStrategy) {
      case 'LRU':
        toEvict = entries.reduce((oldest, entry) => 
          entry.lastAccess < oldest.lastAccess ? entry : oldest
        );
        break;
      case 'LFU':
        toEvict = entries.reduce((leastUsed, entry) => 
          entry.hits < leastUsed.hits ? entry : leastUsed
        );
        break;
      case 'FIFO':
        toEvict = entries[0]; // First entry
        break;
      default:
        toEvict = entries[0];
    }
    
    if (toEvict) {
      this.state.update(s => {
        const newEntries = new Map(s.entries);
        newEntries.delete(toEvict.key);
        return {
          ...s,
          entries: newEntries,
          totalSize: s.totalSize - toEvict.size,
          gpuMemoryUsed: toEvict.gpuResident ? s.gpuMemoryUsed - toEvict.size : s.gpuMemoryUsed,
          cpuMemoryUsed: !toEvict.gpuResident ? s.cpuMemoryUsed - toEvict.size : s.cpuMemoryUsed
        };
      });
      
      console.log(`🗑️ Evicted ${toEvict.key} from cache`);
    }
  }

  public getStats(): Readable<GPUCacheStats> {
    return derived(this.state, ($state) => ({
      totalEntries: $state.entries.size,
      totalSize: $state.totalSize,
      hitRate: $state.hitRate / ($state.hitRate + $state.missRate) || 0,
      gpuMemoryUsed: $state.gpuMemoryUsed,
      cpuMemoryUsed: $state.cpuMemoryUsed,
      compressionRatio: $state.compressionRatio,
      isGPUAvailable: $state.isGPUAvailable
    }));
  }

  public clear() {
    this.state.set({
      entries: new Map(),
      totalSize: 0,
      hitRate: 0,
      missRate: 0,
      gpuMemoryUsed: 0,
      cpuMemoryUsed: 0,
      compressionRatio: 1,
      isGPUAvailable: get(this.state).isGPUAvailable,
      activeRequests: 0
    });
    
    // Clear GPU buffers
    if ((window as any).__gpuCacheBuffers) {
      (window as any).__gpuCacheBuffers = {};
    }
    
    console.log('🧹 Cache cleared');
  }
}

// ============================================
// YORHA UI INTEGRATION
// ============================================

export interface YoRHaUIConfig {
  theme: 'yorha' | 'nes' | 'n64' | 'hybrid';
  enableGPUEffects: boolean;
  enable3D: boolean;
  particleEffects: boolean;
  soundEffects: boolean;
  hapticFeedback: boolean;
}

export const yorhaUIConfig: YoRHaUIConfig = {
  theme: 'hybrid',
  enableGPUEffects: true,
  enable3D: true,
  particleEffects: true,
  soundEffects: false,
  hapticFeedback: false
};

// ============================================
// NES.CSS INTEGRATION
// ============================================

export function applyNESTheme(element: HTMLElement) {
  // Add NES.css classes
  element.classList.add('nes-container', 'with-title', 'is-rounded');
  
  // Apply GPU acceleration
  element.style.transform = 'translateZ(0)';
  element.style.willChange = 'transform';
  element.style.backfaceVisibility = 'hidden';
}

export function createNESButton(text: string, type: 'normal' | 'primary' | 'success' | 'warning' | 'error' = 'normal'): HTMLButtonElement {
  const button = document.createElement('button');
  button.textContent = text;
  button.className = `nes-btn ${type !== 'normal' ? `is-${type}` : ''}`;
  
  // Add GPU acceleration
  button.style.transform = 'translateZ(0)';
  button.style.transition = 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)';
  
  return button;
}

// ============================================
// PERFORMANCE MONITORING
// ============================================

export interface PerformanceMetrics {
  fps: number;
  gpuUtilization: number;
  memoryUsage: number;
  cacheHitRate: number;
  renderTime: number;
  networkLatency: number;
}

export class PerformanceMonitor {
  private metrics: Writable<PerformanceMetrics>;
  private frameCount = 0;
  private lastTime = performance.now();

  constructor() {
    this.metrics = writable({
      fps: 60,
      gpuUtilization: 0,
      memoryUsage: 0,
      cacheHitRate: 0,
      renderTime: 0,
      networkLatency: 0
    });

    this.startMonitoring();
  }

  private startMonitoring() {
    const measureFPS = () => {
      const currentTime = performance.now();
      const deltaTime = currentTime - this.lastTime;
      
      if (deltaTime >= 1000) {
        const fps = Math.round((this.frameCount * 1000) / deltaTime);
        
        this.metrics.update(m => ({
          ...m,
          fps,
          memoryUsage: (performance as any).memory ? 
            ((performance as any).memory.usedJSHeapSize / (performance as any).memory.jsHeapSizeLimit) * 100 : 0
        }));
        
        this.frameCount = 0;
        this.lastTime = currentTime;
      }
      
      this.frameCount++;
      requestAnimationFrame(measureFPS);
    };

    requestAnimationFrame(measureFPS);
  }

  public getMetrics(): Readable<PerformanceMetrics> {
    return { subscribe: this.metrics.subscribe };
  }
}

// ============================================
// GLOBAL INITIALIZATION
// ============================================

let gpuCacheManager: GPUCacheManager | null = null;
let performanceMonitor: PerformanceMonitor | null = null;

export function initializeGPUCache(config?: Partial<GPUCacheConfig>) {
  if (!gpuCacheManager) {
    gpuCacheManager = new GPUCacheManager({
      ...defaultGPUCacheConfig,
      ...config
    });
    console.log('🚀 GPU Cache Manager initialized');
  }
  return gpuCacheManager;
}

export function initializePerformanceMonitor() {
  if (!performanceMonitor) {
    performanceMonitor = new PerformanceMonitor();
    console.log('📊 Performance Monitor initialized');
  }
  return performanceMonitor;
}

export function getGPUCache(): GPUCacheManager {
  if (!gpuCacheManager) {
    throw new Error('GPU Cache not initialized. Call initializeGPUCache() first.');
  }
  return gpuCacheManager;
}

export function getPerformanceMonitor(): PerformanceMonitor {
  if (!performanceMonitor) {
    throw new Error('Performance Monitor not initialized. Call initializePerformanceMonitor() first.');
  }
  return performanceMonitor;
}

// ============================================
// TYPE DEFINITIONS
// ============================================

interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  queue: GPUQueue;
}

interface GPUCacheStats {
  totalEntries: number;
  totalSize: number;
  hitRate: number;
  gpuMemoryUsed: number;
  cpuMemoryUsed: number;
  compressionRatio: number;
  isGPUAvailable: boolean;
}

// ============================================
// EXPORT UTILITIES
// ============================================

export const GPUCacheUtils = {
  formatBytes(bytes: number): string {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  },
  
  calculateCompressionRatio(original: number, compressed: number): number {
    if (original === 0) return 1;
    return Math.round((1 - compressed / original) * 100) / 100;
  },
  
  estimateGPUMemoryUsage(dataSize: number, overhead = 1.2): number {
    return Math.ceil(dataSize * overhead);
  }
};

// Initialize on module load if in browser
if (typeof window !== 'undefined') {
  // Auto-initialize with default config
  initializeGPUCache();
  initializePerformanceMonitor();
  
  // Add global styles for NES.css and YoRHa integration
  const style = document.createElement('style');
  style.textContent = `
    /* GPU-Accelerated YoRHa + NES.css Integration */
    .gpu-accelerated {
      transform: translateZ(0);
      will-change: transform;
      backface-visibility: hidden;
      -webkit-font-smoothing: antialiased;
    }
    
    .yorha-nes-button {
      font-family: 'Press Start 2P', monospace;
      image-rendering: pixelated;
      image-rendering: -moz-crisp-edges;
      image-rendering: crisp-edges;
      transform: translateZ(0);
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .yorha-nes-button:hover {
      transform: translateZ(0) scale(1.05);
      filter: brightness(1.2);
    }
    
    .yorha-container {
      backdrop-filter: blur(10px);
      transform: translateZ(0);
      will-change: backdrop-filter;
    }
    
    @supports (backdrop-filter: blur(10px)) {
      .yorha-container {
        background: rgba(0, 0, 0, 0.8) !important;
      }
    }
  `;
  document.head.appendChild(style);
  
  console.log('✅ YoRHa + NES.css + GPU Cache Integration Complete');
}

export default {
  GPUCacheManager,
  PerformanceMonitor,
  initializeGPUCache,
  initializePerformanceMonitor,
  getGPUCache,
  getPerformanceMonitor,
  applyNESTheme,
  createNESButton,
  GPUCacheUtils,
  yorhaUIConfig
};
