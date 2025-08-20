/**
 * NES-Style GPU Bridge - Integrates NES caching architecture with GPU acceleration
 * Provides 8-bit efficiency optimizations for modern GPU computing
 */

import type { CanvasState } from '$lib/stores/canvas-states';
import type { MultiDimArray, GPUProcessingStats } from '$lib/workers/gpu-tensor-worker';

// NES-style memory hierarchy mapping to modern GPU
interface NESGPUMemoryHierarchy {
  // NES Equivalents → Modern GPU
  prgRom: Float32Array;      // Global Memory (VRAM - The Library)
  chrRom: Uint8ClampedArray; // L2 Cache (Automatic - The Bookshelf)
  ram: Float32Array;         // Shared Memory (Programmable - The Desk)  
  ppu: Int32Array;           // Registers (Per-thread - In Your Hands)
}

// Bit depth profiles for browser optimization
interface BitDepthProfile {
  standard: number;    // 24-bit RGB (16.7M colors) - 99.9% browser support
  modern: number;      // 30-bit HDR (1.07B colors) - 85% modern browser support
  premium: number;     // 48-bit ProPhoto RGB - <5% professional displays
  target: number;      // 32-bit RGBA (our optimization target)
  compressed: number;  // 16-bit (65k colors) for cache efficiency
  minimal: number;     // 8-bit palette (256 colors) NES-style fallback
}

// Cache optimization table (NES-style)
interface CacheTable {
  alphabet: string;
  numbers: string;
  specialChars: string;
  legalTerms: string[];
  commonPhrases: string[];
  nibbleValues: number[];  // 2-bit encoding (4 values)
  byteValues: number[];    // 8-bit encoding (256 values)
}

export class NESStyleGPUBridge {
  private gpuWorker: Worker | null = null;
  private tensorCache: Map<string, CachedTensor> = new Map();
  private bitDepthDetector: BitDepthDetector;
  private memoryHierarchy: NESGPUMemoryHierarchy;
  private cacheTable: CacheTable;
  private stats: BridgeStats;
  
  constructor() {
    this.initializeGPUWorker();
    this.bitDepthDetector = new BitDepthDetector();
    this.cacheTable = this.initializeCacheTable();
    this.stats = this.initializeStats();
    this.memoryHierarchy = this.initializeMemoryHierarchy();
  }

  private initializeGPUWorker(): void {
    try {
      this.gpuWorker = new Worker(
        new URL('../workers/gpu-tensor-worker.ts', import.meta.url),
        { type: 'module' }
      );
      
      this.gpuWorker.postMessage({ type: 'INITIALIZE' });
      
      this.gpuWorker.onmessage = (e) => {
        const { type, data } = e.data;
        
        if (type === 'INITIALIZED') {
          console.log('🎮 NES-style GPU Bridge initialized:', data);
        } else if (type === 'ERROR') {
          console.error('🚨 GPU Worker error:', e.data.error);
        }
      };
      
    } catch (error) {
      console.warn('⚠️ GPU Worker initialization failed:', error);
    }
  }

  private initializeCacheTable(): CacheTable {
    return {
      alphabet: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',           // 26 chars = 5 bits
      numbers: '0123456789',                            // 10 chars = 4 bits
      specialChars: ' .,!?-()[]{}:;"\'',               // 16 chars = 4 bits
      legalTerms: [
        'plaintiff', 'defendant', 'court', 'evidence', 'witness',
        'contract', 'agreement', 'liability', 'damages', 'breach',
        'jurisdiction', 'statute', 'precedent', 'testimony', 'verdict'
      ],
      commonPhrases: [
        'pursuant to', 'in accordance with', 'it is hereby',
        'subject to', 'notwithstanding', 'whereas', 'therefore'
      ],
      nibbleValues: [0, 1, 2, 3],                      // 2-bit encoding
      byteValues: Array.from({ length: 256 }, (_, i) => i) // 8-bit encoding
    };
  }

  private initializeStats(): BridgeStats {
    return {
      totalConversions: 0,
      cacheHitRate: 0,
      averageCompressionRatio: 0,
      bitDepthOptimizations: 0,
      gpuAccelerations: 0,
      nesStyleCacheHits: 0,
      quantizationSavings: 0
    };
  }

  private initializeMemoryHierarchy(): NESGPUMemoryHierarchy {
    return {
      prgRom: new Float32Array(32768),      // 32KB equivalent - global tensor data
      chrRom: new Uint8ClampedArray(8192),  // 8KB equivalent - pattern tables/sprites
      ram: new Float32Array(2048),          // 2KB equivalent - working memory  
      ppu: new Int32Array(64)               // 64 registers equivalent - control state
    };
  }

  // Main entry point: Convert canvas state to GPU-optimized tensor
  async canvasStateToTensor(state: CanvasState): Promise<MultiDimArray> {
    const startTime = performance.now();
    
    try {
      // 1. Extract canvas objects and properties
      const fabricJSON = state.fabricJSON as any;
      const objects = fabricJSON?.objects || [];
      
      // 2. Determine optimal tensor dimensions for GPU processing
      const tensorShape = this.calculateOptimalShape(objects);
      
      // 3. Convert to NES-optimized format
      const nesOptimizedData = await this.optimizeForNESStyle(objects, tensorShape);
      
      // 4. Apply bit-depth quantization
      const quantizedData = this.applyBitDepthOptimization(nesOptimizedData);
      
      // 5. Create GPU-friendly tensor
      const tensor: MultiDimArray = {
        shape: tensorShape,
        data: quantizedData,
        dimensions: tensorShape.length,
        layout: 'nes_optimized',
        cacheKey: this.generateCacheKey(state.id, tensorShape),
        lodLevel: this.determineLODLevel(objects.length),
        timestamp: Date.now()
      };
      
      // Update statistics
      this.stats.totalConversions++;
      const conversionTime = performance.now() - startTime;
      console.log(`🎮 Canvas→Tensor conversion: ${conversionTime.toFixed(2)}ms`);
      
      return tensor;
      
    } catch (error) {
      console.error('🚨 Canvas state conversion failed:', error);
      throw new Error(`Canvas conversion failed: ${error.message}`);
    }
  }

  // Process canvas state with GPU acceleration and NES-style caching
  async processCanvasStateWithGPU(state: CanvasState): Promise<CanvasState> {
    const cacheKey = `canvas_${state.id}_processed`;
    
    // Check NES-style cache first
    const cached = this.checkNESCache(cacheKey);
    if (cached) {
      this.stats.nesStyleCacheHits++;
      return this.tensorToCanvasState(cached.tensor, state);
    }
    
    try {
      // 1. Convert canvas to tensor
      const tensor = await this.canvasStateToTensor(state);
      
      // 2. Process with GPU worker
      if (this.gpuWorker) {
        const processedTensor = await this.processWithGPUWorker(tensor);
        
        // 3. Cache result using NES-style memory hierarchy
        this.cacheInNESHierarchy(cacheKey, processedTensor);
        
        // 4. Convert back to canvas state
        return this.tensorToCanvasState(processedTensor, state);
      } else {
        throw new Error('GPU worker not available');
      }
      
    } catch (error) {
      console.error('🚨 GPU processing failed:', error);
      
      // Fallback to CPU processing with NES optimization
      return this.processFallbackWithNESOptimization(state);
    }
  }

  // Auto-encoder with NES-style bit depth optimization
  async optimizeCanvasState(
    state: CanvasState, 
    targetBitDepth: number = 24
  ): Promise<CanvasState> {
    const startTime = performance.now();
    
    // 1. Detect current browser bit depth capabilities
    const browserCapabilities = this.bitDepthDetector.detect();
    
    // 2. Choose optimal bit depth (NES-style efficiency)
    const optimalBitDepth = Math.min(targetBitDepth, browserCapabilities.totalBits);
    
    // 3. Convert to tensor for processing
    const tensor = await this.canvasStateToTensor(state);
    
    // 4. Apply bit-depth quantization (NES-style)
    const quantizedTensor = this.quantizeTensorBits(tensor, optimalBitDepth);
    
    // 5. Store in appropriate NES memory hierarchy level
    const memoryLevel = this.selectMemoryLevel(quantizedTensor.data.length);
    this.storeInHierarchy(quantizedTensor, memoryLevel);
    
    // 6. Convert back to optimized canvas state
    const optimizedState = await this.tensorToCanvasState(quantizedTensor, state);
    
    const processingTime = performance.now() - startTime;
    this.stats.bitDepthOptimizations++;
    this.stats.quantizationSavings += this.calculateCompressionRatio(tensor, quantizedTensor);
    
    console.log(`🎮 NES optimization: ${processingTime.toFixed(2)}ms, ${optimalBitDepth}-bit`);
    
    return {
      ...optimizedState,
      id: `${optimizedState.id}_nes_optimized`,
      metadata: {
        ...optimizedState.metadata,
        nesOptimized: true,
        bitDepth: optimalBitDepth,
        compressionRatio: this.calculateCompressionRatio(tensor, quantizedTensor),
        processingTime,
        memoryLevel
      }
    };
  }

  // NES-style memory level selection
  private selectMemoryLevel(dataSize: number): keyof NESGPUMemoryHierarchy {
    if (dataSize <= 64) return 'ppu';        // Small data → Registers
    if (dataSize <= 2048) return 'ram';      // Medium data → Working memory
    if (dataSize <= 8192) return 'chrRom';   // Large data → L2 Cache
    return 'prgRom';                         // Huge data → Global memory
  }

  private storeInHierarchy(tensor: MultiDimArray, level: keyof NESGPUMemoryHierarchy): void {
    const hierarchy = this.memoryHierarchy[level];
    
    if (hierarchy instanceof Float32Array) {
      const length = Math.min(tensor.data.length, hierarchy.length);
      hierarchy.set(tensor.data.subarray(0, length));
    } else if (hierarchy instanceof Uint8ClampedArray) {
      // Convert float32 to uint8 for CHR ROM equivalent
      const length = Math.min(tensor.data.length, hierarchy.length);
      for (let i = 0; i < length; i++) {
        hierarchy[i] = Math.round(tensor.data[i] * 255);
      }
    } else if (hierarchy instanceof Int32Array) {
      const length = Math.min(tensor.data.length, hierarchy.length);
      for (let i = 0; i < length; i++) {
        hierarchy[i] = Math.round(tensor.data[i]);
      }
    }
    
    console.log(`📦 Stored ${tensor.data.length} elements in ${level} (NES hierarchy)`);
  }

  // Advanced bit-depth quantization (inspired by NES PPU)
  private quantizeTensorBits(tensor: MultiDimArray, bitDepth: number): MultiDimArray {
    const levels = Math.pow(2, bitDepth) - 1;
    const quantizedData = new Float32Array(tensor.data.length);
    
    // NES-style color quantization approach
    for (let i = 0; i < tensor.data.length; i++) {
      const value = tensor.data[i];
      
      // Normalize to 0-1 range
      const normalized = Math.max(0, Math.min(1, (value + 1) / 2));
      
      // Apply NES-style palette quantization
      let quantized: number;
      
      if (bitDepth <= 8) {
        // 8-bit or less: Use NES palette approach
        quantized = Math.round(normalized * levels) / levels;
      } else if (bitDepth <= 16) {
        // 16-bit: Use dithering for smoother gradients
        const ditherNoise = (Math.random() - 0.5) / levels;
        quantized = Math.round((normalized + ditherNoise) * levels) / levels;
      } else {
        // 24-bit+: Standard quantization
        quantized = Math.round(normalized * levels) / levels;
      }
      
      // Convert back to original range [-1, 1]
      quantizedData[i] = quantized * 2 - 1;
    }
    
    return {
      ...tensor,
      data: quantizedData,
      layout: `nes_quantized_${bitDepth}bit`
    };
  }

  // Calculate optimal tensor shape for GPU processing
  private calculateOptimalShape(objects: any[]): number[] {
    const maxObjects = Math.min(objects.length, 100);        // Limit for memory efficiency
    const propertiesPerObject = 20;                          // Standard properties
    const embeddingDimension = 16;                           // Compact representation
    const timeDimension = 1;                                 // For 4D processing
    
    // Choose shape based on complexity
    if (objects.length <= 10) {
      return [maxObjects, propertiesPerObject, embeddingDimension]; // 3D
    } else {
      return [timeDimension, maxObjects, propertiesPerObject, embeddingDimension]; // 4D
    }
  }

  // Convert objects to NES-optimized format
  private async optimizeForNESStyle(objects: any[], shape: number[]): Promise<Float32Array> {
    const totalElements = shape.reduce((a, b) => a * b, 1);
    const optimizedData = new Float32Array(totalElements);
    
    let writeIndex = 0;
    
    for (let objIndex = 0; objIndex < shape[shape.length - 3]; objIndex++) {
      const obj = objects[objIndex] || {};
      
      // Extract and quantize object properties (NES-style)
      const properties = [
        this.quantizeCoordinate(obj.left || 0),
        this.quantizeCoordinate(obj.top || 0),
        this.quantizeSize(obj.width || 0),
        this.quantizeSize(obj.height || 0),
        this.quantizeScale(obj.scaleX || 1),
        this.quantizeScale(obj.scaleY || 1),
        this.quantizeAngle(obj.angle || 0),
        this.quantizeOpacity(obj.opacity || 1),
        this.quantizeSkew(obj.skewX || 0),
        this.quantizeSkew(obj.skewY || 0),
        this.colorToNESPalette(obj.fill),
        this.colorToNESPalette(obj.stroke),
        this.quantizeStrokeWidth(obj.strokeWidth || 0),
        obj.visible ? 1 : -1,
        obj.selectable ? 1 : -1,
        obj.evented ? 1 : -1,
        // Additional NES-style properties
        this.objectTypeToNumber(obj.type),
        this.quantizeZIndex(obj.zIndex || 0),
        this.quantizeRotation(obj.rotation || 0),
        this.quantizeShadow(obj.shadow)
      ];
      
      // Fill tensor data with NES-optimized values
      for (let propIndex = 0; propIndex < shape[shape.length - 2]; propIndex++) {
        for (let embedIndex = 0; embedIndex < shape[shape.length - 1]; embedIndex++) {
          if (writeIndex < totalElements) {
            const value = propIndex < properties.length ? properties[propIndex] : 0;
            // Apply NES-style encoding pattern
            const encoded = this.applyNESEncoding(value, embedIndex);
            optimizedData[writeIndex++] = encoded;
          }
        }
      }
    }
    
    return optimizedData;
  }

  // NES-style quantization functions
  private quantizeCoordinate(coord: number): number {
    // Quantize to 8-bit precision (0-255 range like NES)
    return Math.round((coord + 1000) / 2000 * 255) / 255 * 2 - 1;
  }

  private quantizeSize(size: number): number {
    return Math.round(Math.min(size, 512) / 512 * 255) / 255;
  }

  private quantizeScale(scale: number): number {
    return Math.round(Math.min(scale, 4) / 4 * 255) / 255;
  }

  private quantizeAngle(angle: number): number {
    return Math.round(((angle % 360) + 360) % 360 / 360 * 255) / 255;
  }

  private quantizeOpacity(opacity: number): number {
    return Math.round(opacity * 255) / 255;
  }

  private quantizeSkew(skew: number): number {
    return Math.round((skew + 45) / 90 * 255) / 255;
  }

  private colorToNESPalette(color: string | undefined): number {
    if (!color) return 0;
    
    if (typeof color === 'string' && color.startsWith('#')) {
      const hex = color.substring(1);
      const num = parseInt(hex, 16);
      
      // Convert to NES-style palette (64 colors)
      const nesColor = this.mapToNESPalette(num);
      return nesColor / 63; // Normalize to [-1, 1]
    }
    
    return 0;
  }

  private mapToNESPalette(rgbValue: number): number {
    // Simulate NES PPU color mapping (simplified)
    const r = (rgbValue >> 16) & 0xFF;
    const g = (rgbValue >> 8) & 0xFF;
    const b = rgbValue & 0xFF;
    
    // Map to NES color space (64 colors)
    const nesR = Math.round(r / 255 * 3);
    const nesG = Math.round(g / 255 * 3);
    const nesB = Math.round(b / 255 * 3);
    
    return nesR * 16 + nesG * 4 + nesB;
  }

  private quantizeStrokeWidth(width: number): number {
    return Math.round(Math.min(width, 32) / 32 * 255) / 255;
  }

  private objectTypeToNumber(type: string | undefined): number {
    const types = ['rect', 'circle', 'triangle', 'text', 'image', 'path', 'group'];
    const index = types.indexOf(type || 'rect');
    return (index !== -1 ? index : 0) / (types.length - 1) * 2 - 1;
  }

  private quantizeZIndex(zIndex: number): number {
    return Math.round((zIndex + 100) / 200 * 255) / 255;
  }

  private quantizeRotation(rotation: number): number {
    return Math.round(((rotation % 360) + 360) % 360 / 360 * 255) / 255;
  }

  private quantizeShadow(shadow: any): number {
    if (!shadow) return 0;
    const blur = Math.min(shadow.blur || 0, 50);
    return Math.round(blur / 50 * 255) / 255;
  }

  private applyNESEncoding(value: number, embedIndex: number): number {
    // Apply NES-style bit manipulation
    const quantized = Math.round((value + 1) * 127.5); // Convert to 0-255
    const nibbleLow = quantized & 0x0F;  // Lower 4 bits
    const nibbleHigh = (quantized & 0xF0) >> 4; // Upper 4 bits
    
    // Alternate between nibbles based on embed index
    const selectedNibble = embedIndex % 2 === 0 ? nibbleLow : nibbleHigh;
    
    // Convert back to [-1, 1] range
    return selectedNibble / 15.0 * 2 - 1;
  }

  // NES-style cache checking
  private checkNESCache(cacheKey: string): CachedTensor | null {
    const cached = this.tensorCache.get(cacheKey);
    
    if (cached && (Date.now() - cached.timestamp) < 300000) { // 5 minutes
      cached.hitCount++;
      return cached;
    }
    
    return null;
  }

  private cacheInNESHierarchy(cacheKey: string, tensor: MultiDimArray): void {
    const cached: CachedTensor = {
      tensor,
      timestamp: Date.now(),
      hitCount: 1,
      memoryLevel: this.selectMemoryLevel(tensor.data.length)
    };
    
    this.tensorCache.set(cacheKey, cached);
    
    // LRU eviction (NES memory constraints simulation)
    if (this.tensorCache.size > 50) {
      const oldestKey = Array.from(this.tensorCache.keys())[0];
      this.tensorCache.delete(oldestKey);
    }
  }

  private async processWithGPUWorker(tensor: MultiDimArray): Promise<MultiDimArray> {
    return new Promise((resolve, reject) => {
      if (!this.gpuWorker) {
        reject(new Error('GPU worker not available'));
        return;
      }
      
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      const timeout = setTimeout(() => {
        reject(new Error('GPU processing timeout'));
      }, 30000); // 30 second timeout
      
      const messageHandler = (e: MessageEvent) => {
        const { type, id, data, error } = e.data;
        
        if (id === requestId) {
          clearTimeout(timeout);
          this.gpuWorker!.removeEventListener('message', messageHandler);
          
          if (type === 'SUCCESS') {
            this.stats.gpuAccelerations++;
            resolve(data);
          } else if (type === 'ERROR') {
            reject(new Error(error));
          }
        }
      };
      
      this.gpuWorker.addEventListener('message', messageHandler);
      this.gpuWorker.postMessage({
        type: 'PROCESS_TENSOR',
        id: requestId,
        data: tensor
      });
    });
  }

  private async tensorToCanvasState(
    tensor: MultiDimArray, 
    originalState: CanvasState
  ): Promise<CanvasState> {
    // Convert processed tensor back to Fabric.js format
    // This is a simplified conversion - full implementation would reconstruct objects
    
    return {
      ...originalState,
      id: `${originalState.id}_nes_gpu_processed`,
      metadata: {
        ...originalState.metadata,
        nesGpuProcessed: true,
        tensorShape: tensor.shape,
        processingLayout: tensor.layout,
        cacheKey: tensor.cacheKey,
        processingTimestamp: Date.now(),
        bitDepthOptimized: tensor.layout.includes('quantized'),
        gpuAccelerated: tensor.layout.includes('webgpu') || tensor.layout.includes('gpu')
      }
    };
  }

  private async processFallbackWithNESOptimization(state: CanvasState): Promise<CanvasState> {
    // CPU fallback with NES-style optimizations
    console.log('🎮 Using NES-style CPU fallback processing');
    
    const tensor = await this.canvasStateToTensor(state);
    const quantized = this.quantizeTensorBits(tensor, 8); // 8-bit NES-style
    
    return this.tensorToCanvasState(quantized, state);
  }

  // Utility functions
  private generateCacheKey(stateId: string, shape: number[]): string {
    return `nes_${stateId}_${shape.join('x')}_${Date.now()}`;
  }

  private determineLODLevel(objectCount: number): number {
    if (objectCount <= 10) return 0;      // Ultra quality
    if (objectCount <= 25) return 1;      // High quality
    if (objectCount <= 50) return 2;      // Medium quality
    return 3;                             // Low quality
  }

  private calculateCompressionRatio(original: MultiDimArray, compressed: MultiDimArray): number {
    const originalSize = original.data.byteLength;
    const compressedSize = compressed.data.byteLength;
    return originalSize / compressedSize;
  }

  private applyBitDepthOptimization(data: Float32Array): Float32Array {
    const browserCapabilities = this.bitDepthDetector.detect();
    
    if (browserCapabilities.totalBits <= 24) {
      // Standard browser - apply 8-bit quantization per channel
      return this.quantizeToNBits(data, 8);
    } else if (browserCapabilities.totalBits <= 30) {
      // Modern browser - apply 10-bit quantization
      return this.quantizeToNBits(data, 10);
    } else {
      // High-end browser - minimal quantization
      return this.quantizeToNBits(data, 16);
    }
  }

  private quantizeToNBits(data: Float32Array, bits: number): Float32Array {
    const levels = Math.pow(2, bits) - 1;
    const quantized = new Float32Array(data.length);
    
    for (let i = 0; i < data.length; i++) {
      const normalized = (data[i] + 1) / 2; // Convert to 0-1
      const quantizedValue = Math.round(normalized * levels) / levels;
      quantized[i] = quantizedValue * 2 - 1; // Convert back to -1,1
    }
    
    return quantized;
  }

  // Public API for statistics
  getStats(): BridgeStats {
    return { ...this.stats };
  }

  getCacheStats(): { size: number; hitRate: number } {
    const totalHits = this.stats.nesStyleCacheHits;
    const totalRequests = this.stats.totalConversions;
    
    return {
      size: this.tensorCache.size,
      hitRate: totalRequests > 0 ? (totalHits / totalRequests) * 100 : 0
    };
  }

  getMemoryUsage(): NESGPUMemoryHierarchy {
    return { ...this.memoryHierarchy };
  }

  clearCache(): void {
    this.tensorCache.clear();
    console.log('🗑️ NES-style cache cleared');
  }

  dispose(): void {
    if (this.gpuWorker) {
      this.gpuWorker.terminate();
      this.gpuWorker = null;
    }
    this.clearCache();
  }
}

// Supporting classes and interfaces
class BitDepthDetector {
  detect(): BitDepthProfile {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Detect color depth capabilities
    const imageData = ctx?.createImageData(1, 1);
    const contextAttributes = ctx?.getContextAttributes?.() || {};
    
    return {
      standard: 24,     // 24-bit RGB
      modern: 30,       // 30-bit HDR
      premium: 48,      // 48-bit ProPhoto RGB
      target: 32,       // 32-bit RGBA (our target)
      compressed: 16,   // 16-bit for cache
      minimal: 8        // 8-bit NES-style
    };
  }
}

interface CachedTensor {
  tensor: MultiDimArray;
  timestamp: number;
  hitCount: number;
  memoryLevel: keyof NESGPUMemoryHierarchy;
}

interface BridgeStats {
  totalConversions: number;
  cacheHitRate: number;
  averageCompressionRatio: number;
  bitDepthOptimizations: number;
  gpuAccelerations: number;
  nesStyleCacheHits: number;
  quantizationSavings: number;
}

// Export the main class and types
export { NESStyleGPUBridge };
export type { BitDepthProfile, CacheTable, NESGPUMemoryHierarchy, BridgeStats };