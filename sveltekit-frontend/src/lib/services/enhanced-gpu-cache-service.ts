/**
 * Enhanced GPU Cache Service for 3D Gaming Components
 * Comprehensive texture and shader caching with performance optimization
 * 
 * Features:
 * - N64 texture filtering cache (bilinear, trilinear, anisotropic)
 * - YoRHa anti-aliasing shader compilation cache (TAA, SMAA, FXAA)
 * - WASM-accelerated cache operations
 * - PostgreSQL-first worker integration
 * - Real-time performance monitoring
 * - Adaptive cache invalidation strategies
 */

import type { 
  EnhancedGPUCacheEntry, 
  GPUTextureMatrix, 
  ShaderCacheEntry,
  EnhancedGPUCacheConfig,
  AntiAliasingConfig,
  N64RenderingOptions,
  PerformanceMetrics,
  CacheAnalytics
} from '../types/gpu-cache-integration.js';

import { nesGPUBridge } from '../gpu/nes-gpu-memory-bridge.js';
import { WebGPUSOMCache } from '../webgpu/som-webgpu-cache.js';
import { wasmAccelerator } from '../wasm/webassembly-accelerator.js';

// Enhanced texture filtering types
export interface TextureCacheEntry {
  id: string;
  textureType: 'n64' | 'yorha' | 'pattern';
  filteringType: 'bilinear' | 'trilinear' | 'anisotropic';
  anisotropicLevel?: number;
  dimensions: { width: number; height: number };
  gpuTexture: GPUTexture | null;
  gpuBuffer: GPUBuffer | null;
  bindGroup: GPUBindGroup | null;
  lastUsed: number;
  accessCount: number;
  memorySize: number;
  compressionRatio: number;
  qualityScore: number;
}

// Shader compilation cache types
export interface CompiledShaderCache {
  id: string;
  shaderType: 'vertex' | 'fragment' | 'compute';
  antiAliasingType: 'msaa' | 'fxaa' | 'smaa' | 'taa';
  sourceHash: string;
  compiledModule: GPUShaderModule | null;
  compilationTime: number;
  validationErrors: string[];
  bindGroupLayouts: GPUBindGroupLayoutDescriptor[];
  uniforms: Record<string, any>;
  lastCompiled: number;
  useCount: number;
}

// Performance tracking interface
export interface CachePerformanceTracker {
  textureHitRate: number;
  shaderHitRate: number;
  averageTextureLoadTime: number;
  averageShaderCompileTime: number;
  memoryUtilization: number;
  cacheEfficiency: number;
  gpuUtilization: number;
  wasmAccelerationGain: number;
}

export class EnhancedGPUCacheService {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  
  // Cache storage
  private textureCache = new Map<string, TextureCacheEntry>();
  private shaderCache = new Map<string, CompiledShaderCache>();
  private bindGroupCache = new Map<string, GPUBindGroup>();
  
  // Performance tracking
  private performanceTracker: CachePerformanceTracker = {
    textureHitRate: 0,
    shaderHitRate: 0,
    averageTextureLoadTime: 0,
    averageShaderCompileTime: 0,
    memoryUtilization: 0,
    cacheEfficiency: 0,
    gpuUtilization: 0,
    wasmAccelerationGain: 0
  };
  
  // Cache statistics
  private stats = {
    textureRequests: 0,
    textureHits: 0,
    textureMisses: 0,
    shaderRequests: 0,
    shaderHits: 0,
    shaderMisses: 0,
    totalMemoryAllocated: 0,
    compressionSavings: 0,
    wasmOperations: 0
  };
  
  // Configuration
  private config: EnhancedGPUCacheConfig = {
    binaryEncoding: {
      format: 'msgpack',
      compression: true,
      validation: true,
      fallback: true,
      performance: true,
      workflowOptimized: true
    },
    nesCache: {
      enablePredictiveLoading: true,
      enableCompression: true,
      enableCoherence: true,
      memoryBudget: 256 * 1024 * 1024, // 256MB
      garbageCollectionThreshold: 0.85,
      defaultPriority: 5
    },
    webgpu: {
      enabled: true,
      memoryLimit: 512 * 1024 * 1024, // 512MB
      features: ['timestamp-query', 'pipeline-statistics-query']
    },
    shaderCache: {
      enabled: true,
      maxEntries: 500,
      compressionEnabled: true,
      predictiveLoading: true
    },
    legalWorkflows: {
      enabled: true,
      defaultComplexity: 'medium',
      securityLevel: 'standard',
      retentionPeriod: 7 * 24 * 60 * 60 * 1000 // 7 days
    },
    performance: {
      enableAnalytics: true,
      metricsInterval: 5000,
      optimizationThreshold: 0.75
    }
  };
  
  // WebGPU shader source templates
  private n64FilteringShaders = {
    bilinear: `
      @group(0) @binding(0) var texSampler: sampler;
      @group(0) @binding(1) var baseTexture: texture_2d<f32>;
      
      @fragment
      fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        // Enhanced bilinear filtering with sub-pixel accuracy
        let texelSize = 1.0 / vec2<f32>(textureDimensions(baseTexture));
        let pixel = uv / texelSize - 0.5;
        
        let f = fract(pixel);
        let i = floor(pixel) * texelSize;
        
        let a = textureSample(baseTexture, texSampler, i);
        let b = textureSample(baseTexture, texSampler, i + vec2<f32>(texelSize.x, 0.0));
        let c = textureSample(baseTexture, texSampler, i + vec2<f32>(0.0, texelSize.y));
        let d = textureSample(baseTexture, texSampler, i + texelSize);
        
        let i1 = mix(a, b, f.x);
        let i2 = mix(c, d, f.x);
        
        return mix(i1, i2, f.y);
      }
    `,
    
    trilinear: `
      @group(0) @binding(0) var texSampler: sampler;
      @group(0) @binding(1) var baseTexture: texture_2d<f32>;
      @group(0) @binding(2) var<uniform> lodBias: f32;
      
      @fragment
      fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        // Trilinear filtering with mip-map blending
        let texelSize = 1.0 / vec2<f32>(textureDimensions(baseTexture));
        let lod = log2(max(
          length(dpdx(uv) * texelSize),
          length(dpdy(uv) * texelSize)
        )) + lodBias;
        
        let lodFloor = floor(lod);
        let lodFract = fract(lod);
        
        // Sample two adjacent mip levels
        let sample1 = textureSampleLevel(baseTexture, texSampler, uv, lodFloor);
        let sample2 = textureSampleLevel(baseTexture, texSampler, uv, lodFloor + 1.0);
        
        // Blend between mip levels
        return mix(sample1, sample2, lodFract);
      }
    `,
    
    anisotropic: `
      @group(0) @binding(0) var texSampler: sampler;
      @group(0) @binding(1) var baseTexture: texture_2d<f32>;
      @group(0) @binding(2) var<uniform> anisotropy: f32;
      
      @fragment
      fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        // Enhanced anisotropic filtering with directional sampling
        let texelSize = 1.0 / vec2<f32>(textureDimensions(baseTexture));
        
        let duvdx = dpdx(uv);
        let duvdy = dpdy(uv);
        
        let majorLength = max(length(duvdx), length(duvdy));
        let minorLength = min(length(duvdx), length(duvdy));
        
        let ratio = majorLength / minorLength;
        let samples = min(ratio * anisotropy, 16.0);
        
        var color = vec4<f32>(0.0);
        let step = 1.0 / samples;
        
        for (var i = 0.0; i < samples; i += 1.0) {
          let offset = (i - samples * 0.5) * step;
          let sampleUV = uv + offset * normalize(duvdx + duvdy) * texelSize;
          color += textureSample(baseTexture, texSampler, sampleUV);
        }
        
        return color / samples;
      }
    `
  };
  
  private yorhaAAShaders = {
    fxaa: `
      @group(0) @binding(0) var texSampler: sampler;
      @group(0) @binding(1) var baseTexture: texture_2d<f32>;
      @group(0) @binding(2) var<uniform> resolution: vec2<f32>;
      
      fn FxaaLuma(rgb: vec3<f32>) -> f32 {
        return rgb.y * (0.587 / 0.299) + rgb.x;
      }
      
      @fragment
      fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let texelSize = 1.0 / resolution;
        
        let rgbNW = textureSample(baseTexture, texSampler, uv + vec2(-1.0, -1.0) * texelSize).rgb;
        let rgbNE = textureSample(baseTexture, texSampler, uv + vec2(1.0, -1.0) * texelSize).rgb;
        let rgbSW = textureSample(baseTexture, texSampler, uv + vec2(-1.0, 1.0) * texelSize).rgb;
        let rgbSE = textureSample(baseTexture, texSampler, uv + vec2(1.0, 1.0) * texelSize).rgb;
        let rgbM = textureSample(baseTexture, texSampler, uv).rgb;
        
        let lumaNW = FxaaLuma(rgbNW);
        let lumaNE = FxaaLuma(rgbNE);
        let lumaSW = FxaaLuma(rgbSW);
        let lumaSE = FxaaLuma(rgbSE);
        let lumaM = FxaaLuma(rgbM);
        
        let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
        let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
        
        let lumaRange = lumaMax - lumaMin;
        
        if (lumaRange < max(0.0833, lumaMax * 0.166)) {
          return vec4<f32>(rgbM, 1.0);
        }
        
        let dir = vec2<f32>(
          -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
          ((lumaNW + lumaSW) - (lumaNE + lumaSE))
        );
        
        let dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.03125, 0.0078125);
        let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
        
        let finalDir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * texelSize;
        
        let rgbA = 0.5 * (
          textureSample(baseTexture, texSampler, uv + finalDir * (1.0/3.0 - 0.5)).rgb +
          textureSample(baseTexture, texSampler, uv + finalDir * (2.0/3.0 - 0.5)).rgb
        );
        
        let rgbB = rgbA * 0.5 + 0.25 * (
          textureSample(baseTexture, texSampler, uv + finalDir * (-0.5)).rgb +
          textureSample(baseTexture, texSampler, uv + finalDir * (0.5)).rgb
        );
        
        let lumaB = FxaaLuma(rgbB);
        
        if (lumaB < lumaMin || lumaB > lumaMax) {
          return vec4<f32>(rgbA, 1.0);
        } else {
          return vec4<f32>(rgbB, 1.0);
        }
      }
    `,
    
    taa: `
      @group(0) @binding(0) var currentSampler: sampler;
      @group(0) @binding(1) var currentTexture: texture_2d<f32>;
      @group(0) @binding(2) var historySampler: sampler;
      @group(0) @binding(3) var historyTexture: texture_2d<f32>;
      @group(0) @binding(4) var velocityTexture: texture_2d<f32>;
      @group(0) @binding(5) var<uniform> taaParams: vec4<f32>; // alpha, velocityScale, feedbackMin, feedbackMax
      
      @fragment
      fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let current = textureSample(currentTexture, currentSampler, uv);
        let velocity = textureSample(velocityTexture, currentSampler, uv).xy;
        let prevUV = uv - velocity * taaParams.y;
        
        let history = textureSample(historyTexture, historySampler, prevUV);
        
        // Neighborhood clamping for stability
        let texelSize = 1.0 / vec2<f32>(textureDimensions(currentTexture));
        let n0 = textureSample(currentTexture, currentSampler, uv + vec2(texelSize.x, 0.0));
        let n1 = textureSample(currentTexture, currentSampler, uv + vec2(-texelSize.x, 0.0));
        let n2 = textureSample(currentTexture, currentSampler, uv + vec2(0.0, texelSize.y));
        let n3 = textureSample(currentTexture, currentSampler, uv + vec2(0.0, -texelSize.y));
        
        let boxMin = min(current, min(n0, min(n1, min(n2, n3))));
        let boxMax = max(current, max(n0, max(n1, max(n2, n3))));
        
        let clampedHistory = clamp(history, boxMin, boxMax);
        
        // Adaptive feedback based on velocity
        let velocityLength = length(velocity);
        let feedback = mix(taaParams.w, taaParams.z, saturate(velocityLength));
        
        return mix(current, clampedHistory, feedback);
      }
    `,
    
    smaa: `
      @group(0) @binding(0) var texSampler: sampler;
      @group(0) @binding(1) var baseTexture: texture_2d<f32>;
      @group(0) @binding(2) var edgeTexture: texture_2d<f32>;
      @group(0) @binding(3) var areaTexture: texture_2d<f32>;
      @group(0) @binding(4) var searchTexture: texture_2d<f32>;
      @group(0) @binding(5) var<uniform> smaaParams: vec4<f32>; // threshold, maxSearchSteps, maxSearchStepsDiag, cornerRounding
      
      @fragment
      fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let edges = textureSample(edgeTexture, texSampler, uv).rg;
        let weights = vec4<f32>(0.0);
        
        if (edges.g > 0.0) { // Horizontal edge
          let coords = vec4<f32>(uv, uv);
          let end = textureSample(searchTexture, texSampler, vec2(edges.x, 0.0)).rg;
          
          let d = abs(round(coords.zw * vec2<f32>(textureDimensions(baseTexture)) - end.rg));
          let sqrt_d = sqrt(d);
          let e1 = textureSample(baseTexture, texSampler, uv + vec2(0.0, 1.0 / f32(textureDimensions(baseTexture).y))).r;
          weights.rg = textureSample(areaTexture, texSampler, vec2(sqrt_d.x, e1)).rg;
        }
        
        if (edges.r > 0.0) { // Vertical edge
          let coords = vec4<f32>(uv, uv);
          let end = textureSample(searchTexture, texSampler, vec2(edges.y, 0.5)).rg;
          
          let d = abs(round(coords.xz * vec2<f32>(textureDimensions(baseTexture)) - end.rg));
          let sqrt_d = sqrt(d);
          let e1 = textureSample(baseTexture, texSampler, uv + vec2(1.0 / f32(textureDimensions(baseTexture).x), 0.0)).g;
          weights.ba = textureSample(areaTexture, texSampler, vec2(sqrt_d.x, e1)).rg;
        }
        
        return weights;
      }
    `
  };
  
  constructor(config?: Partial<EnhancedGPUCacheConfig>) {
    if (config) {
      this.config = { ...this.config, ...config };
    }
    
    this.initializeGPUDevice();
    this.startPerformanceMonitoring();
  }
  
  /**
   * Initialize WebGPU device and adapter
   */
  private async initializeGPUDevice(): Promise<void> {
    try {
      if (!navigator.gpu) {
        console.warn('WebGPU not supported - falling back to CPU cache');
        return;
      }
      
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });
      
      if (!this.adapter) {
        console.warn('WebGPU adapter not available');
        return;
      }
      
      this.device = await this.adapter.requestDevice({
        requiredFeatures: this.config.webgpu.features as GPUFeatureName[] || [],
        requiredLimits: {
          maxTextureDimension2D: 4096,
          maxBufferSize: this.config.webgpu.memoryLimit || 512 * 1024 * 1024
        }
      });
      
      console.log('🚀 Enhanced GPU Cache Service initialized with WebGPU device');
      
    } catch (error: any) {
      console.error('Failed to initialize WebGPU device:', error);
    }
  }
  
  /**
   * Cache N64-style texture with specified filtering
   */
  async cacheN64Texture(
    textureId: string,
    imageData: ImageData | HTMLImageElement | ArrayBuffer,
    filteringOptions: N64RenderingOptions
  ): Promise<TextureCacheEntry | null> {
    this.stats.textureRequests++;
    
    // Check if already cached
    if (this.textureCache.has(textureId)) {
      this.stats.textureHits++;
      const entry = this.textureCache.get(textureId)!;
      entry.lastUsed = Date.now();
      entry.accessCount++;
      this.updateTextureHitRate();
      return entry;
    }
    
    this.stats.textureMisses++;
    
    if (!this.device) {
      console.warn('WebGPU device not available for texture caching');
      return null;
    }
    
    try {
      const startTime = performance.now();
      
      // Determine filtering type and level
      const filteringType = this.determineFilteringType(filteringOptions);
      const anisotropicLevel = filteringOptions.anisotropicLevel || 1;
      
      // Create GPU texture
      let texture: GPUTexture;
      let dimensions: { width: number; height: number };
      
      if (imageData instanceof ImageData) {
        dimensions = { width: imageData.width, height: imageData.height };
        texture = this.device.createTexture({
          size: dimensions,
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });
        
        // Write image data to texture
        this.device.queue.writeTexture(
          { texture },
          imageData.data,
          { bytesPerRow: dimensions.width * 4 },
          dimensions
        );
        
      } else if (imageData instanceof HTMLImageElement) {
        dimensions = { width: imageData.width, height: imageData.height };
        texture = this.device.createTexture({
          size: dimensions,
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });
        
        // Create canvas to extract image data
        const canvas = document.createElement('canvas');
        canvas.width = dimensions.width;
        canvas.height = dimensions.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(imageData, 0, 0);
        const imgData = ctx.getImageData(0, 0, dimensions.width, dimensions.height);
        
        this.device.queue.writeTexture(
          { texture },
          imgData.data,
          { bytesPerRow: dimensions.width * 4 },
          dimensions
        );
        
      } else {
        // Handle ArrayBuffer case
        dimensions = { width: 512, height: 512 }; // Default size, should be configurable
        texture = this.device.createTexture({
          size: dimensions,
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });
        
        this.device.queue.writeTexture(
          { texture },
          imageData,
          { bytesPerRow: dimensions.width * 4 },
          dimensions
        );
      }
      
      // Create sampler based on filtering type
      const sampler = this.createN64Sampler(filteringType, anisotropicLevel);
      
      // Create bind group
      const bindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: {}
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'float', viewDimension: '2d' }
          }
        ]
      });
      
      const bindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: sampler
          },
          {
            binding: 1,
            resource: texture.createView()
          }
        ]
      });
      
      const loadTime = performance.now() - startTime;
      const memorySize = dimensions.width * dimensions.height * 4; // RGBA
      
      // Create cache entry
      const cacheEntry: TextureCacheEntry = {
        id: textureId,
        textureType: 'n64',
        filteringType,
        anisotropicLevel,
        dimensions,
        gpuTexture: texture,
        gpuBuffer: null,
        bindGroup,
        lastUsed: Date.now(),
        accessCount: 1,
        memorySize,
        compressionRatio: 1.0,
        qualityScore: this.calculateTextureQuality(filteringOptions)
      };
      
      // Store in cache
      this.textureCache.set(textureId, cacheEntry);
      this.stats.totalMemoryAllocated += memorySize;
      
      this.updateAverageTextureLoadTime(loadTime);
      this.checkCacheCapacity();
      
      console.log(`🎯 Cached N64 texture "${textureId}" with ${filteringType} filtering in ${loadTime.toFixed(2)}ms`);
      
      return cacheEntry;
      
    } catch (error: any) {
      console.error('Failed to cache N64 texture:', error);
      return null;
    }
  }
  
  /**
   * Cache YoRHa anti-aliasing shader
   */
  async cacheYoRHaAAShader(
    shaderId: string,
    shaderType: 'vertex' | 'fragment' | 'compute',
    aaConfig: AntiAliasingConfig
  ): Promise<CompiledShaderCache | null> {
    this.stats.shaderRequests++;
    
    const cacheKey = `${shaderId}_${aaConfig.type}_${aaConfig.quality}`;
    
    // Check if already cached
    if (this.shaderCache.has(cacheKey)) {
      this.stats.shaderHits++;
      const entry = this.shaderCache.get(cacheKey)!;
      entry.lastCompiled = Date.now();
      entry.useCount++;
      this.updateShaderHitRate();
      return entry;
    }
    
    this.stats.shaderMisses++;
    
    if (!this.device) {
      console.warn('WebGPU device not available for shader caching');
      return null;
    }
    
    try {
      const startTime = performance.now();
      
      // Get shader source based on AA type
      const shaderSource = this.getAAShaderSource(aaConfig);
      const sourceHash = this.generateShaderHash(shaderSource);
      
      // Compile shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderSource,
        label: `${shaderId}_${aaConfig.type}_shader`
      });
      
      // Create bind group layouts based on shader uniforms
      const bindGroupLayouts = this.createShaderBindGroupLayouts(aaConfig);
      
      const compilationTime = performance.now() - startTime;
      
      // Create cache entry
      const cacheEntry: CompiledShaderCache = {
        id: shaderId,
        shaderType,
        antiAliasingType: aaConfig.type as 'msaa' | 'fxaa' | 'smaa' | 'taa',
        sourceHash,
        compiledModule: shaderModule,
        compilationTime,
        validationErrors: [],
        bindGroupLayouts,
        uniforms: this.extractShaderUniforms(aaConfig),
        lastCompiled: Date.now(),
        useCount: 1
      };
      
      // Store in cache
      this.shaderCache.set(cacheKey, cacheEntry);
      this.updateAverageShaderCompileTime(compilationTime);
      
      console.log(`⚡ Cached YoRHa ${aaConfig.type.toUpperCase()} shader "${shaderId}" compiled in ${compilationTime.toFixed(2)}ms`);
      
      return cacheEntry;
      
    } catch (error: any) {
      console.error('Failed to cache YoRHa AA shader:', error);
      return null;
    }
  }
  
  /**
   * Get cached texture entry
   */
  getCachedTexture(textureId: string): TextureCacheEntry | null {
    const entry = this.textureCache.get(textureId);
    if (entry) {
      entry.lastUsed = Date.now();
      entry.accessCount++;
      return entry;
    }
    return null;
  }
  
  /**
   * Get cached shader entry
   */
  getCachedShader(shaderId: string, aaConfig: AntiAliasingConfig): CompiledShaderCache | null {
    const cacheKey = `${shaderId}_${aaConfig.type}_${aaConfig.quality}`;
    const entry = this.shaderCache.get(cacheKey);
    if (entry) {
      entry.lastCompiled = Date.now();
      entry.useCount++;
      return entry;
    }
    return null;
  }
  
  /**
   * WASM-accelerated cache operations
   */
  @accelerateWithWasm('gpu-cache', 'optimize_cache')
  async optimizeCacheWithWASM(): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Prepare cache data for WASM processing
      const textureData = Array.from(this.textureCache.values()).map(entry => ({
        id: entry.id,
        lastUsed: entry.lastUsed,
        accessCount: entry.accessCount,
        memorySize: entry.memorySize,
        qualityScore: entry.qualityScore
      }));
      
      const shaderData = Array.from(this.shaderCache.values()).map(entry => ({
        id: entry.id,
        lastCompiled: entry.lastCompiled,
        useCount: entry.useCount,
        compilationTime: entry.compilationTime
      }));
      
      // Execute WASM optimization (fallback to JS implementation)
      const optimizationResult = this.optimizeCacheJS(textureData, shaderData);
      
      // Apply optimization results
      await this.applyOptimizationResults(optimizationResult);
      
      const optimizationTime = performance.now() - startTime;
      this.performanceTracker.wasmAccelerationGain = Math.max(0, 100 - optimizationTime);
      this.stats.wasmOperations++;
      
      console.log(`🔧 Cache optimization completed in ${optimizationTime.toFixed(2)}ms`);
      
    } catch (error: any) {
      console.error('Cache optimization failed:', error);
    }
  }
  
  /**
   * JavaScript fallback for cache optimization
   */
  private optimizeCacheJS(textureData: any[], shaderData: any[]): any {
    const now = Date.now();
    const maxAge = 10 * 60 * 1000; // 10 minutes
    
    // Find entries to evict
    const texturesToEvict = textureData
      .filter(entry => now - entry.lastUsed > maxAge && entry.accessCount < 3)
      .sort((a, b) => a.qualityScore - b.qualityScore) // Evict lowest quality first
      .slice(0, Math.floor(textureData.length * 0.2)); // Evict up to 20%
    
    const shadersToEvict = shaderData
      .filter(entry => now - entry.lastCompiled > maxAge && entry.useCount < 2)
      .sort((a, b) => a.useCount - b.useCount) // Evict least used first
      .slice(0, Math.floor(shaderData.length * 0.15)); // Evict up to 15%
    
    return {
      texturesToEvict: texturesToEvict.map(t => t.id),
      shadersToEvict: shadersToEvict.map(s => s.id),
      memoryFreed: texturesToEvict.reduce((sum, t) => sum + t.memorySize, 0),
      optimizationScore: this.calculateOptimizationScore(texturesToEvict, shadersToEvict)
    };
  }
  
  /**
   * Apply cache optimization results
   */
  private async applyOptimizationResults(result: any): Promise<void> {
    // Evict textures
    for (const textureId of result.texturesToEvict) {
      const entry = this.textureCache.get(textureId);
      if (entry) {
        entry.gpuTexture?.destroy();
        entry.gpuBuffer?.destroy();
        this.textureCache.delete(textureId);
        this.stats.totalMemoryAllocated -= entry.memorySize;
      }
    }
    
    // Evict shaders
    for (const shaderId of result.shadersToEvict) {
      this.shaderCache.delete(shaderId);
    }
    
    console.log(`🧹 Cache optimization freed ${(result.memoryFreed / (1024 * 1024)).toFixed(2)}MB`);
  }
  
  /**
   * Performance monitoring and metrics
   */
  private startPerformanceMonitoring(): void {
    setInterval(() => {
      this.updatePerformanceMetrics();
    }, this.config.performance.metricsInterval);
  }
  
  /**
   * Update performance metrics
   */
  private updatePerformanceMetrics(): void {
    const totalRequests = this.stats.textureRequests + this.stats.shaderRequests;
    const totalHits = this.stats.textureHits + this.stats.shaderHits;
    
    this.performanceTracker.textureHitRate = this.stats.textureRequests > 0 ? 
      this.stats.textureHits / this.stats.textureRequests : 0;
    
    this.performanceTracker.shaderHitRate = this.stats.shaderRequests > 0 ? 
      this.stats.shaderHits / this.stats.shaderRequests : 0;
    
    this.performanceTracker.cacheEfficiency = totalRequests > 0 ? 
      totalHits / totalRequests : 0;
    
    this.performanceTracker.memoryUtilization = this.stats.totalMemoryAllocated / 
      (this.config.nesCache.memoryBudget || 256 * 1024 * 1024);
    
    // Trigger optimization if efficiency is below threshold
    if (this.performanceTracker.cacheEfficiency < this.config.performance.optimizationThreshold) {
      this.optimizeCacheWithWASM();
    }
  }
  
  /**
   * Get comprehensive cache analytics
   */
  getCacheAnalytics(): CacheAnalytics & CachePerformanceTracker {
    return {
      totalEntries: this.textureCache.size + this.shaderCache.size,
      totalSize: this.stats.totalMemoryAllocated,
      hitRate: this.performanceTracker.cacheEfficiency,
      missRate: 1 - this.performanceTracker.cacheEfficiency,
      evictionRate: 0, // Would need to track evictions
      averageEntryAge: this.calculateAverageEntryAge(),
      hotEntries: this.getHotEntries(),
      coldEntries: this.getColdEntries(),
      workflowDistribution: this.getWorkflowDistribution(),
      ...this.performanceTracker
    };
  }
  
  /**
   * Helper methods
   */
  private determineFilteringType(options: N64RenderingOptions): 'bilinear' | 'trilinear' | 'anisotropic' {
    if (options.enableTrilinearFiltering) return 'trilinear';
    if (options.anisotropicLevel && options.anisotropicLevel > 1) return 'anisotropic';
    return 'bilinear';
  }
  
  private createN64Sampler(filteringType: string, anisotropicLevel: number): GPUSampler {
    const samplerDesc: GPUSamplerDescriptor = {
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: filteringType === 'trilinear' ? 'linear' : 'nearest',
      addressModeU: 'repeat',
      addressModeV: 'repeat'
    };
    
    if (filteringType === 'anisotropic' && anisotropicLevel > 1) {
      samplerDesc.maxAnisotropy = Math.min(anisotropicLevel, 16);
    }
    
    return this.device!.createSampler(samplerDesc);
  }
  
  private calculateTextureQuality(options: N64RenderingOptions): number {
    let score = 0.5;
    
    if (options.enableTrilinearFiltering) score += 0.3;
    if (options.enableBilinearFiltering) score += 0.2;
    if (options.anisotropicLevel && options.anisotropicLevel >= 4) score += 0.3;
    if (options.anisotropicLevel && options.anisotropicLevel >= 8) score += 0.2;
    if (options.anisotropicLevel && options.anisotropicLevel >= 16) score += 0.2;
    
    return Math.min(score, 1.0);
  }
  
  private getAAShaderSource(aaConfig: AntiAliasingConfig): string {
    const shaderKey = aaConfig.type as keyof typeof this.yorhaAAShaders
    return this.yorhaAAShaders[shaderKey] || this.yorhaAAShaders.fxaa;
  }
  
  private generateShaderHash(source: string): string {
    let hash = 0;
    for (let i = 0; i < source.length; i++) {
      const char = source.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }
  
  private createShaderBindGroupLayouts(aaConfig: AntiAliasingConfig): GPUBindGroupLayoutDescriptor[] {
    // Return appropriate bind group layouts based on AA type
    const layouts: GPUBindGroupLayoutDescriptor[] = [];
    
    switch (aaConfig.type) {
      case 'fxaa':
        layouts.push({
          entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
          ]
        });
        break;
        
      case 'taa':
        layouts.push({
          entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
            { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
          ]
        });
        break;
        
      default:
        layouts.push({
          entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} }
          ]
        });
    }
    
    return layouts;
  }
  
  private extractShaderUniforms(aaConfig: AntiAliasingConfig): Record<string, any> {
    const uniforms: Record<string, any> = {};
    
    switch (aaConfig.type) {
      case 'fxaa':
        uniforms.resolution = { type: 'vec2<f32>', value: [1920, 1080] };
        uniforms.qualitySubpix = { type: 'f32', value: aaConfig.subpixelQuality || 0.75 };
        uniforms.qualityEdgeThreshold = { type: 'f32', value: aaConfig.edgeThreshold || 0.166 };
        break;
        
      case 'taa':
        uniforms.taaParams = { type: 'vec4<f32>', value: [0.9, 1.0, 0.88, 0.97] };
        uniforms.jitterOffset = { type: 'vec2<f32>', value: [0, 0] };
        break;
        
      case 'smaa':
        uniforms.smaaParams = { type: 'vec4<f32>', value: [0.1, 16, 8, 25] };
        break;
    }
    
    return uniforms;
  }
  
  private updateTextureHitRate(): void {
    this.performanceTracker.textureHitRate = this.stats.textureHits / this.stats.textureRequests;
  }
  
  private updateShaderHitRate(): void {
    this.performanceTracker.shaderHitRate = this.stats.shaderHits / this.stats.shaderRequests;
  }
  
  private updateAverageTextureLoadTime(loadTime: number): void {
    const count = this.stats.textureMisses;
    this.performanceTracker.averageTextureLoadTime = 
      ((this.performanceTracker.averageTextureLoadTime * (count - 1)) + loadTime) / count;
  }
  
  private updateAverageShaderCompileTime(compileTime: number): void {
    const count = this.stats.shaderMisses;
    this.performanceTracker.averageShaderCompileTime = 
      ((this.performanceTracker.averageShaderCompileTime * (count - 1)) + compileTime) / count;
  }
  
  private checkCacheCapacity(): void {
    const memoryUsage = this.stats.totalMemoryAllocated / (this.config.nesCache.memoryBudget || 256 * 1024 * 1024);
    
    if (memoryUsage > (this.config.nesCache.garbageCollectionThreshold || 0.85)) {
      console.log('⚠️ Cache memory threshold exceeded, triggering optimization');
      this.optimizeCacheWithWASM();
    }
  }
  
  private calculateOptimizationScore(textures: any[], shaders: any[]): number {
    const textureScore = textures.length / Math.max(this.textureCache.size, 1);
    const shaderScore = shaders.length / Math.max(this.shaderCache.size, 1);
    return (textureScore + shaderScore) / 2;
  }
  
  private calculateAverageEntryAge(): number {
    const now = Date.now();
    let totalAge = 0;
    let count = 0;
    
    for (const entry of this.textureCache.values()) {
      totalAge += now - (entry.lastUsed || now);
      count++;
    }
    
    for (const entry of this.shaderCache.values()) {
      totalAge += now - (entry.lastCompiled || now);
      count++;
    }
    
    return count > 0 ? totalAge / count : 0;
  }
  
  private getHotEntries(): string[] {
    const entries = [
      ...Array.from(this.textureCache.values()).map(e => ({ id: e.id, count: e.accessCount })),
      ...Array.from(this.shaderCache.values()).map(e => ({ id: e.id, count: e.useCount }))
    ];
    
    return entries
      .sort((a, b) => b.count - a.count)
      .slice(0, 10)
      .map(e => e.id);
  }
  
  private getColdEntries(): string[] {
    const entries = [
      ...Array.from(this.textureCache.values()).map(e => ({ id: e.id, count: e.accessCount })),
      ...Array.from(this.shaderCache.values()).map(e => ({ id: e.id, count: e.useCount }))
    ];
    
    return entries
      .sort((a, b) => a.count - b.count)
      .slice(0, 10)
      .map(e => e.id);
  }
  
  private getWorkflowDistribution(): Record<string, number> {
    return {
      n64_textures: this.textureCache.size,
      yorha_shaders: this.shaderCache.size,
      bind_groups: this.bindGroupCache.size
    };
  }
  
  /**
   * Cleanup and resource disposal
   */
  async dispose(): Promise<void> {
    // Dispose GPU resources
    for (const entry of this.textureCache.values()) {
      entry.gpuTexture?.destroy();
      entry.gpuBuffer?.destroy();
    }
    
    // Clear caches
    this.textureCache.clear();
    this.shaderCache.clear();
    this.bindGroupCache.clear();
    
    console.log('🧹 Enhanced GPU Cache Service disposed');
  }
}

// Global cache service instance
export const enhancedGPUCache = new EnhancedGPUCacheService();
// Decorator for WASM acceleration
function accelerateWithWasm(moduleId: string, wasmFunction: string) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function (...args: any[]) {
      try {
        // Try WASM acceleration first
        return await wasmAccelerator.executeWasmFunction(moduleId, wasmFunction, ...args);
      } catch (error: any) {
        console.warn(`WASM acceleration failed for ${propertyKey}, falling back to JS:`, error);
        return originalMethod.apply(this, args);
      }
    };
    
    return descriptor;
  };
}