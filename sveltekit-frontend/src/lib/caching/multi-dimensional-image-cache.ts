/**
 * Multi-Dimensional Image Cache with GPU Integration
 * Integrates with NES-GPU Memory Bridge, SOM Neural Networks, Auto-Encoders, and Tensor Tiling
 * for optimal graph traversal visualization image generation and caching
 */

import { SOMNeuralNetwork, type SOMDecomposition } from '../ai/som-neural-network';
import { GraphPatternAutoEncoder, type EncodedGraphPattern } from '../ai/graph-pattern-autoencoder';
import { nesGPUBridge, type GPUTextureMatrix } from '../gpu/nes-gpu-memory-bridge';
import { MultiLayerCache } from '../services/multi-layer-cache';
import { reinforcementLearningCache } from './reinforcement-learning-cache.server';

export interface ImageCacheEntry {
  id: string;
  algorithm: 'dfs' | 'bfs' | 'som' | 'autoencoder' | 'hybrid';
  imageData: string; // Base64 encoded
  dimensions: { width: number; height: number };
  metadata: ImageMetadata;
  compressionData?: {
    original: EncodedGraphPattern;
    som: SOMDecomposition;
    compressed: ArrayBuffer;
  };
  gpuTexture?: GPUTextureMatrix;
  cacheStats: CacheStats;
  timestamp: number;
}

export interface ImageMetadata {
  graphSignature: string;
  nodeCount: number;
  edgeCount: number;
  processingTime: number;
  compressionRatio: number;
  qualityScore: number;
  legalDomain: string;
  patterns: {
    citationDensity: number;
    jurisdictionalSpread: number;
    temporalRange: number;
    complexityIndex: number;
  };
}

export interface CacheStats {
  hitCount: number;
  missCount: number;
  size: number;
  lastAccessed: number;
  generationCost: number;
  compressionEfficiency: number;
}

export interface CacheDimensions {
  temporal: 'recent' | 'historical' | 'archived';
  spatial: 'local' | 'regional' | 'global';
  semantic: 'simple' | 'complex' | 'expert';
  visual: 'thumbnail' | 'standard' | 'hires';
  algorithm: 'dfs' | 'bfs' | 'som' | 'autoencoder' | 'hybrid';
}

export interface CacheLayer {
  name: string;
  capacity: number;
  ttl: number;
  priority: number;
  evictionPolicy: 'lru' | 'lfu' | 'rl' | 'som_guided';
  compression: boolean;
}

export interface MultiDimensionalQuery {
  dimensions: Partial<CacheDimensions>;
  graphSignature: string;
  requiredQuality: number;
  acceptableAge: number;
  preferredAlgorithms: string[];
}

export class MultiDimensionalImageCache {
  private som: SOMNeuralNetwork;
  private autoencoder: GraphPatternAutoEncoder;
  private multiLayerCache: MultiLayerCache;
  private rlCache = reinforcementLearningCache;
  
  // Multi-dimensional storage
  private dimensionalIndices: Map<string, Set<string>> = new Map();
  private imageEntries: Map<string, ImageCacheEntry> = new Map();
  private gpuTextures: Map<string, GPUTextureMatrix> = new Map();
  
  // Cache layers with different characteristics
  private cacheLayers: CacheLayer[] = [
    {
      name: 'gpu_texture',
      capacity: 50, // Limited GPU memory
      ttl: 30000, // 30 seconds
      priority: 10,
      evictionPolicy: 'lru',
      compression: false
    },
    {
      name: 'memory_compressed',
      capacity: 500,
      ttl: 300000, // 5 minutes
      priority: 8,
      evictionPolicy: 'som_guided',
      compression: true
    },
    {
      name: 'lokijs_persistent',
      capacity: 2000,
      ttl: 1800000, // 30 minutes
      priority: 6,
      evictionPolicy: 'rl',
      compression: true
    },
    {
      name: 'redis_shared',
      capacity: 10000,
      ttl: 7200000, // 2 hours
      priority: 4,
      evictionPolicy: 'lfu',
      compression: true
    },
    {
      name: 'disk_archive',
      capacity: 100000,
      ttl: 86400000, // 24 hours
      priority: 2,
      evictionPolicy: 'lru',
      compression: true
    }
  ];

  // Performance metrics
  private metrics = {
    totalQueries: 0,
    cacheHits: 0,
    cacheMisses: 0,
    compressionSavings: 0,
    generationTime: 0,
    retrieval_time: 0,
    gpu_operations: 0,
    som_operations: 0,
    autoencoder_operations: 0
  };

  constructor() {
    this.initializeServices();
  }

  private async initializeServices(): Promise<void> {
    try {
      // Initialize SOM neural network
      this.som = new SOMNeuralNetwork({
        gridSize: { width: 8, height: 8 },
        learningRate: 0.1,
        neighborhoodRadius: 2.0,
        epochs: 50,
        enableGPU: true,
        inputDimension: 256
      });
      await this.som.initialize();

      // Initialize auto-encoder
      this.autoencoder = new GraphPatternAutoEncoder({
        inputDimension: 512,
        hiddenLayers: [256, 128, 64, 32],
        learningRate: 0.001,
        batchSize: 16,
        epochs: 50,
        enableGPU: true,
        compressionTarget: 0.1
      });
      await this.autoencoder.initialize();

      // Initialize multi-layer cache
      this.multiLayerCache = new MultiLayerCache({
        enableRedisCache: true,
        enableLokiCache: true,
        enableMemoryCache: true,
        memoryTTL: 300, // 5 minutes
        lokiTTL: 1800, // 30 minutes
        redisTTL: 7200 // 2 hours
      });
      await this.multiLayerCache.initialize();

      // Initialize RL cache
      await this.rlCache.initialize();

      // Initialize dimensional indices
      this.initializeDimensionalIndices();

      console.log('🚀 Multi-Dimensional Image Cache initialized with SOM, Auto-Encoder, and GPU integration');
    } catch (error) {
      console.error('Failed to initialize Multi-Dimensional Image Cache:', error);
      throw error;
    }
  }

  private initializeDimensionalIndices(): void {
    // Create indices for each dimension
    const dimensions = ['temporal', 'spatial', 'semantic', 'visual', 'algorithm'];
    
    dimensions.forEach(dim => {
      this.dimensionalIndices.set(dim, new Set());
    });

    console.log('📊 Dimensional indices initialized:', dimensions);
  }

  async storeImage(
    imageData: string,
    dimensions: CacheDimensions,
    graphData: any,
    processingMetrics: any
  ): Promise<string> {
    const startTime = performance.now();
    
    try {
      // Generate cache key from dimensions and graph signature
      const cacheKey = this.generateMultiDimensionalKey(dimensions, graphData);
      const graphSignature = this.generateGraphSignature(graphData);

      // Create metadata
      const metadata: ImageMetadata = {
        graphSignature,
        nodeCount: graphData.nodes?.length || 0,
        edgeCount: graphData.edges?.length || 0,
        processingTime: processingMetrics.processingTime || 0,
        compressionRatio: 0,
        qualityScore: processingMetrics.qualityScore || 0.8,
        legalDomain: graphData.metadata?.legalDomain || 'general',
        patterns: {
          citationDensity: this.calculateCitationDensity(graphData),
          jurisdictionalSpread: this.calculateJurisdictionalSpread(graphData),
          temporalRange: this.calculateTemporalRange(graphData),
          complexityIndex: this.calculateComplexityIndex(graphData)
        }
      };

      // Compress image data using auto-encoder and SOM
      const compressionData = await this.compressImageData(imageData, graphData);
      metadata.compressionRatio = compressionData.compressionRatio;

      // Create GPU texture if applicable
      let gpuTexture: GPUTextureMatrix | undefined;
      if (dimensions.visual === 'hires' || dimensions.algorithm === 'som') {
        gpuTexture = await this.createGPUTexture(cacheKey, imageData, {
          width: parseInt(imageData.split(',')[0].split(';')[1]?.split('=')[1]) || 800,
          height: 600
        });
      }

      // Create cache entry
      const entry: ImageCacheEntry = {
        id: cacheKey,
        algorithm: dimensions.algorithm || 'dfs',
        imageData,
        dimensions: {
          width: parseInt(imageData.split(',')[0].split(';')[1]?.split('=')[1]) || 800,
          height: 600
        },
        metadata,
        compressionData,
        gpuTexture,
        cacheStats: {
          hitCount: 0,
          missCount: 0,
          size: imageData.length,
          lastAccessed: Date.now(),
          generationCost: processingMetrics.processingTime || 100,
          compressionEfficiency: compressionData.compressionRatio
        },
        timestamp: Date.now()
      };

      // Store in appropriate cache layers
      await this.storeInLayers(entry, dimensions);

      // Update dimensional indices
      this.updateDimensionalIndices(cacheKey, dimensions);

      // Store in RL cache for learning
      this.rlCache.set(cacheKey, entry);

      // Update metrics
      this.metrics.compressionSavings += (1 - compressionData.compressionRatio) * imageData.length;
      this.metrics.generationTime += performance.now() - startTime;

      console.log(`💾 Image stored: ${cacheKey}, ${compressionData.compressionRatio.toFixed(2)} compression, ${(performance.now() - startTime).toFixed(2)}ms`);
      
      return cacheKey;
    } catch (error) {
      console.error('Failed to store image:', error);
      throw error;
    }
  }

  async queryImage(query: MultiDimensionalQuery): Promise<ImageCacheEntry | null> {
    const startTime = performance.now();
    this.metrics.totalQueries++;

    try {
      // Generate primary cache key
      const primaryKey = this.generateMultiDimensionalKey(query.dimensions as CacheDimensions, { signature: query.graphSignature });

      // Try exact match first
      let entry = await this.getFromLayers(primaryKey);
      
      if (entry) {
        entry.cacheStats.hitCount++;
        entry.cacheStats.lastAccessed = Date.now();
        this.metrics.cacheHits++;
        this.metrics.retrieval_time += performance.now() - startTime;
        
        console.log(`✅ Cache hit: ${primaryKey}`);
        return entry;
      }

      // Try similarity-based matching using SOM
      entry = await this.findSimilarImage(query);
      
      if (entry) {
        entry.cacheStats.hitCount++;
        entry.cacheStats.lastAccessed = Date.now();
        this.metrics.cacheHits++;
        this.metrics.som_operations++;
        console.log(`🧠 SOM-guided hit: ${entry.id}`);
        return entry;
      }

      // Try pattern-based matching using auto-encoder
      entry = await this.findPatternMatchingImage(query);
      
      if (entry) {
        entry.cacheStats.hitCount++;
        entry.cacheStats.lastAccessed = Date.now();
        this.metrics.cacheHits++;
        this.metrics.autoencoder_operations++;
        console.log(`🤖 Auto-encoder guided hit: ${entry.id}`);
        return entry;
      }

      // No match found
      this.metrics.cacheMisses++;
      this.metrics.retrieval_time += performance.now() - startTime;
      
      console.log(`❌ Cache miss for query: ${JSON.stringify(query.dimensions)}`);
      return null;
    } catch (error) {
      console.error('Failed to query image:', error);
      this.metrics.cacheMisses++;
      return null;
    }
  }

  private async compressImageData(imageData: string, graphData: any): Promise<{
    original: EncodedGraphPattern;
    som: SOMDecomposition;
    compressed: ArrayBuffer;
    compressionRatio: number;
  }> {
    try {
      // Convert graph data to format suitable for neural networks
      const graphForProcessing = {
        nodes: graphData.nodes || [],
        edges: graphData.edges || [],
        metadata: {
          totalNodes: graphData.nodes?.length || 0,
          totalEdges: graphData.edges?.length || 0,
          density: graphData.metadata?.density || 0,
          averageDegree: graphData.metadata?.averageDegree || 0,
          legalDomain: graphData.metadata?.legalDomain || 'general',
          timestamp: Date.now()
        }
      };

      // Encode with auto-encoder
      const original = await this.autoencoder.encodeGraphPattern(graphForProcessing);

      // Analyze with SOM
      const somInput = Array.from(original.encodedFeatures).slice(0, 256);
      const som = await this.som.train([somInput]);

      // Compress image data using FlatBuffer via NES-GPU bridge
      const mockDocument = {
        id: `image_${Date.now()}`,
        type: 'evidence' as const,
        priority: 128,
        size: imageData.length,
        confidenceLevel: 0.8,
        riskLevel: 'medium' as const,
        lastAccessed: Date.now(),
        compressed: true,
        metadata: {
          imageData,
          originalSize: imageData.length,
          vectorEmbedding: original.encodedFeatures
        }
      };

      const compressed = await nesGPUBridge.createFlatBufferFromDocument(mockDocument);
      const compressionRatio = compressed.byteLength / imageData.length;

      return {
        original,
        som,
        compressed,
        compressionRatio
      };
    } catch (error) {
      console.error('Failed to compress image data:', error);
      // Return fallback compression
      return {
        original: {} as EncodedGraphPattern,
        som: {} as SOMDecomposition,
        compressed: new ArrayBuffer(imageData.length),
        compressionRatio: 1.0
      };
    }
  }

  private async createGPUTexture(
    cacheKey: string,
    imageData: string,
    dimensions: { width: number; height: number }
  ): Promise<GPUTextureMatrix | null> {
    try {
      // Convert base64 image to float array for GPU texture
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      canvas.width = dimensions.width;
      canvas.height = dimensions.height;

      const img = new Image();
      img.src = imageData;
      
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      ctx.drawImage(img, 0, 0, dimensions.width, dimensions.height);
      const imageDataArray = ctx.getImageData(0, 0, dimensions.width, dimensions.height);
      
      // Convert to float32 array for GPU
      const floatData = new Float32Array(dimensions.width * dimensions.height);
      for (let i = 0; i < floatData.length; i++) {
        // Convert RGBA to grayscale
        const r = imageDataArray.data[i * 4];
        const g = imageDataArray.data[i * 4 + 1];
        const b = imageDataArray.data[i * 4 + 2];
        floatData[i] = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0;
      }

      const gpuTexture = await nesGPUBridge.createRankingTexture(cacheKey, floatData, dimensions);
      
      if (gpuTexture) {
        this.gpuTextures.set(cacheKey, gpuTexture);
        this.metrics.gpu_operations++;
      }

      return gpuTexture;
    } catch (error) {
      console.error('Failed to create GPU texture:', error);
      return null;
    }
  }

  private async storeInLayers(entry: ImageCacheEntry, dimensions: CacheDimensions): Promise<void> {
    // Determine appropriate layers based on dimensions and usage patterns
    const layersToUse = this.selectOptimalLayers(entry, dimensions);

    for (const layer of layersToUse) {
      try {
        const layerKey = `${layer.name}_${entry.id}`;
        
        if (layer.compression && entry.compressionData) {
          // Store compressed version
          if (layer.name === 'redis_shared') {
            await this.multiLayerCache.set('visualization', layerKey, entry.compressionData.compressed, layer.ttl / 1000);
          } else if (layer.name === 'lokijs_persistent') {
            await this.multiLayerCache.set('visualization', layerKey, entry, layer.ttl / 1000);
          }
        } else {
          // Store original version
          if (layer.name === 'memory_compressed') {
            this.imageEntries.set(entry.id, entry);
          }
        }
      } catch (error) {
        console.error(`Failed to store in layer ${layer.name}:`, error);
      }
    }
  }

  private async getFromLayers(cacheKey: string): Promise<ImageCacheEntry | null> {
    // Try layers in priority order
    const sortedLayers = [...this.cacheLayers].sort((a, b) => b.priority - a.priority);

    for (const layer of sortedLayers) {
      try {
        const layerKey = `${layer.name}_${cacheKey}`;
        
        if (layer.name === 'memory_compressed') {
          const entry = this.imageEntries.get(cacheKey);
          if (entry) return entry;
        } else if (layer.name === 'redis_shared' || layer.name === 'lokijs_persistent') {
          const data = await this.multiLayerCache.get('visualization', layerKey);
          if (data) {
            // Reconstruct entry from compressed data if needed
            if (data instanceof ArrayBuffer) {
              return await this.reconstructFromCompressed(data, cacheKey);
            } else if (data && typeof data === 'object') {
              return data as ImageCacheEntry;
            }
          }
        }
      } catch (error) {
        console.error(`Failed to retrieve from layer ${layer.name}:`, error);
      }
    }

    return null;
  }

  private async findSimilarImage(query: MultiDimensionalQuery): Promise<ImageCacheEntry | null> {
    try {
      // Use SOM to find similar cached images
      const somDecomposition = await this.som.getDecomposition();
      if (!somDecomposition) return null;

      // Find entries with similar patterns
      const candidates: { entry: ImageCacheEntry; similarity: number }[] = [];

      for (const entry of this.imageEntries.values()) {
        if (entry.compressionData?.som) {
          const similarity = this.calculateSOMSimilarity(somDecomposition, entry.compressionData.som);
          if (similarity > 0.7) {
            candidates.push({ entry, similarity });
          }
        }
      }

      // Sort by similarity and quality
      candidates.sort((a, b) => {
        const scoreA = a.similarity * a.entry.metadata.qualityScore;
        const scoreB = b.similarity * b.entry.metadata.qualityScore;
        return scoreB - scoreA;
      });

      return candidates.length > 0 ? candidates[0].entry : null;
    } catch (error) {
      console.error('SOM similarity search failed:', error);
      return null;
    }
  }

  private async findPatternMatchingImage(query: MultiDimensionalQuery): Promise<ImageCacheEntry | null> {
    try {
      // Use auto-encoder to find pattern-matching images
      const candidates: { entry: ImageCacheEntry; patternMatch: number }[] = [];

      for (const entry of this.imageEntries.values()) {
        if (entry.compressionData?.original) {
          const patternMatch = this.calculatePatternSimilarity(
            query.graphSignature,
            entry.compressionData.original
          );
          
          if (patternMatch > 0.6) {
            candidates.push({ entry, patternMatch });
          }
        }
      }

      // Sort by pattern match and recency
      candidates.sort((a, b) => {
        const scoreA = a.patternMatch * (1 - (Date.now() - a.entry.timestamp) / 3600000);
        const scoreB = b.patternMatch * (1 - (Date.now() - b.entry.timestamp) / 3600000);
        return scoreB - scoreA;
      });

      return candidates.length > 0 ? candidates[0].entry : null;
    } catch (error) {
      console.error('Pattern matching search failed:', error);
      return null;
    }
  }

  private selectOptimalLayers(entry: ImageCacheEntry, dimensions: CacheDimensions): CacheLayer[] {
    const selectedLayers: CacheLayer[] = [];

    // GPU texture layer for high-quality or SOM-based visualizations
    if (dimensions.visual === 'hires' || dimensions.algorithm === 'som') {
      selectedLayers.push(this.cacheLayers.find(l => l.name === 'gpu_texture')!);
    }

    // Memory layer for frequently accessed images
    if (dimensions.temporal === 'recent' || entry.metadata.patterns.complexityIndex > 0.7) {
      selectedLayers.push(this.cacheLayers.find(l => l.name === 'memory_compressed')!);
    }

    // LokiJS for persistent client-side caching
    selectedLayers.push(this.cacheLayers.find(l => l.name === 'lokijs_persistent')!);

    // Redis for shared caching
    if (dimensions.spatial === 'global' || entry.metadata.nodeCount > 100) {
      selectedLayers.push(this.cacheLayers.find(l => l.name === 'redis_shared')!);
    }

    // Disk archive for long-term storage
    if (entry.metadata.patterns.citationDensity > 0.8) {
      selectedLayers.push(this.cacheLayers.find(l => l.name === 'disk_archive')!);
    }

    return selectedLayers;
  }

  private async reconstructFromCompressed(data: ArrayBuffer, cacheKey: string): Promise<ImageCacheEntry | null> {
    try {
      // Parse FlatBuffer data back to document
      const document = nesGPUBridge.parseFlatBufferToDocument(data);
      if (!document?.metadata?.imageData) return null;

      // Reconstruct entry
      const entry: ImageCacheEntry = {
        id: cacheKey,
        algorithm: 'dfs', // Default
        imageData: document.metadata.imageData,
        dimensions: { width: 800, height: 600 }, // Default
        metadata: {
          graphSignature: cacheKey,
          nodeCount: 0,
          edgeCount: 0,
          processingTime: 0,
          compressionRatio: data.byteLength / document.metadata.imageData.length,
          qualityScore: 0.8,
          legalDomain: 'general',
          patterns: {
            citationDensity: 0,
            jurisdictionalSpread: 0,
            temporalRange: 0,
            complexityIndex: 0
          }
        },
        cacheStats: {
          hitCount: 0,
          missCount: 0,
          size: document.metadata.imageData.length,
          lastAccessed: Date.now(),
          generationCost: 100,
          compressionEfficiency: data.byteLength / document.metadata.imageData.length
        },
        timestamp: document.lastAccessed || Date.now()
      };

      return entry;
    } catch (error) {
      console.error('Failed to reconstruct from compressed data:', error);
      return null;
    }
  }

  // Helper methods
  private generateMultiDimensionalKey(dimensions: CacheDimensions, graphData: any): string {
    const parts = [
      dimensions.temporal || 'recent',
      dimensions.spatial || 'local',
      dimensions.semantic || 'simple',
      dimensions.visual || 'standard',
      dimensions.algorithm || 'dfs',
      graphData.signature || this.generateGraphSignature(graphData)
    ];
    
    return `md_${parts.join('_')}`.replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 64);
  }

  private generateGraphSignature(graphData: any): string {
    if (graphData.signature) return graphData.signature;
    
    const nodeCount = graphData.nodes?.length || 0;
    const edgeCount = graphData.edges?.length || 0;
    const domain = graphData.metadata?.legalDomain || 'general';
    
    return `${nodeCount}_${edgeCount}_${domain}_${Date.now()}`.substring(0, 32);
  }

  private updateDimensionalIndices(cacheKey: string, dimensions: CacheDimensions): void {
    Object.entries(dimensions).forEach(([dim, value]) => {
      if (value) {
        const dimSet = this.dimensionalIndices.get(dim);
        if (dimSet) {
          dimSet.add(`${value}:${cacheKey}`);
        }
      }
    });
  }

  private calculateCitationDensity(graphData: any): number {
    const citationEdges = graphData.edges?.filter((e: any) => e.type === 'cites').length || 0;
    const totalEdges = graphData.edges?.length || 1;
    return citationEdges / totalEdges;
  }

  private calculateJurisdictionalSpread(graphData: any): number {
    const jurisdictions = new Set(
      graphData.nodes?.map((n: any) => n.metadata?.jurisdiction).filter(Boolean) || []
    );
    return Math.min(jurisdictions.size / 10, 1.0); // Normalize to 0-1
  }

  private calculateTemporalRange(graphData: any): number {
    const timestamps = graphData.nodes?.map((n: any) => n.metadata?.timestamp).filter(Boolean) || [];
    if (timestamps.length < 2) return 0;
    
    const range = Math.max(...timestamps) - Math.min(...timestamps);
    return Math.min(range / (365 * 24 * 60 * 60 * 1000), 1.0); // Normalize to years
  }

  private calculateComplexityIndex(graphData: any): number {
    const nodeCount = graphData.nodes?.length || 0;
    const edgeCount = graphData.edges?.length || 0;
    const nodeTypes = new Set(graphData.nodes?.map((n: any) => n.type) || []).size;
    
    return Math.min((edgeCount / nodeCount + nodeTypes / 6) / 2, 1.0);
  }

  private calculateSOMSimilarity(som1: SOMDecomposition, som2: SOMDecomposition): number {
    // Simple similarity calculation between SOM decompositions
    if (!som1.convergenceHistory || !som2.convergenceHistory) return 0;
    
    const len = Math.min(som1.convergenceHistory.length, som2.convergenceHistory.length);
    if (len === 0) return 0;
    
    let similarity = 0;
    for (let i = 0; i < len; i++) {
      const diff = Math.abs(som1.convergenceHistory[i] - som2.convergenceHistory[i]);
      similarity += 1 / (1 + diff); // Convert difference to similarity
    }
    
    return similarity / len;
  }

  private calculatePatternSimilarity(signature: string, pattern: EncodedGraphPattern): number {
    // Simple pattern similarity based on signature matching
    const signatureScore = signature === pattern.patternSignature ? 1.0 : 0.5;
    const compressionScore = 1 - pattern.reconstructionError;
    
    return (signatureScore + compressionScore) / 2;
  }

  // Public API methods
  getMetrics() {
    return {
      ...this.metrics,
      hitRatio: this.metrics.totalQueries > 0 ? this.metrics.cacheHits / this.metrics.totalQueries : 0,
      avgRetrievalTime: this.metrics.totalQueries > 0 ? this.metrics.retrieval_time / this.metrics.totalQueries : 0,
      compressionEfficiency: this.metrics.compressionSavings,
      cacheSize: this.imageEntries.size,
      gpuTextureCount: this.gpuTextures.size,
      layerStats: this.cacheLayers.map(layer => ({
        name: layer.name,
        capacity: layer.capacity,
        priority: layer.priority,
        ttl: layer.ttl
      }))
    };
  }

  async evictExpired(): Promise<number> {
    let evicted = 0;
    const now = Date.now();

    for (const [key, entry] of this.imageEntries.entries()) {
      const age = now - entry.timestamp;
      if (age > 24 * 60 * 60 * 1000) { // 24 hours
        this.imageEntries.delete(key);
        this.gpuTextures.delete(key);
        evicted++;
      }
    }

    console.log(`🧹 Evicted ${evicted} expired entries`);
    return evicted;
  }

  async cleanup(): Promise<void> {
    // Clean up GPU resources
    for (const texture of this.gpuTextures.values()) {
      try {
        texture.texture?.destroy();
        texture.gpuBuffer?.destroy();
      } catch (error) {
        console.error('Error cleaning up GPU texture:', error);
      }
    }

    // Clean up neural networks
    this.som?.cleanup();
    this.autoencoder?.cleanup();

    // Clean up caches
    this.multiLayerCache?.cleanup();

    // Clear maps
    this.imageEntries.clear();
    this.gpuTextures.clear();
    this.dimensionalIndices.clear();

    console.log('🧹 Multi-Dimensional Image Cache cleaned up');
  }
}

// Export singleton instance
export const multiDimensionalImageCache = new MultiDimensionalImageCache();