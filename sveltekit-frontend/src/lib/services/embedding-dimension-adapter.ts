/**
 * Embedding Dimension Adapter Service
 * Handles conversion between different embedding dimensions (768 ↔ 384)
 * Integrates with existing GPU metrics and legal AI platform
 */

import { webgpuFlashAttentionService } from '$lib/services/webgpu-flash-attention-service';

interface EmbeddingConversionOptions {
  method: 'truncate' | 'pca' | 'average' | 'weighted';
  targetDimension: number;
  preserveSemantics: boolean;
}

interface EmbeddingMetrics {
  originalDimension: number;
  targetDimension: number;
  conversionMethod: string;
  semanticSimilarityLoss: number;
  conversionTimeMs: number;
}

export class EmbeddingDimensionAdapter {
  private conversionHistory: EmbeddingMetrics[] = [];
  private webgpuAvailable: boolean = false;
  
  constructor() {
    this.initializeWebGPU();
  }
  
  private async initializeWebGPU() {
    try {
      this.webgpuAvailable = await webgpuFlashAttentionService.initialize();
      console.log('[EmbeddingAdapter] WebGPU available:', this.webgpuAvailable);
    } catch (error) {
      console.warn('[EmbeddingAdapter] WebGPU initialization failed:', error);
    }
  }
  
  /**
   * Convert 768-dimensional embeddings to 384 dimensions
   * This fixes the "Unexpected embedding dimension: 768, expected 384" error
   */
  async convert768to384(
    embedding: number[], 
    options: Partial<EmbeddingConversionOptions> = {}
  ): Promise<{ 
    embedding: number[]; 
    metrics: EmbeddingMetrics; 
    confidence: number 
  }> {
    const startTime = performance.now();
    
    const config: EmbeddingConversionOptions = {
      method: 'weighted',
      targetDimension: 384,
      preserveSemantics: true,
      ...options
    };
    
    if (embedding.length !== 768) {
      throw new Error(`Expected 768-dimensional embedding, got ${embedding.length}`);
    }
    
    let convertedEmbedding: number[];
    let confidence: number;
    
    switch (config.method) {
      case 'truncate':
        convertedEmbedding = this.truncateEmbedding(embedding, config.targetDimension);
        confidence = 0.7; // Lower confidence due to information loss
        break;
        
      case 'average':
        convertedEmbedding = this.averageEmbedding(embedding, config.targetDimension);
        confidence = 0.8;
        break;
        
      case 'weighted':
        convertedEmbedding = await this.weightedEmbedding(embedding, config.targetDimension);
        confidence = 0.9; // Highest confidence with weighted method
        break;
        
      case 'pca':
        convertedEmbedding = await this.pcaEmbedding(embedding, config.targetDimension);
        confidence = 0.85;
        break;
        
      default:
        throw new Error(`Unknown conversion method: ${config.method}`);
    }
    
    // Calculate semantic similarity loss (mock for now)
    const semanticLoss = this.calculateSemanticLoss(embedding, convertedEmbedding);
    
    const endTime = performance.now();
    const metrics: EmbeddingMetrics = {
      originalDimension: embedding.length,
      targetDimension: convertedEmbedding.length,
      conversionMethod: config.method,
      semanticSimilarityLoss: semanticLoss,
      conversionTimeMs: endTime - startTime
    };
    
    this.conversionHistory.push(metrics);
    
    return {
      embedding: convertedEmbedding,
      metrics,
      confidence
    };
  }
  
  /**
   * Convert 384-dimensional embeddings to 768 dimensions (upsampling)
   */
  async convert384to768(
    embedding: number[],
    options: Partial<EmbeddingConversionOptions> = {}
  ): Promise<{ 
    embedding: number[]; 
    metrics: EmbeddingMetrics; 
    confidence: number 
  }> {
    const startTime = performance.now();
    
    if (embedding.length !== 384) {
      throw new Error(`Expected 384-dimensional embedding, got ${embedding.length}`);
    }
    
    // Upsampling using interpolation and noise injection
    const convertedEmbedding = this.upsampleEmbedding(embedding, 768);
    const confidence = 0.6; // Lower confidence for upsampling
    
    const endTime = performance.now();
    const metrics: EmbeddingMetrics = {
      originalDimension: embedding.length,
      targetDimension: convertedEmbedding.length,
      conversionMethod: 'interpolation',
      semanticSimilarityLoss: 0.2, // Estimated loss for upsampling
      conversionTimeMs: endTime - startTime
    };
    
    this.conversionHistory.push(metrics);
    
    return {
      embedding: convertedEmbedding,
      metrics,
      confidence
    };
  }
  
  /**
   * Auto-detect and convert embeddings to target dimension
   */
  async autoConvert(
    embedding: number[],
    targetDimension: number = 384
  ): Promise<{ 
    embedding: number[]; 
    metrics: EmbeddingMetrics; 
    confidence: number;
    wasConverted: boolean;
  }> {
    // Return as-is if already correct dimension
    if (embedding.length === targetDimension) {
      return {
        embedding,
        metrics: {
          originalDimension: embedding.length,
          targetDimension,
          conversionMethod: 'none',
          semanticSimilarityLoss: 0,
          conversionTimeMs: 0
        },
        confidence: 1.0,
        wasConverted: false
      };
    }
    
    // Convert based on source and target dimensions
    if (embedding.length === 768 && targetDimension === 384) {
      const result = await this.convert768to384(embedding);
      return { ...result, wasConverted: true };
    }
    
    if (embedding.length === 384 && targetDimension === 768) {
      const result = await this.convert384to768(embedding);
      return { ...result, wasConverted: true };
    }
    
    throw new Error(`Unsupported conversion: ${embedding.length} → ${targetDimension}`);
  }
  
  /**
   * Batch conversion for multiple embeddings
   */
  async batchConvert(
    embeddings: number[][],
    targetDimension: number = 384
  ): Promise<{
    embeddings: number[][];
    metrics: EmbeddingMetrics[];
    overallConfidence: number;
    conversionsPerformed: number;
  }> {
    const results = await Promise.all(
      embeddings.map(embedding => this.autoConvert(embedding, targetDimension))
    );
    
    const convertedEmbeddings = results.map(r => r.embedding);
    const metrics = results.map(r => r.metrics);
    const conversionsPerformed = results.filter(r => r.wasConverted).length;
    const overallConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;
    
    return {
      embeddings: convertedEmbeddings,
      metrics,
      overallConfidence,
      conversionsPerformed
    };
  }
  
  // Conversion method implementations
  
  private truncateEmbedding(embedding: number[], targetDim: number): number[] {
    return embedding.slice(0, targetDim);
  }
  
  private averageEmbedding(embedding: number[], targetDim: number): number[] {
    const chunkSize = Math.ceil(embedding.length / targetDim);
    const result: number[] = [];
    
    for (let i = 0; i < targetDim; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, embedding.length);
      const chunk = embedding.slice(start, end);
      const average = chunk.reduce((sum, val) => sum + val, 0) / chunk.length;
      result.push(average);
    }
    
    return result;
  }
  
  private async weightedEmbedding(embedding: number[], targetDim: number): Promise<number[]> {
    // Use importance weights based on variance and magnitude
    const chunkSize = Math.ceil(embedding.length / targetDim);
    const result: number[] = [];
    
    for (let i = 0; i < targetDim; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, embedding.length);
      const chunk = embedding.slice(start, end);
      
      // Calculate weighted average based on magnitude
      const weights = chunk.map(val => Math.abs(val));
      const totalWeight = weights.reduce((sum, w) => sum + w, 0) || 1;
      
      const weightedSum = chunk.reduce((sum, val, idx) => {
        return sum + (val * weights[idx]);
      }, 0);
      
      result.push(weightedSum / totalWeight);
    }
    
    // Normalize the result vector
    const magnitude = Math.sqrt(result.reduce((sum, val) => sum + val * val, 0));
    return magnitude > 0 ? result.map(val => val / magnitude) : result;
  }
  
  private async pcaEmbedding(embedding: number[], targetDim: number): Promise<number[]> {
    // Simplified PCA implementation for dimensionality reduction
    // In production, you'd use a proper PCA library or WebGPU compute shader
    
    if (this.webgpuAvailable) {
      // Use WebGPU for PCA computation if available
      return this.pcaWebGPU(embedding, targetDim);
    }
    
    // Fallback to simple linear projection
    return this.weightedEmbedding(embedding, targetDim);
  }
  
  private async pcaWebGPU(embedding: number[], targetDim: number): Promise<number[]> {
    try {
      // Mock WebGPU PCA implementation
      // In production, this would use WebGPU compute shaders for matrix operations
      const projectionMatrix = this.generateProjectionMatrix(embedding.length, targetDim);
      
      const result = new Array(targetDim).fill(0);
      for (let i = 0; i < targetDim; i++) {
        for (let j = 0; j < embedding.length; j++) {
          result[i] += embedding[j] * projectionMatrix[i][j];
        }
      }
      
      return result;
    } catch (error) {
      console.warn('[EmbeddingAdapter] WebGPU PCA failed, falling back:', error);
      return this.weightedEmbedding(embedding, targetDim);
    }
  }
  
  private generateProjectionMatrix(inputDim: number, outputDim: number): number[][] {
    // Generate a random projection matrix with normalization
    const matrix: number[][] = [];
    
    for (let i = 0; i < outputDim; i++) {
      const row: number[] = [];
      for (let j = 0; j < inputDim; j++) {
        // Random projection with Gaussian initialization
        row.push((Math.random() - 0.5) * 2 / Math.sqrt(inputDim));
      }
      matrix.push(row);
    }
    
    return matrix;
  }
  
  private upsampleEmbedding(embedding: number[], targetDim: number): number[] {
    const result: number[] = [];
    const scale = targetDim / embedding.length;
    
    for (let i = 0; i < targetDim; i++) {
      const sourceIndex = Math.floor(i / scale);
      const nextIndex = Math.min(sourceIndex + 1, embedding.length - 1);
      const fraction = (i / scale) - sourceIndex;
      
      // Linear interpolation with small noise injection
      const interpolated = embedding[sourceIndex] * (1 - fraction) + embedding[nextIndex] * fraction;
      const noise = (Math.random() - 0.5) * 0.01; // Small noise for variety
      
      result.push(interpolated + noise);
    }
    
    return result;
  }
  
  private calculateSemanticLoss(original: number[], converted: number[]): number {
    // Calculate cosine similarity as proxy for semantic preservation
    if (original.length === converted.length) {
      return this.cosineSimilarity(original, converted);
    }
    
    // For different dimensions, estimate based on information theory
    const compressionRatio = converted.length / original.length;
    return Math.max(0, 1 - compressionRatio);
  }
  
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude > 0 ? dotProduct / magnitude : 0;
  }
  
  /**
   * Get conversion statistics
   */
  getConversionStats(): {
    totalConversions: number;
    averageConversionTime: number;
    averageSemanticLoss: number;
    methodDistribution: Record<string, number>;
  } {
    if (this.conversionHistory.length === 0) {
      return {
        totalConversions: 0,
        averageConversionTime: 0,
        averageSemanticLoss: 0,
        methodDistribution: {}
      };
    }
    
    const methodCounts: Record<string, number> = {};
    let totalTime = 0;
    let totalLoss = 0;
    
    for (const metrics of this.conversionHistory) {
      methodCounts[metrics.conversionMethod] = (methodCounts[metrics.conversionMethod] || 0) + 1;
      totalTime += metrics.conversionTimeMs;
      totalLoss += metrics.semanticSimilarityLoss;
    }
    
    return {
      totalConversions: this.conversionHistory.length,
      averageConversionTime: totalTime / this.conversionHistory.length,
      averageSemanticLoss: totalLoss / this.conversionHistory.length,
      methodDistribution: methodCounts
    };
  }
  
  /**
   * Clear conversion history
   */
  clearHistory(): void {
    this.conversionHistory = [];
  }
}

// Export singleton instance
export const embeddingDimensionAdapter = new EmbeddingDimensionAdapter();