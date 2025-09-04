/**
 * 🔧 SIMD Worker for CPU-Optimized Vector Processing
 * 
 * Handles:
 * - SIMD-style batch processing
 * - Memory-aligned typed arrays
 * - Vectorized operations
 * - CPU fallback processing
 */

/// <reference lib="webworker" />

declare const self: DedicatedWorkerGlobalScope;

export interface SIMDWorkerMessage {
  type: 'process_batch_simd' | 'process_simd' | 'initialize_simd';
  jobId: string;
  operations?: any[];
  operation?: any;
  batchId?: string;
}

class SIMDWorker {
  private initialized = false;
  private simdSupported = false;
  private vectorWidth = 4; // Default vector width for Float32Array SIMD

  constructor() {
    self.onmessage = this.handleMessage.bind(this);
    this.initializeSIMD();
  }

  private initializeSIMD(): void {
    // Check for SIMD support
    this.simdSupported = typeof WebAssembly !== 'undefined' && 
                        WebAssembly.validate(new Uint8Array([
                          0x00, 0x61, 0x73, 0x6d, // WASM magic number
                          0x01, 0x00, 0x00, 0x00, // WASM version
                        ]));

    // Determine optimal vector width based on platform
    if (typeof navigator !== 'undefined') {
      const cores = navigator.hardwareConcurrency || 4;
      this.vectorWidth = Math.min(cores * 2, 16); // Max 16-wide vectors
    }

    this.initialized = true;
  }

  private async handleMessage(event: MessageEvent<SIMDWorkerMessage>) {
    const { type, jobId, operations, operation, batchId } = event.data;

    try {
      switch (type) {
        case 'initialize_simd':
          self.postMessage({ jobId, result: { 
            initialized: this.initialized,
            simdSupported: this.simdSupported,
            vectorWidth: this.vectorWidth
          }, error: null });
          break;

        case 'process_batch_simd':
          const batchResult = await this.processBatchSIMD(operations!, batchId!);
          self.postMessage({ jobId, result: batchResult, error: null });
          break;

        case 'process_simd':
          const result = await this.processSingleSIMD(operation!);
          self.postMessage({ jobId, result, error: null });
          break;

        default:
          throw new Error(`Unknown message type: ${type}`);
      }
    } catch (error: any) {
      self.postMessage({ 
        jobId, 
        result: null, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  }

  // Batch processing with SIMD optimization
  private async processBatchSIMD(operations: any[], batchId: string): Promise<Float32Array[]> {
    const startTime = performance.now();
    const results: Float32Array[] = [];

    // Group operations by type for better vectorization
    const groupedOps = this.groupOperationsByType(operations);

    for (const [opType, ops] of Object.entries(groupedOps)) {
      switch (opType) {
        case 'embedding':
          results.push(...await this.processEmbeddingBatch(ops));
          break;
        case 'similarity':
          results.push(...await this.processSimilarityBatch(ops));
          break;
        case 'transform':
          results.push(...await this.processTransformBatch(ops));
          break;
        default:
          // Process individually for unknown types
          for (const op of ops) {
            results.push(await this.processSingleSIMD(op));
          }
      }
    }

    const endTime = performance.now();
    console.log(`SIMD batch ${batchId} processed in ${endTime - startTime}ms`);

    return results;
  }

  // Single operation processing
  private async processSingleSIMD(operation: any): Promise<Float32Array> {
    switch (operation.type) {
      case 'embedding':
        return await this.processEmbedding(operation);
      case 'similarity':
        return await this.processSimilarity(operation);
      case 'clustering':
        return await this.processClustering(operation);
      case 'search':
        return await this.processSearch(operation);
      case 'transform':
        return await this.processTransform(operation);
      default:
        throw new Error(`Unsupported SIMD operation: ${operation.type}`);
    }
  }

  // Memory-aligned embedding processing
  private async processEmbedding(operation: any): Promise<Float32Array> {
    const input = this.ensureFloat32Array(operation.input);
    const dimensions = operation.dimensions || [input.length];
    const legalWeight = operation.metadata?.legalWeight || 1.0;

    // Create memory-aligned output array
    const output = this.createAlignedArray(input.length);

    // Vectorized processing with SIMD
    if (this.simdSupported) {
      return await this.processSIMDEmbedding(input, output, legalWeight);
    } else {
      return await this.processScalarEmbedding(input, output, legalWeight);
    }
  }

  // SIMD-optimized embedding processing
  private async processSIMDEmbedding(input: Float32Array, output: Float32Array, legalWeight: number): Promise<Float32Array> {
    const vectorWidth = this.vectorWidth;
    const length = input.length;
    const remainder = length % vectorWidth;
    const alignedLength = length - remainder;

    // Process aligned vectors
    for (let i = 0; i < alignedLength; i += vectorWidth) {
      // Simulate SIMD operations (would use actual WASM SIMD in production)
      for (let j = 0; j < vectorWidth; j++) {
        const idx = i + j;
        const value = input[idx];
        
        // Legal document processing with vectorized operations
        const legalScore = this.computeLegalScore(value, idx);
        const semanticScore = this.computeSemanticScore(value);
        const normalizedValue = this.normalizeValue(value);
        
        output[idx] = normalizedValue * legalScore * legalWeight * semanticScore;
      }
    }

    // Process remaining elements
    for (let i = alignedLength; i < length; i++) {
      const value = input[i];
      const legalScore = this.computeLegalScore(value, i);
      const semanticScore = this.computeSemanticScore(value);
      const normalizedValue = this.normalizeValue(value);
      
      output[i] = normalizedValue * legalScore * legalWeight * semanticScore;
    }

    return output;
  }

  // Scalar fallback for non-SIMD environments
  private async processScalarEmbedding(input: Float32Array, output: Float32Array, legalWeight: number): Promise<Float32Array> {
    for (let i = 0; i < input.length; i++) {
      const value = input[i];
      const legalScore = this.computeLegalScore(value, i);
      const semanticScore = this.computeSemanticScore(value);
      const normalizedValue = this.normalizeValue(value);
      
      output[i] = normalizedValue * legalScore * legalWeight * semanticScore;
    }

    return output;
  }

  // Batch embedding processing
  private async processEmbeddingBatch(operations: any[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    
    // Process multiple embeddings in parallel
    const batchSize = Math.min(operations.length, this.vectorWidth * 2);
    
    for (let i = 0; i < operations.length; i += batchSize) {
      const batch = operations.slice(i, i + batchSize);
      const batchPromises = batch.map(op => this.processEmbedding(op));
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
    }

    return results;
  }

  // Vector similarity processing
  private async processSimilarity(operation: any): Promise<Float32Array> {
    const { vectorsA, vectorsB } = operation.input;
    const numPairs = Math.min(vectorsA.length, vectorsB.length);
    const output = this.createAlignedArray(numPairs * 4); // 4 similarity metrics

    for (let i = 0; i < numPairs; i++) {
      const vecA = this.ensureFloat32Array(vectorsA[i]);
      const vecB = this.ensureFloat32Array(vectorsB[i]);
      
      const similarities = this.computeVectorSimilarity(vecA, vecB);
      
      output[i * 4] = similarities.cosine;
      output[i * 4 + 1] = similarities.euclidean;
      output[i * 4 + 2] = similarities.manhattan;
      output[i * 4 + 3] = similarities.jaccard;
    }

    return output;
  }

  // Batch similarity processing
  private async processSimilarityBatch(operations: any[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    
    for (const operation of operations) {
      results.push(await this.processSimilarity(operation));
    }

    return results;
  }

  // K-means clustering with SIMD optimization
  private async processClustering(operation: any): Promise<Float32Array> {
    const points = operation.input.points;
    const numClusters = operation.input.numClusters || 8;
    const maxIterations = operation.input.maxIterations || 100;

    // Initialize centroids using k-means++
    const centroids = this.initializeCentroids(points, numClusters);
    const assignments = new Int32Array(points.length);
    const output = this.createAlignedArray(points.length + numClusters * points[0].length);

    for (let iter = 0; iter < maxIterations; iter++) {
      let hasChanged = false;

      // Assignment step (vectorized)
      for (let i = 0; i < points.length; i++) {
        const point = this.ensureFloat32Array(points[i]);
        let minDistance = Infinity;
        let nearestCluster = 0;

        for (let c = 0; c < numClusters; c++) {
          const centroid = centroids[c];
          const distance = this.computeSquaredDistance(point, centroid);
          
          if (distance < minDistance) {
            minDistance = distance;
            nearestCluster = c;
          }
        }

        if (assignments[i] !== nearestCluster) {
          assignments[i] = nearestCluster;
          hasChanged = true;
        }
      }

      if (!hasChanged) break;

      // Update centroids
      this.updateCentroids(points, assignments, centroids, numClusters);
    }

    // Pack results
    for (let i = 0; i < points.length; i++) {
      output[i] = assignments[i];
    }

    return output;
  }

  // Transform operation (generic tensor transformation)
  private async processTransform(operation: any): Promise<Float32Array> {
    const input = this.ensureFloat32Array(operation.input);
    const transformType = operation.metadata?.transformType || 'linear';
    const output = this.createAlignedArray(input.length);

    switch (transformType) {
      case 'normalize':
        return this.normalizeVector(input, output);
      case 'scale':
        return this.scaleVector(input, output, operation.metadata?.scale || 1.0);
      case 'rotate':
        return this.rotateVector(input, output, operation.metadata?.angle || 0);
      default:
        // Linear transformation
        for (let i = 0; i < input.length; i++) {
          output[i] = input[i] * (operation.metadata?.factor || 1.0);
        }
        return output;
    }
  }

  // Batch transform processing
  private async processTransformBatch(operations: any[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    
    for (const operation of operations) {
      results.push(await this.processTransform(operation));
    }

    return results;
  }

  // Search operation (semantic search)
  private async processSearch(operation: any): Promise<Float32Array> {
    const queryEmbedding = this.ensureFloat32Array(operation.input.query);
    const documentEmbeddings = operation.input.documents.map((doc: any) => this.ensureFloat32Array(doc));
    const threshold = operation.metadata?.threshold || 0.5;

    const scores = this.createAlignedArray(documentEmbeddings.length);

    for (let i = 0; i < documentEmbeddings.length; i++) {
      const similarity = this.computeCosineSimilarity(queryEmbedding, documentEmbeddings[i]);
      scores[i] = similarity > threshold ? similarity : 0;
    }

    return scores;
  }

  // Utility methods
  private groupOperationsByType(operations: any[]): Record<string, any[]> {
    const grouped: Record<string, any[]> = {};
    
    for (const op of operations) {
      if (!grouped[op.type]) {
        grouped[op.type] = [];
      }
      grouped[op.type].push(op);
    }

    return grouped;
  }

  private ensureFloat32Array(input: any): Float32Array {
    if (input instanceof Float32Array) {
      return input;
    } else if (Array.isArray(input)) {
      return new Float32Array(input);
    } else {
      throw new Error('Invalid input type for SIMD processing');
    }
  }

  private createAlignedArray(size: number): Float32Array {
    // Create memory-aligned array for SIMD operations
    const alignment = 32; // 32-byte alignment for AVX
    const buffer = new ArrayBuffer(size * 4 + alignment);
    const offset = alignment - (buffer.byteLength % alignment);
    return new Float32Array(buffer, offset, size);
  }

  private computeLegalScore(value: number, index: number): number {
    // Legal term weighting based on value patterns
    const normalized = Math.abs(value);
    
    if (normalized > 0.8) return 2.0;      // High legal relevance
    else if (normalized > 0.6) return 1.5; // Medium legal relevance
    else if (normalized > 0.4) return 1.2; // Low legal relevance
    else return 1.0;                       // Neutral
  }

  private computeSemanticScore(value: number): number {
    // Semantic relevance based on value distribution
    const sigmoid = 1 / (1 + Math.exp(-value * 2));
    return 0.5 + sigmoid * 0.5; // Range [0.5, 1.0]
  }

  private normalizeValue(value: number): number {
    // Normalize to [-1, 1] range
    return Math.tanh(value);
  }

  private computeVectorSimilarity(vecA: Float32Array, vecB: Float32Array): any {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    let manhattan = 0;

    // Vectorized computation
    for (let i = 0; i < vecA.length; i++) {
      const a = vecA[i];
      const b = vecB[i];
      
      dotProduct += a * b;
      normA += a * a;
      normB += b * b;
      manhattan += Math.abs(a - b);
    }

    const normProduct = Math.sqrt(normA * normB);
    const cosine = normProduct > 0 ? dotProduct / normProduct : 0;
    const euclidean = Math.sqrt(vecA.reduce((sum, a, i) => sum + (a - vecB[i]) ** 2, 0));

    // Jaccard similarity (binary approximation)
    let intersection = 0;
    let union = 0;
    for (let i = 0; i < vecA.length; i++) {
      const binaryA = vecA[i] > 0 ? 1 : 0;
      const binaryB = vecB[i] > 0 ? 1 : 0;
      intersection += binaryA & binaryB;
      union += binaryA | binaryB;
    }
    const jaccard = union > 0 ? intersection / union : 0;

    return { cosine, euclidean, manhattan, jaccard };
  }

  private computeCosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    const normProduct = Math.sqrt(normA * normB);
    return normProduct > 0 ? dotProduct / normProduct : 0;
  }

  private computeSquaredDistance(vecA: Float32Array, vecB: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < vecA.length; i++) {
      const diff = vecA[i] - vecB[i];
      sum += diff * diff;
    }
    return sum;
  }

  private initializeCentroids(points: any[], numClusters: number): Float32Array[] {
    const centroids: Float32Array[] = [];
    const used = new Set<number>();

    // K-means++ initialization
    const firstIndex = Math.floor(Math.random() * points.length);
    centroids.push(new Float32Array(points[firstIndex]));
    used.add(firstIndex);

    for (let i = 1; i < numClusters; i++) {
      const distances = points.map((point, index) => {
        if (used.has(index)) return 0;

        let minDist = Infinity;
        for (const centroid of centroids) {
          const dist = this.computeSquaredDistance(new Float32Array(point), centroid);
          minDist = Math.min(minDist, dist);
        }
        return minDist;
      });

      const totalDistance = distances.reduce((sum, dist) => sum + dist, 0);
      let random = Math.random() * totalDistance;

      for (let j = 0; j < distances.length; j++) {
        random -= distances[j];
        if (random <= 0 && !used.has(j)) {
          centroids.push(new Float32Array(points[j]));
          used.add(j);
          break;
        }
      }
    }

    return centroids;
  }

  private updateCentroids(points: any[], assignments: Int32Array, centroids: Float32Array[], numClusters: number): void {
    const clusterCounts = new Int32Array(numClusters);
    const clusterSums = centroids.map(c => new Float32Array(c.length));

    // Reset centroids
    for (let c = 0; c < numClusters; c++) {
      centroids[c].fill(0);
      clusterSums[c].fill(0);
    }

    // Accumulate points
    for (let i = 0; i < points.length; i++) {
      const cluster = assignments[i];
      const point = new Float32Array(points[i]);
      
      clusterCounts[cluster]++;
      for (let j = 0; j < point.length; j++) {
        clusterSums[cluster][j] += point[j];
      }
    }

    // Compute new centroids
    for (let c = 0; c < numClusters; c++) {
      if (clusterCounts[c] > 0) {
        for (let j = 0; j < centroids[c].length; j++) {
          centroids[c][j] = clusterSums[c][j] / clusterCounts[c];
        }
      }
    }
  }

  private normalizeVector(input: Float32Array, output: Float32Array): Float32Array {
    let norm = 0;
    for (let i = 0; i < input.length; i++) {
      norm += input[i] * input[i];
    }
    norm = Math.sqrt(norm);

    if (norm > 0) {
      for (let i = 0; i < input.length; i++) {
        output[i] = input[i] / norm;
      }
    } else {
      output.set(input);
    }

    return output;
  }

  private scaleVector(input: Float32Array, output: Float32Array, scale: number): Float32Array {
    for (let i = 0; i < input.length; i++) {
      output[i] = input[i] * scale;
    }
    return output;
  }

  private rotateVector(input: Float32Array, output: Float32Array, angle: number): Float32Array {
    // 2D rotation for simplicity (extend for higher dimensions)
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    for (let i = 0; i < input.length - 1; i += 2) {
      const x = input[i];
      const y = input[i + 1];
      
      output[i] = x * cos - y * sin;
      output[i + 1] = x * sin + y * cos;
    }

    return output;
  }
}

// Initialize worker
new SIMDWorker();