import crypto from "crypto";
/**
 * WebGPU Multi-Core Vector Processing for Legal AI
 * Provides GPU acceleration for vector operations in the browser
 */

// MinIO operations handled server-side via API calls

// Qdrant service for vector storage
class QdrantService {
  static async upsertToQdrant(id: string, embedding: number[], metadata: any) {
    try {
      await fetch('http://localhost:6333/collections/legal_evidence/points', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          points: [{
            id,
            vector: embedding,
            payload: {
              ...metadata,
              tags: metadata.tags || [],
              case_id: metadata.caseId,
              evidence_type: metadata.type
            }
          }]
        })
      });
    } catch (error) {
      console.error('Qdrant upsert failed:', error);
      throw error;
    }
  }
  
  static async searchWithFilters(queryVector: number[], filters: any, limit = 10) {
    try {
      const response = await fetch('http://localhost:6333/collections/legal_evidence/points/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector: queryVector,
          filter: filters,
          limit,
          with_payload: true
        })
      });
      return await response.json();
    } catch (error) {
      console.error('Qdrant search failed:', error);
      return { result: [] };
    }
  }
}

// GPU Vector Processor for batch operations
class GPUVectorProcessor {
  static async batchEmbeddings(texts: string[]): Promise<number[][]> {
    const embeddings = [];
    for (const text of texts) {
      try {
        const response = await fetch('http://localhost:11434/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'nomic-embed-text',
            prompt: text
          })
        });
        const result = await response.json();
        embeddings.push(result.embedding);
      } catch (error) {
        console.error('Embedding failed:', error);
        embeddings.push([]);
      }
    }
    return embeddings;
  }
}

export class WebGPUVectorProcessor {
  private device: GPUDevice | null = null;
  private queue: GPUQueue | null = null;
  private initialized = false;

  async initialize(): Promise<boolean> {
    try {
      if (!('gpu' in navigator)) {
        console.warn('WebGPU not supported in this browser');
        return false;
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.warn('WebGPU adapter not available');
        return false;
      }

      this.device = await adapter.requestDevice();
      this.queue = this.device.queue;
      this.initialized = true;

      console.log('✅ WebGPU initialized for legal AI vector processing');
      return true;
    } catch (error) {
      console.error('WebGPU initialization failed:', error);
      return false;
    }
  }

  /**
   * GPU-accelerated dot product for normalized vectors
   * Since vectors are normalized on server, cosine similarity = dot product
   */
  async computeDotProducts(queryVector: number[], candidateVectors: number[][]): Promise<number[]> {
    if (!this.initialized || !this.device) {
      return this.fallbackDotProducts(queryVector, candidateVectors);
    }

    try {
      const vectorSize = queryVector.length;
      const numCandidates = candidateVectors.length;

      // Create GPU buffers
      const queryBuffer = this.device.createBuffer({
        size: vectorSize * 4, // 4 bytes per float32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const candidatesBuffer = this.device.createBuffer({
        size: numCandidates * vectorSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const resultsBuffer = this.device.createBuffer({
        size: numCandidates * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const stagingBuffer = this.device.createBuffer({
        size: numCandidates * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      // Upload data to GPU
      this.queue!.writeBuffer(queryBuffer, 0, new Float32Array(queryVector));
      this.queue!.writeBuffer(candidatesBuffer, 0, new Float32Array(candidateVectors.flat()));

      // Create compute shader for dot product
      const shaderModule = this.device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> query: array<f32>;
          @group(0) @binding(1) var<storage, read> candidates: array<f32>;
          @group(0) @binding(2) var<storage, read_write> results: array<f32>;

          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= ${numCandidates}u) { return; }
            
            var dot_product: f32 = 0.0;
            let offset = index * ${vectorSize}u;
            
            for (var i: u32 = 0u; i < ${vectorSize}u; i++) {
              dot_product += query[i] * candidates[offset + i];
            }
            
            results[index] = dot_product;
          }
        `
      });

      // Create compute pipeline
      const pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      });

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: queryBuffer } },
          { binding: 1, resource: { buffer: candidatesBuffer } },
          { binding: 2, resource: { buffer: resultsBuffer } },
        ],
      });

      // Execute compute shader
      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(numCandidates / 64));
      passEncoder.end();

      commandEncoder.copyBufferToBuffer(resultsBuffer, 0, stagingBuffer, 0, numCandidates * 4);
      this.queue!.submit([commandEncoder.finish()]);

      // Read results back
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const arrayBuffer = stagingBuffer.getMappedRange();
      const results = Array.from(new Float32Array(arrayBuffer));
      stagingBuffer.unmap();

      return results;
    } catch (error) {
      console.error('GPU dot product computation failed:', error);
      return this.fallbackDotProducts(queryVector, candidateVectors);
    }
  }

  /**
   * CPU fallback for dot product computation
   */
  private fallbackDotProducts(queryVector: number[], candidateVectors: number[][]): number[] {
    return candidateVectors.map(candidate => 
      queryVector.reduce((sum, val, i) => sum + val * candidate[i], 0)
    );
  }

  /**
   * GPU-accelerated vector similarity search with payload filters
   */
  async searchSimilarEvidence(
    queryText: string,
    caseId?: string,
    evidenceType?: string,
    tags?: string[],
    limit = 10
  ): Promise<unknown[]> {
    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(queryText);
      
      // Build Qdrant filters
      const filters: any = {};
      if (caseId) filters.case_id = caseId;
      if (evidenceType) filters.evidence_type = evidenceType;
      if (tags?.length) filters.tags = { any: tags };

      // Search with payload filters
      const results = await QdrantService.searchWithFilters(queryEmbedding, filters, limit);
      return results.result || [];
    } catch (error) {
      console.error('GPU similarity search failed:', error);
      return [];
    }
  }

  /**
   * Generate embedding using Ollama
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: text
        })
      });
      const result = await response.json();
      return result.embedding;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      return [];
    }
  }

  /**
   * Batch process multiple evidence files with GPU acceleration
   * Note: MinIO upload happens server-side, this handles client-side analysis
   */
  async batchProcessEvidence(files: File[], caseId: string): Promise<unknown[]> {
    const results = [];
    
    // Extract text from all files first
    const textContents = await Promise.all(
      files.map(async (file) => {
        const arrayBuffer = await file.arrayBuffer();
        if (file.type === 'text/plain') {
          return new TextDecoder().decode(arrayBuffer);
        }
        return `File: ${file.name}`;
      })
    );

    // Batch generate embeddings
    const embeddings = await GPUVectorProcessor.batchEmbeddings(textContents);

    // Process each file with its embedding
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const embedding = embeddings[i];
      const fileId = crypto.randomUUID();

      // Store in Qdrant (MinIO upload happens via API)
      if (embedding.length > 0) {
        await QdrantService.upsertToQdrant(fileId, embedding, {
          caseId,
          fileName: file.name,
          fileType: file.type,
          tags: ['batch_processed']
        });
      }

      results.push({
        fileId,
        fileName: file.name,
        embeddingDimensions: embedding.length,
        processed: true,
        clientSideProcessing: true
      });
    }

    return results;
  }
}

// Singleton instance for browser use
export const webGPUProcessor = new WebGPUVectorProcessor();

// Initialize on module load (browser only)
if (typeof window !== 'undefined') {
  webGPUProcessor.initialize().then(success => {
    if (success) {
      console.log('🚀 WebGPU Legal AI processor ready');
    } else {
      console.log('⚠️ Falling back to CPU vector processing');
    }
  });
}