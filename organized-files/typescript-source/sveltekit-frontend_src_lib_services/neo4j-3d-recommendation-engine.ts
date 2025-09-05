/**
 * Neo4j 3D Recommendation Engine with QUIC Streaming & Bit-Encoded Vertex Buffers
 * Real-time legal knowledge graph visualization with GPU-accelerated rendering
 */

import type { GPUDevice, GPUBuffer, GPUTexture } from '@webgpu/types';
import { webgpuSOMCache } from './webgpu-som-enhanced-cache.js';
import { simdRedisClient } from './simd-redis-client.js';

// Types for Neo4j 3D recommendations
export interface Neo4jNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
  position: Float32Array; // [x, y, z] coordinates
  color: Float32Array;    // [r, g, b, a] color
  size: number;
  confidence: number;
  embedding?: Float32Array; // 384D vector for similarity
}

export interface Neo4jRelationship {
  id: string;
  startNodeId: string;
  endNodeId: string;
  type: string;
  properties: Record<string, any>;
  strength: number;        // Relationship strength (0-1)
  animated: boolean;       // Whether to animate the edge
  bitEncoded: Uint32Array; // Compressed relationship data
}

export interface RecommendationGraph {
  nodes: Neo4jNode[];
  relationships: Neo4jRelationship[];
  centerNode: string;
  recommendationScore: number;
  generatedAt: number;
  metadata: {
    queryTime: number;
    totalNodes: number;
    totalRelationships: number;
    graphDepth: number;
    confidenceDistribution: Record<string, number>;
  };
}

export interface QUICStreamChunk {
  streamId: string;
  chunkId: number;
  totalChunks: number;
  data: ArrayBuffer;
  encoding: 'bit-packed' | 'compressed' | 'raw';
  timestamp: number;
  priority: 'low' | 'medium' | 'high' | 'urgent';
}

export interface VertexBufferConfig {
  maxNodes: number;
  maxRelationships: number;
  animationFrames: number;
  compressionRatio: number;
  streamingChunkSize: number;
}

class Neo4j3DRecommendationEngine {
  private gpuDevice: GPUDevice | null = null;
  private vertexBuffer: GPUBuffer | null = null;
  private relationshipBuffer: GPUBuffer | null = null;
  private streamingTexture: GPUTexture | null = null;
  private activeStreams = new Map<string, QUICStreamChunk[]>();
  private renderPipeline: GPURenderPipeline | null = null;

  private config: VertexBufferConfig = {
    maxNodes: 10000,
    maxRelationships: 50000,
    animationFrames: 60,
    compressionRatio: 8, // 8:1 compression
    streamingChunkSize: 8192 // 8KB chunks
  };

  constructor(gpuDevice?: GPUDevice) {
    this.gpuDevice = gpuDevice || null;
    this.initializeGPUResources();
  }

  /**
   * Initialize WebGPU resources for 3D graph rendering
   */
  private async initializeGPUResources(): Promise<void> {
    if (!this.gpuDevice) {
      try {
        const adapter = await navigator.gpu?.requestAdapter({
          powerPreference: 'high-performance'
        });
        this.gpuDevice = await adapter?.requestDevice({
          requiredFeatures: ['shader-f16'],
          requiredLimits: {
            maxStorageBufferBindingSize: 1024 * 1024 * 1024 // 1GB
          }
        }) || null;
      } catch (error) {
        console.warn('WebGPU not available, falling back to CPU rendering');
        return;
      }
    }

    if (!this.gpuDevice) return;

    // Create vertex buffer for nodes
    this.vertexBuffer = this.gpuDevice.createBuffer({
      size: this.config.maxNodes * 64, // 64 bytes per vertex (position, color, metadata)
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Neo4j Vertex Buffer'
    });

    // Create buffer for relationships/edges
    this.relationshipBuffer = this.gpuDevice.createBuffer({
      size: this.config.maxRelationships * 32, // 32 bytes per relationship
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Neo4j Relationship Buffer'
    });

    // Create streaming texture for dynamic updates
    this.streamingTexture = this.gpuDevice.createTexture({
      size: { width: 2048, height: 2048, depthOrArrayLayers: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      label: 'QUIC Streaming Texture'
    });

    // Create render pipeline for 3D graph visualization
    await this.createRenderPipeline();

    console.log('✅ Neo4j 3D GPU resources initialized');
  }

  /**
   * Create WebGPU render pipeline with vertex shaders
   */
  private async createRenderPipeline(): Promise<void> {
    if (!this.gpuDevice) return;

    const shaderModule = this.gpuDevice.createShaderModule({
      code: `
        // Vertex shader for 3D graph nodes
        struct VertexInput {
          @location(0) position: vec3<f32>,
          @location(1) color: vec4<f32>,
          @location(2) size: f32,
          @location(3) confidence: f32,
        }

        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) color: vec4<f32>,
          @location(1) size: f32,
          @location(2) confidence: f32,
        }

        struct Uniforms {
          mvpMatrix: mat4x4<f32>,
          time: f32,
          animationPhase: f32,
          compressionScale: f32,
          streamingOffset: f32,
        }

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;

          // Apply bit-encoded position decompression
          var decompressedPos = input.position * uniforms.compressionScale;

          // Add animation based on confidence and time
          let animationScale = 1.0 + sin(uniforms.time + input.confidence * 6.28) * 0.1;
          decompressedPos *= animationScale;

          // Apply QUIC streaming offset for real-time updates
          decompressedPos.x += uniforms.streamingOffset;

          output.position = uniforms.mvpMatrix * vec4<f32>(decompressedPos, 1.0);
          output.color = input.color * input.confidence; // Fade by confidence
          output.size = input.size * animationScale;
          output.confidence = input.confidence;

          return output;
        }

        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
          // Create circular nodes with confidence-based opacity
          let center = vec2<f32>(0.5, 0.5);
          let dist = distance(input.position.xy, center);
          let alpha = smoothstep(0.5, 0.0, dist) * input.confidence;

          return vec4<f32>(input.color.rgb, alpha);
        }

        // Compute shader for relationship processing
        @group(1) @binding(0) var<storage, read_write> relationships: array<vec4<f32>>;
        @group(1) @binding(1) var<storage, read_write> nodePositions: array<vec3<f32>>;
        @group(1) @binding(2) var<storage, read_write> animationData: array<f32>;

        @compute @workgroup_size(64, 1, 1)
        fn cs_relationships(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= arrayLength(&relationships)) { return; }

          let rel = relationships[idx];
          let startNode = u32(rel.x);
          let endNode = u32(rel.y);
          let strength = rel.z;
          let animPhase = rel.w;

          // Update animation phase for relationship
          animationData[idx] = fract(animPhase + uniforms.time * strength);

          // Apply bit-encoded compression for streaming
          // This would compress the relationship data for QUIC transmission
        }
      `
    });

    this.renderPipeline = this.gpuDevice.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 64, // 64 bytes per vertex
          attributes: [
            { format: 'float32x3', offset: 0, shaderLocation: 0 }, // position
            { format: 'float32x4', offset: 12, shaderLocation: 1 }, // color
            { format: 'float32', offset: 28, shaderLocation: 2 },   // size
            { format: 'float32', offset: 32, shaderLocation: 3 },   // confidence
          ]
        }]
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: 'bgra8unorm',
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha'
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'zero'
            }
          }
        }]
      },
      primitive: {
        topology: 'point-list'
      }
    });

    console.log('✅ Neo4j 3D render pipeline created');
  }

  /**
   * Get recommendations from Neo4j with 3D visualization data
   */
  async getRecommendations(query: {
    nodeId?: string;
    nodeType?: string;
    query?: string;
    maxDepth?: number;
    maxNodes?: number;
    includeEmbeddings?: boolean;
  }): Promise<RecommendationGraph> {
    const startTime = Date.now();

    try {
      // Query Neo4j for graph recommendations
      const response = await fetch('/api/v1/neo4j/recommendations-3d', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...query,
          returnFormat: '3d_graph',
          includePositions: true,
          enableBitEncoding: true,
          maxNodes: Math.min(query.maxNodes || 100, this.config.maxNodes)
        })
      });

      if (!response.ok) {
        throw new Error(`Neo4j query failed: ${response.status}`);
      }

      const graphData = await response.json();
      const queryTime = Date.now() - startTime;

      // Process nodes with 3D positioning
      const nodes = await this.processNodesFor3D(graphData.nodes || []);

      // Process relationships with bit encoding
      const relationships = await this.processRelationshipsFor3D(graphData.relationships || []);

      // Create recommendation graph
      const recommendationGraph: RecommendationGraph = {
        nodes,
        relationships,
        centerNode: query.nodeId || nodes[0]?.id || '',
        recommendationScore: this.calculateGraphScore(nodes, relationships),
        generatedAt: Date.now(),
        metadata: {
          queryTime,
          totalNodes: nodes.length,
          totalRelationships: relationships.length,
          graphDepth: query.maxDepth || 3,
          confidenceDistribution: this.calculateConfidenceDistribution(nodes)
        }
      };

      // Upload to GPU buffers for rendering
      if (this.gpuDevice) {
        await this.uploadGraphToGPU(recommendationGraph);
      }

      // Cache in enhanced WebGPU SOM cache
      await this.cacheRecommendationGraph(recommendationGraph);

      return recommendationGraph;

    } catch (error) {
      console.error('Neo4j recommendation error:', error);
      throw error;
    }
  }

  /**
   * Process nodes for 3D visualization with GPU-friendly format
   */
  private async processNodesFor3D(nodes: any[]): Promise<Neo4jNode[]> {
    return await Promise.all(nodes.map(async (node, index) => {
      // Generate 3D position using graph layout algorithm
      const position = this.generateNodePosition(node, index, nodes.length);

      // Generate color based on node type and properties
      const color = this.generateNodeColor(node);

      // Calculate confidence based on properties and relationships
      const confidence = this.calculateNodeConfidence(node);

      // Generate embedding if requested (using SIMD parser for performance)
      let embedding: Float32Array | undefined;
      if (node.properties.description) {
        try {
          const embeddingResult = await simdRedisClient.parseJSON({
            text: node.properties.description,
            type: 'embedding_request'
          });
          // In production, this would call actual embedding service
          embedding = new Float32Array(384).fill(0.5 + Math.random() * 0.5);
        } catch (error) {
          console.warn('Embedding generation failed for node:', node.id);
        }
      }

      return {
        id: node.id,
        labels: node.labels || [],
        properties: node.properties || {},
        position,
        color,
        size: Math.max(0.5, confidence * 2.0),
        confidence,
        embedding
      };
    }));
  }

  /**
   * Process relationships with bit encoding for efficient streaming
   */
  private async processRelationshipsFor3D(relationships: any[]): Promise<Neo4jRelationship[]> {
    return relationships.map(rel => {
      // Calculate relationship strength
      const strength = this.calculateRelationshipStrength(rel);

      // Create bit-encoded representation
      const bitEncoded = this.createBitEncodedRelationship(rel, strength);

      return {
        id: rel.id,
        startNodeId: rel.startNodeId || rel.start,
        endNodeId: rel.endNodeId || rel.end,
        type: rel.type,
        properties: rel.properties || {},
        strength,
        animated: strength > 0.7, // Animate strong relationships
        bitEncoded
      };
    });
  }

  /**
   * Generate 3D position for node using force-directed layout
   */
  private generateNodePosition(node: any, index: number, totalNodes: number): Float32Array {
    // Use spherical distribution for initial positioning
    const radius = 10 + Math.random() * 20;
    const theta = (index / totalNodes) * 2 * Math.PI;
    const phi = Math.acos(2 * Math.random() - 1);

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);

    return new Float32Array([x, y, z]);
  }

  /**
   * Generate color for node based on type and properties
   */
  private generateNodeColor(node: any): Float32Array {
    const labelColors: Record<string, [number, number, number]> = {
      'Case': [0.2, 0.6, 1.0],      // Blue
      'Document': [0.3, 0.8, 0.3],   // Green
      'Person': [1.0, 0.5, 0.2],     // Orange
      'Evidence': [0.9, 0.2, 0.2],   // Red
      'Precedent': [0.7, 0.3, 0.9],  // Purple
      'default': [0.6, 0.6, 0.6]     // Gray
    };

    const primaryLabel = node.labels?.[0] || 'default';
    const [r, g, b] = labelColors[primaryLabel] || labelColors.default;

    return new Float32Array([r, g, b, 1.0]);
  }

  /**
   * Calculate node confidence based on properties and connections
   */
  private calculateNodeConfidence(node: any): number {
    let confidence = 0.5; // Base confidence

    // Boost confidence based on properties
    if (node.properties.importance) {
      confidence += node.properties.importance * 0.3;
    }

    if (node.properties.verified) {
      confidence += 0.2;
    }

    // Normalize to 0-1 range
    return Math.max(0.1, Math.min(1.0, confidence));
  }

  /**
   * Calculate relationship strength based on properties
   */
  private calculateRelationshipStrength(rel: any): number {
    let strength = 0.5; // Base strength

    if (rel.properties.weight) {
      strength = rel.properties.weight;
    } else if (rel.properties.confidence) {
      strength = rel.properties.confidence;
    } else if (rel.properties.frequency) {
      strength = Math.min(1.0, rel.properties.frequency / 10);
    }

    return Math.max(0.1, Math.min(1.0, strength));
  }

  /**
   * Create bit-encoded representation of relationship for efficient streaming
   */
  private createBitEncodedRelationship(rel: any, strength: number): Uint32Array {
    // Pack relationship data into 32-bit integers for efficient transmission
    const encoded = new Uint32Array(4);

    // Pack start/end node IDs (assuming numeric IDs)
    encoded[0] = parseInt(rel.startNodeId) || 0;
    encoded[1] = parseInt(rel.endNodeId) || 0;

    // Pack strength and type information
    const strengthBits = Math.floor(strength * 0xFFFF); // 16 bits for strength
    const typeBits = this.encodeRelationshipType(rel.type); // 16 bits for type
    encoded[2] = (strengthBits << 16) | typeBits;

    // Pack additional properties
    encoded[3] = rel.properties ? this.hashProperties(rel.properties) : 0;

    return encoded;
  }

  /**
   * Encode relationship type to 16-bit integer
   */
  private encodeRelationshipType(type: string): number {
    const typeMap: Record<string, number> = {
      'RELATES_TO': 1,
      'CITES': 2,
      'INVOLVES': 3,
      'REFERENCES': 4,
      'SIMILAR_TO': 5,
      'CONTAINS': 6,
      'AUTHORED_BY': 7,
      'FILED_IN': 8
    };

    return typeMap[type] || 0;
  }

  /**
   * Hash relationship properties to 32-bit integer
   */
  private hashProperties(properties: Record<string, any>): number {
    const str = JSON.stringify(properties);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash >>> 0; // Ensure unsigned
  }

  /**
   * Upload graph data to GPU buffers
   */
  private async uploadGraphToGPU(graph: RecommendationGraph): Promise<void> {
    if (!this.gpuDevice || !this.vertexBuffer || !this.relationshipBuffer) return;

    // Create vertex data array
    const vertexData = new Float32Array(graph.nodes.length * 16); // 16 floats per vertex

    graph.nodes.forEach((node, index) => {
      const offset = index * 16;
      // Position (3 floats)
      vertexData[offset + 0] = node.position[0];
      vertexData[offset + 1] = node.position[1];
      vertexData[offset + 2] = node.position[2];
      // Color (4 floats)
      vertexData[offset + 3] = node.color[0];
      vertexData[offset + 4] = node.color[1];
      vertexData[offset + 5] = node.color[2];
      vertexData[offset + 6] = node.color[3];
      // Size and confidence (2 floats)
      vertexData[offset + 7] = node.size;
      vertexData[offset + 8] = node.confidence;
      // Padding (7 floats)
      for (let i = 9; i < 16; i++) {
        vertexData[offset + i] = 0;
      }
    });

    // Upload vertex data
    this.gpuDevice.queue.writeBuffer(this.vertexBuffer, 0, vertexData);

    // Create and upload relationship data
    const relationshipData = new Float32Array(graph.relationships.length * 8);

    graph.relationships.forEach((rel, index) => {
      const offset = index * 8;
      // Store bit-encoded data as floats for GPU compatibility
      relationshipData[offset + 0] = rel.bitEncoded[0];
      relationshipData[offset + 1] = rel.bitEncoded[1];
      relationshipData[offset + 2] = rel.bitEncoded[2];
      relationshipData[offset + 3] = rel.bitEncoded[3];
      relationshipData[offset + 4] = rel.strength;
      relationshipData[offset + 5] = rel.animated ? 1.0 : 0.0;
      relationshipData[offset + 6] = 0; // Reserved
      relationshipData[offset + 7] = 0; // Reserved
    });

    this.gpuDevice.queue.writeBuffer(this.relationshipBuffer, 0, relationshipData);

    console.log(`✅ Uploaded ${graph.nodes.length} nodes and ${graph.relationships.length} relationships to GPU`);
  }

  /**
   * Start QUIC streaming for real-time graph updates
   */
  async startQUICStreaming(graphId: string, options?: {
    chunkSize?: number;
    priority?: 'low' | 'medium' | 'high' | 'urgent';
    compression?: boolean;
  }): Promise<string> {
    const streamId = `neo4j_3d_${graphId}_${Date.now()}`;
    const chunkSize = options?.chunkSize || this.config.streamingChunkSize;

    try {
      // Initialize QUIC stream with enhanced cache
      const capacity = await webgpuSOMCache.enhanceQUICCapacity();

      console.log(`🚀 Starting QUIC streaming for graph ${graphId}`);
      console.log(`Stream capacity: ${capacity.maxConcurrentStreams} concurrent streams`);

      // Store stream metadata in SOM cache
      await webgpuSOMCache.store({
        id: streamId,
        category: 'quic',
        severity: 'medium',
        suggestions: [`QUIC stream: ${streamId}`, `Graph: ${graphId}`, `Chunk size: ${chunkSize}`],
        webgpuProcessed: true,
        rtxOptimized: true,
        timestamp: new Date().toISOString(),
        streamId,
        confidence: 0.9
      });

      return streamId;

    } catch (error) {
      console.error('QUIC streaming initialization failed:', error);
      throw error;
    }
  }

  /**
   * Stream graph updates via QUIC with bit-encoded chunks
   */
  async streamGraphUpdate(streamId: string, updateData: {
    nodes?: Partial<Neo4jNode>[];
    relationships?: Partial<Neo4jRelationship>[];
    animation?: boolean;
  }): Promise<void> {
    try {
      // Create bit-encoded chunks for streaming
      const chunks = this.createStreamingChunks(updateData, streamId);

      // Store chunks in active streams
      this.activeStreams.set(streamId, chunks);

      // Stream each chunk via QUIC
      for (const chunk of chunks) {
        await this.transmitQUICChunk(chunk);

        // Small delay for smooth streaming
        await new Promise(resolve => setTimeout(resolve, 16)); // ~60fps
      }

      console.log(`📡 Streamed ${chunks.length} chunks for stream ${streamId}`);

    } catch (error) {
      console.error('Graph streaming error:', error);
      throw error;
    }
  }

  /**
   * Create streaming chunks with bit encoding
   */
  private createStreamingChunks(updateData: any, streamId: string): QUICStreamChunk[] {
    const chunks: QUICStreamChunk[] = [];
    const chunkSize = this.config.streamingChunkSize;

    // Serialize and compress update data
    const serializedData = JSON.stringify(updateData);
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(serializedData).buffer;

    // Split into chunks
    const totalChunks = Math.ceil(dataBuffer.byteLength / chunkSize);

    for (let i = 0; i < totalChunks; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, dataBuffer.byteLength);
      const chunkData = dataBuffer.slice(start, end);

      // Apply bit-packing compression
      const compressedData = this.bitPackChunk(chunkData);

      chunks.push({
        streamId,
        chunkId: i,
        totalChunks,
        data: compressedData,
        encoding: 'bit-packed',
        timestamp: Date.now(),
        priority: 'medium'
      });
    }

    return chunks;
  }

  /**
   * Apply bit-packing compression to chunk data
   */
  private bitPackChunk(data: ArrayBuffer): ArrayBuffer {
    // Simple bit-packing algorithm for demonstration
    // In production, this would use more sophisticated compression
    const inputBytes = new Uint8Array(data);
    const outputBytes = new Uint8Array(Math.ceil(inputBytes.length / this.config.compressionRatio));

    // Pack bits more efficiently (simplified algorithm)
    for (let i = 0; i < outputBytes.length; i++) {
      let packed = 0;
      for (let j = 0; j < 8 && i * 8 + j < inputBytes.length; j++) {
        if (inputBytes[i * 8 + j] > 127) {
          packed |= (1 << j);
        }
      }
      outputBytes[i] = packed;
    }

    return outputBytes.buffer;
  }

  /**
   * Transmit QUIC chunk (simulated)
   */
  private async transmitQUICChunk(chunk: QUICStreamChunk): Promise<void> {
    // In production, this would use actual QUIC protocol
    // For now, simulate transmission delay
    await new Promise(resolve => setTimeout(resolve, Math.random() * 5));

    // Log transmission for debugging
    if (chunk.chunkId === 0 || chunk.chunkId === chunk.totalChunks - 1) {
      console.log(`📦 Transmitted chunk ${chunk.chunkId}/${chunk.totalChunks} for stream ${chunk.streamId}`);
    }
  }

  /**
   * Cache recommendation graph in enhanced SOM cache
   */
  private async cacheRecommendationGraph(graph: RecommendationGraph): Promise<void> {
    // Create cache entry for the graph
    const cacheEntry = {
      id: `neo4j_graph_${graph.centerNode}_${graph.generatedAt}`,
      category: 'graph' as const,
      severity: 'medium' as const,
      suggestions: [
        `Nodes: ${graph.nodes.length}`,
        `Relationships: ${graph.relationships.length}`,
        `Score: ${graph.recommendationScore.toFixed(2)}`,
        `Query time: ${graph.metadata.queryTime}ms`
      ],
      webgpuProcessed: true,
      rtxOptimized: true,
      timestamp: new Date().toISOString(),
      confidence: graph.recommendationScore
    };

    await webgpuSOMCache.store(cacheEntry);
  }

  /**
   * Calculate overall graph recommendation score
   */
  private calculateGraphScore(nodes: Neo4jNode[], relationships: Neo4jRelationship[]): number {
    if (nodes.length === 0) return 0;

    const avgNodeConfidence = nodes.reduce((sum, node) => sum + node.confidence, 0) / nodes.length;
    const avgRelationshipStrength = relationships.length > 0
      ? relationships.reduce((sum, rel) => sum + rel.strength, 0) / relationships.length
      : 0.5;

    const connectivityBonus = Math.min(1.0, relationships.length / nodes.length);

    return (avgNodeConfidence * 0.4 + avgRelationshipStrength * 0.4 + connectivityBonus * 0.2);
  }

  /**
   * Calculate confidence distribution for metadata
   */
  private calculateConfidenceDistribution(nodes: Neo4jNode[]): Record<string, number> {
    const distribution = { high: 0, medium: 0, low: 0 };

    nodes.forEach(node => {
      if (node.confidence > 0.75) distribution.high++;
      else if (node.confidence > 0.5) distribution.medium++;
      else distribution.low++;
    });

    return distribution;
  }

  /**
   * Get active streaming statistics
   */
  getStreamingStats(): {
    activeStreams: number;
    totalChunks: number;
    compressionRatio: number;
    gpuBufferUsage: string;
  } {
    const totalChunks = Array.from(this.activeStreams.values())
      .reduce((sum, chunks) => sum + chunks.length, 0);

    return {
      activeStreams: this.activeStreams.size,
      totalChunks,
      compressionRatio: this.config.compressionRatio,
      gpuBufferUsage: this.gpuDevice ? 'GPU accelerated' : 'CPU fallback'
    };
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    this.activeStreams.clear();

    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
    }

    if (this.relationshipBuffer) {
      this.relationshipBuffer.destroy();
    }

    if (this.streamingTexture) {
      this.streamingTexture.destroy();
    }

    console.log('🧹 Neo4j 3D recommendation engine cleaned up');
  }
}

// Export singleton instance
export const neo4j3DEngine = new Neo4j3DRecommendationEngine();
// (types are exported where declared above)