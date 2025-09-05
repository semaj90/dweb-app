// ======================================================================
// GRAPH VERTEX COMPRESSION + CONTEXT EMBEDDING
// Compresses graph context and integrates with canonical result cache
// ======================================================================

import type { CanonicalResult, RankingSet } from './canonical-result-cache.js';

export interface GraphVertex {
  id: string;
  type: 'case' | 'evidence' | 'person' | 'precedent' | 'document';
  properties: Record<string, any>;
  embedding?: Float32Array;        // 384D vector from nomic-embed-text
  relationships: GraphEdge[];
  metadata?: {
    importance: number;           // 0-1 importance score
    confidence: number;          // 0-1 confidence in data
    lastUpdated: number;         // timestamp
    source: string;              // data source
  };
}

export interface GraphEdge {
  target: string;                // Target vertex ID
  type: 'references' | 'contains' | 'related_to' | 'cites' | 'involves';
  weight: number;                // 0-1 relationship strength
  properties?: Record<string, any>;
}

export interface GraphContext {
  vertices: GraphVertex[];
  focusVertex: string;           // Primary vertex of interest
  searchRadius: number;          // Degrees of separation
  totalVertices: number;         // Total in full graph
  timestamp: number;
  version: number;
}

export interface CompressedGraphContext {
  compressed: Uint8Array;        // Bit-packed graph data
  compressionRatio: number;      // Original size / compressed size
  vertexCount: number;
  edgeCount: number;
  embeddingDimensions: number;
  compressionMethod: 'deflate' | 'lz4' | 'custom';
  metadata: {
    focusVertex: string;
    searchRadius: number;
    timestamp: number;
    version: number;
  };
}

export interface GraphAugmentedResult extends CanonicalResult {
  graphContext?: CompressedGraphContext;
  vertexRelevance: number;       // How relevant this result is to graph context
  graphDistance: number;         // Distance from focus vertex
  relationshipTypes: string[];   // Types of relationships involved
}

// Custom bit-packing for graph data
class GraphBitPacker {
  private buffer: ArrayBuffer;
  private view: DataView;
  private bitOffset = 0;
  private byteOffset = 0;

  constructor(estimatedSize: number) {
    this.buffer = new ArrayBuffer(estimatedSize);
    this.view = new DataView(this.buffer);
  }

  // Pack vertex ID with dictionary compression
  packVertexId(id: string, dictionary: Map<string, number>): void {
    let dictIndex = dictionary.get(id);
    if (dictIndex === undefined) {
      dictIndex = dictionary.size;
      dictionary.set(id, dictIndex);
    }
    
    // Use variable-length encoding for dictionary indices
    this.packVariableInt(dictIndex);
  }

  // Pack floating point with quantization
  packQuantizedFloat(value: number, bits: number = 8): void {
    const maxVal = (1 << bits) - 1;
    const quantized = Math.round(value * maxVal);
    this.packBits(quantized, bits);
  }

  // Pack embedding vectors with quantization
  packEmbedding(embedding: Float32Array, quantizationBits: number = 8): void {
    // Quantize to reduce from float32 to 8-bit integers
    const quantizationFactor = (1 << quantizationBits) - 1;
    
    for (let i = 0; i < embedding.length; i++) {
      const quantized = Math.max(0, Math.min(quantizationFactor, 
        Math.round((embedding[i] + 1) / 2 * quantizationFactor))); // Normalize to [0,1]
      this.packBits(quantized, quantizationBits);
    }
  }

  // Pack arbitrary bits
  packBits(value: number, bits: number): void {
    for (let i = bits - 1; i >= 0; i--) {
      const bit = (value >> i) & 1;
      this.setBit(bit === 1);
    }
  }

  // Pack variable-length integers
  packVariableInt(value: number): void {
    while (value >= 128) {
      this.packBits((value & 0x7F) | 0x80, 8);
      value >>>= 7;
    }
    this.packBits(value & 0x7F, 8);
  }

  private setBit(bit: boolean): void {
    const byteIndex = Math.floor(this.bitOffset / 8);
    const bitIndex = this.bitOffset % 8;
    
    if (byteIndex >= this.buffer.byteLength) {
      // Expand buffer if needed
      const newBuffer = new ArrayBuffer(this.buffer.byteLength * 2);
      new Uint8Array(newBuffer).set(new Uint8Array(this.buffer));
      this.buffer = newBuffer;
      this.view = new DataView(this.buffer);
    }
    
    if (bitIndex === 0) {
      this.view.setUint8(byteIndex, bit ? 1 : 0);
    } else {
      const currentByte = this.view.getUint8(byteIndex);
      this.view.setUint8(byteIndex, currentByte | ((bit ? 1 : 0) << (7 - bitIndex)));
    }
    
    this.bitOffset++;
  }

  finalize(): Uint8Array {
    const finalSize = Math.ceil(this.bitOffset / 8);
    return new Uint8Array(this.buffer, 0, finalSize);
  }

  getCompressionRatio(originalSize: number): number {
    const compressedSize = Math.ceil(this.bitOffset / 8);
    return originalSize / compressedSize;
  }
}

class GraphBitUnpacker {
  private view: DataView;
  private bitOffset = 0;

  constructor(private data: Uint8Array) {
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  }

  unpackVariableInt(): number {
    let result = 0;
    let shift = 0;
    
    while (true) {
      const byte = this.unpackBits(8);
      result |= (byte & 0x7F) << shift;
      
      if ((byte & 0x80) === 0) break;
      shift += 7;
    }
    
    return result;
  }

  unpackQuantizedFloat(bits: number = 8): number {
    const maxVal = (1 << bits) - 1;
    const quantized = this.unpackBits(bits);
    return quantized / maxVal;
  }

  unpackEmbedding(dimensions: number, quantizationBits: number = 8): Float32Array {
    const embedding = new Float32Array(dimensions);
    const maxVal = (1 << quantizationBits) - 1;
    
    for (let i = 0; i < dimensions; i++) {
      const quantized = this.unpackBits(quantizationBits);
      embedding[i] = (quantized / maxVal) * 2 - 1; // Denormalize to [-1,1]
    }
    
    return embedding;
  }

  unpackBits(bits: number): number {
    let result = 0;
    
    for (let i = 0; i < bits; i++) {
      const bit = this.getBit();
      result = (result << 1) | (bit ? 1 : 0);
    }
    
    return result;
  }

  private getBit(): boolean {
    const byteIndex = Math.floor(this.bitOffset / 8);
    const bitIndex = this.bitOffset % 8;
    
    if (byteIndex >= this.data.length) {
      throw new Error('Attempted to read beyond buffer');
    }
    
    const byte = this.view.getUint8(byteIndex);
    const bit = (byte >> (7 - bitIndex)) & 1;
    
    this.bitOffset++;
    return bit === 1;
  }
}

export class GraphVertexCompressor {
  constructor(
    private defaultQuantizationBits = 8,
    private enableDeltaCompression = true
  ) {}

  // Compress graph context into binary format
  compressGraphContext(graphContext: GraphContext): CompressedGraphContext {
    const startSize = this.estimateUncompressedSize(graphContext);
    const packer = new GraphBitPacker(Math.ceil(startSize / 4)); // Estimate 4:1 compression

    // Build dictionaries for common strings
    const vertexIdDict = new Map<string, number>();
    const typeDict = new Map<string, number>();
    const propertyKeyDict = new Map<string, number>();

    // Pre-populate dictionaries
    this.buildDictionaries(graphContext, vertexIdDict, typeDict, propertyKeyDict);

    // Pack header
    packer.packVariableInt(1); // Version
    packer.packVariableInt(graphContext.vertices.length);
    packer.packVariableInt(graphContext.searchRadius);
    packer.packVariableInt(graphContext.timestamp);

    // Pack focus vertex
    packer.packVertexId(graphContext.focusVertex, vertexIdDict);

    // Pack dictionaries
    this.packDictionary(packer, vertexIdDict);
    this.packDictionary(packer, typeDict);
    this.packDictionary(packer, propertyKeyDict);

    // Pack vertices with delta compression if enabled
    let previousEmbedding: Float32Array | null = null;
    
    for (const vertex of graphContext.vertices) {
      this.packVertex(packer, vertex, vertexIdDict, typeDict, propertyKeyDict);
      
      // Pack embedding with optional delta compression
      if (vertex.embedding) {
        packer.packBits(1, 1); // Has embedding flag
        
        if (this.enableDeltaCompression && previousEmbedding) {
          // Pack delta from previous embedding
          packer.packBits(1, 1); // Delta flag
          this.packEmbeddingDelta(packer, vertex.embedding, previousEmbedding);
        } else {
          // Pack full embedding
          packer.packBits(0, 1); // No delta flag
          packer.packEmbedding(vertex.embedding, this.defaultQuantizationBits);
        }
        
        previousEmbedding = vertex.embedding;
      } else {
        packer.packBits(0, 1); // No embedding flag
      }
    }

    const compressed = packer.finalize();
    const compressionRatio = packer.getCompressionRatio(startSize);

    return {
      compressed,
      compressionRatio,
      vertexCount: graphContext.vertices.length,
      edgeCount: this.countEdges(graphContext),
      embeddingDimensions: graphContext.vertices[0]?.embedding?.length || 0,
      compressionMethod: 'custom',
      metadata: {
        focusVertex: graphContext.focusVertex,
        searchRadius: graphContext.searchRadius,
        timestamp: graphContext.timestamp,
        version: graphContext.version
      }
    };
  }

  // Decompress graph context from binary format
  decompressGraphContext(compressed: CompressedGraphContext): GraphContext {
    const unpacker = new GraphBitUnpacker(compressed.compressed);

    // Unpack header
    const version = unpacker.unpackVariableInt();
    const vertexCount = unpacker.unpackVariableInt();
    const searchRadius = unpacker.unpackVariableInt();
    const timestamp = unpacker.unpackVariableInt();

    // Unpack dictionaries
    const vertexIdDict = this.unpackStringDictionary(unpacker);
    const typeDict = this.unpackStringDictionary(unpacker);
    const propertyKeyDict = this.unpackStringDictionary(unpacker);

    // Unpack focus vertex
    const focusVertexIndex = unpacker.unpackVariableInt();
    const focusVertex = vertexIdDict[focusVertexIndex];

    // Unpack vertices
    const vertices: GraphVertex[] = [];
    let previousEmbedding: Float32Array | null = null;

    for (let i = 0; i < vertexCount; i++) {
      const vertex = this.unpackVertex(unpacker, vertexIdDict, typeDict, propertyKeyDict);
      
      // Unpack embedding
      const hasEmbedding = unpacker.unpackBits(1) === 1;
      if (hasEmbedding) {
        const isDelta = unpacker.unpackBits(1) === 1;
        
        if (isDelta && previousEmbedding) {
          vertex.embedding = this.unpackEmbeddingDelta(
            unpacker, 
            previousEmbedding, 
            compressed.embeddingDimensions
          );
        } else {
          vertex.embedding = unpacker.unpackEmbedding(
            compressed.embeddingDimensions, 
            this.defaultQuantizationBits
          );
        }
        
        previousEmbedding = vertex.embedding;
      }
      
      vertices.push(vertex);
    }

    return {
      vertices,
      focusVertex,
      searchRadius,
      totalVertices: vertexCount,
      timestamp,
      version: compressed.metadata.version
    };
  }

  // Augment canonical results with graph context
  augmentResultsWithGraph(
    results: CanonicalResult[], 
    graphContext: GraphContext
  ): GraphAugmentedResult[] {
    const compressedGraph = this.compressGraphContext(graphContext);
    
    return results.map(result => {
      const relevance = this.calculateVertexRelevance(result, graphContext);
      const distance = this.calculateGraphDistance(result.docId, graphContext);
      const relationships = this.extractRelationshipTypes(result.docId, graphContext);

      return {
        ...result,
        graphContext: compressedGraph,
        vertexRelevance: relevance,
        graphDistance: distance,
        relationshipTypes: relationships
      };
    });
  }

  // Private helper methods
  private buildDictionaries(
    graphContext: GraphContext,
    vertexIdDict: Map<string, number>,
    typeDict: Map<string, number>,
    propertyKeyDict: Map<string, number>
  ): void {
    for (const vertex of graphContext.vertices) {
      vertexIdDict.set(vertex.id, vertexIdDict.size);
      typeDict.set(vertex.type, typeDict.size);
      
      if (vertex.properties) {
        Object.keys(vertex.properties).forEach(key => {
          propertyKeyDict.set(key, propertyKeyDict.size);
        });
      }
    }
  }

  private packDictionary(packer: GraphBitPacker, dict: Map<string, number>): void {
    const entries = Array.from(dict.entries()).sort((a, b) => a[1] - b[1]);
    packer.packVariableInt(entries.length);
    
    for (const [key] of entries) {
      const keyBytes = new TextEncoder().encode(key);
      packer.packVariableInt(keyBytes.length);
      
      for (const byte of keyBytes) {
        packer.packBits(byte, 8);
      }
    }
  }

  private unpackStringDictionary(unpacker: GraphBitUnpacker): string[] {
    const count = unpacker.unpackVariableInt();
    const dict: string[] = [];
    
    for (let i = 0; i < count; i++) {
      const keyLength = unpacker.unpackVariableInt();
      const keyBytes = new Uint8Array(keyLength);
      
      for (let j = 0; j < keyLength; j++) {
        keyBytes[j] = unpacker.unpackBits(8);
      }
      
      const key = new TextDecoder().decode(keyBytes);
      dict.push(key);
    }
    
    return dict;
  }

  private packVertex(
    packer: GraphBitPacker,
    vertex: GraphVertex,
    vertexIdDict: Map<string, number>,
    typeDict: Map<string, number>,
    propertyKeyDict: Map<string, number>
  ): void {
    // Pack basic vertex info
    packer.packVertexId(vertex.id, vertexIdDict);
    packer.packVariableInt(typeDict.get(vertex.type) || 0);
    
    // Pack metadata if present
    if (vertex.metadata) {
      packer.packBits(1, 1); // Has metadata flag
      packer.packQuantizedFloat(vertex.metadata.importance);
      packer.packQuantizedFloat(vertex.metadata.confidence);
    } else {
      packer.packBits(0, 1); // No metadata flag
    }
    
    // Pack relationships
    packer.packVariableInt(vertex.relationships.length);
    for (const edge of vertex.relationships) {
      packer.packVertexId(edge.target, vertexIdDict);
      packer.packQuantizedFloat(edge.weight);
      // Pack edge type (simplified - would need type dictionary)
      const typeHash = this.simpleHash(edge.type) % 16; // 4 bits for edge type
      packer.packBits(typeHash, 4);
    }
  }

  private unpackVertex(
    unpacker: GraphBitUnpacker,
    vertexIdDict: string[],
    typeDict: string[],
    propertyKeyDict: string[]
  ): GraphVertex {
    // Unpack basic info
    const idIndex = unpacker.unpackVariableInt();
    const typeIndex = unpacker.unpackVariableInt();
    
    const vertex: GraphVertex = {
      id: vertexIdDict[idIndex],
      type: typeDict[typeIndex] as any,
      properties: {},
      relationships: []
    };
    
    // Unpack metadata
    const hasMetadata = unpacker.unpackBits(1) === 1;
    if (hasMetadata) {
      vertex.metadata = {
        importance: unpacker.unpackQuantizedFloat(),
        confidence: unpacker.unpackQuantizedFloat(),
        lastUpdated: Date.now(), // Would be packed/unpacked in full implementation
        source: 'compressed'
      };
    }
    
    // Unpack relationships
    const relationshipCount = unpacker.unpackVariableInt();
    for (let i = 0; i < relationshipCount; i++) {
      const targetIndex = unpacker.unpackVariableInt();
      const weight = unpacker.unpackQuantizedFloat();
      const typeHash = unpacker.unpackBits(4);
      
      vertex.relationships.push({
        target: vertexIdDict[targetIndex],
        type: this.hashToEdgeType(typeHash),
        weight
      });
    }
    
    return vertex;
  }

  private packEmbeddingDelta(
    packer: GraphBitPacker, 
    current: Float32Array, 
    previous: Float32Array
  ): void {
    // Pack differences instead of absolute values
    for (let i = 0; i < current.length; i++) {
      const delta = current[i] - previous[i];
      const quantized = Math.round(delta * 127) + 128; // Map to 0-255 range
      packer.packBits(Math.max(0, Math.min(255, quantized)), 8);
    }
  }

  private unpackEmbeddingDelta(
    unpacker: GraphBitUnpacker, 
    previous: Float32Array, 
    dimensions: number
  ): Float32Array {
    const result = new Float32Array(dimensions);
    
    for (let i = 0; i < dimensions; i++) {
      const quantized = unpacker.unpackBits(8);
      const delta = (quantized - 128) / 127;
      result[i] = previous[i] + delta;
    }
    
    return result;
  }

  private estimateUncompressedSize(graphContext: GraphContext): number {
    // Rough estimate for initial buffer sizing
    let size = 64; // Header
    
    for (const vertex of graphContext.vertices) {
      size += 100; // Base vertex data
      size += vertex.relationships.length * 32; // Relationships
      size += vertex.embedding ? vertex.embedding.length * 4 : 0; // Embedding (float32)
    }
    
    return size;
  }

  private countEdges(graphContext: GraphContext): number {
    return graphContext.vertices.reduce((total, vertex) => 
      total + vertex.relationships.length, 0);
  }

  private calculateVertexRelevance(result: CanonicalResult, graphContext: GraphContext): number {
    const vertex = graphContext.vertices.find(v => v.id === result.docId);
    if (!vertex) return 0;
    
    // Combine result score with vertex importance
    const importance = vertex.metadata?.importance || 0.5;
    return (result.score * 0.7) + (importance * 0.3);
  }

  private calculateGraphDistance(docId: string, graphContext: GraphContext): number {
    // Simple BFS to calculate distance from focus vertex
    const focusVertex = graphContext.vertices.find(v => v.id === graphContext.focusVertex);
    if (!focusVertex) return Infinity;
    
    if (docId === graphContext.focusVertex) return 0;
    
    const visited = new Set<string>();
    const queue: Array<{ id: string; distance: number }> = [{ id: focusVertex.id, distance: 0 }];
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current.id)) continue;
      
      visited.add(current.id);
      
      const vertex = graphContext.vertices.find(v => v.id === current.id);
      if (!vertex) continue;
      
      for (const edge of vertex.relationships) {
        if (edge.target === docId) {
          return current.distance + 1;
        }
        
        if (!visited.has(edge.target)) {
          queue.push({ id: edge.target, distance: current.distance + 1 });
        }
      }
    }
    
    return Infinity;
  }

  private extractRelationshipTypes(docId: string, graphContext: GraphContext): string[] {
    const types = new Set<string>();
    
    for (const vertex of graphContext.vertices) {
      for (const edge of vertex.relationships) {
        if (edge.target === docId || vertex.id === docId) {
          types.add(edge.type);
        }
      }
    }
    
    return Array.from(types);
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) & 0xffffffff;
    }
    return Math.abs(hash);
  }

  private hashToEdgeType(hash: number): GraphEdge['type'] {
    const types: GraphEdge['type'][] = ['references', 'contains', 'related_to', 'cites', 'involves'];
    return types[hash % types.length];
  }
}

// Export singleton instance
export const graphVertexCompressor = new GraphVertexCompressor();