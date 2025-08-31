// Legacy traversal shim removed — using the SoraGraphTraversal class implementation below.
/**
 * Sora - Advanced Graph Traversal Engine
 * Integrates Neo4j, WASM acceleration, and NES memory architecture
 * for ultra-fast legal knowledge graph navigation
 */

import { nesGPUIntegration, type LegalDocument } from '$lib/gpu/nes-gpu-integration.js';
import { nesMemory, type MemoryBank } from '$lib/memory/nes-memory-architecture.js';
import { semanticAnalysisPipeline, type SemanticAnalysisResult } from '$lib/ai/semantic-analysis-pipeline.js';

// Graph traversal types
export interface GraphNode {
  id: string;
  type: 'case' | 'precedent' | 'statute' | 'person' | 'organization' | 'concept';
  label: string;
  properties: Record<string, any>;
  position: { x: number; y: number; z: number };
  metadata: {
    importance: number;
    confidence: number;
    lastAccessed: number;
    memoryBank?: string;
    vectorEmbedding?: Float32Array;
  };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: 'CITES' | 'RELATED_TO' | 'CONFLICTS_WITH' | 'SUPPORTS' | 'MENTIONS' | 'PART_OF';
  weight: number;
  properties: Record<string, any>;
  metadata: {
    confidence: number;
    strength: number;
    bidirectional: boolean;
  };
}

export interface GraphTraversalQuery {
  startNodes: string[];
  maxDepth: number;
  relationshipTypes?: string[];
  filters?: {
    nodeTypes?: string[];
    minImportance?: number;
    dateRange?: { start: Date; end: Date };
    jurisdiction?: string[];
  };
  traversalStrategy: 'breadth_first' | 'depth_first' | 'weighted' | 'semantic_similarity';
  returnLimit?: number;
  useWasmAcceleration?: boolean;
  cacheResults?: boolean;
}

export interface GraphTraversalResult {
  nodes: Map<string, GraphNode>;
  edges: Map<string, GraphEdge>;
  paths: Array<{
    nodes: string[];
    totalWeight: number;
    confidence: number;
  }>;
  statistics: {
    traversalTime: number;
    nodesVisited: number;
    edgesTraversed: number;
    cacheHits: number;
    wasmAccelerated: boolean;
  };
  visualizationData: {
    positions: Float32Array;
    colors: Float32Array;
    connections: Uint32Array;
    metadata: any;
  };
}

export interface Neo4jConnection {
  uri: string;
  user: string;
  password: string;
  database?: string;
}

export class SoraGraphTraversal {
  private neo4jDriver: any = null;
  private wasmModule: any = null;
  private graphCache = new Map<string, GraphTraversalResult>();
  private nodeCache = new Map<string, GraphNode>();
  private edgeCache = new Map<string, GraphEdge>();
  private isInitialized = false;

  // Performance metrics
  private metrics = {
    totalTraversals: 0,
    cacheHits: 0,
    averageTraversalTime: 0,
    wasmAccelerationUsed: 0,
    memoryBankAccess: 0
  };

  constructor(private config: {
    neo4j?: Neo4jConnection;
    enableWasm?: boolean;
    cacheSize?: number;
    memoryIntegration?: boolean;
  } = {}) {
    this.config = {
      enableWasm: true,
      cacheSize: 10000,
      memoryIntegration: true,
      ...config
    };
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      console.log('🌟 Initializing Sora Graph Traversal Engine...');

      // Initialize Neo4j connection (mock for now)
      if (this.config.neo4j) {
        await this.initializeNeo4j();
      }

      // Initialize WASM acceleration module
      if (this.config.enableWasm) {
        await this.initializeWasm();
      }

      // Integrate with NES memory architecture
      if (this.config.memoryIntegration) {
        await this.initializeMemoryIntegration();
      }

      this.isInitialized = true;
      console.log('✅ Sora Graph Traversal Engine initialized');

    } catch (error: any) {
      console.error('❌ Sora initialization failed:', error);
      throw error;
    }
  }

  /**
   * Advanced graph traversal with multiple strategies
   */
  async traverseGraph(query: GraphTraversalQuery): Promise<GraphTraversalResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const startTime = performance.now();
    const cacheKey = this.generateCacheKey(query);

    try {
      // Check cache first
      if (query.cacheResults !== false && this.graphCache.has(cacheKey)) {
        this.metrics.cacheHits++;
        console.log(`⚡ Sora cache hit for query: ${cacheKey}`);
        return this.graphCache.get(cacheKey)!;
      }

      console.log(`🌟 Starting Sora traversal: ${query.traversalStrategy}`);

      // Execute traversal based on strategy
      let result: GraphTraversalResult;

      switch (query.traversalStrategy) {
        case 'breadth_first':
          result = await this.breadthFirstTraversal(query);
          break;
        case 'depth_first':
          result = await this.depthFirstTraversal(query);
          break;
        case 'weighted':
          result = await this.weightedTraversal(query);
          break;
        case 'semantic_similarity':
          result = await this.semanticSimilarityTraversal(query);
          break;
        default:
          throw new Error(`Unknown traversal strategy: ${query.traversalStrategy}`);
      }

      // Enhance result with NES memory data
      if (this.config.memoryIntegration) {
        result = await this.enhanceWithMemoryData(result);
      }

      // Generate visualization data
      result.visualizationData = await this.generateVisualizationData(result);

      const traversalTime = performance.now() - startTime;
      result.statistics.traversalTime = traversalTime;

      // Update metrics
      this.metrics.totalTraversals++;
      this.metrics.averageTraversalTime =
        (this.metrics.averageTraversalTime * (this.metrics.totalTraversals - 1) + traversalTime) /
        this.metrics.totalTraversals;

      // Cache result
      if (query.cacheResults !== false) {
        this.graphCache.set(cacheKey, result);
      }

      console.log(`✅ Sora traversal completed in ${traversalTime.toFixed(2)}ms`);
      return result;

    } catch (error: any) {
      console.error('❌ Sora traversal failed:', error);
      throw error;
    }
  }

  /**
   * Semantic similarity traversal using AI pipeline
   */
  private async semanticSimilarityTraversal(query: GraphTraversalQuery): Promise<GraphTraversalResult> {
    const nodes = new Map<string, GraphNode>();
    const edges = new Map<string, GraphEdge>();
    const paths: Array<{ nodes: string[]; totalWeight: number; confidence: number }> = [];

    let nodesVisited = 0;
    let edgesTraversed = 0;
    let wasmAccelerated = false;

    try {
      // Get semantic embeddings for start nodes
      const startEmbeddings = await this.getSemanticEmbeddings(query.startNodes);

      // Use WASM acceleration if available
      if (this.config.enableWasm && this.wasmModule && query.useWasmAcceleration !== false) {
        const wasmResult = await this.wasmSemanticTraversal(query, startEmbeddings);
        wasmAccelerated = true;
        nodesVisited = wasmResult.nodesVisited;
        edgesTraversed = wasmResult.edgesTraversed;

        // Convert WASM results to our format
        wasmResult.nodes.forEach((node: any) => nodes.set(node.id, this.convertWasmNode(node)));
        wasmResult.edges.forEach((edge: any) => edges.set(edge.id, this.convertWasmEdge(edge)));
        wasmResult.paths.forEach((path: any) => paths.push(this.convertWasmPath(path)));

      } else {
        // Fallback to JavaScript implementation
        const jsResult = await this.jsSemanticTraversal(query, startEmbeddings);
        nodesVisited = jsResult.nodesVisited;
        edgesTraversed = jsResult.edgesTraversed;
        jsResult.nodes.forEach((node, id) => nodes.set(id, node));
        jsResult.edges.forEach((edge, id) => edges.set(id, edge));
        paths.push(...jsResult.paths);
      }

      return {
        nodes,
        edges,
        paths,
        statistics: {
          traversalTime: 0, // Will be set by caller
          nodesVisited,
          edgesTraversed,
          cacheHits: this.metrics.cacheHits,
          wasmAccelerated
        },
        visualizationData: {
          positions: new Float32Array(0),
          colors: new Float32Array(0),
          connections: new Uint32Array(0),
          metadata: {}
        }
      };

    } catch (error: any) {
      console.error('❌ Semantic similarity traversal failed:', error);
      throw error;
    }
  }

  /**
   * WASM-accelerated semantic traversal
   */
  private async wasmSemanticTraversal(query: GraphTraversalQuery, embeddings: Map<string, Float32Array>): Promise<any> {
    if (!this.wasmModule) {
      throw new Error('WASM module not initialized');
    }

    try {
      // Prepare WASM input data
      const inputData = this.prepareWasmInput(query, embeddings);

      // Call WASM function
      const result = await this.wasmModule.semanticTraversal(inputData);

      this.metrics.wasmAccelerationUsed++;
      return result;

    } catch (error: any) {
      console.warn('WASM traversal failed, falling back to JS:', error);
      return this.jsSemanticTraversal(query, embeddings);
    }
  }

  /**
   * JavaScript fallback for semantic traversal
   */
  private async jsSemanticTraversal(query: GraphTraversalQuery, embeddings: Map<string, Float32Array>): Promise<any> {
    const nodes = new Map<string, GraphNode>();
    const edges = new Map<string, GraphEdge>();
    const paths: Array<{ nodes: string[]; totalWeight: number; confidence: number }> = [];

    let nodesVisited = 0;
    let edgesTraversed = 0;

    // Simulate graph traversal with semantic similarity
    for (const startNodeId of query.startNodes) {
      const embedding = embeddings.get(startNodeId);
      if (!embedding) continue;

      // Find semantically similar nodes
      const similarNodes = await this.findSemanticalSimilarNodes(embedding, query);

      for (const similarNode of similarNodes) {
        nodes.set(similarNode.id, similarNode);
        nodesVisited++;

        // Create path
        paths.push({
          nodes: [startNodeId, similarNode.id],
          totalWeight: similarNode.metadata.importance,
          confidence: similarNode.metadata.confidence
        });

        // Find edges between nodes
        const edgesBetween = await this.findEdgesBetween(startNodeId, similarNode.id);
        edgesBetween.forEach(edge => {
          edges.set(edge.id, edge);
          edgesTraversed++;
        });
      }
    }

    return { nodes, edges, paths, nodesVisited, edgesTraversed };
  }

  /**
   * Breadth-first traversal implementation
   */
  private async breadthFirstTraversal(query: GraphTraversalQuery): Promise<GraphTraversalResult> {
    const nodes = new Map<string, GraphNode>();
    const edges = new Map<string, GraphEdge>();
    const paths: Array<{ nodes: string[]; totalWeight: number; confidence: number }> = [];

    const queue: Array<{ nodeId: string; depth: number; path: string[]; weight: number }> = [];
    const visited = new Set<string>();

    let nodesVisited = 0;
    let edgesTraversed = 0;

    // Initialize queue with start nodes
    query.startNodes.forEach(nodeId => {
      queue.push({ nodeId, depth: 0, path: [nodeId], weight: 0 });
    });

    while (queue.length > 0 && nodesVisited < (query.returnLimit || 1000)) {
      const current = queue.shift()!;

      if (visited.has(current.nodeId) || current.depth > query.maxDepth) {
        continue;
      }

      visited.add(current.nodeId);

      // Get node data
      const node = await this.getNodeById(current.nodeId);
      if (node && this.passesFilters(node, query.filters)) {
        nodes.set(current.nodeId, node);
        nodesVisited++;

        if (current.path.length > 1) {
          paths.push({
            nodes: [...current.path],
            totalWeight: current.weight,
            confidence: node.metadata.confidence
          });
        }
      }

      // Get neighbors
      if (current.depth < query.maxDepth) {
        const neighbors = await this.getNodeNeighbors(current.nodeId, query.relationshipTypes);

        for (const { node: neighborNode, edge } of neighbors) {
          if (!visited.has(neighborNode.id)) {
            edges.set(edge.id, edge);
            edgesTraversed++;

            queue.push({
              nodeId: neighborNode.id,
              depth: current.depth + 1,
              path: [...current.path, neighborNode.id],
              weight: current.weight + edge.weight
            });
          }
        }
      }
    }

    return {
      nodes,
      edges,
      paths,
      statistics: {
        traversalTime: 0,
        nodesVisited,
        edgesTraversed,
        cacheHits: this.metrics.cacheHits,
        wasmAccelerated: false
      },
      visualizationData: {
        positions: new Float32Array(0),
        colors: new Float32Array(0),
        connections: new Uint32Array(0),
        metadata: {}
      }
    };
  }

  /**
   * Depth-first traversal implementation
   */
  private async depthFirstTraversal(query: GraphTraversalQuery): Promise<GraphTraversalResult> {
    const nodes = new Map<string, GraphNode>();
    const edges = new Map<string, GraphEdge>();
    const paths: Array<{ nodes: string[]; totalWeight: number; confidence: number }> = [];

    const stack: Array<{ nodeId: string; depth: number; path: string[]; weight: number }> = [];
    const visited = new Set<string>();

    let nodesVisited = 0;
    let edgesTraversed = 0;

    // Initialize stack with start nodes
    query.startNodes.forEach(nodeId => {
      stack.push({ nodeId, depth: 0, path: [nodeId], weight: 0 });
    });

    while (stack.length > 0 && nodesVisited < (query.returnLimit || 1000)) {
      const current = stack.pop()!;

      if (visited.has(current.nodeId) || current.depth > query.maxDepth) {
        continue;
      }

      visited.add(current.nodeId);

      // Get node data
      const node = await this.getNodeById(current.nodeId);
      if (node && this.passesFilters(node, query.filters)) {
        nodes.set(current.nodeId, node);
        nodesVisited++;

        if (current.path.length > 1) {
          paths.push({
            nodes: [...current.path],
            totalWeight: current.weight,
            confidence: node.metadata.confidence
          });
        }
      }

      // Get neighbors (reverse order for DFS)
      if (current.depth < query.maxDepth) {
        const neighbors = await this.getNodeNeighbors(current.nodeId, query.relationshipTypes);

        for (let i = neighbors.length - 1; i >= 0; i--) {
          const { node: neighborNode, edge } = neighbors[i];

          if (!visited.has(neighborNode.id)) {
            edges.set(edge.id, edge);
            edgesTraversed++;

            stack.push({
              nodeId: neighborNode.id,
              depth: current.depth + 1,
              path: [...current.path, neighborNode.id],
              weight: current.weight + edge.weight
            });
          }
        }
      }
    }

    return {
      nodes,
      edges,
      paths,
      statistics: {
        traversalTime: 0,
        nodesVisited,
        edgesTraversed,
        cacheHits: this.metrics.cacheHits,
        wasmAccelerated: false
      },
      visualizationData: {
        positions: new Float32Array(0),
        colors: new Float32Array(0),
        connections: new Uint32Array(0),
        metadata: {}
      }
    };
  }

  /**
   * Weighted traversal using edge weights and node importance
   */
  private async weightedTraversal(query: GraphTraversalQuery): Promise<GraphTraversalResult> {
    // Use Dijkstra's algorithm for weighted shortest paths
    const nodes = new Map<string, GraphNode>();
    const edges = new Map<string, GraphEdge>();
    const paths: Array<{ nodes: string[]; totalWeight: number; confidence: number }> = [];

    const distances = new Map<string, number>();
    const previous = new Map<string, string | null>();
    const unvisited = new Set<string>();

    let nodesVisited = 0;
    let edgesTraversed = 0;

    // Initialize distances
    query.startNodes.forEach(nodeId => {
      distances.set(nodeId, 0);
      previous.set(nodeId, null);
      unvisited.add(nodeId);
    });

    while (unvisited.size > 0 && nodesVisited < (query.returnLimit || 1000)) {
      // Find node with minimum distance
      let currentNode: string | null = null;
      let minDistance = Infinity;

      for (const nodeId of unvisited) {
        const distance = distances.get(nodeId) || Infinity;
        if (distance < minDistance) {
          minDistance = distance;
          currentNode = nodeId;
        }
      }

      if (!currentNode || minDistance === Infinity) break;

      unvisited.delete(currentNode);

      // Get node data
      const node = await this.getNodeById(currentNode);
      if (node && this.passesFilters(node, query.filters)) {
        nodes.set(currentNode, node);
        nodesVisited++;

        // Reconstruct path
        const path = this.reconstructPath(previous, currentNode);
        if (path.length > 1) {
          paths.push({
            nodes: path,
            totalWeight: minDistance,
            confidence: node.metadata.confidence
          });
        }
      }

      // Check neighbors
      const neighbors = await this.getNodeNeighbors(currentNode, query.relationshipTypes);

      for (const { node: neighborNode, edge } of neighbors) {
        if (unvisited.has(neighborNode.id) || !unvisited.has(neighborNode.id)) {
          const alt = minDistance + edge.weight;
          const currentDistance = distances.get(neighborNode.id) || Infinity;

          if (alt < currentDistance) {
            distances.set(neighborNode.id, alt);
            previous.set(neighborNode.id, currentNode);
            unvisited.add(neighborNode.id);
          }

          edges.set(edge.id, edge);
          edgesTraversed++;
        }
      }
    }

    return {
      nodes,
      edges,
      paths,
      statistics: {
        traversalTime: 0,
        nodesVisited,
        edgesTraversed,
        cacheHits: this.metrics.cacheHits,
        wasmAccelerated: false
      },
      visualizationData: {
        positions: new Float32Array(0),
        colors: new Float32Array(0),
        connections: new Uint32Array(0),
        metadata: {}
      }
    };
  }

  /**
   * Generate visualization data for Moogle
   */
  private async generateVisualizationData(result: GraphTraversalResult): Promise<{
    positions: Float32Array;
    colors: Float32Array;
    connections: Uint32Array;
    metadata: any;
  }> {
    const nodeCount = result.nodes.size;
    const edgeCount = result.edges.size;

    // Positions (x, y, z for each node)
    const positions = new Float32Array(nodeCount * 3);
    // Colors (r, g, b, a for each node)
    const colors = new Float32Array(nodeCount * 4);
    // Connections (source, target for each edge)
    const connections = new Uint32Array(edgeCount * 2);

    let nodeIndex = 0;
    const nodeIndexMap = new Map<string, number>();

    // Process nodes
    for (const [nodeId, node] of result.nodes) {
      nodeIndexMap.set(nodeId, nodeIndex);

      // Position
      positions[nodeIndex * 3] = node.position.x;
      positions[nodeIndex * 3 + 1] = node.position.y;
      positions[nodeIndex * 3 + 2] = node.position.z;

      // Color based on node type and importance
      const color = this.getNodeColor(node);
      colors[nodeIndex * 4] = color.r;
      colors[nodeIndex * 4 + 1] = color.g;
      colors[nodeIndex * 4 + 2] = color.b;
      colors[nodeIndex * 4 + 3] = color.a;

      nodeIndex++;
    }

    // Process edges
    let edgeIndex = 0;
    for (const [edgeId, edge] of result.edges) {
      const sourceIndex = nodeIndexMap.get(edge.source);
      const targetIndex = nodeIndexMap.get(edge.target);

      if (sourceIndex !== undefined && targetIndex !== undefined) {
        connections[edgeIndex * 2] = sourceIndex;
        connections[edgeIndex * 2 + 1] = targetIndex;
        edgeIndex++;
      }
    }

    return {
      positions,
      colors,
      connections: connections.slice(0, edgeIndex * 2),
      metadata: {
        nodeCount,
        edgeCount: edgeIndex,
        boundingBox: this.calculateBoundingBox(positions),
        nodeTypes: this.getNodeTypeDistribution(result.nodes),
        edgeTypes: this.getEdgeTypeDistribution(result.edges)
      }
    };
  }

  // Helper methods (mock implementations for now)
  private async initializeNeo4j(): Promise<void> {
    // Mock Neo4j initialization
    console.log('🔗 Neo4j connection initialized (mock)');
  }

  private async initializeWasm(): Promise<void> {
    try {
      // Mock WASM module loading
      this.wasmModule = {
        semanticTraversal: async (inputData: any) => ({
          nodes: [],
          edges: [],
          paths: [],
          nodesVisited: 0,
          edgesTraversed: 0
        })
      };
      console.log('🚀 WASM module initialized (mock)');
    } catch (error: any) {
      console.warn('WASM initialization failed:', error);
    }
  }

  private async initializeMemoryIntegration(): Promise<void> {
    // Integration with NES memory architecture
    console.log('🎮 Memory integration initialized');
  }

  private generateCacheKey(query: GraphTraversalQuery): string {
    return `${query.startNodes.join(',')}_${query.maxDepth}_${query.traversalStrategy}_${JSON.stringify(query.filters)}`;
  }

  private async getSemanticEmbeddings(nodeIds: string[]): Promise<Map<string, Float32Array>> {
    const embeddings = new Map<string, Float32Array>();

    for (const nodeId of nodeIds) {
      // Mock embedding generation
      const embedding = new Float32Array(384);
      for (let i = 0; i < 384; i++) {
        embedding[i] = Math.random() * 2 - 1;
      }
      embeddings.set(nodeId, embedding);
    }

    return embeddings;
  }

  private async findSemanticalSimilarNodes(embedding: Float32Array, query: GraphTraversalQuery): Promise<GraphNode[]> {
    // Mock semantic similarity search
    const similarNodes: GraphNode[] = [];

    for (let i = 0; i < Math.min(10, query.returnLimit || 10); i++) {
      similarNodes.push({
        id: `sim_node_${i}`,
        type: 'concept',
        label: `Similar Concept ${i}`,
        properties: {},
        position: { x: Math.random() * 100, y: Math.random() * 100, z: Math.random() * 100 },
        metadata: {
          importance: Math.random(),
          confidence: Math.random() * 0.3 + 0.7,
          lastAccessed: Date.now(),
          vectorEmbedding: embedding
        }
      });
    }

    return similarNodes;
  }

  private async findEdgesBetween(sourceId: string, targetId: string): Promise<GraphEdge[]> {
    // Mock edge finding
    return [{
      id: `edge_${sourceId}_${targetId}`,
      source: sourceId,
      target: targetId,
      type: 'RELATED_TO',
      weight: Math.random(),
      properties: {},
      metadata: {
        confidence: Math.random() * 0.3 + 0.7,
        strength: Math.random(),
        bidirectional: true
      }
    }];
  }

  private async getNodeById(nodeId: string): Promise<GraphNode | null> {
    // Check cache first
    if (this.nodeCache.has(nodeId)) {
      return this.nodeCache.get(nodeId)!;
    }

    // Mock node retrieval
    const node: GraphNode = {
      id: nodeId,
      type: 'case',
      label: `Node ${nodeId}`,
      properties: { title: `Case ${nodeId}` },
      position: {
        x: Math.random() * 100,
        y: Math.random() * 100,
        z: Math.random() * 100
      },
      metadata: {
        importance: Math.random(),
        confidence: Math.random() * 0.3 + 0.7,
        lastAccessed: Date.now()
      }
    };

    this.nodeCache.set(nodeId, node);
    return node;
  }

  private async getNodeNeighbors(nodeId: string, relationshipTypes?: string[]): Promise<Array<{ node: GraphNode; edge: GraphEdge }>> {
    // Mock neighbor retrieval
    const neighbors: Array<{ node: GraphNode; edge: GraphEdge }> = [];

    for (let i = 0; i < Math.floor(Math.random() * 5) + 1; i++) {
      const neighborId = `neighbor_${nodeId}_${i}`;
      const neighbor = await this.getNodeById(neighborId);

      if (neighbor) {
        const edge: GraphEdge = {
          id: `edge_${nodeId}_${neighborId}`,
          source: nodeId,
          target: neighborId,
          type: 'RELATED_TO',
          weight: Math.random(),
          properties: {},
          metadata: {
            confidence: Math.random() * 0.3 + 0.7,
            strength: Math.random(),
            bidirectional: true
          }
        };

        neighbors.push({ node: neighbor, edge });
      }
    }

    return neighbors;
  }

  private passesFilters(node: GraphNode, filters?: GraphTraversalQuery['filters']): boolean {
    if (!filters) return true;

    if (filters.nodeTypes && !filters.nodeTypes.includes(node.type)) {
      return false;
    }

    if (filters.minImportance && node.metadata.importance < filters.minImportance) {
      return false;
    }

    return true;
  }

  private reconstructPath(previous: Map<string, string | null>, endNode: string): string[] {
    const path: string[] = [];
    let current: string | null = endNode;

    while (current !== null) {
      path.unshift(current);
      current = previous.get(current) || null;
    }

    return path;
  }

  private getNodeColor(node: GraphNode): { r: number; g: number; b: number; a: number } {
    const colors = {
      case: { r: 0.2, g: 0.6, b: 1.0, a: 1.0 },
      precedent: { r: 0.8, g: 0.2, b: 0.2, a: 1.0 },
      statute: { r: 0.2, g: 0.8, b: 0.2, a: 1.0 },
      person: { r: 1.0, g: 0.6, b: 0.2, a: 1.0 },
      organization: { r: 0.6, g: 0.2, b: 0.8, a: 1.0 },
      concept: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }
    };

    return colors[node.type] || colors.concept;
  }

  private calculateBoundingBox(positions: Float32Array): { min: [number, number, number]; max: [number, number, number] } {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < positions.length; i += 3) {
      minX = Math.min(minX, positions[i]);
      minY = Math.min(minY, positions[i + 1]);
      minZ = Math.min(minZ, positions[i + 2]);
      maxX = Math.max(maxX, positions[i]);
      maxY = Math.max(maxY, positions[i + 1]);
      maxZ = Math.max(maxZ, positions[i + 2]);
    }

    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
  }

  private getNodeTypeDistribution(nodes: Map<string, GraphNode>): Record<string, number> {
    const distribution: Record<string, number> = {};

    for (const node of nodes.values()) {
      distribution[node.type] = (distribution[node.type] || 0) + 1;
    }

    return distribution;
  }

  private getEdgeTypeDistribution(edges: Map<string, GraphEdge>): Record<string, number> {
    const distribution: Record<string, number> = {};

    for (const edge of edges.values()) {
      distribution[edge.type] = (distribution[edge.type] || 0) + 1;
    }

    return distribution;
  }

  private async enhanceWithMemoryData(result: GraphTraversalResult): Promise<GraphTraversalResult> {
    // Enhance nodes with NES memory bank information
    for (const [nodeId, node] of result.nodes) {
      const memoryDoc = nesMemory.getDocument(nodeId);
      if (memoryDoc) {
        node.metadata.memoryBank = this.getMemoryBankName(memoryDoc.bankId);
        this.metrics.memoryBankAccess++;
      }
    }

    return result;
  }

  private getMemoryBankName(bankId?: number): string {
    const bankNames = {
      0: 'INTERNAL_RAM',
      1: 'CHR_ROM',
      2: 'PRG_ROM',
      3: 'SAVE_RAM',
      4: 'EXPANSION_ROM'
    };

    return bankNames[bankId as keyof typeof bankNames] || 'UNKNOWN';
  }

  private prepareWasmInput(query: GraphTraversalQuery, embeddings: Map<string, Float32Array>): any {
    return {
      query: JSON.stringify(query),
      embeddings: Array.from(embeddings.entries()).map(([id, emb]) => ({
        id,
        embedding: Array.from(emb)
      }))
    };
  }

  private convertWasmNode(wasmNode: any): GraphNode {
    return wasmNode as GraphNode;
  }

  private convertWasmEdge(wasmEdge: any): GraphEdge {
    return wasmEdge as GraphEdge;
  }

  private convertWasmPath(wasmPath: any): { nodes: string[]; totalWeight: number; confidence: number } {
    return wasmPath;
  }

  /**
   * Get performance metrics
   */
  getMetrics() {
    return { ...this.metrics };
  }

  /**
   * Clear caches
   */
  clearCaches(): void {
    this.graphCache.clear();
    this.nodeCache.clear();
    this.edgeCache.clear();
    console.log('🧹 Sora caches cleared');
  }

  /**
   * Cleanup resources
   */
  async dispose(): Promise<void> {
    this.clearCaches();

    if (this.neo4jDriver) {
      await this.neo4jDriver.close();
    }

    if (this.wasmModule) {
      this.wasmModule = null;
    }

    console.log('🌟 Sora Graph Traversal Engine disposed');
  }
}

// Export singleton instance
export const soraGraphTraversal = new SoraGraphTraversal();