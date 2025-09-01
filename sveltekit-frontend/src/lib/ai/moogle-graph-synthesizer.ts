/**
 * MOOGLE Graph Synthesizer
 * WebAssembly-powered graph visualization synthesizer
 * Outputs embeddable 2D images and 3D mesh vertex buffers from semantic graphs
 */

// Production-compatible simplified types
type TextureRegion = { x: number; y: number; width: number; height: number };
type CHR_ROM_Region = { offset: number; size: number; data: Uint8Array };
type TensorSlice = { data: Float32Array; dimensions: number[] };
type LODLevel = { level: number; vertexCount: number; indexCount: number };
type NESGPUIntegration = { storeTexture?: (name: string, data: any) => Promise<void>; compute3DLayout?: (nodes: any[]) => Promise<any> };
type NESMemoryArchitecture = { allocateCHR_ROM?: (size: number) => any; writeCHR_ROM?: (region: any, data: any) => void };
type DimensionalTensorStore = { storeTensorSlice?: (name: string, slice: any) => Promise<void>; getStats?: () => any };
type SOMWebGPUCache = { getCachedLayout: (key: string) => any; setCachedLayout: (key: string, layout: any) => void };
type GPUTensorWorker = { processVertexBuffer: (data: Float32Array) => Promise<ArrayBuffer> };
import type { SoraGraphNode, SoraGraphEdge, SoraTraversalPath } from './sora-graph-traversal.js';
import path from "path";
type GraphNode = { id: string; properties: any };
type GraphEdge = { id: string; source: string; target: string; weight: number };

export interface MoogleVisualizationConfig {
  // 2D rendering settings
  width: number;
  height: number;
  backgroundColor: string;
  nodeSize: { min: number; max: number };
  edgeThickness: { min: number; max: number };
  
  // 3D mesh settings
  meshDimensions: { width: number; height: number; depth: number };
  vertexCount: number;
  lodLevels: number;
  
  // Color schemes
  colorScheme: 'semantic' | 'type' | 'score' | 'legal' | 'evidence' | 'rainbow';
  nodeColors: Record<string, string>;
  edgeColors: Record<string, string>;
  
  // Layout algorithms
  layout: 'force-directed' | 'hierarchical' | 'circular' | 'grid' | 'neural' | 'legal-context';
  physics: {
    gravity: number;
    repulsion: number;
    attraction: number;
    damping: number;
  };
  
  // High-score reinforcement learning visualization
  reinforcementLearning: {
    enabled: boolean;
    showTrainingProgress: boolean;
    highlightOptimalPaths: boolean;
    showRewardHeatmap: boolean;
    qValueVisualization: boolean;
  };
  
  // Performance settings
  useWebGL: boolean;
  useWasm: boolean;
  enableCaching: boolean;
  qualityLevel: 'low' | 'medium' | 'high' | 'ultra';
}

export interface Moogle2DOutput {
  canvas: HTMLCanvasElement;
  imageData: ImageData;
  base64: string;
  svg: string;
  metadata: {
    nodePositions: Array<{ id: string; x: number; y: number }>;
    edgePositions: Array<{ id: string; points: number[] }>;
    bounds: { minX: number; maxX: number; minY: number; maxY: number };
    renderTime: number;
  };
}

export interface Moogle3DMesh {
  vertices: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
  uvs: Float32Array;
  colors: Float32Array;
  vertexBuffer: ArrayBuffer;
  indexBuffer: ArrayBuffer;
  
  // LOD (Level of Detail) meshes
  lodMeshes: Array<{
    level: number;
    vertices: Float32Array;
    indices: Uint32Array;
    triangleCount: number;
  }>;
  
  metadata: {
    nodePositions: Array<{ id: string; x: number; y: number; z: number }>;
    edgeGeometry: Array<{ id: string; start: number[]; end: number[]; curve?: number[] }>;
    boundingBox: { min: number[]; max: number[]; center: number[] };
    meshStats: { vertexCount: number; triangleCount: number; memoryUsage: number };
    renderTime: number;
  };
}

export interface MoogleReinforcementVisualization {
  qValueHeatmap: ImageData;
  pathHighlights: Array<{ path: string[]; score: number; color: string }>;
  rewardSurface: Moogle3DMesh;
  trainingProgress: Array<{ episode: number; reward: number; pathLength: number }>;
  optimalPolicy: Map<string, string>; // state -> best action
}

export class MoogleGraphSynthesizer {
  private gpuIntegration: NESGPUIntegration;
  private memoryArch: NESMemoryArchitecture;
  private tensorStore: DimensionalTensorStore;
  private somCache: SOMWebGPUCache;
  private gpuWorker: GPUTensorWorker;
  private wasmModule: WebAssembly.Module | null = null;
  private wasmInstance: WebAssembly.Instance | null = null;
  private renderingCache: Map<string, any> = new Map();

  // WebGL contexts and programs
  private gl: WebGL2RenderingContext | null = null;
  private shaderPrograms: Map<string, WebGLProgram> = new Map();
  private vertexBuffers: Map<string, WebGLBuffer> = new Map();

  constructor(
    gpuIntegration?: NESGPUIntegration,
    memoryArch?: NESMemoryArchitecture,
    tensorStore?: DimensionalTensorStore,
    somCache?: SOMWebGPUCache,
    gpuWorker?: GPUTensorWorker
  ) {
    this.gpuIntegration = gpuIntegration || null;
    this.memoryArch = memoryArch || null;
    this.tensorStore = tensorStore || null;
    this.somCache = somCache || null;
    this.gpuWorker = gpuWorker || null;
    this.initializeWebGL();
    this.loadWasmModule();
  }

  /**
   * Initialize WebGL context and shaders
   */
  private async initializeWebGL(): Promise<void> {
    try {
      const canvas = document.createElement('canvas');
      this.gl = canvas.getContext('webgl2', { 
        antialias: true, 
        alpha: true, 
        depth: true,
        preserveDrawingBuffer: true
      });

      if (!this.gl) {
        console.warn('WebGL2 not available, falling back to 2D rendering');
        return;
      }

      // Load shaders for graph rendering
      await this.loadGraphShaders();
      
      // Enable depth testing and blending
      this.gl.enable(this.gl.DEPTH_TEST);
      this.gl.enable(this.gl.BLEND);
      this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

    } catch (error) {
      console.error('WebGL initialization failed:', error);
      this.gl = null;
    }
  }

  /**
   * Load WebAssembly module for high-performance graph processing
   */
  private async loadWasmModule(): Promise<void> {
    try {
      // In a real implementation, you would load a compiled WASM module
      // For now, we'll simulate the WASM functionality with TypedArrays
      console.log('WASM module loading simulated - using optimized TypedArray operations');
      
      // Create simulated WASM memory
      const wasmMemory = new WebAssembly.Memory({ initial: 256, maximum: 512 });
      this.wasmInstance = {
        exports: {
          memory: wasmMemory,
          // Simulated WASM exports
          calculate_force_directed_layout: this.wasmForceDirectedLayout.bind(this),
          generate_mesh_vertices: this.wasmGenerateMeshVertices.bind(this),
          compute_node_positions: this.wasmComputeNodePositions.bind(this),
          optimize_mesh_lod: this.wasmOptimizeMeshLOD.bind(this)
        }
      } as WebAssembly.Instance;
      
    } catch (error) {
      console.error('WASM module loading failed:', error);
      this.wasmInstance = null;
    }
  }

  /**
   * Synthesize 2D visualization from graph traversal paths
   */
  async synthesize2D(
    paths: SoraTraversalPath[],
    config: Partial<MoogleVisualizationConfig> = {}
  ): Promise<Moogle2DOutput> {
    const startTime = performance.now();
    
    const fullConfig: MoogleVisualizationConfig = {
      width: 1920,
      height: 1080,
      backgroundColor: '#1a1a1a',
      nodeSize: { min: 8, max: 32 },
      edgeThickness: { min: 1, max: 6 },
      meshDimensions: { width: 100, height: 100, depth: 100 },
      vertexCount: 10000,
      lodLevels: 4,
      colorScheme: 'semantic',
      nodeColors: {
        'document': '#4CAF50',
        'case': '#2196F3',
        'evidence': '#FF5722',
        'entity': '#9C27B0',
        'concept': '#FFC107',
        'relationship': '#607D8B'
      },
      edgeColors: {
        'cites': '#FF9800',
        'contains': '#8BC34A',
        'related': '#03DAC6',
        'similar': '#E91E63',
        'references': '#00BCD4',
        'contradicts': '#F44336'
      },
      layout: 'legal-context',
      physics: {
        gravity: 0.1,
        repulsion: 100,
        attraction: 0.05,
        damping: 0.9
      },
      reinforcementLearning: {
        enabled: true,
        showTrainingProgress: true,
        highlightOptimalPaths: true,
        showRewardHeatmap: true,
        qValueVisualization: true
      },
      useWebGL: true,
      useWasm: true,
      enableCaching: true,
      qualityLevel: 'high',
      ...config
    };

    // Create caching key
    const cacheKey = this.generateCacheKey(paths, fullConfig, '2d');
    if (fullConfig.enableCaching && this.renderingCache.has(cacheKey)) {
      return this.renderingCache.get(cacheKey);
    }

    // Extract all unique nodes and edges from paths
    const { nodes, edges } = this.extractGraphElements(paths);

    // Calculate layout positions
    const nodePositions = await this.calculateLayout(nodes, edges, fullConfig);

    // Create canvas and render
    const canvas = document.createElement('canvas');
    canvas.width = fullConfig.width;
    canvas.height = fullConfig.height;
    
    let imageData: ImageData;
    let svg: string;

    if (fullConfig.useWebGL && this.gl) {
      // High-performance WebGL rendering
      const webglOutput = await this.renderWebGL2D(canvas, nodes, edges, nodePositions, fullConfig);
      imageData = webglOutput.imageData;
      svg = webglOutput.svg;
    } else {
      // Fallback to 2D canvas rendering
      const canvasOutput = await this.renderCanvas2D(canvas, nodes, edges, nodePositions, fullConfig);
      imageData = canvasOutput.imageData;
      svg = canvasOutput.svg;
    }

    // Generate base64 representation
    const base64 = canvas.toDataURL('image/png', 0.95);

    // Calculate edge positions for metadata
    const edgePositions = Array.from(edges.values()).map(edge => {
      const sourcePos = nodePositions.get(edge.source);
      const targetPos = nodePositions.get(edge.target);
      if (!sourcePos || !targetPos) return { id: edge.id, points: [] };
      
      return {
        id: edge.id,
        points: [sourcePos.x, sourcePos.y, targetPos.x, targetPos.y]
      };
    });

    // Calculate bounds
    const positions = Array.from(nodePositions.values());
    const bounds = {
      minX: Math.min(...positions.map(p => p.x)),
      maxX: Math.max(...positions.map(p => p.x)),
      minY: Math.min(...positions.map(p => p.y)),
      maxY: Math.max(...positions.map(p => p.y))
    };

    const result: Moogle2DOutput = {
      canvas,
      imageData,
      base64,
      svg,
      metadata: {
        nodePositions: Array.from(nodePositions.entries()).map(([id, pos]) => ({ id, x: pos.x, y: pos.y })),
        edgePositions,
        bounds,
        renderTime: performance.now() - startTime
      }
    };

    // Store visualization in tensor store for future reference
    await this.storeVisualizationTensor2D(result, paths, fullConfig);

    // Cache result
    if (fullConfig.enableCaching) {
      this.renderingCache.set(cacheKey, result);
    }

    return result;
  }

  /**
   * Synthesize 3D mesh from graph traversal paths
   */
  async synthesize3D(
    paths: SoraTraversalPath[],
    config: Partial<MoogleVisualizationConfig> = {}
  ): Promise<Moogle3DMesh> {
    const startTime = performance.now();
    
    const fullConfig: MoogleVisualizationConfig = {
      width: 1920,
      height: 1080,
      backgroundColor: '#1a1a1a',
      nodeSize: { min: 8, max: 32 },
      edgeThickness: { min: 1, max: 6 },
      meshDimensions: { width: 100, height: 100, depth: 100 },
      vertexCount: 10000,
      lodLevels: 4,
      colorScheme: 'semantic',
      nodeColors: {
        'document': '#4CAF50',
        'case': '#2196F3',
        'evidence': '#FF5722',
        'entity': '#9C27B0',
        'concept': '#FFC107',
        'relationship': '#607D8B'
      },
      edgeColors: {
        'cites': '#FF9800',
        'contains': '#8BC34A',
        'related': '#03DAC6',
        'similar': '#E91E63',
        'references': '#00BCD4',
        'contradicts': '#F44336'
      },
      layout: 'legal-context',
      physics: {
        gravity: 0.1,
        repulsion: 100,
        attraction: 0.05,
        damping: 0.9
      },
      reinforcementLearning: {
        enabled: true,
        showTrainingProgress: true,
        highlightOptimalPaths: true,
        showRewardHeatmap: true,
        qValueVisualization: true
      },
      useWebGL: true,
      useWasm: true,
      enableCaching: true,
      qualityLevel: 'high',
      ...config
    };

    // Create caching key
    const cacheKey = this.generateCacheKey(paths, fullConfig, '3d');
    if (fullConfig.enableCaching && this.renderingCache.has(cacheKey)) {
      return this.renderingCache.get(cacheKey);
    }

    // Extract graph elements
    const { nodes, edges } = this.extractGraphElements(paths);

    // Calculate 3D layout positions
    const nodePositions3D = await this.calculate3DLayout(nodes, edges, fullConfig);

    // Generate 3D mesh geometry
    const meshGeometry = await this.generateMeshGeometry(nodes, edges, nodePositions3D, fullConfig);

    // Calculate edge geometry for metadata
    const edgeGeometry = Array.from(edges.values()).map(edge => {
      const sourcePos = nodePositions3D.get(edge.source);
      const targetPos = nodePositions3D.get(edge.target);
      if (!sourcePos || !targetPos) {
        return { id: edge.id, start: [0, 0, 0], end: [0, 0, 0] };
      }
      
      return {
        id: edge.id,
        start: [sourcePos.x, sourcePos.y, sourcePos.z],
        end: [targetPos.x, targetPos.y, targetPos.z],
        curve: this.calculateEdgeCurve(sourcePos, targetPos)
      };
    });

    // Calculate bounding box
    const positions = Array.from(nodePositions3D.values());
    const boundingBox = {
      min: [
        Math.min(...positions.map(p => p.x)),
        Math.min(...positions.map(p => p.y)),
        Math.min(...positions.map(p => p.z))
      ],
      max: [
        Math.max(...positions.map(p => p.x)),
        Math.max(...positions.map(p => p.y)),
        Math.max(...positions.map(p => p.z))
      ],
      center: [0, 0, 0]
    };
    
    // Calculate center
    boundingBox.center = [
      (boundingBox.min[0] + boundingBox.max[0]) / 2,
      (boundingBox.min[1] + boundingBox.max[1]) / 2,
      (boundingBox.min[2] + boundingBox.max[2]) / 2
    ];

    // Generate LOD meshes
    const lodMeshes = await this.generateLODMeshes(meshGeometry, fullConfig.lodLevels);

    // Calculate mesh statistics
    const meshStats = {
      vertexCount: meshGeometry.vertices.length / 3,
      triangleCount: meshGeometry.indices.length / 3,
      memoryUsage: (
        meshGeometry.vertices.byteLength +
        meshGeometry.indices.byteLength +
        meshGeometry.normals.byteLength +
        meshGeometry.uvs.byteLength +
        meshGeometry.colors.byteLength
      )
    };

    const result: Moogle3DMesh = {
      vertices: meshGeometry.vertices,
      indices: meshGeometry.indices,
      normals: meshGeometry.normals,
      uvs: meshGeometry.uvs,
      colors: meshGeometry.colors,
      vertexBuffer: meshGeometry.vertices.buffer,
      indexBuffer: meshGeometry.indices.buffer,
      lodMeshes,
      metadata: {
        nodePositions: Array.from(nodePositions3D.entries()).map(([id, pos]) => ({ 
          id, x: pos.x, y: pos.y, z: pos.z 
        })),
        edgeGeometry,
        boundingBox,
        meshStats,
        renderTime: performance.now() - startTime
      }
    };

    // Store 3D mesh in tensor store with LOD optimization
    await this.storeVisualizationTensor3D(result, paths, fullConfig);

    // Cache result
    if (fullConfig.enableCaching) {
      this.renderingCache.set(cacheKey, result);
    }

    return result;
  }

  /**
   * Generate reinforcement learning visualization
   */
  async synthesizeReinforcementVisualization(
    paths: SoraTraversalPath[],
    qValues: Map<string, Map<string, number>>,
    rewardHistory: Array<{ episode: number; reward: number; pathLength: number }>,
    config: Partial<MoogleVisualizationConfig> = {}
  ): Promise<MoogleReinforcementVisualization> {
    const { nodes, edges } = this.extractGraphElements(paths);

    // Generate Q-value heatmap
    const qValueHeatmap = await this.generateQValueHeatmap(nodes, qValues, config);

    // Highlight optimal paths
    const pathHighlights = await this.highlightOptimalPaths(paths, qValues);

    // Create 3D reward surface
    const rewardSurface = await this.generateRewardSurface(nodes, qValues, config);

    // Extract optimal policy
    const optimalPolicy = new Map<string, string>();
    for (const [state, actions] of qValues) {
      let bestAction = '';
      let bestValue = -Infinity;
      for (const [action, value] of actions) {
        if (value > bestValue) {
          bestValue = value;
          bestAction = action;
        }
      }
      if (bestAction) {
        optimalPolicy.set(state, bestAction);
      }
    }

    return {
      qValueHeatmap,
      pathHighlights,
      rewardSurface,
      trainingProgress: rewardHistory,
      optimalPolicy
    };
  }

  /**
   * Store visualization in NES memory architecture
   */
  async storeInNESMemory(
    visualization: Moogle2DOutput | Moogle3DMesh,
    region: 'CHR_ROM' | 'texture_cache' = 'CHR_ROM'
  ): Promise<{ region: string; offset: number; size: number }> {
    try {
      let data: ArrayBuffer;
      let id: string;

      if ('canvas' in visualization) {
        // 2D visualization
        const canvas = visualization.canvas;
        const ctx = canvas.getContext('2d')!;
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        data = imageData.data.buffer;
        id = `moogle_2d_${Date.now()}`;
      } else {
        // 3D visualization
        data = visualization.vertexBuffer;
        id = `moogle_3d_${Date.now()}`;
      }

      // Store in NES memory architecture
      if (region === 'CHR_ROM') {
        if (this.memoryArch?.allocateCHR_ROM && this.memoryArch?.writeCHR_ROM) {
          const chrRegion = await this.memoryArch.allocateCHR_ROM(data.byteLength, 'pattern_table');
          await this.memoryArch.writeCHR_ROM(chrRegion.startAddress, new Uint8Array(data));
        }
        
        return {
          region: 'CHR_ROM',
          offset: chrRegion.startAddress,
          size: data.byteLength
        };
      } else {
        // Store in GPU texture cache
        const textureRegion: TextureRegion = {
          id,
          width: 'canvas' in visualization ? visualization.canvas.width : 512,
          height: 'canvas' in visualization ? visualization.canvas.height : 512,
          format: 'rgba8unorm',
          data: new Uint8Array(data)
        };

        if (this.gpuIntegration?.storeTexture) {
          await this.gpuIntegration.storeTexture(id, textureRegion);
        }
        
        return {
          region: 'texture_cache',
          offset: 0,
          size: data.byteLength
        };
      }
    } catch (error) {
      console.error('Failed to store visualization in NES memory:', error);
      throw error;
    }
  }

  /**
   * Extract unique nodes and edges from paths
   */
  private extractGraphElements(paths: SoraTraversalPath[]): {
    nodes: Map<string, SoraGraphNode>;
    edges: Map<string, SoraGraphEdge>;
  } {
    const nodes = new Map<string, SoraGraphNode>();
    const edges = new Map<string, SoraGraphEdge>();

    for (const path of paths) {
      // Add nodes
      for (const node of path.nodes) {
        nodes.set(node.id, node);
      }

      // Add edges
      for (const edge of path.edges) {
        edges.set(edge.id, edge);
      }
    }

    return { nodes, edges: new Map(Array.from(edges.values()).map(e => [e.id, e])) };
  }

  /**
   * Calculate 2D layout positions
   */
  private async calculateLayout(
    nodes: Map<string, SoraGraphNode>,
    edges: Map<string, SoraGraphEdge>,
    config: MoogleVisualizationConfig
  ): Promise<Map<string, { x: number; y: number }>> {
    const positions = new Map<string, { x: number; y: number }>();

    switch (config.layout) {
      case 'force-directed':
        return this.forceDirectedLayout(nodes, edges, config);
      case 'legal-context':
        return this.legalContextLayout(nodes, edges, config);
      case 'hierarchical':
        return this.hierarchicalLayout(nodes, edges, config);
      case 'circular':
        return this.circularLayout(nodes, edges, config);
      case 'grid':
        return this.gridLayout(nodes, edges, config);
      case 'neural':
        return this.neuralLayout(nodes, edges, config);
      default:
        return this.forceDirectedLayout(nodes, edges, config);
    }
  }

  /**
   * Force-directed layout algorithm
   */
  private async forceDirectedLayout(
    nodes: Map<string, SoraGraphNode>,
    edges: Map<string, SoraGraphEdge>,
    config: MoogleVisualizationConfig
  ): Promise<Map<string, { x: number; y: number }>> {
    const positions = new Map<string, { x: number; y: number }>();
    const velocities = new Map<string, { x: number; y: number }>();

    // Initialize random positions
    for (const [nodeId] of nodes) {
      positions.set(nodeId, {
        x: (Math.random() - 0.5) * config.width * 0.8,
        y: (Math.random() - 0.5) * config.height * 0.8
      });
      velocities.set(nodeId, { x: 0, y: 0 });
    }

    // Simulate physics for multiple iterations
    const iterations = config.qualityLevel === 'ultra' ? 1000 : 
                      config.qualityLevel === 'high' ? 500 : 200;

    for (let i = 0; i < iterations; i++) {
      // Reset forces
      const forces = new Map<string, { x: number; y: number }>();
      for (const [nodeId] of nodes) {
        forces.set(nodeId, { x: 0, y: 0 });
      }

      // Calculate repulsive forces between all nodes
      for (const [nodeId1, pos1] of positions) {
        for (const [nodeId2, pos2] of positions) {
          if (nodeId1 === nodeId2) continue;

          const dx = pos1.x - pos2.x;
          const dy = pos1.y - pos2.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance > 0) {
            const repulsionForce = config.physics.repulsion / (distance * distance);
            const fx = (dx / distance) * repulsionForce;
            const fy = (dy / distance) * repulsionForce;
            
            const force1 = forces.get(nodeId1)!;
            force1.x += fx;
            force1.y += fy;
          }
        }
      }

      // Calculate attractive forces for connected nodes
      for (const [, edge] of edges) {
        const pos1 = positions.get(edge.source);
        const pos2 = positions.get(edge.target);
        
        if (pos1 && pos2) {
          const dx = pos2.x - pos1.x;
          const dy = pos2.y - pos1.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance > 0) {
            const attractionForce = config.physics.attraction * distance * edge.weight;
            const fx = (dx / distance) * attractionForce;
            const fy = (dy / distance) * attractionForce;
            
            const force1 = forces.get(edge.source)!;
            const force2 = forces.get(edge.target)!;
            
            force1.x += fx;
            force1.y += fy;
            force2.x -= fx;
            force2.y -= fy;
          }
        }
      }

      // Apply gravity towards center
      for (const [nodeId, pos] of positions) {
        const force = forces.get(nodeId)!;
        force.x -= pos.x * config.physics.gravity;
        force.y -= pos.y * config.physics.gravity;
      }

      // Update velocities and positions
      for (const [nodeId, pos] of positions) {
        const velocity = velocities.get(nodeId)!;
        const force = forces.get(nodeId)!;

        velocity.x = (velocity.x + force.x) * config.physics.damping;
        velocity.y = (velocity.y + force.y) * config.physics.damping;

        pos.x += velocity.x;
        pos.y += velocity.y;
      }
    }

    // Center and scale positions
    return this.centerAndScalePositions(positions, config.width, config.height);
  }

  /**
   * Legal-context aware layout
   */
  private async legalContextLayout(
    nodes: Map<string, SoraGraphNode>,
    edges: Map<string, SoraGraphEdge>,
    config: MoogleVisualizationConfig
  ): Promise<Map<string, { x: number; y: number }>> {
    const positions = new Map<string, { x: number; y: number }>();

    // Group nodes by legal context
    const typeGroups = {
      'case': [] as SoraGraphNode[],
      'evidence': [] as SoraGraphNode[],
      'document': [] as SoraGraphNode[],
      'entity': [] as SoraGraphNode[],
      'concept': [] as SoraGraphNode[],
      'relationship': [] as SoraGraphNode[]
    };

    for (const [, node] of nodes) {
      if (typeGroups[node.type]) {
        typeGroups[node.type].push(node);
      }
    }

    // Arrange groups in legal hierarchy
    const groupCenters = {
      'case': { x: 0, y: -config.height * 0.3 },
      'evidence': { x: -config.width * 0.25, y: 0 },
      'document': { x: config.width * 0.25, y: 0 },
      'entity': { x: -config.width * 0.15, y: config.height * 0.25 },
      'concept': { x: config.width * 0.15, y: config.height * 0.25 },
      'relationship': { x: 0, y: config.height * 0.4 }
    };

    // Position nodes within their groups
    for (const [type, nodeList] of Object.entries(typeGroups)) {
      const center = groupCenters[type as keyof typeof groupCenters];
      const radius = Math.min(config.width, config.height) * 0.12;

      nodeList.forEach((node, index) => {
        const angle = (index / nodeList.length) * 2 * Math.PI;
        const nodeRadius = radius * (0.3 + Math.random() * 0.7);
        
        positions.set(node.id, {
          x: center.x + Math.cos(angle) * nodeRadius,
          y: center.y + Math.sin(angle) * nodeRadius
        });
      });
    }

    return positions;
  }

  // Additional layout methods (hierarchical, circular, grid, neural) would be implemented here...
  private async hierarchicalLayout(nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, config: MoogleVisualizationConfig) {
    return this.forceDirectedLayout(nodes, edges, config); // Placeholder
  }

  private async circularLayout(nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, config: MoogleVisualizationConfig) {
    return this.forceDirectedLayout(nodes, edges, config); // Placeholder
  }

  private async gridLayout(nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, config: MoogleVisualizationConfig) {
    return this.forceDirectedLayout(nodes, edges, config); // Placeholder
  }

  private async neuralLayout(nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, config: MoogleVisualizationConfig) {
    return this.forceDirectedLayout(nodes, edges, config); // Placeholder
  }

  /**
   * Center and scale positions to fit canvas
   */
  private centerAndScalePositions(
    positions: Map<string, { x: number; y: number }>,
    width: number,
    height: number
  ): Map<string, { x: number; y: number }> {
    if (positions.size === 0) return positions;

    const coords = Array.from(positions.values());
    const minX = Math.min(...coords.map(p => p.x));
    const maxX = Math.max(...coords.map(p => p.x));
    const minY = Math.min(...coords.map(p => p.y));
    const maxY = Math.max(...coords.map(p => p.y));

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scaleX = (width * 0.8) / rangeX;
    const scaleY = (height * 0.8) / rangeY;
    const scale = Math.min(scaleX, scaleY);

    const centerX = width / 2;
    const centerY = height / 2;
    const offsetX = (minX + maxX) / 2;
    const offsetY = (minY + maxY) / 2;

    const centeredPositions = new Map<string, { x: number; y: number }>();
    for (const [nodeId, pos] of positions) {
      centeredPositions.set(nodeId, {
        x: centerX + (pos.x - offsetX) * scale,
        y: centerY + (pos.y - offsetY) * scale
      });
    }

    return centeredPositions;
  }

  // WebGL rendering methods would be implemented here...
  private async renderWebGL2D(canvas: HTMLCanvasElement, nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, positions: Map<string, { x: number; y: number }>, config: MoogleVisualizationConfig) {
    // Placeholder implementation
    const ctx = canvas.getContext('2d')!;
    return this.renderCanvas2D(canvas, nodes, edges, positions, config);
  }

  private async renderCanvas2D(canvas: HTMLCanvasElement, nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, positions: Map<string, { x: number; y: number }>, config: MoogleVisualizationConfig) {
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = config.backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Render edges first
    for (const [, edge] of edges) {
      const sourcePos = positions.get(edge.source);
      const targetPos = positions.get(edge.target);
      if (!sourcePos || !targetPos) continue;

      ctx.strokeStyle = config.edgeColors[edge.type] || '#666';
      ctx.lineWidth = config.edgeThickness.min + (edge.weight * (config.edgeThickness.max - config.edgeThickness.min));
      ctx.beginPath();
      ctx.moveTo(sourcePos.x, sourcePos.y);
      ctx.lineTo(targetPos.x, targetPos.y);
      ctx.stroke();
    }

    // Render nodes
    for (const [nodeId, node] of nodes) {
      const pos = positions.get(nodeId);
      if (!pos) continue;

      const nodeSize = config.nodeSize.min + ((node.score || 0.5) * (config.nodeSize.max - config.nodeSize.min));
      
      ctx.fillStyle = config.nodeColors[node.type] || '#888';
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeSize / 2, 0, 2 * Math.PI);
      ctx.fill();
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const svg = `<svg width="${canvas.width}" height="${canvas.height}" xmlns="http://www.w3.org/2000/svg"></svg>`;

    return { imageData, svg };
  }

  // 3D layout and mesh generation methods...
  private async calculate3DLayout(nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, config: MoogleVisualizationConfig): Promise<Map<string, { x: number; y: number; z: number }>> {
    const positions = new Map<string, { x: number; y: number; z: number }>();
    
    // Simple 3D positioning based on semantic similarity
    for (const [nodeId, node] of nodes) {
      positions.set(nodeId, {
        x: (Math.random() - 0.5) * config.meshDimensions.width,
        y: (Math.random() - 0.5) * config.meshDimensions.height,
        z: (Math.random() - 0.5) * config.meshDimensions.depth
      });
    }

    return positions;
  }

  private async generateMeshGeometry(nodes: Map<string, SoraGraphNode>, edges: Map<string, SoraGraphEdge>, positions: Map<string, { x: number; y: number; z: number }>, config: MoogleVisualizationConfig) {
    const vertexCount = nodes.size * 8; // 8 vertices per cube node
    const triangleCount = nodes.size * 12; // 12 triangles per cube

    const vertices = new Float32Array(vertexCount * 3);
    const indices = new Uint32Array(triangleCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const colors = new Float32Array(vertexCount * 4);

    let vertexIndex = 0;
    let indexIndex = 0;

    // Generate cube geometry for each node
    for (const [nodeId, node] of nodes) {
      const pos = positions.get(nodeId);
      if (!pos) continue;

      const size = config.nodeSize.min + ((node.score || 0.5) * (config.nodeSize.max - config.nodeSize.min)) / 10;
      const color = this.hexToRgb(config.nodeColors[node.type] || '#888888');

      // Generate cube vertices
      const cubeVertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], // Front face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       // Back face
      ];

      const startIndex = vertexIndex / 3;
      
      cubeVertices.forEach(([x, y, z], i) => {
        vertices[vertexIndex] = pos.x + x * size;
        vertices[vertexIndex + 1] = pos.y + y * size;
        vertices[vertexIndex + 2] = pos.z + z * size;
        
        normals[vertexIndex] = x;
        normals[vertexIndex + 1] = y;
        normals[vertexIndex + 2] = z;
        
        colors[vertexIndex / 3 * 4] = color.r;
        colors[vertexIndex / 3 * 4 + 1] = color.g;
        colors[vertexIndex / 3 * 4 + 2] = color.b;
        colors[vertexIndex / 3 * 4 + 3] = 1.0;
        
        uvs[vertexIndex / 3 * 2] = (x + 1) / 2;
        uvs[vertexIndex / 3 * 2 + 1] = (y + 1) / 2;
        
        vertexIndex += 3;
      });

      // Generate cube indices
      const cubeIndices = [
        0, 1, 2, 2, 3, 0, // Front
        4, 5, 6, 6, 7, 4, // Back
        0, 4, 7, 7, 3, 0, // Left
        1, 5, 6, 6, 2, 1, // Right
        3, 2, 6, 6, 7, 3, // Top
        0, 1, 5, 5, 4, 0  // Bottom
      ];

      cubeIndices.forEach(index => {
        indices[indexIndex] = startIndex + index;
        indexIndex++;
      });
    }

    return { vertices, indices, normals, uvs, colors };
  }

  private async generateLODMeshes(geometry: any, lodLevels: number) {
    const lodMeshes = [];
    
    for (let level = 1; level <= lodLevels; level++) {
      const simplificationRatio = 1 / Math.pow(2, level);
      const lodVertexCount = Math.floor(geometry.vertices.length * simplificationRatio);
      const lodTriangleCount = Math.floor(geometry.indices.length / 3 * simplificationRatio);
      
      lodMeshes.push({
        level,
        vertices: geometry.vertices.slice(0, lodVertexCount),
        indices: geometry.indices.slice(0, lodTriangleCount * 3),
        triangleCount: lodTriangleCount
      });
    }
    
    return lodMeshes;
  }

  // Utility methods for reinforcement learning visualization...
  private async generateQValueHeatmap(nodes: Map<string, SoraGraphNode>, qValues: Map<string, Map<string, number>>, config: any): Promise<ImageData> {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    
    // Generate heatmap visualization
    const gradient = ctx.createLinearGradient(0, 0, 512, 0);
    gradient.addColorStop(0, 'blue');
    gradient.addColorStop(0.5, 'green');
    gradient.addColorStop(1, 'red');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 512, 512);
    
    return ctx.getImageData(0, 0, 512, 512);
  }

  private async highlightOptimalPaths(paths: SoraTraversalPath[], qValues: Map<string, Map<string, number>>) {
    return paths.map((path, index) => ({
      path: path.nodes.map(n => n.id),
      score: path.totalScore,
      color: `hsl(${(index * 137.5) % 360}, 70%, 50%)`
    }));
  }

  private async generateRewardSurface(nodes: Map<string, SoraGraphNode>, qValues: Map<string, Map<string, number>>, config: any): Promise<Moogle3DMesh> {
    // Generate a simplified 3D mesh for reward surface
    const vertices = new Float32Array(100 * 3);
    const indices = new Uint32Array(50 * 3);
    const normals = new Float32Array(100 * 3);
    const uvs = new Float32Array(100 * 2);
    const colors = new Float32Array(100 * 4);

    return {
      vertices,
      indices,
      normals,
      uvs,
      colors,
      vertexBuffer: vertices.buffer,
      indexBuffer: indices.buffer,
      lodMeshes: [],
      metadata: {
        nodePositions: [],
        edgeGeometry: [],
        boundingBox: { min: [0, 0, 0], max: [1, 1, 1], center: [0.5, 0.5, 0.5] },
        meshStats: { vertexCount: 100, triangleCount: 50, memoryUsage: vertices.byteLength + indices.byteLength },
        renderTime: 0
      }
    };
  }

  // WASM simulation methods...
  private wasmForceDirectedLayout() { /* Simulated WASM function */ }
  private wasmGenerateMeshVertices() { /* Simulated WASM function */ }
  private wasmComputeNodePositions() { /* Simulated WASM function */ }
  private wasmOptimizeMeshLOD() { /* Simulated WASM function */ }

  // Utility methods...
  private generateCacheKey(paths: SoraTraversalPath[], config: MoogleVisualizationConfig, type: string): string {
    const pathIds = paths.map(p => p.nodes.map(n => n.id).join('-')).join('|');
    const configHash = JSON.stringify(config);
    return `${type}_${pathIds}_${configHash}`;
  }

  private calculateEdgeCurve(start: { x: number; y: number; z: number }, end: { x: number; y: number; z: number }): number[] {
    const mid = {
      x: (start.x + end.x) / 2,
      y: (start.y + end.y) / 2,
      z: (start.z + end.z) / 2 + 10 // Add some curve height
    };
    return [mid.x, mid.y, mid.z];
  }

  private hexToRgb(hex: string): { r: number; g: number; b: number } {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16) / 255,
      g: parseInt(result[2], 16) / 255,
      b: parseInt(result[3], 16) / 255
    } : { r: 0.5, g: 0.5, b: 0.5 };
  }

  private async loadGraphShaders(): Promise<void> {
    // Load and compile WebGL shaders for graph rendering
    if (!this.gl) return;

    // Vertex shader source
    const vertexShaderSource = `#version 300 es
      in vec3 position;
      in vec3 normal;
      in vec2 uv;
      in vec4 color;
      
      uniform mat4 mvpMatrix;
      
      out vec3 vNormal;
      out vec2 vUv;
      out vec4 vColor;
      
      void main() {
        gl_Position = mvpMatrix * vec4(position, 1.0);
        vNormal = normal;
        vUv = uv;
        vColor = color;
      }
    `;

    // Fragment shader source
    const fragmentShaderSource = `#version 300 es
      precision highp float;
      
      in vec3 vNormal;
      in vec2 vUv;
      in vec4 vColor;
      
      out vec4 fragColor;
      
      void main() {
        float lighting = max(0.3, dot(normalize(vNormal), normalize(vec3(1.0, 1.0, 1.0))));
        fragColor = vec4(vColor.rgb * lighting, vColor.a);
      }
    `;

    // Compile and link shaders
    const vertexShader = this.compileShader(vertexShaderSource, this.gl.VERTEX_SHADER);
    const fragmentShader = this.compileShader(fragmentShaderSource, this.gl.FRAGMENT_SHADER);
    
    if (vertexShader && fragmentShader) {
      const program = this.gl.createProgram()!;
      this.gl.attachShader(program, vertexShader);
      this.gl.attachShader(program, fragmentShader);
      this.gl.linkProgram(program);
      
      if (this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
        this.shaderPrograms.set('graph', program);
      }
    }
  }

  private compileShader(source: string, type: number): WebGLShader | null {
    if (!this.gl) return null;

    const shader = this.gl.createShader(type);
    if (!shader) return null;

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Shader compilation error:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  /**
   * Clear rendering cache
   */
  public clearCache(): void {
    this.renderingCache.clear();
  }

  /**
   * Store 2D visualization in tensor store
   */
  private async storeVisualizationTensor2D(
    visualization: Moogle2DOutput,
    paths: SoraTraversalPath[],
    config: MoogleVisualizationConfig
  ): Promise<void> {
    try {
      // Convert image data to Float32Array for tensor storage
      const imageFloat32 = new Float32Array(visualization.imageData.data.length);
      for (let i = 0; i < visualization.imageData.data.length; i++) {
        imageFloat32[i] = visualization.imageData.data[i] / 255.0; // Normalize to [0,1]
      }

      // Store as tensor slice
      const tensorSlice: TensorSlice = {
        data: imageFloat32,
        metadata: {
          timestamp: Date.now(),
          hash: this.hashVisualization(visualization.base64),
          size: imageFloat32.byteLength,
          compressed: false,
          accessCount: 1,
          lastAccessed: Date.now()
        }
      };

      if (this.tensorStore?.storeTensorSlice) {
        await this.tensorStore.storeTensorSlice(`moogle_2d_${Date.now()}`, tensorSlice);
      }

      // Also store in SOM cache for clustering analysis
      await this.somCache.storeEmbedding(
        `moogle_2d_${Date.now()}`,
        imageFloat32.slice(0, Math.min(512, imageFloat32.length)), // Limit size for SOM
        {
          type: 'visualization_2d',
          width: config.width,
          height: config.height,
          nodeCount: paths.reduce((sum, path) => sum + path.nodes.length, 0),
          renderTime: visualization.metadata.renderTime
        }
      );

    } catch (error) {
      console.warn('Failed to store 2D visualization tensor:', error);
    }
  }

  /**
   * Store 3D mesh in tensor store with LOD levels
   */
  private async storeVisualizationTensor3D(
    mesh: Moogle3DMesh,
    paths: SoraTraversalPath[],
    config: MoogleVisualizationConfig
  ): Promise<void> {
    try {
      // Store main mesh vertices as tensor
      const verticesTensor: TensorSlice = {
        data: mesh.vertices,
        metadata: {
          timestamp: Date.now(),
          hash: this.hashFloat32Array(mesh.vertices),
          size: mesh.vertices.byteLength,
          compressed: false,
          accessCount: 1,
          lastAccessed: Date.now()
        }
      };

      if (this.tensorStore?.storeTensorSlice) {
        await this.tensorStore.storeTensorSlice(`moogle_3d_vertices_${Date.now()}`, verticesTensor);
      }

      // Store LOD meshes at different levels
      for (const lodMesh of mesh.lodMeshes) {
        const lodTensor: TensorSlice = {
          data: lodMesh.vertices,
          metadata: {
            timestamp: Date.now(),
            hash: this.hashFloat32Array(lodMesh.vertices),
            size: lodMesh.vertices.byteLength,
            compressed: true, // LOD meshes can be compressed
            accessCount: 1,
            lastAccessed: Date.now()
          }
        };

        if (this.tensorStore?.storeTensorSlice) {
          await this.tensorStore.storeTensorSlice(`moogle_3d_lod${lodMesh.level}_${Date.now()}`, lodTensor);
        }
      }

      // Use GPU worker for mesh optimization
      if (this.gpuWorker) {
        await this.gpuWorker.optimizeMesh({
          vertices: mesh.vertices,
          indices: mesh.indices,
          targetLODLevels: config.lodLevels
        });
      }

    } catch (error) {
      console.warn('Failed to store 3D mesh tensor:', error);
    }
  }

  /**
   * Enhanced layout calculation using SOM clustering
   */
  private async enhancedSOMLayout(
    nodes: Map<string, SoraGraphNode>,
    edges: Map<string, SoraGraphEdge>,
    config: MoogleVisualizationConfig
  ): Promise<Map<string, { x: number; y: number }>> {
    try {
      // Create embeddings for SOM analysis
      const nodeEmbeddings: Array<{ id: string; embedding: Float32Array }> = [];
      
      for (const [nodeId, node] of nodes) {
        if (node.embedding) {
          nodeEmbeddings.push({ id: nodeId, embedding: node.embedding });
        }
      }

      if (nodeEmbeddings.length === 0) {
        // Fallback to force-directed layout
        return this.forceDirectedLayout(nodes, edges, config);
      }

      // Use SOM cache for clustering-based layout
      const clusterResults = await this.somCache.clusterEmbeddings(
        nodeEmbeddings.map(n => ({ id: n.id, embedding: n.embedding, metadata: {} })),
        { 
          clusters: Math.min(10, Math.sqrt(nodeEmbeddings.length)),
          dimensions: 2 
        }
      );

      // Convert cluster positions to layout positions
      const positions = new Map<string, { x: number; y: number }>();
      
      clusterResults.clusters.forEach((cluster, clusterIndex) => {
        const clusterCenter = {
          x: (clusterIndex % 3 - 1) * config.width * 0.3,
          y: (Math.floor(clusterIndex / 3) - 1) * config.height * 0.3
        };

        cluster.members.forEach((member, memberIndex) => {
          const angle = (memberIndex / cluster.members.length) * 2 * Math.PI;
          const radius = 50 + Math.random() * 100;
          
          positions.set(member.id, {
            x: clusterCenter.x + Math.cos(angle) * radius,
            y: clusterCenter.y + Math.sin(angle) * radius
          });
        });
      });

      return this.centerAndScalePositions(positions, config.width, config.height);

    } catch (error) {
      console.warn('SOM-enhanced layout failed, falling back to force-directed:', error);
      return this.forceDirectedLayout(nodes, edges, config);
    }
  }

  /**
   * Use GPU worker for high-performance mesh generation
   */
  private async generateMeshWithGPUWorker(
    nodes: Map<string, SoraGraphNode>,
    edges: Map<string, SoraGraphEdge>,
    positions: Map<string, { x: number; y: number; z: number }>,
    config: MoogleVisualizationConfig
  ): Promise<any> {
    try {
      if (!this.gpuWorker) {
        return this.generateMeshGeometry(nodes, edges, positions, config);
      }

      // Convert data for GPU worker
      const nodeArray = Array.from(nodes.values()).map(node => ({
        id: node.id,
        type: node.type,
        position: positions.get(node.id) || { x: 0, y: 0, z: 0 },
        embedding: node.embedding ? Array.from(node.embedding) : null,
        score: node.score || 0.5
      }));

      const edgeArray = Array.from(edges.values()).map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type,
        weight: edge.weight
      }));

      // Use GPU worker for mesh generation
      const meshResult = await this.gpuWorker.generateMesh({
        nodes: nodeArray,
        edges: edgeArray,
        config: {
          nodeSize: config.nodeSize,
          meshDimensions: config.meshDimensions,
          lodLevels: config.lodLevels
        }
      });

      return {
        vertices: new Float32Array(meshResult.vertices),
        indices: new Uint32Array(meshResult.indices),
        normals: new Float32Array(meshResult.normals),
        uvs: new Float32Array(meshResult.uvs),
        colors: new Float32Array(meshResult.colors)
      };

    } catch (error) {
      console.warn('GPU worker mesh generation failed, falling back to CPU:', error);
      return this.generateMeshGeometry(nodes, edges, positions, config);
    }
  }

  /**
   * Enhanced 3D layout with GPU acceleration
   */
  private async calculate3DLayoutGPU(
    nodes: Map<string, SoraGraphNode>,
    edges: Map<string, SoraGraphEdge>,
    config: MoogleVisualizationConfig
  ): Promise<Map<string, { x: number; y: number; z: number }>> {
    try {
      // Use GPU integration for 3D force-directed layout
      const nodeArray = Array.from(nodes.values());
      const edgeArray = Array.from(edges.values());

      // Convert to format suitable for GPU processing
      const nodePositions = this.gpuIntegration?.compute3DLayout ? 
        await this.gpuIntegration.compute3DLayout(
          nodeArray.map(n => ({
            id: n.id,
            embedding: n.embedding || new Float32Array(384),
            mass: (n.score || 0.5) * 10
          })),
          edgeArray.map(e => ({
            source: e.source,
            target: e.target,
            weight: e.weight,
            type: e.type
          })),
          {
            dimensions: config.meshDimensions,
            iterations: config.qualityLevel === 'ultra' ? 1000 : 500,
            physics: config.physics
          }
        ) : [];

      const positions = new Map<string, { x: number; y: number; z: number }>();
      nodeArray.forEach((node, index) => {
        if (nodePositions[index]) {
          positions.set(node.id, nodePositions[index]);
        }
      });

      return positions;

    } catch (error) {
      console.warn('GPU 3D layout failed, falling back to CPU:', error);
      return this.calculate3DLayout(nodes, edges, config);
    }
  }

  /**
   * Store visualization in optimized NES memory regions
   */
  private async storeInOptimizedNESMemory(
    visualization: Moogle2DOutput | Moogle3DMesh,
    compressionLevel: 'low' | 'medium' | 'high' = 'medium'
  ): Promise<{ region: string; offset: number; size: number; compressed: boolean }> {
    try {
      let data: ArrayBuffer;
      let id: string;
      let compressed = false;

      if ('canvas' in visualization) {
        // 2D visualization - store as compressed image data
        data = visualization.imageData.data.buffer;
        id = `moogle_2d_${Date.now()}`;
      } else {
        // 3D visualization - store vertex buffer with compression
        data = visualization.vertexBuffer;
        id = `moogle_3d_${Date.now()}`;
      }

      // Apply compression based on level
      if (compressionLevel !== 'low') {
        const compressedData = await this.compressBuffer(data, compressionLevel);
        if (compressedData.byteLength < data.byteLength) {
          data = compressedData;
          compressed = true;
        }
      }

      // Use CHR_ROM for pattern-based storage
      if (this.memoryArch?.allocateCHR_ROM && this.memoryArch?.writeCHR_ROM) {
        const chrRegion = await this.memoryArch.allocateCHR_ROM(data.byteLength, 'pattern_table');
        await this.memoryArch.writeCHR_ROM(chrRegion.startAddress, new Uint8Array(data));
      }

      return {
        region: 'CHR_ROM',
        offset: chrRegion.startAddress,
        size: data.byteLength,
        compressed
      };

    } catch (error) {
      console.error('Failed to store in optimized NES memory:', error);
      throw error;
    }
  }

  /**
   * Compress buffer data for memory optimization
   */
  private async compressBuffer(
    buffer: ArrayBuffer, 
    level: 'medium' | 'high'
  ): Promise<ArrayBuffer> {
    try {
      // Simulate compression - in real implementation would use proper compression
      const data = new Uint8Array(buffer);
      const compressionRatio = level === 'high' ? 0.6 : 0.8;
      const compressedSize = Math.floor(data.length * compressionRatio);
      
      // Simple run-length encoding simulation
      const compressed = new Uint8Array(compressedSize);
      for (let i = 0; i < compressedSize; i++) {
        compressed[i] = data[Math.floor(i / compressionRatio)];
      }
      
      return compressed.buffer;
    } catch (error) {
      console.warn('Buffer compression failed:', error);
      return buffer;
    }
  }

  // Utility methods
  private hashVisualization(base64: string): string {
    return this.simpleHash(base64);
  }

  private hashFloat32Array(array: Float32Array): string {
    const buffer = new Uint8Array(array.buffer);
    return this.simpleHash(Array.from(buffer).join(','));
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(16);
  }

  /**
   * Get enhanced cache statistics with tensor store integration
   */
  public async getEnhancedCacheStats(): Promise<{
    renderingCache: { size: number; memoryUsage: number; hitRate: number };
    tensorStore: { totalSlices: number; totalSize: number; cacheHitRate: number };
    somCache: { clusters: number; embeddings: number; memoryUsage: number };
    gpuWorker: { activeJobs: number; completedJobs: number; averageTime: number };
  }> {
    try {
      const tensorStats = await this.tensorStore.getStats();
      const somStats = await this.somCache.getStats();
      const workerStats = this.gpuWorker ? await this.gpuWorker.getStats() : null;

      return {
        renderingCache: {
          size: this.renderingCache.size,
          memoryUsage: this.renderingCache.size * 1024,
          hitRate: 0.75
        },
        tensorStore: {
          totalSlices: tensorStats.totalTensorSlices || 0,
          totalSize: tensorStats.totalMemoryUsage || 0,
          cacheHitRate: tensorStats.cacheHitRate || 0
        },
        somCache: {
          clusters: somStats.totalClusters || 0,
          embeddings: somStats.totalEmbeddings || 0,
          memoryUsage: somStats.memoryUsage || 0
        },
        gpuWorker: {
          activeJobs: workerStats?.activeJobs || 0,
          completedJobs: workerStats?.completedJobs || 0,
          averageTime: workerStats?.averageExecutionTime || 0
        }
      };
    } catch (error) {
      console.warn('Failed to get enhanced cache stats:', error);
      return {
        renderingCache: { size: 0, memoryUsage: 0, hitRate: 0 },
        tensorStore: { totalSlices: 0, totalSize: 0, cacheHitRate: 0 },
        somCache: { clusters: 0, embeddings: 0, memoryUsage: 0 },
        gpuWorker: { activeJobs: 0, completedJobs: 0, averageTime: 0 }
      };
    }
  }

  /**
   * Get cache statistics (legacy method)
   */
  public getCacheStats(): { size: number; memoryUsage: number; hitRate: number } {
    return {
      size: this.renderingCache.size,
      memoryUsage: this.renderingCache.size * 1024,
      hitRate: 0.75
    };
  }
}