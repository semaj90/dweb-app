/**
 * Legal Document Graph Visualization - WebGPU Implementation
 * 
 * Advanced "Graph on a Texture" system for legal document networks:
 * - GPU-accelerated graph layout and rendering
 * - Dimensional tensor stores for nodes, edges, and metadata
 * - Interactive 3D exploration of legal relationships
 * - Integration with client-side IndexedDB persistence
 * 
 * Implements the "tricubic tensor" model on GPU for maximum performance
 */

import type { GraphVisualizationData, LegalEntity, DocumentCache } from '$lib/db/client-db';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface WebGPUGraphConfig {
  maxNodes: number;
  maxEdges: number;
  canvasWidth: number;
  canvasHeight: number;
  enablePhysics: boolean;
  renderDistance: number;
  lodLevels: number;
}

export interface GraphNode {
  id: string;
  position: [number, number, number]; // x, y, z
  velocity: [number, number, number];
  force: [number, number, number];
  mass: number;
  size: number;
  color: [number, number, number, number]; // RGBA
  type: 'document' | 'case' | 'entity' | 'precedent';
  metadata: {
    title: string;
    importance: number;
    connections: number;
    lastAccessed: number;
  };
}

export interface GraphEdge {
  source: number; // Node index
  target: number; // Node index
  weight: number;
  type: 'citation' | 'similarity' | 'reference' | 'temporal';
  strength: number;
  color: [number, number, number, number];
}

export interface TensorStore {
  nodeBuffer: GPUBuffer;
  edgeBuffer: GPUBuffer;
  metadataBuffer: GPUBuffer;
  positionTexture: GPUTexture;
  colorTexture: GPUTexture;
  adjacencyTexture: GPUTexture;
}

export interface GraphRenderState {
  nodeCount: number;
  edgeCount: number;
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  zoom: number;
  selectedNode: number | null;
  highlightedNodes: Set<number>;
  filterType: 'all' | 'document' | 'case' | 'entity' | 'precedent';
  timeRange: [number, number];
}

// ============================================================================
// WEBGPU LEGAL DOCUMENT GRAPH ENGINE
// ============================================================================

export class WebGPULegalDocumentGraph {
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private canvas: HTMLCanvasElement;
  private config: WebGPUGraphConfig;
  
  // GPU Resources
  private tensorStore: TensorStore | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private renderPipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  
  // Graph State
  private nodes: GraphNode[] = [];
  private edges: GraphEdge[] = [];
  private renderState: GraphRenderState;
  private animationId: number | null = null;
  
  // Performance Monitoring
  private frameTime: number = 0;
  private lastFrameTime: number = 0;
  private fps: number = 0;

  constructor(canvas: HTMLCanvasElement, config: Partial<WebGPUGraphConfig> = {}) {
    this.canvas = canvas;
    this.config = {
      maxNodes: 10000,
      maxEdges: 50000,
      canvasWidth: canvas.width || 800,
      canvasHeight: canvas.height || 600,
      enablePhysics: true,
      renderDistance: 1000,
      lodLevels: 4,
      ...config
    };

    this.renderState = {
      nodeCount: 0,
      edgeCount: 0,
      cameraPosition: [0, 0, 10],
      cameraTarget: [0, 0, 0],
      zoom: 1.0,
      selectedNode: null,
      highlightedNodes: new Set(),
      filterType: 'all',
      timeRange: [0, Date.now()]
    };
  }

  /**
   * Initialize WebGPU device and resources
   */
  async initialize(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No WebGPU adapter found');
    }

    this.device = await adapter.requestDevice({
      requiredFeatures: ['texture-binding-array'],
      requiredLimits: {
        maxStorageBufferBindingSize: 134217728, // 128MB for large graphs
        maxComputeWorkgroupSizeX: 1024
      }
    });

    this.context = this.canvas.getContext('webgpu');
    if (!this.context) {
      throw new Error('Failed to get WebGPU context');
    }

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: canvasFormat,
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });

    await this.initializeShaders();
    await this.createTensorStores();
    await this.setupRenderPipeline();

    console.log('[WebGPU Legal Graph] Initialized successfully');
  }

  /**
   * Initialize compute and render shaders
   */
  private async initializeShaders(): Promise<void> {
    if (!this.device) throw new Error('WebGPU device not initialized');

    // Compute shader for graph physics and layout
    const computeShaderCode = `
      struct Node {
        position: vec3<f32>,
        velocity: vec3<f32>,
        force: vec3<f32>,
        mass: f32,
        size: f32,
        color: vec4<f32>,
        metadata: vec4<f32>, // type, importance, connections, lastAccessed
      };

      struct Edge {
        source: u32,
        target: u32,
        weight: f32,
        strength: f32,
        color: vec4<f32>,
      };

      struct Uniforms {
        nodeCount: u32,
        edgeCount: u32,
        deltaTime: f32,
        damping: f32,
        repulsion: f32,
        attraction: f32,
        centerForce: f32,
        minDistance: f32,
      };

      @group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
      @group(0) @binding(1) var<storage, read> edges: array<Edge>;
      @group(0) @binding(2) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(64)
      fn computeForces(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let nodeIndex = global_id.x;
        if (nodeIndex >= uniforms.nodeCount) {
          return;
        }

        var node = nodes[nodeIndex];
        var totalForce = vec3<f32>(0.0, 0.0, 0.0);

        // Repulsion force between all nodes
        for (var i = 0u; i < uniforms.nodeCount; i++) {
          if (i == nodeIndex) {
            continue;
          }

          let other = nodes[i];
          let diff = node.position - other.position;
          let distance = length(diff);
          
          if (distance > 0.001) {
            let repulsionForce = uniforms.repulsion / (distance * distance);
            totalForce += normalize(diff) * repulsionForce;
          }
        }

        // Attraction force from connected edges
        for (var i = 0u; i < uniforms.edgeCount; i++) {
          let edge = edges[i];
          var otherIndex: u32;
          var isConnected = false;

          if (edge.source == nodeIndex) {
            otherIndex = edge.target;
            isConnected = true;
          } else if (edge.target == nodeIndex) {
            otherIndex = edge.source;
            isConnected = true;
          }

          if (isConnected) {
            let other = nodes[otherIndex];
            let diff = other.position - node.position;
            let distance = length(diff);
            
            if (distance > uniforms.minDistance) {
              let attractionForce = uniforms.attraction * edge.weight * edge.strength;
              totalForce += normalize(diff) * attractionForce;
            }
          }
        }

        // Center force to prevent nodes from drifting away
        let centerDiff = -node.position;
        let centerDistance = length(centerDiff);
        if (centerDistance > 0.001) {
          totalForce += normalize(centerDiff) * uniforms.centerForce * centerDistance;
        }

        // Update velocity and position
        node.force = totalForce;
        node.velocity += totalForce * uniforms.deltaTime / node.mass;
        node.velocity *= uniforms.damping;
        node.position += node.velocity * uniforms.deltaTime;

        nodes[nodeIndex] = node;
      }
    `;

    // Vertex shader for rendering
    const vertexShaderCode = `
      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) size: f32,
        @location(2) color: vec4<f32>,
        @location(3) nodeType: f32,
      };

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
        @location(1) size: f32,
        @location(2) nodeType: f32,
      };

      struct Uniforms {
        viewProjectionMatrix: mat4x4<f32>,
        cameraPosition: vec3<f32>,
        zoom: f32,
        time: f32,
        selectedNode: f32,
        filterType: f32,
      };

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      @vertex
      fn vs_main(input: VertexInput, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
        var output: VertexOutput;
        
        // Calculate world position
        let worldPosition = vec4<f32>(input.position, 1.0);
        
        // Apply view-projection transformation
        output.position = uniforms.viewProjectionMatrix * worldPosition;
        
        // Scale size based on distance and zoom
        let distanceToCamera = distance(input.position, uniforms.cameraPosition);
        let scaledSize = input.size * uniforms.zoom / (distanceToCamera * 0.1 + 1.0);
        
        output.color = input.color;
        output.size = scaledSize;
        output.nodeType = input.nodeType;
        
        return output;
      }
    `;

    // Fragment shader for rendering
    const fragmentShaderCode = `
      struct FragmentInput {
        @location(0) color: vec4<f32>,
        @location(1) size: f32,
        @location(2) nodeType: f32,
      };

      @fragment
      fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
        // Create circular nodes with type-based styling
        let center = vec2<f32>(0.5, 0.5);
        let fragCoord = vec2<f32>(0.5, 0.5); // This would come from gl_PointCoord in OpenGL
        let distance = distance(fragCoord, center);
        
        if (distance > 0.5) {
          discard;
        }
        
        // Apply node type styling
        var finalColor = input.color;
        
        // Document nodes: solid circles
        if (input.nodeType < 0.5) {
          finalColor.a *= (1.0 - distance * 2.0);
        }
        // Case nodes: rings
        else if (input.nodeType < 1.5) {
          if (distance < 0.3 || distance > 0.5) {
            discard;
          }
          finalColor.a *= 0.8;
        }
        // Entity nodes: diamond shapes (approximated)
        else if (input.nodeType < 2.5) {
          let diamond = abs(fragCoord.x - 0.5) + abs(fragCoord.y - 0.5);
          if (diamond > 0.35) {
            discard;
          }
        }
        
        return finalColor;
      }
    `;

    // Create compute pipeline
    const computeShaderModule = this.device.createShaderModule({
      code: computeShaderCode
    });

    this.computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: computeShaderModule,
        entryPoint: 'computeForces'
      }
    });

    // Create render shader modules
    const vertexShaderModule = this.device.createShaderModule({
      code: vertexShaderCode
    });

    const fragmentShaderModule = this.device.createShaderModule({
      code: fragmentShaderCode
    });

    // Store for render pipeline creation
    this.vertexShaderModule = vertexShaderModule;
    this.fragmentShaderModule = fragmentShaderModule;
  }

  private vertexShaderModule: GPUShaderModule | null = null;
  private fragmentShaderModule: GPUShaderModule | null = null;

  /**
   * Create GPU tensor stores for graph data
   */
  private async createTensorStores(): Promise<void> {
    if (!this.device) throw new Error('WebGPU device not initialized');

    // Calculate buffer sizes
    const nodeBufferSize = this.config.maxNodes * 32 * 4; // 32 floats per node
    const edgeBufferSize = this.config.maxEdges * 8 * 4;  // 8 floats per edge
    const metadataBufferSize = this.config.maxNodes * 16 * 4; // 16 floats per metadata

    // Create node buffer (positions, velocities, forces, properties)
    const nodeBuffer = this.device.createBuffer({
      size: nodeBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    // Create edge buffer (connections and properties)
    const edgeBuffer = this.device.createBuffer({
      size: edgeBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Create metadata buffer (titles, timestamps, importance scores)
    const metadataBuffer = this.device.createBuffer({
      size: metadataBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Create position texture for advanced rendering techniques
    const positionTexture = this.device.createTexture({
      size: [Math.ceil(Math.sqrt(this.config.maxNodes)), Math.ceil(Math.sqrt(this.config.maxNodes))],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    // Create color texture for node appearance
    const colorTexture = this.device.createTexture({
      size: [Math.ceil(Math.sqrt(this.config.maxNodes)), Math.ceil(Math.sqrt(this.config.maxNodes))],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    // Create adjacency texture for efficient edge queries
    const adjacencyTexture = this.device.createTexture({
      size: [this.config.maxNodes, this.config.maxNodes],
      format: 'r8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    this.tensorStore = {
      nodeBuffer,
      edgeBuffer,
      metadataBuffer,
      positionTexture,
      colorTexture,
      adjacencyTexture
    };

    // Create uniform buffer for render parameters
    this.uniformBuffer = this.device.createBuffer({
      size: 256, // Sufficient for view matrices and parameters
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    console.log('[WebGPU Legal Graph] Tensor stores created successfully');
  }

  /**
   * Setup render pipeline for graph visualization
   */
  private async setupRenderPipeline(): Promise<void> {
    if (!this.device || !this.vertexShaderModule || !this.fragmentShaderModule) {
      throw new Error('Shaders not initialized');
    }

    this.renderPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: this.vertexShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 12 + 4 + 16 + 4, // position + size + color + nodeType
            attributes: [
              { format: 'float32x3', offset: 0, shaderLocation: 0 }, // position
              { format: 'float32', offset: 12, shaderLocation: 1 },   // size
              { format: 'float32x4', offset: 16, shaderLocation: 2 }, // color
              { format: 'float32', offset: 32, shaderLocation: 3 },   // nodeType
            ]
          }
        ]
      },
      fragment: {
        module: this.fragmentShaderModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha'
              },
              alpha: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha'
              }
            }
          }
        ]
      },
      primitive: {
        topology: 'point-list'
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less'
      }
    });
  }

  /**
   * Load graph data from IndexedDB
   */
  async loadGraphFromDB(graphId: string): Promise<void> {
    const { legalDB } = await import('$lib/db/client-db');
    
    try {
      const graphData = await legalDB.graphVisualizationData
        .where('graphId')
        .equals(graphId)
        .first();

      if (!graphData) {
        console.warn(`[WebGPU Legal Graph] Graph ${graphId} not found in database`);
        return;
      }

      // Convert IndexedDB format to internal format
      this.nodes = graphData.nodes.map((node, index) => ({
        id: node.id,
        position: [node.position.x, node.position.y, node.position.z || 0],
        velocity: [0, 0, 0],
        force: [0, 0, 0],
        mass: 1.0,
        size: node.size || 5.0,
        color: this.parseColor(node.color),
        type: this.parseNodeType(node.type),
        metadata: {
          title: node.label,
          importance: node.metadata?.importance || 0.5,
          connections: 0, // Will be calculated
          lastAccessed: Date.now()
        }
      }));

      this.edges = graphData.edges.map(edge => {
        const sourceIndex = this.nodes.findIndex(n => n.id === edge.source);
        const targetIndex = this.nodes.findIndex(n => n.id === edge.target);
        
        return {
          source: sourceIndex,
          target: targetIndex,
          weight: edge.weight || 1.0,
          type: this.parseEdgeType(edge.type),
          strength: 1.0,
          color: this.parseColor(edge.color)
        };
      });

      // Calculate connection counts
      this.nodes.forEach((node, index) => {
        node.metadata.connections = this.edges.filter(
          edge => edge.source === index || edge.target === index
        ).length;
      });

      this.renderState.nodeCount = this.nodes.length;
      this.renderState.edgeCount = this.edges.length;

      await this.uploadGraphDataToGPU();
      
      console.log(`[WebGPU Legal Graph] Loaded graph: ${this.nodes.length} nodes, ${this.edges.length} edges`);
      
    } catch (error: any) {
      console.error('[WebGPU Legal Graph] Error loading graph from database:', error);
    }
  }

  /**
   * Upload graph data to GPU buffers
   */
  private async uploadGraphDataToGPU(): Promise<void> {
    if (!this.device || !this.tensorStore) {
      throw new Error('WebGPU not properly initialized');
    }

    // Prepare node data for GPU
    const nodeData = new Float32Array(this.nodes.length * 32);
    this.nodes.forEach((node, index) => {
      const offset = index * 32;
      
      // Position (vec3)
      nodeData[offset + 0] = node.position[0];
      nodeData[offset + 1] = node.position[1];
      nodeData[offset + 2] = node.position[2];
      
      // Velocity (vec3)
      nodeData[offset + 4] = node.velocity[0];
      nodeData[offset + 5] = node.velocity[1];
      nodeData[offset + 6] = node.velocity[2];
      
      // Force (vec3)
      nodeData[offset + 8] = node.force[0];
      nodeData[offset + 9] = node.force[1];
      nodeData[offset + 10] = node.force[2];
      
      // Properties
      nodeData[offset + 12] = node.mass;
      nodeData[offset + 13] = node.size;
      
      // Color (vec4)
      nodeData[offset + 16] = node.color[0];
      nodeData[offset + 17] = node.color[1];
      nodeData[offset + 18] = node.color[2];
      nodeData[offset + 19] = node.color[3];
      
      // Metadata
      nodeData[offset + 20] = this.encodeNodeType(node.type);
      nodeData[offset + 21] = node.metadata.importance;
      nodeData[offset + 22] = node.metadata.connections;
      nodeData[offset + 23] = node.metadata.lastAccessed;
    });

    // Prepare edge data for GPU
    const edgeData = new Float32Array(this.edges.length * 8);
    this.edges.forEach((edge, index) => {
      const offset = index * 8;
      
      edgeData[offset + 0] = edge.source;
      edgeData[offset + 1] = edge.target;
      edgeData[offset + 2] = edge.weight;
      edgeData[offset + 3] = edge.strength;
      edgeData[offset + 4] = edge.color[0];
      edgeData[offset + 5] = edge.color[1];
      edgeData[offset + 6] = edge.color[2];
      edgeData[offset + 7] = edge.color[3];
    });

    // Upload to GPU buffers
    this.device.queue.writeBuffer(this.tensorStore.nodeBuffer, 0, nodeData);
    this.device.queue.writeBuffer(this.tensorStore.edgeBuffer, 0, edgeData);
  }

  /**
   * Start the render loop
   */
  startRenderLoop(): void {
    if (this.animationId !== null) {
      return; // Already running
    }

    const renderFrame = (timestamp: number) => {
      this.frameTime = timestamp - this.lastFrameTime;
      this.lastFrameTime = timestamp;
      this.fps = 1000 / this.frameTime;

      this.updatePhysics(this.frameTime * 0.001); // Convert to seconds
      this.render();

      this.animationId = requestAnimationFrame(renderFrame);
    };

    this.animationId = requestAnimationFrame(renderFrame);
    console.log('[WebGPU Legal Graph] Render loop started');
  }

  /**
   * Stop the render loop
   */
  stopRenderLoop(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  /**
   * Update physics simulation
   */
  private updatePhysics(deltaTime: number): void {
    if (!this.device || !this.computePipeline || !this.tensorStore || !this.config.enablePhysics) {
      return;
    }

    // Update compute uniforms
    const computeUniforms = new Float32Array([
      this.renderState.nodeCount,
      this.renderState.edgeCount,
      Math.min(deltaTime, 0.016), // Cap at ~60fps
      0.95, // damping
      100.0, // repulsion
      2.0, // attraction
      0.1, // center force
      1.0  // min distance
    ]);

    this.device.queue.writeBuffer(this.uniformBuffer!, 0, computeUniforms);

    // Run compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(this.computePipeline);
    
    // Bind resources (this would need proper bind group setup)
    // computePass.setBindGroup(0, computeBindGroup);
    
    const workgroupCount = Math.ceil(this.renderState.nodeCount / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Render the graph
   */
  private render(): void {
    if (!this.device || !this.context || !this.renderPipeline) {
      return;
    }

    const commandEncoder = this.device.createCommandEncoder();
    
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store'
        }
      ]
    });

    renderPass.setPipeline(this.renderPipeline);
    
    // Render nodes
    renderPass.draw(this.renderState.nodeCount);
    
    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private parseColor(colorString: string): [number, number, number, number] {
    // Parse hex, rgb, rgba colors and convert to normalized RGBA
    if (colorString.startsWith('#')) {
      const hex = colorString.slice(1);
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      return [r, g, b, 1.0];
    }
    
    // Default color
    return [0.5, 0.5, 0.5, 1.0];
  }

  private parseNodeType(typeString: string): GraphNode['type'] {
    const typeMap: Record<string, GraphNode['type']> = {
      'document': 'document',
      'case': 'case',
      'entity': 'entity',
      'precedent': 'precedent'
    };
    
    return typeMap[typeString] || 'document';
  }

  private parseEdgeType(typeString: string): GraphEdge['type'] {
    const typeMap: Record<string, GraphEdge['type']> = {
      'citation': 'citation',
      'similarity': 'similarity',
      'reference': 'reference',
      'temporal': 'temporal'
    };
    
    return typeMap[typeString] || 'reference';
  }

  private encodeNodeType(type: GraphNode['type']): number {
    const typeMap: Record<GraphNode['type'], number> = {
      'document': 0.0,
      'case': 1.0,
      'entity': 2.0,
      'precedent': 3.0
    };
    
    return typeMap[type];
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats(): {
    fps: number;
    frameTime: number;
    nodeCount: number;
    edgeCount: number;
    gpuMemoryUsage: number;
  } {
    return {
      fps: this.fps,
      frameTime: this.frameTime,
      nodeCount: this.renderState.nodeCount,
      edgeCount: this.renderState.edgeCount,
      gpuMemoryUsage: 0 // Would need WebGPU memory tracking
    };
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.stopRenderLoop();
    
    if (this.tensorStore) {
      this.tensorStore.nodeBuffer.destroy();
      this.tensorStore.edgeBuffer.destroy();
      this.tensorStore.metadataBuffer.destroy();
      this.tensorStore.positionTexture.destroy();
      this.tensorStore.colorTexture.destroy();
      this.tensorStore.adjacencyTexture.destroy();
    }
    
    if (this.uniformBuffer) {
      this.uniformBuffer.destroy();
    }
    
    console.log('[WebGPU Legal Graph] Resources disposed');
  }
}