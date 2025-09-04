
/**
 * Self-Organizing Neural Sprite Engine
 * NES-inspired rapid state switching with AI prediction and multi-core processing
 *
 * Core concept: Pre-computed JSON states stored as "sprites" in Loki.js,
 * with local LLM predicting next states for ultra-fast animations
 * Enhanced with Self-Organizing Maps (SOM) and multi-core worker orchestration
 */

import Loki from "lokijs";
import { writable, derived, type Readable } from "svelte/store";
import { fabric } from "fabric";
import { BrowserCacheManager } from "./browser-cache-manager";
import { ShaderCache } from "./webgl-shader-cache";
import { MatrixTransformLib } from "./matrix-transform-lib";

// Types for our "sprite sheet" system
export interface CanvasSprite {
  id: string;
  name: string; // e.g., 'idle', 'text_creation', 'object_move'
  sequence: number; // Frame number in animation sequence
  jsonState: string; // Serialized Fabric.js canvas state
  metadata: {
    objects: number;
    complexity: number;
    duration?: number; // How long this frame should display
    triggers?: string[]; // What user actions can trigger this state
  };
  embedding?: number[]; // Vector embedding for AI similarity search
  createdAt: number;
  usageCount: number;
  predictedNext?: string[]; // AI-predicted likely next states
}

export interface UserActivity {
  id: string;
  action: string;
  context: Record<string, any>;
  timestamp: number;
  canvasState?: string; // Current canvas state when action occurred
  sequence: number; // Position in user's action sequence
}

export interface AnimationSequence {
  id: string;
  name: string;
  frames: string[]; // Array of sprite IDs
  loop: boolean;
  fps: number;
  triggers: string[];
  confidence: number; // AI confidence this sequence will be used
}

// Self-Organizing Map Node
export interface SOMNode {
  id: string;
  position: { x: number; y: number };
  weights: Float32Array;
  activationHistory: number[];
  connectedSprites: string[];
  learningRate: number;
  neighborhoodRadius: number;
}

// Multi-core Processing Task
export interface ProcessingTask {
  id: string;
  type: "sprite_analysis" | "som_update" | "prediction" | "optimization";
  priority: "low" | "medium" | "high" | "critical";
  data: any;
  workerId?: number;
  startTime?: number;
  completionTime?: number;
  result?: unknown;
}

// Performance Analytics
export interface PerformanceMetrics {
  frameRate: number;
  cacheHitRate: number;
  workerUtilization: number;
  somConvergence: number;
  memoryUsage: number;
  cpuUsage: number;
  predictiveAccuracy: number;
  selfOrganizationEfficiency: number;
}

// Main Self-Organizing Neural Sprite Engine class
export class NeuralSpriteEngine {
  private db: Loki;
  private sprites: any;
  private activities: any;
  private sequences: any;
  private canvas: fabric.Canvas;
  private aiWorker?: Worker;

  // Self-Organizing Map components
  private somNodes: Map<string, SOMNode> = new Map();
  private somGridSize: { width: number; height: number } = {
    width: 10,
    height: 10,
  };
  private globalLearningRate = 0.1;
  private neighborhoodDecay = 0.95;

  // Multi-core processing system
  private workerPool: Worker[] = [];
  private taskQueue: ProcessingTask[] = [];
  private activeWorkers = 0;
  private completedTasks: Map<string, any> = new Map();
  private maxWorkers: number;

  // Enhanced caching systems
  private shaderCache: ShaderCache;
  private browserCache: BrowserCacheManager;
  private matrixLib: MatrixTransformLib;
  private webglContext?: WebGL2RenderingContext;

  // Stores for reactive Svelte integration
  public currentState = writable<string>("idle");
  public isAnimating = writable<boolean>(false);
  public cacheHitRate = writable<number>(1.0);
  public predictedStates = writable<string[]>([]);

  // Enhanced stores for self-organizing features
  public somVisualization = writable<{
    nodes: Array<{ x: number; y: number; activation: number; sprites: number }>;
    connections: Array<{ from: string; to: string; strength: number }>;
  }>({ nodes: [], connections: [] });

  public multiCoreMetrics = writable<{
    activeWorkers: number;
    queueDepth: number;
    tasksPerSecond: number;
    averageTaskTime: number;
    cpuUtilization: number;
  }>({
    activeWorkers: 0,
    queueDepth: 0,
    tasksPerSecond: 0,
    averageTaskTime: 0,
    cpuUtilization: 0,
  });

  public selfOrganizationStats = writable<{
    convergenceRate: number;
    adaptationSpeed: number;
    clusterFormation: number;
    neuralEfficiency: number;
  }>({
    convergenceRate: 0,
    adaptationSpeed: 0,
    clusterFormation: 0,
    neuralEfficiency: 0,
  });

  // Performance metrics (NES-inspired)
  private startTime: number;
  private frameCount = 0;
  private cacheHits = 0;
  private cacheMisses = 0;

  constructor(canvas: fabric.Canvas, maxWorkers?: number) {
    this.canvas = canvas;
    this.maxWorkers = maxWorkers || navigator.hardwareConcurrency || 4;

    this.initializeDatabase();
    this.initializeAIWorker();
    this.initializeSelfOrganizingMap();
    this.initializeMultiCoreProcessing();
    this.setupPerformanceMonitoring();
    this.initializeEnhancedCaching();
  }

  private initializeDatabase(): void {
    this.db = new Loki("neural-sprite-cache.db", {
      autoload: true,
      autoloadCallback: this.databaseInitialize.bind(this),
      autosave: true,
      autosaveInterval: 4000,
    } as any);
  }

  private databaseInitialize(): void {
    // Initialize collections if they don't exist
    this.sprites =
      this.db.getCollection<CanvasSprite>("sprites") ||
      this.db.addCollection<CanvasSprite>("sprites", { indices: ["name", "sequence"] });

    this.activities =
      this.db.getCollection<UserActivity>("activities") ||
      this.db.addCollection<UserActivity>("activities", { indices: ["action", "timestamp"] });

    this.sequences =
      this.db.getCollection<AnimationSequence>("sequences") ||
      this.db.addCollection<AnimationSequence>("sequences", { indices: ["name", "confidence"] });

    // Load default "idle" state if database is empty
    if (this.sprites.count() === 0) {
      this.createDefaultSprites();
    }
  }

  private initializeAIWorker(): void {
    if (typeof Worker !== "undefined") {
      this.aiWorker = new Worker("/workers/neural-predictor.js");
      this.aiWorker.onmessage = this.handleAIWorkerMessage.bind(this);
    }
  }

  private setupPerformanceMonitoring(): void {
    // Enhanced performance monitoring with self-organizing features
    setInterval(() => {
      const hitRate = this.cacheHits / (this.cacheHits + this.cacheMisses);
      this.cacheHitRate.set(isNaN(hitRate) ? 1.0 : hitRate);

      // Update multi-core and SOM metrics
      this.updateMultiCoreMetrics();
      this.updateSOMVisualization();
      this.updateSelfOrganizationStats();

      // Reset counters every second (like NES frame counting)
      this.frameCount = 0;
    }, 1000);
  }

  private initializeEnhancedCaching(): void {
    // Initialize WebGL2 context for shader caching
    const canvasElement = this.canvas.getElement();
    this.webglContext = canvasElement.getContext("webgl2", {
      preserveDrawingBuffer: true,
      powerPreference: "high-performance", // Prefer dedicated GPU
      antialias: false, // Disable for performance
      alpha: false,
    }) as WebGL2RenderingContext;

    // Initialize NVIDIA shader cache (WebGL2 program caching)
    this.shaderCache = new ShaderCache(this.webglContext, {
      enableNVIDIAOptimizations: true,
      cacheSize: 50, // Cache up to 50 compiled shader programs
      persistToDisk: true, // Use browser storage for shader persistence
    });

    // Initialize browser cache manager for sprite JSON
    this.browserCache = new BrowserCacheManager({
      cachePrefix: "neural-sprite-",
      maxCacheSize: 100 * 1024 * 1024, // 100MB cache limit
      enableCompression: true,
      enableServiceWorkerIntegration: true,
    });

    // Initialize lightweight matrix transform library (10kb)
    this.matrixLib = new MatrixTransformLib({
      enableGPUAcceleration: true,
      optimizeForCSS: true,
      cacheTransforms: true,
    });
  }

  /**
   * Initialize Self-Organizing Map for sprite clustering
   */
  private initializeSelfOrganizingMap(): void {
    console.log("🧠 Initializing Self-Organizing Map...");

    // Create SOM grid
    for (let y = 0; y < this.somGridSize.height; y++) {
      for (let x = 0; x < this.somGridSize.width; x++) {
        const nodeId = `som_${x}_${y}`;
        const node: SOMNode = {
          id: nodeId,
          position: { x, y },
          weights: new Float32Array(16), // 16-dimensional feature space
          activationHistory: [],
          connectedSprites: [],
          learningRate: this.globalLearningRate,
          neighborhoodRadius:
            Math.min(this.somGridSize.width, this.somGridSize.height) / 2,
        };

        // Initialize weights randomly
        for (let i = 0; i < node.weights.length; i++) {
          node.weights[i] = Math.random() * 0.5 - 0.25;
        }

        this.somNodes.set(nodeId, node);
      }
    }

    console.log(`✅ SOM initialized with ${this.somNodes.size} nodes`);
  }

  /**
   * Initialize multi-core processing worker pool
   */
  private initializeMultiCoreProcessing(): void {
    console.log(`🔧 Initializing ${this.maxWorkers} worker threads...`);

    // Create worker code for sprite processing
    const workerCode = `
      // Multi-core Sprite Processing Worker
      let isInitialized = false;

      function initializeWorker() {
        if (isInitialized) return;

        // Worker-specific initialization
        console.log('Worker initialized');
        isInitialized = true;
      }

      function processSprite(sprite) {
        // Extract features from sprite for SOM processing
        const features = new Float32Array(16);

        try {
          const state = JSON.parse(sprite.jsonState);
          const objects = state.objects || [];

          // Feature extraction
          features[0] = objects.length / 10; // Object count normalized
          features[1] = sprite.metadata.complexity / 100; // Complexity
          features[2] = sprite.usageCount / 10; // Usage frequency
          features[3] = Math.random(); // Mock feature

          // Document type encoding
          const docTypes = ['contract', 'case_law', 'evidence', 'statute', 'memo'];
          const typeIndex = docTypes.indexOf(sprite.documentType || 'contract');
          features[4] = typeIndex / docTypes.length;

          // Fill remaining features with processed data
          for (let i = 5; i < 16; i++) {
            features[i] = Math.sin(i * sprite.metadata.complexity * 0.01);
          }

        } catch (error: any) {
          // Fallback features
          for (let i = 0; i < 16; i++) {
            features[i] = Math.random() * 0.5;
          }
        }

        return features;
      }

      function updateSOMNode(node, inputVector, learningRate, neighborhoodRadius) {
        // Calculate distance from node
        let distance = 0;
        for (let i = 0; i < node.weights.length; i++) {
          const diff = inputVector[i] - node.weights[i];
          distance += diff * diff;
        }
        distance = Math.sqrt(distance);

        // Calculate neighborhood influence
        const influence = Math.exp(-(distance * distance) / (2 * neighborhoodRadius * neighborhoodRadius));

        // Update weights
        const adjustedLearningRate = learningRate * influence;
        for (let i = 0; i < node.weights.length; i++) {
          node.weights[i] += adjustedLearningRate * (inputVector[i] - node.weights[i]);
        }

        return node;
      }

      function predictNextStates(sprite, recentActivities) {
        // Simple pattern-based prediction
        const predictions = [];

        // Based on current sprite type and recent activities
        if (sprite.name.includes('idle') && recentActivities.length > 0) {
          predictions.push('text_creation', 'object_move');
        } else if (sprite.name.includes('text')) {
          predictions.push('idle', 'object_move');
        } else {
          predictions.push('idle');
        }

        return predictions;
      }

      self.onmessage = function(e) {
        const { id, type, data } = e.data;
        let result = null;

        try {
          initializeWorker();

          switch (type) {
            case 'sprite_analysis':
              result = {
                spriteId: data.sprite.id,
                features: processSprite(data.sprite),
                complexity: data.sprite.metadata.complexity,
                timestamp: Date.now()
              };
              break;

            case 'som_update':
              result = {
                nodeId: data.node.id,
                updatedNode: updateSOMNode(
                  data.node,
                  data.inputVector,
                  data.learningRate,
                  data.neighborhoodRadius
                )
              };
              break;

            case 'prediction':
              result = {
                spriteId: data.sprite.id,
                predictedStates: predictNextStates(data.sprite, data.recentActivities),
                confidence: 0.7 + Math.random() * 0.2
              };
              break;

            case 'optimization':
              // Optimize sprite loading order based on usage patterns
              result = {
                optimizedOrder: data.sprites.sort((a, b) => b.usageCount - a.usageCount),
                cacheRecommendations: data.sprites.slice(0, 10).map((s: any) => s.id)
              };
              break;

            default:
              throw new Error('Unknown task type: ' + type);
          }

          self.postMessage({ id, result, error: null });

        } catch (error: any) {
          self.postMessage({ id, result: null, error: error.message });
        }
      };
    `;

    const workerBlob = new Blob([workerCode], {
      type: "application/javascript",
    });
    const workerUrl = URL.createObjectURL(workerBlob);

    // Create worker pool
    for (let i = 0; i < this.maxWorkers; i++) {
      const worker = new Worker(workerUrl);

      worker.onmessage = (e: any) => {
        const { id, result, error } = e.data;

        if (error) {
          console.error(`Worker task ${id} failed:`, error);
        } else {
          this.completedTasks.set(id, result);
          this.handleWorkerResult(id, result);
        }

        this.activeWorkers--;
        this.processNextTask(); // Process next queued task
      };

      this.workerPool.push(worker);
    }

    URL.revokeObjectURL(workerUrl);
    console.log(
      `✅ Worker pool initialized with ${this.workerPool.length} workers`
    );
  }

  /**
   * Submit task to worker pool
   */
  private submitTask(task: ProcessingTask): void {
    task.startTime = Date.now();
    this.taskQueue.push(task);
    this.processNextTask();
  }

  /**
   * Process next task in queue
   */
  private processNextTask(): void {
    if (this.taskQueue.length === 0 || this.activeWorkers >= this.maxWorkers) {
      return;
    }

    // Sort by priority
    this.taskQueue.sort((a, b) => {
      const priorities = { critical: 4, high: 3, medium: 2, low: 1 };
      return priorities[b.priority] - priorities[a.priority];
    });

    const task = this.taskQueue.shift();
    if (!task) return;

    // Find available worker
    const workerIndex = this.activeWorkers % this.workerPool.length;
    const worker = this.workerPool[workerIndex];

    task.workerId = workerIndex;
    this.activeWorkers++;

    // Send task to worker
    worker.postMessage({
      id: task.id,
      type: task.type,
      data: task.data,
    });

    this.updateMultiCoreMetrics();
  }

  /**
   * Handle completed worker task results
   */
  private handleWorkerResult(taskId: string, result: any): void {
    const task = this.taskQueue.find((t) => t.id === taskId);
    if (task) {
      task.completionTime = Date.now();
      task.result = result;
    }

    // Process result based on task type
    if (result.spriteId && result.features) {
      // Update sprite with extracted features
      this.updateSpriteFeatures(result.spriteId, result.features);
    }

    if (result.nodeId && result.updatedNode) {
      // Update SOM node
      this.somNodes.set(result.nodeId, result.updatedNode);
    }

    if (result.predictedStates) {
      // Update prediction cache
      this.updatePredictionCache(result.spriteId, result.predictedStates);
    }

    if (result.optimizedOrder) {
      // Update loading optimization
      this.updateLoadingOptimizations(
        result.optimizedOrder,
        result.cacheRecommendations
      );
    }
  }

  /**
   * Update sprite features from worker analysis
   */
  private updateSpriteFeatures(spriteId: string, features: Float32Array): void {
    const sprite = this.sprites.findOne({ id: spriteId });
    if (sprite) {
      sprite.embedding = Array.from(features);
      this.sprites.update(sprite);

      // Find best matching SOM node
      this.findBestMatchingSOMNode(spriteId, features);
    }
  }

  /**
   * Find best matching SOM node ID for sprite (helper method)
   */
  private findBestSOMNodeForSprite(spriteId: string): string | null {
    // Check if sprite already has an assigned SOM node
    for (const node of this.somNodes.values()) {
      if (node.connectedSprites.includes(spriteId)) {
        return node.id;
      }
    }
    return null;
  }

  /**
   * Find best matching SOM node for sprite
   */
  private findBestMatchingSOMNode(
    spriteId: string,
    features: Float32Array
  ): void {
    let bestNode: SOMNode | null = null;
    let bestDistance = Infinity;

    for (const node of this.somNodes.values()) {
      let distance = 0;
      for (let i = 0; i < features.length; i++) {
        const diff = features[i] - node.weights[i];
        distance += diff * diff;
      }
      distance = Math.sqrt(distance);

      if (distance < bestDistance) {
        bestDistance = distance;
        bestNode = node;
      }
    }

    if (bestNode) {
      // Add sprite to node's connected sprites
      if (!bestNode.connectedSprites.includes(spriteId)) {
        bestNode.connectedSprites.push(spriteId);
      }

      // Update node activation
      bestNode.activationHistory.push(1.0);
      if (bestNode.activationHistory.length > 100) {
        bestNode.activationHistory.shift();
      }

      // Submit SOM update task
      this.submitTask({
        id: `som_update_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: "som_update",
        priority: "medium",
        data: {
          node: bestNode,
          inputVector: features,
          learningRate: this.globalLearningRate,
          neighborhoodRadius: bestNode.neighborhoodRadius,
        },
      });
    }
  }

  /**
   * Update prediction cache
   */
  private updatePredictionCache(spriteId: string, predictions: string[]): void {
    const sprite = this.sprites.findOne({ id: spriteId });
    if (sprite) {
      sprite.predictedNext = predictions;
      this.sprites.update(sprite);
    }
  }

  /**
   * Update loading optimizations
   */
  private updateLoadingOptimizations(
    optimizedOrder: any[],
    cacheRecommendations: string[]
  ): void {
    // Cache recommended sprites
    this.preCacheRecommendedStates(cacheRecommendations);
  }

  /**
   * Update multi-core metrics
   */
  private updateMultiCoreMetrics(): void {
    const queueDepth = this.taskQueue.length;
    const completedTasksInLastSecond = Array.from(
      this.completedTasks.values()
    ).filter(
      (result: any) => Date.now() - (result.timestamp || 0) < 1000
    ).length;

    // Calculate average task time
    const recentTasks = Array.from(this.completedTasks.values()).slice(-10);
    const avgTaskTime =
      recentTasks.length > 0
        ? recentTasks.reduce(
            (sum: number, task: any) => sum + (task.processingTime || 0),
            0
          ) / recentTasks.length
        : 0;

    this.multiCoreMetrics.set({
      activeWorkers: this.activeWorkers,
      queueDepth,
      tasksPerSecond: completedTasksInLastSecond,
      averageTaskTime: avgTaskTime,
      cpuUtilization: (this.activeWorkers / this.maxWorkers) * 100,
    });
  }

  /**
   * Update SOM visualization data
   */
  private updateSOMVisualization(): void {
    const nodes = Array.from(this.somNodes.values()).map((node) => ({
      x: node.position.x,
      y: node.position.y,
      activation:
        node.activationHistory.length > 0
          ? node.activationHistory[node.activationHistory.length - 1]
          : 0,
      sprites: node.connectedSprites.length,
    }));

    // Generate connections between nearby nodes
    const connections: Array<{ from: string; to: string; strength: number }> =
      [];
    for (const nodeA of this.somNodes.values()) {
      for (const nodeB of this.somNodes.values()) {
        if (nodeA.id !== nodeB.id) {
          const distance = Math.sqrt(
            Math.pow(nodeA.position.x - nodeB.position.x, 2) +
              Math.pow(nodeA.position.y - nodeB.position.y, 2)
          );

          if (distance <= 1.5) {
            // Only connect nearby nodes
            const strength = Math.max(0, 1 - distance / 1.5);
            connections.push({
              from: nodeA.id,
              to: nodeB.id,
              strength,
            });
          }
        }
      }
    }

    this.somVisualization.set({ nodes, connections });
  }

  /**
   * Calculate self-organization statistics
   */
  private updateSelfOrganizationStats(): void {
    const nodes = Array.from(this.somNodes.values());

    // Calculate convergence rate
    let convergedNodes = 0;
    for (const node of nodes) {
      if (node.activationHistory.length >= 10) {
        const recent = node.activationHistory.slice(-10);
        const variance = this.calculateVariance(recent);
        if (variance < 0.1) convergedNodes++;
      }
    }
    const convergenceRate = (convergedNodes / nodes.length) * 100;

    // Calculate adaptation speed
    const adaptationSpeed = this.globalLearningRate * 100;

    // Calculate cluster formation
    const activeClusters = nodes.filter(
      (node) => node.connectedSprites.length > 0
    ).length;
    const clusterFormation = (activeClusters / nodes.length) * 100;

    // Calculate neural efficiency
    const totalConnections = nodes.reduce(
      (sum, node) => sum + node.connectedSprites.length,
      0
    );
    const neuralEfficiency = Math.min(
      100,
      (totalConnections / this.sprites.count()) * 50
    );

    this.selfOrganizationStats.set({
      convergenceRate,
      adaptationSpeed,
      clusterFormation,
      neuralEfficiency,
    });
  }

  /**
   * Calculate variance for convergence analysis
   */
  private calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map((val) => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / values.length;
  }

  // Core "sprite sheet" operations with SOM integration
  public async captureCurrentState(
    name: string,
    triggers: string[] = []
  ): Promise<string> {
    const jsonState = JSON.stringify(this.canvas.toJSON());
    const complexity = this.calculateComplexity(jsonState);

    const sprite: CanvasSprite = {
      id: `sprite_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      sequence: this.getNextSequenceNumber(name),
      jsonState,
      metadata: {
        objects: this.canvas.getObjects().length,
        complexity,
        triggers,
      },
      createdAt: Date.now(),
      usageCount: 0,
    };

    this.sprites.insert(sprite);

    // Submit sprite analysis task to worker pool for SOM processing
    this.submitTask({
      id: `sprite_analysis_${sprite.id}`,
      type: "sprite_analysis",
      priority: "medium",
      data: { sprite },
    });

    // Submit prediction task for AI analysis
    const recentActivities = this.getRecentActivities(5);
    this.submitTask({
      id: `prediction_${sprite.id}`,
      type: "prediction",
      priority: "low",
      data: { sprite, recentActivities },
    });

    // Send to AI worker for embedding generation (legacy support)
    if (this.aiWorker) {
      this.aiWorker.postMessage({
        type: "GENERATE_EMBEDDING",
        sprite,
      });
    }

    console.log(
      `📸 Captured sprite: ${name} (complexity: ${complexity}) with SOM analysis queued`
    );
    return sprite.id;
  }

  // NES-style rapid state switching with enhanced caching and multi-core processing
  public async loadSprite(spriteId: string): Promise<boolean> {
    const loadStartTime = Date.now();

    // 1. Try browser cache first (fastest)
    let sprite = await this.browserCache.getSprite(spriteId);

    if (!sprite) {
      // 2. Fallback to Loki.js database
      sprite = this.sprites.findOne({ id: spriteId });

      if (!sprite) {
        this.cacheMisses++;
        console.warn(`❌ Sprite not found: ${spriteId}`);
        return false;
      }

      // Cache in browser for next time
      await this.browserCache.cacheSprite(sprite);
    }

    this.cacheHits++;
    sprite.usageCount++;
    this.sprites.update(sprite);

    // Submit optimization task to worker pool for usage pattern analysis
    this.submitTask({
      id: `optimization_${spriteId}`,
      type: "optimization",
      priority: "low",
      data: {
        sprites: this.sprites.find().slice(0, 20), // Top 20 sprites for analysis
        currentSprite: sprite,
      },
    });

    // 3. Enhanced loading with GPU acceleration and multi-core processing
    return new Promise(async (resolve) => {
      try {
        // Pre-warm shader programs if needed (async optimization)
        const shouldPrecompile =
          this.webglContext && sprite.metadata.complexity > 10;
        const precompilePromise = shouldPrecompile
          ? this.shaderCache.precompileForSprite(sprite)
          : Promise.resolve();

        // Generate CSS transforms using lightweight matrix library
        const transforms = this.matrixLib.generateCSSTransforms(
          sprite.jsonState
        );

        // Wait for shader precompilation if needed
        await precompilePromise;

        // Apply transforms to canvas container for hardware acceleration
        if (transforms.css3d) {
          const canvasContainer = this.canvas.getElement().parentElement;
          if (canvasContainer) {
            canvasContainer.style.transform = transforms.css3d;
            canvasContainer.style.willChange = "transform";

            // Add self-organizing visual hints for neural feedback
            canvasContainer.setAttribute(
              "data-som-node",
              this.findBestSOMNodeForSprite(spriteId) || "unassigned"
            );
          }
        }

        // Load the actual canvas state with error handling
        this.canvas.loadFromJSON(sprite.jsonState, () => {
          try {
            // Apply GPU-accelerated rendering if available
            if (this.webglContext && transforms.webgl) {
              this.shaderCache.applyTransforms({
                matrix: transforms.webgl,
                opacity: 1.0,
                blend: "normal",
              });
            }

            this.canvas.renderAll();
            this.currentState.set(sprite.name);
            this.frameCount++;

            // Log successful load for performance tracking
            const loadTime = Date.now() - loadStartTime;
            console.log(
              `✅ Loaded sprite: ${sprite.name} (${loadTime}ms, complexity: ${sprite.metadata.complexity})`
            );

            // Clean up transform hints for next frame
            setTimeout(() => {
              const canvasContainer = this.canvas.getElement().parentElement;
              if (canvasContainer) {
                canvasContainer.style.willChange = "auto";
                canvasContainer.removeAttribute("data-som-node");
              }
            }, 50);

            resolve(true);
          } catch (renderError) {
            console.error(
              `❌ Render error for sprite ${spriteId}:`,
              renderError
            );
            resolve(false);
          }
        });
      } catch (loadError) {
        console.error(`❌ Load error for sprite ${spriteId}:`, loadError);
        resolve(false);
      }
    });
  }

  // Play animation sequence (like NES sprite animation) with SOM-aware optimization
  public async playAnimation(sequenceName: string): Promise<void> {
    const sequence = this.sequences.findOne({ name: sequenceName });

    if (!sequence) {
      console.warn(`Animation sequence '${sequenceName}' not found`);
      return;
    }

    console.log(
      `🎬 Starting animation: ${sequenceName} (${sequence.frames.length} frames, ${sequence.fps} FPS)`
    );
    this.isAnimating.set(true);

    // Pre-analyze animation sequence for SOM optimization
    this.submitTask({
      id: `anim_optimization_${sequenceName}`,
      type: "optimization",
      priority: "high",
      data: {
        sprites: sequence.frames
          .map((frameId) => this.sprites.findOne({ id: frameId }))
          .filter(Boolean),
        animationType: "sequence",
        expectedFps: sequence.fps,
      },
    });

    let frameIndex = 0;
    const frameInterval = 1000 / sequence.fps; // Convert FPS to milliseconds
    const animationStartTime = Date.now();

    const playFrame = async (): Promise<any> => {
      if (frameIndex >= sequence.frames.length) {
        if (sequence.loop) {
          frameIndex = 0;
          console.log(`🔄 Looping animation: ${sequenceName}`);
        } else {
          this.isAnimating.set(false);
          const animationDuration = Date.now() - animationStartTime;
          console.log(
            `✅ Animation complete: ${sequenceName} (${animationDuration}ms)`
          );
          return;
        }
      }

      const spriteId = sequence.frames[frameIndex];
      const frameStartTime = Date.now();

      const success = await this.loadSprite(spriteId);

      if (!success) {
        console.warn(
          `⚠️ Failed to load frame ${frameIndex} (${spriteId}) in animation ${sequenceName}`
        );
      }

      const frameTime = Date.now() - frameStartTime;

      // Adaptive frame timing based on load performance
      const adaptiveInterval = Math.max(frameInterval, frameTime * 1.2);

      frameIndex++;

      if (frameIndex < sequence.frames.length || sequence.loop) {
        setTimeout(playFrame, adaptiveInterval);
      } else {
        this.isAnimating.set(false);
        const totalDuration = Date.now() - animationStartTime;
        console.log(
          `🎯 Animation finished: ${sequenceName} (${totalDuration}ms total)`
        );
      }
    };

    // Start animation with initial frame
    playFrame();
  }

  // AI-driven behavior learning with SOM integration
  public logUserActivity(
    action: string,
    context: Record<string, any> = {}
  ): void {
    const currentCanvasState = this.getCurrentStateName();

    const activity: UserActivity = {
      id: `activity_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      action,
      context,
      timestamp: Date.now(),
      canvasState: currentCanvasState,
      sequence: this.activities.count(),
    };

    this.activities.insert(activity);

    // Submit activity analysis to worker pool for pattern recognition
    const recentActivities = this.getRecentActivities(10);
    this.submitTask({
      id: `activity_analysis_${activity.id}`,
      type: "prediction",
      priority: "medium",
      data: {
        activity,
        recentActivities,
        currentSprite: this.sprites.findOne({ name: currentCanvasState }),
        contextInfo: {
          totalActivities: this.activities.count(),
          sessionDuration: Date.now() - this.startTime,
          averageComplexity: this.calculateAverageComplexity(),
        },
      },
    });

    // Send to AI worker for pattern analysis (legacy support)
    if (this.aiWorker) {
      this.aiWorker.postMessage({
        type: "ANALYZE_PATTERN",
        activity,
        recentActivities,
      });
    }

    console.log(
      `👤 User activity logged: ${action} in state ${currentCanvasState} (sequence: ${activity.sequence})`
    );
  }

  // AI worker message handling
  private handleAIWorkerMessage(event: MessageEvent): void {
    const { type, data } = event.data;

    switch (type) {
      case "EMBEDDING_GENERATED":
        this.updateSpriteEmbedding(data.spriteId, data.embedding);
        break;

      case "PATTERN_PREDICTION":
        this.handlePatternPrediction(data);
        break;

      case "CACHE_RECOMMENDATION":
        this.preCacheRecommendedStates(data.stateIds);
        break;

      case "NEW_SEQUENCE_GENERATED":
        this.registerAIGeneratedSequence(data.sequence);
        break;
    }
  }

  // Create new animation sequences based on AI learning
  private registerAIGeneratedSequence(sequenceData: any): void {
    const sequence: AnimationSequence = {
      id: `ai_seq_${Date.now()}`,
      name: `ai_generated_${sequenceData.pattern}`,
      frames: sequenceData.spriteIds,
      loop: sequenceData.loop || false,
      fps: sequenceData.fps || 24,
      triggers: sequenceData.triggers || [],
      confidence: sequenceData.confidence || 0.7,
    };

    this.sequences.insert(sequence);
  }

  // Predictive caching based on AI analysis
  private preCacheRecommendedStates(stateIds: string[]): void {
    this.predictedStates.set(stateIds);

    // Pre-warm these states in background
    stateIds.forEach((stateId) => {
      const sprite = this.sprites.findOne({ id: stateId });
      if (sprite) {
        // Pre-parse JSON in background (like loading sprite data into VRAM)
        try {
          JSON.parse(sprite.jsonState);
        } catch (e: any) {
          console.warn(`Invalid sprite JSON for ${stateId}`);
        }
      }
    });
  }

  // Utility methods
  private calculateComplexity(jsonState: string): number {
    // Simple complexity calculation based on JSON size and object count
    const stateObj = JSON.parse(jsonState);
    const objects = stateObj.objects?.length || 0;
    const jsonSize = jsonState.length;

    return Math.floor(objects * 10 + jsonSize / 1000);
  }

  private calculateAverageComplexity(): number {
    const allSprites = this.sprites.find();
    if (allSprites.length === 0) return 0;

    const totalComplexity = allSprites.reduce(
      (sum, sprite) => sum + sprite.metadata.complexity,
      0
    );
    return Math.round((totalComplexity / allSprites.length) * 100) / 100; // Round to 2 decimal places
  }

  private getNextSequenceNumber(name: string): number {
    const existingSprites = this.sprites.find({ name });
    return existingSprites.length;
  }

  private getCurrentStateName(): string {
    let currentState: string;
    this.currentState.subscribe((state) => (currentState = state))();
    return currentState!;
  }

  private getRecentActivities(count: number): UserActivity[] {
    return (this.activities as any)
      .chain()
      .simplesort("timestamp", true)
      .limit(count)
      .data();
  }

  private updateSpriteEmbedding(spriteId: string, embedding: number[]): void {
    const sprite = this.sprites.findOne({ id: spriteId });
    if (sprite) {
      sprite.embedding = embedding;
      this.sprites.update(sprite);
    }
  }

  private handlePatternPrediction(data: any): void {
    // Update predicted next states based on AI analysis
    this.predictedStates.set(data.predictedStates || []);
  }

  private createDefaultSprites(): void {
    // Create basic "idle" sprite
    const idleState = JSON.stringify(this.canvas.toJSON());
    this.sprites.insert({
      id: "sprite_idle_default",
      name: "idle",
      sequence: 0,
      jsonState: idleState,
      metadata: {
        objects: 0,
        complexity: 1,
        triggers: ["init", "reset"],
      },
      createdAt: Date.now(),
      usageCount: 0,
    });
  }

  // Public API for Svelte components
  public getAvailableStates(): string[] {
    return Array.from(
      new Set(this.sprites.find().map((sprite) => sprite.name))
    );
  }

  public getAnimationSequences(): AnimationSequence[] {
    return this.sequences.find();
  }

  public getCacheStats(): { hits: number; misses: number; hitRate: number } {
    const hitRate = this.cacheHits / (this.cacheHits + this.cacheMisses);
    return {
      hits: this.cacheHits,
      misses: this.cacheMisses,
      hitRate: isNaN(hitRate) ? 1.0 : hitRate,
    };
  }

  // Enhanced cleanup with multi-core system shutdown
  public destroy(): void {
    console.log("🛑 Shutting down Neural Sprite Engine...");

    // Terminate AI worker
    if (this.aiWorker) {
      this.aiWorker.terminate();
      console.log("✅ AI worker terminated");
    }

    // Terminate all workers in the pool
    console.log(`🔄 Terminating ${this.workerPool.length} worker threads...`);
    for (const worker of this.workerPool) {
      worker.terminate();
    }
    this.workerPool = [];
    this.activeWorkers = 0;
    this.taskQueue = [];
    this.completedTasks.clear();
    console.log("✅ Worker pool terminated");

    // Clear SOM nodes
    this.somNodes.clear();
    console.log("✅ SOM nodes cleared");

    // Clear database
    if (this.db) {
      // Loki doesn't have a close method, just clear collections
      const collections = (this.db as any).collections || [];
      collections.forEach((col: any) => {
        this.db.removeCollection(col.name);
      });
      console.log("✅ Database collections cleared");
    }

    // Reset performance counters
    this.frameCount = 0;
    this.cacheHits = 0;
    this.cacheMisses = 0;

    // Clean up WebGL context
    if (this.webglContext) {
      const loseContext = this.webglContext.getExtension("WEBGL_lose_context");
      if (loseContext) {
        loseContext.loseContext();
      }
      console.log("✅ WebGL context cleaned up");
    }

    console.log("🎯 Neural Sprite Engine shutdown complete");
  }
}

// Factory function for Svelte integration
export function createNeuralSpriteEngine(
  canvas: fabric.Canvas
): NeuralSpriteEngine {
  return new NeuralSpriteEngine(canvas);
}

// Enhanced derived stores for performance monitoring with SOM integration
export function createPerformanceStores(engine: NeuralSpriteEngine) {
  return {
    // Core engine stores
    currentState: engine.currentState,
    isAnimating: engine.isAnimating,
    cacheHitRate: engine.cacheHitRate,
    predictedStates: engine.predictedStates,

    // Self-organizing features
    somVisualization: engine.somVisualization,
    multiCoreMetrics: engine.multiCoreMetrics,
    selfOrganizationStats: engine.selfOrganizationStats,

    // Derived performance metrics with SOM awareness
    performanceGrade: derived(
      [engine.cacheHitRate, engine.selfOrganizationStats],
      ([$hitRate, $somStats]) => {
        const baseScore = $hitRate;
        const somBonus = $somStats.neuralEfficiency / 1000; // Small bonus for neural efficiency
        const totalScore = baseScore + somBonus;

        if (totalScore >= 0.95) return "S+"; // Perfect with SOM optimization
        if (totalScore >= 0.9) return "S"; // Perfect
        if (totalScore >= 0.85) return "A"; // Excellent
        if (totalScore >= 0.75) return "B"; // Good
        if (totalScore >= 0.65) return "C"; // Average
        return "D"; // Needs optimization
      }
    ),

    isOptimized: derived(
      [engine.cacheHitRate, engine.multiCoreMetrics],
      ([$hitRate, $multiCore]) =>
        $hitRate >= 0.9 && $multiCore.cpuUtilization < 80
    ),

    systemHealth: derived(
      [engine.multiCoreMetrics, engine.selfOrganizationStats],
      ([$multiCore, $somStats]) => ({
        cpu:
          $multiCore.cpuUtilization < 70
            ? "excellent"
            : $multiCore.cpuUtilization < 85
              ? "good"
              : "warning",
        workers:
          $multiCore.activeWorkers /
            ($multiCore.activeWorkers + $multiCore.queueDepth) >
          0.7
            ? "efficient"
            : "overloaded",
        neural:
          $somStats.convergenceRate > 80
            ? "converged"
            : $somStats.convergenceRate > 50
              ? "learning"
              : "initializing",
        overall:
          $multiCore.cpuUtilization < 80 && $somStats.neuralEfficiency > 60
            ? "healthy"
            : "degraded",
      })
    ),

    adaptiveMetrics: derived(
      [
        engine.cacheHitRate,
        engine.multiCoreMetrics,
        engine.selfOrganizationStats,
      ],
      ([$hitRate, $multiCore, $somStats]) => ({
        efficiency: Math.round(
          ($hitRate * 0.4 +
            ($multiCore.tasksPerSecond / 10) * 0.3 +
            ($somStats.neuralEfficiency / 100) * 0.3) *
            100
        ),
        adaptability: Math.round(
          ($somStats.adaptationSpeed * $somStats.clusterFormation) / 10
        ),
        predictiveAccuracy: Math.round(
          $somStats.convergenceRate * 0.8 + $hitRate * 20
        ),
        resourceUtilization: Math.round(
          (100 - $multiCore.cpuUtilization) * 0.6 +
            $somStats.neuralEfficiency * 0.4
        ),
      })
    ),
  };
}
