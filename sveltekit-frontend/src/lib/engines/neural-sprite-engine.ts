/**
 * Neural Sprite Sheet Engine
 * NES-inspired rapid state switching with AI prediction
 *
 * Core concept: Pre-computed JSON states stored as "sprites" in Loki.js,
 * with local LLM predicting next states for ultra-fast animations
 */

import Loki, { type Collection } from "lokijs";
import { writable, derived, type Readable } from "svelte/store";
import { fabric } from "fabric";
import { ShaderCache } from "./webgl-shader-cache";
import { BrowserCacheManager } from "./browser-cache-manager";
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

// Main Neural Sprite Engine class
export class NeuralSpriteEngine {
  private db: Loki;
  private sprites: Collection<CanvasSprite>;
  private activities: Collection<UserActivity>;
  private sequences: Collection<AnimationSequence>;
  private canvas: fabric.Canvas;
  private aiWorker?: Worker;

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

  // Performance metrics (NES-inspired)
  private frameCount = 0;
  private cacheHits = 0;
  private cacheMisses = 0;

  constructor(canvas: fabric.Canvas) {
    this.canvas = canvas;
    this.initializeDatabase();
    this.initializeAIWorker();
    this.setupPerformanceMonitoring();
    this.initializeEnhancedCaching();
  }

  private initializeDatabase(): void {
    this.db = new Loki("neural-sprite-cache.db", {
      persistenceMethod: "localStorage",
      autoload: true,
      autoloadCallback: this.databaseInitialize.bind(this),
      autosave: true,
      autosaveInterval: 4000,
    });
  }

  private databaseInitialize(): void {
    // Initialize collections if they don't exist
    this.sprites =
      this.db.getCollection("sprites") ||
      this.db.addCollection("sprites", { indices: ["name", "sequence"] });

    this.activities =
      this.db.getCollection("activities") ||
      this.db.addCollection("activities", { indices: ["action", "timestamp"] });

    this.sequences =
      this.db.getCollection("sequences") ||
      this.db.addCollection("sequences", { indices: ["name", "confidence"] });

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
    // NES-style performance monitoring: 60fps target
    setInterval(() => {
      const hitRate = this.cacheHits / (this.cacheHits + this.cacheMisses);
      this.cacheHitRate.set(isNaN(hitRate) ? 1.0 : hitRate);

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

  // Core "sprite sheet" operations
  public async captureCurrentState(
    name: string,
    triggers: string[] = [],
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

    // Send to AI worker for embedding generation
    if (this.aiWorker) {
      this.aiWorker.postMessage({
        type: "GENERATE_EMBEDDING",
        sprite,
      });
    }

    return sprite.id;
  }

  // NES-style rapid state switching with enhanced caching
  public async loadSprite(spriteId: string): Promise<boolean> {
    // 1. Try browser cache first (fastest)
    let sprite = await this.browserCache.getSprite(spriteId);

    if (!sprite) {
      // 2. Fallback to Loki.js database
      sprite = this.sprites.findOne({ id: spriteId });

      if (!sprite) {
        this.cacheMisses++;
        return false;
      }

      // Cache in browser for next time
      await this.browserCache.cacheSprite(sprite);
    }

    this.cacheHits++;
    sprite.usageCount++;
    this.sprites.update(sprite);

    // 3. Enhanced loading with GPU acceleration
    return new Promise(async (resolve) => {
      // Pre-warm shader programs if needed
      if (this.webglContext && sprite.metadata.complexity > 10) {
        await this.shaderCache.precompileForSprite(sprite);
      }

      // Generate CSS transforms using lightweight matrix library
      const transforms = this.matrixLib.generateCSSTransforms(sprite.jsonState);

      // Apply transforms to canvas container for hardware acceleration
      if (transforms.css3d) {
        const canvasContainer = this.canvas.getElement().parentElement;
        if (canvasContainer) {
          canvasContainer.style.transform = transforms.css3d;
          canvasContainer.style.willChange = "transform";
        }
      }

      // Load the actual canvas state
      this.canvas.loadFromJSON(sprite.jsonState, () => {
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

        // Clean up transform hints for next frame
        setTimeout(() => {
          const canvasContainer = this.canvas.getElement().parentElement;
          if (canvasContainer) {
            canvasContainer.style.willChange = "auto";
          }
        }, 50);

        resolve(true);
      });
    });
  }

  // Play animation sequence (like NES sprite animation)
  public async playAnimation(sequenceName: string): Promise<void> {
    const sequence = this.sequences.findOne({ name: sequenceName });

    if (!sequence) {
      console.warn(`Animation sequence '${sequenceName}' not found`);
      return;
    }

    this.isAnimating.set(true);

    let frameIndex = 0;
    const frameInterval = 1000 / sequence.fps; // Convert FPS to milliseconds

    const playFrame = async () => {
      if (frameIndex >= sequence.frames.length) {
        if (sequence.loop) {
          frameIndex = 0;
        } else {
          this.isAnimating.set(false);
          return;
        }
      }

      const spriteId = sequence.frames[frameIndex];
      await this.loadSprite(spriteId);
      frameIndex++;

      if (frameIndex < sequence.frames.length || sequence.loop) {
        setTimeout(playFrame, frameInterval);
      } else {
        this.isAnimating.set(false);
      }
    };

    playFrame();
  }

  // AI-driven behavior learning
  public logUserActivity(
    action: string,
    context: Record<string, any> = {},
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

    // Send to AI worker for pattern analysis
    if (this.aiWorker) {
      this.aiWorker.postMessage({
        type: "ANALYZE_PATTERN",
        activity,
        recentActivities: this.getRecentActivities(10),
      });
    }
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
        } catch (e) {
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
    return this.activities
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
      new Set(this.sprites.find().map((sprite) => sprite.name)),
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

  // Cleanup
  public destroy(): void {
    if (this.aiWorker) {
      this.aiWorker.terminate();
    }

    if (this.db) {
      this.db.close();
    }
  }
}

// Factory function for Svelte integration
export function createNeuralSpriteEngine(
  canvas: fabric.Canvas,
): NeuralSpriteEngine {
  return new NeuralSpriteEngine(canvas);
}

// Derived stores for performance monitoring
export function createPerformanceStores(engine: NeuralSpriteEngine) {
  return {
    currentState: engine.currentState,
    isAnimating: engine.isAnimating,
    cacheHitRate: engine.cacheHitRate,
    predictedStates: engine.predictedStates,

    // Derived performance metrics
    performanceGrade: derived(engine.cacheHitRate, ($hitRate) => {
      if ($hitRate >= 0.95) return "S"; // Perfect (NES-style grading)
      if ($hitRate >= 0.85) return "A"; // Excellent
      if ($hitRate >= 0.75) return "B"; // Good
      if ($hitRate >= 0.65) return "C"; // Average
      return "D"; // Needs optimization
    }),

    isOptimized: derived(engine.cacheHitRate, ($hitRate) => $hitRate >= 0.9),
  };
}
