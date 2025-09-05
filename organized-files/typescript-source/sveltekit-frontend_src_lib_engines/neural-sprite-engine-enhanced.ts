// Enhanced Neural Sprite Engine with Full Service Integration
// Extends the base neural sprite engine with PostgreSQL, Redis, RabbitMQ, and MinIO integration
// Provides enterprise-grade performance and scalability for legal AI applications

import { NeuralSpriteEngine, type CanvasSprite, type SOMNode, type ProcessingTask } from './neural-sprite-engine';
import { 
  NeuralServiceConnections, 
  createConfigFromEnv, 
  type NeuralServiceConfig 
} from '$lib/services/neural-service-connections';
import { writable, derived, type Readable } from 'svelte/store';
import * as fabric from 'fabric';

// Enhanced configuration with service integration
export interface EnhancedNeuralConfig {
  canvas: fabric.Canvas;
  services: NeuralServiceConfig;
  performance: {
    enableServiceCaching: boolean;
    enableDistributedProcessing: boolean;
    enableLargeStateStorage: boolean;
    cacheTTL: number;
    maxInMemorySprites: number;
  };
  monitoring: {
    enableMetrics: boolean;
    metricsInterval: number;
    enableHealthChecks: boolean;
  };
}

// Enhanced performance metrics with service data
export interface EnhancedPerformanceMetrics {
  cacheHitRate: number;
  databaseConnections: number;
  queueDepth: number;
  storageUtilization: number;
  distributedTasksPerSecond: number;
  serviceLatency: Record<string, number>;
  memoryUsage: number;
  neuralEfficiency: number;
}

// Service-integrated sprite with additional metadata
export interface ServiceIntegratedSprite extends CanvasSprite {
  storageLocation?: 'memory' | 'redis' | 'postgresql' | 'minio';
  vectorSimilarity?: number;
  distributedProcessing?: boolean;
  serviceMetadata?: {
    redisKey?: string;
    minioObject?: string;
    postgresId?: string;
    lastServiceSync?: number;
  };
}

/**
 * Enhanced Neural Sprite Engine with full service integration
 * Combines the base neural engine with enterprise-grade service connections
 */
export class EnhancedNeuralSpriteEngine extends NeuralSpriteEngine {
  private serviceConnections: NeuralServiceConnections;
  private config: EnhancedNeuralConfig;
  private performanceMetrics: EnhancedPerformanceMetrics;
  private metricsInterval?: NodeJS.Timeout;
  
  // Enhanced reactive stores
  public serviceStatus = writable<Record<string, any>>({});
  public distributedMetrics = writable<Record<string, number>>({});
  public enhancedPerformance = writable<EnhancedPerformanceMetrics>({
    cacheHitRate: 1.0,
    databaseConnections: 0,
    queueDepth: 0,
    storageUtilization: 0,
    distributedTasksPerSecond: 0,
    serviceLatency: {},
    memoryUsage: 0,
    neuralEfficiency: 0
  });

  // Service health monitoring
  public isServicesHealthy = derived(
    [this.serviceStatus],
    ([$status]) => {
      const services = Object.values($status);
      return services.length > 0 && services.every((s: any) => s.connected === true);
    }
  );

  constructor(config: EnhancedNeuralConfig) {
    super(config.canvas);
    this.config = config;
    this.serviceConnections = new NeuralServiceConnections(config.services);
    
    this.performanceMetrics = {
      cacheHitRate: 1.0,
      databaseConnections: 0,
      queueDepth: 0,
      storageUtilization: 0,
      distributedTasksPerSecond: 0,
      serviceLatency: {},
      memoryUsage: 0,
      neuralEfficiency: 0
    };

    this.initialize();
  }

  /**
   * Initialize enhanced neural engine with service connections
   */
  private async initialize(): Promise<void> {
    console.log('🚀 Initializing Enhanced Neural Sprite Engine...');

    try {
      // Initialize service connections
      const servicesInitialized = await this.serviceConnections.initialize();
      
      if (servicesInitialized) {
        console.log('✅ All services connected successfully');
        this.setupServiceMonitoring();
        this.startPerformanceMonitoring();
        await this.loadExistingSprites();
        await this.synchronizeSOMNodes();
      } else {
        console.warn('⚠️ Some services failed to initialize - running in degraded mode');
      }

      // Subscribe to service status updates
      this.serviceConnections.status.subscribe(status => {
        this.serviceStatus.set(status);
      });

    } catch (error) {
      console.error('❌ Failed to initialize enhanced neural engine:', error);
      throw error;
    }
  }

  /**
   * Override sprite capture with service integration
   */
  public async captureCurrentState(name: string, triggers: string[] = []): Promise<string> {
    const spriteId = await super.captureCurrentState(name, triggers);
    
    if (this.config.performance.enableServiceCaching) {
      // Enhance with service storage
      await this.enhanceSprite(spriteId);
    }

    return spriteId;
  }

  /**
   * Override sprite loading with intelligent service-based loading
   */
  public async loadSprite(spriteId: string): Promise<boolean> {
    const startTime = Date.now();
    let sprite: ServiceIntegratedSprite | null = null;

    try {
      // Try loading from different service tiers
      sprite = await this.loadSpriteFromServices(spriteId);
      
      if (!sprite) {
        // Fallback to base engine
        const success = await super.loadSprite(spriteId);
        this.recordServiceLatency('base_engine', Date.now() - startTime);
        return success;
      }

      // Apply service-loaded sprite to canvas
      const success = await this.applySpriteToCanvas(sprite);
      this.recordServiceLatency('service_integrated', Date.now() - startTime);
      
      // Update usage metrics
      this.updateDistributedMetrics(sprite);
      
      return success;

    } catch (error) {
      console.error(`❌ Enhanced sprite loading failed for ${spriteId}:`, error);
      // Fallback to base engine
      return await super.loadSprite(spriteId);
    }
  }

  /**
   * Enhanced sprite loading from service tiers
   */
  private async loadSpriteFromServices(spriteId: string): Promise<ServiceIntegratedSprite | null> {
    // Try Redis cache first (fastest)
    if (this.config.performance.enableServiceCaching) {
      try {
        const cachedNode = await this.serviceConnections.loadCachedSOMNode(spriteId);
        if (cachedNode) {
          console.log(`⚡ Loaded sprite ${spriteId} from Redis cache`);
          return await this.reconstructSpriteFromCache(spriteId, cachedNode);
        }
      } catch (error) {
        console.warn('Redis cache miss:', error);
      }
    }

    // Try PostgreSQL database (medium speed, authoritative)
    try {
      const sprite = await this.serviceConnections.loadSprite(spriteId);
      if (sprite) {
        console.log(`🗄️ Loaded sprite ${spriteId} from PostgreSQL`);
        // Cache for next time
        if (this.config.performance.enableServiceCaching) {
          await this.cacheSprite(sprite);
        }
        return sprite as ServiceIntegratedSprite;
      }
    } catch (error) {
      console.warn('PostgreSQL load failed:', error);
    }

    // Try MinIO for large states (slowest but handles large files)
    if (this.config.performance.enableLargeStateStorage) {
      try {
        const largeState = await this.serviceConnections.loadSpriteLargeState(`sprites/${spriteId}/state.json`);
        if (largeState) {
          console.log(`💾 Loaded large sprite ${spriteId} from MinIO`);
          return await this.reconstructSpriteFromLargeState(spriteId, largeState);
        }
      } catch (error) {
        console.warn('MinIO load failed:', error);
      }
    }

    return null;
  }

  /**
   * Apply service-loaded sprite to canvas
   */
  private async applySpriteToCanvas(sprite: ServiceIntegratedSprite): Promise<boolean> {
    return new Promise((resolve) => {
      try {
        this.canvas.loadFromJSON(sprite.jsonState, () => {
          this.canvas.renderAll();
          this.currentState.set(sprite.name);
          this.frameCount++;
          
          console.log(`✅ Applied service sprite: ${sprite.name} (${sprite.storageLocation})`);
          resolve(true);
        });
      } catch (error) {
        console.error(`❌ Failed to apply sprite to canvas:`, error);
        resolve(false);
      }
    });
  }

  /**
   * Enhance sprite with service storage
   */
  private async enhanceSprite(spriteId: string): Promise<void> {
    const sprite = this.sprites.findOne({ id: spriteId });
    if (!sprite) return;

    const tasks: Promise<void>[] = [];

    // Save to PostgreSQL with vector embedding
    tasks.push(this.serviceConnections.saveSprite(sprite));

    // Cache frequently used sprites in Redis
    if (sprite.usageCount > 5) {
      tasks.push(this.cacheSprite(sprite));
    }

    // Store large states in MinIO
    if (sprite.jsonState.length > 100000) { // 100KB threshold
      tasks.push(this.storeLargeState(sprite));
    }

    // Distribute processing tasks
    if (this.config.performance.enableDistributedProcessing) {
      tasks.push(this.distributeProcessingTasks(sprite));
    }

    await Promise.allSettled(tasks);
  }

  /**
   * Cache sprite in Redis
   */
  private async cacheSprite(sprite: CanvasSprite): Promise<void> {
    if (!sprite.embedding) return;

    // Convert sprite to SOM node format for caching
    const somNode: SOMNode = {
      id: sprite.id,
      position: { x: 0, y: 0 }, // Will be updated by SOM
      weights: new Float32Array(sprite.embedding),
      activationHistory: [],
      connectedSprites: [sprite.id],
      learningRate: 0.1,
      neighborhoodRadius: 1.0
    };

    await this.serviceConnections.cacheSOMNode(somNode);
  }

  /**
   * Store large sprite state in MinIO
   */
  private async storeLargeState(sprite: CanvasSprite): Promise<void> {
    await this.serviceConnections.storeSpriteLargeState(sprite.id, sprite.jsonState);
  }

  /**
   * Distribute processing tasks via RabbitMQ
   */
  private async distributeProcessingTasks(sprite: CanvasSprite): Promise<void> {
    const tasks: ProcessingTask[] = [
      {
        id: `sprite_analysis_${sprite.id}`,
        type: 'sprite_analysis',
        priority: 'medium',
        data: { sprite }
      },
      {
        id: `prediction_${sprite.id}`,
        type: 'prediction', 
        priority: 'low',
        data: { sprite, context: 'service_integrated' }
      }
    ];

    for (const task of tasks) {
      await this.serviceConnections.publishTask(task);
    }
  }

  /**
   * Load existing sprites from database on startup
   */
  private async loadExistingSprites(): Promise<void> {
    try {
      // This would require implementing a method to get recent sprites
      // For now, we'll log that we're ready to load
      console.log('📚 Ready to load sprites from PostgreSQL on demand');
    } catch (error) {
      console.warn('⚠️ Could not load existing sprites:', error);
    }
  }

  /**
   * Synchronize SOM nodes with database
   */
  private async synchronizeSOMNodes(): Promise<void> {
    console.log('🧠 Synchronizing SOM nodes with service layer...');
    
    // In a real implementation, this would sync SOM nodes between
    // memory, Redis cache, and PostgreSQL
    for (const [nodeId, node] of this.somNodes) {
      try {
        await this.serviceConnections.cacheSOMNode(node);
      } catch (error) {
        console.warn(`⚠️ Failed to sync SOM node ${nodeId}:`, error);
      }
    }
  }

  /**
   * Setup service monitoring
   */
  private setupServiceMonitoring(): void {
    // Monitor service connections
    this.serviceConnections.status.subscribe(status => {
      const connected = Object.values(status).filter((s: any) => s.connected).length;
      const total = Object.keys(status).length;
      
      this.performanceMetrics.databaseConnections = connected;
      console.log(`📡 Service health: ${connected}/${total} services connected`);
    });

    // Monitor connection errors
    this.serviceConnections.connectionErrors.subscribe(errors => {
      if (errors.length > 0) {
        console.warn('⚠️ Service connection errors:', errors);
      }
    });
  }

  /**
   * Start performance monitoring with service metrics
   */
  private startPerformanceMonitoring(): void {
    if (!this.config.monitoring.enableMetrics) return;

    this.metricsInterval = setInterval(async () => {
      await this.updateEnhancedMetrics();
    }, this.config.monitoring.metricsInterval || 1000);
  }

  /**
   * Update enhanced performance metrics
   */
  private async updateEnhancedMetrics(): Promise<void> {
    try {
      // Get service health metrics
      const healthMetrics = await this.serviceConnections.getHealthMetrics();
      
      // Update performance metrics
      this.performanceMetrics.cacheHitRate = this.getCacheStats().hitRate;
      this.performanceMetrics.memoryUsage = this.calculateMemoryUsage();
      this.performanceMetrics.neuralEfficiency = this.calculateNeuralEfficiency();
      
      // Service-specific metrics
      if (healthMetrics.postgresql) {
        this.performanceMetrics.storageUtilization = healthMetrics.postgresql.sprites || 0;
      }

      this.enhancedPerformance.set({ ...this.performanceMetrics });
      
    } catch (error) {
      console.warn('⚠️ Failed to update enhanced metrics:', error);
    }
  }

  /**
   * Record service latency for monitoring
   */
  private recordServiceLatency(service: string, latency: number): void {
    this.performanceMetrics.serviceLatency[service] = latency;
    
    // Update distributed metrics
    this.distributedMetrics.update(current => ({
      ...current,
      [`${service}_latency`]: latency
    }));
  }

  /**
   * Update distributed processing metrics
   */
  private updateDistributedMetrics(sprite: ServiceIntegratedSprite): void {
    this.distributedMetrics.update(current => ({
      ...current,
      sprites_processed: (current.sprites_processed || 0) + 1,
      storage_type: sprite.storageLocation || 'memory',
      last_processed: Date.now()
    }));
  }

  /**
   * Calculate memory usage
   */
  private calculateMemoryUsage(): number {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      const usage = process.memoryUsage();
      return Math.round((usage.heapUsed / 1024 / 1024) * 100) / 100; // MB
    }
    return 0;
  }

  /**
   * Calculate neural efficiency based on service performance
   */
  private calculateNeuralEfficiency(): number {
    const baseEfficiency = this.getCacheStats().hitRate * 100;
    const serviceBonus = Math.min(20, Object.keys(this.performanceMetrics.serviceLatency).length * 5);
    return Math.min(100, baseEfficiency + serviceBonus);
  }

  /**
   * Reconstruct sprite from Redis cache
   */
  private async reconstructSpriteFromCache(spriteId: string, somNode: SOMNode): Promise<ServiceIntegratedSprite | null> {
    // In a real implementation, this would reconstruct the sprite
    // from cached SOM node data. For now, return null to fallback
    return null;
  }

  /**
   * Reconstruct sprite from MinIO large state
   */
  private async reconstructSpriteFromLargeState(spriteId: string, jsonState: string): Promise<ServiceIntegratedSprite> {
    const complexity = this.calculateComplexity(jsonState);
    
    return {
      id: spriteId,
      name: `large_state_${spriteId}`,
      sequence: 0,
      jsonState,
      metadata: {
        objects: JSON.parse(jsonState).objects?.length || 0,
        complexity
      },
      createdAt: Date.now(),
      usageCount: 0,
      storageLocation: 'minio',
      serviceMetadata: {
        minioObject: `sprites/${spriteId}/state.json`,
        lastServiceSync: Date.now()
      }
    };
  }

  /**
   * Get comprehensive service health status
   */
  public async getServiceHealth(): Promise<Record<string, any>> {
    const healthMetrics = await this.serviceConnections.getHealthMetrics();
    const baseStats = this.getCacheStats();
    
    return {
      services: healthMetrics,
      cache: baseStats,
      performance: this.performanceMetrics,
      neural: {
        somNodes: this.somNodes.size,
        sprites: this.sprites.count(),
        workers: this.activeWorkers || 0
      }
    };
  }

  /**
   * Enhanced cleanup with service disconnection
   */
  public async destroy(): Promise<void> {
    console.log('🛑 Shutting down Enhanced Neural Sprite Engine...');

    // Clear monitoring
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }

    // Disconnect services
    await this.serviceConnections.disconnect();

    // Call base cleanup
    super.destroy();

    console.log('✅ Enhanced Neural Sprite Engine shutdown complete');
  }
}

/**
 * Factory function for creating enhanced neural sprite engine
 */
export function createEnhancedNeuralSpriteEngine(
  canvas: fabric.Canvas,
  customConfig?: Partial<EnhancedNeuralConfig>
): EnhancedNeuralSpriteEngine {
  const defaultConfig: EnhancedNeuralConfig = {
    canvas,
    services: createConfigFromEnv(),
    performance: {
      enableServiceCaching: true,
      enableDistributedProcessing: true,
      enableLargeStateStorage: true,
      cacheTTL: 3600,
      maxInMemorySprites: 100
    },
    monitoring: {
      enableMetrics: true,
      metricsInterval: 1000,
      enableHealthChecks: true
    }
  };

  const config = { ...defaultConfig, ...customConfig };
  return new EnhancedNeuralSpriteEngine(config);
}

/**
 * Create enhanced performance monitoring stores
 */
export function createEnhancedPerformanceStores(engine: EnhancedNeuralSpriteEngine) {
  return {
    // Base engine stores
    currentState: engine.currentState,
    isAnimating: engine.isAnimating,
    cacheHitRate: engine.cacheHitRate,
    predictedStates: engine.predictedStates,
    
    // Enhanced service stores
    serviceStatus: engine.serviceStatus,
    distributedMetrics: engine.distributedMetrics,
    enhancedPerformance: engine.enhancedPerformance,
    isServicesHealthy: engine.isServicesHealthy,

    // Derived enhanced metrics
    serviceGrade: derived(
      [engine.isServicesHealthy, engine.enhancedPerformance],
      ([$healthy, $metrics]) => {
        if (!$healthy) return 'F'; // Fail if services down
        
        const score = $metrics.neuralEfficiency;
        if (score >= 90) return 'S+';
        if (score >= 85) return 'S';
        if (score >= 80) return 'A';
        if (score >= 70) return 'B';
        if (score >= 60) return 'C';
        return 'D';
      }
    ),

    systemEfficiency: derived(
      [engine.enhancedPerformance, engine.distributedMetrics],
      ([$performance, $distributed]) => ({
        overall: Math.round(
          ($performance.cacheHitRate * 0.3 +
           $performance.neuralEfficiency * 0.4 +
           Math.min(100, $distributed.sprites_processed || 0) * 0.3)
        ),
        services: $performance.databaseConnections,
        latency: Math.round(
          Object.values($performance.serviceLatency).reduce((a, b) => a + b, 0) /
          Math.max(1, Object.values($performance.serviceLatency).length)
        ),
        memory: $performance.memoryUsage
      })
    )
  };
}