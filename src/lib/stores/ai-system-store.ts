
/**
 * Comprehensive AI System Store
 * Manages the entire AI system lifecycle with Windows-native optimizations
 */

import { writable } from 'svelte/store';
import type { Writable } from 'svelte/store';
import { browser } from '$app/environment';

// Import the comprehensive AI system
import { MathOptimizedAISystem } from '../integration/math-optimized-ai-system';

interface SystemConfig {
  windowsOptimizations: {
    enableGPUAcceleration: boolean;
    enableSIMD: boolean;
    maxWorkerThreads: number;
    enableWebAssembly: boolean;
  };
  performance: {
    enableJITCompilation: boolean;
    cacheStrategy: 'memory' | 'disk' | 'hybrid';
    enableRealTimeMetrics: boolean;
  };
}

interface SystemState {
  initialized: boolean;
  health: unknown;
  performance: unknown;
  components: Record<string, 'active' | 'inactive' | 'error'>;
  metrics: {
    cpuUsage: number;
    memoryUsage: number;
    gpuUsage?: number;
    cacheHitRate: number;
    activeConnections: number;
    processingQueue: number;
  };
  recommendations: unknown[];
  errors: unknown[];
}

class AISystemStore {
  private system: MathOptimizedAISystem | null = null;
  private systemState: Writable<SystemState>;
  private metricsInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.systemState = writable<SystemState>({
      initialized: false,
      health: {},
      performance: {},
      components: {},
      metrics: {
        cpuUsage: 0,
        memoryUsage: 0,
        cacheHitRate: 0,
        activeConnections: 0,
        processingQueue: 0
      },
      recommendations: [],
      errors: []
    });
  }

  async initialize(config: SystemConfig) {
    if (!browser) return;

    try {
      console.log('🚀 Initializing MathOptimizedAISystem...');
      
      // Create and initialize the comprehensive AI system
      this.system = new MathOptimizedAISystem(config);
      
      // Set up event listeners
      this.setupEventListeners();
      
      // Start metrics collection
      this.startMetricsCollection();
      
      // Update state
      this.systemState.update(state => ({
        ...state,
        initialized: true,
        components: {
          streamingServer: 'active',
          cacheManager: 'active',
          analyticsService: 'active',
          recommendationEngine: 'active',
          semanticSearch: 'active',
          aiOrchestration: 'active',
          extendedThinking: 'active'
        }
      }));
      
      console.log('✅ AI System initialized successfully');
      
    } catch (error) {
      console.error('❌ AI System initialization failed:', error);
      
      this.systemState.update(state => ({
        ...state,
        initialized: false,
        errors: [...state.errors, { timestamp: Date.now(), error: error.message }]
      }));
      
      throw error;
    }
  }

  private setupEventListeners() {
    if (!this.system) return;

    // Listen for system health updates
    this.system.on('health-updated', (health) => {
      this.systemState.update(state => ({
        ...state,
        health
      }));
    });

    // Listen for performance metrics
    this.system.on('performance-updated', (metrics) => {
      this.systemState.update(state => ({
        ...state,
        performance: metrics,
        metrics: {
          ...state.metrics,
          cpuUsage: metrics.cpu || 0,
          memoryUsage: metrics.memory || 0,
          gpuUsage: metrics.gpu || 0
        }
      }));
    });

    // Listen for quality changes
    this.system.on('quality-changed', (data) => {
      console.log(`🎯 Quality adjusted to ${data.quality} based on performance`);
    });

    // Listen for component status changes
    this.system.on('component-status', (data) => {
      this.systemState.update(state => ({
        ...state,
        components: {
          ...state.components,
          [data.component]: data.status
        }
      }));
    });

    // Listen for errors
    this.system.on('error', (error) => {
      this.systemState.update(state => ({
        ...state,
        errors: [...state.errors.slice(-9), { // Keep last 10 errors
          timestamp: Date.now(),
          component: error.component || 'system',
          message: error.message || error.toString()
        }]
      }));
    });
  }

  private startMetricsCollection() {
    this.metricsInterval = setInterval(async () => {
      if (!this.system) return;

      try {
        const health = this.system.getSystemHealth();
        const recommendations = await this.getRecommendations();

        this.systemState.update(state => ({
          ...state,
          health,
          recommendations,
          metrics: {
            ...state.metrics,
            cpuUsage: health.performance?.cpuUsage?.user || 0,
            memoryUsage: health.performance?.memoryUsage?.heapUsed || 0,
            activeConnections: health.performance?.activeWorkers || 0,
            cacheHitRate: this.calculateCacheHitRate(health),
            processingQueue: health.performance?.queueDepths?.total || 0
          }
        }));

      } catch (error) {
        console.warn('Metrics collection failed:', error);
      }
    }, 2000); // Every 2 seconds
  }

  private calculateCacheHitRate(health: unknown): number {
    // Calculate cache hit rate from system health data
    const cacheStats = health.performance?.cacheStats;
    if (!cacheStats) return 0;

    const hits = cacheStats.hits || 0;
    const total = (cacheStats.hits || 0) + (cacheStats.misses || 0);
    
    return total > 0 ? (hits / total) * 100 : 0;
  }

  // Public API methods
  async processDocument(documentId: string, content: string, options: unknown = {}) {
    if (!this.system) throw new Error('System not initialized');
    return this.system.processDocumentOptimized(documentId, content, options);
  }

  async performAnalysis(sessionId: string, documents: unknown[], options: unknown = {}) {
    if (!this.system) throw new Error('System not initialized');
    return this.system.performComprehensiveAnalysis(sessionId, documents, options);
  }

  async getSystemHealth() {
    if (!this.system) return {};
    return this.system.getSystemHealth();
  }

  async getPerformanceMetrics() {
    if (!this.system) return {};
    
    const health = this.system.getSystemHealth();
    return {
      timestamp: Date.now(),
      cpu: health.performance?.cpuUsage || {},
      memory: health.performance?.memoryUsage || {},
      workers: health.performance?.activeWorkers || 0,
      cache: {
        hitRate: this.calculateCacheHitRate(health),
        size: health.performance?.cacheSize || 0
      },
      components: health.components || {},
      uptime: health.system?.uptime || 0
    };
  }

  async getRecommendations() {
    if (!this.system) return [];

    // Mock recommendations based on system state - would integrate with actual recommendation engine
    const health = this.system.getSystemHealth();
    const recommendations = [];

    // Performance-based recommendations
    const memUsage = health.performance?.memoryUsage?.heapUsed || 0;
    if (memUsage > 512) { // > 512MB
      recommendations.push({
        id: 'memory-optimization',
        type: 'performance',
        priority: 'medium',
        title: 'Memory Usage High',
        description: 'Consider optimizing memory usage or increasing heap size',
        action: 'Optimize cache settings',
        confidence: 0.8
      });
    }

    const cpuUsage = health.performance?.cpuUsage?.user || 0;
    if (cpuUsage > 80) {
      recommendations.push({
        id: 'cpu-optimization',
        type: 'performance',
        priority: 'high',
        title: 'High CPU Usage',
        description: 'CPU usage is above 80%. Consider optimizing worker threads.',
        action: 'Adjust worker thread configuration',
        confidence: 0.9
      });
    }

    // Component-based recommendations
    const components = health.components || {};
    Object.entries(components).forEach(([name, status]) => {
      if (status === 'inactive' || status === 'error') {
        recommendations.push({
          id: `component-${name}`,
          type: 'system',
          priority: 'critical',
          title: `${name} Component Issue`,
          description: `The ${name} component is ${status}`,
          action: `Restart ${name} service`,
          confidence: 0.95
        });
      }
    });

    return recommendations;
  }

  logAnalysis(result: unknown) {
    console.log('📊 Analysis logged:', result);
    
    // Update analytics through the system
    if (this.system && this.system.components?.analyticsService) {
      // Would integrate with actual analytics service
    }
  }

  logInteraction(interaction: unknown) {
    console.log('👤 Interaction logged:', interaction);
    
    // Update user behavior tracking
    if (this.system && this.system.components?.recommendationEngine) {
      // Would integrate with actual recommendation engine
    }
  }

  // Store subscription methods
  subscribe(callback: (value: SystemState) => void) {
    return this.systemState.subscribe(callback);
  }

  get systemState() {
    return this.systemState;
  }

  async shutdown() {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }

    if (this.system) {
      await this.system.shutdown();
      this.system = null;
    }

    this.systemState.set({
      initialized: false,
      health: {},
      performance: {},
      components: {},
      metrics: {
        cpuUsage: 0,
        memoryUsage: 0,
        cacheHitRate: 0,
        activeConnections: 0,
        processingQueue: 0
      },
      recommendations: [],
      errors: []
    });

    console.log('🔄 AI System shutdown complete');
  }
}

// Export factory function to create store instances
export function createAISystemStore() {
  return new AISystemStore();
}

// Export types
export type { SystemConfig, SystemState };