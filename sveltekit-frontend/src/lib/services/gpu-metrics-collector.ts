/**
 * GPU Metrics Collector
 * Integrates frontend visual effects performance data with Go GPUMetricsCollector service
 * Provides comprehensive performance monitoring for N64 visual effects system
 */

export interface GPUMetrics {
  // Performance metrics
  fps: number;
  frameTime: number; // milliseconds
  memoryUsage: number; // MB
  
  // Effect status
  effectsActive: string[];
  qualityLevel: 'low' | 'medium' | 'high' | 'ultra';
  
  // Hardware detection
  gpuRenderer?: string;
  gpuVendor?: string;
  
  // Context information
  timestamp: number;
  sessionId: string;
  userId?: string;
  
  // Browser/system info
  userAgent: string;
  screenResolution: string;
  devicePixelRatio: number;
  
  // Performance thresholds
  performanceGrade: 'A+' | 'A' | 'B' | 'C' | 'D' | 'F';
  adaptiveQualityChanges: number;
}

export interface GPUMetricsConfig {
  // Collection settings
  collectionInterval: number; // milliseconds
  batchSize: number;
  maxRetries: number;
  
  // API endpoints
  localEndpoint: string;
  goServiceEndpoint: string;
  fallbackEndpoint?: string;
  
  // Performance thresholds
  fpsThresholds: {
    excellent: number; // 60+
    good: number;      // 45+
    fair: number;      // 30+
    poor: number;      // 15+
  };
  
  // Privacy settings
  includeUserAgent: boolean;
  includeUserId: boolean;
  anonymizeData: boolean;
}

export class GPUMetricsCollector {
  private config: GPUMetricsConfig;
  private sessionId: string;
  private metricsQueue: GPUMetrics[] = [];
  private collectionTimer: number | null = null;
  private adaptiveQualityChanges = 0;
  
  // WebGL context for hardware detection
  private gl: WebGLRenderingContext | null = null;
  private gpuInfo: { renderer?: string; vendor?: string } = {};
  
  constructor(config: Partial<GPUMetricsConfig> = {}) {
    this.config = {
      collectionInterval: 5000, // 5 seconds
      batchSize: 10,
      maxRetries: 3,
      localEndpoint: '/api/metrics/gpu',
      goServiceEndpoint: 'http://localhost:8094/gpu-metrics',
      fpsThresholds: {
        excellent: 60,
        good: 45,
        fair: 30,
        poor: 15
      },
      includeUserAgent: true,
      includeUserId: true,
      anonymizeData: false,
      ...config
    };
    
    this.sessionId = this.generateSessionId();
    this.initializeGPUDetection();
  }
  
  /**
   * Initialize WebGL context for hardware detection
   */
  private initializeGPUDetection(): void {
    try {
      const canvas = document.createElement('canvas');
      this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      
      if (this.gl) {
        // Get GPU info using WebGL extensions
        const debugInfo = this.gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          this.gpuInfo.renderer = this.gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
          this.gpuInfo.vendor = this.gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        }
      }
    } catch (error) {
      console.warn('GPU detection failed:', error);
    }
  }
  
  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    return `gpu_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * Calculate performance grade based on FPS and frame time
   */
  private calculatePerformanceGrade(fps: number, frameTime: number): 'A+' | 'A' | 'B' | 'C' | 'D' | 'F' {
    const { fpsThresholds } = this.config;
    
    if (fps >= fpsThresholds.excellent && frameTime <= 16.67) return 'A+';
    if (fps >= fpsThresholds.excellent) return 'A';
    if (fps >= fpsThresholds.good) return 'B';
    if (fps >= fpsThresholds.fair) return 'C';
    if (fps >= fpsThresholds.poor) return 'D';
    return 'F';
  }
  
  /**
   * Collect current GPU metrics from effect manager
   */
  collectMetrics(
    performanceData: { fps: number; frameTime: number; memoryUsage: number },
    effectData: { activeEffects: string[]; qualityLevel: string },
    userId?: string
  ): GPUMetrics {
    const metrics: GPUMetrics = {
      // Performance data
      fps: performanceData.fps,
      frameTime: performanceData.frameTime,
      memoryUsage: performanceData.memoryUsage,
      
      // Effect data
      effectsActive: effectData.activeEffects,
      qualityLevel: effectData.qualityLevel as 'low' | 'medium' | 'high' | 'ultra',
      
      // Hardware info
      gpuRenderer: this.gpuInfo.renderer,
      gpuVendor: this.gpuInfo.vendor,
      
      // Context
      timestamp: Date.now(),
      sessionId: this.sessionId,
      userId: this.config.includeUserId ? userId : undefined,
      
      // Browser/system
      userAgent: this.config.includeUserAgent ? navigator.userAgent : '',
      screenResolution: `${screen.width}x${screen.height}`,
      devicePixelRatio: window.devicePixelRatio || 1,
      
      // Performance analysis
      performanceGrade: this.calculatePerformanceGrade(performanceData.fps, performanceData.frameTime),
      adaptiveQualityChanges: this.adaptiveQualityChanges
    };
    
    // Anonymize if requested
    if (this.config.anonymizeData) {
      metrics.userId = undefined;
      metrics.userAgent = 'anonymized';
    }
    
    return metrics;
  }
  
  /**
   * Record adaptive quality change
   */
  recordQualityChange(): void {
    this.adaptiveQualityChanges++;
  }
  
  /**
   * Add metrics to collection queue
   */
  addMetrics(metrics: GPUMetrics): void {
    this.metricsQueue.push(metrics);
    
    // Auto-flush if queue is full
    if (this.metricsQueue.length >= this.config.batchSize) {
      this.flushMetrics();
    }
  }
  
  /**
   * Send metrics to backend services
   */
  private async flushMetrics(): Promise<void> {
    if (this.metricsQueue.length === 0) return;
    
    const batch = [...this.metricsQueue];
    this.metricsQueue = [];
    
    try {
      // Try Go service first (primary)
      await this.sendToGoService(batch);
    } catch (goError) {
      console.warn('Go service unavailable, trying local endpoint:', goError);
      
      try {
        // Fallback to SvelteKit endpoint
        await this.sendToLocalEndpoint(batch);
      } catch (localError) {
        console.error('All GPU metrics endpoints failed:', { goError, localError });
        
        // Re-queue metrics for retry (with limit)
        if (batch.length <= this.config.maxRetries * this.config.batchSize) {
          this.metricsQueue.unshift(...batch);
        }
      }
    }
  }
  
  /**
   * Send metrics directly to Go GPUMetricsCollector service
   */
  private async sendToGoService(metrics: GPUMetrics[]): Promise<void> {
    const response = await fetch(this.config.goServiceEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': this.sessionId
      },
      body: JSON.stringify({
        timestamp: Date.now(),
        sessionId: this.sessionId,
        metricsCount: metrics.length,
        metrics: metrics
      })
    });
    
    if (!response.ok) {
      throw new Error(`Go service responded with ${response.status}: ${response.statusText}`);
    }
  }
  
  /**
   * Send metrics to SvelteKit endpoint (which forwards to Go service)
   */
  private async sendToLocalEndpoint(metrics: GPUMetrics[]): Promise<void> {
    const response = await fetch(this.config.localEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': this.sessionId
      },
      body: JSON.stringify({
        timestamp: Date.now(),
        sessionId: this.sessionId,
        metricsCount: metrics.length,
        metrics: metrics
      })
    });
    
    if (!response.ok) {
      throw new Error(`Local endpoint responded with ${response.status}: ${response.statusText}`);
    }
  }
  
  /**
   * Start automatic metrics collection
   */
  startCollection(): void {
    if (this.collectionTimer) {
      this.stopCollection();
    }
    
    this.collectionTimer = window.setInterval(() => {
      this.flushMetrics();
    }, this.config.collectionInterval);
  }
  
  /**
   * Stop automatic metrics collection
   */
  stopCollection(): void {
    if (this.collectionTimer) {
      clearInterval(this.collectionTimer);
      this.collectionTimer = null;
    }
    
    // Flush remaining metrics
    this.flushMetrics();
  }
  
  /**
   * Get current queue status
   */
  getStatus(): {
    queueLength: number;
    sessionId: string;
    gpuInfo: { renderer?: string; vendor?: string };
    isCollecting: boolean;
    adaptiveQualityChanges: number;
  } {
    return {
      queueLength: this.metricsQueue.length,
      sessionId: this.sessionId,
      gpuInfo: this.gpuInfo,
      isCollecting: this.collectionTimer !== null,
      adaptiveQualityChanges: this.adaptiveQualityChanges
    };
  }
  
  /**
   * Reset metrics collection
   */
  reset(): void {
    this.stopCollection();
    this.metricsQueue = [];
    this.adaptiveQualityChanges = 0;
    this.sessionId = this.generateSessionId();
  }
  
  /**
   * Clean up resources
   */
  destroy(): void {
    this.stopCollection();
    this.metricsQueue = [];
    this.gl = null;
  }
}

// Singleton instance for global use
export const gpuMetricsCollector = new GPUMetricsCollector();

// Helper function to integrate with existing effect manager
export function integrateWithEffectManager(effectManager: any, userId?: string): () => void {
  let lastMetricsTime = 0;
  const METRICS_INTERVAL = 1000; // Collect every second
  
  // Start automatic collection
  gpuMetricsCollector.startCollection();
  
  // Monitor for quality changes
  const originalSetQuality = effectManager.setQuality;
  effectManager.setQuality = function(quality: string) {
    gpuMetricsCollector.recordQualityChange();
    return originalSetQuality.call(this, quality);
  };
  
  // Periodic metrics collection
  const collectLoop = () => {
    const now = Date.now();
    if (now - lastMetricsTime >= METRICS_INTERVAL) {
      const performanceData = effectManager.getPerformanceMetrics();
      const effectData = {
        activeEffects: Object.keys(effectManager.activeEffects || {}),
        qualityLevel: effectManager.getQuality()
      };
      
      const metrics = gpuMetricsCollector.collectMetrics(
        performanceData,
        effectData,
        userId
      );
      
      gpuMetricsCollector.addMetrics(metrics);
      lastMetricsTime = now;
    }
    
    requestAnimationFrame(collectLoop);
  };
  
  // Start collection loop
  collectLoop();
  
  // Return cleanup function
  return () => {
    gpuMetricsCollector.stopCollection();
  };
}