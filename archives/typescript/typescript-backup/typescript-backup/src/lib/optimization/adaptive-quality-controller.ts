// Adaptive Quality Controller for AI System Performance Management
export class AdaptiveQualityController {
  private qualityLevel: number = 0.8;
  private performanceMetrics: Map<string, number> = new Map();
  
  constructor(private initialQuality: number = 0.8) {
    this.qualityLevel = initialQuality;
  }

  getQualityLevel(): number {
    return this.qualityLevel;
  }

  updateQuality(metrics: { latency?: number; throughput?: number; accuracy?: number }): void {
    // Adaptive quality adjustment based on performance metrics
    const { latency = 0, throughput = 0, accuracy = 0 } = metrics;
    
    this.performanceMetrics.set('latency', latency);
    this.performanceMetrics.set('throughput', throughput);
    this.performanceMetrics.set('accuracy', accuracy);

    // Adjust quality based on performance
    if (latency > 1000) { // High latency
      this.qualityLevel = Math.max(0.1, this.qualityLevel - 0.1);
    } else if (latency < 100) { // Low latency
      this.qualityLevel = Math.min(1.0, this.qualityLevel + 0.05);
    }
  }

  getPerformanceMetrics(): Record<string, number> {
    return Object.fromEntries(this.performanceMetrics);
  }

  reset(): void {
    this.qualityLevel = this.initialQuality;
    this.performanceMetrics.clear();
  }

  /**
   * Current quality level as string for math-optimized system
   */
  get currentQuality(): string {
    if (this.qualityLevel < 0.3) return 'low';
    if (this.qualityLevel < 0.7) return 'standard';
    return 'high';
  }

  /**
   * Adjust quality based on metrics and return quality level
   */
  adjustQuality(metrics: any): string {
    const cpuUsage = metrics.cpuUsage || 0;
    const memoryUsage = metrics.memoryUsage || 0;
    
    if (cpuUsage > 85 || memoryUsage > 1500) {
      this.qualityLevel = Math.max(0.1, this.qualityLevel - 0.2);
    } else if (cpuUsage < 50 && memoryUsage < 1000) {
      this.qualityLevel = Math.min(1.0, this.qualityLevel + 0.1);
    } else {
      // Maintain current level for moderate load
    }
    
    return this.currentQuality;
  }
}