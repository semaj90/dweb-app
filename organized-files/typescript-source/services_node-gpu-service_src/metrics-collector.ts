/**
 * Metrics Collector - Performance monitoring and reporting
 */

import { Logger } from './logger';

interface PerformanceMetrics {
  embeddingsProcessed: number;
  totalProcessingTime: number;
  avgProcessingTime: number;
  gpuUtilization: number;
  memoryUsage: number;
  cacheHits: number;
  cacheMisses: number;
  cacheHitRate: number;
  errorCount: number;
  uptime: number;
}

interface MetricPoint {
  timestamp: number;
  value: number;
  tags?: Record<string, string>;
}

export class MetricsCollector {
  private logger: Logger;
  private metrics: Map<string, MetricPoint[]> = new Map();
  private startTime: number;
  private isRunning: boolean = false;
  private intervalId: NodeJS.Timeout | null = null;

  // Performance counters
  private embeddingCount: number = 0;
  private totalProcessingTime: number = 0;
  private cacheHits: number = 0;
  private cacheMisses: number = 0;
  private errorCount: number = 0;

  constructor(logger: Logger) {
    this.logger = logger;
    this.startTime = Date.now();
  }

  start(interval: number = 5000): void {
    if (this.isRunning) {
      this.logger.warn('Metrics collector is already running');
      return;
    }

    this.isRunning = true;
    this.logger.info(`📊 Starting metrics collection (interval: ${interval}ms)`);

    this.intervalId = setInterval(() => {
      this.collectSystemMetrics();
    }, interval);
  }

  stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.logger.info('📊 Stopped metrics collection');
  }

  recordEmbedding(processingTime: number, cacheHit: boolean = false): void {
    this.embeddingCount++;
    this.totalProcessingTime += processingTime;
    
    if (cacheHit) {
      this.cacheHits++;
    } else {
      this.cacheMisses++;
    }

    this.recordMetric('embedding_processing_time', processingTime, {
      cache_hit: cacheHit.toString()
    });
  }

  recordError(errorType: string = 'unknown'): void {
    this.errorCount++;
    this.recordMetric('error_count', 1, { type: errorType });
  }

  private collectSystemMetrics(): void {
    const now = Date.now();
    
    // Memory usage
    const memUsage = process.memoryUsage();
    this.recordMetric('memory_heap_used', memUsage.heapUsed / 1024 / 1024); // MB
    this.recordMetric('memory_heap_total', memUsage.heapTotal / 1024 / 1024); // MB
    this.recordMetric('memory_rss', memUsage.rss / 1024 / 1024); // MB

    // CPU usage (simplified)
    const cpuUsage = process.cpuUsage();
    this.recordMetric('cpu_user', cpuUsage.user / 1000); // ms
    this.recordMetric('cpu_system', cpuUsage.system / 1000); // ms

    // Uptime
    this.recordMetric('uptime', (now - this.startTime) / 1000); // seconds

    // Processing rates
    const uptime = (now - this.startTime) / 1000;
    const embeddingRate = this.embeddingCount / uptime;
    this.recordMetric('embedding_rate', embeddingRate); // embeddings/second

    // Cache metrics
    const totalCacheOps = this.cacheHits + this.cacheMisses;
    const hitRate = totalCacheOps > 0 ? (this.cacheHits / totalCacheOps) * 100 : 0;
    this.recordMetric('cache_hit_rate', hitRate);
  }

  private recordMetric(name: string, value: number, tags?: Record<string, string>): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }

    const metricPoints = this.metrics.get(name)!;
    metricPoints.push({
      timestamp: Date.now(),
      value,
      tags
    });

    // Keep only last 1000 points per metric
    if (metricPoints.length > 1000) {
      metricPoints.shift();
    }
  }

  getMetrics(): PerformanceMetrics {
    const uptime = (Date.now() - this.startTime) / 1000;
    const avgProcessingTime = this.embeddingCount > 0 ? this.totalProcessingTime / this.embeddingCount : 0;
    const totalCacheOps = this.cacheHits + this.cacheMisses;
    const cacheHitRate = totalCacheOps > 0 ? (this.cacheHits / totalCacheOps) * 100 : 0;

    return {
      embeddingsProcessed: this.embeddingCount,
      totalProcessingTime: this.totalProcessingTime,
      avgProcessingTime,
      gpuUtilization: this.estimateGpuUtilization(),
      memoryUsage: process.memoryUsage().heapUsed / 1024 / 1024,
      cacheHits: this.cacheHits,
      cacheMisses: this.cacheMisses,
      cacheHitRate,
      errorCount: this.errorCount,
      uptime
    };
  }

  getMetricHistory(name: string, limit: number = 100): MetricPoint[] {
    const points = this.metrics.get(name) || [];
    return points.slice(-limit);
  }

  getMetricSummary(name: string, windowMs: number = 60000): {
    avg: number;
    min: number;
    max: number;
    count: number;
  } | null {
    const points = this.metrics.get(name);
    if (!points || points.length === 0) {
      return null;
    }

    const now = Date.now();
    const windowStart = now - windowMs;
    const recentPoints = points.filter(p => p.timestamp >= windowStart);

    if (recentPoints.length === 0) {
      return null;
    }

    const values = recentPoints.map(p => p.value);
    return {
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length
    };
  }

  private estimateGpuUtilization(): number {
    // Simplified GPU utilization estimation
    // In real implementation, this would query GPU stats
    const recentEmbeddings = this.embeddingCount;
    const uptime = (Date.now() - this.startTime) / 1000;
    const rate = recentEmbeddings / uptime;
    
    // Estimate utilization based on processing rate
    // Assuming max rate of ~100 embeddings/second for 100% utilization
    return Math.min((rate / 100) * 100, 100);
  }

  exportMetrics(format: 'json' | 'prometheus' = 'json'): string {
    if (format === 'prometheus') {
      return this.exportPrometheusFormat();
    }

    const metrics = this.getMetrics();
    const allMetrics: Record<string, any> = {};
    
    for (const [name, points] of this.metrics) {
      allMetrics[name] = {
        current: points[points.length - 1]?.value || 0,
        history: points.slice(-10) // Last 10 points
      };
    }

    return JSON.stringify({
      summary: metrics,
      detailed: allMetrics,
      timestamp: new Date().toISOString()
    }, null, 2);
  }

  private exportPrometheusFormat(): string {
    const metrics = this.getMetrics();
    const lines: string[] = [];

    // Add metrics in Prometheus format
    lines.push(`# HELP embeddings_processed_total Total number of embeddings processed`);
    lines.push(`# TYPE embeddings_processed_total counter`);
    lines.push(`embeddings_processed_total ${metrics.embeddingsProcessed}`);

    lines.push(`# HELP avg_processing_time_ms Average processing time in milliseconds`);
    lines.push(`# TYPE avg_processing_time_ms gauge`);
    lines.push(`avg_processing_time_ms ${metrics.avgProcessingTime}`);

    lines.push(`# HELP gpu_utilization_percent GPU utilization percentage`);
    lines.push(`# TYPE gpu_utilization_percent gauge`);
    lines.push(`gpu_utilization_percent ${metrics.gpuUtilization}`);

    lines.push(`# HELP memory_usage_mb Memory usage in megabytes`);
    lines.push(`# TYPE memory_usage_mb gauge`);
    lines.push(`memory_usage_mb ${metrics.memoryUsage}`);

    lines.push(`# HELP cache_hit_rate_percent Cache hit rate percentage`);
    lines.push(`# TYPE cache_hit_rate_percent gauge`);
    lines.push(`cache_hit_rate_percent ${metrics.cacheHitRate}`);

    lines.push(`# HELP error_count_total Total number of errors`);
    lines.push(`# TYPE error_count_total counter`);
    lines.push(`error_count_total ${metrics.errorCount}`);

    return lines.join('\n') + '\n';
  }

  reset(): void {
    this.embeddingCount = 0;
    this.totalProcessingTime = 0;
    this.cacheHits = 0;
    this.cacheMisses = 0;
    this.errorCount = 0;
    this.metrics.clear();
    this.startTime = Date.now();
    
    this.logger.info('📊 Metrics reset');
  }
}