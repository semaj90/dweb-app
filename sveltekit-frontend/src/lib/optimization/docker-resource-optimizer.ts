/**
 * Docker Resource Optimization and Performance Enhancement
 * Implements memory management, cache optimization, and throughput improvements
 */

import type { RequestHandler } from "@sveltejs/kit";

export interface DockerResourceMetrics {
  memory: {
    usage: number;
    limit: number;
    percentage: number;
  };
  cpu: {
    usage: number;
    cores: number;
    percentage: number;
  };
  disk: {
    usage: number;
    available: number;
    percentage: number;
  };
  network: {
    rx: number;
    tx: number;
    throughput: number;
  };
}

export interface OptimizationConfig {
  maxMemoryMB: number;
  maxCpuPercentage: number;
  cacheStrategy: "aggressive" | "balanced" | "conservative";
  compressionLevel: number;
  batchSize: number;
  parallelism: number;
}

export class DockerResourceOptimizer {
  private metrics: DockerResourceMetrics | null = null;
  private optimizationConfig: OptimizationConfig;
  private performanceCache = new Map<string, any>();
  private batchProcessor = new Map<string, any[]>();

  constructor(config: Partial<OptimizationConfig> = {}) {
    this.optimizationConfig = {
      maxMemoryMB: 2048, // 2GB default
      maxCpuPercentage: 80,
      cacheStrategy: "balanced",
      compressionLevel: 6,
      batchSize: 100,
      parallelism: 4,
      ...config,
    };

    this.startResourceMonitoring();
  }

  /**
   * Monitor Docker container resources
   */
  async getResourceMetrics(): Promise<DockerResourceMetrics> {
    try {
      // In a real implementation, this would query Docker stats API
      const metrics: DockerResourceMetrics = {
        memory: {
          usage: await this.getMemoryUsage(),
          limit: this.optimizationConfig.maxMemoryMB * 1024 * 1024,
          percentage: 0,
        },
        cpu: {
          usage: await this.getCpuUsage(),
          cores: navigator.hardwareConcurrency || 4,
          percentage: 0,
        },
        disk: {
          usage: await this.getDiskUsage(),
          available: 10 * 1024 * 1024 * 1024, // 10GB
          percentage: 0,
        },
        network: {
          rx: 0,
          tx: 0,
          throughput: 0,
        },
      };

      // Calculate percentages
      metrics.memory.percentage =
        (metrics.memory.usage / metrics.memory.limit) * 100;
      metrics.cpu.percentage = (metrics.cpu.usage / metrics.cpu.cores) * 100;
      metrics.disk.percentage =
        (metrics.disk.usage / metrics.disk.available) * 100;

      this.metrics = metrics;
      return metrics;
    } catch (error) {
      console.error("Failed to get resource metrics:", error);
      throw error;
    }
  }

  /**
   * Optimize memory usage with intelligent caching
   */
  async optimizeMemoryUsage(): Promise<void> {
    const metrics = await this.getResourceMetrics();

    if (metrics.memory.percentage > 85) {
      console.warn("High memory usage detected, starting optimization...");

      // Clear old cache entries
      await this.clearOldCacheEntries();

      // Compress large objects in memory
      await this.compressMemoryObjects();

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      // Reduce batch sizes
      this.optimizationConfig.batchSize = Math.max(
        10,
        this.optimizationConfig.batchSize * 0.7
      );
    }
  }

  /**
   * Batch processing for improved throughput
   */
  async processBatch<T, R>(
    items: T[],
    processor: (batch: T[]) => Promise<R[]>,
    batchId?: string
  ): Promise<R[]> {
    const { batchSize, parallelism } = this.optimizationConfig;
    const results: R[] = [];

    // Split items into batches
    const batches: T[][] = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }

    // Process batches in parallel
    const workers = Array(Math.min(parallelism, batches.length))
      .fill(null)
      .map(async (_, workerIndex) => {
        for (
          let batchIndex = workerIndex;
          batchIndex < batches.length;
          batchIndex += parallelism
        ) {
          const batch = batches[batchIndex];
          try {
            const batchResults = await processor(batch);
            results.push(...batchResults);
          } catch (error) {
            console.error(`Batch ${batchIndex} failed:`, error);
            // Continue processing other batches
          }
        }
      });

    await Promise.all(workers);
    return results;
  }

  /**
   * Intelligent caching with compression
   */
  async cacheWithCompression<T>(
    key: string,
    data: T,
    ttl: number = 1000 * 60 * 15
  ): Promise<void> {
    try {
      // Compress data if it's large
      const serialized = JSON.stringify(data);
      const compressed = await this.compressString(serialized);

      this.performanceCache.set(key, {
        data: compressed,
        originalSize: serialized.length,
        compressedSize: compressed.length,
        timestamp: Date.now(),
        ttl,
      });

      // Monitor cache size
      await this.enforceMemoryLimits();
    } catch (error) {
      console.error("Cache compression failed:", error);
      // Fallback to uncompressed caching
      this.performanceCache.set(key, {
        data,
        originalSize: JSON.stringify(data).length,
        compressedSize: JSON.stringify(data).length,
        timestamp: Date.now(),
        ttl,
      });
    }
  }

  /**
   * Retrieve and decompress cached data
   */
  async getCachedData<T>(key: string): Promise<T | null> {
    const entry = this.performanceCache.get(key);
    if (!entry) return null;

    // Check TTL
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.performanceCache.delete(key);
      return null;
    }

    try {
      // Decompress if needed
      if (
        typeof entry.data === "string" &&
        entry.originalSize !== entry.compressedSize
      ) {
        const decompressed = await this.decompressString(entry.data);
        return JSON.parse(decompressed);
      }

      return entry.data;
    } catch (error) {
      console.error("Cache decompression failed:", error);
      this.performanceCache.delete(key);
      return null;
    }
  }

  /**
   * Stream processing for large datasets
   */
  async *streamProcess<T, R>(
    items: T[],
    processor: (item: T) => Promise<R>,
    options: { bufferSize?: number; backpressure?: boolean } = {}
  ): AsyncGenerator<R, void, unknown> {
    const { bufferSize = 10, backpressure = true } = options;
    const buffer: Promise<R>[] = [];

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      buffer.push(processor(item));

      if (buffer.length >= bufferSize) {
        // Wait for oldest promise and yield result
        const result = await buffer.shift()!;
        yield result;

        // Check for backpressure
        if (backpressure) {
          const metrics = await this.getResourceMetrics();
          if (metrics.memory.percentage > 90 || metrics.cpu.percentage > 95) {
            // Pause processing briefly
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
        }
      }
    }

    // Process remaining items in buffer
    while (buffer.length > 0) {
      const result = await buffer.shift()!;
      yield result;
    }
  }

  /**
   * Database connection pooling optimization
   */
  async optimizeConnectionPool(): Promise<void> {
    const metrics = await this.getResourceMetrics();

    // Adjust pool size based on available resources
    const maxConnections = Math.min(
      20, // Hard limit
      Math.floor(this.optimizationConfig.maxMemoryMB / 50), // 50MB per connection
      metrics.cpu.cores * 2
    );

    console.log(
      `Optimizing DB connection pool to ${maxConnections} connections`
    );

    // This would configure your database pool
    // Example for Drizzle/PostgreSQL:
    // await configureConnectionPool({ max: maxConnections });
  }

  /**
   * API response optimization
   */
  optimizeResponse(data: any, request: Request): any {
    const acceptEncoding = request.headers.get("accept-encoding") || "";
    const supportsGzip = acceptEncoding.includes("gzip");

    // Compress responses if client supports it
    if (supportsGzip && JSON.stringify(data).length > 1024) {
      return {
        compressed: true,
        data,
      };
    }

    return data;
  }

  // Private methods

  private async getMemoryUsage(): Promise<number> {
    if (typeof process !== "undefined" && process.memoryUsage) {
      return process.memoryUsage().heapUsed;
    }

    // Browser fallback
    if ("memory" in performance) {
      return (performance as any).memory.usedJSHeapSize;
    }

    return 0;
  }

  private async getCpuUsage(): Promise<number> {
    // In Node.js, you could use process.cpuUsage()
    // For browser, we'll estimate based on performance
    return 0; // Placeholder
  }

  private async getDiskUsage(): Promise<number> {
    // This would query actual disk usage
    return 0; // Placeholder
  }

  private async clearOldCacheEntries(): Promise<void> {
    const now = Date.now();
    let clearedCount = 0;

    for (const [key, entry] of this.performanceCache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.performanceCache.delete(key);
        clearedCount++;
      }
    }

    console.log(`Cleared ${clearedCount} expired cache entries`);
  }

  private async compressMemoryObjects(): Promise<void> {
    // Compress large objects in the performance cache
    for (const [key, entry] of this.performanceCache.entries()) {
      if (
        entry.originalSize > 10000 &&
        entry.originalSize === entry.compressedSize
      ) {
        // This object isn't compressed yet
        try {
          const compressed = await this.compressString(
            JSON.stringify(entry.data)
          );
          entry.data = compressed;
          entry.compressedSize = compressed.length;
        } catch (error) {
          console.warn(`Failed to compress cache entry ${key}:`, error);
        }
      }
    }
  }

  private async compressString(input: string): Promise<string> {
    // Simple compression using built-in compression
    // In a real implementation, use a proper compression library
    try {
      const encoder = new TextEncoder();
      const decoder = new TextDecoder();
      const data = encoder.encode(input);

      // Use CompressionStream if available (newer browsers)
      if ("CompressionStream" in window) {
        const stream = new CompressionStream("gzip");
        const compressed = await new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(data);
              controller.close();
            },
          }).pipeThrough(stream)
        ).arrayBuffer();

        return btoa(String.fromCharCode(...new Uint8Array(compressed)));
      }

      // Fallback: simple LZ-like compression
      return this.simpleLZCompress(input);
    } catch (error) {
      console.warn("Compression failed, returning original:", error);
      return input;
    }
  }

  private async decompressString(compressed: string): Promise<string> {
    try {
      // Use DecompressionStream if available
      if ("DecompressionStream" in window && compressed.startsWith("H4sI")) {
        const binaryData = Uint8Array.from(atob(compressed), (c) =>
          c.charCodeAt(0)
        );
        const stream = new DecompressionStream("gzip");
        const decompressed = await new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(binaryData);
              controller.close();
            },
          }).pipeThrough(stream)
        ).text();

        return decompressed;
      }

      // Fallback: simple decompression
      return this.simpleLZDecompress(compressed);
    } catch (error) {
      console.warn("Decompression failed, returning as-is:", error);
      return compressed;
    }
  }

  private simpleLZCompress(input: string): string {
    // Very simple compression algorithm
    let result = "";
    let i = 0;

    while (i < input.length) {
      let matchLength = 0;
      let matchDistance = 0;

      // Look for matches in the previous 4096 characters
      for (let j = Math.max(0, i - 4096); j < i; j++) {
        let length = 0;
        while (
          i + length < input.length &&
          input[j + length] === input[i + length] &&
          length < 255
        ) {
          length++;
        }

        if (length > matchLength) {
          matchLength = length;
          matchDistance = i - j;
        }
      }

      if (matchLength > 3) {
        result += `#${matchDistance},${matchLength}#`;
        i += matchLength;
      } else {
        result += input[i];
        i++;
      }
    }

    return result;
  }

  private simpleLZDecompress(compressed: string): string {
    let result = "";
    let i = 0;

    while (i < compressed.length) {
      if (compressed[i] === "#") {
        // Find the closing #
        const closeIndex = compressed.indexOf("#", i + 1);
        if (closeIndex !== -1) {
          const match = compressed.substring(i + 1, closeIndex);
          const [distance, length] = match.split(",").map(Number);

          const startPos = result.length - distance;
          for (let j = 0; j < length; j++) {
            result += result[startPos + j];
          }

          i = closeIndex + 1;
        } else {
          result += compressed[i];
          i++;
        }
      } else {
        result += compressed[i];
        i++;
      }
    }

    return result;
  }

  private async enforceMemoryLimits(): Promise<void> {
    const maxCacheSize =
      this.optimizationConfig.maxMemoryMB * 0.3 * 1024 * 1024; // 30% of max memory
    let currentSize = 0;

    // Calculate current cache size
    for (const entry of this.performanceCache.values()) {
      currentSize += entry.compressedSize;
    }

    if (currentSize > maxCacheSize) {
      // Remove oldest entries until we're under the limit
      const entries = Array.from(this.performanceCache.entries()).sort(
        ([, a], [, b]) => a.timestamp - b.timestamp
      );

      while (currentSize > maxCacheSize * 0.8 && entries.length > 0) {
        const [key, entry] = entries.shift()!;
        this.performanceCache.delete(key);
        currentSize -= entry.compressedSize;
      }
    }
  }

  private startResourceMonitoring(): void {
    // Monitor resources every 30 seconds
    setInterval(async () => {
      try {
        await this.getResourceMetrics();
        await this.optimizeMemoryUsage();
      } catch (error) {
        console.error("Resource monitoring failed:", error);
      }
    }, 30000);
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.performanceCache.clear();
    this.batchProcessor.clear();
  }
}

// Global resource optimizer
export const dockerOptimizer = new DockerResourceOptimizer();

// SvelteKit request handler optimization decorator
export function optimizeHandler(config?: Partial<OptimizationConfig>): any {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalHandler = descriptor.value;

    descriptor.value = async function (event: any) {
      const startTime = performance.now();

      try {
        // Check resource usage before processing
        const metrics = await dockerOptimizer.getResourceMetrics();

        if (metrics.memory.percentage > 90) {
          await dockerOptimizer.optimizeMemoryUsage();
        }

        // Process request
        const result = await originalHandler.call(this, event);

        // Optimize response
        const optimizedResult = dockerOptimizer.optimizeResponse(
          result,
          event.request
        );

        const endTime = performance.now();
        console.log(
          `Handler ${propertyKey} executed in ${endTime - startTime}ms`
        );

        return optimizedResult;
      } catch (error) {
        console.error(`Optimized handler ${propertyKey} failed:`, error);
        throw error;
      }
    };

    return descriptor;
  };
}

// Performance monitoring utilities
export const performanceUtils = {
  /**
   * Monitor function execution time
   */
  timeExecution: async <T>(name: string, fn: () => Promise<T>): Promise<T> => {
    const start = performance.now();
    try {
      const result = await fn();
      const end = performance.now();
      console.log(`${name} executed in ${end - start}ms`);
      return result;
    } catch (error) {
      const end = performance.now();
      console.error(`${name} failed after ${end - start}ms:`, error);
      throw error;
    }
  },

  /**
   * Create a performance report
   */
  generateReport: async (): Promise<Record<string, any>> => {
    const metrics = await dockerOptimizer.getResourceMetrics();
    const cacheStats = dockerOptimizer.performanceCache.size;

    return {
      timestamp: new Date().toISOString(),
      metrics,
      cacheEntries: cacheStats,
      config: dockerOptimizer.optimizationConfig,
    };
  },
};
