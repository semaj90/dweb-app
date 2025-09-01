import type { RequestHandler } from './$types.js';

/**
 * SvelteKit API Route: GPU Tensor Processing
 * Integrates with Go GPU microservice and provides load balancing
 */


import { ensureError } from '$lib/utils/ensure-error';
import { json, error } from '@sveltejs/kit';
import { dev } from '$app/environment';
import { URL } from "url";

// GPU service pool for load balancing
const gpuServicePool = [
  'http://localhost:8095',  // Primary GPU service
  'http://localhost:8096',  // Secondary GPU service
  'http://localhost:8097',  // Tertiary GPU service
];

// Service health tracking
export interface ServiceHealth {
  url: string;
  healthy: boolean;
  lastCheck: number;
  responseTime: number;
  errorCount: number;
}

class GPUServiceManager {
  private serviceHealth: Map<string, ServiceHealth> = new Map();
  private currentServiceIndex = 0;
  private healthCheckInterval: NodeJS.Timeout | null = null;

  constructor() {
    // Initialize health tracking for all services
    gpuServicePool.forEach(url => {
      this.serviceHealth.set(url, {
        url,
        healthy: true,
        lastCheck: 0,
        responseTime: 0,
        errorCount: 0
      });
    });

    // Start periodic health checks
    this.startHealthChecks();
  }

  private startHealthChecks(): void {
    this.healthCheckInterval = setInterval(async () => {
      await this.checkAllServicesHealth();
    }, 30000); // Check every 30 seconds
  }

  private async checkAllServicesHealth(): Promise<void> {
    const healthPromises = gpuServicePool.map(url => this.checkServiceHealth(url));
    await Promise.all(healthPromises);
  }

  private async checkServiceHealth(url: string): Promise<void> {
    const startTime = Date.now();
    const health = this.serviceHealth.get(url)!;

    try {
      const response = await fetch(`${url}/health`, {
        method: 'GET',
        timeout: 5000 // 5 second timeout
      } as RequestInit);

      if (response.ok) {
        health.healthy = true;
        health.responseTime = Date.now() - startTime;
        health.errorCount = Math.max(0, health.errorCount - 1); // Reduce error count on success
      } else {
        health.healthy = false;
        health.errorCount++;
      }
    } catch (error: any) {
      health.healthy = false;
      health.errorCount++;
    }

    health.lastCheck = Date.now();
  }

  getHealthyService(): string | null {
    // Get healthy services sorted by response time
    const healthyServices = Array.from(this.serviceHealth.values())
      .filter(service => service.healthy)
      .sort((a, b) => a.responseTime - b.responseTime);

    if (healthyServices.length === 0) {
      return null;
    }

    // Round-robin with preference for fastest services
    const service = healthyServices[this.currentServiceIndex % healthyServices.length];
    this.currentServiceIndex++;
    return service.url;
  }

  getServiceForHash(hash: string): string {
    const healthyServices = Array.from(this.serviceHealth.values())
      .filter(service => service.healthy);

    if (healthyServices.length === 0) {
      return gpuServicePool[0]; // Fallback to primary
    }

    // Consistent hashing
    const hashCode = this.hashString(hash);
    const serviceIndex = hashCode % healthyServices.length;
    return healthyServices[serviceIndex].url;
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  getHealthStats() {
    return Array.from(this.serviceHealth.values()).map(health => ({
      url: health.url,
      healthy: health.healthy,
      responseTime: health.responseTime,
      errorCount: health.errorCount,
      lastCheck: health.lastCheck
    }));
  }

  cleanup(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
  }
}

// Global service manager instance
let serviceManager: GPUServiceManager;

// Initialize service manager in development
if (dev) {
  serviceManager = new GPUServiceManager();
} else {
  // In production, you might want to use a different initialization strategy
  serviceManager = new GPUServiceManager();
}

// Request processing statistics
export interface ProcessingStats {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  cacheHits: number;
  averageProcessingTime: number;
}

const stats: ProcessingStats = {
  totalRequests: 0,
  successfulRequests: 0,
  failedRequests: 0,
  cacheHits: 0,
  averageProcessingTime: 0
};

// POST: Process tensor with GPU acceleration
export const POST: RequestHandler = async ({ request, getClientAddress, url }) => {
  const startTime = Date.now();
  stats.totalRequests++;

  try {
    const tensorData = await request.json();

    // Validate tensor data structure
    if (!tensorData.shape || !tensorData.data) {
      stats.failedRequests++;
      throw error(400, ensureError({
        message: 'Invalid tensor data: missing shape or data fields'
      }));
    }

    // Validate tensor shape
    if (!Array.isArray(tensorData.shape) || tensorData.shape.some((dim: any) => typeof dim !== 'number' || dim <= 0)) {
      stats.failedRequests++;
      throw error(400, ensureError({
        message: 'Invalid tensor shape: must be array of positive integers'
      }));
    }

    // Validate tensor data
    const expectedSize = tensorData.shape.reduce((a: number, b: number) => a * b, 1);
    if (!Array.isArray(tensorData.data) || tensorData.data.length !== expectedSize) {
      stats.failedRequests++;
      throw error(400, ensureError({
        message: 'Tensor data size mismatch'
      }));
    }

    // Generate cache key for consistent routing
    const cacheKey = generateCacheKey(tensorData);

    // Enhance tensor data with metadata
    const enhancedTensorData = {
      ...tensorData,
      cacheKey,
      timestamp: Date.now(),
      requestId: generateRequestId(),
      clientAddress: getClientAddress(),
      userAgent: request.headers.get('user-agent'),
      context: 'legal-ai-processing',
      dimensions: tensorData.shape.length,
      layout: tensorData.layout || 'standard',
      lodLevel: tensorData.lodLevel || 0
    };

    // Select appropriate GPU service
    const targetService = serviceManager.getServiceForHash(cacheKey);

    if (!targetService) {
      stats.failedRequests++;
      throw error(503, ensureError({
        message: 'All GPU services unavailable'
      }));
    }

    // Process with primary service
    const result = await processWithService(targetService, enhancedTensorData);

    // Update statistics
    const processingTime = Date.now() - startTime;
    updateProcessingStats(processingTime, result.cache_hit || false);
    stats.successfulRequests++;

    return json({
      success: true,
      data: result.data,
      metadata: {
        processingTime,
        cacheHit: result.cache_hit || false,
        service: targetService,
        route: result.route || generateRouteHash(cacheKey),
        requestId: enhancedTensorData.requestId,
        tensorStats: result.metadata?.tensorStats,
        optimizationLevel: result.metadata?.optimizationLevel || 'standard',
        gpuMemoryUsed: result.metadata?.gpuMemoryUsed || 0
      },
      stats: {
        totalRequests: stats.totalRequests,
        successRate: (stats.successfulRequests / stats.totalRequests) * 100,
        averageProcessingTime: stats.averageProcessingTime,
        cacheHitRate: (stats.cacheHits / stats.totalRequests) * 100
      }
    });

  } catch (err: any) {
    stats.failedRequests++;
    console.error('GPU tensor processing error:', err);

    if (err.status) {
      // Re-throw SvelteKit errors
      throw err;
    }

    throw error(500, ensureError({
      message: `Processing failed: ${(err as Error).message}`
    }));
  }
};

// GET: Retrieve processing statistics and service health
export const GET: RequestHandler = async ({ url }) => {
  const statsType = url.searchParams.get('type');

  try {
    switch (statsType) {
      case 'health':
        return json({
          serviceHealth: serviceManager.getHealthStats(),
          timestamp: Date.now()
        });

      case 'stats':
        return json({
          processing: {
            totalRequests: stats.totalRequests,
            successfulRequests: stats.successfulRequests,
            failedRequests: stats.failedRequests,
            successRate: stats.totalRequests > 0 ?
              (stats.successfulRequests / stats.totalRequests) * 100 : 0,
            cacheHitRate: stats.totalRequests > 0 ?
              (stats.cacheHits / stats.totalRequests) * 100 : 0,
            averageProcessingTime: stats.averageProcessingTime
          },
          services: serviceManager.getHealthStats(),
          timestamp: Date.now()
        });

      case 'full':
      default:
        // Get detailed stats from primary GPU service
        const primaryService = gpuServicePool[0];
        let serviceStats = null;

        try {
          const response = await fetch(`${primaryService}/stats`);
          if (response.ok) {
            serviceStats = await response.json();
          }
        } catch (error: any) {
          console.warn('Failed to fetch service stats:', error);
        }

        return json({
          api: {
            processing: stats,
            services: serviceManager.getHealthStats()
          },
          gpuService: serviceStats,
          timestamp: Date.now(),
          uptime: process.uptime(),
          environment: dev ? 'development' : 'production'
        });
    }
  } catch (err: any) {
    console.error('Stats retrieval error:', err);
    throw error(500, ensureError({
      message: `Stats retrieval failed: ${err.message}`,
      code: 'STATS_ERROR'
    }));
  }
};

// DELETE: Clear caches and reset statistics (development only)
export const DELETE: RequestHandler = async ({ url }) => {
  if (!dev) {
    throw error(403, ensureError({
      message: 'Cache clearing only available in development mode',
      code: 'PRODUCTION_PROTECTION'
    }));
  }

  try {
    const clearType = url.searchParams.get('type') || 'cache';

    if (clearType === 'stats' || clearType === 'all') {
      // Reset API statistics
      stats.totalRequests = 0;
      stats.successfulRequests = 0;
      stats.failedRequests = 0;
      stats.cacheHits = 0;
      stats.averageProcessingTime = 0;
    }

    if (clearType === 'cache' || clearType === 'all') {
      // Clear caches in GPU services
      const clearPromises = gpuServicePool.map(async (serviceUrl) => {
        try {
          const response = await fetch(`${serviceUrl}/stats`, {
            method: 'DELETE'
          });
          return { service: serviceUrl, success: response.ok };
        } catch (error: any) {
          return { service: serviceUrl, success: false, error: error.message };
        }
      });

      const results = await Promise.all(clearPromises);

      return json({
        success: true,
        message: `${clearType} cleared successfully`,
        details: {
          clearedType: clearType,
          serviceResults: results
        }
      });
    }

    return json({
      success: true,
      message: 'Operation completed'
    });

  } catch (err: any) {
    console.error('Cache clearing error:', err);
    throw error(500, ensureError({
      message: `Cache clearing failed: ${err.message}`,
      code: 'CACHE_CLEAR_ERROR'
    }));
  }
};

// Helper functions
async function processWithService(serviceUrl: string, tensorData: any): Promise<any> {
  const maxRetries = 2;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(`${serviceUrl}/process-tensor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': tensorData.requestId,
          'X-Client-IP': tensorData.clientAddress,
          'X-Cache-Key': tensorData.cacheKey,
          'X-Attempt': (attempt + 1).toString(),
          'User-Agent': tensorData.userAgent || 'SvelteKit-API'
        },
        body: JSON.stringify(tensorData),
        // Add timeout
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });

      if (!response.ok) {
        throw new Error(`GPU service error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(`GPU processing failed: ${result.error || 'Unknown error'}`);
      }

      return result;

    } catch (error: any) {
      lastError = error as Error;
      console.warn(`Attempt ${attempt + 1} failed for service ${serviceUrl}:`, error.message);

      if (attempt < maxRetries - 1) {
        // Wait before retry with exponential backoff
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }
  }

  // All attempts failed, try fallback services
  const healthyServices = serviceManager.getHealthStats()
    .filter(s => s.healthy && s.url !== serviceUrl)
    .sort((a, b) => a.responseTime - b.responseTime);

  for (const fallbackService of healthyServices) {
    try {
      console.log(`Trying fallback service: ${fallbackService.url}`);

      const response = await fetch(`${fallbackService.url}/process-tensor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': tensorData.requestId,
          'X-Fallback': 'true'
        },
        body: JSON.stringify(tensorData),
        signal: AbortSignal.timeout(15000) // Shorter timeout for fallback
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          return result;
        }
      }
    } catch (error: any) {
      console.warn(`Fallback service ${fallbackService.url} failed:`, error.message);
    }
  }

  throw lastError || new Error('All GPU services failed');
}

function generateCacheKey(tensorData: any): string {
  const shapeStr = tensorData.shape.join('x');
  const layout = tensorData.layout || 'standard';
  const lodLevel = tensorData.lodLevel || 0;
  const dataHash = hashArray(tensorData.data, 100); // Hash first 100 elements

  const key = `${shapeStr}_${layout}_${lodLevel}_${dataHash}`;
  return btoa(key).replace(/[+/=]/g, ''); // Base64 encode and remove special chars
}

function hashArray(arr: number[], sampleSize: number): string {
  const sample = arr.slice(0, Math.min(sampleSize, arr.length));
  let hash = 0;

  for (let i = 0; i < sample.length; i++) {
    const value = Math.round(sample[i] * 1000); // Precision to 3 decimals
    hash = ((hash << 5) - hash) + value;
    hash = hash & hash; // Convert to 32-bit integer
  }

  return Math.abs(hash).toString(36);
}

function generateRequestId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substr(2, 9);
  return `req_${timestamp}_${random}`;
}

function generateRouteHash(cacheKey: string): string {
  let hash = 0;
  for (let i = 0; i < cacheKey.length; i++) {
    const char = cacheKey.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return `route_${Math.abs(hash).toString(36)}`;
}

function updateProcessingStats(processingTime: number, cacheHit: boolean): void {
  if (cacheHit) {
    stats.cacheHits++;
  }

  // Update average processing time
  if (stats.successfulRequests > 0) {
    const totalTime = stats.averageProcessingTime * (stats.successfulRequests - 1);
    stats.averageProcessingTime = (totalTime + processingTime) / stats.successfulRequests;
  } else {
    stats.averageProcessingTime = processingTime;
  }
}

// Cleanup on process exit
if (typeof process !== 'undefined') {
  process.on('exit', () => {
    serviceManager?.cleanup();
  });

  process.on('SIGINT', () => {
    serviceManager?.cleanup();
    process.exit(0);
  });
}