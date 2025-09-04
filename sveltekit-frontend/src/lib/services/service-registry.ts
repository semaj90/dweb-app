/**
 * Unified Service Registry
 * Centralized service health monitoring and caching for the YoRHa Legal AI Platform
 * Provides background health checks and unified service status
 */

import { browser } from '$app/environment';

interface ServiceHealth {
  status: 'online' | 'offline' | 'degraded' | 'unknown';
  lastCheck: number;
  responseTime: number;
  error?: string;
  metadata?: Record<string, any>;
}

interface SystemStatus {
  redis: boolean;
  postgres: boolean;
  minio: boolean;
  ollama: boolean;
  neo4j: boolean;
  services: Record<string, ServiceHealth>;
  lastUpdate: number;
  healthScore: number;
}

class ServiceRegistry {
  private cache: Map<string, any> = new Map();
  private healthChecks: Map<string, ServiceHealth> = new Map();
  private checkInterval: number | null = null;
  private isChecking = false;

  // Service endpoints for health checks
  private readonly services = {
    redis: { url: '/api/test-health', method: 'GET' },
    postgres: { url: '/api/test-health', method: 'GET' },
    minio: { url: '/api/test-health', method: 'GET' },
    ollama: { url: '/api/test-health', method: 'GET' },
    neo4j: { url: '/api/test-health', method: 'GET' },
    gallery: { url: '/api/gallery', method: 'GET' },
    legalPlatform: { url: '/api/v2/legal-platform', method: 'POST', body: { action: 'health', entity: 'case' } },
    contextMCP: { url: '/api/test-context7', method: 'GET' },
    aiUnified: { url: '/api/ai/unified', method: 'POST', body: { query: 'health check' } }
  };

  constructor() {
    if (browser) {
      this.startHealthChecking();
    }
  }

  /**
   * Start periodic health checking
   */
  private startHealthChecking() {
    if (this.checkInterval) return;

    // Initial check
    this.performHealthChecks();

    // Set up periodic checks every 30 seconds
    this.checkInterval = window.setInterval(() => {
      this.performHealthChecks();
    }, 30000);
  }

  /**
   * Stop health checking
   */
  stopHealthChecking() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  /**
   * Perform health checks for all services
   */
  private async performHealthChecks() {
    if (this.isChecking) return;
    this.isChecking = true;

    const checks = Object.entries(this.services).map(async ([serviceName, config]) => {
      const startTime = Date.now();
      let health: ServiceHealth;

      try {
        const response = await fetch(config.url, {
          method: config.method,
          headers: {
            'Content-Type': 'application/json'
          },
          body: config.body ? JSON.stringify(config.body) : undefined,
          signal: AbortSignal.timeout(10000) // 10 second timeout
        });

        const responseTime = Date.now() - startTime;

        if (response.ok) {
          const data = await response.json().catch(() => ({}));
          health = {
            status: 'online',
            lastCheck: Date.now(),
            responseTime,
            metadata: data
          };
        } else {
          health = {
            status: 'degraded',
            lastCheck: Date.now(),
            responseTime,
            error: `HTTP ${response.status}: ${response.statusText}`
          };
        }
      } catch (error: any) {
        health = {
          status: 'offline',
          lastCheck: Date.now(),
          responseTime: Date.now() - startTime,
          error: error.message || 'Service unavailable'
        };
      }

      this.healthChecks.set(serviceName, health);
    });

    await Promise.allSettled(checks);
    this.isChecking = false;

    // Update cached system status
    this.updateSystemStatus();
  }

  /**
   * Update the cached system status based on health checks
   */
  private updateSystemStatus() {
    const services: Record<string, ServiceHealth> = {};
    let totalScore = 0;
    let serviceCount = 0;

    for (const [serviceName, health] of this.healthChecks) {
      services[serviceName] = health;
      
      // Calculate health score contribution
      if (health.status === 'online') {
        totalScore += 100;
      } else if (health.status === 'degraded') {
        totalScore += 50;
      }
      serviceCount++;
    }

    const healthScore = serviceCount > 0 ? Math.round(totalScore / serviceCount) : 0;

    const systemStatus: SystemStatus = {
      redis: this.getServiceStatus('redis') === 'online',
      postgres: this.getServiceStatus('postgres') === 'online',
      minio: this.getServiceStatus('minio') === 'online',
      ollama: this.getServiceStatus('ollama') === 'online',
      neo4j: this.getServiceStatus('neo4j') === 'online',
      services,
      lastUpdate: Date.now(),
      healthScore
    };

    this.cache.set('system:status', systemStatus);
  }

  /**
   * Get system status (cached with automatic refresh)
   */
  async getSystemStatus(): Promise<SystemStatus> {
    const cached = this.cache.get('system:status');
    
    if (cached && (Date.now() - cached.lastUpdate) < 30000) {
      return cached;
    }

    // If no recent cached data, perform immediate check
    if (!this.isChecking) {
      await this.performHealthChecks();
    }

    return this.cache.get('system:status') || {
      redis: false,
      postgres: false,
      minio: false,
      ollama: false,
      neo4j: false,
      services: {},
      lastUpdate: Date.now(),
      healthScore: 0
    };
  }

  /**
   * Get specific service health
   */
  getServiceHealth(serviceName: string): ServiceHealth | null {
    return this.healthChecks.get(serviceName) || null;
  }

  /**
   * Get service status shorthand
   */
  getServiceStatus(serviceName: string): 'online' | 'offline' | 'degraded' | 'unknown' {
    const health = this.healthChecks.get(serviceName);
    return health?.status || 'unknown';
  }

  /**
   * Cache arbitrary data with TTL
   */
  setCache(key: string, value: any, ttlMs = 300000): void { // 5 minute default TTL
    this.cache.set(key, {
      value,
      expires: Date.now() + ttlMs
    });
  }

  /**
   * Get cached data
   */
  getCache<T>(key: string): T | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    if (Date.now() > cached.expires) {
      this.cache.delete(key);
      return null;
    }

    return cached.value;
  }

  /**
   * Clear all cached data
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get service registry statistics
   */
  getStats() {
    return {
      cacheSize: this.cache.size,
      healthChecksCount: this.healthChecks.size,
      isHealthChecking: this.isChecking,
      lastHealthCheck: Math.max(...Array.from(this.healthChecks.values()).map(h => h.lastCheck)),
      services: Array.from(this.healthChecks.keys())
    };
  }

  /**
   * Test specific service connectivity
   */
  async testService(serviceName: string): Promise<ServiceHealth> {
    const config = this.services[serviceName as keyof typeof this.services];
    if (!config) {
      throw new Error(`Unknown service: ${serviceName}`);
    }

    const startTime = Date.now();
    try {
      const response = await fetch(config.url, {
        method: config.method,
        headers: {
          'Content-Type': 'application/json'
        },
        body: config.body ? JSON.stringify(config.body) : undefined,
        signal: AbortSignal.timeout(10000)
      });

      const responseTime = Date.now() - startTime;
      
      if (response.ok) {
        const data = await response.json().catch(() => ({}));
        const health: ServiceHealth = {
          status: 'online',
          lastCheck: Date.now(),
          responseTime,
          metadata: data
        };
        
        this.healthChecks.set(serviceName, health);
        return health;
      } else {
        const health: ServiceHealth = {
          status: 'degraded',
          lastCheck: Date.now(),
          responseTime,
          error: `HTTP ${response.status}: ${response.statusText}`
        };
        
        this.healthChecks.set(serviceName, health);
        return health;
      }
    } catch (error: any) {
      const health: ServiceHealth = {
        status: 'offline',
        lastCheck: Date.now(),
        responseTime: Date.now() - startTime,
        error: error.message || 'Service unavailable'
      };
      
      this.healthChecks.set(serviceName, health);
      return health;
    }
  }

  /**
   * Force immediate health check refresh
   */
  async refreshHealthChecks(): Promise<void> {
    await this.performHealthChecks();
  }
}

// Export singleton instance
export const serviceRegistry = new ServiceRegistry();
// Export types for use in components
export type { SystemStatus, ServiceHealth };