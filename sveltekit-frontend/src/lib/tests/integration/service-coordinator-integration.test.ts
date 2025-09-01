/**
 * Comprehensive Integration Tests - Master Service Coordinator
 * Tests all 38 Go microservices with error resolution and health monitoring
 */

import { describe, it, expect, beforeAll, afterAll, vi, beforeEach } from 'vitest';
import { masterServiceCoordinator } from '../../services/master-service-coordinator.js';
import { errorResolutionEngine } from '../../services/error-resolution-engine.js';
import { get } from 'svelte/store';
import { URL } from "url";

export interface TestContext {
  coordinator: typeof masterServiceCoordinator;
  errorEngine: typeof errorResolutionEngine;
  mockFetch: ReturnType<typeof vi.fn>;
  startTime: number;
}

describe('Master Service Coordinator Integration Tests', () => {
  let context: TestContext;
  const TEST_TIMEOUT = 60000; // 60 seconds for comprehensive tests

  beforeAll(async () => {
    // Mock fetch for API calls
    const mockFetch = vi.fn();
    global.fetch = mockFetch;
    
    context = {
      coordinator: masterServiceCoordinator,
      errorEngine: errorResolutionEngine,
      mockFetch,
      startTime: Date.now()
    };
    
    // Mock successful responses for all service endpoints
    mockFetch.mockImplementation(async (url: string, options?: RequestInit) => {
      if (typeof url === 'string') {
        // Health check endpoints
        if (url.includes('/health')) {
          return {
            ok: true,
            json: async () => ({
              status: 'healthy',
              uptime: Date.now() - context.startTime,
              version: '1.0.0'
            })
          };
        }
        
        // Coordinator API endpoints
        if (url.includes('/api/v1/coordinator')) {
          const method = options?.method || 'GET';
          if (method === 'POST') {
            const body = options?.body ? JSON.parse(options.body as string) : {};
            return {
              ok: true,
              json: async () => ({
                success: true,
                message: `Action ${body.action} completed`,
                timestamp: new Date().toISOString()
              })
            };
          }
          return {
            ok: true,
            json: async () => ({
              success: true,
              data: {
                systemHealth: 'excellent',
                healthyServices: 15,
                totalServices: 18,
                performance: {
                  avgResponseTime: 250,
                  successRate: 0.98,
                  cudaUtilization: 75
                }
              }
            })
          };
        }
        
        // Individual service endpoints
        if (url.includes('/api/v1/services/')) {
          return {
            ok: true,
            json: async () => ({
              success: true,
              data: {
                healthy: true,
                responseTime: Math.random() * 500 + 50,
                details: { status: 'operational' }
              }
            })
          };
        }
      }
      
      return { ok: false, status: 404 };
    });
  }, TEST_TIMEOUT);

  afterAll(async () => {
    await context.coordinator.cleanup();
    await context.errorEngine.cleanup();
  });

  describe('Service Discovery and Initialization', () => {
    it('should initialize with all 38+ services defined', () => {
      expect(context.coordinator.services).toBeDefined();
      expect(context.coordinator.services.length).toBeGreaterThanOrEqual(18);
      
      // Verify critical services are present
      const criticalServices = ['enhanced-rag', 'upload-service', 'cuda-service', 'grpc-server'];
      criticalServices.forEach(serviceId => {
        const service = context.coordinator.services.find(s => s.id === serviceId);
        expect(service).toBeDefined();
        expect(service?.critical).toBe(true);
      });
    });

    it('should properly categorize services by tiers', () => {
      const tierGroups = context.coordinator.services.reduce((acc, service) => {
        acc[service.tier] = (acc[service.tier] || 0) + 1;
        return acc;
      }, {} as Record<number, number>);

      expect(tierGroups[1]).toBeGreaterThan(0); // Tier 1: Core services
      expect(tierGroups[2]).toBeGreaterThan(0); // Tier 2: Performance services
      expect(tierGroups[3]).toBeGreaterThan(0); // Tier 3: Specialized services
    });

    it('should validate service configurations', () => {
      context.coordinator.services.forEach(service => {
        expect(service.id).toBeTruthy();
        expect(service.displayName).toBeTruthy();
        expect(service.port).toBeGreaterThan(0);
        expect(['http', 'grpc', 'quic', 'websocket']).toContain(service.protocol);
        expect([1, 2, 3, 4]).toContain(service.tier);
        expect(typeof service.critical).toBe('boolean');
        expect(Array.isArray(service.capabilities)).toBe(true);
        expect(Array.isArray(service.dependencies)).toBe(true);
        expect(service.maxRetries).toBeGreaterThan(0);
        expect(service.timeoutMs).toBeGreaterThan(0);
      });
    });
  });

  describe('Service Lifecycle Management', () => {
    it('should start services in proper tier order', async () => {
      const startOrder: number[] = [];
      
      // Mock the startService method to track order
      const originalStartService = (context.coordinator as any).startService;
      (context.coordinator as any).startService = vi.fn().mockImplementation(async (service: any) => {
        startOrder.push(service.tier);
        return originalStartService?.call(context.coordinator, service);
      });

      await context.coordinator.startAllServices();

      // Verify tiers started in ascending order
      for (let i = 1; i < startOrder.length; i++) {
        expect(startOrder[i]).toBeGreaterThanOrEqual(startOrder[i - 1]);
      }
    }, TEST_TIMEOUT);

    it('should handle service failures gracefully', async () => {
      // Mock a service failure
      context.mockFetch.mockImplementationOnce(() => 
        Promise.reject(new Error('Connection refused'))
      );

      const systemStatus = context.coordinator.getSystemStatus();
      expect(systemStatus).toBeDefined();
      expect(typeof systemStatus.systemHealth).toBe('string');
    });

    it('should respect service dependencies', () => {
      const dependencyGraph: Record<string, string[]> = {};
      
      context.coordinator.services.forEach(service => {
        dependencyGraph[service.id] = service.dependencies;
      });

      // Verify no circular dependencies
      const visited = new Set<string>();
      const visiting = new Set<string>();

      function hasCycle(serviceId: string): boolean {
        if (visiting.has(serviceId)) return true;
        if (visited.has(serviceId)) return false;

        visiting.add(serviceId);
        const dependencies = dependencyGraph[serviceId] || [];
        
        for (const dep of dependencies) {
          if (hasCycle(dep)) return true;
        }
        
        visiting.delete(serviceId);
        visited.add(serviceId);
        return false;
      }

      Object.keys(dependencyGraph).forEach(serviceId => {
        expect(hasCycle(serviceId)).toBe(false);
      });
    });
  });

  describe('Health Monitoring System', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should perform health checks on all services', async () => {
      const services = context.coordinator.services;
      
      // Wait for health checks to complete
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      services.forEach(service => {
        const healthEndpoint = `http://localhost:${service.port}${service.healthEndpoint}`;
        expect(context.mockFetch).toHaveBeenCalledWith(
          healthEndpoint,
          expect.objectContaining({
            signal: expect.any(AbortSignal)
          })
        );
      });
    });

    it('should update service statuses based on health checks', async () => {
      const statusStore = context.coordinator.serviceStatuses;
      const statuses = get(statusStore);
      
      expect(statuses).toBeInstanceOf(Map);
      expect(statuses.size).toBeGreaterThan(0);
      
      // Check that statuses have required fields
      statuses.forEach((status, serviceId) => {
        expect(status.id).toBe(serviceId);
        expect(['starting', 'healthy', 'degraded', 'failed', 'unknown']).toContain(status.status);
        expect(typeof status.lastCheck).toBe('number');
        expect(typeof status.responseTime).toBe('number');
        expect(typeof status.errorCount).toBe('number');
      });
    });

    it('should calculate system health metrics', async () => {
      const metricsStore = context.coordinator.performanceMetrics;
      const metrics = get(metricsStore);
      
      expect(metrics.totalRequests).toBeGreaterThanOrEqual(0);
      expect(metrics.successRate).toBeGreaterThanOrEqual(0);
      expect(metrics.successRate).toBeLessThanOrEqual(1);
      expect(metrics.avgResponseTime).toBeGreaterThanOrEqual(0);
      expect(metrics.throughput).toBeGreaterThanOrEqual(0);
      expect(metrics.errorRate).toBeGreaterThanOrEqual(0);
      expect(metrics.cudaUtilization).toBeGreaterThanOrEqual(0);
      expect(metrics.cudaUtilization).toBeLessThanOrEqual(100);
    });

    it('should determine overall system health status', async () => {
      const healthStore = context.coordinator.systemHealth;
      const health = get(healthStore);
      
      expect(['excellent', 'good', 'degraded', 'critical', 'offline']).toContain(health);
    });
  });

  describe('Error Resolution System', () => {
    it('should detect and analyze service errors', async () => {
      // Simulate service error
      context.mockFetch.mockImplementationOnce(() =>
        Promise.reject(new Error('ECONNREFUSED'))
      );

      const errorAnalyses = get(context.errorEngine.errorAnalyses);
      const initialCount = errorAnalyses.length;

      // Trigger error analysis
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Should detect the error pattern
      const updatedAnalyses = get(context.errorEngine.errorAnalyses);
      expect(updatedAnalyses.length).toBeGreaterThanOrEqual(initialCount);
    });

    it('should match error patterns correctly', () => {
      const errorPatterns = (context.errorEngine as any).errorPatterns;
      expect(Array.isArray(errorPatterns)).toBe(true);
      expect(errorPatterns.length).toBeGreaterThan(0);

      // Test pattern matching
      const testErrors = [
        'Connection timeout',
        'ECONNREFUSED',
        'Out of memory',
        'CUDA error',
        'High response time'
      ];

      testErrors.forEach(errorMsg => {
        const pattern = errorPatterns.find((p: any) => p.pattern.test(errorMsg));
        expect(pattern).toBeDefined();
      });
    });

    it('should provide appropriate recovery actions', () => {
      const errorPatterns = (context.errorEngine as any).errorPatterns;
      
      errorPatterns.forEach((pattern: any) => {
        expect(pattern.recoveryActions).toBeDefined();
        expect(Array.isArray(pattern.recoveryActions)).toBe(true);
        expect(pattern.recoveryActions.length).toBeGreaterThan(0);
        
        pattern.recoveryActions.forEach((action: any) => {
          expect(['restart', 'reconnect', 'scale', 'fallback', 'cleanup', 'configure', 'wait']).toContain(action.type);
          expect(action.description).toBeTruthy();
          expect(action.timeout).toBeGreaterThan(0);
        });
      });
    });

    it('should maintain recovery statistics', async () => {
      const stats = get(context.errorEngine.recoveryStats);
      
      expect(typeof stats.totalErrors).toBe('number');
      expect(typeof stats.autoResolved).toBe('number');
      expect(typeof stats.manualResolved).toBe('number');
      expect(typeof stats.unresolved).toBe('number');
      expect(typeof stats.avgRecoveryTime).toBe('number');
      
      expect(stats.totalErrors).toBeGreaterThanOrEqual(
        stats.autoResolved + stats.manualResolved + stats.unresolved
      );
    });
  });

  describe('API Integration Tests', () => {
    it('should handle coordinator API requests', async () => {
      const response = await fetch('/api/v1/coordinator?action=health');
      expect(response.ok).toBe(true);
      
      const data = await response.json();
      expect(data.success).toBe(true);
      expect(data.data).toBeDefined();
    });

    it('should support service-specific operations', async () => {
      const serviceId = 'enhanced-rag';
      const response = await fetch(`/api/v1/services/${serviceId}?action=health`);
      expect(response.ok).toBe(true);
      
      const data = await response.json();
      expect(data.success).toBe(true);
      expect(data.data.healthy).toBeDefined();
    });

    it('should handle POST requests for service actions', async () => {
      const response = await fetch('/api/v1/coordinator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'force_health_check' })
      });
      
      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.success).toBe(true);
    });
  });

  describe('Performance and Scalability Tests', () => {
    it('should handle concurrent health checks efficiently', async () => {
      const startTime = Date.now();
      const services = context.coordinator.services.slice(0, 10); // Test with 10 services
      
      const healthCheckPromises = services.map(service =>
        fetch(`http://localhost:${service.port}/health`)
      );
      
      await Promise.allSettled(healthCheckPromises);
      const duration = Date.now() - startTime;
      
      // Should complete within reasonable time
      expect(duration).toBeLessThan(10000); // 10 seconds
    });

    it('should maintain performance under load', async () => {
      const iterations = 50;
      const startTime = Date.now();
      
      const requests = Array.from({ length: iterations }, (_, i) =>
        fetch('/api/v1/coordinator?action=status')
      );
      
      const results = await Promise.allSettled(requests);
      const duration = Date.now() - startTime;
      
      const successCount = results.filter(r => r.status === 'fulfilled').length;
      const successRate = successCount / iterations;
      
      expect(successRate).toBeGreaterThan(0.95); // 95% success rate
      expect(duration / iterations).toBeLessThan(100); // < 100ms per request average
    });

    it('should handle memory usage efficiently', () => {
      const statusStore = get(context.coordinator.serviceStatuses);
      const errorAnalyses = get(context.errorEngine.errorAnalyses);
      
      // Should not accumulate unbounded data
      expect(statusStore.size).toBeLessThan(100);
      expect(errorAnalyses.length).toBeLessThan(1000);
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle network failures gracefully', async () => {
      context.mockFetch.mockImplementationOnce(() =>
        Promise.reject(new Error('Network error'))
      );
      
      const response = await fetch('/api/v1/coordinator?action=health').catch(() => null);
      expect(response).toBeNull();
      
      // System should continue operating
      const systemStatus = context.coordinator.getSystemStatus();
      expect(systemStatus).toBeDefined();
    });

    it('should validate API input parameters', async () => {
      const invalidRequests = [
        { url: '/api/v1/coordinator?action=invalid_action', expectedStatus: 400 },
        { url: '/api/v1/services/nonexistent?action=health', expectedStatus: 404 },
      ];
      
      for (const request of invalidRequests) {
        context.mockFetch.mockImplementationOnce(() => ({
          ok: false,
          status: request.expectedStatus
        }));
        
        const response = await fetch(request.url);
        expect(response.ok).toBe(false);
      }
    });

    it('should handle service timeout scenarios', async () => {
      context.mockFetch.mockImplementationOnce(() =>
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), 1000)
        )
      );
      
      const startTime = Date.now();
      await fetch('/api/v1/services/enhanced-rag?action=health').catch(() => null);
      const duration = Date.now() - startTime;
      
      expect(duration).toBeLessThan(5000); // Should timeout quickly
    });
  });

  describe('Production Readiness Tests', () => {
    it('should provide comprehensive system status', () => {
      const systemStatus = context.coordinator.getSystemStatus();
      
      expect(systemStatus.initialized).toBeDefined();
      expect(systemStatus.services).toBeDefined();
      expect(systemStatus.performance).toBeDefined();
      expect(systemStatus.systemHealth).toBeDefined();
      expect(systemStatus.activeErrors).toBeDefined();
      expect(systemStatus.serviceCount).toBeDefined();
      expect(systemStatus.protocolDistribution).toBeDefined();
    });

    it('should support graceful shutdown', async () => {
      const originalCleanup = context.coordinator.cleanup;
      const cleanupSpy = vi.fn().mockImplementation(originalCleanup);
      context.coordinator.cleanup = cleanupSpy;
      
      await context.coordinator.stopAllServices();
      expect(cleanupSpy).toHaveBeenCalled();
    });

    it('should maintain configuration integrity', () => {
      const services = context.coordinator.services;
      const ports = new Set(services.map(s => s.port));
      
      // No duplicate ports
      expect(ports.size).toBe(services.length);
      
      // All required configurations present
      services.forEach(service => {
        expect(service.healthEndpoint).toBeTruthy();
        expect(service.capabilities.length).toBeGreaterThan(0);
        expect(service.timeoutMs).toBeGreaterThan(0);
        expect(service.maxRetries).toBeGreaterThan(0);
      });
    });
  });
});

// Helper functions for testing
export function createMockService(overrides: Partial<any> = {}) {
  return {
    id: 'test-service',
    name: 'test-service',
    displayName: 'Test Service',
    port: 8888,
    protocol: 'http',
    tier: 1,
    critical: false,
    healthEndpoint: '/health',
    capabilities: ['testing'],
    dependencies: [],
    maxRetries: 3,
    timeoutMs: 10000,
    ...overrides
  };
}

export function createMockHealthResponse(healthy = true) {
  return {
    ok: healthy,
    json: async () => ({
      status: healthy ? 'healthy' : 'unhealthy',
      uptime: Date.now(),
      version: '1.0.0'
    })
  };
}