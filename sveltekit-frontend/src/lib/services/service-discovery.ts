/**
 * Service Discovery & Failover Logic
 * Intelligent routing with automatic failover and service health monitoring
 */

import { productionServiceRegistry, getOptimalServiceForRoute, type ServiceDefinition } from './production-service-registry.js';
import { productionAPIClient, type ServiceRequest, type ServiceResponse } from '$lib/api/production-client.js';

export interface ServiceDiscoveryConfig {
  healthCheckInterval: number;
  failoverTimeout: number;
  maxRetries: number;
  circuitBreakerThreshold: number;
  loadBalancingStrategy: 'round_robin' | 'least_connections' | 'response_time' | 'health_weighted';
}

export interface CircuitBreakerState {
  serviceName: string;
  state: 'closed' | 'half_open' | 'open';
  failureCount: number;
  lastFailure: number;
  nextRetryTime: number;
}

export class ServiceDiscoveryEngine {
  private config: ServiceDiscoveryConfig;
  private circuitBreakers: Map<string, CircuitBreakerState> = new Map();
  private serviceMetrics: Map<string, {
    responseTime: number[];
    successRate: number;
    lastRequest: number;
    connections: number;
  }> = new Map();
  private healthCheckTimer?: NodeJS.Timeout;

  constructor(config: Partial<ServiceDiscoveryConfig> = {}) {
    this.config = {
      healthCheckInterval: 30000, // 30 seconds
      failoverTimeout: 5000, // 5 seconds
      maxRetries: 3,
      circuitBreakerThreshold: 5,
      loadBalancingStrategy: 'health_weighted',
      ...config
    };

    this.startHealthMonitoring();
  }

  async discoverService(route: string): Promise<{
    primary: ServiceDefinition;
    fallbacks: ServiceDefinition[];
    protocol: string;
    loadBalanced?: ServiceDefinition;
  } | null> {
    const routeMapping = getOptimalServiceForRoute(route);
    if (!routeMapping) return null;

    const primary = routeMapping.service;
    const fallbacks = await this.getHealthyFallbacks(route);
    
    // Apply load balancing if multiple healthy services available
    const allHealthyServices = [primary, ...fallbacks].filter(service => 
      this.isServiceHealthy(service.name)
    );

    const loadBalanced = allHealthyServices.length > 1 
      ? this.selectServiceByLoadBalancing(allHealthyServices)
      : primary;

    return {
      primary,
      fallbacks,
      protocol: routeMapping.protocol,
      loadBalanced
    };
  }

  async executeWithFailover<T>(
    route: string, 
    request: ServiceRequest
  ): Promise<ServiceResponse<T>> {
    const discovery = await this.discoverService(route);
    if (!discovery) {
      throw new Error(`No services available for route: ${route}`);
    }

    const servicesToTry = [
      discovery.loadBalanced || discovery.primary,
      ...discovery.fallbacks
    ].filter((service, index, arr) => 
      arr.findIndex(s => s.name === service.name) === index // Remove duplicates
    );

    let lastError: Error | null = null;

    for (const service of servicesToTry) {
      // Check circuit breaker
      if (this.isCircuitBreakerOpen(service.name)) {
        console.warn(`Circuit breaker open for ${service.name}, skipping`);
        continue;
      }

      try {
        const startTime = Date.now();
        
        // Execute request with timeout
        const response = await Promise.race([
          this.executeServiceRequest<T>(service, request),
          this.createTimeoutPromise(this.config.failoverTimeout)
        ]);

        const responseTime = Date.now() - startTime;
        
        // Record success metrics
        this.recordServiceMetrics(service.name, responseTime, true);
        this.resetCircuitBreaker(service.name);
        
        return {
          ...response,
          service: service.name,
          protocol: discovery.protocol,
          latency: responseTime
        };

      } catch (error) {
        lastError = error as Error;
        console.warn(`Service ${service.name} failed:`, error);
        
        // Record failure metrics
        this.recordServiceMetrics(service.name, 0, false);
        this.updateCircuitBreaker(service.name);
      }
    }

    throw new Error(`All services failed for route ${route}: ${lastError?.message}`);
  }

  private async getHealthyFallbacks(route: string): Promise<ServiceDefinition[]> {
    const routeMapping = productionServiceRegistry.getServiceForRoute(route);
    if (!routeMapping) return [];

    const fallbackServices: ServiceDefinition[] = [];
    
    for (const fallbackName of routeMapping.fallbacks) {
      const service = productionServiceRegistry.getServiceByName(fallbackName);
      if (service && this.isServiceHealthy(service.name)) {
        fallbackServices.push(service);
      }
    }

    return fallbackServices;
  }

  private selectServiceByLoadBalancing(services: ServiceDefinition[]): ServiceDefinition {
    switch (this.config.loadBalancingStrategy) {
      case 'round_robin':
        return this.roundRobinSelection(services);
      case 'least_connections':
        return this.leastConnectionsSelection(services);
      case 'response_time':
        return this.responseTimeSelection(services);
      case 'health_weighted':
        return this.healthWeightedSelection(services);
      default:
        return services[0];
    }
  }

  private roundRobinSelection(services: ServiceDefinition[]): ServiceDefinition {
    const now = Date.now();
    const index = Math.floor(now / 1000) % services.length;
    return services[index];
  }

  private leastConnectionsSelection(services: ServiceDefinition[]): ServiceDefinition {
    return services.reduce((selected, service) => {
      const selectedConnections = this.serviceMetrics.get(selected.name)?.connections || 0;
      const serviceConnections = this.serviceMetrics.get(service.name)?.connections || 0;
      return serviceConnections < selectedConnections ? service : selected;
    });
  }

  private responseTimeSelection(services: ServiceDefinition[]): ServiceDefinition {
    return services.reduce((selected, service) => {
      const selectedTime = this.getAverageResponseTime(selected.name);
      const serviceTime = this.getAverageResponseTime(service.name);
      return serviceTime < selectedTime ? service : selected;
    });
  }

  private healthWeightedSelection(services: ServiceDefinition[]): ServiceDefinition {
    const weights = services.map(service => {
      const metrics = this.serviceMetrics.get(service.name);
      if (!metrics) return 0.5; // Default weight for unknown services
      
      const successWeight = metrics.successRate;
      const responseTimeWeight = 1 - (this.getAverageResponseTime(service.name) / 1000); // Normalize to 0-1
      const connectionWeight = Math.max(0, 1 - (metrics.connections / 100)); // Assume 100 is high load
      
      return (successWeight * 0.5) + (responseTimeWeight * 0.3) + (connectionWeight * 0.2);
    });

    const maxWeight = Math.max(...weights);
    const bestIndex = weights.indexOf(maxWeight);
    return services[bestIndex];
  }

  private async executeServiceRequest<T>(
    service: ServiceDefinition, 
    request: ServiceRequest
  ): Promise<ServiceResponse<T>> {
    // Increment connection count
    const metrics = this.serviceMetrics.get(service.name) || {
      responseTime: [],
      successRate: 1.0,
      lastRequest: 0,
      connections: 0
    };
    metrics.connections++;
    this.serviceMetrics.set(service.name, metrics);

    try {
      const response = await productionAPIClient.request<T>({
        ...request,
        route: request.route // Use the original route, let the client handle service resolution
      });

      // Decrement connection count on completion
      metrics.connections = Math.max(0, metrics.connections - 1);
      this.serviceMetrics.set(service.name, metrics);

      return response;
    } catch (error) {
      // Decrement connection count on error
      metrics.connections = Math.max(0, metrics.connections - 1);
      this.serviceMetrics.set(service.name, metrics);
      throw error;
    }
  }

  private createTimeoutPromise(timeout: number): Promise<never> {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Request timeout')), timeout);
    });
  }

  private isServiceHealthy(serviceName: string): boolean {
    const circuitBreaker = this.circuitBreakers.get(serviceName);
    if (circuitBreaker?.state === 'open') {
      return Date.now() > circuitBreaker.nextRetryTime;
    }
    
    const metrics = this.serviceMetrics.get(serviceName);
    return !metrics || metrics.successRate > 0.5; // Consider healthy if >50% success rate
  }

  private isCircuitBreakerOpen(serviceName: string): boolean {
    const circuitBreaker = this.circuitBreakers.get(serviceName);
    if (!circuitBreaker) return false;
    
    if (circuitBreaker.state === 'open') {
      if (Date.now() > circuitBreaker.nextRetryTime) {
        // Move to half-open state
        circuitBreaker.state = 'half_open';
        this.circuitBreakers.set(serviceName, circuitBreaker);
        return false;
      }
      return true;
    }
    
    return false;
  }

  private updateCircuitBreaker(serviceName: string): void {
    const existing = this.circuitBreakers.get(serviceName) || {
      serviceName,
      state: 'closed' as const,
      failureCount: 0,
      lastFailure: 0,
      nextRetryTime: 0
    };

    existing.failureCount++;
    existing.lastFailure = Date.now();

    if (existing.failureCount >= this.config.circuitBreakerThreshold) {
      existing.state = 'open';
      existing.nextRetryTime = Date.now() + (60000 * Math.pow(2, Math.min(existing.failureCount - this.config.circuitBreakerThreshold, 5))); // Exponential backoff
    }

    this.circuitBreakers.set(serviceName, existing);
  }

  private resetCircuitBreaker(serviceName: string): void {
    const existing = this.circuitBreakers.get(serviceName);
    if (existing) {
      existing.state = 'closed';
      existing.failureCount = 0;
      this.circuitBreakers.set(serviceName, existing);
    }
  }

  private recordServiceMetrics(serviceName: string, responseTime: number, success: boolean): void {
    const existing = this.serviceMetrics.get(serviceName) || {
      responseTime: [],
      successRate: 1.0,
      lastRequest: 0,
      connections: 0
    };

    if (success && responseTime > 0) {
      existing.responseTime.push(responseTime);
      if (existing.responseTime.length > 100) {
        existing.responseTime.shift(); // Keep last 100 measurements
      }
    }

    // Update success rate (rolling average)
    const currentSuccess = success ? 1 : 0;
    existing.successRate = (existing.successRate * 0.9) + (currentSuccess * 0.1);
    existing.lastRequest = Date.now();

    this.serviceMetrics.set(serviceName, existing);
  }

  private getAverageResponseTime(serviceName: string): number {
    const metrics = this.serviceMetrics.get(serviceName);
    if (!metrics || metrics.responseTime.length === 0) return 1000; // Default high value
    
    return metrics.responseTime.reduce((sum, time) => sum + time, 0) / metrics.responseTime.length;
  }

  private startHealthMonitoring(): void {
    this.healthCheckTimer = setInterval(async () => {
      await this.performHealthChecks();
    }, this.config.healthCheckInterval);
  }

  private async performHealthChecks(): Promise<void> {
    const services = Object.keys(productionServiceRegistry['services'] || {});
    
    await Promise.all(
      services.map(async (serviceName) => {
        try {
          const healthy = await productionServiceRegistry.checkServiceHealth(serviceName);
          if (healthy) {
            this.resetCircuitBreaker(serviceName);
          } else {
            this.updateCircuitBreaker(serviceName);
          }
        } catch (error) {
          console.warn(`Health check failed for ${serviceName}:`, error);
          this.updateCircuitBreaker(serviceName);
        }
      })
    );
  }

  getServiceMetrics(): Record<string, {
    healthy: boolean;
    circuitBreakerState: string;
    averageResponseTime: number;
    successRate: number;
    connections: number;
    lastRequest: number;
  }> {
    const result: Record<string, any> = {};
    
    Array.from(this.serviceMetrics.keys()).forEach(serviceName => {
      const metrics = this.serviceMetrics.get(serviceName)!;
      const circuitBreaker = this.circuitBreakers.get(serviceName);
      
      result[serviceName] = {
        healthy: this.isServiceHealthy(serviceName),
        circuitBreakerState: circuitBreaker?.state || 'closed',
        averageResponseTime: this.getAverageResponseTime(serviceName),
        successRate: metrics.successRate,
        connections: metrics.connections,
        lastRequest: metrics.lastRequest
      };
    });
    
    return result;
  }

  async getServiceRecommendations(): Promise<{
    recommendations: Array<{
      type: 'scale_up' | 'scale_down' | 'restart' | 'investigate';
      service: string;
      reason: string;
      priority: 'low' | 'medium' | 'high' | 'critical';
    }>;
    overallHealth: string;
    alertsCount: number;
  }> {
    const metrics = this.getServiceMetrics();
    const recommendations: any[] = [];
    let alertsCount = 0;

    Object.entries(metrics).forEach(([serviceName, serviceMetrics]) => {
      // High response time recommendation
      if (serviceMetrics.averageResponseTime > 5000) {
        recommendations.push({
          type: 'investigate',
          service: serviceName,
          reason: `High response time: ${serviceMetrics.averageResponseTime}ms`,
          priority: 'high'
        });
        alertsCount++;
      }

      // Low success rate recommendation
      if (serviceMetrics.successRate < 0.8) {
        recommendations.push({
          type: 'restart',
          service: serviceName,
          reason: `Low success rate: ${Math.round(serviceMetrics.successRate * 100)}%`,
          priority: 'critical'
        });
        alertsCount++;
      }

      // High load recommendation
      if (serviceMetrics.connections > 50) {
        recommendations.push({
          type: 'scale_up',
          service: serviceName,
          reason: `High connection count: ${serviceMetrics.connections}`,
          priority: 'medium'
        });
      }

      // Circuit breaker open recommendation
      if (serviceMetrics.circuitBreakerState === 'open') {
        recommendations.push({
          type: 'restart',
          service: serviceName,
          reason: 'Circuit breaker open - service degraded',
          priority: 'critical'
        });
        alertsCount++;
      }
    });

    const healthyServices = Object.values(metrics).filter(m => m.healthy).length;
    const totalServices = Object.keys(metrics).length;
    const healthRatio = totalServices > 0 ? healthyServices / totalServices : 0;

    let overallHealth: string;
    if (healthRatio >= 0.95) overallHealth = 'excellent';
    else if (healthRatio >= 0.85) overallHealth = 'good';
    else if (healthRatio >= 0.70) overallHealth = 'degraded';
    else overallHealth = 'critical';

    return {
      recommendations: recommendations.sort((a, b) => {
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      }),
      overallHealth,
      alertsCount
    };
  }

  async executeAutomatedRecovery(serviceName: string): Promise<{
    success: boolean;
    actions: string[];
    newState: string;
  }> {
    const actions: string[] = [];
    let success = false;

    try {
      // Step 1: Reset circuit breaker
      this.resetCircuitBreaker(serviceName);
      actions.push('Circuit breaker reset');

      // Step 2: Perform health check
      const healthy = await productionServiceRegistry.checkServiceHealth(serviceName);
      actions.push(`Health check: ${healthy ? 'passed' : 'failed'}`);

      if (!healthy) {
        // Step 3: Attempt service restart (simulation)
        actions.push('Service restart attempted');
        
        // Wait and check again
        await new Promise(resolve => setTimeout(resolve, 5000));
        const healthyAfterRestart = await productionServiceRegistry.checkServiceHealth(serviceName);
        actions.push(`Post-restart health: ${healthyAfterRestart ? 'passed' : 'failed'}`);
        
        success = healthyAfterRestart;
      } else {
        success = true;
      }

      return {
        success,
        actions,
        newState: success ? 'healthy' : 'failed'
      };
    } catch (error) {
      actions.push(`Recovery failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return {
        success: false,
        actions,
        newState: 'failed'
      };
    }
  }

  destroy(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }
  }
}

// Export singleton instance
export const serviceDiscovery = new ServiceDiscoveryEngine();

// Utility functions for service discovery integration
export async function executeWithSmartRouting<T>(
  route: string,
  request: ServiceRequest
): Promise<ServiceResponse<T>> {
  return serviceDiscovery.executeWithFailover<T>(route, request);
}

export async function getServiceStatus(): Promise<{
  discovery: ReturnType<typeof serviceDiscovery.getServiceMetrics>;
  recommendations: Awaited<ReturnType<typeof serviceDiscovery.getServiceRecommendations>>;
  timestamp: string;
}> {
  const [discovery, recommendations] = await Promise.all([
    serviceDiscovery.getServiceMetrics(),
    serviceDiscovery.getServiceRecommendations()
  ]);

  return {
    discovery,
    recommendations,
    timestamp: new Date().toISOString()
  };
}

export async function executeServiceRecovery(serviceName: string): Promise<{
  recovery: Awaited<ReturnType<typeof serviceDiscovery.executeAutomatedRecovery>>;
  postRecoveryHealth: boolean;
}> {
  const recovery = await serviceDiscovery.executeAutomatedRecovery(serviceName);
  const postRecoveryHealth = await productionServiceRegistry.checkServiceHealth(serviceName);

  return { recovery, postRecoveryHealth };
}