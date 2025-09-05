#!/usr/bin/env node

/**
 * Legal AI Service Dependency Manager
 * 
 * Manages service dependencies, startup order, health checks,
 * and automatic recovery for the Legal AI orchestration system.
 */

class ServiceDependencyManager {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.dependencyGraph = new Map();
    this.serviceStates = new Map();
    this.retryAttempts = new Map();
    this.maxRetries = 3;
    this.healthCheckInterval = 30000; // 30 seconds
    this.dependencyCheckInterval = 15000; // 15 seconds
    
    this.buildDependencyGraph();
    this.initializeServiceStates();
  }

  buildDependencyGraph() {
    console.log('[DEPENDENCY-MANAGER] Building service dependency graph...');
    
    const services = this.orchestrator.config.orchestration.services;
    
    // Build dependency relationships
    Object.entries(services).forEach(([serviceId, config]) => {
      const dependencies = config.dependencies || [];
      this.dependencyGraph.set(serviceId, {
        dependencies,
        dependents: [],
        config
      });
    });
    
    // Build reverse dependencies (dependents)
    this.dependencyGraph.forEach((service, serviceId) => {
      service.dependencies.forEach(depId => {
        const dependency = this.dependencyGraph.get(depId);
        if (dependency) {
          dependency.dependents.push(serviceId);
        }
      });
    });
    
    console.log(`[DEPENDENCY-MANAGER] Built dependency graph for ${this.dependencyGraph.size} services`);
    this.logDependencyGraph();
  }

  initializeServiceStates() {
    this.dependencyGraph.forEach((service, serviceId) => {
      this.serviceStates.set(serviceId, {
        status: 'stopped',
        health: 'unknown',
        lastHealthCheck: null,
        startTime: null,
        restartCount: 0,
        errors: []
      });
      this.retryAttempts.set(serviceId, 0);
    });
  }

  logDependencyGraph() {
    console.log('[DEPENDENCY-MANAGER] Service Dependency Graph:');
    this.dependencyGraph.forEach((service, serviceId) => {
      const deps = service.dependencies.length > 0 ? service.dependencies.join(', ') : 'none';
      const dependents = service.dependents.length > 0 ? service.dependents.join(', ') : 'none';
      console.log(`  ${serviceId}:`);
      console.log(`    Dependencies: ${deps}`);
      console.log(`    Dependents: ${dependents}`);
    });
  }

  getStartupOrder() {
    console.log('[DEPENDENCY-MANAGER] Calculating optimal startup order...');
    
    const visited = new Set();
    const visiting = new Set();
    const order = [];
    
    const visit = (serviceId) => {
      if (visiting.has(serviceId)) {
        throw new Error(`Circular dependency detected involving ${serviceId}`);
      }
      if (visited.has(serviceId)) {
        return;
      }
      
      visiting.add(serviceId);
      
      const service = this.dependencyGraph.get(serviceId);
      if (service) {
        service.dependencies.forEach(depId => {
          visit(depId);
        });
      }
      
      visiting.delete(serviceId);
      visited.add(serviceId);
      order.push(serviceId);
    };
    
    // Visit all services
    this.dependencyGraph.forEach((service, serviceId) => {
      if (!visited.has(serviceId)) {
        visit(serviceId);
      }
    });
    
    console.log('[DEPENDENCY-MANAGER] Startup order:', order);
    return order;
  }

  getShutdownOrder() {
    // Shutdown is reverse of startup order
    const startupOrder = this.getStartupOrder();
    return startupOrder.reverse();
  }

  async startServicesInOrder() {
    console.log('[DEPENDENCY-MANAGER] Starting services in dependency order...');
    
    const startupOrder = this.getStartupOrder();
    
    for (const serviceId of startupOrder) {
      if (this.orchestrator.isShuttingDown) break;
      
      try {
        await this.startServiceWithDependencies(serviceId);
      } catch (error) {
        console.error(`[DEPENDENCY-MANAGER] Failed to start ${serviceId}:`, error.message);
        
        // Check if we should continue or abort
        const shouldContinue = await this.handleStartupFailure(serviceId, error);
        if (!shouldContinue) {
          throw new Error(`Critical service ${serviceId} failed to start`);
        }
      }
    }
    
    console.log('[DEPENDENCY-MANAGER] All services started successfully');
  }

  async startServiceWithDependencies(serviceId) {
    const service = this.dependencyGraph.get(serviceId);
    if (!service) {
      throw new Error(`Service ${serviceId} not found in dependency graph`);
    }
    
    // Check if already running
    const state = this.serviceStates.get(serviceId);
    if (state.status === 'running' && state.health === 'healthy') {
      console.log(`[DEPENDENCY-MANAGER] Service ${serviceId} already running`);
      return;
    }
    
    // Verify dependencies are healthy
    await this.verifyDependencies(serviceId);
    
    // Start the service
    console.log(`[DEPENDENCY-MANAGER] Starting service: ${serviceId}`);
    this.updateServiceState(serviceId, { status: 'starting' });
    
    try {
      await this.orchestrator.startService(serviceId);
      this.updateServiceState(serviceId, { 
        status: 'running',
        startTime: Date.now(),
        restartCount: state.restartCount + 1
      });
      
      // Wait for service to be healthy
      await this.waitForServiceHealth(serviceId);
      
      console.log(`[DEPENDENCY-MANAGER] Service ${serviceId} started successfully`);
      this.retryAttempts.set(serviceId, 0); // Reset retry count on success
      
    } catch (error) {
      this.updateServiceState(serviceId, { 
        status: 'failed',
        errors: [...state.errors, { error: error.message, timestamp: Date.now() }]
      });
      throw error;
    }
  }

  async verifyDependencies(serviceId) {
    const service = this.dependencyGraph.get(serviceId);
    const dependencies = service.dependencies;
    
    if (dependencies.length === 0) {
      return; // No dependencies to verify
    }
    
    console.log(`[DEPENDENCY-MANAGER] Verifying dependencies for ${serviceId}: ${dependencies.join(', ')}`);
    
    for (const depId of dependencies) {
      const depState = this.serviceStates.get(depId);
      
      if (!depState || depState.status !== 'running' || depState.health !== 'healthy') {
        console.log(`[DEPENDENCY-MANAGER] Dependency ${depId} not healthy, starting...`);
        await this.startServiceWithDependencies(depId);
      }
    }
    
    console.log(`[DEPENDENCY-MANAGER] All dependencies verified for ${serviceId}`);
  }

  async waitForServiceHealth(serviceId, timeout = 60000) {
    const startTime = Date.now();
    
    console.log(`[DEPENDENCY-MANAGER] Waiting for ${serviceId} to be healthy...`);
    
    while (Date.now() - startTime < timeout) {
      if (this.orchestrator.isShuttingDown) return;
      
      try {
        const isHealthy = await this.orchestrator.healthMonitor.checkService(serviceId);
        if (isHealthy) {
          this.updateServiceState(serviceId, { health: 'healthy', lastHealthCheck: Date.now() });
          console.log(`[DEPENDENCY-MANAGER] Service ${serviceId} is healthy`);
          return;
        }
      } catch (error) {
        // Still waiting for service to be ready
      }
      
      await this.delay(2000);
    }
    
    throw new Error(`Service ${serviceId} failed to become healthy within ${timeout}ms`);
  }

  async handleStartupFailure(serviceId, error) {
    const retryCount = this.retryAttempts.get(serviceId) || 0;
    
    console.log(`[DEPENDENCY-MANAGER] Handling startup failure for ${serviceId} (attempt ${retryCount + 1}/${this.maxRetries})`);
    
    if (retryCount < this.maxRetries) {
      this.retryAttempts.set(serviceId, retryCount + 1);
      
      console.log(`[DEPENDENCY-MANAGER] Retrying ${serviceId} in 5 seconds...`);
      await this.delay(5000);
      
      try {
        await this.startServiceWithDependencies(serviceId);
        return true; // Continue with other services
      } catch (retryError) {
        return await this.handleStartupFailure(serviceId, retryError);
      }
    } else {
      // Max retries exceeded
      const service = this.dependencyGraph.get(serviceId);
      const isCritical = this.isCriticalService(serviceId);
      
      if (isCritical) {
        console.error(`[DEPENDENCY-MANAGER] Critical service ${serviceId} failed to start after ${this.maxRetries} attempts`);
        return false; // Abort startup
      } else {
        console.warn(`[DEPENDENCY-MANAGER] Non-critical service ${serviceId} failed to start, continuing...`);
        return true; // Continue with other services
      }
    }
  }

  isCriticalService(serviceId) {
    const criticalServices = ['nats', 'kratos', 'windows-services'];
    return criticalServices.includes(serviceId);
  }

  async stopServicesInOrder() {
    console.log('[DEPENDENCY-MANAGER] Stopping services in reverse dependency order...');
    
    const shutdownOrder = this.getShutdownOrder();
    
    for (const serviceId of shutdownOrder) {
      try {
        await this.stopServiceWithDependents(serviceId);
      } catch (error) {
        console.error(`[DEPENDENCY-MANAGER] Error stopping ${serviceId}:`, error.message);
      }
    }
    
    console.log('[DEPENDENCY-MANAGER] All services stopped');
  }

  async stopServiceWithDependents(serviceId) {
    const service = this.dependencyGraph.get(serviceId);
    if (!service) return;
    
    // Stop dependents first
    for (const dependentId of service.dependents) {
      const dependentState = this.serviceStates.get(dependentId);
      if (dependentState && dependentState.status === 'running') {
        console.log(`[DEPENDENCY-MANAGER] Stopping dependent ${dependentId} before ${serviceId}`);
        await this.stopServiceWithDependents(dependentId);
      }
    }
    
    // Stop the service itself
    const state = this.serviceStates.get(serviceId);
    if (state && state.status === 'running') {
      console.log(`[DEPENDENCY-MANAGER] Stopping service: ${serviceId}`);
      this.updateServiceState(serviceId, { status: 'stopping' });
      
      try {
        await this.orchestrator.stopService(serviceId);
        this.updateServiceState(serviceId, { status: 'stopped', health: 'unknown' });
        console.log(`[DEPENDENCY-MANAGER] Service ${serviceId} stopped successfully`);
      } catch (error) {
        this.updateServiceState(serviceId, { status: 'failed' });
        throw error;
      }
    }
  }

  startDependencyMonitoring() {
    console.log('[DEPENDENCY-MANAGER] Starting dependency monitoring...');
    
    // Monitor service health
    setInterval(async () => {
      await this.performDependencyHealthCheck();
    }, this.dependencyCheckInterval);
    
    console.log('[DEPENDENCY-MANAGER] Dependency monitoring active');
  }

  async performDependencyHealthCheck() {
    for (const [serviceId, state] of this.serviceStates) {
      if (state.status === 'running') {
        try {
          const isHealthy = await this.orchestrator.healthMonitor.checkService(serviceId);
          this.updateServiceState(serviceId, { 
            health: isHealthy ? 'healthy' : 'unhealthy',
            lastHealthCheck: Date.now()
          });
          
          if (!isHealthy) {
            console.warn(`[DEPENDENCY-MANAGER] Service ${serviceId} is unhealthy`);
            await this.handleUnhealthyService(serviceId);
          }
          
        } catch (error) {
          this.updateServiceState(serviceId, { 
            health: 'error',
            lastHealthCheck: Date.now()
          });
          console.error(`[DEPENDENCY-MANAGER] Health check failed for ${serviceId}:`, error.message);
        }
      }
    }
  }

  async handleUnhealthyService(serviceId) {
    const state = this.serviceStates.get(serviceId);
    const service = this.dependencyGraph.get(serviceId);
    
    console.log(`[DEPENDENCY-MANAGER] Handling unhealthy service: ${serviceId}`);
    
    // Check if dependents are affected
    const affectedDependents = [];
    for (const dependentId of service.dependents) {
      const dependentState = this.serviceStates.get(dependentId);
      if (dependentState && dependentState.status === 'running') {
        affectedDependents.push(dependentId);
      }
    }
    
    if (affectedDependents.length > 0) {
      console.log(`[DEPENDENCY-MANAGER] Services depending on ${serviceId}: ${affectedDependents.join(', ')}`);
    }
    
    // Attempt restart if retry count allows
    const retryCount = this.retryAttempts.get(serviceId) || 0;
    if (retryCount < this.maxRetries) {
      console.log(`[DEPENDENCY-MANAGER] Attempting to restart ${serviceId}...`);
      this.retryAttempts.set(serviceId, retryCount + 1);
      
      try {
        await this.orchestrator.stopService(serviceId);
        await this.delay(3000);
        await this.startServiceWithDependencies(serviceId);
      } catch (error) {
        console.error(`[DEPENDENCY-MANAGER] Failed to restart ${serviceId}:`, error.message);
      }
    }
  }

  updateServiceState(serviceId, updates) {
    const currentState = this.serviceStates.get(serviceId) || {};
    const newState = { ...currentState, ...updates };
    this.serviceStates.set(serviceId, newState);
    
    // Emit state change event
    this.orchestrator.serviceRegistry.updateStatus(serviceId, newState.status);
  }

  getDependencyStatus() {
    const status = {
      services: Object.fromEntries(this.serviceStates),
      dependencyGraph: {},
      healthSummary: {
        total: this.serviceStates.size,
        running: 0,
        healthy: 0,
        unhealthy: 0,
        stopped: 0,
        failed: 0
      }
    };
    
    // Build dependency graph summary
    this.dependencyGraph.forEach((service, serviceId) => {
      status.dependencyGraph[serviceId] = {
        dependencies: service.dependencies,
        dependents: service.dependents
      };
    });
    
    // Calculate health summary
    this.serviceStates.forEach((state) => {
      switch (state.status) {
        case 'running':
          status.healthSummary.running++;
          if (state.health === 'healthy') status.healthSummary.healthy++;
          if (state.health === 'unhealthy') status.healthSummary.unhealthy++;
          break;
        case 'stopped':
          status.healthSummary.stopped++;
          break;
        case 'failed':
          status.healthSummary.failed++;
          break;
      }
    });
    
    return status;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = ServiceDependencyManager;