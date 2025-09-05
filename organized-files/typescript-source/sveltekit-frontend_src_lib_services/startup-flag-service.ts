/**
 * Startup Flag Service
 * Essential for automation - provides ready flags and service health monitoring
 */

import { writeFile, readFile, mkdir, stat } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';
import { uploadTelemetry } from './upload-telemetry-service';
import { gpuEmbeddingService } from './gpu-semantic-embedding-service';

export interface ServiceStatus {
  name: string;
  port?: number;
  status: 'starting' | 'ready' | 'failed' | 'unknown';
  health: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
  lastCheck: number;
  startupTime?: number;
  retryCount: number;
  isOptional: boolean;
  dependsOn: string[];
  endpoint?: string;
}

export interface StartupSummary {
  timestamp: number;
  sessionId: string;
  environment: string;
  totalServices: number;
  readyServices: number;
  failedServices: number;
  optionalServices: number;
  startupDuration: number;
  services: Record<string, ServiceStatus>;
  flags: {
    readyFlag: boolean;
    allCriticalReady: boolean;
    allOptionalAttempted: boolean;
  };
}

export interface StartupDiff {
  timestamp: number;
  previousSummary?: StartupSummary;
  currentSummary: StartupSummary;
  changes: {
    newServices: string[];
    removedServices: string[];
    statusChanges: Array<{
      service: string;
      from: string;
      to: string;
    }>;
    performanceChanges: Array<{
      service: string;
      startupTimeChange: number;
      retryCountChange: number;
    }>;
  };
  regressions: Array<{
    service: string;
    issue: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
  }>;
}

class StartupFlagService {
  private logsDir = join(process.cwd(), 'logs');
  private readyFlagPath = join(this.logsDir, 'ready.flag');
  private summaryPath = join(this.logsDir, 'startup-summary.json');
  private previousSummaryPath = join(this.logsDir, 'startup-summary-previous.json');
  private diffPath = join(this.logsDir, 'startup-diff.json');
  
  private sessionId: string;
  private startTime: number;
  private services: Map<string, ServiceStatus> = new Map();
  private isShuttingDown = false;

  // Service definitions with dependencies and retry policies
  private serviceDefinitions: Array<Omit<ServiceStatus, 'status' | 'health' | 'lastCheck' | 'startupTime' | 'retryCount'>> = [
    // Critical Infrastructure
    {
      name: 'postgres',
      port: 5432,
      isOptional: false,
      dependsOn: [],
      endpoint: 'postgresql://localhost:5432'
    },
    {
      name: 'redis',
      port: 6379,
      isOptional: false,
      dependsOn: [],
      endpoint: 'redis://localhost:6379'
    },
    
    // AI Services
    {
      name: 'ollama-primary',
      port: 11434,
      isOptional: false,
      dependsOn: [],
      endpoint: 'http://localhost:11434'
    },
    {
      name: 'gpu-embedding-service',
      isOptional: false,
      dependsOn: ['ollama-primary'],
      endpoint: 'internal'
    },
    
    // Go Microservices
    {
      name: 'enhanced-rag',
      port: 8094,
      isOptional: false,
      dependsOn: ['postgres', 'ollama-primary'],
      endpoint: 'http://localhost:8094'
    },
    {
      name: 'upload-service',
      port: 8093,
      isOptional: true,
      dependsOn: ['postgres', 'redis'],
      endpoint: 'http://localhost:8093'
    },
    
    // Optional Services
    {
      name: 'neo4j',
      port: 7474,
      isOptional: true,
      dependsOn: [],
      endpoint: 'http://localhost:7474'
    },
    {
      name: 'qdrant',
      port: 6333,
      isOptional: true,
      dependsOn: [],
      endpoint: 'http://localhost:6333'
    },
    {
      name: 'minio',
      port: 9000,
      isOptional: true,
      dependsOn: [],
      endpoint: 'http://localhost:9000'
    },
    
    // Frontend
    {
      name: 'sveltekit',
      port: 5173,
      isOptional: false,
      dependsOn: ['enhanced-rag', 'gpu-embedding-service'],
      endpoint: 'http://localhost:5173'
    }
  ];

  constructor() {
    this.sessionId = `startup-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`;
    this.startTime = Date.now();
    this.initializeServices();
    this.ensureLogsDirectory();
  }

  /**
   * Initialize service tracking
   */
  private initializeServices(): void {
    for (const serviceDef of this.serviceDefinitions) {
      this.services.set(serviceDef.name, {
        ...serviceDef,
        status: 'starting',
        health: 'unknown',
        lastCheck: 0,
        retryCount: 0
      });
    }
  }

  /**
   * Ensure logs directory exists
   */
  private async ensureLogsDirectory(): Promise<void> {
    try {
      if (!existsSync(this.logsDir)) {
        await mkdir(this.logsDir, { recursive: true });
      }
    } catch (error) {
      console.warn('Failed to create logs directory:', error);
    }
  }

  /**
   * Start monitoring all services with retry logic
   */
  async startMonitoring(): Promise<void> {
    console.log(`🚀 Starting service monitoring (Session: ${this.sessionId})`);
    
    uploadTelemetry.customEvent('startup_monitoring_start', {
      sessionId: this.sessionId,
      totalServices: this.serviceDefinitions.length,
      criticalServices: this.serviceDefinitions.filter(s => !s.isOptional).length,
      optionalServices: this.serviceDefinitions.filter(s => s.isOptional).length
    });

    // Start health checking loop
    this.startHealthCheckLoop();
    
    // Wait for critical services to be ready
    await this.waitForCriticalServices();
    
    // Attempt optional services with retry logic
    await this.attemptOptionalServices();
    
    // Generate startup summary and diff
    await this.generateStartupSummary();
    await this.generateStartupDiff();
    
    // Set ready flag
    await this.setReadyFlag();
  }

  /**
   * Health check loop for all services
   */
  private startHealthCheckLoop(): void {
    const checkInterval = setInterval(async () => {
      if (this.isShuttingDown) {
        clearInterval(checkInterval);
        return;
      }

      for (const [name, service] of this.services.entries()) {
        await this.checkServiceHealth(name, service);
      }
    }, 5000); // Check every 5 seconds
  }

  /**
   * Check health of individual service
   */
  private async checkServiceHealth(name: string, service: ServiceStatus): Promise<void> {
    const startCheck = Date.now();
    
    try {
      let isHealthy = false;
      let responseTime = 0;

      if (service.endpoint === 'internal') {
        // Internal service checks
        if (name === 'gpu-embedding-service') {
          const status = await gpuEmbeddingService.getStatus();
          isHealthy = status.initialized && status.ollamaConnected;
        }
      } else if (service.endpoint && service.endpoint.startsWith('http')) {
        // HTTP health checks
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        try {
          const response = await fetch(`${service.endpoint}/api/health`, {
            method: 'GET',
            signal: controller.signal
          });
          responseTime = Date.now() - startCheck;
          isHealthy = response.ok;
          clearTimeout(timeoutId);
        } catch (fetchError) {
          // Try alternative health endpoints
          try {
            const response = await fetch(`${service.endpoint}`, {
              method: 'HEAD',
              signal: controller.signal
            });
            responseTime = Date.now() - startCheck;
            isHealthy = response.ok;
          } catch {
            isHealthy = false;
          }
          clearTimeout(timeoutId);
        }
      } else if (service.port) {
        // TCP port checks
        isHealthy = await this.checkTCPPort(service.port);
        responseTime = Date.now() - startCheck;
      }

      // Update service status
      const previousStatus = service.status;
      service.status = isHealthy ? 'ready' : (service.status === 'starting' ? 'starting' : 'failed');
      service.health = this.calculateHealthGrade(responseTime, isHealthy);
      service.lastCheck = Date.now();

      // Track startup time
      if (previousStatus === 'starting' && service.status === 'ready' && !service.startupTime) {
        service.startupTime = Date.now() - this.startTime;
        console.log(`✅ ${name} ready in ${service.startupTime}ms`);
      }

      // Update service in map
      this.services.set(name, service);

    } catch (error) {
      console.warn(`Health check failed for ${name}:`, error);
      service.status = 'failed';
      service.health = 'critical';
      service.lastCheck = Date.now();
      this.services.set(name, service);
    }
  }

  /**
   * Check if TCP port is accessible
   */
  private async checkTCPPort(port: number): Promise<boolean> {
    try {
      const response = await fetch(`http://localhost:${port}`, {
        method: 'HEAD',
        signal: AbortSignal.timeout(3000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Calculate health grade based on response time and availability
   */
  private calculateHealthGrade(responseTime: number, isHealthy: boolean): ServiceStatus['health'] {
    if (!isHealthy) return 'critical';
    if (responseTime < 50) return 'excellent';
    if (responseTime < 200) return 'good';
    if (responseTime < 1000) return 'fair';
    return 'poor';
  }

  /**
   * Wait for all critical services to be ready
   */
  private async waitForCriticalServices(): Promise<void> {
    const criticalServices = Array.from(this.services.entries())
      .filter(([_, service]) => !service.isOptional);
    
    const timeout = 120000; // 2 minute timeout
    const startWait = Date.now();
    
    while (Date.now() - startWait < timeout) {
      const readyCritical = criticalServices.filter(([_, service]) => service.status === 'ready');
      
      if (readyCritical.length === criticalServices.length) {
        console.log(`✅ All ${criticalServices.length} critical services ready`);
        return;
      }
      
      const remaining = criticalServices
        .filter(([_, service]) => service.status !== 'ready')
        .map(([name]) => name);
      
      console.log(`⏳ Waiting for critical services: ${remaining.join(', ')}`);
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    throw new Error('Timeout waiting for critical services');
  }

  /**
   * Attempt optional services with retry logic
   */
  private async attemptOptionalServices(): Promise<void> {
    const optionalServices = Array.from(this.services.entries())
      .filter(([_, service]) => service.isOptional);
    
    console.log(`🔄 Attempting ${optionalServices.length} optional services with retry logic`);
    
    for (const [name, service] of optionalServices) {
      await this.retryOptionalService(name, service);
    }
  }

  /**
   * Retry logic for optional services
   */
  private async retryOptionalService(name: string, service: ServiceStatus): Promise<void> {
    const maxRetries = 3;
    const baseDelay = 5000; // 5 seconds
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      service.retryCount = attempt;
      this.services.set(name, service);
      
      await this.checkServiceHealth(name, service);
      const updatedService = this.services.get(name)!;
      
      if (updatedService.status === 'ready') {
        console.log(`✅ ${name} ready after ${attempt + 1} attempts`);
        return;
      }
      
      if (attempt < maxRetries - 1) {
        const delay = baseDelay * Math.pow(2, attempt); // Exponential backoff
        console.log(`⏳ ${name} failed, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    console.log(`⚠️ ${name} failed after ${maxRetries} attempts (optional service)`);
    service.status = 'failed';
    service.retryCount = maxRetries;
    this.services.set(name, service);
  }

  /**
   * Generate comprehensive startup summary
   */
  private async generateStartupSummary(): Promise<StartupSummary> {
    const services: Record<string, ServiceStatus> = {};
    for (const [name, service] of this.services.entries()) {
      services[name] = { ...service };
    }
    
    const readyServices = Array.from(this.services.values()).filter(s => s.status === 'ready').length;
    const failedServices = Array.from(this.services.values()).filter(s => s.status === 'failed').length;
    const optionalServices = Array.from(this.services.values()).filter(s => s.isOptional).length;
    const criticalReady = Array.from(this.services.values())
      .filter(s => !s.isOptional && s.status === 'ready').length;
    const totalCritical = Array.from(this.services.values()).filter(s => !s.isOptional).length;
    
    const summary: StartupSummary = {
      timestamp: Date.now(),
      sessionId: this.sessionId,
      environment: process.env.NODE_ENV || 'development',
      totalServices: this.services.size,
      readyServices,
      failedServices,
      optionalServices,
      startupDuration: Date.now() - this.startTime,
      services,
      flags: {
        readyFlag: criticalReady === totalCritical,
        allCriticalReady: criticalReady === totalCritical,
        allOptionalAttempted: true
      }
    };
    
    // Save summary
    try {
      await writeFile(this.summaryPath, JSON.stringify(summary, null, 2));
      console.log(`📊 Startup summary saved: ${this.summaryPath}`);
    } catch (error) {
      console.warn('Failed to save startup summary:', error);
    }
    
    return summary;
  }

  /**
   * Generate startup diff comparing with previous run
   */
  private async generateStartupDiff(): Promise<void> {
    try {
      // Load current summary
      const currentSummary = JSON.parse(await readFile(this.summaryPath, 'utf-8')) as StartupSummary;
      
      // Load previous summary if exists
      let previousSummary: StartupSummary | undefined;
      if (existsSync(this.previousSummaryPath)) {
        try {
          previousSummary = JSON.parse(await readFile(this.previousSummaryPath, 'utf-8')) as StartupSummary;
        } catch (error) {
          console.warn('Failed to load previous summary:', error);
        }
      }
      
      // Generate diff
      const diff: StartupDiff = {
        timestamp: Date.now(),
        previousSummary,
        currentSummary,
        changes: {
          newServices: [],
          removedServices: [],
          statusChanges: [],
          performanceChanges: []
        },
        regressions: []
      };
      
      if (previousSummary) {
        // Detect changes
        const currentServices = new Set(Object.keys(currentSummary.services));
        const previousServices = new Set(Object.keys(previousSummary.services));
        
        diff.changes.newServices = Array.from(currentServices).filter(s => !previousServices.has(s));
        diff.changes.removedServices = Array.from(previousServices).filter(s => !currentServices.has(s));
        
        // Status and performance changes
        for (const serviceName of currentServices) {
          const current = currentSummary.services[serviceName];
          const previous = previousSummary.services[serviceName];
          
          if (previous) {
            // Status changes
            if (current.status !== previous.status) {
              diff.changes.statusChanges.push({
                service: serviceName,
                from: previous.status,
                to: current.status
              });
            }
            
            // Performance changes
            if (current.startupTime && previous.startupTime) {
              const timeChange = current.startupTime - previous.startupTime;
              const retryChange = current.retryCount - previous.retryCount;
              
              if (Math.abs(timeChange) > 1000 || retryChange !== 0) {
                diff.changes.performanceChanges.push({
                  service: serviceName,
                  startupTimeChange: timeChange,
                  retryCountChange: retryChange
                });
              }
              
              // Detect regressions
              if (timeChange > 5000) { // >5s slower
                diff.regressions.push({
                  service: serviceName,
                  issue: `Startup time increased by ${timeChange}ms`,
                  severity: timeChange > 30000 ? 'critical' : 'high'
                });
              }
              
              if (retryChange > 0) {
                diff.regressions.push({
                  service: serviceName,
                  issue: `Retry count increased by ${retryChange}`,
                  severity: current.isOptional ? 'medium' : 'high'
                });
              }
            }
            
            // Health regressions
            if (current.status === 'failed' && previous.status === 'ready') {
              diff.regressions.push({
                service: serviceName,
                issue: 'Service that was previously ready is now failed',
                severity: current.isOptional ? 'medium' : 'critical'
              });
            }
          }
        }
        
        // Overall performance regression
        if (currentSummary.startupDuration > previousSummary.startupDuration + 10000) {
          diff.regressions.push({
            service: 'overall',
            issue: `Total startup time increased by ${currentSummary.startupDuration - previousSummary.startupDuration}ms`,
            severity: 'medium'
          });
        }
      }
      
      // Save diff
      await writeFile(this.diffPath, JSON.stringify(diff, null, 2));
      
      // Move current summary to previous for next run
      await writeFile(this.previousSummaryPath, JSON.stringify(currentSummary, null, 2));
      
      // Log important changes
      if (diff.changes.statusChanges.length > 0) {
        console.log('📊 Service status changes:', diff.changes.statusChanges);
      }
      
      if (diff.regressions.length > 0) {
        console.warn('⚠️ Performance regressions detected:', diff.regressions);
      }
      
      console.log(`📊 Startup diff saved: ${this.diffPath}`);
      
    } catch (error) {
      console.warn('Failed to generate startup diff:', error);
    }
  }

  /**
   * Set ready flag file for automation
   */
  private async setReadyFlag(): Promise<void> {
    const criticalServices = Array.from(this.services.values()).filter(s => !s.isOptional);
    const readyCritical = criticalServices.filter(s => s.status === 'ready');
    
    if (readyCritical.length === criticalServices.length) {
      try {
        const flagContent = JSON.stringify({
          timestamp: Date.now(),
          sessionId: this.sessionId,
          readyServices: readyCritical.map(s => s.name),
          startupDuration: Date.now() - this.startTime,
          environment: process.env.NODE_ENV || 'development'
        }, null, 2);
        
        await writeFile(this.readyFlagPath, flagContent);
        console.log(`🚩 Ready flag set: ${this.readyFlagPath}`);
        
        uploadTelemetry.customEvent('startup_ready_flag_set', {
          sessionId: this.sessionId,
          startupDuration: Date.now() - this.startTime,
          readyServices: readyCritical.length,
          totalServices: this.services.size
        });
        
      } catch (error) {
        console.warn('Failed to set ready flag:', error);
      }
    } else {
      console.warn(`❌ Ready flag not set - ${readyCritical.length}/${criticalServices.length} critical services ready`);
    }
  }

  /**
   * Check if environment is ready (for automation)
   */
  async isReady(): Promise<boolean> {
    try {
      if (!existsSync(this.readyFlagPath)) {
        return false;
      }
      
      const flagContent = JSON.parse(await readFile(this.readyFlagPath, 'utf-8'));
      const flagAge = Date.now() - flagContent.timestamp;
      
      // Flag expires after 30 minutes
      return flagAge < 30 * 60 * 1000;
    } catch {
      return false;
    }
  }

  /**
   * Get current service status summary
   */
  getServiceSummary(): StartupSummary {
    const services: Record<string, ServiceStatus> = {};
    for (const [name, service] of this.services.entries()) {
      services[name] = { ...service };
    }
    
    const readyServices = Array.from(this.services.values()).filter(s => s.status === 'ready').length;
    const failedServices = Array.from(this.services.values()).filter(s => s.status === 'failed').length;
    const optionalServices = Array.from(this.services.values()).filter(s => s.isOptional).length;
    
    return {
      timestamp: Date.now(),
      sessionId: this.sessionId,
      environment: process.env.NODE_ENV || 'development',
      totalServices: this.services.size,
      readyServices,
      failedServices,
      optionalServices,
      startupDuration: Date.now() - this.startTime,
      services,
      flags: {
        readyFlag: existsSync(this.readyFlagPath),
        allCriticalReady: Array.from(this.services.values())
          .filter(s => !s.isOptional)
          .every(s => s.status === 'ready'),
        allOptionalAttempted: true
      }
    };
  }

  /**
   * Shutdown monitoring and cleanup
   */
  async shutdown(): Promise<void> {
    this.isShuttingDown = true;
    
    // Remove ready flag
    try {
      if (existsSync(this.readyFlagPath)) {
        const { unlink } = await import('fs/promises');
        await unlink(this.readyFlagPath);
        console.log('🚩 Ready flag removed');
      }
    } catch (error) {
      console.warn('Failed to remove ready flag:', error);
    }
    
    uploadTelemetry.customEvent('startup_monitoring_shutdown', {
      sessionId: this.sessionId,
      totalUptime: Date.now() - this.startTime
    });
  }
}

// Export singleton instance
export const startupFlagService = new StartupFlagService();

// Cleanup on process exit
if (typeof process !== 'undefined') {
  process.on('SIGINT', () => startupFlagService.shutdown());
  process.on('SIGTERM', () => startupFlagService.shutdown());
  process.on('beforeExit', () => startupFlagService.shutdown());
}