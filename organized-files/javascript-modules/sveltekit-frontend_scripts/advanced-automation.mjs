#!/usr/bin/env node
/**
 * Advanced Automation System
 * Orchestrates service startup, health monitoring, autosolve, and GPU task dispatch
 */

import { spawn, exec } from 'child_process';
import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fetch from 'node-fetch';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

// Configuration
const CONFIG = {
  services: {
    autoStartTimeout: 120000, // 2 minutes
    healthCheckInterval: 30000, // 30 seconds
    autosolveThreshold: 5,
    metricsInterval: 3000 // 3 seconds
  },
  
  endpoints: {
    cluster: 'http://localhost:5173/api/v1/cluster/health',
    autosolve: 'http://localhost:5173/api/context7-autosolve',
    gpu: 'http://localhost:5173/api/v1/gpu/orchestrate',
    metrics: 'http://localhost:5173/api/v1/cluster/metrics'
  },
  
  files: {
    clusterMetrics: join(rootDir, '.vscode', 'cluster-metrics.json'),
    autosolveReport: join(rootDir, '.vscode', 'auto-solve-report.json'),
    automationLog: join(rootDir, 'logs', 'automation.log')
  }
};

class AdvancedAutomationSystem {
  constructor() {
    this.services = new Map();
    this.healthStatus = new Map();
    this.metrics = {
      startTime: new Date(),
      serviceRestarts: 0,
      autosolveRuns: 0,
      gpuTasks: 0,
      errors: []
    };
    this.isRunning = false;
  }

  async start() {
    console.log('🚀 Starting Advanced Automation System...');
    
    try {
      await this.ensureDirectories();
      await this.startAllServices();
      this.startHealthMonitoring();
      this.startMetricsCollection();
      this.startAutosolveLoop();
      
      this.isRunning = true;
      console.log('✅ Advanced Automation System is fully operational');
      
      // Keep the process alive
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());
      
    } catch (error) {
      console.error('❌ Failed to start automation system:', error);
      process.exit(1);
    }
  }

  async ensureDirectories() {
    const dirs = [
      join(rootDir, 'logs'),
      join(rootDir, '.vscode')
    ];
    
    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        // Directory might already exist
      }
    }
  }

  async startAllServices() {
    console.log('⚡ Starting all services with npm run dev:full...');
    
    return new Promise((resolve, reject) => {
      const devProcess = spawn('npm', ['run', 'dev:full'], {
        cwd: rootDir,
        stdio: 'pipe',
        shell: true
      });
      
      let startupOutput = '';
      let servicesStarted = 0;
      const requiredServices = 10; // Adjust based on your service count
      
      devProcess.stdout.on('data', (data) => {
        const output = data.toString();
        startupOutput += output;
        
        // Count service startup messages
        if (output.includes('✅') || output.includes('Running')) {
          servicesStarted++;
        }
        
        // Log important startup messages
        if (output.includes('Error') || output.includes('Failed')) {
          this.logError('Service startup error', output);
        }
      });
      
      devProcess.stderr.on('data', (data) => {
        const error = data.toString();
        this.logError('Service startup stderr', error);
      });
      
      // Wait for services to start or timeout
      const timeout = setTimeout(() => {
        if (servicesStarted >= requiredServices * 0.8) { // 80% success rate acceptable
          console.log(`✅ Started ${servicesStarted}/${requiredServices} services`);
          resolve();
        } else {
          reject(new Error(`Only ${servicesStarted}/${requiredServices} services started`));
        }
      }, CONFIG.services.autoStartTimeout);
      
      // Check if enough services started earlier
      const checkInterval = setInterval(() => {
        if (servicesStarted >= requiredServices) {
          clearTimeout(timeout);
          clearInterval(checkInterval);
          console.log(`✅ All ${servicesStarted} services started successfully`);
          resolve();
        }
      }, 2000);
    });
  }

  startHealthMonitoring() {
    console.log('🔍 Starting health monitoring...');
    
    setInterval(async () => {
      try {
        await this.checkAllServicesHealth();
        await this.restartFailedServices();
      } catch (error) {
        this.logError('Health monitoring error', error);
      }
    }, CONFIG.services.healthCheckInterval);
  }

  async checkAllServicesHealth() {
    try {
      const response = await fetch(CONFIG.endpoints.cluster, {
        timeout: 10000
      });
      
      if (response.ok) {
        const healthData = await response.json();
        this.updateHealthStatus(healthData);
        return healthData;
      } else {
        throw new Error(`Health check failed: ${response.status}`);
      }
    } catch (error) {
      this.logError('Health check failed', error);
      return null;
    }
  }

  updateHealthStatus(healthData) {
    if (healthData.cluster && healthData.cluster.services) {
      for (const [serviceName, isHealthy] of Object.entries(healthData.cluster.services)) {
        const previousStatus = this.healthStatus.get(serviceName);
        this.healthStatus.set(serviceName, isHealthy);
        
        // Log status changes
        if (previousStatus !== undefined && previousStatus !== isHealthy) {
          const status = isHealthy ? 'recovered' : 'failed';
          console.log(`🔄 Service ${serviceName} ${status}`);
          this.logActivity(`Service ${serviceName} ${status}`);
        }
      }
    }
  }

  async restartFailedServices() {
    const failedServices = [];
    
    for (const [serviceName, isHealthy] of this.healthStatus.entries()) {
      if (!isHealthy) {
        failedServices.push(serviceName);
      }
    }
    
    if (failedServices.length > 0) {
      console.log(`🔧 Attempting to restart ${failedServices.length} failed services...`);
      
      for (const serviceName of failedServices) {
        try {
          await this.restartService(serviceName);
          this.metrics.serviceRestarts++;
        } catch (error) {
          this.logError(`Failed to restart ${serviceName}`, error);
        }
      }
    }
  }

  async restartService(serviceName) {
    // Trigger service restart via cluster API
    try {
      const response = await fetch(CONFIG.endpoints.cluster, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'restart_service',
          services: [serviceName]
        })
      });
      
      if (response.ok) {
        console.log(`✅ Service ${serviceName} restart triggered`);
      } else {
        throw new Error(`Restart failed: ${response.status}`);
      }
    } catch (error) {
      this.logError(`Service restart failed for ${serviceName}`, error);
    }
  }

  startMetricsCollection() {
    console.log('📊 Starting metrics collection...');
    
    setInterval(async () => {
      try {
        await this.collectAndSaveMetrics();
      } catch (error) {
        this.logError('Metrics collection error', error);
      }
    }, CONFIG.services.metricsInterval);
  }

  async collectAndSaveMetrics() {
    try {
      // Collect cluster metrics
      const metricsResponse = await fetch(CONFIG.endpoints.metrics, {
        timeout: 5000
      });
      
      if (metricsResponse.ok) {
        const metricsData = await metricsResponse.json();
        
        // Add automation system metrics
        const enhancedMetrics = {
          ...metricsData,
          automation: {
            uptime: Date.now() - this.metrics.startTime.getTime(),
            serviceRestarts: this.metrics.serviceRestarts,
            autosolveRuns: this.metrics.autosolveRuns,
            gpuTasks: this.metrics.gpuTasks,
            lastUpdate: new Date().toISOString()
          }
        };
        
        // Save to file
        await fs.writeFile(
          CONFIG.files.clusterMetrics,
          JSON.stringify(enhancedMetrics, null, 2)
        );
      }
    } catch (error) {
      this.logError('Metrics collection failed', error);
    }
  }

  startAutosolveLoop() {
    console.log('🤖 Starting autosolve maintenance loop...');
    
    // Run autosolve every 5 minutes
    setInterval(async () => {
      try {
        await this.runAutosolve();
      } catch (error) {
        this.logError('Autosolve error', error);
      }
    }, 300000); // 5 minutes
    
    // Also run immediately
    setTimeout(() => this.runAutosolve(), 10000); // Wait 10 seconds for services to stabilize
  }

  async runAutosolve() {
    console.log('🔧 Running autosolve maintenance cycle...');
    
    try {
      // Check if autosolve should run
      const shouldRun = await this.shouldRunAutosolve();
      
      if (!shouldRun) {
        console.log('⏭️ Autosolve skipped (threshold not met)');
        return;
      }
      
      // Trigger autosolve via GPU orchestrator
      const response = await fetch(CONFIG.endpoints.gpu, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'autosolve',
          data: {
            threshold: CONFIG.services.autosolveThreshold,
            includeClusterMetrics: true,
            forceRun: false
          }
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        this.metrics.autosolveRuns++;
        
        console.log('✅ Autosolve completed successfully');
        
        // Save autosolve report
        await this.saveAutosolveReport(result);
        
        // Process autosolve recommendations
        await this.processAutosolveRecommendations(result);
        
      } else {
        throw new Error(`Autosolve failed: ${response.status}`);
      }
      
    } catch (error) {
      this.logError('Autosolve execution failed', error);
    }
  }

  async shouldRunAutosolve() {
    try {
      // Check TypeScript errors
      const tsCheckResult = await this.runCommand('npm run check:ultra-fast');
      const errorCount = this.parseErrorCount(tsCheckResult);
      
      return errorCount >= CONFIG.services.autosolveThreshold;
    } catch (error) {
      // If we can't check errors, run autosolve as a safety measure
      return true;
    }
  }

  parseErrorCount(output) {
    const errorMatch = output.match(/(\d+)\s+error/i);
    return errorMatch ? parseInt(errorMatch[1]) : 0;
  }

  async saveAutosolveReport(result) {
    const report = {
      timestamp: new Date().toISOString(),
      success: result.success,
      autosolve: result.autosolve,
      cluster: result.cluster,
      automation: {
        triggeredBy: 'automation-system',
        runNumber: this.metrics.autosolveRuns
      }
    };
    
    await fs.writeFile(
      CONFIG.files.autosolveReport,
      JSON.stringify(report, null, 2)
    );
  }

  async processAutosolveRecommendations(result) {
    if (result.autosolve && result.autosolve.recommendations) {
      for (const recommendation of result.autosolve.recommendations) {
        console.log(`💡 Autosolve recommendation: ${recommendation}`);
        
        // Implement automatic actions based on recommendations
        if (recommendation.includes('restart')) {
          await this.handleRestartRecommendation(recommendation);
        } else if (recommendation.includes('scale')) {
          await this.handleScaleRecommendation(recommendation);
        }
      }
    }
  }

  async handleRestartRecommendation(recommendation) {
    // Extract service name and restart if safe
    console.log(`🔄 Processing restart recommendation: ${recommendation}`);
    // Implementation depends on specific recommendation format
  }

  async handleScaleRecommendation(recommendation) {
    // Handle scaling recommendations
    console.log(`📈 Processing scale recommendation: ${recommendation}`);
    // Implementation depends on specific recommendation format
  }

  // GPU Task Management
  async dispatchGPUTask(taskType, data, config = {}) {
    try {
      const response = await fetch(CONFIG.endpoints.gpu, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'gpu_task',
          data: {
            taskType,
            taskData: data,
            priority: config.priority || 'medium',
            context: config.context || {}
          },
          config
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        this.metrics.gpuTasks++;
        return result;
      } else {
        throw new Error(`GPU task failed: ${response.status}`);
      }
    } catch (error) {
      this.logError('GPU task dispatch failed', error);
      throw error;
    }
  }

  // Utility methods
  async runCommand(command) {
    return new Promise((resolve, reject) => {
      exec(command, { cwd: rootDir }, (error, stdout, stderr) => {
        if (error) {
          reject(error);
        } else {
          resolve(stdout + stderr);
        }
      });
    });
  }

  logActivity(message) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${message}`);
    this.appendToLog(`[${timestamp}] ACTIVITY: ${message}`);
  }

  logError(context, error) {
    const timestamp = new Date().toISOString();
    const errorMessage = error instanceof Error ? error.message : String(error);
    
    console.error(`[${timestamp}] ERROR: ${context} - ${errorMessage}`);
    this.metrics.errors.push({ timestamp, context, error: errorMessage });
    
    // Keep only last 100 errors
    if (this.metrics.errors.length > 100) {
      this.metrics.errors = this.metrics.errors.slice(-100);
    }
    
    this.appendToLog(`[${timestamp}] ERROR: ${context} - ${errorMessage}`);
  }

  async appendToLog(message) {
    try {
      await fs.appendFile(CONFIG.files.automationLog, message + '\n');
    } catch (error) {
      // Silently fail if logging fails
    }
  }

  async shutdown() {
    console.log('🛑 Shutting down Advanced Automation System...');
    this.isRunning = false;
    
    // Save final metrics
    await this.collectAndSaveMetrics();
    
    // Log shutdown
    this.logActivity('Advanced Automation System shutdown');
    
    process.exit(0);
  }

  // Public API for external control
  async getStatus() {
    return {
      isRunning: this.isRunning,
      metrics: this.metrics,
      healthStatus: Object.fromEntries(this.healthStatus),
      uptime: Date.now() - this.metrics.startTime.getTime()
    };
  }
}

// Command line interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const automation = new AdvancedAutomationSystem();
  
  const command = process.argv[2];
  
  switch (command) {
    case 'start':
      automation.start();
      break;
      
    case 'autosolve':
      automation.runAutosolve().then(() => process.exit(0));
      break;
      
    case 'health':
      automation.checkAllServicesHealth().then(health => {
        console.log(JSON.stringify(health, null, 2));
        process.exit(0);
      });
      break;
      
    case 'status':
      automation.getStatus().then(status => {
        console.log(JSON.stringify(status, null, 2));
        process.exit(0);
      });
      break;
      
    default:
      console.log(`
Advanced Automation System

Usage:
  node scripts/advanced-automation.mjs start     - Start full automation system
  node scripts/advanced-automation.mjs autosolve - Run autosolve cycle
  node scripts/advanced-automation.mjs health    - Check service health
  node scripts/advanced-automation.mjs status    - Get system status
`);
      process.exit(1);
  }
}

export { AdvancedAutomationSystem };