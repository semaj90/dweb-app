#!/usr/bin/env node

/**
 * Legal AI Fully Wired Orchestration System
 * 
 * This is the complete, production-ready orchestration system that integrates
 * all components into a unified, enterprise-grade platform.
 * 
 * Components Integrated:
 * - Service Discovery & Registration
 * - Inter-Service Communication (NATS)
 * - Health Monitoring & Alerting
 * - Dependency Management
 * - Configuration Management
 * - Comprehensive Logging (ELK Stack)
 * - Message Routing & Coordination
 * - Real-time Monitoring Dashboard
 */

const LegalAIOrchestrationController = require('./orchestration-controller');
const LegalAIMessageRouter = require('./message-routing-config');
const ServiceDependencyManager = require('./service-dependency-manager');
const ConfigurationManager = require('./configuration-manager');
const LegalAILoggingIntegration = require('./logging-integration');

class WiredOrchestrationSystem {
  constructor() {
    this.isInitialized = false;
    this.isRunning = false;
    this.startTime = null;
    
    // Core components
    this.orchestrationController = null;
    this.messageRouter = null;
    this.dependencyManager = null;
    this.configurationManager = null;
    this.loggingIntegration = null;
    
    // System state
    this.systemHealth = {
      status: 'initializing',
      components: {},
      lastUpdate: null
    };
    
    this.bindProcessHandlers();
  }

  async initialize() {
    console.log('================================================================================');
    console.log('                 Legal AI Enterprise Orchestration System');
    console.log('                           Fully Wired Integration');
    console.log('================================================================================');
    console.log();
    
    try {
      this.startTime = Date.now();
      
      // Initialize Configuration Manager first
      await this.initializeConfigurationManager();
      
      // Initialize Logging Integration
      await this.initializeLoggingIntegration();
      
      // Initialize Orchestration Controller
      await this.initializeOrchestrationController();
      
      // Initialize Message Router
      await this.initializeMessageRouter();
      
      // Initialize Dependency Manager
      await this.initializeDependencyManager();
      
      // Wire components together
      await this.wireComponents();
      
      // Start system monitoring
      this.startSystemMonitoring();
      
      this.isInitialized = true;
      this.logSystemEvent('SYSTEM_INITIALIZED', 'All components initialized successfully');
      
      console.log('================================================================================');
      console.log('✅ Legal AI Orchestration System fully initialized and ready');
      console.log(`   Initialization time: ${Date.now() - this.startTime}ms`);
      console.log('================================================================================');
      
    } catch (error) {
      this.logSystemError('INITIALIZATION_FAILED', error);
      throw error;
    }
  }

  async initializeConfigurationManager() {
    console.log('[SYSTEM] Initializing Configuration Manager...');
    
    this.configurationManager = new ConfigurationManager();
    
    // Subscribe to configuration changes
    this.configurationManager.subscribe('main', (newConfig, oldConfig) => {
      this.handleConfigurationChange('main', newConfig, oldConfig);
    });
    
    this.updateComponentHealth('configuration-manager', 'healthy');
    console.log('[SYSTEM] ✅ Configuration Manager initialized');
  }

  async initializeLoggingIntegration() {
    console.log('[SYSTEM] Initializing Logging Integration...');
    
    // Create placeholder orchestrator for logging
    const orchestratorPlaceholder = {
      config: this.configurationManager.getConfiguration(),
      messageRouter: null,
      serviceRegistry: { size: () => 0 },
      services: new Map(),
      healthMonitor: { getOverallHealth: () => ({ services: { healthy: 0 } }) }
    };
    
    this.loggingIntegration = new LegalAILoggingIntegration(orchestratorPlaceholder);
    
    this.updateComponentHealth('logging-integration', 'healthy');
    console.log('[SYSTEM] ✅ Logging Integration initialized');
  }

  async initializeOrchestrationController() {
    console.log('[SYSTEM] Initializing Orchestration Controller...');
    
    this.orchestrationController = new LegalAIOrchestrationController();
    
    // Inject logging integration
    this.orchestrationController.loggingIntegration = this.loggingIntegration;
    
    await this.orchestrationController.initialize();
    
    this.updateComponentHealth('orchestration-controller', 'healthy');
    console.log('[SYSTEM] ✅ Orchestration Controller initialized');
  }

  async initializeMessageRouter() {
    console.log('[SYSTEM] Initializing Message Router...');
    
    this.messageRouter = new LegalAIMessageRouter();
    await this.messageRouter.initialize();
    
    this.updateComponentHealth('message-router', 'healthy');
    console.log('[SYSTEM] ✅ Message Router initialized');
  }

  async initializeDependencyManager() {
    console.log('[SYSTEM] Initializing Dependency Manager...');
    
    this.dependencyManager = new ServiceDependencyManager(this.orchestrationController);
    this.dependencyManager.startDependencyMonitoring();
    
    this.updateComponentHealth('dependency-manager', 'healthy');
    console.log('[SYSTEM] ✅ Dependency Manager initialized');
  }

  async wireComponents() {
    console.log('[SYSTEM] Wiring components together...');
    
    // Wire orchestration controller with other components
    this.orchestrationController.messageRouter = this.messageRouter;
    this.orchestrationController.dependencyManager = this.dependencyManager;
    this.orchestrationController.configurationManager = this.configurationManager;
    
    // Wire message router with orchestration controller
    this.messageRouter.orchestrator = this.orchestrationController;
    
    // Wire dependency manager with message router
    this.dependencyManager.messageRouter = this.messageRouter;
    
    // Update logging integration with real orchestrator
    this.loggingIntegration.orchestrator = this.orchestrationController;
    
    // Set up cross-component event handling
    this.setupCrossComponentEvents();
    
    console.log('[SYSTEM] ✅ All components wired together');
  }

  setupCrossComponentEvents() {
    // Configuration change propagation
    this.configurationManager.subscribe('*', (newConfig, oldConfig, serviceName) => {
      this.propagateConfigurationChange(serviceName, newConfig, oldConfig);
    });
    
    // Health status propagation
    if (this.orchestrationController.healthMonitor) {
      // Set up health status forwarding to message router
      setInterval(() => {
        const healthStatus = this.orchestrationController.healthMonitor.getOverallHealth();
        this.broadcastHealthStatus(healthStatus);
      }, 30000);
    }
    
    // Log message forwarding from all components
    this.setupLogMessageForwarding();
  }

  setupLogMessageForwarding() {
    // Intercept console.log from all components and route through logging integration
    const originalConsoleLog = console.log;
    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    
    console.log = (...args) => {
      originalConsoleLog.apply(console, args);
      if (this.loggingIntegration) {
        this.loggingIntegration.log('info', args.join(' '), {}, 'system');
      }
    };
    
    console.error = (...args) => {
      originalConsoleError.apply(console, args);
      if (this.loggingIntegration) {
        this.loggingIntegration.log('error', args.join(' '), {}, 'system');
      }
    };
    
    console.warn = (...args) => {
      originalConsoleWarn.apply(console, args);
      if (this.loggingIntegration) {
        this.loggingIntegration.log('warn', args.join(' '), {}, 'system');
      }
    };
  }

  async startServices() {
    if (!this.isInitialized) {
      throw new Error('System must be initialized before starting services');
    }
    
    console.log('[SYSTEM] Starting all services in dependency order...');
    this.isRunning = true;
    
    try {
      // Use dependency manager to start services
      await this.dependencyManager.startServicesInOrder();
      
      this.logSystemEvent('SERVICES_STARTED', 'All services started successfully');
      console.log('[SYSTEM] ✅ All services started successfully');
      
    } catch (error) {
      this.isRunning = false;
      this.logSystemError('SERVICE_START_FAILED', error);
      throw error;
    }
  }

  async stopServices() {
    if (!this.isRunning) {
      console.log('[SYSTEM] Services are not running');
      return;
    }
    
    console.log('[SYSTEM] Stopping all services...');
    
    try {
      // Use dependency manager to stop services in reverse order
      await this.dependencyManager.stopServicesInOrder();
      
      this.isRunning = false;
      this.logSystemEvent('SERVICES_STOPPED', 'All services stopped successfully');
      console.log('[SYSTEM] ✅ All services stopped successfully');
      
    } catch (error) {
      this.logSystemError('SERVICE_STOP_FAILED', error);
      throw error;
    }
  }

  startSystemMonitoring() {
    console.log('[SYSTEM] Starting system monitoring...');
    
    // System health monitoring
    setInterval(() => {
      this.updateSystemHealth();
    }, 15000); // Every 15 seconds
    
    // Performance monitoring
    setInterval(() => {
      this.logPerformanceMetrics();
    }, 60000); // Every minute
    
    // Resource monitoring
    setInterval(() => {
      this.monitorSystemResources();
    }, 30000); // Every 30 seconds
    
    console.log('[SYSTEM] ✅ System monitoring active');
  }

  updateSystemHealth() {
    const health = {
      status: this.isRunning ? 'running' : 'stopped',
      uptime: Date.now() - this.startTime,
      components: { ...this.systemHealth.components },
      services: this.dependencyManager ? this.dependencyManager.getDependencyStatus() : {},
      resources: {
        memory: process.memoryUsage(),
        cpu: process.cpuUsage()
      },
      lastUpdate: new Date().toISOString()
    };
    
    this.systemHealth = health;
    
    // Broadcast health update
    if (this.messageRouter?.natsConnection) {
      this.messageRouter.publishSystemEvent('health-update', health);
    }
  }

  logPerformanceMetrics() {
    if (!this.loggingIntegration) return;
    
    const metrics = {
      uptime: Date.now() - this.startTime,
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      system: {
        initialized: this.isInitialized,
        running: this.isRunning,
        components_healthy: Object.values(this.systemHealth.components).filter(c => c === 'healthy').length,
        total_components: Object.keys(this.systemHealth.components).length
      }
    };
    
    this.loggingIntegration.logPerformance('system-metrics', 0, metrics);
  }

  monitorSystemResources() {
    const usage = process.memoryUsage();
    const memoryUsagePercent = (usage.heapUsed / usage.heapTotal) * 100;
    
    // Alert on high memory usage
    if (memoryUsagePercent > 80) {
      this.logSystemEvent('HIGH_MEMORY_USAGE', `Memory usage at ${memoryUsagePercent.toFixed(2)}%`, {
        memory: usage,
        threshold: 80
      });
    }
    
    // Alert on low available memory
    if (usage.heapTotal - usage.heapUsed < 100 * 1024 * 1024) { // Less than 100MB available
      this.logSystemEvent('LOW_MEMORY_WARNING', 'Available memory below 100MB', {
        memory: usage
      });
    }
  }

  // Event handling methods
  handleConfigurationChange(serviceName, newConfig, oldConfig) {
    this.logSystemEvent('CONFIGURATION_CHANGED', `Configuration updated for ${serviceName}`, {
      service: serviceName,
      changes: this.getConfigurationDiff(oldConfig, newConfig)
    });
    
    // Notify affected services
    this.propagateConfigurationChange(serviceName, newConfig, oldConfig);
  }

  propagateConfigurationChange(serviceName, newConfig, oldConfig) {
    if (this.messageRouter?.natsConnection) {
      this.messageRouter.publishSystemEvent('configuration-updated', {
        service: serviceName,
        config: newConfig,
        timestamp: new Date().toISOString()
      });
    }
  }

  broadcastHealthStatus(healthStatus) {
    if (this.messageRouter?.natsConnection) {
      this.messageRouter.publishSystemEvent('health-broadcast', healthStatus);
    }
  }

  getConfigurationDiff(oldConfig, newConfig) {
    // Simple diff implementation
    const changes = {};
    
    const checkDiff = (obj1, obj2, path = '') => {
      Object.keys(obj2 || {}).forEach(key => {
        const currentPath = path ? `${path}.${key}` : key;
        
        if (typeof obj2[key] === 'object' && obj2[key] !== null && !Array.isArray(obj2[key])) {
          checkDiff(obj1?.[key], obj2[key], currentPath);
        } else if (obj1?.[key] !== obj2[key]) {
          changes[currentPath] = { old: obj1?.[key], new: obj2[key] };
        }
      });
    };
    
    checkDiff(oldConfig, newConfig);
    return changes;
  }

  updateComponentHealth(componentName, status) {
    this.systemHealth.components[componentName] = status;
    this.systemHealth.lastUpdate = new Date().toISOString();
  }

  // Logging helper methods
  logSystemEvent(eventType, message, data = {}) {
    if (this.loggingIntegration) {
      this.loggingIntegration.log('info', message, {
        type: 'system-event',
        event: eventType,
        ...data
      }, 'orchestration.system');
    }
  }

  logSystemError(errorType, error, data = {}) {
    if (this.loggingIntegration) {
      this.loggingIntegration.logError(`System error: ${errorType}`, error, {
        type: 'system-error',
        error_type: errorType,
        ...data
      });
    }
  }

  // Public API methods
  getSystemStatus() {
    return {
      ...this.systemHealth,
      initialized: this.isInitialized,
      running: this.isRunning,
      startup_time: this.startTime,
      components: {
        orchestration_controller: this.orchestrationController ? 'initialized' : 'not_initialized',
        message_router: this.messageRouter ? 'initialized' : 'not_initialized',
        dependency_manager: this.dependencyManager ? 'initialized' : 'not_initialized',
        configuration_manager: this.configurationManager ? 'initialized' : 'not_initialized',
        logging_integration: this.loggingIntegration ? 'initialized' : 'not_initialized'
      }
    };
  }

  getDetailedStatus() {
    return {
      system: this.getSystemStatus(),
      services: this.dependencyManager?.getDependencyStatus() || {},
      configuration: this.configurationManager?.getConfigurationSummary() || {},
      messaging: this.messageRouter?.getRoutingStats() || {},
      health: this.orchestrationController?.healthMonitor?.getOverallHealth() || {}
    };
  }

  // Process event handlers
  bindProcessHandlers() {
    process.on('SIGINT', () => this.handleShutdown('SIGINT'));
    process.on('SIGTERM', () => this.handleShutdown('SIGTERM'));
    process.on('uncaughtException', (error) => this.handleUncaughtException(error));
    process.on('unhandledRejection', (reason, promise) => this.handleUnhandledRejection(reason, promise));
  }

  async handleShutdown(signal) {
    console.log(`[SYSTEM] Received ${signal}, initiating graceful shutdown...`);
    
    try {
      await this.shutdown();
      console.log('[SYSTEM] Graceful shutdown completed');
      process.exit(0);
    } catch (error) {
      console.error('[SYSTEM] Error during shutdown:', error);
      process.exit(1);
    }
  }

  handleUncaughtException(error) {
    console.error('[SYSTEM] Uncaught Exception:', error);
    this.logSystemError('UNCAUGHT_EXCEPTION', error);
    
    // Attempt graceful shutdown
    this.shutdown().finally(() => {
      process.exit(1);
    });
  }

  handleUnhandledRejection(reason, promise) {
    console.error('[SYSTEM] Unhandled Rejection at:', promise, 'reason:', reason);
    this.logSystemError('UNHANDLED_REJECTION', new Error(reason));
  }

  async shutdown() {
    this.logSystemEvent('SHUTDOWN_INITIATED', 'System shutdown initiated');
    
    try {
      // Stop services first
      if (this.isRunning) {
        await this.stopServices();
      }
      
      // Shutdown components
      if (this.messageRouter) {
        await this.messageRouter.shutdown();
      }
      
      if (this.loggingIntegration) {
        await this.loggingIntegration.shutdown();
      }
      
      this.logSystemEvent('SHUTDOWN_COMPLETED', 'System shutdown completed successfully');
      
    } catch (error) {
      this.logSystemError('SHUTDOWN_ERROR', error);
      throw error;
    }
  }
}

// Export for use as module
module.exports = WiredOrchestrationSystem;

// Main execution
if (require.main === module) {
  const system = new WiredOrchestrationSystem();
  
  async function startSystem() {
    try {
      await system.initialize();
      await system.startServices();
      
      console.log('\n🚀 Legal AI Enterprise Orchestration System is fully operational!');
      console.log('\n📊 Management Dashboard: http://localhost:8000/status');
      console.log('📈 Real-time Metrics: ws://localhost:8000');
      console.log('🔍 Health Monitoring: http://localhost:8000/health');
      console.log('\n✨ All orchestration components are wired and running!');
      
    } catch (error) {
      console.error('\n❌ Failed to start Legal AI Orchestration System:', error);
      process.exit(1);
    }
  }
  
  startSystem();
}