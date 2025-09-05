// Enhanced timing and reporting for Legal AI System
// Provides detailed performance metrics and service coordination

export class SystemReporter {
  constructor() {
    this.phases = new Map();
    this.startTime = Date.now();
    this.metrics = {
      services: new Map(),
      phases: [],
      errors: [],
      warnings: []
    };
  }

  startPhase(name, description) {
    const phase = {
      name,
      description,
      startTime: Date.now(),
      endTime: null,
      duration: null,
      status: 'running'
    };
    
    this.phases.set(name, phase);
    console.log(`🔄 [${name}] ${description}`);
    return phase;
  }

  endPhase(name, status = 'completed') {
    const phase = this.phases.get(name);
    if (!phase) return null;

    phase.endTime = Date.now();
    phase.duration = phase.endTime - phase.startTime;
    phase.status = status;
    
    const durationSec = (phase.duration / 1000).toFixed(2);
    const emoji = status === 'completed' ? '✅' : status === 'failed' ? '❌' : '⚠️';
    
    console.log(`${emoji} [${name}] ${phase.description} - ${durationSec}s`);
    this.metrics.phases.push(phase);
    
    return phase;
  }

  recordService(name, port, status, details = {}) {
    this.metrics.services.set(name, {
      name,
      port,
      status,
      timestamp: new Date().toISOString(),
      ...details
    });
  }

  recordError(error, context = '') {
    const errorRecord = {
      message: error.message || error,
      context,
      timestamp: new Date().toISOString(),
      stack: error.stack
    };
    
    this.metrics.errors.push(errorRecord);
    console.error(`❌ ERROR [${context}]: ${error.message || error}`);
  }

  recordWarning(warning, context = '') {
    const warningRecord = {
      message: warning,
      context,
      timestamp: new Date().toISOString()
    };
    
    this.metrics.warnings.push(warningRecord);
    console.warn(`⚠️ WARNING [${context}]: ${warning}`);
  }

  generateReport() {
    const totalTime = Date.now() - this.startTime;
    const servicesArray = Array.from(this.metrics.services.values());
    const healthyServices = servicesArray.filter(s => s.status === 'healthy');
    
    const report = {
      summary: {
        startTime: new Date(this.startTime).toISOString(),
        totalDuration: (totalTime / 1000).toFixed(2) + 's',
        status: this.metrics.errors.length === 0 ? 'success' : 'partial',
        servicesHealthy: healthyServices.length,
        servicesTotal: servicesArray.length,
        healthScore: Math.round((healthyServices.length / servicesArray.length) * 100)
      },
      phases: this.metrics.phases.map(p => ({
        name: p.name,
        description: p.description,
        duration: (p.duration / 1000).toFixed(2) + 's',
        status: p.status
      })),
      services: servicesArray,
      errors: this.metrics.errors,
      warnings: this.metrics.warnings,
      recommendations: this.generateRecommendations()
    };

    return report;
  }

  generateRecommendations() {
    const recommendations = [];
    
    // Check for failed services
    const failedServices = Array.from(this.metrics.services.values())
      .filter(s => s.status !== 'healthy');
    
    if (failedServices.length > 0) {
      recommendations.push({
        type: 'service_failure',
        message: `${failedServices.length} services are not healthy`,
        services: failedServices.map(s => s.name),
        action: 'Check logs and restart failed services'
      });
    }

    // Check for slow phases
    const slowPhases = this.metrics.phases.filter(p => p.duration > 30000);
    if (slowPhases.length > 0) {
      recommendations.push({
        type: 'performance',
        message: 'Some phases took longer than expected',
        phases: slowPhases.map(p => p.name),
        action: 'Consider optimizing startup sequence or checking system resources'
      });
    }

    // Check for errors
    if (this.metrics.errors.length > 0) {
      recommendations.push({
        type: 'errors',
        message: `${this.metrics.errors.length} errors occurred during startup`,
        action: 'Review error logs and fix underlying issues'
      });
    }

    return recommendations;
  }

  printDetailedSummary() {
    const report = this.generateReport();
    
    console.log('\n========================================');
    console.log('📊 DETAILED STARTUP REPORT');
    console.log('========================================');
    console.log(`🕐 Total Time: ${report.summary.totalDuration}`);
    console.log(`📊 Health Score: ${report.summary.healthScore}%`);
    console.log(`🟢 Services: ${report.summary.servicesHealthy}/${report.summary.servicesTotal}`);
    
    console.log('\n📋 Phase Breakdown:');
    report.phases.forEach(phase => {
      const emoji = phase.status === 'completed' ? '✅' : phase.status === 'failed' ? '❌' : '⚠️';
      console.log(`   ${emoji} ${phase.name}: ${phase.duration}`);
    });

    if (report.errors.length > 0) {
      console.log(`\n❌ Errors (${report.errors.length}):`);
      report.errors.forEach(error => {
        console.log(`   • [${error.context}] ${error.message}`);
      });
    }

    if (report.warnings.length > 0) {
      console.log(`\n⚠️ Warnings (${report.warnings.length}):`);
      report.warnings.forEach(warning => {
        console.log(`   • [${warning.context}] ${warning.message}`);
      });
    }

    if (report.recommendations.length > 0) {
      console.log('\n💡 Recommendations:');
      report.recommendations.forEach(rec => {
        console.log(`   • ${rec.message}`);
        console.log(`     Action: ${rec.action}`);
      });
    }

    console.log('========================================\n');
    
    return report;
  }
}

// Enhanced service coordinator that works with your existing API structure
export class ServiceCoordinator {
  constructor(reporter) {
    this.reporter = reporter;
    this.dependencies = new Map();
    this.setupDependencies();
  }

  setupDependencies() {
    // Define service dependencies based on your API structure
    this.dependencies.set('postgresql', {
      required: true,
      dependents: ['sveltekit', 'enhanced_rag'],
      healthEndpoint: null // TCP check only
    });

    this.dependencies.set('ollama', {
      required: true,
      dependents: ['sveltekit', 'enhanced_rag'],
      healthEndpoint: 'http://localhost:11434/api/tags'
    });

    this.dependencies.set('redis', {
      required: false,
      dependents: ['sveltekit'],
      healthEndpoint: null // TCP check only
    });

    this.dependencies.set('qdrant', {
      required: false,
      dependents: ['enhanced_rag'],
      healthEndpoint: 'http://localhost:6333/collections'
    });

    this.dependencies.set('minio', {
      required: false,
      dependents: ['sveltekit'],
      healthEndpoint: 'http://localhost:9000/minio/health/live'
    });

    this.dependencies.set('enhanced_rag', {
      required: false,
      dependents: [],
      healthEndpoint: 'http://localhost:8094/health'
    });

    this.dependencies.set('gpu_orchestrator', {
      required: false,
      dependents: [],
      healthEndpoint: 'http://localhost:8095/health'
    });

    this.dependencies.set('sveltekit', {
      required: true,
      dependents: [],
      healthEndpoint: 'http://localhost:5173/api/health'
    });
  }

  async validateDependencies() {
    const issues = [];
    
    for (const [service, config] of this.dependencies) {
      const serviceStatus = this.reporter.metrics.services.get(service);
      
      if (config.required && (!serviceStatus || serviceStatus.status !== 'healthy')) {
        issues.push({
          type: 'missing_required',
          service,
          message: `Required service ${service} is not healthy`
        });
      }

      // Check if dependents are affected
      if (serviceStatus && serviceStatus.status !== 'healthy') {
        for (const dependent of config.dependents) {
          const dependentStatus = this.reporter.metrics.services.get(dependent);
          if (dependentStatus && dependentStatus.status === 'healthy') {
            issues.push({
              type: 'dependency_warning',
              service: dependent,
              dependency: service,
              message: `${dependent} depends on unhealthy service ${service}`
            });
          }
        }
      }
    }

    return issues;
  }

  getStartupOrder() {
    // Return services in dependency order
    return [
      'postgresql',
      'redis', 
      'ollama',
      'qdrant',
      'minio',
      'enhanced_rag',
      'gpu_orchestrator',
      'sveltekit'
    ];
  }

  async validateApiEndpoints() {
    const endpoints = [
      'http://localhost:5173/api/health',
      'http://localhost:5173/api/system/status',
      'http://localhost:5173/api/cases',
      'http://localhost:5173/api/evidence',
      'http://localhost:5173/api/chat',
      'http://localhost:5173/api/search'
    ];

    const results = [];
    
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint, { 
          method: 'GET',
          timeout: 5000
        });
        
        results.push({
          endpoint,
          status: response.status,
          ok: response.ok,
          responseTime: Date.now() - Date.now() // Simple timing
        });
        
        if (response.ok) {
          console.log(`✓ API ready: ${endpoint}`);
        } else {
          console.log(`⚠ API issue: ${endpoint} (${response.status})`);
        }
      } catch (error) {
        results.push({
          endpoint,
          status: 'error',
          ok: false,
          error: error.message
        });
        console.log(`✗ API failed: ${endpoint} - ${error.message}`);
      }
    }

    return results;
  }
}

// Integrated timing wrapper for the main orchestrator
export function withTiming(serviceManager) {
  const reporter = new SystemReporter();
  const coordinator = new ServiceCoordinator(reporter);
  
  // Wrap service manager methods with timing
  const originalMethods = {};
  
  const methodsToWrap = [
    'startPostgreSQL',
    'startRedis', 
    'startOllama',
    'startOptionalServices',
    'startGoMicroservices',
    'startSvelteKitApp',
    'runDatabaseMigrations'
  ];

  methodsToWrap.forEach(methodName => {
    originalMethods[methodName] = serviceManager[methodName];
    
    serviceManager[methodName] = async function(...args) {
      const phaseName = methodName.replace('start', '').replace('run', '');
      reporter.startPhase(phaseName, `Starting ${phaseName}`);
      
      try {
        const result = await originalMethods[methodName].apply(this, args);
        reporter.endPhase(phaseName, result ? 'completed' : 'failed');
        return result;
      } catch (error) {
        reporter.recordError(error, phaseName);
        reporter.endPhase(phaseName, 'failed');
        throw error;
      }
    };
  });

  // Add enhanced reporting methods
  serviceManager.getReport = () => reporter.generateReport();
  serviceManager.printDetailedSummary = () => reporter.printDetailedSummary();
  serviceManager.validateDependencies = () => coordinator.validateDependencies();
  serviceManager.validateApiEndpoints = () => coordinator.validateApiEndpoints();
  serviceManager.reporter = reporter;
  serviceManager.coordinator = coordinator;

  return serviceManager;
}
