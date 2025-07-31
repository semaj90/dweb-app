/**
 * Health Service for Enhanced RAG Backend
 * Monitors system health, performance metrics, and service availability
 */

export class HealthService {
  constructor(services) {
    this.services = services;
    this.healthStatus = {
      overall: 'unknown',
      lastCheck: null,
      components: {}
    };
    
    // Health check intervals (in milliseconds)
    this.intervals = {
      database: 30000,    // 30 seconds
      ollama: 60000,      // 1 minute
      cache: 45000,       // 45 seconds
      vector: 120000      // 2 minutes
    };

    this.healthHistory = [];
    this.maxHistoryLength = 100;
  }

  /**
   * Initialize health monitoring
   */
  async initialize() {
    console.log('💊 Initializing health monitoring...');
    
    // Start periodic health checks
    this.startPeriodicHealthChecks();
    
    // Initial health check
    await this.performFullHealthCheck();
    
    console.log('✅ Health monitoring initialized');
    return true;
  }

  /**
   * Start periodic health checks for all services
   */
  startPeriodicHealthChecks() {
    // Database health check
    setInterval(async () => {
      await this.checkDatabaseHealth();
    }, this.intervals.database);

    // Ollama health check
    setInterval(async () => {
      await this.checkOllamaHealth();
    }, this.intervals.ollama);

    // Cache health check
    setInterval(async () => {
      await this.checkCacheHealth();
    }, this.intervals.cache);

    // Vector service health check
    setInterval(async () => {
      await this.checkVectorHealth();
    }, this.intervals.vector);

    console.log('⏰ Periodic health checks started');
  }

  /**
   * Perform comprehensive health check
   */
  async performFullHealthCheck() {
    const startTime = Date.now();
    const checks = [];

    try {
      // Run all health checks in parallel
      checks.push(this.checkDatabaseHealth());
      checks.push(this.checkOllamaHealth());
      checks.push(this.checkCacheHealth());
      checks.push(this.checkVectorHealth());
      checks.push(this.checkSystemResources());

      await Promise.allSettled(checks);

      // Calculate overall health
      this.calculateOverallHealth();

      // Record health check in history
      this.recordHealthCheck(Date.now() - startTime);

      this.healthStatus.lastCheck = new Date().toISOString();

      return this.healthStatus;
    } catch (error) {
      console.error('Full health check failed:', error);
      this.healthStatus.overall = 'unhealthy';
      this.healthStatus.error = error.message;
      return this.healthStatus;
    }
  }

  /**
   * Check database health
   */
  async checkDatabaseHealth() {
    const component = 'database';
    const startTime = Date.now();

    try {
      if (!this.services.database) {
        throw new Error('Database service not available');
      }

      // Test database connection with a simple query
      const stats = await this.services.database.getHealthStats();
      const responseTime = Date.now() - startTime;

      this.healthStatus.components[component] = {
        status: 'healthy',
        responseTime,
        details: {
          totalDocuments: stats.totalDocuments,
          indexedDocuments: stats.indexedDocuments,
          totalChunks: stats.totalChunks,
          queriesLast24h: stats.queriesLast24h,
          avgProcessingTime: stats.avgProcessingTime,
          pendingJobs: stats.pendingJobs,
          runningJobs: stats.runningJobs
        },
        lastCheck: new Date().toISOString()
      };

      // Check for concerning metrics
      const warnings = [];
      if (stats.pendingJobs > 100) warnings.push('High number of pending jobs');
      if (stats.runningJobs > 10) warnings.push('Many jobs currently running');
      if (stats.avgProcessingTime > 5000) warnings.push('High average processing time');
      
      if (warnings.length > 0) {
        this.healthStatus.components[component].warnings = warnings;
      }

    } catch (error) {
      console.error('Database health check failed:', error);
      this.healthStatus.components[component] = {
        status: 'unhealthy',
        error: error.message,
        lastCheck: new Date().toISOString()
      };
    }
  }

  /**
   * Check Ollama service health
   */
  async checkOllamaHealth() {
    const component = 'ollama';
    const startTime = Date.now();

    try {
      if (!this.services.ollama) {
        throw new Error('Ollama service not available');
      }

      // Test Ollama connection and get stats
      const isHealthy = await this.services.ollama.healthCheck();
      const stats = await this.services.ollama.getStats();
      const responseTime = Date.now() - startTime;

      if (!isHealthy) {
        throw new Error('Ollama service is not responding');
      }

      this.healthStatus.components[component] = {
        status: 'healthy',
        responseTime,
        details: {
          baseUrl: stats.baseUrl,
          defaultModel: stats.defaultModel,
          embeddingModel: stats.embeddingModel,
          availableModels: stats.availableModels,
          models: stats.models
        },
        lastCheck: new Date().toISOString()
      };

      // Check for warnings
      const warnings = [];
      if (stats.availableModels === 0) warnings.push('No models available');
      if (responseTime > 5000) warnings.push('Slow response time');
      
      if (warnings.length > 0) {
        this.healthStatus.components[component].warnings = warnings;
      }

    } catch (error) {
      console.error('Ollama health check failed:', error);
      this.healthStatus.components[component] = {
        status: 'unhealthy',
        error: error.message,
        lastCheck: new Date().toISOString()
      };
    }
  }

  /**
   * Check cache service health
   */
  async checkCacheHealth() {
    const component = 'cache';
    const startTime = Date.now();

    try {
      if (!this.services.cache) {
        throw new Error('Cache service not available');
      }

      // Test cache with a simple operation
      const testKey = 'health_check_' + Date.now();
      const testValue = { timestamp: Date.now() };
      
      await this.services.cache.set(testKey, testValue, 60);
      const retrieved = await this.services.cache.get(testKey);
      await this.services.cache.delete(testKey);

      if (!retrieved || retrieved.timestamp !== testValue.timestamp) {
        throw new Error('Cache read/write test failed');
      }

      const stats = await this.services.cache.getStats();
      const responseTime = Date.now() - startTime;

      this.healthStatus.components[component] = {
        status: 'healthy',
        responseTime,
        details: {
          connected: stats?.connected || this.services.cache.connected,
          memory: stats?.memory,
          stats: stats?.stats
        },
        lastCheck: new Date().toISOString()
      };

    } catch (error) {
      console.error('Cache health check failed:', error);
      this.healthStatus.components[component] = {
        status: 'degraded', // Cache failures are not critical
        error: error.message,
        lastCheck: new Date().toISOString()
      };
    }
  }

  /**
   * Check vector service health
   */
  async checkVectorHealth() {
    const component = 'vector';
    const startTime = Date.now();

    try {
      if (!this.services.vector) {
        throw new Error('Vector service not available');
      }

      // Test vector operations
      const testText = 'Health check test document';
      const embedding = await this.services.vector.generateEmbedding(testText);
      
      if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
        throw new Error('Vector embedding generation failed');
      }

      const responseTime = Date.now() - startTime;

      this.healthStatus.components[component] = {
        status: 'healthy',
        responseTime,
        details: {
          embeddingDimensions: embedding.length,
          embeddingModel: this.services.ollama?.config?.embeddingModel || 'unknown'
        },
        lastCheck: new Date().toISOString()
      };

      // Check for warnings
      const warnings = [];
      if (responseTime > 10000) warnings.push('Slow embedding generation');
      if (embedding.length !== 384) warnings.push('Unexpected embedding dimensions');
      
      if (warnings.length > 0) {
        this.healthStatus.components[component].warnings = warnings;
      }

    } catch (error) {
      console.error('Vector health check failed:', error);
      this.healthStatus.components[component] = {
        status: 'unhealthy',
        error: error.message,
        lastCheck: new Date().toISOString()
      };
    }
  }

  /**
   * Check system resources
   */
  async checkSystemResources() {
    const component = 'system';
    
    try {
      const memoryUsage = process.memoryUsage();
      const cpuUsage = process.cpuUsage();
      const uptime = process.uptime();

      // Convert memory to MB
      const memoryMB = {
        rss: Math.round(memoryUsage.rss / 1024 / 1024),
        heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024),
        heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024),
        external: Math.round(memoryUsage.external / 1024 / 1024),
        arrayBuffers: Math.round(memoryUsage.arrayBuffers / 1024 / 1024)
      };

      this.healthStatus.components[component] = {
        status: 'healthy',
        details: {
          memory: memoryMB,
          uptime: {
            seconds: Math.round(uptime),
            hours: Math.round(uptime / 3600 * 100) / 100
          },
          nodeVersion: process.version,
          platform: process.platform,
          arch: process.arch
        },
        lastCheck: new Date().toISOString()
      };

      // Check for resource warnings
      const warnings = [];
      if (memoryMB.heapUsed > 1000) warnings.push('High memory usage');
      if (memoryMB.rss > 2000) warnings.push('High RSS memory usage');
      
      if (warnings.length > 0) {
        this.healthStatus.components[component].warnings = warnings;
        this.healthStatus.components[component].status = 'degraded';
      }

    } catch (error) {
      console.error('System resource check failed:', error);
      this.healthStatus.components[component] = {
        status: 'unknown',
        error: error.message,
        lastCheck: new Date().toISOString()
      };
    }
  }

  /**
   * Calculate overall system health
   */
  calculateOverallHealth() {
    const components = Object.values(this.healthStatus.components);
    
    if (components.length === 0) {
      this.healthStatus.overall = 'unknown';
      return;
    }

    const unhealthyCount = components.filter(c => c.status === 'unhealthy').length;
    const degradedCount = components.filter(c => c.status === 'degraded').length;
    const healthyCount = components.filter(c => c.status === 'healthy').length;

    if (unhealthyCount > 0) {
      // Critical services down
      const criticalUnhealthy = ['database', 'ollama', 'vector'].some(critical => 
        this.healthStatus.components[critical]?.status === 'unhealthy'
      );
      
      this.healthStatus.overall = criticalUnhealthy ? 'critical' : 'unhealthy';
    } else if (degradedCount > 0) {
      this.healthStatus.overall = 'degraded';
    } else if (healthyCount === components.length) {
      this.healthStatus.overall = 'healthy';
    } else {
      this.healthStatus.overall = 'unknown';
    }
  }

  /**
   * Record health check in history
   */
  recordHealthCheck(duration) {
    const record = {
      timestamp: new Date().toISOString(),
      duration,
      overall: this.healthStatus.overall,
      componentCount: Object.keys(this.healthStatus.components).length,
      unhealthyComponents: Object.entries(this.healthStatus.components)
        .filter(([_, component]) => component.status === 'unhealthy')
        .map(([name, _]) => name)
    };

    this.healthHistory.unshift(record);
    
    // Keep history size manageable
    if (this.healthHistory.length > this.maxHistoryLength) {
      this.healthHistory = this.healthHistory.slice(0, this.maxHistoryLength);
    }
  }

  /**
   * Get current health status
   */
  getHealthStatus() {
    return {
      ...this.healthStatus,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get health history
   */
  getHealthHistory(limit = 10) {
    return this.healthHistory.slice(0, limit);
  }

  /**
   * Get detailed system metrics
   */
  async getDetailedMetrics() {
    try {
      await this.performFullHealthCheck();
      
      return {
        health: this.healthStatus,
        history: this.getHealthHistory(20),
        trends: this.calculateHealthTrends(),
        recommendations: this.generateHealthRecommendations()
      };
    } catch (error) {
      console.error('Failed to get detailed metrics:', error);
      return {
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Calculate health trends from history
   */
  calculateHealthTrends() {
    if (this.healthHistory.length < 2) {
      return { trend: 'insufficient_data' };
    }

    const recent = this.healthHistory.slice(0, 10);
    const healthyCount = recent.filter(r => r.overall === 'healthy').length;
    const unhealthyCount = recent.filter(r => r.overall === 'unhealthy' || r.overall === 'critical').length;
    
    const healthPercentage = (healthyCount / recent.length) * 100;
    const avgDuration = recent.reduce((sum, r) => sum + r.duration, 0) / recent.length;

    return {
      healthPercentage: Math.round(healthPercentage),
      avgCheckDuration: Math.round(avgDuration),
      trend: healthPercentage > 80 ? 'improving' : 
             healthPercentage > 50 ? 'stable' : 'declining',
      recentChecks: recent.length
    };
  }

  /**
   * Generate health recommendations
   */
  generateHealthRecommendations() {
    const recommendations = [];
    const components = this.healthStatus.components;

    // Database recommendations
    if (components.database?.details?.pendingJobs > 50) {
      recommendations.push({
        component: 'database',
        priority: 'medium',
        message: 'High number of pending jobs - consider scaling processing capacity'
      });
    }

    // Ollama recommendations
    if (components.ollama?.details?.availableModels === 0) {
      recommendations.push({
        component: 'ollama',
        priority: 'high',
        message: 'No models available - pull required models using ollama pull'
      });
    }

    // Cache recommendations
    if (components.cache?.status === 'degraded') {
      recommendations.push({
        component: 'cache',
        priority: 'low',
        message: 'Cache service degraded - system will work but performance may be reduced'
      });
    }

    // System recommendations
    if (components.system?.details?.memory?.heapUsed > 1000) {
      recommendations.push({
        component: 'system',
        priority: 'medium',
        message: 'High memory usage detected - monitor for memory leaks'
      });
    }

    // Vector service recommendations
    if (components.vector?.responseTime > 10000) {
      recommendations.push({
        component: 'vector',
        priority: 'medium',
        message: 'Slow embedding generation - check Ollama service performance'
      });
    }

    return recommendations;
  }

  /**
   * Force health check for specific component
   */
  async checkComponent(componentName) {
    switch (componentName) {
      case 'database':
        await this.checkDatabaseHealth();
        break;
      case 'ollama':
        await this.checkOllamaHealth();
        break;
      case 'cache':
        await this.checkCacheHealth();
        break;
      case 'vector':
        await this.checkVectorHealth();
        break;
      case 'system':
        await this.checkSystemResources();
        break;
      default:
        throw new Error(`Unknown component: ${componentName}`);
    }

    return this.healthStatus.components[componentName];
  }

  /**
   * Get health summary for dashboard
   */
  getHealthSummary() {
    const components = Object.entries(this.healthStatus.components);
    
    return {
      overall: this.healthStatus.overall,
      lastCheck: this.healthStatus.lastCheck,
      totalComponents: components.length,
      healthyComponents: components.filter(([_, c]) => c.status === 'healthy').length,
      degradedComponents: components.filter(([_, c]) => c.status === 'degraded').length,
      unhealthyComponents: components.filter(([_, c]) => c.status === 'unhealthy').length,
      criticalIssues: this.healthStatus.overall === 'critical',
      uptime: this.healthStatus.components.system?.details?.uptime?.hours || 0
    };
  }
}