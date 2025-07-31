/**
 * Health API Routes
 * System health monitoring and diagnostics endpoints
 */

import express from 'express';

const router = express.Router();

export function createHealthRoutes(services) {
  const { health, database, ollama, cache, vector } = services;

  /**
   * GET / - Basic health check
   */
  router.get('/', async (req, res) => {
    try {
      const healthStatus = health.getHealthStatus();

      res.status(healthStatus.overall === 'healthy' ? 200 : 503).json({
        status: healthStatus.overall,
        timestamp: healthStatus.lastCheck,
        components: Object.fromEntries(
          Object.entries(healthStatus.components).map(([name, component]) => [
            name,
            {
              status: component.status,
              responseTime: component.responseTime,
              lastCheck: component.lastCheck,
              warnings: component.warnings || []
            }
          ])
        ),
        uptime: process.uptime(),
        version: process.env.npm_package_version || '1.0.0'
      });

    } catch (error) {
      console.error('Health check failed:', error);
      res.status(503).json({
        status: 'error',
        error: 'Health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /detailed - Detailed health metrics
   */
  router.get('/detailed', async (req, res) => {
    try {
      const metrics = await health.getDetailedMetrics();

      res.json({
        success: true,
        ...metrics,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Detailed health check failed:', error);
      res.status(500).json({
        success: false,
        error: 'Detailed health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /summary - Health summary for dashboard
   */
  router.get('/summary', async (req, res) => {
    try {
      const summary = health.getHealthSummary();

      res.json({
        success: true,
        summary,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Health summary failed:', error);
      res.status(500).json({
        success: false,
        error: 'Health summary failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /history - Health check history
   */
  router.get('/history', async (req, res) => {
    try {
      const { limit = 10 } = req.query;
      const history = health.getHealthHistory(parseInt(limit));

      res.json({
        success: true,
        history,
        count: history.length,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Health history retrieval failed:', error);
      res.status(500).json({
        success: false,
        error: 'Health history retrieval failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * POST /check - Force health check
   */
  router.post('/check', async (req, res) => {
    try {
      const { component } = req.body;

      if (component) {
        // Check specific component
        const result = await health.checkComponent(component);
        
        res.json({
          success: true,
          component,
          result,
          timestamp: new Date().toISOString()
        });
      } else {
        // Full health check
        const result = await health.performFullHealthCheck();
        
        res.json({
          success: true,
          result,
          timestamp: new Date().toISOString()
        });
      }

    } catch (error) {
      console.error('Forced health check failed:', error);
      res.status(500).json({
        success: false,
        error: 'Forced health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /components - List all components and their status
   */
  router.get('/components', async (req, res) => {
    try {
      const healthStatus = health.getHealthStatus();

      const components = Object.entries(healthStatus.components).map(([name, component]) => ({
        name,
        status: component.status,
        responseTime: component.responseTime,
        lastCheck: component.lastCheck,
        warnings: component.warnings || [],
        details: component.details || {},
        error: component.error
      }));

      res.json({
        success: true,
        components,
        overall: healthStatus.overall,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Components listing failed:', error);
      res.status(500).json({
        success: false,
        error: 'Components listing failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /database - Database-specific health information
   */
  router.get('/database', async (req, res) => {
    try {
      const stats = await database.getHealthStats();

      res.json({
        success: true,
        database: {
          status: stats ? 'healthy' : 'unhealthy',
          statistics: stats,
          connection: 'active',
          version: 'PostgreSQL with pgvector'
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Database health check failed:', error);
      res.status(500).json({
        success: false,
        error: 'Database health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /ollama - Ollama service health information
   */
  router.get('/ollama', async (req, res) => {
    try {
      const stats = await ollama.getStats();

      res.json({
        success: true,
        ollama: {
          status: stats.isHealthy ? 'healthy' : 'unhealthy',
          baseUrl: stats.baseUrl,
          models: {
            default: stats.defaultModel,
            embedding: stats.embeddingModel,
            available: stats.availableModels,
            list: stats.models || []
          },
          error: stats.error
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Ollama health check failed:', error);
      res.status(500).json({
        success: false,
        error: 'Ollama health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /cache - Cache service health information
   */
  router.get('/cache', async (req, res) => {
    try {
      const stats = await cache.getStats();

      res.json({
        success: true,
        cache: {
          status: cache.connected ? 'healthy' : 'unhealthy',
          connected: cache.connected,
          statistics: stats ? {
            memory: stats.memory,
            stats: stats.stats
          } : null
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Cache health check failed:', error);
      res.status(500).json({
        success: false,
        error: 'Cache health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /vector - Vector service health information
   */
  router.get('/vector', async (req, res) => {
    try {
      // Test vector service by generating a test embedding
      const testEmbedding = await vector.generateEmbedding('Health check test');
      
      res.json({
        success: true,
        vector: {
          status: 'healthy',
          embeddingDimensions: testEmbedding.length,
          service: 'active'
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Vector service health check failed:', error);
      res.status(500).json({
        success: false,
        error: 'Vector service health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /system - System resource information
   */
  router.get('/system', async (req, res) => {
    try {
      const memoryUsage = process.memoryUsage();
      const cpuUsage = process.cpuUsage();

      // Convert memory to MB
      const memoryMB = {
        rss: Math.round(memoryUsage.rss / 1024 / 1024),
        heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024),
        heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024),
        external: Math.round(memoryUsage.external / 1024 / 1024),
        arrayBuffers: Math.round(memoryUsage.arrayBuffers / 1024 / 1024)
      };

      const uptime = process.uptime();

      res.json({
        success: true,
        system: {
          status: 'healthy',
          memory: memoryMB,
          cpu: {
            user: cpuUsage.user,
            system: cpuUsage.system
          },
          uptime: {
            seconds: Math.round(uptime),
            hours: Math.round(uptime / 3600 * 100) / 100,
            formatted: formatUptime(uptime)
          },
          process: {
            pid: process.pid,
            version: process.version,
            platform: process.platform,
            arch: process.arch
          },
          environment: process.env.NODE_ENV || 'development'
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('System health check failed:', error);
      res.status(500).json({
        success: false,
        error: 'System health check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * GET /metrics - Prometheus-style metrics (basic implementation)
   */
  router.get('/metrics', async (req, res) => {
    try {
      const healthStatus = health.getHealthStatus();
      const memoryUsage = process.memoryUsage();
      const uptime = process.uptime();

      // Simple metrics in Prometheus format
      const metrics = [
        `# HELP rag_system_up System up status`,
        `# TYPE rag_system_up gauge`,
        `rag_system_up{service="enhanced_rag"} ${healthStatus.overall === 'healthy' ? 1 : 0}`,
        '',
        `# HELP rag_memory_usage_bytes Memory usage in bytes`,
        `# TYPE rag_memory_usage_bytes gauge`,
        `rag_memory_usage_bytes{type="rss"} ${memoryUsage.rss}`,
        `rag_memory_usage_bytes{type="heap_total"} ${memoryUsage.heapTotal}`,
        `rag_memory_usage_bytes{type="heap_used"} ${memoryUsage.heapUsed}`,
        '',
        `# HELP rag_uptime_seconds Process uptime in seconds`,
        `# TYPE rag_uptime_seconds counter`,
        `rag_uptime_seconds ${uptime}`,
        '',
        `# HELP rag_component_health Component health status`,
        `# TYPE rag_component_health gauge`,
        ...Object.entries(healthStatus.components).map(([name, component]) => 
          `rag_component_health{component="${name}"} ${component.status === 'healthy' ? 1 : 0}`
        )
      ];

      res.set('Content-Type', 'text/plain');
      res.send(metrics.join('\n'));

    } catch (error) {
      console.error('Metrics generation failed:', error);
      res.status(500).send('# Error generating metrics');
    }
  });

  /**
   * GET /version - Service version information
   */
  router.get('/version', (req, res) => {
    res.json({
      success: true,
      version: {
        service: 'Enhanced RAG Backend',
        version: process.env.npm_package_version || '1.0.0',
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        environment: process.env.NODE_ENV || 'development'
      },
      timestamp: new Date().toISOString()
    });
  });

  return router;
}

// Helper function to format uptime
function formatUptime(seconds) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

  return parts.join(' ');
}