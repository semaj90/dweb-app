// AI Synthesis MCP Server
// Integrates with Context7 and provides AI synthesis capabilities

const express = require('express');
const cors = require('cors');
const { EventEmitter } = require('events');

const app = express();
const PORT = process.env.AI_SYNTHESIS_PORT || 8200;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Event emitter for real-time updates
const eventBus = new EventEmitter();

// Service connections
const services = {
  context7: 'http://localhost:4000',
  enhancedRAG: 'http://localhost:8094',
  gpuOrchestrator: 'http://localhost:8095',
  ollama: 'http://localhost:11434',
  redis: 'localhost:6379',
  synthesis: 'http://localhost:5173/api/ai-synthesizer'
};

// Health tracking
const health = {
  status: 'initializing',
  uptime: 0,
  services: {},
  capabilities: [],
  requests: {
    total: 0,
    successful: 0,
    failed: 0
  }
};

// Start time
const startTime = Date.now();

// ===== ENDPOINTS =====

// Health check
app.get('/health', (req, res) => {
  health.uptime = (Date.now() - startTime) / 1000;
  res.json(health);
});

// MCP Capabilities
app.get('/capabilities', (req, res) => {
  res.json({
    name: 'ai-synthesis-mcp',
    version: '1.0.0',
    type: 'mcp-server',
    capabilities: [
      {
        id: 'synthesis',
        name: 'AI Synthesis',
        description: 'Comprehensive AI-powered document synthesis',
        methods: ['synthesize', 'stream', 'analyze']
      },
      {
        id: 'caching',
        name: 'Intelligent Caching',
        description: 'Multi-tier caching with Redis and LRU',
        methods: ['get', 'set', 'invalidate', 'warm']
      },
      {
        id: 'monitoring',
        name: 'Performance Monitoring',
        description: 'Real-time metrics and alerts',
        methods: ['metrics', 'alerts', 'logs']
      },
      {
        id: 'feedback',
        name: 'Feedback Loop',
        description: 'Machine learning from user interactions',
        methods: ['record', 'process', 'recommend']
      },
      {
        id: 'ollama',
        name: 'Local LLM',
        description: 'Ollama integration for local inference',
        methods: ['generate', 'embed', 'models']
      }
    ],
    endpoints: {
      synthesis: '/api/synthesize',
      stream: '/api/stream',
      cache: '/api/cache',
      feedback: '/api/feedback',
      monitor: '/api/monitor',
      ollama: '/api/ollama'
    }
  });
});

// Main synthesis endpoint
app.post('/api/synthesize', async (req, res) => {
  health.requests.total++;
  
  try {
    const { query, context, options } = req.body;
    
    // Forward to main synthesis API
    const response = await fetch(`${services.synthesis}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        context,
        options: {
          ...options,
          mcpIntegration: true
        }
      })
    });
    
    const result = await response.json();
    
    if (response.ok) {
      health.requests.successful++;
      
      // Emit event for monitoring
      eventBus.emit('synthesis:complete', {
        requestId: result.metadata?.requestId,
        processingTime: result.metadata?.processingTime,
        confidence: result.metadata?.confidence
      });
      
      res.json({
        success: true,
        ...result
      });
    } else {
      throw new Error(result.error || 'Synthesis failed');
    }
    
  } catch (error) {
    health.requests.failed++;
    console.error('[MCP] Synthesis error:', error);
    
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ===== SERVICE HEALTH CHECKS =====

async function checkServiceHealth() {
  const checks = [
    { name: 'context7', url: `${services.context7}/health` },
    { name: 'enhancedRAG', url: `${services.enhancedRAG}/health` },
    { name: 'gpuOrchestrator', url: `${services.gpuOrchestrator}/health` },
    { name: 'ollama', url: `${services.ollama}/api/tags` },
    { name: 'synthesis', url: `${services.synthesis}/health` }
  ];
  
  for (const check of checks) {
    try {
      const response = await fetch(check.url, { 
        method: 'GET',
        signal: AbortSignal.timeout(2000)
      });
      
      health.services[check.name] = {
        status: response.ok ? 'healthy' : 'unhealthy',
        lastCheck: new Date()
      };
    } catch (error) {
      health.services[check.name] = {
        status: 'offline',
        lastCheck: new Date(),
        error: error.message
      };
    }
  }
  
  // Update overall health status
  const healthyCount = Object.values(health.services)
    .filter(s => s.status === 'healthy').length;
  
  health.status = healthyCount >= 3 ? 'healthy' : 
                  healthyCount >= 1 ? 'degraded' : 'unhealthy';
  
  // Update capabilities based on available services
  health.capabilities = [];
  if (health.services.synthesis?.status === 'healthy') {
    health.capabilities.push('synthesis', 'streaming', 'caching', 'monitoring');
  }
  if (health.services.ollama?.status === 'healthy') {
    health.capabilities.push('local-llm');
  }
  if (health.services.context7?.status === 'healthy') {
    health.capabilities.push('autosolve', 'documentation');
  }
}

// ===== SERVER STARTUP =====

app.listen(PORT, () => {
  console.log(`🚀 AI Synthesis MCP Server running on port ${PORT}`);
  console.log(`📍 Health: http://localhost:${PORT}/health`);
  console.log(`📍 Capabilities: http://localhost:${PORT}/capabilities`);
  
  // Start health monitoring
  setInterval(checkServiceHealth, 10000); // Check every 10 seconds
  checkServiceHealth(); // Initial check
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n👋 AI Synthesis MCP Server shutting down...');
  process.exit(0);
});

// Export for testing
module.exports = { app, eventBus, health };
