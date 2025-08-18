#!/usr/bin/env node

/**
 * MCP Context7 Wrapper Server
 * Properly formatted MCP server for Claude Code integration
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  McpError,
  ErrorCode
} from '@modelcontextprotocol/sdk/types.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class MCPContext7Server {
  constructor() {
    this.projectRoot = process.env.PROJECT_ROOT || path.join(__dirname, '..');
    this.ollamaEndpoint = process.env.OLLAMA_ENDPOINT || 'http://localhost:11434';
    this.databaseUrl = process.env.DATABASE_URL;
    
    this.server = new Server(
      {
        name: 'context7-mcp-server',
        version: '1.0.0',
        description: 'Context7 MCP Server for Legal AI Development',
      },
      {
        capabilities: {
          tools: true,
          resources: true,
        },
      }
    );

    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'analyze_codebase',
          description: 'Analyze SvelteKit codebase structure and patterns',
          inputSchema: {
            type: 'object',
            properties: {
              path: { type: 'string', description: 'Path to analyze' },
              depth: { type: 'number', description: 'Analysis depth' }
            },
            required: ['path']
          }
        },
        {
          name: 'check_services',
          description: 'Check status of running services (Ollama, PostgreSQL, Redis)',
          inputSchema: {
            type: 'object',
            properties: {},
            required: []
          }
        },
        {
          name: 'generate_recommendations',
          description: 'Generate development recommendations based on error patterns',
          inputSchema: {
            type: 'object',
            properties: {
              errorLog: { type: 'string', description: 'Error log content' },
              context: { type: 'string', description: 'Additional context' }
            },
            required: ['errorLog']
          }
        },
        {
          name: 'get_context7_status',
          description: 'Get comprehensive Context7 system status',
          inputSchema: {
            type: 'object',
            properties: {},
            required: []
          }
        },
        {
          name: 'parse_json_simd',
          description: 'Parse JSON using SIMD acceleration for high-performance processing',
          inputSchema: {
            type: 'object',
            properties: {
              data: { type: 'string', description: 'JSON string to parse' },
              parser: { 
                type: 'string', 
                enum: ['simd', 'fastjson', 'gjson', 'standard'],
                description: 'Parser type to use'
              },
              schema: { type: 'string', description: 'Optional JSON schema for validation' }
            },
            required: ['data']
          }
        },
        {
          name: 'parse_tensor',
          description: 'Parse and process tensor data with operations like reshape, transpose',
          inputSchema: {
            type: 'object',
            properties: {
              shape: { type: 'array', items: { type: 'integer' } },
              data: { type: 'array', items: { type: 'number' } },
              dtype: { type: 'string', enum: ['float32', 'float64', 'int32', 'int64'] },
              op: { 
                type: 'string', 
                enum: ['create', 'reshape', 'transpose', 'multiply', 'add'],
                description: 'Tensor operation to perform'
              },
              metadata: { type: 'object', description: 'Additional metadata for operations' }
            },
            required: ['shape', 'data']
          }
        },
        {
          name: 'generate_llama_response',
          description: 'Generate text using LLAMA models via Go-LLAMA integration',
          inputSchema: {
            type: 'object',
            properties: {
              prompt: { type: 'string', description: 'Text prompt for generation' },
              model: { type: 'string', description: 'LLAMA model to use (default: gemma2:2b)' },
              max_tokens: { type: 'integer', description: 'Maximum tokens to generate' },
              options: { type: 'object', description: 'Additional model options' }
            },
            required: ['prompt']
          }
        },
        {
          name: 'get_multicore_performance',
          description: 'Analyze performance metrics across multicore Context7 workers',
          inputSchema: {
            type: 'object',
            properties: {
              include_workers: { type: 'boolean', description: 'Include individual worker stats' },
              include_load_balancer: { type: 'boolean', description: 'Include load balancer metrics' }
            }
          }
        }
      ]
    }));

    // List available resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: [
        {
          uri: 'context7://codebase-structure',
          name: 'Codebase Structure',
          description: 'Current project structure and file organization',
          mimeType: 'application/json'
        },
        {
          uri: 'context7://service-status',
          name: 'Service Status',
          description: 'Status of all running services',
          mimeType: 'application/json'
        },
        {
          uri: 'context7://error-patterns',
          name: 'Error Patterns',
          description: 'Common error patterns and solutions',
          mimeType: 'application/json'
        },
        {
          uri: 'context7://best-practices',
          name: 'Best Practices',
          description: 'Context7 development best practices',
          mimeType: 'text/markdown'
        }
      ]
    }));

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'analyze_codebase':
            return await this.analyzeCodebase(args);
          case 'check_services':
            return await this.checkServices();
          case 'generate_recommendations':
            return await this.generateRecommendations(args);
          case 'get_context7_status':
            return await this.getContext7Status();
          case 'parse_json_simd':
            return await this.parseJSONSIMD(args);
          case 'parse_tensor':
            return await this.parseTensor(args);
          case 'generate_llama_response':
            return await this.generateLLAMAResponse(args);
          case 'get_multicore_performance':
            return await this.getMulticorePerformance(args);
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
        }
      } catch (error) {
        console.error(`Error in tool ${name}:`, error);
        throw new McpError(
          ErrorCode.InternalError,
          `Tool execution failed: ${error.message}`
        );
      }
    });

    // Handle resource reads
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      try {
        switch (uri) {
          case 'context7://codebase-structure':
            return await this.getCodebaseStructure();
          case 'context7://service-status':
            return await this.getServiceStatus();
          case 'context7://error-patterns':
            return await this.getErrorPatterns();
          case 'context7://best-practices':
            return await this.getBestPractices();
          default:
            throw new McpError(ErrorCode.InvalidRequest, `Unknown resource: ${uri}`);
        }
      } catch (error) {
        console.error(`Error reading resource ${uri}:`, error);
        throw new McpError(
          ErrorCode.InternalError,
          `Resource read failed: ${error.message}`
        );
      }
    });
  }

  async analyzeCodebase(args) {
    const { path: analyzePath = '.', depth = 2 } = args;
    const fullPath = path.resolve(this.projectRoot, analyzePath);

    try {
      const structure = await this.scanDirectory(fullPath, depth);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              path: analyzePath,
              structure,
              analysis: {
                totalFiles: this.countFiles(structure),
                fileTypes: this.getFileTypes(structure),
                recommendations: this.generateCodebaseRecommendations(structure)
              }
            }, null, 2)
          }
        ]
      };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to analyze codebase: ${error.message}`);
    }
  }

  async checkServices() {
    const services = {
      ollama: await this.checkOllama(),
      postgresql: await this.checkPostgreSQL(),
      redis: await this.checkRedis(),
      context7_workers: await this.checkContext7Workers(),
      recommendation_service: await this.checkRecommendationService()
    };

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            timestamp: new Date().toISOString(),
            services,
            overall_status: Object.values(services).every(s => s.status === 'healthy') ? 'healthy' : 'degraded'
          }, null, 2)
        }
      ]
    };
  }

  async generateRecommendations(args) {
    const { errorLog, context = '' } = args;
    
    // Parse common error patterns
    const recommendations = [];
    
    if (errorLog.includes('TS2322')) {
      recommendations.push({
        error: 'Type assignment error',
        solution: 'Check type definitions and ensure proper type casting',
        confidence: 0.9
      });
    }
    
    if (errorLog.includes('TS2304')) {
      recommendations.push({
        error: 'Cannot find name',
        solution: 'Add proper imports or type declarations',
        confidence: 0.85
      });
    }
    
    if (errorLog.includes('svelte-check')) {
      recommendations.push({
        error: 'Svelte compilation issues',
        solution: 'Check component syntax and props usage',
        confidence: 0.8
      });
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            errorLog: errorLog.substring(0, 200) + '...',
            context,
            recommendations,
            generated_at: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }

  async getContext7Status() {
    const status = {
      system: 'Context7 Legal AI Platform',
      version: '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      project_root: this.projectRoot,
      services: {
        mcp_server: 'running',
        ollama_endpoint: this.ollamaEndpoint,
        database_url: this.databaseUrl ? 'configured' : 'not configured'
      },
      features: [
        'SvelteKit 5 + TypeScript',
        'PostgreSQL with pgvector',
        'Ollama LLM integration', 
        'Enhanced RAG pipeline',
        'Context7 workers',
        'Error-to-vector recommendations'
      ]
    };

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(status, null, 2)
        }
      ]
    };
  }

  async scanDirectory(dirPath, maxDepth, currentDepth = 0) {
    if (currentDepth >= maxDepth) return null;
    
    try {
      const items = await fs.readdir(dirPath);
      const structure = {};
      
      for (const item of items.slice(0, 20)) { // Limit to prevent huge responses
        if (item.startsWith('.')) continue;
        
        const itemPath = path.join(dirPath, item);
        const stat = await fs.stat(itemPath);
        
        if (stat.isDirectory()) {
          structure[item] = await this.scanDirectory(itemPath, maxDepth, currentDepth + 1);
        } else {
          structure[item] = 'file';
        }
      }
      
      return structure;
    } catch (error) {
      return { error: error.message };
    }
  }

  countFiles(structure) {
    if (!structure) return 0;
    let count = 0;
    for (const [key, value] of Object.entries(structure)) {
      if (value === 'file') {
        count++;
      } else if (typeof value === 'object') {
        count += this.countFiles(value);
      }
    }
    return count;
  }

  getFileTypes(structure) {
    const types = new Set();
    this.extractFileTypes(structure, types);
    return Array.from(types);
  }

  extractFileTypes(structure, types) {
    if (!structure) return;
    for (const [key, value] of Object.entries(structure)) {
      if (value === 'file') {
        const ext = path.extname(key);
        if (ext) types.add(ext);
      } else if (typeof value === 'object') {
        this.extractFileTypes(value, types);
      }
    }
  }

  generateCodebaseRecommendations(structure) {
    const recommendations = [];
    
    if (structure.src && structure.src['app.html'] === 'file') {
      recommendations.push('SvelteKit project detected');
    }
    
    if (structure['package.json'] === 'file') {
      recommendations.push('Node.js project with proper package management');
    }
    
    return recommendations;
  }

  async checkOllama() {
    try {
      const response = await fetch(`${this.ollamaEndpoint}/api/tags`);
      return {
        status: response.ok ? 'healthy' : 'unhealthy',
        endpoint: this.ollamaEndpoint,
        response_code: response.status
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        endpoint: this.ollamaEndpoint,
        error: error.message
      };
    }
  }

  async checkPostgreSQL() {
    try {
      // Simple check - try to connect
      const response = await fetch('http://localhost:8096/status');
      const data = await response.json();
      return {
        status: 'healthy',
        note: 'Checked via recommendation service'
      };
    } catch (error) {
      return {
        status: 'unknown',
        error: error.message
      };
    }
  }

  async checkRedis() {
    return {
      status: 'assumed_healthy',
      note: 'Redis check not implemented'
    };
  }

  async checkContext7Workers() {
    const workers = [];
    for (let port = 4100; port <= 4107; port++) {
      try {
        const response = await fetch(`http://localhost:${port}/health`, { 
          timeout: 1000 
        });
        workers.push({
          port,
          status: response.ok ? 'healthy' : 'unhealthy'
        });
      } catch (error) {
        workers.push({
          port,
          status: 'unhealthy',
          error: error.message
        });
      }
    }
    
    return {
      workers,
      healthy_count: workers.filter(w => w.status === 'healthy').length,
      total_count: workers.length
    };
  }

  async checkRecommendationService() {
    try {
      const response = await fetch('http://localhost:8096/health');
      const data = await response.json();
      return {
        status: 'healthy',
        service: data.service,
        endpoint: 'http://localhost:8096'
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        endpoint: 'http://localhost:8096',
        error: error.message
      };
    }
  }

  async getCodebaseStructure() {
    const structure = await this.scanDirectory(this.projectRoot, 3);
    return {
      contents: [
        {
          type: 'text',
          text: JSON.stringify(structure, null, 2),
          mimeType: 'application/json'
        }
      ]
    };
  }

  async getServiceStatus() {
    const status = await this.checkServices();
    return {
      contents: [
        {
          type: 'text',
          text: status.content[0].text,
          mimeType: 'application/json'
        }
      ]
    };
  }

  async getErrorPatterns() {
    const patterns = {
      typescript_errors: [
        'TS2322: Type assignment mismatch',
        'TS2304: Cannot find name',
        'TS2339: Property does not exist'
      ],
      svelte_errors: [
        'Component compilation errors',
        'Store reactivity issues',
        'Props validation failures'
      ],
      solutions: {
        'TS2322': 'Add type assertions or fix type definitions',
        'TS2304': 'Import missing dependencies or add type declarations',
        'TS2339': 'Check object structure or add optional chaining'
      }
    };

    return {
      contents: [
        {
          type: 'text',
          text: JSON.stringify(patterns, null, 2),
          mimeType: 'application/json'
        }
      ]
    };
  }

  async getBestPractices() {
    const practices = `# Context7 Development Best Practices

## SvelteKit 5 + TypeScript

1. **Component Structure**
   - Use TypeScript interfaces for props
   - Implement proper error boundaries
   - Follow Svelte 5 runes patterns

2. **State Management** 
   - Use stores for global state
   - Implement XState for complex flows
   - Handle async states properly

3. **Database Integration**
   - Use Drizzle ORM for type safety
   - Implement connection pooling
   - Handle migrations properly

4. **AI Integration**
   - Optimize Ollama model loading
   - Implement proper error handling
   - Use streaming for better UX

5. **Testing**
   - Test components with Vitest
   - Mock external dependencies
   - Implement E2E tests`;

    return {
      contents: [
        {
          type: 'text',
          text: practices,
          mimeType: 'text/markdown'
        }
      ]
    };
  }

  // Multicore integration methods
  async parseJSONSIMD(args) {
    const { data, parser = 'simd', schema } = args;
    
    try {
      const response = await fetch('http://localhost:8095/parse/json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data, parser, schema })
      });

      if (!response.ok) {
        throw new Error(`Context7 multicore service responded with ${response.status}`);
      }

      const result = await response.json();
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: result.success,
              parser_used: result.parser,
              data: result.data,
              performance: {
                parse_time_ns: result.parse_time_ns,
                throughput_mbps: result.throughput_mbps,
                memory_used: result.memory_used
              },
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: false,
              error: error.message,
              fallback: 'Context7 multicore service unavailable',
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    }
  }

  async parseTensor(args) {
    const { shape, data, dtype = 'float64', op = 'create', metadata } = args;
    
    try {
      const response = await fetch('http://localhost:8095/parse/tensor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shape, data, dtype, op, metadata })
      });

      if (!response.ok) {
        throw new Error(`Context7 multicore service responded with ${response.status}`);
      }

      const result = await response.json();
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: result.success,
              tensor_info: {
                shape: result.shape,
                dtype: result.dtype,
                operation: result.operation,
                data_length: result.data?.length || 0
              },
              performance: {
                process_time_ns: result.process_time_ns,
                memory_used: result.memory_used
              },
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: false,
              error: error.message,
              fallback: 'Context7 multicore service unavailable',
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    }
  }

  async generateLLAMAResponse(args) {
    const { prompt, model = 'gemma2:2b', max_tokens, options } = args;
    
    try {
      const response = await fetch('http://localhost:8095/llama', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model, max_tokens, options })
      });

      if (!response.ok) {
        throw new Error(`Context7 multicore service responded with ${response.status}`);
      }

      const result = await response.json();
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: result.success,
              response: result.response,
              llama_info: {
                model_used: result.model,
                token_count: result.token_count,
                process_time_ns: result.process_time_ns
              },
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } catch (error) {
      // Fallback to direct Ollama if multicore service is unavailable
      try {
        const ollamaResponse = await fetch(`${this.ollamaEndpoint}/api/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: model,
            prompt: prompt,
            stream: false,
            options: options
          })
        });

        if (ollamaResponse.ok) {
          const ollamaResult = await ollamaResponse.json();
          return {
            content: [
              {
                type: 'text',
                text: JSON.stringify({
                  success: true,
                  response: ollamaResult.response,
                  fallback: 'Direct Ollama integration used',
                  model: model,
                  timestamp: new Date().toISOString()
                }, null, 2)
              }
            ]
          };
        }
      } catch (ollamaError) {
        // Ignore fallback error
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: false,
              error: error.message,
              fallback: 'Both multicore service and Ollama unavailable',
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    }
  }

  async getMulticorePerformance(args) {
    const { include_workers = true, include_load_balancer = true } = args;
    
    const metrics = {
      timestamp: new Date().toISOString(),
      services: {},
      workers: {},
      aggregated: {
        total_healthy: 0,
        total_services: 0,
        average_response_time: 0
      }
    };

    // Check multicore service
    try {
      const multicoreResponse = await fetch('http://localhost:8095/metrics');
      if (multicoreResponse.ok) {
        metrics.services.context7_multicore = await multicoreResponse.json();
        metrics.aggregated.total_services++;
        metrics.aggregated.total_healthy++;
      }
    } catch (error) {
      metrics.services.context7_multicore = { error: error.message };
      metrics.aggregated.total_services++;
    }

    // Check load balancer if requested
    if (include_load_balancer) {
      try {
        const lbResponse = await fetch('http://localhost:8099/status');
        if (lbResponse.ok) {
          metrics.services.load_balancer = await lbResponse.json();
          metrics.aggregated.total_services++;
          metrics.aggregated.total_healthy++;
        }
      } catch (error) {
        metrics.services.load_balancer = { error: error.message };
        metrics.aggregated.total_services++;
      }
    }

    // Check individual workers if requested
    if (include_workers) {
      for (let port = 4100; port <= 4107; port++) {
        const workerId = `worker_${port - 4099}`;
        try {
          const workerResponse = await fetch(`http://localhost:${port}/metrics`, { 
            timeout: 3000 
          });
          if (workerResponse.ok) {
            metrics.workers[workerId] = await workerResponse.json();
            metrics.aggregated.total_services++;
            metrics.aggregated.total_healthy++;
          }
        } catch (error) {
          metrics.workers[workerId] = { error: error.message, port };
          metrics.aggregated.total_services++;
        }
      }
    }

    // Calculate health percentage
    metrics.aggregated.health_percentage = metrics.aggregated.total_services > 0 
      ? (metrics.aggregated.total_healthy / metrics.aggregated.total_services) * 100 
      : 0;

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(metrics, null, 2)
        }
      ]
    };
  }

  async start() {
    console.error('Starting Context7 MCP Server...');
    
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    
    console.error('Context7 MCP Server ready and connected!');
  }
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Start the server
const server = new MCPContext7Server();
server.start().catch((error) => {
  console.error('Failed to start server:', error);
  process.exit(1);
});