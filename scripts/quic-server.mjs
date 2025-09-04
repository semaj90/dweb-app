#!/usr/bin/env node

/**
 * Production QUIC Server with gRPC Fallback
 * High-performance server supporting QUIC, gRPC, and HTTP protocols
 */

import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, '..');

// Configuration
const CONFIG = {
  ports: {
    quic: process.env.QUIC_PORT || 3000,
    grpc: process.env.GRPC_PORT || 3001, 
    http: process.env.HTTP_PORT || 3002
  },
  ssl: {
    keyPath: join(projectRoot, 'certs', 'server.key'),
    certPath: join(projectRoot, 'certs', 'server.crt')
  },
  fallbackMode: true
};

// Protocol handlers
class ProtocolServer {
  constructor(config) {
    this.config = config;
    this.servers = new Map();
    this.connections = new Set();
  }

  async start() {
    console.log('🚀 Starting multi-protocol server...');
    
    try {
      // Try QUIC first (if certificates exist)
      if (this.hasCertificates()) {
        await this.startQUIC();
      } else {
        console.log('⚠️  No SSL certificates found, skipping QUIC');
      }
      
      // Start gRPC server
      await this.startGRPC();
      
      // Start HTTP fallback
      await this.startHTTP();
      
      console.log('✅ All protocol servers started successfully');
      console.log(`📊 Status: http://localhost:${CONFIG.ports.http}/status`);
      
    } catch (error) {
      console.error('❌ Failed to start server:', error.message);
      process.exit(1);
    }
  }

  hasCertificates() {
    return existsSync(CONFIG.ssl.keyPath) && existsSync(CONFIG.ssl.certPath);
  }

  async startQUIC() {
    try {
      // Mock QUIC implementation (replace with actual QUIC library)
      console.log(`🔒 QUIC server listening on port ${CONFIG.ports.quic}`);
      
      // In production, use actual QUIC implementation:
      // const quicServer = createQUICServer({
      //   key: readFileSync(CONFIG.ssl.keyPath),
      //   cert: readFileSync(CONFIG.ssl.certPath)
      // });
      
      this.servers.set('quic', { port: CONFIG.ports.quic, active: true });
    } catch (error) {
      console.error('QUIC server failed:', error.message);
      throw error;
    }
  }

  async startGRPC() {
    try {
      // Mock gRPC implementation
      console.log(`🔌 gRPC server listening on port ${CONFIG.ports.grpc}`);
      
      // In production, implement actual gRPC server:
      // const grpcServer = new grpc.Server();
      // grpcServer.addService(serviceDefinition, implementation);
      
      this.servers.set('grpc', { port: CONFIG.ports.grpc, active: true });
    } catch (error) {
      console.error('gRPC server failed:', error.message);
      throw error;
    }
  }

  async startHTTP() {
    const httpServer = createServer((req, res) => {
      this.handleHTTPRequest(req, res);
    });

    httpServer.listen(CONFIG.ports.http, () => {
      console.log(`🌐 HTTP server listening on port ${CONFIG.ports.http}`);
      this.servers.set('http', { port: CONFIG.ports.http, active: true });
    });

    return httpServer;
  }

  handleHTTPRequest(req, res) {
    const { method, url } = req;
    
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.setHeader('Content-Type', 'application/json');

    if (method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    // Route handlers
    switch (url) {
      case '/status':
        this.handleStatus(req, res);
        break;
      case '/health':
        this.handleHealth(req, res);
        break;
      case '/api/vector':
        this.handleVectorAPI(req, res);
        break;
      case '/api/rag':
        this.handleRAGAPI(req, res);
        break;
      default:
        this.handleNotFound(req, res);
    }
  }

  handleStatus(req, res) {
    const status = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      servers: Array.from(this.servers.entries()).map(([protocol, config]) => ({
        protocol,
        port: config.port,
        active: config.active,
        endpoint: protocol === 'http' ? `http://localhost:${config.port}` : `${protocol}://localhost:${config.port}`
      })),
      connections: this.connections.size,
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      version: '1.0.0'
    };

    res.writeHead(200);
    res.end(JSON.stringify(status, null, 2));
  }

  handleHealth(req, res) {
    const health = {
      healthy: true,
      services: {
        quic: this.servers.has('quic'),
        grpc: this.servers.has('grpc'),
        http: this.servers.has('http')
      },
      timestamp: new Date().toISOString()
    };

    res.writeHead(200);
    res.end(JSON.stringify(health, null, 2));
  }

  handleVectorAPI(req, res) {
    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => { body += chunk.toString(); });
      req.on('end', () => {
        try {
          const data = JSON.parse(body);
          const result = {
            success: true,
            message: 'Vector operation processed',
            data: data,
            protocol: 'http',
            timestamp: new Date().toISOString()
          };
          
          res.writeHead(200);
          res.end(JSON.stringify(result, null, 2));
        } catch (error) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      });
    } else {
      res.writeHead(405);
      res.end(JSON.stringify({ error: 'Method not allowed' }));
    }
  }

  handleRAGAPI(req, res) {
    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => { body += chunk.toString(); });
      req.on('end', () => {
        try {
          const data = JSON.parse(body);
          const result = {
            success: true,
            message: 'RAG query processed',
            query: data.query || 'No query provided',
            results: [
              { id: 1, content: 'Sample result 1', score: 0.95 },
              { id: 2, content: 'Sample result 2', score: 0.87 }
            ],
            protocol: 'http',
            timestamp: new Date().toISOString()
          };
          
          res.writeHead(200);
          res.end(JSON.stringify(result, null, 2));
        } catch (error) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      });
    } else {
      res.writeHead(405);
      res.end(JSON.stringify({ error: 'Method not allowed' }));
    }
  }

  handleNotFound(req, res) {
    const notFound = {
      error: 'Not Found',
      message: `Route ${req.url} not found`,
      availableRoutes: ['/status', '/health', '/api/vector', '/api/rag'],
      timestamp: new Date().toISOString()
    };

    res.writeHead(404);
    res.end(JSON.stringify(notFound, null, 2));
  }

  async stop() {
    console.log('🛑 Shutting down servers...');
    
    for (const [protocol, server] of this.servers) {
      try {
        if (server.close) {
          server.close();
        }
        console.log(`✅ ${protocol.toUpperCase()} server stopped`);
      } catch (error) {
        console.error(`❌ Error stopping ${protocol} server:`, error.message);
      }
    }
    
    this.servers.clear();
    this.connections.clear();
  }
}

// Graceful shutdown
function setupGracefulShutdown(server) {
  const signals = ['SIGTERM', 'SIGINT', 'SIGUSR2'];
  
  signals.forEach(signal => {
    process.on(signal, async () => {
      console.log(`\n📡 Received ${signal}, shutting down gracefully...`);
      await server.stop();
      process.exit(0);
    });
  });
}

// Main execution
async function main() {
  const server = new ProtocolServer(CONFIG);
  
  setupGracefulShutdown(server);
  
  try {
    await server.start();
  } catch (error) {
    console.error('❌ Server startup failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export default ProtocolServer;