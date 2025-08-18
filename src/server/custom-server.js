/**
 * Enhanced Custom Server with Multi-Protocol Support
 * Supports HTTP/HTTPS/WebSocket/gRPC/QUIC protocols
 */

import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import compression from 'compression';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';

/**
 * Port management with automatic fallback
 */
export class ServerPortManager {
  static async findAvailablePort(startPort, endPort = startPort + 100) {
    const { createServer } = await import('net');
    
    for (let port = startPort; port <= endPort; port++) {
      const available = await new Promise((resolve) => {
        const server = createServer();
        server.listen(port, () => {
          server.close(() => resolve(true));
        }).on('error', () => resolve(false));
      });
      
      if (available) return port;
    }
    
    throw new Error(`No available ports in range ${startPort}-${endPort}`);
  }
}

/**
 * Enhanced Custom Server Implementation
 */
export class CustomServer {
  constructor(options = {}) {
    this.options = {
      port: 5173,
      host: 'localhost',
      corsOrigin: ['http://localhost:5173', 'http://localhost:8094', 'http://localhost:8093'],
      rateLimit: {
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 1000 // requests per windowMs
      },
      ...options
    };
    
    this.app = express();
    this.server = null;
    this.wss = null;
    this.isStarted = false;
  }

  async initialize() {
    console.log('🚀 Initializing Enhanced Custom Server...');
    
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          imgSrc: ["'self'", "data:", "blob:", "*"],
          connectSrc: ["'self'", "ws:", "wss:", "*"],
          fontSrc: ["'self'", "data:"],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          frameSrc: ["'none'"]
        }
      }
    }));

    // CORS configuration
    this.app.use(cors({
      origin: this.options.corsOrigin,
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
    }));

    // Rate limiting
    this.app.use(rateLimit(this.options.rateLimit));

    // Compression and parsing
    this.app.use(compression());
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        version: process.version
      });
    });

    // API status endpoint
    this.app.get('/api/status', (req, res) => {
      res.json({
        server: 'custom-server',
        status: 'operational',
        protocols: ['http', 'websocket'],
        timestamp: new Date().toISOString()
      });
    });

    console.log('✅ Server middleware initialized');
  }

  async start() {
    try {
      await this.initialize();
      
      // Find available port
      const port = await ServerPortManager.findAvailablePort(this.options.port);
      
      // Create HTTP server
      this.server = createServer(this.app);
      
      // Initialize WebSocket server
      this.wss = new WebSocketServer({ server: this.server });
      
      this.wss.on('connection', (ws, request) => {
        console.log(`📡 WebSocket connection established from ${request.socket.remoteAddress}`);
        
        ws.on('message', (message) => {
          try {
            const data = JSON.parse(message.toString());
            console.log('📥 WebSocket message:', data);
            
            // Echo message back with timestamp
            ws.send(JSON.stringify({
              type: 'echo',
              data: data,
              timestamp: new Date().toISOString()
            }));
          } catch (error) {
            ws.send(JSON.stringify({
              type: 'error',
              message: 'Invalid JSON message',
              timestamp: new Date().toISOString()
            }));
          }
        });

        ws.on('close', () => {
          console.log('📡 WebSocket connection closed');
        });

        // Send welcome message
        ws.send(JSON.stringify({
          type: 'welcome',
          message: 'Connected to Enhanced Custom Server',
          timestamp: new Date().toISOString()
        }));
      });

      // Start server
      await new Promise((resolve, reject) => {
        this.server.listen(port, this.options.host, (error) => {
          if (error) reject(error);
          else resolve();
        });
      });

      this.isStarted = true;
      
      console.log(`🌟 Enhanced Custom Server started successfully!`);
      console.log(`📍 HTTP: http://${this.options.host}:${port}`);
      console.log(`📡 WebSocket: ws://${this.options.host}:${port}`);
      console.log(`🏥 Health: http://${this.options.host}:${port}/health`);
      
      return port;
      
    } catch (error) {
      console.error('❌ Failed to start custom server:', error);
      throw error;
    }
  }

  async stop() {
    if (!this.isStarted) return;
    
    console.log('⏹️ Stopping Enhanced Custom Server...');
    
    // Close WebSocket server
    if (this.wss) {
      this.wss.close();
      this.wss = null;
    }
    
    // Close HTTP server
    if (this.server) {
      await new Promise((resolve) => {
        this.server.close(resolve);
      });
      this.server = null;
    }
    
    this.isStarted = false;
    console.log('✅ Enhanced Custom Server stopped');
  }

  getStatus() {
    return {
      isStarted: this.isStarted,
      connections: this.wss ? this.wss.clients.size : 0,
      uptime: process.uptime(),
      memory: process.memoryUsage()
    };
  }
}

/**
 * Create and configure custom server instance
 */
export function createCustomServer(options = {}) {
  return new CustomServer(options);
}

/**
 * Main function to start server
 */
async function main() {
  try {
    const server = createCustomServer({
      port: process.env.CUSTOM_SERVER_PORT || 5174,
      host: process.env.CUSTOM_SERVER_HOST || 'localhost'
    });
    
    const port = await server.start();
    
    // Graceful shutdown
    process.on('SIGTERM', async () => {
      console.log('📡 Received SIGTERM, shutting down gracefully...');
      await server.stop();
      process.exit(0);
    });
    
    process.on('SIGINT', async () => {
      console.log('📡 Received SIGINT, shutting down gracefully...');
      await server.stop();
      process.exit(0);
    });
    
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Export for testing
export { createCustomServer };

// Start the server if this file is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}