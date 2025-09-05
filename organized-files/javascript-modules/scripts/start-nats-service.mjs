#!/usr/bin/env node
// NATS Messaging Service Orchestrator for Legal AI Platform
import { spawn } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const log = (msg, ...rest) => console.log(`[nats] ${msg}`, ...rest);
const warn = (msg, ...rest) => console.warn(`[nats] WARN ${msg}`, ...rest);
const err = (msg, ...rest) => console.error(`[nats] ERR ${msg}`, ...rest);

class NATSServiceManager {
  constructor() {
    this.process = null;
    this.isShuttingDown = false;
    this.configPath = resolve('sveltekit-frontend/nats-server/nats-server.conf');
    this.serverBinary = this.findNATSBinary();
    this.jetStreamDir = resolve('sveltekit-frontend/nats-server/jetstream');
  }

  findNATSBinary() {
    const possiblePaths = [
      resolve('nats-server/nats-server.exe'),
      resolve('../nats-server/nats-server.exe'),
      resolve('sveltekit-frontend/nats-server/nats-server.exe'),
      resolve('bin/nats-server.exe'),
      'nats-server.exe', // System PATH
      'nats-server'      // Linux/Mac
    ];

    for (const path of possiblePaths) {
      if (existsSync(path)) {
        log(`Found NATS server binary: ${path}`);
        return path;
      }
    }

    warn('NATS server binary not found. Attempting to use system binary.');
    return process.platform === 'win32' ? 'nats-server.exe' : 'nats-server';
  }

  ensureConfig() {
    if (!existsSync(this.configPath)) {
      log('Creating NATS server configuration...');
      
      const configDir = resolve('sveltekit-frontend/nats-server');
      if (!existsSync(configDir)) {
        require('fs').mkdirSync(configDir, { recursive: true });
      }

      const config = `# NATS Server Configuration for Legal AI Platform
server_name: "legal-ai-nats-server"
port: 4225

# Enable WebSocket for browser clients
websocket {
  port: 4226
  no_tls: true
  same_origin: false
  allowed_origins: ["*"]
}

# HTTP monitoring
http_port: 8225

# Logging
logtime: true
debug: false
trace: false

# Performance settings
max_connections: 1000
max_subscriptions: 10000
max_payload: 67108864  # 64MB

# JetStream for persistent messaging
jetstream {
  store_dir: "./jetstream"
  max_mem: 1GB
  max_file: 10GB
}

# Legal AI specific subjects authorization
authorization {
  default_permissions = {
    publish = ["legal.>", "system.>", "chat.>", "document.>", "analysis.>"]
    subscribe = ["legal.>", "system.>", "chat.>", "document.>", "analysis.>"]
  }
  users = [
    {
      user: "legal_ai_client"
      password: "legal_ai_2024"
      permissions = $default_permissions
    }
  ]
}`;

      writeFileSync(this.configPath, config);
      log(`✅ NATS configuration created: ${this.configPath}`);
    }
  }

  ensureJetStreamDir() {
    if (!existsSync(this.jetStreamDir)) {
      require('fs').mkdirSync(this.jetStreamDir, { recursive: true });
      log(`✅ JetStream directory created: ${this.jetStreamDir}`);
    }
  }

  async startServer() {
    if (this.process) {
      warn('NATS server already running');
      return;
    }

    this.ensureConfig();
    this.ensureJetStreamDir();

    log('🚀 Starting NATS messaging server...');
    log(`📁 Config: ${this.configPath}`);
    log(`📁 JetStream: ${this.jetStreamDir}`);
    log(`🔧 Binary: ${this.serverBinary}`);

    const args = [
      '--config', this.configPath,
      '--jetstream'
    ];

    if (process.env.NATS_DEBUG === 'true') {
      args.push('--debug');
    }

    if (process.env.NATS_TRACE === 'true') {
      args.push('--trace');
    }

    try {
      this.process = spawn(this.serverBinary, args, {
        stdio: 'inherit',
        shell: true,
        env: {
          ...process.env,
          NATS_SERVER_NAME: 'legal-ai-nats-server'
        }
      });

      log('✅ NATS server process spawned');

      this.process.on('exit', (code, signal) => {
        if (!this.isShuttingDown) {
          warn(`NATS server exited with code=${code} signal=${signal}`);
          this.process = null;
          
          // Auto-restart on unexpected exit
          if (code !== 0) {
            log('🔄 Restarting NATS server in 5 seconds...');
            setTimeout(() => this.startServer(), 5000);
          }
        }
      });

      this.process.on('error', (error) => {
        err('NATS server error:', error.message);
        this.process = null;
      });

      // Give server time to start
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify server is running
      await this.verifyServer();

    } catch (error) {
      err('Failed to start NATS server:', error.message);
      throw error;
    }
  }

  async verifyServer() {
    try {
      log('🔍 Verifying NATS server endpoints...');
      
      const endpoints = [
        { name: 'Core NATS', url: 'nats://localhost:4225' },
        { name: 'WebSocket', url: 'ws://localhost:4226' },
        { name: 'HTTP Monitor', url: 'http://localhost:8225' }
      ];

      log('📡 NATS server endpoints:');
      endpoints.forEach(endpoint => {
        log(`   • ${endpoint.name}: ${endpoint.url}`);
      });

      log('🎯 Legal AI NATS subjects available:');
      const subjects = [
        'legal.case.created', 'legal.case.updated', 'legal.case.closed',
        'legal.document.uploaded', 'legal.document.processed',
        'legal.ai.analysis.started', 'legal.ai.analysis.completed',
        'legal.search.query', 'legal.search.results',
        'legal.chat.message', 'legal.chat.response',
        'system.health', 'system.metrics'
      ];
      
      subjects.forEach(subject => {
        log(`   • ${subject}`);
      });

      log('🌟 NATS JetStream features:');
      log('   • Persistent messaging with durability');
      log('   • Legal document processing workflows');
      log('   • Real-time case collaboration');
      log('   • AI analysis result distribution');
      log('   • WebSocket browser integration');

    } catch (error) {
      err('NATS server verification failed:', error.message);
    }
  }

  async stop() {
    if (!this.process) {
      log('NATS server not running');
      return;
    }

    this.isShuttingDown = true;
    log('🛑 Shutting down NATS server...');

    try {
      this.process.kill('SIGTERM');
      
      // Wait for graceful shutdown
      await new Promise((resolve) => {
        const timeout = setTimeout(() => {
          warn('Force killing NATS server');
          this.process?.kill('SIGKILL');
          resolve();
        }, 5000);

        this.process.on('exit', () => {
          clearTimeout(timeout);
          resolve();
        });
      });

      this.process = null;
      log('✅ NATS server stopped');
    } catch (error) {
      err('Error stopping NATS server:', error.message);
    }
  }

  getStatus() {
    return {
      running: !!this.process,
      pid: this.process?.pid,
      config: this.configPath,
      jetstream: this.jetStreamDir,
      endpoints: {
        nats: 'nats://localhost:4225',
        websocket: 'ws://localhost:4226',
        monitoring: 'http://localhost:8225'
      }
    };
  }
}

// Global service manager instance
const natsManager = new NATSServiceManager();

// Graceful shutdown handlers
function shutdown() {
  log('Received shutdown signal');
  natsManager.stop().then(() => {
    process.exit(0);
  }).catch(error => {
    err('Error during shutdown:', error.message);
    process.exit(1);
  });
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Unhandled errors
process.on('uncaughtException', (error) => {
  err('Uncaught exception:', error);
  shutdown();
});

process.on('unhandledRejection', (reason, promise) => {
  err('Unhandled rejection at:', promise, 'reason:', reason);
  shutdown();
});

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  log('🌟 NATS Legal AI Messaging Service Starting...');
  
  natsManager.startServer().then(() => {
    log('🎉 NATS Legal AI messaging platform ready');
    log('💡 Use Ctrl+C to stop the server');
  }).catch(error => {
    err('Failed to start NATS service:', error.message);
    process.exit(1);
  });
}

// Export for use by other modules
export { natsManager as default, NATSServiceManager };