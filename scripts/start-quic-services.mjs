#!/usr/bin/env node
// QUIC Services Orchestrator for Legal AI Platform
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const log = (msg, ...rest) => console.log(`[quic] ${msg}`, ...rest);
const warn = (msg, ...rest) => console.warn(`[quic] WARN ${msg}`, ...rest);
const err = (msg, ...rest) => console.error(`[quic] ERR ${msg}`, ...rest);

// QUIC service configuration matching tasks.json patterns
const quicServices = [
  {
    name: 'quic-gateway',
    cmd: resolve('../go-microservice/bin/quic-gateway.exe'),
    args: ['--port=8097', '--http3-port=8098'],
    port: 8097,
    description: 'QUIC Gateway Service - Ultra-low latency transport'
  },
  {
    name: 'quic-tensor-server',
    cmd: resolve('../quic-services/quic-tensor-server.exe'),
    args: ['--port=8098', '--workers=2'],
    port: 8098,
    description: 'QUIC Tensor Server - GPU-accelerated inference'
  },
  {
    name: 'quic-vector-proxy',
    cmd: resolve('../go-microservice/bin/quic-vector-proxy.exe'),
    args: ['--port=8099', '--backend-pool=localhost:6333'],
    port: 8099,
    description: 'QUIC Vector Proxy - High-speed vector operations'
  },
  {
    name: 'quic-ai-stream',
    cmd: resolve('../quic-services/quic-ai-stream.exe'),
    args: ['--port=8100', '--stream-buffer=1024'],
    port: 8100,
    description: 'QUIC AI Stream - Real-time AI response streaming'
  }
];

const processes = new Map();
let shuttingDown = false;

function checkBinary(service) {
  if (!existsSync(service.cmd)) {
    warn(`Binary missing for ${service.name}: ${service.cmd}`);
    return false;
  }
  return true;
}

function spawnQuicService(service) {
  if (!checkBinary(service)) {
    return null;
  }

  log(`Starting ${service.name}: ${service.description}`);
  
  const env = {
    ...process.env,
    QUIC_PORT: String(service.port),
    QUIC_LOG_LEVEL: process.env.QUIC_LOG_LEVEL || 'info',
    QUIC_CERT_PATH: process.env.QUIC_CERT_PATH || './certs/server.crt',
    QUIC_KEY_PATH: process.env.QUIC_KEY_PATH || './certs/server.key'
  };

  const child = spawn(service.cmd, service.args, {
    stdio: 'inherit',
    shell: true,
    env: env
  });

  processes.set(service.name, { service, child, restarts: 0 });
  
  log(`✅ ${service.name} spawned on port ${service.port}`);

  child.on('exit', (code, signal) => {
    if (!shuttingDown) {
      warn(`${service.name} exited with code=${code} signal=${signal}`);
      processes.delete(service.name);
      
      const entry = processes.get(service.name) || { restarts: 0 };
      const restartDelay = Math.min(10000, 1000 * Math.pow(2, entry.restarts || 0));
      entry.restarts = (entry.restarts || 0) + 1;
      
      if (entry.restarts <= 5) {
        log(`Restarting ${service.name} in ${restartDelay}ms (attempt ${entry.restarts})`);
        setTimeout(() => spawnQuicService(service), restartDelay).unref();
      } else {
        err(`${service.name} failed too many times, giving up`);
      }
    }
  });

  child.on('error', (error) => {
    err(`${service.name} error:`, error.message);
  });

  return child;
}

async function startAllQuicServices() {
  log('🚀 Starting QUIC Services Suite for Legal AI Platform');
  log(`🔧 Services to start: ${quicServices.length}`);
  
  // Check for required certificates
  const certPath = process.env.QUIC_CERT_PATH || './certs/server.crt';
  const keyPath = process.env.QUIC_KEY_PATH || './certs/server.key';
  
  if (!existsSync(certPath) || !existsSync(keyPath)) {
    warn('QUIC certificates not found. Some services may fall back to insecure mode.');
    warn(`Expected: ${certPath}, ${keyPath}`);
  }

  let started = 0;
  let failed = 0;

  for (const service of quicServices) {
    try {
      const child = spawnQuicService(service);
      if (child) {
        started++;
        // Small delay between service starts to avoid port conflicts
        await new Promise(resolve => setTimeout(resolve, 500));
      } else {
        failed++;
      }
    } catch (error) {
      err(`Failed to start ${service.name}:`, error.message);
      failed++;
    }
  }

  log(`🎉 QUIC Services startup complete: ${started} started, ${failed} failed`);
  
  if (started > 0) {
    log('📡 Available QUIC endpoints:');
    quicServices.forEach(service => {
      if (processes.has(service.name)) {
        log(`   • ${service.name}: quic://localhost:${service.port}`);
      }
    });
  }
}

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  
  log('🛑 Shutting down QUIC services...');
  
  for (const { child } of processes.values()) {
    try {
      child.kill('SIGTERM');
    } catch (error) {
      // Ignore kill errors during shutdown
    }
  }
  
  setTimeout(() => {
    log('QUIC services shutdown complete');
    process.exit(0);
  }, 2000).unref();
}

// Handle graceful shutdown
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

// Start services
startAllQuicServices().catch(error => {
  err('Failed to start QUIC services:', error);
  process.exit(1);
});