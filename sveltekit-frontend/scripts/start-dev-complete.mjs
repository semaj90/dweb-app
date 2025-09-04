#!/usr/bin/env node

/**
 * Complete Development Environment Startup
 * Wires up all services for Legal AI Platform development
 */

import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const execAsync = promisify(exec);

// Prefer postgres superuser for local dev unless explicitly overridden
if (!process.env.DATABASE_URL) {
  process.env.DATABASE_URL = 'postgresql://postgres:123456@localhost:5432/legal_ai_db';
}

const services = {
  databases: [
    { name: 'PostgreSQL', command: 'net start postgresql-x64-17', port: 5432, optional: false },
    { name: 'Redis', command: 'redis-server', port: 6379, optional: false },
    { name: 'Neo4j', command: 'powershell -Command "Start-Service neo4j"', port: 7474, optional: true }
  ],
  ai: [
    { name: 'Ollama', command: 'ollama serve', port: 11434, optional: false },
    { name: 'Qdrant', command: '.\\qdrant-windows\\qdrant.exe', port: 6333, optional: true },
    { name: 'MinIO', command: 'minio.exe server ./minio-data --address :9000 --console-address :9001', port: 9000, optional: true }
  ],
  goServices: [
    { name: 'Main Service', path: 'go-microservice', binary: 'bin/main-service.exe', port: 8080, optional: false },
    { name: 'Upload Service', path: 'go-microservice', binary: 'bin/upload-service.exe', port: 8093, optional: false },
    { name: 'gRPC Server', path: 'go-microservice', binary: 'bin/grpc-server.exe', port: 8084, optional: false },
    { name: 'Summarizer Service', path: 'go-microservice', binary: 'bin/summarizer-service.exe', port: 8092, optional: true },
    { name: 'Enhanced RAG', path: 'go-microservice', binary: 'bin/enhanced-rag.exe', port: 8094, optional: true },
    { name: 'Load Balancer', path: 'go-microservice', binary: 'bin/load-balancer.exe', port: 8099, optional: true }
  ],
  frontend: [
    { name: 'SvelteKit', command: 'npm run dev -- --host 0.0.0.0', path: 'sveltekit-frontend', port: 5173, optional: false }
  ]
};

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function isProcessRunning(processName) {
  return new Promise((resolve) => {
    exec(`tasklist | findstr "${processName}"`, (error) => {
      resolve(!error);
    });
  });
}

function isPortInUse(port) {
  return new Promise((resolve) => {
    exec(`netstat -an | findstr ":${port}"`, (error) => {
      resolve(!error);
    });
  });
}

async function buildGoServices() {
  log('🔨 Building Go microservices...', 'cyan');

  const buildCommands = [
    'go build -o bin/main-service.exe ./main.go',
    'go build -o bin/upload-service.exe ./cmd/upload-service/main.go',
    'go build -o bin/grpc-server.exe ./cmd/grpc-server/main.go',
    'go build -o bin/summarizer-service.exe ./cmd/summarizer-service/main.go'
  ];

  for (const command of buildCommands) {
    try {
      log(`  Building: ${command}`, 'yellow');
      await execAsync(command, { cwd: path.join(process.cwd(), '..', 'go-microservice') });
      log(`  ✅ Built successfully`, 'green');
    } catch (error) {
      log(`  ❌ Build failed: ${error.message}`, 'red');
    }
  }
}

async function startService(service, category) {
  const serviceName = service.name;

  // Check if service is already running
  if (service.port) {
    const portInUse = await isPortInUse(service.port);
    if (portInUse) {
      log(`  ✅ ${serviceName} already running on port ${service.port}`, 'green');
      return true;
    }
  }

  try {
    let command = service.command;
    let cwd = process.cwd();

    if (service.path) {
      cwd = path.join(process.cwd(), '..', service.path);
    }

    if (service.binary) {
      command = service.binary;
      const binaryPath = path.join(cwd, service.binary);
      if (!fs.existsSync(binaryPath)) {
        if (!service.optional) {
          log(`  ❌ ${serviceName} binary not found: ${binaryPath}`, 'red');
          return false;
        } else {
          log(`  ⚠️  ${serviceName} binary not found (optional service)`, 'yellow');
          return true;
        }
      }
    }

    log(`  🚀 Starting ${serviceName}...`, 'cyan');

    const process = spawn('cmd', ['/c', command], {
      cwd,
      detached: true,
      stdio: 'ignore'
    });

    // Give the service time to start
    await new Promise(resolve => setTimeout(resolve, 2000));

    if (service.port) {
      const portInUse = await isPortInUse(service.port);
      if (portInUse) {
        log(`  ✅ ${serviceName} started on port ${service.port}`, 'green');
        return true;
      } else {
        log(`  ❌ ${serviceName} failed to start on port ${service.port}`, 'red');
        return false;
      }
    } else {
      log(`  ✅ ${serviceName} started`, 'green');
      return true;
    }
  } catch (error) {
    if (!service.optional) {
      log(`  ❌ Failed to start ${serviceName}: ${error.message}`, 'red');
      return false;
    } else {
      log(`  ⚠️  Failed to start ${serviceName} (optional): ${error.message}`, 'yellow');
      return true;
    }
  }
}

async function healthCheck() {
  log('🏥 Performing health checks...', 'cyan');

  const healthChecks = [
    { name: 'Ollama', url: 'http://localhost:11434/api/tags' },
    { name: 'Qdrant', url: 'http://localhost:6333' },
    { name: 'Main Service', url: 'http://localhost:8080/health' },
    { name: 'Upload Service', url: 'http://localhost:8093/health' },
    { name: 'SvelteKit', url: 'http://localhost:5173' }
  ];

  for (const check of healthChecks) {
    try {
      await execAsync(`curl -s ${check.url}`, { timeout: 5000 });
      log(`  ✅ ${check.name}: Healthy`, 'green');
    } catch (error) {
      log(`  ❌ ${check.name}: Not responding`, 'red');
    }
  }
}

async function main() {
  log('', 'reset');
  log('================================================================================', 'bright');
  log('🚀 LEGAL AI PLATFORM - COMPLETE DEVELOPMENT STARTUP', 'bright');
  log('================================================================================', 'bright');
  log('', 'reset');

  // Ensure directories exist
  const requiredDirs = ['minio-data', 'redis-windows', 'qdrant-windows'];
  for (const dir of requiredDirs) {
    const dirPath = path.join(process.cwd(), '..', dir);
    if (!fs.existsSync(dirPath)) {
      log(`📁 Creating directory: ${dir}`, 'yellow');
      fs.mkdirSync(dirPath, { recursive: true });
    }
  }

  // Build Go services
  await buildGoServices();

  // Start database services
  log('🗄️  Starting database services...', 'magenta');
  for (const service of services.databases) {
    await startService(service, 'database');
  }

  // Start AI services
  log('🧠 Starting AI services...', 'magenta');
  for (const service of services.ai) {
    await startService(service, 'ai');
  }

  // Wait for services to initialize
  log('⏱️  Waiting for services to initialize...', 'yellow');
  await new Promise(resolve => setTimeout(resolve, 5000));

  // Start Go microservices
  log('⚙️  Starting Go microservices...', 'magenta');
  for (const service of services.goServices) {
    await startService(service, 'go');
  }

  // Start frontend
  log('🎨 Starting frontend...', 'magenta');
  for (const service of services.frontend) {
    await startService(service, 'frontend');
  }

  // Wait for everything to stabilize
  await new Promise(resolve => setTimeout(resolve, 8000));

  // Health check
  await healthCheck();

  // Generate readiness flag if all core services appear up
  try {
    const readinessFlag = path.join(process.cwd(), '..', 'platform-ready.flag');
    const meta = {
      generatedAt: new Date().toISOString(),
      databaseUrl: process.env.DATABASE_URL,
      services: {
        frontend: await isPortInUse(5173),
        main: await isPortInUse(8080),
        upload: await isPortInUse(8093),
        postgres: await isPortInUse(5432),
        redis: await isPortInUse(6379),
        ollama: await isPortInUse(11434)
      }
    };
    const allCore = Object.values(meta.services).every(Boolean);
    meta.allCore = allCore;
    fs.writeFileSync(readinessFlag, JSON.stringify(meta, null, 2));
    if (allCore) {
      log('🟢 platform-ready.flag created (all core services up)', 'green');
    } else {
      log('🟡 platform-ready.flag created (some services missing)', 'yellow');
    }
  } catch (e) {
    log(`⚠️ Failed to write readiness flag: ${e.message}`, 'yellow');
  }

  log('', 'reset');
  log('================================================================================', 'bright');
  log('🎉 LEGAL AI PLATFORM STARTUP COMPLETE!', 'green');
  log('================================================================================', 'bright');
  log('', 'reset');

  log('📍 Access Points:', 'cyan');
  log('   Frontend:        http://localhost:5173', 'bright');
  log('   Main Service:    http://localhost:8080', 'bright');
  log('   Upload API:      http://localhost:8093/upload', 'bright');
  log('   gRPC Server:     http://localhost:8084', 'bright');
  log('   Ollama API:      http://localhost:11434', 'bright');
  log('   MinIO Console:   http://localhost:9001', 'bright');
  log('   Qdrant API:      http://localhost:6333', 'bright');
  log('   Neo4j Browser:   http://localhost:7474', 'bright');
  log('', 'reset');

  log('💾 Database Connections:', 'cyan');
  log('   PostgreSQL:      postgresql://postgres:123456@localhost:5432/legal_ai_db', 'bright');
  log('   Redis:           redis://localhost:6379', 'bright');
  log('', 'reset');

  log('Happy coding! 🚀', 'green');
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  log('', 'reset');
  log('👋 Shutdown signal received. Services will continue running.', 'yellow');
  log('Use MANAGE-SERVICES.bat stop to stop all services.', 'cyan');
  process.exit(0);
});

main().catch(error => {
  log(`❌ Startup failed: ${error.message}`, 'red');
  process.exit(1);
});