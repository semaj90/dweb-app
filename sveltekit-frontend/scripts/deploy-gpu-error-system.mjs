#!/usr/bin/env node

// ======================================================================
// GPU ERROR SYSTEM DEPLOYMENT SCRIPT
// Deploy and configure the complete GPU Loki error orchestrator system
// ======================================================================

import { spawn } from 'child_process';
import { writeFile, readFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';

const DEPLOYMENT_CONFIG = {
  services: [
    {
      name: 'SvelteKit Frontend',
      command: 'npm run dev',
      port: 5173,
      healthCheck: 'http://localhost:5173',
      required: true
    },
    {
      name: 'Ollama Server',
      command: 'ollama serve',
      port: 11434,
      healthCheck: 'http://localhost:11434/api/tags',
      required: false
    }
  ],
  directories: [
    '.vscode',
    'static/gpu-cache',
    'logs'
  ],
  configFiles: [
    {
      path: '.vscode/gpu-error-config.json',
      content: {
        gpuAcceleration: true,
        maxWorkers: 4,
        batchSize: 50,
        cacheSize: 10000,
        ollamaModel: 'gemma3-legal',
        logLevel: 'info'
      }
    }
  ]
};

class GPUErrorSystemDeployer {
  constructor() {
    this.deploymentLog = [];
    this.services = new Map();
  }

  async deploy() {
    console.log('🚀 Deploying GPU Error System...');

    try {
      await this.createDirectories();
      await this.createConfigFiles();
      await this.checkDependencies();
      await this.initializeServices();
      await this.runTests();
      await this.generateDeploymentReport();

      console.log('✅ Deployment completed successfully!');
      return true;
    } catch (error) {
      console.error('❌ Deployment failed:', error);
      await this.cleanup();
      return false;
    }
  }

  async createDirectories() {
    console.log('📁 Creating required directories...');

    for (const dir of DEPLOYMENT_CONFIG.directories) {
      if (!existsSync(dir)) {
        await mkdir(dir, { recursive: true });
        this.log(`Created directory: ${dir}`);
      } else {
        this.log(`Directory exists: ${dir}`);
      }
    }
  }

  async createConfigFiles() {
    console.log('⚙️ Creating configuration files...');

    for (const config of DEPLOYMENT_CONFIG.configFiles) {
      await writeFile(config.path, JSON.stringify(config.content, null, 2));
      this.log(`Created config file: ${config.path}`);
    }
  }

  async checkDependencies() {
    console.log('🔍 Checking dependencies...');

    const dependencies = [
      'src/lib/services/gpu-loki-error-orchestrator.ts',
      'src/lib/services/parallel-error-analyzer.ts',
      'src/lib/services/ai-error-fixer.ts',
      'src/lib/workers/error-analysis-worker.ts',
      'src/lib/stores/enhancedLokiStore.ts'
    ];

    for (const dep of dependencies) {
      if (!existsSync(dep)) {
        throw new Error(`Missing dependency: ${dep}`);
      }
      this.log(`✓ Dependency found: ${dep}`);
    }
  }

  async initializeServices() {
    console.log('🔧 Initializing services...');

    // Check if SvelteKit is already running
    const isRunning = await this.checkServiceHealth('http://localhost:5173');

    if (isRunning) {
      console.log('✓ SvelteKit is already running');
      this.log('SvelteKit service already active');
    } else {
      console.log('⚠️ SvelteKit not running - manual start required');
      this.log('SvelteKit service requires manual start');
    }

    // Check Ollama
    const ollamaRunning = await this.checkServiceHealth('http://localhost:11434/api/tags');

    if (ollamaRunning) {
      console.log('✓ Ollama is running');
      this.log('Ollama service active');
    } else {
      console.log('⚠️ Ollama not running - optional service');
      this.log('Ollama service not available (optional)');
    }
  }

  async runTests() {
    console.log('🧪 Running deployment tests...');

    try {
      // Import and run the test system
      const { GPUErrorSystemTester } = await import('./test-gpu-error-system.mjs');
      const tester = new GPUErrorSystemTester();
      const results = await tester.runAllTests();

      this.log(`Tests completed: ${results.passed}/${results.total} passed`);

      if (results.failed > 0) {
        console.log('⚠️ Some tests failed - system may have limited functionality');
      }

      return results;
    } catch (error) {
      console.error('Test execution failed:', error);
      this.log(`Test execution failed: ${error.message}`);
      return null;
    }
  }

  async generateDeploymentReport() {
    const report = {
      timestamp: new Date().toISOString(),
      status: 'deployed',
      services: await this.getServiceStatus(),
      configuration: DEPLOYMENT_CONFIG,
      logs: this.deploymentLog,
      nextSteps: [
        'Run `npm run dev` to start SvelteKit if not already running',
        'Ensure Ollama is running with `ollama serve` for AI features',
        'Test error processing with `npm run check`',
        'Monitor system with the GPU error processor API'
      ]
    };

    await writeFile('.vscode/gpu-error-deployment-report.json', JSON.stringify(report, null, 2));
    console.log('📄 Deployment report saved to .vscode/gpu-error-deployment-report.json');
  }

  async getServiceStatus() {
    const status = {};

    for (const service of DEPLOYMENT_CONFIG.services) {
      status[service.name] = {
        running: await this.checkServiceHealth(service.healthCheck),
        port: service.port,
        required: service.required
      };
    }

    return status;
  }

  async checkServiceHealth(url) {
    try {
      const response = await fetch(url);
      return response.ok;
    } catch {
      return false;
    }
  }

  async cleanup() {
    console.log('🧹 Cleaning up failed deployment...');

    // Kill any services we started
    for (const [name, process] of this.services) {
      try {
        process.kill();
        console.log(`Stopped service: ${name}`);
      } catch {
        // Process may have already stopped
      }
    }
  }

  log(message) {
    const timestamp = new Date().toISOString();
    this.deploymentLog.push(`[${timestamp}] ${message}`);
    console.log(`  ${message}`);
  }
}

// ======================================================================
// CLI INTERFACE
// ======================================================================

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'deploy';

  switch (command) {
    case 'deploy':
      await deploySystem();
      break;

    case 'test':
      await testSystem();
      break;

    case 'status':
      await checkStatus();
      break;

    case 'help':
      printHelp();
      break;

    default:
      console.error(`Unknown command: ${command}`);
      printHelp();
      process.exit(1);
  }
}

async function deploySystem() {
  const deployer = new GPUErrorSystemDeployer();
  const success = await deployer.deploy();
  process.exit(success ? 0 : 1);
}

async function testSystem() {
  console.log('🧪 Running system tests only...');

  try {
    const { GPUErrorSystemTester } = await import('./test-gpu-error-system.mjs');
    const tester = new GPUErrorSystemTester();
    const results = await tester.runAllTests();

    process.exit(results.failed === 0 ? 0 : 1);
  } catch (error) {
    console.error('Test execution failed:', error);
    process.exit(1);
  }
}

async function checkStatus() {
  console.log('📊 Checking system status...');

  const services = [
    { name: 'SvelteKit', url: 'http://localhost:5173' },
    { name: 'GPU Error API', url: 'http://localhost:5173/api/gpu-error-processor' },
    { name: 'Ollama', url: 'http://localhost:11434/api/tags' }
  ];

  for (const service of services) {
    const status = await checkServiceHealth(service.url);
    const icon = status ? '✅' : '❌';
    console.log(`  ${icon} ${service.name}: ${status ? 'Running' : 'Not Running'}`);
  }

  // Check file system
  const files = [
    'src/lib/services/gpu-loki-error-orchestrator.ts',
    'src/lib/services/parallel-error-analyzer.ts',
    'src/lib/services/ai-error-fixer.ts'
  ];

  console.log('\n📁 File System:');
  for (const file of files) {
    const exists = existsSync(file);
    const icon = exists ? '✅' : '❌';
    console.log(`  ${icon} ${file}`);
  }
}

async function checkServiceHealth(url) {
  try {
    const response = await fetch(url);
    return response.ok;
  } catch {
    return false;
  }
}

function printHelp() {
  console.log(`
GPU Error System Deployment Tool

Usage:
  node deploy-gpu-error-system.mjs [command]

Commands:
  deploy    Deploy the complete GPU error system (default)
  test      Run system tests only
  status    Check system status
  help      Show this help message

Examples:
  node deploy-gpu-error-system.mjs deploy
  node deploy-gpu-error-system.mjs test
  node deploy-gpu-error-system.mjs status
  `);
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('Deployment script failed:', error);
    process.exit(1);
  });
}

export { GPUErrorSystemDeployer };