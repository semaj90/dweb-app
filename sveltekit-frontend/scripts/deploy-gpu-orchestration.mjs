#!/usr/bin/env node

/**
 * GPU-Accelerated Orchestration Deployment Script
 * Deploys and manages the complete orchestration system
 */

import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import { existsSync } from 'fs';
import { readFile, writeFile } from 'fs/promises';
import path from 'path';

const execAsync = promisify(exec);

const COLORS = {
  GREEN: '\x1b[32m',
  RED: '\x1b[31m',
  YELLOW: '\x1b[33m',
  BLUE: '\x1b[34m',
  RESET: '\x1b[0m',
  BOLD: '\x1b[1m'
};

const CONFIG = {
  orchestratorPort: 8094,
  errorProcessorPort: 8095,
  svelteKitPort: 5173,
  ollamaPort: 11434,
  models: {
    primary: 'gemma3-legal',
    embedding: 'nomic-embed-text',
    blocked: ['gemma3:2b', 'gemma3:8b', 'gemma3:27b', 'gemma2*', 'gemma*']
  },
  timeout: 30000
};

/**
 * Main deployment function
 */
async function main() {
  console.log(`${COLORS.BOLD}${COLORS.BLUE}🚀 GPU-Accelerated Orchestration Deployment${COLORS.RESET}\n`);

  try {
    const action = process.argv[2] || 'deploy';
    
    switch (action) {
      case 'deploy':
        await deployOrchestration();
        break;
      case 'start':
        await startServices();
        break;
      case 'stop':
        await stopServices();
        break;
      case 'status':
        await checkStatus();
        break;
      case 'test':
        await testDeployment();
        break;
      default:
        console.log(`Usage: node deploy-gpu-orchestration.mjs [deploy|start|stop|status|test]`);
        process.exit(1);
    }

  } catch (error) {
    console.error(`${COLORS.RED}❌ Deployment failed:${COLORS.RESET}`, error.message);
    process.exit(1);
  }
}

/**
 * Deploy the complete orchestration system
 */
async function deployOrchestration() {
  console.log(`${COLORS.GREEN}🔧 Deploying GPU orchestration system...${COLORS.RESET}\n`);

  const steps = [
    { name: 'Validate Environment', fn: validateEnvironment },
    { name: 'Check Model Constraints', fn: checkModelConstraints },
    { name: 'Verify Go Services', fn: verifyGoServices },
    { name: 'Test FlashAttention', fn: testFlashAttention },
    { name: 'Configure MCP', fn: configureMCP },
    { name: 'Start Orchestration', fn: startOrchestration },
    { name: 'Verify Deployment', fn: verifyDeployment }
  ];

  let successful = 0;

  for (const step of steps) {
    try {
      console.log(`${COLORS.BLUE}⚡ ${step.name}...${COLORS.RESET}`);
      await step.fn();
      console.log(`${COLORS.GREEN}✅ ${step.name} completed${COLORS.RESET}`);
      successful++;
    } catch (error) {
      console.error(`${COLORS.RED}❌ ${step.name} failed:${COLORS.RESET}`, error.message);
      break;
    }
  }

  if (successful === steps.length) {
    console.log(`\n${COLORS.BOLD}${COLORS.GREEN}🎉 GPU orchestration deployment complete!${COLORS.RESET}`);
    await displayStatus();
  } else {
    console.log(`\n${COLORS.YELLOW}⚠️ Deployment partially complete (${successful}/${steps.length} steps)${COLORS.RESET}`);
  }
}

/**
 * Validate development environment
 */
async function validateEnvironment() {
  // Check Node.js version
  const nodeVersion = process.version;
  const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
  if (majorVersion < 18) {
    throw new Error(`Node.js 18+ required, found ${nodeVersion}`);
  }

  // Check key files
  const requiredFiles = [
    'src/lib/services/nodejs-orchestrator.ts',
    'src/lib/services/flashattention2-rtx3060.ts',
    '.vscode/mcp.json',
    'package.json'
  ];

  for (const file of requiredFiles) {
    if (!existsSync(file)) {
      throw new Error(`Required file missing: ${file}`);
    }
  }

  console.log(`  📋 Environment validated (Node ${nodeVersion})`);
}

/**
 * Check model constraints and Ollama service
 */
async function checkModelConstraints() {
  try {
    const response = await fetch(`http://localhost:${CONFIG.ollamaPort}/api/tags`, {
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      throw new Error('Ollama service not available');
    }

    const data = await response.json();
    const models = data.models?.map(m => m.name) || [];

    // Check required models
    const hasGemma3Legal = models.some(name => name.includes('gemma3-legal'));
    const hasNomicEmbed = models.some(name => name.includes('nomic-embed'));

    if (!hasGemma3Legal) {
      throw new Error('gemma3-legal model not found. Run: ollama pull gemma3-legal');
    }

    if (!hasNomicEmbed) {
      throw new Error('nomic-embed-text model not found. Run: ollama pull nomic-embed-text');
    }

    // Check for blocked models
    const blockedModels = models.filter(name => 
      CONFIG.models.blocked.some(blocked => 
        blocked.endsWith('*') ? name.startsWith(blocked.slice(0, -1)) : name === blocked
      )
    );

    if (blockedModels.length > 0) {
      console.log(`  ${COLORS.YELLOW}⚠️ Blocked models detected: ${blockedModels.join(', ')}${COLORS.RESET}`);
    }

    console.log(`  📊 Models validated: gemma3-legal ✅, nomic-embed ✅`);

  } catch (error) {
    throw new Error(`Model constraint check failed: ${error.message}`);
  }
}

/**
 * Verify Go microservices availability
 */
async function verifyGoServices() {
  const services = [
    { name: 'Enhanced RAG', path: '../go-microservice/bin/enhanced-rag.exe', port: 8094 },
    { name: 'Upload Service', path: '../go-microservice/bin/upload-service.exe', port: 8093 }
  ];

  for (const service of services) {
    if (existsSync(service.path)) {
      console.log(`  📦 ${service.name}: Binary found at ${service.path}`);
    } else {
      console.log(`  ${COLORS.YELLOW}⚠️ ${service.name}: Binary not found, may need compilation${COLORS.RESET}`);
    }

    // Check if service is running
    try {
      const response = await fetch(`http://localhost:${service.port}/health`, {
        signal: AbortSignal.timeout(2000)
      });
      if (response.ok) {
        console.log(`  ✅ ${service.name}: Service running on port ${service.port}`);
      }
    } catch {
      console.log(`  🔴 ${service.name}: Service not running on port ${service.port}`);
    }
  }
}

/**
 * Test FlashAttention GPU processing
 */
async function testFlashAttention() {
  try {
    const response = await fetch(`http://localhost:${CONFIG.svelteKitPort}/api/gpu-status`, {
      signal: AbortSignal.timeout(5000)
    });

    if (response.ok) {
      const gpuStatus = await response.json();
      console.log(`  ⚡ FlashAttention: ${gpuStatus.flashAttentionEnabled ? 'Enabled' : 'Disabled'}`);
      console.log(`  🖥️ GPU Device: ${gpuStatus.device || 'RTX3060Ti'}`);
    } else {
      console.log(`  ${COLORS.YELLOW}⚠️ FlashAttention: Service not responding${COLORS.RESET}`);
    }
  } catch (error) {
    console.log(`  ${COLORS.YELLOW}⚠️ FlashAttention test failed: ${error.message}${COLORS.RESET}`);
  }
}

/**
 * Configure MCP integration
 */
async function configureMCP() {
  const mcpConfigPath = '.vscode/mcp.json';

  if (!existsSync(mcpConfigPath)) {
    throw new Error('MCP configuration file not found');
  }

  try {
    const mcpConfig = JSON.parse(await readFile(mcpConfigPath, 'utf-8'));
    
    // Verify model constraints are configured
    if (mcpConfig.modelConstraints?.enforceModelList) {
      console.log(`  🔗 MCP: Model constraints enforced`);
    } else {
      console.log(`  ${COLORS.YELLOW}⚠️ MCP: Model constraints not enforced${COLORS.RESET}`);
    }

    // Verify orchestration is enabled
    if (mcpConfig.orchestration?.enabled) {
      console.log(`  🎯 MCP: Orchestration enabled`);
    } else {
      console.log(`  ${COLORS.YELLOW}⚠️ MCP: Orchestration disabled${COLORS.RESET}`);
    }

  } catch (error) {
    throw new Error(`MCP configuration validation failed: ${error.message}`);
  }
}

/**
 * Start orchestration services
 */
async function startOrchestration() {
  console.log(`  🚀 Starting orchestration services...`);

  // The services are managed by the deployment API
  try {
    const response = await fetch(`http://localhost:${CONFIG.svelteKitPort}/api/gpu-orchestration/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        action: 'start',
        config: CONFIG
      })
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`  ✅ Orchestration API responded: ${result.message}`);
    } else {
      console.log(`  ${COLORS.YELLOW}⚠️ Orchestration API not available, starting locally${COLORS.RESET}`);
    }
  } catch (error) {
    console.log(`  ${COLORS.YELLOW}⚠️ Starting local orchestration fallback${COLORS.RESET}`);
  }
}

/**
 * Verify deployment success
 */
async function verifyDeployment() {
  console.log(`  🔍 Verifying deployment...`);

  const checks = [
    { name: 'SvelteKit', url: `http://localhost:${CONFIG.svelteKitPort}` },
    { name: 'Ollama', url: `http://localhost:${CONFIG.ollamaPort}/api/tags` },
    { name: 'Enhanced RAG', url: `http://localhost:${CONFIG.orchestratorPort}/health` }
  ];

  let passedChecks = 0;

  for (const check of checks) {
    try {
      const response = await fetch(check.url, { signal: AbortSignal.timeout(3000) });
      if (response.ok) {
        console.log(`    ✅ ${check.name}: Running`);
        passedChecks++;
      } else {
        console.log(`    🔴 ${check.name}: Error (${response.status})`);
      }
    } catch {
      console.log(`    🔴 ${check.name}: Not responding`);
    }
  }

  if (passedChecks < checks.length) {
    throw new Error(`Verification failed: ${passedChecks}/${checks.length} services running`);
  }

  console.log(`  🎯 Deployment verified: ${passedChecks}/${checks.length} services running`);
}

/**
 * Start services individually
 */
async function startServices() {
  console.log(`${COLORS.GREEN}🚀 Starting orchestration services...${COLORS.RESET}\n`);

  // Call deployment API to start services
  try {
    const response = await fetch(`http://localhost:${CONFIG.svelteKitPort}/api/gpu-orchestration/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'start' })
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`${COLORS.GREEN}✅ Services started successfully${COLORS.RESET}`);
      console.log(JSON.stringify(result, null, 2));
    } else {
      throw new Error(`API responded with ${response.status}`);
    }
  } catch (error) {
    console.error(`${COLORS.RED}❌ Failed to start services:${COLORS.RESET}`, error.message);
  }
}

/**
 * Stop orchestration services
 */
async function stopServices() {
  console.log(`${COLORS.YELLOW}🛑 Stopping orchestration services...${COLORS.RESET}\n`);

  try {
    const response = await fetch(`http://localhost:${CONFIG.svelteKitPort}/api/gpu-orchestration/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'stop' })
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`${COLORS.GREEN}✅ Services stopped${COLORS.RESET}`);
      console.log(JSON.stringify(result, null, 2));
    } else {
      throw new Error(`API responded with ${response.status}`);
    }
  } catch (error) {
    console.error(`${COLORS.RED}❌ Failed to stop services:${COLORS.RESET}`, error.message);
  }
}

/**
 * Check orchestration status
 */
async function checkStatus() {
  console.log(`${COLORS.BLUE}📊 Checking orchestration status...${COLORS.RESET}\n`);

  try {
    const response = await fetch(`http://localhost:${CONFIG.svelteKitPort}/api/gpu-orchestration/deploy`);
    
    if (response.ok) {
      const status = await response.json();
      await displayStatus(status);
    } else {
      throw new Error(`Status API responded with ${response.status}`);
    }
  } catch (error) {
    console.error(`${COLORS.RED}❌ Failed to get status:${COLORS.RESET}`, error.message);
  }
}

/**
 * Test the deployment with sample requests
 */
async function testDeployment() {
  console.log(`${COLORS.BLUE}🧪 Testing GPU orchestration deployment...${COLORS.RESET}\n`);

  const tests = [
    {
      name: 'Orchestrator Health',
      url: `http://localhost:${CONFIG.orchestratorPort}/health`,
      expected: 200
    },
    {
      name: 'Model Validation',
      url: `http://localhost:${CONFIG.ollamaPort}/api/tags`,
      expected: 200,
      validator: (data) => {
        const models = data.models?.map(m => m.name) || [];
        return models.some(name => name.includes('gemma3-legal'));
      }
    },
    {
      name: 'FlashAttention GPU',
      url: `http://localhost:${CONFIG.svelteKitPort}/api/gpu-status`,
      expected: 200
    }
  ];

  let passedTests = 0;

  for (const test of tests) {
    try {
      const response = await fetch(test.url, { signal: AbortSignal.timeout(5000) });
      
      if (response.status === test.expected) {
        if (test.validator) {
          const data = await response.json();
          if (test.validator(data)) {
            console.log(`  ${COLORS.GREEN}✅ ${test.name}: PASS${COLORS.RESET}`);
            passedTests++;
          } else {
            console.log(`  ${COLORS.RED}❌ ${test.name}: Validation failed${COLORS.RESET}`);
          }
        } else {
          console.log(`  ${COLORS.GREEN}✅ ${test.name}: PASS${COLORS.RESET}`);
          passedTests++;
        }
      } else {
        console.log(`  ${COLORS.RED}❌ ${test.name}: HTTP ${response.status}${COLORS.RESET}`);
      }
    } catch (error) {
      console.log(`  ${COLORS.RED}❌ ${test.name}: ${error.message}${COLORS.RESET}`);
    }
  }

  console.log(`\n${COLORS.BOLD}Test Results: ${passedTests}/${tests.length} passed${COLORS.RESET}`);
}

/**
 * Display comprehensive status
 */
async function displayStatus(status = null) {
  if (!status) {
    try {
      const response = await fetch(`http://localhost:${CONFIG.svelteKitPort}/api/gpu-orchestration/deploy`);
      status = response.ok ? await response.json() : null;
    } catch {
      status = null;
    }
  }

  console.log(`\n${COLORS.BOLD}📊 GPU Orchestration Status:${COLORS.RESET}`);
  console.log(`${COLORS.BLUE}════════════════════════════${COLORS.RESET}`);

  if (status) {
    const indicators = {
      orchestrator: status.orchestrator === 'running' ? '🟢' : '🔴',
      errorProcessor: status.errorProcessor === 'running' ? '🟢' : '🔴',
      flashAttention: status.flashAttention === 'enabled' ? '⚡' : '🔴',
      mcp: status.mcp === 'connected' ? '🔗' : '🔴',
      gemma3Legal: status.models?.gemma3Legal === 'available' ? '🧠' : '🔴',
      nomicEmbed: status.models?.nomicEmbed === 'available' ? '📊' : '🔴'
    };

    console.log(`${indicators.orchestrator} Orchestrator: ${status.orchestrator}`);
    console.log(`${indicators.errorProcessor} Error Processor: ${status.errorProcessor}`);
    console.log(`${indicators.flashAttention} FlashAttention: ${status.flashAttention}`);
    console.log(`${indicators.mcp} MCP Integration: ${status.mcp}`);
    console.log(`${indicators.gemma3Legal} Gemma3-Legal: ${status.models?.gemma3Legal || 'unknown'}`);
    console.log(`${indicators.nomicEmbed} Nomic-Embed: ${status.models?.nomicEmbed || 'unknown'}`);
    console.log(`🖥️ GPU: ${status.gpu?.device} (${status.gpu?.memory})`);
  } else {
    console.log(`${COLORS.YELLOW}⚠️ Status not available - services may not be running${COLORS.RESET}`);
  }

  console.log(`${COLORS.BLUE}════════════════════════════${COLORS.RESET}\n`);
}

// Run main function
main().catch(console.error);