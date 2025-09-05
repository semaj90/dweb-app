#!/usr/bin/env node

/**
 * Enhanced RAG Integration Setup Script (Fixed)
 * Sets up and validates the enhanced MCP integration system
 */

import { spawn } from "child_process";
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const workspaceRoot = process.cwd();

/**
 * Enhanced RAG System Integration Setup
 */
async function setupEnhancedRAGSystem() {
  console.log("🚀 Starting Enhanced RAG System Integration Setup...\n");

  try {
    // Step 1: Validate environment
    console.log("🔍 Step 1: Validating environment and dependencies...");
    await validateEnvironment();
    
    // Step 2: Check service availability  
    console.log("\n🔍 Step 2: Checking service availability...");
    await checkServiceAvailability();
    
    // Step 3: Initialize library sync service
    console.log("\n📚 Step 3: Initializing Library Sync Service...");
    await initializeLibrarySyncService();
    
    // Step 4: Test enhanced MCP integration
    console.log("\n🧪 Step 4: Testing Enhanced MCP Integration...");
    await testEnhancedMCPIntegration();
    
    // Step 5: Validate cluster system
    console.log("\n⚡ Step 5: Validating cluster system...");
    await validateClusterSystem();
    
    // Step 6: Setup frontend integration
    console.log("\n🎨 Step 6: Setting up frontend integration...");
    await setupFrontendIntegration();
    
    console.log("\n🎉 Enhanced RAG System Integration Complete!");
    console.log("\n📋 Next Steps:");
    console.log("1. Run: powershell .\\setup-enhanced-rag-fixed.ps1");
    console.log("2. Start services: .\\start-enhanced-rag.bat");
    console.log("3. Test system: powershell .\\test-enhanced-rag.ps1");
    console.log("4. Access demo: http://localhost:5173/ai/enhanced-mcp");
    
  } catch (error) {
    console.error("💥 Integration setup failed:", error);
    process.exit(1);
  }
}

async function validateEnvironment() {
  const requiredFiles = [
    'context7-mcp-server.js',
    'test-cluster-simple.js',
    'sveltekit-frontend/src/lib/components/ai/EnhancedMCPIntegration.svelte',
    'sveltekit-frontend/src/routes/ai/enhanced-mcp/+page.svelte',
    'src/lib/services/library-sync-service.js'
  ];
  
  const requiredDirectories = [
    'sveltekit-frontend/src/lib/components/ai',
    'sveltekit-frontend/src/routes/ai/enhanced-mcp',
    '.vscode/extensions/mcp-context7-assistant/src'
  ];
  
  // Check files
  for (const file of requiredFiles) {
    try {
      await fs.access(file);
      console.log(`✅ Found: ${file}`);
    } catch (error) {
      console.log(`❌ Missing: ${file}`);
      throw new Error(`Required file missing: ${file}`);
    }
  }
  
  // Check directories
  for (const dir of requiredDirectories) {
    try {
      await fs.access(dir);
      console.log(`✅ Directory exists: ${dir}`);
    } catch (error) {
      console.log(`❌ Missing directory: ${dir}`);
      throw new Error(`Required directory missing: ${dir}`);
    }
  }
  
  console.log("✅ Environment validation passed");
}

async function checkServiceAvailability() {
  const services = [
    { name: 'Context7 MCP Server', url: 'http://localhost:40000/health' },
    { name: 'Ollama Service', url: 'http://localhost:11434/api/tags' }
  ];
  
  for (const service of services) {
    try {
      const response = await fetchWithTimeout(service.url, { 
        method: 'GET'
      }, 5000);
      
      if (response.ok) {
        console.log(`✅ ${service.name} is running`);
      } else {
        console.log(`⚠️ ${service.name} returned status: ${response.status}`);  
      }
    } catch (error) {
      console.log(`❌ ${service.name} is not available (${error.message})`);
      console.log(`   This service will need to be started manually`);
    }
  }
}

async function initializeLibrarySyncService() {
  try {
    // Dynamic import to handle the service
    const { LibrarySyncService } = await import('./src/lib/services/library-sync-service.js');
    
    const syncService = new LibrarySyncService({
      baseUrl: 'http://localhost:40000',
      timeout: 10000,
      retries: 2
    });
    
    const initialized = await syncService.initialize();
    if (initialized) {
      console.log("✅ Library Sync Service initialized successfully");
      
      // Get status and cleanup
      const status = await syncService.getStatus();
      console.log(`   Status: Connected=${status.isConnected}, URL=${status.baseUrl}`);
      
      syncService.dispose();
    } else {
      console.log("⚠️ Library Sync Service initialization failed (service may be offline)");
    }
    
  } catch (error) {
    console.log("⚠️ Could not test Library Sync Service:", error.message);
  }
}

async function testEnhancedMCPIntegration() {
  const testEndpoints = [
    { 
      name: 'Enhanced RAG Query',
      url: 'http://localhost:40000/mcp/enhanced-rag/query',
      method: 'POST',
      body: { query: 'integration test', caseId: 'test-case', maxResults: 1 }
    },
    {
      name: 'Memory Graph',
      url: 'http://localhost:40000/mcp/memory/read-graph', 
      method: 'POST',
      body: {}
    },
    {
      name: 'Context7 Library Resolution',
      url: 'http://localhost:40000/mcp/context7/resolve-library-id',
      method: 'POST', 
      body: { libraryName: 'sveltekit' }
    }
  ];
  
  let passedTests = 0;
  for (const test of testEndpoints) {
    try {
      const response = await fetchWithTimeout(test.url, {
        method: test.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(test.body)
      }, 5000);
      
      if (response.ok) {
        console.log(`✅ ${test.name} - PASSED`);
        passedTests++;
      } else {
        console.log(`❌ ${test.name} - HTTP ${response.status}`);
      }
    } catch (error) {
      console.log(`❌ ${test.name} - ERROR: ${error.message}`);
    }
  }
  
  console.log(`📊 MCP Integration Tests: ${passedTests}/${testEndpoints.length} passed`);
  
  if (passedTests === 0) {
    console.log("⚠️ No MCP endpoints available - start Context7 MCP server first");
  }
}

async function validateClusterSystem() {
  try {
    // Check if cluster performance results exist
    const clusterResultPath = 'cluster-performance-simple.json';
    
    try {
      await fs.access(clusterResultPath);
      const resultData = await fs.readFile(clusterResultPath, 'utf8');
      const results = JSON.parse(resultData);
      
      if (results.status === 'working') {
        console.log(`✅ Cluster system validated - ${results.results.successfulRequests} successful requests`);
        console.log(`   Success rate: ${(results.results.successfulRequests / results.results.totalRequests * 100).toFixed(1)}%`);
      } else {
        console.log("⚠️ Cluster system status unclear");
      }
    } catch (error) {
      console.log("⚠️ Cluster performance results not found");
      console.log("   Run: node test-cluster-simple.js");
    }
    
  } catch (error) {
    console.log("❌ Cluster validation failed:", error.message);
  }
}

async function setupFrontendIntegration() {
  try {
    // Check if SvelteKit development server is running
    try {
      const response = await fetchWithTimeout('http://localhost:5173', {}, 2000);
      console.log("✅ SvelteKit development server is running");
      console.log("   Enhanced MCP Demo: http://localhost:5173/ai/enhanced-mcp");
    } catch (error) {
      console.log("❌ SvelteKit development server is not running");
      console.log("   Start with: cd sveltekit-frontend; npm run dev");
    }
    
    // Verify integration components exist
    const integrationFiles = [
      'sveltekit-frontend/src/lib/components/ai/EnhancedMCPIntegration.svelte',
      'sveltekit-frontend/src/routes/ai/enhanced-mcp/+page.svelte',
      'sveltekit-frontend/src/routes/api/mcp/status/+server.ts'
    ];
    
    for (const file of integrationFiles) {
      try {
        await fs.access(file);
        console.log(`✅ Integration component: ${path.basename(file)}`);
      } catch (error) {
        console.log(`❌ Missing integration component: ${file}`);
      }
    }
    
  } catch (error) {
    console.log("❌ Frontend integration setup failed:", error.message);
  }
}

// Helper function for fetch with timeout (Node.js compatibility)
async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const { default: nodeFetch } = await import('node-fetch');
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await nodeFetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// Run the setup
setupEnhancedRAGSystem().catch(console.error);