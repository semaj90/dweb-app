#!/usr/bin/env node
// Comprehensive System Verification Script - Improved Version

import { exec } from 'child_process';
import { promisify } from 'util';
import net from 'net';

const execAsync = promisify(exec);

console.log('🔍 COMPREHENSIVE SYSTEM VERIFICATION');
console.log('=====================================\n');

// Colors for output
const colors = {
  green: (text) => `\x1b[32m${text}\x1b[0m`,
  red: (text) => `\x1b[31m${text}\x1b[0m`,
  yellow: (text) => `\x1b[33m${text}\x1b[0m`,
  blue: (text) => `\x1b[34m${text}\x1b[0m`,
  cyan: (text) => `\x1b[36m${text}\x1b[0m`,
  magenta: (text) => `\x1b[35m${text}\x1b[0m`
};

// Verification results
const results = {
  timestamp: new Date().toISOString(),
  services: {},
  database: {},
  apis: {},
  processing: {},
  recommendations: []
};

// Helper function to check if port is in use
function checkPort(port, host = 'localhost') {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let isOpen = false;
    
    socket.setTimeout(3000);
    
    socket.on('connect', () => {
      isOpen = true;
      socket.end();
    });
    
    socket.on('timeout', () => {
      socket.destroy();
    });
    
    socket.on('error', () => {
      // Port is not open
    });
    
    socket.on('close', () => {
      resolve(isOpen);
    });
    
    socket.connect(port, host);
  });
}

// Helper function to test HTTP endpoint
async function testEndpoint(url, options = {}) {
  const method = options.method || 'GET';
  const headers = options.headers || {};
  const body = options.body || null;
  
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    
    const fetchOptions = {
      method,
      headers: { 'Content-Type': 'application/json', ...headers },
      signal: controller.signal
    };
    
    if (body) {
      fetchOptions.body = JSON.stringify(body);
    }
    
    const response = await fetch(url, fetchOptions);
    clearTimeout(timeout);
    
    let data = null;
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      try {
        data = await response.json();
      } catch {
        data = await response.text();
      }
    } else {
      data = await response.text();
    }
    
    return { 
      success: response.ok, 
      status: response.status, 
      data,
      headers: Object.fromEntries(response.headers.entries())
    };
  } catch (error) {
    return { 
      success: false, 
      error: error.message,
      type: error.name
    };
  }
}

// Check Windows services
async function checkWindowsService(serviceName) {
  try {
    const { stdout } = await execAsync(`sc query "${serviceName}" 2>nul`);
    return stdout.includes('RUNNING');
  } catch {
    return false;
  }
}

// Check process by name
async function checkProcess(processName) {
  try {
    const { stdout } = await execAsync(`tasklist /FI "IMAGENAME eq ${processName}" 2>nul`);
    return stdout.toLowerCase().includes(processName.toLowerCase());
  } catch {
    return false;
  }
}

// Main verification process
async function runVerification() {
  
  // 1. Check Core Services
  console.log(colors.blue('📋 CHECKING CORE SERVICES'));
  console.log('==========================');
  
  const services = [
    { name: 'PostgreSQL', port: 5432, serviceName: 'postgresql-x64-15', processName: 'postgres.exe' },
    { name: 'Redis', port: 6379, serviceName: 'Redis', processName: 'redis-server.exe' },
    { name: 'Ollama', port: 11434, processName: 'ollama.exe' },
    { name: 'Dev Server', port: 5173 }
  ];
  
  for (const service of services) {
    const portOpen = await checkPort(service.port);
    let serviceRunning = false;
    let processRunning = false;
    
    if (service.serviceName) {
      serviceRunning = await checkWindowsService(service.serviceName);
    }
    
    if (service.processName) {
      processRunning = await checkProcess(service.processName);
    }
    
    const isRunning = portOpen || serviceRunning || processRunning;
    results.services[service.name] = {
      running: isRunning,
      port: service.port,
      portOpen,
      serviceRunning,
      processRunning
    };
    
    const status = isRunning ? colors.green('✅ RUNNING') : colors.red('❌ NOT RUNNING');
    const portStatus = portOpen ? colors.green('✓') : colors.red('✗');
    
    console.log(`${service.name.padEnd(15)} | Port ${String(service.port).padEnd(5)} [${portStatus}] | ${status}`);
    
    if (!isRunning) {
      results.recommendations.push(`Start ${service.name} service`);
    }
  }
  
  // 2. Database Connection Test
  console.log('\n' + colors.blue('🗄️  DATABASE CONNECTION'));
  console.log('========================');
  
  if (results.services['PostgreSQL']?.portOpen) {
    console.log(colors.green('✅ PostgreSQL port is open'));
    results.database.accessible = true;
    
    // Test actual connection
    try {
      const { stdout } = await execAsync('psql -U postgres -h localhost -p 5432 -c "SELECT version();" 2>nul');
      if (stdout) {
        console.log(colors.green('✅ Database connection successful'));
        results.database.connectable = true;
      }
    } catch {
      console.log(colors.yellow('⚠️  Could not verify database connection'));
      results.database.connectable = false;
    }
  } else {
    console.log(colors.red('❌ PostgreSQL not accessible'));
    results.database.accessible = false;
  }
  
  // 3. API Health Checks
  console.log('\n' + colors.blue('🌐 API HEALTH CHECKS'));
  console.log('======================');
  
  const apiEndpoints = [
    { 
      name: 'Dev Server Root', 
      url: 'http://localhost:5173/',
      critical: true
    },
    { 
      name: 'OCR Service', 
      url: 'http://localhost:5173/api/ocr/langextract',
      method: 'GET'
    },
    { 
      name: 'Embeddings Service', 
      url: 'http://localhost:5173/api/embeddings/generate',
      method: 'GET'
    },
    { 
      name: 'Search Service', 
      url: 'http://localhost:5173/api/documents/search',
      method: 'GET'
    },
    { 
      name: 'Storage Service', 
      url: 'http://localhost:5173/api/documents/store',
      method: 'GET'
    },
    {
      name: 'AI Upload Demo',
      url: 'http://localhost:5173/ai-upload-demo',
      method: 'GET'
    }
  ];
  
  for (const api of apiEndpoints) {
    const result = await testEndpoint(api.url, { method: api.method });
    results.apis[api.name] = result;
    
    const status = result.success ? 
      colors.green('✅ HEALTHY') : 
      result.error?.includes('ECONNREFUSED') ? 
        colors.red('❌ UNREACHABLE') : 
        colors.yellow(`⚠️  HTTP ${result.status || 'ERROR'}`);
    
    console.log(`${api.name.padEnd(20)} | ${status}`);
    
    if (result.error) {
      console.log(`${' '.repeat(23)}| ${colors.red(result.error)}`);
      if (api.critical) {
        results.recommendations.push('Start the development server with: npm run dev');
      }
    }
  }
  
  // 4. Ollama Model Check
  console.log('\n' + colors.blue('🤖 AI MODEL STATUS'));
  console.log('====================');
  
  if (results.services['Ollama']?.portOpen) {
    const ollamaTest = await testEndpoint('http://localhost:11434/api/tags');
    
    if (ollamaTest.success) {
      console.log(colors.green('✅ Ollama API accessible'));
      
      if (ollamaTest.data?.models) {
        const models = ollamaTest.data.models;
        console.log(`Found ${models.length} model(s):`);
        
        const requiredModels = ['nomic-embed-text', 'llama3.2'];
        const installedModels = models.map(m => m.name);
        
        for (const model of requiredModels) {
          const hasModel = installedModels.some(m => m.includes(model));
          if (hasModel) {
            console.log(`  ${colors.green('✓')} ${model}`);
          } else {
            console.log(`  ${colors.red('✗')} ${model} - ${colors.yellow('NOT INSTALLED')}`);
            results.recommendations.push(`Install model: ollama pull ${model}`);
          }
        }
        
        results.processing.ollamaModels = installedModels;
      }
    } else {
      console.log(colors.red('❌ Ollama API not responding'));
    }
  } else {
    console.log(colors.red('❌ Ollama not running'));
    results.recommendations.push('Start Ollama service');
  }
  
  // 5. Test Real Processing (if dev server is running)
  if (results.apis['Dev Server Root']?.success) {
    console.log('\n' + colors.blue('🧪 TESTING REAL PROCESSING'));
    console.log('============================');
    
    // Test embedding generation
    const embeddingTest = await testEndpoint(
      'http://localhost:5173/api/embeddings/generate',
      {
        method: 'POST',
        body: {
          text: 'Test legal contract document verification',
          model: 'nomic-embed-text'
        }
      }
    );
    
    if (embeddingTest.success && embeddingTest.data?.success) {
      console.log(colors.green('✅ Real embedding generation working'));
      console.log(`   Model: ${embeddingTest.data.model || 'nomic-embed-text'}`);
      console.log(`   Dimensions: ${embeddingTest.data.dimensions || 'unknown'}`);
      results.processing.embeddings = true;
    } else {
      console.log(colors.red('❌ Embedding generation failed'));
      results.processing.embeddings = false;
      
      if (embeddingTest.data?.error) {
        console.log(`   Error: ${embeddingTest.data.error}`);
      }
    }
  }
  
  // 6. System Summary
  console.log('\n' + colors.cyan('📊 VERIFICATION SUMMARY'));
  console.log('=========================');
  
  const serviceCount = Object.values(results.services).filter(s => s.running).length;
  const apiCount = Object.values(results.apis).filter(r => r.success).length;
  const totalServices = Object.keys(results.services).length;
  const totalApis = Object.keys(results.apis).length;
  
  console.log(`Services Running: ${serviceCount}/${totalServices}`);
  console.log(`APIs Healthy: ${apiCount}/${totalApis}`);
  console.log(`Database: ${results.database.accessible ? 'Accessible' : 'Not Accessible'}`);
  console.log(`Processing: ${results.processing.embeddings ? 'Working' : 'Not Working'}`);
  
  // Overall Status
  const allGood = serviceCount === totalServices && 
                  apiCount === totalApis && 
                  results.database.accessible &&
                  results.processing.embeddings;
  
  if (allGood) {
    console.log('\n' + colors.green('🎉 SYSTEM FULLY OPERATIONAL!'));
    console.log(colors.green('✅ All services running'));
    console.log(colors.green('✅ All APIs healthy'));
    console.log(colors.green('✅ Database accessible'));
    console.log(colors.green('✅ Real processing working'));
    console.log('\n' + colors.magenta('🔗 Ready to use at: http://localhost:5173/ai-upload-demo'));
  } else {
    console.log('\n' + colors.yellow('⚠️  SYSTEM PARTIALLY OPERATIONAL'));
    
    // Recommendations
    if (results.recommendations.length > 0) {
      console.log('\n' + colors.yellow('💡 RECOMMENDED ACTIONS:'));
      results.recommendations.forEach((rec, i) => {
        console.log(`${i + 1}. ${rec}`);
      });
    }
    
    console.log('\n' + colors.blue('🔧 QUICK START COMMANDS:'));
    console.log('1. Start all services: .\\START-REAL-SYSTEM.bat');
    console.log('2. Start dev server: npm run dev');
    console.log('3. Check Ollama: ollama list');
    console.log('4. Test database: psql -U postgres -c "\\l"');
  }
  
  // Save results to file
  console.log('\n' + colors.cyan('💾 Saving detailed results...'));
  const fs = await import('fs').then(m => m.promises);
  const resultsPath = 'verification-results.json';
  await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));
  console.log(`Results saved to: ${resultsPath}`);
}

// Run the verification
runVerification().catch(error => {
  console.error(colors.red('\n❌ Verification script error:'), error.message);
  process.exit(1);
});
