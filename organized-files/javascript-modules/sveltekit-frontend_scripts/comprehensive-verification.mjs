#!/usr/bin/env node
// Comprehensive System Verification Script

import { spawn } from 'child_process';

console.log('🔍 COMPREHENSIVE SYSTEM VERIFICATION');
console.log('====================================\n');

// Colors for output
const colors = {
  green: (text) => `\x1b[32m${text}\x1b[0m`,
  red: (text) => `\x1b[31m${text}\x1b[0m`,
  yellow: (text) => `\x1b[33m${text}\x1b[0m`,
  blue: (text) => `\x1b[34m${text}\x1b[0m`,
  cyan: (text) => `\x1b[36m${text}\x1b[0m`
};

// Verification results
const results = {
  services: {},
  database: {},
  apis: {},
  processing: {}
};

// Helper function to check if port is in use
function checkPort(port) {
  return new Promise((resolve) => {
    const cmd = spawn('netstat', ['-an'], { shell: true });
    let output = '';
    
    cmd.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    cmd.on('close', () => {
      const isListening = output.includes(`:${port}`) && output.includes('LISTENING');
      resolve(isListening);
    });
    
    cmd.on('error', () => resolve(false));
  });
}

// Helper function to test HTTP endpoint
async function testEndpoint(url, method = 'GET') {
  try {
    const response = await fetch(url, { method, timeout: 5000 });
    return { success: response.ok, status: response.status, data: response.ok ? await response.json() : null };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// 1. Check Services
console.log(colors.blue('📋 CHECKING SERVICES'));
console.log('=====================');

const servicePorts = [
  { name: 'PostgreSQL', port: 5432 },
  { name: 'Redis', port: 6379 },
  { name: 'Ollama', port: 11434 },
  { name: 'Dev Server', port: 5173 }
];

for (const service of servicePorts) {
  const isRunning = await checkPort(service.port);
  results.services[service.name] = isRunning;
  
  const status = isRunning ? colors.green('✅ RUNNING') : colors.red('❌ NOT RUNNING');
  console.log(`${service.name.padEnd(15)} | Port ${service.port} | ${status}`);
}

console.log('\n' + colors.blue('📊 API HEALTH CHECKS'));
console.log('======================');

// 2. Check API Health
const apiEndpoints = [
  { name: 'OCR Health', url: 'http://localhost:5173/api/ocr/langextract' },
  { name: 'Embeddings Health', url: 'http://localhost:5173/api/embeddings/generate' },
  { name: 'Search Health', url: 'http://localhost:5173/api/documents/search' },
  { name: 'Storage Health', url: 'http://localhost:5173/api/documents/store' }
];

for (const api of apiEndpoints) {
  const result = await testEndpoint(api.url);
  results.apis[api.name] = result;
  
  const status = result.success ? colors.green('✅ HEALTHY') : colors.red('❌ UNHEALTHY');
  console.log(`${api.name.padEnd(20)} | ${status}`);
  
  if (result.success && result.data) {
    console.log(`${' '.repeat(23)}| Status: ${result.data.status || 'unknown'}`);
  } else if (!result.success) {
    console.log(`${' '.repeat(23)}| Error: ${result.error || 'HTTP ' + result.status}`);
  }
}

console.log('\n' + colors.blue('🧪 TESTING REAL PROCESSING'));
console.log('============================');

// 3. Test Real Processing
console.log('Testing real embedding generation...');
const embeddingTest = await testEndpoint('http://localhost:5173/api/embeddings/generate', 'POST');
if (embeddingTest.success) {
  try {
    const testResponse = await fetch('http://localhost:5173/api/embeddings/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: 'Test legal contract document',
        model: 'nomic-embed-text'
      })
    });
    
    if (testResponse.ok) {
      const data = await testResponse.json();
      if (data.success && data.embedding) {
        console.log(colors.green('✅ Real embedding generation working'));
        console.log(`   Dimensions: ${data.dimensions}`);
        console.log(`   Model: ${data.model}`);
        results.processing.embeddings = true;
      } else {
        console.log(colors.red('❌ Embedding generation failed'));
        results.processing.embeddings = false;
      }
    }
  } catch (error) {
    console.log(colors.red('❌ Embedding test failed: ' + error.message));
    results.processing.embeddings = false;
  }
} else {
  console.log(colors.red('❌ Embeddings API not accessible'));
  results.processing.embeddings = false;
}

// 4. Summary
console.log('\n' + colors.cyan('📈 VERIFICATION SUMMARY'));
console.log('========================');

const serviceCount = Object.values(results.services).filter(Boolean).length;
const apiCount = Object.values(results.apis).filter(r => r.success).length;
const totalServices = Object.keys(results.services).length;
const totalApis = Object.keys(results.apis).length;

console.log(`Services Running: ${serviceCount}/${totalServices}`);
console.log(`APIs Healthy: ${apiCount}/${totalApis}`);
console.log(`Real Processing: ${results.processing.embeddings ? 'Working' : 'Not Working'}`);

if (serviceCount === totalServices && apiCount === totalApis && results.processing.embeddings) {
  console.log('\n' + colors.green('🎉 SYSTEM FULLY OPERATIONAL!'));
  console.log(colors.green('✅ All services running'));
  console.log(colors.green('✅ All APIs healthy'));
  console.log(colors.green('✅ Real processing working'));
  console.log('\n' + colors.blue('🔗 Ready to use at: http://localhost:5173/ai-upload-demo'));
} else {
  console.log('\n' + colors.yellow('⚠️  System partially operational'));
  
  if (serviceCount < totalServices) {
    console.log(colors.red('❌ Some services not running'));
    Object.entries(results.services).forEach(([name, running]) => {
      if (!running) console.log(`   - ${name} needs to be started`);
    });
  }
  
  if (apiCount < totalApis) {
    console.log(colors.red('❌ Some APIs not responding'));
  }
  
  if (!results.processing.embeddings) {
    console.log(colors.red('❌ Real processing not working'));
  }
  
  console.log('\n' + colors.yellow('💡 Next steps:'));
  console.log('1. Start missing services');
  console.log('2. Run: npm run dev');
  console.log('3. Check: START-REAL-SYSTEM.bat');
}

console.log('\n' + colors.cyan('📊 Detailed Results:'));
console.log(JSON.stringify(results, null, 2));
