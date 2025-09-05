#!/usr/bin/env node
/**
 * Quick Evidence System Test
 * Tests core functionality with the test PDF
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = 'http://localhost:5173';
const TEST_PDF = path.join(__dirname, '../../lawpdfs/test-document.pdf');

function log(message, color = 'reset') {
  const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m'
  };
  console.log(`${colors[color]}${message}${colors.reset}`);
}

async function testServerHealth() {
  log('\n🔍 Testing Server Health...', 'cyan');
  
  try {
    const response = await fetch(BASE_URL);
    if (response.ok) {
      log('✅ SvelteKit server is running', 'green');
      return true;
    } else {
      log(`❌ Server responded with status: ${response.status}`, 'red');
      return false;
    }
  } catch (error) {
    log(`❌ Server connection failed: ${error.message}`, 'red');
    return false;
  }
}

async function testFileExists() {
  log('\n📄 Checking Test Files...', 'cyan');
  
  if (fs.existsSync(TEST_PDF)) {
    const stats = fs.statSync(TEST_PDF);
    log(`✅ Test PDF found: ${TEST_PDF} (${stats.size} bytes)`, 'green');
    return true;
  } else {
    log(`❌ Test PDF not found: ${TEST_PDF}`, 'red');
    return false;
  }
}

async function testBasicAPI() {
  log('\n🌐 Testing Basic API Routes...', 'cyan');
  
  const routes = [
    '/',
    '/api/health',
    '/evidence',
    '/cases'
  ];
  
  let passed = 0;
  for (const route of routes) {
    try {
      const response = await fetch(`${BASE_URL}${route}`);
      if (response.status !== 404) {
        log(`✅ ${route} - Status: ${response.status}`, 'green');
        passed++;
      } else {
        log(`❌ ${route} - Not found (404)`, 'red');
      }
    } catch (error) {
      log(`❌ ${route} - Error: ${error.message}`, 'red');
    }
  }
  
  return passed > 0;
}

async function testFileUpload() {
  log('\n📤 Testing File Upload...', 'cyan');
  
  if (!fs.existsSync(TEST_PDF)) {
    log('❌ Test PDF not found for upload test', 'red');
    return false;
  }
  
  try {
    const fileBuffer = fs.readFileSync(TEST_PDF);
    const formData = new FormData();
    const file = new File([fileBuffer], 'test-document.pdf', { type: 'application/pdf' });
    formData.append('files', file);
    formData.append('caseId', 'TEST-001');
    
    // Try multiple upload endpoints
    const endpoints = ['/api/upload', '/api/evidence/upload', '/api/ai/lawpdfs'];
    
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(`${BASE_URL}${endpoint}`, {
          method: 'POST',
          body: formData
        });
        
        log(`   Testing ${endpoint} - Status: ${response.status}`, response.ok ? 'green' : 'yellow');
        
        if (response.ok) {
          const result = await response.json();
          log(`✅ File upload successful at ${endpoint}`, 'green');
          log(`   Result: ${JSON.stringify(result, null, 2)}`, 'blue');
          return true;
        }
      } catch (error) {
        log(`   ${endpoint} - Error: ${error.message}`, 'yellow');
      }
    }
    
    log('⚠️  No upload endpoints are currently working', 'yellow');
    return false;
    
  } catch (error) {
    log(`❌ File upload test failed: ${error.message}`, 'red');
    return false;
  }
}

async function main() {
  log('🚀 Quick Evidence System Test', 'blue');
  log('=' .repeat(50), 'blue');
  
  const results = {
    serverHealth: await testServerHealth(),
    fileExists: await testFileExists(),
    basicAPI: await testBasicAPI(),
    fileUpload: await testFileUpload()
  };
  
  log('\n📊 Test Results:', 'cyan');
  log('=' .repeat(50), 'cyan');
  
  Object.entries(results).forEach(([test, passed]) => {
    const status = passed ? '✅ PASS' : '❌ FAIL';
    const color = passed ? 'green' : 'red';
    log(`${status} ${test}`, color);
  });
  
  const totalPassed = Object.values(results).filter(Boolean).length;
  const totalTests = Object.keys(results).length;
  
  log(`\nOverall: ${totalPassed}/${totalTests} tests passed`, totalPassed === totalTests ? 'green' : 'yellow');
  
  if (results.serverHealth && results.fileExists) {
    log('\n🎯 System is ready for testing!', 'green');
    log('Next steps:', 'blue');
    log('1. Open browser: http://localhost:5173', 'blue');
    log('2. Test evidence upload manually', 'blue');
    log('3. Check API endpoints work', 'blue');
  } else {
    log('\n⚠️  System needs setup:', 'yellow');
    if (!results.serverHealth) log('- Start SvelteKit dev server: npm run dev', 'yellow');
    if (!results.fileExists) log('- Test PDF created at: ' + TEST_PDF, 'yellow');
  }
  
  process.exit(totalPassed === totalTests ? 0 : 1);
}

main().catch(error => {
  log(`❌ Fatal error: ${error.message}`, 'red');
  process.exit(1);
});