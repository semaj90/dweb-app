#!/usr/bin/env node
/**
 * Enhanced File Upload Integration Test
 * Tests XState + WebSocket + MCP integration
 */

import fs from 'fs';
import fetch from 'node-fetch';

console.log('🧪 Enhanced File Upload Integration Test');
console.log('========================================\n');

// Test configuration
const config = {
  uploadService: 'http://localhost:8093',
  ragService: 'http://localhost:8094',
  wsEndpoint: 'ws://localhost:8094/ws',
  testUserId: 'test-user-123',
  testCaseId: 'test-case-456'
};

// Test results
const results = {
  servicesHealthy: false,
  mcpEndpointsAccessible: false,
  xstateTypingWorking: false,
  websocketConnectable: false,
  overallStatus: 'UNKNOWN'
};

async function testServiceHealth() {
  console.log('📡 Testing service health...');
  
  try {
    // Test upload service
    const uploadHealth = await fetch(`${config.uploadService}/health`);
    const uploadData = await uploadHealth.json();
    console.log(`  ✅ Upload Service (8093): ${uploadData.status}`);
    
    // Test RAG service
    const ragHealth = await fetch(`${config.ragService}/health`);
    const ragData = await ragHealth.json();
    console.log(`  ✅ RAG Service (8094): ${ragData.status}`);
    
    results.servicesHealthy = true;
    console.log('  🎉 All services healthy!\n');
  } catch (error) {
    console.log(`  ❌ Service health check failed: ${error.message}\n`);
  }
}

async function testMCPEndpoints() {
  console.log('🔗 Testing MCP endpoint accessibility...');
  
  const endpoints = [
    { service: 'upload', url: `${config.uploadService}/upload`, method: 'POST' },
    { service: 'extract-text', url: `${config.uploadService}/extract-text`, method: 'POST' },
    { service: 'process', url: `${config.ragService}/process`, method: 'POST' }
  ];
  
  let accessible = 0;
  
  for (const endpoint of endpoints) {
    try {
      // Just test if endpoint exists (expect 400 for missing data, not 404)
      const response = await fetch(endpoint.url, { 
        method: endpoint.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      
      if (response.status !== 404) {
        console.log(`  ✅ ${endpoint.service}: accessible`);
        accessible++;
      } else {
        console.log(`  ❌ ${endpoint.service}: not found`);
      }
    } catch (error) {
      console.log(`  ⚠️ ${endpoint.service}: ${error.message}`);
    }
  }
  
  results.mcpEndpointsAccessible = accessible > 0;
  console.log(`  📊 ${accessible}/${endpoints.length} endpoints accessible\n`);
}

async function testXStateTyping() {
  console.log('🎭 Testing XState typing helpers...');
  
  try {
    // Check if documentUploadMachine file exists and has proper exports
    const machinePath = './src/lib/state/documentUploadMachine.ts';
    if (fs.existsSync(machinePath)) {
      const content = fs.readFileSync(machinePath, 'utf8');
      
      const requiredExports = [
        'DocumentUploadStateValue',
        'getStateValue',
        'isInErrorState',
        'isInProcessingState',
        'documentUploadMachine'
      ];
      
      let foundExports = 0;
      for (const exportName of requiredExports) {
        if (content.includes(exportName)) {
          foundExports++;
          console.log(`  ✅ ${exportName}: found`);
        } else {
          console.log(`  ❌ ${exportName}: missing`);
        }
      }
      
      results.xstateTypingWorking = foundExports === requiredExports.length;
      console.log(`  📊 ${foundExports}/${requiredExports.length} exports found\n`);
    } else {
      console.log('  ❌ documentUploadMachine.ts not found\n');
    }
  } catch (error) {
    console.log(`  ❌ XState typing test failed: ${error.message}\n`);
  }
}

function testWebSocketConnectivity() {
  console.log('🔌 Testing WebSocket connectivity...');
  
  return new Promise((resolve) => {
    try {
      // Note: We can't test actual WebSocket in Node.js easily without ws package
      // But we can check if the HTTP server responds to WebSocket upgrade requests
      const ws = new URL(config.wsEndpoint);
      
      fetch(`http://${ws.host}/ws`, {
        headers: {
          'Connection': 'Upgrade',
          'Upgrade': 'websocket'
        }
      }).then(response => {
        if (response.status === 400 || response.status === 426) {
          // Expected response for WebSocket upgrade attempt via HTTP
          console.log('  ✅ WebSocket endpoint responding to upgrade requests');
          results.websocketConnectable = true;
        } else {
          console.log(`  ⚠️ Unexpected response: ${response.status}`);
        }
        console.log('');
        resolve();
      }).catch(error => {
        console.log(`  ❌ WebSocket test failed: ${error.message}\n`);
        resolve();
      });
    } catch (error) {
      console.log(`  ❌ WebSocket connectivity test failed: ${error.message}\n`);
      resolve();
    }
  });
}

function calculateOverallStatus() {
  console.log('📋 Integration Test Results');
  console.log('============================');
  
  const checks = [
    { name: 'Services Healthy', status: results.servicesHealthy },
    { name: 'MCP Endpoints', status: results.mcpEndpointsAccessible },
    { name: 'XState Typing', status: results.xstateTypingWorking },
    { name: 'WebSocket Ready', status: results.websocketConnectable }
  ];
  
  let passedChecks = 0;
  
  for (const check of checks) {
    const icon = check.status ? '✅' : '❌';
    console.log(`${icon} ${check.name}: ${check.status ? 'PASS' : 'FAIL'}`);
    if (check.status) passedChecks++;
  }
  
  console.log(`\n📊 Overall Score: ${passedChecks}/${checks.length}`);
  
  if (passedChecks === checks.length) {
    results.overallStatus = 'EXCELLENT';
    console.log('🎉 EXCELLENT: All systems operational!');
  } else if (passedChecks >= 3) {
    results.overallStatus = 'GOOD';
    console.log('✨ GOOD: Most systems working, minor issues to resolve');
  } else if (passedChecks >= 2) {
    results.overallStatus = 'FAIR';
    console.log('⚠️ FAIR: Some systems working, significant issues to resolve');
  } else {
    results.overallStatus = 'POOR';
    console.log('❌ POOR: Major integration issues detected');
  }
  
  console.log('\n🚀 Next Steps:');
  if (!results.servicesHealthy) {
    console.log('  - Start services: npm run dev:services');
  }
  if (!results.mcpEndpointsAccessible) {
    console.log('  - Verify MCP endpoint configuration');
  }
  if (!results.xstateTypingWorking) {
    console.log('  - Check XState machine imports and exports');
  }
  if (!results.websocketConnectable) {
    console.log('  - Verify WebSocket server configuration');
  }
  
  if (results.overallStatus === 'EXCELLENT') {
    console.log('  🎯 Ready for production testing!');
  }
}

// Run all tests
async function runTests() {
  await testServiceHealth();
  await testMCPEndpoints();
  await testXStateTyping();
  await testWebSocketConnectivity();
  calculateOverallStatus();
}

runTests().catch(console.error);