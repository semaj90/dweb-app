#!/usr/bin/env node

/**
 * Start All Services Script
 * Launches frontend on port 5173 and proxy on port 5180
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Legal AI Platform - Starting All Services...\n');

// Function to start a service
function startService(name, command, args, options = {}) {
  console.log(`📡 Starting ${name}...`);
  
  const service = spawn(command, args, {
    stdio: 'pipe',
    shell: true,
    cwd: options.cwd || process.cwd(),
    ...options
  });
  
  service.stdout.on('data', (data) => {
    console.log(`[${name}] ${data.toString().trim()}`);
  });
  
  service.stderr.on('data', (data) => {
    console.error(`[${name}] ERROR: ${data.toString().trim()}`);
  });
  
  service.on('close', (code) => {
    console.log(`[${name}] Process exited with code ${code}`);
  });
  
  return service;
}

// Start frontend service
console.log('🎯 Starting Frontend Service (Port 5173)...');
const frontend = startService('Frontend', 'npm', ['run', 'dev'], {
  cwd: path.join(process.cwd(), 'sveltekit-frontend')
});

// Wait for frontend to start
setTimeout(() => {
  console.log('\n🎯 Starting Port 5180 Proxy...');
  
  // Start proxy service
  const proxy = startService('Proxy', 'node', ['start-port-5180.cjs'], {
    cwd: process.cwd()
  });
  
  console.log('\n✅ All services started!');
  console.log('🌐 Frontend: http://localhost:5173');
  console.log('🌐 Port 5180: http://localhost:5180');
  console.log('\n📋 Service Status:');
  console.log('   • Frontend: Starting on port 5173');
  console.log('   • Proxy: Starting on port 5180');
  console.log('   • Both ports will serve the same content');
  
}, 15000); // Wait 15 seconds for frontend to start

// Handle process termination
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down all services...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down all services...');
  process.exit(0);
});
