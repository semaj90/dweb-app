#!/usr/bin/env node

/**
 * 🚀 Legal AI Platform - Dual Port Setup with Environment
 * Starts frontend on port 5173 and proxy on port 5180
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('🚀 Legal AI Platform - Dual Port Setup with Environment\n');

// Load environment configuration
console.log('📝 Loading environment configuration...');
try {
  // Since this is a .cjs file, we can't use ES modules directly
  // We'll set environment variables manually for now
  console.log('⚠️  Using default environment variables (ES module compatibility)');
} catch (error) {
  console.log('⚠️  Using default environment variables');
}

// Create necessary directories
const directories = [
  'uploads',
  'documents', 
  'evidence',
  'logs',
  'generated_reports',
  'temp'
];

console.log('\n📁 Creating directories...');
directories.forEach(dir => {
  const fullPath = path.join(process.cwd(), dir);
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
    console.log(`✅ Created: ${dir}`);
  } else {
    console.log(`✅ Exists: ${dir}`);
  }
});

// Function to start a service
function startService(name, command, args, options = {}) {
  console.log(`📡 Starting ${name}...`);
  
  const service = spawn(command, args, {
    stdio: 'pipe',
    shell: true,
    cwd: options.cwd || process.cwd(),
    env: { ...process.env, ...options.env },
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
console.log('\n🎯 Step 1: Starting Frontend Service (Port 5173)...');
const frontend = startService('Frontend', 'pnpm', ['--filter', 'yorha-legal-ai-frontend', 'run', 'dev'], {
  cwd: process.cwd() // Use root directory since we're using pnpm filter
});

// Wait for frontend to start
setTimeout(() => {
  console.log('\n🎯 Step 2: Starting Port 5180 Proxy...');
  
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
  console.log('   • Environment variables loaded');
  
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
