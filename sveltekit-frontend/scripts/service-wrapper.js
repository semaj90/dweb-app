/**
 * Windows Service Wrapper for Legal AI Platform
 * Wraps Node.js applications to run as Windows Services using node-windows
 */

const Service = require('node-windows').Service;
const path = require('path');
const fs = require('fs');

// Service configuration
const SERVICE_CONFIG = {
  'sveltekit': {
    name: 'LegalAI-SvelteKit',
    description: 'Legal AI Platform - SvelteKit Frontend Server',
    script: path.join(__dirname, '..', 'build', 'index.js'),
    nodeOptions: ['--max-old-space-size=4096'],
    env: [
      {
        name: 'NODE_ENV',
        value: 'production'
      },
      {
        name: 'PORT',
        value: '5173'
      },
      {
        name: 'ORIGIN',
        value: 'http://localhost:5173'
      }
    ],
    logpath: path.join(__dirname, '..', '..', 'logs'),
    logmode: 'rotate',
    logOnAs: {
      domain: '',
      account: '',
      password: ''
    },
    abortOnError: 5,
    maxRestarts: 10,
    maxRetries: 3,
    wait: 2,
    grow: 0.5
  }
};

function createLogDirectory(logPath) {
  if (!fs.existsSync(logPath)) {
    fs.mkdirSync(logPath, { recursive: true });
    console.log(`✅ Created log directory: ${logPath}`);
  }
}

function installService(serviceName) {
  const config = SERVICE_CONFIG[serviceName];
  if (!config) {
    console.error(`❌ Unknown service: ${serviceName}`);
    process.exit(1);
  }

  // Ensure script file exists
  if (!fs.existsSync(config.script)) {
    console.error(`❌ Script file not found: ${config.script}`);
    console.error('Please run "npm run build" first to create the build directory.');
    process.exit(1);
  }

  // Create log directory
  createLogDirectory(config.logpath);

  // Create service
  const svc = new Service({
    name: config.name,
    description: config.description,
    script: config.script,
    nodeOptions: config.nodeOptions,
    env: config.env,
    logpath: config.logpath,
    logmode: config.logmode,
    abortOnError: config.abortOnError,
    maxRestarts: config.maxRestarts,
    maxRetries: config.maxRetries,
    wait: config.wait,
    grow: config.grow
  });

  // Listen for install event
  svc.on('install', () => {
    console.log(`✅ Service ${config.name} installed successfully`);
    console.log(`   Description: ${config.description}`);
    console.log(`   Script: ${config.script}`);
    console.log(`   Logs: ${config.logpath}`);
    
    // Optionally start the service after installation
    if (process.argv.includes('--start')) {
      console.log('▶️ Starting service...');
      svc.start();
    }
  });

  // Listen for start event
  svc.on('start', () => {
    console.log(`✅ Service ${config.name} started successfully`);
  });

  // Listen for errors
  svc.on('error', (err) => {
    console.error(`❌ Service error: ${err}`);
  });

  // Install the service
  console.log(`🔧 Installing service: ${config.name}`);
  svc.install();
}

function uninstallService(serviceName) {
  const config = SERVICE_CONFIG[serviceName];
  if (!config) {
    console.error(`❌ Unknown service: ${serviceName}`);
    process.exit(1);
  }

  // Create service object
  const svc = new Service({
    name: config.name,
    script: config.script
  });

  // Listen for uninstall event
  svc.on('uninstall', () => {
    console.log(`✅ Service ${config.name} uninstalled successfully`);
  });

  // Listen for errors
  svc.on('error', (err) => {
    console.error(`❌ Service error: ${err}`);
  });

  // Uninstall the service
  console.log(`🗑️ Uninstalling service: ${config.name}`);
  svc.uninstall();
}

function showUsage() {
  console.log(`
🔧 Legal AI Platform Service Wrapper
===================================

Usage: node service-wrapper.js <action> <service> [options]

Actions:
  install   - Install Windows service
  uninstall - Remove Windows service

Services:
  sveltekit - SvelteKit Frontend Server

Options:
  --start   - Start service after installation

Examples:
  node service-wrapper.js install sveltekit
  node service-wrapper.js install sveltekit --start
  node service-wrapper.js uninstall sveltekit

Prerequisites:
  - Run as Administrator
  - Ensure 'npm run build' has been executed
  - Install node-windows: npm install -g node-windows
`);
}

function checkAdministrator() {
  const isWindows = process.platform === 'win32';
  if (!isWindows) {
    console.error('❌ This service wrapper is designed for Windows only.');
    process.exit(1);
  }

  // Note: This is a simplified check. In production, you might want more robust detection
  const userInfo = process.env.USERNAME;
  const isAdmin = process.env.USERPROFILE && process.env.USERPROFILE.includes('Administrator');
  
  if (!isAdmin) {
    console.warn('⚠️ This script should be run as Administrator for proper service installation.');
    console.warn('   Please run Command Prompt or PowerShell as Administrator.');
  }
}

// Main execution
if (process.argv.length < 4) {
  showUsage();
  process.exit(1);
}

const action = process.argv[2];
const serviceName = process.argv[3];

// Check prerequisites
checkAdministrator();

// Validate node-windows availability
try {
  require('node-windows');
} catch (err) {
  console.error('❌ node-windows package not found.');
  console.error('   Please install it globally: npm install -g node-windows');
  process.exit(1);
}

// Execute action
switch (action.toLowerCase()) {
  case 'install':
    installService(serviceName);
    break;
  
  case 'uninstall':
    uninstallService(serviceName);
    break;
  
  default:
    console.error(`❌ Invalid action: ${action}`);
    showUsage();
    process.exit(1);
}