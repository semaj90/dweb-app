#!/usr/bin/env node
/**
 * Startup Automation Script
 * Reliable startup management with ready flag monitoring and retry logic
 */

import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const logsDir = join(rootDir, 'logs');

// Configuration
const CONFIG = {
  maxWaitTime: 120000, // 2 minutes max wait
  checkInterval: 2000, // Check every 2 seconds
  retryAttempts: 3,
  retryDelay: 5000, // 5 seconds between retries
  requiredServices: [
    'postgres',
    'redis', 
    'ollama-primary',
    'gpu-embedding-service',
    'enhanced-rag',
    'sveltekit'
  ],
  optionalServices: [
    'upload-service',
    'neo4j',
    'qdrant',
    'minio'
  ]
};

/**
 * Main startup automation function
 */
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'start';
  
  console.log(`🚀 Legal AI Platform Startup Automation`);
  console.log(`Command: ${command}`);
  console.log(`Timestamp: ${new Date().toISOString()}\n`);
  
  try {
    switch (command) {
      case 'start':
        await startSystem();
        break;
      case 'check':
        await checkReadiness();
        break;
      case 'wait':
        await waitForReady();
        break;
      case 'status':
        await showStatus();
        break;
      case 'diff':
        await showStartupDiff();
        break;
      case 'retry':
        await retryFailedServices();
        break;
      default:
        showHelp();
    }
  } catch (error) {
    console.error('❌ Startup automation failed:', error.message);
    process.exit(1);
  }
}

/**
 * Start the entire system with monitoring
 */
async function startSystem() {
  console.log('🔄 Starting Legal AI Platform...\n');
  
  // 1. Start SvelteKit dev server (this will initialize startup flag service)
  console.log('📦 Starting SvelteKit frontend...');
  const svelteProcess = spawn('npm', ['run', 'dev'], {
    cwd: rootDir,
    stdio: 'inherit',
    shell: true
  });
  
  // Give SvelteKit time to initialize
  await new Promise(resolve => setTimeout(resolve, 5000));
  
  // 2. Initialize startup monitoring via API
  console.log('🔧 Initializing startup monitoring...');
  try {
    const response = await fetch('http://localhost:5173/api/v1/startup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'start' })
    });
    
    if (response.ok) {
      console.log('✅ Startup monitoring initiated');
    } else {
      console.warn('⚠️ Failed to initialize startup monitoring');
    }
  } catch (error) {
    console.warn('⚠️ Startup monitoring API not available yet');
  }
  
  // 3. Wait for system readiness
  console.log('\n⏳ Waiting for system readiness...');
  const isReady = await waitForReady();
  
  if (isReady) {
    console.log('\n🎉 Legal AI Platform is ready!');
    console.log('📊 Access points:');
    console.log('  • Frontend: http://localhost:5173');
    console.log('  • API Status: http://localhost:5173/api/v1/startup');
    console.log('  • Telemetry: http://localhost:5173/api/v1/telemetry/upload');
    console.log('  • Embeddings: http://localhost:5173/api/v1/embeddings');
    showFinalStatus();
  } else {
    console.log('\n❌ System failed to become ready within timeout');
    await showStatus();
    process.exit(1);
  }
}

/**
 * Check current readiness status
 */
async function checkReadiness() {
  try {
    const flagPath = join(logsDir, 'ready.flag');
    
    if (existsSync(flagPath)) {
      const flagContent = JSON.parse(await readFile(flagPath, 'utf-8'));
      const age = Date.now() - flagContent.timestamp;
      const ageMinutes = Math.floor(age / 60000);
      
      console.log('✅ System is ready!');
      console.log(`   Flag age: ${ageMinutes} minutes`);
      console.log(`   Session: ${flagContent.sessionId}`);
      console.log(`   Startup duration: ${flagContent.startupDuration}ms`);
      console.log(`   Ready services: ${flagContent.readyServices.join(', ')}`);
      return true;
    } else {
      console.log('❌ System is not ready (no flag file)');
      return false;
    }
  } catch (error) {
    console.log('❌ Cannot determine readiness:', error.message);
    return false;
  }
}

/**
 * Wait for system to become ready
 */
async function waitForReady() {
  const startTime = Date.now();
  let attempt = 0;
  
  while (Date.now() - startTime < CONFIG.maxWaitTime) {
    attempt++;
    
    try {
      // Check via API first (more reliable)
      const response = await fetch('http://localhost:5173/api/v1/startup?action=status', {
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.ready) {
          const elapsed = Date.now() - startTime;
          console.log(`✅ System ready after ${elapsed}ms (${attempt} checks)`);
          return true;
        } else {
          const ready = data.summary.readyServices;
          const total = data.summary.totalServices;
          console.log(`⏳ Progress: ${ready}/${total} services ready (attempt ${attempt})`);
        }
      } else {
        // Fallback to file check
        const fileReady = await checkReadiness();
        if (fileReady) {
          return true;
        }
      }
    } catch (error) {
      // API not available, try file check
      const fileReady = await checkReadiness();
      if (fileReady) {
        return true;
      }
      
      if (attempt === 1) {
        console.log('⏳ Waiting for API to become available...');
      }
    }
    
    await new Promise(resolve => setTimeout(resolve, CONFIG.checkInterval));
  }
  
  console.log(`❌ Timeout after ${CONFIG.maxWaitTime}ms`);
  return false;
}

/**
 * Show detailed system status
 */
async function showStatus() {
  try {
    const response = await fetch('http://localhost:5173/api/v1/startup?action=health');
    
    if (response.ok) {
      const data = await response.json();
      
      console.log(`📊 System Health: ${data.health.toUpperCase()}`);
      console.log(`🚩 Ready Flag: ${data.ready ? '✅ Set' : '❌ Not set'}\n`);
      
      console.log('📋 Critical Services:');
      for (const [name, service] of Object.entries(data.criticalServices)) {
        const statusIcon = service.status === 'ready' ? '✅' : 
                          service.status === 'starting' ? '⏳' : '❌';
        const healthIcon = service.health === 'excellent' ? '🟢' :
                          service.health === 'good' ? '🟡' :
                          service.health === 'fair' ? '🟠' : '🔴';
        
        const startupTime = service.startupTime ? `(${service.startupTime}ms)` : '';
        console.log(`  ${statusIcon} ${healthIcon} ${name}: ${service.status} ${startupTime}`);
      }
    } else {
      console.log('❌ Cannot fetch status - API not available');
      await checkReadiness();
    }
  } catch (error) {
    console.log('❌ Status check failed:', error.message);
    await checkReadiness();
  }
}

/**
 * Show startup performance diff
 */
async function showStartupDiff() {
  try {
    const response = await fetch('http://localhost:5173/api/v1/startup?action=diff');
    
    if (response.ok) {
      const data = await response.json();
      
      if (data.diff) {
        const diff = data.diff;
        
        console.log('📊 Startup Performance Diff:');
        console.log(`   Current vs Previous (${new Date(diff.timestamp).toLocaleString()})\n`);
        
        if (diff.changes.statusChanges.length > 0) {
          console.log('🔄 Status Changes:');
          for (const change of diff.changes.statusChanges) {
            console.log(`   • ${change.service}: ${change.from} → ${change.to}`);
          }
        }
        
        if (diff.changes.performanceChanges.length > 0) {
          console.log('\n⏱️  Performance Changes:');
          for (const change of diff.changes.performanceChanges) {
            const timeChange = change.startupTimeChange;
            const timeIcon = timeChange > 0 ? '🔴' : '🟢';
            console.log(`   ${timeIcon} ${change.service}: ${timeChange > 0 ? '+' : ''}${timeChange}ms`);
          }
        }
        
        if (diff.regressions.length > 0) {
          console.log('\n⚠️ Regressions Detected:');
          for (const regression of diff.regressions) {
            const severityIcon = regression.severity === 'critical' ? '🔴' :
                               regression.severity === 'high' ? '🟠' :
                               regression.severity === 'medium' ? '🟡' : '🟢';
            console.log(`   ${severityIcon} ${regression.service}: ${regression.issue}`);
          }
        } else {
          console.log('\n✅ No performance regressions detected');
        }
        
      } else {
        console.log('📊 No startup diff data available (first run)');
      }
    } else {
      console.log('❌ Cannot fetch diff data');
    }
  } catch (error) {
    console.log('❌ Diff check failed:', error.message);
  }
}

/**
 * Retry failed optional services
 */
async function retryFailedServices() {
  console.log('🔄 Retrying failed optional services...\n');
  
  try {
    // Get current status
    const response = await fetch('http://localhost:5173/api/v1/startup?action=status');
    const data = await response.json();
    
    const failedOptional = Object.entries(data.summary.services)
      .filter(([_, service]) => service.isOptional && service.status === 'failed')
      .map(([name]) => name);
    
    if (failedOptional.length === 0) {
      console.log('✅ No failed optional services to retry');
      return;
    }
    
    console.log(`🔄 Retrying ${failedOptional.length} failed optional services:`);
    console.log(`   ${failedOptional.join(', ')}\n`);
    
    // Trigger retry via startup service
    await fetch('http://localhost:5173/api/v1/startup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'start' })
    });
    
    // Wait for retry completion
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Check results
    await showStatus();
    
  } catch (error) {
    console.log('❌ Retry failed:', error.message);
  }
}

/**
 * Show final status summary
 */
async function showFinalStatus() {
  try {
    const response = await fetch('http://localhost:5173/api/v1/startup?action=status');
    const data = await response.json();
    
    console.log('\n📈 Final Status Summary:');
    console.log(`   Total Services: ${data.summary.totalServices}`);
    console.log(`   Ready: ${data.summary.readyServices}`);
    console.log(`   Failed: ${data.summary.failedServices}`);
    console.log(`   Startup Duration: ${data.summary.startupDuration}ms`);
    console.log(`   Environment: ${data.summary.environment}`);
    
    // Show logs location
    console.log('\n📁 Logs and Artifacts:');
    console.log(`   Ready flag: logs/ready.flag`);
    console.log(`   Startup summary: logs/startup-summary.json`);
    console.log(`   Performance diff: logs/startup-diff.json`);
    
  } catch (error) {
    console.log('⚠️ Cannot show final status:', error.message);
  }
}

/**
 * Show help information
 */
function showHelp() {
  console.log('🚀 Legal AI Platform Startup Automation\n');
  console.log('Usage: node startup-automation.mjs [command]\n');
  console.log('Commands:');
  console.log('  start    - Start the entire system and wait for readiness');
  console.log('  check    - Check if system is currently ready');
  console.log('  wait     - Wait for system to become ready');
  console.log('  status   - Show detailed system status');
  console.log('  diff     - Show startup performance diff');
  console.log('  retry    - Retry failed optional services');
  console.log('  help     - Show this help message\n');
  console.log('Examples:');
  console.log('  node startup-automation.mjs start');
  console.log('  node startup-automation.mjs check');
  console.log('  node startup-automation.mjs wait && echo "System ready!"');
}

// Handle process signals
process.on('SIGINT', () => {
  console.log('\n🛑 Startup automation interrupted');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Startup automation terminated');
  process.exit(0);
});

// Run main function
main().catch(error => {
  console.error('💥 Startup automation crashed:', error);
  process.exit(1);
});