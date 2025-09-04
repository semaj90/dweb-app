#!/usr/bin/env node

/**
 * Startup Verification Script
 * Checks all system components and configurations
 */

import { execSync } from 'child_process';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🚀 YoRHa Legal AI Platform - Startup Verification\n');

// Color codes for output
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function checkService(name, host, port, protocol = 'http') {
  return new Promise((resolve) => {
    try {
      if (protocol === 'tcp') {
        // For TCP services like PostgreSQL and Redis
        execSync(`powershell -Command "Test-NetConnection -ComputerName ${host} -Port ${port} -InformationLevel Quiet"`, 
          { stdio: 'pipe', timeout: 5000 });
        resolve({ name, status: 'online', host, port });
      } else {
        // For HTTP services
        execSync(`curl -s --max-time 5 ${protocol}://${host}:${port} > nul 2>&1`, 
          { stdio: 'pipe', timeout: 5000 });
        resolve({ name, status: 'online', host, port });
      }
    } catch (error) {
      resolve({ name, status: 'offline', host, port, error: error.message });
    }
  });
}

async function verifyEnvironment() {
  log('📋 Environment Configuration Check', 'blue');
  
  try {
    const envPath = join(__dirname, '.env');
    const envContent = readFileSync(envPath, 'utf8');
    
    const requiredVars = [
      'DATABASE_URL',
      'POSTGRES_USER', 
      'POSTGRES_PASSWORD',
      'MINIO_HOST',
      'REDIS_URL'
    ];
    
    const envLines = envContent.split('\n');
    const envVars = {};
    
    envLines.forEach(line => {
      if (line.includes('=') && !line.startsWith('#')) {
        const [key, value] = line.split('=', 2);
        envVars[key.trim()] = value.trim();
      }
    });
    
    let allPresent = true;
    requiredVars.forEach(varName => {
      if (envVars[varName]) {
        log(`  ✅ ${varName}: Set`, 'green');
      } else {
        log(`  ❌ ${varName}: Missing`, 'red');
        allPresent = false;
      }
    });
    
    log(`\n📊 Database Configuration:`, 'blue');
    log(`  🗄️  Database: ${envVars.POSTGRES_DB || 'legal_ai_db'}`);
    log(`  👤 User: ${envVars.POSTGRES_USER || 'postgres'}`);
    log(`  🏠 Host: ${envVars.POSTGRES_HOST || 'localhost'}:${envVars.POSTGRES_PORT || '5432'}`);
    
    return allPresent;
  } catch (error) {
    log('❌ Environment file check failed:', 'red');
    log(`   ${error.message}`);
    return false;
  }
}

async function verifyServices() {
  log('\n🔍 Service Connectivity Check', 'blue');
  
  const services = [
    { name: 'PostgreSQL', host: 'localhost', port: 5432, protocol: 'tcp' },
    { name: 'Redis', host: 'localhost', port: 6379, protocol: 'tcp' },
    { name: 'MinIO', host: 'localhost', port: 9000, protocol: 'http' },
    { name: 'Ollama', host: 'localhost', port: 11434, protocol: 'http' },
    { name: 'Qdrant', host: 'localhost', port: 6333, protocol: 'http' },
    { name: 'Neo4j', host: 'localhost', port: 7474, protocol: 'http' }
  ];
  
  const results = await Promise.all(
    services.map(service => checkService(service.name, service.host, service.port, service.protocol))
  );
  
  let onlineCount = 0;
  results.forEach(result => {
    if (result.status === 'online') {
      log(`  ✅ ${result.name}: Online (${result.host}:${result.port})`, 'green');
      onlineCount++;
    } else {
      log(`  ❌ ${result.name}: Offline (${result.host}:${result.port})`, 'red');
    }
  });
  
  log(`\n📊 Service Status: ${onlineCount}/${results.length} services online`);
  return { online: onlineCount, total: results.length, results };
}

async function verifyBuild() {
  log('\n🔨 Build Verification', 'blue');
  
  try {
    // Check if node_modules exists
    execSync('test -d node_modules', { stdio: 'pipe' });
    log('  ✅ Dependencies installed', 'green');
    
    // Try TypeScript check (non-blocking)
    try {
      execSync('npm run check:typescript', { stdio: 'pipe', timeout: 30000 });
      log('  ✅ TypeScript compilation successful', 'green');
      return true;
    } catch (error) {
      log('  ⚠️ TypeScript compilation has errors (non-blocking)', 'yellow');
      return true; // Still allow startup
    }
  } catch (error) {
    log('  ❌ Build verification failed', 'red');
    log(`     ${error.message}`);
    return false;
  }
}

async function testDatabaseConnection() {
  log('\n🗄️  Database Connection Test', 'blue');
  
  try {
    // Try to connect to PostgreSQL using psql
    const testQuery = 'SELECT version(), current_database(), current_user';
    const result = execSync(`psql postgresql://postgres:123456@localhost:5432/legal_ai_db -c "${testQuery}" -t`, 
      { encoding: 'utf8', timeout: 10000 });
    
    log('  ✅ PostgreSQL connection successful', 'green');
    log('  📋 Database info:', 'blue');
    
    const lines = result.trim().split('\n').filter(line => line.trim());
    lines.forEach(line => {
      if (line.trim()) {
        log(`     ${line.trim()}`);
      }
    });
    
    // Test pgvector extension
    try {
      const vectorCheck = execSync(`psql postgresql://postgres:123456@localhost:5432/legal_ai_db -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as pgvector_installed" -t`, 
        { encoding: 'utf8', timeout: 5000 });
      
      const hasVector = vectorCheck.trim().includes('t');
      if (hasVector) {
        log('  ✅ pgvector extension installed', 'green');
      } else {
        log('  ⚠️ pgvector extension not installed', 'yellow');
      }
    } catch (vectorError) {
      log('  ⚠️ Could not check pgvector extension', 'yellow');
    }
    
    return true;
  } catch (error) {
    log('  ❌ PostgreSQL connection failed', 'red');
    log(`     Error: ${error.message.split('\n')[0]}`);
    return false;
  }
}

async function generateStartupReport() {
  log('\n' + '='.repeat(60), 'bold');
  log('🎯 STARTUP VERIFICATION REPORT', 'bold');
  log('='.repeat(60), 'bold');
  
  const envOk = await verifyEnvironment();
  const serviceStatus = await verifyServices();
  const buildOk = await verifyBuild();
  const dbOk = await testDatabaseConnection();
  
  log('\n📊 Overall System Status:', 'bold');
  
  const checks = [
    { name: 'Environment Configuration', status: envOk },
    { name: 'Build System', status: buildOk },
    { name: 'Database Connection', status: dbOk },
    { name: 'Core Services', status: serviceStatus.online >= 3 } // At least 3 services
  ];
  
  let passedChecks = 0;
  checks.forEach(check => {
    if (check.status) {
      log(`  ✅ ${check.name}`, 'green');
      passedChecks++;
    } else {
      log(`  ❌ ${check.name}`, 'red');
    }
  });
  
  const overallHealth = passedChecks / checks.length;
  
  log(`\n🎯 System Health: ${Math.round(overallHealth * 100)}%`, 'bold');
  
  if (overallHealth >= 0.75) {
    log('✅ System is ready for startup!', 'green');
    log('\n🚀 Start the application with: npm run dev', 'blue');
  } else if (overallHealth >= 0.5) {
    log('⚠️ System has some issues but may still work', 'yellow');
    log('💡 Check the failed components above', 'blue');
  } else {
    log('❌ System has critical issues', 'red');
    log('🔧 Fix the failed components before starting', 'blue');
  }
  
  log('\n📍 Quick Access Links (once started):');
  log('   🌐 Main App: http://localhost:5173');
  log('   🎮 GPU Cache Demo: http://localhost:5173/demo/gpu-cache-integration');
  log('   📊 System Status: http://localhost:5173/status');
  log('   🧪 Integration Tests: http://localhost:5173/test/integration');
  log('   ❤️ Health Check: http://localhost:5173/api/health');
}

// Run the verification
await generateStartupReport();