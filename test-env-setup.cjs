#!/usr/bin/env node

/**
 * 🧪 Test Environment Setup and Dual-Port Configuration
 * This script tests the environment loading and verifies the setup
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('🧪 Testing Environment Setup and Dual-Port Configuration\n');

// Test 1: Load environment configuration
console.log('📝 Test 1: Loading Environment Configuration...');
try {
  // Since this is a .cjs file, we can't use ES modules directly
  // We'll check if the file exists and show a note
  if (fs.existsSync('./env-config.mjs')) {
    console.log('✅ Environment configuration file exists (env-config.mjs)');
    console.log('   ⚠️  Note: This is an ES module, use start:env for full environment loading');
  } else {
    console.log('❌ Environment configuration file missing');
  }
  
  // Check if any environment variables are set
  const keyVars = ['FRONTEND_PORT', 'PROXY_PORT', 'DATABASE_URL', 'REDIS_URL'];
  let hasEnvVars = false;
  keyVars.forEach(varName => {
    if (process.env[varName]) {
      console.log(`   • ${varName}: ${process.env[varName]}`);
      hasEnvVars = true;
    }
  });
  
  if (!hasEnvVars) {
    console.log('   ℹ️  No environment variables currently set (will be loaded by start:env)');
  }
} catch (error) {
  console.log('❌ Error checking environment configuration:', error.message);
}

// Test 2: Check required files exist
console.log('\n📁 Test 2: Checking Required Files...');
const requiredFiles = [
  'start-port-5180.cjs',
  'env-config.mjs',
  'start-dual-ports-with-env.cjs'
];

requiredFiles.forEach(file => {
  const filePath = path.join(process.cwd(), file);
  if (fs.existsSync(filePath)) {
    console.log(`   ✅ ${file} exists`);
  } else {
    console.log(`   ❌ ${file} missing`);
  }
});

// Test 3: Check directories
console.log('\n📂 Test 3: Checking Required Directories...');
const requiredDirs = [
  'sveltekit-frontend',
  'uploads',
  'documents',
  'evidence',
  'logs',
  'generated_reports'
];

requiredDirs.forEach(dir => {
  const dirPath = path.join(process.cwd(), dir);
  if (fs.existsSync(dirPath)) {
    console.log(`   ✅ ${dir}/ exists`);
  } else {
    console.log(`   ❌ ${dir}/ missing`);
  }
});

// Test 4: Check npm scripts
console.log('\n📦 Test 4: Checking NPM Scripts...');
try {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const scripts = packageJson.scripts || {};
  
  const requiredScripts = ['dev:dual', 'start:dual', 'start:env'];
  requiredScripts.forEach(script => {
    if (scripts[script]) {
      console.log(`   ✅ ${script} script exists`);
    } else {
      console.log(`   ❌ ${script} script missing`);
    }
  });
} catch (error) {
  console.log('   ❌ Failed to read package.json:', error.message);
}

// Test 5: Check frontend package.json
console.log('\n🎨 Test 5: Checking Frontend Package.json...');
try {
  const frontendPackageJson = JSON.parse(fs.readFileSync('sveltekit-frontend/package.json', 'utf8'));
  const frontendScripts = frontendPackageJson.scripts || {};
  
  const requiredFrontendScripts = ['dev:dual', 'start:dual', 'start:env'];
  requiredFrontendScripts.forEach(script => {
    if (frontendScripts[script]) {
      console.log(`   ✅ ${script} script exists in frontend`);
    } else {
      console.log(`   ❌ ${script} script missing in frontend`);
    }
  });
} catch (error) {
  console.log('   ❌ Failed to read frontend package.json:', error.message);
}

// Test 6: Check if ports are available
console.log('\n🔌 Test 6: Checking Port Availability...');
try {
  const netstatOutput = execSync('netstat -ano | findstr ":5173\\|:5180"', { encoding: 'utf8' });
  if (netstatOutput.trim()) {
    console.log('   ⚠️  Ports 5173 or 5180 are already in use:');
    console.log(netstatOutput);
  } else {
    console.log('   ✅ Ports 5173 and 5180 are available');
  }
} catch (error) {
  console.log('   ✅ Ports 5173 and 5180 are available (no existing services)');
}

console.log('\n🎉 Environment Setup Test Complete!');
console.log('\n📋 Next Steps:');
console.log('   1. Run: npm run start:env (for environment + dual ports)');
console.log('   2. Run: npm run dev:dual (for dual ports only)');
console.log('   3. Run: npm run start:dual (alternative dual port command)');
console.log('   4. Run: npm run dev:proxy (proxy only, if frontend is running)');
console.log('\n🌐 Expected URLs:');
console.log('   • Frontend: http://localhost:5173');
console.log('   • Proxy: http://localhost:5180');
console.log('\n📖 For more options, see: COMPLETE-DUAL-PORT-SETUP.md');
