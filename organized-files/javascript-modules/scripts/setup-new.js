#!/usr/bin/env node

import { execSync } from 'child_process';
import { existsSync } from 'fs';

function runCommand(command, description) {
  try {
    console.log(`🔧 ${description}...`);
    execSync(command, { stdio: 'inherit' });
    console.log(`✅ ${description} completed\n`);
  } catch (error) {
    console.error(`❌ ${description} failed:`, error.message);
    process.exit(1);
  }
}

function checkPrerequisites() {
  console.log('🔍 Checking prerequisites...\n');

  // Check Docker
  try {
    execSync('docker --version', { stdio: 'pipe' });
    console.log('✅ Docker is installed');
  } catch {
    console.error('❌ Docker is not installed or not running');
    process.exit(1);
  }

  // Check Docker Compose
  try {
    execSync('docker-compose --version', { stdio: 'pipe' });
    console.log('✅ Docker Compose is available');
  } catch {
    console.error('❌ Docker Compose is not available');
    process.exit(1);
  }

  // Check Node.js
  try {
    execSync('node --version', { stdio: 'pipe' });
    console.log('✅ Node.js is installed');
  } catch {
    console.error('❌ Node.js is not installed');
    process.exit(1);
  }

  console.log();
}

function main() {
  console.log('🚀 Legal AI Assistant Setup');
  console.log('============================\n');

  checkPrerequisites();

  // Install dependencies
  if (existsSync('sveltekit-frontend/package.json')) {
    runCommand(
      'cd sveltekit-frontend && npm ci',
      'Installing SvelteKit dependencies'
    );
  }

  // Pull Docker images
  runCommand(
    'docker-compose -f docker-compose-fixed.yml pull',
    'Pulling Docker images'
  );

  // Build custom images
  runCommand(
    'docker-compose -f docker-compose-fixed.yml build',
    'Building custom Docker images'
  );

  console.log('🎉 Setup completed successfully!');
  console.log('\n📝 Next steps:');
  console.log('   1. Run: npm run docker:up');
  console.log('   2. Wait for services to start');
  console.log('   3. Run: npm run health');
  console.log('   4. Access app at: http://localhost:5173');
}

main();
