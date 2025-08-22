#!/usr/bin/env node

/**
 * Fix Dependencies and Start Services
 * Legal AI Platform Recovery Script
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

console.log('🔧 Legal AI Platform - Dependency Fix & Service Startup\n');

class DependencyFixer {
  constructor() {
    this.projectRoot = process.cwd();
    this.frontendDir = path.join(this.projectRoot, 'sveltekit-frontend');
  }

  // Check if a command exists
  checkCommand(command) {
    try {
      execSync(`${command} --version`, { stdio: 'pipe' });
      return true;
    } catch {
      return false;
    }
  }

  // Get available package manager
  getPackageManager() {
    if (this.checkCommand('pnpm')) return 'pnpm';
    if (this.checkCommand('yarn')) return 'yarn';
    if (this.checkCommand('npm')) return 'npm';
    return null;
  }

  // Clean corrupted dependencies
  cleanDependencies() {
    console.log('🧹 Cleaning corrupted dependencies...\n');
    
    const dirsToClean = [
      path.join(this.frontendDir, 'node_modules'),
      path.join(this.frontendDir, '.svelte-kit'),
      path.join(this.frontendDir, 'dist'),
      path.join(this.projectRoot, 'node_modules')
    ];

    for (const dir of dirsToClean) {
      if (fs.existsSync(dir)) {
        try {
          console.log(`Removing: ${dir}`);
          fs.rmSync(dir, { recursive: true, force: true });
        } catch (error) {
          console.log(`Warning: Could not remove ${dir}: ${error.message}`);
        }
      }
    }

    // Remove lock files
    const lockFiles = [
      path.join(this.frontendDir, 'package-lock.json'),
      path.join(this.frontendDir, 'yarn.lock'),
      path.join(this.frontendDir, 'pnpm-lock.yaml'),
      path.join(this.projectRoot, 'package-lock.json'),
      path.join(this.projectRoot, 'yarn.lock'),
      path.join(this.projectRoot, 'pnpm-lock.yaml')
    ];

    for (const file of lockFiles) {
      if (fs.existsSync(file)) {
        try {
          console.log(`Removing: ${file}`);
          fs.unlinkSync(file);
        } catch (error) {
          console.log(`Warning: Could not remove ${file}: ${error.message}`);
        }
      }
    }

    console.log('✅ Dependencies cleaned\n');
  }

  // Install dependencies
  async installDependencies() {
    console.log('📦 Installing dependencies...\n');
    
    const packageManager = this.getPackageManager();
    if (!packageManager) {
      console.log('❌ No package manager found. Installing npm...');
      try {
        execSync('npm install -g npm@latest', { stdio: 'inherit' });
      } catch (error) {
        console.log('❌ Failed to install npm');
        return false;
      }
    }

    console.log(`Using package manager: ${packageManager}\n`);

    try {
      // Install root dependencies
      console.log('Installing root dependencies...');
      if (packageManager === 'pnpm') {
        execSync('pnpm install', { stdio: 'inherit', cwd: this.projectRoot });
      } else if (packageManager === 'yarn') {
        execSync('yarn install', { stdio: 'inherit', cwd: this.projectRoot });
      } else {
        execSync('npm install', { stdio: 'inherit', cwd: this.projectRoot });
      }

      // Install frontend dependencies
      console.log('\nInstalling frontend dependencies...');
      if (packageManager === 'pnpm') {
        execSync('pnpm install', { stdio: 'inherit', cwd: this.frontendDir });
      } else if (packageManager === 'yarn') {
        execSync('yarn install', { stdio: 'inherit', cwd: this.frontendDir });
      } else {
        execSync('npm install', { stdio: 'inherit', cwd: this.frontendDir });
      }

      console.log('✅ Dependencies installed successfully\n');
      return true;
    } catch (error) {
      console.log(`❌ Failed to install dependencies: ${error.message}\n`);
      return false;
    }
  }

  // Start services
  async startServices() {
    console.log('🚀 Starting services...\n');

    const services = [
      {
        name: 'Frontend',
        command: 'npm run dev',
        cwd: this.frontendDir,
        port: 5180
      },
      {
        name: 'QUIC Gateway',
        command: 'go run quic-services/main.go',
        cwd: path.join(this.projectRoot, 'quic-services'),
        port: 8447
      }
    ];

    for (const service of services) {
      try {
        console.log(`Starting ${service.name}...`);
        
        // Check if service is already running
        try {
          const result = execSync(`netstat -ano | findstr ":${service.port}"`, { encoding: 'utf8' });
          if (result.trim()) {
            console.log(`✅ ${service.name} already running on port ${service.port}`);
            continue;
          }
        } catch {
          // Port not in use, continue to start service
        }

        // Start service in background
        const child = spawn(service.command, [], {
          cwd: service.cwd,
          stdio: 'pipe',
          shell: true,
          detached: true
        });

        // Wait a bit for service to start
        await new Promise(resolve => setTimeout(resolve, 5000));

        // Check if service started successfully
        try {
          const checkResult = execSync(`netstat -ano | findstr ":${service.port}"`, { encoding: 'utf8' });
          if (checkResult.trim()) {
            console.log(`✅ ${service.name} started successfully on port ${service.port}`);
          } else {
            console.log(`⚠️ ${service.name} may not have started (port ${service.port} not listening)`);
          }
        } catch {
          console.log(`⚠️ ${service.name} may not have started (port ${service.port} not listening)`);
        }

      } catch (error) {
        console.log(`❌ Failed to start ${service.name}: ${error.message}`);
      }
    }

    console.log('\n✅ Service startup completed\n');
  }

  // Verify services
  async verifyServices() {
    console.log('🔍 Verifying services...\n');

    const services = [
      { name: 'Frontend', port: 5180 },
      { name: 'Upload Service', port: 8093 },
      { name: 'RAG Service', port: 8094 },
      { name: 'QUIC Gateway', port: 8447 },
      { name: 'PostgreSQL', port: 5432 },
      { name: 'Redis', port: 6379 }
    ];

    for (const service of services) {
      try {
        const result = execSync(`netstat -ano | findstr ":${service.port}"`, { encoding: 'utf8' });
        if (result.trim()) {
          console.log(`✅ ${service.name}: Port ${service.port} - LISTENING`);
        } else {
          console.log(`❌ ${service.name}: Port ${service.port} - NOT LISTENING`);
        }
      } catch {
        console.log(`❌ ${service.name}: Port ${service.port} - NOT LISTENING`);
      }
    }

    console.log('\n✅ Service verification completed\n');
  }

  // Run full recovery
  async runRecovery() {
    console.log('🚀 Starting full system recovery...\n');

    try {
      // Step 1: Clean dependencies
      this.cleanDependencies();

      // Step 2: Install dependencies
      const installSuccess = await this.installDependencies();
      if (!installSuccess) {
        console.log('❌ Dependency installation failed. Recovery cannot continue.');
        return false;
      }

      // Step 3: Start services
      await this.startServices();

      // Step 4: Verify services
      await this.verifyServices();

      console.log('🎉 System recovery completed successfully!\n');
      console.log('📋 Next steps:');
      console.log('   1. Test frontend at http://localhost:5180');
      console.log('   2. Run TypeScript check: npm run check:typescript');
      console.log('   3. Test API endpoints');
      console.log('   4. Run comprehensive tests');

      return true;
    } catch (error) {
      console.log(`❌ Recovery failed: ${error.message}`);
      return false;
    }
  }
}

// Run if called directly
if (require.main === module) {
  const fixer = new DependencyFixer();
  fixer.runRecovery().catch(console.error);
}

module.exports = DependencyFixer;
