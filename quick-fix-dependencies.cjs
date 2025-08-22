#!/usr/bin/env node

/**
 * Quick Fix for Critical Dependencies
 * Legal AI Platform - Emergency Recovery
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🚨 Legal AI Platform - Quick Dependency Fix\n');

class QuickFixer {
  constructor() {
    this.projectRoot = process.cwd();
    this.frontendDir = path.join(this.projectRoot, 'sveltekit-frontend');
  }

  // Fix the corrupted postgres-js dependency
  fixPostgresJs() {
    console.log('🔧 Fixing corrupted postgres-js dependency...\n');
    
    try {
      // Remove the problematic package from package.json
      const packageJsonPath = path.join(this.frontendDir, 'package.json');
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
      
      if (packageJson.dependencies && packageJson.dependencies['postgres-js']) {
        console.log('Removing corrupted postgres-js dependency...');
        delete packageJson.dependencies['postgres-js'];
        fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
        console.log('✅ postgres-js removed from package.json');
      }

      // Also check for any other problematic packages
      const problematicPackages = ['postgres-js', 'postgres'];
      for (const pkg of problematicPackages) {
        if (packageJson.dependencies && packageJson.dependencies[pkg]) {
          console.log(`Removing ${pkg} dependency...`);
          delete packageJson.dependencies[pkg];
        }
      }
      
      fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
      console.log('✅ Problematic packages removed\n');
      
    } catch (error) {
      console.log(`❌ Error fixing package.json: ${error.message}`);
    }
  }

  // Clean and reinstall with npm instead of pnpm
  async cleanAndReinstall() {
    console.log('🧹 Cleaning and reinstalling with npm...\n');
    
    try {
      // Remove node_modules and lock files
      const dirsToClean = [
        path.join(this.frontendDir, 'node_modules'),
        path.join(this.frontendDir, '.svelte-kit'),
        path.join(this.projectRoot, 'node_modules')
      ];

      for (const dir of dirsToClean) {
        if (fs.existsSync(dir)) {
          console.log(`Removing: ${dir}`);
          try {
            fs.rmSync(dir, { recursive: true, force: true });
          } catch (error) {
            console.log(`Warning: Could not remove ${dir}: ${error.message}`);
          }
        }
      }

      // Remove lock files
      const lockFiles = [
        path.join(this.frontendDir, 'package-lock.json'),
        path.join(this.frontendDir, 'pnpm-lock.yaml'),
        path.join(this.projectRoot, 'package-lock.json'),
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

      console.log('✅ Cleanup completed\n');

      // Install with npm
      console.log('Installing root dependencies with npm...');
      execSync('npm install', { stdio: 'inherit', cwd: this.projectRoot });

      console.log('\nInstalling frontend dependencies with npm...');
      execSync('npm install', { stdio: 'inherit', cwd: this.frontendDir });

      console.log('✅ Dependencies installed successfully\n');
      return true;

    } catch (error) {
      console.log(`❌ Installation failed: ${error.message}`);
      return false;
    }
  }

  // Start frontend service
  startFrontend() {
    console.log('🚀 Starting frontend service...\n');
    
    try {
      // Check if TypeScript is available
      const tsPath = path.join(this.frontendDir, 'node_modules', '.bin', 'tsc.cmd');
      if (fs.existsSync(tsPath)) {
        console.log('✅ TypeScript found');
      } else {
        console.log('⚠️ TypeScript not found, trying to start anyway');
      }

      // Start the frontend
      console.log('Starting SvelteKit dev server...');
      const child = require('child_process').spawn('npm', ['run', 'dev'], {
        cwd: this.frontendDir,
        stdio: 'inherit',
        shell: true
      });

      console.log('✅ Frontend service started');
      console.log('📱 Access at: http://localhost:5180');
      
      return child;
    } catch (error) {
      console.log(`❌ Failed to start frontend: ${error.message}`);
      return null;
    }
  }

  // Run quick fix
  async runQuickFix() {
    console.log('🚀 Starting quick dependency fix...\n');

    try {
      // Step 1: Fix corrupted dependencies
      this.fixPostgresJs();

      // Step 2: Clean and reinstall
      const installSuccess = await this.cleanAndReinstall();
      if (!installSuccess) {
        console.log('❌ Installation failed. Trying alternative approach...');
        return false;
      }

      // Step 3: Start frontend
      const frontendProcess = this.startFrontend();
      if (!frontendProcess) {
        console.log('❌ Frontend startup failed');
        return false;
      }

      console.log('\n🎉 Quick fix completed successfully!\n');
      console.log('📋 Next steps:');
      console.log('   1. Frontend should be running at http://localhost:5180');
      console.log('   2. Test TypeScript compilation: npm run check:typescript');
      console.log('   3. Test API endpoints');
      console.log('   4. Run comprehensive tests');

      return true;
    } catch (error) {
      console.log(`❌ Quick fix failed: ${error.message}`);
      return false;
    }
  }
}

// Run if called directly
if (require.main === module) {
  const fixer = new QuickFixer();
  fixer.runQuickFix().catch(console.error);
}

module.exports = QuickFixer;
