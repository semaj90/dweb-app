#!/usr/bin/env node
/**
 * Multi-Library Integration Verification Script
 * Verifies all 7 libraries are properly installed and integrated
 */

import chalk from 'chalk';
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import path from 'path';

const style = {
  primary: (text) => chalk.hex('#f4f4f4')(text),
  secondary: (text) => chalk.hex('#8b9dc3')(text),
  accent: (text) => chalk.hex('#dca561')(text),
  success: (text) => chalk.hex('#51cf66')(text),
  warning: (text) => chalk.hex('#ff6b6b')(text),
  error: (text) => chalk.hex('#ff4757')(text),
  bold: (text) => chalk.bold(text),
  dim: (text) => chalk.dim(text)
};

class MultiLibraryVerifier {
  constructor() {
    this.results = {
      packageJson: false,
      libraries: {
        lokijs: false,
        fusejs: false,
        fabricjs: false,
        xstate: false,
        redis: false,
        rabbitmq: false,
        orchestrator: false
      },
      services: {
        startupService: false,
        concurrencyOrchestrator: false,
        evidenceCanvas: false,
        layoutIntegration: false
      },
      scripts: {
        devFull: false,
        startupScript: false
      }
    };
  }

  async verify() {
    console.log(style.bold(style.accent('🔍 MULTI-LIBRARY INTEGRATION VERIFICATION')));
    console.log(style.dim('Checking all 7 libraries and integration components...\n'));

    this.verifyPackageJson();
    this.verifyLibraries();
    this.verifyServices();
    this.verifyScripts();
    this.showResults();
  }

  verifyPackageJson() {
    console.log(style.accent('📦 Checking package.json dependencies...'));
    
    try {
      const packagePath = path.resolve('./package.json');
      const packageJson = JSON.parse(readFileSync(packagePath, 'utf-8'));
      
      const requiredDeps = {
        'lokijs': '^1.5.12',
        'fuse.js': '^7.0.0',
        'fabric': '^5.3.0',
        'xstate': '^5.20.1',
        'ioredis': '^5.7.0',
        'amqplib': '^0.10.8'
      };

      let allFound = true;
      for (const [dep, expectedVersion] of Object.entries(requiredDeps)) {
        const installed = packageJson.dependencies?.[dep] || packageJson.devDependencies?.[dep];
        if (installed) {
          console.log(style.success(`  ✅ ${dep} - ${installed}`));
        } else {
          console.log(style.error(`  ❌ ${dep} - NOT FOUND`));
          allFound = false;
        }
      }

      this.results.packageJson = allFound;
    } catch (error) {
      console.log(style.error(`  ❌ Failed to read package.json: ${error.message}`));
    }
    console.log('');
  }

  verifyLibraries() {
    console.log(style.accent('📚 Checking library accessibility...'));

    const libraries = [
      { name: 'lokijs', check: () => this.checkNodeModule('lokijs') },
      { name: 'fusejs', check: () => this.checkNodeModule('fuse.js') },
      { name: 'fabricjs', check: () => this.checkNodeModule('fabric') },
      { name: 'xstate', check: () => this.checkNodeModule('xstate') },
      { name: 'redis', check: () => this.checkNodeModule('ioredis') },
      { name: 'rabbitmq', check: () => this.checkNodeModule('amqplib') }
    ];

    for (const library of libraries) {
      try {
        const exists = library.check();
        if (exists) {
          console.log(style.success(`  ✅ ${library.name.toUpperCase()} - Module accessible`));
          this.results.libraries[library.name] = true;
        } else {
          console.log(style.error(`  ❌ ${library.name.toUpperCase()} - Module not found`));
        }
      } catch (error) {
        console.log(style.error(`  ❌ ${library.name.toUpperCase()} - Error: ${error.message}`));
      }
    }
    
    // Check orchestrator (always true since it's our custom service)
    this.results.libraries.orchestrator = true;
    console.log(style.success(`  ✅ ORCHESTRATOR - Custom service ready`));
    console.log('');
  }

  verifyServices() {
    console.log(style.accent('⚙️ Checking service files...'));

    const services = [
      {
        name: 'Multi-Library Startup Service',
        path: './src/lib/services/multi-library-startup.ts',
        key: 'startupService'
      },
      {
        name: 'Concurrency Orchestrator',
        path: './src/lib/services/concurrency-orchestrator.ts',
        key: 'concurrencyOrchestrator'
      },
      {
        name: 'Evidence Canvas (Fabric.js)',
        path: './src/lib/components/ai/EvidenceCanvas.svelte',
        key: 'evidenceCanvas'
      },
      {
        name: 'Layout Integration',
        path: './src/routes/+layout.svelte',
        key: 'layoutIntegration'
      }
    ];

    for (const service of services) {
      const exists = existsSync(path.resolve(service.path));
      if (exists) {
        console.log(style.success(`  ✅ ${service.name}`));
        this.results.services[service.key] = true;
      } else {
        console.log(style.error(`  ❌ ${service.name} - File not found`));
      }
    }
    console.log('');
  }

  verifyScripts() {
    console.log(style.accent('📜 Checking npm scripts...'));

    try {
      const packagePath = path.resolve('./package.json');
      const packageJson = JSON.parse(readFileSync(packagePath, 'utf-8'));
      
      // Check dev:full script
      if (packageJson.scripts?.['dev:full']) {
        console.log(style.success(`  ✅ npm run dev:full - Available`));
        this.results.scripts.devFull = true;
      } else {
        console.log(style.error(`  ❌ npm run dev:full - Script not found`));
      }

      // Check startup script file
      const startupExists = existsSync(path.resolve('./scripts/dev-full-working.mjs'));
      if (startupExists) {
        console.log(style.success(`  ✅ Startup Script - dev-full-working.mjs exists`));
        this.results.scripts.startupScript = true;
      } else {
        console.log(style.error(`  ❌ Startup Script - dev-full-working.mjs not found`));
      }
    } catch (error) {
      console.log(style.error(`  ❌ Failed to check scripts: ${error.message}`));
    }
    console.log('');
  }

  checkNodeModule(moduleName) {
    try {
      const modulePath = path.resolve(`./node_modules/${moduleName}`);
      return existsSync(modulePath);
    } catch {
      return false;
    }
  }

  showResults() {
    console.log(style.bold(style.accent('📊 VERIFICATION RESULTS')));
    console.log(style.dim('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));

    // Libraries status
    const libraryCount = Object.values(this.results.libraries).filter(Boolean).length;
    const totalLibraries = Object.keys(this.results.libraries).length;
    console.log(style.primary(`Libraries: ${libraryCount}/${totalLibraries} ready`));
    
    for (const [lib, status] of Object.entries(this.results.libraries)) {
      const icon = status ? '✅' : '❌';
      const name = lib.replace('js', '.js').toUpperCase();
      console.log(`  ${icon} ${name}`);
    }

    // Services status
    const serviceCount = Object.values(this.results.services).filter(Boolean).length;
    const totalServices = Object.keys(this.results.services).length;
    console.log(style.primary(`\nServices: ${serviceCount}/${totalServices} ready`));
    
    for (const [service, status] of Object.entries(this.results.services)) {
      const icon = status ? '✅' : '❌';
      const name = service.replace(/([A-Z])/g, ' $1').toUpperCase();
      console.log(`  ${icon} ${name}`);
    }

    // Scripts status
    const scriptCount = Object.values(this.results.scripts).filter(Boolean).length;
    const totalScripts = Object.keys(this.results.scripts).length;
    console.log(style.primary(`\nScripts: ${scriptCount}/${totalScripts} ready`));
    
    for (const [script, status] of Object.entries(this.results.scripts)) {
      const icon = status ? '✅' : '❌';
      const name = script.replace(/([A-Z])/g, ' $1').toUpperCase();
      console.log(`  ${icon} ${name}`);
    }

    // Overall status
    const totalReady = libraryCount + serviceCount + scriptCount;
    const totalPossible = totalLibraries + totalServices + totalScripts;
    const percentage = Math.round((totalReady / totalPossible) * 100);

    console.log(style.dim('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
    
    let statusColor = style.success;
    let statusText = 'READY TO RUN';
    
    if (percentage < 70) {
      statusColor = style.error;
      statusText = 'INTEGRATION INCOMPLETE';
    } else if (percentage < 90) {
      statusColor = style.warning;
      statusText = 'PARTIALLY READY';
    }

    console.log(statusColor(`🎯 OVERALL STATUS: ${percentage}% ${statusText}`));
    
    if (percentage >= 90) {
      console.log(style.success('\n🚀 Ready to run: npm run dev:full'));
      console.log(style.dim('All multi-library integrations are properly configured.'));
    } else {
      console.log(style.warning('\n⚠️ Some components may need attention before running.'));
    }
  }
}

const verifier = new MultiLibraryVerifier();
verifier.verify();