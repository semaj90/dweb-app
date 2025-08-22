#!/usr/bin/env node

/**
 * Comprehensive Testing & Monitoring Framework
 * Legal AI Platform - Production Readiness Suite
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const http = require('http');

console.log('🚀 Legal AI Platform - Comprehensive Testing & Monitoring Framework\n');

class SystemHealthMonitor {
  constructor() {
    this.services = {
      frontend: { port: 5173, status: 'unknown' },
      uploadService: { port: 8093, status: 'unknown' },
      ragService: { port: 8094, status: 'unknown' },
      quicGateway: { port: 8447, status: 'unknown' },
      postgres: { port: 5432, status: 'unknown' },
      redis: { port: 6379, status: 'unknown' }
    };
    this.testResults = [];
  }

  // Check if a port is listening
  async checkPort(port) {
    return new Promise((resolve) => {
      const req = http.request({
        host: 'localhost',
        port: port,
        path: '/',
        method: 'GET',
        timeout: 2000
      }, (res) => {
        resolve({ status: 'listening', code: res.statusCode });
      });

      req.on('error', (err) => {
        if (err.code === 'ECONNREFUSED') {
          resolve({ status: 'closed', error: err.code });
        } else {
          resolve({ status: 'error', error: err.message });
        }
      });

      req.on('timeout', () => {
        req.destroy();
        resolve({ status: 'timeout' });
      });

      req.end();
    });
  }

  // Check all services
  async checkAllServices() {
    console.log('🔍 Checking Service Health...\n');
    
    for (const [service, config] of Object.entries(this.services)) {
      try {
        const result = await this.checkPort(config.port);
        config.status = result.status;
        config.lastCheck = new Date().toISOString();
        
        const statusIcon = result.status === 'listening' ? '✅' : '❌';
        console.log(`${statusIcon} ${service}: Port ${config.port} - ${result.status}`);
        
        if (result.status !== 'listening') {
          this.testResults.push({
            service,
            port: config.port,
            status: result.status,
            error: result.error,
            timestamp: new Date().toISOString()
          });
        }
      } catch (error) {
        console.log(`❌ ${service}: Port ${config.port} - Error: ${error.message}`);
        config.status = 'error';
      }
    }
    
    console.log('');
  }

  // Check TypeScript compilation
  async checkTypeScript() {
    console.log('🔍 Checking TypeScript Compilation...\n');
    
    try {
      // Check if we can find TypeScript
      const tsConfigPath = path.join(process.cwd(), 'sveltekit-frontend', 'tsconfig.json');
      if (!fs.existsSync(tsConfigPath)) {
        console.log('❌ TypeScript config not found');
        return false;
      }

      // Try to run TypeScript check
      const result = execSync('cd sveltekit-frontend && npx tsc --noEmit --skipLibCheck', {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      
      console.log('✅ TypeScript compilation successful');
      return true;
    } catch (error) {
      console.log('❌ TypeScript compilation failed');
      console.log('Error:', error.message);
      
      this.testResults.push({
        test: 'typescript',
        status: 'failed',
        error: error.message,
        timestamp: new Date().toISOString()
      });
      
      return false;
    }
  }

  // Check file structure
  checkFileStructure() {
    console.log('🔍 Checking File Structure...\n');
    
    const criticalFiles = [
      'sveltekit-frontend/src/routes/+page.svelte',
      'sveltekit-frontend/src/lib/components',
      'go-microservice/main.go',
      'package.json',
      'go.mod'
    ];

    for (const file of criticalFiles) {
      if (fs.existsSync(file)) {
        console.log(`✅ ${file}`);
      } else {
        console.log(`❌ ${file} - Missing`);
        this.testResults.push({
          test: 'file_structure',
          file,
          status: 'missing',
          timestamp: new Date().toISOString()
        });
      }
    }
    
    console.log('');
  }

  // Check dependencies
  checkDependencies() {
    console.log('🔍 Checking Dependencies...\n');
    
    try {
      const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      console.log(`✅ Root package.json: ${packageJson.name} v${packageJson.version}`);
      
      const frontendPackageJson = JSON.parse(fs.readFileSync('sveltekit-frontend/package.json', 'utf8'));
      console.log(`✅ Frontend package.json: ${frontendPackageJson.name} v${frontendPackageJson.version}`);
      
      // Check if node_modules exists
      if (fs.existsSync('sveltekit-frontend/node_modules')) {
        console.log('✅ Frontend node_modules exists');
      } else {
        console.log('❌ Frontend node_modules missing');
        this.testResults.push({
          test: 'dependencies',
          issue: 'node_modules_missing',
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      console.log(`❌ Error reading package files: ${error.message}`);
    }
    
    console.log('');
  }

  // Generate health report
  generateHealthReport() {
    console.log('📊 System Health Report\n');
    
    const operationalServices = Object.values(this.services).filter(s => s.status === 'listening').length;
    const totalServices = Object.keys(this.services).length;
    const healthPercentage = Math.round((operationalServices / totalServices) * 100);
    
    console.log(`🏥 Overall Health: ${healthPercentage}% (${operationalServices}/${totalServices} services operational)\n`);
    
    if (this.testResults.length > 0) {
      console.log('🚨 Issues Found:');
      this.testResults.forEach(result => {
        console.log(`   • ${result.service || result.test}: ${result.status} - ${result.error || 'See details above'}`);
      });
    } else {
      console.log('✅ No critical issues found');
    }
    
    console.log('\n🎯 Recommendations:');
    
    if (healthPercentage < 100) {
      console.log('   1. Resolve service port conflicts');
      console.log('   2. Restart failed services');
      console.log('   3. Check service logs for errors');
    }
    
    if (this.testResults.some(r => r.test === 'typescript')) {
      console.log('   4. Fix remaining TypeScript errors');
      console.log('   5. Reinstall dependencies if needed');
    }
    
    console.log('   6. Run comprehensive integration tests');
    console.log('   7. Implement monitoring dashboard');
  }

  // Run all checks
  async runFullDiagnostic() {
    console.log('🚀 Starting Comprehensive System Diagnostic...\n');
    
    await this.checkAllServices();
    this.checkFileStructure();
    this.checkDependencies();
    await this.checkTypeScript();
    
    console.log('='.repeat(60));
    this.generateHealthReport();
    
    // Save results to file
    const reportPath = `system-health-report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      services: this.services,
      testResults: this.testResults,
      summary: {
        operationalServices: Object.values(this.services).filter(s => s.status === 'listening').length,
        totalServices: Object.keys(this.services).length,
        issuesFound: this.testResults.length
      }
    }, null, 2));
    
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
  }
}

// Run if called directly
if (require.main === module) {
  const monitor = new SystemHealthMonitor();
  monitor.runFullDiagnostic().catch(console.error);
}

module.exports = SystemHealthMonitor;
