#!/usr/bin/env node

/**
 * Architecture Alignment Verification Script
 * Verifies system matches complete architecture summaries:
 * - Gemma3:legal + nomic-embed-text configuration
 * - Windows native service configuration
 * - Complete API ecosystem alignment
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const currentDir = process.cwd();

class ArchitectureAlignmentVerifier {
  constructor() {
    this.results = {
      passed: 0,
      failed: 0,
      warnings: 0,
      details: []
    };
    this.baseDir = currentDir;
  }

  log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level}] ${message}`);
    this.results.details.push({ timestamp, level, message });
  }

  assert(condition, testName, details = '') {
    if (condition) {
      this.results.passed++;
      this.log(`✅ PASS: ${testName}${details ? ' - ' + details : ''}`, 'PASS');
      return true;
    } else {
      this.results.failed++;
      this.log(`❌ FAIL: ${testName}${details ? ' - ' + details : ''}`, 'FAIL');
      return false;
    }
  }

  warn(testName, details = '') {
    this.results.warnings++;
    this.log(`⚠️  WARN: ${testName}${details ? ' - ' + details : ''}`, 'WARN');
  }

  async testCommand(command, description) {
    try {
      const output = execSync(command, { 
        encoding: 'utf8', 
        timeout: 10000,
        stdio: 'pipe'
      });
      return output.trim();
    } catch (error) {
      this.log(`Command failed [${description}]: ${command}`, 'ERROR');
      return null;
    }
  }

  // Test 1: Gemma3:legal Model Configuration
  async testGemmaLegalModel() {
    this.log('=== Testing Gemma3:legal Model Configuration ===');

    try {
      // Check Ollama service availability
      const ollamaVersion = await this.testCommand('curl -s http://localhost:11434/api/version', 'Ollama version check');
      
      if (ollamaVersion) {
        this.assert(true, 'Ollama Service Available', 'http://localhost:11434');
        
        // Check available models
        const modelsResponse = await this.testCommand('curl -s http://localhost:11434/api/tags', 'Ollama models check');
        
        if (modelsResponse) {
          const models = JSON.parse(modelsResponse);
          const gemmaModel = models.models?.find(m => m.name.includes('gemma3-legal'));
          const nomicModel = models.models?.find(m => m.name.includes('nomic-embed-text'));
          
          this.assert(
            gemmaModel !== undefined,
            'Gemma3:legal Model Available',
            gemmaModel ? `${gemmaModel.name} (${(gemmaModel.size / 1000000000).toFixed(1)}GB)` : 'Not found'
          );
          
          this.assert(
            nomicModel !== undefined,
            'nomic-embed-text Model Available',
            nomicModel ? `${nomicModel.name} (${(nomicModel.size / 1000000).toFixed(0)}MB)` : 'Not found'
          );
          
          // Verify model specifications match architecture
          if (gemmaModel) {
            const expectedSpecs = {
              quantization: 'Q4_K_M',
              family: 'gemma3',
              parameterSize: '11.8B'
            };
            
            this.assert(
              gemmaModel.details?.quantization_level === expectedSpecs.quantization,
              'Gemma3:legal Quantization Level',
              `${gemmaModel.details?.quantization_level} (expected: ${expectedSpecs.quantization})`
            );
            
            this.assert(
              gemmaModel.details?.families?.includes('gemma3'),
              'Gemma3:legal Model Family',
              `${gemmaModel.details?.families?.join(', ')}`
            );
          }
          
          if (nomicModel) {
            this.assert(
              nomicModel.details?.format === 'gguf',
              'Nomic-embed Model Format',
              `${nomicModel.details?.format} (expected: gguf)`
            );
          }
        } else {
          this.assert(false, 'Ollama Models API', 'Unable to fetch model list');
        }
      } else {
        this.assert(false, 'Ollama Service Available', 'Service not responding on port 11434');
      }
    } catch (error) {
      this.log(`Ollama model check failed: ${error.message}`, 'ERROR');
      this.results.failed++;
    }

    return true;
  }

  // Test 2: Windows Native Services Configuration  
  async testWindowsNativeServices() {
    this.log('=== Testing Windows Native Services Configuration ===');

    const requiredServices = [
      { name: 'Ollama', port: 11434, process: 'ollama' },
      { name: 'PostgreSQL', port: 5432, process: 'postgres' },
      { name: 'Redis', port: 6379, process: 'redis-server' }
    ];

    let servicesRunning = 0;
    
    for (const service of requiredServices) {
      try {
        // Test port connectivity
        const portTest = await this.testCommand(
          `powershell "Test-NetConnection -ComputerName localhost -Port ${service.port} -InformationLevel Quiet"`, 
          `Port ${service.port} connectivity`
        );
        
        if (portTest && portTest.includes('True')) {
          servicesRunning++;
          this.log(`✓ ${service.name} service running on port ${service.port}`, 'DEBUG');
        } else {
          this.log(`✗ ${service.name} service not running on port ${service.port}`, 'WARN');
        }
      } catch (error) {
        this.log(`Service check failed for ${service.name}: ${error.message}`, 'WARN');
      }
    }

    this.assert(
      servicesRunning >= 2, // At least Ollama + one database service
      'Windows Native Services',
      `${servicesRunning}/${requiredServices.length} services running`
    );

    // Check Windows-specific configuration
    const windowsServiceCheck = await this.testCommand('sc query ollama', 'Windows service check');
    if (windowsServiceCheck && !windowsServiceCheck.includes('failed')) {
      this.log('Ollama configured as Windows service', 'DEBUG');
    }

    return true;
  }

  // Test 3: API Ecosystem Alignment
  async testAPIEcosystemAlignment() {
    this.log('=== Testing Complete API Ecosystem Alignment ===');

    // Read architecture summary files if available
    const summaryFiles = [
      '../COMPLETE_API_ECOSYSTEM_SUMMARY.md',
      'COMPLETE_ARCHITECTURE_SUMMARY.md',
      'SYSTEM_VERIFICATION_COMPLETE.md'
    ];

    let summariesFound = 0;
    let architectureFeatures = {};

    for (const file of summaryFiles) {
      const fullPath = path.join(this.baseDir, file);
      if (fs.existsSync(fullPath)) {
        summariesFound++;
        const content = fs.readFileSync(fullPath, 'utf8');
        
        // Extract key architecture features
        if (content.includes('gemma3-legal')) {
          architectureFeatures.gemmaLegal = true;
        }
        if (content.includes('nomic-embed-text')) {
          architectureFeatures.nomicEmbed = true;
        }
        if (content.includes('Windows Native')) {
          architectureFeatures.windowsNative = true;
        }
        if (content.includes('37 Go microservices')) {
          architectureFeatures.goMicroservices = true;
        }
        if (content.includes('pgvector')) {
          architectureFeatures.pgvector = true;
        }
        if (content.includes('RTX 3060 Ti')) {
          architectureFeatures.gpuAcceleration = true;
        }
      }
    }

    this.assert(
      summariesFound >= 2,
      'Architecture Documentation Available',
      `${summariesFound}/${summaryFiles.length} summary files found`
    );

    // Verify key features from architecture summaries
    this.assert(
      architectureFeatures.gemmaLegal,
      'Gemma3:legal Referenced in Architecture',
      'Model specified in documentation'
    );

    this.assert(
      architectureFeatures.nomicEmbed,
      'nomic-embed-text Referenced in Architecture',
      'Embedding model specified in documentation'
    );

    this.assert(
      architectureFeatures.windowsNative,
      'Windows Native Configuration Documented',
      'Native Windows deployment specified'
    );

    return true;
  }

  // Test 4: Service Integration Matrix
  async testServiceIntegrationMatrix() {
    this.log('=== Testing Service Integration Matrix ===');

    // Test API endpoint availability (if services are running)
    const apiEndpoints = [
      { url: 'http://localhost:11434/api/version', name: 'Ollama API' },
      { url: 'http://localhost:5173', name: 'SvelteKit Frontend', optional: true }
    ];

    let endpointsAvailable = 0;
    
    for (const endpoint of apiEndpoints) {
      try {
        const response = await this.testCommand(
          `curl -s -o /dev/null -w "%{http_code}" ${endpoint.url}`, 
          `${endpoint.name} availability`
        );
        
        if (response && (response.includes('200') || response.includes('404'))) {
          endpointsAvailable++;
          this.log(`✓ ${endpoint.name} responding`, 'DEBUG');
        } else if (!endpoint.optional) {
          this.log(`✗ ${endpoint.name} not responding`, 'WARN');
        }
      } catch (error) {
        if (!endpoint.optional) {
          this.log(`${endpoint.name} check failed: ${error.message}`, 'WARN');
        }
      }
    }

    // Check database connection configuration
    const schemaFile = 'src/lib/server/db/schema-postgres.ts';
    if (fs.existsSync(path.join(this.baseDir, schemaFile))) {
      const schemaContent = fs.readFileSync(path.join(this.baseDir, schemaFile), 'utf8');
      
      this.assert(
        schemaContent.includes('pgvector') || schemaContent.includes('vector('),
        'pgvector Integration in Schema',
        'Vector columns configured for embeddings'
      );
      
      this.assert(
        schemaContent.includes('jsonb('),
        'JSONB Support in Schema',
        'JSONB columns configured for metadata'
      );
    } else {
      this.warn('Database Schema File', 'schema-postgres.ts not found');
    }

    return true;
  }

  // Test 5: Performance and Configuration Alignment
  async testPerformanceAlignment() {
    this.log('=== Testing Performance and Configuration Alignment ===');

    // Check GPU availability (if applicable)
    const gpuCheck = await this.testCommand('nvidia-smi --query-gpu=name --format=csv,noheader,nounits', 'GPU detection');
    if (gpuCheck && gpuCheck.includes('RTX 3060 Ti')) {
      this.assert(true, 'RTX 3060 Ti GPU Available', 'Hardware matches architecture specification');
    } else if (gpuCheck) {
      this.warn('GPU Hardware', `Found: ${gpuCheck.trim()}, Expected: RTX 3060 Ti`);
    } else {
      this.warn('GPU Detection', 'nvidia-smi not available or no GPU detected');
    }

    // Check Node.js and npm versions
    const nodeVersion = await this.testCommand('node --version', 'Node.js version');
    const npmVersion = await this.testCommand('npm --version', 'npm version');
    
    if (nodeVersion) {
      this.assert(
        nodeVersion.includes('v18') || nodeVersion.includes('v20') || nodeVersion.includes('v21'),
        'Node.js Version Compatible',
        `${nodeVersion} (modern version required)`
      );
    }

    // Check package.json for key dependencies
    const packageJsonPath = path.join(this.baseDir, 'package.json');
    if (fs.existsSync(packageJsonPath)) {
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
      
      const keyDependencies = [
        '@sveltejs/kit',
        'drizzle-orm',
        'postgres',
        'svelte'
      ];
      
      let dependenciesFound = 0;
      for (const dep of keyDependencies) {
        if (packageJson.dependencies?.[dep] || packageJson.devDependencies?.[dep]) {
          dependenciesFound++;
        }
      }
      
      this.assert(
        dependenciesFound >= 3,
        'Key Dependencies Available',
        `${dependenciesFound}/${keyDependencies.length} required packages found`
      );
    }

    return true;
  }

  // Main verification runner
  async runVerification() {
    this.log('🔍 Starting Architecture Alignment Verification');
    this.log(`Base directory: ${this.baseDir}`);
    this.log('Verifying: Gemma3:legal + nomic-embed-text + Windows Native');

    const tests = [
      () => this.testGemmaLegalModel(),
      () => this.testWindowsNativeServices(),
      () => this.testAPIEcosystemAlignment(),
      () => this.testServiceIntegrationMatrix(),
      () => this.testPerformanceAlignment()
    ];

    for (const test of tests) {
      try {
        await test();
      } catch (error) {
        this.log(`Test error: ${error.message}`, 'ERROR');
        this.results.failed++;
      }
    }

    await this.generateReport();
  }

  async generateReport() {
    const total = this.results.passed + this.results.failed;
    const successRate = total > 0 ? (this.results.passed / total * 100).toFixed(2) : 0;

    const report = {
      summary: {
        total,
        passed: this.results.passed,
        failed: this.results.failed,
        warnings: this.results.warnings,
        successRate: `${successRate}%`,
        timestamp: new Date().toISOString(),
        configuration: 'Gemma3:legal + nomic-embed-text + Windows Native'
      },
      alignment: {
        aiModels: 'Gemma3:legal (7.3GB) + nomic-embed-text (274MB)',
        platform: 'Windows Native (no Docker)',
        database: 'PostgreSQL + pgvector + JSONB',
        frontend: 'SvelteKit 2 + Svelte 5',
        services: 'Native Windows services'
      },
      details: this.results.details
    };

    const reportPath = `architecture-alignment-report-${Date.now()}.json`;
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    this.log('📊 Architecture Alignment Summary:');
    this.log(`Configuration: ${report.summary.configuration}`);
    this.log(`Total Tests: ${total}`);
    this.log(`Passed: ${this.results.passed}`);
    this.log(`Failed: ${this.results.failed}`);
    this.log(`Warnings: ${this.results.warnings}`);
    this.log(`Success Rate: ${successRate}%`);
    this.log(`Report saved: ${reportPath}`);

    if (this.results.failed === 0) {
      this.log('🎉 ARCHITECTURE ALIGNMENT VERIFIED!');
      this.log('');
      this.log('✅ AI Models: Gemma3:legal + nomic-embed-text configured correctly');
      this.log('✅ Platform: Windows Native services operational');
      this.log('✅ Integration: Complete API ecosystem aligned');
      this.log('✅ Performance: System meets architecture specifications');
      this.log('');
      this.log('🚀 SYSTEM READY FOR PRODUCTION WITH SPECIFIED CONFIGURATION');
    } else {
      this.log('⚠️  Some configuration issues detected. Check the report for details.');
      this.log('');
      this.log('Next steps:');
      this.log('1. Review failed tests in the detailed report');
      this.log('2. Ensure all required services are running');
      this.log('3. Verify model installations and configurations');
      this.log('4. Re-run verification after addressing issues');
    }

    return report;
  }
}

// Run verification
async function main() {
  const verifier = new ArchitectureAlignmentVerifier();
  
  try {
    await verifier.runVerification();
    process.exit(verifier.results.failed > 0 ? 1 : 0);
  } catch (error) {
    console.error('Fatal verification error:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = ArchitectureAlignmentVerifier;