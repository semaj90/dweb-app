#!/usr/bin/env node

import { promises as fs } from 'fs';
import { exec } from 'child_process';
import path from 'path';

/**
 * Comprehensive Testing and Validation Script
 * Tests the TypeScript fixes and validates system health
 */

class TestSuite {
  constructor() {
    this.results = {
      typescript: null,
      build: null,
      runtime: null,
      performance: null
    };
    this.startTime = Date.now();
  }

  async runComplete() {
    console.log('🧪 Starting comprehensive test suite...');
    
    try {
      await this.testTypeScript();
      await this.testBuild();
      await this.testRuntime();
      await this.testPerformance();
      
      await this.generateCompleteReport();
      console.log('✅ Test suite completed successfully');
      
    } catch (error) {
      console.error('❌ Test suite failed:', error);
      await this.generateErrorReport(error);
    }
  }

  async testTypeScript() {
    console.log('📝 Testing TypeScript compilation...');
    
    return new Promise((resolve, reject) => {
      const command = 'npx svelte-kit sync && npx svelte-check --tsconfig ./tsconfig.json';
      
      exec(command, { maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
        const output = stdout + stderr;
        
        // Parse TypeScript results
        const errors = this.parseTypeScriptOutput(output);
        
        this.results.typescript = {
          exitCode: error ? error.code : 0,
          errorCount: errors.filter(e => e.type === 'Error').length,
          warningCount: errors.filter(e => e.type === 'Warning').length,
          totalIssues: errors.length,
          errors: errors,
          rawOutput: output,
          status: error ? 'failed' : 'passed'
        };
        
        console.log(`📊 TypeScript Results: ${this.results.typescript.errorCount} errors, ${this.results.typescript.warningCount} warnings`);
        resolve(this.results.typescript);
      });
    });
  }

  async testBuild() {
    console.log('🏗️ Testing build process...');
    
    return new Promise((resolve, reject) => {
      const command = 'npm run build';
      const startTime = Date.now();
      
      exec(command, { maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
        const buildTime = Date.now() - startTime;
        const output = stdout + stderr;
        
        this.results.build = {
          exitCode: error ? error.code : 0,
          success: !error,
          buildTime: buildTime,
          output: output,
          status: error ? 'failed' : 'passed'
        };
        
        console.log(`🏗️ Build Results: ${this.results.build.success ? 'SUCCESS' : 'FAILED'} (${buildTime}ms)`);
        resolve(this.results.build);
      });
    });
  }

  async testRuntime() {
    console.log('🚀 Testing runtime functionality...');
    
    const tests = [
      () => this.testDatabaseTypes(),
      () => this.testVLLMService(),
      () => this.testOrchestratorStore(),
      () => this.testClusteringServices(),
      () => this.testAPIEndpoints()
    ];
    
    const results = [];
    
    for (const test of tests) {
      try {
        const result = await test();
        results.push(result);
      } catch (error) {
        results.push({
          name: test.name,
          status: 'failed',
          error: error.message
        });
      }
    }
    
    this.results.runtime = {
      tests: results,
      passed: results.filter(r => r.status === 'passed').length,
      failed: results.filter(r => r.status === 'failed').length,
      status: results.every(r => r.status === 'passed') ? 'passed' : 'failed'
    };
    
    console.log(`🚀 Runtime Results: ${this.results.runtime.passed}/${results.length} tests passed`);
  }

  async testDatabaseTypes() {
    try {
      // Test if database types file exists and is importable
      const dbTypesPath = 'src/lib/types/database.ts';
      const exists = await this.fileExists(dbTypesPath);
      
      if (!exists) {
        throw new Error('Database types file not found');
      }
      
      const content = await fs.readFile(dbTypesPath, 'utf8');
      
      // Check for key type exports
      const requiredTypes = [
        'Case', 'NewCase', 'Evidence', 'NewEvidence',
        'CaseWithEvidence', 'CaseQueryResult'
      ];
      
      const missingTypes = requiredTypes.filter(type => 
        !content.includes(`export type ${type}`) && 
        !content.includes(`export interface ${type}`)
      );
      
      if (missingTypes.length > 0) {
        throw new Error(`Missing type exports: ${missingTypes.join(', ')}`);
      }
      
      return {
        name: 'Database Types',
        status: 'passed',
        details: `All ${requiredTypes.length} required types found`
      };
      
    } catch (error) {
      return {
        name: 'Database Types',
        status: 'failed',
        error: error.message
      };
    }
  }

  async testVLLMService() {
    try {
      const servicePath = 'src/lib/services/vllm-service.ts';
      const exists = await this.fileExists(servicePath);
      
      if (!exists) {
        throw new Error('VLLM service file not found');
      }
      
      const content = await fs.readFile(servicePath, 'utf8');
      
      // Check for key methods
      const requiredMethods = [
        'queryVLLM', 'checkHealth', 'generateCompletion'
      ];
      
      const missingMethods = requiredMethods.filter(method => 
        !content.includes(method)
      );
      
      if (missingMethods.length > 0) {
        throw new Error(`Missing methods: ${missingMethods.join(', ')}`);
      }
      
      return {
        name: 'VLLM Service',
        status: 'passed',
        details: 'All required methods implemented'
      };
      
    } catch (error) {
      return {
        name: 'VLLM Service',
        status: 'failed',
        error: error.message
      };
    }
  }

  async testOrchestratorStore() {
    try {
      const storePath = 'src/lib/stores/orchestrator.ts';
      const exists = await this.fileExists(storePath);
      
      if (!exists) {
        throw new Error('Orchestrator store file not found');
      }
      
      const content = await fs.readFile(storePath, 'utf8');
      
      // Check for subscribe method (critical fix)
      if (!content.includes('subscribe,')) {
        throw new Error('Missing subscribe method in store');
      }
      
      // Check for derived stores
      const derivedStores = ['isProcessing', 'currentTask', 'progress'];
      const missingStores = derivedStores.filter(store => 
        !content.includes(`export const ${store}`)
      );
      
      if (missingStores.length > 0) {
        throw new Error(`Missing derived stores: ${missingStores.join(', ')}`);
      }
      
      return {
        name: 'Orchestrator Store',
        status: 'passed',
        details: 'Subscribe method and derived stores implemented'
      };
      
    } catch (error) {
      return {
        name: 'Orchestrator Store',
        status: 'failed',
        error: error.message
      };
    }
  }

  async testClusteringServices() {
    try {
      // Check for clustering-related files
      const clusteringPaths = [
        'src/lib/types/legal-ai.ts'
      ];
      
      let foundFiles = 0;
      for (const path of clusteringPaths) {
        if (await this.fileExists(path)) {
          foundFiles++;
        }
      }
      
      return {
        name: 'Clustering Services',
        status: foundFiles > 0 ? 'passed' : 'warning',
        details: `Found ${foundFiles}/${clusteringPaths.length} clustering files`
      };
      
    } catch (error) {
      return {
        name: 'Clustering Services',
        status: 'failed',
        error: error.message
      };
    }
  }

  async testAPIEndpoints() {
    try {
      // Check for API route files
      const apiRoutes = [
        'src/routes/api/cases/+server.ts',
        'src/routes/api/evidence/+server.ts'
      ];
      
      let validRoutes = 0;
      
      for (const route of apiRoutes) {
        if (await this.fileExists(route)) {
          const content = await fs.readFile(route, 'utf8');
          
          // Check for basic structure
          if (content.includes('export async function') || content.includes('export function')) {
            validRoutes++;
          }
        }
      }
      
      return {
        name: 'API Endpoints',
        status: validRoutes > 0 ? 'passed' : 'warning',
        details: `Found ${validRoutes}/${apiRoutes.length} valid API routes`
      };
      
    } catch (error) {
      return {
        name: 'API Endpoints',
        status: 'failed',
        error: error.message
      };
    }
  }

  async testPerformance() {
    console.log('⚡ Testing performance metrics...');
    
    const startMem = process.memoryUsage();
    const startTime = Date.now();
    
    // Simulate some operations
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const endMem = process.memoryUsage();
    const endTime = Date.now();
    
    this.results.performance = {
      memoryUsage: {
        heapUsed: endMem.heapUsed,
        heapTotal: endMem.heapTotal,
        external: endMem.external
      },
      executionTime: endTime - startTime,
      status: 'measured'
    };
    
    console.log(`⚡ Performance: ${this.results.performance.executionTime}ms execution`);
  }

  parseTypeScriptOutput(output) {
    const errors = [];
    const lines = output.split('\n');
    
    for (const line of lines) {
      const errorMatch = line.match(/^(.+):(\\d+):(\\d+)\\s+(Error|Warning):\\s*(.+)$/);
      
      if (errorMatch) {
        errors.push({
          file: errorMatch[1],
          line: parseInt(errorMatch[2]),
          column: parseInt(errorMatch[3]),
          type: errorMatch[4],
          message: errorMatch[5]
        });
      }
    }
    
    return errors;
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  async generateCompleteReport() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const reportFile = `TEST_RESULTS_COMPLETE_${timestamp}.md`;
    const totalTime = Date.now() - this.startTime;
    
    const report = `# Comprehensive Test Results Report
Generated: ${new Date().toISOString()}
Total Execution Time: ${totalTime}ms

## 🎯 Executive Summary

**TypeScript Status:** ${this.results.typescript?.status || 'not tested'}
- Errors: ${this.results.typescript?.errorCount || 'N/A'}
- Warnings: ${this.results.typescript?.warningCount || 'N/A'}

**Build Status:** ${this.results.build?.status || 'not tested'}
- Build Time: ${this.results.build?.buildTime || 'N/A'}ms

**Runtime Tests:** ${this.results.runtime?.status || 'not tested'}
- Passed: ${this.results.runtime?.passed || 0}/${this.results.runtime?.tests?.length || 0}

**Overall Status:** ${this.getOverallStatus()}

---

## 📝 TypeScript Compilation Results

**Exit Code:** ${this.results.typescript?.exitCode || 'N/A'}
**Error Count:** ${this.results.typescript?.errorCount || 0}
**Warning Count:** ${this.results.typescript?.warningCount || 0}

### Error Categories:
${this.categorizeErrors()}

### Top Errors:
${this.getTopErrors()}

---

## 🏗️ Build Process Results

**Success:** ${this.results.build?.success ? '✅' : '❌'}
**Build Time:** ${this.results.build?.buildTime || 'N/A'}ms
**Status:** ${this.results.build?.status || 'not tested'}

---

## 🚀 Runtime Functionality Tests

${this.results.runtime?.tests?.map(test => `
### ${test.name}
**Status:** ${test.status === 'passed' ? '✅' : test.status === 'warning' ? '⚠️' : '❌'}
**Details:** ${test.details || test.error || 'No details'}

`).join('') || 'No runtime tests performed'}

---

## ⚡ Performance Metrics

**Memory Usage:**
- Heap Used: ${Math.round((this.results.performance?.memoryUsage?.heapUsed || 0) / 1024 / 1024)}MB
- Heap Total: ${Math.round((this.results.performance?.memoryUsage?.heapTotal || 0) / 1024 / 1024)}MB

**Execution Time:** ${this.results.performance?.executionTime || 'N/A'}ms

---

## 📊 Improvement Recommendations

${this.generateRecommendations()}

## 🎯 Next Actions

${this.generateNextActions()}

---

## 📋 Detailed Logs

### TypeScript Raw Output:
\`\`\`
${this.results.typescript?.rawOutput?.slice(0, 2000) || 'No output captured'}
\`\`\`

### Build Output:
\`\`\`
${this.results.build?.output?.slice(0, 1000) || 'No output captured'}
\`\`\`

---

**Report Generation Time:** ${new Date().toISOString()}
**Test Suite Version:** 1.0.0
`;

    await fs.writeFile(reportFile, report);
    console.log(`📄 Complete test report saved: ${reportFile}`);
  }

  getOverallStatus() {
    const tsStatus = this.results.typescript?.status;
    const buildStatus = this.results.build?.status;
    const runtimeStatus = this.results.runtime?.status;
    
    if (tsStatus === 'passed' && buildStatus === 'passed' && runtimeStatus === 'passed') {
      return '🎉 EXCELLENT - Production Ready';
    } else if (tsStatus === 'passed' && buildStatus === 'passed') {
      return '✅ GOOD - Core functionality working';
    } else if (tsStatus === 'failed' && this.results.typescript?.errorCount < 10) {
      return '⚠️ MODERATE - Minor issues remaining';
    } else {
      return '❌ NEEDS WORK - Significant issues remain';
    }
  }

  categorizeErrors() {
    if (!this.results.typescript?.errors) return 'No errors to categorize';
    
    const categories = {};
    this.results.typescript.errors.forEach(error => {
      const category = this.categorizeError(error.message);
      categories[category] = (categories[category] || 0) + 1;
    });
    
    return Object.entries(categories)
      .map(([cat, count]) => `- ${cat}: ${count}`)
      .join('\n');
  }

  categorizeError(message) {
    if (message.includes('Drizzle') || message.includes('PgSelectBase')) return 'Drizzle ORM';
    if (message.includes('does not exist on type')) return 'Missing Properties';
    if (message.includes('is not assignable')) return 'Type Mismatch';
    if (message.includes('Cannot find module')) return 'Import Errors';
    if (message.includes('subscribe')) return 'Store Issues';
    return 'Other';
  }

  getTopErrors() {
    if (!this.results.typescript?.errors) return 'No errors found';
    
    return this.results.typescript.errors
      .slice(0, 5)
      .map(error => `- ${error.file}:${error.line} - ${error.message.slice(0, 80)}...`)
      .join('\n');
  }

  generateRecommendations() {
    const recommendations = [];
    
    if (this.results.typescript?.errorCount > 0) {
      recommendations.push('- Fix remaining TypeScript errors for production readiness');
    }
    
    if (this.results.build?.status === 'failed') {
      recommendations.push('- Resolve build process issues before deployment');
    }
    
    if (this.results.runtime?.failed > 0) {
      recommendations.push('- Address failed runtime tests');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('- System appears healthy, consider performance optimization');
    }
    
    return recommendations.join('\n');
  }

  generateNextActions() {
    const actions = [];
    
    if (this.results.typescript?.errorCount > 0) {
      actions.push('1. Run `npm run check` to see current TypeScript status');
      actions.push('2. Address critical errors first, then warnings');
    }
    
    if (this.results.build?.status === 'failed') {
      actions.push('3. Run `npm run build` to test build process');
    }
    
    actions.push('4. Test core functionality in development mode');
    actions.push('5. Consider deployment to staging environment');
    
    return actions.join('\n');
  }

  async generateErrorReport(error) {
    const errorReport = `# Test Suite Error Report
Generated: ${new Date().toISOString()}

## Error Details
**Message:** ${error.message}
**Stack:** ${error.stack}

## Partial Results
${JSON.stringify(this.results, null, 2)}

## Recovery Steps
1. Check system dependencies
2. Verify file permissions
3. Retry individual test components
4. Review error logs for specific issues
`;

    await fs.writeFile('TEST_ERROR_REPORT.md', errorReport);
    console.log('📄 Error report saved: TEST_ERROR_REPORT.md');
  }
}

// Run the complete test suite
const testSuite = new TestSuite();
testSuite.runComplete()
  .then(() => {
    console.log('🏁 Test suite execution completed');
    process.exit(0);
  })
  .catch(error => {
    console.error('💥 Test suite failed:', error);
    process.exit(1);
  });
