#!/usr/bin/env node

/**
 * AutoSolve Runner - Comprehensive npm integration for AutoSolve system
 * Integrates with GPU orchestrator, Enhanced RAG, and VS Code MCP extension
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class AutoSolveRunner {
  constructor() {
    this.services = {
      gpu_orchestrator: 'http://localhost:8095',
      enhanced_rag: 'http://localhost:8094', 
      sveltekit: 'http://localhost:5173',
      ollama: 'http://localhost:11434',
      postgresql: 'localhost:5432',
      redis: 'localhost:6379'
    };
    this.results = [];
  }

  async checkServices() {
    console.log('🔍 Checking AutoSolve services...\n');
    
    for (const [name, url] of Object.entries(this.services)) {
      try {
        if (name === 'postgresql' || name === 'redis') {
          console.log(`✅ ${name}: Assumed running (${url})`);
          continue;
        }
        
        const response = await fetch(`${url}/api/status`).catch(() => 
          fetch(`${url}/health`).catch(() => 
            fetch(`${url}/api/tags`).catch(() => null)
          )
        );
        
        if (response && response.ok) {
          console.log(`✅ ${name}: Running (${url})`);
        } else {
          console.log(`⚠️  ${name}: Available but may need restart (${url})`);
        }
      } catch (error) {
        console.log(`❌ ${name}: Not responding (${url})`);
      }
    }
  }

  async runTypeScriptCheck() {
    console.log('\n🔧 Running TypeScript check...\n');
    
    return new Promise((resolve) => {
      const checkProcess = spawn('npm', ['run', 'check'], {
        stdio: 'pipe',
        shell: true
      });
      
      let stdout = '';
      let stderr = '';
      
      checkProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      checkProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      checkProcess.on('close', (code) => {
        const output = stdout + stderr;
        const errorMatch = output.match(/(\d+)\s+errors?/i);
        const warningMatch = output.match(/(\d+)\s+warnings?/i);
        
        const errors = errorMatch ? parseInt(errorMatch[1]) : 0;
        const warnings = warningMatch ? parseInt(warningMatch[1]) : 0;
        
        console.log(`📊 TypeScript Check Results:`);
        console.log(`   Errors: ${errors}`);
        console.log(`   Warnings: ${warnings}`);
        console.log(`   Exit Code: ${code}`);
        
        this.results.push({
          type: 'typescript_check',
          errors,
          warnings,
          exit_code: code,
          timestamp: new Date().toISOString()
        });
        
        resolve({ errors, warnings, code });
      });
    });
  }

  async triggerAutoSolveQueries() {
    console.log('\n🤖 Triggering AutoSolve processing queries...\n');
    
    const queries = [
      {
        name: 'UI Component Fixes',
        query: 'fix svelte 5 runes TypeScript errors in UI components',
        context: 'Component prop binding and event handler optimization'
      },
      {
        name: 'Module Import Resolution', 
        query: 'resolve TypeScript module import and type definition errors',
        context: 'Missing dependencies and type resolution issues'
      },
      {
        name: 'Database Integration',
        query: 'fix PostgreSQL pgvector and Drizzle ORM integration errors',
        context: 'Vector service and database schema validation'
      },
      {
        name: 'VS Code Extension',
        query: 'optimize VS Code MCP extension command registration and functionality',
        context: 'Extension build and MCP server integration'
      }
    ];
    
    for (const queryConfig of queries) {
      try {
        console.log(`🔄 Processing: ${queryConfig.name}`);
        
        const response = await fetch(`${this.services.gpu_orchestrator}/api/enhanced-rag`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: queryConfig.query,
            context: queryConfig.context,
            enable_som: true,
            enable_attention: true,
            priority: 'high'
          })
        });
        
        if (response.ok) {
          const result = await response.json();
          console.log(`   ✅ Processed in ${result.processing_time_ms}ms (GPU: ${result.gpu_accelerated})`);
          
          this.results.push({
            type: 'autosolve_query',
            name: queryConfig.name,
            processing_time: result.processing_time_ms,
            gpu_accelerated: result.gpu_accelerated,
            success: true,
            timestamp: new Date().toISOString()
          });
        } else {
          console.log(`   ❌ Failed: ${response.status}`);
        }
      } catch (error) {
        console.log(`   ❌ Error: ${error.message}`);
      }
    }
  }

  async testVSCodeExtension() {
    console.log('\n🔌 Testing VS Code MCP Extension...\n');
    
    return new Promise((resolve) => {
      const testProcess = spawn('node', ['test-mcp-extension.mjs'], {
        stdio: 'pipe',
        shell: true
      });
      
      let output = '';
      
      testProcess.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      testProcess.stderr.on('data', (data) => {
        output += data.toString();
      });
      
      testProcess.on('close', (code) => {
        const scoreMatch = output.match(/(\d+)\/(\d+)\s+tests\s+passed/i);
        const percentMatch = output.match(/(\d+)%/i);
        
        const passed = scoreMatch ? parseInt(scoreMatch[1]) : 0;
        const total = scoreMatch ? parseInt(scoreMatch[2]) : 6;
        const percentage = percentMatch ? parseInt(percentMatch[1]) : 0;
        
        console.log(`📊 VS Code Extension Test Results:`);
        console.log(`   Tests Passed: ${passed}/${total}`);
        console.log(`   Success Rate: ${percentage}%`);
        console.log(`   Status: ${percentage >= 80 ? '✅ Excellent' : percentage >= 60 ? '⚠️ Good' : '❌ Needs Improvement'}`);
        
        this.results.push({
          type: 'vscode_extension_test',
          tests_passed: passed,
          total_tests: total,
          success_rate: percentage,
          exit_code: code,
          timestamp: new Date().toISOString()
        });
        
        resolve({ passed, total, percentage });
      });
    });
  }

  async generateSummaryReport() {
    console.log('\n📈 AutoSolve System Summary Report\n');
    console.log('=' .repeat(60));
    
    const typescriptResult = this.results.find(r => r.type === 'typescript_check');
    const autosolveQueries = this.results.filter(r => r.type === 'autosolve_query');
    const vscodeResult = this.results.find(r => r.type === 'vscode_extension_test');
    
    // TypeScript Status
    if (typescriptResult) {
      console.log(`🔧 TypeScript Compilation:`);
      console.log(`   Errors: ${typescriptResult.errors} | Warnings: ${typescriptResult.warnings}`);
      console.log(`   Status: ${typescriptResult.errors < 50 ? '✅ Excellent' : typescriptResult.errors < 200 ? '⚠️ Improving' : '❌ Needs Work'}`);
    }
    
    // AutoSolve Processing
    if (autosolveQueries.length > 0) {
      const successfulQueries = autosolveQueries.filter(q => q.success);
      const avgProcessingTime = autosolveQueries.reduce((sum, q) => sum + (q.processing_time || 0), 0) / autosolveQueries.length;
      
      console.log(`\n🤖 AutoSolve Processing:`);
      console.log(`   Queries Processed: ${successfulQueries.length}/${autosolveQueries.length}`);
      console.log(`   Average Processing Time: ${avgProcessingTime.toFixed(2)}ms`);
      console.log(`   GPU Acceleration: ${autosolveQueries.every(q => q.gpu_accelerated) ? '✅ Active' : '⚠️ Partial'}`);
    }
    
    // VS Code Extension
    if (vscodeResult) {
      console.log(`\n🔌 VS Code Extension:`);
      console.log(`   Functionality: ${vscodeResult.success_rate}%`);
      console.log(`   Tests: ${vscodeResult.tests_passed}/${vscodeResult.total_tests} passing`);
      console.log(`   Status: ${vscodeResult.success_rate >= 80 ? '✅ Production Ready' : vscodeResult.success_rate >= 60 ? '⚠️ Functional' : '❌ Needs Fixes'}`);
    }
    
    // Overall System Health
    console.log(`\n🎯 Overall System Status:`);
    const overallHealth = this.calculateOverallHealth();
    console.log(`   Health Score: ${overallHealth}%`);
    console.log(`   Status: ${overallHealth >= 90 ? '🎉 Excellent' : overallHealth >= 70 ? '✅ Good' : overallHealth >= 50 ? '⚠️ Fair' : '❌ Needs Attention'}`);
    
    console.log('\n' + '=' .repeat(60));
    console.log(`✨ AutoSolve System Report Generated: ${new Date().toLocaleString()}`);
    
    // Save detailed results
    fs.writeFileSync(
      path.join(__dirname, '..', 'autosolve-results.json'),
      JSON.stringify(this.results, null, 2)
    );
    
    return overallHealth;
  }

  calculateOverallHealth() {
    const weights = {
      typescript: 0.3,
      autosolve: 0.4, 
      vscode: 0.3
    };
    
    let score = 0;
    
    // TypeScript score (inverse of error count, max 100)
    const typescriptResult = this.results.find(r => r.type === 'typescript_check');
    if (typescriptResult) {
      const tsScore = Math.max(0, 100 - (typescriptResult.errors / 10));
      score += tsScore * weights.typescript;
    }
    
    // AutoSolve score
    const autosolveQueries = this.results.filter(r => r.type === 'autosolve_query');
    if (autosolveQueries.length > 0) {
      const successRate = autosolveQueries.filter(q => q.success).length / autosolveQueries.length;
      score += (successRate * 100) * weights.autosolve;
    }
    
    // VS Code score
    const vscodeResult = this.results.find(r => r.type === 'vscode_extension_test');
    if (vscodeResult) {
      score += vscodeResult.success_rate * weights.vscode;
    }
    
    return Math.round(score);
  }

  async run() {
    console.log('🚀 Starting AutoSolve Comprehensive Test Suite\n');
    console.log('=' .repeat(60));
    
    try {
      // 1. Check service availability
      await this.checkServices();
      
      // 2. Run TypeScript check
      await this.runTypeScriptCheck();
      
      // 3. Trigger AutoSolve queries
      await this.triggerAutoSolveQueries();
      
      // 4. Test VS Code extension
      await this.testVSCodeExtension();
      
      // 5. Generate summary report
      const overallHealth = await this.generateSummaryReport();
      
      // Return exit code based on health
      process.exit(overallHealth >= 70 ? 0 : 1);
      
    } catch (error) {
      console.error('❌ AutoSolve runner failed:', error);
      process.exit(1);
    }
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const runner = new AutoSolveRunner();
  runner.run();
}

export default AutoSolveRunner;