#!/usr/bin/env node

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

class TestRunner {
  constructor() {
    this.results = {
      passed: 0,
      failed: 0,
      skipped: 0,
      total: 0
    };
    this.startTime = Date.now();
  }

  log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
  }

  async runCommand(command, args = [], options = {}) {
    return new Promise((resolve) => {
      const process = spawn(command, args, {
        stdio: 'pipe',
        shell: true,
        ...options
      });

      let output = '';
      let errorOutput = '';

      process.stdout.on('data', (data) => {
        output += data.toString();
      });

      process.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      process.on('close', (code) => {
        resolve({
          code,
          output,
          errorOutput,
          success: code === 0
        });
      });
    });
  }

  async checkPrerequisites() {
    this.log('\\n🔍 Checking Prerequisites', 'cyan');
    this.log('=' .repeat(50), 'cyan');

    const checks = [
      {
        name: 'Node.js',
        command: 'node',
        args: ['--version'],
        required: true
      },
      {
        name: 'npm',
        command: 'npm',
        args: ['--version'],
        required: true
      },
      {
        name: 'PostgreSQL',
        command: '"C:\\\\Program Files\\\\PostgreSQL\\\\17\\\\bin\\\\psql.exe"',
        args: ['-U', 'postgres', '-h', 'localhost', '-c', 'SELECT version();'],
        required: false,
        env: { PGPASSWORD: '123456' }
      },
      {
        name: 'Ollama',
        command: 'curl',
        args: ['-s', 'http://localhost:11434/api/version'],
        required: false
      }
    ];

    let allRequiredPassed = true;

    for (const check of checks) {
      const result = await this.runCommand(check.command, check.args, {
        env: { ...process.env, ...check.env },
        timeout: 5000
      });

      if (result.success) {
        this.log(`✅ ${check.name} - Available`, 'green');
        if (check.name === 'Node.js') {
          this.log(`   Version: ${result.output.trim()}`, 'green');
        }
      } else {
        const status = check.required ? 'REQUIRED - MISSING' : 'Optional - Not Available';
        const color = check.required ? 'red' : 'yellow';
        this.log(`${check.required ? '❌' : '⚠️'} ${check.name} - ${status}`, color);
        
        if (check.required) {
          allRequiredPassed = false;
        }
      }
    }

    if (!allRequiredPassed) {
      this.log('\\n❌ Required prerequisites missing. Please install missing dependencies.', 'red');
      process.exit(1);
    }

    this.log('\\n✅ Prerequisites check completed', 'green');
    return true;
  }

  async runTestSuite(suiteName, testFile, options = {}) {
    this.log(`\\n🧪 Running ${suiteName}`, 'blue');
    this.log('-'.repeat(50), 'blue');

    const playwrightArgs = [
      'test',
      testFile,
      '--reporter=list',
      '--timeout=30000'
    ];

    if (options.headed) {
      playwrightArgs.push('--headed');
    }

    if (options.browser) {
      playwrightArgs.push(`--project=${options.browser}`);
    }

    const result = await this.runCommand('npx', ['playwright', ...playwrightArgs], {
      cwd: __dirname,
      timeout: 120000 // 2 minutes timeout
    });

    // Parse Playwright output for results
    const output = result.output + result.errorOutput;
    
    // Count test results from Playwright output
    const passedMatches = output.match(/✓|passed/gi) || [];
    const failedMatches = output.match(/✗|failed/gi) || [];
    const skippedMatches = output.match(/skipped/gi) || [];

    const suitePassed = passedMatches.length;
    const suiteFailed = failedMatches.length;
    const suiteSkipped = skippedMatches.length;

    this.results.passed += suitePassed;
    this.results.failed += suiteFailed;
    this.results.skipped += suiteSkipped;
    this.results.total += (suitePassed + suiteFailed + suiteSkipped);

    if (result.success) {
      this.log(`✅ ${suiteName} completed successfully`, 'green');
      this.log(`   Passed: ${suitePassed}, Failed: ${suiteFailed}, Skipped: ${suiteSkipped}`, 'green');
    } else {
      this.log(`⚠️ ${suiteName} completed with issues`, 'yellow');
      this.log(`   Passed: ${suitePassed}, Failed: ${suiteFailed}, Skipped: ${suiteSkipped}`, 'yellow');
      
      // Show last few lines of error output
      const errorLines = result.errorOutput.split('\\n').slice(-5).filter(line => line.trim());
      if (errorLines.length > 0) {
        this.log('   Last errors:', 'red');
        errorLines.forEach(line => this.log(`     ${line}`, 'red'));
      }
    }

    return result;
  }

  async runDevelopmentStatusCheck() {
    this.log('\\n🔧 Running Development Status Check', 'cyan');
    this.log('-'.repeat(50), 'cyan');

    const statusResult = await this.runCommand('npm', ['run', 'dev:status'], {
      cwd: __dirname,
      timeout: 15000
    });

    if (statusResult.success) {
      this.log('✅ Development environment status check completed', 'green');
      
      // Extract key information from status output
      const output = statusResult.output;
      if (output.includes('PostgreSQL')) {
        this.log('   📊 PostgreSQL status included', 'green');
      }
      if (output.includes('Ollama')) {
        this.log('   🤖 Ollama status included', 'green');
      }
      if (output.includes('SvelteKit')) {
        this.log('   ⚡ SvelteKit status included', 'green');
      }
    } else {
      this.log('⚠️ Development status check had issues', 'yellow');
    }

    return statusResult;
  }

  async generateTestReport() {
    const endTime = Date.now();
    const duration = ((endTime - this.startTime) / 1000).toFixed(2);

    const report = {
      timestamp: new Date().toISOString(),
      duration: `${duration}s`,
      results: this.results,
      environment: {
        node_version: process.version,
        platform: process.platform,
        cwd: process.cwd()
      }
    };

    // Ensure reports directory exists
    try {
      await fs.mkdir('.test-reports', { recursive: true });
    } catch (error) {
      // Directory might already exist
    }

    const reportFile = path.join('.test-reports', `development-tests-${Date.now()}.json`);
    await fs.writeFile(reportFile, JSON.stringify(report, null, 2));

    this.log(`\\n📊 Test report saved to: ${reportFile}`, 'magenta');
    return report;
  }

  async run() {
    this.log('🚀 Legal AI Development Environment Test Suite', 'bright');
    this.log('='.repeat(60), 'bright');
    this.log(`Started at: ${new Date().toLocaleString()}`, 'cyan');

    try {
      // Check prerequisites
      await this.checkPrerequisites();

      // Run development status check
      await this.runDevelopmentStatusCheck();

      // Define test suites
      const testSuites = [
        {
          name: 'Development Environment',
          file: 'tests/development-environment.spec.ts',
          critical: true
        },
        {
          name: 'PostgreSQL CRUD & pgvector',
          file: 'tests/postgresql-crud-pgvector.spec.ts',
          critical: true
        },
        {
          name: 'Enhanced RAG System',
          file: 'tests/enhanced-rag-system.spec.ts',
          critical: false
        },
        {
          name: 'Claude Vector CLI',
          file: 'tests/claude-vector-cli.spec.ts',
          critical: false
        },
        {
          name: 'SvelteKit Endpoints',
          file: 'tests/sveltekit-endpoints.spec.ts',
          critical: false
        }
      ];

      // Run test suites
      for (const suite of testSuites) {
        try {
          await this.runTestSuite(suite.name, suite.file);
        } catch (error) {
          this.log(`❌ Failed to run ${suite.name}: ${error.message}`, 'red');
          if (suite.critical) {
            this.log('⚠️ Critical test suite failed - continuing with remaining tests', 'yellow');
          }
        }
      }

      // Generate final report
      this.log('\\n📋 Final Results', 'bright');
      this.log('='.repeat(30), 'bright');
      
      const totalTests = this.results.total;
      const successRate = totalTests > 0 ? ((this.results.passed / totalTests) * 100).toFixed(1) : 0;
      
      this.log(`Total Tests: ${totalTests}`, 'cyan');
      this.log(`Passed: ${this.results.passed}`, 'green');
      this.log(`Failed: ${this.results.failed}`, this.results.failed > 0 ? 'red' : 'cyan');
      this.log(`Skipped: ${this.results.skipped}`, 'yellow');
      this.log(`Success Rate: ${successRate}%`, successRate >= 80 ? 'green' : 'yellow');

      const report = await this.generateTestReport();

      // Final status
      if (this.results.failed === 0) {
        this.log('\\n🎉 All tests completed successfully!', 'green');
        this.log('✅ Development environment is ready for use', 'green');
      } else if (this.results.passed > this.results.failed) {
        this.log('\\n⚠️ Tests completed with some issues', 'yellow');
        this.log('💡 Check the failed tests - some features may need attention', 'yellow');
      } else {
        this.log('\\n❌ Multiple test failures detected', 'red');
        this.log('🔧 Development environment may need configuration', 'red');
      }

      this.log(`\\n⏱️ Total execution time: ${report.duration}`, 'magenta');

    } catch (error) {
      this.log(`\\n❌ Test runner failed: ${error.message}`, 'red');
      process.exit(1);
    }
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  headed: args.includes('--headed') || args.includes('-h'),
  browser: args.find(arg => arg.startsWith('--browser='))?.split('=')[1] || 'chromium',
  verbose: args.includes('--verbose') || args.includes('-v')
};

// Create and run test runner
const runner = new TestRunner();
runner.run().catch(error => {
  console.error('\\n❌ Fatal error:', error);
  process.exit(1);
});