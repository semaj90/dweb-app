#!/usr/bin/env node
/**
 * Production Test Suite Runner
 * Automated testing for all routes, components, database operations, and native Windows services
 */

import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';

const TEST_CONFIG = {
  baseURL: process.env.TEST_BASE_URL || 'http://localhost:5173',
  dbURL: process.env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db',
  timeout: 60000,
  retries: 2
};

class ProductionTestSuite {
  constructor() {
    this.spinner = ora();
    this.results = {
      passed: 0,
      failed: 0,
      skipped: 0,
      details: []
    };
  }

  async run() {
    console.log(chalk.cyan.bold('🚀 Production Readiness Test Suite'));
    console.log(chalk.gray(`Testing against: ${TEST_CONFIG.baseURL}`));
    console.log('='.repeat(60));

    try {
      await this.checkPrerequisites();
      await this.runRouteHealthChecks();
      await this.runDatabaseTests();
      await this.runE2ETests();
      await this.runPerformanceTests();
      await this.runServiceIntegrationTests();
      
      this.printResults();
    } catch (error) {
      this.spinner.fail(chalk.red(`Test suite failed: ${error.message}`));
      process.exit(1);
    }
  }

  async checkPrerequisites() {
    this.spinner.start('Checking prerequisites...');
    
    const checks = [
      { name: 'Node.js', command: 'node --version' },
      { name: 'npm', command: 'npm --version' },
      { name: 'Playwright', command: 'npx playwright --version' },
      { name: 'PostgreSQL', test: () => this.checkPostgreSQL() },
      { name: 'Redis', test: () => this.checkRedis() },
      { name: 'SvelteKit Dev Server', test: () => this.checkDevServer() }
    ];

    for (const check of checks) {
      try {
        if (check.command) {
          execSync(check.command, { stdio: 'pipe' });
        } else if (check.test) {
          await check.test();
        }
        this.results.passed++;
      } catch (error) {
        this.results.failed++;
        this.results.details.push(`❌ ${check.name}: ${error.message}`);
      }
    }

    this.spinner.succeed('Prerequisites checked');
  }

  async checkPostgreSQL() {
    try {
      const { default: postgres } = await import('postgres');
      const sql = postgres(TEST_CONFIG.dbURL);
      await sql`SELECT 1`;
      await sql.end();
      return true;
    } catch (error) {
      throw new Error(`PostgreSQL connection failed: ${error.message}`);
    }
  }

  async checkRedis() {
    try {
      const { createClient } = await import('redis');
      const client = createClient({ url: 'redis://localhost:6379' });
      await client.connect();
      await client.ping();
      await client.disconnect();
      return true;
    } catch (error) {
      throw new Error(`Redis connection failed: ${error.message}`);
    }
  }

  async checkDevServer() {
    try {
      const response = await fetch(`${TEST_CONFIG.baseURL}/api/health`);
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      return true;
    } catch (error) {
      throw new Error(`Dev server not accessible: ${error.message}`);
    }
  }

  async runRouteHealthChecks() {
    this.spinner.start('Testing route health...');
    
    const routes = [
      '/',
      '/auth/login',
      '/auth/register', 
      '/dashboard',
      '/cases',
      '/evidence',
      '/demos',
      '/demos/yorha-command-center',
      '/demos/enhanced-legal-ai',
      '/api/health',
      '/api/v1/cases',
      '/api/v1/users'
    ];

    let passed = 0;
    const failed = [];

    for (const route of routes) {
      try {
        const response = await fetch(`${TEST_CONFIG.baseURL}${route}`, {
          timeout: 10000
        });
        
        if (response.status < 400 || response.status === 401) {
          passed++;
        } else {
          failed.push(`${route}: HTTP ${response.status}`);
        }
      } catch (error) {
        failed.push(`${route}: ${error.message}`);
      }
    }

    this.results.passed += passed;
    this.results.failed += failed.length;
    this.results.details.push(
      `✅ Routes passed: ${passed}/${routes.length}`,
      ...failed.map(f => `❌ ${f}`)
    );

    this.spinner.succeed(`Route health checked (${passed}/${routes.length} passed)`);
  }

  async runDatabaseTests() {
    this.spinner.start('Testing database operations...');
    
    try {
      const { default: postgres } = await import('postgres');
      const sql = postgres(TEST_CONFIG.dbURL);

      // Test basic connectivity
      await sql`SELECT 1`;
      
      // Test schema exists
      const tables = await sql`
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
      `;
      
      const expectedTables = ['users', 'cases', 'evidence', 'documents'];
      const existingTables = tables.map(t => t.table_name);
      const missingTables = expectedTables.filter(t => !existingTables.includes(t));
      
      if (missingTables.length > 0) {
        throw new Error(`Missing tables: ${missingTables.join(', ')}`);
      }

      // Test pgvector extension
      const extensions = await sql`SELECT * FROM pg_extension WHERE extname = 'vector'`;
      if (extensions.length === 0) {
        throw new Error('pgvector extension not installed');
      }

      // Test vector operations
      await sql`
        CREATE TABLE IF NOT EXISTS test_vectors (
          id serial PRIMARY KEY,
          embedding vector(384)
        )
      `;
      
      await sql`
        INSERT INTO test_vectors (embedding) 
        VALUES ('[${Array(384).fill(0.1).join(',')}]')
      `;
      
      const vectorTest = await sql`
        SELECT id, embedding <-> '[${Array(384).fill(0.1).join(',')}]' as distance 
        FROM test_vectors LIMIT 1
      `;
      
      if (vectorTest.length === 0) {
        throw new Error('Vector similarity search failed');
      }

      // Cleanup
      await sql`DROP TABLE IF EXISTS test_vectors`;
      await sql.end();

      this.results.passed++;
      this.results.details.push('✅ Database operations: All tests passed');
      this.spinner.succeed('Database tests passed');

    } catch (error) {
      this.results.failed++;
      this.results.details.push(`❌ Database operations: ${error.message}`);
      this.spinner.fail('Database tests failed');
    }
  }

  async runE2ETests() {
    this.spinner.start('Running E2E tests...');
    
    try {
      // Run the existing Playwright tests
      const testCommand = 'npx playwright test --reporter=json --output=test-results/e2e-results.json';
      
      const result = execSync(testCommand, { 
        stdio: 'pipe',
        encoding: 'utf8',
        timeout: 120000 // 2 minutes
      });

      // Parse results if JSON reporter was used
      try {
        const resultsPath = 'test-results/e2e-results.json';
        if (fs.existsSync(resultsPath)) {
          const testResults = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
          
          this.results.passed += testResults.stats?.passed || 0;
          this.results.failed += testResults.stats?.failed || 0;
          this.results.skipped += testResults.stats?.skipped || 0;
        }
      } catch (parseError) {
        console.warn('Could not parse E2E test results');
      }

      this.results.details.push('✅ E2E tests: Completed successfully');
      this.spinner.succeed('E2E tests completed');

    } catch (error) {
      this.results.failed++;
      this.results.details.push(`❌ E2E tests: ${error.message}`);
      this.spinner.fail('E2E tests failed');
    }
  }

  async runPerformanceTests() {
    this.spinner.start('Running performance tests...');
    
    const performanceMetrics = [];
    const routes = ['/', '/dashboard', '/cases'];

    for (const route of routes) {
      try {
        const startTime = Date.now();
        const response = await fetch(`${TEST_CONFIG.baseURL}${route}`);
        const endTime = Date.now();
        
        const loadTime = endTime - startTime;
        performanceMetrics.push({ route, loadTime, status: response.status });
        
        // Performance budget: 5 seconds
        if (loadTime > 5000) {
          this.results.failed++;
          this.results.details.push(`❌ Performance: ${route} took ${loadTime}ms (> 5000ms)`);
        } else {
          this.results.passed++;
        }
      } catch (error) {
        this.results.failed++;
        this.results.details.push(`❌ Performance: ${route} failed - ${error.message}`);
      }
    }

    const avgLoadTime = performanceMetrics.reduce((sum, m) => sum + m.loadTime, 0) / performanceMetrics.length;
    this.results.details.push(`📊 Average load time: ${Math.round(avgLoadTime)}ms`);

    this.spinner.succeed('Performance tests completed');
  }

  async runServiceIntegrationTests() {
    this.spinner.start('Testing service integrations...');
    
    const services = [
      { name: 'Ollama', url: 'http://localhost:11434/api/tags' },
      { name: 'MinIO', url: 'http://localhost:9000/minio/health/live' },
      { name: 'Qdrant', url: 'http://localhost:6333/collections' },
      { name: 'Neo4j', url: 'http://localhost:7474/db/data/' }
    ];

    for (const service of services) {
      try {
        const response = await fetch(service.url, { timeout: 5000 });
        
        if (response.ok) {
          this.results.passed++;
          this.results.details.push(`✅ ${service.name}: Online`);
        } else {
          this.results.failed++;
          this.results.details.push(`❌ ${service.name}: HTTP ${response.status}`);
        }
      } catch (error) {
        this.results.skipped++;
        this.results.details.push(`⚠️ ${service.name}: Not accessible (${error.message})`);
      }
    }

    this.spinner.succeed('Service integration tests completed');
  }

  printResults() {
    console.log('\n' + '='.repeat(60));
    console.log(chalk.cyan.bold('🎯 Test Results Summary'));
    console.log('='.repeat(60));
    
    const total = this.results.passed + this.results.failed + this.results.skipped;
    const passRate = total > 0 ? Math.round((this.results.passed / total) * 100) : 0;
    
    console.log(chalk.green(`✅ Passed: ${this.results.passed}`));
    console.log(chalk.red(`❌ Failed: ${this.results.failed}`));
    console.log(chalk.yellow(`⚠️ Skipped: ${this.results.skipped}`));
    console.log(chalk.blue(`📊 Pass Rate: ${passRate}%`));
    
    console.log('\n' + chalk.cyan('Detailed Results:'));
    console.log('-'.repeat(40));
    this.results.details.forEach(detail => console.log(detail));
    
    if (this.results.failed === 0) {
      console.log(chalk.green.bold('\n🎉 All critical tests passed! Production ready.'));
    } else {
      console.log(chalk.red.bold(`\n⚠️ ${this.results.failed} tests failed. Review before deploying.`));
    }

    // Generate report file
    const reportPath = 'production-test-report.json';
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      config: TEST_CONFIG,
      results: this.results
    }, null, 2));
    
    console.log(chalk.gray(`\n📄 Full report saved to: ${reportPath}`));
  }
}

// Command line interface
if (process.argv[1] === new URL(import.meta.url).pathname) {
  const testSuite = new ProductionTestSuite();
  testSuite.run().catch(error => {
    console.error(chalk.red('Test suite crashed:', error));
    process.exit(1);
  });
}

export { ProductionTestSuite };