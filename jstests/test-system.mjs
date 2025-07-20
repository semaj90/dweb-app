#!/usr/bin/env node
// Comprehensive system test script

import { exec } from 'child_process';
import { promisify } from 'util';
import fetch from 'node-fetch';
import chalk from 'chalk';

const execAsync = promisify(exec);

class SystemTester {
  constructor() {
    this.results = {
      docker: { status: 'pending', details: '' },
      postgres: { status: 'pending', details: '' },
      pgvector: { status: 'pending', details: '' },
      redis: { status: 'pending', details: '' },
      qdrant: { status: 'pending', details: '' },
      ollama: { status: 'pending', details: '' },
      webApp: { status: 'pending', details: '' },
      vectorApi: { status: 'pending', details: '' },
    };
  }

  async log(message, type = 'info') {
    const colors = {
      info: chalk.blue,
      success: chalk.green,
      error: chalk.red,
      warning: chalk.yellow,
    };
    console.log(colors[type](`${type.toUpperCase()}: ${message}`));
  }

  async testDocker() {
    try {
      await execAsync('docker version');
      const { stdout } = await execAsync('docker ps --format "{{.Names}}"');
      const containers = stdout.trim().split('\n').filter(Boolean);
      
      this.results.docker = {
        status: 'success',
        details: `Docker running with ${containers.length} containers`,
      };
      await this.log('Docker is running', 'success');
    } catch (error) {
      this.results.docker = {
        status: 'error',
        details: 'Docker not running or not installed',
      };
      await this.log('Docker test failed', 'error');
    }
  }

  async testPostgres() {
    try {
      const { stdout } = await execAsync(
        'docker exec prosecutor_postgres psql -U postgres -d prosecutor_db -c "SELECT version();"'
      );
      
      if (stdout.includes('PostgreSQL')) {
        this.results.postgres = {
          status: 'success',
          details: 'PostgreSQL is running and accessible',
        };
        await this.log('PostgreSQL is healthy', 'success');
      }
    } catch (error) {
      this.results.postgres = {
        status: 'error',
        details: 'Cannot connect to PostgreSQL',
      };
      await this.log('PostgreSQL test failed', 'error');
    }
  }

  async testPgVector() {
    try {
      const { stdout } = await execAsync(
        `docker exec prosecutor_postgres psql -U postgres -d prosecutor_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"`
      );
      
      if (stdout.includes('vector')) {
        this.results.pgvector = {
          status: 'success',
          details: 'pgvector extension is installed',
        };
        await this.log('pgvector extension found', 'success');
      } else {
        // Try to create it
        await execAsync(
          'docker exec prosecutor_postgres psql -U postgres -d prosecutor_db -c "CREATE EXTENSION IF NOT EXISTS vector;"'
        );
        this.results.pgvector = {
          status: 'success',
          details: 'pgvector extension installed',
        };
      }
    } catch (error) {
      this.results.pgvector = {
        status: 'error',
        details: 'pgvector extension not available',
      };
      await this.log('pgvector test failed', 'error');
    }
  }

  async testRedis() {
    try {
      const { stdout } = await execAsync('docker exec prosecutor_redis redis-cli ping');
      
      if (stdout.trim() === 'PONG') {
        this.results.redis = {
          status: 'success',
          details: 'Redis is responding',
        };
        await this.log('Redis is healthy', 'success');
      }
    } catch (error) {
      this.results.redis = {
        status: 'error',
        details: 'Redis not responding',
      };
      await this.log('Redis test failed', 'error');
    }
  }

  async testQdrant() {
    try {
      const response = await fetch('http://localhost:6333/healthz');
      const data = await response.json();
      
      if (data.title === 'qdrant - vector search engine') {
        this.results.qdrant = {
          status: 'success',
          details: `Qdrant version ${data.version || 'unknown'}`,
        };
        await this.log('Qdrant is healthy', 'success');
      }
    } catch (error) {
      this.results.qdrant = {
        status: 'error',
        details: 'Qdrant not accessible',
      };
      await this.log('Qdrant test failed', 'error');
    }
  }

  async testOllama() {
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      const data = await response.json();
      
      const models = data.models || [];
      this.results.ollama = {
        status: 'success',
        details: `Ollama running with ${models.length} models: ${models.map(m => m.name).join(', ')}`,
      };
      await this.log(`Ollama is healthy with ${models.length} models`, 'success');
    } catch (error) {
      this.results.ollama = {
        status: 'error',
        details: 'Ollama not accessible',
      };
      await this.log('Ollama test failed', 'error');
    }
  }

  async testWebApp() {
    try {
      const response = await fetch('http://localhost:5173');
      
      if (response.ok) {
        this.results.webApp = {
          status: 'success',
          details: 'Web application is running',
        };
        await this.log('Web app is accessible', 'success');
      } else {
        this.results.webApp = {
          status: 'warning',
          details: 'Web app returned non-200 status',
        };
      }
    } catch (error) {
      this.results.webApp = {
        status: 'error',
        details: 'Web app not running (run npm run dev)',
      };
      await this.log('Web app not accessible', 'warning');
    }
  }

  async testVectorApi() {
    try {
      const response = await fetch('http://localhost:5173/api/vector');
      const data = await response.json();
      
      if (data.success) {
        this.results.vectorApi = {
          status: 'success',
          details: 'Vector API is functioning',
        };
        await this.log('Vector API is healthy', 'success');
      }
    } catch (error) {
      this.results.vectorApi = {
        status: 'error',
        details: 'Vector API not accessible',
      };
      await this.log('Vector API test failed', 'error');
    }
  }

  async runAllTests() {
    console.log(chalk.bold.cyan('\n🧪 Running Comprehensive System Tests\n'));
    
    await this.testDocker();
    await this.testPostgres();
    await this.testPgVector();
    await this.testRedis();
    await this.testQdrant();
    await this.testOllama();
    await this.testWebApp();
    await this.testVectorApi();
    
    this.printReport();
  }

  printReport() {
    console.log(chalk.bold.cyan('\n📊 System Test Report\n'));
    
    const table = Object.entries(this.results).map(([service, result]) => {
      const status = result.status === 'success' ? chalk.green('✅ PASS') :
                     result.status === 'warning' ? chalk.yellow('⚠️  WARN') :
                     chalk.red('❌ FAIL');
      
      return {
        Service: service.charAt(0).toUpperCase() + service.slice(1),
        Status: status,
        Details: result.details,
      };
    });
    
    console.table(table);
    
    const allPassed = Object.values(this.results).every(r => r.status === 'success');
    
    if (allPassed) {
      console.log(chalk.bold.green('\n✨ All systems operational! You can start development.\n'));
      console.log(chalk.cyan('Run: npm run dev'));
    } else {
      console.log(chalk.bold.yellow('\n⚠️  Some services need attention.\n'));
      console.log(chalk.cyan('Quick fixes:'));
      console.log(chalk.white('  1. Start Docker: docker-compose up -d'));
      console.log(chalk.white('  2. Run setup: .\\setup-complete-with-ollama.ps1'));
      console.log(chalk.white('  3. Start dev server: npm run dev'));
    }
  }
}

// Check if running directly
if (process.argv[1] === import.meta.url.slice(7)) {
  const tester = new SystemTester();
  tester.runAllTests().catch(console.error);
}

export default SystemTester;
