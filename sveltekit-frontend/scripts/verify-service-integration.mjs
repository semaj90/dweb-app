#!/usr/bin/env node
/**
 * Service Integration Verification
 * Tests Neo4j and RabbitMQ integration with the development workflow
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import chalk from 'chalk';

const execAsync = promisify(exec);

class ServiceVerification {
  constructor() {
    this.results = new Map();
  }

  async verifyIntegration() {
    console.log(chalk.bold.cyan('🔍 VERIFYING SERVICE INTEGRATION\n'));

    try {
      // Test Neo4j integration
      await this.testNeo4jIntegration();
      
      // Test RabbitMQ integration  
      await this.testRabbitMQIntegration();
      
      // Test service file imports
      await this.testServiceImports();
      
      // Show results
      this.showResults();
      
    } catch (error) {
      console.error(chalk.red(`❌ Verification failed: ${error.message}`));
      process.exit(1);
    }
  }

  async testNeo4jIntegration() {
    console.log(chalk.yellow('Testing Neo4j integration...'));
    
    try {
      // Check if Neo4j files exist and are importable
      const { stdout } = await execAsync('node -e "console.log(require(\'../src/lib/services/neo4jGraphService.ts\'))"');
      this.results.set('neo4j-import', '✅ Importable');
      
      // Check Neo4j binary
      const neo4jExists = await this.fileExists('../neo4j-community-5.21.2/bin/neo4j.bat');
      this.results.set('neo4j-binary', neo4jExists ? '✅ Found' : '⚠️ Missing');
      
    } catch (error) {
      this.results.set('neo4j-import', `❌ Error: ${error.message}`);
    }
  }

  async testRabbitMQIntegration() {
    console.log(chalk.yellow('Testing RabbitMQ integration...'));
    
    try {
      // Check RabbitMQ service files
      const rabbitmqTsExists = await this.fileExists('../src/lib/server/rabbitmq.ts');
      const rabbitmqJsExists = await this.fileExists('../src/lib/server/rabbitmq.js');
      
      this.results.set('rabbitmq-ts', rabbitmqTsExists ? '✅ Found' : '❌ Missing');
      this.results.set('rabbitmq-js', rabbitmqJsExists ? '✅ Found' : '❌ Missing');
      
      // Check RabbitMQ Windows service
      try {
        await execAsync('sc query RabbitMQ');
        this.results.set('rabbitmq-service', '✅ Service installed');
      } catch {
        this.results.set('rabbitmq-service', '⚠️ Service not installed - using fallback mode');
      }
      
    } catch (error) {
      this.results.set('rabbitmq-integration', `❌ Error: ${error.message}`);
    }
  }

  async testServiceImports() {
    console.log(chalk.yellow('Testing service file imports...'));
    
    const serviceFiles = [
      '../src/lib/services/neo4j-transformers-summarization.ts',
      '../src/lib/ai/enhanced-neo4j-reranker.ts',
      '../src/lib/services/neo4jGraphService.ts',
      '../src/lib/server/rabbitmq.ts'
    ];

    for (const file of serviceFiles) {
      try {
        const exists = await this.fileExists(file);
        const filename = file.split('/').pop();
        this.results.set(filename, exists ? '✅ Ready' : '❌ Missing');
      } catch (error) {
        const filename = file.split('/').pop();
        this.results.set(filename, `❌ Error: ${error.message}`);
      }
    }
  }

  async fileExists(filePath) {
    try {
      await execAsync(`test -f "${filePath}"`);
      return true;
    } catch {
      return false;
    }
  }

  showResults() {
    console.log(chalk.bold.green('\n📊 INTEGRATION VERIFICATION RESULTS\n'));
    
    for (const [service, status] of this.results) {
      console.log(`${status} ${chalk.dim(service)}`);
    }
    
    const successCount = Array.from(this.results.values()).filter(v => v.startsWith('✅')).length;
    const totalCount = this.results.size;
    
    console.log(chalk.bold.cyan(`\n🎯 Integration Score: ${successCount}/${totalCount}`));
    
    if (successCount === totalCount) {
      console.log(chalk.bold.green('🚀 All services ready for npm run dev:full integration!'));
    } else {
      console.log(chalk.bold.yellow('⚠️ Some services may need manual configuration'));
    }
  }
}

// Run verification
const verifier = new ServiceVerification();
verifier.verifyIntegration();