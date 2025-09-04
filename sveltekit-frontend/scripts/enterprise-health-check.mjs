#!/usr/bin/env node
// Enterprise Services Health Check
// Quick health verification for npm scripts

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  bright: '\x1b[1m'
};

const services = [
  {
    name: 'PostgreSQL',
    command: 'pg_isready -h localhost -p 5432',
    endpoint: 'localhost:5432',
    expectedOutput: 'accepting connections'
  },
  {
    name: 'Redis',
    command: 'redis-cli ping',
    endpoint: 'localhost:6379',
    expectedOutput: 'PONG'
  },
  {
    name: 'RabbitMQ',
    command: 'rabbitmq-diagnostics status',
    endpoint: 'localhost:5672',
    expectedOutput: 'Status of node'
  },
  {
    name: 'Enhanced RAG',
    command: 'curl -s http://localhost:8094/health',
    endpoint: 'http://localhost:8094',
    expectedOutput: 'status',
    timeout: 3000
  },
  {
    name: 'Upload Service',
    command: 'curl -s http://localhost:8093/health',
    endpoint: 'http://localhost:8093',
    expectedOutput: 'status',
    timeout: 3000
  },
  {
    name: 'SvelteKit Dev',
    command: 'curl -s http://localhost:5173/',
    endpoint: 'http://localhost:5173',
    expectedOutput: 'html',
    timeout: 3000
  }
];

async function checkService(service) {
  try {
    const { stdout, stderr } = await execAsync(service.command, { 
      timeout: service.timeout || 5000 
    });
    const output = stdout + stderr;
    
    if (service.expectedOutput && !output.toLowerCase().includes(service.expectedOutput.toLowerCase())) {
      return { 
        name: service.name, 
        status: 'unhealthy', 
        message: 'Service responding but output unexpected',
        endpoint: service.endpoint 
      };
    }
    
    return { 
      name: service.name, 
      status: 'healthy', 
      message: 'Service operational',
      endpoint: service.endpoint 
    };
  } catch (error) {
    return { 
      name: service.name, 
      status: 'error', 
      message: error.message.includes('timeout') ? 'Service timeout' : 'Service not accessible',
      endpoint: service.endpoint 
    };
  }
}

async function runHealthCheck() {
  console.log(`${colors.blue}${colors.bright}Enterprise Services Health Check${colors.reset}`);
  console.log('━'.repeat(60));
  
  const results = await Promise.all(services.map(checkService));
  
  let healthyCount = 0;
  let totalCount = results.length;
  
  for (const result of results) {
    const statusIcon = result.status === 'healthy' ? '✓' : 
                      result.status === 'unhealthy' ? '⚠' : '✗';
    const statusColor = result.status === 'healthy' ? colors.green :
                       result.status === 'unhealthy' ? colors.yellow : colors.red;
    
    console.log(`${statusColor}${statusIcon}${colors.reset} ${result.name.padEnd(15)} ${result.endpoint.padEnd(25)} ${result.message}`);
    
    if (result.status === 'healthy') healthyCount++;
  }
  
  console.log('━'.repeat(60));
  
  const healthPercentage = Math.round((healthyCount / totalCount) * 100);
  const overallStatus = healthPercentage >= 80 ? 'EXCELLENT' :
                       healthPercentage >= 60 ? 'GOOD' :
                       healthPercentage >= 40 ? 'FAIR' : 'POOR';
  
  const overallColor = healthPercentage >= 80 ? colors.green :
                      healthPercentage >= 60 ? colors.blue :
                      healthPercentage >= 40 ? colors.yellow : colors.red;
  
  console.log(`Overall Health: ${overallColor}${overallStatus}${colors.reset} (${healthyCount}/${totalCount} services healthy - ${healthPercentage}%)`);
  
  if (healthPercentage < 80) {
    console.log(`\n${colors.yellow}Recommendations:${colors.reset}`);
    
    for (const result of results) {
      if (result.status !== 'healthy') {
        console.log(`• ${result.name}: ${result.message}`);
        
        if (result.name === 'PostgreSQL') {
          console.log('  → Install: https://www.postgresql.org/download/windows/');
        } else if (result.name === 'Redis') {
          console.log('  → Install: https://github.com/microsoftarchive/redis/releases');
        } else if (result.name === 'RabbitMQ') {
          console.log('  → Install: https://www.rabbitmq.com/install-windows.html');
        }
      }
    }
    
    console.log(`\n${colors.blue}Setup Command:${colors.reset} npm run dev:enterprise:setup`);
  }
  
  console.log(`\n${colors.green}Ready for development!${colors.reset} Run: npm run dev:enterprise`);
  
  // Exit with appropriate code
  process.exit(healthPercentage >= 60 ? 0 : 1);
}

runHealthCheck().catch(error => {
  console.error(`${colors.red}Health check failed: ${error.message}${colors.reset}`);
  process.exit(1);
});