#!/usr/bin/env node

/**
 * YoRHa Legal AI - Comprehensive Health Check
 * 
 * Deep system validation and diagnostics:
 * - Service connectivity and API validation
 * - Database schema and data integrity
 * - Performance benchmarking
 * - Security auditing
 * - Configuration validation
 * 
 * @author YoRHa Legal AI Team
 * @version 2.0.0
 */

import 'zx/globals';
import chalk from 'chalk';
import ora from 'ora';
import fetch from 'node-fetch';
import { WebSocket } from 'ws';
import { program } from 'commander';

// Comprehensive health check configuration
const HEALTH_CONFIG = {
  checks: {
    connectivity: {
      name: 'Service Connectivity',
      priority: 'critical',
      timeout: 15000,
      checks: [
        {
          name: 'PostgreSQL Connection',
          test: async () => {
            const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT version();" -t`;
            return {
              success: result.exitCode === 0,
              details: result.stdout.trim(),
              latency: await measureQueryLatency()
            };
          }
        },
        {
          name: 'Redis Connection',
          test: async () => {
            const start = Date.now();
            const result = await $`echo "ping" | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
            return {
              success: result.stdout.includes('PONG'),
              details: result.stdout.trim(),
              latency: Date.now() - start
            };
          }
        },
        {
          name: 'Ollama API',
          test: async () => {
            const start = Date.now();
            const response = await fetch('http://localhost:11434/api/version', {
              timeout: 10000,
              signal: AbortSignal.timeout(10000)
            });
            const data = await response.json();
            return {
              success: response.ok,
              details: data.version || 'Version info unavailable',
              latency: Date.now() - start
            };
          }
        },
        {
          name: 'Go Microservice API',
          test: async () => {
            const start = Date.now();
            const response = await fetch('http://localhost:8080/api/health', {
              timeout: 10000,
              signal: AbortSignal.timeout(10000)
            });
            const data = await response.json();
            return {
              success: response.ok && data.status === 'healthy',
              details: data.message || 'Health check response',
              latency: Date.now() - start
            };
          }
        },
        {
          name: 'SvelteKit Frontend',
          test: async () => {
            const port = process.env.NODE_ENV === 'production' ? 3000 : 5173;
            const start = Date.now();
            const response = await fetch(`http://localhost:${port}/`, {
              timeout: 15000,
              signal: AbortSignal.timeout(15000)
            });
            return {
              success: response.ok,
              details: `HTTP ${response.status} - ${response.statusText}`,
              latency: Date.now() - start
            };
          }
        }
      ]
    },
    database: {
      name: 'Database Integrity',
      priority: 'critical',
      timeout: 30000,
      checks: [
        {
          name: 'pgvector Extension',
          test: async () => {
            try {
              const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT * FROM pg_extension WHERE extname = 'vector';" -t`;
              return {
                success: result.stdout.includes('vector'),
                details: result.stdout.trim() || 'pgvector extension not found'
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'Database Schema',
          test: async () => {
            try {
              const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "\\dt" -t`;
              const tables = result.stdout.split('\n').filter(line => line.includes('public')).length;
              return {
                success: tables > 0,
                details: `${tables} tables found`
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'Sample Data Query',
          test: async () => {
            try {
              const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT COUNT(*) FROM legal_documents;" -t`;
              const count = parseInt(result.stdout.trim()) || 0;
              return {
                success: true,
                details: `${count} legal documents in database`
              };
            } catch (error) {
              return { success: false, details: 'Could not query legal_documents table' };
            }
          }
        },
        {
          name: 'Vector Embedding Test',
          test: async () => {
            try {
              const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT COUNT(*) FROM legal_documents WHERE embedding IS NOT NULL;" -t`;
              const count = parseInt(result.stdout.trim()) || 0;
              return {
                success: true,
                details: `${count} documents with vector embeddings`
              };
            } catch (error) {
              return { success: false, details: 'Could not check vector embeddings' };
            }
          }
        }
      ]
    },
    api: {
      name: 'API Functionality',
      priority: 'high',
      timeout: 20000,
      checks: [
        {
          name: 'Document Upload API',
          test: async () => {
            try {
              // Test with a small sample document
              const testDocument = JSON.stringify({
                title: 'Health Check Document',
                content: 'This is a test document for health checking.',
                caseId: 'HEALTH-CHECK-001',
                type: 'legal'
              });
              
              const response = await fetch('http://localhost:8080/api/v1/documents', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: testDocument,
                timeout: 15000,
                signal: AbortSignal.timeout(15000)
              });
              
              return {
                success: response.ok || response.status === 400, // 400 might be validation error, which is expected
                details: `HTTP ${response.status} - API responding`
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'Search API',
          test: async () => {
            try {
              const searchQuery = JSON.stringify({
                query: 'legal document search test',
                limit: 5
              });
              
              const response = await fetch('http://localhost:8080/api/v1/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: searchQuery,
                timeout: 15000,
                signal: AbortSignal.timeout(15000)
              });
              
              return {
                success: response.ok || response.status === 404, // 404 might indicate no documents, which is fine
                details: `HTTP ${response.status} - Search API responding`
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'Ollama Model Loading',
          test: async () => {
            try {
              const response = await fetch('http://localhost:11434/api/tags');
              const data = await response.json();
              const modelCount = data.models ? data.models.length : 0;
              
              return {
                success: response.ok,
                details: `${modelCount} models available`
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'WebSocket Connection',
          test: async () => {
            return new Promise((resolve) => {
              const timeout = setTimeout(() => {
                resolve({ success: false, details: 'WebSocket connection timeout' });
              }, 10000);
              
              try {
                const ws = new WebSocket('ws://localhost:8080/ws');
                
                ws.on('open', () => {
                  clearTimeout(timeout);
                  ws.close();
                  resolve({ success: true, details: 'WebSocket connection successful' });
                });
                
                ws.on('error', (error) => {
                  clearTimeout(timeout);
                  resolve({ success: false, details: error.message });
                });
              } catch (error) {
                clearTimeout(timeout);
                resolve({ success: false, details: error.message });
              }
            });
          }
        }
      ]
    },
    performance: {
      name: 'Performance Benchmarks',
      priority: 'medium',
      timeout: 45000,
      checks: [
        {
          name: 'Database Query Performance',
          test: async () => {
            const start = Date.now();
            try {
              await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT COUNT(*) FROM legal_documents;" -t`;
              const queryTime = Date.now() - start;
              
              return {
                success: queryTime < 5000, // Should complete in under 5 seconds
                details: `Query completed in ${queryTime}ms`,
                benchmark: queryTime
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'API Response Time',
          test: async () => {
            const start = Date.now();
            try {
              const response = await fetch('http://localhost:8080/api/health');
              const responseTime = Date.now() - start;
              
              return {
                success: response.ok && responseTime < 2000,
                details: `API responded in ${responseTime}ms`,
                benchmark: responseTime
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'Frontend Load Time',
          test: async () => {
            const start = Date.now();
            try {
              const port = process.env.NODE_ENV === 'production' ? 3000 : 5173;
              const response = await fetch(`http://localhost:${port}/`);
              const loadTime = Date.now() - start;
              
              return {
                success: response.ok && loadTime < 5000,
                details: `Frontend loaded in ${loadTime}ms`,
                benchmark: loadTime
              };
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        },
        {
          name: 'LLM Inference Speed',
          test: async () => {
            const start = Date.now();
            try {
              const response = await fetch('http://localhost:11434/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model: 'llama3.1:8b',
                  prompt: 'Hello',
                  stream: false
                }),
                timeout: 30000,
                signal: AbortSignal.timeout(30000)
              });
              
              const inferenceTime = Date.now() - start;
              
              if (response.ok) {
                return {
                  success: inferenceTime < 30000,
                  details: `LLM inference completed in ${inferenceTime}ms`,
                  benchmark: inferenceTime
                };
              } else {
                return {
                  success: false,
                  details: `HTTP ${response.status} - Model may not be loaded`
                };
              }
            } catch (error) {
              return { success: false, details: error.message };
            }
          }
        }
      ]
    },
    security: {
      name: 'Security Validation',
      priority: 'high',
      timeout: 15000,
      checks: [
        {
          name: 'Database Authentication',
          test: async () => {
            try {
              // Test with wrong credentials (should fail)
              const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U wronguser -d legal_ai_db -h localhost -c "SELECT 1;" -t`.catch(() => ({ exitCode: 1 }));
              
              return {
                success: result.exitCode !== 0, // Should fail with wrong credentials
                details: result.exitCode !== 0 ? 'Authentication properly enforced' : 'Security concern: weak authentication'
              };
            } catch (error) {
              return { success: true, details: 'Authentication properly enforced' };
            }
          }
        },
        {
          name: 'API Rate Limiting',
          test: async () => {
            try {
              // Make rapid requests to test rate limiting
              const requests = Array(10).fill().map(() => 
                fetch('http://localhost:8080/api/health', { timeout: 5000 })
              );
              
              const responses = await Promise.allSettled(requests);
              const rateLimited = responses.some(r => 
                r.status === 'fulfilled' && r.value.status === 429
              );
              
              return {
                success: true, // Rate limiting is optional, so we don't fail
                details: rateLimited ? 'Rate limiting active' : 'No rate limiting detected'
              };
            } catch (error) {
              return { success: true, details: 'Could not test rate limiting' };
            }
          }
        },
        {
          name: 'HTTPS/SSL Configuration',
          test: async () => {
            // Check if services are properly configured for production
            const isProduction = process.env.NODE_ENV === 'production';
            
            return {
              success: true, // This is informational
              details: isProduction ? 
                'Production mode - ensure HTTPS is configured' : 
                'Development mode - HTTPS not required'
            };
          }
        }
      ]
    }
  }
};

// Enhanced logging
const log = {
  timestamp: () => new Date().toISOString(),
  info: (msg) => console.log(`[${log.timestamp()}]`, chalk.blue('ℹ'), msg),
  success: (msg) => console.log(`[${log.timestamp()}]`, chalk.green('✓'), msg),
  error: (msg) => console.log(`[${log.timestamp()}]`, chalk.red('✗'), msg),
  warn: (msg) => console.log(`[${log.timestamp()}]`, chalk.yellow('⚠'), msg),
  debug: (msg) => process.env.DEBUG && console.log(`[${log.timestamp()}]`, chalk.gray('🔍'), msg)
};

// Utility functions
async function measureQueryLatency() {
  const start = Date.now();
  try {
    await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT 1;" -t`;
    return Date.now() - start;
  } catch {
    return null;
  }
}

// Health check execution
async function runHealthCheck(checkConfig) {
  const results = [];
  const spinner = ora(`🔍 Running ${checkConfig.name}...`).start();
  
  try {
    for (const check of checkConfig.checks) {
      const checkSpinner = ora(`  ${check.name}...`).start();
      
      try {
        const result = await Promise.race([
          check.test(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Check timeout')), checkConfig.timeout)
          )
        ]);
        
        result.name = check.name;
        result.timestamp = Date.now();
        results.push(result);
        
        if (result.success) {
          checkSpinner.succeed(`${check.name} - ${result.details}`);
        } else {
          checkSpinner.fail(`${check.name} - ${result.details}`);
        }
        
      } catch (error) {
        const failedResult = {
          name: check.name,
          success: false,
          details: error.message,
          timestamp: Date.now()
        };
        results.push(failedResult);
        checkSpinner.fail(`${check.name} - ${error.message}`);
      }
      
      // Brief pause between checks
      await sleep(500);
    }
    
    const successful = results.filter(r => r.success).length;
    const total = results.length;
    
    if (successful === total) {
      spinner.succeed(`${checkConfig.name} - All ${total} checks passed`);
    } else {
      spinner.warn(`${checkConfig.name} - ${successful}/${total} checks passed`);
    }
    
  } catch (error) {
    spinner.fail(`${checkConfig.name} failed: ${error.message}`);
  }
  
  return results;
}

// Comprehensive health check runner
async function runFullHealthCheck(options = {}) {
  console.log(chalk.cyan.bold('🏥 YoRHa Legal AI - Comprehensive Health Check\n'));
  
  const startTime = Date.now();
  const allResults = new Map();
  
  // Filter checks based on options
  let checksToRun = Object.entries(HEALTH_CONFIG.checks);
  
  if (options.category) {
    checksToRun = checksToRun.filter(([name]) => name === options.category);
  }
  
  if (options.priority) {
    checksToRun = checksToRun.filter(([, config]) => config.priority === options.priority);
  }
  
  // Run health checks
  for (const [checkName, checkConfig] of checksToRun) {
    const results = await runHealthCheck(checkConfig);
    allResults.set(checkName, {
      category: checkName,
      priority: checkConfig.priority,
      results
    });
  }
  
  // Generate comprehensive report
  console.log(chalk.cyan('\n📊 Health Check Summary:'));
  
  let totalChecks = 0;
  let totalPassed = 0;
  let criticalFailed = 0;
  const benchmarks = {};
  
  for (const [category, data] of allResults) {
    const passed = data.results.filter(r => r.success).length;
    const total = data.results.length;
    const failed = total - passed;
    
    totalChecks += total;
    totalPassed += passed;
    
    if (data.priority === 'critical' && failed > 0) {
      criticalFailed += failed;
    }
    
    // Collect benchmarks
    data.results.forEach(result => {
      if (result.benchmark) {
        benchmarks[result.name] = result.benchmark;
      }
    });
    
    const statusColor = failed === 0 ? chalk.green : 
                       data.priority === 'critical' ? chalk.red : chalk.yellow;
    
    console.log(`  ${category.padEnd(20)} ${statusColor(`${passed}/${total} passed`)} ${chalk.gray(`(${data.priority})`)}`);
    
    // Show failed checks
    data.results.filter(r => !r.success).forEach(result => {
      console.log(`    ${chalk.red('✗')} ${result.name}: ${chalk.gray(result.details)}`);
    });
  }
  
  // Overall assessment
  const overallHealth = totalPassed / totalChecks;
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  
  console.log(chalk.cyan('\n🎯 Overall Assessment:'));
  
  let healthColor, healthStatus, exitCode;
  
  if (criticalFailed > 0) {
    healthColor = chalk.red;
    healthStatus = 'CRITICAL ISSUES DETECTED';
    exitCode = 2;
  } else if (overallHealth >= 0.9) {
    healthColor = chalk.green;
    healthStatus = 'EXCELLENT HEALTH';
    exitCode = 0;
  } else if (overallHealth >= 0.7) {
    healthColor = chalk.yellow;
    healthStatus = 'GOOD HEALTH WITH MINOR ISSUES';
    exitCode = 0;
  } else {
    healthColor = chalk.red;
    healthStatus = 'SIGNIFICANT ISSUES DETECTED';
    exitCode = 1;
  }
  
  console.log(`  Status:        ${healthColor(healthStatus)}`);
  console.log(`  Success Rate:  ${healthColor(`${Math.round(overallHealth * 100)}%`)} (${totalPassed}/${totalChecks})`);
  console.log(`  Duration:      ${chalk.blue(`${duration}s`)}`);
  console.log(`  Critical:      ${criticalFailed > 0 ? chalk.red(`${criticalFailed} failed`) : chalk.green('All passed')}`);
  
  // Show benchmarks
  if (Object.keys(benchmarks).length > 0) {
    console.log(chalk.cyan('\n⏱️  Performance Benchmarks:'));
    for (const [check, time] of Object.entries(benchmarks)) {
      const timeColor = time < 1000 ? chalk.green : 
                       time < 5000 ? chalk.yellow : chalk.red;
      console.log(`  ${check.padEnd(25)} ${timeColor(`${time}ms`)}`);
    }
  }
  
  // Recommendations
  if (overallHealth < 1) {
    console.log(chalk.cyan('\n💡 Recommendations:'));
    
    if (criticalFailed > 0) {
      console.log('  🚨 Critical issues must be resolved before production use');
      console.log('  🔧 Check service configurations and dependencies');
      console.log('  📋 Review failed checks above for specific actions');
    }
    
    if (Object.values(benchmarks).some(t => t > 5000)) {
      console.log('  ⚡ Consider performance optimization for slow components');
    }
    
    console.log('  📊 Run: npm run status --watch for continuous monitoring');
    console.log('  🔄 Run: npm run health --category=connectivity to focus on specific areas');
  }
  
  // Output JSON if requested
  if (options.json) {
    const jsonReport = {
      timestamp: new Date().toISOString(),
      duration: parseFloat(duration),
      summary: {
        totalChecks,
        totalPassed,
        criticalFailed,
        overallHealth: Math.round(overallHealth * 100),
        status: healthStatus
      },
      categories: Object.fromEntries(allResults),
      benchmarks
    };
    
    console.log('\n' + JSON.stringify(jsonReport, null, 2));
  }
  
  return { exitCode, results: allResults, overallHealth };
}

// Main function
async function main() {
  program
    .option('-c, --category <category>', 'Run specific category only (connectivity, database, api, performance, security)')
    .option('-p, --priority <priority>', 'Filter by priority (critical, high, medium, low)')
    .option('-j, --json', 'Output detailed JSON report')
    .option('-q, --quick', 'Run only critical checks')
    .option('-v, --verbose', 'Verbose output with debug information')
    .parse();

  const options = program.opts();
  
  if (options.verbose) {
    process.env.DEBUG = '1';
  }
  
  if (options.quick) {
    options.priority = 'critical';
  }
  
  try {
    const { exitCode } = await runFullHealthCheck(options);
    process.exit(exitCode);
  } catch (error) {
    log.error(`Health check failed: ${error.message}`);
    process.exit(3);
  }
}

// Helper function
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Handle CLI help
if (process.argv.includes('--help')) {
  console.log(`
YoRHa Legal AI Comprehensive Health Check

Usage: npm run health [options]

Options:
  -c, --category <name>    Run specific category only
  -p, --priority <level>   Filter by priority level
  -j, --json              Output detailed JSON report
  -q, --quick             Run only critical checks
  -v, --verbose           Verbose output with debug information
  --help                  Show this help message

Categories:
  connectivity            Service connectivity and basic API checks
  database                Database integrity and schema validation
  api                     API functionality and endpoint testing
  performance             Performance benchmarking and response times
  security                Security validation and configuration checks

Priority Levels:
  critical               Must pass for system to be operational
  high                   Important for proper functioning
  medium                 Performance and optimization related
  low                    Nice-to-have and informational

Examples:
  npm run health                           # Full health check
  npm run health --quick                   # Critical checks only
  npm run health --category database       # Database checks only
  npm run health --priority high --json    # High priority with JSON output
  npm run health --verbose                 # Detailed debug output

Exit Codes:
  0 = All checks passed or minor issues only
  1 = Significant issues detected
  2 = Critical issues detected
  3 = Health check system failure
`);
  process.exit(0);
}

// Run the health check
main();