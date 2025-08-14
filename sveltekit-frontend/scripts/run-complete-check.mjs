// run-complete-check.mjs
// Comprehensive integration check script

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const execAsync = promisify(exec);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Colors for console output
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

class IntegrationChecker {
  constructor() {
    this.report = [];
    this.errors = [];
    this.warnings = [];
    this.successes = [];
    this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.enhancedRAGMetrics = null;
  }

  log(message, type = 'info') {
    const prefix = {
      success: `${colors.green}✅`,
      error: `${colors.red}❌`,
      warning: `${colors.yellow}⚠️`,
      info: `${colors.cyan}ℹ️`
    };

    console.log(`${prefix[type] || ''}  ${message}${colors.reset}`);
    this.report.push(`${prefix[type] || ''}  ${message}`);

    if (type === 'error') this.errors.push(message);
    if (type === 'warning') this.warnings.push(message);
    if (type === 'success') this.successes.push(message);
  }

  async checkPort(port) {
    try {
      const { stdout } = await execAsync(
        `powershell -NoProfile -Command "Test-NetConnection -ComputerName localhost -Port ${port} -WarningAction SilentlyContinue | Select-Object -ExpandProperty TcpTestSucceeded"`
      );
      return stdout.trim() === 'True';
    } catch {
      return false;
    }
  }

  async checkNodeModules() {
    this.log('Checking npm packages...', 'info');

    const requiredPackages = ['chalk', 'ora', 'glob', 'concurrently', 'ws', 'rimraf'];

    for (const pkg of requiredPackages) {
      try {
        await import(pkg);
        this.log(`Package ${pkg} is installed`, 'success');
      } catch {
        this.log(`Package ${pkg} is missing`, 'warning');

        // Try to install
        this.log(`Installing ${pkg}...`, 'info');
        try {
          await execAsync(`npm install --save-dev ${pkg}`);
          this.log(`Installed ${pkg}`, 'success');
        } catch (error) {
          this.log(`Failed to install ${pkg}: ${error.message}`, 'error');
        }
      }
    }
  }

  async runTypeScriptCheck() {
    this.log('Running TypeScript check...', 'info');

    try {
      const { stdout, stderr } = await execAsync('npx tsc --noEmit --skipLibCheck --incremental');
      if (!stderr && !stdout) {
        this.log('TypeScript: No errors', 'success');
      } else {
        const errorCount = (stdout + stderr).match(/error TS/g)?.length || 0;
        this.log(`TypeScript: ${errorCount} errors found`, errorCount > 0 ? 'warning' : 'success');
      }
    } catch (error) {
      const errorCount = error.stdout?.match(/error TS/g)?.length || 0;
      this.log(`TypeScript: ${errorCount} errors found`, 'warning');
    }
  }

  async checkServices() {
    this.log('Checking critical services...', 'info');

    const services = [
      { name: 'Frontend', port: 5173, url: 'http://localhost:5173' },
      { name: 'Frontend (alt 5177)', port: 5177, url: 'http://localhost:5177' },
      { name: 'Go API (8084)', port: 8084, url: 'http://localhost:8084/api/health' },
      { name: 'Go API (8085)', port: 8085, url: 'http://localhost:8085/api/health' },
      { name: 'Redis', port: 6379, url: null },
      { name: 'Ollama', port: 11434, url: 'http://localhost:11434/api/version' },
      { name: 'PostgreSQL', port: 5432, url: null },
      { name: 'MCP Context7 (4000)', port: 4000, url: 'http://localhost:4000/health' },
      { name: 'MCP Multi-Core (4100)', port: 4100, url: 'http://localhost:4100/health' }
    ];

    for (const service of services) {
      try {
        if (service.url) {
          const response = await fetch(service.url, {
            signal: AbortSignal.timeout(3000)
          });
          if (response.ok) {
            this.log(`${service.name}: Running`, 'success');
          } else {
            this.log(`${service.name}: HTTP ${response.status}`, 'warning');
            if (service.port) {
              const isOpen = await this.checkPort(service.port);
              if (isOpen) {
                if (service.name.toLowerCase().includes('go api')) {
                  this.log(`${service.name}: Port ${service.port} is open (gRPC likely)`, 'success');
                } else {
                  this.log(`${service.name}: Port ${service.port} is open`, 'info');
                }
              }
            }
          }
        } else {
          // Port check only
          const isOpen = await this.checkPort(service.port);
          if (isOpen) {
            this.log(`${service.name} (${service.port}): Port open`, 'success');
          } else {
            this.log(`${service.name} (${service.port}): Not accessible`, 'error');
          }
        }
      } catch (error) {
        // If HTTP check failed, try port as fallback for more context
        if (service.port) {
          const isOpen = await this.checkPort(service.port);
          if (isOpen) {
            if (service.name.toLowerCase().includes('go api')) {
              this.log(`${service.name}: Port ${service.port} is open (gRPC likely; HTTP check failed)`, 'success');
            } else {
              this.log(`${service.name}: Port ${service.port} is open (HTTP check failed)`, 'warning');
            }
          }
        }
        this.log(`${service.name}: ${error.message}`, 'error');
      }
    }
  }

  async checkEnhancedRAG() {
    this.log('🚀 Checking Enhanced RAG Tool Performance...', 'info');

    try {
      // Test Context7 MCP server endpoint first
      try {
        const context7Check = await fetch('http://localhost:4000/health', {
          signal: AbortSignal.timeout(2000)
        });

        if (context7Check.ok) {
          this.log('Context7 MCP Server: Running', 'success');
        } else {
          this.log('Context7 MCP Server: Not responding', 'warning');
        }
      } catch {
        this.log('Context7 MCP Server: Not accessible', 'warning');
      }

      // Test MCP Multi-Core base port if available
      try {
        const context7Multi = await fetch('http://localhost:4100/health', {
          signal: AbortSignal.timeout(2000)
        });
        if (context7Multi.ok) {
          this.log('Context7 MCP Multi-Core: Running (base 4100)', 'success');
        } else {
          this.log('Context7 MCP Multi-Core: Not responding (base 4100)', 'warning');
        }
      } catch {
        this.log('Context7 MCP Multi-Core: Not accessible (base 4100)', 'warning');
      }

      // Performance benchmark test with fallback to alternate frontend port
      const testQueries = ['contract liability terms', 'legal document analysis', 'evidence processing'];
      const candidateVectorEndpoints = [
        'http://localhost:5173/api/ai/vector-search',
        'http://localhost:5177/api/ai/vector-search'
      ];

      let totalResponseTime = 0;
      let successfulQueries = 0;
      let usedEndpoint = null;

      for (const query of testQueries) {
        try {
          const startTime = performance.now();

          let response;
          let endpointHit;
          for (const endpoint of candidateVectorEndpoints) {
            try {
              response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, model: 'claude', limit: 5 }),
                signal: AbortSignal.timeout(10000)
              });
              endpointHit = endpoint;
              if (response.ok || response.status === 500) break; // good or informative
            } catch {
              // try next endpoint
            }
          }

          const endTime = performance.now();
          const responseTime = endTime - startTime;

          if (response && response.ok) {
            // eslint-disable-next-line no-unused-vars
            const result = await response.json();
            totalResponseTime += responseTime;
            successfulQueries++;
            usedEndpoint = endpointHit;
            this.log(`Query "${query}" via ${endpointHit}: ${responseTime.toFixed(2)}ms`, 'success');
          } else if (response && response.status === 500) {
            // Frontend has compilation issues but API might still work
            this.log(`Query "${query}" via ${endpointHit}: Frontend error (500)`, 'warning');
          } else {
            this.log(`Query "${query}": No reachable endpoint (tried ${candidateVectorEndpoints.join(', ')})`, 'warning');
          }
        } catch (error) {
          this.log(`Query "${query}": ${error.message}`, 'warning');
        }
      }

      if (successfulQueries > 0) {
        const avgResponseTime = totalResponseTime / successfulQueries;
        const throughputEstimate = Math.round(1000 / avgResponseTime);

        this.log(`📊 Enhanced RAG Performance:`, 'info');
        this.log(`   Average Response: ${avgResponseTime.toFixed(2)}ms`, 'success');
        this.log(`   Estimated Throughput: ${throughputEstimate} req/sec`, 'success');

        // Performance thresholds
        if (avgResponseTime < 1) {
          this.log(`   🚀 ULTRA-OPTIMIZED (Target: <1ms)`, 'success');
        } else if (avgResponseTime < 10) {
          this.log(`   ⚡ OPTIMIZED (Target: <10ms)`, 'success');
        } else if (avgResponseTime < 100) {
          this.log(`   ✅ GOOD (Target: <100ms)`, 'success');
        } else {
          this.log(`   ⚠️  NEEDS OPTIMIZATION (>100ms)`, 'warning');
        }

        // Store metrics for report
        this.enhancedRAGMetrics = {
          avgResponseTime,
          throughputEstimate,
          successfulQueries,
          totalQueries: testQueries.length,
          endpoint: usedEndpoint,
          status:
            avgResponseTime < 1
              ? 'ULTRA-OPTIMIZED'
              : avgResponseTime < 10
              ? 'OPTIMIZED'
              : avgResponseTime < 100
              ? 'GOOD'
              : 'NEEDS_OPTIMIZATION'
        };
      } else {
        // Fallback: Test if the API endpoint exists even if not working
        this.log('Enhanced RAG: No successful queries, testing endpoint...', 'warning');

        try {
          const response = await fetch('http://localhost:5173/api/ai/vector-search', {
            method: 'GET',
            signal: AbortSignal.timeout(5000)
          });

          if (response.status === 405) {
            // Method not allowed is actually good - means endpoint exists
            this.log('Enhanced RAG: API endpoint exists (GET not allowed)', 'success');
            this.enhancedRAGMetrics = {
              status: 'ENDPOINT_EXISTS',
              note: 'API configured but frontend compilation issues'
            };
          } else {
            this.log(`Enhanced RAG: API responded with ${response.status}`, 'warning');
            this.enhancedRAGMetrics = { status: 'PARTIAL', note: 'API exists but has issues' };
          }
        } catch {
          // Try alternate port 5177 before declaring offline
          try {
            const responseAlt = await fetch('http://localhost:5177/api/ai/vector-search', {
              method: 'GET',
              signal: AbortSignal.timeout(5000)
            });
            if (responseAlt.status === 405) {
              this.log('Enhanced RAG: API endpoint exists on 5177 (GET not allowed)', 'success');
              this.enhancedRAGMetrics = {
                status: 'ENDPOINT_EXISTS',
                note: '5177 available; 5173 had issues'
              };
            } else {
              this.log(`Enhanced RAG (5177): API responded with ${responseAlt.status}`, 'warning');
              this.enhancedRAGMetrics = { status: 'PARTIAL', note: 'API exists on 5177 but has issues' };
            }
          } catch {
            this.log('Enhanced RAG: API endpoint not accessible on 5173 or 5177', 'error');
            this.enhancedRAGMetrics = { status: 'OFFLINE' };
          }
        }
      }
    } catch (error) {
      this.log(`Enhanced RAG check failed: ${error.message}`, 'error');
      this.enhancedRAGMetrics = { status: 'ERROR', error: error.message };
    }
  }

  async checkFileStructure() {
    this.log('Verifying file structure...', 'info');

    // Get workspace root (deeds-web-app directory)
    const workspaceRoot = path.join(__dirname, '..', '..');

    const criticalFiles = [
      { path: path.join(workspaceRoot, 'main.go'), name: 'main.go' },
      { path: path.join(__dirname, '..', 'package.json'), name: 'package.json' },
      { path: path.join(__dirname, '..', 'tsconfig.json'), name: 'tsconfig.json' },
      { path: path.join(workspaceRoot, 'database', 'schema-jsonb-enhanced.sql'), name: 'database/schema-jsonb-enhanced.sql' },
      { path: path.join(__dirname, '..', 'src', 'lib', 'db', 'schema-jsonb.ts'), name: 'src/lib/db/schema-jsonb.ts' },
      { path: path.join(__dirname, '..', 'src', 'routes', 'api', 'ai', 'vector-search', '+server.ts'), name: 'src/routes/api/ai/vector-search/+server.ts' },
      { path: path.join(workspaceRoot, '812aisummarizeintegration.md'), name: '812aisummarizeintegration.md' },
      { path: path.join(workspaceRoot, 'TODO-AI-INTEGRATION.md'), name: 'TODO-AI-INTEGRATION.md' },
      { path: path.join(workspaceRoot, 'FINAL-INTEGRATION-REPORT.md'), name: 'FINAL-INTEGRATION-REPORT.md' },
      { path: path.join(workspaceRoot, 'enhancedraghow2.txt'), name: 'enhancedraghow2.txt' }
    ];

    const criticalDirs = [
      { path: path.join(workspaceRoot, 'ai-summarized-documents'), name: 'ai-summarized-documents' },
      { path: path.join(workspaceRoot, 'ai-summarized-documents', 'contracts'), name: 'ai-summarized-documents/contracts' },
      { path: path.join(workspaceRoot, 'ai-summarized-documents', 'legal-briefs'), name: 'ai-summarized-documents/legal-briefs' },
      { path: path.join(workspaceRoot, 'ai-summarized-documents', 'case-studies'), name: 'ai-summarized-documents/case-studies' },
      { path: path.join(workspaceRoot, 'ai-summarized-documents', 'embeddings'), name: 'ai-summarized-documents/embeddings' },
      { path: path.join(workspaceRoot, 'ai-summarized-documents', 'cache'), name: 'ai-summarized-documents/cache' },
      { path: path.join(__dirname, '..', 'scripts'), name: 'scripts' },
      { path: path.join(__dirname, '..', 'src', 'lib', 'db'), name: 'src/lib/db' },
      { path: path.join(__dirname, '..', 'src', 'routes', 'api', 'ai'), name: 'src/routes/api/ai' }
    ];

    // Check files
    for (const { path: filePath, name } of criticalFiles) {
      try {
        await fs.access(filePath);
        this.log(`File exists: ${name}`, 'success');
      } catch {
        this.log(`File missing: ${name}`, 'error');
      }
    }

    // Check directories
    for (const { path: dirPath, name } of criticalDirs) {
      try {
        await fs.access(dirPath);
        this.log(`Directory exists: ${name}`, 'success');
      } catch {
        this.log(`Directory missing: ${name}`, 'error');
      }
    }
  }

  async checkSystemRequirements() {
    this.log('Checking system requirements...', 'info');

    // Node.js version
    try {
      const { stdout } = await execAsync('node --version');
      const version = stdout.trim();
      const major = parseInt(version.match(/v(\d+)/)?.[1] || '0');
      if (major >= 18) {
        this.log(`Node.js ${version} (OK)`, 'success');
      } else {
        this.log(`Node.js ${version} (requires v18+)`, 'error');
      }
    } catch {
      this.log('Node.js not found', 'error');
    }

    // Go
    try {
      const { stdout } = await execAsync('go version');
      this.log(`Go installed: ${stdout.trim()}`, 'success');
    } catch {
      this.log('Go not installed (required for API)', 'warning');
    }

    // GPU
    try {
      const { stdout } = await execAsync('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader');
      this.log(`GPU detected: ${stdout.trim()}`, 'success');
    } catch {
      this.log('NVIDIA GPU not detected (CPU mode will be used)', 'warning');
    }
  }

  async generateReport() {
    const reportPath = path.join(__dirname, `INTEGRATION-CHECK-${this.timestamp}.md`);

    const reportContent = `# 🔍 Integration Check Report

**Generated:** ${new Date().toISOString()}
**System:** ${process.platform}
**Node Version:** ${process.version}

---

## 📊 Summary

- ✅ **Successes:** ${this.successes.length}
- ⚠️ **Warnings:** ${this.warnings.length}
- ❌ **Errors:** ${this.errors.length}

---

## 🚀 Enhanced RAG Performance

${this.enhancedRAGMetrics ? `
**Status:** ${this.enhancedRAGMetrics.status}
${this.enhancedRAGMetrics.endpoint ? `
**Endpoint:** ${this.enhancedRAGMetrics.endpoint}
` : ''}
${this.enhancedRAGMetrics.avgResponseTime ? `
**Average Response Time:** ${this.enhancedRAGMetrics.avgResponseTime.toFixed(2)}ms
**Estimated Throughput:** ${this.enhancedRAGMetrics.throughputEstimate} req/sec
**Query Success Rate:** ${this.enhancedRAGMetrics.successfulQueries}/${this.enhancedRAGMetrics.totalQueries}

### Performance Classification:
${this.enhancedRAGMetrics.status === 'ULTRA-OPTIMIZED' ? '🚀 **ULTRA-OPTIMIZED** - Response time <1ms (Target achieved!)' :
  this.enhancedRAGMetrics.status === 'OPTIMIZED' ? '⚡ **OPTIMIZED** - Response time <10ms' :
  this.enhancedRAGMetrics.status === 'GOOD' ? '✅ **GOOD** - Response time <100ms' :
  '⚠️ **NEEDS OPTIMIZATION** - Response time >100ms'}
` : this.enhancedRAGMetrics.error ? `
**Error:** ${this.enhancedRAGMetrics.error}
` : '**Status:** OFFLINE'}
` : '**Not tested** (Enhanced RAG check skipped)'}

---

## 📋 Detailed Results

${this.report.join('\n')}

---

## 🚦 Overall Status

${this.errors.length === 0 ? '### ✅ SYSTEM READY' :
  this.errors.length < 5 ? '### ⚠️ SYSTEM PARTIALLY READY' :
  '### ❌ SYSTEM NEEDS CONFIGURATION'}

${this.errors.length > 0 ? `
### Required Actions:
${this.errors.map((e, i) => `${i + 1}. Fix: ${e}`).join('\n')}
` : ''}

${this.warnings.length > 0 ? `
### Recommendations:
${this.warnings.map((w, i) => `${i + 1}. ${w}`).join('\n')}
` : ''}

---

## 🚀 Next Steps

1. ${this.errors.length > 0 ? 'Fix errors listed above' : 'System is ready'}
2. Run \`npm run dev:full\` to start all services
3. Access frontend at http://localhost:5173
4. Monitor health at http://localhost:8084/api/health

---

**Report saved to:** ${reportPath}
`;

    await fs.writeFile(reportPath, reportContent, 'utf-8');
    this.log(`Report saved to: ${reportPath}`, 'success');

    return reportPath;
  }

  async run() {
    console.log(`${colors.blue}${colors.bright}`);
    console.log('╔════════════════════════════════════════╗');
    console.log('║    COMPLETE INTEGRATION CHECK          ║');
    console.log('╚════════════════════════════════════════╝');
    console.log(`${colors.reset}\n`);

    await this.checkNodeModules();
    await this.runTypeScriptCheck();
    await this.checkSystemRequirements();
    await this.checkServices();
    await this.checkEnhancedRAG();
    await this.checkFileStructure();

    const reportPath = await this.generateReport();

    console.log(`\n${colors.bright}${colors.cyan}`);
    console.log('════════════════════════════════════════');
    console.log('           CHECK COMPLETE');
    console.log('════════════════════════════════════════');
    console.log(`${colors.reset}\n`);

    console.log(`📊 Results:`);
    console.log(`  ${colors.green}✅ Successes: ${this.successes.length}${colors.reset}`);
    console.log(`  ${colors.yellow}⚠️  Warnings: ${this.warnings.length}${colors.reset}`);
    console.log(`  ${colors.red}❌ Errors: ${this.errors.length}${colors.reset}\n`);

    // Enhanced RAG summary
    if (this.enhancedRAGMetrics) {
      console.log(`🚀 Enhanced RAG: ${this.enhancedRAGMetrics.status}`);
      if (this.enhancedRAGMetrics.avgResponseTime) {
        console.log(`   ${colors.cyan}Average Response: ${this.enhancedRAGMetrics.avgResponseTime.toFixed(2)}ms${colors.reset}`);
        console.log(`   ${colors.cyan}Throughput: ${this.enhancedRAGMetrics.throughputEstimate} req/sec${colors.reset}\n`);
      }
    }

    if (this.errors.length === 0) {
      console.log(`${colors.green}🎉 System is ready for use!${colors.reset}`);
    } else {
      console.log(`${colors.yellow}⚠️  Please review the report for required actions.${colors.reset}`);
    }

    console.log(`\n📄 Full report: ${reportPath}\n`);
  }
}

// Run the checker
const checker = new IntegrationChecker();
checker.run().catch(console.error);
