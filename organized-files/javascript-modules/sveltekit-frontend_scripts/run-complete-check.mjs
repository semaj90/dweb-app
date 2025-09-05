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

  async checkNodeModules() {
    this.log('Checking npm packages...', 'info');
    
    const requiredPackages = [
      'chalk', 'ora', 'glob', 'concurrently', 'ws', 'rimraf'
    ];
    
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
    this.log('Checking services...', 'info');
    
    const services = [
      { name: 'Frontend', port: 5173, url: 'http://localhost:5173' },
      { name: 'Go API', port: 8084, url: 'http://localhost:8084/api/health' },
      { name: 'Redis', port: 6379, command: 'redis-cli ping' },
      { name: 'Ollama', port: 11434, url: 'http://localhost:11434/api/tags' },
      { name: 'PostgreSQL', port: 5432, command: 'pg_isready' }
    ];
    
    for (const service of services) {
      if (service.command) {
        try {
          await execAsync(service.command);
          this.log(`${service.name} is running`, 'success');
        } catch {
          this.log(`${service.name} is not available`, 'warning');
        }
      } else if (service.url) {
        try {
          const response = await fetch(service.url, { 
            method: 'GET',
            signal: AbortSignal.timeout(2000)
          });
          if (response.ok) {
            this.log(`${service.name} is running on port ${service.port}`, 'success');
          } else {
            this.log(`${service.name} responded with status ${response.status}`, 'warning');
          }
        } catch {
          this.log(`${service.name} is not available on port ${service.port}`, 'warning');
        }
      }
    }
  }

  async checkFileStructure() {
    this.log('Verifying file structure...', 'info');
    
    const criticalFiles = [
      '../main.go',
      'package.json',
      'tsconfig.json',
      '../database/schema-jsonb-enhanced.sql',
      'src/lib/db/schema-jsonb.ts',
      'src/routes/api/ai/vector-search/+server.ts',
      '../812aisummarizeintegration.md',
      '../TODO-AI-INTEGRATION.md',
      '../FINAL-INTEGRATION-REPORT.md'
    ];
    
    const criticalDirs = [
      '../ai-summarized-documents',
      '../ai-summarized-documents/contracts',
      '../ai-summarized-documents/legal-briefs',
      '../ai-summarized-documents/case-studies',
      '../ai-summarized-documents/embeddings',
      '../ai-summarized-documents/cache',
      'scripts',
      'src/lib/db',
      'src/routes/api/ai'
    ];
    
    // Check files
    for (const file of criticalFiles) {
      const fullPath = path.join(__dirname, file);
      try {
        await fs.access(fullPath);
        this.log(`File exists: ${file}`, 'success');
      } catch {
        this.log(`File missing: ${file}`, 'error');
      }
    }
    
    // Check directories
    for (const dir of criticalDirs) {
      const fullPath = path.join(__dirname, dir);
      try {
        await fs.access(fullPath);
        this.log(`Directory exists: ${dir}`, 'success');
      } catch {
        this.log(`Directory missing: ${dir}`, 'error');
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
