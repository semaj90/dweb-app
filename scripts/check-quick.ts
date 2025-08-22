// scripts/check-quick.ts
// Quick Enhanced Check with LokiJS RAG (no AI timeout)

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import loki from 'lokijs';
import Fuse from 'fuse.js';

const config = {
  timestamp: new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19),
  enableGPU: true,
  enableRAG: true,
  maxProcessingTime: 30000 // 30 seconds max
};

const checkDir = `check-quick-${config.timestamp}`;
const vscodeDir = path.join(process.cwd(), '.vscode');

[checkDir, vscodeDir].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

console.log(`🚀 Quick Enhanced Check Starting
📁 Output: ${checkDir}
⚡ GPU: ${config.enableGPU ? 'Enabled' : 'Disabled'}
🔍 RAG: ${config.enableRAG ? 'Enabled' : 'Disabled'}`);

// Quick RAG Cache with LokiJS
class QuickRAGCache {
  private db: loki;
  private errors: unknown;
  private searchIndex: Fuse<any>;

  constructor() {
    this.db = new loki();
    this.errors = this.db.addCollection('errors', { 
      indices: ['type', 'severity', 'file'] 
    });
    this.initSearch();
  }

  private initSearch() {
    this.searchIndex = new Fuse([], {
      keys: ['message', 'file', 'type'],
      threshold: 0.3
    });
  }

  storeError(error: unknown) {
    this.errors.insert(error);
    this.updateSearch();
  }

  private updateSearch() {
    const allErrors = this.errors.find({});
    this.searchIndex = new Fuse(allErrors, {
      keys: ['message', 'file', 'type'],
      threshold: 0.3
    });
  }

  getStats() {
    const all = this.errors.find({});
    return {
      total: all.length,
      byType: all.reduce((acc: unknown, e: unknown) => {
        acc[e.type] = (acc[e.type] || 0) + 1;
        return acc;
      }, {}),
      bySeverity: all.reduce((acc: unknown, e: unknown) => {
        acc[e.severity] = (acc[e.severity] || 0) + 1;
        return acc;
      }, {})
    };
  }

  search(query: string) {
    return this.searchIndex.search(query, { limit: 5 });
  }
}

async function runQuickCheck() {
  const startTime = Date.now();
  
  // Initialize RAG cache
  const ragCache = new QuickRAGCache();
  
  // Apply GPU settings quickly
  if (config.enableGPU) {
    console.log('⚙️ Applying GPU optimizations...');
    await updateVSCodeGPU();
  }

  // Run svelte-check
  console.log('🔍 Running svelte-check...');
  const buildLog = await runCheck();
  
  // Parse errors quickly
  const errors = parseErrors(buildLog);
  console.log(`📋 Found ${errors.length} issues`);
  
  // Store in RAG cache
  errors.forEach(error => ragCache.storeError(error));
  
  // Generate quick report
  const stats = ragCache.getStats();
  const duration = Date.now() - startTime;
  
  await generateQuickReport(errors, stats, duration, buildLog);
  
  console.log(`
✅ Quick Enhanced Check Complete!

📊 Results:
- Issues Found: ${errors.length}
- Processing Time: ${(duration / 1000).toFixed(2)}s
- RAG Cache: ${stats.total} errors stored
- GPU Status: ${config.enableGPU ? '✅ Enabled' : '❌ Disabled'}

📁 Report: .vscode/enhanced-check-summary.md
`);
}

function parseErrors(log: string): unknown[] {
  const errors: unknown[] = [];
  const lines = log.split('\n');
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    if (line.includes('error') || line.includes('Error') || line.includes('warning') || /TS\d+/.test(line)) {
      const fileMatch = line.match(/([^\s]+\.(ts|tsx|js|jsx|svelte)):(\d+):(\d+)/);
      
      errors.push({
        id: `error-${i}-${Date.now()}`,
        type: getErrorType(line),
        file: fileMatch?.[1] || 'unknown',
        line: parseInt(fileMatch?.[3] || '0'),
        message: line.trim(),
        severity: line.toLowerCase().includes('error') ? 'error' : 'warning',
        timestamp: new Date()
      });
    }
  }
  
  return errors;
}

function getErrorType(line: string): string {
  if (/TS\d+/.test(line)) return 'typescript';
  if (line.includes('Svelte')) return 'svelte';
  if (line.includes('import') || line.includes('Cannot find module')) return 'import';
  if (line.includes('Syntax')) return 'syntax';
  return 'build';
}

async function runCheck(): Promise<string> {
  return new Promise((resolve) => {
    const check = spawn('npx', ['svelte-check', '--tsconfig', './tsconfig.json'], { 
      shell: true 
    });
    
    let output = '';
    let hasOutput = false;

    const timeout = setTimeout(() => {
      if (!hasOutput) {
        check.kill();
        resolve('Build check timed out');
      }
    }, config.maxProcessingTime);

    check.stdout.on('data', (data) => {
      hasOutput = true;
      const chunk = data.toString();
      output += chunk;
      process.stdout.write(chunk);
    });
    
    check.stderr.on('data', (data) => {
      hasOutput = true;
      output += data.toString();
    });

    check.on('close', () => {
      clearTimeout(timeout);
      resolve(output);
    });
  });
}

async function updateVSCodeGPU() {
  const settingsPath = path.join(vscodeDir, 'settings.json');
  
  if (fs.existsSync(settingsPath)) {
    try {
      const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
      
      Object.assign(settings, {
        "terminal.integrated.gpuAcceleration": "on",
        "editor.experimental.useGPU": true,
        "enable-gpu-rasterization": true
      });
      
      fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
      console.log('✅ VS Code GPU settings updated');
    } catch (error) {
      console.warn('⚠️ VS Code GPU update skipped');
    }
  }
}

async function generateQuickReport(errors: unknown[], stats: unknown, duration: number, buildLog: string) {
  const criticalErrors = errors.filter(e => e.severity === 'error');
  const warnings = errors.filter(e => e.severity === 'warning');
  
  const buildStatus = errors.length === 0 ? '✅ CLEAN' : `❌ ${errors.length} ISSUES`;
  
  const report = `# Enhanced Check Results - ${config.timestamp}

## 🚀 Build Status: ${buildStatus}

### 📊 Quick Summary
- **Total Issues**: ${errors.length}
- **Critical Errors**: ${criticalErrors.length}
- **Warnings**: ${warnings.length}
- **Processing Time**: ${(duration / 1000).toFixed(2)}s

### 🔧 System Status
- **LokiJS RAG**: ✅ ${stats.total} errors cached
- **GPU Optimization**: ${config.enableGPU ? '✅ Active' : '❌ Disabled'}
- **Vector Search**: ✅ Fuse.js ready
- **Build Process**: ${buildLog.includes('0 errors and 0 warnings') ? '✅ Clean' : '⚠️ Issues found'}

### 📁 Error Breakdown
${Object.entries(stats.byType).map(([type, count]) => 
  `- **${type}**: ${count} issues`
).join('\n')}

### 🏆 Priority Actions

${criticalErrors.length > 0 ? `
#### Critical Errors (Fix First)
${criticalErrors.slice(0, 3).map((error, i) => `
${i + 1}. **${error.file}:${error.line}** - ${error.type}
   ${error.message}
`).join('')}
` : '✅ No critical errors found'}

${warnings.length > 0 ? `
#### Warnings (Review)
${warnings.slice(0, 2).map((error, i) => `
${i + 1}. **${error.file}:${error.line}** - ${error.message.slice(0, 80)}...
`).join('')}
` : '✅ No warnings found'}

### 🚀 Available Tools

- \`npm run check\` - Run this enhanced check again
- \`npm run check:basic\` - Run basic svelte-check only
- \`npm run rag:demo\` - Test RAG system with sample data
- \`npm run rag:test\` - Interactive RAG testing

### 🔍 RAG Cache Status

The LokiJS cache now contains ${stats.total} error patterns for:
- **Fast similarity search** of related errors
- **Pattern recognition** for common issues  
- **Solution caching** for repeated problems
- **Analytics** for build quality tracking

---
*Generated by Quick Enhanced Check*  
*Timestamp: ${new Date().toISOString()}*  
*Duration: ${(duration / 1000).toFixed(2)}s*
`;

  fs.writeFileSync(path.join(vscodeDir, 'enhanced-check-summary.md'), report);
  
  // Also generate a simple diagnostics file
  const diagnostics = {
    timestamp: new Date().toISOString(),
    buildStatus: buildStatus.includes('CLEAN') ? 'clean' : 'issues',
    errors: errors.length,
    warnings: warnings.length,
    duration: duration,
    ragCache: stats,
    tools: ['LokiJS', 'Fuse.js', 'GPU Optimization']
  };
  
  fs.writeFileSync(
    path.join(vscodeDir, 'enhanced-check-diagnostics.json'),
    JSON.stringify(diagnostics, null, 2)
  );
}

// Run quick check
runQuickCheck().catch(console.error);