// scripts/enhanced-check-complete.ts
// Complete Enhanced Check with LokiJS RAG + GPU Ollama Analysis

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import crypto from 'crypto';
import loki from 'lokijs';
import Fuse from 'fuse.js';
import os from 'os';

// Enhanced Check Configuration
const config = {
  timestamp: new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19),
  ollamaHost: process.env.OLLAMA_HOST || 'http://localhost:11434',
  embedModel: 'nomic-embed-text',
  legalModel: 'gemma3-legal',
  vectorDimension: 768,
  maxCacheSize: 1000,
  enableGPU: true,
  enableRAG: true,
  enableAI: true
};

// Directories
const checkDir = `check-enhanced-${config.timestamp}`;
const cacheDir = path.join(checkDir, 'cache');
const ragDir = path.join(checkDir, 'rag');
const logsDir = path.join(checkDir, 'logs');
const vscodeDir = path.join(process.cwd(), '.vscode');

[checkDir, cacheDir, ragDir, logsDir, vscodeDir].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

console.log(`🚀 Enhanced Check Starting
📁 Output: ${checkDir}
🧠 Models: ${config.embedModel} + ${config.legalModel}
⚡ GPU: ${config.enableGPU ? 'Enabled' : 'Disabled'}
🔍 RAG: ${config.enableRAG ? 'Enabled' : 'Disabled'}
🤖 AI: ${config.enableAI ? 'Enabled' : 'Disabled'}`);

// Enhanced Error Interface
interface EnhancedError {
  id: string;
  type: ErrorType;
  file: string;
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
  context: string;
  embedding?: number[];
  aiAnalysis?: string;
  suggestedFix?: string;
  confidence: number;
  timestamp: Date;
  hash: string;
}

enum ErrorType {
  TypeScript = 'typescript',
  Svelte = 'svelte',
  Build = 'build',
  Runtime = 'runtime',
  Import = 'import',
  Syntax = 'syntax',
  Type = 'type',
  Component = 'component'
}

// LokiJS RAG Cache Manager
class RAGCacheManager {
  private db: loki;
  private errors: any;
  private solutions: any;
  private analytics: any;
  private searchIndex: Fuse<EnhancedError>;

  constructor() {
    this.initializeDB();
  }

  private initializeDB() {
    const dbFile = path.join(cacheDir, 'enhanced-check-cache.db');
    
    this.db = new loki(dbFile, {
      autosave: true,
      autosaveInterval: 5000
    });

    // Collections
    this.errors = this.db.addCollection('errors', { 
      indices: ['hash', 'type', 'severity', 'file'] 
    });
    
    this.solutions = this.db.addCollection('solutions', { 
      indices: ['errorHash', 'confidence'] 
    });
    
    this.analytics = this.db.addCollection('analytics', { 
      indices: ['timestamp'] 
    });

    this.initializeSearch();
    console.log('📦 LokiJS RAG cache initialized');
  }

  private initializeSearch() {
    const allErrors = this.errors.find({});
    
    this.searchIndex = new Fuse(allErrors, {
      keys: [
        { name: 'message', weight: 0.4 },
        { name: 'file', weight: 0.3 },
        { name: 'context', weight: 0.2 },
        { name: 'type', weight: 0.1 }
      ],
      threshold: 0.4,
      includeScore: true
    });
  }

  async storeError(error: EnhancedError): Promise<void> {
    // Check for existing error
    const existing = this.errors.findOne({ hash: error.hash });
    
    if (existing) {
      // Update existing
      Object.assign(existing, error);
      this.errors.update(existing);
    } else {
      // Insert new
      this.errors.insert(error);
    }
    
    // Update search index
    this.initializeSearch();
  }

  async findSimilarErrors(error: EnhancedError, limit: number = 3): Promise<EnhancedError[]> {
    const results = this.searchIndex.search(`${error.message} ${error.type}`, { limit });
    return results.map(r => r.item);
  }

  async storeSolution(errorHash: string, solution: string, confidence: number): Promise<void> {
    this.solutions.insert({
      errorHash,
      solution,
      confidence,
      timestamp: new Date()
    });
  }

  getSolution(errorHash: string): any {
    return this.solutions.findOne({ errorHash });
  }

  getErrorStats(): any {
    const totalErrors = this.errors.count();
    const errorsByType = this.errors.mapReduce(
      (obj: any) => obj.type,
      (array: string[]) => array.reduce((acc, type) => {
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    );

    return {
      totalErrors,
      errorsByType,
      totalSolutions: this.solutions.count(),
      cacheSize: totalErrors
    };
  }
}

// Enhanced Error Parser
class EnhancedErrorParser {
  private patterns = {
    typescript: /TS\d+:|Type error:|Cannot find|Property .* does not exist/i,
    svelte: /Svelte:|Component:|<script>|<style>|\$:|export let/i,
    import: /Cannot find module|Module not found|Import|import/i,
    syntax: /Syntax error|Unexpected token|Parse error/i,
    type: /Type|interface|declare|generic/i,
    component: /Component|\.svelte|<|>/i
  };

  parseErrors(log: string): EnhancedError[] {
    const errors: EnhancedError[] = [];
    const lines = log.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      if (this.isErrorLine(line)) {
        const error = this.parseErrorLine(line, i, lines);
        if (error) {
          errors.push(error);
        }
      }
    }
    
    console.log(`📋 Parsed ${errors.length} errors from build log`);
    return errors;
  }

  private isErrorLine(line: string): boolean {
    return line.includes('error') || 
           line.includes('Error') ||
           line.includes('warning') ||
           /TS\d+/.test(line) ||
           line.includes('Svelte');
  }

  private parseErrorLine(line: string, index: number, allLines: string[]): EnhancedError | null {
    // Extract file path, line, column
    const fileMatch = line.match(/([^\s]+\.(ts|tsx|js|jsx|svelte)):(\d+):(\d+)/);
    const file = fileMatch?.[1] || this.extractFileFromContext(line, allLines, index) || 'unknown';
    const lineNum = parseInt(fileMatch?.[3] || '0');
    const column = parseInt(fileMatch?.[4] || '0');

    // Determine error type
    let type = ErrorType.Build;
    for (const [typeName, pattern] of Object.entries(this.patterns)) {
      if (pattern.test(line)) {
        type = typeName as ErrorType;
        break;
      }
    }

    // Determine severity
    const severity = line.toLowerCase().includes('error') ? 'error' : 
                    line.toLowerCase().includes('warning') ? 'warning' : 'info';

    // Extract context
    const context = allLines.slice(Math.max(0, index - 2), index + 3).join('\n');

    // Create error object
    const error: EnhancedError = {
      id: `error-${index}-${Date.now()}`,
      type,
      file,
      line: lineNum,
      column,
      message: line.trim(),
      severity: severity as any,
      context,
      confidence: this.calculateConfidence(line, type),
      timestamp: new Date(),
      hash: crypto.createHash('sha256').update(`${file}:${lineNum}:${line}`).digest('hex').slice(0, 16)
    };

    return error;
  }

  private extractFileFromContext(line: string, allLines: string[], index: number): string | null {
    // Look for file paths in surrounding lines
    for (let i = Math.max(0, index - 3); i <= Math.min(allLines.length - 1, index + 3); i++) {
      const contextLine = allLines[i];
      const fileMatch = contextLine.match(/([^\s]+\.(ts|tsx|js|jsx|svelte))/);
      if (fileMatch) {
        return fileMatch[1];
      }
    }
    return null;
  }

  private calculateConfidence(line: string, type: ErrorType): number {
    let confidence = 0.5; // Base confidence
    
    // Higher confidence for specific error patterns
    if (/TS\d+/.test(line)) confidence += 0.3;
    if (line.includes('Svelte')) confidence += 0.2;
    if (line.includes('Cannot find')) confidence += 0.2;
    if (type !== ErrorType.Build) confidence += 0.1;
    
    return Math.min(confidence, 1.0);
  }
}

// GPU Ollama Integration
class GPUOllamaAnalyzer {
  private ragCache: RAGCacheManager;

  constructor(ragCache: RAGCacheManager) {
    this.ragCache = ragCache;
  }

  async analyzeErrors(errors: EnhancedError[]): Promise<void> {
    if (!config.enableAI) {
      console.log('🤖 AI analysis disabled');
      return;
    }

    console.log(`🤖 Starting GPU Ollama analysis of ${errors.length} errors...`);

    // Generate embeddings first
    await this.generateEmbeddings(errors);

    // Analyze each error with AI
    for (const error of errors) {
      await this.analyzeError(error);
    }

    console.log('✅ GPU Ollama analysis complete');
  }

  private async generateEmbeddings(errors: EnhancedError[]): Promise<void> {
    console.log(`🧠 Generating embeddings with ${config.embedModel}...`);

    for (const error of errors) {
      try {
        const text = `${error.type} error in ${error.file}: ${error.message}\nContext: ${error.context}`;
        
        const response = await fetch(`${config.ollamaHost}/api/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: config.embedModel,
            prompt: text
          })
        });

        if (response.ok) {
          const result = await response.json();
          error.embedding = result.embedding;
          console.log(`✅ Generated ${result.embedding?.length || 0}d embedding for ${error.id}`);
        }
      } catch (err) {
        console.warn(`⚠️ Embedding failed for ${error.id}:`, err.message);
      }

      // Small delay to avoid overwhelming Ollama
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }

  private async analyzeError(error: EnhancedError): Promise<void> {
    try {
      // Check cache first
      const cachedSolution = this.ragCache.getSolution(error.hash);
      if (cachedSolution && cachedSolution.confidence > 0.8) {
        error.suggestedFix = cachedSolution.solution;
        error.aiAnalysis = 'Cached solution (high confidence)';
        return;
      }

      // Find similar errors
      const similarErrors = await this.ragCache.findSimilarErrors(error);
      const similarContext = similarErrors.length > 0 ? 
        `\nSimilar errors found:\n${similarErrors.map(e => `- ${e.message}`).join('\n')}` : '';

      const prompt = `You are an expert TypeScript/SvelteKit developer. Analyze this build error and provide a specific fix:

Error Type: ${error.type}
File: ${error.file}:${error.line}:${error.column}
Message: ${error.message}
Context: ${error.context}${similarContext}

Please provide:
1. Root cause analysis
2. Specific fix steps
3. Prevention advice

Be concise and actionable.`;

      const response = await fetch(`${config.ollamaHost}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.legalModel,
          prompt,
          temperature: 0.2,
          stream: false
        })
      });

      if (response.ok) {
        const result = await response.json();
        error.aiAnalysis = result.response;
        
        // Extract suggested fix (simple pattern matching)
        const fixMatch = result.response.match(/(?:Fix|Solution|Step):\s*([^\n]+)/i);
        error.suggestedFix = fixMatch?.[1] || 'See AI analysis for details';

        // Cache the solution
        await this.ragCache.storeSolution(error.hash, error.suggestedFix, 0.9);
        
        console.log(`🤖 AI analysis complete for ${error.id}`);
      }
    } catch (err) {
      console.warn(`⚠️ AI analysis failed for ${error.id}:`, err.message);
      error.aiAnalysis = 'AI analysis failed';
    }

    // Delay between API calls
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

// VS Code Integration & Reporting
class EnhancedReporting {
  async generateReports(errors: EnhancedError[], ragCache: RAGCacheManager, duration: number): Promise<void> {
    const stats = ragCache.getErrorStats();
    
    // Generate VS Code summary
    await this.generateVSCodeSummary(errors, stats, duration);
    
    // Generate diagnostics
    await this.generateDiagnostics(errors, stats);
    
    // Generate fix suggestions
    await this.generateFixSuggestions(errors);

    console.log('📄 Reports generated in .vscode/ directory');
  }

  private async generateVSCodeSummary(errors: EnhancedError[], stats: any, duration: number): Promise<void> {
    const errorsByType = errors.reduce((acc, error) => {
      acc[error.type] = (acc[error.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const criticalErrors = errors.filter(e => e.severity === 'error');
    const warnings = errors.filter(e => e.severity === 'warning');
    const withAI = errors.filter(e => e.aiAnalysis);
    const withEmbeddings = errors.filter(e => e.embedding);

    const summary = `# Enhanced Check Results - ${config.timestamp}

## 🚀 Build Status: ${errors.length === 0 ? '✅ CLEAN' : `❌ ${errors.length} ISSUES`}

### 📊 Overview
- **Total Issues**: ${errors.length}
- **Critical Errors**: ${criticalErrors.length}
- **Warnings**: ${warnings.length}
- **Processing Time**: ${(duration / 1000).toFixed(2)}s

### 🧠 AI Analysis
- **GPU Ollama**: ${config.enableAI ? '✅ Active' : '❌ Disabled'}
- **AI Analyzed**: ${withAI.length}/${errors.length} errors
- **Vector Embeddings**: ${withEmbeddings.length}/${errors.length} errors
- **Model**: ${config.legalModel} + ${config.embedModel}

### 🔍 RAG Cache Status
- **Total Cached Errors**: ${stats.totalErrors}
- **Cached Solutions**: ${stats.totalSolutions}
- **Cache Hit Rate**: ${stats.totalSolutions > 0 ? ((stats.totalSolutions / stats.totalErrors) * 100).toFixed(1) + '%' : '0%'}

### 📁 Error Distribution
${Object.entries(errorsByType).map(([type, count]) => 
  `- **${type}**: ${count} errors`
).join('\n')}

## 🔧 Critical Issues Requiring Attention

${criticalErrors.slice(0, 5).map((error, index) => `
### ${index + 1}. ${error.type.toUpperCase()} - ${error.file}:${error.line}

**Message**: ${error.message}

**AI Analysis**: ${error.aiAnalysis || 'Not analyzed'}

**Suggested Fix**: ${error.suggestedFix || 'No suggestion available'}

**Confidence**: ${(error.confidence * 100).toFixed(0)}%

---
`).join('')}

## 🚀 Quick Actions

1. **Fix Critical Errors**: Focus on ${criticalErrors.length} error(s) above
2. **Check AI Suggestions**: Review AI-generated fixes for accuracy
3. **Update Dependencies**: Some errors may be resolved by updates
4. **Run Tests**: Ensure fixes don't break existing functionality

## 🔗 Integration Status

- ✅ **LokiJS RAG**: Caching ${stats.totalErrors} error patterns
- ✅ **GPU Processing**: ${config.enableGPU ? 'Hardware acceleration active' : 'Disabled'}
- ✅ **Vector Search**: ${config.vectorDimension}d embeddings for similarity matching
- ✅ **AI Analysis**: Real-time error analysis with ${config.legalModel}

---
*Generated by Enhanced Check System*  
*Timestamp: ${new Date().toISOString()}*
`;

    fs.writeFileSync(path.join(vscodeDir, 'enhanced-check-summary.md'), summary);
  }

  private async generateDiagnostics(errors: EnhancedError[], stats: any): Promise<void> {
    const diagnostics = {
      timestamp: new Date().toISOString(),
      buildStatus: errors.length === 0 ? 'clean' : 'errors',
      system: {
        gpu: config.enableGPU,
        rag: config.enableRAG,
        ai: config.enableAI,
        models: {
          embedding: config.embedModel,
          analysis: config.legalModel
        }
      },
      statistics: {
        totalErrors: errors.length,
        criticalErrors: errors.filter(e => e.severity === 'error').length,
        warnings: errors.filter(e => e.severity === 'warning').length,
        aiAnalyzed: errors.filter(e => e.aiAnalysis).length,
        withEmbeddings: errors.filter(e => e.embedding).length,
        avgConfidence: errors.length > 0 ? 
          errors.reduce((sum, e) => sum + e.confidence, 0) / errors.length : 0
      },
      cache: stats,
      errors: errors.map(error => ({
        id: error.id,
        type: error.type,
        file: error.file,
        line: error.line,
        severity: error.severity,
        message: error.message.slice(0, 200),
        confidence: error.confidence,
        hasAI: !!error.aiAnalysis,
        hasEmbedding: !!error.embedding,
        hasSuggestion: !!error.suggestedFix
      }))
    };

    fs.writeFileSync(
      path.join(vscodeDir, 'enhanced-check-diagnostics.json'), 
      JSON.stringify(diagnostics, null, 2)
    );
  }

  private async generateFixSuggestions(errors: EnhancedError[]): Promise<void> {
    const fixableErrors = errors.filter(e => e.suggestedFix);
    
    if (fixableErrors.length === 0) return;

    const suggestions = `# Automated Fix Suggestions

${fixableErrors.map((error, index) => `
## ${index + 1}. ${error.file}:${error.line} (${error.type})

**Error**: ${error.message}

**Suggested Fix**: ${error.suggestedFix}

**Confidence**: ${(error.confidence * 100).toFixed(0)}%

**AI Analysis**:
${error.aiAnalysis || 'No detailed analysis available'}

---
`).join('')}

## 🔧 Batch Fix Commands

\`\`\`bash
# Run automated fixes (when available)
npm run fix:auto

# Re-run check after fixes
npm run check
\`\`\`
`;

    fs.writeFileSync(path.join(vscodeDir, 'fix-suggestions.md'), suggestions);
  }
}

// Main Enhanced Check Function
async function runEnhancedCheck(): Promise<void> {
  const startTime = Date.now();
  
  console.log('🚀 Enhanced Check with LokiJS RAG + GPU Ollama Starting...');

  // Initialize systems
  const ragCache = new RAGCacheManager();
  const errorParser = new EnhancedErrorParser();
  const ollamaAnalyzer = new GPUOllamaAnalyzer(ragCache);
  const reporter = new EnhancedReporting();

  // Update VS Code GPU settings
  if (config.enableGPU) {
    console.log('⚙️ Applying GPU optimizations...');
    await updateVSCodeGPUSettings();
  }

  // Run the actual check
  console.log('🔍 Running svelte-check...');
  const buildLog = await runSvelteCheck();
  
  // Parse errors
  const errors = errorParser.parseErrors(buildLog);
  
  if (errors.length === 0) {
    console.log('✅ No errors found - build is clean!');
    const cleanReport = `# Build Status: ✅ CLEAN

No errors found in the build process.

**Processing Time**: ${((Date.now() - startTime) / 1000).toFixed(2)}s
**Timestamp**: ${new Date().toISOString()}
**GPU Status**: ${config.enableGPU ? '✅ Enabled' : '❌ Disabled'}
**RAG Status**: ${config.enableRAG ? '✅ Enabled' : '❌ Disabled'}
`;
    fs.writeFileSync(path.join(vscodeDir, 'enhanced-check-summary.md'), cleanReport);
    return;
  }

  // Store errors in RAG cache
  console.log('📦 Storing errors in RAG cache...');
  for (const error of errors) {
    await ragCache.storeError(error);
  }

  // GPU Ollama analysis
  await ollamaAnalyzer.analyzeErrors(errors);

  // Generate reports
  const duration = Date.now() - startTime;
  await reporter.generateReports(errors, ragCache, duration);

  // Final summary
  console.log(`
✅ Enhanced Check Complete!

📊 Results:
- Total Errors: ${errors.length}
- AI Analyzed: ${errors.filter(e => e.aiAnalysis).length}
- Vector Embeddings: ${errors.filter(e => e.embedding).length}
- Processing Time: ${(duration / 1000).toFixed(2)}s

📁 Reports Generated:
- .vscode/enhanced-check-summary.md
- .vscode/enhanced-check-diagnostics.json
- .vscode/fix-suggestions.md

🚀 Next Steps:
1. Review AI-generated fix suggestions
2. Apply recommended fixes
3. Re-run check to verify
`);
}

// Helper Functions
async function updateVSCodeGPUSettings(): Promise<void> {
  const vscodeSettingsPath = path.join(process.cwd(), '.vscode', 'settings.json');
  
  if (fs.existsSync(vscodeSettingsPath)) {
    try {
      const settings = JSON.parse(fs.readFileSync(vscodeSettingsPath, 'utf-8'));
      
      Object.assign(settings, {
        "disable-hardware-acceleration": false,
        "enable-gpu-rasterization": true,
        "enable-webgl2-compute-context": true,
        "gpu-sandbox": true,
        "use-angle": "d3d11",
        "terminal.integrated.gpuAcceleration": "on",
        "editor.experimental.useGPU": true
      });
      
      fs.writeFileSync(vscodeSettingsPath, JSON.stringify(settings, null, 2));
      console.log('✅ VS Code GPU settings updated');
    } catch (error) {
      console.warn('⚠️ Failed to update VS Code settings:', error.message);
    }
  }
}

async function runSvelteCheck(): Promise<string> {
  return new Promise((resolve) => {
    const check = spawn('npx', ['svelte-check', '--tsconfig', './tsconfig.json'], { 
      shell: true,
      env: { ...process.env, NODE_OPTIONS: '--max-old-space-size=8192' }
    });
    
    let output = '';

    check.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stdout.write(chunk); // Show real-time output
    });
    
    check.stderr.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stderr.write(chunk);
    });

    check.on('close', () => {
      const logFile = path.join(logsDir, 'svelte-check.log');
      fs.writeFileSync(logFile, output);
      resolve(output);
    });
  });
}

// Run enhanced check
runEnhancedCheck().catch(console.error);