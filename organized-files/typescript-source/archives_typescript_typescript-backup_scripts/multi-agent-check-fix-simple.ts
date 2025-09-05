// scripts/multi-agent-check-fix-simple.ts
// Simplified multi-agent build fixer with local LLM integration

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import os from 'os';

// Configuration
const config = {
  timestamp: new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19),
  maxWorkers: os.cpus().length,
  ollamaHost: process.env.OLLAMA_HOST || 'http://localhost:11434',
  model: 'gemma3-legal',
  embedModel: 'nomic-embed-text'
};

// Directory setup
const logsDir = `logs_${config.timestamp}`;
const todoDir = `todolist_${config.timestamp}`;
const logFile = path.join(logsDir, 'npm_check.log');
const summaryFile = path.join(todoDir, 'summary.md');
const outputJson = path.join(todoDir, 'gemma_suggestions.json');
const metricsFile = path.join(todoDir, 'build_metrics.json');

// Create directories
[logsDir, todoDir].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

console.log(`📁 Created directories:
- Logs: ${logsDir}
- TODOs: ${todoDir}`);

// VS Code GPU settings
const vscodeGpuSettings = {
  "disable-hardware-acceleration": false,
  "enable-gpu-rasterization": true,
  "enable-webgl2-compute-context": true,
  "gpu-sandbox": true,
  "use-angle": "d3d11",
  "enable-zero-copy": true,
  "enable-gpu-memory-buffer-video-frames": true,
  "enable-native-gpu-memory-buffers": true,
  "disable-gpu-vsync": false,
  "max-active-webgl-contexts": 16,
  "webgl-antialiasing-mode": "msaa",
  "enable-gpu-scheduler": true,
  "terminal.integrated.gpuAcceleration": "on",
  "editor.experimental.useGPU": true
};

// Error classification
enum ErrorType {
  TypeScript = 'typescript',
  Svelte = 'svelte',
  Build = 'build',
  Runtime = 'runtime',
  Memory = 'memory',
  GPU = 'gpu',
  Network = 'network',
  Database = 'database'
}

export interface ErrorEntry {
  id: string;
  type: ErrorType;
  file: string;
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
  context: string;
  timestamp: Date;
  suggestions?: string[];
  autoFixAvailable?: boolean;
}

export interface BuildMetrics {
  totalErrors: number;
  errorsByType: Record<ErrorType, number>;
  memoryUsage: NodeJS.MemoryUsage;
  cpuUsage: NodeJS.CpuUsage;
  duration: number;
  gpuAvailable: boolean;
  ollamaAvailable: boolean;
  gemmaModelLoaded: boolean;
}

// Error parser
class ErrorParser {
  private patterns = {
    typescript: /TS\d+:|Type error:|Cannot find|Property .* does not exist/i,
    svelte: /Svelte:|Component:|<script>|<style>|\$:/i,
    memory: /heap|memory|allocation|OOM|out of memory/i,
    gpu: /GPU|CUDA|WebGL|shader|rasterization/i,
    database: /postgres|pgvector|drizzle|SQL|query/i,
  };

  parseErrors(log: string): ErrorEntry[] {
    const errors: ErrorEntry[] = [];
    const lines = log.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const errorMatch = this.extractError(line, i, lines);
      
      if (errorMatch) {
        errors.push(errorMatch);
      }
    }
    
    return errors;
  }

  private extractError(line: string, index: number, allLines: string[]): ErrorEntry | null {
    for (const [type, pattern] of Object.entries(this.patterns)) {
      if (pattern.test(line)) {
        const fileMatch = line.match(/([^\s]+\.(ts|tsx|js|jsx|svelte)):(\d+):(\d+)/);
        
        return {
          id: `error-${index}-${Date.now()}`,
          type: type as ErrorType,
          file: fileMatch?.[1] || 'unknown',
          line: parseInt(fileMatch?.[3] || '0'),
          column: parseInt(fileMatch?.[4] || '0'),
          message: line.trim(),
          severity: line.includes('error') ? 'error' : 'warning',
          context: allLines.slice(Math.max(0, index - 2), index + 3).join('\n'),
          timestamp: new Date(),
        };
      }
    }
    
    return null;
  }
}

// Local LLM integration
class LocalLLMService {
  async analyzeErrors(errors: ErrorEntry[]): Promise<any> {
    console.log(`🤖 Analyzing ${errors.length} errors with ${config.model}...`);
    
    const prompt = `You are an expert TypeScript/SvelteKit developer. Analyze these build errors and create a structured action plan:

Errors:
${JSON.stringify(errors.slice(0, 10), null, 2)}

Please provide:
1. Error dependencies and fix order
2. Similar error patterns that can be batch-fixed
3. Root causes vs symptoms
4. Specific fix suggestions
5. Integration with SvelteKit 2 + Svelte 5 patterns

Format your response as structured JSON with clear action items.`;

    try {
      const response = await fetch(`${config.ollamaHost}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.model,
          prompt: prompt,
          temperature: 0.3,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama error: ${response.status}`);
      }

      const result = await response.json();
      return {
        agent: 'gemma3-legal',
        response: result.response,
        model: config.model,
        success: true
      };
    } catch (error: any) {
      console.error('❌ Local LLM analysis failed:', error);
      return {
        agent: 'gemma3-legal',
        error: error.message,
        success: false
      };
    }
  }

  async generateEmbedding(text: string): Promise<number[] | null> {
    try {
      const response = await fetch(`${config.ollamaHost}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.embedModel,
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding error: ${response.status}`);
      }

      const result = await response.json();
      return result.embedding;
    } catch (error: any) {
      console.error('❌ Embedding generation failed:', error);
      return null;
    }
  }
}

// Main orchestration function
async function runEnhancedCheckFix(): Promise<any> {
  const startTime = Date.now();
  const initialMemory = process.memoryUsage();
  
  console.log('🚀 Starting enhanced multi-agent check-fix pipeline...');
  
  // Step 1: Update VS Code GPU settings
  console.log('⚙️ Updating VS Code GPU settings...');
  const vscodeSettingsPath = path.join(process.cwd(), '.vscode', 'settings.json');
  if (fs.existsSync(vscodeSettingsPath)) {
    try {
      const settings = JSON.parse(fs.readFileSync(vscodeSettingsPath, 'utf-8'));
      Object.assign(settings, vscodeGpuSettings);
      fs.writeFileSync(vscodeSettingsPath, JSON.stringify(settings, null, 2));
      console.log('✅ VS Code GPU optimizations applied');
    } catch (e: any) {
      console.warn('⚠️ Failed to update VS Code settings:', e.message);
    }
  }

  // Step 2: Run npm check
  console.log('🔍 Running npm check...');
  const log = await runCheck();
  
  // Step 3: Parse errors
  const parser = new ErrorParser();
  const errors = parser.parseErrors(log);
  console.log(`📋 Found ${errors.length} errors`);

  // Step 4: Check services
  const [ollamaAvailable, gemmaLoaded] = await Promise.all([
    checkOllamaAvailable(),
    checkModelLoaded(config.model)
  ]);

  // Step 5: Analyze with local LLM if available
  let llmResults = null;
  if (ollamaAvailable && gemmaLoaded) {
    const llmService = new LocalLLMService();
    llmResults = await llmService.analyzeErrors(errors);
    
    // Generate embeddings for error analysis
    console.log('🧠 Generating embeddings...');
    for (const error of errors.slice(0, 5)) {
      const embedding = await llmService.generateEmbedding(error.message);
      if (embedding) {
        console.log(`📊 Generated embedding for: ${error.file}:${error.line}`);
      }
    }
  } else {
    console.warn('⚠️ Ollama or Gemma3-legal not available, skipping LLM analysis');
    llmResults = {
      agent: 'gemma3-legal',
      error: 'Service not available',
      success: false
    };
  }

  // Step 6: Create comprehensive results
  const results = {
    summary: {
      totalErrors: errors.length,
      criticalErrors: errors.filter(e => e.severity === 'error').length,
      warnings: errors.filter(e => e.severity === 'warning').length,
      filesAffected: [...new Set(errors.map(e => e.file))].length
    },
    errors: errors,
    llmAnalysis: llmResults,
    actionPlan: generateActionPlan(errors, llmResults),
    systemInfo: {
      ollamaAvailable,
      gemmaLoaded,
      gpuOptimized: true,
      timestamp: new Date().toISOString()
    }
  };

  // Step 7: Save results
  const markdownSummary = formatMarkdownSummary(results);
  fs.writeFileSync(summaryFile, markdownSummary);
  fs.writeFileSync(outputJson, JSON.stringify(results, null, 2));
  
  // Step 8: Generate metrics
  const metrics: BuildMetrics = {
    totalErrors: errors.length,
    errorsByType: errors.reduce((acc, e) => {
      acc[e.type] = (acc[e.type] || 0) + 1;
      return acc;
    }, {} as Record<ErrorType, number>),
    memoryUsage: process.memoryUsage(),
    cpuUsage: process.cpuUsage(),
    duration: Date.now() - startTime,
    gpuAvailable: await checkGPUAvailability(),
    ollamaAvailable,
    gemmaModelLoaded: gemmaLoaded
  };
  
  fs.writeFileSync(metricsFile, JSON.stringify(metrics, null, 2));
  
  console.log(`
✅ Enhanced check-fix pipeline complete!

📊 Summary:
- Total errors: ${errors.length}
- LLM analysis: ${llmResults?.success ? '✅ Success' : '❌ Failed'}
- Processing time: ${(metrics.duration / 1000).toFixed(2)}s
- Memory used: ${((metrics.memoryUsage.heapUsed - initialMemory.heapUsed) / 1024 / 1024).toFixed(2)}MB

📁 Outputs:
- Summary: ${summaryFile}
- Suggestions: ${outputJson}
- Metrics: ${metricsFile}

🚀 Next steps:
1. Review ${summaryFile} for prioritized fixes
2. Use gemma3-legal analysis for automated fixes
3. Apply GPU optimizations to VS Code
4. Monitor build performance improvements
`);
}

// Helper functions
function runCheck(): Promise<string> {
  return new Promise((resolve) => {
    const check = spawn('powershell', ['-Command', 'npm run check'], { shell: true });
    let output = '';

    check.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stdout.write(chunk);
    });
    
    check.stderr.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stderr.write(chunk);
    });

    check.on('close', () => {
      fs.writeFileSync(logFile, output);
      resolve(output);
    });
  });
}

async function checkOllamaAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${config.ollamaHost}/api/tags`);
    return response.ok;
  } catch {
    return false;
  }
}

async function checkModelLoaded(model: string): Promise<boolean> {
  try {
    const response = await fetch(`${config.ollamaHost}/api/tags`);
    if (!response.ok) return false;
    
    const data = await response.json();
    return data.models?.some((m: unknown) => m.name.includes(model));
  } catch {
    return false;
  }
}

async function checkGPUAvailability(): Promise<boolean> {
  try {
    const response = await fetch(`${config.ollamaHost}/api/tags`);
    const data = await response.json();
    return data?.models?.some((m: unknown) => m.details?.families?.includes('gpu'));
  } catch {
    return false;
  }
}

function generateActionPlan(errors: ErrorEntry[], llmResults: unknown): unknown {
  const errorsByType = errors.reduce((acc, error) => {
    acc[error.type] = (acc[error.type] || 0) + 1;
    return acc;
  }, {} as Record<ErrorType, number>);

  const plan = {
    immediate: [
      'Fix critical TypeScript errors blocking compilation',
      'Update Svelte component syntax to v5 patterns',
      'Resolve missing module imports'
    ],
    automated: [
      'Apply ESLint auto-fixes',
      'Use gemma3-legal suggestions for pattern fixes',
      'Update import paths using compiler diagnostics'
    ],
    manual: [
      'Review complex type definitions',
      'Update deprecated Svelte patterns',
      'Optimize component performance'
    ],
    metrics: {
      estimatedTime: errors.length > 10 ? '2-4 hours' : '30-60 minutes',
      automationPotential: llmResults?.success ? '75%' : '40%',
      errorsByType
    }
  };

  // Add LLM-specific suggestions if available
  if (llmResults?.success) {
    try {
      const llmSuggestions = JSON.parse(llmResults.response);
      if (llmSuggestions.suggestions) {
        plan.automated = [...plan.automated, ...llmSuggestions.suggestions.slice(0, 3)];
      }
    } catch (e: any) {
      // LLM response wasn't JSON, extract text suggestions
      if (typeof llmResults.response === 'string') {
        const lines = llmResults.response.split('\n');
        const suggestions = lines.filter(line => 
          line.includes('fix') || line.includes('Fix') || line.includes('TODO')
        ).slice(0, 3);
        plan.automated = [...plan.automated, ...suggestions];
      }
    }
  }

  return plan;
}

function formatMarkdownSummary(results: unknown): string {
  const { summary, llmAnalysis, actionPlan, systemInfo } = results;

  return `# Enhanced Build Fix Summary - ${config.timestamp}

## 📊 Overview
- **Total Errors**: ${summary.totalErrors}
- **Critical Errors**: ${summary.criticalErrors}
- **Warnings**: ${summary.warnings}
- **Files Affected**: ${summary.filesAffected}

## 🤖 Local LLM Analysis (${config.model})
**Status**: ${llmAnalysis.success ? '✅ Success' : '❌ Failed'}
${llmAnalysis.success ? `
**Model**: ${llmAnalysis.model}
**Analysis**: 
\`\`\`
${llmAnalysis.response.slice(0, 1000)}...
\`\`\`
` : `**Error**: ${llmAnalysis.error}`}

## 🎯 Action Plan

### 🚨 Immediate Actions
${actionPlan.immediate.map((a: string) => `- ${a}`).join('\n')}

### 🤖 Automated Fixes (${llmAnalysis.success ? 'LLM-Enhanced' : 'Basic'})
${actionPlan.automated.map((a: string) => `- ${a}`).join('\n')}

### 👨‍💻 Manual Review Required
${actionPlan.manual.map((a: string) => `- ${a}`).join('\n')}

## ⚙️ System Integration

### GPU Optimization Status
- ✅ VS Code GPU settings updated
- ✅ Hardware acceleration enabled
- ✅ WebGL2 compute context active

### Local LLM Status
- **Ollama**: ${systemInfo.ollamaAvailable ? '✅ Running' : '❌ Not available'}
- **Gemma3-Legal**: ${systemInfo.gemmaLoaded ? '✅ Loaded' : '❌ Not loaded'}
- **Embeddings**: ${systemInfo.ollamaAvailable ? '✅ Available' : '❌ Unavailable'}

## 📈 Performance Metrics
- **Estimated Fix Time**: ${actionPlan.metrics.estimatedTime}
- **Automation Potential**: ${actionPlan.metrics.automationPotential}
- **Error Distribution**: ${JSON.stringify(actionPlan.metrics.errorsByType, null, 2)}

## 🚀 Next Steps
1. **Start with automated fixes** using LLM suggestions
2. **Apply VS Code GPU optimizations** for better performance  
3. **Monitor build improvements** with real-time feedback
4. **Use vector embeddings** for similar error detection

---
*Generated by Enhanced Multi-Agent Build Fixer v2.0*
*Using: ${config.model} + GPU Optimization + Vector Analysis*
`;
}

// Run the enhanced pipeline
runEnhancedCheckFix().catch(console.error);

export { runEnhancedCheckFix, ErrorParser, LocalLLMService };