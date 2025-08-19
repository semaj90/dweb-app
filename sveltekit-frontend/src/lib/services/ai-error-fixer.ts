// ======================================================================
// AI-POWERED ERROR FIXING PIPELINE
// Automated TypeScript error resolution with LLM assistance
// ======================================================================

import { gpuLokiErrorAPI } from './gpu-loki-error-orchestrator.js';
import { parallelAnalysisAPI } from './parallel-error-analyzer.js';
import { browser } from '$app/environment';
import { writable, derived } from 'svelte/store';

interface FixAttempt {
  id: string;
  errorId: string;
  strategy: string;
  originalCode: string;
  fixedCode: string;
  confidence: number;
  applied: boolean;
  result: 'success' | 'failed' | 'partial';
  timestamp: Date;
  llmModel?: string;
}

interface ErrorFix {
  errorId: string;
  file: string;
  line: number;
  originalText: string;
  fixedText: string;
  strategy: string;
  confidence: number;
  reasoning: string;
  dependencies: string[];
  validated: boolean;
}

interface AIFixConfig {
  model: 'gemma3-legal';
  endpoint: string;
  maxRetries: number;
  confidenceThreshold: number;
  batchSize: number;
  validateFixes: boolean;
  embeddingModel: 'nomic-embed-text';
}

class AIErrorFixer {
  private config: AIFixConfig = {
    model: 'gemma3-legal',
    endpoint: 'http://localhost:11434/api/generate',
    maxRetries: 3,
    confidenceThreshold: 0.7,
    batchSize: 10,
    validateFixes: true,
    embeddingModel: 'nomic-embed-text'
  };

  private fixHistory = new Map<string, FixAttempt[]>();
  private ollama = this.initializeOllama();

  private initializeOllama() {
    return {
      async generate(prompt: string, model: string = 'gemma3-legal:latest') {
        try {
          const response = await fetch('http://localhost:11434/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model,
              prompt,
              stream: false,
              options: {
                temperature: 0.1, // Low temperature for consistent fixes
                top_p: 0.9,
                max_tokens: 1000
              }
            })
          });

          if (!response.ok) throw new Error('Ollama request failed');
          
          const data = await response.json();
          return data.response || '';
        } catch (error) {
          console.error('Ollama generation failed:', error);
          return '';
        }
      }
    };
  }

  async fixErrors(errors: any[]): Promise<ErrorFix[]> {
    if (!errors.length) return [];

    console.log(`🔧 AI fixing ${errors.length} errors...`);
    const startTime = performance.now();

    // Filter fixable errors
    const fixableErrors = errors.filter(error => error.fixable && error.confidence > this.config.confidenceThreshold);
    
    if (!fixableErrors.length) {
      console.log('ℹ️ No fixable errors found');
      return [];
    }

    console.log(`🎯 Attempting to fix ${fixableErrors.length} errors`);

    // Process errors in batches
    const batches = this.createBatches(fixableErrors, this.config.batchSize);
    const allFixes: ErrorFix[] = [];

    for (const batch of batches) {
      const batchFixes = await this.processBatch(batch);
      allFixes.push(...batchFixes);
    }

    const processingTime = performance.now() - startTime;
    console.log(`✅ AI fixing completed in ${processingTime.toFixed(2)}ms`);

    return allFixes;
  }

  private async processBatch(errors: any[]): Promise<ErrorFix[]> {
    const fixes: ErrorFix[] = [];

    for (const error of errors) {
      try {
        const fix = await this.generateFix(error);
        if (fix) {
          fixes.push(fix);
          
          // Cache the fix attempt
          await this.cacheFixAttempt(error.id, fix);
        }
      } catch (error) {
        console.error('Error fixing failed:', error);
      }
    }

    return fixes;
  }

  private async generateFix(error: any): Promise<ErrorFix | null> {
    // Check cache first
    const cachedFix = await this.getCachedFix(error.id);
    if (cachedFix) return cachedFix;

    // Generate new fix using AI
    const fix = await this.generateAIFix(error);
    
    if (this.config.validateFixes && fix) {
      fix.validated = await this.validateFix(fix);
    }

    return fix;
  }

  private async generateAIFix(error: any): Promise<ErrorFix | null> {
    const prompt = this.createFixPrompt(error);
    
    try {
      const response = await this.ollama.generate(prompt, 'gemma3-legal:latest');
      
      if (!response) return null;

      return this.parseFixResponse(error, response);
    } catch (error) {
      console.error('AI fix generation failed:', error);
      return null;
    }
  }

  private createFixPrompt(error: any): string {
    return `You are a TypeScript expert. Fix this error:

Error: ${error.code} - ${error.message}
File: ${error.file}
Line: ${error.line}
Category: ${error.category}

Context around line ${error.line}:
\`\`\`typescript
// Line ${error.line - 1}: 
// Line ${error.line}: ${error.originalCode || '// Code not available'}
// Line ${error.line + 1}:
\`\`\`

Provide ONLY the fixed code for line ${error.line}, with this format:
FIXED_CODE: [your fix here]
REASONING: [brief explanation]
CONFIDENCE: [0.0-1.0]

Common fixes for ${error.code}:
${this.getCommonFixes(error.code)}`;
  }

  private getCommonFixes(code: string): string {
    const fixes: Record<string, string> = {
      'TS1434': '- Remove unexpected keyword\n- Fix identifier syntax\n- Check for typos',
      'TS2304': '- Add missing import\n- Declare the variable\n- Check spelling',
      'TS2307': '- Fix module path\n- Install missing package\n- Check file exists',
      'TS2457': '- Rename type alias\n- Use different name\n- Avoid reserved keywords',
      'TS1005': '- Add missing semicolon\n- Add missing comma\n- Check syntax',
      'TS1128': '- Add missing declaration\n- Complete the statement\n- Fix syntax'
    };

    return fixes[code] || '- Manual review required\n- Check TypeScript documentation';
  }

  private parseFixResponse(error: any, response: string): ErrorFix | null {
    try {
      const fixedCodeMatch = response.match(/FIXED_CODE:\s*(.+?)(?:\n|$)/);
      const reasoningMatch = response.match(/REASONING:\s*(.+?)(?:\n|$)/);
      const confidenceMatch = response.match(/CONFIDENCE:\s*([\d.]+)/);

      if (!fixedCodeMatch) return null;

      const fix: ErrorFix = {
        errorId: error.id,
        file: error.file,
        line: error.line,
        originalText: error.originalCode || '',
        fixedText: fixedCodeMatch[1].trim(),
        strategy: this.getFixStrategy(error.code),
        confidence: parseFloat(confidenceMatch?.[1] || '0.5'),
        reasoning: reasoningMatch?.[1]?.trim() || 'AI generated fix',
        dependencies: error.dependencies || [],
        validated: false
      };

      return fix;
    } catch (error) {
      console.error('Failed to parse fix response:', error);
      return null;
    }
  }

  private getFixStrategy(code: string): string {
    const strategies: Record<string, string> = {
      'TS1434': 'syntax_cleanup',
      'TS2304': 'add_import',
      'TS2307': 'fix_module_path', 
      'TS2457': 'rename_type',
      'TS1005': 'add_punctuation',
      'TS1128': 'add_declaration'
    };

    return strategies[code] || 'manual_fix';
  }

  private async validateFix(fix: ErrorFix): Promise<boolean> {
    // Basic validation checks
    if (!fix.fixedText || fix.fixedText === fix.originalText) return false;
    if (fix.confidence < this.config.confidenceThreshold) return false;

    // Syntax validation
    try {
      // Check for common syntax issues
      if (fix.strategy === 'add_punctuation' && !fix.fixedText.match(/[;,]/)) return false;
      if (fix.strategy === 'add_import' && !fix.fixedText.includes('import')) return false;
      
      return true;
    } catch (error) {
      return false;
    }
  }

  private async getCachedFix(errorId: string): Promise<ErrorFix | null> {
    try {
      // Check cache using enhanced Loki
      const cached = await gpuLokiErrorAPI.getStats();
      return null; // Implement cache retrieval
    } catch {
      return null;
    }
  }

  private async cacheFixAttempt(errorId: string, fix: ErrorFix) {
    const attempt: FixAttempt = {
      id: `fix_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      errorId,
      strategy: fix.strategy,
      originalCode: fix.originalText,
      fixedCode: fix.fixedText,
      confidence: fix.confidence,
      applied: false,
      result: 'success',
      timestamp: new Date(),
      llmModel: this.config.model
    };

    const history = this.fixHistory.get(errorId) || [];
    history.push(attempt);
    this.fixHistory.set(errorId, history);
  }

  private createBatches<T>(items: T[], batchSize: number): T[][] {
    const batches = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }
    return batches;
  }

  async applyFixes(fixes: ErrorFix[]): Promise<{applied: number, failed: number, results: any[]}> {
    console.log(`📝 Applying ${fixes.length} fixes...`);
    
    const results = [];
    let applied = 0;
    let failed = 0;

    for (const fix of fixes) {
      try {
        if (fix.validated && fix.confidence >= this.config.confidenceThreshold) {
          const result = await this.applyFix(fix);
          results.push(result);
          
          if (result.success) {
            applied++;
          } else {
            failed++;
          }
        } else {
          results.push({
            errorId: fix.errorId,
            success: false,
            reason: 'Fix not validated or confidence too low'
          });
          failed++;
        }
      } catch (error) {
        console.error(`Failed to apply fix for ${fix.errorId}:`, error);
        failed++;
      }
    }

    console.log(`✅ Applied: ${applied}, Failed: ${failed}`);
    return { applied, failed, results };
  }

  private async applyFix(fix: ErrorFix): Promise<{errorId: string, success: boolean, reason?: string}> {
    try {
      // Read the file
      const response = await fetch(`/api/files/read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: fix.file })
      });

      if (!response.ok) {
        return { errorId: fix.errorId, success: false, reason: 'Could not read file' };
      }

      const { content } = await response.json();
      const lines = content.split('\n');
      
      // Apply the fix
      if (fix.line <= lines.length) {
        lines[fix.line - 1] = fix.fixedText;
        
        // Write the file back
        const writeResponse = await fetch(`/api/files/write`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file: fix.file,
            content: lines.join('\n')
          })
        });

        if (writeResponse.ok) {
          return { errorId: fix.errorId, success: true };
        } else {
          return { errorId: fix.errorId, success: false, reason: 'Could not write file' };
        }
      } else {
        return { errorId: fix.errorId, success: false, reason: 'Line number out of range' };
      }
    } catch (error) {
      return { errorId: fix.errorId, success: false, reason: String(error) };
    }
  }

  getFixHistory(errorId?: string): FixAttempt[] {
    if (errorId) {
      return this.fixHistory.get(errorId) || [];
    } else {
      return Array.from(this.fixHistory.values()).flat();
    }
  }

  getStats() {
    const allAttempts = this.getFixHistory();
    return {
      totalAttempts: allAttempts.length,
      successfulFixes: allAttempts.filter(a => a.result === 'success').length,
      failedFixes: allAttempts.filter(a => a.result === 'failed').length,
      averageConfidence: allAttempts.reduce((sum, a) => sum + a.confidence, 0) / allAttempts.length || 0,
      appliedFixes: allAttempts.filter(a => a.applied).length
    };
  }
}

// ======================================================================
// STORE INTEGRATION
// ======================================================================

export const aiErrorFixer = new AIErrorFixer();

export const errorFixerStore = writable({
  initialized: false,
  fixing: false,
  fixes: [] as ErrorFix[],
  appliedFixes: 0,
  failedFixes: 0,
  stats: {
    totalAttempts: 0,
    successfulFixes: 0,
    failedFixes: 0,
    averageConfidence: 0,
    appliedFixes: 0
  }
});

export const fixerProgressStore = derived(errorFixerStore, ($store) => ({
  active: $store.fixing,
  totalFixes: $store.fixes.length,
  applied: $store.appliedFixes,
  failed: $store.failedFixes,
  successRate: $store.appliedFixes / ($store.appliedFixes + $store.failedFixes) || 0
}));

// ======================================================================
// PUBLIC API
// ======================================================================

export const aiErrorFixerAPI = {
  async initialize() {
    await gpuLokiErrorAPI.initialize();
    await parallelAnalysisAPI.initialize();
    errorFixerStore.update(state => ({ ...state, initialized: true }));
  },

  async processAndFixErrors(tscOutput: string) {
    errorFixerStore.update(state => ({ ...state, fixing: true }));

    try {
      // 1. Process errors with GPU orchestrator
      const analysisResults = await gpuLokiErrorAPI.processErrors(tscOutput);
      
      // 2. Generate fixes with AI
      const fixes = await aiErrorFixer.fixErrors(analysisResults);
      
      // 3. Apply validated fixes
      const applyResults = await aiErrorFixer.applyFixes(fixes);
      
      // 4. Update store
      const stats = aiErrorFixer.getStats();
      errorFixerStore.update(state => ({
        ...state,
        fixing: false,
        fixes,
        appliedFixes: applyResults.applied,
        failedFixes: applyResults.failed,
        stats
      }));

      return {
        totalErrors: analysisResults.length,
        fixableErrors: fixes.length,
        appliedFixes: applyResults.applied,
        failedFixes: applyResults.failed,
        fixes: fixes
      };
    } catch (error) {
      console.error('Error fixing pipeline failed:', error);
      errorFixerStore.update(state => ({ ...state, fixing: false }));
      throw error;
    }
  },

  async getStats() {
    return aiErrorFixer.getStats();
  },

  async getFixHistory(errorId?: string) {
    return aiErrorFixer.getFixHistory(errorId);
  }
};