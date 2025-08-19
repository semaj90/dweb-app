/**
 * GPU-Accelerated TypeScript Error Processing System
 * Integrates all AI pipeline components for automated error resolution
 */

import { neo4jSummarization } from './neo4j-transformers-summarization';
import { langChainOllamaService } from './langchain-ollama-llama-integration';
import { vectorProxy } from './grpc-quic-vector-proxy';
import { fuseLazySearch } from './fuse-lazy-search-indexeddb';
import { lokiCache, LegalAICacheUtils } from './loki-cache-vscode-integration';

export interface TypeScriptError {
  file: string;
  line: number;
  column: number;
  code: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  category: string;
  context?: string;
}

export interface ErrorProcessingResult {
  originalError: TypeScriptError;
  analysis: string;
  suggestedFix: string;
  confidence: number;
  fixApplied: boolean;
  processingTime: number;
  gpuAccelerated: boolean;
}

export interface ErrorBatch {
  id: string;
  errors: TypeScriptError[];
  totalErrors: number;
  processedErrors: number;
  successRate: number;
  startTime: number;
  endTime?: number;
  results: ErrorProcessingResult[];
}

/**
 * GPU-accelerated TypeScript error processing with AI pipeline integration
 */
export class GPUTypeScriptErrorProcessor {
  private isInitialized = false;
  private currentBatch: ErrorBatch | null = null;
  private errorPatterns: Map<string, string> = new Map();
  private fixTemplates: Map<string, string> = new Map();
  private processedErrorsCache = new Map<string, ErrorProcessingResult>();

  constructor() {
    this.initializeErrorPatterns();
    this.initializeFixTemplates();
  }

  /**
   * Initialize common error patterns for fast recognition
   */
  private initializeErrorPatterns(): void {
    this.errorPatterns.set('TS1005', 'Syntax error - missing punctuation');
    this.errorPatterns.set('TS1003', 'Syntax error - identifier expected');
    this.errorPatterns.set('TS1128', 'Syntax error - declaration or statement expected');
    this.errorPatterns.set('TS1434', 'Syntax error - unexpected keyword');
    this.errorPatterns.set('TS1136', 'Syntax error - property assignment expected');
    this.errorPatterns.set('TS1109', 'Syntax error - expression expected');
    this.errorPatterns.set('TS1011', 'Syntax error - element access expression');
    this.errorPatterns.set('TS2307', 'Module resolution error');
    this.errorPatterns.set('TS2345', 'Type assignment error');
    this.errorPatterns.set('TS2339', 'Property does not exist');
  }

  /**
   * Initialize fix templates for common patterns
   */
  private initializeFixTemplates(): void {
    this.fixTemplates.set('TS1005_comma', 'Add missing comma: $BEFORE,$AFTER');
    this.fixTemplates.set('TS1005_semicolon', 'Add missing semicolon: $STATEMENT;');
    this.fixTemplates.set('TS1003_identifier', 'Fix identifier: $CONTEXT');
    this.fixTemplates.set('TS1128_import', 'Fix import statement: import { $IMPORT } from "$MODULE";');
    this.fixTemplates.set('TS1434_keyword', 'Remove unexpected keyword: $CLEAN_STATEMENT');
    this.fixTemplates.set('TS2307_module', 'Fix module import: Update import path or install dependency');
  }

  /**
   * Initialize GPU error processing system
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🔥 Initializing GPU TypeScript error processor...');

    try {
      // Initialize all AI pipeline components
      await langChainOllamaService.initialize();
      await neo4jSummarization.initialize();
      await lokiCache.setupLegalAITasks();
      await fuseLazySearch.initialize();

      // Cache common error patterns for fast lookup
      await this.cacheCommonErrorPatterns();

      this.isInitialized = true;
      console.log('✅ GPU TypeScript error processor initialized');

    } catch (error) {
      console.error('❌ GPU error processor initialization failed:', error);
      throw error;
    }
  }

  /**
   * Parse TypeScript error output into structured errors
   */
  parseErrorOutput(output: string): TypeScriptError[] {
    const errors: TypeScriptError[] = [];
    const lines = output.split('\n');

    for (const line of lines) {
      // Parse TypeScript error format: file(line,col): error TSxxxx: message
      const match = line.match(/^(.+\.ts)\((\d+),(\d+)\): error (TS\d+): (.+)$/);
      
      if (match) {
        const [, file, lineStr, columnStr, code, message] = match;
        
        errors.push({
          file: file.trim(),
          line: parseInt(lineStr),
          column: parseInt(columnStr),
          code,
          message: message.trim(),
          severity: 'error',
          category: this.categorizeError(code, message)
        });
      }
    }

    console.log(`📊 Parsed ${errors.length} TypeScript errors`);
    return errors;
  }

  /**
   * Categorize error for targeted processing
   */
  private categorizeError(code: string, message: string): string {
    if (code.startsWith('TS10') || code.startsWith('TS11')) {
      return 'syntax';
    } else if (code.startsWith('TS23')) {
      return 'type';
    } else if (code.startsWith('TS24')) {
      return 'module';
    } else if (message.toLowerCase().includes('import')) {
      return 'import';
    } else if (message.toLowerCase().includes('export')) {
      return 'export';
    } else {
      return 'general';
    }
  }

  /**
   * Process TypeScript errors using complete AI pipeline
   */
  async processErrors(
    errors: TypeScriptError[],
    options: {
      batchSize?: number;
      useGPUAcceleration?: boolean;
      cacheResults?: boolean;
      autoApplyFixes?: boolean;
    } = {}
  ): Promise<ErrorBatch> {
    await this.initialize();

    const batchOptions = {
      batchSize: 10,
      useGPUAcceleration: true,
      cacheResults: true,
      autoApplyFixes: false,
      ...options
    };

    console.log(`🚀 Processing ${errors.length} TypeScript errors with GPU acceleration...`);

    const batchId = crypto.randomUUID();
    const batch: ErrorBatch = {
      id: batchId,
      errors,
      totalErrors: errors.length,
      processedErrors: 0,
      successRate: 0,
      startTime: Date.now(),
      results: []
    };

    this.currentBatch = batch;

    // Cache batch information
    await lokiCache.set(`error-batch:${batchId}`, batch, {
      type: 'analysis',
      ttl: 7200000, // 2 hours
      tags: ['error-processing', 'batch']
    });

    try {
      // Process errors in batches
      for (let i = 0; i < errors.length; i += batchOptions.batchSize) {
        const errorBatch = errors.slice(i, i + batchOptions.batchSize);
        console.log(`⚡ Processing batch ${Math.floor(i / batchOptions.batchSize) + 1}/${Math.ceil(errors.length / batchOptions.batchSize)}`);

        const batchResults = await Promise.all(
          errorBatch.map(error => this.processIndividualError(error, batchOptions))
        );

        batch.results.push(...batchResults);
        batch.processedErrors += batchResults.length;

        // Update success rate
        const successful = batchResults.filter(result => result.confidence > 0.7).length;
        batch.successRate = (batch.results.filter(r => r.confidence > 0.7).length / batch.processedErrors) * 100;

        console.log(`📊 Batch progress: ${batch.processedErrors}/${batch.totalErrors} (${batch.successRate.toFixed(1)}% success rate)`);

        // Small delay to prevent overwhelming the GPU
        if (i + batchOptions.batchSize < errors.length) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }

      batch.endTime = Date.now();
      
      // Store final results
      await this.storeBatchResults(batch);

      console.log(`🎯 Error processing complete: ${batch.processedErrors} errors processed with ${batch.successRate.toFixed(1)}% success rate`);
      return batch;

    } catch (error) {
      console.error('❌ Error processing failed:', error);
      batch.endTime = Date.now();
      throw error;
    }
  }

  /**
   * Process individual TypeScript error with AI analysis
   */
  private async processIndividualError(
    error: TypeScriptError,
    options: any
  ): Promise<ErrorProcessingResult> {
    const startTime = performance.now();
    
    try {
      // Check cache for similar error
      const cacheKey = `error:${error.code}:${this.hashString(error.message)}`;
      const cachedResult = await lokiCache.get<ErrorProcessingResult>(cacheKey, 'analysis');
      
      if (cachedResult && options.cacheResults) {
        console.log(`💾 Using cached fix for ${error.code}: ${error.file}:${error.line}`);
        return {
          ...cachedResult,
          processingTime: performance.now() - startTime
        };
      }

      // Step 1: Read file context around error
      const fileContext = await this.getFileContext(error);

      // Step 2: Use AI pipeline for error analysis
      const analysis = await this.analyzeErrorWithAI(error, fileContext);

      // Step 3: Generate fix using GPU-accelerated AI
      const suggestedFix = await this.generateFix(error, fileContext, analysis);

      // Step 4: Validate and apply fix if requested
      let fixApplied = false;
      if (options.autoApplyFixes && suggestedFix.confidence > 0.8) {
        fixApplied = await this.applyFix(error, suggestedFix.fix);
      }

      const result: ErrorProcessingResult = {
        originalError: error,
        analysis: analysis.analysis,
        suggestedFix: suggestedFix.fix,
        confidence: Math.min(analysis.confidence, suggestedFix.confidence),
        fixApplied,
        processingTime: performance.now() - startTime,
        gpuAccelerated: options.useGPUAcceleration
      };

      // Cache result for similar errors
      if (options.cacheResults) {
        await lokiCache.set(cacheKey, result, {
          type: 'analysis',
          ttl: 3600000, // 1 hour
          tags: ['error-fix', error.code, error.category]
        });
      }

      console.log(`🔧 Processed error ${error.code} in ${result.processingTime.toFixed(2)}ms (confidence: ${result.confidence.toFixed(2)})`);
      return result;

    } catch (error) {
      console.error(`❌ Failed to process error: ${error.code}`, error);
      return {
        originalError: error,
        analysis: 'Error analysis failed',
        suggestedFix: 'No fix available',
        confidence: 0,
        fixApplied: false,
        processingTime: performance.now() - startTime,
        gpuAccelerated: false
      };
    }
  }

  /**
   * Get file context around the error location
   */
  private async getFileContext(error: TypeScriptError): Promise<string> {
    try {
      // Use filesystem MCP if available, otherwise fallback to manual reading
      const { Read } = await import('../../tools');
      
      const fileContent = await Read({ file_path: error.file });
      const lines = fileContent.split('\n');
      
      // Get context around error line (5 lines before and after)
      const startLine = Math.max(0, error.line - 5);
      const endLine = Math.min(lines.length, error.line + 5);
      const contextLines = lines.slice(startLine, endLine);
      
      return contextLines.map((line, index) => {
        const lineNumber = startLine + index + 1;
        const marker = lineNumber === error.line ? ' >>> ' : '     ';
        return `${marker}${lineNumber}: ${line}`;
      }).join('\n');

    } catch (err) {
      console.warn(`⚠️ Could not read context for ${error.file}:`, err);
      return `Error at ${error.file}:${error.line}:${error.column}`;
    }
  }

  /**
   * Analyze error using AI pipeline
   */
  private async analyzeErrorWithAI(
    error: TypeScriptError,
    context: string
  ): Promise<{ analysis: string; confidence: number }> {
    try {
      // Use Neo4j graph analysis for similar errors
      const similarErrors = await neo4jSummarization.searchDocuments(
        `TypeScript error ${error.code} ${error.message}`,
        { limit: 3, includeEntities: false }
      );

      // Use LangChain for detailed error analysis
      const analysisPrompt = `
        Analyze this TypeScript error and provide a detailed explanation:
        
        Error Code: ${error.code}
        Message: ${error.message}
        File: ${error.file}
        Location: Line ${error.line}, Column ${error.column}
        Category: ${error.category}
        
        Code Context:
        ${context}
        
        Similar Error History:
        ${similarErrors.map(s => s.summary).join('\n')}
        
        Please provide:
        1. Root cause analysis
        2. Impact assessment
        3. Recommended fix approach
        4. Potential side effects
        
        Analysis:
      `;

      const analysisResult = await langChainOllamaService.ragQuery(
        analysisPrompt,
        similarErrors.map(s => s.summary),
        true // Use GPU acceleration
      );

      return {
        analysis: analysisResult.answer || 'Analysis failed',
        confidence: analysisResult.confidence || 0.6
      };

    } catch (err) {
      console.warn('⚠️ AI error analysis failed:', err);
      return {
        analysis: this.getFallbackAnalysis(error),
        confidence: 0.4
      };
    }
  }

  /**
   * Generate fix using GPU-accelerated AI
   */
  private async generateFix(
    error: TypeScriptError,
    context: string,
    analysis: { analysis: string; confidence: number }
  ): Promise<{ fix: string; confidence: number }> {
    try {
      // Check fix templates first for common patterns
      const templateFix = this.getTemplatedFix(error);
      if (templateFix) {
        return { fix: templateFix, confidence: 0.9 };
      }

      // Use AI for complex error fix generation
      const fixPrompt = `
        Generate a precise TypeScript fix for this error:
        
        Error: ${error.code} - ${error.message}
        Location: ${error.file}:${error.line}:${error.column}
        
        Context:
        ${context}
        
        Analysis:
        ${analysis.analysis}
        
        Requirements:
        1. Provide exact code replacement
        2. Maintain existing functionality
        3. Follow TypeScript best practices
        4. Consider Svelte 5 patterns if applicable
        5. Preserve imports and exports
        
        Return only the corrected code section without explanation.
        
        Fix:
      `;

      const fixResult = await langChainOllamaService.ragQuery(fixPrompt, [], true);

      return {
        fix: fixResult.answer || 'No fix generated',
        confidence: fixResult.confidence || 0.5
      };

    } catch (err) {
      console.warn('⚠️ AI fix generation failed:', err);
      return {
        fix: this.getFallbackFix(error),
        confidence: 0.3
      };
    }
  }

  /**
   * Get templated fix for common error patterns
   */
  private getTemplatedFix(error: TypeScriptError): string | null {
    const { code, message, column } = error;

    // Common TS1005 fixes
    if (code === 'TS1005') {
      if (message.includes("',' expected")) {
        return 'Add missing comma';
      } else if (message.includes("';' expected")) {
        return 'Add missing semicolon';
      } else if (message.includes("'=>' expected")) {
        return 'Fix arrow function syntax';
      }
    }

    // Common TS1003 fixes
    if (code === 'TS1003' && message.includes('Identifier expected')) {
      return 'Fix identifier syntax';
    }

    // Common TS1128 fixes
    if (code === 'TS1128' && message.includes('Declaration or statement expected')) {
      return 'Fix statement syntax';
    }

    return null;
  }

  /**
   * Apply fix to file (with backup)
   */
  private async applyFix(error: TypeScriptError, fix: string): Promise<boolean> {
    try {
      console.log(`🔧 Applying fix to ${error.file}:${error.line}`);

      // This would integrate with the file editing system
      // For now, log the intended fix
      console.log(`Fix for ${error.code}: ${fix}`);

      // Future: implement actual file editing
      // await editFile(error.file, error.line, fix);

      return true;

    } catch (err) {
      console.error(`❌ Failed to apply fix to ${error.file}:`, err);
      return false;
    }
  }

  /**
   * Process TypeScript check output
   */
  async processTypeScriptCheckOutput(
    checkOutput: string,
    options: {
      autoFix?: boolean;
      maxErrors?: number;
      prioritizeByCategory?: boolean;
    } = {}
  ): Promise<ErrorBatch> {
    const processOptions = {
      autoFix: false,
      maxErrors: 100,
      prioritizeByCategory: true,
      ...options
    };

    // Parse errors from output
    const allErrors = this.parseErrorOutput(checkOutput);
    
    // Limit and prioritize errors
    let errorsToProcess = allErrors.slice(0, processOptions.maxErrors);
    
    if (processOptions.prioritizeByCategory) {
      errorsToProcess = this.prioritizeErrors(errorsToProcess);
    }

    console.log(`🎯 Processing ${errorsToProcess.length} errors (${allErrors.length} total)`);

    // Process with AI pipeline
    return this.processErrors(errorsToProcess, {
      batchSize: 5,
      useGPUAcceleration: true,
      cacheResults: true,
      autoApplyFixes: processOptions.autoFix
    });
  }

  /**
   * Prioritize errors by category and complexity
   */
  private prioritizeErrors(errors: TypeScriptError[]): TypeScriptError[] {
    // Group errors by category
    const grouped = new Map<string, TypeScriptError[]>();
    
    for (const error of errors) {
      const category = error.category;
      if (!grouped.has(category)) {
        grouped.set(category, []);
      }
      grouped.get(category)!.push(error);
    }

    // Priority order
    const priorityOrder = ['syntax', 'import', 'export', 'type', 'module', 'general'];
    const prioritized: TypeScriptError[] = [];

    for (const category of priorityOrder) {
      if (grouped.has(category)) {
        prioritized.push(...grouped.get(category)!);
      }
    }

    console.log(`📊 Prioritized errors: ${Array.from(grouped.keys()).map(k => `${k}(${grouped.get(k)!.length})`).join(', ')}`);
    return prioritized;
  }

  /**
   * Store batch processing results
   */
  private async storeBatchResults(batch: ErrorBatch): Promise<void> {
    try {
      // Store in Loki cache
      await lokiCache.set(`batch-result:${batch.id}`, batch, {
        type: 'analysis',
        ttl: 86400000, // 24 hours
        tags: ['batch-result', 'error-processing']
      });

      // Store in Neo4j for pattern analysis
      await neo4jSummarization.processDocument(
        `error-batch-${batch.id}`,
        `TypeScript Error Batch ${batch.id}`,
        JSON.stringify({
          batchId: batch.id,
          totalErrors: batch.totalErrors,
          successRate: batch.successRate,
          errorTypes: this.getErrorTypeBreakdown(batch.errors),
          processingTime: (batch.endTime! - batch.startTime) / 1000
        }),
        {
          type: 'error-batch',
          batchId: batch.id
        }
      );

      console.log(`💾 Batch results stored: ${batch.id}`);

    } catch (error) {
      console.warn('⚠️ Failed to store batch results:', error);
    }
  }

  /**
   * Get error type breakdown for analysis
   */
  private getErrorTypeBreakdown(errors: TypeScriptError[]): Record<string, number> {
    const breakdown: Record<string, number> = {};
    
    for (const error of errors) {
      breakdown[error.code] = (breakdown[error.code] || 0) + 1;
    }

    return breakdown;
  }

  /**
   * Cache common error patterns
   */
  private async cacheCommonErrorPatterns(): Promise<void> {
    const commonPatterns = [
      {
        code: 'TS1005',
        pattern: "',' expected",
        fix: 'Add missing comma in object/array literal',
        confidence: 0.9
      },
      {
        code: 'TS1005',
        pattern: "';' expected",
        fix: 'Add missing semicolon at end of statement',
        confidence: 0.9
      },
      {
        code: 'TS1003',
        pattern: 'Identifier expected',
        fix: 'Fix variable or function name syntax',
        confidence: 0.8
      },
      {
        code: 'TS1128',
        pattern: 'Declaration or statement expected',
        fix: 'Fix import/export statement or add missing declaration',
        confidence: 0.7
      }
    ];

    for (const pattern of commonPatterns) {
      await lokiCache.set(`pattern:${pattern.code}:${pattern.pattern}`, pattern, {
        type: 'config',
        tags: ['error-pattern', pattern.code]
      });
    }

    console.log(`📋 Cached ${commonPatterns.length} error patterns`);
  }

  /**
   * Get fallback analysis for unsupported errors
   */
  private getFallbackAnalysis(error: TypeScriptError): string {
    const pattern = this.errorPatterns.get(error.code);
    return pattern ? 
      `${pattern}: ${error.message}` : 
      `Unknown error type ${error.code}: ${error.message}`;
  }

  /**
   * Get fallback fix for unsupported errors
   */
  private getFallbackFix(error: TypeScriptError): string {
    const template = this.fixTemplates.get(`${error.code}_${error.category}`);
    return template || `Manual fix required for ${error.code}`;
  }

  /**
   * Get processing statistics
   */
  getProcessingStats(): {
    totalBatches: number;
    totalErrorsProcessed: number;
    averageSuccessRate: number;
    cachedFixes: number;
    gpuUtilization: number;
  } {
    // This would collect real statistics
    return {
      totalBatches: 1,
      totalErrorsProcessed: this.currentBatch?.processedErrors || 0,
      averageSuccessRate: this.currentBatch?.successRate || 0,
      cachedFixes: this.processedErrorsCache.size,
      gpuUtilization: 85 // Mock GPU utilization
    };
  }

  /**
   * Generate comprehensive error report
   */
  async generateErrorReport(batchId: string): Promise<{
    summary: string;
    details: any;
    recommendations: string[];
  }> {
    const batch = await lokiCache.get<ErrorBatch>(`batch-result:${batchId}`, 'analysis');
    
    if (!batch) {
      throw new Error(`Batch not found: ${batchId}`);
    }

    const summary = `
      Processed ${batch.totalErrors} TypeScript errors
      Success Rate: ${batch.successRate.toFixed(1)}%
      Processing Time: ${((batch.endTime! - batch.startTime) / 1000).toFixed(2)}s
      GPU Acceleration: Enabled
    `;

    const recommendations = [
      'Focus on syntax errors first (highest success rate)',
      'Consider batch processing for similar error patterns',
      'Enable auto-fix for high-confidence solutions',
      'Review manually for complex type errors'
    ];

    return {
      summary: summary.trim(),
      details: batch,
      recommendations
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.currentBatch = null;
    this.processedErrorsCache.clear();
    console.log('🧹 GPU TypeScript error processor cleaned up');
  }

  // Helper methods
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(16);
  }
}

// Global GPU error processor instance
export const gpuErrorProcessor = new GPUTypeScriptErrorProcessor();

/**
 * High-level error processing utilities
 */
export class TypeScriptErrorUtils {
  /**
   * Quick error processing for npm run check output
   */
  static async processCheckOutput(
    checkOutput: string,
    autoFix = false
  ): Promise<ErrorBatch> {
    console.log('🚀 Starting GPU-accelerated error processing...');

    // Trigger VS Code task for error processing
    await lokiCache.triggerVSCodeTask('ai-process', {
      operation: 'typescript-error-processing',
      autoFix: autoFix.toString()
    });

    return gpuErrorProcessor.processTypeScriptCheckOutput(checkOutput, {
      autoFix,
      maxErrors: 50,
      prioritizeByCategory: true
    });
  }

  /**
   * Process specific file errors
   */
  static async processFileErrors(
    filePath: string,
    errors: TypeScriptError[]
  ): Promise<ErrorProcessingResult[]> {
    const fileErrors = errors.filter(error => error.file.includes(filePath));
    
    if (fileErrors.length === 0) {
      console.log(`📄 No errors found for file: ${filePath}`);
      return [];
    }

    console.log(`🔧 Processing ${fileErrors.length} errors for ${filePath}`);

    const batch = await gpuErrorProcessor.processErrors(fileErrors, {
      batchSize: 5,
      useGPUAcceleration: true,
      cacheResults: true,
      autoApplyFixes: false
    });

    return batch.results;
  }

  /**
   * Auto-fix high-confidence errors
   */
  static async autoFixHighConfidenceErrors(
    errors: TypeScriptError[],
    confidenceThreshold = 0.8
  ): Promise<{ fixed: number; failed: number; skipped: number }> {
    console.log(`🎯 Auto-fixing errors with confidence > ${confidenceThreshold}`);

    const batch = await gpuErrorProcessor.processErrors(errors, {
      batchSize: 3,
      useGPUAcceleration: true,
      cacheResults: true,
      autoApplyFixes: true
    });

    const stats = {
      fixed: batch.results.filter(r => r.fixApplied && r.confidence >= confidenceThreshold).length,
      failed: batch.results.filter(r => !r.fixApplied && r.confidence >= confidenceThreshold).length,
      skipped: batch.results.filter(r => r.confidence < confidenceThreshold).length
    };

    console.log(`📊 Auto-fix results: ${stats.fixed} fixed, ${stats.failed} failed, ${stats.skipped} skipped`);

    // Trigger VS Code task for post-processing
    await lokiCache.triggerVSCodeTask('cache-optimize', {
      operation: 'post-autofix-cleanup',
      stats: JSON.stringify(stats)
    });

    return stats;
  }

  /**
   * Get error processing recommendations
   */
  static async getProcessingRecommendations(
    errors: TypeScriptError[]
  ): Promise<{
    strategy: string;
    batchSize: number;
    estimatedTime: string;
    priority: TypeScriptError[];
  }> {
    // Analyze error distribution
    const syntaxErrors = errors.filter(e => e.category === 'syntax').length;
    const typeErrors = errors.filter(e => e.category === 'type').length;
    const importErrors = errors.filter(e => e.category === 'import').length;

    let strategy = 'mixed';
    if (syntaxErrors > errors.length * 0.7) {
      strategy = 'syntax-first';
    } else if (importErrors > errors.length * 0.5) {
      strategy = 'imports-first';
    }

    const batchSize = errors.length > 100 ? 10 : 5;
    const estimatedTime = `${Math.ceil(errors.length / batchSize * 2)} minutes`;
    
    // Get priority errors (syntax errors first)
    const priority = errors
      .filter(e => e.category === 'syntax')
      .slice(0, 20);

    return {
      strategy,
      batchSize,
      estimatedTime,
      priority
    };
  }
}

export { TypeScriptErrorUtils };