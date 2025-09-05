#!/usr/bin/env node

/**
 * Automated TypeScript Error Analysis Pipeline
 * Captures npm run check errors, processes with Go SIMD parser, and feeds to Claude AI
 */

import { spawn } from 'child_process';
import { writeFile, readFile } from 'fs/promises';
import { join } from 'path';
import fetch from 'node-fetch';

const CONFIG = {
  goServiceUrl: 'http://localhost:8082',
  outputDir: './error-analysis',
  claudeApiKey: process.env.CLAUDE_API_KEY || '',
  projectType: 'sveltekit-legal-ai'
};

class ErrorAnalyzer {
  constructor() {
    this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.results = {};
  }

  async runAnalysis() {
    console.log('🚀 Starting automated error analysis pipeline...');
    
    try {
      // Step 1: Capture TypeScript errors
      console.log('📋 Step 1: Capturing TypeScript errors...');
      const errorLog = await this.captureTypeScriptErrors();
      
      // Step 2: Parse with Go SIMD JSON service
      console.log('🔍 Step 2: Parsing errors with Go SIMD service...');
      const parsedErrors = await this.parseErrorsWithGo(errorLog);
      
      // Step 3: Generate Claude AI analysis
      console.log('🤖 Step 3: Generating Claude AI analysis...');
      const claudeAnalysis = await this.generateClaudeAnalysis(parsedErrors);
      
      // Step 4: Save comprehensive results
      console.log('💾 Step 4: Saving analysis results...');
      await this.saveResults({
        timestamp: this.timestamp,
        errorLog,
        parsedErrors,
        claudeAnalysis,
        summary: this.generateSummary(parsedErrors, claudeAnalysis)
      });
      
      console.log('✅ Error analysis pipeline completed successfully!');
      console.log(`📁 Results saved to: ${CONFIG.outputDir}/analysis-${this.timestamp}.json`);
      
      return this.results;
      
    } catch (error) {
      console.error('❌ Error analysis pipeline failed:', error);
      throw error;
    }
  }

  async captureTypeScriptErrors() {
    return new Promise((resolve, reject) => {
      const process = spawn('npm', ['run', 'check:base'], {
        cwd: process.cwd(),
        shell: true
      });
      
      let stdout = '';
      let stderr = '';
      
      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      process.on('close', (code) => {
        // TypeScript errors are expected, so we capture both stdout and stderr
        const fullOutput = stdout + stderr;
        resolve(fullOutput);
      });
      
      process.on('error', (error) => {
        reject(error);
      });
      
      // Set timeout for long-running checks
      setTimeout(() => {
        process.kill('SIGTERM');
        resolve(stdout + stderr);
      }, 120000); // 2 minutes timeout
    });
  }

  async parseErrorsWithGo(errorLog) {
    try {
      const response = await fetch(`${CONFIG.goServiceUrl}/api/parse-errors`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          errorLog,
          context: 'SvelteKit 2 + Svelte 5 Legal AI Application'
        })
      });
      
      if (!response.ok) {
        throw new Error(`Go service responded with status: ${response.status}`);
      }
      
      return await response.json();
      
    } catch (error) {
      console.warn('⚠️ Go service unavailable, falling back to basic parsing...');
      return this.basicErrorParsing(errorLog);
    }
  }

  basicErrorParsing(errorLog) {
    // Fallback basic parsing if Go service is unavailable
    const lines = errorLog.split('\n');
    const errors = [];
    const errorPattern = /^(.+?)\((\d+),(\d+)\): error (TS\d+): (.+)$/;
    
    for (const line of lines) {
      const match = line.match(errorPattern);
      if (match) {
        errors.push({
          file: match[1],
          line: parseInt(match[2]),
          column: parseInt(match[3]),
          errorCode: match[4],
          message: match[5],
          category: this.categorizeError(match[4]),
          severity: this.getSeverity(match[4])
        });
      }
    }
    
    return {
      timestamp: new Date().toISOString(),
      totalErrors: errors.length,
      errors,
      categories: this.groupByCategory(errors),
      priorities: this.groupBySeverity(errors)
    };
  }

  categorizeError(errorCode) {
    const categories = {
      'TS2339': 'property_access',
      'TS2345': 'argument_type',
      'TS2322': 'assignment_type',
      'TS2554': 'argument_count',
      'TS2307': 'module_resolution',
      'TS2305': 'export_member',
      'TS2353': 'object_literal',
      'TS2341': 'private_property',
      'TS2484': 'export_conflict'
    };
    return categories[errorCode] || 'unknown';
  }

  getSeverity(errorCode) {
    const critical = ['TS2307', 'TS2305', 'TS2484'];
    const high = ['TS2339', 'TS2345', 'TS2322', 'TS2554'];
    
    if (critical.includes(errorCode)) return 'critical';
    if (high.includes(errorCode)) return 'high';
    return 'medium';
  }

  groupByCategory(errors) {
    return errors.reduce((acc, error) => {
      acc[error.category] = (acc[error.category] || 0) + 1;
      return acc;
    }, {});
  }

  groupBySeverity(errors) {
    return errors.reduce((acc, error) => {
      acc[error.severity] = (acc[error.severity] || 0) + 1;
      return acc;
    }, {});
  }

  async generateClaudeAnalysis(parsedErrors) {
    if (!CONFIG.claudeApiKey) {
      console.warn('⚠️ No Claude API key provided, skipping AI analysis');
      return {
        analysis: 'Claude API key not provided',
        suggestions: ['Configure CLAUDE_API_KEY environment variable'],
        priorities: ['Fix critical errors first', 'Address module resolution issues'],
        fixPlan: ['Review error patterns', 'Apply systematic fixes']
      };
    }

    try {
      // Generate Claude prompt using Go service
      const promptResponse = await fetch(`${CONFIG.goServiceUrl}/api/claude-prompt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          errors: parsedErrors.errors?.slice(0, 20) || [], // Limit to top 20 errors
          context: 'SvelteKit 2 + Svelte 5 Legal AI application with enhanced RAG, XState machines, and AI services',
          projectType: CONFIG.projectType
        })
      });

      if (!promptResponse.ok) {
        throw new Error('Failed to generate Claude prompt');
      }

      const { prompt } = await promptResponse.json();

      // Send to Claude AI (mock response for now - integrate with actual Claude API)
      const claudeResponse = await this.callClaudeAPI(prompt);

      return claudeResponse;

    } catch (error) {
      console.error('Failed to generate Claude analysis:', error);
      return {
        analysis: `Analysis failed: ${error.message}`,
        suggestions: ['Review error logs manually', 'Check service connectivity'],
        priorities: ['Fix critical blocking errors first'],
        fixPlan: ['Manual review required due to API failure']
      };
    }
  }

  async callClaudeAPI(prompt) {
    // Mock Claude API response - replace with actual Claude API integration
    // This would use the Anthropic Claude API in production
    return {
      analysis: `Analyzed ${this.results?.totalErrors || 'unknown'} TypeScript errors. Main issues: schema mismatches, XState configuration, and import resolution.`,
      suggestions: [
        'Update database schema types to match API expectations',
        'Fix XState machine actor configurations',
        'Resolve module import/export issues',
        'Add missing type definitions'
      ],
      priorities: [
        'Critical: Fix module resolution errors (TS2307, TS2305)',
        'High: Resolve property access errors (TS2339)',
        'Medium: Fix type assignment issues (TS2322)'
      ],
      fixPlan: [
        '1. Fix database schema imports and type definitions',
        '2. Update XState machines to v5 syntax',
        '3. Resolve API service type mismatches',
        '4. Add missing export members',
        '5. Run incremental checks after each fix'
      ]
    };
  }

  generateSummary(parsedErrors, claudeAnalysis) {
    return {
      totalErrors: parsedErrors.totalErrors || 0,
      criticalCount: parsedErrors.priorities?.critical || 0,
      highCount: parsedErrors.priorities?.high || 0,
      mediumCount: parsedErrors.priorities?.medium || 0,
      topCategories: Object.entries(parsedErrors.categories || {})
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([category, count]) => ({ category, count })),
      claudeSuggestions: claudeAnalysis.suggestions?.length || 0,
      analysisQuality: parsedErrors.totalErrors > 0 ? 'detailed' : 'none',
      recommendedAction: this.getRecommendedAction(parsedErrors)
    };
  }

  getRecommendedAction(parsedErrors) {
    const critical = parsedErrors.priorities?.critical || 0;
    const total = parsedErrors.totalErrors || 0;

    if (critical > 10) return 'immediate_action_required';
    if (total > 50) return 'systematic_cleanup_needed';
    if (total > 10) return 'incremental_fixes_recommended';
    return 'maintenance_mode';
  }

  async saveResults(results) {
    this.results = results;
    
    // Ensure output directory exists
    try {
      await import('fs').then(fs => fs.promises.mkdir(CONFIG.outputDir, { recursive: true }));
    } catch (error) {
      console.warn('Could not create output directory:', error);
    }

    const outputFile = join(CONFIG.outputDir, `analysis-${this.timestamp}.json`);
    
    try {
      await writeFile(outputFile, JSON.stringify(results, null, 2));
      console.log(`📁 Results saved to: ${outputFile}`);
    } catch (error) {
      console.error('Failed to save results:', error);
    }

    // Also save a latest.json for easy access
    try {
      const latestFile = join(CONFIG.outputDir, 'latest.json');
      await writeFile(latestFile, JSON.stringify(results, null, 2));
    } catch (error) {
      console.warn('Could not save latest.json:', error);
    }
  }
}

// CLI interface
async function main() {
  const analyzer = new ErrorAnalyzer();
  
  try {
    const results = await analyzer.runAnalysis();
    
    // Print summary
    console.log('\n📊 Analysis Summary:');
    console.log(`Total Errors: ${results.summary.totalErrors}`);
    console.log(`Critical: ${results.summary.criticalCount}`);
    console.log(`High: ${results.summary.highCount}`);
    console.log(`Medium: ${results.summary.mediumCount}`);
    console.log(`Recommended Action: ${results.summary.recommendedAction}`);
    
    if (results.claudeAnalysis.fixPlan) {
      console.log('\n🔧 Fix Plan:');
      results.claudeAnalysis.fixPlan.forEach((step, index) => {
        console.log(`${index + 1}. ${step}`);
      });
    }
    
    process.exit(0);
    
  } catch (error) {
    console.error('Analysis failed:', error);
    process.exit(1);
  }
}

// Export for use as module
export { ErrorAnalyzer, CONFIG };

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}