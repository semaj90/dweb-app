#!/usr/bin/env node

/**
 * Error Processor Daemon
 * Automated npm run check:auto with intelligent error processing
 */

import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import fetch from 'node-fetch';
import chokidar from 'chokidar';

const CHECK_INTERVAL = parseInt(process.env.CHECK_INTERVAL) || 300000; // 5 minutes
const AUTO_FIX_ENABLED = process.env.AUTO_FIX_ENABLED === 'true';
// Default now points at 8100 (dynamic service may shift from 8099 to 8100)
let RECOMMENDATION_ENDPOINT = process.env.RECOMMENDATION_ENDPOINT || 'http://localhost:8100/api/process-error-log';
const RECOMMENDATION_META_FILE = path.join('./logs','recommendation-service.json');
const RECOMMENDATION_HISTORY_FILE = path.join('./logs','recommendations-history.jsonl');
const LOG_DIRECTORY = process.env.LOG_DIRECTORY || './error-logs';
const CONFIDENCE_THRESHOLD = parseFloat(process.env.CONFIDENCE_THRESHOLD) || 0.85;

class ErrorProcessorDaemon {
  constructor() {
    this.isProcessing = false;
    this.lastCheckTime = 0;
    this.errorPatterns = new Map();
    this.autoFixCount = 0;
    this.totalErrorsFound = 0;
    this.watcher = null;

    this.initialize();
  }

  async initialize() {
    console.log('🤖 Error Processor Daemon starting...');
    console.log(`⏰ Check interval: ${CHECK_INTERVAL / 1000}s`);
    console.log(`🔧 Auto-fix enabled: ${AUTO_FIX_ENABLED}`);
    console.log(`🎯 Confidence threshold: ${CONFIDENCE_THRESHOLD}`);
    console.log(`📁 Log directory: ${LOG_DIRECTORY}`);
    console.log(`🌐 Recommendation endpoint: ${RECOMMENDATION_ENDPOINT}`);

    // Ensure log directory exists
    try {
      await fs.mkdir(LOG_DIRECTORY, { recursive: true });
    } catch (error) {
      console.error('Failed to create log directory:', error);
    }

    // Start file watcher for real-time error processing
    this.startFileWatcher();

    // Start periodic checks
    this.startPeriodicChecks();

    // Load existing error patterns
    await this.loadErrorPatterns();

    console.log('✅ Error Processor Daemon ready');
  }

  startFileWatcher() {
    console.log('👁️ Starting file watcher for error logs...');

    this.watcher = chokidar.watch(LOG_DIRECTORY, {
      ignored: /^\./, // ignore dotfiles
      persistent: true,
      ignoreInitial: true
    });

    this.watcher.on('add', async (filePath) => {
      if (path.extname(filePath) === '.log') {
        console.log(`📄 New error log detected: ${path.basename(filePath)}`);
        await this.processErrorLogFile(filePath);
      }
    });

    this.watcher.on('change', async (filePath) => {
      if (path.extname(filePath) === '.log') {
        console.log(`📝 Error log updated: ${path.basename(filePath)}`);
        await this.processErrorLogFile(filePath);
      }
    });
  }

  startPeriodicChecks() {
    console.log('⏰ Starting periodic error checks...');

    // Initial check
    setTimeout(() => this.runErrorCheck(), 5000);

    // Periodic checks
    setInterval(() => {
      this.runErrorCheck();
    }, CHECK_INTERVAL);
  }

  async runErrorCheck() {
    if (this.isProcessing) {
      console.log('⚠️ Error check already in progress, skipping...');
      return;
    }

    this.isProcessing = true;
    const startTime = Date.now();

    try {
      console.log('🔍 Running npm check:auto...');

      // Run the error check command
      const result = await this.executeCommand('npm', ['run', 'check:auto']);

      if (result.stderr || result.stdout.includes('error')) {
        console.log('❌ Errors detected during check');
        this.totalErrorsFound++;
      } else {
        console.log('✅ No errors found');
      }

      const duration = Date.now() - startTime;
      console.log(`⏱️ Check completed in ${duration}ms`);

    } catch (error) {
      console.error('❌ Error check failed:', error.message);
    } finally {
      this.isProcessing = false;
      this.lastCheckTime = Date.now();
    }
  }

  async processErrorLogFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf8');
      const fileName = path.basename(filePath);

      console.log(`📊 Processing error log: ${fileName} (${content.length} chars)`);

      // Send to recommendation service
      let recommendations = await this.getRecommendations(fileName, content);

      // Fallback: generate local recommendations from TS error patterns if remote produced none
      if (!recommendations || recommendations.length === 0) {
        const local = this.generateLocalRecommendations(content);
        if (local.length) {
          console.log(`🧪 Generated ${local.length} local recommendations (TS analysis)`);
          recommendations = local;
        }
      }

      if (recommendations && recommendations.length > 0) {
        console.log(`💡 Received ${recommendations.length} recommendations`);

        if (AUTO_FIX_ENABLED) {
          await this.applyAutoFixes(recommendations);
        }

        // Store error patterns for learning
        await this.storeErrorPatterns(content, recommendations);
      } else {
        console.log('🤷 No recommendations received');
      }

    } catch (error) {
      console.error('❌ Error processing log file:', error);
    }
  }

  async getRecommendations(logFile, content) {
    const maxAttempts = 5;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      // Refresh dynamic port if metadata file exists and no explicit override
      if (!process.env.RECOMMENDATION_ENDPOINT) {
        try {
          const metaRaw = await fs.readFile(RECOMMENDATION_META_FILE,'utf8');
          const meta = JSON.parse(metaRaw);
            if (meta?.port) {
              const dynamicEndpoint = `http://localhost:${meta.port}/api/process-error-log`;
              if (dynamicEndpoint !== RECOMMENDATION_ENDPOINT) {
                console.log(`🔄 Updating recommendation endpoint to ${dynamicEndpoint}`);
                RECOMMENDATION_ENDPOINT = dynamicEndpoint;
              }
            }
        } catch(_) { /* ignore if missing */ }
      }
      try {
        const response = await fetch(RECOMMENDATION_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ logFile, content, timestamp: new Date().toISOString() }),
          timeout: 30000
        });
        if (response.status === 404) {
          console.log('⚠️ Recommendation endpoint 404. Attempting to start local service...');
          this.spawnRecommendationService();
          await this.delay(300 * attempt);
          continue;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        const recs = result.recommendations || [];
        await this.appendRecommendationHistory({
          ts: new Date().toISOString(),
          logFile,
            attempt,
          count: recs.length,
          sample: content.slice(0,400),
          recommendations: recs
        });
        return recs;
      } catch (error) {
        console.warn(`⚠️ Attempt ${attempt} recommendation fetch failed: ${error.message}`);
        await this.delay(Math.min(250 * attempt, 1500));
      }
    }
    console.error('❌ Exhausted recommendation attempts');
    return [];
  }

  async appendRecommendationHistory(entry){
    try {
      await fs.appendFile(RECOMMENDATION_HISTORY_FILE, JSON.stringify(entry) + '\n','utf8');
    } catch(e){ /* non-fatal */ }
  }

  delay(ms){ return new Promise(r=>setTimeout(r,ms)); }

  spawnRecommendationService() {
    if (this._recommendationStarting) return;
    this._recommendationStarting = true;
    const child = spawn('node', ['scripts/recommendation-service.mjs'], { shell: true, stdio: 'inherit' });
    child.on('exit', code => {
      this._recommendationStarting = false;
      if (code !== 0) console.log('❌ Recommendation service exited with code', code);
    });
  }

  generateLocalRecommendations(content) {
    const recs = [];
    const push = (issue, autoFixable, confidence, codeFix, file=null) => recs.push({ issue, autoFixable, confidence, codeFix, file });

    // TS2304 / TS2552 Cannot find name
    const missingNameRegex = /error TS(?:2304|2552): Cannot find name '([A-Za-z0-9_]+)'/g;
    let m;
    let count = 0;
    while ((m = missingNameRegex.exec(content)) && count < 15) {
      push(`Missing symbol ${m[1]}`, false, 0.9, `// import or declare ${m[1]}`);
      count++;
    }

    // TS2339 Property does not exist
    const propRegex = /error TS2339: Property '([A-Za-z0-9_]+)' does not exist on type '([^']+)'/g;
    count = 0;
    while ((m = propRegex.exec(content)) && count < 15) {
      push(`Property ${m[1]} missing on ${m[2]}`, false, 0.75, 'add type assertion');
      count++;
    }

    // TS7006 Parameter implicitly has an 'any' type
    const paramRegex = /error TS7006: Parameter '([A-Za-z0-9_]+)' implicitly has an 'any' type/g;
    count = 0;
    while ((m = paramRegex.exec(content)) && count < 20) {
      push(`Parameter ${m[1]} implicitly any`, true, 0.85, 'add type assertion');
      count++;
    }

    // TS6133 Unused variable
    const unusedRegex = /error TS6133: '([A-Za-z0-9_]+)' is declared but its value is never read/g;
    count = 0;
    while ((m = unusedRegex.exec(content)) && count < 20) {
      push(`Unused identifier ${m[1]}`, true, 0.88, `remove variable: ${m[1]}`);
      count++;
    }

    // TS2322 Type not assignable
    const assignRegex = /error TS2322: Type '([^']+)' is not assignable to type '([^']+)'/g;
    count = 0;
    while ((m = assignRegex.exec(content)) && count < 10) {
      push(`Type mismatch ${m[1]} → ${m[2]}`, false, 0.7, 'add type assertion');
      count++;
    }
    return recs;
  }

  async applyAutoFixes(recommendations) {
    console.log('🔧 Applying automatic fixes...');

    let fixesApplied = 0;

    for (const recommendation of recommendations) {
      if (recommendation.autoFixable && recommendation.confidence >= CONFIDENCE_THRESHOLD) {
        try {
          await this.applyFix(recommendation);
          fixesApplied++;
          this.autoFixCount++;
          console.log(`✅ Auto-fixed: ${recommendation.issue}`);
        } catch (error) {
          console.error(`❌ Failed to apply fix for ${recommendation.issue}:`, error);
        }
      } else {
        console.log(`⚠️ Skipped fix for ${recommendation.issue} (confidence: ${recommendation.confidence})`);
      }
    }

    if (fixesApplied > 0) {
      console.log(`🎉 Applied ${fixesApplied} automatic fixes`);

      // Run check again to verify fixes
      setTimeout(() => {
        console.log('🔄 Re-running check to verify fixes...');
        this.runErrorCheck();
      }, 5000);
    }
  }

  async applyFix(recommendation) {
    const { file, codeFix, issue } = recommendation;

    if (!file || !codeFix) {
      throw new Error('Invalid recommendation format');
    }

    console.log(`🔧 Applying fix to ${file}: ${issue}`);

    try {
      // Read the current file content
      const filePath = path.resolve(file);
      const content = await fs.readFile(filePath, 'utf8');

      // Apply the fix (this is a simplified implementation)
      const fixedContent = this.applyCodeFix(content, codeFix, issue);

      // Create backup
      const backupPath = `${filePath}.backup.${Date.now()}`;
      await fs.writeFile(backupPath, content);

      // Write the fixed content
      await fs.writeFile(filePath, fixedContent);

      console.log(`✅ Fix applied to ${file}, backup saved as ${path.basename(backupPath)}`);

    } catch (error) {
      throw new Error(`Failed to apply fix: ${error.message}`);
    }
  }

  applyCodeFix(content, codeFix, issue) {
    // This is a simplified fix application
    // In a real implementation, this would use AST manipulation

    if (issue.includes('Type') && issue.includes('not assignable')) {
      // Type error fix
      return this.fixTypeError(content, codeFix);
    } else if (issue.includes('Cannot find name')) {
      // Import missing dependency
      return this.fixMissingImport(content, codeFix);
    } else if (issue.includes('never used')) {
      // Remove unused variable
      return this.fixUnusedVariable(content, codeFix);
    }

    // Default: try to replace based on codeFix
    if (codeFix.includes('replace:')) {
      const [, oldCode, newCode] = codeFix.match(/replace:(.*?)\s*->\s*(.*)/);
      return content.replace(oldCode, newCode);
    }

    return content;
  }

  fixTypeError(content, codeFix) {
    // Simple type error fixes
    if (codeFix.includes('add type assertion')) {
      // Add type assertion as string
      return content.replace(/= (\d+)/, '= $1 as string');
    } else if (codeFix.includes('change to const')) {
      return content.replace(/let\s+(\w+)/, 'const $1');
    }
    return content;
  }

  fixMissingImport(content, codeFix) {
    if (codeFix.includes('import')) {
      const lines = content.split('\n');
      lines.splice(0, 0, codeFix);
      return lines.join('\n');
    }
    return content;
  }

  fixUnusedVariable(content, codeFix) {
    if (codeFix.includes('remove variable')) {
      const variableName = codeFix.match(/variable:\s*(\w+)/)?.[1];
      if (variableName) {
        const regex = new RegExp(`^.*\\b${variableName}\\b.*$`, 'gm');
        return content.replace(regex, '');
      }
    }
    return content;
  }

  async storeErrorPatterns(content, recommendations) {
    try {
      // Extract error patterns for future learning
      const patterns = this.extractErrorPatterns(content);

      for (const pattern of patterns) {
        const key = this.generatePatternKey(pattern);
        if (!this.errorPatterns.has(key)) {
          this.errorPatterns.set(key, {
            pattern,
            count: 0,
            recommendations: [],
            lastSeen: Date.now()
          });
        }

        const stored = this.errorPatterns.get(key);
        stored.count++;
        stored.lastSeen = Date.now();
        stored.recommendations.push(...recommendations);
      }

      // Save to file for persistence
      await this.saveErrorPatterns();

    } catch (error) {
      console.error('❌ Failed to store error patterns:', error);
    }
  }

  extractErrorPatterns(content) {
    const patterns = [];

    // TypeScript error patterns
    const tsErrors = content.match(/error TS\d+: .+/g);
    if (tsErrors) {
      patterns.push(...tsErrors.map(err => ({
        type: 'typescript',
        code: err.match(/TS(\d+)/)?.[1],
        message: err,
        severity: 'error'
      })));
    }

    // ESLint error patterns
    const eslintErrors = content.match(/error\s+.+\s+@typescript-eslint\/.+/g);
    if (eslintErrors) {
      patterns.push(...eslintErrors.map(err => ({
        type: 'eslint',
        rule: err.match(/@typescript-eslint\/(.+)$/)?.[1],
        message: err,
        severity: 'error'
      })));
    }

    return patterns;
  }

  generatePatternKey(pattern) {
    return `${pattern.type}_${pattern.code || pattern.rule}_${pattern.severity}`;
  }

  async loadErrorPatterns() {
    const patternFile = path.join(LOG_DIRECTORY, 'error-patterns.json');

    try {
      const data = await fs.readFile(patternFile, 'utf8');
      const patterns = JSON.parse(data);

      for (const [key, value] of Object.entries(patterns)) {
        this.errorPatterns.set(key, value);
      }

      console.log(`📚 Loaded ${this.errorPatterns.size} error patterns`);

    } catch (error) {
      console.log('📚 No existing error patterns found, starting fresh');
    }
  }

  async saveErrorPatterns() {
    const patternFile = path.join(LOG_DIRECTORY, 'error-patterns.json');

    try {
      const patterns = Object.fromEntries(this.errorPatterns);
      await fs.writeFile(patternFile, JSON.stringify(patterns, null, 2));

    } catch (error) {
      console.error('❌ Failed to save error patterns:', error);
    }
  }

  async executeCommand(command, args) {
    return new Promise((resolve, reject) => {
      const process = spawn(command, args, {
        shell: true,
        cwd: path.resolve('.')
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
        resolve({ code, stdout, stderr });
      });

      process.on('error', (error) => {
        reject(error);
      });
    });
  }

  getStatus() {
    return {
      status: 'running',
      isProcessing: this.isProcessing,
      lastCheckTime: new Date(this.lastCheckTime).toISOString(),
      totalErrorsFound: this.totalErrorsFound,
      autoFixCount: this.autoFixCount,
      errorPatternsLearned: this.errorPatterns.size,
      config: {
        checkInterval: CHECK_INTERVAL,
        autoFixEnabled: AUTO_FIX_ENABLED,
        confidenceThreshold: CONFIDENCE_THRESHOLD
      }
    };
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('🛑 Error Processor Daemon shutting down...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('🛑 Received SIGTERM, shutting down gracefully...');
  process.exit(0);
});

// Start the daemon
const daemon = new ErrorProcessorDaemon();

// Expose status endpoint if running as HTTP service
if (process.env.HTTP_STATUS_PORT) {
  const http = require('http');
  const port = parseInt(process.env.HTTP_STATUS_PORT);

  http.createServer((req, res) => {
    if (req.url === '/status') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(daemon.getStatus()));
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  }).listen(port, () => {
    console.log(`📊 Status endpoint available at http://localhost:${port}/status`);
  });
}