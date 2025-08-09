// @ts-nocheck
// Vite VS Code Error Logger Plugin
// Integrates Vite errors directly into VS Code for enhanced development workflow

import { type Plugin, type ViteDevServer } from 'vite';
import { type ErrorPayload } from 'vite';
import { resolve, dirname, relative } from 'path';
import { writeFileSync, existsSync, mkdirSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';

interface ErrorLogEntry {
  timestamp: string;
  level: 'error' | 'warn' | 'info';
  message: string;
  stack?: string;
  file?: string;
  line?: number;
  column?: number;
  source?: string;
  frame?: string;
  buildTarget?: string;
  buildPhase?: string;
  suggestion?: string;
}

interface VSCodeErrorConfig {
  enabled?: boolean;
  logFile?: string;
  maxEntries?: number;
  includeWarnings?: boolean;
  includeSourceMaps?: boolean;
  autoOpenProblems?: boolean;
  notificationLevel?: 'all' | 'errors-only' | 'none';
  integrateTasks?: boolean;
  generateDiagnostics?: boolean;
}

interface DiagnosticInfo {
  uri: string;
  range: {
    start: { line: number; character: number };
    end: { line: number; character: number };
  };
  severity: number; // 1=Error, 2=Warning, 3=Info, 4=Hint
  message: string;
  source: string;
  code?: string;
}

export function vscodeErrorLogger(options: VSCodeErrorConfig = {}): Plugin {
  const config: Required<VSCodeErrorConfig> = {
    enabled: true,
    logFile: resolve(process.cwd(), '.vscode/vite-errors.json'),
    maxEntries: 1000,
    includeWarnings: true,
    includeSourceMaps: true,
    autoOpenProblems: false,
    notificationLevel: 'errors-only',
    integrateTasks: true,
    generateDiagnostics: true,
    ...options
  };

  let server: ViteDevServer;
  let errorLog: ErrorLogEntry[] = [];
  let diagnostics: DiagnosticInfo[] = [];

  // Load existing error log
  function loadErrorLog() {
    try {
      if (existsSync(config.logFile)) {
        const data = readFileSync(config.logFile, 'utf-8');
        errorLog = JSON.parse(data).slice(-config.maxEntries);
      }
    } catch (error) {
      console.warn('Failed to load existing error log:', error);
      errorLog = [];
    }
  }

  // Save error log to file
  function saveErrorLog() {
    try {
      const logDir = dirname(config.logFile);
      if (!existsSync(logDir)) {
        mkdirSync(logDir, { recursive: true });
      }

      const logData = {
        metadata: {
          lastUpdated: new Date().toISOString(),
          totalEntries: errorLog.length,
          viteVersion: server?.config?.plugins?.find(p: any => p.name === 'vite:build')?.name || 'unknown',
          projectRoot: server?.config?.root || process.cwd()
        },
        errors: errorLog.slice(-config.maxEntries),
        diagnostics: config.generateDiagnostics ? diagnostics : []
      };

      writeFileSync(config.logFile, JSON.stringify(logData, null, 2));

      // Also create VS Code tasks integration
      if (config.integrateTasks) {
        createVSCodeTasks();
      }

      // Generate diagnostics file for VS Code extension integration
      if (config.generateDiagnostics) {
        createDiagnosticsFile();
      }
    } catch (error) {
      console.error('Failed to save error log:', error);
    }
  }

  // Create VS Code tasks for error navigation
  function createVSCodeTasks() {
    const tasksFile = resolve(process.cwd(), '.vscode/tasks.json');
    const tasks = {
      version: "2.0.0",
      tasks: [
        {
          label: "View Vite Errors",
          type: "shell",
          command: "code",
          args: [config.logFile],
          group: "test",
          presentation: {
            reveal: "always",
            panel: "new"
          },
          problemMatcher: []
        },
        {
          label: "Clear Vite Error Log",
          type: "shell",
          command: "echo",
          args: ["'{\"errors\": [], \"diagnostics\": []}'", ">", config.logFile],
          group: "test",
          presentation: {
            reveal: "silent"
          },
          problemMatcher: []
        },
        {
          label: "Restart Vite with Clean Logs",
          type: "shell",
          command: "npm",
          args: ["run", "dev"],
          group: "build",
          dependsOn: "Clear Vite Error Log",
          presentation: {
            reveal: "always",
            panel: "new"
          },
          problemMatcher: [
            {
              owner: "vite",
              fileLocation: ["relative", "${workspaceFolder}"],
              pattern: {
                regexp: "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)$",
                file: 1,
                line: 2,
                column: 3,
                severity: 4,
                message: 5
              }
            }
          ]
        }
      ]
    };

    try {
      const tasksDir = dirname(tasksFile);
      if (!existsSync(tasksDir)) {
        mkdirSync(tasksDir, { recursive: true });
      }
      writeFileSync(tasksFile, JSON.stringify(tasks, null, 2));
    } catch (error) {
      console.warn('Failed to create VS Code tasks:', error);
    }
  }

  // Create diagnostics file for VS Code language server integration
  function createDiagnosticsFile() {
    const diagnosticsFile = resolve(process.cwd(), '.vscode/vite-diagnostics.json');
    
    try {
      const diagnosticsData = {
        version: "1.0.0",
        timestamp: new Date().toISOString(),
        diagnostics: diagnostics,
        summary: {
          totalErrors: diagnostics.filter(d: any => d.severity === 1).length,
          totalWarnings: diagnostics.filter(d: any => d.severity === 2).length,
          totalInfo: diagnostics.filter(d: any => d.severity === 3).length,
          affectedFiles: Array.from(new Set(diagnostics.map(d: any => d.uri))).length
        }
      };

      writeFileSync(diagnosticsFile, JSON.stringify(diagnosticsData, null, 2));
    } catch (error) {
      console.warn('Failed to create diagnostics file:', error);
    }
  }

  // Add error to log
  function addError(entry: ErrorLogEntry) {
    errorLog.push(entry);
    
    // Convert to diagnostic if possible
    if (config.generateDiagnostics && entry.file && entry.line) {
      diagnostics.push({
        uri: `file://${resolve(entry.file)}`,
        range: {
          start: { line: (entry.line || 1) - 1, character: (entry.column || 1) - 1 },
          end: { line: (entry.line || 1) - 1, character: (entry.column || 1) + 10 }
        },
        severity: entry.level === 'error' ? 1 : entry.level === 'warn' ? 2 : 3,
        message: entry.message,
        source: 'vite',
        code: entry.buildPhase || 'build'
      });
    }

    saveErrorLog();

    // Console output with enhanced formatting
    if (config.notificationLevel === 'all' || 
        (config.notificationLevel === 'errors-only' && entry.level === 'error')) {
      
      const timestamp = new Date().toLocaleTimeString();
      const level = entry.level.toUpperCase().padEnd(5);
      const fileInfo = entry.file ? ` (${relative(process.cwd(), entry.file)}:${entry.line})` : '';
      
      console.log(`\n🔧 [${timestamp}] ${level}${fileInfo}`);
      console.log(`   ${entry.message}`);
      
      if (entry.suggestion) {
        console.log(`   💡 Suggestion: ${entry.suggestion}`);
      }
      
      if (entry.frame) {
        console.log(`   📄 Context:\n${entry.frame}`);
      }
    }
  }

  // Parse Vite error payload
  function parseViteError(err: ErrorPayload['err']): ErrorLogEntry {
    const entry: ErrorLogEntry = {
      timestamp: new Date().toISOString(),
      level: 'error',
      message: err.message || 'Unknown error',
      stack: err.stack,
      buildPhase: 'build'
    };

    // Extract file information from stack or error
    if (err.loc) {
      entry.file = err.loc.file;
      entry.line = err.loc.line;
      entry.column = err.loc.column;
    }

    // Extract source code frame
    if (err.frame) {
      entry.frame = err.frame;
    }

    // Add intelligent suggestions based on error patterns
    entry.suggestion = generateSuggestion(err.message, err.stack);

    return entry;
  }

  // Generate intelligent suggestions based on error patterns
  function generateSuggestion(message: string, stack?: string): string {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('cannot resolve module') || lowerMessage.includes('module not found')) {
      return 'Check if the import path is correct and the module is installed. Run `npm install` if needed.';
    }
    
    if (lowerMessage.includes('unexpected token') || lowerMessage.includes('syntax error')) {
      return 'Check for syntax errors, missing brackets, or incorrect TypeScript/JavaScript syntax.';
    }
    
    if (lowerMessage.includes('svelte') && lowerMessage.includes('compilation')) {
      return 'Check Svelte component syntax. Ensure you\'re using Svelte 5 patterns like $props() and $state().';
    }
    
    if (lowerMessage.includes('typescript') || lowerMessage.includes('type')) {
      return 'Check TypeScript types and imports. Run `npm run check` for detailed type checking.';
    }
    
    if (lowerMessage.includes('css') || lowerMessage.includes('style')) {
      return 'Check CSS syntax and UnoCSS class names. Verify Tailwind/UnoCSS configuration.';
    }
    
    if (lowerMessage.includes('network') || lowerMessage.includes('fetch')) {
      return 'Check network connectivity and API endpoints. Verify proxy configuration in vite.config.ts.';
    }
    
    if (lowerMessage.includes('permission') || lowerMessage.includes('access')) {
      return 'Check file permissions and ensure the development server has proper access rights.';
    }
    
    return 'Check the error details and stack trace for more information.';
  }

  // Plugin implementation
  return {
    name: 'vscode-error-logger',
    
    configureServer(devServer) {
      server = devServer;
      loadErrorLog();
      
      // Listen for Vite build errors
      devServer.ws.on('vite:error', (data: ErrorPayload) => {
        addError(parseViteError(data.err));
      });
      
      // Listen for HMR errors
      devServer.ws.on('vite:invalidate', (data) => {
        if (data.message) {
          addError({
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `HMR: ${data.message}`,
            buildPhase: 'hmr'
          });
        }
      });
    },
    
    buildStart() {
      // Clear previous build errors
      errorLog = errorLog.filter(entry: any => entry.buildPhase !== 'build');
      diagnostics = diagnostics.filter(d: any => d.source !== 'vite-build');
      
      addError({
        timestamp: new Date().toISOString(),
        level: 'info',
        message: 'Build started',
        buildPhase: 'build-start'
      });
    },
    
    buildEnd(error) {
      if (error) {
        addError({
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `Build failed: ${error.message}`,
          stack: error.stack,
          buildPhase: 'build-end',
          suggestion: generateSuggestion(error.message, error.stack)
        });
      } else {
        addError({
          timestamp: new Date().toISOString(),
          level: 'info',
          message: 'Build completed successfully',
          buildPhase: 'build-end'
        });
      }
    },
    
    // Handle transform errors
    transform(code, id) {
      try {
        // This hook runs for every file transformation
        // We can catch compilation errors here
        return null;
      } catch (error) {
        addError({
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `Transform error in ${id}: ${(error as Error).message}`,
          file: id,
          stack: (error as Error).stack,
          buildPhase: 'transform',
          suggestion: generateSuggestion((error as Error).message, (error as Error).stack)
        });
        throw error;
      }
    },
    
    // Handle load errors
    load(id) {
      try {
        return null;
      } catch (error) {
        addError({
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `Load error for ${id}: ${(error as Error).message}`,
          file: id,
          stack: (error as Error).stack,
          buildPhase: 'load',
          suggestion: generateSuggestion((error as Error).message, (error as Error).stack)
        });
        throw error;
      }
    }
  };
}

// Export utility functions for manual error logging
export function logCustomError(message: string, file?: string, line?: number, level: 'error' | 'warn' | 'info' = 'error') {
  const entry: ErrorLogEntry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    file,
    line,
    buildPhase: 'custom',
    suggestion: generateSuggestion(message)
  };
  
  // This would need to be integrated with the plugin instance
  console.log(`[Custom ${level.toUpperCase()}] ${message}`);
}

function generateSuggestion(message: string): string {
  const lowerMessage = message.toLowerCase();
  
  if (lowerMessage.includes('cannot resolve module') || lowerMessage.includes('module not found')) {
    return 'Check if the import path is correct and the module is installed. Run `npm install` if needed.';
  }
  
  if (lowerMessage.includes('unexpected token') || lowerMessage.includes('syntax error')) {
    return 'Check for syntax errors, missing brackets, or incorrect TypeScript/JavaScript syntax.';
  }
  
  if (lowerMessage.includes('svelte') && lowerMessage.includes('compilation')) {
    return 'Check Svelte component syntax. Ensure you\'re using Svelte 5 patterns like $props() and $state().';
  }
  
  if (lowerMessage.includes('typescript') || lowerMessage.includes('type')) {
    return 'Check TypeScript types and imports. Run `npm run check` for detailed type checking.';
  }
  
  if (lowerMessage.includes('css') || lowerMessage.includes('style')) {
    return 'Check CSS syntax and UnoCSS class names. Verify Tailwind/UnoCSS configuration.';
  }
  
  return 'Check the error details and stack trace for more information.';
}

// Export default configuration
export const defaultVSCodeErrorConfig: VSCodeErrorConfig = {
  enabled: true,
  logFile: resolve(process.cwd(), '.vscode/vite-errors.json'),
  maxEntries: 1000,
  includeWarnings: true,
  includeSourceMaps: true,
  autoOpenProblems: false,
  notificationLevel: 'errors-only',
  integrateTasks: true,
  generateDiagnostics: true
};