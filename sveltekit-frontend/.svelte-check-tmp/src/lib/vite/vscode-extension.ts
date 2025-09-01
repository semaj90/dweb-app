
// VS Code Extension Integration for Vite Error Logger
// Provides real-time error notifications and quick navigation

import { existsSync, readFileSync, watchFile } from "fs";
import { resolve } from "path";

export interface VSCodeCommand {
  command: string;
  args?: unknown[];
}

export interface VSCodeNotification {
  message: string;
  type: 'error' | 'warning' | 'info';
  actions?: Array<{
    title: string;
    command: VSCodeCommand;
  }>;
}

export class VSCodeIntegration {
  private logFile: string;
  private isWatching = false;
  private callbacks: Array<(errors: any[]) => void> = [];

  constructor(logFile?: string) {
    this.logFile = logFile || resolve(process.cwd(), '.vscode/vite-errors.json');
  }

  // Start watching for error log changes
  startWatching() {
    if (this.isWatching || !existsSync(this.logFile)) {
      return;
    }

    this.isWatching = true;
    
    watchFile(this.logFile, { interval: 1000 }, () => {
      this.handleLogUpdate();
    });

    console.log(`📟 VS Code integration started - watching ${this.logFile}`);
  }

  // Stop watching
  stopWatching() {
    this.isWatching = false;
    // Note: fs.watchFile doesn't return a watcher to close in Node.js
    console.log('📟 VS Code integration stopped');
  }

  // Handle log file updates
  private handleLogUpdate() {
    try {
      const data = readFileSync(this.logFile, 'utf-8');
      const logData = JSON.parse(data);
      
      // Get recent errors (last 5 minutes)
      const recentErrors = logData.errors.filter((error: any) => {
        const errorTime = new Date(error.timestamp);
        const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
        return errorTime > fiveMinutesAgo;
      });

      if (recentErrors.length > 0) {
        this.notifyErrors(recentErrors);
      }

      // Notify callbacks
      this.callbacks.forEach((callback: any) => callback(logData.errors));
      
    } catch (error: any) {
      console.warn('Failed to parse error log:', error);
    }
  }

  // Send error notifications
  private notifyErrors(errors: any[]) {
    const errorCount = errors.filter((e: any) => e.level === 'error').length;
    const warningCount = errors.filter((e: any) => e.level === 'warn').length;

    if (errorCount > 0) {
      this.sendNotification({
        message: `Vite: ${errorCount} error(s) detected`,
        type: 'error',
        actions: [
          {
            title: 'View Errors',
            command: { command: 'workbench.action.tasks.runTask', args: ['View Vite Errors'] }
          },
          {
            title: 'Clear Log',
            command: { command: 'workbench.action.tasks.runTask', args: ['Clear Vite Error Log'] }
          }
        ]
      });
    } else if (warningCount > 0) {
      this.sendNotification({
        message: `Vite: ${warningCount} warning(s) detected`,
        type: 'warning',
        actions: [
          {
            title: 'View Warnings',
            command: { command: 'workbench.action.tasks.runTask', args: ['View Vite Errors'] }
          }
        ]
      });
    }
  }

  // Send notification to VS Code
  private sendNotification(notification: VSCodeNotification) {
    // In a real VS Code extension, this would use the VS Code API
    // For now, we'll use console output with special formatting
    const timestamp = new Date().toLocaleTimeString();
    const icon = notification.type === 'error' ? '🚨' : notification.type === 'warning' ? '⚠️' : 'ℹ️';
    
    console.log(`\n${icon} [${timestamp}] VS Code Notification: ${notification.message}`);
    
    if (notification.actions) {
      notification.actions.forEach((action, index) => {
        console.log(`   ${index + 1}. ${action.title} (${action.command.command})`);
      });
    }
  }

  // Register callback for error updates
  onErrorUpdate(callback: (errors: any[]) => void) {
    this.callbacks.push(callback);
  }

  // Get current errors
  getCurrentErrors() {
    try {
      if (existsSync(this.logFile)) {
        const data = readFileSync(this.logFile, 'utf-8');
        return JSON.parse(data);
      }
    } catch (error: any) {
      console.warn('Failed to read current errors:', error);
    }
    return { errors: [], diagnostics: [] };
  }

  // Generate problem matcher for VS Code tasks
  static generateProblemMatcher() {
    return {
      owner: 'vite-error-logger',
      fileLocation: ['relative', '${workspaceFolder}'],
      pattern: [
        {
          regexp: '^ERROR\\s+(.+):(\\d+):(\\d+)\\s+(.+)$',
          file: 1,
          line: 2,
          column: 3,
          message: 4,
          severity: 'error'
        },
        {
          regexp: '^WARN\\s+(.+):(\\d+):(\\d+)\\s+(.+)$',
          file: 1,
          line: 2,
          column: 3,
          message: 4,
          severity: 'warning'
        }
      ]
    };
  }

  // Generate VS Code settings for the integration
  static generateVSCodeSettings() {
    return {
      'files.associations': {
        'vite-errors.json': 'json',
        'vite-diagnostics.json': 'json'
      },
      'json.schemas': [
        {
          fileMatch: ['vite-errors.json'],
          schema: {
            type: 'object',
            properties: {
              metadata: {
                type: 'object',
                properties: {
                  lastUpdated: { type: 'string' },
                  totalEntries: { type: 'number' },
                  viteVersion: { type: 'string' },
                  projectRoot: { type: 'string' }
                }
              },
              errors: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    timestamp: { type: 'string' },
                    level: { type: 'string', enum: ['error', 'warn', 'info'] },
                    message: { type: 'string' },
                    file: { type: 'string' },
                    line: { type: 'number' },
                    column: { type: 'number' },
                    stack: { type: 'string' },
                    suggestion: { type: 'string' }
                  }
                }
              }
            }
          }
        }
      ],
      'problems.decorations.enabled': true,
      'problems.sortOrder': 'severity'
    };
  }
}

// Error navigation utilities
export class ErrorNavigator {
  private errors: any[] = [];

  constructor(private integration: VSCodeIntegration) {
    integration.onErrorUpdate((errors) => {
      this.errors = errors;
    });
  }

  // Navigate to next error
  nextError() {
    const errorWithFile = this.errors.find((e: any) => e.level === 'error' && e.file);
    if (errorWithFile) {
      this.openFile(errorWithFile.file, errorWithFile.line, errorWithFile.column);
    }
  }

  // Navigate to previous error
  previousError() {
    const errors = this.errors.filter((e: any) => e.level === 'error' && e.file).reverse();
    const errorWithFile = errors[0];
    if (errorWithFile) {
      this.openFile(errorWithFile.file, errorWithFile.line, errorWithFile.column);
    }
  }

  // Open file at specific location
  private openFile(file: string, line?: number, column?: number) {
    const location = line ? `:${line}${column ? `:${column}` : ''}` : '';
    console.log(`📂 Opening file: ${file}${location}`);
    
    // In a real VS Code extension, this would use:
    // vscode.window.showTextDocument(vscode.Uri.file(file), {
    //   selection: new vscode.Range(line - 1, column - 1, line - 1, column - 1)
    // });
  }

  // Get error summary
  getErrorSummary() {
    const summary = {
      total: this.errors.length,
      errors: this.errors.filter((e: any) => e.level === 'error').length,
      warnings: this.errors.filter((e: any) => e.level === 'warn').length,
      info: this.errors.filter((e: any) => e.level === 'info').length,
      files: Array.from(new Set(this.errors.filter((e: any) => e.file).map((e: any) => e.file))).length,
      recent: this.errors.filter((e: any) => {
        const errorTime = new Date(e.timestamp);
        const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
        return errorTime > oneHourAgo;
      }).length
    };

    return summary;
  }
}

// Auto-fix suggestions
export class AutoFixSuggestions {
  static getSuggestions(error: any): Array<{ title: string; command: string; args?: unknown[] }> {
    const suggestions = [];
    const message = error.message.toLowerCase();

    if (message.includes('module not found') || message.includes('cannot resolve')) {
      suggestions.push({
        title: 'Install missing dependencies',
        command: 'npm install',
        args: []
      });
      
      suggestions.push({
        title: 'Check import paths',
        command: 'editor.action.quickFix',
        args: []
      });
    }

    if (message.includes('typescript') || message.includes('type')) {
      suggestions.push({
        title: 'Run TypeScript check',
        command: 'workbench.action.tasks.runTask',
        args: ['npm: check']
      });
      
      suggestions.push({
        title: 'Generate missing types',
        command: 'typescript.generateGettersAndSetters',
        args: []
      });
    }

    if (message.includes('svelte')) {
      suggestions.push({
        title: 'Check Svelte syntax',
        command: 'svelte.restartLanguageServer',
        args: []
      });
      
      suggestions.push({
        title: 'Update to Svelte 5 patterns',
        command: 'editor.action.codeAction',
        args: [{ kind: 'refactor.rewrite' }]
      });
    }

    if (message.includes('css') || message.includes('style')) {
      suggestions.push({
        title: 'Check UnoCSS configuration',
        command: 'editor.action.formatDocument',
        args: []
      });
    }

    return suggestions;
  }
}

// Export default integration instance
export const vscodeIntegration = new VSCodeIntegration();
export const errorNavigator = new ErrorNavigator(vscodeIntegration);