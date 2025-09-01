
import { resolve, dirname } from 'path';
import { writeFileSync, existsSync, mkdirSync, readFileSync } from 'fs';

/**
 * Simple VS Code Error Logger plugin for Vite
 * - Writes a JSON log to .vscode/vite-errors.json
 * - Listens for devServer ws 'vite:error' and 'vite:warning'
 * - Records buildStart/buildEnd events
 */
export function vscodeErrorLogger(options: any = {}) {
  const config = {
    enabled: true,
    logFile: resolve(process.cwd(), '.vscode/vite-errors.json'),
    maxEntries: 500,
    includeWarnings: true,
    includeSourceMaps: true,
    ...options
  };

  let server: any = undefined;
  let errorLog: any = { metadata: { lastUpdated: new Date().toISOString(), version: 1 }, errors: [] };

  function loadLog() {
    try {
      if (existsSync(config.logFile)) {
        const raw = readFileSync(config.logFile, 'utf8');
        const parsed = JSON.parse(raw);
        // Normalize shape so later code can safely read metadata and errors
        errorLog = {
          metadata: {
            lastUpdated: parsed?.metadata?.lastUpdated || new Date().toISOString(),
            version: parsed?.metadata?.version || 1
          },
          errors: Array.isArray(parsed?.errors) ? parsed.errors : []
        };
      }
    } catch (e: any) {
    // ignore
    }
  }

  function saveLog() {
    try {
      const dir = dirname(config.logFile);
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      // Ensure metadata and errors exist before mutating
      errorLog.metadata = errorLog.metadata || { version: 1 };
      errorLog.metadata.lastUpdated = new Date().toISOString();
      errorLog.errors = Array.isArray(errorLog.errors) ? errorLog.errors : [];
      writeFileSync(config.logFile, JSON.stringify(errorLog, null, 2));
    } catch (e: any) {
    // ignore
    }
  }

  function pushEntry(entry: any) {
    if (!config.enabled) return;
    errorLog.errors = errorLog.errors || [];
    errorLog.errors.unshift(entry);
    if (errorLog.errors.length > config.maxEntries) errorLog.errors.length = config.maxEntries;
    saveLog();
  }

  function normalizeViteError(err: any) {
    const entry: any = {
      timestamp: new Date().toISOString(),
      level: 'error',
      message: err?.message || err?.text || String(err),
      stack: err?.stack || err?.stackStr,
      file: err?.id || err?.loc?.file || undefined,
      line: err?.loc?.line || err?.loc?.lineNumber || undefined,
      column: err?.loc?.column || undefined,
      frame: err?.frame || undefined,
      plugin: err?.plugin || undefined,
      buildPhase: 'vite'
    };
    return entry;
  }

  return {
    name: 'vscode-error-logger',
    configureServer(srv: any) {
      server = srv;
      loadLog();

      try {
        server.ws.on('vite:error', (payload: any) => {
          const e = normalizeViteError(payload.err || payload);
          pushEntry(e);
        });

        if (config.includeWarnings) {
          server.ws.on('vite:warning', (payload: any) => {
            const e = normalizeViteError(payload.warn || payload);
            e.level = 'warn';
            pushEntry(e);
          });
        }
      } catch (e: any) {
        // ignore websocket attach errors
      }
    },

    buildStart() {
      pushEntry({ timestamp: new Date().toISOString(), level: 'info', message: 'Build started', buildPhase: 'build' });
    },

    buildEnd(error: any) {
      if (error) {
        pushEntry({ timestamp: new Date().toISOString(), level: 'error', message: error.message || String(error), stack: error.stack, buildPhase: 'build' });
      } else {
        pushEntry({ timestamp: new Date().toISOString(), level: 'info', message: 'Build completed', buildPhase: 'build' });
      }
    }
  };
}

export const defaultVSCodeErrorConfig = {
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