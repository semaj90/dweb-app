<!-- Vite Error Logger Demo Page -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { logCustomError } from '$lib/vite/vscode-error-logger';
  import { vscodeIntegration, errorNavigator } from '$lib/vite/vscode-extension';

  let errorLog: any[] = [];
  let errorStats = { total: 0, errors: 0, warnings: 0, info: 0 };
  let isWatching = false;

  // Demo error generators
  const demoErrors = [
    {
      level: 'error' as const,
      message: 'Cannot find module "@/components/NonExistentComponent"',
      file: 'src/routes/demo/+page.svelte',
      line: 42,
      suggestion: 'Check if the import path is correct and the module exists.'
    },
    {
      level: 'warn' as const,
      message: 'Unused variable "unusedVar" in function scope',
      file: 'src/lib/utils/demo-utils.ts',
      line: 15,
      suggestion: 'Remove unused variables or prefix with underscore if intentionally unused.'
    },
    {
      level: 'error' as const,
      message: 'Type error: Property "nonExistentProp" does not exist on type',
      file: 'src/lib/components/DemoComponent.svelte',
      line: 28,
      suggestion: 'Check TypeScript types and ensure the property exists on the object.'
    },
    {
      level: 'info' as const,
      message: 'HMR: File changed, hot reloading...',
      file: 'src/app.html',
      line: 1,
      suggestion: 'File change detected. No action needed.'
    }
  ];

  function loadErrorLog() {
    const currentErrors = vscodeIntegration.getCurrentErrors();
    errorLog = currentErrors.errors || [];
    updateStats();
  }

  function updateStats() {
    errorStats = {
      total: errorLog.length,
      errors: errorLog.filter(e => e.level === 'error').length,
      warnings: errorLog.filter(e => e.level === 'warn').length,
      info: errorLog.filter(e => e.level === 'info').length
    };
  }

  function generateDemoError() {
    const randomError = demoErrors[Math.floor(Math.random() * demoErrors.length)];
    
    // Simulate error logging (this would normally be done by the Vite plugin)
    const errorEntry = {
      ...randomError,
      timestamp: new Date().toISOString(),
      buildPhase: 'demo',
      id: Math.random().toString(36).substr(2, 9)
    };

    errorLog = [errorEntry, ...errorLog];
    updateStats();
    
    console.log(`🔧 Demo Error Generated: ${randomError.message}`);
  }

  function clearErrors() {
    errorLog = [];
    updateStats();
    console.log('🧹 Error log cleared');
  }

  function startWatching() {
    if (!isWatching) {
      vscodeIntegration.startWatching();
      vscodeIntegration.onErrorUpdate((errors) => {
        errorLog = errors;
        updateStats();
      });
      isWatching = true;
      console.log('👀 Started watching for error log changes');
    }
  }

  function stopWatching() {
    if (isWatching) {
      vscodeIntegration.stopWatching();
      isWatching = false;
      console.log('⏹️ Stopped watching for error log changes');
    }
  }

  function getErrorIcon(level: string) {
    switch (level) {
      case 'error': return '🚨';
      case 'warn': return '⚠️';
      case 'info': return 'ℹ️';
      default: return '📝';
    }
  }

  function getErrorColor(level: string) {
    switch (level) {
      case 'error': return 'text-red-600 bg-red-50 border-red-200';
      case 'warn': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'info': return 'text-blue-600 bg-blue-50 border-blue-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  }

  function formatTimestamp(timestamp: string) {
    return new Date(timestamp).toLocaleTimeString();
  }

  onMount(() => {
    loadErrorLog();
  });
</script>

<svelte:head>
  <title>Vite Error Logger Demo</title>
  <meta name="description" content="Interactive demonstration of the Vite error logging system with VS Code integration" />
</svelte:head>

<main class="min-h-screen bg-gray-50 py-8">
  <div class="container mx-auto px-4 max-w-6xl">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">
        🔧 Vite Error Logger Demo
      </h1>
      <p class="text-gray-600 text-lg">
        Interactive demonstration of the Vite error logging system with real-time VS Code integration
      </p>
    </div>

    <!-- Controls -->
    <div class="bg-white rounded-lg shadow-md p-6 mb-8">
      <h2 class="text-xl font-semibold mb-4">Controls</h2>
      <div class="flex flex-wrap gap-4">
        <button
          class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          onclick={generateDemoError}
        >
          🎲 Generate Demo Error
        </button>
        
        <button
          class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
          onclick={loadErrorLog}
        >
          🔄 Reload Error Log
        </button>
        
        <button
          class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
          onclick={clearErrors}
        >
          🧹 Clear Errors
        </button>
        
        <button
          class="px-4 py-2 {isWatching ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-purple-600 hover:bg-purple-700'} text-white rounded-md transition-colors"
          onclick={isWatching ? stopWatching : startWatching}
        >
          {isWatching ? '⏹️ Stop Watching' : '👀 Start Watching'}
        </button>
      </div>
    </div>

    <!-- Statistics -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <div class="bg-white rounded-lg shadow-md p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">Total Errors</p>
            <p class="text-2xl font-bold text-gray-900">{errorStats.total}</p>
          </div>
          <div class="text-3xl">📊</div>
        </div>
      </div>
      
      <div class="bg-white rounded-lg shadow-md p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-red-600">Errors</p>
            <p class="text-2xl font-bold text-red-700">{errorStats.errors}</p>
          </div>
          <div class="text-3xl">🚨</div>
        </div>
      </div>
      
      <div class="bg-white rounded-lg shadow-md p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-yellow-600">Warnings</p>
            <p class="text-2xl font-bold text-yellow-700">{errorStats.warnings}</p>
          </div>
          <div class="text-3xl">⚠️</div>
        </div>
      </div>
      
      <div class="bg-white rounded-lg shadow-md p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-blue-600">Info</p>
            <p class="text-2xl font-bold text-blue-700">{errorStats.info}</p>
          </div>
          <div class="text-3xl">ℹ️</div>
        </div>
      </div>
    </div>

    <!-- Error Log -->
    <div class="bg-white rounded-lg shadow-md overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200">
        <h2 class="text-xl font-semibold text-gray-900">Error Log</h2>
        <p class="text-sm text-gray-600 mt-1">
          Real-time error tracking with VS Code integration
        </p>
      </div>
      
      <div class="max-h-96 overflow-y-auto">
        {#if errorLog.length === 0}
          <div class="p-8 text-center text-gray-500">
            <div class="text-4xl mb-4">📝</div>
            <p class="text-lg">No errors in log</p>
            <p class="text-sm">Generate some demo errors to see the logging system in action</p>
          </div>
        {:else}
          <div class="divide-y divide-gray-200">
            {#each errorLog as error}
              <div class="p-4 hover:bg-gray-50 transition-colors">
                <div class="flex items-start space-x-3">
                  <div class="text-xl">{getErrorIcon(error.level)}</div>
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center justify-between">
                      <p class="text-sm font-medium text-gray-900 truncate">
                        {error.message}
                      </p>
                      <div class="flex items-center space-x-2">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getErrorColor(error.level)} border">
                          {error.level.toUpperCase()}
                        </span>
                        <span class="text-xs text-gray-500">
                          {formatTimestamp(error.timestamp)}
                        </span>
                      </div>
                    </div>
                    
                    {#if error.file}
                      <p class="text-sm text-gray-600 mt-1">
                        📄 {error.file}{error.line ? `:${error.line}` : ''}{error.column ? `:${error.column}` : ''}
                      </p>
                    {/if}
                    
                    {#if error.suggestion}
                      <div class="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
                        <p class="text-sm text-blue-700">
                          💡 <strong>Suggestion:</strong> {error.suggestion}
                        </p>
                      </div>
                    {/if}
                    
                    {#if error.buildPhase}
                      <p class="text-xs text-gray-500 mt-1">
                        🔧 Build Phase: {error.buildPhase}
                      </p>
                    {/if}
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- VS Code Integration Info -->
    <div class="mt-8 bg-indigo-50 rounded-lg p-6">
      <h3 class="text-lg font-semibold text-indigo-900 mb-4">🔗 VS Code Integration</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 class="font-medium text-indigo-800 mb-2">Available Tasks</h4>
          <ul class="text-sm text-indigo-700 space-y-1">
            <li>• <code>View Vite Errors</code> - Open error log in editor</li>
            <li>• <code>Clear Vite Error Log</code> - Clear all logged errors</li>
            <li>• <code>Restart Vite with Clean Logs</code> - Fresh start</li>
            <li>• <code>Analyze Error Patterns</code> - Generate error statistics</li>
          </ul>
        </div>
        <div>
          <h4 class="font-medium text-indigo-800 mb-2">File Locations</h4>
          <ul class="text-sm text-indigo-700 space-y-1">
            <li>• <code>.vscode/vite-errors.json</code> - Error log file</li>
            <li>• <code>.vscode/vite-diagnostics.json</code> - Diagnostics data</li>
            <li>• <code>.vscode/error-report.json</code> - Generated reports</li>
            <li>• <code>.vscode/tasks.json</code> - VS Code tasks</li>
          </ul>
        </div>
      </div>
      
      <div class="mt-4 p-4 bg-indigo-100 rounded">
        <p class="text-sm text-indigo-800">
          <strong>💡 Tip:</strong> Use <code>Ctrl+Shift+P</code> and search for "Tasks: Run Task" to access Vite error logger commands.
          The error logger automatically integrates with VS Code's Problems panel and provides intelligent error suggestions.
        </p>
      </div>
    </div>

    <!-- Technical Details -->
    <div class="mt-8 bg-gray-100 rounded-lg p-6">
      <h3 class="text-lg font-semibold text-gray-900 mb-4">🛠️ Technical Details</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 class="font-medium text-gray-800 mb-2">Features</h4>
          <ul class="text-sm text-gray-700 space-y-1">
            <li>✅ Real-time error capture during development</li>
            <li>✅ Intelligent error suggestions and fixes</li>
            <li>✅ VS Code Problems panel integration</li>
            <li>✅ File and line navigation</li>
            <li>✅ Error pattern analysis</li>
            <li>✅ Build phase tracking</li>
          </ul>
        </div>
        <div>
          <h4 class="font-medium text-gray-800 mb-2">Configuration</h4>
          <ul class="text-sm text-gray-700 space-y-1">
            <li>• Plugin: <code>vscodeErrorLogger()</code></li>
            <li>• Max entries: 500 errors</li>
            <li>• Includes warnings and source maps</li>
            <li>• Auto-generates VS Code tasks</li>
            <li>• JSON schema validation</li>
            <li>• Problem matcher integration</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</main>

<style>
  :global(body) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  
  code {
    background-color: rgba(0, 0, 0, 0.1);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-size: 0.875em;
  }
</style>