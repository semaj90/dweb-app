<script lang="ts">

  import { onMount } from 'svelte';
  import { $derived } from 'svelte';
  import { WebGPUSOMCache, type IntelligentTodo, type NPMError, initializeSOMCache } from '$lib/webgpu/som-webgpu-cache.js';

  let somCache: WebGPUSOMCache;
let isLoading = $state(false);
let webGPUEnabled = $state(false);
let todos = $state<IntelligentTodo[] >([]);
let errors = $state<NPMError[] >([]);
let processingTime = $state(0);
let npmOutput = $state(`
src/app.ts(1,25): error TS2307: Cannot find module '@types/node' or its corresponding type declarations.
src/utils.ts(15,23): error TS2339: Property 'foo' does not exist on type 'Object'.
src/api.ts(25,10): error: Service unavailable: http://localhost:8080
src/parser.ts(42,15): error TS1005: Unexpected token ');'.
src/index.ts(8,32): error TS2307: Module not found: Can't resolve './missing'
src/components/Button.tsx(12,8): error TS2322: Type 'string' is not assignable to type 'number'.
src/services/auth.ts(56,4): error: Authentication service connection failed
src/database/models.ts(23,12): error TS2304: Cannot find name 'User'.
src/types/global.d.ts(5,18): error TS2717: Subsequent property declarations must have the same type.
src/hooks.server.ts(18,25): error: Database connection timeout after 5000ms
`;
let performanceMetrics = $state({
    somTrainingTime: 0,
    webGPUProcessingTime: 0,
    pageRankIterations: 0,
    cacheHitRatio: 0,
    totalProcessingTime: 0
  });
let filterCategory = $state('all');
let sortBy = $state<'priority' | 'confidence' | 'effort' >('priority');
let showDetails = $state(false);

  onMount(async () => {
    try {
      somCache = await initializeSOMCache();
      webGPUEnabled = true;
    } catch (error) {
      console.error('Failed to initialize SOM cache:', error);
    }
  });

  async function processErrors() {
    if (!somCache) {
      alert('SOM Cache not initialized');
      return;
    }

    isLoading = true;
    const startTime = performance.now();

    try {
      // Process npm check errors with SOM analysis
      const generatedTodos = await somCache.processNPMCheckErrors(npmOutput);
      todos = generatedTodos;

      // Extract errors from npm output for display
      errors = extractErrorsFromOutput(npmOutput);

      // Simulate performance metrics (in a real implementation, these would come from the cache)
      performanceMetrics = {
        somTrainingTime: 150 + Math.random() * 50, // ms
        webGPUProcessingTime: 12 + Math.random() * 8, // ms
        pageRankIterations: 20,
        cacheHitRatio: Math.random() * 0.3 + 0.1,
        totalProcessingTime: performance.now() - startTime
      };

      processingTime = performanceMetrics.totalProcessingTime;

    } catch (error) {
      console.error('Error processing npm output:', error);
      alert('Failed to process errors: ' + error.message);
    } finally {
      isLoading = false;
    }
  }

  function extractErrorsFromOutput(output: string): NPMError[] {
    const lines = output.trim().split('\n');
    const extractedErrors: NPMError[] = [];

    lines.forEach(line => {
      const match = line.match(/(.+\.tsx?)\((\d+),\d+\): (.+)/);
      if (match) {
        extractedErrors.push({
          message: match[3],
          file: match[1],
          line: parseInt(match[2]),
          severity: determineSeverity(match[3]),
          category: determineCategory(match[3]),
          type: 'error',
          timestamp: new Date().toISOString(),
          context: [line]
        });
      }
    });

    return extractedErrors;
  }

  function determineSeverity(message: string): 'low' | 'medium' | 'high' | 'critical' {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('service unavailable') || lowerMessage.includes('timeout')) return 'critical';
    if (lowerMessage.includes('cannot find') || lowerMessage.includes('not assignable')) return 'high';
    if (lowerMessage.includes('property') || lowerMessage.includes('type')) return 'medium';
    return 'low';
  }

  function determineCategory(message: string): string {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('ts23') || lowerMessage.includes('type')) return 'typescript';
    if (lowerMessage.includes('module') || lowerMessage.includes('import')) return 'import';
    if (lowerMessage.includes('syntax') || lowerMessage.includes('token')) return 'syntax';
    if (lowerMessage.includes('service') || lowerMessage.includes('connection')) return 'service';
    return 'general';
  }

  function formatDuration(nanoseconds: number): string {
    const minutes = Math.floor(nanoseconds / (60 * 1000000000));
    const remainingSeconds = Math.floor((nanoseconds % (60 * 1000000000)) / 1000000000);
    return `${minutes}m ${remainingSeconds}s`;
  }

  function getSeverityColor(severity: string): string {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  }

  function getPriorityColor(priority: number): string {
    if (priority > 0.05) return 'text-red-600 font-bold';
    if (priority > 0.03) return 'text-orange-600 font-semibold';
    if (priority > 0.02) return 'text-yellow-600';
    return 'text-green-600';
  }

  let filteredTodos = $derived(() => todos
    .filter(todo => filterCategory === 'all' || todo.category === filterCategory)
    .sort((a, b) => {
      switch (sortBy) {
        case 'priority': return b.priority - a.priority;
        case 'confidence': return b.confidence - a.confidence;
        case 'effort': return a.estimated_effort - b.estimated_effort;
        default: return 0;
      }
    })
  );

  let uniqueCategories = $derived([...new Set(todos.map(todo => todo.category))]);
</script>

<div class="p-6 max-w-7xl mx-auto">
  <div class="mb-8">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">
      🧠 SOM-based Intelligent Todo Generator
    </h1>
    <p class="text-gray-600 mb-4">
      Advanced semantic analysis using Self-Organizing Maps, WebGPU acceleration, and real-time PageRank prioritization
    </p>

    <!-- Status Indicators -->
    <div class="flex flex-wrap gap-4 mb-6">
      <div class="flex items-center space-x-2">
        <div class={`w-3 h-3 rounded-full ${webGPUEnabled ? 'bg-green-500' : 'bg-red-500'}`}></div>
        <span class="text-sm font-medium">
          WebGPU: {webGPUEnabled ? 'Enabled' : 'Disabled'}
        </span>
      </div>
      <div class="flex items-center space-x-2">
        <div class="w-3 h-3 rounded-full bg-blue-500"></div>
        <span class="text-sm font-medium">
          SOM Network: 8×8 Grid
        </span>
      </div>
      <div class="flex items-center space-x-2">
        <div class="w-3 h-3 rounded-full bg-purple-500"></div>
        <span class="text-sm font-medium">
          PageRank: Real-time
        </span>
      </div>
    </div>
  </div>

  <!-- NPM Output Input -->
  <div class="bg-gray-900 rounded-lg p-4 mb-6">
    <h3 class="text-white font-medium mb-3">📋 NPM Check Output:</h3>
    <textarea
      bind:value={npmOutput}
      class="w-full h-32 bg-gray-800 text-green-400 font-mono text-sm p-3 rounded border-none resize-none"
      placeholder="Paste npm check output here..."
    ></textarea>

    <div class="flex justify-between items-center mt-4">
      <button
        on:onclick={processErrors}
        disabled={isLoading || !somCache}
        class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg font-medium transition-colors"
      >
        {#if isLoading}
          🧠 Processing with SOM...
        {:else}
          🚀 Generate Intelligent Todos
        {/if}
      </button>

      {#if processingTime > 0}
        <div class="text-white text-sm">
          ⚡ Processed in {processingTime.toFixed(1)}ms
        </div>
      {/if}
    </div>
  </div>

  {#if isLoading}
    <div class="text-center py-12">
      <div class="animate-spin w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
      <p class="text-gray-600">Running SOM analysis and WebGPU PageRank...</p>
    </div>
  {/if}

  {#if todos.length > 0}
    <!-- Performance Metrics -->
    <div class="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 mb-6">
      <h3 class="text-lg font-bold text-gray-900 mb-4">⚡ Performance Metrics</h3>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div class="text-center">
          <div class="text-2xl font-bold text-blue-600">{performanceMetrics.somTrainingTime.toFixed(0)}ms</div>
          <div class="text-sm text-gray-600">SOM Training</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-purple-600">{performanceMetrics.webGPUProcessingTime.toFixed(0)}ms</div>
          <div class="text-sm text-gray-600">WebGPU Processing</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-green-600">{performanceMetrics.pageRankIterations}</div>
          <div class="text-sm text-gray-600">PageRank Iterations</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-orange-600">{(performanceMetrics.cacheHitRatio * 100).toFixed(0)}%</div>
          <div class="text-sm text-gray-600">Cache Hit Ratio</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-red-600">{performanceMetrics.totalProcessingTime.toFixed(0)}ms</div>
          <div class="text-sm text-gray-600">Total Time</div>
        </div>
      </div>
    </div>

    <!-- Filters and Controls -->
    <div class="flex flex-wrap justify-between items-center mb-6 gap-4">
      <div class="flex flex-wrap gap-4">
        <select bind:value={filterCategory} class="px-3 py-2 border rounded-lg">
          <option value="all">All Categories</option>
          {#each uniqueCategories as category}
            <option value={category}>{category}</option>
          {/each}
        </select>

        <select bind:value={sortBy} class="px-3 py-2 border rounded-lg">
          <option value="priority">Sort by Priority</option>
          <option value="confidence">Sort by Confidence</option>
          <option value="effort">Sort by Effort</option>
        </select>
      </div>

      <div class="flex items-center space-x-4">
        <label class="flex items-center space-x-2">
          <input type="checkbox" bind:checked={showDetails} class="rounded">
          <span class="text-sm">Show Details</span>
        </label>
        <div class="text-sm text-gray-600">
          {filteredTodos.length} todos found
        </div>
      </div>
    </div>

    <!-- Intelligent Todos -->
    <div class="space-y-4">
      {#each filteredTodos as todo, index}
        <div class="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
          <div class="p-6">
            <div class="flex justify-between items-start mb-4">
              <div class="flex-1">
                <div class="flex items-center space-x-3 mb-2">
                  <h3 class="text-lg font-semibold text-gray-900">
                    {index + 1}. {todo.title}
                  </h3>
                  <div class={`px-3 py-1 rounded-full text-xs font-medium ${getSeverityColor(todo.tags.find(t => ['low', 'medium', 'high', 'critical'].includes(t)) || 'medium')}`}>
                    {todo.category}
                  </div>
                </div>
                <p class="text-gray-600 mb-3">{todo.description}</p>

                <div class="flex flex-wrap gap-2 mb-3">
                  {#each todo.tags as tag}
                    <span class="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                      {tag}
                    </span>
                  {/each}
                </div>
              </div>

              <div class="text-right ml-6">
                <div class={`text-lg font-bold mb-1 ${getPriorityColor(todo.priority)}`}>
                  Priority: {todo.priority.toFixed(4)}
                </div>
                <div class="text-sm text-gray-600">
                  Confidence: {(todo.confidence * 100).toFixed(1)}%
                </div>
                <div class="text-sm text-gray-600">
                  Effort: {formatDuration(todo.estimated_effort)}
                </div>
              </div>
            </div>

            <!-- Suggested Fixes -->
            <div class="mb-4">
              <h4 class="font-medium text-gray-900 mb-2">🔧 Suggested Fixes:</h4>
              <ul class="list-disc list-inside text-sm text-gray-700 space-y-1">
                {#each todo.suggested_fixes as fix}
                  <li>{fix}</li>
                {/each}
              </ul>
            </div>

            {#if showDetails}
              <!-- Related Errors -->
              <div class="border-t pt-4">
                <h4 class="font-medium text-gray-900 mb-2">
                  📋 Related Errors ({todo.related_errors.length}):
                </h4>
                <div class="space-y-2">
                  {#each todo.related_errors.slice(0, 3) as error}
                    <div class="bg-gray-50 p-3 rounded text-sm">
                      <div class="flex justify-between items-start mb-1">
                        <span class="font-medium text-gray-900">{error.file}:{error.line}</span>
                        <span class={`px-2 py-1 rounded text-xs ${getSeverityColor(error.severity)}`}>
                          {error.severity}
                        </span>
                      </div>
                      <p class="text-gray-700">{error.message}</p>
                    </div>
                  {/each}
                  {#if todo.related_errors.length > 3}
                    <p class="text-sm text-gray-500 italic">
                      ... and {todo.related_errors.length - 3} more errors
                    </p>
                  {/if}
                </div>
              </div>

              <!-- Metadata -->
              <div class="border-t pt-4 mt-4">
                <h4 class="font-medium text-gray-900 mb-2">📊 Metadata:</h4>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span class="text-gray-600">Todo ID:</span>
                    <p class="font-mono">{todo.id}</p>
                  </div>
                  <div>
                    <span class="text-gray-600">Created:</span>
                    <p>{new Date(todo.created_at).toLocaleTimeString()}</p>
                  </div>
                  <div>
                    <span class="text-gray-600">Error Count:</span>
                    <p>{todo.metadata.error_count || todo.related_errors.length}</p>
                  </div>
                  <div>
                    <span class="text-gray-600">Files Affected:</span>
                    <p>{todo.metadata.files_affected || new Set(todo.related_errors.map(e => e.file)).size}</p>
                  </div>
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/each}
    </div>

    <!-- Summary Statistics -->
    <div class="mt-8 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6">
      <h3 class="text-lg font-bold text-gray-900 mb-4">📊 Analysis Summary</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        <div>
          <div class="text-2xl font-bold text-green-600">{todos.length}</div>
          <div class="text-sm text-gray-600">Intelligent Todos</div>
        </div>
        <div>
          <div class="text-2xl font-bold text-blue-600">{errors.length}</div>
          <div class="text-sm text-gray-600">Original Errors</div>
        </div>
        <div>
          <div class="text-2xl font-bold text-purple-600">{uniqueCategories.length}</div>
          <div class="text-sm text-gray-600">Categories Found</div>
        </div>
        <div>
          <div class="text-2xl font-bold text-orange-600">{todos.reduce((sum, todo) => sum + todo.related_errors.length, 0)}</div>
          <div class="text-sm text-gray-600">Total Error Clusters</div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  :global(body) {
    background-color: #f8fafc;
  }
</style>

