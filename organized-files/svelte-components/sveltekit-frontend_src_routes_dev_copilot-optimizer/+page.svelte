<!--
  Copilot Index Optimizer - Development Interface
  Real-time optimization testing and monitoring for GitHub Copilot context enhancement
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { Dialog } from 'bits-ui';

  // Component state
  let optimizationStatus = $state('idle');
  let optimizationResults = $state(null);
  let performanceMetrics = $state(null);
  let copilotContent = $state('');
  let searchQuery = $state('');
  let searchResults = $state([]);
  let selectedTab = $state('optimization');
  let errorMessage = $state('');
  let isLoading = $state(false);

  // Configuration state
  let optimizationConfig = $state({
    enableContext7Boost: true,
    enableSemanticClustering: true,
    enablePatternRecognition: true,
    enablePerformanceOptimization: true,
    minRelevanceThreshold: 0.7,
    compressionRatio: 0.8,
  });

  // Real-time metrics
  let realTimeMetrics = $state({
    totalOptimizations: 0,
    avgOptimizationTime: 0,
    cacheHitRate: 0,
    lastUpdated: null,
  });

  // Load initial data
  onMount(async () => {
    await loadCopilotContent();
    await loadSystemStatus();

    // Start real-time metrics polling
    const metricsInterval = setInterval(loadMetrics, 5000);

    return () => {
      clearInterval(metricsInterval);
    };
  });

  /**
   * Load copilot.md content
   */
  async function loadCopilotContent() {
    try {
      isLoading = true;
      const response = await fetch('/api/copilot/optimize?action=load_copilot');

      if (!response.ok) {
        throw new Error(`Failed to load: ${response.status}`);
      }

      const data = await response.json();
      copilotContent = data.content;

    } catch (error) {
      errorMessage = `Failed to load copilot content: ${error.message}`;
    } finally {
      isLoading = false;
    }
  }

  /**
   * Optimize the copilot index
   */
  async function optimizeIndex() {
    try {
      isLoading = true;
      optimizationStatus = 'optimizing';
      errorMessage = '';

      const response = await fetch('/api/copilot/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'optimize_index',
          content: copilotContent,
          options: optimizationConfig,
        }),
      });

      if (!response.ok) {
        throw new Error(`Optimization failed: ${response.status}`);
      }

      const data = await response.json();
      optimizationResults = data;
      optimizationStatus = 'completed';

      // Update metrics
      await loadMetrics();

    } catch (error) {
      optimizationStatus = 'error';
      errorMessage = error.message;
    } finally {
      isLoading = false;
    }
  }

  /**
   * Perform semantic search
   */
  async function performSearch() {
    if (!searchQuery.trim()) return;

    try {
      isLoading = true;

      const response = await fetch('/api/copilot/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'semantic_search',
          content: searchQuery,
          options: {
            limit: 10,
            includePatterns: true,
            boostContext7: true,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
      }

      const data = await response.json();
      searchResults = data.results;

    } catch (error) {
      errorMessage = `Search failed: ${error.message}`;
    } finally {
      isLoading = false;
    }
  }

  /**
   * Load system status and metrics
   */
  async function loadSystemStatus() {
    try {
      const response = await fetch('/api/copilot/optimize?action=status');
      const data = await response.json();

      realTimeMetrics.lastUpdated = new Date().toLocaleTimeString();

    } catch (error) {
      console.error('Failed to load status:', error);
    }
  }

  /**
   * Load performance metrics
   */
  async function loadMetrics() {
    try {
      const response = await fetch('/api/copilot/optimize?action=metrics');
      const data = await response.json();

      performanceMetrics = data;
      realTimeMetrics = {
        totalOptimizations: data.optimizer.totalOptimizations || 0,
        avgOptimizationTime: data.optimizer.avgOptimizationTime || 0,
        cacheHitRate: data.cache.hitRate || 0,
        lastUpdated: new Date().toLocaleTimeString(),
      };

    } catch (error) {
      console.error('Failed to load metrics:', error);
    }
  }

  /**
   * Generate test suggestions
   */
  async function generateSuggestions() {
    const testCode = `// Test Svelte 5 component
<script lang="ts">
  
  let count = $state(0);

  // Need suggestions here
</script>`.trim();

    try {
      isLoading = true;

      const response = await fetch('/api/copilot/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'generate_suggestions',
          content: {
            currentCode: testCode,
            cursor: { line: 5, character: 25 },
            language: 'svelte',
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`Suggestion generation failed: ${response.status}`);
      }

      const data = await response.json();
      searchResults = data.suggestions.map((suggestion, index) => ({
        id: `suggestion_${index}`,
        document: {
          title: `Suggestion: ${suggestion.category}`,
          content: suggestion.text,
        },
        score: suggestion.confidence,
        explanation: `${suggestion.category} suggestion (Priority: ${suggestion.priority})`,
        context7Pattern: suggestion.context7Pattern,
      }));

    } catch (error) {
      errorMessage = `Suggestion generation failed: ${error.message}`;
    } finally {
      isLoading = false;
    }
  }

  /**
   * Export optimization results
   */
  function exportResults() {
    if (!optimizationResults) return;

    const dataStr = JSON.stringify(optimizationResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `copilot-optimization-${Date.now()}.json`;
    link.click();
  }

  /**
   * Reset optimization
   */
  function resetOptimization() {
    optimizationResults = null;
    optimizationStatus = 'idle';
    searchResults = [];
    errorMessage = '';
  }

  /**
   * Format file size
   */
  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  /**
   * Format time duration
   */
  function formatTime(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(1)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  }
</script>

<svelte:head>
  <title>Copilot Index Optimizer - Dev Tools</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
  <!-- Header -->
  <header class="border-b border-purple-500/20 bg-black/20 backdrop-blur-md">
    <div class="max-w-7xl mx-auto px-4 py-4">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-purple-100">Copilot Index Optimizer</h1>
          <p class="text-purple-300 text-sm">SIMD JSON Processing + Vector Embeddings</p>
        </div>

        <!-- Real-time metrics -->
        <div class="flex gap-4 text-sm">
          <div class="text-center">
            <div class="text-purple-100 font-semibold">{realTimeMetrics.totalOptimizations}</div>
            <div class="text-purple-400">Optimizations</div>
          </div>
          <div class="text-center">
            <div class="text-purple-100 font-semibold">
              {formatTime(realTimeMetrics.avgOptimizationTime)}
            </div>
            <div class="text-purple-400">Avg Time</div>
          </div>
          <div class="text-center">
            <div class="text-purple-100 font-semibold">
              {(realTimeMetrics.cacheHitRate * 100).toFixed(1)}%
            </div>
            <div class="text-purple-400">Cache Hit</div>
          </div>
          {#if realTimeMetrics.lastUpdated}
            <div class="text-center">
              <div class="text-purple-100 font-semibold">{realTimeMetrics.lastUpdated}</div>
              <div class="text-purple-400">Last Update</div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </header>

  <!-- Main content -->
  <main class="max-w-7xl mx-auto px-4 py-6">
    <!-- Error display -->
    {#if errorMessage}
      <div class="mb-6 p-4 bg-red-900/30 border border-red-500/50 rounded-lg text-red-100">
        <strong>Error:</strong> {errorMessage}
        <button
          onclick={() => errorMessage = ''}
          class="ml-2 text-red-300 hover:text-red-100"
        >
          ✕
        </button>
      </div>
    {/if}

    <!-- Tab navigation -->
    <div class="mb-6">
      <nav class="flex space-x-1 bg-black/20 p-1 rounded-lg">
        {#each [
          { id: 'optimization', label: 'Index Optimization' },
          { id: 'search', label: 'Semantic Search' },
          { id: 'suggestions', label: 'Code Suggestions' },
          { id: 'metrics', label: 'Performance Metrics' },
          { id: 'config', label: 'Configuration' }
        ] as tab}
          <button
            onclick={() => selectedTab = tab.id}
            class="px-4 py-2 rounded-md text-sm font-medium transition-all
              {selectedTab === tab.id
                ? 'bg-purple-600 text-white'
                : 'text-purple-300 hover:text-white hover:bg-purple-700/30'
              }"
          >
            {tab.label}
          </button>
        {/each}
      </nav>
    </div>

    <!-- Tab content -->
    {#if selectedTab === 'optimization'}
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Copilot content editor -->
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-xl font-semibold text-purple-100">Copilot Content</h2>
            <button
              onclick={loadCopilotContent}
              disabled={isLoading}
              class="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50
                     text-white rounded-lg transition-colors"
            >
              {isLoading ? 'Loading...' : 'Reload'}
            </button>
          </div>

          <textarea
            bind:value={copilotContent}
            placeholder="Paste your copilot.md content here..."
            class="w-full h-96 p-4 bg-black/30 border border-purple-500/30 rounded-lg
                   text-purple-100 placeholder:text-purple-400 resize-none"
          ></textarea>

          <div class="flex gap-2">
            <button
              onclick={optimizeIndex}
              disabled={!copilotContent || isLoading}
              class="flex-1 px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600
                     hover:from-purple-700 hover:to-blue-700 disabled:opacity-50
                     text-white rounded-lg transition-all font-medium"
            >
              {optimizationStatus === 'optimizing' ? 'Optimizing...' : 'Optimize Index'}
            </button>

            {#if optimizationResults}
              <button
                onclick={exportResults}
                class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
              >
                Export
              </button>

              <button
                onclick={resetOptimization}
                class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
              >
                Reset
              </button>
            {/if}
          </div>
        </div>

        <!-- Optimization results -->
        <div class="space-y-4">
          <h2 class="text-xl font-semibold text-purple-100">Optimization Results</h2>

          {#if optimizationStatus === 'idle'}
            <div class="p-8 text-center text-purple-400 bg-black/20 rounded-lg border border-purple-500/20">
              <div class="text-4xl mb-2">⚡</div>
              <p>Ready to optimize your copilot index</p>
              <p class="text-sm mt-1">Load content and click "Optimize Index" to begin</p>
            </div>

          {:else if optimizationStatus === 'optimizing'}
            <div class="p-8 text-center text-purple-300 bg-black/20 rounded-lg border border-purple-500/20">
              <div class="animate-spin text-4xl mb-2">⚙️</div>
              <p>Processing with SIMD JSON parser...</p>
              <p class="text-sm mt-1">Generating vector embeddings and semantic clusters</p>
            </div>

          {:else if optimizationStatus === 'completed' && optimizationResults}
            <div class="space-y-4">
              <!-- Summary stats -->
              <div class="grid grid-cols-2 gap-4">
                <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
                  <div class="text-2xl font-bold text-purple-100">
                    {optimizationResults.summary.totalEntries}
                  </div>
                  <div class="text-purple-400 text-sm">Index Entries</div>
                </div>

                <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
                  <div class="text-2xl font-bold text-purple-100">
                    {optimizationResults.summary.indexSize}
                  </div>
                  <div class="text-purple-400 text-sm">Total Size</div>
                </div>

                <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
                  <div class="text-2xl font-bold text-purple-100">
                    {optimizationResults.summary.semanticClusters}
                  </div>
                  <div class="text-purple-400 text-sm">SOM Clusters</div>
                </div>

                <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
                  <div class="text-2xl font-bold text-purple-100">
                    {formatTime(optimizationResults.summary.optimizationTime)}
                  </div>
                  <div class="text-purple-400 text-sm">Process Time</div>
                </div>
              </div>

              <!-- Detailed results -->
              <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
                <h3 class="text-lg font-semibold text-purple-100 mb-3">Optimization Details</h3>
                <div class="space-y-2 text-sm">
                  <div class="flex justify-between">
                    <span class="text-purple-400">Cache Hit Rate:</span>
                    <span class="text-purple-100">
                      {(optimizationResults.summary.cacheHitRate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-purple-400">Compression Savings:</span>
                    <span class="text-purple-100">{optimizationResults.summary.compressionSavings}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-purple-400">Context7 Patterns:</span>
                    <span class="text-purple-100">{optimizationResults.summary.context7Patterns}</span>
                  </div>
                </div>
              </div>
            </div>

          {:else if optimizationStatus === 'error'}
            <div class="p-8 text-center text-red-300 bg-red-900/20 rounded-lg border border-red-500/30">
              <div class="text-4xl mb-2">❌</div>
              <p>Optimization failed</p>
              <p class="text-sm mt-1">Check the error message above for details</p>
            </div>
          {/if}
        </div>
      </div>

    {:else if selectedTab === 'search'}
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Search interface -->
        <div class="space-y-4">
          <h2 class="text-xl font-semibold text-purple-100">Semantic Search</h2>

          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-purple-300 mb-2">
                Search Query
              </label>
              <div class="flex gap-2">
                <input
                  type="text"
                  bind:value={searchQuery}
                  placeholder="Enter search query (e.g., 'Svelte 5 props patterns')"
                  class="flex-1 px-4 py-2 bg-black/30 border border-purple-500/30 rounded-lg
                         text-purple-100 placeholder:text-purple-400"
                  onkeydown={(e) => e.key === 'Enter' && performSearch()}
                />
                <button
                  onclick={performSearch}
                  disabled={!searchQuery.trim() || isLoading}
                  class="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50
                         text-white rounded-lg transition-colors"
                >
                  Search
                </button>
              </div>
            </div>

            <!-- Quick search examples -->
            <div class="grid grid-cols-2 gap-2">
              {#each [
                'Svelte 5 runes patterns',
                'SvelteKit load functions',
                'Drizzle ORM schemas',
                'Vector embeddings'
              ] as example}
                <button
                  onclick={() => { searchQuery = example; performSearch(); }}
                  class="p-2 text-left text-sm bg-black/20 hover:bg-purple-700/20
                         text-purple-300 hover:text-purple-100 rounded border border-purple-500/20
                         transition-colors"
                >
                  {example}
                </button>
              {/each}
            </div>
          </div>
        </div>

        <!-- Search results -->
        <div class="space-y-4">
          <h2 class="text-xl font-semibold text-purple-100">Search Results</h2>

          {#if searchResults.length === 0}
            <div class="p-8 text-center text-purple-400 bg-black/20 rounded-lg border border-purple-500/20">
              <div class="text-4xl mb-2">🔍</div>
              <p>No search results yet</p>
              <p class="text-sm mt-1">Enter a query to search the optimized index</p>
            </div>
          {:else}
            <div class="space-y-3 max-h-96 overflow-y-auto">
              {#each searchResults as result, index}
                <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
                  <div class="flex items-start justify-between mb-2">
                    <h3 class="font-semibold text-purple-100 text-sm">
                      {result.document?.title || `Result ${index + 1}`}
                    </h3>
                    <div class="text-xs px-2 py-1 bg-purple-600 text-white rounded">
                      {(result.score * 100).toFixed(1)}%
                    </div>
                  </div>

                  <p class="text-purple-300 text-sm mb-2 line-clamp-3">
                    {result.document?.content || result.text | | "No content"}
                  </p>

                  <div class="flex items-center justify-between text-xs">
                    <span class="text-purple-400">
                      {result.explanation | | "Semantic match"}
                    </span>
                    {#if result.context7Pattern}
                      <span class="px-2 py-1 bg-blue-600/20 text-blue-300 rounded">
                        Context7: {result.context7Pattern}
                      </span>
                    {/if}
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>

    {:else if selectedTab === 'suggestions'}
      <div class="space-y-6">
        <div class="flex items-center justify-between">
          <h2 class="text-xl font-semibold text-purple-100">Code Suggestions</h2>
          <button
            onclick={generateSuggestions}
            disabled={isLoading}
            class="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50
                   text-white rounded-lg transition-colors"
          >
            {isLoading ? 'Generating...' : 'Generate Test Suggestions'}
          </button>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Test code context -->
          <div class="space-y-4">
            <h3 class="text-lg font-semibold text-purple-100">Test Context</h3>
            <pre class="p-4 bg-black/40 rounded-lg border border-purple-500/30 text-purple-100 text-sm overflow-x-auto"><code>{`// Test Svelte 5 component
<script lang="ts">
  let { data = [] } = $props();
  let count = $state(0);

  // Cursor position - suggestions generated here
  |
</script>
`}</code></pre>

            <div class="text-sm text-purple-400">
              The system analyzes this context and generates Context7-aware suggestions
              based on Svelte 5 patterns, legal AI domain knowledge, and semantic clustering.
            </div>
          </div>

          <!-- Generated suggestions -->
          <div class="space-y-4">
            <h3 class="text-lg font-semibold text-purple-100">Generated Suggestions</h3>

            {#if searchResults.length === 0}
              <div class="p-8 text-center text-purple-400 bg-black/20 rounded-lg border border-purple-500/20">
                <div class="text-4xl mb-2">💡</div>
                <p>No suggestions generated yet</p>
                <p class="text-sm mt-1">Click "Generate Test Suggestions" to see Context7-aware suggestions</p>
              </div>
            {:else}
              <div class="space-y-3 max-h-96 overflow-y-auto">
                {#each searchResults as suggestion, index}
                  <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
                    <div class="flex items-start justify-between mb-2">
                      <div class="text-sm font-semibold text-purple-100">
                        Suggestion {index + 1}
                      </div>
                      <div class="flex gap-2">
                        <div class="text-xs px-2 py-1 bg-green-600 text-white rounded">
                          {(suggestion.score * 100).toFixed(0)}% confidence
                        </div>
                        {#if suggestion.context7Pattern}
                          <div class="text-xs px-2 py-1 bg-blue-600 text-white rounded">
                            {suggestion.context7Pattern}
                          </div>
                        {/if}
                      </div>
                    </div>

                    <pre class="text-purple-300 text-sm bg-black/20 p-2 rounded border overflow-x-auto"><code>{suggestion.document?.content}</code></pre>

                    <div class="mt-2 text-xs text-purple-400">
                      {suggestion.explanation}
                    </div>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      </div>

    {:else if selectedTab === 'metrics'}
      <!-- Performance metrics -->
      {#if performanceMetrics}
        <div class="space-y-6">
          <h2 class="text-xl font-semibold text-purple-100">Performance Metrics</h2>

          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <!-- Optimizer metrics -->
            <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
              <h3 class="text-lg font-semibold text-purple-100 mb-3">Index Optimizer</h3>
              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span class="text-purple-400">Total Optimizations:</span>
                  <span class="text-purple-100">{performanceMetrics.optimizer.totalOptimizations}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Cache Hit Rate:</span>
                  <span class="text-purple-100">
                    {(performanceMetrics.optimizer.cacheHitRate * 100).toFixed(1)}%
                  </span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Avg Optimization Time:</span>
                  <span class="text-purple-100">
                    {formatTime(performanceMetrics.optimizer.avgOptimizationTime)}
                  </span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Pattern Matches:</span>
                  <span class="text-purple-100">{performanceMetrics.optimizer.patternMatches}</span>
                </div>
              </div>
            </div>

            <!-- SIMD processor metrics -->
            <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
              <h3 class="text-lg font-semibold text-purple-100 mb-3">SIMD Processor</h3>
              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span class="text-purple-400">Total Processed:</span>
                  <span class="text-purple-100">{performanceMetrics.simd.totalProcessed}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Cache Hits:</span>
                  <span class="text-purple-100">{performanceMetrics.simd.cacheHits}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Avg Parse Time:</span>
                  <span class="text-purple-100">
                    {formatTime(performanceMetrics.simd.avgParseTime)}
                  </span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Avg Embedding Time:</span>
                  <span class="text-purple-100">
                    {formatTime(performanceMetrics.simd.avgEmbeddingTime)}
                  </span>
                </div>
              </div>
            </div>

            <!-- System metrics -->
            <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
              <h3 class="text-lg font-semibold text-purple-100 mb-3">System</h3>
              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span class="text-purple-400">Memory Usage:</span>
                  <span class="text-purple-100">
                    {formatSize(performanceMetrics.system.memoryUsage.heapUsed)}
                  </span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Cache Size:</span>
                  <span class="text-purple-100">{performanceMetrics.cache.size} entries</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Node Version:</span>
                  <span class="text-purple-100">{performanceMetrics.system.nodeVersion}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-purple-400">Platform:</span>
                  <span class="text-purple-100">{performanceMetrics.system.platform}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      {:else}
        <div class="p-8 text-center text-purple-400 bg-black/20 rounded-lg border border-purple-500/20">
          <div class="text-4xl mb-2">📊</div>
          <p>Loading performance metrics...</p>
        </div>
      {/if}

    {:else if selectedTab === 'config'}
      <!-- Configuration -->
      <div class="max-w-2xl space-y-6">
        <h2 class="text-xl font-semibold text-purple-100">Optimization Configuration</h2>

        <div class="space-y-4">
          <!-- Boolean options -->
          {#each [
            { key: 'enableContext7Boost', label: 'Enable Context7 Pattern Boosting', description: 'Boost relevance for Context7-compatible patterns' },
            { key: 'enableSemanticClustering', label: 'Enable Semantic Clustering', description: 'Use SOM clustering for better organization' },
            { key: 'enablePatternRecognition', label: 'Enable Pattern Recognition', description: 'Recognize and boost known code patterns' },
            { key: 'enablePerformanceOptimization', label: 'Enable Performance Optimization', description: 'Apply compression and sorting optimizations' }
          ] as option}
            <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
              <label class="flex items-center justify-between">
                <div>
                  <div class="font-medium text-purple-100">{option.label}</div>
                  <div class="text-sm text-purple-400">{option.description}</div>
                </div>
                <input
                  type="checkbox"
                  bind:checked={optimizationConfig[option.key]}
                  class="w-5 h-5 text-purple-600 bg-black/30 border-purple-500 rounded
                         focus:ring-purple-500 focus:ring-2"
                />
              </label>
            </div>
          {/each}

          <!-- Numeric options -->
          <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
            <label class="block">
              <div class="font-medium text-purple-100 mb-1">Minimum Relevance Threshold</div>
              <div class="text-sm text-purple-400 mb-2">Filter out results below this relevance score</div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                bind:value={optimizationConfig.minRelevanceThreshold}
                class="w-full h-2 bg-black/30 rounded-lg appearance-none cursor-pointer slider"
              />
              <div class="text-sm text-purple-300 mt-1">
                Current: {optimizationConfig.minRelevanceThreshold}
              </div>
            </label>
          </div>

          <div class="p-4 bg-black/30 rounded-lg border border-purple-500/30">
            <label class="block">
              <div class="font-medium text-purple-100 mb-1">Compression Ratio</div>
              <div class="text-sm text-purple-400 mb-2">Reduce index size for better performance</div>
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.1"
                bind:value={optimizationConfig.compressionRatio}
                class="w-full h-2 bg-black/30 rounded-lg appearance-none cursor-pointer slider"
              />
              <div class="text-sm text-purple-300 mt-1">
                Current: {optimizationConfig.compressionRatio}
                ({(100 - optimizationConfig.compressionRatio * 100).toFixed(0)}% reduction)
              </div>
            </label>
          </div>
        </div>

        <!-- Save configuration -->
        <div class="flex gap-2">
          <button
            onclick={() => {
              localStorage.setItem('copilot-optimization-config', JSON.stringify(optimizationConfig));
              alert('Configuration saved!');
            }}
            class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
          >
            Save Configuration
          </button>

          <button
            onclick={() => {
              const saved = localStorage.getItem('copilot-optimization-config');
              if (saved) {
                optimizationConfig = JSON.parse(saved);
                alert('Configuration loaded!');
              }
            }}
            class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Load Saved
          </button>

          <button
            onclick={() => {
              optimizationConfig = {
                enableContext7Boost: true,
                enableSemanticClustering: true,
                enablePatternRecognition: true,
                enablePerformanceOptimization: true,
                minRelevanceThreshold: 0.7,
                compressionRatio: 0.8,
              };
            }}
            class="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
          >
            Reset to Defaults
          </button>
        </div>
      </div>
    {/if}
  </main>
</div>

<style>
  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical
    overflow: hidden
  }

  .slider::-webkit-slider-thumb {
    appearance: none
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: #7c3aed;
    cursor: pointer
    border: 2px solid #a855f7;
  }

  .slider::-webkit-slider-thumb:hover {
    background: #8b5cf6;
  }
</style>