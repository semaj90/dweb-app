<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<!-- EnhancedRAGInterface.svelte - SvelteKit 2.0 Advanced RAG Interface -->
<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { enhancedRAGStore } from '$lib/stores/enhanced-rag-store.js';
  import { Card, CardHeader, CardTitle, CardContent } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Badge } from '$lib/components/ui/badge';
  import { Progress } from '$lib/components/ui/progress';
  import {
    Search,
    Brain,
    Zap,
    Database,
    TrendingUp,
    Cpu,
    ChartBar,
    Lightbulb,
    RefreshCw,
    Settings,
    Sparkles,
    Target
  } from 'lucide-svelte';
  import type { SearchResult } from '$lib/types/rag.js';

  // Reactive state using Svelte 5 runes
  let searchQuery = $state('');
  let showAdvancedOptions = $state(false);
  let selectedClusters = $state<string[]>([]);
  let optimizationLevel = $state<'basic' | 'enhanced' | 'neural'>('enhanced');

  // Derived reactive state from store
  let ragState = $derived(enhancedRAGStore.state);
  let performanceMetrics = $derived(enhancedRAGStore.performanceMetrics);
  let optimizedResults = $derived(enhancedRAGStore.optimizedResults);
  let intelligentSuggestions = $derived(enhancedRAGStore.intelligentSuggestions);
  const search = enhancedRAGStore.search;
  const optimizeCache = enhancedRAGStore.optimizeCache;

  // Performance monitoring
  let searchStartTime = $state(0);
  let searchDuration = $state(0);

  // Advanced UI state
  let visualizationMode = $state<'list' | 'clusters' | 'neural' | 'performance'>('list');
  let autoOptimization = $state(true);

  onMount(() => {
    // Initialize the RAG system
    console.log('Enhanced RAG Interface initialized');

    // Start real-time performance monitoring
    const interval = setInterval(() => {
      updatePerformanceVisualization();
    }, 1000);

    return () => clearInterval(interval);
  });
  async function handleSearch() {
    if (!searchQuery.trim()) return;

    searchStartTime = performance.now();

    try {
      await search(searchQuery, {
        limit: 20,
        enableClustering: true,
        enableReranking: true,
        enableMLOptimization: true,
        cacheStrategy: optimizationLevel === 'neural' ? 'aggressive' : 'balanced'
      });

      searchDuration = performance.now() - searchStartTime;

      // Auto-scroll to results after search
      await tick();
      document.querySelector('#search-results')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });

    } catch (error) {
      console.error('Search failed:', error);
    }
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSearch();
    }
  }

  function selectSuggestion(suggestion: string) {
    searchQuery = suggestion;
    handleSearch();
  }

  function handleClusterSelect(clusterId: string) {
    if (selectedClusters.includes(clusterId)) {
      selectedClusters = selectedClusters.filter(id => id !== clusterId);
    } else {
      selectedClusters = [...selectedClusters, clusterId];
    }
  }

  async function handleOptimization() {
    await optimizeCache();
  }

  function updatePerformanceVisualization() {
    // Update real-time performance visualizations
    // This would update charts, graphs, etc.
  }

  function formatDuration(ms: number): string {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  function formatMemory(bytes: number): string {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)}MB`;
  }

  // Reactive calculations
  let searchPerformance = $derived({
    isSearching: ragState.isLoading,
    duration: searchDuration,
    resultsCount: optimizedResults.length,
    cacheHitRate: ragState.cacheMetrics.hitRate,
    memoryUsage: ragState.cacheMetrics.memoryUsageMB,
    predictionAccuracy: ragState.cacheMetrics.predictionAccuracy
  });

  let clusterVisualization = $derived(
    ragState.somClusters.map(cluster => ({
      ...cluster,
      isSelected: selectedClusters.includes(cluster.id),
      relevantResults: optimizedResults.filter(result => result.clusterId === cluster.id)
    }))
  );

  let performanceIndicators = $derived({
    throughput: performanceMetrics.throughputQPS,
    efficiency: performanceMetrics.memoryEfficiency,
    cacheOptimization: ragState.cacheMetrics.hitRate * 100,
    neuralAccuracy: ragState.cacheMetrics.predictionAccuracy * 100
  });
</script>

<!-- Main Interface -->
<div class="enhanced-rag-interface">
  <!-- Header with Real-time Status -->
  <Card class="mb-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
    <CardHeader>
      <div class="flex items-center justify-between">
        <div>
          <CardTitle class="flex items-center gap-2">
            <Brain class="h-6 w-6 text-blue-600" />
            Enhanced RAG System
            <Badge variant="outline" class="ml-2">
              {ragState.status.version}
            </Badge>
          </CardTitle>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Neural-powered legal research with SOM clustering and ML optimization
          </p>
        </div>

        <!-- Real-time Performance Indicators -->
        <div class="flex items-center gap-4">
          <div class="text-center">
            <div class="text-2xl font-bold text-green-600">
              {Math.round(performanceIndicators.efficiency * 100)}%
            </div>
            <div class="text-xs text-gray-500">Efficiency</div>
          </div>

          <div class="text-center">
            <div class="text-2xl font-bold text-blue-600">
              {Math.round(performanceIndicators.cacheOptimization)}%
            </div>
            <div class="text-xs text-gray-500">Cache Hit</div>
          </div>

          <div class="text-center">
            <div class="text-2xl font-bold text-purple-600">
              {Math.round(performanceIndicators.neuralAccuracy)}%
            </div>
            <div class="text-xs text-gray-500">ML Accuracy</div>
          </div>
        </div>
      </div>
    </CardHeader>
  </Card>

  <!-- Enhanced Search Interface -->
  <Card class="mb-6">
    <CardContent class="p-6">
      <!-- Main Search Bar -->
      <div class="relative mb-4">
        <div class="flex gap-2">
          <div class="relative flex-1">
            <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              bind:value={searchQuery}
              keydown={handleKeyPress}
              placeholder="Ask anything about your legal documents..."
              class="pl-10 pr-4 py-3 text-lg"
              disabled={ragState.isLoading}
            />

            <!-- Loading indicator -->
            {#if ragState.isLoading}
              <div class="absolute right-3 top-1/2 transform -translate-y-1/2">
                <RefreshCw class="h-4 w-4 animate-spin text-blue-600" />
              </div>
            {/if}
          </div>

          <Button
            on:on:click={handleSearch}
            disabled={!searchQuery.trim() || ragState.isLoading}
            class="px-6"
          >
            {ragState.isLoading ? 'Searching...' : 'Search'}
          </Button>

          <Button
            variant="outline"
            on:on:click={() => showAdvancedOptions = !showAdvancedOptions}
          >
            <Settings class="h-4 w-4" />
          </Button>
        </div>

        <!-- Real-time Search Performance -->
        {#if ragState.isLoading || searchDuration > 0}
          <div class="mt-2 text-sm text-gray-600 flex items-center gap-4">
            <span>Search time: {formatDuration(searchDuration)}</span>
            <span>Cache hit rate: {Math.round(ragState.cacheMetrics.hitRate * 100)}%</span>
            <span>Memory: {formatMemory(ragState.cacheMetrics.memoryUsageMB * 1024 * 1024)}</span>
          </div>
        {/if}
      </div>

      <!-- Intelligent Suggestions (Real-time) -->
      {#if intelligentSuggestions.length > 0}
        <div class="mb-4">
          <div class="flex items-center gap-2 mb-2">
            <Lightbulb class="h-4 w-4 text-yellow-500" />
            <span class="text-sm font-medium">Intelligent Suggestions</span>
          </div>
          <div class="flex flex-wrap gap-2">
            {#each intelligentSuggestions as suggestion}
              <Button
                variant="ghost"
                size="sm"
                on:on:click={() => selectSuggestion(suggestion)}
                class="h-8 px-3 bg-yellow-50 hover:bg-yellow-100 text-yellow-800 border border-yellow-200"
              >
                <Sparkles class="h-3 w-3 mr-1" />
                {suggestion}
              </Button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Did You Mean Suggestions -->
      {#if ragState.didYouMean.length > 0}
        <div class="mb-4">
          <div class="text-sm text-gray-600">
            Did you mean:
            {#each ragState.didYouMean as suggestion, i}
              <button
                on:onclick={() => selectSuggestion(suggestion)}
                class="text-blue-600 hover:text-blue-800 underline ml-1"
              >
                {suggestion}
              </button>
              {#if i < ragState.didYouMean.length - 1}, {/if}
            {/each}
          </div>
        </div>
      {/if}

      <!-- Advanced Options Panel -->
      {#if showAdvancedOptions}
        <div class="mt-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <!-- Optimization Level -->
            <div>
              <label class="block text-sm font-medium mb-2">Optimization Level</label>
              <select bind:value={optimizationLevel} class="w-full p-2 border rounded">
                <option value="basic">Basic</option>
                <option value="enhanced">Enhanced</option>
                <option value="neural">Neural</option>
              </select>
            </div>

            <!-- Visualization Mode -->
            <div>
              <label class="block text-sm font-medium mb-2">View Mode</label>
              <select bind:value={visualizationMode} class="w-full p-2 border rounded">
                <option value="list">List View</option>
                <option value="clusters">Cluster View</option>
                <option value="neural">Neural View</option>
                <option value="performance">Performance</option>
              </select>
            </div>

            <!-- Auto Optimization -->
            <div>
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={autoOptimization} />
                <span class="text-sm">Auto Optimization</span>
              </label>
              <Button
                variant="outline"
                size="sm"
                on:on:click={handleOptimization}
                class="mt-2 w-full"
              >
                <Zap class="h-4 w-4 mr-1" />
                Optimize Now
              </Button>
            </div>
          </div>
        </div>
      {/if}
    </CardContent>
  </Card>

  <!-- Results Section -->
  {#if optimizedResults.length > 0}
    <div id="search-results">
      <!-- Results Header with Filters -->
      <Card class="mb-4">
        <CardContent class="p-4">
          <div class="flex items-center justify-between">
            <div>
              <h3 class="font-semibold">
                {optimizedResults.length} results
                {#if ragState.currentQuery}for "{ragState.currentQuery}"{/if}
              </h3>
              <p class="text-sm text-gray-600">
                Found in {formatDuration(searchDuration)} •
                Cache hit rate: {Math.round(ragState.cacheMetrics.hitRate * 100)}%
              </p>
            </div>

            <!-- View Toggle -->
            <div class="flex gap-2">
              <Button
                variant={visualizationMode === 'list' ? 'default' : 'outline'}
                size="sm"
                on:on:click={() => visualizationMode = 'list'}
              >
                List
              </Button>
              <Button
                variant={visualizationMode === 'clusters' ? 'default' : 'outline'}
                size="sm"
                on:on:click={() => visualizationMode = 'clusters'}
              >
                <Target class="h-4 w-4 mr-1" />
                Clusters
              </Button>
              <Button
                variant={visualizationMode === 'performance' ? 'default' : 'outline'}
                size="sm"
                on:on:click={() => visualizationMode = 'performance'}
              >
                <BarChart3 class="h-4 w-4 mr-1" />
                Analytics
                <ChartBar class="h-4 w-4 mr-1" />
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Dynamic Content Based on View Mode -->
      {#if visualizationMode === 'list'}
        <!-- Enhanced Results List -->
        <div class="space-y-4">
          {#each optimizedResults as result, index}
            <Card class="hover:shadow-lg transition-shadow">
              <CardContent class="p-6">
                <div class="flex items-start justify-between mb-3">
                  <div class="flex-1">
                    <h4 class="font-semibold text-lg mb-1">
                      {result.document.title || `Document ${result.id}`}
                    </h4>
                    <div class="flex items-center gap-4 text-sm text-gray-600">
                      <span>Relevance: {Math.round(result.score * 100)}%</span>
                      {#if result.cacheLayer}
                        <Badge variant="outline" class="text-xs">
                          {result.cacheLayer}
                        </Badge>
                      {/if}
                      {#if result.clusterId}
                        <Badge variant="secondary" class="text-xs">
                          Cluster {result.clusterId}
                        </Badge>
                      {/if}
                    </div>
                  </div>

                  <!-- Neural Prediction Indicator -->
                  {#if result.legalRelevance}
                    <div class="text-right">
                      <div class="text-2xl font-bold text-blue-600">
                        {Math.round(result.legalRelevance.overall * 100)}
                      </div>
                      <div class="text-xs text-gray-500">Legal Score</div>
                    </div>
                  {/if}
                </div>

                <!-- Document Preview -->
                {#if result.highlights && result.highlights.length > 0}
                  <div class="mb-3">
                    <div class="text-sm bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded border-l-4 border-yellow-400">
                      {#each result.highlights.slice(0, 2) as highlight}
                        <p class="mb-1 last:mb-0">...{@html highlight}...</p>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Metadata Tags -->
                {#if result.document.metadata.tags}
                  <div class="flex flex-wrap gap-1">
                    {#each result.document.metadata.tags.slice(0, 5) as tag}
                      <Badge variant="outline" class="text-xs">
                        {tag}
                      </Badge>
                    {/each}
                  </div>
                {/if}
              </CardContent>
            </Card>
          {/each}
        </div>

      {:else if visualizationMode === 'clusters'}
        <!-- SOM Cluster Visualization -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {#each clusterVisualization as cluster}
            <Card class="cursor-pointer transition-all {cluster.isSelected ? 'ring-2 ring-blue-500' : ''}"
                  on:on:click={() => handleClusterSelect(cluster.id)}>
              <CardHeader>
                <CardTitle class="flex items-center justify-between">
                  <span>Cluster {cluster.id}</span>
                  <Badge variant="secondary">
                    {cluster.relevantResults.length} results
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div class="space-y-2">
                  <div class="flex flex-wrap gap-1">
                    {#each cluster.keywords.slice(0, 6) as keyword}
                      <Badge variant="outline" class="text-xs">
                        {keyword}
                      </Badge>
                    {/each}
                  </div>

                  <div class="text-sm text-gray-600">
                    Coherence: {Math.round(cluster.coherence * 100)}% •
                    Avg Relevance: {Math.round(cluster.avgRelevance * 100)}%
                  </div>

                  <Progress value={cluster.coherence * 100} class="h-2" />
                </div>
              </CardContent>
            </Card>
          {/each}
        </div>

      {:else if visualizationMode === 'performance'}
        <!-- Performance Analytics Dashboard -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent class="p-4 text-center">
              <Cpu class="h-8 w-8 mx-auto text-blue-600 mb-2" />
              <div class="text-2xl font-bold">{performanceMetrics.totalQueries}</div>
              <div class="text-sm text-gray-600">Total Queries</div>
            </CardContent>
          </Card>

          <Card>
            <CardContent class="p-4 text-center">
              <Zap class="h-8 w-8 mx-auto text-green-600 mb-2" />
              <div class="text-2xl font-bold">{Math.round(performanceMetrics.throughputQPS)}</div>
              <div class="text-sm text-gray-600">Queries/sec</div>
            </CardContent>
          </Card>

          <Card>
            <CardContent class="p-4 text-center">
              <Database class="h-8 w-8 mx-auto text-purple-600 mb-2" />
              <div class="text-2xl font-bold">{performanceMetrics.cacheSize}</div>
              <div class="text-sm text-gray-600">Cache Entries</div>
            </CardContent>
          </Card>

          <Card>
            <CardContent class="p-4 text-center">
              <TrendingUp class="h-8 w-8 mx-auto text-orange-600 mb-2" />
              <div class="text-2xl font-bold">
                {formatDuration(performanceMetrics.averageResponseTime)}
              </div>
              <div class="text-sm text-gray-600">Avg Response</div>
            </CardContent>
          </Card>
        </div>

        <!-- Real-time Performance Chart -->
        <Card>
          <CardHeader>
            <CardTitle>Real-time Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="h-64 bg-gray-50 dark:bg-gray-900 rounded flex items-center justify-center">
              <p class="text-gray-500">Performance chart would be rendered here</p>
            </div>
          </CardContent>
        </Card>
      {/if}
    </div>
  {/if}

  <!-- Error Display -->
  {#if ragState.error}
    <Card class="border-red-200 bg-red-50 dark:bg-red-950">
      <CardContent class="p-4">
        <div class="flex items-center gap-2 text-red-800 dark:text-red-200">
          <span class="font-medium">Error:</span>
          <span>{ragState.error}</span>
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  .enhanced-rag-interface {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
  }

  @keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.5); }
    50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8); }
  }

  .enhanced-rag-interface :global(.searching) {
    animation: pulse-glow 2s infinite;
  }
</style>


