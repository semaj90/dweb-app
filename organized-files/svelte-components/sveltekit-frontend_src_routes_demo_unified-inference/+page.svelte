<!--
  Unified Inference Pipeline Demo
  Showcases complete production inference system with Redis caching
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { 
    unifiedInferenceClient, 
    isProcessing, 
    lastResponse, 
    pipelineHealth, 
    pipelineStats 
  } from '$lib/services/unified-inference-client';
  import Button from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/Badge.svelte';

  // Demo state
  let { 
    selectedDemo = $state('generation'),
    demoText = $state('Analyze the legal implications of force majeure clauses in commercial contracts.'),
    demoResults = $state(null),
    embeddingTexts = $state(['contract', 'liability', 'evidence', 'jurisdiction']),
    similarityVector = $state([]),
    showAdvanced = $state(false),
    cacheStrategy = $state('hybrid'),
    demoHistory = $state([])
  } = $props();

  // Reactive computations using $derived
  const healthStatus = $derived($pipelineHealth?.status || 'unknown')
  const isHealthy = $derived(healthStatus === 'healthy')
  const totalRequests = $derived($pipelineStats?.requests.total || 0)
  const cacheHitRate = $derived($pipelineStats?.performance.cacheHitRate || 0)
  const avgLatency = $derived($pipelineStats?.performance.averageLatency || 0)

  // Demo options
  const demoTypes = [
    { id: 'generation', label: 'Text Generation', icon: '🤖' },
    { id: 'embedding', label: 'Embeddings', icon: '🧮' },
    { id: 'similarity', label: 'Similarity Search', icon: '🔍' },
    { id: 'legal_analysis', label: 'Legal Analysis', icon: '⚖️' },
    { id: 'tokenization', label: 'Tokenization', icon: '✂️' }
  ];

  const cacheStrategies = [
    { value: 'hybrid', label: 'Hybrid (Client + Server)' },
    { value: 'client-only', label: 'Client Only (XState + Loki)' },
    { value: 'server-only', label: 'Server Only (Redis)' },
    { value: 'no-cache', label: 'No Cache' }
  ];

  // Sample legal documents for similarity search
  const sampleDocuments = [
    'Software licensing agreement with unlimited user provisions',
    'Commercial lease contract with force majeure clauses',
    'Employment contract with non-compete restrictions',
    'Service agreement with liability limitations',
    'Purchase agreement with warranty disclaimers'
  ];

  onMount(async () => {
    // Initialize the unified client
    await unifiedInferenceClient.initialize();
    
    // Generate some sample embeddings for similarity demo
    if (selectedDemo === 'similarity') {
      await generateSampleEmbeddings();
    }
  });

  // Demo execution functions
  async function runTextGeneration() {
    try {
      demoResults = null;
      const response = await unifiedInferenceClient.generateText(demoText, {
        model: 'gemma3-legal',
        temperature: 0.7,
        maxTokens: 300,
        cacheStrategy: cacheStrategy as any
      });
      
      demoResults = response;
      addToHistory('Text Generation', response);
    } catch (error) {
      console.error('Generation demo error:', error);
      demoResults = { error: error.message };
    }
  }

  async function runEmbedding() {
    try {
      demoResults = null;
      const response = await unifiedInferenceClient.generateEmbeddings(
        embeddingTexts,
        'nomic-embed-text',
        true
      );
      
      demoResults = response;
      addToHistory('Embeddings', response);
    } catch (error) {
      console.error('Embedding demo error:', error);
      demoResults = { error: error.message };
    }
  }

  async function runSimilaritySearch() {
    try {
      if (similarityVector.length === 0) {
        await generateSampleEmbeddings();
      }
      
      demoResults = null;
      const response = await unifiedInferenceClient.searchSimilarDocuments(
        similarityVector,
        undefined, // caseId
        'legal_document',
        5
      );
      
      demoResults = response;
      addToHistory('Similarity Search', response);
    } catch (error) {
      console.error('Similarity demo error:', error);
      demoResults = { error: error.message };
    }
  }

  async function runLegalAnalysis() {
    try {
      demoResults = null;
      const response = await unifiedInferenceClient.analyzeLegalDocument(
        'demo-doc-001',
        demoText,
        'summary'
      );
      
      demoResults = response;
      addToHistory('Legal Analysis', response);
    } catch (error) {
      console.error('Legal analysis demo error:', error);
      demoResults = { error: error.message };
    }
  }

  async function runTokenization() {
    try {
      demoResults = null;
      const response = await unifiedInferenceClient.processRequest({
        type: 'tokenize',
        prompt: demoText,
        metadata: { model: 'legal-tokenizer' },
        cacheStrategy: cacheStrategy as any
      });
      
      demoResults = response;
      addToHistory('Tokenization', response);
    } catch (error) {
      console.error('Tokenization demo error:', error);
      demoResults = { error: error.message };
    }
  }

  async function generateSampleEmbeddings() {
    try {
      const response = await unifiedInferenceClient.generateEmbeddings([demoText]);
      if (response.data?.embeddings?.length > 0) {
        similarityVector = response.data.embeddings[0].vector || [];
      }
    } catch (error) {
      console.error('Sample embedding generation error:', error);
    }
  }

  function addToHistory(type: string, response: any) {
    demoHistory = [{
      type,
      timestamp: new Date().toLocaleTimeString(),
      cached: response.cached,
      processingTime: response.processingTime,
      cacheLevel: response.cacheLevel
    }, ...demoHistory.slice(0, 9)]; // Keep last 10
  }

  async function runDemo() {
    switch (selectedDemo) {
      case 'generation':
        await runTextGeneration();
        break;
      case 'embedding':
        await runEmbedding();
        break;
      case 'similarity':
        await runSimilaritySearch();
        break;
      case 'legal_analysis':
        await runLegalAnalysis();
        break;
      case 'tokenization':
        await runTokenization();
        break;
    }
  }

  function formatLatency(ms: number): string {
    return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(2)}s`;
  }

  function getHealthColor(status: string): string {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  }
</script>

<div class="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-white mb-4">
        🚀 Unified Inference Pipeline Demo
      </h1>
      <p class="text-xl text-blue-200 mb-6">
        Complete production system with Redis caching, GPU acceleration, and multi-level optimization
      </p>
      
      <!-- Health Status -->
      <div class="flex justify-center items-center gap-4 mb-6">
        <Badge class={`${getHealthColor(healthStatus)} bg-opacity-20`}>
          System: {healthStatus.toUpperCase()}
        </Badge>
        <Badge class="text-blue-400 bg-blue-900 bg-opacity-30">
          Requests: {totalRequests}
        </Badge>
        <Badge class="text-green-400 bg-green-900 bg-opacity-30">
          Cache Hit: {cacheHitRate.toFixed(1)}%
        </Badge>
        <Badge class="text-purple-400 bg-purple-900 bg-opacity-30">
          Avg Latency: {formatLatency(avgLatency)}
        </Badge>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Control Panel -->
      <div class="lg:col-span-1">
        <Card class="bg-gray-800 border-blue-500">
          <CardHeader>
            <CardTitle class="text-white">Demo Controls</CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <!-- Demo Type Selection -->
            <div>
              <label class="text-white text-sm font-medium mb-2 block">Demo Type</label>
              <div class="grid grid-cols-1 gap-2">
                {#each demoTypes as demo}
                  <button
                    class={`p-3 rounded-lg border text-left transition-all ${
                      selectedDemo === demo.id
                        ? 'bg-blue-600 border-blue-400 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600'
                    }`}
                    onclick={() => { selectedDemo = demo.id; }}
                  >
                    <span class="text-lg mr-2">{demo.icon}</span>
                    {demo.label}
                  </button>
                {/each}
              </div>
            </div>

            <!-- Cache Strategy -->
            <div>
              <label class="text-white text-sm font-medium mb-2 block">Cache Strategy</label>
              <select 
                bind:value={cacheStrategy}
                class="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
              >
                {#each cacheStrategies as strategy}
                  <option value={strategy.value}>{strategy.label}</option>
                {/each}
              </select>
            </div>

            <!-- Input Text -->
            {#if selectedDemo === 'generation' || selectedDemo === 'legal_analysis' || selectedDemo === 'tokenization'}
              <div>
                <label class="text-white text-sm font-medium mb-2 block">Input Text</label>
                <textarea 
                  bind:value={demoText}
                  class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white h-24 resize-none"
                  placeholder="Enter your text here..."
                ></textarea>
              </div>
            {/if}

            <!-- Embedding Texts -->
            {#if selectedDemo === 'embedding'}
              <div>
                <label class="text-white text-sm font-medium mb-2 block">Texts to Embed</label>
                <div class="space-y-2">
                  {#each embeddingTexts as text, i}
                    <input 
                      bind:value={embeddingTexts[i]}
                      class="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      placeholder="Enter text..."
                    />
                  {/each}
                  <Button
                    onclick={() => { embeddingTexts = [...embeddingTexts, '']; }}
                    class="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    Add Text
                  </Button>
                </div>
              </div>
            {/if}

            <!-- Run Demo Button -->
            <Button
              onclick={runDemo}
              disabled={$isProcessing}
              class="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50"
            >
              {#if $isProcessing}
                <span class="animate-spin mr-2">⚡</span>
                Processing...
              {:else}
                🚀 Run Demo
              {/if}
            </Button>

            <!-- Advanced Options Toggle -->
            <Button
              onclick={() => { showAdvanced = !showAdvanced; }}
              class="w-full bg-gray-600 hover:bg-gray-700"
            >
              {showAdvanced ? '📋 Hide' : '⚙️ Show'} Advanced Options
            </Button>
          </CardContent>
        </Card>

        <!-- Service Status -->
        <Card class="bg-gray-800 border-green-500 mt-6">
          <CardHeader>
            <CardTitle class="text-white">Service Status</CardTitle>
          </CardHeader>
          <CardContent>
            {#if $pipelineHealth}
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span class="text-gray-300">Redis Cache</span>
                  <Badge class={$pipelineHealth.services.redis ? 'text-green-400 bg-green-900' : 'text-red-400 bg-red-900'}>
                    {$pipelineHealth.services.redis ? '✅ Online' : '❌ Offline'}
                  </Badge>
                </div>
                <div class="flex justify-between">
                  <span class="text-gray-300">Go Gateway</span>
                  <Badge class={$pipelineHealth.services.goGateway ? 'text-green-400 bg-green-900' : 'text-red-400 bg-red-900'}>
                    {$pipelineHealth.services.goGateway ? '✅ Online' : '❌ Offline'}
                  </Badge>
                </div>
                <div class="flex justify-between">
                  <span class="text-gray-300">Python GPU</span>
                  <Badge class={$pipelineHealth.services.pythonGpu ? 'text-green-400 bg-green-900' : 'text-red-400 bg-red-900'}>
                    {$pipelineHealth.services.pythonGpu ? '✅ Online' : '❌ Offline'}
                  </Badge>
                </div>
                <div class="flex justify-between">
                  <span class="text-gray-300">Client Cache</span>
                  <Badge class={$pipelineHealth.services.clientCache ? 'text-green-400 bg-green-900' : 'text-red-400 bg-red-900'}>
                    {$pipelineHealth.services.clientCache ? '✅ Online' : '❌ Offline'}
                  </Badge>
                </div>
              </div>
            {:else}
              <div class="text-gray-400">Loading service status...</div>
            {/if}
          </CardContent>
        </Card>
      </div>

      <!-- Results Panel -->
      <div class="lg:col-span-2">
        <Card class="bg-gray-800 border-purple-500">
          <CardHeader>
            <CardTitle class="text-white flex items-center gap-2">
              📊 Demo Results
              {#if $lastResponse?.cached}
                <Badge class="text-green-400 bg-green-900">
                  {$lastResponse.cacheLevel?.toUpperCase()} CACHE HIT
                </Badge>
              {/if}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {#if $isProcessing}
              <div class="flex items-center justify-center h-64">
                <div class="animate-spin text-4xl">⚡</div>
                <span class="ml-4 text-xl text-blue-400">Processing request...</span>
              </div>
            {:else if demoResults}
              {#if demoResults.error}
                <div class="bg-red-900 bg-opacity-30 border border-red-600 rounded-lg p-4">
                  <h3 class="text-red-400 font-semibold mb-2">Error</h3>
                  <p class="text-red-300">{demoResults.error}</p>
                </div>
              {:else}
                <div class="space-y-4">
                  <!-- Response Metadata -->
                  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div class="bg-gray-700 rounded-lg p-3">
                      <div class="text-gray-400 text-sm">Request ID</div>
                      <div class="text-white font-mono text-sm">{demoResults.requestId?.slice(0, 12)}...</div>
                    </div>
                    <div class="bg-gray-700 rounded-lg p-3">
                      <div class="text-gray-400 text-sm">Processing Time</div>
                      <div class="text-white">{formatLatency(demoResults.processingTime)}</div>
                    </div>
                    <div class="bg-gray-700 rounded-lg p-3">
                      <div class="text-gray-400 text-sm">Cache Status</div>
                      <div class={demoResults.cached ? 'text-green-400' : 'text-yellow-400'}>
                        {demoResults.cached ? 'HIT' : 'MISS'}
                      </div>
                    </div>
                    <div class="bg-gray-700 rounded-lg p-3">
                      <div class="text-gray-400 text-sm">Cache Level</div>
                      <div class="text-purple-400">{demoResults.cacheLevel || 'none'}</div>
                    </div>
                  </div>

                  <!-- Response Data -->
                  <div class="bg-gray-700 rounded-lg p-4">
                    <h3 class="text-white font-semibold mb-3">Response Data</h3>
                    <pre class="bg-gray-900 rounded p-4 text-green-400 text-sm overflow-auto max-h-64">
{JSON.stringify(demoResults.data, null, 2)}</pre>
                  </div>
                </div>
              {/if}
            {:else}
              <div class="flex items-center justify-center h-64 text-gray-400">
                <div class="text-center">
                  <div class="text-4xl mb-4">🚀</div>
                  <p>Select a demo type and click "Run Demo" to see results</p>
                </div>
              </div>
            {/if}
          </CardContent>
        </Card>

        <!-- Demo History -->
        {#if demoHistory.length > 0}
          <Card class="bg-gray-800 border-yellow-500 mt-6">
            <CardHeader>
              <CardTitle class="text-white">Recent Demo History</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="space-y-2">
                {#each demoHistory as entry}
                  <div class="flex items-center justify-between bg-gray-700 rounded p-3">
                    <div class="flex items-center gap-3">
                      <span class="text-blue-400 font-medium">{entry.type}</span>
                      <span class="text-gray-400 text-sm">{entry.timestamp}</span>
                    </div>
                    <div class="flex items-center gap-2">
                      <Badge class={entry.cached ? 'text-green-400 bg-green-900' : 'text-yellow-400 bg-yellow-900'}>
                        {entry.cached ? 'CACHED' : 'FRESH'}
                      </Badge>
                      <span class="text-purple-400 text-sm">{formatLatency(entry.processingTime)}</span>
                    </div>
                  </div>
                {/each}
              </div>
            </CardContent>
          </Card>
        {/if}
      </div>
    </div>

    <!-- Advanced Options Panel -->
    {#if showAdvanced}
      <Card class="bg-gray-800 border-orange-500 mt-6">
        <CardHeader>
          <CardTitle class="text-white">🔧 Advanced Pipeline Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <!-- Cache Statistics -->
            <div>
              <h3 class="text-white font-semibold mb-3">Cache Statistics</h3>
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span class="text-gray-300">Hit Rate</span>
                  <span class="text-green-400">{cacheHitRate.toFixed(1)}%</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-gray-300">Total Requests</span>
                  <span class="text-blue-400">{totalRequests}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-gray-300">Avg Latency</span>
                  <span class="text-purple-400">{formatLatency(avgLatency)}</span>
                </div>
              </div>
            </div>

            <!-- Sample Documents -->
            <div>
              <h3 class="text-white font-semibold mb-3">Sample Documents</h3>
              <div class="space-y-1 max-h-32 overflow-y-auto">
                {#each sampleDocuments as doc}
                  <div class="text-gray-300 text-sm bg-gray-700 rounded p-2">
                    {doc}
                  </div>
                {/each}
              </div>
            </div>

            <!-- Quick Actions -->
            <div>
              <h3 class="text-white font-semibold mb-3">Quick Actions</h3>
              <div class="space-y-2">
                <Button
                  onclick={() => unifiedInferenceClient.prefetchLegalQueries('DEMO-CASE-001', ['contract', 'evidence'])}
                  class="w-full bg-green-600 hover:bg-green-700 text-sm"
                >
                  🔄 Prefetch Legal Queries
                </Button>
                <Button
                  onclick={() => { demoHistory = []; }}
                  class="w-full bg-red-600 hover:bg-red-700 text-sm"
                >
                  🗑️ Clear History
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    {/if}
  </div>
</div>

<style>
  :global(body) {
    background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
  }
</style>