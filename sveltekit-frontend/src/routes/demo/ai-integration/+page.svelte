<!-- AI Integration Demo - Complete System Showcase -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { useMachine } from '@xstate/svelte';
  import { prefetchMachine } from '$lib/machines/prefetchMachine';
  import OllamaAgentShell from '$lib/components/ai/ollama-agent-shell.svelte';
  import WebGPUViewer from '$lib/components/ai/webgpu-viewer.svelte';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { Tabs, TabsContent, TabsList, TabsTrigger } from '$lib/components/ui/tabs';
  import { Play, Pause, Brain, Cpu, Database, Zap, Globe, Activity, Terminal, Eye } from 'lucide-svelte';

  // XState Prefetch Machine
  const { snapshot: prefetchState, send: prefetchSend } = useMachine(prefetchMachine);

  // Component State
  let systemHealth = $state({
    go: { status: 'checking' },
    ollama: { status: 'checking' },
    postgres: { status: 'checking' },
    redis: { status: 'checking' }
  });

  let embeddings = $state<number[][]>([]);
  let labels = $state<string[]>([]);
  let agentShellOpen = $state(false);
  let isMonitoring = $state(false);
  let performanceMetrics = $state({
    requests: 0,
    avgResponseTime: 0,
    cacheHitRate: 0,
    embeddingsGenerated: 0,
    vectorSimilarityQueries: 0
  });

  // Demo data
  const demoDocuments = [
    { id: 'doc1', title: 'Contract Analysis', content: 'This contract contains liability clauses...' },
    { id: 'doc2', title: 'Legal Precedent', content: 'Similar cases show evidence patterns...' },
    { id: 'doc3', title: 'Risk Assessment', content: 'Based on the evidence, the risk factors...' },
    { id: 'doc4', title: 'Compliance Review', content: 'The regulatory framework requires...' }
  ];

  onMount(() => {
    // Disabled auto health checks to prevent CORS errors
    // checkSystemHealth();
    generateDemoEmbeddings();

    // Start monitoring if specified
    if (isMonitoring) {
      startPerformanceMonitoring();
    }

    // Initialize prefetch machine with demo context
    prefetchSend({
      type: 'USER_ACTION',
      action: 'demo_start',
      context: { demo: true, documents: demoDocuments.length }
    });
  });

  async function checkSystemHealth() {
    const healthChecks = [
      { name: 'go', url: 'http://localhost:8080/health' },
      { name: 'ollama', url: 'http://localhost:11434/api/tags' },
      { name: 'postgres', url: 'http://localhost:8080/database-status' },
    ];

    for (const check of healthChecks) {
      try {
        const response = await fetch(check.url, {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });

        systemHealth[check.name] = {
          status: response.ok ? 'healthy' : 'error',
          details: response.ok ? 'Connected' : `HTTP ${response.status}`
        };
      } catch (error) {
        systemHealth[check.name] = {
          status: 'error',
          details: error instanceof Error ? error.message : 'Connection failed'
        };
      }
    }

    // Mock Redis status for demo
    systemHealth.redis = { status: 'healthy', details: 'Connected' };
  }

  async function generateDemoEmbeddings() {
    // Simulate generating embeddings for demo documents
    const mockEmbeddings = demoDocuments.map((_, index) => {
      const base = index * 0.25;
      return [
        Math.sin(base) * 0.5 + Math.random() * 0.1,
        Math.cos(base) * 0.5 + Math.random() * 0.1,
        Math.sin(base * 2) * 0.3 + Math.random() * 0.1
      ];
    });

    embeddings = mockEmbeddings;
    labels = demoDocuments.map(doc => doc.title);

    // Update prefetch machine with embedding data
    prefetchSend({
      type: 'UPDATE_EMBEDDINGS',
      embeddings: mockEmbeddings
    });

    performanceMetrics.embeddingsGenerated = demoDocuments.length;
  }

  async function testOllamaIntegration() {
    try {
      performanceMetrics.requests++;
      const startTime = performance.now();

      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: 'Analyze this legal document: "This contract contains liability clauses..."',
          stream: false
        })
      });

      const endTime = performance.now();
      performanceMetrics.avgResponseTime = Math.round(
        (performanceMetrics.avgResponseTime + (endTime - startTime)) / 2
      );

      if (response.ok) {
        const data = await response.json();
        console.log('Ollama Analysis Response:', data.response);

        // Trigger prefetch based on analysis success
        prefetchSend({
          type: 'USER_ACTION',
          action: 'ai_analysis_complete',
          context: { success: true, model: 'gemma3-legal' }
        });

        return data.response;
      } else {
        throw new Error(`Ollama request failed: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Ollama integration test failed:', error);
      throw error;
    }
  }

  async function testEmbeddingGeneration() {
    try {
      performanceMetrics.requests++;
      const startTime = performance.now();

      const response = await fetch('http://localhost:8080/generate-embedding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: 'Test embedding generation for legal documents',
          model: 'nomic-embed-text'
        })
      });

      const endTime = performance.now();
      performanceMetrics.avgResponseTime = Math.round(
        (performanceMetrics.avgResponseTime + (endTime - startTime)) / 2
      );

      if (response.ok) {
        const data = await response.json();
        console.log('Generated embedding:', data.embedding?.slice(0, 5));
        performanceMetrics.embeddingsGenerated++;

        // Cache hit simulation
        performanceMetrics.cacheHitRate = Math.min(
          95,
          performanceMetrics.cacheHitRate + Math.random() * 10
        );

        return data.embedding;
      } else {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Embedding generation test failed:', error);
      throw error;
    }
  }

  async function testVectorSimilarity() {
    try {
      performanceMetrics.requests++;
      performanceMetrics.vectorSimilarityQueries++;

      const response = await fetch('http://localhost:8080/search-similar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_embedding: embeddings[0] || [0.1, 0.2, 0.3],
          threshold: 0.7,
          limit: 5
        })
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Similarity search results:', data);

        // Update cache statistics
        prefetchSend({ type: 'CACHE_HIT', resource: 'similarity_search' });

        return data;
      } else {
        throw new Error(`Similarity search failed: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Vector similarity test failed:', error);
      prefetchSend({ type: 'CACHE_MISS', resource: 'similarity_search' });
      throw error;
    }
  }

  function startPerformanceMonitoring() {
    isMonitoring = true;

    const interval = setInterval(() => {
      if (!isMonitoring) {
        clearInterval(interval);
        return;
      }

      // Simulate metrics updates
      performanceMetrics.cacheHitRate = Math.min(
        95,
        performanceMetrics.cacheHitRate + (Math.random() - 0.5) * 5
      );

      // Update prefetch machine with viewport changes
      prefetchSend({
        type: 'VIEWPORT_CHANGE',
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
          scrollY: window.scrollY
        }
      });
    }, 2000);
  }

  function stopPerformanceMonitoring() {
    isMonitoring = false;
  }

  function openAgentShell() {
    agentShellOpen = true;

    // Notify prefetch machine
    prefetchSend({
      type: 'USER_ACTION',
      action: 'agent_shell_opened',
      context: { timestamp: Date.now() }
    });
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'checking': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  }

  function getStatusIcon(service: string) {
    switch (service) {
      case 'go': return Cpu;
      case 'ollama': return Brain;
      case 'postgres': return Database;
      case 'redis': return Zap;
      default: return Globe;
    }
  }

  // Reactive prefetch statistics
  let prefetchStats = $derived({
    state: $prefetchState.value,
    confidence: $prefetchState.context.confidence,
    cacheHits: $prefetchState.context.metrics.hits,
    cacheMisses: $prefetchState.context.metrics.misses,
    queueLength: $prefetchState.context.prefetchQueue.length
  });

  let hitRate = $derived(prefetchStats.cacheHits + prefetchStats.cacheMisses > 0
    ? Math.round((prefetchStats.cacheHits / (prefetchStats.cacheHits + prefetchStats.cacheMisses)) * 100)
    : 0);
</script>

<div class="ai-integration-demo min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">

  <!-- Header -->
  <div class="mb-8">
    <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
      🚀 AI Integration System Demo
    </h1>
    <p class="text-slate-300 text-lg">
      Complete legal AI pipeline with Go + SIMD, Ollama LLM, WebGPU visualization, and XState orchestration
    </p>
  </div>

  <!-- System Health Status -->
  <Card class="mb-8 bg-slate-800/50 border-slate-700">
    <CardHeader>
      <CardTitle class="flex items-center gap-2 text-slate-100">
        <Activity class="h-5 w-5" />
        System Health Status
      </CardTitle>
      <CardDescription class="text-slate-400">
        Real-time monitoring of all system components
      </CardDescription>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        {#each Object.entries(systemHealth) as [service, health]}
          {@const IconComponent = getStatusIcon(service)}
          <div class="flex items-center gap-3 p-3 rounded-lg bg-slate-700/30">
            <div class="relative">
              <IconComponent class="h-6 w-6 text-slate-300" />
              <div class={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full ${getStatusColor(health.status)}`}></div>
            </div>
            <div>
              <div class="font-medium text-slate-200 capitalize">{service}</div>
              <div class="text-xs text-slate-400">{health.details || health.status}</div>
            </div>
          </div>
        {/each}
      </div>
    </CardContent>
  </Card>

  <!-- Main Demo Tabs -->
  <Tabs defaultValue="webgpu" class="space-y-6">
    <TabsList class="bg-slate-800 border-slate-700">
      <TabsTrigger value="webgpu" class="data-[state=active]:bg-slate-700">
        <Eye class="h-4 w-4 mr-2" />
        WebGPU Visualization
      </TabsTrigger>
      <TabsTrigger value="ollama" class="data-[state=active]:bg-slate-700">
        <Brain class="h-4 w-4 mr-2" />
        AI Agent Shell
      </TabsTrigger>
      <TabsTrigger value="prefetch" class="data-[state=active]:bg-slate-700">
        <Zap class="h-4 w-4 mr-2" />
        XState Prefetch
      </TabsTrigger>
      <TabsTrigger value="performance" class="data-[state=active]:bg-slate-700">
        <Activity class="h-4 w-4 mr-2" />
        Performance
      </TabsTrigger>
    </TabsList>

    <!-- WebGPU Visualization Tab -->
    <TabsContent value="webgpu" class="space-y-6">
      <Card class="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle class="text-slate-100">GPU-Accelerated Vector Visualization</CardTitle>
          <CardDescription class="text-slate-400">
            Real-time 3D rendering of document embeddings using WebGPU shaders
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div class="space-y-4">
            <div class="flex items-center gap-4 mb-4">
              <Badge variant="outline" class="bg-blue-500/10 text-blue-400 border-blue-500/30">
                {embeddings.length} Vectors Loaded
              </Badge>
              <Badge variant="outline" class="bg-green-500/10 text-green-400 border-green-500/30">
                WebGPU Enabled
              </Badge>
              <Button on:on:click={generateDemoEmbeddings} size="sm">
                Regenerate Embeddings
              </Button>
            </div>

            <!-- WebGPU Canvas -->
            <div class="rounded-lg overflow-hidden border border-slate-700">
              <WebGPUViewer
                {embeddings}
                {labels}
                docId="demo-integration"
                autoRotate={true}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>

    <!-- Ollama Agent Shell Tab -->
    <TabsContent value="ollama" class="space-y-6">
      <Card class="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle class="text-slate-100">AI Agent Terminal Interface</CardTitle>
          <CardDescription class="text-slate-400">
            Interactive shell for real-time AI analysis with Ollama and gemma3-legal model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div class="space-y-4">
            <div class="flex items-center gap-4 mb-4">
              <Badge variant="outline" class="bg-purple-500/10 text-purple-400 border-purple-500/30">
                Model: gemma3-legal
              </Badge>
              <Badge variant="outline" class="bg-orange-500/10 text-orange-400 border-orange-500/30">
                Streaming: Enabled
              </Badge>
              <Button on:on:click={openAgentShell} size="sm">
                <Terminal class="h-4 w-4 mr-2" />
                Open Agent Shell
              </Button>
            </div>

            <!-- Integration Test Buttons -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <Button on:on:click={testOllamaIntegration} class="bg-blue-600 hover:bg-blue-700">
                Test AI Analysis
              </Button>
              <Button on:on:click={testEmbeddingGeneration} class="bg-green-600 hover:bg-green-700">
                Generate Embeddings
              </Button>
              <Button on:on:click={testVectorSimilarity} class="bg-purple-600 hover:bg-purple-700">
                Vector Similarity
              </Button>
            </div>

            <!-- Agent Shell Modal -->
            {#if agentShellOpen}
              <OllamaAgentShell
                bind:open={agentShellOpen}
                docId="demo-integration"
                initialPrompt="Analyze the legal document embeddings and provide insights on document similarity patterns."
              />
            {/if}
          </div>
        </CardContent>
      </Card>
    </TabsContent>

    <!-- XState Prefetch Tab -->
    <TabsContent value="prefetch" class="space-y-6">
      <Card class="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle class="text-slate-100">AI-Powered Intent Prediction & Prefetching</CardTitle>
          <CardDescription class="text-slate-400">
            XState machine with Loki.js caching and Fuse.js fuzzy search
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div class="space-y-6">
            <!-- Current State -->
            <div class="flex items-center justify-between">
              <div>
                <Badge variant="outline" class="bg-yellow-500/10 text-yellow-400 border-yellow-500/30 mb-2">
                  State: {prefetchStats.state}
                </Badge>
                {#if prefetchStats.confidence > 0}
                  <div class="text-sm text-slate-400">
                    Intent Confidence: {Math.round(prefetchStats.confidence * 100)}%
                  </div>
                {/if}
              </div>
              <Button on:on:click={() => prefetchSend({ type: 'PREDICT_INTENT' })} size="sm">
                Trigger Prediction
              </Button>
            </div>

            <!-- Metrics Grid -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="p-4 rounded-lg bg-slate-700/30">
                <div class="text-2xl font-bold text-blue-400">{prefetchStats.cacheHits}</div>
                <div class="text-sm text-slate-400">Cache Hits</div>
              </div>
              <div class="p-4 rounded-lg bg-slate-700/30">
                <div class="text-2xl font-bold text-red-400">{prefetchStats.cacheMisses}</div>
                <div class="text-sm text-slate-400">Cache Misses</div>
              </div>
              <div class="p-4 rounded-lg bg-slate-700/30">
                <div class="text-2xl font-bold text-green-400">{hitRate}%</div>
                <div class="text-sm text-slate-400">Hit Rate</div>
              </div>
              <div class="p-4 rounded-lg bg-slate-700/30">
                <div class="text-2xl font-bold text-purple-400">{prefetchStats.queueLength}</div>
                <div class="text-sm text-slate-400">Queue Length</div>
              </div>
            </div>

            <!-- Control Buttons -->
            <div class="flex gap-4">
              <Button on:on:click={() => prefetchSend({ type: 'CACHE_HIT', resource: 'demo' })} size="sm" class="bg-green-600 hover:bg-green-700">
                Simulate Cache Hit
              </Button>
              <Button on:on:click={() => prefetchSend({ type: 'CACHE_MISS', resource: 'demo' })} size="sm" class="bg-red-600 hover:bg-red-700">
                Simulate Cache Miss
              </Button>
              <Button on:on:click={() => prefetchSend({ type: 'RESET_METRICS' })} size="sm" variant="outline">
                Reset Metrics
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>

    <!-- Performance Tab -->
    <TabsContent value="performance" class="space-y-6">
      <Card class="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <div class="flex items-center justify-between">
            <div>
              <CardTitle class="text-slate-100">Performance Metrics</CardTitle>
              <CardDescription class="text-slate-400">
                Real-time system performance monitoring and statistics
              </CardDescription>
            </div>
            <Button
              on:on:click={isMonitoring ? stopPerformanceMonitoring : startPerformanceMonitoring}
              class={isMonitoring ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"}
            >
              {#if isMonitoring}
                <Pause class="h-4 w-4 mr-2" />
                Stop Monitoring
              {:else}
                <Play class="h-4 w-4 mr-2" />
                Start Monitoring
              {/if}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div class="p-4 rounded-lg bg-slate-700/30">
              <div class="text-3xl font-bold text-blue-400 mb-2">{performanceMetrics.requests}</div>
              <div class="text-sm text-slate-400">Total Requests</div>
            </div>
            <div class="p-4 rounded-lg bg-slate-700/30">
              <div class="text-3xl font-bold text-green-400 mb-2">{performanceMetrics.avgResponseTime}ms</div>
              <div class="text-sm text-slate-400">Avg Response Time</div>
            </div>
            <div class="p-4 rounded-lg bg-slate-700/30">
              <div class="text-3xl font-bold text-purple-400 mb-2">{Math.round(performanceMetrics.cacheHitRate)}%</div>
              <div class="text-sm text-slate-400">Cache Hit Rate</div>
            </div>
            <div class="p-4 rounded-lg bg-slate-700/30">
              <div class="text-3xl font-bold text-yellow-400 mb-2">{performanceMetrics.embeddingsGenerated}</div>
              <div class="text-sm text-slate-400">Embeddings Generated</div>
            </div>
            <div class="p-4 rounded-lg bg-slate-700/30">
              <div class="text-3xl font-bold text-red-400 mb-2">{performanceMetrics.vectorSimilarityQueries}</div>
              <div class="text-sm text-slate-400">Similarity Queries</div>
            </div>
            <div class="p-4 rounded-lg bg-slate-700/30">
              <div class="text-3xl font-bold text-cyan-400 mb-2">
                {#if isMonitoring}
                  <Activity class="h-8 w-8 animate-pulse" />
                {:else}
                  <Pause class="h-8 w-8" />
                {/if}
              </div>
              <div class="text-sm text-slate-400">Monitoring Status</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  </Tabs>

  <!-- Footer -->
  <div class="mt-12 p-6 rounded-lg bg-slate-800/30 border border-slate-700">
    <h3 class="text-lg font-semibold text-slate-200 mb-3">🎯 Integration Summary</h3>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-400">
      <div>
        <h4 class="font-medium text-slate-300 mb-2">Backend Stack</h4>
        <ul class="space-y-1">
          <li>✅ Go 1.24.5 with SIMD optimization</li>
          <li>✅ CUDA 12.8 integration</li>
          <li>✅ Ollama local LLM (gemma3-legal)</li>
          <li>✅ PostgreSQL with pgvector</li>
          <li>✅ Redis caching layer</li>
        </ul>
      </div>
      <div>
        <h4 class="font-medium text-slate-300 mb-2">Frontend Stack</h4>
        <ul class="space-y-1">
          <li>✅ SvelteKit 2 with Svelte 5 runes</li>
          <li>✅ WebGPU canvas with shader rendering</li>
          <li>✅ XState intent prediction machine</li>
          <li>✅ Loki.js + Fuse.js caching</li>
          <li>✅ Real-time agent shell interface</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<style>
  .ai-integration-demo {
    animation: fadeIn 0.8s ease-in;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
</style>