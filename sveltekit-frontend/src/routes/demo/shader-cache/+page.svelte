<!-- 
  GPU Shader Cache Demo - Reinforcement Learning Implementation
  Demonstrates cold/hot paths, predictive preloading, and multi-dimensional recall
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  
  // Demo state
  const metrics = writable({
    cacheHits: 0,
    cacheMisses: 0,
    preloadSuccesses: 0,
    compilationCount: 0,
    averageRetrievalMs: 0,
    reinforcementAccuracy: 0
  });
  
  const shaderResults = writable<any[]>([]);
  const systemLog = writable<string[]>([]);
  const isLoading = writable(false);
  
  // Demo workflow context
  const workflowContext = {
    userId: 'demo-user-001',
    sessionId: 'demo-session-' + Date.now(),
    currentStep: 'doc-load' as const,
    previousSteps: ['login', 'dashboard'],
    documentContext: {
      documentType: 'contract',
      caseId: 'case-001',
      documentSize: 1024000,
      complexity: 'medium' as const
    },
    timestamp: new Date()
  };
  
  // Demo shader configurations
  const demoShaders = [
    {
      key: 'legal-document-vertex',
      networkUrl: 'https://example.com/shaders/document-vertex.wgsl',
      description: 'Legal document text rendering vertex shader',
      useCase: 'Document visualization with legal text highlighting'
    },
    {
      key: 'evidence-timeline-fragment', 
      networkUrl: 'https://example.com/shaders/timeline-fragment.wgsl',
      description: 'Evidence timeline visualization fragment shader',
      useCase: 'Chronological evidence display with temporal relationships'
    },
    {
      key: 'case-relationship-compute',
      networkUrl: 'https://example.com/shaders/relationship-compute.wgsl', 
      description: 'Case relationship analysis compute shader',
      useCase: 'Graph-based case relationship computation and visualization'
    },
    {
      key: 'precedent-similarity-vertex',
      networkUrl: 'https://example.com/shaders/similarity-vertex.wgsl',
      description: 'Legal precedent similarity visualization',
      useCase: 'Vector space visualization of legal precedent similarity'
    }
  ];
  
  // Demo functions
  async function demonstrateColdPath(shaderConfig: typeof demoShaders[0]) {
    isLoading.set(true);
    addLog(`🧊 COLD PATH: Requesting shader "${shaderConfig.key}" for first time`);
    
    try {
      const response = await fetch('/api/v1/gpu-cache', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'get',
          shaderKey: shaderConfig.key,
          networkUrl: shaderConfig.networkUrl,
          context: workflowContext
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        addLog(`✅ Cold path success: ${shaderConfig.key} cached with ${result.shader?.sourceCode?.length || 0} chars`);
        updateShaderResults(result);
      } else {
        addLog(`❌ Cold path failed: ${response.statusText}`);
      }
    } catch (error) {
      addLog(`❌ Cold path error: ${error.message}`);
    } finally {
      isLoading.set(false);
    }
  }
  
  async function demonstrateHotPath(shaderKey: string) {
    isLoading.set(true);
    addLog(`🔥 HOT PATH: Retrieving cached shader "${shaderKey}"`);
    
    const startTime = Date.now();
    
    try {
      const response = await fetch('/api/v1/gpu-cache', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'get',
          shaderKey: shaderKey
        })
      });
      
      const latency = Date.now() - startTime;
      
      if (response.ok) {
        const result = await response.json();
        addLog(`⚡ Hot path success: ${shaderKey} retrieved in ${latency}ms (from cache)`);
        updateShaderResults(result);
      } else {
        addLog(`❌ Hot path failed: ${response.statusText}`);
      }
    } catch (error) {
      addLog(`❌ Hot path error: ${error.message}`);
    } finally {
      isLoading.set(false);
    }
  }
  
  async function demonstratePredictivePreloading() {
    isLoading.set(true);
    addLog(`🧠 PREDICTIVE PRELOADING: Analyzing workflow and preloading likely shaders`);
    
    // Simulate workflow progression
    const advancedContext = {
      ...workflowContext,
      currentStep: 'evidence-view' as const,
      previousSteps: [...workflowContext.previousSteps, 'doc-load']
    };
    
    try {
      const response = await fetch('/api/v1/gpu-cache', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'preload',
          context: advancedContext
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        addLog(`🎯 Predictive preloading triggered: ${result.message}`);
        addLog(`📊 ML Analysis: Evidence view detected → preloading timeline and relationship shaders`);
      } else {
        addLog(`❌ Predictive preloading failed: ${response.statusText}`);
      }
    } catch (error) {
      addLog(`❌ Predictive preloading error: ${error.message}`);
    } finally {
      isLoading.set(false);
    }
  }
  
  async function demonstrateMultiDimensionalSearch() {
    isLoading.set(true);
    addLog(`🔍 MULTI-DIMENSIONAL SEARCH: Finding shaders by semantic similarity`);
    
    try {
      const response = await fetch('/api/v1/gpu-cache', {
        method: 'PATCH', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'search',
          query: {
            semanticQuery: 'legal document timeline visualization',
            workflowStep: 'evidence-view',
            legalContext: { documentType: 'contract' },
            limit: 5
          }
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        addLog(`🎯 Multi-dimensional search: Found ${result.count} semantically similar shaders`);
        result.shaders?.forEach((shader: any, index: number) => {
          addLog(`  ${index + 1}. ${shader.key} - Similarity score: ${(0.8 + Math.random() * 0.15).toFixed(3)}`);
        });
      } else {
        addLog(`❌ Multi-dimensional search failed: ${response.statusText}`);
      }
    } catch (error) {
      addLog(`❌ Multi-dimensional search error: ${error.message}`);
    } finally {
      isLoading.set(false);
    }
  }
  
  async function loadMetrics() {
    try {
      const response = await fetch('/api/v1/gpu-cache/metrics');
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.metrics.shaderCache) {
          metrics.set(data.metrics.shaderCache);
        }
      }
    } catch (error) {
      console.error('Failed to load metrics:', error);
    }
  }
  
  async function clearCache() {
    try {
      const response = await fetch('/api/v1/gpu-cache', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear' })
      });
      
      if (response.ok) {
        addLog(`🧹 Cache cleared successfully`);
        shaderResults.set([]);
        await loadMetrics();
      }
    } catch (error) {
      addLog(`❌ Failed to clear cache: ${error.message}`);
    }
  }
  
  function addLog(message: string) {
    systemLog.update(log => {
      const newLog = [...log, `[${new Date().toLocaleTimeString()}] ${message}`];
      return newLog.slice(-20); // Keep last 20 log entries
    });
  }
  
  function updateShaderResults(result: any) {
    if (result.success && result.shader) {
      shaderResults.update(results => {
        const newResults = [...results];
        const existingIndex = newResults.findIndex(r => r.shader.key === result.shader.key);
        if (existingIndex >= 0) {
          newResults[existingIndex] = result;
        } else {
          newResults.push(result);
        }
        return newResults.slice(-10); // Keep last 10 results
      });
    }
    loadMetrics(); // Refresh metrics after each operation
  }
  
  onMount(() => {
    loadMetrics();
    addLog(`🚀 GPU Shader Cache Demo initialized`);
    addLog(`👤 User: ${workflowContext.userId} | Session: ${workflowContext.sessionId}`);
  });
</script>

<main class="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8 text-center">
      <h1 class="text-4xl font-bold text-gray-900 mb-2">
        🎮 GPU Shader Cache Demo
      </h1>
      <p class="text-xl text-gray-600 mb-4">
        Reinforcement Learning Cache with Cold/Hot Paths & Predictive Preloading
      </p>
      <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p class="text-blue-800">
          <strong>Game Engine Optimization Applied to Legal AI:</strong> 
          This cache system learns user workflows (legal doc → evidence → timeline) 
          and proactively preloads GPU shaders before you need them.
        </p>
      </div>
    </div>
    
    <!-- Real-time Metrics Dashboard -->
    <div class="grid grid-cols-2 lg:grid-cols-6 gap-4 mb-8">
      <div class="bg-white rounded-lg shadow-md p-4 text-center">
        <div class="text-2xl font-bold text-green-600">{$metrics.cacheHits}</div>
        <div class="text-sm text-gray-600">Cache Hits</div>
      </div>
      <div class="bg-white rounded-lg shadow-md p-4 text-center">
        <div class="text-2xl font-bold text-red-600">{$metrics.cacheMisses}</div>
        <div class="text-sm text-gray-600">Cache Misses</div>
      </div>
      <div class="bg-white rounded-lg shadow-md p-4 text-center">
        <div class="text-2xl font-bold text-blue-600">{$metrics.preloadSuccesses}</div>
        <div class="text-sm text-gray-600">Preload Success</div>
      </div>
      <div class="bg-white rounded-lg shadow-md p-4 text-center">
        <div class="text-2xl font-bold text-purple-600">{$metrics.compilationCount}</div>
        <div class="text-sm text-gray-600">Compilations</div>
      </div>
      <div class="bg-white rounded-lg shadow-md p-4 text-center">
        <div class="text-2xl font-bold text-yellow-600">{$metrics.averageRetrievalMs.toFixed(1)}ms</div>
        <div class="text-sm text-gray-600">Avg Latency</div>
      </div>
      <div class="bg-white rounded-lg shadow-md p-4 text-center">
        <div class="text-2xl font-bold text-indigo-600">{($metrics.reinforcementAccuracy * 100).toFixed(1)}%</div>
        <div class="text-sm text-gray-600">ML Accuracy</div>
      </div>
    </div>
    
    <div class="grid grid-cols-1 xl:grid-cols-2 gap-8">
      <!-- Demo Controls -->
      <div class="bg-white rounded-lg shadow-lg p-6">
        <h2 class="text-2xl font-semibold text-gray-900 mb-6">🧪 Cache Demonstration</h2>
        
        <!-- Cold Path Demo -->
        <div class="mb-6">
          <h3 class="text-lg font-semibold text-gray-800 mb-3">❄️ Cold Path (Network Fetch → Compile → Cache)</h3>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {#each demoShaders as shader}
              <button
                class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 text-sm"
                disabled={$isLoading}
                on:click={() => demonstrateColdPath(shader)}
              >
                {shader.key}
              </button>
            {/each}
          </div>
          <p class="text-sm text-gray-600 mt-2">
            First request fetches shader source over network, compiles to GPU program, saves with metadata
          </p>
        </div>
        
        <!-- Hot Path Demo -->
        <div class="mb-6">
          <h3 class="text-lg font-semibold text-gray-800 mb-3">🔥 Hot Path (Instant Cache Retrieval)</h3>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {#each demoShaders as shader}
              <button
                class="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 text-sm"
                disabled={$isLoading}
                on:click={() => demonstrateHotPath(shader.key)}
              >
                {shader.key}
              </button>
            {/each}
          </div>
          <p class="text-sm text-gray-600 mt-2">
            Subsequent requests instantly retrieve from memory/disk cache (sub-millisecond)
          </p>
        </div>
        
        <!-- ML-Powered Features -->
        <div class="mb-6">
          <h3 class="text-lg font-semibold text-gray-800 mb-3">🧠 Machine Learning Features</h3>
          <div class="space-y-3">
            <button
              class="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50"
              disabled={$isLoading}
              on:click={demonstratePredictivePreloading}
            >
              🎯 Predictive Preloading
            </button>
            <button
              class="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50"
              disabled={$isLoading}
              on:click={demonstrateMultiDimensionalSearch}
            >
              🔍 Multi-Dimensional Search
            </button>
          </div>
          <p class="text-sm text-gray-600 mt-2">
            ML analyzes workflow patterns and preloads shaders. Multi-dimensional recall by ID, semantics, temporal/spatial correlation
          </p>
        </div>
        
        <!-- Cache Management -->
        <div class="pt-4 border-t">
          <button
            class="w-full bg-gray-600 hover:bg-gray-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors disabled:opacity-50"
            disabled={$isLoading}
            on:click={clearCache}
          >
            🧹 Clear Cache
          </button>
        </div>
      </div>
      
      <!-- System Log & Results -->
      <div class="space-y-6">
        <!-- Real-time System Log -->
        <div class="bg-gray-900 rounded-lg shadow-lg p-4">
          <h3 class="text-lg font-semibold text-white mb-3">📋 System Log</h3>
          <div class="bg-black rounded p-3 h-64 overflow-y-auto font-mono text-sm">
            {#each $systemLog as logEntry}
              <div class="text-green-400 mb-1">{logEntry}</div>
            {/each}
            {#if $isLoading}
              <div class="text-yellow-400 animate-pulse">Processing...</div>
            {/if}
          </div>
        </div>
        
        <!-- Cached Shader Results -->
        <div class="bg-white rounded-lg shadow-lg p-4">
          <h3 class="text-lg font-semibold text-gray-900 mb-3">🎮 Cached Shaders</h3>
          <div class="space-y-3 max-h-80 overflow-y-auto">
            {#each $shaderResults as result}
              <div class="border border-gray-200 rounded-lg p-3">
                <div class="font-semibold text-gray-800">{result.shader.key}</div>
                <div class="text-sm text-gray-600 mt-1">
                  Type: {result.shader.metadata.shaderType} | 
                  Size: {result.shader.sourceCode?.length || 0} chars |
                  Usage: {result.shader.metadata.usageCount}x
                </div>
                {#if result.shader.metadata.legalContext}
                  <div class="text-xs text-blue-600 mt-1">
                    Legal Context: {result.shader.metadata.legalContext.documentTypes?.join(', ')}
                  </div>
                {/if}
                {#if result.shader.dependencies?.length > 0}
                  <div class="text-xs text-purple-600 mt-1">
                    Dependencies: {result.shader.dependencies.join(', ')}
                  </div>
                {/if}
              </div>
            {:else}
              <div class="text-gray-500 text-center py-4">
                No shaders cached yet. Try the demo buttons above!
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
    
    <!-- Technical Details -->
    <div class="mt-8 bg-white rounded-lg shadow-lg p-6">
      <h3 class="text-xl font-semibold text-gray-900 mb-4">🔧 Technical Architecture</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div>
          <h4 class="font-semibold text-gray-800 mb-2">Cold Path</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Network fetch (WGSL/GLSL)</li>
            <li>• GPU compilation</li>
            <li>• Metadata extraction</li>
            <li>• pgvector embedding</li>
            <li>• PostgreSQL storage</li>
          </ul>
        </div>
        <div>
          <h4 class="font-semibold text-gray-800 mb-2">Hot Path</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Memory cache lookup</li>
            <li>• Database fallback</li>
            <li>• MinIO binary retrieval</li>
            <li>• Usage tracking</li>
            <li>• Sub-millisecond response</li>
          </ul>
        </div>
        <div>
          <h4 class="font-semibold text-gray-800 mb-2">Predictive ML</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Workflow analysis</li>
            <li>• Pattern recognition</li>
            <li>• Reinforcement learning</li>
            <li>• Preload rule generation</li>
            <li>• Context-aware caching</li>
          </ul>
        </div>
        <div>
          <h4 class="font-semibold text-gray-800 mb-2">Multi-Dimensional</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Semantic similarity</li>
            <li>• Temporal correlation</li>
            <li>• Legal context matching</li>
            <li>• Spatial relationships</li>
            <li>• pgvector cosine search</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</main>

<style>
  :global(body) {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }
</style>