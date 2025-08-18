<!--
  High-Performance AI Assistant Demo
  Showcases the complete blueprint implementation with real-time 3D visualization
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import Enhanced3DCanvas from '$lib/components/ai/Enhanced3DCanvas.svelte';
  import { somTopicModeler } from '$lib/services/som-topic-modeler';
  import { interactionTracker } from '$lib/services/interaction-tracker';
  import { proactiveCache } from '$lib/services/proactive-cache';
  import { multiDimensionalEncoder } from '$lib/services/multi-dimensional-encoder';
  import { webGPUProcessor } from '$lib/services/webgpu-processor';
  import type { DocumentNode } from '$lib/types/ai';

  // State
  let isInitialized = false;
  let documents: DocumentNode[] = [];
  let selectedDocument: DocumentNode | null = null;
  let aiResponse = '';
  let isProcessing = false;
  let userQuery = '';
  
  // Demo data
  let systemStatus = {
    webgpu: 'initializing',
    som: 'ready',
    cache: 'warming',
    tracking: 'active',
    encoder: 'ready'
  };

  // Statistics
  let stats = {
    documentsProcessed: 0,
    interactionsTracked: 0,
    cacheHitRate: '0%',
    somAccuracy: '0%',
    avgResponseTime: '0ms'
  };

  // Demo controls
  let enableGPU = true;
  let enableTracking = true;
  let showDebugInfo = false;
  let animationSpeed = 1.0;

  onMount(async () => {
    await initializeAIAssistant();
    await loadDemoDocuments();
    startRealTimeUpdates();
  });

  onDestroy(() => {
    stopRealTimeUpdates();
  });

  async function initializeAIAssistant() {
    console.log('🚀 Initializing High-Performance AI Assistant...');

    try {
      // Initialize core services
      systemStatus.webgpu = 'initializing';
      const webgpuReady = await webGPUProcessor.initialize();
      systemStatus.webgpu = webgpuReady ? 'ready' : 'fallback';

      systemStatus.cache = 'initializing';
      await proactiveCache.initialize();
      systemStatus.cache = 'ready';

      systemStatus.encoder = 'initializing';
      await multiDimensionalEncoder.initialize();
      systemStatus.encoder = 'ready';

      if (enableTracking) {
        interactionTracker.startTracking();
        systemStatus.tracking = 'active';
      }

      isInitialized = true;
      console.log('✅ AI Assistant initialization complete');
    } catch (error) {
      console.error('❌ Initialization failed:', error);
      systemStatus.webgpu = 'error';
    }
  }

  async function loadDemoDocuments() {
    // Generate demo documents with embeddings
    const demoTypes = ['contract', 'evidence', 'motion', 'brief', 'regulation'];
    const newDocuments: DocumentNode[] = [];

    for (let i = 0; i < 50; i++) {
      const type = demoTypes[i % demoTypes.length];
      const embedding = new Float32Array(768);
      
      // Generate realistic embeddings with clustering
      const cluster = Math.floor(i / 10);
      const baseAngle = (cluster * Math.PI * 2) / 5;
      const radius = 2 + Math.random() * 3;
      
      for (let j = 0; j < 768; j++) {
        if (j < 3) {
          // Encode position in first 3 dimensions
          embedding[j] = radius * Math.cos(baseAngle + (j * 0.1)) + Math.random() * 0.5;
        } else {
          embedding[j] = Math.random() * 2 - 1;
        }
      }

      const doc: DocumentNode = {
        id: `doc_${i}`,
        title: `Legal Document ${i + 1}`,
        type,
        content: `This is demo content for ${type} document ${i + 1}. Contains legal text and analysis.`,
        embedding,
        metadata: {
          created: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000),
          size: Math.floor(Math.random() * 50000) + 5000,
          keywords: [`keyword${i % 10}`, `legal${i % 15}`, `case${i % 8}`]
        },
        somPosition: undefined // Will be calculated by SOM
      };

      newDocuments.push(doc);
    }

    documents = newDocuments;
    stats.documentsProcessed = documents.length;

    // Train SOM on document embeddings
    console.log('🧠 Training SOM on document embeddings...');
    await somTopicModeler.trainOnDocuments(
      documents.map(doc => ({
        id: doc.id,
        embedding: doc.embedding,
        text: doc.content
      }))
    );

    // Update document positions based on SOM
    const documentMappings = somTopicModeler.getDocumentMappings();
    documentMappings.subscribe(mappings => {
      for (const doc of documents) {
        const mapping = mappings.get(doc.id);
        if (mapping) {
          doc.somPosition = mapping.bestMatchingUnit;
        }
      }
      documents = [...documents]; // Trigger reactivity
    })();

    console.log('📊 Demo documents loaded and SOM training complete');
  }

  let updateTimer: NodeJS.Timeout;

  function startRealTimeUpdates() {
    updateTimer = setInterval(updateStatistics, 1000);
  }

  function stopRealTimeUpdates() {
    if (updateTimer) {
      clearInterval(updateTimer);
    }
  }

  function updateStatistics() {
    // Update interaction tracking stats
    const interactionHistory = interactionTracker.getInteractions();
    interactionHistory.subscribe(interactions => {
      stats.interactionsTracked = interactions.length;
    })();

    // Update cache stats
    const cacheStats = proactiveCache.getStats();
    cacheStats.subscribe(cacheData => {
      stats.cacheHitRate = (cacheData.hitRate * 100).toFixed(1) + '%';
      stats.avgResponseTime = cacheData.averageLatency.toFixed(0) + 'ms';
    })();

    // Update SOM accuracy (simulated)
    const topicInsights = somTopicModeler.getTopicInsights();
    topicInsights.subscribe(insights => {
      stats.somAccuracy = (85 + Math.random() * 10).toFixed(1) + '%';
    })();
  }

  async function processUserQuery() {
    if (!userQuery.trim() || isProcessing) return;

    isProcessing = true;
    aiResponse = '';

    try {
      console.log('🎯 Processing user query:', userQuery);

      // Record interaction
      interactionTracker.recordInteraction({
        type: 'search',
        timestamp: Date.now(),
        position: { x: 0, y: 0 },
        target: 'search_input',
        metadata: { query: userQuery }
      });

      // Simulate AI processing with streaming response
      const response = await simulateAIResponse(userQuery);
      
      // Stream the response
      for (let i = 0; i < response.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 20));
        aiResponse += response[i];
      }

    } catch (error) {
      console.error('Query processing failed:', error);
      aiResponse = 'Sorry, I encountered an error processing your query.';
    } finally {
      isProcessing = false;
    }
  }

  async function simulateAIResponse(query: string): Promise<string> {
    // Simulate semantic search and AI analysis
    const relevantDocs = documents.slice(0, 3);
    
    const responses = [
      `Based on my analysis of your legal documents, I found ${relevantDocs.length} relevant items. `,
      `The documents show patterns related to "${query}" with high confidence. `,
      `Key insights include contractual obligations, regulatory compliance, and precedent analysis. `,
      `Would you like me to dive deeper into any specific aspect?`
    ];

    return responses.join('');
  }

  function handleDocumentSelected(event: CustomEvent) {
    const documentId = event.detail.documentId;
    selectedDocument = documents.find(doc => doc.id === documentId) || null;
    
    if (selectedDocument) {
      console.log('📄 Document selected:', selectedDocument.title);
      
      // Trigger proactive caching for related documents
      proactiveCache.warmupArea(
        selectedDocument.somPosition?.x || 0,
        selectedDocument.somPosition?.y || 0,
        3
      );
    }
  }

  function toggleGPUAcceleration() {
    enableGPU = !enableGPU;
    systemStatus.webgpu = enableGPU ? 'ready' : 'disabled';
  }

  function toggleInteractionTracking() {
    enableTracking = !enableTracking;
    
    if (enableTracking) {
      interactionTracker.startTracking();
      systemStatus.tracking = 'active';
    } else {
      interactionTracker.stopTracking();
      systemStatus.tracking = 'disabled';
    }
  }

  function regenerateSOM() {
    console.log('🔄 Regenerating SOM...');
    loadDemoDocuments();
  }

  function clearCache() {
    proactiveCache.clearCache();
    console.log('🗑️ Cache cleared');
  }
</script>

<svelte:head>
  <title>High-Performance AI Assistant Demo</title>
  <meta name="description" content="Advanced AI assistant with 3D visualization, SOM topic modeling, and proactive caching" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
  <!-- Header -->
  <header class="bg-black/20 backdrop-blur-sm border-b border-purple-500/30 p-4">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">High-Performance AI Assistant</h1>
        <p class="text-purple-300 text-sm">Next-generation legal document analysis with 3D visualization</p>
      </div>
      
      <div class="flex items-center gap-4">
        <!-- System Status -->
        <div class="flex gap-2">
          <div class="px-2 py-1 rounded text-xs font-mono" class:bg-green-600={systemStatus.webgpu === 'ready'} class:bg-yellow-600={systemStatus.webgpu === 'fallback'} class:bg-red-600={systemStatus.webgpu === 'error'}>
            GPU: {systemStatus.webgpu}
          </div>
          <div class="px-2 py-1 rounded text-xs font-mono bg-green-600">
            SOM: {systemStatus.som}
          </div>
          <div class="px-2 py-1 rounded text-xs font-mono" class:bg-green-600={systemStatus.cache === 'ready'} class:bg-yellow-600={systemStatus.cache === 'warming'}>
            Cache: {systemStatus.cache}
          </div>
          <div class="px-2 py-1 rounded text-xs font-mono" class:bg-green-600={systemStatus.tracking === 'active'} class:bg-gray-600={systemStatus.tracking === 'disabled'}>
            Tracking: {systemStatus.tracking}
          </div>
        </div>
        
        <button 
          class="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
          on:click={() => showDebugInfo = !showDebugInfo}
        >
          {showDebugInfo ? 'Hide' : 'Show'} Debug
        </button>
      </div>
    </div>
  </header>

  <div class="max-w-7xl mx-auto p-4 grid grid-cols-12 gap-4 h-[calc(100vh-80px)]">
    <!-- Main 3D Canvas -->
    <div class="col-span-8">
      <div class="bg-black/40 rounded-lg overflow-hidden h-full">
        <Enhanced3DCanvas 
          {documents}
          enableGPUAcceleration={enableGPU}
          enableInteractionTracking={enableTracking}
          on:documentSelected={handleDocumentSelected}
        />
      </div>
    </div>

    <!-- Sidebar -->
    <div class="col-span-4 space-y-4">
      <!-- AI Chat Interface -->
      <div class="bg-black/40 rounded-lg p-4 backdrop-blur-sm border border-purple-500/30">
        <h3 class="text-lg font-semibold text-white mb-3">AI Assistant</h3>
        
        <!-- Query Input -->
        <div class="flex gap-2 mb-4">
          <input
            type="text"
            bind:value={userQuery}
            placeholder="Ask about your legal documents..."
            class="flex-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded text-white placeholder-slate-400"
            on:keydown={(e) => e.key === 'Enter' && processUserQuery()}
          />
          <button
            class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white disabled:opacity-50"
            disabled={isProcessing}
            on:click={processUserQuery}
          >
            {isProcessing ? '...' : 'Ask'}
          </button>
        </div>

        <!-- AI Response -->
        <div class="bg-slate-800/50 rounded p-3 min-h-[120px] text-sm text-gray-200">
          {#if isProcessing}
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
              <span>AI is thinking...</span>
            </div>
          {:else if aiResponse}
            <div in:fade={{ duration: 300 }}>
              {aiResponse}
            </div>
          {:else}
            <div class="text-gray-500 italic">
              Ask me anything about your legal documents. I can analyze content, find similar cases, and provide insights.
            </div>
          {/if}
        </div>
      </div>

      <!-- Selected Document Info -->
      {#if selectedDocument}
        <div class="bg-black/40 rounded-lg p-4 backdrop-blur-sm border border-purple-500/30" in:fly={{ y: 20, duration: 300 }}>
          <h3 class="text-lg font-semibold text-white mb-2">Selected Document</h3>
          <div class="space-y-2 text-sm">
            <div><span class="text-purple-300">Title:</span> {selectedDocument.title}</div>
            <div><span class="text-purple-300">Type:</span> {selectedDocument.type}</div>
            <div><span class="text-purple-300">Size:</span> {selectedDocument.metadata?.size} bytes</div>
            <div><span class="text-purple-300">SOM Position:</span> 
              {selectedDocument.somPosition ? `(${selectedDocument.somPosition.x}, ${selectedDocument.somPosition.y})` : 'Not mapped'}
            </div>
            <div class="text-xs text-gray-400 mt-2">
              {selectedDocument.content?.substring(0, 150)}...
            </div>
          </div>
        </div>
      {/if}

      <!-- Statistics -->
      <div class="bg-black/40 rounded-lg p-4 backdrop-blur-sm border border-purple-500/30">
        <h3 class="text-lg font-semibold text-white mb-3">System Statistics</h3>
        <div class="grid grid-cols-2 gap-3 text-sm">
          <div class="bg-slate-800/50 rounded p-2">
            <div class="text-purple-300 text-xs">Documents</div>
            <div class="text-white font-semibold">{stats.documentsProcessed}</div>
          </div>
          <div class="bg-slate-800/50 rounded p-2">
            <div class="text-purple-300 text-xs">Interactions</div>
            <div class="text-white font-semibold">{stats.interactionsTracked}</div>
          </div>
          <div class="bg-slate-800/50 rounded p-2">
            <div class="text-purple-300 text-xs">Cache Hit Rate</div>
            <div class="text-white font-semibold">{stats.cacheHitRate}</div>
          </div>
          <div class="bg-slate-800/50 rounded p-2">
            <div class="text-purple-300 text-xs">SOM Accuracy</div>
            <div class="text-white font-semibold">{stats.somAccuracy}</div>
          </div>
        </div>
      </div>

      <!-- Controls -->
      <div class="bg-black/40 rounded-lg p-4 backdrop-blur-sm border border-purple-500/30">
        <h3 class="text-lg font-semibold text-white mb-3">Controls</h3>
        <div class="space-y-3">
          <label class="flex items-center gap-2 text-sm text-gray-300">
            <input 
              type="checkbox" 
              bind:checked={enableGPU}
              on:change={toggleGPUAcceleration}
              class="rounded"
            />
            GPU Acceleration
          </label>
          
          <label class="flex items-center gap-2 text-sm text-gray-300">
            <input 
              type="checkbox" 
              bind:checked={enableTracking}
              on:change={toggleInteractionTracking}
              class="rounded"
            />
            Interaction Tracking
          </label>

          <div class="flex gap-2">
            <button
              class="flex-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs"
              on:click={regenerateSOM}
            >
              Regenerate SOM
            </button>
            <button
              class="flex-1 px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs"
              on:click={clearCache}
            >
              Clear Cache
            </button>
          </div>
        </div>
      </div>

      <!-- Debug Info -->
      {#if showDebugInfo}
        <div class="bg-black/40 rounded-lg p-4 backdrop-blur-sm border border-orange-500/30" in:fade={{ duration: 200 }}>
          <h3 class="text-lg font-semibold text-orange-400 mb-3">Debug Information</h3>
          <div class="text-xs text-gray-400 font-mono space-y-1">
            <div>WebGPU Available: {navigator.gpu ? 'Yes' : 'No'}</div>
            <div>Documents Loaded: {documents.length}</div>
            <div>SOM Training: {systemStatus.som}</div>
            <div>Cache Status: {systemStatus.cache}</div>
            <div>Average Response: {stats.avgResponseTime}</div>
            <div>Memory Usage: ~{(documents.length * 3072 / 1024).toFixed(1)}KB</div>
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: 'Inter', system-ui, sans-serif;
  }
</style>