<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import LegalDocumentGraphViewer from '$lib/components/visualization/LegalDocumentGraphViewer.svelte';
  import { legalDB } from '$lib/db/client-db.js';

  // Demo state management
  const demoStats = writable({
    totalCacheHits: 0,
    totalServerFetches: 0,
    averageCacheTime: 0,
    averageServerTime: 0,
    cacheSizeKB: 0,
    totalInteractions: 0
  });

  const systemStatus = writable({
    indexedDBReady: false,
    webGPUSupported: false,
    graphInitialized: false,
    apiEndpointsActive: false
  });

  let graphViewer: LegalDocumentGraphViewer;
let showPerformanceMetrics = $state(true);
let showArchitectureOverview = $state(true);

  onMount(async () => {
    await initializeDemo();
    await checkSystemStatus();
  });

  async function initializeDemo() {
    try {
      // Initialize some sample data in IndexedDB for demonstration
      await populateSampleData();
      console.log('✅ Demo initialization complete');
    } catch (error) {
      console.error('Demo initialization failed:', error);
    }
  }

  async function checkSystemStatus() {
    try {
      // Check IndexedDB
      const testDoc = await legalDB.documentCache.limit(1).toArray();
      systemStatus.update(s => ({ ...s, indexedDBReady: true }));

      // Check WebGPU support
      const webGPUSupported = 'gpu' in navigator;
      systemStatus.update(s => ({ ...s, webGPUSupported }));

      // Check API endpoints
      try {
        const response = await fetch('/api/document/test-doc');
        systemStatus.update(s => ({ ...s, apiEndpointsActive: response.status !== 404 }));
      } catch {
        systemStatus.update(s => ({ ...s, apiEndpointsActive: false }));
      }

    } catch (error) {
      console.warn('System status check failed:', error);
    }
  }

  async function populateSampleData() {
    // Sample legal documents for cache demonstration
    const sampleDocs = [
      {
        id: Date.now(),
        documentId: 'doc-uuid-12345',
        title: 'Commercial Lease Agreement - Indemnification Clause Analysis',
        content: 'This commercial lease agreement contains comprehensive indemnification clauses that protect both lessor and lessee from various liability scenarios. The indemnification provision specifically addresses environmental hazards, third-party claims, and property damage situations...',
        documentType: 'contract',
        metadata: {
          cached_at: new Date(),
          sample_document: true,
          vector_embedding: true,
          related_count: 8
        },
        hash: 'hash_sample_12345',
        lastAccessed: new Date(),
        cacheSize: 2456
      },
      {
        id: Date.now() + 1,
        documentId: 'doc-uuid-67890',
        title: 'Legal Precedent: Liability Limitation in Contract Law',
        content: 'Supreme Court ruling establishing precedent for liability limitation clauses in commercial contracts. The court determined that indemnification provisions must be specific and clearly delineated to be enforceable...',
        documentType: 'precedent',
        metadata: {
          cached_at: new Date(),
          sample_document: true,
          vector_embedding: true,
          related_count: 12
        },
        hash: 'hash_sample_67890',
        lastAccessed: new Date(),
        cacheSize: 3102
      },
      {
        id: Date.now() + 2,
        documentId: 'case-uuid-11111',
        title: 'Case Study: Breach of Contract with Indemnification Claims',
        content: 'Comprehensive case analysis involving breach of contract where indemnification clauses were invoked. The case demonstrates the practical application of liability protection provisions in real-world scenarios...',
        documentType: 'case',
        metadata: {
          cached_at: new Date(),
          sample_document: true,
          vector_embedding: true,
          related_count: 6
        },
        hash: 'hash_sample_11111',
        lastAccessed: new Date(),
        cacheSize: 1892
      }
    ];

    // Populate IndexedDB cache
    for (const doc of sampleDocs) {
      try {
        await legalDB.documentCache.put(doc);
      } catch (error) {
        console.warn('Failed to cache sample document:', error);
      }
    }

    console.log(`📦 Populated ${sampleDocs.length} sample documents in IndexedDB cache`);
  }

  async function clearCache() {
    try {
      await legalDB.documentCache.clear();
      demoStats.update(stats => ({
        ...stats,
        totalCacheHits: 0,
        totalServerFetches: 0,
        cacheSizeKB: 0,
        totalInteractions: 0
      }));
      console.log('🗑️ Cache cleared');
    } catch (error) {
      console.error('Failed to clear cache:', error);
    }
  }

  async function refreshSystemStatus() {
    await checkSystemStatus();

    // Update cache size
    try {
      const allDocs = await legalDB.documentCache.toArray();
      const totalSize = allDocs.reduce((sum, doc) => sum + (doc.cacheSize || 0), 0);
      demoStats.update(stats => ({
        ...stats,
        cacheSizeKB: Math.round(totalSize / 1024)
      }));
    } catch (error) {
      console.warn('Failed to calculate cache size:', error);
    }
  }

  function simulateNodeClick(nodeId: string) {
    if (graphViewer) {
      // Simulate clicking on a specific node
      graphViewer.focusOnNode(nodeId);
      console.log(`🎯 Simulated click on node: ${nodeId}`);
    }
  }
</script>

<svelte:head>
  <title>Hybrid Cache-First Architecture Demo - Legal AI Platform</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 py-8">
  <div class="container mx-auto px-4">

    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">
        🔄 Hybrid Cache-First Architecture Demo
      </h1>
      <p class="text-xl text-gray-600 mb-6">
        Interactive demonstration of the "Fast Path / Slow Path" document retrieval system
      </p>

      <!-- System Status -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div class="bg-white rounded-lg p-4 text-center border border-gray-200">
          <div class="text-2xl mb-2">
            {$systemStatus.indexedDBReady ? '✅' : '❌'}
          </div>
          <div class="text-sm font-medium">IndexedDB</div>
        </div>

        <div class="bg-white rounded-lg p-4 text-center border border-gray-200">
          <div class="text-2xl mb-2">
            {$systemStatus.webGPUSupported ? '✅' : '⚠️'}
          </div>
          <div class="text-sm font-medium">WebGPU</div>
        </div>

        <div class="bg-white rounded-lg p-4 text-center border border-gray-200">
          <div class="text-2xl mb-2">
            {$systemStatus.graphInitialized ? '✅' : '🔄'}
          </div>
          <div class="text-sm font-medium">Graph Engine</div>
        </div>

        <div class="bg-white rounded-lg p-4 text-center border border-gray-200">
          <div class="text-2xl mb-2">
            {$systemStatus.apiEndpointsActive ? '✅' : '❌'}
          </div>
          <div class="text-sm font-medium">API Endpoints</div>
        </div>
      </div>
    </div>

    <!-- Architecture Overview -->
    {#if showArchitectureOverview}
      <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-2xl font-semibold">🏗️ Architecture Overview</h2>
          <button
            onclick={() => showArchitectureOverview = false}
            class="text-gray-400 hover:text-gray-600"
          >
            ×
          </button>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Fast Path -->
          <div class="bg-green-50 rounded-lg p-4 border border-green-200">
            <h3 class="text-lg font-semibold text-green-800 mb-3">
              ⚡ Fast Path (IndexedDB Cache)
            </h3>
            <ul class="text-sm text-green-700 space-y-2">
              <li>• &lt;5ms response time</li>
              <li>• Immediate UI updates</li>
              <li>• Offline-first capability</li>
              <li>• 9 reactive tables</li>
              <li>• Intelligent cleanup</li>
            </ul>
          </div>

          <!-- Slow Path -->
          <div class="bg-blue-50 rounded-lg p-4 border border-blue-200">
            <h3 class="text-lg font-semibold text-blue-800 mb-3">
              🌐 Slow Path (Server APIs)
            </h3>
            <ul class="text-sm text-blue-700 space-y-2">
              <li>• PostgreSQL + pgvector</li>
              <li>• Neo4j graph traversal</li>
              <li>• GPU-accelerated search</li>
              <li>• ~50ms comprehensive analysis</li>
              <li>• Automatic caching</li>
            </ul>
          </div>

          <!-- Visualization -->
          <div class="bg-purple-50 rounded-lg p-4 border border-purple-200">
            <h3 class="text-lg font-semibold text-purple-800 mb-3">
              🎨 WebGPU Visualization
            </h3>
            <ul class="text-sm text-purple-700 space-y-2">
              <li>• Real-time 60+ FPS rendering</li>
              <li>• Interactive node selection</li>
              <li>• GPU-based ray casting</li>
              <li>• Smooth camera animations</li>
              <li>• Relationship highlighting</li>
            </ul>
          </div>
        </div>

        <!-- Flow Diagram -->
        <div class="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 class="font-semibold text-gray-800 mb-3">Interaction Flow:</h4>
          <div class="text-sm text-gray-600 font-mono">
            User Click → Node Detection → Cache Check →
            {#if Math.random() > 0.5}
              <span class="text-green-600 font-semibold">CACHE HIT (&lt;5ms)</span> → UI Update
            {:else}
              <span class="text-blue-600 font-semibold">CACHE MISS</span> → Server Fetch (~50ms) → Cache Store → UI Update
            {/if}
          </div>
        </div>
      </div>
    {/if}

    <!-- Performance Metrics -->
    {#if showPerformanceMetrics}
      <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-2xl font-semibold">📊 Performance Metrics</h2>
          <div class="space-x-2">
            <button
              onclick={refreshSystemStatus}
              class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600"
            >
              🔄 Refresh
            </button>
            <button
              onclick={clearCache}
              class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600"
            >
              🗑️ Clear Cache
            </button>
            <button
              onclick={() => showPerformanceMetrics = false}
              class="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div class="text-center">
            <div class="text-2xl font-bold text-green-600">{$demoStats.totalCacheHits}</div>
            <div class="text-xs text-gray-600">Cache Hits</div>
          </div>

          <div class="text-center">
            <div class="text-2xl font-bold text-blue-600">{$demoStats.totalServerFetches}</div>
            <div class="text-xs text-gray-600">Server Fetches</div>
          </div>

          <div class="text-center">
            <div class="text-2xl font-bold text-purple-600">
              {$demoStats.averageCacheTime.toFixed(1)}ms
            </div>
            <div class="text-xs text-gray-600">Avg Cache Time</div>
          </div>

          <div class="text-center">
            <div class="text-2xl font-bold text-orange-600">
              {$demoStats.averageServerTime.toFixed(1)}ms
            </div>
            <div class="text-xs text-gray-600">Avg Server Time</div>
          </div>

          <div class="text-center">
            <div class="text-2xl font-bold text-gray-600">{$demoStats.cacheSizeKB}KB</div>
            <div class="text-xs text-gray-600">Cache Size</div>
          </div>

          <div class="text-center">
            <div class="text-2xl font-bold text-indigo-600">{$demoStats.totalInteractions}</div>
            <div class="text-xs text-gray-600">Total Clicks</div>
          </div>
        </div>

        <!-- Cache Hit Rate Visualization -->
        <div class="mt-6">
          <h4 class="font-medium text-gray-800 mb-2">Cache Hit Rate</h4>
          <div class="w-full bg-gray-200 rounded-full h-3">
            <div
              class="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all duration-300"
              style="width: {$demoStats.totalCacheHits + $demoStats.totalServerFetches > 0
                ? ($demoStats.totalCacheHits / ($demoStats.totalCacheHits + $demoStats.totalServerFetches)) * 100
                : 0}%"
            ></div>
          </div>
          <div class="text-sm text-gray-600 mt-1">
            {$demoStats.totalCacheHits + $demoStats.totalServerFetches > 0
              ? (($demoStats.totalCacheHits / ($demoStats.totalCacheHits + $demoStats.totalServerFetches)) * 100).toFixed(1)
              : 0}%
            cache hit rate
          </div>
        </div>
      </div>
    {/if}

    <!-- Interactive Demo Controls -->
    <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
      <h2 class="text-2xl font-semibold mb-4">🎮 Interactive Demo Controls</h2>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onclick={() => simulateNodeClick('doc-uuid-12345')}
          class="bg-green-500 text-white py-3 px-4 rounded-lg hover:bg-green-600 transition-colors"
        >
          📄 Click Contract Document
          <div class="text-sm text-green-100 mt-1">Should hit cache (fast)</div>
        </button>

        <button
          onclick={() => simulateNodeClick('precedent-uuid-22222')}
          class="bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 transition-colors"
        >
          ⚖️ Click Legal Precedent
          <div class="text-sm text-blue-100 mt-1">May need server fetch (slow)</div>
        </button>

        <button
          onclick={() => simulateNodeClick('case-uuid-' + Date.now())}
          class="bg-purple-500 text-white py-3 px-4 rounded-lg hover:bg-purple-600 transition-colors"
        >
          🔍 Click Random Node
          <div class="text-sm text-purple-100 mt-1">Demonstrates cache miss</div>
        </button>
      </div>

      <div class="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 class="font-medium text-gray-800 mb-2">How to Test:</h4>
        <ol class="text-sm text-gray-600 space-y-1">
          <li>1. Click "Contract Document" - Should load instantly from cache</li>
          <li>2. Click "Legal Precedent" - May trigger server fetch if not cached</li>
          <li>3. Click "Random Node" - Will definitely trigger server fetch</li>
          <li>4. Watch the performance metrics update in real-time</li>
          <li>5. Try clicking the same node twice to see cache benefit</li>
        </ol>
      </div>
    </div>

    <!-- WebGPU Graph Visualization -->
    <div class="bg-white rounded-lg shadow-lg overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200">
        <h2 class="text-2xl font-semibold">🌐 Interactive Legal Document Network</h2>
        <p class="text-gray-600 mt-2">
          Click on nodes to trigger the hybrid cache-first architecture.
          Watch the console and metrics for real-time performance data.
        </p>
      </div>

      <div class="p-6">
        <div class="border border-gray-200 rounded-lg overflow-hidden">
          <LegalDocumentGraphViewer
            bind:this={graphViewer}
            width={1000}
            height={600}
            enablePhysics={true}
            enableStreaming={true}
            maxNodes={1000}
            on:nodeInteraction={(event) => {
              // Track interaction metrics
              demoStats.update(stats => ({
                ...stats,
                totalInteractions: stats.totalInteractions + 1
              }));

              systemStatus.update(s => ({ ...s, graphInitialized: true }));
            }}
            on:cacheHit={(event) => {
              demoStats.update(stats => ({
                ...stats,
                totalCacheHits: stats.totalCacheHits + 1,
                averageCacheTime: (stats.averageCacheTime + event.detail.timing) / 2
              }));
            }}
            on:serverFetch={(event) => {
              demoStats.update(stats => ({
                ...stats,
                totalServerFetches: stats.totalServerFetches + 1,
                averageServerTime: (stats.averageServerTime + event.detail.timing) / 2
              }));
            }}
          />
        </div>
      </div>
    </div>

    <!-- Technical Implementation Details -->
    <div class="mt-8 bg-gray-900 text-gray-100 rounded-lg p-6">
      <h3 class="text-xl font-semibold mb-4">💻 Technical Implementation</h3>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
        <div>
          <h4 class="font-semibold text-gray-300 mb-2">Fast Path Implementation:</h4>
          <pre class="text-xs text-gray-300 bg-gray-800 p-3 rounded overflow-x-auto"><code>// Cache-first node click handler
async function onNodeClick(docId: string) &#123;
  let document = await db.documents.get(docId);

  if (document) &#123;
    // ✅ CACHE HIT! (&lt;5ms)
    displayDocumentDetails(document);
  &#125; else &#123;
    // ❌ CACHE MISS! Fetch from server
    await fetchAndCacheDocument(docId);
  &#125;
&#125;</code></pre>
        </div>

        <div>
          <h4 class="font-semibold text-gray-300 mb-2">Server Integration:</h4>
          <pre class="text-xs text-gray-300 bg-gray-800 p-3 rounded overflow-x-auto"><code>// /api/document/[id]/+server.ts
export async function GET(&#123; params &#125;) &#123;
  // 1. PostgreSQL + pgvector similarity
  // 2. Neo4j graph relationships
  // 3. GPU-accelerated analysis
  // 4. Comprehensive metadata

  return json(&#123;
    document, relatedDocuments,
    graphConnections, caseAssociations
  &#125;);
&#125;</code></pre>
        </div>
      </div>
    </div>

  </div>
</div>