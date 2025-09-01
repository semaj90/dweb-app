<!--
  Graph Explorer Component - Unified Architecture Demo
  
  Demonstrates the complete integration:
  Neo4j → Embeddings → Quantization → WebGPU Textures → LokiJS → IndexedDB → Reactive UI
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { db, chatHistory, graphNodes } from '../db/dexie-integration';
  import { integratedSearch } from '../storage/integrated-search-engine';
  import { graphTextureManager } from '../webgpu/graph-texture-layout';
  import { unifiedDimensionalStore } from '../storage/unified-dimensional-store';
  import { vectorQuantization } from '../storage/vector-quantization';

  // ========================================================================
  // REACTIVE STATE
  // ========================================================================
  
  // Reactive queries from Dexie - automatically update UI
  let messages = $derived($chatHistory || [];);
  let nodes = $derived($graphNodes || [];);
  
  // Component state
let searchQuery = $state('');
let searchResults = $state<any[] >([]);
let isSearching = $state(false);
let selectedNode = $state<any >(null);
let viewport = $state({ x: 0, y: 0, width: 1000, height: 1000 });
let performanceStats = $state<any >(null);
  
  // GPU canvas for visualization
let canvas = $state<HTMLCanvasElement;
let animationFrame = $state<number;
  
  // >(>(======================================================================
  // LIFECYCLE
  // ========================================================================
  
  onMount(async () => {
    try {
      console.log('🚀 Initializing Graph Explorer...')));
      
      // Initialize all systems
      await Promise.all([
        integratedSearch.initialize(),
        graphTextureManager.initialize(),
        unifiedDimensionalStore.initializeStorage()
      ]);
      
      // Load initial graph data
      await loadGraphData();
      
      // Start render loop
      startRenderLoop();
      
      // Load performance stats
      await updatePerformanceStats();
      
      console.log('✅ Graph Explorer ready');
    } catch (error) {
      console.error('❌ Failed to initialize Graph Explorer:', error);
    }
  });
  
  onDestroy(() => {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
  });
  
  // ========================================================================
  // DATA LOADING
  // ========================================================================
  
  async function loadGraphData() {
    try {
      // Load from database
      await graphTextureManager.loadGraphData(viewport);
      
      // Get database stats
      const stats = await db.getDatabaseStats();
      console.log('Database stats:', stats);
      
    } catch (error) {
      console.error('Failed to load graph data:', error);
    }
  }
  
  async function addSampleData() {
    try {
      // Add sample chat message
      await db.addChatMessage({
        role: 'user',
        content: searchQuery || 'Sample legal query about contract liability',
        metadata: {
          legalContext: {
            documentType: 'contract',
            jurisdiction: 'california',
            practiceArea: 'corporate'
          }
        }
      });
      
      // Add sample graph node
      await db.addGraphNode({
        nodeId: `node_${Date.now()}`,
        label: 'Sample Legal Node',
        position: { 
          x: Math.random() * viewport.width, 
          y: Math.random() * viewport.height 
        },
        embedding: Array.from({ length: 384 }, () => Math.random() * 2 - 1),
        rankingMatrix: Array.from({ length: 16 }, () => Math.random()),
        varianceMatrix: Array.from({ length: 16 }, () => Math.random() * 0.1),
        metadata: {
          documentType: 'contract',
          jurisdiction: 'california',
          practiceArea: 'corporate',
          confidence: 0.85,
          lastUpdated: new Date()
        },
        connections: []
      });
      
      console.log('✅ Sample data added');
      await updatePerformanceStats();
      
    } catch (error) {
      console.error('Failed to add sample data:', error);
    }
  }
  
  // ========================================================================
  // SEARCH FUNCTIONALITY
  // ========================================================================
  
  async function handleSearch() {
    if (!searchQuery.trim()) return;
    
    isSearching = true;
    try {
      // Add search to chat history
      await db.addChatMessage({
        role: 'user',
        content: searchQuery,
        metadata: {
          legalContext: {
            documentType: 'search',
            practiceArea: 'general'
          }
        }
      });
      
      // Perform integrated search
      const result = await integratedSearch.search({
        text: searchQuery,
        filters: {
          confidenceThreshold: 0.7
        },
        options: {
          searchStrategy: 'hybrid',
          useQuantizedVectors: true,
          includeReranking: true,
          maxResults: 20
        }
      });
      
      searchResults = result.results;
      
      // Add AI response to chat
      await db.addChatMessage({
        role: 'assistant',
        content: `Found ${result.results.length} results in ${result.metrics.totalTime}ms`,
        metadata: {
          responseTime: result.metrics.totalTime,
          legalContext: {
            documentType: 'search_results'
          }
        }
      });
      
      console.log('Search completed:', result);
      
    } catch (error) {
      console.error('Search failed:', error);
      await db.addChatMessage({
        role: 'system',
        content: `Search failed: ${error.message}`
      });
    } finally {
      isSearching = false;
    }
  }
  
  // ========================================================================
  // VIEWPORT MANAGEMENT
  // ========================================================================
  
  async function updateViewport(newViewport: typeof viewport) {
    viewport = newViewport;
    await graphTextureManager.updateViewport(viewport);
    await updatePerformanceStats();
  }
  
  function handleCanvasInteraction(event: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert to viewport coordinates
    const viewportX = viewport.x + (x / rect.width) * viewport.width;
    const viewportY = viewport.y + (y / rect.height) * viewport.height;
    
    // Find closest node
    const closestNode = nodes.reduce((closest, node) => {
      if (!node.position) return closest;
      
      const dx = node.position.x - viewportX;
      const dy = node.position.y - viewportY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      return distance < (closest.distance || Infinity) 
        ? { node, distance }
        : closest;
    }, { node: null, distance: Infinity });
    
    if (closestNode.node && closestNode.distance < 50) {
      selectedNode = closestNode.node;
    }
  }
  
  // ========================================================================
  // PERFORMANCE MONITORING
  // ========================================================================
  
  async function updatePerformanceStats() {
    try {
      const [
        dbStats,
        searchStats,
        gpuStats,
        storageStats,
        quantizationStats
      ] = await Promise.all([
        db.getDatabaseStats(),
        integratedSearch.getPerformanceAnalytics(),
        graphTextureManager.getPerformanceStats(),
        unifiedDimensionalStore.getStorageStats(),
        Promise.resolve(vectorQuantization.getQuantizationStats())
      ]);
      
      performanceStats = {
        database: dbStats,
        search: searchStats,
        gpu: gpuStats,
        storage: storageStats,
        quantization: quantizationStats
      };
      
    } catch (error) {
      console.error('Failed to update performance stats:', error);
    }
  }
  
  // ========================================================================
  // RENDERING
  // ========================================================================
  
  function startRenderLoop() {
    function render() {
      if (canvas && nodes.length > 0) {
        renderGraph();
      }
      animationFrame = requestAnimationFrame(render);
    }
    render();
  }
  
  function renderGraph() {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set up viewport
    const scaleX = canvas.width / viewport.width;
    const scaleY = canvas.height / viewport.height;
    
    // Render nodes
    for (const node of nodes) {
      if (!node.position) continue;
      
      const screenX = (node.position.x - viewport.x) * scaleX;
      const screenY = (node.position.y - viewport.y) * scaleY;
      
      // Skip nodes outside viewport
      if (screenX < -10 || screenX > canvas.width + 10 || 
          screenY < -10 || screenY > canvas.height + 10) continue;
      
      // Node color based on confidence
      const confidence = node.metadata?.confidence || 0;
      const hue = confidence * 120; // Green = high confidence, Red = low confidence
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      
      // Highlight selected node
      if (selectedNode?.nodeId === node.nodeId) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(screenX, screenY, 12, 0, Math.PI * 2);
        ctx.stroke();
      }
      
      // Draw node
      ctx.beginPath();
      ctx.arc(screenX, screenY, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Label
      if (confidence > 0.8) {
        ctx.fillStyle = '#333';
        ctx.font = '12px monospace';
        ctx.fillText(node.label.substring(0, 10), screenX + 12, screenY + 4);
      }
    }
  }
  
  // ========================================================================
  // CLEANUP
  // ========================================================================
  
  async function clearAllData() {
    if (confirm('Clear all data? This cannot be undone.')) {
      await Promise.all([
        db.chatHistory.clear(),
        db.graphNodes.clear(),
        db.graphEdges.clear(),
        db.cache.clear(),
        integratedSearch.reset(),
        graphTextureManager.cleanup(),
        unifiedDimensionalStore.clearAllStorage()
      ]);
      
      searchResults = [];
      selectedNode = null;
      await updatePerformanceStats();
      
      console.log('✅ All data cleared');
    }
  }
</script>

<!-- ============================================================================ -->
<!-- COMPONENT TEMPLATE -->
<!-- ============================================================================ -->

<div class="graph-explorer">
  <header class="controls">
    <div class="search-section">
      <input 
        bind:value={searchQuery}
        onkeypress={e => e.key === 'Enter' && handleSearch()}
        placeholder="Search legal documents, cases, precedents..."
        class="search-input"
        disabled={isSearching}
      />
      <button 
        on:onclick={handleSearch}
        disabled={isSearching || !searchQuery.trim()}
        class="search-btn"
      >
        {isSearching ? 'Searching...' : 'Search'}
      </button>
    </div>
    
    <div class="actions">
      <button on:onclick={addSampleData} class="action-btn">
        Add Sample Data
      </button>
      <button on:onclick={() => updatePerformanceStats()} class="action-btn">
        Refresh Stats
      </button>
      <button on:onclick={clearAllData} class="action-btn danger">
        Clear All
      </button>
    </div>
  </header>

  <main class="main-content">
    <!-- Graph Visualization Canvas -->
    <section class="graph-section">
      <h3>Interactive Graph ({nodes.length} nodes)</h3>
      <canvas 
        bind:this={canvas}
        on:onclick={handleCanvasInteraction}
        width="800"
        height="600"
        class="graph-canvas"
      ></canvas>
      
      {#if selectedNode}
        <div class="node-details">
          <h4>Selected Node: {selectedNode.label}</h4>
          <p>Confidence: {(selectedNode.metadata?.confidence * 100).toFixed(1)}%</p>
          <p>Type: {selectedNode.metadata?.documentType || 'Unknown'}</p>
          <p>Position: ({selectedNode.position?.x.toFixed(0)}, {selectedNode.position?.y.toFixed(0)})</p>
        </div>
      {/if}
    </section>

    <!-- Search Results -->
    {#if searchResults.length > 0}
      <section class="results-section">
        <h3>Search Results ({searchResults.length})</h3>
        <div class="results-grid">
          {#each searchResults as result}
            <div class="result-card">
              <div class="result-score">{(result.score * 100).toFixed(1)}%</div>
              <div class="result-content">
                <h4>{result.id}</h4>
                <p>{result.content}</p>
                <div class="result-meta">
                  <span class="meta-tag">{result.metadata.documentType}</span>
                  <span class="meta-tag">{result.metadata.source}</span>
                  {#if result.metadata.legalAnalysis}
                    <span class="risk-{result.metadata.legalAnalysis.riskLevel}">
                      {result.metadata.legalAnalysis.riskLevel} risk
                    </span>
                  {/if}
                </div>
              </div>
            </div>
          {/each}
        </div>
      </section>
    {/if}
  </main>

  <!-- Performance Dashboard -->
  {#if performanceStats}
    <aside class="stats-panel">
      <h3>System Performance</h3>
      
      <div class="stat-group">
        <h4>Database</h4>
        <p>Chat History: {performanceStats.database?.chatHistory || 0}</p>
        <p>Graph Nodes: {performanceStats.database?.graphNodes || 0}</p>
        <p>Cache Entries: {performanceStats.database?.cache?.entries || 0}</p>
      </div>
      
      {#if performanceStats.search}
        <div class="stat-group">
          <h4>Search Performance</h4>
          <p>Queries: {performanceStats.search.totalQueries}</p>
          <p>Avg Response: {performanceStats.search.averageResponseTime}ms</p>
          <p>Cache Hit Rate: {performanceStats.search.cacheHitRate}</p>
        </div>
      {/if}
      
      <div class="stat-group">
        <h4>GPU Memory</h4>
        <p>LOD Levels: {performanceStats.gpu?.lodLevels || 0}</p>
        <p>Memory: {((performanceStats.gpu?.memoryUsage || 0) / (1024 * 1024)).toFixed(1)}MB</p>
        <p>Total Nodes: {performanceStats.gpu?.totalNodes || 0}</p>
      </div>
      
      {#if performanceStats.quantization?.totalVectors > 0}
        <div class="stat-group">
          <h4>Vector Quantization</h4>
          <p>Vectors: {performanceStats.quantization.totalVectors}</p>
          <p>Compression: {performanceStats.quantization.compressionRatio.toFixed(1)}x</p>
          <p>Memory Saved: {(performanceStats.quantization.memoryReduction * 100).toFixed(1)}%</p>
        </div>
      {/if}
    </aside>
  {/if}

  <!-- Chat History (Reactive) -->
  <aside class="chat-panel">
    <h3>Chat History ({messages.length})</h3>
    <div class="chat-messages">
      {#each messages.slice(-10) as message}
        <div class="message message-{message.role}">
          <div class="message-meta">
            <span class="role">{message.role}</span>
            <span class="time">{message.timestamp.toLocaleTimeString()}</span>
          </div>
          <div class="message-content">{message.content}</div>
          {#if message.metadata?.legalContext}
            <div class="legal-context">
              {#if message.metadata.legalContext.documentType}
                <span class="context-tag">{message.metadata.legalContext.documentType}</span>
              {/if}
              {#if message.metadata.legalContext.jurisdiction}
                <span class="context-tag">{message.metadata.legalContext.jurisdiction}</span>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  </aside>
</div>

<!-- ============================================================================ -->
<!-- COMPONENT STYLES -->
<!-- ============================================================================ -->

<style>
  .graph-explorer {
    display: grid;
    grid-template-columns: 1fr 300px;
    grid-template-rows: auto 1fr auto;
    grid-template-areas: 
      "controls controls"
      "main stats"
      "main chat";
    height: 100vh;
    gap: 1rem;
    padding: 1rem;
    background: #f8f9fa;
    font-family: 'Inter', monospace;
  }

  .controls {
    grid-area: controls;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  .search-section {
    display: flex;
    gap: 0.5rem;
    flex: 1;
    max-width: 600px;
  }

  .search-input {
    flex: 1;
    padding: 0.75rem;
    border: 2px solid #e1e5e9;
    border-radius: 6px;
    font-size: 14px;
    font-family: inherit;
  }

  .search-input:focus {
    outline: none;
    border-color: #007bff;
  }

  .search-btn {
    padding: 0.75rem 1.5rem;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
  }

  .search-btn:disabled {
    background: #6c757d;
    cursor: not-allowed;
  }

  .actions {
    display: flex;
    gap: 0.5rem;
  }

  .action-btn {
    padding: 0.5rem 1rem;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
  }

  .action-btn:hover {
    background: #e9ecef;
  }

  .action-btn.danger {
    background: #dc3545;
    color: white;
    border-color: #dc3545;
  }

  .main-content {
    grid-area: main;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    overflow: auto;
  }

  .graph-section {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  .graph-canvas {
    width: 100%;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    cursor: crosshair;
    background: #fafbfc;
  }

  .node-details {
    margin-top: 1rem;
    padding: 0.75rem;
    background: #f8f9fa;
    border-radius: 4px;
    font-size: 13px;
  }

  .results-section {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }

  .result-card {
    display: flex;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 6px;
    border-left: 4px solid #007bff;
  }

  .result-score {
    font-weight: bold;
    color: #007bff;
    margin-right: 1rem;
    min-width: 50px;
  }

  .result-content h4 {
    margin: 0 0 0.5rem 0;
    font-size: 14px;
  }

  .result-content p {
    margin: 0 0 0.5rem 0;
    font-size: 13px;
    color: #6c757d;
  }

  .result-meta {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .meta-tag {
    padding: 0.25rem 0.5rem;
    background: #e9ecef;
    border-radius: 3px;
    font-size: 11px;
    color: #495057;
  }

  .risk-low { background: #d4edda; color: #155724; }
  .risk-medium { background: #fff3cd; color: #856404; }
  .risk-high { background: #f8d7da; color: #721c24; }

  .stats-panel {
    grid-area: stats;
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-size: 13px;
  }

  .stat-group {
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e9ecef;
  }

  .stat-group h4 {
    margin: 0 0 0.5rem 0;
    color: #495057;
  }

  .stat-group p {
    margin: 0.25rem 0;
    display: flex;
    justify-content: space-between;
  }

  .chat-panel {
    grid-area: chat;
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    max-height: 400px;
    overflow-y: auto;
  }

  .chat-messages {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 1rem;
  }

  .message {
    padding: 0.75rem;
    border-radius: 6px;
    font-size: 13px;
  }

  .message-user {
    background: #e3f2fd;
    margin-left: 2rem;
  }

  .message-assistant {
    background: #f3e5f5;
    margin-right: 2rem;
  }

  .message-system {
    background: #fff3e0;
    font-style: italic;
  }

  .message-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 11px;
    color: #6c757d;
  }

  .role {
    font-weight: 600;
    text-transform: uppercase;
  }

  .legal-context {
    margin-top: 0.5rem;
    display: flex;
    gap: 0.25rem;
    flex-wrap: wrap;
  }

  .context-tag {
    padding: 0.125rem 0.375rem;
    background: #007bff;
    color: white;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 500;
  }

  h3 {
    margin: 0 0 1rem 0;
    color: #212529;
    font-size: 16px;
  }

  h4 {
    margin: 0 0 0.5rem 0;
    color: #495057;
    font-size: 14px;
  }
</style>