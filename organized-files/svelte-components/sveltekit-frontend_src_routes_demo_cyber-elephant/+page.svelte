<!-- Cyber Elephant - Ancient Psychic Tandem War Elephant -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import { fabric } from 'fabric';
  import { createNeuralSpriteEngine, createPerformanceStores } from '$lib/engines/neural-sprite-engine';
  import { createCyberElephant3D, generateMockDocuments, type DocumentPoint } from '$lib/engines/cyber-elephant-3d';
  import { tensorCoreUpscaler } from '$lib/services/tensor-upscaler-service';
  import { 
    Eye, Zap, Search, Brain, Gauge, Activity, 
    Database, Cpu, HardDrive, Network, Target 
  } from 'lucide-svelte';

  let canvas2d: HTMLCanvasElement
  let canvas3d: HTMLCanvasElement
  let fabricCanvas: fabric.Canvas;
  let neuralEngine = $state<any>();
  let cyberElephant = $state<any>();
  let performanceStores = $state<any>();
  
  // Demo state using Svelte 5 runes
  let isSystemActive = $state(false);
  let documentsLoaded = $state(0);
  let selectedVisualizationMode = $state('3d_scatter');
  let searchQuery = $state('');
  let searchResults = writable<any[]>([]);
  let selectedDocuments = writable<DocumentPoint[]>([]);
  
  // Performance metrics
  let systemMetrics = {
    neuralFPS: 60,
    spatialFPS: 60,
    cacheHitRate: 100,
    bvhSearchTime: 0.5,
    tensorUpscaling: true,
    gpuAcceleration: true
  };
  
  // Visualization modes
  const visualizationModes = [
    { id: '3d_scatter', name: '🌌 3D Document Galaxy', description: 'Documents as stars in semantic space' },
    { id: 'neural_network', name: '🧠 Neural Network View', description: 'Connected document clusters' },
    { id: 'legal_timeline', name: '⚖️ Legal Timeline', description: 'Chronological case law visualization' },
    { id: 'galaxy', name: '🌀 Spiral Galaxy Mode', description: 'Auto-rotating cosmic view' }
  ];
  
  // Integration demonstration states
  let currentDemoStage = $state('initialization');
  const demoStages = [
    'initialization',
    'neural_learning', 
    '3d_projection',
    'bvh_building',
    'semantic_search',
    'tensor_upscaling',
    'complete_system'
  ];
  
  onMount(async () => {
    await initializeSystem();
    startDemoSequence();
  });
  
  onDestroy(() => {
    if (neuralEngine) neuralEngine.destroy();
    if (cyberElephant) cyberElephant.destroy();
  });
  
  /**
   * Initialize the complete Cyber Elephant system
   */
  async function initializeSystem() {
    console.log('🐘 Initializing Ancient Psychic Tandem Cyber Elephant...');
    
    // 1. Initialize Neural Sprite Engine (2D learning system)
    fabricCanvas = new fabric.Canvas(canvas2d, {
      width: 400,
      height: 300,
      backgroundColor: '#0a0a1a',
      selection: false
    });
    
    neuralEngine = createNeuralSpriteEngine(fabricCanvas);
    performanceStores = createPerformanceStores(neuralEngine);
    
    // 2. Initialize Cyber Elephant 3D Engine
    cyberElephant = createCyberElephant3D(canvas3d, {
      visualizationMode: selectedVisualizationMode,
      maxDocuments: 1000,
      bvhMaxDepth: 10
    });
    
    // 3. Connect Neural Engine to 3D Engine for enhanced learning
    cyberElephant.connectNeuralEngine(neuralEngine);
    
    // 4. Initialize Tensor Upscaler integration
    await tensorCoreUpscaler.start();
    await tensorCoreUpscaler.initializeNeuralSpriteCache();
    
    // 5. Load demo documents
    const mockDocs = generateMockDocuments(500);
    await cyberElephant.loadDocuments(mockDocs);
    documentsLoaded = mockDocs.length;
    
    // 6. Start animation loop
    cyberElephant.animate();
    
    // 7. Subscribe to reactive stores
    setupReactiveSubscriptions();
    
    console.log('✅ Cyber Elephant system initialized successfully');
  }
  
  /**
   * Setup reactive subscriptions between systems
   */
  function setupReactiveSubscriptions() {
    // Connect 3D search results to neural learning
    cyberElephant.searchResults.subscribe((results: any[]) => {
      searchResults.set(results);
      
      // Log search interactions for neural learning
      if (results.length > 0 && neuralEngine) {
        neuralEngine.logUserActivity('3d_search_performed', {
          resultCount: results.length,
          visualizationMode: selectedVisualizationMode,
          queryType: 'spatial_similarity'
        });
      }
    });
    
    // Connect document selections to neural engine
    cyberElephant.selectedDocuments.subscribe((docs: DocumentPoint[]) => {
      selectedDocuments.set(docs);
      
      if (docs.length > 0 && neuralEngine) {
        for (const doc of docs) {
          neuralEngine.logUserActivity('document_selected', {
            documentId: doc.id,
            documentType: doc.metadata.type,
            clusterId: doc.clusterId,
            selectionMethod: '3d_click'
          });
        }
      }
    });
  }
  
  /**
   * Demonstration sequence showing system integration
   */
  async function startDemoSequence() {
    if (!isSystemActive) return;
    
    const stageActions = {
      initialization: async () => {
        await neuralEngine.captureCurrentState('system_startup', ['demo', 'initialization']);
        await delay(2000);
      },
      
      neural_learning: async () => {
        // Simulate neural learning activity
        for (let i = 0; i < 5; i++) {
          await neuralEngine.logUserActivity('demo_interaction', {
            stage: 'neural_learning',
            iteration: i
          });
          await delay(500);
        }
      },
      
      '3d_projection': async () => {
        // Change visualization mode to show 3D projection
        await cyberElephant.updateConfig({ visualizationMode: 'neural_network' });
        await delay(2000);
      },
      
      bvh_building: async () => {
        // Demonstrate BVH spatial search
        const center = { x: 0, y: 0, z: 0 };
        const results = cyberElephant.findNearestDocuments(center, 10);
        console.log(`🌳 BVH search found ${results.length} documents`);
        await delay(2000);
      },
      
      semantic_search: async () => {
        // Perform semantic search demonstration
        await performSemanticSearch('contract liability');
        await delay(2000);
      },
      
      tensor_upscaling: async () => {
        // Demonstrate tensor upscaling integration
        await tensorCoreUpscaler.optimizeForLegalDocuments();
        const stats = tensorCoreUpscaler.getNeuralSpriteCacheStats();
        console.log('🎯 Tensor upscaling stats:', stats);
        await delay(2000);
      },
      
      complete_system: async () => {
        console.log('🏆 Complete Cyber Elephant system demonstration finished!');
      }
    };
    
    for (const stage of demoStages) {
      if (!isSystemActive) break;
      
      currentDemoStage = stage;
      await stageActions[stage]();
    }
  }
  
  /**
   * Perform semantic search integration
   */
  async function performSemanticSearch(query: string) {
    console.log(`🔍 Performing semantic search: "${query}"`);
    
    // Log search in neural engine for learning
    neuralEngine.logUserActivity('semantic_search', {
      query,
      timestamp: Date.now(),
      searchType: 'text_to_3d'
    });
    
    // Simulate search results (in real implementation, would use embedding similarity)
    const mockResults = Array.from({ length: 5 }, (_, i) => ({
      document: {
        id: `search_result_${i}`,
        metadata: {
          title: `Search Result ${i + 1}`,
          type: 'contract',
          relevance: 0.9 - (i * 0.1)
        }
      },
      similarity: 0.95 - (i * 0.05),
      relevanceRank: i + 1
    }));
    
    searchResults.set(mockResults);
  }
  
  /**
   * Toggle system activation
   */
  function toggleSystem() {
    isSystemActive = !isSystemActive;
    
    if (isSystemActive) {
      startDemoSequence();
    }
  }
  
  /**
   * Change visualization mode
   */
  function changeVisualizationMode(mode: string) {
    selectedVisualizationMode = mode;
    if (cyberElephant) {
      cyberElephant.updateConfig({ visualizationMode: mode });
    }
  }
  
  /**
   * Utility delay function
   */
  function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  // Reactive derived stores
  let totalSystemScore = $derived(() => {
    const neuralScore = systemMetrics.cacheHitRate;
    const spatialScore = (100 - systemMetrics.bvhSearchTime * 10);
    const tensorScore = systemMetrics.tensorUpscaling ? 100 : 50;
    return Math.round((neuralScore + spatialScore + tensorScore) / 3);
  });
  
  let systemGrade = $derived(() => {
    const score = totalSystemScore();
    if (score >= 95) return 'S+';
    if (score >= 90) return 'S';
    if (score >= 85) return 'A';
    if (score >= 75) return 'B';
    if (score >= 65) return 'C';
    return 'D';
  });
</script>

<svelte:head>
  <title>Cyber Elephant - Ancient Psychic Tandem War Elephant</title>
</svelte:head>

<div class="cyber-elephant-container">
  <!-- System Header -->
  <header class="system-header">
    <div class="title-section">
      <h1 class="system-title">🐘 ANCIENT PSYCHIC TANDEM CYBER ELEPHANT</h1>
      <p class="system-subtitle">Revolutionary 3D Document Intelligence with Neural Learning</p>
    </div>
    
    <div class="system-status">
      <div class="status-indicator" class:active={isSystemActive}>
        <div class="pulse-ring"></div>
        <div class="status-text">
          {isSystemActive ? 'SYSTEM ACTIVE' : 'STANDBY MODE'}
        </div>
      </div>
      
      <div class="system-metrics">
        <div class="metric">
          <Gauge size="16" />
          <span>Score: {totalSystemScore()}</span>
        </div>
        <div class="metric">
          <Target size="16" />
          <span>Grade: {systemGrade()}</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Interface -->
  <div class="main-interface">
    <!-- Control Panel -->
    <aside class="control-panel">
      <div class="panel-section">
        <h3>🎮 System Control</h3>
        <button 
          class="cyber-btn primary" 
          class:active={isSystemActive}
          onclick={toggleSystem}
        >
          {isSystemActive ? 'DEACTIVATE' : 'ACTIVATE ELEPHANT'}
        </button>
        
        <div class="demo-stage">
          <span>Demo Stage:</span>
          <div class="stage-indicator">{currentDemoStage.replace('_', ' ').toUpperCase()}</div>
        </div>
      </div>

      <div class="panel-section">
        <h3>👁️ Visualization Mode</h3>
        <div class="mode-selector">
          {#each visualizationModes as mode}
            <button 
              class="mode-btn"
              class:selected={selectedVisualizationMode === mode.id}
              onclick={() => changeVisualizationMode(mode.id)}
            >
              <div class="mode-name">{mode.name}</div>
              <div class="mode-desc">{mode.description}</div>
            </button>
          {/each}
        </div>
      </div>

      <div class="panel-section">
        <h3>🔍 Semantic Search</h3>
        <div class="search-controls">
          <input 
            type="text" 
            bind:value={searchQuery}
            placeholder="Search documents..."
            class="search-input"
          />
          <button 
            class="cyber-btn secondary"
            onclick={() => performSemanticSearch(searchQuery)}
          >
            <Search size="16" />
            Search
          </button>
        </div>
        
        {#if $searchResults.length > 0}
          <div class="search-results">
            <h4>Results ({$searchResults.length})</h4>
            {#each $searchResults as result}
              <div class="result-item">
                <div class="result-title">{result.document.metadata.title}</div>
                <div class="result-score">Similarity: {Math.round(result.similarity * 100)}%</div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <div class="panel-section">
        <h3>🧠 Neural Intelligence</h3>
        <div class="intelligence-stats">
          <div class="stat-row">
            <Brain size="14" />
            <span>Learning Rate:</span>
            <span class="stat-value">Real-time</span>
          </div>
          <div class="stat-row">
            <Activity size="14" />
            <span>Cache Hit Rate:</span>
            <span class="stat-value">{Math.round(systemMetrics.cacheHitRate)}%</span>
          </div>
          <div class="stat-row">
            <Zap size="14" />
            <span>Tensor Upscaling:</span>
            <span class="stat-value">{systemMetrics.tensorUpscaling ? 'Active' : 'Inactive'}</span>
          </div>
        </div>
      </div>
    </aside>

    <!-- Dual Canvas System -->
    <main class="canvas-system">
      <div class="canvas-header">
        <h2>🎯 Integrated Neural + Spatial Intelligence</h2>
        <div class="canvas-stats">
          <span>Documents: {documentsLoaded}</span>
          <span>•</span>
          <span>Neural FPS: {systemMetrics.neuralFPS}</span>
          <span>•</span>
          <span>Spatial FPS: {systemMetrics.spatialFPS}</span>
          <span>•</span>
          <span>BVH Search: {systemMetrics.bvhSearchTime}ms</span>
        </div>
      </div>

      <div class="dual-canvas">
        <!-- Neural Sprite Engine (2D Learning) -->
        <div class="canvas-container neural-canvas">
          <div class="canvas-label">
            <Brain size="16" />
            <span>Neural Sprite Engine</span>
            <div class="performance-badge">{performanceStores?.performanceGrade || 'S+'}</div>
          </div>
          <canvas bind:this={canvas2d} class="sprite-canvas"></canvas>
          <div class="canvas-overlay">
            <div class="tech-info">
              <div>• Self-Organizing Maps</div>
              <div>• Multi-core Processing</div>
              <div>• WebGL2 Shader Cache</div>
              <div>• AI Pattern Recognition</div>
            </div>
          </div>
        </div>

        <!-- Cyber Elephant 3D (Spatial Visualization) -->
        <div class="canvas-container spatial-canvas">
          <div class="canvas-label">
            <Eye size="16" />
            <span>3D Document Space</span>
            <div class="performance-badge">{cyberElephant?.getStatus()?.performanceGrade || 'A'}</div>
          </div>
          <canvas bind:this={canvas3d} class="elephant-canvas"></canvas>
          <div class="canvas-overlay">
            <div class="tech-info">
              <div>• BVH Spatial Search</div>
              <div>• Three.js Visualization</div>
              <div>• Tensor Core Optimization</div>
              <div>• Real-time Interaction</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Integration Demonstration -->
      <div class="integration-demo">
        <h3>🔗 System Integration Flow</h3>
        <div class="demo-flow">
          {#each demoStages as stage, index}
            <div 
              class="flow-step"
              class:active={currentDemoStage === stage}
              class:completed={demoStages.indexOf(currentDemoStage) > index}
            >
              <div class="step-number">{index + 1}</div>
              <div class="step-name">{stage.replace('_', ' ')}</div>
            </div>
            {#if index < demoStages.length - 1}
              <div class="flow-arrow">→</div>
            {/if}
          {/each}
        </div>
      </div>
    </main>
  </div>

  <!-- Performance Dashboard -->
  <footer class="performance-dashboard">
    <div class="dashboard-section">
      <h4>⚡ System Performance</h4>
      <div class="performance-grid">
        <div class="perf-card">
          <Cpu size="20" />
          <div class="perf-value">{systemMetrics.neuralFPS}</div>
          <div class="perf-label">Neural FPS</div>
        </div>
        <div class="perf-card">
          <Database size="20" />
          <div class="perf-value">{systemMetrics.spatialFPS}</div>
          <div class="perf-label">Spatial FPS</div>
        </div>
        <div class="perf-card">
          <HardDrive size="20" />
          <div class="perf-value">{Math.round(systemMetrics.cacheHitRate)}%</div>
          <div class="perf-label">Cache Hit</div>
        </div>
        <div class="perf-card">
          <Network size="20" />
          <div class="perf-value">{systemMetrics.bvhSearchTime}ms</div>
          <div class="perf-label">Search Time</div>
        </div>
      </div>
    </div>

    <div class="dashboard-section">
      <h4>🎯 Integration Status</h4>
      <div class="integration-status">
        <div class="status-item">
          <div class="status-dot active"></div>
          <span>Neural Sprite Engine: Active</span>
        </div>
        <div class="status-item">
          <div class="status-dot active"></div>
          <span>3D Visualization: Running</span>
        </div>
        <div class="status-item">
          <div class="status-dot active"></div>
          <span>Tensor Upscaler: Optimizing</span>
        </div>
        <div class="status-item">
          <div class="status-dot active"></div>
          <span>BVH Acceleration: Ready</span>
        </div>
      </div>
    </div>
  </footer>
</div>

<style>
  /* @unocss-include */
  .cyber-elephant-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #00ff88;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    padding: 1rem;
  }

  .system-header {
    display: flex
    justify-content: space-between;
    align-items: center
    padding: 2rem;
    background: rgba(0, 0, 0, 0.8);
    border: 2px solid #00ff88;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
  }

  .system-title {
    font-size: 2rem;
    font-weight: bold
    margin: 0 0 0.5rem 0;
    text-shadow: 0 0 20px #00ff88;
    animation: elephant-glow 3s ease-in-out infinite alternate;
  }

  .system-subtitle {
    font-size: 0.9rem;
    opacity: 0.8;
    margin: 0;
  }

  .system-status {
    display: flex
    align-items: center
    gap: 2rem;
  }

  .status-indicator {
    display: flex
    align-items: center
    gap: 1rem;
    position: relative
  }

  .pulse-ring {
    width: 20px;
    height: 20px;
    border: 2px solid #ff4444;
    border-radius: 50%;
    position: relative
  }

  .status-indicator.active .pulse-ring {
    border-color: #00ff88;
    animation: pulse-ring 2s infinite;
  }

  .system-metrics {
    display: flex
    gap: 1rem;
  }

  .metric {
    display: flex
    align-items: center
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid #00ff88;
    border-radius: 6px;
    font-size: 0.8rem;
  }

  .main-interface {
    display: grid
    grid-template-columns: 350px 1fr;
    gap: 2rem;
    min-height: 800px;
  }

  .control-panel {
    background: rgba(0, 0, 0, 0.9);
    border: 2px solid #00ff88;
    border-radius: 12px;
    padding: 1.5rem;
    overflow-y: auto
  }

  .panel-section {
    margin-bottom: 2rem;
  }

  .panel-section h3 {
    font-size: 1rem;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #00ff88;
  }

  .cyber-btn {
    background: rgba(0, 255, 136, 0.1);
    border: 2px solid #00ff88;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #00ff88;
    font-family: inherit
    font-size: 0.9rem;
    font-weight: bold
    cursor: pointer
    display: flex
    align-items: center
    justify-content: center
    gap: 0.5rem;
    transition: all 0.3s ease;
    width: 100%;
  }

  .cyber-btn:hover {
    background: rgba(0, 255, 136, 0.2);
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3);
  }

  .cyber-btn.primary {
    background: rgba(0, 255, 136, 0.2);
  }

  .cyber-btn.primary.active {
    background: rgba(255, 68, 68, 0.2);
    border-color: #ff4444;
    color: #ff4444;
  }

  .cyber-btn.secondary {
    background: rgba(0, 170, 255, 0.1);
    border-color: #00aaff;
    color: #00aaff;
  }

  .demo-stage {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(0, 100, 255, 0.1);
    border: 1px solid #0064ff;
    border-radius: 6px;
    text-align: center
  }

  .stage-indicator {
    font-weight: bold
    color: #00aaff;
    margin-top: 0.5rem;
  }

  .mode-selector {
    display: flex
    flex-direction: column
    gap: 0.8rem;
  }

  .mode-btn {
    background: rgba(0, 0, 0, 0.5);
    border: 2px solid #333;
    border-radius: 8px;
    padding: 1rem;
    color: #00ff88;
    font-family: inherit
    cursor: pointer
    transition: all 0.3s ease;
    text-align: left
  }

  .mode-btn:hover {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
  }

  .mode-btn.selected {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.2);
    box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
  }

  .mode-name {
    font-weight: bold
    margin-bottom: 0.3rem;
  }

  .mode-desc {
    font-size: 0.8rem;
    opacity: 0.7;
  }

  .search-controls {
    display: flex
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .search-input {
    flex: 1;
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid #333;
    border-radius: 6px;
    padding: 0.8rem;
    color: #00ff88;
    font-family: inherit
  }

  .search-input:focus {
    outline: none
    border-color: #00ff88;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
  }

  .search-results {
    margin-top: 1rem;
  }

  .search-results h4 {
    font-size: 0.9rem;
    margin: 0 0 0.8rem 0;
    color: #00aaff;
  }

  .result-item {
    background: rgba(0, 100, 255, 0.1);
    border: 1px solid #0064ff;
    border-radius: 4px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
  }

  .result-title {
    font-weight: bold
    margin-bottom: 0.3rem;
  }

  .result-score {
    font-size: 0.8rem;
    opacity: 0.8;
  }

  .intelligence-stats {
    display: flex
    flex-direction: column
    gap: 0.8rem;
  }

  .stat-row {
    display: flex
    align-items: center
    gap: 0.8rem;
    font-size: 0.8rem;
  }

  .stat-value {
    margin-left: auto
    font-weight: bold
    color: #00aaff;
  }

  .canvas-system {
    background: rgba(0, 0, 0, 0.9);
    border: 2px solid #00ff88;
    border-radius: 12px;
    padding: 1.5rem;
    display: flex
    flex-direction: column
  }

  .canvas-header {
    display: flex
    justify-content: space-between;
    align-items: center
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #00ff88;
  }

  .canvas-header h2 {
    font-size: 1.3rem;
    margin: 0;
  }

  .canvas-stats {
    display: flex
    gap: 1rem;
    font-size: 0.8rem;
    opacity: 0.8;
  }

  .dual-canvas {
    display: grid
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
  }

  .canvas-container {
    position: relative
    background: rgba(0, 0, 0, 0.8);
    border: 2px solid #333;
    border-radius: 8px;
    padding: 1rem;
  }

  .neural-canvas {
    border-color: #00ff88;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
  }

  .spatial-canvas {
    border-color: #00aaff;
    box-shadow: 0 0 20px rgba(0, 170, 255, 0.2);
  }

  .canvas-label {
    display: flex
    align-items: center
    gap: 0.8rem;
    margin-bottom: 1rem;
    font-weight: bold
  }

  .performance-badge {
    margin-left: auto
    background: rgba(0, 255, 136, 0.2);
    border: 1px solid #00ff88;
    border-radius: 4px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    font-weight: bold
  }

  .sprite-canvas, .elephant-canvas {
    width: 100%;
    height: 300px;
    border: 1px solid #444;
    border-radius: 4px;
    background: #0a0a1a;
  }

  .canvas-overlay {
    position: absolute
    top: 1rem;
    right: 1rem;
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid #333;
    border-radius: 4px;
    padding: 0.8rem;
  }

  .tech-info {
    font-size: 0.7rem;
    opacity: 0.7;
  }

  .tech-info div {
    margin-bottom: 0.3rem;
  }

  .integration-demo {
    margin-top: auto
  }

  .integration-demo h3 {
    font-size: 1rem;
    margin: 0 0 1rem 0;
  }

  .demo-flow {
    display: flex
    align-items: center
    gap: 0.5rem;
    flex-wrap: wrap
  }

  .flow-step {
    display: flex
    flex-direction: column
    align-items: center
    padding: 0.8rem;
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid #333;
    border-radius: 6px;
    min-width: 80px;
    transition: all 0.3s ease;
  }

  .flow-step.active {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
  }

  .flow-step.completed {
    border-color: #00aaff;
    background: rgba(0, 170, 255, 0.1);
  }

  .step-number {
    font-weight: bold
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
  }

  .step-name {
    font-size: 0.7rem;
    text-align: center
    text-transform: uppercase
  }

  .flow-arrow {
    font-size: 1.2rem;
    color: #00ff88;
  }

  .performance-dashboard {
    margin-top: 2rem;
    background: rgba(0, 0, 0, 0.9);
    border: 2px solid #00ff88;
    border-radius: 12px;
    padding: 2rem;
    display: grid
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
  }

  .dashboard-section h4 {
    font-size: 1.1rem;
    margin: 0 0 1.5rem 0;
    color: #00ff88;
  }

  .performance-grid {
    display: grid
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
  }

  .perf-card {
    background: rgba(0, 255, 136, 0.05);
    border: 1px solid #00ff88;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center
    display: flex
    flex-direction: column
    align-items: center
    gap: 0.8rem;
  }

  .perf-value {
    font-size: 1.5rem;
    font-weight: bold
    color: #00ff88;
  }

  .perf-label {
    font-size: 0.8rem;
    opacity: 0.8;
  }

  .integration-status {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .status-item {
    display: flex
    align-items: center
    gap: 1rem;
    font-size: 0.9rem;
  }

  .status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #666;
  }

  .status-dot.active {
    background: #00ff88;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
  }

  /* Animations */
  @keyframes elephant-glow {
    from {
      text-shadow: 0 0 20px #00ff88;
    }
    to {
      text-shadow: 0 0 40px #00ff88, 0 0 60px #00ff88;
    }
  }

  @keyframes pulse-ring {
    0% {
      transform: scale(0.8);
      opacity: 1;
    }
    100% {
      transform: scale(2);
      opacity: 0;
    }
  }

  /* Responsive Design */
  @media (max-width: 1400px) {
    .main-interface {
      grid-template-columns: 300px 1fr;
    }
    
    .dual-canvas {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 1000px) {
    .main-interface {
      grid-template-columns: 1fr;
    }
    
    .performance-dashboard {
      grid-template-columns: 1fr;
    }
    
    .performance-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 600px) {
    .system-header {
      flex-direction: column
      gap: 1rem;
      text-align: center
    }
    
    .performance-grid {
      grid-template-columns: 1fr;
    }
  }
</style>