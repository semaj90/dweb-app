<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { writable, type Writable } from 'svelte/store';
  import { SoraGraphTraversal, type SoraTraversalPath, type SoraTraversalOptions } from '$lib/ai/sora-graph-traversal.js';
  import { MoogleGraphSynthesizer, type MoogleVisualizationConfig, type Moogle2DOutput, type Moogle3DMesh } from '$lib/ai/moogle-graph-synthesizer.js';
  import { NESGPUIntegration } from '$lib/gpu/nes-gpu-integration.js';
  import { NESMemoryArchitecture } from '$lib/memory/nes-memory-architecture.js';
  import { SemanticAnalysisPipeline } from '$lib/ai/semantic-analysis-pipeline.js';
  import { DimensionalTensorStore } from '$lib/webgpu/dimensional-tensor-store.js';
  import { SOMWebGPUCache } from '$lib/webgpu/som-webgpu-cache.js';
  import { GPUTensorWorker } from '$lib/workers/gpu-tensor-worker.js';
  import { LegalAIReranker } from '$lib/ai/custom-reranker.js';
  
  // Props
  interface Props {
    query?: string;
    startNodeId?: string;
    neo4jDriver?: any;
    config?: Partial<SoraTraversalOptions & MoogleVisualizationConfig>;
    mode?: '2d' | '3d' | 'both';
    width?: number;
    height?: number;
    enableReinforcementLearning?: boolean;
    enableGPUAcceleration?: boolean;
    theme?: 'dark' | 'light' | 'legal';
    interactive?: boolean;
  }

  let {
    query = '',
    startNodeId = '',
    neo4jDriver = null,
    config = {},
    mode = '2d',
    width = 800,
    height = 600,
    enableReinforcementLearning = true,
    enableGPUAcceleration = true,
    theme = 'legal',
    interactive = true
  }: Props = $props();
  // Event dispatcher
  const dispatch = createEventDispatcher<{
    pathSelected: { path: SoraTraversalPath };
    nodeClicked: { nodeId: string; nodeType: string };
    visualizationComplete: { visualization: Moogle2DOutput | Moogle3DMesh };
    error: { message: string; error: any };
  }>();

  // State stores
  const loading: Writable<boolean> = writable(false);
  const paths: Writable<SoraTraversalPath[]> = writable([]);
  const visualization2D: Writable<Moogle2DOutput | null> = writable(null);
  const visualization3D: Writable<Moogle3DMesh | null> = writable(null);
  const stats: Writable<any> = writable({});
  const error: Writable<string | null> = writable(null);

  // Component instances
let soraTraversal = $state<SoraGraphTraversal | null >(null);
let moogleSynthesizer = $state<MoogleGraphSynthesizer | null >(null);
let canvas2D = $state<HTMLCanvasElement;
  let canvas3D: HTMLCanvasElement;
  let container: HTMLDivElement;

  // GPU and memory integrations
  let gpuIntegration: NESGPUIntegration | null >(null);
let memoryArch = $state<NESMemoryArchitecture | null >(null);
let semanticPipeline = $state<SemanticAnalysisPipeline | null >(null);
let tensorStore = $state<DimensionalTensorStore | null >(null);
let somCache = $state<SOMWebGPUCache | null >(null);
let gpuWorker = $state<GPUTensorWorker | null >(null);
let reranker = $state<LegalAIReranker | null >(null);

  // Theme configurations
  const themes = {
    dark: {
      backgroundColor: '#1a1a1a',
      nodeColors: {
        document: '#4CAF50',
        case: '#2196F3',
        evidence: '#FF5722',
        entity: '#9C27B0',
        concept: '#FFC107',
        relationship: '#607D8B'
      },
      edgeColors: {
        cites: '#FF9800',
        contains: '#8BC34A',
        related: '#03DAC6',
        similar: '#E91E63',
        references: '#00BCD4',
        contradicts: '#F44336'
      }
    },
    light: {
      backgroundColor: '#ffffff',
      nodeColors: {
        document: '#2E7D32',
        case: '#1565C0',
        evidence: '#D32F2F',
        entity: '#7B1FA2',
        concept: '#F57C00',
        relationship: '#455A64'
      },
      edgeColors: {
        cites: '#E65100',
        contains: '#558B2F',
        related: '#00695C',
        similar: '#AD1457',
        references: '#0097A7',
        contradicts: '#C62828'
      }
    },
    legal: {
      backgroundColor: '#0f1419',
      nodeColors: {
        document: '#4a9eff',
        case: '#ff6b35',
        evidence: '#f7931e',
        entity: '#c77dff',
        concept: '#06ffa5',
        relationship: '#87ceeb'
      },
      edgeColors: {
        cites: '#ff9f40',
        contains: '#4bc0c0',
        related: '#ff6384',
        similar: '#36a2eb',
        references: '#9966ff',
        contradicts: '#ff4757'
      }
    }
  };

  // Reactive statements
  $: currentTheme = themes[theme];
  $: traversalConfig = {
    maxDepth: 5,
    maxNodes: 100,
    scoreThreshold: 0.6,
    traversalStrategy: 'reinforcement' as const,
    semanticFiltering: true,
    useGPUAcceleration: enableGPUAcceleration,
    reinforcementLearning: {
      enabled: enableReinforcementLearning,
      explorationRate: 0.1,
      learningRate: 0.01,
      discountFactor: 0.95
    },
    ...config
  } as SoraTraversalOptions;

  $: visualizationConfig = {
    width,
    height,
    backgroundColor: currentTheme.backgroundColor,
    nodeColors: currentTheme.nodeColors,
    edgeColors: currentTheme.edgeColors,
    nodeSize: { min: 8, max: 32 },
    edgeThickness: { min: 1, max: 6 },
    meshDimensions: { width: 100, height: 100, depth: 100 },
    vertexCount: 10000,
    lodLevels: 4,
    colorScheme: 'semantic' as const,
    layout: 'legal-context' as const,
    physics: {
      gravity: 0.1,
      repulsion: 100,
      attraction: 0.05,
      damping: 0.9
    },
    reinforcementLearning: {
      enabled: enableReinforcementLearning,
      showTrainingProgress: true,
      highlightOptimalPaths: true,
      showRewardHeatmap: true,
      qValueVisualization: true
    },
    useWebGL: true,
    useWasm: true,
    enableCaching: true,
    qualityLevel: 'high' as const,
    ...config
  } as MoogleVisualizationConfig;

  onMount(async () => {
    try {
      await initializeComponents();
      if (query && startNodeId) {
        await performGraphTraversal();
      }
    } catch (err) {
      console.error('Sora component initialization failed:', err);
      error.set(`Initialization failed: ${err.message}`);
      dispatch('error', { message: 'Component initialization failed', error: err });
    }
  });

  onDestroy(() => {
    cleanup();
  });

  async function initializeComponents(): Promise<void> {
    try {
      // Initialize GPU and memory architecture
      gpuIntegration = new NESGPUIntegration();
      memoryArch = new NESMemoryArchitecture();
      semanticPipeline = new SemanticAnalysisPipeline();
      tensorStore = new DimensionalTensorStore({
        documents: 1000,
        chunks: 10000,
        representations: 100,
        maxLOD: 4
      });
      somCache = new SOMWebGPUCache();
      reranker = new LegalAIReranker();

      // Initialize GPU worker if available
      if (enableGPUAcceleration && typeof Worker !== 'undefined') {
        gpuWorker = new GPUTensorWorker();
      }

      // Initialize Sora graph traversal
      if (neo4jDriver) {
        soraTraversal = new SoraGraphTraversal(
          neo4jDriver,
          gpuIntegration,
          memoryArch,
          semanticPipeline,
          tensorStore,
          reranker
        );
      }

      // Initialize Moogle synthesizer
      moogleSynthesizer = new MoogleGraphSynthesizer(
        gpuIntegration,
        memoryArch,
        tensorStore,
        somCache,
        gpuWorker
      );

    } catch (error) {
      throw new Error(`Component initialization failed: ${error.message}`);
    }
  }

  async function performGraphTraversal(): Promise<void> {
    if (!soraTraversal || !moogleSynthesizer) {
      error.set('Components not initialized');
      return;
    }

    try {
      loading.set(true);
      error.set(null);

      // Perform graph traversal
      const traversalPaths = await soraTraversal.traverseGraph(
        startNodeId,
        query,
        traversalConfig
      );

      paths.set(traversalPaths);

      // Generate visualizations
      if (mode === '2d' || mode === 'both') {
        const viz2D = await moogleSynthesizer.synthesize2D(traversalPaths, visualizationConfig);
        visualization2D.set(viz2D);
        renderCanvas2D(viz2D);
        dispatch('visualizationComplete', { visualization: viz2D });
      }

      if (mode === '3d' || mode === 'both') {
        const viz3D = await moogleSynthesizer.synthesize3D(traversalPaths, visualizationConfig);
        visualization3D.set(viz3D);
        renderCanvas3D(viz3D);
        dispatch('visualizationComplete', { visualization: viz3D });
      }

      // Update statistics
      const reinforcementStats = soraTraversal.getReinforcementStats();
      const tensorStats = await soraTraversal.getTensorStats();
      const cacheStats = await moogleSynthesizer.getEnhancedCacheStats();

      stats.set({
        paths: traversalPaths.length,
        totalNodes: reinforcementStats.totalNodes,
        avgVisitCount: reinforcementStats.avgVisitCount,
        tensorSlices: tensorStats.totalSlices,
        cacheHitRate: cacheStats.renderingCache.hitRate,
        renderTime: $visualization2D?.metadata.renderTime || $visualization3D?.metadata.renderTime || 0
      });

    } catch (err) {
      console.error('Graph traversal failed:', err);
      error.set(`Traversal failed: ${err.message}`);
      dispatch('error', { message: 'Graph traversal failed', error: err });
    } finally {
      loading.set(false);
    }
  }

  function renderCanvas2D(viz: Moogle2DOutput): void {
    if (!canvas2D) return;

    const ctx = canvas2D.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);

    // Draw the visualization
    ctx.putImageData(viz.imageData, 0, 0);

    // Add interactive overlays if enabled
    if (interactive) {
      addInteractiveOverlays2D(ctx, viz);
    }
  }

  function renderCanvas3D(viz: Moogle3DMesh): void {
    if (!canvas3D) return;

    // For WebGL rendering, we would set up a proper 3D context
    // This is a simplified implementation
    const ctx = canvas3D.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = visualizationConfig.backgroundColor;
    ctx.fillRect(0, 0, canvas3D.width, canvas3D.height);

    // Render simplified 3D projection
    renderSimple3DProjection(ctx, viz);
  }

  function addInteractiveOverlays2D(ctx: CanvasRenderingContext2D, viz: Moogle2DOutput): void {
    // Add hover regions for nodes
    viz.metadata.nodePositions.forEach((nodePos) => {
      const nodeSize = 16;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(nodePos.x, nodePos.y, nodeSize, 0, 2 * Math.PI);
      ctx.stroke();
    });
  }

  function renderSimple3DProjection(ctx: CanvasRenderingContext2D, viz: Moogle3DMesh): void {
    // Simple orthographic projection of 3D points
    viz.metadata.nodePositions.forEach((nodePos) => {
      const projectedX = nodePos.x + width / 2;
      const projectedY = nodePos.y + height / 2;
      
      ctx.fillStyle = currentTheme.nodeColors.document;
      ctx.beginPath();
      ctx.arc(projectedX, projectedY, 6, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  function handleCanvasClick(event: MouseEvent, is3D: boolean = false): void {
    if (!interactive) return;

    const rect = (event.target as HTMLCanvasElement).getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked node
    const viz = is3D ? $visualization3D : $visualization2D;
    if (!viz) return;

    const clickedNode = viz.metadata.nodePositions.find(nodePos => {
      const distance = Math.sqrt(
        Math.pow(x - nodePos.x, 2) + Math.pow(y - nodePos.y, 2)
      );
      return distance < 20; // Click tolerance
    });

    if (clickedNode) {
      dispatch('nodeClicked', { 
        nodeId: clickedNode.id, 
        nodeType: 'unknown' // Would need to track node types in metadata
      });
    }
  }

  function handlePathSelection(pathIndex: number): void {
    const selectedPath = $paths[pathIndex];
    if (selectedPath) {
      dispatch('pathSelected', { path: selectedPath });
    }
  }

  function cleanup(): void {
    // Cleanup GPU resources
    if (gpuWorker) {
      gpuWorker.terminate?.();
    }
    
    // Clear caches
    if (soraTraversal) {
      soraTraversal.clearCache();
    }
    
    if (moogleSynthesizer) {
      moogleSynthesizer.clearCache();
    }
  }

  // Public methods for external control
  export async function refresh(): Promise<void> {
    if (query && startNodeId) {
      await performGraphTraversal();
    }
  }

  export function clearVisualization(): void {
    paths.set([]);
    visualization2D.set(null);
    visualization3D.set(null);
    stats.set({});
    error.set(null);
  }

  export function exportVisualization(format: 'png' | 'svg' | 'json' = 'png'): string | null {
    const viz = $visualization2D;
    if (!viz) return null;

    switch (format) {
      case 'png':
        return viz.base64;
      case 'svg':
        return viz.svg;
      case 'json':
        return JSON.stringify({
          paths: $paths,
          metadata: viz.metadata
        }, null, 2);
      default:
        return null;
    }
  }
</script>

<div 
  class="sora-graph-visualization" 
  class:loading={$loading}
  class:error={$error}
  bind:this={container}
  style="width: {width}px; height: {height}px;"
>
  <!-- Loading indicator -->
  {#if $loading}
    <div class="loading-overlay">
      <div class="spinner"></div>
      <p>Traversing graph with Sora AI...</p>
      <div class="loading-stats">
        {#if enableReinforcementLearning}
          <span class="loading-detail">🧠 Reinforcement Learning Active</span>
        {/if}
        {#if enableGPUAcceleration}
          <span class="loading-detail">⚡ GPU Acceleration Enabled</span>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Error display -->
  {#if $error}
    <div class="error-overlay">
      <h3>⚠️ Visualization Error</h3>
      <p>{$error}</p>
      <button on:on:onclick={() => error.set(null)}>Dismiss</button>
    </div>
  {/if}

  <!-- 2D Visualization -->
  {#if (mode === '2d' || mode === 'both') && !$loading}
    <div class="canvas-container" class:hidden={mode === '3d'}>
      <canvas
        bind:this={canvas2D}
        width={width}
        height={height}
        class="visualization-canvas canvas-2d"
        on:on:onclick={(e) => handleCanvasClick(e, false)}
      ></canvas>
      
      <!-- 2D Controls -->
      <div class="canvas-controls">
        <button class="control-btn" title="Zoom In">🔍+</button>
        <button class="control-btn" title="Zoom Out">🔍-</button>
        <button class="control-btn" title="Reset View">⟲</button>
        <button class="control-btn" title="Export">💾</button>
      </div>
    </div>
  {/if}

  <!-- 3D Visualization -->
  {#if (mode === '3d' || mode === 'both') && !$loading}
    <div class="canvas-container" class:hidden={mode === '2d'}>
      <canvas
        bind:this={canvas3D}
        width={width}
        height={height}
        class="visualization-canvas canvas-3d"
        on:on:onclick={(e) => handleCanvasClick(e, true)}
      ></canvas>
      
      <!-- 3D Controls -->
      <div class="canvas-controls">
        <button class="control-btn" title="Rotate">🔄</button>
        <button class="control-btn" title="Pan">👆</button>
        <button class="control-btn" title="LOD Toggle">📊</button>
        <button class="control-btn" title="Mesh Export">📦</button>
      </div>
    </div>
  {/if}

  <!-- Mode switcher for 'both' mode -->
  {#if mode === 'both' && !$loading}
    <div class="mode-switcher">
      <button class="mode-btn" class:active={true}>2D</button>
      <button class="mode-btn" class:active={false}>3D</button>
    </div>
  {/if}

  <!-- Path explorer panel -->
  {#if $paths.length > 0 && interactive}
    <div class="path-explorer">
      <h4>🛤️ Traversal Paths ({$paths.length})</h4>
      <div class="path-list">
        {#each $paths.slice(0, 5) as path, index}
          <div 
            class="path-item"
            class:high-score={path.totalScore > 0.8}
            on:on:click={() => handlePathSelection(index)}
          >
            <div class="path-header">
              <span class="path-score">Score: {path.totalScore.toFixed(3)}</span>
              <span class="path-length">Nodes: {path.nodes.length}</span>
            </div>
            <div class="path-preview">
              {path.nodes.slice(0, 3).map(n => n.type).join(' → ')}
              {path.nodes.length > 3 ? '...' : ''}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Statistics panel -->
  {#if Object.keys($stats).length > 0}
    <div class="stats-panel">
      <h4>📊 Performance Stats</h4>
      <div class="stat-grid">
        <div class="stat-item">
          <span class="stat-label">Paths Found</span>
          <span class="stat-value">{$stats.paths}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Graph Nodes</span>
          <span class="stat-value">{$stats.totalNodes}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Cache Hit Rate</span>
          <span class="stat-value">{($stats.cacheHitRate * 100).toFixed(1)}%</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Render Time</span>
          <span class="stat-value">{$stats.renderTime?.toFixed(0)}ms</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .sora-graph-visualization {
    position: relative;
    border-radius: 8px;
    background: var(--bg-color, #0f1419);
    border: 1px solid var(--border-color, #2a2a2a);
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
  }

  .loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(15, 20, 25, 0.95);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 100;
    color: #4a9eff;
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #2a2a2a;
    border-top: 3px solid #4a9eff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 16px;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .loading-stats {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 12px;
    opacity: 0.8;
  }

  .loading-detail {
    background: rgba(74, 158, 255, 0.2);
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid rgba(74, 158, 255, 0.3);
  }

  .error-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(255, 71, 87, 0.95);
    color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    z-index: 100;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }

  .canvas-container {
    position: relative;
    width: 100%;
    height: 100%;
  }

  .canvas-container.hidden {
    display: none;
  }

  .visualization-canvas {
    width: 100%;
    height: 100%;
    cursor: crosshair;
    transition: opacity 0.3s ease;
  }

  .visualization-canvas:hover {
    opacity: 0.95;
  }

  .canvas-controls {
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    gap: 4px;
    z-index: 10;
  }

  .control-btn {
    background: rgba(42, 42, 42, 0.9);
    border: 1px solid rgba(74, 158, 255, 0.3);
    color: #4a9eff;
    padding: 6px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s ease;
  }

  .control-btn:hover {
    background: rgba(74, 158, 255, 0.2);
    border-color: #4a9eff;
    transform: translateY(-1px);
  }

  .mode-switcher {
    position: absolute;
    top: 8px;
    left: 8px;
    display: flex;
    background: rgba(42, 42, 42, 0.9);
    border-radius: 6px;
    padding: 2px;
    z-index: 10;
  }

  .mode-btn {
    background: transparent;
    border: none;
    color: #87ceeb;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s ease;
  }

  .mode-btn.active {
    background: #4a9eff;
    color: white;
  }

  .path-explorer {
    position: absolute;
    bottom: 8px;
    left: 8px;
    background: rgba(15, 20, 25, 0.95);
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 12px;
    max-width: 300px;
    max-height: 200px;
    overflow-y: auto;
    z-index: 10;
  }

  .path-explorer h4 {
    margin: 0 0 8px 0;
    color: #4a9eff;
    font-size: 12px;
    font-weight: 600;
  }

  .path-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .path-item {
    background: rgba(42, 42, 42, 0.5);
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 11px;
  }

  .path-item:hover {
    background: rgba(74, 158, 255, 0.1);
    border-color: rgba(74, 158, 255, 0.3);
  }

  .path-item.high-score {
    border-color: rgba(6, 255, 165, 0.4);
    background: rgba(6, 255, 165, 0.1);
  }

  .path-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    color: #87ceeb;
  }

  .path-score {
    font-weight: 600;
  }

  .path-length {
    opacity: 0.8;
  }

  .path-preview {
    color: #c77dff;
    font-family: monospace;
    font-size: 10px;
    opacity: 0.9;
  }

  .stats-panel {
    position: absolute;
    bottom: 8px;
    right: 8px;
    background: rgba(15, 20, 25, 0.95);
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 12px;
    min-width: 200px;
    z-index: 10;
  }

  .stats-panel h4 {
    margin: 0 0 8px 0;
    color: #4a9eff;
    font-size: 12px;
    font-weight: 600;
  }

  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .stat-label {
    font-size: 10px;
    color: #87ceeb;
    opacity: 0.8;
  }

  .stat-value {
    font-size: 12px;
    color: #06ffa5;
    font-weight: 600;
    font-family: monospace;
  }

  /* Theme overrides */
  :global(.sora-graph-visualization[data-theme="light"]) {
    --bg-color: #ffffff;
    --border-color: #e0e0e0;
  }

  :global(.sora-graph-visualization[data-theme="dark"]) {
    --bg-color: #1a1a1a;
    --border-color: #333333;
  }

  :global(.sora-graph-visualization[data-theme="legal"]) {
    --bg-color: #0f1419;
    --border-color: #2a2a2a;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .path-explorer {
      max-width: 250px;
      bottom: 4px;
      left: 4px;
    }

    .stats-panel {
      min-width: 180px;
      bottom: 4px;
      right: 4px;
    }

    .canvas-controls {
      top: 4px;
      right: 4px;
    }

    .mode-switcher {
      top: 4px;
      left: 4px;
    }
  }
</style>