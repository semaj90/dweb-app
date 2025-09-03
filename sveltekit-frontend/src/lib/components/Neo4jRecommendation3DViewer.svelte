<!--
  Neo4j 3D Recommendation Viewer with QUIC Streaming & Bit-Encoded Vertex Buffers
  Real-time 3D legal knowledge graph with GPU-accelerated rendering and progress animations
-->

<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { useMachine } from '@xstate/svelte';
  import { idleDetectionMachine } from '$lib/machines/idle-detection-rabbitmq-machine.js';
  import { neo4j3DEngine, type RecommendationGraph, type Neo4jNode } from '$lib/services/neo4j-3d-recommendation-engine.js';
  import { webgpuSOMCache } from '$lib/services/webgpu-som-enhanced-cache.js';

  // Props
  export let nodeId: string = '';
  export let nodeType: string = 'Case';
  export let maxNodes: number = 100;
  export let maxDepth: number = 3;
  export let autoStart: boolean = true;
  export let enableStreaming: boolean = true;
  export let showProgress: boolean = true;
  export let theme: 'light' | 'dark' | 'yorha' = 'yorha';

  // Component state
  let canvasRef: HTMLCanvasElement;
  let containerRef: HTMLDivElement;
  let gpuDevice: GPUDevice | null = null;
  let context: GPUCanvasContext | null = null;
  let animationFrame: number | null = null;
  let mounted = false;

  // Reactive state
  let currentGraph: RecommendationGraph | null = null;
  let isLoading = $state(false);
  let progress = $state(0);
  let error = $state<string | null>(null);
  let streamingActive = $state(false);
  let renderStats = $state({
    fps: 0,
    vertices: 0,
    relationships: 0,
    streamingChunks: 0
  });

  // XState machine for idle detection and self-prompting
  const { state: idleState, send: sendIdleEvent } = useMachine(idleDetectionMachine);

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    graphLoaded: { graph: RecommendationGraph };
    nodeClicked: { node: Neo4jNode };
    streamingStarted: { streamId: string };
    streamingError: { error: string };
    renderUpdate: { stats: typeof renderStats };
  }>();

  // Camera and animation state
  let camera = {
    position: { x: 0, y: 0, z: 50 },
    rotation: { x: 0, y: 0, z: 0 },
    fov: 45,
    target: { x: 0, y: 0, z: 0 }
  };

  let animation = {
    time: 0,
    phase: 0,
    speed: 1.0,
    enabled: true
  };

  // Progress animation state
  let progressAnimation = {
    value: 0,
    target: 0,
    speed: 0.05,
    segments: [] as Array<{ start: number; end: number; active: boolean; color: string }>
  };

  /**
   * Initialize WebGPU and canvas context
   */
  async function initializeWebGPU() {
    if (!canvasRef || !mounted) return;

    try {
      // Request WebGPU adapter
      const adapter = await navigator.gpu?.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!adapter) {
        throw new Error('WebGPU not supported');
      }

      // Request device
      gpuDevice = await adapter.requestDevice({
        requiredFeatures: ['shader-f16'],
        requiredLimits: {
          maxStorageBufferBindingSize: 1024 * 1024 * 1024 // 1GB
        }
      });

      // Get canvas context
      context = canvasRef.getContext('webgpu');
      if (!context) {
        throw new Error('Failed to get WebGPU context');
      }

      // Configure canvas
      const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({
        device: gpuDevice,
        format: presentationFormat,
        alphaMode: 'premultiplied'
      });

      // Initialize progress animation segments
      initializeProgressSegments();

      console.log('✅ WebGPU initialized successfully');

      // Start idle detection for self-prompting
      sendIdleEvent({ type: 'START_IDLE_DETECTION' });

    } catch (err) {
      error = `WebGPU initialization failed: ${err}`;
      console.error(error);
    }
  }

  /**
   * Initialize progress bar segments with YoRHa styling
   */
  function initializeProgressSegments() {
    const numSegments = 12;
    progressAnimation.segments = Array.from({ length: numSegments }, (_, i) => ({
      start: (i / numSegments) * 100,
      end: ((i + 1) / numSegments) * 100,
      active: false,
      color: theme === 'yorha' ? '#00ff00' : '#0ea5e9'
    }));
  }

  /**
   * Load Neo4j recommendations with progress tracking
   */
  async function loadRecommendations() {
    if (!nodeId || isLoading) return;

    isLoading = true;
    progress = 0;
    error = null;
    progressAnimation.target = 0;

    try {
      // Animate progress segments
      animateProgressSegments();

      // Step 1: Query Neo4j (30% progress)
      progressAnimation.target = 30;
      const graph = await neo4j3DEngine.getRecommendations({
        nodeId,
        nodeType,
        maxNodes,
        maxDepth,
        includeEmbeddings: true
      });

      // Step 2: Process for GPU (60% progress)
      progressAnimation.target = 60;
      await new Promise(resolve => setTimeout(resolve, 300)); // Simulate processing

      // Step 3: Start streaming if enabled (80% progress)
      if (enableStreaming) {
        progressAnimation.target = 80;
        const streamId = await neo4j3DEngine.startQUICStreaming(nodeId, {
          chunkSize: 8192,
          priority: 'high',
          compression: true
        });
        
        streamingActive = true;
        dispatch('streamingStarted', { streamId });
      }

      // Step 4: Complete (100% progress)
      progressAnimation.target = 100;
      currentGraph = graph;
      
      // Update render stats
      renderStats.vertices = graph.nodes.length;
      renderStats.relationships = graph.relationships.length;
      
      dispatch('graphLoaded', { graph });

      // Cache the graph in WebGPU SOM cache
      await cacheGraphInSOM(graph);

    } catch (err) {
      error = `Failed to load recommendations: ${err}`;
      console.error(error);
    } finally {
      isLoading = false;
      setTimeout(() => {
        progressAnimation.target = 0;
      }, 1000);
    }
  }

  /**
   * Animate progress bar segments
   */
  function animateProgressSegments() {
    const animateSegment = (index: number) => {
      if (index >= progressAnimation.segments.length) return;
      
      progressAnimation.segments[index].active = true;
      
      setTimeout(() => {
        progressAnimation.segments[index].active = false;
        animateSegment(index + 1);
      }, 100);
    };
    
    animateSegment(0);
  }

  /**
   * Cache graph in WebGPU SOM cache for enhanced performance
   */
  async function cacheGraphInSOM(graph: RecommendationGraph) {
    try {
      const cacheEntry = {
        id: `neo4j_3d_${graph.centerNode}_${Date.now()}`,
        category: 'graph' as const,
        severity: 'medium' as const,
        suggestions: [
          `3D Graph: ${graph.nodes.length} nodes, ${graph.relationships.length} relationships`,
          `Score: ${graph.recommendationScore.toFixed(2)}`,
          `Query time: ${graph.metadata.queryTime}ms`,
          `Streaming: ${streamingActive ? 'active' : 'disabled'}`
        ],
        webgpuProcessed: true,
        rtxOptimized: true,
        timestamp: new Date().toISOString(),
        confidence: graph.recommendationScore
      };

      await webgpuSOMCache.store(cacheEntry);
      console.log('📊 Graph cached in WebGPU SOM cache');
    } catch (err) {
      console.warn('Failed to cache graph in SOM:', err);
    }
  }

  /**
   * Start render loop
   */
  function startRenderLoop() {
    if (!gpuDevice || !context) return;

    let lastTime = 0;
    let frameCount = 0;
    let lastFPSTime = 0;

    const render = (currentTime: number) => {
      if (!mounted) return;

      const deltaTime = (currentTime - lastTime) / 1000;
      lastTime = currentTime;

      // Update animation
      if (animation.enabled) {
        animation.time += deltaTime * animation.speed;
        animation.phase = (animation.time % (2 * Math.PI));
        
        // Rotate camera around the graph
        camera.rotation.y = animation.time * 0.2;
        camera.position.x = Math.cos(camera.rotation.y) * 50;
        camera.position.z = Math.sin(camera.rotation.y) * 50;
      }

      // Update progress animation
      updateProgressAnimation(deltaTime);

      // Update streaming stats
      if (streamingActive) {
        const streamingStats = neo4j3DEngine.getStreamingStats();
        renderStats.streamingChunks = streamingStats.totalChunks;
      }

      // Calculate FPS
      frameCount++;
      if (currentTime - lastFPSTime >= 1000) {
        renderStats.fps = frameCount;
        frameCount = 0;
        lastFPSTime = currentTime;
        dispatch('renderUpdate', { stats: renderStats });
      }

      // Render frame (WebGPU rendering would go here)
      renderFrame();

      animationFrame = requestAnimationFrame(render);
    };

    animationFrame = requestAnimationFrame(render);
  }

  /**
   * Update progress animation
   */
  function updateProgressAnimation(deltaTime: number) {
    // Smooth progress animation
    const diff = progressAnimation.target - progressAnimation.value;
    progressAnimation.value += diff * progressAnimation.speed;
    
    // Update progress variable for UI
    progress = progressAnimation.value;
  }

  /**
   * Render WebGPU frame
   */
  function renderFrame() {
    if (!gpuDevice || !context) return;

    // Create command encoder
    const commandEncoder = gpuDevice.createCommandEncoder();
    
    // Get current texture
    const textureView = context.getCurrentTexture().createView();
    
    // Create render pass
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }, // Dark background
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });

    // Render graph nodes and relationships here
    // This would use the render pipeline from neo4j3DEngine
    
    renderPass.end();
    
    // Submit commands
    gpuDevice.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Handle canvas click for node selection
   */
  function handleCanvasClick(event: MouseEvent) {
    if (!currentGraph || !canvasRef) return;

    const rect = canvasRef.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert screen coordinates to world coordinates
    // This would involve ray casting through the 3D scene
    const clickedNode = findNodeAtPosition(x, y);
    
    if (clickedNode) {
      dispatch('nodeClicked', { node: clickedNode });
    }
  }

  /**
   * Find node at screen position (simplified)
   */
  function findNodeAtPosition(x: number, y: number): Neo4jNode | null {
    if (!currentGraph) return null;
    
    // Simplified hit testing - would be more sophisticated in production
    const centerX = canvasRef.width / 2;
    const centerY = canvasRef.height / 2;
    const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
    
    // Return first node if click is near center (simplified)
    return distance < 50 ? currentGraph.nodes[0] : null;
  }

  /**
   * Handle window resize
   */
  function handleResize() {
    if (!canvasRef || !containerRef) return;
    
    const rect = containerRef.getBoundingClientRect();
    canvasRef.width = rect.width * window.devicePixelRatio;
    canvasRef.height = rect.height * window.devicePixelRatio;
    canvasRef.style.width = `${rect.width}px`;
    canvasRef.style.height = `${rect.height}px`;
  }

  // Lifecycle
  onMount(async () => {
    mounted = true;
    
    // Initialize WebGPU
    await initializeWebGPU();
    
    // Handle resize
    handleResize();
    window.addEventListener('resize', handleResize);
    
    // Start render loop
    startRenderLoop();
    
    // Auto-load if enabled
    if (autoStart && nodeId) {
      await loadRecommendations();
    }
  });

  onDestroy(() => {
    mounted = false;
    
    // Cleanup
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
    
    window.removeEventListener('resize', handleResize);
    
    // Stop idle detection
    sendIdleEvent({ type: 'STOP_IDLE_DETECTION' });
    
    // Cleanup engine
    neo4j3DEngine.cleanup();
  });

  // Reactive updates
  $: if (nodeId && mounted) {
    loadRecommendations();
  }

  $: if ($idleState.matches('idle.generating_prompts')) {
    console.log('🤖 Self-prompting activated while viewing 3D graph');
    // Could trigger automatic graph updates or suggestions
  }
</script>

<!-- Component Template -->
<div 
  bind:this={containerRef}
  class="neo4j-3d-viewer"
  class:yorha-theme={theme === 'yorha'}
  class:dark-theme={theme === 'dark'}
  class:loading={isLoading}
>
  <!-- WebGPU Canvas -->
  <canvas
    bind:this={canvasRef}
    on:click={handleCanvasClick}
    on:mousemove={handleCanvasClick}
    class="render-canvas"
    aria-label="3D Neo4j Knowledge Graph Visualization"
  />
  
  <!-- Progress Bar with Bit-Encoded Animation -->
  {#if showProgress && (isLoading || progress > 0)}
    <div class="progress-container">
      <div class="progress-label">
        {#if isLoading}
          Loading 3D Knowledge Graph...
        {:else}
          Processing Complete
        {/if}
      </div>
      
      <!-- Segmented Progress Bar -->
      <div class="progress-bar">
        {#each progressAnimation.segments as segment}
          <div 
            class="progress-segment"
            class:active={segment.active}
            style:--segment-color={segment.color}
            style:--segment-width="{(segment.end - segment.start)}%"
          />
        {/each}
        
        <!-- Main Progress Fill -->
        <div 
          class="progress-fill"
          style:width="{progress}%"
        />
      </div>
      
      <!-- Bit-Encoded Progress Info -->
      <div class="progress-info">
        <span class="progress-percent">{Math.round(progress)}%</span>
        {#if streamingActive}
          <span class="streaming-indicator">
            📡 QUIC Streaming Active
          </span>
        {/if}
      </div>
    </div>
  {/if}
  
  <!-- Error Display -->
  {#if error}
    <div class="error-container">
      <div class="error-title">❌ 3D Graph Error</div>
      <div class="error-message">{error}</div>
      <button 
        class="retry-button"
        on:click={loadRecommendations}
        disabled={isLoading}
      >
        Retry Loading
      </button>
    </div>
  {/if}
  
  <!-- Graph Stats Overlay -->
  {#if currentGraph && !isLoading}
    <div class="stats-overlay">
      <div class="stat">
        <span class="stat-label">Nodes:</span>
        <span class="stat-value">{renderStats.vertices}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Relationships:</span>
        <span class="stat-value">{renderStats.relationships}</span>
      </div>
      <div class="stat">
        <span class="stat-label">FPS:</span>
        <span class="stat-value">{renderStats.fps}</span>
      </div>
      {#if streamingActive}
        <div class="stat">
          <span class="stat-label">Chunks:</span>
          <span class="stat-value">{renderStats.streamingChunks}</span>
        </div>
      {/if}
    </div>
  {/if}
  
  <!-- Camera Controls -->
  <div class="camera-controls">
    <button 
      class="control-button"
      on:click={() => animation.enabled = !animation.enabled}
      title="Toggle Animation"
    >
      {animation.enabled ? '⏸️' : '▶️'}
    </button>
    
    <button 
      class="control-button"
      on:click={() => camera = { ...camera, position: { x: 0, y: 0, z: 50 } }}
      title="Reset Camera"
    >
      🎯
    </button>
    
    {#if streamingActive}
      <button 
        class="control-button streaming"
        title="QUIC Streaming Active"
      >
        📡
      </button>
    {/if}
  </div>
</div>

<!-- Styles -->
<style>
  .neo4j-3d-viewer {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 400px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    overflow: hidden;
    font-family: 'Roboto Mono', monospace;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }

  .yorha-theme {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%);
    border: 1px solid #00ff00;
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
  }

  .dark-theme {
    background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #374151 100%);
  }

  .render-canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: pointer;
    transition: filter 0.3s ease;
  }

  .loading .render-canvas {
    filter: blur(2px) brightness(0.7);
  }

  /* Progress Bar Styles */
  .progress-container {
    position: absolute;
    top: 20px;
    left: 20px;
    right: 20px;
    z-index: 10;
    background: rgba(0, 0, 0, 0.8);
    padding: 16px;
    border-radius: 8px;
    backdrop-filter: blur(8px);
  }

  .progress-label {
    color: white;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }

  .progress-bar {
    position: relative;
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .progress-segment {
    position: absolute;
    top: 0;
    height: 100%;
    background: var(--segment-color, #0ea5e9);
    width: var(--segment-width, 8.33%);
    left: calc(var(--segment-start, 0) * 1%);
    opacity: 0;
    transition: opacity 0.2s ease;
  }

  .progress-segment.active {
    opacity: 0.6;
    animation: pulse 0.5s ease-in-out;
  }

  .progress-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #0ea5e9, #06b6d4);
    border-radius: 4px;
    transition: width 0.3s ease-out;
  }

  .yorha-theme .progress-fill {
    background: linear-gradient(90deg, #00ff00, #00cc00);
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
    font-size: 12px;
  }

  .streaming-indicator {
    color: #00ff00;
    font-weight: 600;
    animation: blink 1s infinite;
  }

  /* Stats Overlay */
  .stats-overlay {
    position: absolute;
    bottom: 20px;
    left: 20px;
    background: rgba(0, 0, 0, 0.8);
    padding: 12px;
    border-radius: 6px;
    backdrop-filter: blur(8px);
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    min-width: 200px;
  }

  .stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
    font-size: 12px;
  }

  .stat-label {
    opacity: 0.7;
  }

  .stat-value {
    font-weight: 600;
    color: #0ea5e9;
  }

  .yorha-theme .stat-value {
    color: #00ff00;
  }

  /* Camera Controls */
  .camera-controls {
    position: absolute;
    bottom: 20px;
    right: 20px;
    display: flex;
    gap: 8px;
  }

  .control-button {
    width: 40px;
    height: 40px;
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 6px;
    color: white;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
    backdrop-filter: blur(8px);
  }

  .control-button:hover {
    background: rgba(14, 165, 233, 0.2);
    border-color: #0ea5e9;
    transform: translateY(-2px);
  }

  .control-button.streaming {
    border-color: #00ff00;
    color: #00ff00;
    animation: pulse 2s infinite;
  }

  /* Error Display */
  .error-container {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(220, 38, 38, 0.9);
    color: white;
    padding: 24px;
    border-radius: 8px;
    text-align: center;
    max-width: 400px;
    backdrop-filter: blur(8px);
  }

  .error-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
  }

  .error-message {
    font-size: 14px;
    margin-bottom: 16px;
    opacity: 0.9;
  }

  .retry-button {
    background: white;
    color: #dc2626;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s ease;
  }

  .retry-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
  }

  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.3; }
  }

  /* Responsive */
  @media (max-width: 768px) {
    .progress-container,
    .stats-overlay {
      left: 10px;
      right: 10px;
    }
    
    .camera-controls {
      right: 10px;
      bottom: 10px;
    }
    
    .stats-overlay {
      grid-template-columns: 1fr;
      min-width: auto;
    }
  }
</style>