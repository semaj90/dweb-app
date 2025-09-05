<!-- GPU Cluster Acceleration Demo -->
<!-- Real-time GPU-accelerated legal AI visualizations -->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import Card from '$lib/components/ui/Card.svelte';
  import { createGPUClusterManager, checkGPUCapabilities } from '$lib/services/gpu-cluster-acceleration';
  import { createWebGLShaderCache, LEGAL_AI_SHADERS } from '$lib/utils/webgl-shader-cache';
  import { Activity, Cpu, Zap, Eye, BarChart3, Network, Clock } from 'lucide-svelte';

  // GPU system state
  let gpuManager: any = null;
  let shaderCache: any = null;
  let gpuCapabilities = $state({
    webgl: false,
    webgl2: false,
    webgpu: false,
    extensions: [] as string[]
  });

  // Canvas and WebGL context
  let canvas: HTMLCanvasElement
  let gl: WebGL2RenderingContext | null = null;

  // Demo state
  let isInitialized = $state(false);
  let activeVisualization = $state<string>('attentionHeatmap');
  let isRendering = $state(false);
  let animationFrame: number = 0;

  // Performance metrics
  let gpuMetrics = $state({
    totalContexts: 0,
    activeContexts: 0,
    totalShaders: 0,
    cacheHitRate: 0,
    compilationTime: 0,
    frameRate: 0,
    contextSwitches: 0
  });

  let shaderMetrics = $state({
    totalShaders: 0,
    cacheHits: 0,
    cacheMisses: 0,
    averageCompilationTime: 0,
    memoryUsage: 0
  });

  // Demo data
  let attentionData = $state<Float32Array>(new Float32Array(0));
  let documentData = $state<Float32Array>(new Float32Array(0));
  let timelineData = $state<Float32Array>(new Float32Array(0));

  onMount(async () => {
    await initializeGPUDemo();
  });

  onDestroy(() => {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }

    if (gpuManager) {
      gpuManager.destroy();
    }

    if (shaderCache) {
      shaderCache.cleanup();
    }
  });

  async function initializeGPUDemo() {
    try {
      console.log('🎮 Initializing GPU Demo...');

      // Check GPU capabilities
      gpuCapabilities = await checkGPUCapabilities();
      console.log('GPU Capabilities:', gpuCapabilities);

      if (!gpuCapabilities.webgl && !gpuCapabilities.webgl2) {
        throw new Error('WebGL not supported');
      }

      // Initialize WebGL context
      if (canvas) {
        gl = canvas.getContext('webgl2') || canvas.getContext('webgl') as WebGL2RenderingContext;

        if (!gl) {
          throw new Error('Failed to get WebGL context');
        }

        // Setup WebGL state
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.clearColor(0.05, 0.05, 0.1, 1.0);

        console.log('✅ WebGL context initialized');
      }

      // Initialize GPU cluster manager
      gpuManager = createGPUClusterManager();

      // Initialize shader cache
      if (gl) {
        shaderCache = createWebGLShaderCache(gl);
      }

      // Generate demo data
      generateDemoData();

      // Subscribe to metrics
      if (gpuManager) {
        gpuManager.getMetrics().subscribe((metrics: any) => {
          gpuMetrics = {
            totalContexts: metrics.totalContexts,
            activeContexts: metrics.activeContexts,
            totalShaders: metrics.totalShaders,
            cacheHitRate: metrics.cacheHitRate * 100,
            compilationTime: metrics.compilationTime,
            frameRate: metrics.performance.frameRate,
            contextSwitches: metrics.performance.contextSwitches
          };
        });
      }

      if (shaderCache) {
        shaderCache.getMetrics().subscribe((metrics: any) => {
          shaderMetrics = {
            totalShaders: metrics.totalShaders,
            cacheHits: metrics.cacheHits,
            cacheMisses: metrics.cacheMisses,
            averageCompilationTime: metrics.averageCompilationTime,
            memoryUsage: metrics.memoryUsage
          };
        });
      }

      isInitialized = true;
      console.log('✅ GPU Demo initialized successfully');

    } catch (error) {
      console.error('❌ GPU Demo initialization failed:', error);
    }
  }

  function generateDemoData() {
    // Generate attention weight data (simulating transformer attention)
    const attentionSize = 64 * 64; // 64x64 attention matrix
    attentionData = new Float32Array(attentionSize * 3); // x, y, attention

    for (let i = 0; i < attentionSize; i++) {
      const x = (i % 64) / 63 * 2 - 1; // -1 to 1
      const y = Math.floor(i / 64) / 63 * 2 - 1;
      const attention = Math.random() * Math.exp(-((x*x + y*y) * 2)); // Gaussian-like

      attentionData[i * 3] = x;
      attentionData[i * 3 + 1] = y;
      attentionData[i * 3 + 2] = attention;
    }

    // Generate document network data (simulating legal document relationships)
    const docCount = 100;
    documentData = new Float32Array(docCount * 7); // x, y, z, r, g, b, pagerank

    for (let i = 0; i < docCount; i++) {
      const angle = (i / docCount) * Math.PI * 2;
      const radius = 0.5 + Math.random() * 0.3;
      const pageRank = Math.random();

      documentData[i * 7] = Math.cos(angle) * radius;     // x
      documentData[i * 7 + 1] = Math.sin(angle) * radius; // y
      documentData[i * 7 + 2] = (Math.random() - 0.5) * 0.2; // z
      documentData[i * 7 + 3] = 0.3 + pageRank * 0.7;    // r
      documentData[i * 7 + 4] = 0.2 + pageRank * 0.3;    // g
      documentData[i * 7 + 5] = 0.8 - pageRank * 0.3;    // b
      documentData[i * 7 + 6] = pageRank;                 // pageRank
    }

    // Generate timeline data (simulating evidence timeline)
    const timelineCount = 50;
    timelineData = new Float32Array(timelineCount * 6); // x, y, timestamp, importance, r, g, b

    for (let i = 0; i < timelineCount; i++) {
      const t = i / (timelineCount - 1);
      const importance = Math.random();

      timelineData[i * 6] = t * 2 - 1;                    // x (time axis)
      timelineData[i * 6 + 1] = (Math.random() - 0.5) * 0.5; // y (random offset)
      timelineData[i * 6 + 2] = t;                        // timestamp
      timelineData[i * 6 + 3] = importance;               // importance
      timelineData[i * 6 + 4] = importance;               // r
      timelineData[i * 6 + 5] = 0.5 + importance * 0.5;  // g
      timelineData[i * 6 + 6] = 1.0 - importance * 0.5;  // b
    }

    console.log('📊 Demo data generated');
  }

  async function startVisualization(type: string) {
    if (!isInitialized || !gl || !shaderCache) return;

    activeVisualization = type;
    isRendering = true;

    try {
      // Get appropriate shader program
      const shaderId = `legal-ai-${type}`;
      const shaderProgram = await shaderCache.getShaderProgram(shaderId);

      console.log(`🎨 Starting ${type} visualization`);

      // Start render loop
      renderLoop();

    } catch (error) {
      console.error(`Failed to start ${type} visualization:`, error);
    }
  }

  function renderLoop() {
    if (!isRendering || !gl || !shaderCache) return;

    // Clear canvas
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const currentTime = Date.now() * 0.001; // Convert to seconds

    try {
      switch (activeVisualization) {
        case 'attentionHeatmap':
          renderAttentionHeatmap(currentTime);
          break;
        case 'documentNetwork':
          renderDocumentNetwork(currentTime);
          break;
        case 'evidenceTimeline':
          renderEvidenceTimeline(currentTime);
          break;
        case 'textFlow':
          renderTextFlow(currentTime);
          break;
      }
    } catch (error) {
      console.error('Render error:', error);
    }

    // Continue render loop
    if (isRendering) {
      animationFrame = requestAnimationFrame(renderLoop);
    }
  }

  async function renderAttentionHeatmap(time: number) {
    if (!gl || !shaderCache) return;

    try {
      const program = await shaderCache.getShaderProgram('legal-ai-attentionHeatmap');

      // Create vertex buffer if needed
      const positionBuffer = shaderCache.createVertexBuffer(attentionData);

      // Set up rendering state
      gl.useProgram(program.program);

      // Set uniforms
      const uniforms = {
        u_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], // Identity matrix
        u_time: time,
        u_scale: 0.2,
        u_lowColor: [0.1, 0.1, 0.8],
        u_highColor: [0.8, 0.2, 0.2],
        u_intensity: 1.0
      };

      shaderCache.setUniforms(program, uniforms);

      // Set up vertex attributes
      const attributes = {
        a_position: { buffer: positionBuffer, size: 2 },
        a_texCoord: { buffer: positionBuffer, size: 2 },
        a_attention: { buffer: positionBuffer, size: 1, offset: 2 * 4 }
      };

      shaderCache.setupVertexAttributes(program, attributes);

      // Render
      gl.drawArrays(gl.POINTS, 0, attentionData.length / 3);

    } catch (error) {
      console.error('Attention heatmap render error:', error);
    }
  }

  async function renderDocumentNetwork(time: number) {
    if (!gl || !shaderCache) return;

    try {
      const program = await shaderCache.getShaderProgram('legal-ai-documentNetwork');

      const positionBuffer = shaderCache.createVertexBuffer(documentData);

      gl.useProgram(program.program);

      const uniforms = {
        u_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        u_time: time,
        u_nodeSize: 10.0,
        u_alpha: 0.8
      };

      shaderCache.setUniforms(program, uniforms);

      const attributes = {
        a_position: { buffer: positionBuffer, size: 3 },
        a_color: { buffer: positionBuffer, size: 3, offset: 3 * 4 },
        a_pageRank: { buffer: positionBuffer, size: 1, offset: 6 * 4 }
      };

      shaderCache.setupVertexAttributes(program, attributes);

      gl.drawArrays(gl.POINTS, 0, documentData.length / 7);

    } catch (error) {
      console.error('Document network render error:', error);
    }
  }

  async function renderEvidenceTimeline(time: number) {
    if (!gl || !shaderCache) return;

    try {
      const program = await shaderCache.getShaderProgram('legal-ai-evidenceTimeline');

      const positionBuffer = shaderCache.createVertexBuffer(timelineData);

      gl.useProgram(program.program);

      const uniforms = {
        u_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        u_currentTime: (time * 0.1) % 1.0,
        u_timeRange: 1.0,
        u_alpha: 0.8
      };

      shaderCache.setUniforms(program, uniforms);

      const attributes = {
        a_position: { buffer: positionBuffer, size: 2 },
        a_timestamp: { buffer: positionBuffer, size: 1, offset: 2 * 4 },
        a_importance: { buffer: positionBuffer, size: 1, offset: 3 * 4 },
        a_evidenceColor: { buffer: positionBuffer, size: 3, offset: 4 * 4 }
      };

      shaderCache.setupVertexAttributes(program, attributes);

      gl.drawArrays(gl.POINTS, 0, timelineData.length / 7);

    } catch (error) {
      console.error('Evidence timeline render error:', error);
    }
  }

  async function renderTextFlow(time: number) {
    // Simplified text flow rendering
    if (!gl || !shaderCache) return;

    try {
      const program = await shaderCache.getShaderProgram('legal-ai-textFlow');
      // Implementation would be similar to other renderers

    } catch (error) {
      console.error('Text flow render error:', error);
    }
  }

  function stopVisualization() {
    isRendering = false;
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
  }

  async function executeGPUWorkload() {
    if (!gpuManager) return;

    try {
      const workload = {
        id: `demo_${Date.now()}`,
        type: 'vector-processing' as const,
        priority: 'high' as const,
        data: new Float32Array([1, 2, 3, 4, 5]),
        shaderProgram: 'vector-normalize',
        expectedDuration: 10,
        callback: (result: any) => {
          console.log('GPU workload result:', result);
        }
      };

      const result = await gpuManager.executeWorkload(workload);
      console.log('🔥 GPU workload completed:', result);

    } catch (error) {
      console.error('GPU workload failed:', error);
    }
  }

  function formatBytes(bytes: number): string {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  function formatPercentage(value: number): string {
    return `${value.toFixed(1)}%`;
  }
</script>

<svelte:head>
  <title>GPU Cluster Demo - Legal AI</title>
  <meta name="description" content="Real-time GPU-accelerated legal AI visualizations with multi-cluster support">
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-8">
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold mb-4">
        <span class="bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
          GPU Cluster Acceleration Demo
        </span>
      </h1>
      <p class="text-xl text-gray-300 max-w-4xl mx-auto">
        Real-time WebGL/WebGPU accelerated legal AI visualizations with multi-cluster GPU context switching
      </p>
    </div>

    <!-- GPU Capabilities -->
    <Card class="p-6 mb-8 bg-slate-800/30 border-slate-600">
      <h2 class="text-xl font-bold mb-4 flex items-center gap-2">
        <Cpu class="h-5 w-5" />
        GPU Capabilities
      </h2>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={gpuCapabilities.webgl} class:bg-red-100={!gpuCapabilities.webgl}>
            <div class="h-4 w-4" class:text-green-600={gpuCapabilities.webgl} class:text-red-600={!gpuCapabilities.webgl}>
              <Activity class="h-4 w-4" />
            </div>
          </div>
          <div>
            <p class="font-medium">WebGL</p>
            <p class="text-sm text-gray-400">{gpuCapabilities.webgl ? 'Supported' : 'Not Available'}</p>
          </div>
        </div>

        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={gpuCapabilities.webgl2} class:bg-red-100={!gpuCapabilities.webgl2}>
            <div class="h-4 w-4" class:text-green-600={gpuCapabilities.webgl2} class:text-red-600={!gpuCapabilities.webgl2}>
              <Zap class="h-4 w-4" />
            </div>
          </div>
          <div>
            <p class="font-medium">WebGL 2</p>
            <p class="text-sm text-gray-400">{gpuCapabilities.webgl2 ? 'Supported' : 'Not Available'}</p>
          </div>
        </div>

        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={gpuCapabilities.webgpu} class:bg-red-100={!gpuCapabilities.webgpu}>
            <div class="h-4 w-4" class:text-green-600={gpuCapabilities.webgpu} class:text-red-600={!gpuCapabilities.webgpu}>
              <Cpu class="h-4 w-4" />
            </div>
          </div>
          <div>
            <p class="font-medium">WebGPU</p>
            <p class="text-sm text-gray-400">{gpuCapabilities.webgpu ? 'Supported' : 'Not Available'}</p>
          </div>
        </div>
      </div>

      {#if gpuCapabilities.extensions.length > 0}
        <div class="mt-4">
          <p class="text-sm font-medium mb-2">WebGL Extensions:</p>
          <div class="flex flex-wrap gap-2">
            {#each gpuCapabilities.extensions.slice(0, 10) as extension}
              <span class="px-2 py-1 text-xs bg-slate-700 rounded">{extension}</span>
            {/each}
            {#if gpuCapabilities.extensions.length > 10}
              <span class="px-2 py-1 text-xs bg-slate-600 rounded">+{gpuCapabilities.extensions.length - 10} more</span>
            {/if}
          </div>
        </div>
      {/if}
    </Card>

    <!-- Performance Metrics -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <!-- GPU Metrics -->
      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg bg-green-100">
            <Cpu class="h-5 w-5 text-green-600" />
          </div>
          <div>
            <h3 class="font-semibold text-white">GPU Contexts</h3>
            <p class="text-sm text-gray-400">{gpuMetrics.activeContexts}/{gpuMetrics.totalContexts}</p>
          </div>
        </div>
      </Card>

      <!-- Shader Cache -->
      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg bg-blue-100">
            <Zap class="h-5 w-5 text-blue-600" />
          </div>
          <div>
            <h3 class="font-semibold text-white">Shaders</h3>
            <p class="text-sm text-gray-400">{shaderMetrics.totalShaders} cached</p>
          </div>
        </div>
      </Card>

      <!-- Cache Hit Rate -->
      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg bg-purple-100">
            <BarChart3 class="h-5 w-5 text-purple-600" />
          </div>
          <div>
            <h3 class="font-semibold text-white">Cache Hit Rate</h3>
            <p class="text-sm text-gray-400">{formatPercentage(gpuMetrics.cacheHitRate)}</p>
          </div>
        </div>
      </Card>

      <!-- Frame Rate -->
      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg bg-orange-100">
            <Activity class="h-5 w-5 text-orange-600" />
          </div>
          <div>
            <h3 class="font-semibold text-white">Frame Rate</h3>
            <p class="text-sm text-gray-400">{gpuMetrics.frameRate.toFixed(0)} FPS</p>
          </div>
        </div>
      </Card>
    </div>
  </div>

  <!-- Main Content -->
  <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- Visualization Canvas -->
    <Card class="p-6 bg-slate-800/30 border-slate-600">
      <h2 class="text-xl font-bold mb-4 flex items-center gap-2">
        <Eye class="h-5 w-5" />
        Legal AI Visualizations
      </h2>

      <div class="space-y-4">
        <!-- Canvas -->
        <div class="relative">
          <canvas
            bind:this={canvas}
            width="500"
            height="400"
            class="w-full border border-slate-600 rounded bg-black"
            style="max-width: 100%; height: auto"
          ></canvas>

          {#if !isInitialized}
            <div class="absolute inset-0 flex items-center justify-center bg-black/50 rounded">
              <div class="text-white text-center">
                <Activity class="h-8 w-8 mx-auto mb-2 animate-spin" />
                <p>Initializing GPU...</p>
              </div>
            </div>
          {/if}
        </div>

        <!-- Visualization Controls -->
        <div class="grid grid-cols-2 gap-2">
          <Button
            onclick={() => startVisualization('attentionHeatmap')}
            disabled={!isInitialized}
            variant={activeVisualization === 'attentionHeatmap' ? 'default' : 'outline'}
            class="text-sm"
          >
            Attention Heatmap
          </Button>

          <Button
            onclick={() => startVisualization('documentNetwork')}
            disabled={!isInitialized}
            variant={activeVisualization === 'documentNetwork' ? 'default' : 'outline'}
            class="text-sm"
          >
            Document Network
          </Button>

          <Button
            onclick={() => startVisualization('evidenceTimeline')}
            disabled={!isInitialized}
            variant={activeVisualization === 'evidenceTimeline' ? 'default' : 'outline'}
            class="text-sm"
          >
            Evidence Timeline
          </Button>

          <Button
            onclick={() => startVisualization('textFlow')}
            disabled={!isInitialized}
            variant={activeVisualization === 'textFlow' ? 'default' : 'outline'}
            class="text-sm"
          >
            Text Flow
          </Button>
        </div>

        <!-- Render Controls -->
        <div class="flex gap-2">
          {#if isRendering}
            <Button onclick={stopVisualization} class="bg-red-600 hover:bg-red-700">
              Stop Rendering
            </Button>
          {/if}

          <Button
            onclick={executeGPUWorkload}
            disabled={!isInitialized}
            variant="outline"
            class="text-white border-slate-600 hover:bg-slate-700"
          >
            Execute GPU Workload
          </Button>
        </div>
      </div>
    </Card>

    <!-- Performance Dashboard -->
    <div class="space-y-6">
      <!-- GPU Cluster Metrics -->
      <Card class="p-6 bg-slate-800/30 border-slate-600">
        <h3 class="text-xl font-bold mb-4 flex items-center gap-2">
          <Network class="h-5 w-5" />
          GPU Cluster Metrics
        </h3>

        <div class="space-y-4">
          <div class="grid grid-cols-2 gap-4">
            <div class="bg-slate-700/50 p-3 rounded">
              <p class="text-xs text-gray-400">Context Switches</p>
              <p class="text-lg font-bold">{gpuMetrics.contextSwitches}</p>
            </div>

            <div class="bg-slate-700/50 p-3 rounded">
              <p class="text-xs text-gray-400">Compilation Time</p>
              <p class="text-lg font-bold">{gpuMetrics.compilationTime}ms</p>
            </div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between text-sm">
              <span>Shader Cache Hit Rate</span>
              <span>{formatPercentage(gpuMetrics.cacheHitRate)}</span>
            </div>
            <div class="w-full bg-slate-700 rounded-full h-2">
              <div
                class="bg-green-500 h-2 rounded-full transition-all duration-500"
                style="width: {gpuMetrics.cacheHitRate}%"
              ></div>
            </div>
          </div>
        </div>
      </Card>

      <!-- Shader Cache Stats -->
      <Card class="p-6 bg-slate-800/30 border-slate-600">
        <h3 class="text-xl font-bold mb-4 flex items-center gap-2">
          <BarChart3 class="h-5 w-5" />
          Shader Cache Stats
        </h3>

        <div class="space-y-3">
          <div class="flex justify-between">
            <span class="text-gray-400">Total Shaders:</span>
            <span class="font-bold">{shaderMetrics.totalShaders}</span>
          </div>

          <div class="flex justify-between">
            <span class="text-gray-400">Cache Hits:</span>
            <span class="font-bold text-green-400">{shaderMetrics.cacheHits}</span>
          </div>

          <div class="flex justify-between">
            <span class="text-gray-400">Cache Misses:</span>
            <span class="font-bold text-red-400">{shaderMetrics.cacheMisses}</span>
          </div>

          <div class="flex justify-between">
            <span class="text-gray-400">Avg Compilation:</span>
            <span class="font-bold">{shaderMetrics.averageCompilationTime.toFixed(1)}ms</span>
          </div>

          <div class="flex justify-between">
            <span class="text-gray-400">Memory Usage:</span>
            <span class="font-bold">{formatBytes(shaderMetrics.memoryUsage)}</span>
          </div>
        </div>
      </Card>

      <!-- Available Shaders -->
      <Card class="p-6 bg-slate-800/30 border-slate-600">
        <h3 class="text-xl font-bold mb-4 flex items-center gap-2">
          <Clock class="h-5 w-5" />
          Available Shaders
        </h3>

        <div class="space-y-2">
          {#each Object.keys(LEGAL_AI_SHADERS) as shaderName}
            <div class="flex justify-between items-center py-2 px-3 bg-slate-700/50 rounded">
              <span class="font-mono text-sm">{shaderName}</span>
              <span class="text-xs text-gray-400">Legal AI</span>
            </div>
          {/each}
        </div>
      </Card>
    </div>
  </div>
</div>

<style>
  /* Custom WebGL canvas styling */
  canvas {
    image-rendering: pixelated
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
  }
</style>