<script lang="ts">
  import { onMount } from 'svelte';
  import { initializeGPUCache, initializePerformanceMonitor, GPUCacheUtils } from '$lib/services/gpu-cache-integration';
  
  // GPU Cache state
  let gpuCache = $state(null);
  let performanceMonitor = $state(null);
  let cacheStats = $state({
    totalEntries: 0,
    totalSize: 0,
    hitRate: 0,
    gpuMemoryUsed: 0,
    cpuMemoryUsed: 0,
    isGPUAvailable: false
  });
  
  let performanceMetrics = $state({
    fps: 60,
    gpuUtilization: 0,
    memoryUsage: 0,
    cacheHitRate: 0
  });
  
  // Test data
  let testResults = $state([]);
  let isTestRunning = $state(false);
  
  // NES.css styles
  let selectedTheme = $state('hybrid');
  let enableGPUEffects = $state(true);
  
  onMount(() => {
    // Initialize GPU cache
    gpuCache = initializeGPUCache({
      maxMemory: 1024, // 1GB for testing
      cacheStrategy: 'LRU',
      enableCompression: true,
      gpuAcceleration: true,
      rtx3060TiOptimization: true
    });
    
    // Initialize performance monitor
    performanceMonitor = initializePerformanceMonitor();
    
    // Subscribe to stats
    const statsUnsubscribe = gpuCache.getStats().subscribe(stats => {
      cacheStats = stats;
    });
    
    const metricsUnsubscribe = performanceMonitor.getMetrics().subscribe(metrics => {
      performanceMetrics = metrics;
    });
    
    console.log('✅ YoRHa N64 Test Page initialized with GPU cache');
    
    return () => {
      statsUnsubscribe();
      metricsUnsubscribe();
    };
  });
  
  async function runGPUCacheTest() {
    isTestRunning = true;
    testResults = [];
    
    // Test 1: Basic cache operations
    const test1Start = performance.now();
    await gpuCache.set('test-key-1', { data: 'Test data for GPU cache', array: new Array(1000).fill(1) });
    const cachedData1 = await gpuCache.get('test-key-1');
    const test1Time = performance.now() - test1Start;
    
    testResults = [...testResults, {
      name: 'Basic Cache Operations',
      status: cachedData1 ? '✅ Passed' : '❌ Failed',
      time: `${test1Time.toFixed(2)}ms`,
      details: cachedData1 ? 'Data cached and retrieved successfully' : 'Failed to retrieve cached data'
    }];
    
    // Test 2: Large data caching
    const test2Start = performance.now();
    const largeData = {
      vectors: new Array(10000).fill(0).map(() => Math.random()),
      metadata: { test: true, timestamp: Date.now() }
    };
    await gpuCache.set('large-data', largeData, { compress: true, gpu: true });
    const cachedLarge = await gpuCache.get('large-data');
    const test2Time = performance.now() - test2Start;
    
    testResults = [...testResults, {
      name: 'Large Data with Compression',
      status: cachedLarge ? '✅ Passed' : '❌ Failed',
      time: `${test2Time.toFixed(2)}ms`,
      details: `Cached ${GPUCacheUtils.formatBytes(JSON.stringify(largeData).length)} of data`
    }];
    
    // Test 3: Cache hit rate
    const test3Start = performance.now();
    for (let i = 0; i < 10; i++) {
      await gpuCache.get('test-key-1'); // Should be hits
    }
    for (let i = 0; i < 5; i++) {
      await gpuCache.get(`non-existent-${i}`); // Should be misses
    }
    const test3Time = performance.now() - test3Start;
    
    testResults = [...testResults, {
      name: 'Cache Hit Rate Test',
      status: '✅ Passed',
      time: `${test3Time.toFixed(2)}ms`,
      details: `Hit rate: ${(cacheStats.hitRate * 100).toFixed(1)}%`
    }];
    
    // Test 4: GPU acceleration check
    const test4Result = cacheStats.isGPUAvailable ? 
      '✅ GPU acceleration available' : 
      '⚠️ Running in CPU-only mode';
    
    testResults = [...testResults, {
      name: 'GPU Acceleration Status',
      status: cacheStats.isGPUAvailable ? '✅ Enabled' : '⚠️ CPU Mode',
      time: 'N/A',
      details: test4Result
    }];
    
    isTestRunning = false;
  }
  
  function clearCache() {
    gpuCache?.clear();
    testResults = [];
  }
  
  function applyTheme(theme: string) {
    selectedTheme = theme;
    document.body.className = `theme-${theme} ${enableGPUEffects ? 'gpu-accelerated' : ''}`;
  }
</script>

<div class="container mx-auto p-6 max-w-6xl">
  <!-- Header with NES.css styling -->
  <div class="nes-container with-title is-centered mb-8">
    <p class="title">YoRHa N64 GPU Cache Integration Test</p>
    <h1 class="text-2xl font-bold text-yorha-accent-warm mb-2">
      🎮 N64 + YoRHa UI + NES.css + GPU Cache
    </h1>
    <p class="text-sm text-gray-500">
      Complete integration test for all UI systems with GPU acceleration
    </p>
  </div>

  <!-- Theme Selector -->
  <div class="nes-container with-title mb-6">
    <p class="title">Theme Configuration</p>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <button 
        class="nes-btn {selectedTheme === 'yorha' ? 'is-primary' : ''}"
        on:on:onclick={() => applyTheme('yorha')}
      >
        YoRHa
      </button>
      <button 
        class="nes-btn {selectedTheme === 'nes' ? 'is-success' : ''}"
        on:on:onclick={() => applyTheme('nes')}
      >
        NES
      </button>
      <button 
        class="nes-btn {selectedTheme === 'n64' ? 'is-warning' : ''}"
        on:on:onclick={() => applyTheme('n64')}
      >
        N64
      </button>
      <button 
        class="nes-btn {selectedTheme === 'hybrid' ? 'is-error' : ''}"
        on:on:onclick={() => applyTheme('hybrid')}
      >
        Hybrid
      </button>
    </div>
    
    <div class="mt-4">
      <label>
        <input 
          type="checkbox" 
          class="nes-checkbox"
          bind:checked={enableGPUEffects}
          onchange={() => applyTheme(selectedTheme)}
        />
        <span>Enable GPU Effects</span>
      </label>
    </div>
  </div>

  <!-- GPU Cache Status -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
    <!-- Cache Statistics -->
    <div class="nes-container with-title">
      <p class="title">📊 Cache Statistics</p>
      <div class="space-y-2">
        <div class="flex justify-between">
          <span>Total Entries:</span>
          <span class="nes-text is-primary">{cacheStats.totalEntries}</span>
        </div>
        <div class="flex justify-between">
          <span>Total Size:</span>
          <span class="nes-text is-primary">{GPUCacheUtils.formatBytes(cacheStats.totalSize)}</span>
        </div>
        <div class="flex justify-between">
          <span>Hit Rate:</span>
          <span class="nes-text {cacheStats.hitRate > 0.8 ? 'is-success' : cacheStats.hitRate > 0.5 ? 'is-warning' : 'is-error'}">
            {(cacheStats.hitRate * 100).toFixed(1)}%
          </span>
        </div>
        <div class="flex justify-between">
          <span>GPU Memory:</span>
          <span class="nes-text is-primary">{GPUCacheUtils.formatBytes(cacheStats.gpuMemoryUsed)}</span>
        </div>
        <div class="flex justify-between">
          <span>CPU Memory:</span>
          <span class="nes-text is-primary">{GPUCacheUtils.formatBytes(cacheStats.cpuMemoryUsed)}</span>
        </div>
        <div class="flex justify-between">
          <span>GPU Available:</span>
          <span class="nes-text {cacheStats.isGPUAvailable ? 'is-success' : 'is-error'}">
            {cacheStats.isGPUAvailable ? '✅ Yes' : '❌ No'}
          </span>
        </div>
      </div>
    </div>

    <!-- Performance Metrics -->
    <div class="nes-container with-title">
      <p class="title">⚡ Performance Metrics</p>
      <div class="space-y-2">
        <div class="flex justify-between">
          <span>FPS:</span>
          <span class="nes-text {performanceMetrics.fps >= 60 ? 'is-success' : performanceMetrics.fps >= 30 ? 'is-warning' : 'is-error'}">
            {performanceMetrics.fps}
          </span>
        </div>
        <div class="flex justify-between">
          <span>GPU Usage:</span>
          <span class="nes-text is-primary">{performanceMetrics.gpuUtilization.toFixed(1)}%</span>
        </div>
        <div class="flex justify-between">
          <span>Memory Usage:</span>
          <span class="nes-text is-primary">{performanceMetrics.memoryUsage.toFixed(1)}%</span>
        </div>
        <div class="flex justify-between">
          <span>Cache Hit Rate:</span>
          <span class="nes-text is-primary">{(performanceMetrics.cacheHitRate * 100).toFixed(1)}%</span>
        </div>
      </div>
      
      <!-- FPS Progress Bar -->
      <div class="mt-4">
        <label>FPS Performance</label>
        <progress 
          class="nes-progress {performanceMetrics.fps >= 60 ? 'is-success' : performanceMetrics.fps >= 30 ? 'is-warning' : 'is-error'}" 
          value={performanceMetrics.fps} 
          max="120"
        ></progress>
      </div>
    </div>
  </div>

  <!-- Test Controls -->
  <div class="nes-container with-title mb-6">
    <p class="title">🧪 GPU Cache Tests</p>
    <div class="flex gap-4">
      <button 
        class="nes-btn is-primary"
        on:on:onclick={runGPUCacheTest}
        disabled={isTestRunning}
      >
        {isTestRunning ? 'Running Tests...' : 'Run GPU Cache Tests'}
      </button>
      <button 
        class="nes-btn is-warning"
        on:on:onclick={clearCache}
        disabled={isTestRunning}
      >
        Clear Cache
      </button>
    </div>
  </div>

  <!-- Test Results -->
  {#if testResults.length > 0}
    <div class="nes-container with-title">
      <p class="title">📋 Test Results</p>
      <div class="nes-table-responsive">
        <table class="nes-table is-bordered is-centered">
          <thead>
            <tr>
              <th>Test Name</th>
              <th>Status</th>
              <th>Time</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {#each testResults as result}
              <tr>
                <td>{result.name}</td>
                <td>{result.status}</td>
                <td>{result.time}</td>
                <td>{result.details}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>
  {/if}

  <!-- YoRHa UI Components Test -->
  <div class="nes-container with-title mt-6">
    <p class="title">🎮 YoRHa UI Components</p>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="nes-container is-rounded">
        <p>YoRHa Button</p>
        <button class="nes-btn yorha-button">
          <span>EXECUTE</span>
        </button>
      </div>
      
      <div class="nes-container is-rounded">
        <p>NES Input</p>
        <div class="nes-field">
          <input type="text" class="nes-input" placeholder="Enter command...">
        </div>
      </div>
      
      <div class="nes-container is-rounded">
        <p>N64 Select</p>
        <div class="nes-select">
          <select>
            <option>Option 1</option>
            <option>Option 2</option>
            <option>Option 3</option>
          </select>
        </div>
      </div>
    </div>
  </div>

  <!-- GPU Shader Effects Demo -->
  <div class="nes-container with-title mt-6">
    <p class="title">✨ GPU Shader Effects</p>
    <div class="gpu-effects-demo">
      <div class="shader-box gpu-accelerated">
        <div class="glow-effect">GPU Accelerated Glow</div>
      </div>
      <div class="shader-box gpu-accelerated">
        <div class="scan-lines">Scan Lines Effect</div>
      </div>
      <div class="shader-box gpu-accelerated">
        <div class="pixel-effect">Pixelated Effect</div>
      </div>
    </div>
  </div>
</div>

<style>
  /* YoRHa Custom Styles */
  .yorha-button {
    background: linear-gradient(135deg, #4a4a4a 0%, #6b6b6b 100%);
    color: #ffd700;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    position: relative;
    overflow: hidden;
    transform: translateZ(0);
    will-change: transform;
  }
  
  .yorha-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.3), transparent);
    transition: left 0.5s;
  }
  
  .yorha-button:hover::before {
    left: 100%;
  }
  
  /* GPU Effects */
  .gpu-effects-demo {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }
  
  .shader-box {
    padding: 2rem;
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
    border: 2px solid #4a4a4a;
    border-radius: 8px;
    text-align: center;
    color: #ffd700;
    font-weight: bold;
    text-transform: uppercase;
    transform: translateZ(0);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  .shader-box:hover {
    transform: translateZ(0) scale(1.05);
    border-color: #ffd700;
    box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
  }
  
  .glow-effect {
    text-shadow: 0 0 10px currentColor, 0 0 20px currentColor, 0 0 30px currentColor;
    animation: glow-pulse 2s ease-in-out infinite;
  }
  
  @keyframes glow-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
  
  .scan-lines {
    position: relative;
    overflow: hidden;
  }
  
  .scan-lines::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      transparent 50%,
      rgba(255, 215, 0, 0.03) 50%
    );
    background-size: 100% 4px;
    pointer-events: none;
    animation: scan 8s linear infinite;
  }
  
  @keyframes scan {
    0% { transform: translateY(0); }
    100% { transform: translateY(20px); }
  }
  
  .pixel-effect {
    image-rendering: pixelated;
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
    font-family: 'Press Start 2P', monospace;
    font-size: 12px;
  }
  
  /* GPU Acceleration */
  .gpu-accelerated {
    transform: translateZ(0);
    will-change: transform, opacity;
    backface-visibility: hidden;
    -webkit-font-smoothing: antialiased;
  }
  
  /* Theme-specific overrides */
  .theme-yorha {
    --primary-color: #4a4a4a;
    --accent-color: #ffd700;
  }
  
  .theme-nes {
    --primary-color: #212529;
    --accent-color: #92cc41;
  }
  
  .theme-n64 {
    --primary-color: #3d5afe;
    --accent-color: #ff4081;
  }
  
  .theme-hybrid {
    --primary-color: #4a4a4a;
    --accent-color: #92cc41;
  }
</style>
