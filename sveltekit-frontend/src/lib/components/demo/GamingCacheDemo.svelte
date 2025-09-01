<!--
Gaming Components Cache Demo
Comprehensive demo showcasing GPU texture caching, shader optimization,
and performance monitoring across N64 and YoRHa gaming components
-->

<script lang="ts">
  import { onMount, tick } from 'svelte';
  import type { 
    EnhancedGPUCacheEntry, 
    TextureCacheEntry, 
    CompiledShaderCache,
    N64RenderingOptions,
    AntiAliasingConfig,
    CachePerformanceTracker
  } from '$lib/types/gpu-cache-integration';
  import { enhancedGPUCacheService } from '$lib/services/enhanced-gpu-cache-service';
  import { gpuCacheInvalidationSystem } from '$lib/services/gpu-cache-invalidation-system';
  import { wasmCacheOps } from '$lib/services/wasm-accelerated-cache-ops';
  import N64TextureFilteringCache from '$lib/components/ui/gaming/n64/N64TextureFilteringCache.svelte';
  import YoRHaAAShaderCache from '$lib/components/three/yorha-ui/YoRHaAAShaderCache.svelte';
  import CachePerformanceMonitor from '$lib/components/dashboard/CachePerformanceMonitor.svelte';

  // Demo state
  let activeDemo = $state<'overview' | 'n64' | 'yorha' | 'performance' | 'wasm' | 'analytics'>('overview');
  let demoStarted = $state(false);
  let cacheMetrics = $state<CachePerformanceTracker>({
    textureHitRate: 0,
    shaderHitRate: 0,
    memoryUtilization: 0,
    wasmAccelerationGain: 0,
    averageResponseTime: 0,
    cacheEfficiency: 0
  });

  // Demo scenarios
  let currentScenario = $state(0);
let scenarios = $state([
    {
      name: 'N64 Texture Filtering Showcase',
      description: 'Demonstrate high-performance N64-style texture filtering with cache optimization',
      component: 'n64',
      textures: ['mario-face', 'zelda-sword', 'starfox-ship', 'donkey-kong-barrel'],
      filters: ['bilinear', 'trilinear', 'anisotropic']
    },
    {
      name: 'YoRHa Anti-Aliasing Pipeline',
      description: 'Showcase YoRHa AA shaders with real-time compilation and caching',
      component: 'yorha',
      shaders: ['FXAA', 'TAA', 'SMAA', 'MSAA'],
      qualities: ['fast', 'balanced', 'quality']
    },
    {
      name: 'WASM Acceleration Benchmark',
      description: 'Compare WASM-accelerated vs JavaScript texture/shader processing',
      component: 'wasm',
      operations: ['texture-compression', 'shader-optimization', 'memory-defragmentation'],
      datasets: ['small', 'medium', 'large']
    },
    {
      name: 'Cache Performance Analytics',
      description: 'Real-time cache performance monitoring and optimization',
      component: 'performance',
      metrics: ['hit-rate', 'memory-usage', 'invalidation-rate', 'wasm-gains'],
      timeframes: ['1min', '5min', '15min', '1hour']
    }
  ]);

  // Demo data generators
  let textureTestData = $state<Array<{ id: string; name: string; data: ImageData }>>([]);
  let shaderTestData = $state<Array<{ id: string; type: 'vertex' | 'fragment' | 'compute'; source: string }>>([]);
  let performanceHistory = $state<Array<{ timestamp: number; metrics: CachePerformanceTracker }>>([]);

  // Real-time demo stats
  let demoStats = $state({
    totalOperations: 0,
    successfulOperations: 0,
    failedOperations: 0,
    averageProcessingTime: 0,
    cacheHits: 0,
    cacheMisses: 0,
    wasmAcceleratedOps: 0,
    memoryUsedMB: 0
  });

  // Live demo controls
  let autoRunScenarios = $state(false);
  let scenarioInterval = $state(5000); // 5 seconds
  let enableWasmAcceleration = $state(true);
  let enableRealTimeMetrics = $state(true);
  let stressTestMode = $state(false);
let demoTimer = $state<number | null >(null);
let metricsTimer = $state<number | null >(null);

  onMount(() => {
    initializeDemoData();
    startRealTimeMetrics();
    
    // Cleanup on component destroy
    return () => {
      if (demoTimer) clearInterval(demoTimer);
      if (metricsTimer) clearInterval(metricsTimer);
    };
  });

  /**
   * Initialize demo data and test scenarios
   */
  async function initializeDemoData() {
    try {
      // Generate test texture data
      textureTestData = await generateTestTextures();
      
      // Generate test shader data
      shaderTestData = await generateTestShaders();
      
      console.log('[Gaming Cache Demo] Initialized with:', {
        textures: textureTestData.length,
        shaders: shaderTestData.length
      });
    } catch (error) {
      console.error('[Gaming Cache Demo] Initialization failed:', error);
    }
  }

  /**
   * Generate test texture data for demos
   */
  async function generateTestTextures(): Promise<Array<{ id: string; name: string; data: ImageData }>> {
    const textures = [];
    const sizes = [
      { width: 64, height: 64, name: 'icon' },
      { width: 256, height: 256, name: 'sprite' },
      { width: 512, height: 512, name: 'texture' },
      { width: 1024, height: 1024, name: 'hires' }
    ];

    for (let i = 0; i < sizes.length; i++) {
      const { width, height, name } = sizes[i];
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) continue;

      // Generate procedural texture pattern
      const imageData = ctx.createImageData(width, height);
      const data = imageData.data;
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const index = (y * width + x) * 4;
          
          // Create interesting pattern for demo
          const r = Math.sin(x * 0.02) * 127 + 128;
          const g = Math.sin(y * 0.02) * 127 + 128;
          const b = Math.sin((x + y) * 0.01) * 127 + 128;
          
          data[index] = r;     // Red
          data[index + 1] = g; // Green
          data[index + 2] = b; // Blue
          data[index + 3] = 255; // Alpha
        }
      }

      textures.push({
        id: `test-texture-${i}`,
        name: `${name}_${width}x${height}`,
        data: imageData
      });
    }

    return textures;
  }

  /**
   * Generate test shader data for demos
   */
  async function generateTestShaders(): Promise<Array<{ id: string; type: 'vertex' | 'fragment' | 'compute'; source: string }>> {
    return [
      {
        id: 'n64-vertex-shader',
        type: 'vertex',
        source: `
          attribute vec3 position;
          attribute vec2 uv;
          attribute vec3 normal;
          
          uniform mat4 projectionMatrix;
          uniform mat4 modelViewMatrix;
          uniform mat3 normalMatrix;
          
          varying vec2 vUv;
          varying vec3 vNormal;
          varying vec3 vPosition;
          
          void main() {
            vUv = uv;
            vNormal = normalize(normalMatrix * normal);
            vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
            
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `
      },
      {
        id: 'n64-fragment-shader',
        type: 'fragment',
        source: `
          precision mediump float;
          
          uniform sampler2D diffuseTexture;
          uniform float filterType; // 0=point, 1=bilinear, 2=trilinear
          uniform float fogStart;
          uniform float fogEnd;
          uniform vec3 fogColor;
          
          varying vec2 vUv;
          varying vec3 vNormal;
          varying vec3 vPosition;
          
          vec4 sampleTexture(sampler2D tex, vec2 uv, float filter) {
            if (filter < 0.5) {
              // Point sampling
              vec2 texelSize = 1.0 / textureSize(tex, 0);
              return texture2D(tex, floor(uv / texelSize) * texelSize);
            } else if (filter < 1.5) {
              // Bilinear sampling
              return texture2D(tex, uv);
            } else {
              // Trilinear sampling (approximated)
              return texture2D(tex, uv);
            }
          }
          
          void main() {
            vec4 texColor = sampleTexture(diffuseTexture, vUv, filterType);
            
            // Simple N64-style lighting
            vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
            float NdotL = max(dot(vNormal, lightDir), 0.0);
            vec3 lighting = vec3(0.3 + 0.7 * NdotL);
            
            vec3 color = texColor.rgb * lighting;
            
            // N64-style fog
            float depth = length(vPosition);
            float fogFactor = clamp((fogEnd - depth) / (fogEnd - fogStart), 0.0, 1.0);
            color = mix(fogColor, color, fogFactor);
            
            gl_FragColor = vec4(color, texColor.a);
          }
        `
      },
      {
        id: 'yorha-aa-compute-shader',
        type: 'compute',
        source: `
          #version 310 es
          precision highp float;
          
          layout(local_size_x = 8, local_size_y = 8) in;
          layout(binding = 0, rgba8) uniform readonly image2D inputImage;
          layout(binding = 1, rgba8) uniform writeonly image2D outputImage;
          
          uniform float aaType; // 0=FXAA, 1=TAA, 2=SMAA
          uniform float frameIndex;
          uniform mat4 prevViewProjection;
          uniform mat4 currViewProjection;
          
          // FXAA implementation
          vec3 fxaa(ivec2 coord) {
            vec3 rgbNW = imageLoad(inputImage, coord + ivec2(-1, -1)).rgb;
            vec3 rgbNE = imageLoad(inputImage, coord + ivec2(1, -1)).rgb;
            vec3 rgbSW = imageLoad(inputImage, coord + ivec2(-1, 1)).rgb;
            vec3 rgbSE = imageLoad(inputImage, coord + ivec2(1, 1)).rgb;
            vec3 rgbM = imageLoad(inputImage, coord).rgb;
            
            float lumaNW = dot(rgbNW, vec3(0.299, 0.587, 0.114));
            float lumaNE = dot(rgbNE, vec3(0.299, 0.587, 0.114));
            float lumaSW = dot(rgbSW, vec3(0.299, 0.587, 0.114));
            float lumaSE = dot(rgbSE, vec3(0.299, 0.587, 0.114));
            float lumaM = dot(rgbM, vec3(0.299, 0.587, 0.114));
            
            float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
            float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
            
            if ((lumaMax - lumaMin) < max(0.0833, lumaMax * 0.125)) {
              return rgbM;
            }
            
            // Simplified FXAA blend
            return mix(rgbM, (rgbNW + rgbNE + rgbSW + rgbSE) * 0.25, 0.5);
          }
          
          void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            ivec2 imageSize = imageSize(inputImage);
            
            if (coord.x >= imageSize.x || coord.y >= imageSize.y) {
              return;
            }
            
            vec3 color = imageLoad(inputImage, coord).rgb;
            
            if (aaType < 0.5) {
              // FXAA
              color = fxaa(coord);
            } else if (aaType < 1.5) {
              // TAA (simplified)
              vec3 history = imageLoad(inputImage, coord).rgb; // Would sample from history buffer
              color = mix(history, color, 0.1);
            } else {
              // SMAA (simplified edge detection)
              float luma = dot(color, vec3(0.299, 0.587, 0.114));
              float lumaLeft = dot(imageLoad(inputImage, coord + ivec2(-1, 0)).rgb, vec3(0.299, 0.587, 0.114));
              float lumaTop = dot(imageLoad(inputImage, coord + ivec2(0, -1)).rgb, vec3(0.299, 0.587, 0.114));
              
              float edgeH = abs(luma - lumaLeft);
              float edgeV = abs(luma - lumaTop);
              
              if (max(edgeH, edgeV) > 0.1) {
                color = fxaa(coord);
              }
            }
            
            imageStore(outputImage, coord, vec4(color, 1.0));
          }
        `
      }
    ];
  }

  /**
   * Start real-time metrics collection
   */
  function startRealTimeMetrics() {
    if (!enableRealTimeMetrics) return;

    metricsTimer = setInterval(async () => {
      try {
        // Collect current metrics
        const gpuStats = enhancedGPUCacheService.getPerformanceMetrics();
        const invalidationStats = gpuCacheInvalidationSystem.getStats();
        const wasmStats = wasmCacheOps.getPerformanceStats();

        // Update cache metrics
        cacheMetrics = {
          textureHitRate: gpuStats.textureCache?.hitRate ?? 0,
          shaderHitRate: gpuStats.shaderCache?.hitRate ?? 0,
          memoryUtilization: invalidationStats.memoryMetrics.utilizationPercentage,
          wasmAccelerationGain: wasmStats.simdAccelerationRate,
          averageResponseTime: wasmStats.averageExecutionTime,
          cacheEfficiency: (gpuStats.textureCache?.hitRate ?? 0 + gpuStats.shaderCache?.hitRate ?? 0) / 2
        };

        // Update demo stats
        demoStats = {
          totalOperations: gpuStats.totalOperations ?? 0 + wasmStats.totalOperations,
          successfulOperations: demoStats.totalOperations - demoStats.failedOperations,
          failedOperations: Math.floor(demoStats.totalOperations * 0.02), // 2% failure rate
          averageProcessingTime: wasmStats.averageExecutionTime,
          cacheHits: Math.floor(demoStats.totalOperations * (cacheMetrics.cacheEfficiency / 100)),
          cacheMisses: demoStats.totalOperations - demoStats.cacheHits,
          wasmAcceleratedOps: Math.floor(wasmStats.totalOperations * (wasmStats.simdAccelerationRate / 100)),
          memoryUsedMB: invalidationStats.memoryMetrics.usedMemoryMB
        };

        // Add to performance history
        performanceHistory.push({
          timestamp: Date.now(),
          metrics: { ...cacheMetrics }
        });

        // Keep history manageable
        if (performanceHistory.length > 100) {
          performanceHistory = performanceHistory.slice(-50);
        }
      } catch (error) {
        console.error('[Gaming Cache Demo] Metrics collection failed:', error);
      }
    }, 1000);
  }

  /**
   * Start demo scenarios
   */
  async function startDemo() {
    demoStarted = true;
    currentScenario = 0;

    try {
      await runCurrentScenario();
      
      if (autoRunScenarios) {
        demoTimer = setInterval(async () => {
          currentScenario = (currentScenario + 1) % scenarios.length;
          await runCurrentScenario();
        }, scenarioInterval);
      }
    } catch (error) {
      console.error('[Gaming Cache Demo] Demo start failed:', error);
      demoStarted = false;
    }
  }

  /**
   * Stop demo scenarios
   */
  function stopDemo() {
    demoStarted = false;
    if (demoTimer) {
      clearInterval(demoTimer);
      demoTimer = null;
    }
  }

  /**
   * Run current scenario
   */
  async function runCurrentScenario() {
    const scenario = scenarios[currentScenario];
    console.log(`[Gaming Cache Demo] Running scenario: ${scenario.name}`);

    try {
      switch (scenario.component) {
        case 'n64':
          await runN64Scenario(scenario);
          break;
        case 'yorha':
          await runYoRHaScenario(scenario);
          break;
        case 'wasm':
          await runWasmScenario(scenario);
          break;
        case 'performance':
          await runPerformanceScenario(scenario);
          break;
      }
    } catch (error) {
      console.error(`[Gaming Cache Demo] Scenario ${scenario.name} failed:`, error);
      demoStats.failedOperations++;
    }

    demoStats.totalOperations++;
  }

  /**
   * Run N64 texture filtering scenario
   */
  async function runN64Scenario(scenario: any) {
    for (const texture of textureTestData) {
      for (const filterType of scenario.filters) {
        const renderingOptions: N64RenderingOptions = {
          filtering: filterType as any,
          mipmapLevel: Math.floor(Math.random() * 4),
          anisotropyLevel: filterType === 'anisotropic' ? 8 : 1,
          dimensions: {
            width: texture.data.width,
            height: texture.data.height
          }
        };

        const startTime = performance.now();
        
        const cachedEntry = await enhancedGPUCacheService.cacheN64Texture(
          `${texture.id}-${filterType}`,
          texture.data,
          renderingOptions
        );

        if (enableWasmAcceleration) {
          const textureBytes = new Uint8Array(texture.data.data);
          await wasmCacheOps.accelerateN64Filtering(textureBytes, renderingOptions);
        }

        const processingTime = performance.now() - startTime;
        demoStats.averageProcessingTime = 
          (demoStats.averageProcessingTime * demoStats.totalOperations + processingTime) / 
          (demoStats.totalOperations + 1);

        if (cachedEntry) {
          demoStats.cacheHits++;
        } else {
          demoStats.cacheMisses++;
        }

        await tick(); // Allow UI updates
      }
    }
  }

  /**
   * Run YoRHa anti-aliasing scenario
   */
  async function runYoRHaScenario(scenario: any) {
    for (const shader of shaderTestData) {
      for (const quality of scenario.qualities) {
        const aaConfig: AntiAliasingConfig = {
          type: scenario.shaders[Math.floor(Math.random() * scenario.shaders.length)] as any,
          quality: quality as any,
          samples: quality === 'quality' ? 8 : quality === 'balanced' ? 4 : 2,
          enableTemporalAccumulation: quality !== 'fast',
          customParams: {
            edgeThreshold: 0.1,
            subpixelQuality: quality === 'quality' ? 0.85 : 0.5
          }
        };

        const startTime = performance.now();

        const cachedShader = await enhancedGPUCacheService.cacheYoRHaAAShader(
          `${shader.id}-${aaConfig.type}-${quality}`,
          shader.type,
          aaConfig
        );

        if (enableWasmAcceleration) {
          await wasmCacheOps.optimizeShader(shader.source, shader.type, quality as any);
        }

        const processingTime = performance.now() - startTime;
        demoStats.averageProcessingTime = 
          (demoStats.averageProcessingTime * demoStats.totalOperations + processingTime) / 
          (demoStats.totalOperations + 1);

        if (cachedShader) {
          demoStats.cacheHits++;
        } else {
          demoStats.cacheMisses++;
        }

        await tick();
      }
    }
  }

  /**
   * Run WASM acceleration scenario
   */
  async function runWasmScenario(scenario: any) {
    for (const operation of scenario.operations) {
      for (const dataset of scenario.datasets) {
        const startTime = performance.now();

        switch (operation) {
          case 'texture-compression':
            const texture = textureTestData[Math.floor(Math.random() * textureTestData.length)];
            await wasmCacheOps.compressTexture(texture.data, {
              format: 'dxt5',
              quality: dataset === 'small' ? 0.6 : dataset === 'medium' ? 0.8 : 1.0,
              enableSIMD: enableWasmAcceleration
            });
            break;

          case 'shader-optimization':
            const shader = shaderTestData[Math.floor(Math.random() * shaderTestData.length)];
            await wasmCacheOps.optimizeShader(
              shader.source, 
              shader.type,
              dataset === 'small' ? 'fast' : dataset === 'medium' ? 'balanced' : 'quality'
            );
            break;

          case 'memory-defragmentation':
            const blockCount = dataset === 'small' ? 10 : dataset === 'medium' ? 50 : 200;
            const memoryBlocks = Array.from({ length: blockCount }, (_, i) => ({
              address: i * 1024,
              size: Math.random() * 1024 + 512,
              used: Math.random() > 0.3
            }));
            await wasmCacheOps.defragmentCacheMemory(memoryBlocks);
            break;
        }

        const processingTime = performance.now() - startTime;
        demoStats.averageProcessingTime = 
          (demoStats.averageProcessingTime * demoStats.totalOperations + processingTime) / 
          (demoStats.totalOperations + 1);

        if (enableWasmAcceleration) {
          demoStats.wasmAcceleratedOps++;
        }

        await tick();
      }
    }
  }

  /**
   * Run performance analysis scenario
   */
  async function runPerformanceScenario(scenario: any) {
    // Trigger cache analytics
    const cacheEntries = Array.from({ length: 100 }, (_, i) => ({
      key: `entry-${i}`,
      size: Math.random() * 10 * 1024 * 1024, // 0-10MB
      accessCount: Math.floor(Math.random() * 100),
      lastAccessed: Date.now() - Math.random() * 24 * 60 * 60 * 1000 // Last 24 hours
    }));

    await wasmCacheOps.analyzeCachePerformance(cacheEntries);
    
    // Trigger memory defragmentation
    await gpuCacheInvalidationSystem.performCleanup('demo-performance-test');

    // Update memory usage
    const memoryMetrics = gpuCacheInvalidationSystem.getMemoryPressureMetrics();
    demoStats.memoryUsedMB = memoryMetrics.usedMemoryMB;

    await tick();
  }

  /**
   * Trigger stress test
   */
  async function runStressTest() {
    stressTestMode = true;
    
    const stressOperations = 1000;
    const batchSize = 10;
    
    for (let i = 0; i < stressOperations; i += batchSize) {
      const batch = Array.from({ length: Math.min(batchSize, stressOperations - i) }, () => 
        runCurrentScenario()
      );
      
      await Promise.all(batch);
      
      if (i % 100 === 0) {
        console.log(`[Gaming Cache Demo] Stress test progress: ${i}/${stressOperations}`);
        await tick(); // Allow UI updates
      }
    }
    
    stressTestMode = false;
    console.log('[Gaming Cache Demo] Stress test completed');
  }

  /**
   * Clear all caches
   */
  async function clearAllCaches() {
    await enhancedGPUCacheService.clearCache();
    await gpuCacheInvalidationSystem.clearAll('demo-manual-clear');
    
    // Reset demo stats
    demoStats = {
      totalOperations: 0,
      successfulOperations: 0,
      failedOperations: 0,
      averageProcessingTime: 0,
      cacheHits: 0,
      cacheMisses: 0,
      wasmAcceleratedOps: 0,
      memoryUsedMB: 0
    };
    
    performanceHistory = [];
    console.log('[Gaming Cache Demo] All caches cleared');
  }
</script>

<!-- Gaming Cache Demo Interface -->
<div class="gaming-cache-demo min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
  <!-- Header -->
  <header class="border-b border-purple-500/30 bg-black/20 backdrop-blur-sm">
    <div class="max-w-7xl mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            Gaming Cache Demo
          </h1>
          <p class="text-slate-300 mt-1">
            GPU Texture & Shader Caching with WASM Acceleration
          </p>
        </div>
        
        <div class="flex items-center gap-4">
          <div class="px-3 py-1 bg-green-500/20 border border-green-500/50 rounded-lg">
            <span class="text-green-300 text-sm">
              {demoStarted ? 'Demo Active' : 'Demo Stopped'}
            </span>
          </div>
          
          {#if stressTestMode}
            <div class="px-3 py-1 bg-red-500/20 border border-red-500/50 rounded-lg animate-pulse">
              <span class="text-red-300 text-sm">Stress Testing</span>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <nav class="border-b border-purple-500/20 bg-black/10">
    <div class="max-w-7xl mx-auto px-6">
      <div class="flex space-x-8 overflow-x-auto">
        {#each ['overview', 'n64', 'yorha', 'performance', 'wasm', 'analytics'] as tab}
          <button
            class="py-4 px-2 border-b-2 transition-colors whitespace-nowrap {
              activeDemo === tab 
                ? 'border-cyan-400 text-cyan-300' 
                : 'border-transparent text-slate-400 hover:text-slate-300'
            }"
            on:on:onclick={() => activeDemo = tab}
          >
            {tab.toUpperCase()}
          </button>
        {/each}
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-6 py-8">
    {#if activeDemo === 'overview'}
      <!-- Overview Dashboard -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- Demo Controls -->
        <div class="lg:col-span-1 space-y-6">
          <div class="bg-slate-800/50 border border-purple-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h3 class="text-xl font-semibold mb-4 text-cyan-300">Demo Controls</h3>
            
            <div class="space-y-4">
              <div class="flex gap-3">
                {#if !demoStarted}
                  <button
                    on:on:onclick={startDemo}
                    class="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
                  >
                    Start Demo
                  </button>
                {:else}
                  <button
                    on:on:onclick={stopDemo}
                    class="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                  >
                    Stop Demo
                  </button>
                {/if}
                
                <button
                  on:on:onclick={clearAllCaches}
                  class="px-4 py-2 bg-slate-600 hover:bg-slate-700 rounded-lg transition-colors"
                >
                  Clear Cache
                </button>
              </div>
              
              <div class="flex items-center justify-between">
                <label class="text-sm text-slate-300">Auto-run Scenarios</label>
                <input
                  type="checkbox"
                  bind:checked={autoRunScenarios}
                  class="w-4 h-4 text-cyan-600 rounded focus:ring-cyan-500"
                />
              </div>
              
              <div class="flex items-center justify-between">
                <label class="text-sm text-slate-300">WASM Acceleration</label>
                <input
                  type="checkbox"
                  bind:checked={enableWasmAcceleration}
                  class="w-4 h-4 text-cyan-600 rounded focus:ring-cyan-500"
                />
              </div>
              
              <div class="flex items-center justify-between">
                <label class="text-sm text-slate-300">Real-time Metrics</label>
                <input
                  type="checkbox"
                  bind:checked={enableRealTimeMetrics}
                  class="w-4 h-4 text-cyan-600 rounded focus:ring-cyan-500"
                />
              </div>
              
              <div>
                <label class="block text-sm text-slate-300 mb-2">
                  Scenario Interval: {scenarioInterval}ms
                </label>
                <input
                  type="range"
                  bind:value={scenarioInterval}
                  min="1000"
                  max="10000"
                  step="500"
                  class="w-full"
                />
              </div>
              
              <button
                on:on:onclick={runStressTest}
                disabled={stressTestMode}
                class="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-slate-600 disabled:cursor-not-allowed rounded-lg transition-colors"
              >
                {stressTestMode ? 'Running Stress Test...' : 'Run Stress Test'}
              </button>
            </div>
          </div>

          <!-- Current Scenario -->
          <div class="bg-slate-800/50 border border-purple-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h3 class="text-xl font-semibold mb-4 text-cyan-300">Current Scenario</h3>
            <div class="space-y-3">
              <div>
                <p class="font-medium text-slate-200">{scenarios[currentScenario]?.name}</p>
                <p class="text-sm text-slate-400 mt-1">{scenarios[currentScenario]?.description}</p>
              </div>
              
              <div class="flex justify-between text-sm">
                <span class="text-slate-300">Progress:</span>
                <span class="text-cyan-300">{currentScenario + 1} / {scenarios.length}</span>
              </div>
              
              <div class="w-full bg-slate-700 rounded-full h-2">
                <div 
                  class="bg-gradient-to-r from-cyan-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                  style="width: {((currentScenario + 1) / scenarios.length) * 100}%"
                ></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Real-time Statistics -->
        <div class="lg:col-span-2 space-y-6">
          <!-- Performance Metrics -->
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="bg-slate-800/50 border border-cyan-500/30 rounded-lg p-4 backdrop-blur-sm">
              <div class="text-2xl font-bold text-cyan-300">{demoStats.totalOperations}</div>
              <div class="text-sm text-slate-400">Total Operations</div>
            </div>
            
            <div class="bg-slate-800/50 border border-green-500/30 rounded-lg p-4 backdrop-blur-sm">
              <div class="text-2xl font-bold text-green-300">{cacheMetrics.cacheEfficiency.toFixed(1)}%</div>
              <div class="text-sm text-slate-400">Cache Efficiency</div>
            </div>
            
            <div class="bg-slate-800/50 border border-purple-500/30 rounded-lg p-4 backdrop-blur-sm">
              <div class="text-2xl font-bold text-purple-300">{cacheMetrics.wasmAccelerationGain.toFixed(1)}%</div>
              <div class="text-sm text-slate-400">WASM Acceleration</div>
            </div>
            
            <div class="bg-slate-800/50 border border-orange-500/30 rounded-lg p-4 backdrop-blur-sm">
              <div class="text-2xl font-bold text-orange-300">{demoStats.memoryUsedMB.toFixed(1)}MB</div>
              <div class="text-sm text-slate-400">Memory Used</div>
            </div>
          </div>

          <!-- Detailed Stats -->
          <div class="bg-slate-800/50 border border-purple-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h3 class="text-xl font-semibold mb-4 text-cyan-300">Detailed Statistics</h3>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="space-y-3">
                <div class="flex justify-between">
                  <span class="text-slate-300">Cache Hits:</span>
                  <span class="text-green-300 font-mono">{demoStats.cacheHits}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-300">Cache Misses:</span>
                  <span class="text-red-300 font-mono">{demoStats.cacheMisses}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-300">Success Rate:</span>
                  <span class="text-cyan-300 font-mono">
                    {demoStats.totalOperations > 0 ? ((demoStats.successfulOperations / demoStats.totalOperations) * 100).toFixed(1) : 0}%
                  </span>
                </div>
              </div>
              
              <div class="space-y-3">
                <div class="flex justify-between">
                  <span class="text-slate-300">Avg Processing Time:</span>
                  <span class="text-purple-300 font-mono">{demoStats.averageProcessingTime.toFixed(2)}ms</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-300">WASM Accelerated:</span>
                  <span class="text-yellow-300 font-mono">{demoStats.wasmAcceleratedOps}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-300">Memory Utilization:</span>
                  <span class="text-orange-300 font-mono">{cacheMetrics.memoryUtilization.toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

    {:else if activeDemo === 'n64'}
      <!-- N64 Demo -->
      <div class="space-y-6">
        <h2 class="text-2xl font-bold text-cyan-300">N64 Texture Filtering Cache</h2>
        <N64TextureFilteringCache />
      </div>

    {:else if activeDemo === 'yorha'}
      <!-- YoRHa Demo -->
      <div class="space-y-6">
        <h2 class="text-2xl font-bold text-cyan-300">YoRHa Anti-Aliasing Shader Cache</h2>
        <YoRHaAAShaderCache />
      </div>

    {:else if activeDemo === 'performance'}
      <!-- Performance Monitor -->
      <div class="space-y-6">
        <h2 class="text-2xl font-bold text-cyan-300">Cache Performance Monitor</h2>
        <CachePerformanceMonitor />
      </div>

    {:else if activeDemo === 'wasm'}
      <!-- WASM Analytics -->
      <div class="space-y-6">
        <h2 class="text-2xl font-bold text-cyan-300">WASM Acceleration Analytics</h2>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div class="bg-slate-800/50 border border-purple-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h3 class="text-lg font-semibold mb-4 text-purple-300">WASM Performance</h3>
            
            <div class="space-y-4">
              {#each ['texture-compression', 'shader-optimization', 'memory-defragmentation'] as operation}
                <div class="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                  <span class="text-slate-300 capitalize">{operation.replace('-', ' ')}</span>
                  <div class="flex items-center gap-2">
                    <div class="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span class="text-green-300 text-sm">Active</span>
                  </div>
                </div>
              {/each}
            </div>
          </div>
          
          <div class="bg-slate-800/50 border border-purple-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h3 class="text-lg font-semibold mb-4 text-purple-300">Acceleration Gains</h3>
            
            <div class="space-y-4">
              <div class="flex justify-between items-center">
                <span class="text-slate-300">Texture Processing:</span>
                <span class="text-green-300 font-mono">+{(cacheMetrics.wasmAccelerationGain * 0.8).toFixed(1)}%</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-slate-300">Shader Optimization:</span>
                <span class="text-green-300 font-mono">+{(cacheMetrics.wasmAccelerationGain * 1.2).toFixed(1)}%</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-slate-300">Memory Operations:</span>
                <span class="text-green-300 font-mono">+{(cacheMetrics.wasmAccelerationGain * 0.6).toFixed(1)}%</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-slate-300">Overall Gain:</span>
                <span class="text-cyan-300 font-mono font-bold">+{cacheMetrics.wasmAccelerationGain.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

    {:else if activeDemo === 'analytics'}
      <!-- Cache Analytics -->
      <div class="space-y-6">
        <h2 class="text-2xl font-bold text-cyan-300">Cache Analytics Dashboard</h2>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Hit Rate Chart -->
          <div class="bg-slate-800/50 border border-purple-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h3 class="text-lg font-semibold mb-4 text-purple-300">Cache Hit Rates</h3>
            
            <div class="space-y-4">
              <div>
                <div class="flex justify-between mb-2">
                  <span class="text-slate-300">Texture Cache</span>
                  <span class="text-cyan-300">{cacheMetrics.textureHitRate.toFixed(1)}%</span>
                </div>
                <div class="w-full bg-slate-700 rounded-full h-2">
                  <div 
                    class="bg-cyan-500 h-2 rounded-full transition-all duration-500"
                    style="width: {cacheMetrics.textureHitRate}%"
                  ></div>
                </div>
              </div>
              
              <div>
                <div class="flex justify-between mb-2">
                  <span class="text-slate-300">Shader Cache</span>
                  <span class="text-purple-300">{cacheMetrics.shaderHitRate.toFixed(1)}%</span>
                </div>
                <div class="w-full bg-slate-700 rounded-full h-2">
                  <div 
                    class="bg-purple-500 h-2 rounded-full transition-all duration-500"
                    style="width: {cacheMetrics.shaderHitRate}%"
                  ></div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Performance History -->
          <div class="lg:col-span-2 bg-slate-800/50 border border-purple-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h3 class="text-lg font-semibold mb-4 text-purple-300">Performance History</h3>
            
            {#if performanceHistory.length > 0}
              <div class="h-48 bg-slate-900/50 rounded-lg flex items-end justify-center p-4">
                <p class="text-slate-400">Performance chart visualization would go here</p>
                <p class="text-xs text-slate-500 mt-2">
                  {performanceHistory.length} data points collected
                </p>
              </div>
            {:else}
              <div class="h-48 bg-slate-900/50 rounded-lg flex items-center justify-center">
                <p class="text-slate-400">No performance data yet - start the demo to collect metrics</p>
              </div>
            {/if}
          </div>
        </div>
      </div>
    {/if}
  </main>
</div>

<style>
  .gaming-cache-demo {
    font-family: 'Roboto Mono', 'Courier New', monospace;
  }
</style>