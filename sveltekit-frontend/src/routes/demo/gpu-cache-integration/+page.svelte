<!--
  GPU Cache Integration Demo Page
  Showcases the complete GPU cache system integration with gaming UI progression
-->

<script lang="ts">
  import GPUCacheIntegrationDemo from '$lib/components/ui/gaming/demo/GPUCacheIntegrationDemo.svelte';
  import { page } from '$app/stores';
  import { onMount } from 'svelte';

  let debugMode = $state(false);
  let showAdvancedMetrics = $state(false);

  onMount(() => {
    // Check URL params for demo configuration
    const urlParams = new URLSearchParams(window.location.search);
    debugMode = urlParams.get('debug') === 'true';
    showAdvancedMetrics = urlParams.get('advanced') === 'true';
  });
</script>

<svelte:head>
  <title>GPU Cache Integration Demo - YoRHa Legal AI</title>
  <meta name="description" content="Interactive demonstration of the GPU cache system with gaming UI progression from 8-bit to YoRHa era" />
</svelte:head>

<div class="demo-page-container">
  <!-- Page Header -->
  <header class="demo-header mb-8">
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-4xl font-bold text-white mb-2">GPU Cache Integration Demo</h1>
        <p class="text-gray-300 text-lg">
          Interactive showcase of the unified GPU cache system with gaming UI progression
        </p>
      </div>
      <div class="demo-controls">
        <label class="flex items-center gap-2 text-white">
          <input 
            type="checkbox" 
            bind:checked={debugMode}
            class="gpu-cache-checkbox"
          />
          Debug Mode
        </label>
        <label class="flex items-center gap-2 text-white">
          <input 
            type="checkbox" 
            bind:checked={showAdvancedMetrics}
            class="gpu-cache-checkbox"
          />
          Advanced Metrics
        </label>
      </div>
    </div>
  </header>

  <!-- Main Demo Component -->
  <GPUCacheIntegrationDemo 
    showProgressionDemo={true}
    enableRealTimeMetrics={true}
    {debugMode}
  />

  <!-- Technical Details Section -->
  <section class="technical-details mt-12">
    <h2 class="text-2xl font-bold text-white mb-6">Technical Implementation</h2>
    <div class="details-grid">
      <div class="detail-card">
        <h3 class="text-lg font-semibold text-white mb-3">NES Memory Architecture</h3>
        <ul class="text-gray-300 space-y-1 text-sm">
          <li>• PRG_ROM: Program code storage</li>
          <li>• CHR_ROM: Character/graphics data</li>
          <li>• RAM: Working memory (2KB)</li>
          <li>• PPU_MEMORY: Picture processing unit</li>
          <li>• SPRITE_MEMORY: Object attribute memory</li>
          <li>• PALETTE_MEMORY: Color palette data</li>
        </ul>
      </div>

      <div class="detail-card">
        <h3 class="text-lg font-semibold text-white mb-3">GPU Cache Integration</h3>
        <ul class="text-gray-300 space-y-1 text-sm">
          <li>• WebGPU acceleration support</li>
          <li>• WASM-powered cache operations</li>
          <li>• Binary encoding optimization</li>
          <li>• Real-time performance monitoring</li>
          <li>• Adaptive quality adjustment</li>
          <li>• XState machine orchestration</li>
        </ul>
      </div>

      <div class="detail-card">
        <h3 class="text-lg font-semibold text-white mb-3">N64 Texture Filtering</h3>
        <ul class="text-gray-300 space-y-1 text-sm">
          <li>• Bilinear filtering support</li>
          <li>• Trilinear mipmap filtering</li>
          <li>• Anisotropic filtering (16x)</li>
          <li>• Texture compression (S3TC, ASTC)</li>
          <li>• Multi-threaded processing</li>
          <li>• GPU memory optimization</li>
        </ul>
      </div>

      <div class="detail-card">
        <h3 class="text-lg font-semibold text-white mb-3">YoRHa Quantum Effects</h3>
        <ul class="text-gray-300 space-y-1 text-sm">
          <li>• Holographic interface elements</li>
          <li>• Quantum particle animations</li>
          <li>• Advanced shader effects</li>
          <li>• Real-time ray tracing</li>
          <li>• Procedural texture generation</li>
          <li>• AI-powered visual optimization</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- Performance Characteristics -->
  {#if showAdvancedMetrics}
    <section class="performance-section mt-12">
      <h2 class="text-2xl font-bold text-white mb-6">Performance Characteristics</h2>
      <div class="performance-grid">
        <div class="performance-chart">
          <h4 class="text-white font-semibold mb-2">Cache Hit Rates by Era</h4>
          <div class="chart-container">
            <div class="chart-bar" style="height: 60%;">
              <span class="chart-label">NES (60%)</span>
            </div>
            <div class="chart-bar" style="height: 75%;">
              <span class="chart-label">SNES (75%)</span>
            </div>
            <div class="chart-bar" style="height: 90%;">
              <span class="chart-label">N64 (90%)</span>
            </div>
            <div class="chart-bar" style="height: 95%;">
              <span class="chart-label">YoRHa (95%)</span>
            </div>
          </div>
        </div>

        <div class="performance-metrics">
          <h4 class="text-white font-semibold mb-2">System Metrics</h4>
          <div class="metric-row">
            <span class="metric-name">Memory Usage:</span>
            <span class="metric-value">124 MB / 512 MB</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">GPU Utilization:</span>
            <span class="metric-value">67%</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Cache Efficiency:</span>
            <span class="metric-value">92.3%</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Frame Rate:</span>
            <span class="metric-value">60 FPS</span>
          </div>
        </div>
      </div>
    </section>
  {/if}

  <!-- Code Examples -->
  <section class="code-examples mt-12">
    <h2 class="text-2xl font-bold text-white mb-6">Usage Examples</h2>
    <div class="code-tabs">
      <div class="code-tab active">CSS Integration</div>
      <div class="code-example">
        <pre><code class="language-css">{`/* Using GPU Cache CSS Custom Properties */
.my-component {
  background: var(--gpu-cache-bg-primary);
  border: 1px solid var(--gpu-cache-border-primary);
  color: var(--gpu-cache-text-primary);
}

/* NES Memory Bank Styling */
.nes-memory-bank.nes-prg_rom {
  border-color: var(--nes-prg-rom-color);
  background: var(--nes-prg-rom-bg);
}

/* Cache Status Indicators */
.cache-status.idle {
  background: var(--gpu-cache-state-idle);
}

.cache-status.processing {
  background: var(--gpu-cache-state-processing);
  animation: var(--gpu-cache-animation-processing);
}`}</code></pre>
      </div>
    </div>
  </section>
</div>

<style>
  .demo-page-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    padding: 2rem;
  }

  .demo-header {
    border-bottom: 1px solid var(--gpu-cache-border-primary);
    padding-bottom: 2rem;
  }

  .demo-controls {
    display: flex;
    gap: 1rem;
    align-items: center;
  }

  .gpu-cache-checkbox {
    accent-color: var(--gpu-cache-accent-primary);
  }

  .details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
  }

  .detail-card {
    background: var(--gpu-cache-bg-secondary);
    border: 1px solid var(--gpu-cache-border-secondary);
    padding: 1.5rem;
    border-radius: 8px;
  }

  .performance-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }

  .chart-container {
    display: flex;
    align-items: end;
    gap: 1rem;
    height: 200px;
    padding: 1rem;
    background: var(--gpu-cache-bg-tertiary);
    border-radius: 4px;
  }

  .chart-bar {
    flex: 1;
    background: linear-gradient(180deg, var(--gpu-cache-accent-primary), var(--gpu-cache-accent-secondary));
    border-radius: 2px 2px 0 0;
    position: relative;
    display: flex;
    align-items: end;
    justify-content: center;
    padding-bottom: 0.5rem;
  }

  .chart-label {
    color: white;
    font-size: 0.75rem;
    font-weight: bold;
    writing-mode: vertical-rl;
    text-orientation: mixed;
  }

  .performance-metrics {
    background: var(--gpu-cache-bg-secondary);
    border: 1px solid var(--gpu-cache-border-secondary);
    padding: 1.5rem;
    border-radius: 8px;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--gpu-cache-border-tertiary);
  }

  .metric-name {
    color: var(--gpu-cache-text-secondary);
  }

  .metric-value {
    color: var(--gpu-cache-text-primary);
    font-family: monospace;
    font-weight: bold;
  }

  .code-example {
    background: var(--gpu-cache-bg-tertiary);
    border: 1px solid var(--gpu-cache-border-secondary);
    border-radius: 0 4px 4px 4px;
    overflow: auto;
  }

  .code-example pre {
    margin: 0;
    padding: 1.5rem;
  }

  .code-example code {
    color: var(--gpu-cache-text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.875rem;
    line-height: 1.5;
  }

  .code-tabs {
    margin-top: 1rem;
  }

  .code-tab {
    background: var(--gpu-cache-bg-secondary);
    border: 1px solid var(--gpu-cache-border-secondary);
    border-bottom: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px 4px 0 0;
    color: var(--gpu-cache-text-primary);
    font-weight: bold;
    display: inline-block;
  }

  .code-tab.active {
    background: var(--gpu-cache-accent-primary);
    color: var(--gpu-cache-bg-primary);
  }
</style>