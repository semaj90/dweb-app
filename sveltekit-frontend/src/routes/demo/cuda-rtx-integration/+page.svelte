<script lang="ts">
  import { onMount } from 'svelte';
  import { cudaService } from '$lib/services/cuda-service';
  import type { CudaHealthStatus, CudaStats } from '$lib/services/cuda-service';

  let health = $state<CudaHealthStatus | null>(null);
  let stats = $state<CudaStats | null>(null);
  let tensorCoreInfo = $state<any>(null);
  let loading = $state(true);
  let computeResult = $state<any>(null);
  let computeLoading = $state(false);
  let testQuery = $state('Analyze legal document with RTX tensor core optimization');

  onMount(async () => {
    await loadCudaInfo();
    setInterval(loadCudaInfo, 5000); // Refresh every 5 seconds
  });

  async function loadCudaInfo() {
    try {
      loading = true;
      const [healthResult, statsResult, tensorResult] = await Promise.all([
        cudaService.getHealth(),
        cudaService.getStats(),
        cudaService.getTensorCoreInfo()
      ]);

      health = healthResult;
      stats = 'cuda_stats' in statsResult ? statsResult : null;
      tensorCoreInfo = tensorResult;
    } catch (error) {
      console.error('Failed to load CUDA info:', error);
    } finally {
      loading = false;
    }
  }

  async function testCompute() {
    if (!testQuery.trim()) return;
    
    computeLoading = true;
    computeResult = null;

    try {
      const result = await cudaService.enhancedCompute(testQuery, {
        use_tensor_cores: true,
        quantization: '4bit',
        negative_latent_space: true,
        graph_dimensions: '4D'
      });

      computeResult = result;
    } catch (error) {
      computeResult = {
        error: 'Computation failed',
        details: error instanceof Error ? error.message : String(error)
      };
    } finally {
      computeLoading = false;
    }
  }
</script>

<div class="min-h-screen bg-nier-bg-primary text-nier-text-primary p-golden-lg">
  <div class="container mx-auto max-w-6xl">
    <!-- Header -->
    <header class="mb-golden-xl">
      <h1 class="text-4xl font-bold text-nier-accent-warm mb-golden-md uppercase tracking-wider">
        🚀 CUDA RTX Integration Demo
      </h1>
      <p class="text-nier-text-secondary mb-golden-md">
        Real-time RTX 3060 Ti Tensor Core processing with 4-bit quantization and negative latent space
      </p>
      
      {#if loading}
        <div class="bg-yellow-500/20 text-yellow-400 border-yellow-500/30 border px-golden-md py-golden-sm rounded">
          🔄 Loading CUDA system information...
        </div>
      {:else if health?.status === 'healthy'}
        <div class="bg-green-500/20 text-green-400 border-green-500/30 border px-golden-md py-golden-sm rounded">
          ✅ CUDA System Online - RTX Tensor Cores Active
        </div>
      {:else}
        <div class="bg-red-500/20 text-red-400 border-red-500/30 border px-golden-md py-golden-sm rounded">
          ❌ CUDA System Unavailable
        </div>
      {/if}
    </header>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-golden-lg">
      <!-- System Status Panel -->
      <div class="bg-nier-bg-secondary border border-nier-border-primary rounded p-golden-lg">
        <h2 class="text-2xl font-bold text-nier-accent-warm mb-golden-md uppercase">
          🎯 System Status
        </h2>
        
        {#if health}
          <div class="space-y-golden-sm">
            <div class="flex justify-between">
              <span class="text-nier-text-secondary">Status:</span>
              <span class="font-mono text-nier-accent-warm">{health.status.toUpperCase()}</span>
            </div>
            
            {#if health.gpu_info}
              <div class="flex justify-between">
                <span class="text-nier-text-secondary">GPU Count:</span>
                <span class="font-mono text-nier-text-primary">{health.gpu_info.device_count}</span>
              </div>
              
              <div class="flex justify-between">
                <span class="text-nier-text-secondary">Compute Capability:</span>
                <span class="font-mono text-nier-accent-warm">{health.gpu_info.compute_capability}</span>
              </div>
              
              <div class="flex justify-between">
                <span class="text-nier-text-secondary">Queue Size:</span>
                <span class="font-mono text-nier-text-primary">{health.gpu_info.queue_size}</span>
              </div>
            {/if}
          </div>
        {/if}

        {#if tensorCoreInfo}
          <div class="mt-golden-md pt-golden-md border-t border-nier-border-muted">
            <h3 class="text-lg font-bold text-nier-accent-warm mb-golden-sm uppercase">
              ⚡ Tensor Core Info
            </h3>
            <div class="space-y-golden-xs">
              <div class="flex justify-between">
                <span class="text-nier-text-secondary">Generation:</span>
                <span class="font-mono text-nier-accent-cool">{tensorCoreInfo.tensor_core_generation}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-nier-text-secondary">Performance:</span>
                <span class="font-mono text-green-400">{tensorCoreInfo.performance_estimate}</span>
              </div>
            </div>
          </div>
        {/if}
      </div>

      <!-- Performance Stats Panel -->
      <div class="bg-nier-bg-secondary border border-nier-border-primary rounded p-golden-lg">
        <h2 class="text-2xl font-bold text-nier-accent-warm mb-golden-md uppercase">
          📊 Performance Stats
        </h2>
        
        {#if stats}
          <div class="space-y-golden-sm">
            <div class="flex justify-between">
              <span class="text-nier-text-secondary">Uptime:</span>
              <span class="font-mono text-nier-text-primary">{stats.cuda_stats.uptime}</span>
            </div>
            
            <div class="flex justify-between">
              <span class="text-nier-text-secondary">Cache Hit Rate:</span>
              <span class="font-mono text-nier-accent-warm">{(stats.cuda_stats.cache_stats.hit_rate * 100).toFixed(1)}%</span>
            </div>
            
            <div class="flex justify-between">
              <span class="text-nier-text-secondary">Queue Status:</span>
              <span class="font-mono text-green-400">{stats.cuda_stats.queue_stats.online ? 'ONLINE' : 'OFFLINE'}</span>
            </div>

            {#if stats.cuda_stats.t5_config}
              <div class="mt-golden-md pt-golden-md border-t border-nier-border-muted">
                <h3 class="text-lg font-bold text-nier-accent-warm mb-golden-sm uppercase">
                  🧠 T5 Model Config
                </h3>
                <div class="space-y-golden-xs">
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Model Size:</span>
                    <span class="font-mono text-nier-text-primary">{stats.cuda_stats.t5_config.model_size.toUpperCase()}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Hidden Size:</span>
                    <span class="font-mono text-nier-text-primary">{stats.cuda_stats.t5_config.hidden_size}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">GPU Enabled:</span>
                    <span class="font-mono text-green-400">{stats.cuda_stats.t5_config.use_gpu ? 'YES' : 'NO'}</span>
                  </div>
                </div>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    </div>

    <!-- Compute Test Panel -->
    <div class="mt-golden-lg bg-nier-bg-secondary border border-nier-border-primary rounded p-golden-lg">
      <h2 class="text-2xl font-bold text-nier-accent-warm mb-golden-md uppercase">
        🔬 RTX Tensor Core Compute Test
      </h2>
      
      <div class="space-y-golden-md">
        <div>
          <label for="testQuery" class="block text-nier-text-secondary mb-golden-sm font-bold">
            Test Query (with 4-bit quantization & negative latent space):
          </label>
          <textarea
            id="testQuery"
            bind:value={testQuery}
            class="w-full bg-nier-bg-tertiary border border-nier-border-muted rounded p-golden-md text-nier-text-primary font-mono"
            rows="3"
            placeholder="Enter a query to test RTX tensor core processing..."
          ></textarea>
        </div>

        <button
          onclick={testCompute}
          disabled={computeLoading || !health?.cuda_available}
          class="bg-gradient-to-r from-nier-accent-warm to-nier-accent-cool text-nier-bg-primary font-bold px-golden-lg py-golden-md rounded uppercase tracking-wide hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {#if computeLoading}
            🔄 Processing with RTX Tensor Cores...
          {:else}
            ⚡ Run RTX Compute Test
          {/if}
        </button>

        {#if computeResult}
          <div class="bg-nier-bg-tertiary border border-nier-border-muted rounded p-golden-md">
            <h3 class="text-lg font-bold text-nier-accent-warm mb-golden-sm uppercase">
              📋 Compute Result
            </h3>
            <pre class="text-sm text-nier-text-primary font-mono whitespace-pre-wrap overflow-x-auto">
{JSON.stringify(computeResult, null, 2)}
            </pre>
          </div>
        {/if}
      </div>
    </div>

    <!-- Technical Details -->
    <div class="mt-golden-lg bg-nier-bg-secondary border border-nier-border-primary rounded p-golden-lg">
      <h2 class="text-2xl font-bold text-nier-accent-warm mb-golden-md uppercase">
        🔧 Technical Implementation
      </h2>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-golden-md text-sm">
        <div>
          <h3 class="font-bold text-nier-accent-warm mb-golden-sm">RTX Optimization Features:</h3>
          <ul class="space-y-1 text-nier-text-secondary">
            <li>✅ 4-bit quantization (75% memory reduction)</li>
            <li>✅ Negative latent space for 3D/4D graph search</li>
            <li>✅ Multi-stream CUDA processing</li>
            <li>✅ Tensor Core acceleration</li>
            <li>✅ Memory pool optimization</li>
          </ul>
        </div>
        
        <div>
          <h3 class="font-bold text-nier-accent-warm mb-golden-sm">Architecture:</h3>
          <ul class="space-y-1 text-nier-text-secondary">
            <li>🎯 SvelteKit → API → CUDA Server</li>
            <li>🎯 RTX 3060 Ti (8.6 compute capability)</li>
            <li>🎯 Real-time health monitoring</li>
            <li>🎯 TypeScript service integration</li>
            <li>🎯 Enhanced error handling</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* Custom scrollbar for code blocks */
  pre::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  
  pre::-webkit-scrollbar-track {
    background: rgba(61, 61, 61, 0.2);
  }
  
  pre::-webkit-scrollbar-thumb {
    background: rgba(213, 182, 120, 0.5);
    border-radius: 4px;
  }
  
  pre::-webkit-scrollbar-thumb:hover {
    background: rgba(213, 182, 120, 0.8);
  }
</style>