<!--
  WebAssembly GPU Demo Page
  Comprehensive demonstration of browser-native GPU acceleration
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { browser } from '$app/environment';
  import WasmGpuDemo from '$lib/components/demo/WasmGpuDemo.svelte';
  import { WasmGpuHelpers } from '$lib/wasm/gpu-wasm-init';
  
  let webGpuSupported = false;
  let supportCheckComplete = false;
  
  onMount(async () => {
    if (browser) {
      // Check WebGPU support
      webGpuSupported = await WasmGpuHelpers.validateWebGpuSupport();
      supportCheckComplete = true;
      
      console.log('🎮 WebGPU support check:', webGpuSupported ? 'Supported' : 'Not supported');
    }
  });
</script>

<svelte:head>
  <title>WebAssembly GPU Demo - Legal AI</title>
  <meta name="description" content="Browser-native GPU acceleration with WebAssembly for legal AI applications" />
</svelte:head>

<main class="min-h-screen bg-gray-900">
  {#if !supportCheckComplete}
    <!-- Loading state -->
    <div class="flex items-center justify-center min-h-screen">
      <div class="text-center">
        <div class="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-400 mx-auto mb-4"></div>
        <h2 class="text-xl font-semibold text-white mb-2">Checking WebGPU Support...</h2>
        <p class="text-gray-400">Validating browser capabilities</p>
      </div>
    </div>
  {:else if !webGpuSupported}
    <!-- WebGPU not supported -->
    <div class="flex items-center justify-center min-h-screen">
      <div class="max-w-2xl mx-auto p-8 text-center">
        <div class="text-red-400 text-6xl mb-6">⚠️</div>
        <h1 class="text-3xl font-bold text-white mb-4">WebGPU Not Supported</h1>
        <p class="text-gray-300 mb-6 leading-relaxed">
          Your browser doesn't support WebGPU, which is required for this demonstration. 
          WebGPU enables high-performance GPU compute operations directly in the browser.
        </p>
        
        <div class="bg-gray-800 rounded-lg p-6 text-left mb-6">
          <h3 class="text-lg font-semibold text-yellow-400 mb-3">Requirements:</h3>
          <ul class="space-y-2 text-gray-300">
            <li class="flex items-center">
              <span class="text-green-400 mr-2">✓</span>
              Chrome 113+ or Edge 113+ (recommended)
            </li>
            <li class="flex items-center">
              <span class="text-green-400 mr-2">✓</span>
              Firefox Nightly with WebGPU enabled
            </li>
            <li class="flex items-center">
              <span class="text-green-400 mr-2">✓</span>
              Modern GPU with Vulkan/DirectX 12 support
            </li>
            <li class="flex items-center">
              <span class="text-green-400 mr-2">✓</span>
              Updated graphics drivers
            </li>
          </ul>
        </div>
        
        <div class="bg-blue-900/20 border border-blue-700 rounded-lg p-4 mb-6">
          <h4 class="text-blue-400 font-semibold mb-2">Chrome Setup Instructions:</h4>
          <ol class="text-sm text-gray-300 text-left space-y-1">
            <li>1. Open Chrome flags: <code class="bg-gray-700 px-1 rounded">chrome://flags</code></li>
            <li>2. Search for "WebGPU"</li>
            <li>3. Enable "Unsafe WebGPU" flag</li>
            <li>4. Restart Chrome</li>
          </ol>
        </div>
        
        <div class="flex gap-4 justify-center">
          <a 
            href="/" 
            class="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            Return Home
          </a>
          <button 
            onclick={() => window.location.reload()} 
            class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Check Again
          </button>
        </div>
      </div>
    </div>
  {:else}
    <!-- WebGPU supported - show demo -->
    <WasmGpuDemo />
  {/if}
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
  }
</style>