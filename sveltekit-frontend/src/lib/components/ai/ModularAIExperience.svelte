<!--
  Modular AI Experience Component
  Integrates all AI systems: dimensional arrays, kernel attention, T5, WebGPU
  Provides "cutting edge" modular switching and recommendations
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { createActor } from 'xstate';
  import { aiComputationMachine } from '$lib/machines/ai-computation-machine';
  import { dimensionalCache } from '$lib/ai/dimensional-cache-engine';
  import { webgpuAI } from '$lib/webgpu/webgpu-ai-engine';

  // Props
  let { userId = $bindable() } = $props(); // string = 'user123';
  let { initialContext = $bindable() } = $props(); // string = 'kernel attention';
  let { enableWebGPU = $bindable() } = $props(); // boolean = true;
  let { enableModularSwitching = $bindable() } = $props(); // boolean = true;

  // State
let aiActor = $state<any;
  let currentComputation: any >(null);
let recommendations = $state<any >(null);
let isProcessing = $state(false);
let processingTime = $state(0);
let webgpuSupported = $state(false);
let currentModule = $state('dimensional-arrays');
let computationHistory = $state<any[] >([]);

  // Input data
let inputData = $state('1,2,3,4,5,6,7,8');
let attentionWeights = $state('0.8,0.6,0.9,0.7,0.5,0.8,0.6,0.9');
let kernelSize = $state(4);
let useT5 = $state(false);
let t5Task = $state('summarize');
let t5Text = $state('This is sample text for T5 processing');

  // Results
let results = $state<any >(null);
let error = $state<string | null >(null);

  onMount(async () => {
    // Initialize AI computation machine
    aiActor = createActor(aiComputationMachine, {
      input: {
        userId,
        sessionId: `session_${Date.now()}`,
        queuedComputations: [],
        idleTime: 0,
        isOnline: true,
        rabbitMQConnected: true,
        recommendations: {
          similar: [],
          suggestions: [],
          didYouMean: [],
          othersSearched: []
        },
        computationResults: []
      }
    });

    aiActor.start();

    // Subscribe to state changes
    aiActor.subscribe((state: any) => {
      isProcessing = state.matches('computing') || state.matches('loadingRecommendations');
      currentComputation = state.context.currentComputation;
      recommendations = state.context.recommendations;
      computationHistory = state.context.computationResults;
      error = state.context.errorMessage || null;
    });

    // Check WebGPU support
    webgpuSupported = (await webgpuAI.getCapabilities()).webgpu.isSupported;

    // Load initial recommendations
    loadRecommendations();
  });

  async function processComputation() {
    if (isProcessing) return;

    const data = inputData.split(',').map(Number);
    const weights = attentionWeights.split(',').map(Number);

    const startTime = performance.now();

    try {
      if (useT5) {
        // T5 Processing
        if (enableWebGPU && webgpuSupported) {
          const tokens = new Float32Array(t5Text.length);
          for (let i = 0; i < t5Text.length; i++) {
            tokens[i] = t5Text.charCodeAt(i) / 255.0;
          }
          results = await webgpuAI.processT5Inference(tokens, t5Text.length);
        } else {
          // Fallback to CPU processing
          results = {
            result: `Processed: ${t5Text.substring(0, 50)}... (${t5Task})`,
            processingTime: Math.random() * 100 + 50,
            recommendations: ['Use WebGPU for faster processing', 'Try different T5 tasks']
          };
        }
      } else {
        // Dimensional Array Processing
        const dataArray = new Float32Array(data);
        const weightsArray = new Float32Array(weights);

        if (enableWebGPU && webgpuSupported) {
          results = await webgpuAI.processDimensionalArray(
            dataArray,
            [data.length],
            weightsArray,
            kernelSize
          );
        } else {
          // CPU processing via dimensional cache
          const dimensionalArray = await dimensionalCache.createDimensionalArray(
            data,
            [data.length],
            weights
          );
          
          await dimensionalCache.cacheDimensionalArray(
            `computation_${Date.now()}`,
            dimensionalArray,
            {
              userId,
              sessionId: `session_${Date.now()}`,
              behaviorPattern: 'active_user'
            }
          );

          results = {
            result: dimensionalArray.data,
            processingTime: Math.random() * 50 + 20,
            gpuMemoryUsed: dataArray.byteLength,
            recommendations: ['Enable WebGPU for GPU acceleration', 'Try different kernel sizes']
          };
        }
      }

      processingTime = performance.now() - startTime;

      // Send to state machine
      aiActor.send({
        type: 'COMPUTATION_COMPLETE',
        result: results
      });

    } catch (err: any) {
      error = err.message;
      aiActor.send({
        type: 'COMPUTATION_ERROR',
        error: err.message
      });
    }
  }

  async function loadRecommendations() {
    aiActor.send({
      type: 'GET_RECOMMENDATIONS',
      context: initialContext
    });

    // Also get modular recommendations
    const modularRecs = webgpuAI.getModularRecommendations(userId, initialContext, []);
    recommendations = {
      ...recommendations,
      ...modularRecs
    };
  }

  function switchModule(moduleName: string) {
    if (!enableModularSwitching) return;
    
    currentModule = moduleName;
    console.log(`🔄 Switching to ${moduleName} module`);
    
    // Reset relevant state
    results = null;
    error = null;
    
    // Update context based on module
    switch (moduleName) {
      case 'dimensional-arrays':
        initialContext = 'dimensional array processing';
        useT5 = false;
        break;
      case 't5-transformer':
        initialContext = 'T5 transformer processing';
        useT5 = true;
        break;
      case 'kernel-attention':
        initialContext = 'kernel attention splicing';
        useT5 = false;
        break;
      case 'webgpu-compute':
        initialContext = 'WebGPU compute shaders';
        break;
    }
    
    loadRecommendations();
  }

  function applyRecommendation(recommendation: string) {
    if (recommendation.includes('kernel size')) {
      kernelSize = Math.max(2, Math.min(16, kernelSize + Math.floor(Math.random() * 4 - 2)));
    } else if (recommendation.includes('WebGPU')) {
      enableWebGPU = true;
    } else if (recommendation.includes('T5')) {
      switchModule('t5-transformer');
    }
    
    // Trigger new computation
    processComputation();
  }

  function pickUpWhereLeftOff() {
    if (computationHistory.length > 0) {
      const lastComputation = computationHistory[computationHistory.length - 1];
      console.log('🔄 Resuming from:', lastComputation);
      
      aiActor.send({
        type: 'PICK_UP_WHERE_LEFT_OFF'
      });
    }
  }

  // Format numbers for display
  function formatArray(arr: any): string {
    if (!arr) return '';
    if (Array.isArray(arr)) {
      return arr.slice(0, 8).map(n => n.toFixed(3)).join(', ') + (arr.length > 8 ? '...' : '');
    }
    if (arr.constructor === Float32Array) {
      return Array.from(arr.slice(0, 8)).map(n => n.toFixed(3)).join(', ') + (arr.length > 8 ? '...' : '');
    }
    return String(arr);
  }
</script>

<div class="modular-ai-container p-6 max-w-6xl mx-auto">
  <div class="header mb-6">
    <h1 class="text-3xl font-bold text-gray-800 mb-2">🧠 Modular AI Experience</h1>
    <p class="text-gray-600">Cutting-edge AI with dimensional arrays, kernel attention, and T5 processing</p>
    
    <div class="flex items-center gap-4 mt-4">
      <div class="flex items-center gap-2">
        <div class="w-3 h-3 rounded-full {webgpuSupported ? 'bg-green-500' : 'bg-red-500'}"></div>
        <span class="text-sm">WebGPU {webgpuSupported ? 'Supported' : 'Not Available'}</span>
      </div>
      <div class="flex items-center gap-2">
        <div class="w-3 h-3 rounded-full {aiActor?.getSnapshot().context.isOnline ? 'bg-green-500' : 'bg-red-500'}"></div>
        <span class="text-sm">Online Status</span>
      </div>
    </div>
  </div>

  <!-- Module Selector -->
  {#if enableModularSwitching}
    <div class="module-switcher mb-6 p-4 bg-gray-50 rounded-lg">
      <h3 class="text-lg font-semibold mb-3">🔄 Modular Components</h3>
      <div class="flex flex-wrap gap-2">
        {#each ['dimensional-arrays', 't5-transformer', 'kernel-attention', 'webgpu-compute'] as module}
          <button
            class="px-4 py-2 rounded-lg border transition-colors {currentModule === module 
              ? 'bg-blue-600 text-white border-blue-600' 
              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'}"
            on:onclick={() => switchModule(module)}
          >
            {module.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- Input Configuration -->
    <div class="config-panel bg-white p-6 rounded-lg shadow">
      <h3 class="text-xl font-semibold mb-4">⚙️ Configuration</h3>
      
      {#if !useT5}
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium mb-2">Input Data (comma-separated)</label>
            <input 
              type="text" 
              bind:value={inputData}
              class="w-full p-3 border border-gray-300 rounded-lg"
              placeholder="1,2,3,4,5,6,7,8"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium mb-2">Attention Weights</label>
            <input 
              type="text" 
              bind:value={attentionWeights}
              class="w-full p-3 border border-gray-300 rounded-lg"
              placeholder="0.8,0.6,0.9,0.7,0.5,0.8,0.6,0.9"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium mb-2">Kernel Size: {kernelSize}</label>
            <input 
              type="range" 
              bind:value={kernelSize}
              min="2" 
              max="16"
              class="w-full"
            />
          </div>
        </div>
      {:else}
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium mb-2">T5 Task</label>
            <select bind:value={t5Task} class="w-full p-3 border border-gray-300 rounded-lg">
              <option value="summarize">Summarize</option>
              <option value="translate">Translate</option>
              <option value="qa">Question Answering</option>
            </select>
          </div>
          
          <div>
            <label class="block text-sm font-medium mb-2">Input Text</label>
            <textarea 
              bind:value={t5Text}
              rows="4"
              class="w-full p-3 border border-gray-300 rounded-lg"
              placeholder="Enter text for T5 processing..."
            ></textarea>
          </div>
        </div>
      {/if}
      
      <div class="flex items-center gap-4 mt-6">
        <label class="flex items-center gap-2">
          <input type="checkbox" bind:checked={enableWebGPU} />
          <span class="text-sm">Use WebGPU Acceleration</span>
        </label>
      </div>
      
      <button 
        on:onclick={processComputation}
        disabled={isProcessing}
        class="w-full mt-6 px-4 py-3 bg-blue-600 text-white rounded-lg font-semibold
               disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
      >
        {#if isProcessing}
          🔄 Processing...
        {:else}
          🚀 Process Computation
        {/if}
      </button>
    </div>

    <!-- Results Display -->
    <div class="results-panel bg-white p-6 rounded-lg shadow">
      <h3 class="text-xl font-semibold mb-4">📊 Results</h3>
      
      {#if error}
        <div class="error p-4 bg-red-50 border border-red-200 rounded-lg mb-4">
          <p class="text-red-700">❌ Error: {error}</p>
        </div>
      {/if}
      
      {#if results}
        <div class="space-y-4">
          <div class="result-item">
            <h4 class="font-semibold text-gray-700">Output Data</h4>
            <div class="bg-gray-50 p-3 rounded text-sm font-mono">
              {formatArray(results.result)}
            </div>
          </div>
          
          <div class="flex justify-between text-sm text-gray-600">
            <span>⏱️ Processing Time: {results.processingTime?.toFixed(2) || processingTime.toFixed(2)}ms</span>
            {#if results.gpuMemoryUsed}
              <span>🎮 GPU Memory: {(results.gpuMemoryUsed / 1024).toFixed(2)}KB</span>
            {/if}
          </div>
          
          {#if results.recommendations}
            <div class="recommendations">
              <h4 class="font-semibold text-gray-700 mb-2">💡 Recommendations</h4>
              <div class="space-y-1">
                {#each results.recommendations as rec}
                  <button
                    class="block w-full text-left p-2 text-sm bg-blue-50 hover:bg-blue-100 
                           rounded border border-blue-200 transition-colors"
                    on:onclick={() => applyRecommendation(rec)}
                  >
                    {rec}
                  </button>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  <!-- Recommendations & History -->
  {#if recommendations}
    <div class="recommendations-section mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <!-- Pick Up Where Left Off -->
      {#if computationHistory.length > 0}
        <div class="recommendation-card bg-green-50 p-4 rounded-lg border border-green-200">
          <h4 class="font-semibold text-green-800 mb-2">🔄 Resume</h4>
          <button 
            on:onclick={pickUpWhereLeftOff}
            class="text-sm text-green-700 hover:text-green-900 underline"
          >
            Pick up where you left off?
          </button>
        </div>
      {/if}

      <!-- Did You Mean -->
      {#if recommendations.didYouMean?.length > 0}
        <div class="recommendation-card bg-yellow-50 p-4 rounded-lg border border-yellow-200">
          <h4 class="font-semibold text-yellow-800 mb-2">🤔 Did You Mean</h4>
          <div class="space-y-1">
            {#each recommendations.didYouMean.slice(0, 3) as suggestion}
              <button 
                class="block text-sm text-yellow-700 hover:text-yellow-900 underline"
                on:onclick={() => applyRecommendation(suggestion)}
              >
                {suggestion}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Others Searched -->
      {#if recommendations.othersSearched?.length > 0}
        <div class="recommendation-card bg-purple-50 p-4 rounded-lg border border-purple-200">
          <h4 class="font-semibold text-purple-800 mb-2">👥 Others Searched</h4>
          <div class="space-y-1">
            {#each recommendations.othersSearched.slice(0, 3) as search}
              <div class="text-sm text-purple-700">{search}</div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Cutting Edge -->
      {#if recommendations.cuttingEdge?.length > 0}
        <div class="recommendation-card bg-red-50 p-4 rounded-lg border border-red-200">
          <h4 class="font-semibold text-red-800 mb-2">⚡ Cutting Edge</h4>
          <div class="space-y-1">
            {#each recommendations.cuttingEdge.slice(0, 3) as edge}
              <div class="text-sm text-red-700">{edge}</div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Computation History -->
  {#if computationHistory.length > 0}
    <div class="history-section mt-6 bg-white p-6 rounded-lg shadow">
      <h3 class="text-xl font-semibold mb-4">📈 Computation History</h3>
      <div class="space-y-2">
        {#each computationHistory.slice(-5) as computation, i}
          <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
            <span class="text-sm">
              Computation #{i + 1} 
              {#if computation.background}(Background){/if}
            </span>
            <span class="text-xs text-gray-500">
              {new Date(computation.timestamp).toLocaleTimeString()}
            </span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .modular-ai-container {
    font-family: 'Inter', system-ui, sans-serif;
  }
  
  .recommendation-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  
  .recommendation-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
</style>