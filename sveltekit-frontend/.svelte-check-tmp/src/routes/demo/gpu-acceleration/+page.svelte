<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import type { CUDAResponse } from '$lib/services/gpu-service-router.js';
  import type { AttentionResult, LegalContextAnalysis } from '$lib/services/flashattention2-rtx3060.js';
  
  // Reactive stores for GPU status and results
  const gpuStatus = writable<any>(null);
  const cudaHealth = writable<boolean>(false);
  const processingResults = writable<any[]>([]);
  const isProcessing = writable<boolean>(false);
  const performanceMetrics = writable<any>(null);
  
  let testText = "The defendant's indemnification clause shall survive termination of this agreement and remain in full force and effect, providing liability coverage for all preceding actions.";
  let selectedService = 'enhanced-rag';
  let selectedOperation = 'legal_analysis';
  let selectedPriority: 'high' | 'normal' | 'low' = 'high';
  
  const availableServices = [
    { value: 'enhanced-rag', label: 'Enhanced RAG (Port 8094)' },
    { value: 'legal-ai', label: 'Legal AI Service (Port 8202)' },
    { value: 'gpu-indexer', label: 'GPU Indexer (Port 8220)' },
    { value: 'typescript-optimizer', label: 'TypeScript Optimizer (Port 5173)' },
    { value: 'ai-summary', label: 'AI Summary (Port 8096)' },
    { value: 'kratos-server', label: 'Kratos gRPC (Port 50051)' }
  ];
  
  const availableOperations = [
    { value: 'legal_analysis', label: 'Legal Document Analysis' },
    { value: 'embedding', label: 'Text Embedding Generation' },
    { value: 'similarity', label: 'Similarity Search' },
    { value: 'vector_search', label: 'Vector Database Search' },
    { value: 'som_train', label: 'Self-Organizing Map Training' }
  ];
  
  // Check GPU status and CUDA health on mount
  onMount(async () => {
    await checkGPUStatus();
  });
  
  async function checkGPUStatus() {
    try {
      const response = await fetch('/api/gpu/cuda-status');
      const data = await response.json();
      
      gpuStatus.set(data.gpu_status);
      cudaHealth.set(data.cuda.available);
      
      if (data.gpu_status?.performanceMetrics) {
        performanceMetrics.set(data.gpu_status.performanceMetrics);
      }
      
    } catch (error) {
      console.error('Failed to check GPU status:', error);
      cudaHealth.set(false);
    }
  }
  
  async function processWithCUDA() {
    isProcessing.set(true);
    
    try {
      // Convert text to simple numeric array for processing
      const textData = Array.from(testText).map((char, i) => char.charCodeAt(0) + i * 0.01);
      
      const response = await fetch('/api/gpu/cuda-status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          service: selectedService,
          operation: selectedOperation,
          priority: selectedPriority,
          data: textData,
          metadata: {
            text: testText,
            timestamp: Date.now(),
            rtx_3060_ti: true,
            legal_domain: true
          }
        })
      });
      
      const result = await response.json();
      
      // Add to results
      processingResults.update(results => [
        {
          timestamp: new Date().toISOString(),
          service: selectedService,
          operation: selectedOperation,
          priority: selectedPriority,
          success: result.success,
          processingTime: result.result?.processingTime || 0,
          memoryUsed: result.result?.memoryUsed || 0,
          result: result.result,
          error: result.error
        },
        ...results
      ]);
      
      // Refresh status to get updated metrics
      await checkGPUStatus();
      
    } catch (error) {
      console.error('GPU processing failed:', error);
      processingResults.update(results => [
        {
          timestamp: new Date().toISOString(),
          service: selectedService,
          operation: selectedOperation,
          priority: selectedPriority,
          success: false,
          error: error instanceof Error ? error.message : String(error)
        },
        ...results
      ]);
    }
    
    isProcessing.set(false);
  }
  
  async function testFlashAttention2() {
    isProcessing.set(true);
    
    try {
      // Test FlashAttention2 directly
      const response = await fetch('/api/gpu/flash-attention', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: testText,
          context: ["Legal contract analysis", "Indemnification clause review"],
          analysisType: 'legal'
        })
      });
      
      const result = await response.json();
      
      processingResults.update(results => [
        {
          timestamp: new Date().toISOString(),
          service: 'flashattention2',
          operation: 'legal_text_processing',
          priority: 'high',
          success: result.success,
          processingTime: result.processingTime,
          memoryUsage: result.memoryUsage,
          result: result,
          type: 'FlashAttention2 RTX 3060 Ti'
        },
        ...results
      ]);
      
    } catch (error) {
      console.error('FlashAttention2 test failed:', error);
    }
    
    isProcessing.set(false);
  }
  
  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }
</script>

<svelte:head>
  <title>GPU Acceleration Demo - Legal AI Platform</title>
</svelte:head>

<div class="container mx-auto px-4 py-8 max-w-7xl">
  <div class="mb-8">
    <h1 class="text-4xl font-bold mb-4">🚀 GPU Acceleration Demo</h1>
    <p class="text-xl text-gray-600 mb-6">
      RTX 3060 Ti CUDA Integration with 37+ Go Microservices
    </p>
    
    <!-- GPU Status Card -->
    <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
      <h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
        <span class="w-3 h-3 rounded-full {$cudaHealth ? 'bg-green-500' : 'bg-red-500'}"></span>
        GPU & CUDA Status
      </h2>
      
      {#if $gpuStatus}
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="bg-gray-50 rounded p-4">
            <h3 class="font-semibold text-gray-700 mb-2">Memory Usage</h3>
            <p class="text-lg">
              {formatBytes($gpuStatus.memoryUsage.total - $gpuStatus.memoryUsage.available)} / 
              {formatBytes($gpuStatus.memoryUsage.total)}
            </p>
            <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div 
                class="bg-blue-600 h-2 rounded-full" 
                style="width: {((($gpuStatus.memoryUsage.total - $gpuStatus.memoryUsage.available) / $gpuStatus.memoryUsage.total) * 100).toFixed(1)}%"
              ></div>
            </div>
          </div>
          
          <div class="bg-gray-50 rounded p-4">
            <h3 class="font-semibold text-gray-700 mb-2">Active Services</h3>
            <p class="text-lg">{$gpuStatus.activeServices.length} Services</p>
            <p class="text-sm text-gray-600">GPU-accelerated microservices</p>
          </div>
          
          <div class="bg-gray-50 rounded p-4">
            <h3 class="font-semibold text-gray-700 mb-2">CUDA Worker</h3>
            <p class="text-sm font-mono break-all">{$gpuStatus.cudaWorkerPath}</p>
            <p class="text-sm {$cudaHealth ? 'text-green-600' : 'text-red-600'} mt-1">
              {$cudaHealth ? '✅ Healthy' : '❌ Unavailable'}
            </p>
          </div>
        </div>
      {:else}
        <p class="text-gray-500">Loading GPU status...</p>
      {/if}
    </div>
  </div>
  
  <!-- GPU Processing Interface -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
    
    <!-- Control Panel -->
    <div class="bg-white rounded-lg shadow-lg p-6">
      <h2 class="text-2xl font-semibold mb-6">🎛️ GPU Processing Control</h2>
      
      <!-- Test Text Input -->
      <div class="mb-4">
        <label for="test-text" class="block text-sm font-medium text-gray-700 mb-2">
          Legal Text for Analysis
        </label>
        <textarea
          id="test-text"
          bind:value={testText}
          rows="4"
          class="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Enter legal text for GPU processing..."
        ></textarea>
      </div>
      
      <!-- Service Selection -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Service</label>
          <select bind:value={selectedService} class="w-full border border-gray-300 rounded-md px-3 py-2">
            {#each availableServices as service}
              <option value={service.value}>{service.label}</option>
            {/each}
          </select>
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Operation</label>
          <select bind:value={selectedOperation} class="w-full border border-gray-300 rounded-md px-3 py-2">
            {#each availableOperations as operation}
              <option value={operation.value}>{operation.label}</option>
            {/each}
          </select>
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Priority</label>
          <select bind:value={selectedPriority} class="w-full border border-gray-300 rounded-md px-3 py-2">
            <option value="high">High (Direct CUDA)</option>
            <option value="normal">Normal (Service Route)</option>
            <option value="low">Low (Queue)</option>
          </select>
        </div>
      </div>
      
      <!-- Processing Buttons -->
      <div class="space-y-4">
        <button
          click={processWithCUDA}
          disabled={$isProcessing || !$cudaHealth}
          class="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {#if $isProcessing}
            <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
            Processing with CUDA...
          {:else}
            ⚡ Process with CUDA Worker
          {/if}
        </button>
        
        <button
          click={testFlashAttention2}
          disabled={$isProcessing}
          class="w-full bg-purple-600 text-white py-3 px-4 rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          🧠 Test FlashAttention2 RTX 3060 Ti
        </button>
        
        <button
          click={checkGPUStatus}
          class="w-full bg-gray-600 text-white py-3 px-4 rounded-md hover:bg-gray-700"
        >
          🔄 Refresh GPU Status
        </button>
      </div>
    </div>
    
    <!-- Results Panel -->
    <div class="bg-white rounded-lg shadow-lg p-6">
      <h2 class="text-2xl font-semibold mb-6">📊 Processing Results</h2>
      
      {#if $performanceMetrics && $performanceMetrics.length > 0}
        <div class="mb-6 p-4 bg-blue-50 rounded-lg">
          <h3 class="font-semibold text-blue-800 mb-2">Recent Performance Metrics</h3>
          <div class="grid grid-cols-2 gap-4 text-sm">
            {#each $performanceMetrics.slice(0, 3) as metric}
              <div>
                <p class="text-blue-600 font-medium">{metric.service}</p>
                <p class="text-gray-600">
                  {formatDuration(metric.duration)} | 
                  {formatBytes(metric.memoryUsed)}
                </p>
              </div>
            {/each}
          </div>
        </div>
      {/if}
      
      <div class="space-y-4 max-h-96 overflow-y-auto">
        {#each $processingResults as result, i}
          <div class="border border-gray-200 rounded-lg p-4 {result.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}">
            <div class="flex justify-between items-start mb-2">
              <div>
                <p class="font-semibold {result.success ? 'text-green-800' : 'text-red-800'}">
                  {result.service} - {result.operation}
                </p>
                <p class="text-sm text-gray-600">{result.timestamp}</p>
              </div>
              <span class="px-2 py-1 text-xs rounded {
                result.priority === 'high' ? 'bg-red-100 text-red-800' :
                result.priority === 'normal' ? 'bg-yellow-100 text-yellow-800' :
                'bg-gray-100 text-gray-800'
              }">
                {result.priority}
              </span>
            </div>
            
            {#if result.success}
              <div class="grid grid-cols-2 gap-4 text-sm">
                {#if result.processingTime}
                  <p><span class="font-medium">Time:</span> {formatDuration(result.processingTime)}</p>
                {/if}
                {#if result.memoryUsed}
                  <p><span class="font-medium">Memory:</span> {formatBytes(result.memoryUsed)}</p>
                {/if}
              </div>
              
              {#if result.result && result.result.vector}
                <p class="text-xs text-gray-600 mt-2">
                  Vector Result: [{result.result.vector.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]
                </p>
              {/if}
            {:else}
              <p class="text-red-600 text-sm mt-2">Error: {result.error}</p>
            {/if}
          </div>
        {/each}
        
        {#if $processingResults.length === 0}
          <p class="text-gray-500 text-center py-8">No processing results yet. Run a GPU operation to see results.</p>
        {/if}
      </div>
    </div>
  </div>
  
  <!-- Service Architecture Overview -->
  <div class="mt-8 bg-white rounded-lg shadow-lg p-6">
    <h2 class="text-2xl font-semibold mb-6">🏗️ GPU Service Architecture</h2>
    
    {#if $gpuStatus}
      <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        {#each $gpuStatus.activeServices as service}
          <div class="bg-gray-100 rounded-lg p-3 text-center">
            <p class="text-sm font-semibold text-gray-800">{service}</p>
            <p class="text-xs text-gray-600 mt-1">GPU Ready</p>
          </div>
        {/each}
      </div>
      
      <div class="mt-6 p-4 bg-blue-50 rounded-lg">
        <h3 class="font-semibold text-blue-800 mb-2">🎯 Integration Status</h3>
        <ul class="text-sm text-blue-700 space-y-1">
          <li>✅ FlashAttention2 RTX 3060 Ti Service - Memory-efficient attention</li>
          <li>✅ CUDA Worker Integration - JSON I/O with cuda-worker.exe</li>
          <li>✅ Matrix Transform Library - Hardware-accelerated transformations</li>
          <li>✅ GPU Service Router - 37+ Go microservices coordination</li>
          <li>✅ Performance Monitoring - Real-time metrics and health checks</li>
        </ul>
      </div>
    {/if}
  </div>
</div>

<style>
  :global(body) {
    background-color: #f9fafb;
  }
</style>