<!--
  WebAssembly GPU Initialization Demo
  Real-time monitoring and testing interface for the WASM GPU system
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { createWasmGpuService, WasmGpuHelpers } from '$lib/wasm/gpu-wasm-init';
  
  // Initialize WASM GPU service with RTX 3060 configuration
  const wasmGpu = createWasmGpuService(WasmGpuHelpers.rtx3060Config());
  
  // Reactive stores
  const { initStatus, performanceMetrics, resourceStatus } = wasmGpu.stores;
  const { isReady, isRtx3060, systemHealth, performance } = wasmGpu.derived;
  
  // Demo state
  let benchmarkRunning = $state(false);
  let benchmarkResults: { operation: string; time: number; throughput: number }[] = $state([]);
let testVectorCount = $state(100);
let testDimensions = $state(384);
  let similarityResults: Float32Array | null = $state(null);
  
  // Legal AI test scenarios
  const legalTestScenarios = [
    {
      name: 'Contract Analysis',
      description: 'Similarity search across contract clauses',
      vectorCount: 150,
      dimensions: 384,
      expectedTime: 5 // ms
    },
    {
      name: 'Case Law Search',
      description: 'Semantic search through legal precedents',
      vectorCount: 500,
      dimensions: 768,
      expectedTime: 15 // ms
    },
    {
      name: 'Evidence Classification',
      description: 'Document type classification using embeddings',
      vectorCount: 200,
      dimensions: 512,
      expectedTime: 8 // ms
    }
  ];
  
  let selectedScenario = $state(legalTestScenarios[0]);
  
  onMount(async () => {
    console.log('🎮 WASM GPU Demo component mounted');
    
    // Wait for initialization
    const unsubscribe = isReady.subscribe(ready => {
      if (ready) {
        console.log('✅ WASM GPU system ready for demos');
        unsubscribe();
      }
    });
  });
  
  /**
   * Run comprehensive benchmark suite
   */
  async function runBenchmark() {
    if (!$isReady || benchmarkRunning) return;
    
    benchmarkRunning = true;
    benchmarkResults = [];
    
    try {
      console.log('🏃 Starting WASM GPU benchmark suite...');
      
      // Test 1: Vector similarity computation
      const vectors1 = WasmGpuHelpers.createTestVectors(testVectorCount, testDimensions);
      const vectors2 = WasmGpuHelpers.createTestVectors(testVectorCount, testDimensions);
      
      const startTime = performance.now();
      const similarities = await wasmGpu.computeVectorSimilarity(vectors1, vectors2, testDimensions);
      const computeTime = performance.now() - startTime;
      
      const dataSize = (vectors1.length + vectors2.length + similarities.length) * 4; // bytes
      const throughput = (dataSize / 1024 / 1024) / (computeTime / 1000); // MB/s
      
      benchmarkResults.push({
        operation: 'Vector Similarity',
        time: computeTime,
        throughput
      });
      
      similarityResults = similarities;
      
      // Test 2: Memory bandwidth test
      const largeVectors1 = WasmGpuHelpers.createTestVectors(1000, 768);
      const largeVectors2 = WasmGpuHelpers.createTestVectors(1000, 768);
      
      const memoryStart = performance.now();
      await wasmGpu.computeVectorSimilarity(largeVectors1, largeVectors2, 768);
      const memoryTime = performance.now() - memoryStart;
      
      const largeDataSize = (largeVectors1.length + largeVectors2.length) * 4;
      const memoryThroughput = (largeDataSize / 1024 / 1024) / (memoryTime / 1000);
      
      benchmarkResults.push({
        operation: 'Memory Bandwidth',
        time: memoryTime,
        throughput: memoryThroughput
      });
      
      // Test 3: Legal AI scenario
      const scenarioStart = performance.now();
      const scenarioVectors1 = WasmGpuHelpers.createTestVectors(
        selectedScenario.vectorCount,
        selectedScenario.dimensions
      );
      const scenarioVectors2 = WasmGpuHelpers.createTestVectors(
        selectedScenario.vectorCount,
        selectedScenario.dimensions
      );
      
      await wasmGpu.computeVectorSimilarity(scenarioVectors1, scenarioVectors2, selectedScenario.dimensions);
      const scenarioTime = performance.now() - scenarioStart;
      
      const scenarioDataSize = (scenarioVectors1.length + scenarioVectors2.length) * 4;
      const scenarioThroughput = (scenarioDataSize / 1024 / 1024) / (scenarioTime / 1000);
      
      benchmarkResults.push({
        operation: selectedScenario.name,
        time: scenarioTime,
        throughput: scenarioThroughput
      });
      
      console.log('🎯 Benchmark results:', benchmarkResults);
      
    } catch (error) {
      console.error('❌ Benchmark failed:', error);
    } finally {
      benchmarkRunning = false;
    }
  }
  
  /**
   * Run specific legal AI scenario
   */
  async function runLegalScenario() {
    if (!$isReady || benchmarkRunning) return;
    
    benchmarkRunning = true;
    
    try {
      console.log(`🏛️ Running legal AI scenario: ${selectedScenario.name}`);
      
      const vectors1 = WasmGpuHelpers.createTestVectors(
        selectedScenario.vectorCount,
        selectedScenario.dimensions
      );
      const vectors2 = WasmGpuHelpers.createTestVectors(
        selectedScenario.vectorCount,
        selectedScenario.dimensions
      );
      
      const startTime = performance.now();
      const results = await wasmGpu.computeVectorSimilarity(vectors1, vectors2, selectedScenario.dimensions);
      const executionTime = performance.now() - startTime;
      
      similarityResults = results;
      
      // Find top similarities
      const topSimilarities = Array.from(results)
        .map((similarity, index) => ({ similarity, index }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 10);
      
      console.log(`✅ ${selectedScenario.name} completed in ${Math.round(executionTime)}ms`);
      console.log('🔍 Top similarities:', topSimilarities);
      
    } catch (error) {
      console.error(`❌ Legal scenario failed: ${selectedScenario.name}`, error);
    } finally {
      benchmarkRunning = false;
    }
  }
  
  /**
   * Get status color based on system health
   */
  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': case 'optimal': case 'good': case 'efficient':
        return 'text-green-400';
      case 'warning': case 'high': case 'overhead':
        return 'text-yellow-400';
      case 'error': case 'critical':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  }
  
  /**
   * Format throughput for display
   */
  function formatThroughput(throughput: number): string {
    if (throughput > 1000) {
      return `${(throughput / 1000).toFixed(1)} GB/s`;
    }
    return `${throughput.toFixed(1)} MB/s`;
  }
  
  /**
   * Get performance grade color
   */
  function getGradeColor(grade: string): string {
    switch (grade) {
      case 'S': return 'text-purple-400';
      case 'A': return 'text-green-400';
      case 'B': return 'text-blue-400';
      case 'C': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }
</script>

<div class="wasm-gpu-demo p-6 bg-gray-900 text-white min-h-screen">
  <div class="max-w-7xl mx-auto">
    <header class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
        WebAssembly GPU System
      </h1>
      <p class="text-gray-400 text-lg">
        Browser-native GPU acceleration without Node.js overhead for legal AI applications
      </p>
    </header>

    <!-- System Status -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <!-- Initialization Status -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-xl font-semibold mb-4 text-blue-400">Initialization Status</h3>
        <div class="space-y-3">
          <div class="flex justify-between items-center">
            <span>Phase:</span>
            <span class="font-mono text-sm px-2 py-1 bg-gray-700 rounded">
              {$initStatus.phase}
            </span>
          </div>
          <div class="flex justify-between items-center">
            <span>Progress:</span>
            <span class="text-green-400">{$initStatus.progress}%</span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-2">
            <div 
              class="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
              style="width: {$initStatus.progress}%"
            ></div>
          </div>
          <div class="text-sm text-gray-400 mt-2">
            {$initStatus.message}
          </div>
          {#if $initStatus.error}
            <div class="text-red-400 text-sm bg-red-900/20 p-2 rounded">
              {$initStatus.error}
            </div>
          {/if}
        </div>
      </div>

      <!-- Device Information -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-xl font-semibold mb-4 text-green-400">Device Information</h3>
        {#if $initStatus.deviceInfo}
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-400">Name:</span>
              <span class="font-mono">{$initStatus.deviceInfo.name}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Vendor:</span>
              <span>{$initStatus.deviceInfo.vendor}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">RTX 3060:</span>
              <span class="{$isRtx3060 ? 'text-green-400' : 'text-yellow-400'}">
                {$isRtx3060 ? '✅ Detected' : '⚠️ Not detected'}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">WASM Compatible:</span>
              <span class="text-green-400">
                {$initStatus.deviceInfo.wasmCompatible ? '✅ Yes' : '❌ No'}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Max Buffer:</span>
              <span class="font-mono">
                {Math.round($initStatus.deviceInfo.maxBufferSize / (1024 * 1024))} MB
              </span>
            </div>
          </div>
        {:else}
          <div class="text-gray-500">Device information not available</div>
        {/if}
      </div>

      <!-- System Health -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-xl font-semibold mb-4 text-purple-400">System Health</h3>
        <div class="space-y-3">
          <div class="flex justify-between items-center">
            <span>Overall:</span>
            <span class="{getStatusColor($systemHealth.overall)} font-semibold">
              {$systemHealth.overall.toUpperCase()}
            </span>
          </div>
          <div class="flex justify-between items-center">
            <span>GPU:</span>
            <span class="{getStatusColor($systemHealth.gpu)}">
              {$systemHealth.gpu}
            </span>
          </div>
          <div class="flex justify-between items-center">
            <span>Memory:</span>
            <span class="{getStatusColor($systemHealth.memory)}">
              {$systemHealth.memory}
            </span>
          </div>
          <div class="flex justify-between items-center">
            <span>WASM:</span>
            <span class="{getStatusColor($systemHealth.wasm)}">
              {$systemHealth.wasm}
            </span>
          </div>
          <div class="flex justify-between items-center">
            <span>Performance:</span>
            <span class="{getGradeColor($performance.grade)} text-xl font-bold">
              {$performance.grade}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Performance Metrics -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div class="text-blue-400 text-sm font-semibold mb-1">GPU Utilization</div>
        <div class="text-2xl font-bold">{Math.round($performanceMetrics.gpuUtilization)}%</div>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div class="text-green-400 text-sm font-semibold mb-1">Throughput</div>
        <div class="text-2xl font-bold">{formatThroughput($performanceMetrics.throughputMBps)}</div>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div class="text-purple-400 text-sm font-semibold mb-1">WASM Memory</div>
        <div class="text-2xl font-bold">{Math.round($resourceStatus.wasmMemoryUsage)} MB</div>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div class="text-yellow-400 text-sm font-semibold mb-1">Operations</div>
        <div class="text-2xl font-bold">{$performanceMetrics.totalOperations}</div>
      </div>
    </div>

    <!-- Legal AI Test Scenarios -->
    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
      <h3 class="text-xl font-semibold mb-4 text-yellow-400">Legal AI Test Scenarios</h3>
      
      <div class="mb-4">
        <label class="block text-sm font-medium mb-2">Select Test Scenario:</label>
        <select 
          bind:value={selectedScenario}
          class="bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 w-full"
          disabled={benchmarkRunning}
        >
          {#each legalTestScenarios as scenario}
            <option value={scenario}>{scenario.name}</option>
          {/each}
        </select>
      </div>
      
      <div class="bg-gray-700 rounded-lg p-4 mb-4">
        <h4 class="font-semibold mb-2">{selectedScenario.name}</h4>
        <p class="text-gray-300 text-sm mb-3">{selectedScenario.description}</p>
        <div class="grid grid-cols-3 gap-4 text-sm">
          <div>
            <span class="text-gray-400">Vectors:</span>
            <span class="ml-2 font-mono">{selectedScenario.vectorCount}</span>
          </div>
          <div>
            <span class="text-gray-400">Dimensions:</span>
            <span class="ml-2 font-mono">{selectedScenario.dimensions}</span>
          </div>
          <div>
            <span class="text-gray-400">Expected:</span>
            <span class="ml-2 font-mono">{selectedScenario.expectedTime}ms</span>
          </div>
        </div>
      </div>
      
      <div class="flex gap-4">
        <button
          on:onclick={runLegalScenario}
          disabled={!$isReady || benchmarkRunning}
          class="px-6 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
        >
          {#if benchmarkRunning}
            Running...
          {:else}
            Run Legal Scenario
          {/if}
        </button>
        
        <button
          on:onclick={runBenchmark}
          disabled={!$isReady || benchmarkRunning}
          class="px-6 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
        >
          {#if benchmarkRunning}
            Running...
          {:else}
            Run Full Benchmark
          {/if}
        </button>
      </div>
    </div>

    <!-- Benchmark Results -->
    {#if benchmarkResults.length > 0}
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
        <h3 class="text-xl font-semibold mb-4 text-green-400">Benchmark Results</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-gray-700">
                <th class="text-left py-2 px-4">Operation</th>
                <th class="text-right py-2 px-4">Time (ms)</th>
                <th class="text-right py-2 px-4">Throughput</th>
                <th class="text-right py-2 px-4">Performance</th>
              </tr>
            </thead>
            <tbody>
              {#each benchmarkResults as result}
                <tr class="border-b border-gray-700/50">
                  <td class="py-2 px-4 font-medium">{result.operation}</td>
                  <td class="py-2 px-4 text-right font-mono">{Math.round(result.time)}</td>
                  <td class="py-2 px-4 text-right font-mono">{formatThroughput(result.throughput)}</td>
                  <td class="py-2 px-4 text-right">
                    <span class="{result.throughput > 2000 ? 'text-green-400' : result.throughput > 1000 ? 'text-yellow-400' : 'text-red-400'}">
                      {result.throughput > 2000 ? 'Excellent' : result.throughput > 1000 ? 'Good' : 'Poor'}
                    </span>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    {/if}

    <!-- Similarity Results -->
    {#if similarityResults}
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-xl font-semibold mb-4 text-blue-400">Latest Similarity Results</h3>
        <div class="grid grid-cols-2 gap-4 mb-4">
          <div>
            <span class="text-gray-400">Result Count:</span>
            <span class="ml-2 font-mono">{similarityResults.length}</span>
          </div>
          <div>
            <span class="text-gray-400">Data Size:</span>
            <span class="ml-2 font-mono">{Math.round(similarityResults.byteLength / 1024)} KB</span>
          </div>
        </div>
        
        <div class="bg-gray-700 rounded-lg p-4">
          <h4 class="font-semibold mb-2">Top 10 Similarities</h4>
          <div class="grid grid-cols-2 gap-2 text-sm font-mono">
            {#each Array.from(similarityResults).slice(0, 10) as similarity, index}
              <div class="flex justify-between">
                <span class="text-gray-400">#{index + 1}:</span>
                <span class="{similarity > 0.8 ? 'text-green-400' : similarity > 0.6 ? 'text-yellow-400' : 'text-gray-300'}">
                  {similarity.toFixed(4)}
                </span>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

