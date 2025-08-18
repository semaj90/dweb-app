<!-- WebGPU Acceleration Demo Component -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly, scale } from 'svelte/transition';
  import { 
    Zap, 
    Cpu, 
    Activity, 
    BarChart3, 
    Database, 
    Search, 
    FileText, 
    Brain,
    Clock,
    TrendingUp,
    CheckCircle,
    AlertTriangle,
    Play,
    Pause,
    RotateCcw,
    Settings,
    Monitor
  } from 'lucide-svelte';

  // Import WebGPU services and stores
  import { 
    webgpuManager, 
    webgpuCapabilities, 
    webgpuMetrics, 
    webgpuStatus,
    webgpuHelpers 
  } from '../webgpu/webgpu-manager';
  import { 
    webgpuRAGService, 
    gpuRagMetrics, 
    gpuRagStatus 
  } from '../webgpu/webgpu-rag-service';

  // Component state
  let selectedDemo = $state<'capabilities' | 'vector' | 'rag' | 'performance'>('capabilities');
  let isRunning = $state(false);
  let demoResults = $state<any[]>([]);
  let benchmarkResults = $state<any>(null);

  // Test data
  let testVectorSize = $state(384);
  let testDocumentCount = $state(10);
  let testQuery = $state('contract liability and damages');

  // Reactive state access
  let capabilities = $state($webgpuCapabilities);
  let metrics = $state($webgpuMetrics);
  let status = $state($webgpuStatus);
  let ragMetrics = $state($gpuRagMetrics);
  let ragStatus = $state($gpuRagStatus);

  // Subscribe to store updates
  webgpuCapabilities.subscribe(c => capabilities = c);
  webgpuMetrics.subscribe(m => metrics = m);
  webgpuStatus.subscribe(s => status = s);
  gpuRagMetrics.subscribe(r => ragMetrics = r);
  gpuRagStatus.subscribe(r => ragStatus = r);

  onMount(async () => {
    console.log('🎮 WebGPU Demo component mounted');
    
    // Initialize WebGPU services
    await initializeServices();
  });

  async function initializeServices() {
    try {
      console.log('🔄 Initializing WebGPU services...');
      
      const gpuReady = await webgpuManager.initialize();
      if (gpuReady) {
        await webgpuRAGService.initialize();
        await webgpuRAGService.warmup();
        console.log('✓ WebGPU services ready');
      } else {
        console.warn('⚠️ WebGPU not available');
      }
    } catch (error) {
      console.error('❌ Service initialization failed:', error);
    }
  }

  // Demo functions
  async function runVectorDemo() {
    isRunning = true;
    demoResults = [];
    
    try {
      console.log('🔄 Running vector operations demo...');

      // Generate test vectors
      const vectorA = new Float32Array(testVectorSize);
      const vectorB = new Float32Array(testVectorSize);
      
      for (let i = 0; i < testVectorSize; i++) {
        vectorA[i] = Math.random() * 2 - 1;
        vectorB[i] = Math.random() * 2 - 1;
      }

      // GPU similarity computation
      const gpuStart = performance.now();
      const gpuSimilarity = await webgpuHelpers.computeSimilarity(vectorA, vectorB);
      const gpuTime = performance.now() - gpuStart;

      // CPU similarity computation for comparison
      const cpuStart = performance.now();
      const cpuSimilarity = await computeSimilarityCPU(vectorA, vectorB);
      const cpuTime = performance.now() - cpuStart;

      demoResults = [
        {
          type: 'Vector Similarity',
          gpu: { result: gpuSimilarity.toFixed(6), time: gpuTime.toFixed(2) },
          cpu: { result: cpuSimilarity.toFixed(6), time: cpuTime.toFixed(2) },
          speedup: (cpuTime / gpuTime).toFixed(2)
        }
      ];

      // Batch vector test
      if (capabilities.isSupported) {
        const queryVector = new Float32Array(testVectorSize);
        const docVectors = [];
        
        for (let i = 0; i < testDocumentCount; i++) {
          const vec = new Float32Array(testVectorSize);
          for (let j = 0; j < testVectorSize; j++) {
            vec[j] = Math.random() * 2 - 1;
          }
          docVectors.push(vec);
        }

        const batchStart = performance.now();
        const batchResults = await webgpuHelpers.batchSimilarity(
          queryVector, 
          docVectors, 
          { topK: 5, threshold: 0.0 }
        );
        const batchTime = performance.now() - batchStart;

        demoResults.push({
          type: 'Batch Search',
          gpu: { 
            result: `${batchResults.length} results`, 
            time: batchTime.toFixed(2),
            details: `Top similarity: ${batchResults[0]?.similarity.toFixed(4) || 'N/A'}`
          },
          cpu: { result: 'N/A', time: 'N/A' },
          speedup: 'GPU Only'
        });
      }

    } catch (error) {
      console.error('Vector demo failed:', error);
      demoResults = [{ type: 'Error', gpu: { result: error.message }, cpu: { result: 'N/A' }, speedup: 'N/A' }];
    } finally {
      isRunning = false;
    }
  }

  async function runRAGDemo() {
    isRunning = true;
    demoResults = [];

    try {
      console.log('🔄 Running RAG demo...');

      // Simulate document processing
      const sampleDocuments = [
        {
          id: 'doc1',
          content: 'Employment contract with liability clauses and termination procedures. The employer shall not be liable for damages exceeding the agreed compensation.',
          metadata: { type: 'contract', category: 'employment' }
        },
        {
          id: 'doc2', 
          content: 'Personal injury case involving negligence and damages. The defendant failed to exercise reasonable care resulting in significant harm.',
          metadata: { type: 'case', category: 'personal-injury' }
        },
        {
          id: 'doc3',
          content: 'Intellectual property agreement covering patent rights and licensing terms. Violations may result in statutory damages and injunctive relief.',
          metadata: { type: 'agreement', category: 'ip' }
        }
      ];

      // Process documents
      const processingResults = [];
      for (const doc of sampleDocuments) {
        const result = await webgpuRAGService.processDocument(doc.id, doc.content, doc.metadata);
        processingResults.push({
          documentId: doc.id,
          ...result
        });
      }

      // Perform semantic search
      const searchStart = performance.now();
      const searchResult = await webgpuRAGService.semanticSearch(testQuery, {
        topK: 5,
        threshold: 0.3,
        useGPU: true
      });
      const searchTime = performance.now() - searchStart;

      demoResults = [
        {
          type: 'Document Processing',
          gpu: { 
            result: `${processingResults.length} docs processed`,
            time: processingResults.reduce((sum, r) => sum + r.processingTime, 0).toFixed(2),
            details: `Total chunks: ${processingResults.reduce((sum, r) => sum + r.chunks, 0)}`
          },
          cpu: { result: 'N/A', time: 'N/A' },
          speedup: 'GPU Accelerated'
        },
        {
          type: 'Semantic Search',
          gpu: { 
            result: `${searchResult.results.length} results`,
            time: searchTime.toFixed(2),
            details: `Top match: ${(searchResult.results[0]?.similarity * 100).toFixed(1)}% similarity`
          },
          cpu: { result: 'Fallback available', time: 'N/A' },
          speedup: searchResult.usedGPU ? 'GPU Used' : 'CPU Fallback'
        }
      ];

    } catch (error) {
      console.error('RAG demo failed:', error);
      demoResults = [{ type: 'Error', gpu: { result: error.message }, cpu: { result: 'N/A' }, speedup: 'N/A' }];
    } finally {
      isRunning = false;
    }
  }

  async function runBenchmark() {
    isRunning = true;
    
    try {
      console.log('🔄 Running comprehensive benchmark...');

      const benchmark = {
        vectorOperations: { gpu: 0, cpu: 0, speedup: 1 },
        batchProcessing: { gpu: 0, cpu: 0, speedup: 1 },
        memoryThroughput: { gpu: 0, cpu: 0, speedup: 1 },
        overall: { score: 0, recommendation: '' }
      };

      // Vector operations benchmark
      const vectors = Array(100).fill(null).map(() => {
        const vec = new Float32Array(384);
        for (let i = 0; i < 384; i++) {
          vec[i] = Math.random() * 2 - 1;
        }
        return vec;
      });

      // GPU benchmark
      const gpuStart = performance.now();
      for (let i = 0; i < vectors.length - 1; i++) {
        await webgpuHelpers.computeSimilarity(vectors[i], vectors[i + 1]);
      }
      benchmark.vectorOperations.gpu = performance.now() - gpuStart;

      // CPU benchmark
      const cpuStart = performance.now();
      for (let i = 0; i < vectors.length - 1; i++) {
        await computeSimilarityCPU(vectors[i], vectors[i + 1]);
      }
      benchmark.vectorOperations.cpu = performance.now() - cpuStart;
      
      benchmark.vectorOperations.speedup = benchmark.vectorOperations.cpu / benchmark.vectorOperations.gpu;

      // Batch processing benchmark
      const queryVec = vectors[0];
      const docVecs = vectors.slice(1, 51);

      const batchGpuStart = performance.now();
      await webgpuHelpers.batchSimilarity(queryVec, docVecs, { topK: 10 });
      benchmark.batchProcessing.gpu = performance.now() - batchGpuStart;

      const batchCpuStart = performance.now();
      // Simulate CPU batch processing
      for (const docVec of docVecs) {
        await computeSimilarityCPU(queryVec, docVec);
      }
      benchmark.batchProcessing.cpu = performance.now() - batchCpuStart;
      
      benchmark.batchProcessing.speedup = benchmark.batchProcessing.cpu / benchmark.batchProcessing.gpu;

      // Calculate overall score
      const avgSpeedup = (benchmark.vectorOperations.speedup + benchmark.batchProcessing.speedup) / 2;
      benchmark.overall.score = Math.min(100, avgSpeedup * 10);
      
      if (avgSpeedup > 5) {
        benchmark.overall.recommendation = 'Excellent GPU acceleration - use GPU for all operations';
      } else if (avgSpeedup > 2) {
        benchmark.overall.recommendation = 'Good GPU performance - use GPU for batch operations';
      } else if (avgSpeedup > 1.2) {
        benchmark.overall.recommendation = 'Moderate GPU benefits - use selectively';
      } else {
        benchmark.overall.recommendation = 'Limited GPU benefits - consider CPU fallback';
      }

      benchmarkResults = benchmark;

    } catch (error) {
      console.error('Benchmark failed:', error);
      benchmarkResults = { error: error.message };
    } finally {
      isRunning = false;
    }
  }

  async function computeSimilarityCPU(a: Float32Array, b: Float32Array): Promise<number> {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  function getStatusIcon(isSupported: boolean, isActive: boolean) {
    if (!isSupported) return AlertTriangle;
    if (isActive) return Activity;
    return CheckCircle;
  }

  function getStatusColor(isSupported: boolean, isActive: boolean) {
    if (!isSupported) return 'text-red-500';
    if (isActive) return 'text-blue-500';
    return 'text-green-500';
  }

  function resetDemo() {
    demoResults = [];
    benchmarkResults = null;
    webgpuManager.getMetrics();
  }
</script>

<div class="webgpu-demo bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 min-h-screen p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-8">
    <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class="w-12 h-12 bg-gradient-to-br from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
            <Zap class="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100">WebGPU Acceleration Demo</h1>
            <p class="text-slate-600 dark:text-slate-400">GPU-Powered Legal AI Computing</p>
          </div>
        </div>
        
        <!-- System Status -->
        <div class="flex items-center space-x-4">
          {@const StatusIcon = getStatusIcon(capabilities.isSupported, status.isReady)}
          <div class="flex items-center space-x-2 px-4 py-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
            <StatusIcon class="w-5 h-5 {getStatusColor(capabilities.isSupported, status.isReady)}" />
            <div>
              <p class="text-sm font-medium text-slate-900 dark:text-slate-100">
                {capabilities.isSupported ? (status.isReady ? 'GPU Ready' : 'GPU Available') : 'GPU Not Supported'}
              </p>
              <p class="text-xs text-slate-600 dark:text-slate-400">
                {capabilities.features.size} features, {Object.keys(capabilities.limits).length} limits
              </p>
            </div>
          </div>
          
          <button
            on:click={resetDemo}
            class="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            <RotateCcw class="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-8">
    <!-- Demo Controls -->
    <div class="lg:col-span-1 space-y-6">
      <!-- Demo Selection -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">GPU Demos</h3>
        
        <div class="space-y-3">
          {#each [
            { id: 'capabilities', label: 'GPU Capabilities', icon: Monitor, desc: 'Hardware features & limits' },
            { id: 'vector', label: 'Vector Operations', icon: Brain, desc: 'Similarity & embeddings' },
            { id: 'rag', label: 'RAG Processing', icon: Database, desc: 'Document analysis' },
            { id: 'performance', label: 'Performance', icon: TrendingUp, desc: 'Benchmarks & metrics' }
          ] as demo}
            <button
              on:click={() => selectedDemo = demo.id}
              class="w-full flex items-center space-x-3 p-3 rounded-lg transition-colors {
                selectedDemo === demo.id
                  ? 'bg-blue-100 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                  : 'hover:bg-slate-100 dark:hover:bg-slate-700'
              }"
            >
              <svelte:component this={demo.icon} class="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <div class="text-left">
                <p class="font-medium text-slate-900 dark:text-slate-100">{demo.label}</p>
                <p class="text-sm text-slate-600 dark:text-slate-400">{demo.desc}</p>
              </div>
            </button>
          {/each}
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Actions</h3>
        
        <div class="space-y-3">
          <button
            on:click={runVectorDemo}
            disabled={isRunning || !capabilities.isSupported}
            class="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
          >
            {#if isRunning}
              <Activity class="w-4 h-4 animate-spin" />
            {:else}
              <Brain class="w-4 h-4" />
            {/if}
            <span>Vector Demo</span>
          </button>
          
          <button
            on:click={runRAGDemo}
            disabled={isRunning || !capabilities.isSupported}
            class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
          >
            {#if isRunning}
              <Activity class="w-4 h-4 animate-spin" />
            {:else}
              <Database class="w-4 h-4" />
            {/if}
            <span>RAG Demo</span>
          </button>
          
          <button
            on:click={runBenchmark}
            disabled={isRunning || !capabilities.isSupported}
            class="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
          >
            {#if isRunning}
              <Activity class="w-4 h-4 animate-spin" />
            {:else}
              <BarChart3 class="w-4 h-4" />
            {/if}
            <span>Benchmark</span>
          </button>
        </div>
      </div>

      <!-- Test Parameters -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Parameters</h3>
        
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Vector Size
            </label>
            <input
              type="number"
              bind:value={testVectorSize}
              min="128"
              max="1024"
              step="64"
              class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Document Count
            </label>
            <input
              type="number"
              bind:value={testDocumentCount}
              min="5"
              max="100"
              step="5"
              class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Test Query
            </label>
            <input
              bind:value={testQuery}
              class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Main Demo Area -->
    <div class="lg:col-span-3 space-y-6">
      <!-- Selected Demo Content -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4 capitalize">
          {selectedDemo} Demo
        </h3>
        
        {#if selectedDemo === 'capabilities'}
          <div class="space-y-6">
            <!-- GPU Features -->
            <div>
              <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">GPU Features</h4>
              <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                {#each Array.from(capabilities.features) as feature}
                  <div class="p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
                    <p class="text-sm font-medium text-slate-900 dark:text-slate-100">{feature}</p>
                  </div>
                {/each}
              </div>
            </div>

            <!-- GPU Limits -->
            <div>
              <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">Key Limits</h4>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                  <p class="text-sm text-slate-600 dark:text-slate-400">Max Workgroups</p>
                  <p class="text-lg font-semibold text-slate-900 dark:text-slate-100">
                    {capabilities.maxComputeWorkgroupsPerDimension.toLocaleString()}
                  </p>
                </div>
                <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                  <p class="text-sm text-slate-600 dark:text-slate-400">Max Invocations</p>
                  <p class="text-lg font-semibold text-slate-900 dark:text-slate-100">
                    {capabilities.maxComputeInvocationsPerWorkgroup.toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        {:else if selectedDemo === 'vector'}
          <div class="space-y-4">
            {#if demoResults.length > 0}
              <div class="space-y-3">
                {#each demoResults as result}
                  <div class="p-4 border border-slate-200 dark:border-slate-600 rounded-lg">
                    <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">{result.type}</h4>
                    <div class="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p class="text-slate-600 dark:text-slate-400">GPU</p>
                        <p class="font-semibold text-green-600">{result.gpu.result}</p>
                        <p class="text-xs text-slate-500">{result.gpu.time}ms</p>
                        {#if result.gpu.details}
                          <p class="text-xs text-slate-500">{result.gpu.details}</p>
                        {/if}
                      </div>
                      <div>
                        <p class="text-slate-600 dark:text-slate-400">CPU</p>
                        <p class="font-semibold text-blue-600">{result.cpu.result}</p>
                        <p class="text-xs text-slate-500">{result.cpu.time}ms</p>
                      </div>
                      <div>
                        <p class="text-slate-600 dark:text-slate-400">Speedup</p>
                        <p class="font-semibold text-purple-600">{result.speedup}x</p>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            {:else}
              <div class="text-center py-8">
                <Brain class="w-12 h-12 text-slate-400 mx-auto mb-4" />
                <p class="text-slate-600 dark:text-slate-400">Click "Vector Demo" to run GPU vector operations</p>
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'rag'}
          <div class="space-y-4">
            {#if demoResults.length > 0}
              <div class="space-y-3">
                {#each demoResults as result}
                  <div class="p-4 border border-slate-200 dark:border-slate-600 rounded-lg">
                    <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">{result.type}</h4>
                    <div class="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p class="text-slate-600 dark:text-slate-400">Result</p>
                        <p class="font-semibold text-slate-900 dark:text-slate-100">{result.gpu.result}</p>
                        {#if result.gpu.details}
                          <p class="text-xs text-slate-500">{result.gpu.details}</p>
                        {/if}
                      </div>
                      <div>
                        <p class="text-slate-600 dark:text-slate-400">Processing Time</p>
                        <p class="font-semibold text-green-600">{result.gpu.time}ms</p>
                      </div>
                      <div>
                        <p class="text-slate-600 dark:text-slate-400">Acceleration</p>
                        <p class="font-semibold text-blue-600">{result.speedup}</p>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            {:else}
              <div class="text-center py-8">
                <Database class="w-12 h-12 text-slate-400 mx-auto mb-4" />
                <p class="text-slate-600 dark:text-slate-400">Click "RAG Demo" to run GPU-accelerated document processing</p>
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'performance'}
          <div class="space-y-6">
            <!-- Real-time Metrics -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Total Tasks</p>
                <p class="text-2xl font-bold text-slate-900 dark:text-slate-100">{metrics.totalTasks}</p>
              </div>
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Throughput</p>
                <p class="text-2xl font-bold text-green-600">{metrics.throughput.toFixed(1)}/s</p>
              </div>
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Avg Time</p>
                <p class="text-2xl font-bold text-blue-600">{metrics.averageExecutionTime.toFixed(1)}ms</p>
              </div>
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Success Rate</p>
                <p class="text-2xl font-bold text-purple-600">
                  {metrics.totalTasks > 0 ? ((metrics.completedTasks / metrics.totalTasks) * 100).toFixed(1) : 0}%
                </p>
              </div>
            </div>

            <!-- Benchmark Results -->
            {#if benchmarkResults}
              <div class="p-6 border border-slate-200 dark:border-slate-600 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-4">Benchmark Results</h4>
                
                {#if benchmarkResults.error}
                  <div class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p class="text-red-700 dark:text-red-300">Error: {benchmarkResults.error}</p>
                  </div>
                {:else}
                  <div class="space-y-4">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                        <p class="text-sm text-slate-600 dark:text-slate-400 mb-2">Vector Operations</p>
                        <p class="font-semibold text-slate-900 dark:text-slate-100">
                          {benchmarkResults.vectorOperations.speedup.toFixed(2)}x speedup
                        </p>
                        <p class="text-xs text-slate-500">
                          GPU: {benchmarkResults.vectorOperations.gpu.toFixed(1)}ms, 
                          CPU: {benchmarkResults.vectorOperations.cpu.toFixed(1)}ms
                        </p>
                      </div>
                      
                      <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                        <p class="text-sm text-slate-600 dark:text-slate-400 mb-2">Batch Processing</p>
                        <p class="font-semibold text-slate-900 dark:text-slate-100">
                          {benchmarkResults.batchProcessing.speedup.toFixed(2)}x speedup
                        </p>
                        <p class="text-xs text-slate-500">
                          GPU: {benchmarkResults.batchProcessing.gpu.toFixed(1)}ms, 
                          CPU: {benchmarkResults.batchProcessing.cpu.toFixed(1)}ms
                        </p>
                      </div>
                    </div>
                    
                    <div class="p-4 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg">
                      <p class="text-sm text-slate-600 dark:text-slate-400 mb-2">Overall Performance Score</p>
                      <p class="text-2xl font-bold text-slate-900 dark:text-slate-100">
                        {benchmarkResults.overall.score.toFixed(0)}/100
                      </p>
                      <p class="text-sm text-slate-600 dark:text-slate-400 mt-2">
                        {benchmarkResults.overall.recommendation}
                      </p>
                    </div>
                  </div>
                {/if}
              </div>
            {:else}
              <div class="text-center py-8">
                <BarChart3 class="w-12 h-12 text-slate-400 mx-auto mb-4" />
                <p class="text-slate-600 dark:text-slate-400">Click "Benchmark" to run performance tests</p>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .webgpu-demo {
    font-family: system-ui, -apple-system, sans-serif;
  }
</style>