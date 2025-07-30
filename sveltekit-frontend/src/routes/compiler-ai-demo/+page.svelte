<!-- Compiler AI Feedback Loop Demo -->
<!-- Real-time demonstration of AI-driven development architecture -->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import Card from '$lib/components/ui/Card.svelte';
  import { createEnhancedRAGEngine } from '$lib/services/enhanced-rag-pagerank';
  import { createCompilerFeedbackLoop, type CompilerEvent, type PatchCandidate, type SOMCluster } from '$lib/services/compiler-feedback-loop';
  import { AlertCircle, CheckCircle, Cpu, Zap, Activity, Code, Brain, Target } from 'lucide-svelte';

  // System instances
  let ragEngine: any = null;
  let feedbackLoop: any = null;
  let systemInitialized = $state(false);
  
  // Real-time state
  let events = $state<CompilerEvent[]>([]);
  let patches = $state<PatchCandidate[]>([]);
  let clusters = $state<SOMCluster[]>([]);
  let performance = $state({
    totalEvents: 0,
    successfulPatches: 0,
    averageProcessingTime: 0,
    clusterCount: 0
  });
  
  // Demo controls
  let isMonitoring = $state(false);
  let selectedEvent: CompilerEvent | null = $state(null);
  let selectedPatch: PatchCandidate | null = $state(null);

  onMount(async () => {
    try {
      console.log('🚀 Initializing Compiler AI Demo...');
      
      // Initialize Enhanced RAG Engine
      ragEngine = createEnhancedRAGEngine({
        enablePageRank: true,
        enableUserFeedback: true,
        enableRealTimeUpdates: true,
        vectorDimensions: 384,
        maxDocuments: 1000
      });

      // Initialize Compiler Feedback Loop
      feedbackLoop = createCompilerFeedbackLoop(ragEngine);
      
      // Subscribe to reactive stores
      feedbackLoop.events.subscribe((newEvents: CompilerEvent[]) => {
        events = newEvents;
      });
      
      feedbackLoop.patches.subscribe((newPatches: PatchCandidate[]) => {
        patches = newPatches;
      });
      
      feedbackLoop.clusters.subscribe((newClusters: SOMCluster[]) => {
        clusters = newClusters;
      });
      
      feedbackLoop.performance.subscribe((newPerformance: any) => {
        performance = newPerformance;
      });

      systemInitialized = true;
      console.log('✅ Compiler AI Demo initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize Compiler AI Demo:', error);
    }
  });

  onDestroy(() => {
    if (feedbackLoop && isMonitoring) {
      feedbackLoop.stopMonitoring();
    }
  });

  async function startMonitoring() {
    if (!feedbackLoop) return;
    
    try {
      await feedbackLoop.startMonitoring();
      isMonitoring = true;
      console.log('🎯 Compiler monitoring started');
    } catch (error) {
      console.error('❌ Failed to start monitoring:', error);
    }
  }

  function stopMonitoring() {
    if (!feedbackLoop) return;
    
    feedbackLoop.stopMonitoring();
    isMonitoring = false;
    console.log('⏹️ Compiler monitoring stopped');
  }

  function addTestError() {
    if (!feedbackLoop) return;
    
    const testEvent: CompilerEvent = {
      type: 'ERROR_DETECTED',
      logs: [{
        id: `test_${Date.now()}`,
        timestamp: Date.now(),
        level: 'error',
        message: 'Property \'nonExistent\' does not exist on type \'TestInterface\'',
        file: 'src/test/demo.ts',
        line: Math.floor(Math.random() * 100) + 1,
        code: 'const value = obj.nonExistent;',
        metadata: {
          component: 'TypeScript',
          phase: 'type-checking',
          category: 'type'
        }
      }],
      performance: {
        compilationTime: Math.random() * 2000 + 500,
        memoryUsage: Math.random() * 100 + 20,
        errorCount: 1,
        warningCount: Math.floor(Math.random() * 3)
      }
    };
    
    feedbackLoop.addEvent(testEvent);
  }

  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString();
  }

  function getEventIcon(event: CompilerEvent) {
    switch (event.type) {
      case 'ERROR_DETECTED': return AlertCircle;
      case 'PATCH_SUGGESTED': return CheckCircle;
      case 'COMPILE_COMPLETE': return Code;
      default: return Activity;
    }
  }

  function getPatchIcon(patch: PatchCandidate) {
    switch (patch.category) {
      case 'fix': return AlertCircle;
      case 'optimization': return Zap;
      case 'refactor': return Brain;
      case 'enhancement': return Target;
      default: return Code;
    }
  }

  function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  }

  function generateHeatmapGradient(weights: Float32Array): string {
    if (!weights || weights.length === 0) return 'rgba(0,0,0,0.1)';
    
    const maxWeight = Math.max(...Array.from(weights));
    const colors = [];
    
    for (let i = 0; i < Math.min(10, weights.length); i++) {
      const intensity = weights[i] / maxWeight;
      colors.push(`rgba(59, 130, 246, ${intensity * 0.8})`);
    }
    
    return `linear-gradient(90deg, ${colors.join(', ')})`;
  }
</script>

<svelte:head>
  <title>Compiler AI Demo - AI-Driven Development</title>
  <meta name="description" content="Real-time demonstration of AI-driven compiler feedback loop with vector embeddings, clustering, and patch generation">
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-8">
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold mb-4">
        <span class="bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Compiler AI Feedback Loop
        </span>
      </h1>
      <p class="text-xl text-gray-300 max-w-3xl mx-auto">
        Real-time AI-driven development architecture with vector embeddings, self-organizing map clustering, 
        enhanced RAG search, and multi-agent patch generation.
      </p>
    </div>

    <!-- System Status -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={systemInitialized} class:bg-red-100={!systemInitialized}>
            <Cpu class="h-5 w-5 {systemInitialized ? 'text-green-600' : 'text-red-600'}" />
          </div>
          <div>
            <h3 class="font-semibold text-white">System Status</h3>
            <p class="text-sm text-gray-400">{systemInitialized ? 'Ready' : 'Initializing'}</p>
          </div>
        </div>
      </Card>

      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={isMonitoring} class:bg-red-100={!isMonitoring}>
            <Activity class="h-5 w-5 {isMonitoring ? 'text-green-600' : 'text-red-600'}" />
          </div>
          <div>
            <h3 class="font-semibold text-white">Monitoring</h3>
            <p class="text-sm text-gray-400">{isMonitoring ? 'Active' : 'Inactive'}</p>
          </div>
        </div>
      </Card>

      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg bg-blue-100">
            <Brain class="h-5 w-5 text-blue-600" />
          </div>
          <div>
            <h3 class="font-semibold text-white">AI Patches</h3>
            <p class="text-sm text-gray-400">{performance.successfulPatches} generated</p>
          </div>
        </div>
      </Card>

      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg bg-purple-100">
            <Target class="h-5 w-5 text-purple-600" />
          </div>
          <div>
            <h3 class="font-semibold text-white">Clusters</h3>
            <p class="text-sm text-gray-400">{performance.clusterCount} patterns</p>
          </div>
        </div>
      </Card>
    </div>

    <!-- Controls -->
    <div class="flex gap-4 mb-8 justify-center">
      <Button 
        onclick={startMonitoring} 
        disabled={!systemInitialized || isMonitoring}
        class="bg-green-600 hover:bg-green-700"
      >
        Start Monitoring
      </Button>
      
      <Button 
        onclick={stopMonitoring} 
        disabled={!isMonitoring}
        class="bg-red-600 hover:bg-red-700"
      >
        Stop Monitoring
      </Button>
      
      <Button 
        onclick={addTestError} 
        disabled={!systemInitialized}
        class="bg-orange-600 hover:bg-orange-700"
      >
        Add Test Error
      </Button>
    </div>
  </div>

  <!-- Main Content -->
  <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Real-time Events -->
    <div class="lg:col-span-1">
      <Card class="p-6 bg-slate-800/30 border-slate-600 h-96">
        <h2 class="text-xl font-bold mb-4 flex items-center gap-2">
          <Activity class="h-5 w-5" />
          Real-time Events
        </h2>
        
        <div class="space-y-3 max-h-80 overflow-y-auto">
          {#each events.slice(-10).reverse() as event}
            {@const IconComponent = getEventIcon(event)}
            <div 
              class="p-3 rounded-lg bg-slate-700/50 border border-slate-600 cursor-pointer hover:bg-slate-700 transition-colors"
              onclick={() => selectedEvent = event}
              class:ring-2={selectedEvent?.logs[0]?.id === event.logs[0]?.id}
              class:ring-blue-500={selectedEvent?.logs[0]?.id === event.logs[0]?.id}
            >
              <div class="flex items-start gap-3">
                <IconComponent class="h-4 w-4 mt-1 text-orange-400" />
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-medium text-white truncate">
                    {event.logs[0]?.message || 'Unknown event'}
                  </p>
                  <p class="text-xs text-gray-400">
                    {event.logs[0]?.file || 'Unknown file'} • {formatTimestamp(event.logs[0]?.timestamp || Date.now())}
                  </p>
                  {#if event.clusterId}
                    <span class="inline-block px-2 py-1 text-xs bg-purple-600 rounded mt-2">
                      Cluster: {event.clusterId.slice(-8)}
                    </span>
                  {/if}
                </div>
              </div>
            </div>
          {/each}
          
          {#if events.length === 0}
            <div class="text-center py-8 text-gray-400">
              <Activity class="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No events yet. Start monitoring to see real-time compiler feedback.</p>
            </div>
          {/if}
        </div>
      </Card>
    </div>

    <!-- AI Patches -->
    <div class="lg:col-span-1">
      <Card class="p-6 bg-slate-800/30 border-slate-600 h-96">
        <h2 class="text-xl font-bold mb-4 flex items-center gap-2">
          <Brain class="h-5 w-5" />
          AI Generated Patches
        </h2>
        
        <div class="space-y-3 max-h-80 overflow-y-auto">
          {#each patches.slice(-5).reverse() as patch}
            {@const IconComponent = getPatchIcon(patch)}
            <div 
              class="p-3 rounded-lg bg-slate-700/50 border border-slate-600 cursor-pointer hover:bg-slate-700 transition-colors"
              onclick={() => selectedPatch = patch}
              class:ring-2={selectedPatch?.id === patch.id}
              class:ring-green-500={selectedPatch?.id === patch.id}
            >
              <div class="flex items-start gap-3">
                <IconComponent class="h-4 w-4 mt-1 text-blue-400" />
                <div class="flex-1 min-w-0">
                  <div class="flex items-center justify-between mb-1">
                    <p class="text-sm font-medium text-white truncate">
                      {patch.description}
                    </p>
                    <span class="text-xs font-bold {getConfidenceColor(patch.confidence)}">
                      {(patch.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p class="text-xs text-gray-400 mb-2">
                    {patch.affectedFiles.join(', ')} • {patch.agentSource}
                  </p>
                  <div class="flex gap-2">
                    <span class="px-2 py-1 text-xs bg-gray-600 rounded">
                      {patch.category}
                    </span>
                    <span class="px-2 py-1 text-xs bg-gray-600 rounded">
                      {patch.estimatedImpact} impact
                    </span>
                  </div>
                  {#if patch.testResults}
                    <div class="mt-2 text-xs">
                      <span class="text-{patch.testResults.passed ? 'green' : 'red'}-400">
                        Tests: {patch.testResults.passed ? 'Passed' : 'Failed'}
                      </span>
                      <span class="text-gray-400 ml-2">
                        Coverage: {patch.testResults.coverage.toFixed(1)}%
                      </span>
                    </div>
                  {/if}
                </div>
              </div>
            </div>
          {/each}
          
          {#if patches.length === 0}
            <div class="text-center py-8 text-gray-400">
              <Brain class="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No patches generated yet. Compiler errors will trigger AI patch suggestions.</p>
            </div>
          {/if}
        </div>
      </Card>
    </div>

    <!-- Performance & Clusters -->
    <div class="lg:col-span-1">
      <Card class="p-6 bg-slate-800/30 border-slate-600 h-96">
        <h2 class="text-xl font-bold mb-4 flex items-center gap-2">
          <Target class="h-5 w-5" />
          Performance & Clusters
        </h2>
        
        <div class="space-y-4">
          <!-- Performance Metrics -->
          <div class="grid grid-cols-2 gap-3">
            <div class="bg-slate-700/50 p-3 rounded">
              <p class="text-xs text-gray-400">Total Events</p>
              <p class="text-lg font-bold">{performance.totalEvents}</p>
            </div>
            <div class="bg-slate-700/50 p-3 rounded">
              <p class="text-xs text-gray-400">Avg Processing</p>
              <p class="text-lg font-bold">{performance.averageProcessingTime.toFixed(0)}ms</p>
            </div>
            <div class="bg-slate-700/50 p-3 rounded">
              <p class="text-xs text-gray-400">Success Rate</p>
              <p class="text-lg font-bold text-green-400">
                {performance.totalEvents > 0 ? ((performance.successfulPatches / performance.totalEvents) * 100).toFixed(0) : 0}%
              </p>
            </div>
            <div class="bg-slate-700/50 p-3 rounded">
              <p class="text-xs text-gray-400">Clusters</p>
              <p class="text-lg font-bold text-purple-400">{performance.clusterCount}</p>
            </div>
          </div>

          <!-- SOM Clusters -->
          <div>
            <h3 class="font-semibold mb-2">Error Pattern Clusters</h3>
            <div class="space-y-2 max-h-40 overflow-y-auto">
              {#each clusters.slice(0, 5) as cluster}
                <div class="bg-slate-700/50 p-2 rounded text-xs">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">Cluster {cluster.id.slice(-8)}</span>
                    <span class="text-gray-400">{cluster.frequency}x</span>
                  </div>
                  <p class="text-gray-400 mt-1">{cluster.errorPattern}</p>
                  <p class="text-purple-400 mt-1">{cluster.members.length} members</p>
                </div>
              {/each}
              
              {#if clusters.length === 0}
                <p class="text-gray-400 text-center py-4">No clusters formed yet</p>
              {/if}
            </div>
          </div>
        </div>
      </Card>
    </div>
  </div>

  <!-- Detailed Views -->
  {#if selectedEvent || selectedPatch}
    <div class="max-w-7xl mx-auto mt-6">
      {#if selectedEvent}
        <Card class="p-6 bg-slate-800/30 border-slate-600 mb-6">
          <h3 class="text-lg font-bold mb-4">Event Details</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 class="font-semibold mb-2">Compiler Log</h4>
              <div class="bg-slate-900 p-4 rounded font-mono text-sm">
                <p class="text-red-400">{selectedEvent.logs[0]?.level.toUpperCase()}: {selectedEvent.logs[0]?.message}</p>
                <p class="text-gray-400">File: {selectedEvent.logs[0]?.file}:{selectedEvent.logs[0]?.line}</p>
                {#if selectedEvent.logs[0]?.code}
                  <p class="text-yellow-400 mt-2">Code: {selectedEvent.logs[0].code}</p>
                {/if}
              </div>
            </div>
            <div>
              <h4 class="font-semibold mb-2">Performance Impact</h4>
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span>Compilation Time:</span>
                  <span class="font-mono">{selectedEvent.performance.compilationTime}ms</span>
                </div>
                <div class="flex justify-between">
                  <span>Memory Usage:</span>
                  <span class="font-mono">{selectedEvent.performance.memoryUsage.toFixed(1)}MB</span>
                </div>
                <div class="flex justify-between">
                  <span>Error Count:</span>
                  <span class="font-mono text-red-400">{selectedEvent.performance.errorCount}</span>
                </div>
                <div class="flex justify-between">
                  <span>Warning Count:</span>
                  <span class="font-mono text-yellow-400">{selectedEvent.performance.warningCount}</span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      {/if}

      {#if selectedPatch}
        <Card class="p-6 bg-slate-800/30 border-slate-600">
          <h3 class="text-lg font-bold mb-4">Patch Details</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 class="font-semibold mb-2">Generated Patch</h4>
              <div class="bg-slate-900 p-4 rounded font-mono text-sm max-h-60 overflow-y-auto">
                <pre class="text-green-400">{selectedPatch.diff}</pre>
              </div>
              
              {#if selectedPatch.attentionWeights}
                <div class="mt-4">
                  <h4 class="font-semibold mb-2">Attention Heatmap</h4>
                  <div 
                    class="h-8 rounded"
                    style="background: {generateHeatmapGradient(selectedPatch.attentionWeights.weights)}"
                  ></div>
                  <p class="text-xs text-gray-400 mt-1">
                    Focus areas: {selectedPatch.attentionWeights.focusAreas.length} regions identified
                  </p>
                </div>
              {/if}
            </div>
            <div>
              <h4 class="font-semibold mb-2">Patch Metadata</h4>
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span>Confidence:</span>
                  <span class="font-bold {getConfidenceColor(selectedPatch.confidence)}">
                    {(selectedPatch.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div class="flex justify-between">
                  <span>Category:</span>
                  <span class="font-mono">{selectedPatch.category}</span>
                </div>
                <div class="flex justify-between">
                  <span>Impact:</span>
                  <span class="font-mono">{selectedPatch.estimatedImpact}</span>
                </div>
                <div class="flex justify-between">
                  <span>Agent Source:</span>
                  <span class="font-mono">{selectedPatch.agentSource}</span>
                </div>
              </div>
              
              <div class="mt-4">
                <h4 class="font-semibold mb-2">Affected Files</h4>
                <div class="space-y-1">
                  {#each selectedPatch.affectedFiles as file}
                    <div class="text-sm font-mono bg-slate-700/50 px-2 py-1 rounded">
                      {file}
                    </div>
                  {/each}
                </div>
              </div>
            </div>
          </div>
        </Card>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* Custom scrollbar for dark theme */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: rgba(51, 65, 85, 0.3);
    border-radius: 3px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background: rgba(71, 85, 105, 0.8);
    border-radius: 3px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: rgba(71, 85, 105, 1);
  }
</style>