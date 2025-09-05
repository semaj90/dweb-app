<!-- Phase 13 Enhanced Features Demo -->
<!-- Complete integration of XState, WebGL, Enhanced RAG, and Context7 MCP -->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import Card from '$lib/components/ui/Card.svelte';
  import { createPhase13Integration } from '$lib/state/phase13StateMachine';
  import { createStatelessAPICoordinator } from '$lib/services/stateless-api-coordinator';
  import { createEnhancedRAGEngine, RAGHelpers } from '$lib/services/enhanced-rag-pagerank';
  import { createContext7Phase13Integration } from '$lib/services/context7-phase13-integration';
  import { CheckCircle, AlertCircle, Cpu, Database, Zap, Activity } from 'lucide-svelte';

  // Phase 13 system instances
  let canvas: HTMLCanvasElement
  let phase13System: any = null;
  let apiCoordinator: any = null;
  let ragEngine: any = null;
  let context7Integration: any = null;

  // Demo state
  let systemInitialized = $state(false);
  let webglReady = $state(false);
  let apiActive = $state(false);
  let ragActive = $state(false);
  let context7Active = $state(false);

  // Performance metrics
  let frameRate = $state(0);
  let apiThroughput = $state(0);
  let ragQueryTime = $state(0);
  let pageRankScore = $state(0);
  let feedbackCount = $state(0);

  // Demo data
  let searchQuery = $state("contract liability clauses");
  let searchResults = $state<any[]>([]);
  let recommendations = $state<any[]>([]);
  let systemHealth = $state(0);

  // Initialize Phase 13 system
  onMount(async () => {
    try {
      console.log('🚀 Initializing Phase 13 Enhanced Features...');

      // 1. Initialize API Coordinator
      apiCoordinator = createStatelessAPICoordinator({
        enableRedis: true,
        enableNATS: true,
        enableWebSocket: true,
        taskTimeout: 30000
      });
      
      // 2. Initialize Enhanced RAG Engine
      ragEngine = createEnhancedRAGEngine(apiCoordinator.coordinator);
      
      // 3. Initialize Context7 MCP Integration
      context7Integration = createContext7Phase13Integration({
        enableSemanticSearch: true,
        enableMemoryGraph: true,
        enableAgentOrchestration: true,
        enableBestPractices: true
      }, ragEngine.engine, apiCoordinator.coordinator);

      // 4. Initialize Phase 13 State Machine with WebGL
      if (canvas) {
        phase13System = createPhase13Integration(canvas);
        
        // Subscribe to system stores
        phase13System.stores.webglStatus.subscribe((status: any) => {
          webglReady = status.initialized && status.streaming;
        });

        phase13System.stores.apiCoordination.subscribe((coordination: any) => {
          apiActive = coordination.active;
        });

        phase13System.stores.performanceMetrics.subscribe((metrics: any) => {
          frameRate = metrics.frameRate;
        });
      }

      // Subscribe to API coordinator stores
      apiCoordinator.stores.systemHealth.subscribe((health: any) => {
        systemHealth = health.overall;
      });

      apiCoordinator.stores.throughputMetrics.subscribe((metrics: any) => {
        apiThroughput = metrics.tasksPerSecond;
      });

      // Subscribe to RAG engine stores
      ragEngine.stores.queryResults.subscribe((results: any) => {
        if (results.size > 0) {
          const latestResults = Array.from(results.values()).pop() || [];
          searchResults = Array.isArray(latestResults) ? latestResults.slice(0, 5) : [];
        }
      });

      ragEngine.stores.feedbackMetrics.subscribe((metrics: any) => {
        feedbackCount = metrics.totalVotes;
        pageRankScore = metrics.averageRelevance;
      });

      // Subscribe to Context7 integration stores
      context7Integration.stores.activeRecommendations.subscribe((recs: any) => {
        recommendations = Array.isArray(recs) ? recs.slice(0, 3) : [];
      });

      context7Integration.stores.integrationStatus.subscribe((status: any) => {
        context7Active = status.overall === "HEALTHY";
      });

      // Add sample documents to RAG
      addSampleDocuments();

      systemInitialized = true;
      console.log('✅ Phase 13 system initialized successfully');

    } catch (error) {
      console.error('❌ Phase 13 initialization failed:', error);
    }
  });

  onDestroy(() => {
    if (phase13System) phase13System.destroy?.();
    if (apiCoordinator) apiCoordinator.destroy();
    if (ragEngine) ragEngine.destroy();
    if (context7Integration) context7Integration.destroy();
  });

  // Add sample legal documents
  function addSampleDocuments() {
    if (!ragEngine) return;

    const sampleDocs = [
      {
        title: "Employment Contract Template",
        content: "This employment contract contains liability clauses for workplace safety, intellectual property protection, and confidentiality agreements. The contractor assumes liability for damages caused by negligence.",
        type: "CONTRACT" as const,
        metadata: { jurisdiction: "Federal", caseId: "CASE-2024-001" }
      },
      {
        title: "Product Liability Case Study",
        content: "In this landmark case, the court established precedent for manufacturer liability in cases of defective products. The ruling clarifies burden of proof requirements for consumer protection claims.",
        type: "CASE_LAW" as const,
        metadata: { jurisdiction: "Federal", caseId: "CASE-2024-002" }
      },
      {
        title: "Consumer Protection Statute",
        content: "Federal regulations governing consumer rights and business liability. Includes provisions for warranty claims, return policies, and damage compensation procedures.",
        type: "STATUTE" as const,
        metadata: { jurisdiction: "Federal" }
      }
    ];

    for (const doc of sampleDocs) {
      const fullDoc = RAGHelpers.createDocument(
        doc.content,
        doc.title,
        doc.type,
        doc.metadata
      );
      ragEngine.addDocument(fullDoc);
    }
  }

  // Demo functions
  async function startWebGLDemo() {
    if (!phase13System) return;
    
    try {
      // Generate sample vertex data
      const vertices = new Float32Array([
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0
      ]);
      
      phase13System.startVertexStreaming(vertices);
      console.log('🎮 WebGL vertex streaming started');
    } catch (error) {
      console.error('WebGL demo failed:', error);
    }
  }

  async function startAPICoordination() {
    if (!phase13System) return;
    
    try {
      phase13System.startAPICoordination();
      
      // Submit sample tasks
      if (apiCoordinator) {
        await apiCoordinator.submitTask({
          type: "LEGAL_ANALYSIS",
          payload: { caseId: "CASE-2024-001", query: searchQuery },
          priority: "HIGH",
          maxRetries: 2,
          timeout: 30000,
          metadata: { estimatedDuration: 15000 }
        });
      }
      
      console.log('📡 API coordination started');
    } catch (error) {
      console.error('API coordination failed:', error);
    }
  }

  async function performEnhancedRAGSearch() {
    if (!ragEngine) return;
    
    try {
      ragActive = true;
      const startTime = Date.now();
      
      const query = RAGHelpers.createLegalQuery(searchQuery, {
        caseId: "CASE-2024-001",
        jurisdiction: "Federal",
        documentTypes: ["CONTRACT", "CASE_LAW", "STATUTE"],
        maxResults: 10
      });

      const results = await ragEngine.queryDocuments({
        ...query,
        id: `query_${Date.now()}`,
        timestamp: Date.now(),
        sessionId: "demo_session"
      });

      ragQueryTime = Date.now() - startTime;
      searchResults = results.slice(0, 5);
      
      console.log('🔍 Enhanced RAG search completed:', results.length, 'results');
    } catch (error) {
      console.error('RAG search failed:', error);
    } finally {
      ragActive = false;
    }
  }

  async function getContext7Recommendations() {
    if (!context7Integration) return;
    
    try {
      const recs = await context7Integration.getRecommendations(
        `Legal AI system optimization for: ${searchQuery}`,
        {
          priority: "HIGH",
          includeImplementationPlan: true,
          maxRecommendations: 5
        }
      );
      
      recommendations = recs.slice(0, 3);
      console.log('🧠 Context7 recommendations received:', recs.length);
    } catch (error) {
      console.error('Context7 recommendations failed:', error);
    }
  }

  async function submitPositiveFeedback(resultIndex: number) {
    if (!ragEngine || !searchResults[resultIndex]) return;
    
    try {
      await ragEngine.submitFeedback({
        queryId: "demo_query",
        documentId: searchResults[resultIndex].document.id,
        vote: "POSITIVE",
        relevanceScore: 0.9,
        context: {
          queryText: searchQuery,
          resultPosition: resultIndex,
          timeSpentViewing: 5000
        }
      });
      
      // Update PageRank
      phase13System.updatePageRank(searchResults[resultIndex].document.id, 0.1);
      
      console.log('👍 Positive feedback submitted');
    } catch (error) {
      console.error('Feedback submission failed:', error);
    }
  }

  async function runFullDemo() {
    try {
      console.log('🚀 Starting full Phase 13 demo...');
      
      await startWebGLDemo();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await startAPICoordination();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await performEnhancedRAGSearch();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await getContext7Recommendations();
      
      console.log('✅ Full Phase 13 demo completed successfully!');
    } catch (error) {
      console.error('❌ Full demo failed:', error);
    }
  }
</script>

<svelte:head>
  <title>Phase 13 Enhanced Features Demo - Deeds Legal AI</title>
  <meta name="description" content="Demonstration of Phase 13 enhanced features with XState, WebGL, Enhanced RAG, and Context7 MCP integration" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
  <div class="container mx-auto px-4 py-8">
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
        🚀 Phase 13 Enhanced Features Demo
      </h1>
      <p class="text-xl text-gray-300 max-w-4xl mx-auto">
        XState + WebGL Vertex Streaming + Enhanced RAG + Context7 MCP Integration
      </p>
    </div>

    <!-- System Status -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={webglReady} class:bg-red-100={!webglReady}>
            <Cpu class="h-5 w-5 {webglReady ? 'text-green-600' : 'text-red-600'}" />
          </div>
          <div>
            <h3 class="font-semibold text-white">WebGL Streaming</h3>
            <p class="text-sm text-gray-400">{frameRate}fps</p>
          </div>
        </div>
      </Card>

      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={apiActive} class:bg-red-100={!apiActive}>
            <Database class="h-5 w-5 {apiActive ? 'text-green-600' : 'text-red-600'}" />
          </div>
          <div>
            <h3 class="font-semibold text-white">API Coordination</h3>
            <p class="text-sm text-gray-400">{apiThroughput.toFixed(1)} tasks/s</p>
          </div>
        </div>
      </Card>

      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={ragActive || searchResults.length > 0} class:bg-red-100={!ragActive && searchResults.length === 0}>
            <Zap class="h-5 w-5 {(ragActive || searchResults.length > 0) ? 'text-green-600' : 'text-red-600'}" />
          </div>
          <div>
            <h3 class="font-semibold text-white">Enhanced RAG</h3>
            <p class="text-sm text-gray-400">{ragQueryTime}ms</p>
          </div>
        </div>
      </Card>

      <Card class="p-4 bg-slate-800/50 border-slate-700">
        <div class="flex items-center gap-3">
          <div class="p-2 rounded-lg" class:bg-green-100={context7Active} class:bg-red-100={!context7Active}>
            <Activity class="h-5 w-5 {context7Active ? 'text-green-600' : 'text-red-600'}" />
          </div>
          <div>
            <h3 class="font-semibold text-white">Context7 MCP</h3>
            <p class="text-sm text-gray-400">{systemHealth}% health</p>
          </div>
        </div>
      </Card>
    </div>

    <!-- WebGL Canvas -->
    <Card class="p-6 mb-8 bg-slate-800/50 border-slate-700">
      <h3 class="text-lg font-semibold text-white mb-4">WebGL Vertex Streaming Canvas</h3>
      <div class="flex gap-4 items-start">
        <canvas 
          bind:this={canvas}
          width="400" 
          height="300" 
          class="border border-slate-600 rounded bg-black"
        ></canvas>
        <div class="flex flex-col gap-2">
          <Button 
            onclick={startWebGLDemo} 
            disabled={!systemInitialized}
            variant="outline"
            class="text-white border-slate-600 hover:bg-slate-700"
          >
            Start WebGL Demo
          </Button>
          <p class="text-sm text-gray-400">
            Frame Rate: {frameRate} fps<br>
            Status: {webglReady ? 'Streaming' : 'Idle'}
          </p>
        </div>
      </div>
    </Card>

    <!-- Control Panel -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
      <!-- Search Controls -->
      <Card class="p-6 bg-slate-800/50 border-slate-700">
        <h3 class="text-lg font-semibold text-white mb-4">Enhanced RAG Search</h3>
        <div class="space-y-4">
          <input 
            bind:value={searchQuery}
            placeholder="Enter legal search query..."
            class="w-full p-3 rounded border border-slate-600 bg-slate-700 text-white placeholder-gray-400"
          />
          <div class="flex gap-2">
            <Button 
              onclick={performEnhancedRAGSearch}
              disabled={!systemInitialized || ragActive}
              class="bg-blue-600 hover:bg-blue-700"
            >
              {ragActive ? 'Searching...' : 'Search'}
            </Button>
            <Button 
              onclick={getContext7Recommendations}
              disabled={!systemInitialized}
              variant="outline"
              class="text-white border-slate-600 hover:bg-slate-700"
            >
              Get Recommendations
            </Button>
          </div>
          <div class="text-sm text-gray-400">
            PageRank Score: {pageRankScore.toFixed(3)} | Feedback: {feedbackCount} votes
          </div>
        </div>
      </Card>

      <!-- System Controls -->
      <Card class="p-6 bg-slate-800/50 border-slate-700">
        <h3 class="text-lg font-semibold text-white mb-4">System Controls</h3>
        <div class="space-y-4">
          <Button 
            onclick={startAPICoordination}
            disabled={!systemInitialized || apiActive}
            class="w-full bg-green-600 hover:bg-green-700"
          >
            {apiActive ? 'API Active' : 'Start API Coordination'}
          </Button>
          <Button 
            onclick={runFullDemo}
            disabled={!systemInitialized}
            class="w-full bg-purple-600 hover:bg-purple-700"
          >
            Run Full Demo
          </Button>
          <div class="text-sm text-gray-400">
            System Status: {systemInitialized ? 'Initialized' : 'Initializing...'}<br>
            Health: {systemHealth}%
          </div>
        </div>
      </Card>
    </div>

    <!-- Results Display -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- Search Results -->
      <Card class="p-6 bg-slate-800/50 border-slate-700">
        <h3 class="text-lg font-semibold text-white mb-4">Search Results</h3>
        <div class="space-y-3 max-h-96 overflow-y-auto">
          {#each searchResults as result, index}
            <div class="p-3 rounded border border-slate-600 bg-slate-700/50">
              <div class="flex justify-between items-start mb-2">
                <h4 class="font-medium text-white">{result.document.title}</h4>
                <span class="text-xs px-2 py-1 rounded bg-blue-600 text-white">
                  {(result.finalScore * 100).toFixed(1)}%
                </span>
              </div>
              <p class="text-sm text-gray-300 mb-2">
                {result.document.content.substring(0, 150)}...
              </p>
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-400">
                  PageRank: {result.pageRankBoost?.toFixed(3) || '0.000'}
                </span>
                <Button 
                  size="sm" 
                  onclick={() => submitPositiveFeedback(index)}
                  class="text-xs bg-green-600 hover:bg-green-700"
                >
                  👍 Relevant
                </Button>
              </div>
            </div>
          {:else}
            <p class="text-gray-400 text-center py-8">No search results yet. Try performing a search.</p>
          {/each}
        </div>
      </Card>

      <!-- AI Recommendations -->
      <Card class="p-6 bg-slate-800/50 border-slate-700">
        <h3 class="text-lg font-semibold text-white mb-4">Context7 MCP Recommendations</h3>
        <div class="space-y-3 max-h-96 overflow-y-auto">
          {#each recommendations as rec}
            <div class="p-3 rounded border border-slate-600 bg-slate-700/50">
              <div class="flex justify-between items-start mb-2">
                <span class="text-xs px-2 py-1 rounded bg-purple-600 text-white capitalize">
                  {rec.agent}
                </span>
                <span class="text-xs px-2 py-1 rounded text-white" 
                      class:bg-red-600={rec.priority === 'CRITICAL'}
                      class:bg-orange-600={rec.priority === 'HIGH'}
                      class:bg-yellow-600={rec.priority === 'MEDIUM'}
                      class:bg-green-600={rec.priority === 'LOW'}>
                  {rec.priority}
                </span>
              </div>
              <p class="text-sm text-white font-medium mb-1">{rec.recommendation}</p>
              <p class="text-xs text-gray-300 mb-2">{rec.reasoning}</p>
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-400">
                  Confidence: {(rec.confidence * 100).toFixed(1)}%
                </span>
                <span class="text-xs text-gray-400">
                  Impact: {(rec.estimatedImpact * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          {:else}
            <p class="text-gray-400 text-center py-8">No recommendations yet. Click "Get Recommendations" above.</p>
          {/each}
        </div>
      </Card>
    </div>

    <!-- Footer -->
    <div class="mt-12 text-center text-gray-400">
      <p class="text-sm">
        Phase 13 Enhanced Features Demo - Powered by SvelteKit 2, XState, WebGL2, Enhanced RAG, and Context7 MCP
      </p>
      <p class="text-xs mt-2">
        🚀 Features: Vertex Streaming, Stateless APIs, PageRank Feedback, Agent Orchestration, Memory Graph
      </p>
    </div>
  </div>
</div>

<style>
  :global(body) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
</style>