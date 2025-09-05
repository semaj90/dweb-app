<!-- Phase 3: Live Agent Integration Demo -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { 
    liveAgentOrchestrator, 
    createAgentRequest,
    quickAnalyze,
    quickSummarize,
    quickEmbed,
    quickSearch,
    type AgentRequest,
    type OrchestrationResult 
  } from '$lib/services/live-agent-orchestrator.js';

  // Reactive state
  let connectionStats = $state(liveAgentOrchestrator.getConnectionStats());
  let connectionStatus = $state('');
  let agentHealth = $state({});
  let activeRequests = $state([]);

  // Demo form state
  let testInput = $state('Legal contract analysis for compliance review');
  let selectedAgents = $state(['go-llama', 'ollama-direct']);
  let requestType = $state('analyze');
  let isLoading = $state(false);
  let results: OrchestrationResult[] = $state([]);

  // Available options
  const requestTypes = [
    { value: 'analyze', label: '🔍 Analyze' },
    { value: 'summarize', label: '📝 Summarize' },
    { value: 'embed', label: '🧠 Generate Embeddings' },
    { value: 'search', label: '🔎 Search' }
  ];

  const availableAgents = [
    { value: 'go-llama', label: 'Go + Llama (Recommended)', color: 'bg-blue-500' },
    { value: 'ollama-direct', label: 'Ollama Direct', color: 'bg-green-500' },
    { value: 'context7', label: 'Context7 MCP', color: 'bg-purple-500' },
    { value: 'rag', label: 'Enhanced RAG', color: 'bg-orange-500' }
  ];

  // Subscribe to orchestrator stores
  onMount(async () => {
    console.log('🚀 Initializing Live Agent Demo');
    
    // Subscribe to reactive stores
    connectionStats.subscribe((stats) => {
      console.log('📊 Connection stats updated:', stats);
    });

    liveAgentOrchestrator.connectionStatus.subscribe((status) => {
      connectionStatus = status;
    });

    liveAgentOrchestrator.agentHealth.subscribe((health) => {
      agentHealth = health;
    });

    liveAgentOrchestrator.activeRequests.subscribe((requests) => {
      activeRequests = requests;
    });

    // Test connection
    const isConnected = await liveAgentOrchestrator.testConnection();
    console.log('🔗 Connection test result:', isConnected);
  });

  onDestroy(() => {
    liveAgentOrchestrator.disconnect();
  });

  // Event handlers
  async function testAgentOrchestration() {
    if (!testInput.trim()) return;
    
    isLoading = true;
    console.log(`🎼 Testing ${requestType} with agents:`, selectedAgents);

    try {
      let result: OrchestrationResult
      
      switch (requestType) {
        case 'analyze':
          result = await quickAnalyze(testInput, selectedAgents);
          break;
        case 'summarize':
          result = await quickSummarize(testInput, selectedAgents);
          break;
        case 'embed':
          result = await quickEmbed(testInput, selectedAgents);
          break;
        case 'search':
          result = await quickSearch(testInput, selectedAgents);
          break;
        default:
          const request = createAgentRequest(requestType as any, { text: testInput }, { agents: selectedAgents });
          result = await liveAgentOrchestrator.orchestrateAgents(request);
      }

      results = [result, ...results.slice(0, 4)]; // Keep last 5 results
      
    } catch (error) {
      console.error('❌ Agent orchestration error:', error);
      alert(`Error: ${error.message}`);
    } finally {
      isLoading = false;
    }
  }

  function toggleAgent(agentValue: string) {
    if (selectedAgents.includes(agentValue)) {
      selectedAgents = selectedAgents.filter(a => a !== agentValue);
    } else {
      selectedAgents = [...selectedAgents, agentValue];
    }
  }

  function getAgentStatusColor(agentName: string): string {
    const status = agentHealth[agentName] || agentHealth[agentName.replace('-', '-')] || 'down';
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'degraded': return 'text-yellow-400';
      default: return 'text-red-400';
    }
  }

  function getAgentStatusIcon(agentName: string): string {
    const status = agentHealth[agentName] || agentHealth[agentName.replace('-', '-')] || 'down';
    switch (status) {
      case 'healthy': return '✅';
      case 'degraded': return '⚠️';
      default: return '❌';
    }
  }

  function formatProcessingTime(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }
</script>

<svelte:head>
  <title>Live Agent Integration Demo - Phase 3</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-6xl mx-auto">
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
        🤖 Live Agent Integration Demo
      </h1>
      <p class="text-gray-400 mt-2">Phase 3: Real-time Multi-Agent Orchestration</p>
    </div>

    <!-- Connection Status -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      <div class="bg-gray-800 p-4 rounded-lg">
        <h3 class="text-sm font-medium text-gray-400 mb-2">Connection Status</h3>
        <div class="flex items-center space-x-2">
          <div class={`w-3 h-3 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-400' :
            connectionStatus === 'connecting' ? 'bg-yellow-400' :
            connectionStatus === 'error' ? 'bg-red-400' : 'bg-gray-400'
          }`}></div>
          <span class="capitalize">{connectionStatus}</span>
        </div>
      </div>

      <div class="bg-gray-800 p-4 rounded-lg">
        <h3 class="text-sm font-medium text-gray-400 mb-2">Agent Health</h3>
        <div class="text-lg font-semibold">
          {Object.entries(agentHealth).filter(([_, status]) => status === 'healthy').length} / 
          {Object.keys(agentHealth).length} Healthy
        </div>
      </div>

      <div class="bg-gray-800 p-4 rounded-lg">
        <h3 class="text-sm font-medium text-gray-400 mb-2">Active Requests</h3>
        <div class="text-lg font-semibold">{activeRequests.length}</div>
      </div>
    </div>

    <!-- Agent Health Dashboard -->
    <div class="bg-gray-800 p-6 rounded-lg mb-8">
      <h2 class="text-xl font-semibold mb-4">Agent Status Dashboard</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        {#each availableAgents as agent}
          <div class="bg-gray-700 p-3 rounded-lg">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-medium">{agent.label}</span>
              <span class={getAgentStatusColor(agent.value)}>
                {getAgentStatusIcon(agent.value)}
              </span>
            </div>
            <div class={`text-xs ${getAgentStatusColor(agent.value)}`}>
              {agentHealth[agent.value] || agentHealth[agent.value.replace('-', '')] || 'down'}
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- Demo Interface -->
    <div class="bg-gray-800 p-6 rounded-lg mb-8">
      <h2 class="text-xl font-semibold mb-4">Test Agent Orchestration</h2>
      
      <!-- Input Section -->
      <div class="space-y-4 mb-6">
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Test Input</label>
          <textarea
            bind:value={testInput}
            class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows="3"
            placeholder="Enter text for agent analysis..."
          ></textarea>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Request Type</label>
            <select
              bind:value={requestType}
              class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              {#each requestTypes as type}
                <option value={type.value}>{type.label}</option>
              {/each}
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Select Agents</label>
            <div class="flex flex-wrap gap-2">
              {#each availableAgents as agent}
                <button
                  onclick={() => toggleAgent(agent.value)}
                  class={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedAgents.includes(agent.value)
                      ? `${agent.color} text-white`
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {agent.label}
                  {#if selectedAgents.includes(agent.value)}
                    <span class="ml-1">✓</span>
                  {/if}
                </button>
              {/each}
            </div>
          </div>
        </div>
      </div>

      <!-- Action Button -->
      <button
        onclick={testAgentOrchestration}
        disabled={isLoading || !testInput.trim() || selectedAgents.length === 0}
        class="w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 disabled:from-gray-600 disabled:to-gray-600 text-white py-3 px-6 rounded-lg font-medium transition-colors"
      >
        {#if isLoading}
          <div class="flex items-center justify-center space-x-2">
            <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            <span>Orchestrating Agents...</span>
          </div>
        {:else}
          🚀 Test Agent Orchestration
        {/if}
      </button>
    </div>

    <!-- Results Section -->
    {#if results.length > 0}
      <div class="space-y-6">
        <h2 class="text-xl font-semibold">Orchestration Results</h2>
        
        {#each results as result, index}
          <div class="bg-gray-800 p-6 rounded-lg">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-medium">Result #{results.length - index}</h3>
              <div class="text-sm text-gray-400">
                ID: {result.requestId.slice(-8)}
              </div>
            </div>

            <!-- Summary Stats -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div class="bg-gray-700 p-3 rounded">
                <div class="text-sm text-gray-400">Total Time</div>
                <div class="font-semibold">{formatProcessingTime(result.totalTime)}</div>
              </div>
              <div class="bg-gray-700 p-3 rounded">
                <div class="text-sm text-gray-400">Success Rate</div>
                <div class="font-semibold">{(result.successRate * 100).toFixed(0)}%</div>
              </div>
              <div class="bg-gray-700 p-3 rounded">
                <div class="text-sm text-gray-400">Best Agent</div>
                <div class="font-semibold">{result.bestAgent || 'N/A'}</div>
              </div>
              <div class="bg-gray-700 p-3 rounded">
                <div class="text-sm text-gray-400">Agents Used</div>
                <div class="font-semibold">{result.responses.length}</div>
              </div>
            </div>

            <!-- Agent Responses -->
            <div class="space-y-3">
              <h4 class="font-medium text-gray-300">Agent Responses</h4>
              {#each result.responses as response}
                <div class={`p-3 rounded border-l-4 ${
                  response.status === 'completed' ? 'border-green-400 bg-green-900/20' :
                  response.status === 'error' ? 'border-red-400 bg-red-900/20' :
                  'border-yellow-400 bg-yellow-900/20'
                }`}>
                  <div class="flex items-center justify-between mb-2">
                    <span class="font-medium">{response.agent}</span>
                    <div class="flex items-center space-x-2">
                      {#if response.confidence}
                        <span class="text-sm text-gray-400">
                          Confidence: {(response.confidence * 100).toFixed(0)}%
                        </span>
                      {/if}
                      <span class="text-sm text-gray-400">
                        {formatProcessingTime(response.processingTime)}
                      </span>
                      <span class={`text-sm ${
                        response.status === 'completed' ? 'text-green-400' :
                        response.status === 'error' ? 'text-red-400' :
                        'text-yellow-400'
                      }`}>
                        {response.status}
                      </span>
                    </div>
                  </div>
                  
                  {#if response.error}
                    <div class="text-red-300 text-sm">Error: {response.error}</div>
                  {:else if response.result}
                    <div class="bg-gray-900 p-2 rounded text-sm font-mono overflow-x-auto">
                      {JSON.stringify(response.result, null, 2)}
                    </div>
                  {/if}
                </div>
              {/each}
            </div>

            <!-- Synthesized Result -->
            {#if result.synthesized}
              <div class="mt-4">
                <h4 class="font-medium text-gray-300 mb-2">Synthesized Result</h4>
                <div class="bg-gray-900 p-3 rounded">
                  <pre class="text-sm overflow-x-auto">{JSON.stringify(result.synthesized, null, 2)}</pre>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  /* Custom scrollbar for code blocks */
  pre::-webkit-scrollbar {
    height: 8px;
  }
  
  pre::-webkit-scrollbar-track {
    background: #374151;
  }
  
  pre::-webkit-scrollbar-thumb {
    background: #6B7280;
    border-radius: 4px;
  }
  
  pre::-webkit-scrollbar-thumb:hover {
    background: #9CA3AF;
  }
</style>