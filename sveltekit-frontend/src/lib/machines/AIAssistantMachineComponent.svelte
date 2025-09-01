<script lang="ts">
  import { useMachine } from '@xstate/svelte';
  import { aiAssistantMachine, type AIAssistantContext } from './aiAssistantMachine.js';
  
  interface Props {
    initialContext?: Partial<AIAssistantContext>;
    enableStreamingMode?: boolean;
    preferredProtocol?: 'http' | 'grpc' | 'quic' | 'websocket';
  }
  
  let {
    initialContext = {},
    enableStreamingMode = false,
    preferredProtocol = 'http'
  }: Props = $props();

  // Create machine instance with initial context
  const machineWithContext = aiAssistantMachine.provide({
    context: {
      // Default context merged with props
      currentQuery: '',
      response: '',
      conversationHistory: [],
      sessionId: `session-${Date.now()}`,
      isProcessing: false,
      model: 'gemma3-legal',
      temperature: 0.7,
      maxTokens: 2048,
      availableModels: [
        { name: 'gemma3-legal', displayName: 'Gemma 3 Legal', capabilities: ['text', 'legal'] },
        { name: 'nomic-embed-text', displayName: 'Nomic Embeddings', capabilities: ['embeddings'] }
      ],
      modelLoadBalancing: false,
      databaseConnected: true,
      vectorSearchEnabled: true,
      databasePerformance: {
        queryLatency: 45,
        connectionPool: 8,
        cacheHitRatio: 0.85
      },
      vectorIndexStatus: {
        totalVectors: 15432,
        indexHealth: 'excellent',
        lastUpdated: new Date().toISOString()
      },
      context7Available: true,
      context7Cache: new Map(),
      currentDocuments: [],
      currentImages: [],
      processingQueue: [],
      gpuProcessingEnabled: true,
      serviceHealth: {
        overallHealth: 'excellent',
        services: {
          'enhanced-rag': { status: 'healthy', latency: 25 },
          'upload-service': { status: 'healthy', latency: 15 },
          'vector-service': { status: 'healthy', latency: 35 }
        }
      },
      preferredProtocol,
      activeProtocol: preferredProtocol,
      serviceLoadBalancer: {
        strategy: 'round_robin',
        currentIndex: 0
      },
      circuitBreakers: new Map(),
      natsConnected: true,
      activeStreaming: enableStreamingMode,
      streamBuffer: '',
      collaborationUsers: [],
      ...initialContext
    }
  });

  const { state, send } = useMachine(machineWithContext);
  
  let queryInput = $state('');

  // Helper functions for interaction
  function submitQuery() {
    if (queryInput.trim()) {
      send({ type: 'QUERY', query: queryInput.trim() });
      queryInput = '';
    }
  }

  function clearConversation() {
    send({ type: 'CLEAR_HISTORY' });
  }

  function toggleStreaming() {
    send({ type: 'TOGGLE_STREAMING' });
  }

  // Get status indicators
  let isIdle = $derived(state.value === 'idle');
  let isProcessing = $derived(state.value === 'processing');
  let isStreaming = $derived(state.context.activeStreaming);
  let currentState = $derived(state.value as string);
  let context = $derived(state.context);
</script>

<div class="ai-assistant-machine-demo max-w-4xl mx-auto p-6 space-y-6">
  <!-- Machine Status Header -->
  <div class="bg-gray-900 text-white p-4 rounded-lg">
    <div class="flex items-center justify-between mb-4">
      <h1 class="text-2xl font-bold">AI Assistant Machine - XState 5</h1>
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2">
          <div class="w-3 h-3 rounded-full {currentState === 'idle' ? 'bg-green-500' : currentState === 'processing' ? 'bg-yellow-500' : 'bg-red-500'}"></div>
          <span class="text-sm font-mono uppercase">{currentState}</span>
        </div>
        <div class="text-sm">
          Protocol: <span class="font-mono text-blue-300">{context.activeProtocol}</span>
        </div>
      </div>
    </div>
    
    <!-- Service Health Indicators -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
      <div class="bg-gray-800 p-3 rounded">
        <div class="font-semibold mb-2">Database</div>
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full {context.databaseConnected ? 'bg-green-500' : 'bg-red-500'}"></div>
          <span>{context.databaseConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <div class="text-gray-400 mt-1">
          Latency: {context.databasePerformance?.queryLatency || 0}ms
        </div>
      </div>
      
      <div class="bg-gray-800 p-3 rounded">
        <div class="font-semibold mb-2">Vector Search</div>
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full {context.vectorSearchEnabled ? 'bg-green-500' : 'bg-red-500'}"></div>
          <span>{context.vectorSearchEnabled ? 'Enabled' : 'Disabled'}</span>
        </div>
        <div class="text-gray-400 mt-1">
          Vectors: {context.vectorIndexStatus?.totalVectors?.toLocaleString() || 0}
        </div>
      </div>
      
      <div class="bg-gray-800 p-3 rounded">
        <div class="font-semibold mb-2">NATS Messaging</div>
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full {context.natsConnected ? 'bg-green-500' : 'bg-red-500'}"></div>
          <span>{context.natsConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <div class="text-gray-400 mt-1">
          Streaming: {isStreaming ? 'Active' : 'Inactive'}
        </div>
      </div>
    </div>
  </div>

  <!-- Query Interface -->
  <div class="bg-white border rounded-lg p-6">
    <h2 class="text-xl font-semibold mb-4">AI Query Interface</h2>
    
    <div class="space-y-4">
      <div class="flex gap-2">
        <input
          type="text"
          bind:value={queryInput}
          placeholder="Enter your legal AI query..."
          class="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          disabled={isProcessing}
          keydown={(e) => e.key === 'Enter' && submitQuery()}
        />
        <button
          on:onclick={submitQuery}
          disabled={isProcessing || !queryInput.trim()}
          class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isProcessing ? 'Processing...' : 'Submit'}
        </button>
      </div>
      
      <div class="flex gap-2">
        <button
          on:onclick={toggleStreaming}
          class="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
        >
          {isStreaming ? 'Disable' : 'Enable'} Streaming
        </button>
        <button
          on:onclick={clearConversation}
          class="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
        >
          Clear History
        </button>
      </div>
    </div>
  </div>

  <!-- Current Processing -->
  {#if context.processingQueue && context.processingQueue.length > 0}
    <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4">
      <h3 class="font-semibold text-yellow-800 mb-2">Processing Queue</h3>
      {#each context.processingQueue as job}
        <div class="bg-white p-3 rounded mb-2">
          <div class="flex items-center justify-between mb-2">
            <span class="font-medium">{job.type?.replace(/_/g, ' ').toUpperCase() || 'Unknown Job'}</span>
            <span class="text-sm text-gray-500">{Math.round((job.progress || 0) * 100)}%</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div 
              class="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style="width: {(job.progress || 0) * 100}%"
            ></div>
          </div>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Conversation History -->
  {#if context.conversationHistory && context.conversationHistory.length > 0}
    <div class="bg-white border rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">Conversation History</h2>
      <div class="space-y-4 max-h-96 overflow-y-auto">
        {#each context.conversationHistory as entry}
          <div class="flex gap-3 p-3 rounded-lg {entry.type === 'user' ? 'bg-blue-50' : entry.type === 'ai' ? 'bg-green-50' : 'bg-red-50'}">
            <div class="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center {entry.type === 'user' ? 'bg-blue-600 text-white' : entry.type === 'ai' ? 'bg-green-600 text-white' : 'bg-red-600 text-white'}">
              {entry.type === 'user' ? 'U' : entry.type === 'ai' ? 'AI' : '!'}
            </div>
            <div class="flex-1">
              <div class="text-sm text-gray-500 mb-1">
                {new Date(entry.timestamp).toLocaleTimeString()}
                {entry.userId && ` • User: ${entry.userId}`}
              </div>
              <div class="text-gray-900">{entry.content}</div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Current Response -->
  {#if context.response}
    <div class="bg-green-50 border rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">AI Response</h2>
      <div class="prose max-w-none">
        <div class="whitespace-pre-wrap">{context.response}</div>
        {#if isStreaming && context.streamBuffer}
          <div class="mt-4 p-3 bg-white rounded border-l-4 border-green-400">
            <div class="text-sm text-gray-600 mb-2">Streaming...</div>
            <div class="whitespace-pre-wrap">{context.streamBuffer}</div>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Document Processing -->
  {#if context.currentDocuments && context.currentDocuments.length > 0}
    <div class="bg-white border rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">Document Analysis</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        {#each context.currentDocuments as doc}
          <div class="border rounded-lg p-4">
            <div class="font-medium mb-2">{doc.filename}</div>
            <div class="text-sm text-gray-600 mb-2">
              {doc.contentType} • {(doc.size / 1024).toFixed(1)} KB
            </div>
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full {doc.status === 'analyzed' ? 'bg-green-500' : doc.status === 'analyzing' ? 'bg-yellow-500' : 'bg-gray-400'}"></div>
              <span class="text-sm capitalize">{doc.status}</span>
            </div>
            {#if doc.aiAnalysis}
              <div class="mt-3 text-sm">
                <div class="font-medium">Risk Level: <span class="px-2 py-1 rounded text-xs {doc.aiAnalysis.riskLevel === 'low' ? 'bg-green-100 text-green-800' : doc.aiAnalysis.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'}">{doc.aiAnalysis.riskLevel}</span></div>
                <div class="mt-1">Confidence: {Math.round(doc.aiAnalysis.confidence * 100)}%</div>
                {#if doc.aiAnalysis.keyTerms}
                  <div class="mt-2">
                    <div class="font-medium mb-1">Key Terms:</div>
                    <div class="flex flex-wrap gap-1">
                      {#each doc.aiAnalysis.keyTerms as term}
                        <span class="px-2 py-1 bg-gray-100 rounded text-xs">{term}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Collaboration Users -->
  {#if context.collaborationUsers && context.collaborationUsers.length > 0}
    <div class="bg-white border rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">Collaboration Users</h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        {#each context.collaborationUsers as user}
          <div class="flex items-center gap-3 p-3 border rounded-lg">
            <div class="w-8 h-8 rounded-full bg-gray-600 text-white flex items-center justify-center text-sm font-bold">
              {user.name.split(' ').map(n => n[0]).join('')}
            </div>
            <div>
              <div class="font-medium">{user.name}</div>
              <div class="text-sm text-gray-600">{user.role}</div>
              <div class="flex items-center gap-1 mt-1">
                <div class="w-2 h-2 rounded-full {user.status === 'active' ? 'bg-green-500' : 'bg-gray-400'}"></div>
                <span class="text-xs">{user.status}</span>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Context7 Analysis -->
  {#if context.context7Analysis}
    <div class="bg-blue-50 border rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">Context7 Analysis</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 class="font-medium mb-2">Framework Details</h3>
          <div class="space-y-2 text-sm">
            <div><strong>Framework:</strong> {context.context7Analysis.framework}</div>
            <div><strong>Version:</strong> {context.context7Analysis.version}</div>
            <div><strong>Libraries:</strong> {context.context7Analysis.libraries?.join(', ')}</div>
            <div><strong>Patterns:</strong> {context.context7Analysis.patterns?.join(', ')}</div>
          </div>
        </div>
        <div>
          <h3 class="font-medium mb-2">Recommendations</h3>
          <ul class="list-disc list-inside text-sm space-y-1">
            {#each context.context7Analysis.recommendations || [] as recommendation}
              <li>{recommendation}</li>
            {/each}
          </ul>
        </div>
      </div>
    </div>
  {/if}

  <!-- Debug Information -->
  <details class="bg-gray-50 border rounded-lg p-4">
    <summary class="cursor-pointer font-medium">Debug Information</summary>
    <div class="mt-4 p-4 bg-white rounded border">
      <h4 class="font-medium mb-2">Machine State</h4>
      <pre class="text-xs overflow-x-auto">{JSON.stringify(state, null, 2)}</pre>
    </div>
  </details>
</div>

<style>
  .ai-assistant-machine-demo {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  .prose {
    max-width: none;
  }
  
  .prose pre {
    background: #f5f5f5;
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
  }
</style>