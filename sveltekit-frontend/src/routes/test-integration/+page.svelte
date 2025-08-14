<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';

  // Test state
  let testResults: any = {};
  let isRunning = false;
  let currentTest = '';
  let testLog: string[] = [];

  // Service status
  let ollamaStatus = 'unknown';
  let svelteKitStatus = 'unknown';
  let modelStatus = 'unknown';

  // Test chat
  let chatMessage = 'What are the essential elements of a valid contract?';
  let chatResponse = '';
  let chatLoading = false;

  onMount(() => {
    runHealthChecks();
  });

  async function runHealthChecks() {
    isRunning = true;
    testLog = [];

    addLog('🔍 Starting health checks...');

    // Test Ollama service
    currentTest = 'Ollama Service';
    try {
      const response = await fetch('http://localhost:11434/api/version');
      if (response.ok) {
        const data = await response.json();
        ollamaStatus = 'connected';
        addLog(`✅ Ollama service running (v${data.version})`);
      } else {
        ollamaStatus = 'error';
        addLog(`❌ Ollama service HTTP ${response.status}`);
      }
    } catch (error) {
      ollamaStatus = 'disconnected';
      addLog(`❌ Ollama service not accessible: ${error}`);
    }        // Test SvelteKit API
        currentTest = 'SvelteKit API';
        try {
          const response = await fetch('/api/ai/test-ollama');
          if (response.ok) {
            const data = await response.json();
            svelteKitStatus = 'connected';
            testResults.api = data;
            addLog(`✅ SvelteKit API responding`);
            addLog(`ℹ️  Model: ${data.ollama?.gemma3Model || 'Unknown'}`);
          } else {
            svelteKitStatus = 'error';
            addLog(`❌ SvelteKit API HTTP ${response.status}`);
          }
        } catch (error) {
          svelteKitStatus = 'error';
          addLog(`❌ SvelteKit API error: ${error}`);
        }

    // Test model availability
    currentTest = 'Model Status';
    if (ollamaStatus === 'connected') {
      try {
        const response = await fetch('http://localhost:11434/api/tags');
        if (response.ok) {
          const data = await response.json();
          const models = data.models?.map((m: any) => m.name) || [];
          const hasGemma3Legal = models.some((m: string) => m.includes('gemma3-legal'));
          const hasGemma = models.some((m: string) => m.includes('gemma'));

          if (hasGemma3Legal) {
            modelStatus = 'ready';
            addLog(`✅ Gemma3 Legal model available`);
          } else if (hasGemma) {
            modelStatus = 'fallback';
            addLog(`⚠️  Gemma model found but not gemma3-legal`);
          } else {
            modelStatus = 'missing';
            addLog(`❌ No Gemma models found`);
          }

          testResults.models = models;
          addLog(`ℹ️  Available models: ${models.length}`);
        }
      } catch (error) {
        modelStatus = 'error';
        addLog(`❌ Model check failed: ${error}`);
      }
    } else {
      modelStatus = 'unavailable';
      addLog(`⚠️  Cannot check models - Ollama not accessible`);
    }

    currentTest = '';
    isRunning = false;
    addLog('🏁 Health checks completed');
  }

  async function testChatAPI() {
    if (!chatMessage.trim()) return;

    chatLoading = true;
    chatResponse = '';
    addLog(`🧪 Testing chat with: "${chatMessage}"`);

    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: chatMessage,
          model: 'gemma3-legal',
          temperature: 0.1,
          maxTokens: 400
        })
      });

      if (response.ok) {
        const data = await response.json();
        chatResponse = data.response;
        addLog(`✅ Chat API successful (${data.metadata?.executionTime || 'N/A'}ms)`);
        addLog(`ℹ️  Provider: ${data.metadata?.provider || 'Unknown'}`);

        if (data.metadata?.fallbackReason) {
          addLog(`⚠️  Fallback reason: ${data.metadata.fallbackReason}`);
        }
      } else {
        const errorData = await response.json();
        addLog(`❌ Chat API failed: ${errorData.error}`);
        chatResponse = `Error: ${errorData.error}`;
      }
    } catch (error) {
      addLog(`❌ Chat API error: ${error}`);
      chatResponse = `Network error: ${error}`;
    }

    chatLoading = false;
  }

  async function testDirectOllama() {
    addLog('🧪 Testing direct Ollama generation...');

    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: 'Briefly explain contract law.',
          stream: false,
          options: {
            temperature: 0.1,
            num_predict: 100
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        addLog(`✅ Direct Ollama successful`);
        addLog(`ℹ️  Response: ${data.response?.substring(0, 100)}...`);
      } else {
        addLog(`❌ Direct Ollama failed: HTTP ${response.status}`);
      }
    } catch (error) {
      addLog(`❌ Direct Ollama error: ${error}`);
    }
  }

  function addLog(message: string) {
    testLog = [...testLog, `${new Date().toLocaleTimeString()}: ${message}`];
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'connected':
      case 'ready':
        return 'text-green-600 bg-green-50';
      case 'fallback':
        return 'text-yellow-600 bg-yellow-50';
      case 'error':
      case 'missing':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'connected':
      case 'ready':
        return '✅';
      case 'fallback':
        return '⚠️';
      case 'error':
      case 'missing':
        return '❌';
      default:
        return '🔄';
    }
  }
</script>

<svelte:head>
  <title>Gemma3 Integration Test - Legal AI Assistant</title>
  <meta name="description" content="Integration testing page for Gemma3 Legal AI model with SvelteKit and Bits UI" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
  <div class="container mx-auto px-4 py-8">

    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-2">
        🤖 Gemma3 Legal AI Integration Test
      </h1>
      <p class="text-lg text-gray-600">
        Comprehensive testing for Ollama, SvelteKit API, and model integration
      </p>
    </div>

    <!-- Status Dashboard -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">

      <!-- Ollama Status -->
      <div class="bg-white rounded-lg shadow-lg p-6 border">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-gray-900">Ollama Service</h3>
          <span class={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(ollamaStatus)}`}>
            {getStatusIcon(ollamaStatus)} {ollamaStatus}
          </span>
        </div>
        <p class="text-sm text-gray-600 mb-2">Local LLM inference engine</p>
        <p class="text-xs text-gray-500">http://localhost:11434</p>
      </div>

      <!-- SvelteKit API Status -->
      <div class="bg-white rounded-lg shadow-lg p-6 border">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-gray-900">SvelteKit API</h3>
          <span class={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(svelteKitStatus)}`}>
            {getStatusIcon(svelteKitStatus)} {svelteKitStatus}
          </span>
        </div>
        <p class="text-sm text-gray-600 mb-2">Frontend API endpoints</p>
        <p class="text-xs text-gray-500">/api/ai/*</p>
      </div>

      <!-- Model Status -->
      <div class="bg-white rounded-lg shadow-lg p-6 border">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-gray-900">Gemma3 Model</h3>
          <span class={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(modelStatus)}`}>
            {getStatusIcon(modelStatus)} {modelStatus}
          </span>
        </div>
        <p class="text-sm text-gray-600 mb-2">Legal AI model</p>
        <p class="text-xs text-gray-500">gemma3-legal</p>
      </div>
    </div>

    <!-- Control Panel -->
    <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
      <h2 class="text-2xl font-bold text-gray-900 mb-4">🧪 Test Controls</h2>

      <div class="flex flex-wrap gap-4 mb-6">
        <button
          onclick={runHealthChecks}
          disabled={isRunning}
          class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
        >
          {isRunning ? '🔄 Running...' : '🔍 Run Health Checks'}
        </button>

        <button
          onclick={testDirectOllama}
          disabled={isRunning || ollamaStatus !== 'connected'}
          class="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition-colors"
        >
          🧪 Test Direct Ollama
        </button>
      </div>

      {#if currentTest}
        <div class="mb-4" transition:fade>
          <div class="flex items-center text-blue-600">
            <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
            Testing: {currentTest}
          </div>
        </div>
      {/if}
    </div>

    <!-- Chat Testing -->
    <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
      <h2 class="text-2xl font-bold text-gray-900 mb-4">💬 Chat API Testing</h2>

      <div class="mb-4">
        <label for="chatMessage" class="block text-sm font-medium text-gray-700 mb-2">
          Test Message
        </label>
        <div class="flex gap-2">
          <input
            id="chatMessage"
            bind:value={chatMessage}
            placeholder="Enter a legal question..."
            class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            onclick={testChatAPI}
            disabled={chatLoading || !chatMessage.trim()}
            class="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 transition-colors"
          >
            {chatLoading ? '🔄' : '💬'} Test Chat
          </button>
        </div>
      </div>

      {#if chatResponse}
        <div class="mt-4 p-4 bg-gray-50 rounded-lg" transitionfly={{ y: 20 }}>
          <h4 class="font-semibold text-gray-900 mb-2">AI Response:</h4>
          <div class="text-gray-700 whitespace-pre-wrap">{chatResponse}</div>
        </div>
      {/if}
    </div>

    <!-- System Information -->
    {#if testResults.api || testResults.models}
      <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
        <h2 class="text-2xl font-bold text-gray-900 mb-4">📊 System Information</h2>

        {#if testResults.api}
          <div class="mb-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-2">API Response</h3>
            <pre class="bg-gray-100 p-4 rounded-lg text-sm overflow-x-auto">{JSON.stringify(testResults.api, null, 2)}</pre>
          </div>
        {/if}

        {#if testResults.models}
          <div class="mb-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-2">Available Models</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {#each testResults.models as model}
                <div class="px-3 py-2 bg-gray-100 rounded text-sm font-mono">
                  {model}
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Test Log -->
    <div class="bg-white rounded-lg shadow-lg p-6">
      <h2 class="text-2xl font-bold text-gray-900 mb-4">📋 Test Log</h2>

      <div class="bg-black text-green-400 p-4 rounded-lg font-mono text-sm max-h-64 overflow-y-auto">
        {#each testLog as logEntry}
          <div class="mb-1">{logEntry}</div>
        {:else}
          <div class="text-gray-500">No test logs yet. Run health checks to start testing.</div>
        {/each}
      </div>
    </div>

    <!-- Instructions -->
    <div class="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold text-blue-900 mb-2">📝 Testing Instructions</h3>
      <ul class="text-blue-800 space-y-1 text-sm">
        <li>• <strong>Health Checks:</strong> Verify all services are running</li>
        <li>• <strong>Chat Test:</strong> Test the complete API chain with a legal question</li>
        <li>• <strong>Direct Ollama:</strong> Test model directly without SvelteKit</li>
        <li>• <strong>Expected Behavior:</strong> Legal responses with proper terminology</li>
        <li>• <strong>Troubleshooting:</strong> Check TROUBLESHOOTING_GUIDE.md for issues</li>
      </ul>
    </div>

    <!-- Quick Links -->
    <div class="mt-6 text-center">
      <div class="flex flex-wrap justify-center gap-4 text-sm">
        <a href="/" class="text-blue-600 hover:underline">← Back to Main App</a>
        <a href="/api/ai/test-ollama" class="text-blue-600 hover:underline" target="_blank">API Endpoint</a>
        <a href="http://localhost:11434/api/tags" class="text-blue-600 hover:underline" target="_blank">Ollama Models</a>
      </div>
    </div>
  </div>
</div>

<style>
  :global(body) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  }
</style>
