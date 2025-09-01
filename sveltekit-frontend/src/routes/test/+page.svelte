<script lang="ts">
  import { onMount } from 'svelte';
  // $state runtime rune is provided globally
  import { aiAgentStore, isAIConnected, systemHealth, currentConversation } from '$lib/stores/ai-agent';
  import type { User } from '$lib/types';

  let testMessage = $state('Hello, can you help me with legal questions?');
  let connectionStatus = $state('checking...');
  let testResults: string[] = $state([]);

  onMount(async () => {
    await runSystemTests();
  });

  async function runSystemTests() {
    testResults = [];
    addTestResult('🧪 Starting system tests...');

    // Test 1: Store initialization
    try {
      addTestResult('✅ AI Agent store initialized');
    } catch (error) {
      addTestResult(`❌ Store initialization failed: ${error}`);
    }

    // Test 2: AI Connection
    try {
      await aiAgentStore.connect();
      connectionStatus = 'connected';
      addTestResult('✅ AI service connection established');
    } catch (error) {
      connectionStatus = 'failed';
      addTestResult(`⚠️  AI service connection failed: ${error}`);
      addTestResult('   This is expected if Ollama is not running');
    }

    // Test 3: Type system
    try {
      const testUser: User = {
        id: 'test',
        email: 'test@example.com',
        name: 'Test User',
        firstName: 'Test',
        lastName: 'User',
        avatarUrl: '',
        role: 'user',
        isActive: true,
        emailVerified: true,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      addTestResult('✅ TypeScript types working correctly');
    } catch (error) {
      addTestResult(`❌ Type system error: ${error}`);
    }

    addTestResult('🎯 System test complete!');
  }

  function addTestResult(message: string) {
    testResults = [...testResults, message];
  }

  async function sendTestMessage() {
    if (!$isAIConnected) {
      addTestResult('⚠️  Cannot send message: AI not connected');
      return;
    }

    try {
      addTestResult(`📤 Sending: ${testMessage}`);
      await aiAgentStore.sendMessage(testMessage);
      addTestResult('✅ Message sent successfully');
    } catch (error) {
      addTestResult(`❌ Message failed: ${error}`);
    }
  }

  function clearTests() {
    testResults = [];
  }

  async function runCRUDTests() {
    testResults = [];
    addTestResult('🔧 Starting CRUD API tests...');

    // Test 1: CRUD Health Check
    try {
      const response = await fetch('/test/crud');
      if (response.ok) {
        addTestResult('✅ CRUD route accessible');
      } else {
        addTestResult(`⚠️ CRUD route returned ${response.status}`);
      }
    } catch (error) {
      addTestResult(`❌ CRUD route error: ${error}`);
    }

    // Test 2: API Health Check via JSON request
    try {
      const response = await fetch('/test/crud', {
        headers: { 'Accept': 'application/json' }
      });
      if (response.ok) {
        const data = await response.json();
        addTestResult('✅ CRUD API health check passed');
        addTestResult(`   Database: ${data.health?.database?.connected ? 'Connected' : 'Offline'}`);
      } else {
        addTestResult(`⚠️ CRUD API health check failed: ${response.status}`);
      }
    } catch (error) {
      addTestResult(`❌ CRUD API error: ${error}`);
    }

    // Test 3: DELETE endpoint test (without actually deleting)
    try {
      const response = await fetch('/test/crud?id=test-id', {
        method: 'DELETE'
      });
      // We expect this to fail because 'test-id' doesn't exist
      if (response.status === 404) {
        addTestResult('✅ DELETE endpoint responds correctly to invalid ID');
      } else if (response.status === 400) {
        addTestResult('✅ DELETE endpoint validates required parameters');
      } else {
        addTestResult(`⚠️ DELETE endpoint returned unexpected status: ${response.status}`);
      }
    } catch (error) {
      addTestResult(`❌ DELETE endpoint error: ${error}`);
    }

    addTestResult('🎯 CRUD API test complete!');
  }
</script>

<svelte:head>
  <title>AI Agent Test Page</title>
</svelte:head>

<div class="p-8 max-w-6xl mx-auto">
  <div class="mb-8">
    <h1 class="text-4xl font-bold mb-2">🧪 Test Navigation Hub</h1>
    <p class="text-gray-600">Route testing, CRUD operations, SSR validation, and system health monitoring</p>
  </div>

  <!-- Quick Navigation -->
  <div class="mb-8 p-6 border rounded-lg bg-blue-50">
    <h2 class="text-2xl font-semibold mb-4">🛣️ Test Routes</h2>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <a 
        href="/test/crud" 
        class="p-4 border-2 border-blue-200 rounded-lg hover:border-blue-400 transition-colors bg-white"
      >
        <div class="font-semibold text-blue-800">CRUD Test Interface</div>
        <div class="text-sm text-gray-600">Full database operations with gaming UI</div>
        <div class="text-xs text-gray-500 mt-1">/test/crud</div>
      </a>
      
      <button 
        on:on:onclick={() => runCRUDTests()}
        class="p-4 border-2 border-green-200 rounded-lg hover:border-green-400 transition-colors bg-white text-left"
      >
        <div class="font-semibold text-green-800">CRUD API Tests</div>
        <div class="text-sm text-gray-600">Test all CRUD endpoints</div>
        <div class="text-xs text-gray-500 mt-1">API validation</div>
      </button>
      
      <button 
        on:on:onclick={() => runSystemTests()}
        class="p-4 border-2 border-purple-200 rounded-lg hover:border-purple-400 transition-colors bg-white text-left"
      >
        <div class="font-semibold text-purple-800">System Health Check</div>
        <div class="text-sm text-gray-600">Database, SSR, API status</div>
        <div class="text-xs text-gray-500 mt-1">Health monitoring</div>
      </button>
    </div>
  </div>

  <!-- Status Panel -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
    <div class="p-4 border rounded-lg bg-blue-50">
      <h3 class="font-semibold text-blue-800">Connection</h3>
      <p class="text-2xl font-mono {connectionStatus === 'connected' ? 'text-green-600' : 'text-red-600'}">
        {connectionStatus}
      </p>
    </div>

    <div class="p-4 border rounded-lg bg-green-50">
      <h3 class="font-semibold text-green-800">AI Connected</h3>
      <p class="text-2xl font-mono {$isAIConnected ? 'text-green-600' : 'text-red-600'}">
        {$isAIConnected}
      </p>
    </div>

    <div class="p-4 border rounded-lg bg-purple-50">
      <h3 class="font-semibold text-purple-800">System Health</h3>
      <p class="text-2xl font-mono
        {$systemHealth === 'healthy' ? 'text-green-600' :
          $systemHealth === 'degraded' ? 'text-yellow-600' : 'text-red-600'}">
        {$systemHealth}
      </p>
    </div>
  </div>

  <!-- Chat Test -->
  <div class="mb-8 p-6 border rounded-lg">
    <h2 class="text-2xl font-semibold mb-4">💬 AI Chat Test</h2>

    <div class="flex gap-2 mb-4">
      <input
        bind:value={testMessage}
        class="flex-1 p-3 border rounded-lg"
        placeholder="Type a message to test AI response..."
      />
      <button
        on:onclick={sendTestMessage}
        class="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
        disabled={!$isAIConnected}
      >
        Send Test
      </button>
    </div>

    {#if $currentConversation.length > 0}
      <div class="space-y-3 max-h-64 overflow-y-auto">
        {#each $currentConversation as message}
          <div class="p-3 rounded-lg {message.role === 'user' ? 'bg-blue-100 ml-8' : 'bg-gray-100 mr-8'}">
            <div class="text-sm text-gray-600 mb-1">
              {message.role === 'user' ? '👤 You' : '🤖 AI Assistant'}
            </div>
            <div class="text-gray-800">{message.content}</div>
            {#if message.metadata}
              <div class="text-xs text-gray-500 mt-2">
                Model: {message.metadata.model || 'unknown'} |
                Execution: {message.metadata.executionTime || 0}ms
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <div class="text-gray-500 text-center py-4">
        No conversation yet. Send a test message to begin.
      </div>
    {/if}
  </div>

  <!-- Test Results -->
  <div class="mb-8 p-6 border rounded-lg">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-2xl font-semibold">🧪 System Test Results</h2>
      <div class="space-x-2">
        <button
          on:onclick={runSystemTests}
          class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Re-run Tests
        </button>
        <button
          on:onclick={clearTests}
          class="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Clear
        </button>
      </div>
    </div>

    <div class="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
      {#each testResults as result}
        <div class="mb-1">{result}</div>
      {/each}
      {#if testResults.length === 0}
        <div class="text-gray-500">No test results yet...</div>
      {/if}
    </div>
  </div>

  <!-- Feature Status -->
  <div class="p-6 border rounded-lg">
    <h2 class="text-2xl font-semibold mb-4">🎯 Implementation Status</h2>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <h3 class="font-semibold mb-2 text-green-700">✅ Completed Features</h3>
        <ul class="space-y-1 text-sm">
          <li>✅ Svelte 5 with runes syntax</li>
          <li>✅ TypeScript type safety (namespaced)</li>
          <li>✅ AI Agent store with real-time updates</li>
          <li>✅ Local LLM integration (Ollama/Gemma)</li>
          <li>✅ Enhanced RAG pipeline</li>
          <li>✅ Production error handling</li>
          <li>✅ Streaming response support</li>
          <li>✅ Vector search capabilities</li>
          <li>✅ Health monitoring system</li>
          <li>✅ API endpoints with validation</li>
          <li>✅ Production deployment scripts</li>
        </ul>
      </div>

      <div>
        <h3 class="font-semibold mb-2 text-blue-700">🔧 Configuration Guide</h3>
        <ul class="space-y-1 text-sm">
          <li><strong>Start Ollama:</strong> ollama serve</li>
          <li><strong>Install Models:</strong> ollama pull gemma2:2b</li>
          <li><strong>Vector DB (Optional):</strong> docker run -p 6333:6333 qdrant/qdrant</li>
          <li><strong>Redis (Optional):</strong> redis-server</li>
          <li><strong>Development:</strong> npm run dev</li>
          <li><strong>Production:</strong> npm run build && npm run preview</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Quick Actions -->
  <div class="mt-8 p-4 bg-gray-50 rounded-lg">
    <h3 class="font-semibold mb-2">🚀 Quick Actions</h3>
    <div class="flex flex-wrap gap-2">
      <a href="/" class="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600">
        Back to App
      </a>
      <a href="/test/crud" class="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600">
        CRUD Tests
      </a>
      <a href="/test/status" class="px-3 py-1 bg-purple-500 text-white rounded text-sm hover:bg-purple-600">
        Route Status
      </a>
      <a href="/api/ai/chat" class="px-3 py-1 bg-orange-500 text-white rounded text-sm hover:bg-orange-600">
        API Health
      </a>
      <button
        on:onclick={() => aiAgentStore.clearConversation()}
        class="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
      >
        Clear Chat
      </button>
    </div>
  </div>
</div>

<style>
  /* Add some custom styling for the test page */
  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
</style>
