<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/ui/enhanced/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/enhanced/Card.svelte';
  import { Badge } from '$lib/components/ui/modular/Badge.svelte';
  import { cn } from '$lib/utils';

  // Svelte 5 runes
  let systemStatus = $state<Record<string, any>>({});
  let authStatus = $state<any>(null);
  let testResults = $state<Record<string, any>>({});
  let isRunning = $state(false);

  const tests = [
    {
      name: 'Authentication Debug',
      endpoint: '/api/auth/debug',
      description: 'Check authentication status and development flags'
    },
    {
      name: 'Development Auth Creation',
      endpoint: '/api/dev-auth?seed=true',
      description: 'Create development session with sample data'
    },
    {
      name: 'Enhanced RAG Health',
      endpoint: 'http://localhost:8094/health',
      description: 'Go microservice health check'
    },
    {
      name: 'Upload Service Health', 
      endpoint: 'http://localhost:8093/health',
      description: 'File upload service health'
    },
    {
      name: 'Ollama API',
      endpoint: 'http://localhost:11434/api/tags',
      description: 'AI model availability'
    },
    {
      name: 'SSE Chat API',
      endpoint: '/api/ai/chat-sse',
      method: 'POST',
      body: { message: 'Test SSE streaming', model: 'gemma3-legal:latest' },
      description: 'Server-Sent Events streaming test'
    }
  ];

  async function runTest(test: any) {
    try {
      const options: RequestInit = {
        method: test.method || 'GET',
        headers: { 'Content-Type': 'application/json' }
      };

      if (test.body) {
        options.body = JSON.stringify(test.body);
      }

      const response = await fetch(test.endpoint, options);
      let data;
      
      try {
        data = await response.json();
      } catch {
        data = await response.text();
      }

      testResults[test.name] = {
        success: response.ok,
        status: response.status,
        data: data,
        endpoint: test.endpoint,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      testResults[test.name] = {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        endpoint: test.endpoint,
        timestamp: new Date().toISOString()
      };
    }

    // Trigger reactivity
    testResults = { ...testResults };
  }

  async function runAllTests() {
    isRunning = true;
    testResults = {};

    for (const test of tests) {
      await runTest(test);
      await new Promise(resolve => setTimeout(resolve, 300));
    }

    isRunning = false;
  }

  async function checkAuthStatus() {
    try {
      const response = await fetch('/api/auth/debug');
      authStatus = await response.json();
    } catch (error) {
      console.error('Auth status check failed:', error);
    }
  }

  async function createDevSession() {
    try {
      const response = await fetch('/api/dev-auth?seed=true');
      const result = await response.json();
      
      if (result.success) {
        await checkAuthStatus();
      }
      
      return result;
    } catch (error) {
      console.error('Dev session creation failed:', error);
    }
  }

  async function clearSession() {
    try {
      const response = await fetch('/api/dev-auth', { method: 'POST' });
      const result = await response.json();
      
      if (result.success) {
        await checkAuthStatus();
      }
      
      return result;
    } catch (error) {
      console.error('Session clear failed:', error);
    }
  }

  onMount(() => {
    checkAuthStatus();
    runAllTests();
  });
</script>

<div class="min-h-screen bg-gradient-to-br from-purple-50 via-white to-pink-50 p-6">
  <div class="max-w-6xl mx-auto">
    <!-- Header -->
    <div class="mb-8 text-center">
      <h1 class="text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
        🎯 Legal AI Platform System Status
      </h1>
      <p class="text-xl text-gray-600">Complete Development Environment Health Check</p>
      <p class="text-lg text-gray-500 mt-2">SvelteKit 2 + Svelte 5 + Go Microservices + PostgreSQL + Ollama + SSE</p>
    </div>

    <!-- Quick Actions -->
    <Card class="mb-8 p-6">
      {#snippet children()}
        <CardHeader class="mb-4">
          {#snippet children()}
            <CardTitle>🚀 Development Actions</CardTitle>
          {/snippet}
        </CardHeader>
        <CardContent>
          {#snippet children()}
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Button onclick={runAllTests} disabled={isRunning} variant="default">
                {#snippet children()}
                  {isRunning ? '⏳ Running...' : '🔄 Run All Tests'}
                {/snippet}
              </Button>
              
              <Button onclick={createDevSession} variant="outline">
                {#snippet children()}🔑 Create Dev Session{/snippet}
              </Button>
              
              <Button onclick={checkAuthStatus} variant="outline">
                {#snippet children()}👤 Check Auth Status{/snippet}
              </Button>
              
              <Button onclick={clearSession} variant="destructive">
                {#snippet children()}🚪 Clear Session{/snippet}
              </Button>
            </div>
          {/snippet}
        </CardContent>
      {/snippet}
    </Card>

    <!-- Authentication Status -->
    <Card class="mb-8 p-6">
      {#snippet children()}
        <CardHeader class="mb-4">
          {#snippet children()}
            <CardTitle>🔐 Authentication Status</CardTitle>
          {/snippet}
        </CardHeader>
        <CardContent>
          {#snippet children()}
            {#if authStatus}
              <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="space-y-3">
                  <div class="flex items-center gap-3">
                    <Badge variant={authStatus.hasUser ? 'success' : 'secondary'}>
                      {#snippet children()}
                        {authStatus.hasUser ? '✅ Authenticated' : '❌ Not Authenticated'}
                      {/snippet}
                    </Badge>
                    {#if authStatus.user}
                      <span class="text-sm text-gray-600">
                        User ID: {authStatus.user.id.substring(0, 8)}...
                      </span>
                    {/if}
                  </div>
                  
                  <p class="text-sm bg-blue-50 p-3 rounded-lg">
                    💡 {authStatus.hint}
                  </p>
                </div>
                
                <div class="bg-gray-50 p-4 rounded-lg">
                  <h4 class="font-semibold mb-2">Request Info</h4>
                  <p class="text-sm text-gray-600">Request ID: {authStatus.requestId}</p>
                  {#if authStatus.serviceRoute}
                    <p class="text-sm text-gray-600">Service Route: {authStatus.serviceRoute}</p>
                  {/if}
                </div>
              </div>
            {:else}
              <p class="text-gray-500">Loading authentication status...</p>
            {/if}
          {/snippet}
        </CardContent>
      {/snippet}
    </Card>

    <!-- System Tests -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      {#each tests as test}
        {@const result = testResults[test.name]}
        <Card class="p-6">
          {#snippet children()}
            <CardHeader class="mb-4">
              {#snippet children()}
                <div class="flex items-start justify-between">
                  <div>
                    <CardTitle class="text-lg">{test.name}</CardTitle>
                    <p class="text-sm text-gray-600 mt-1">{test.description}</p>
                    <p class="text-xs text-gray-500 mt-1 font-mono">
                      {test.method || 'GET'} {test.endpoint}
                    </p>
                  </div>
                  <Badge variant={result?.success ? 'success' : result ? 'destructive' : 'secondary'}>
                    {#snippet children()}
                      {result?.success ? '✅ PASS' : result ? '❌ FAIL' : '⏳ PENDING'}
                    {/snippet}
                  </Badge>
                </div>
              {/snippet}
            </CardHeader>
            
            <CardContent>
              {#snippet children()}
                {#if result}
                  <div class="space-y-3">
                    <div class="flex items-center gap-2">
                      <span class="text-sm font-medium">Status:</span>
                      <Badge variant={result.success ? 'outline' : 'destructive'}>
                        {#snippet children()}
                          {result.status || 'N/A'}
                        {/snippet}
                      </Badge>
                    </div>

                    {#if result.error}
                      <div class="bg-red-50 p-3 rounded-lg">
                        <p class="text-sm text-red-700 font-medium">Error:</p>
                        <p class="text-sm text-red-600">{result.error}</p>
                      </div>
                    {/if}

                    {#if result.data && result.success}
                      <details class="bg-green-50 p-3 rounded-lg">
                        <summary class="text-sm font-medium text-green-700 cursor-pointer">
                          ✅ Response Data
                        </summary>
                        <pre class="text-xs mt-2 overflow-auto max-h-32 bg-white p-2 rounded border">
{typeof result.data === 'object' ? JSON.stringify(result.data, null, 2) : result.data}
                        </pre>
                      </details>
                    {/if}

                    <p class="text-xs text-gray-500">
                      Tested: {new Date(result.timestamp).toLocaleString()}
                    </p>
                  </div>
                {:else}
                  <p class="text-gray-400 text-sm">Waiting for test to run...</p>
                {/if}
              {/snippet}
            </CardContent>
          {/snippet}
        </Card>
      {/each}
    </div>

    <!-- URLs and Access Points -->
    <Card class="p-6">
      {#snippet children()}
        <CardHeader class="mb-4">
          {#snippet children()}
            <CardTitle>🌐 Available Endpoints & Demos</CardTitle>
          {/snippet}
        </CardHeader>
        <CardContent>
          {#snippet children()}
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <!-- Frontend Demos -->
              <div class="bg-blue-50 p-4 rounded-lg">
                <h4 class="font-semibold text-blue-800 mb-2">🎨 Frontend Demos</h4>
                <ul class="text-sm space-y-1">
                  <li><a href="/dev-demo" class="text-blue-600 hover:underline">Development Demo</a></li>
                  <li><a href="/ai-assistant" class="text-blue-600 hover:underline">AI Assistant (SSE)</a></li>
                  <li><a href="/test-ai-assistant" class="text-blue-600 hover:underline">Integration Tests</a></li>
                  <li><a href="/system-status" class="text-blue-600 hover:underline">System Status (This Page)</a></li>
                </ul>
              </div>

              <!-- API Endpoints -->
              <div class="bg-green-50 p-4 rounded-lg">
                <h4 class="font-semibold text-green-800 mb-2">🔗 API Endpoints</h4>
                <ul class="text-sm space-y-1">
                  <li><code class="bg-white px-1 rounded">/api/auth/debug</code></li>
                  <li><code class="bg-white px-1 rounded">/api/dev-auth</code></li>
                  <li><code class="bg-white px-1 rounded">/api/cases</code></li>
                  <li><code class="bg-white px-1 rounded">/api/ai/chat-sse</code></li>
                </ul>
              </div>

              <!-- Go Services -->
              <div class="bg-purple-50 p-4 rounded-lg">
                <h4 class="font-semibold text-purple-800 mb-2">⚡ Go Services</h4>
                <ul class="text-sm space-y-1">
                  <li><code class="bg-white px-1 rounded">:8094 Enhanced RAG</code></li>
                  <li><code class="bg-white px-1 rounded">:8093 Upload Service</code></li>
                  <li><code class="bg-white px-1 rounded">:11434 Ollama API</code></li>
                  <li><code class="bg-white px-1 rounded">:5432 PostgreSQL</code></li>
                </ul>
              </div>
            </div>

            <div class="mt-6 p-4 bg-yellow-50 rounded-lg">
              <h4 class="font-semibold text-yellow-800 mb-2">🎯 Current Status Summary</h4>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span class="font-medium">Development Server:</span>
                  <p class="text-yellow-700">http://localhost:5176</p>
                </div>
                <div>
                  <span class="font-medium">Authentication:</span>
                  <p class="text-yellow-700">DEV_BYPASS_AUTH enabled</p>
                </div>
                <div>
                  <span class="font-medium">Database:</span>
                  <p class="text-yellow-700">35 tables ready</p>
                </div>
                <div>
                  <span class="font-medium">AI Models:</span>
                  <p class="text-yellow-700">gemma3-legal + nomic-embed</p>
                </div>
              </div>
            </div>
          {/snippet}
        </CardContent>
      {/snippet}
    </Card>
  </div>
</div>