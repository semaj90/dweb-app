<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/ui/enhanced/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/enhanced/Card.svelte';
  import { Badge } from '$lib/components/ui/modular/Badge.svelte';
  import { cn } from '$lib/utils';

  // Svelte 5 runes for reactive state
  let testResults = $state<Record<string, any>>({});
  let isRunning = $state(false);
  let currentTest = $state('');

  const tests = [
    {
      name: 'Database Connection',
      endpoint: '/api/db/health',
      description: 'Test PostgreSQL + pgvector connection'
    },
    {
      name: 'Enhanced RAG Service',
      endpoint: 'http://localhost:8094/health',
      description: 'Test Go microservice health'
    },
    {
      name: 'Upload Service',
      endpoint: 'http://localhost:8093/health',
      description: 'Test file upload service'
    },
    {
      name: 'Ollama Models',
      endpoint: 'http://localhost:11434/api/tags',
      description: 'Check available AI models'
    },
    {
      name: 'Chat API',
      endpoint: '/api/ai/chat',
      method: 'POST',
      body: {
        message: 'Test message',
        model: 'gemma3-legal:latest',
        stream: false
      },
      description: 'Test basic chat functionality'
    },
    {
      name: 'SSE Chat API',
      endpoint: '/api/ai/chat-sse',
      method: 'POST',
      body: {
        message: 'Test SSE streaming',
        model: 'gemma3-legal:latest',
        useRAG: true
      },
      description: 'Test Server-Sent Events streaming'
    }
  ];

  async function runTest(test: any) {
    currentTest = test.name;
    
    try {
      const options: RequestInit = {
        method: test.method || 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      };

      if (test.body) {
        options.body = JSON.stringify(test.body);
      }

      const response = await fetch(test.endpoint, options);
      const data = await response.json();

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
      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    currentTest = '';
    isRunning = false;
  }

  async function testSSEStream() {
    currentTest = 'SSE Streaming Test';
    
    try {
      const response = await fetch('/api/ai/chat-sse', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: 'What are the key elements of contract formation?',
          model: 'gemma3-legal:latest',
          useRAG: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      let streamData = '';
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        for (let i = 0; i < 10; i++) { // Read first 10 chunks
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          streamData += chunk;
        }
        reader.cancel();
      }

      testResults['SSE Stream Test'] = {
        success: true,
        status: response.status,
        streamSample: streamData.substring(0, 500) + '...',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      testResults['SSE Stream Test'] = {
        success: false,
        error: error instanceof Error ? error.message : 'Stream test failed',
        timestamp: new Date().toISOString()
      };
    }

    testResults = { ...testResults };
    currentTest = '';
  }

  onMount(() => {
    // Auto-run basic tests on mount
    runAllTests();
  });
</script>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
  <div class="max-w-6xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
        AI Assistant Integration Test
      </h1>
      <p class="text-gray-600">Test suite for SvelteKit frontend + Go microservices + Database integration</p>
    </div>

    <!-- Control Panel -->
    <Card class="mb-6 p-6">
      {#snippet children()}
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-xl font-semibold">Test Controls</h2>
          <div class="flex gap-3">
            <Button
              variant="outline"
              onclick={runAllTests}
              disabled={isRunning}
            >
              {#snippet children()}
                {isRunning ? 'Running Tests...' : 'Run All Tests'}
              {/snippet}
            </Button>
            <Button
              variant="default"
              onclick={testSSEStream}
              disabled={isRunning}
            >
              {#snippet children()}Test SSE Stream{/snippet}
            </Button>
          </div>
        </div>

        {#if currentTest}
          <div class="flex items-center gap-2 text-blue-600">
            <div class="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            <span>Testing: {currentTest}</span>
          </div>
        {/if}
      {/snippet}
    </Card>

    <!-- Test Results -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {#each tests as test}
        {@const result = testResults[test.name]}
        <Card class="p-6">
          {#snippet children()}
            <div class="flex items-start justify-between mb-4">
              <div>
                <h3 class="text-lg font-semibold">{test.name}</h3>
                <p class="text-sm text-gray-600">{test.description}</p>
                <p class="text-xs text-gray-500 mt-1">{test.endpoint}</p>
              </div>
              <Badge variant={result?.success ? 'success' : result ? 'destructive' : 'secondary'}>
                {#snippet children()}
                  {result?.success ? 'PASS' : result ? 'FAIL' : 'PENDING'}
                {/snippet}
              </Badge>
            </div>

            {#if result}
              <div class="space-y-2 text-sm">
                <div>
                  <span class="font-medium">Status:</span>
                  <span class={cn(
                    "ml-2",
                    result.success ? "text-green-600" : "text-red-600"
                  )}>
                    {result.status || 'N/A'} {result.success ? '✓' : '✗'}
                  </span>
                </div>

                {#if result.error}
                  <div>
                    <span class="font-medium text-red-600">Error:</span>
                    <span class="ml-2 text-red-600">{result.error}</span>
                  </div>
                {/if}

                {#if result.data && typeof result.data === 'object'}
                  <details class="mt-2">
                    <summary class="font-medium cursor-pointer">Response Data</summary>
                    <pre class="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto max-h-40">
{JSON.stringify(result.data, null, 2)}
                    </pre>
                  </details>
                {/if}

                {#if result.streamSample}
                  <details class="mt-2">
                    <summary class="font-medium cursor-pointer">Stream Sample</summary>
                    <pre class="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto max-h-40">
{result.streamSample}
                    </pre>
                  </details>
                {/if}

                <div class="text-xs text-gray-500">
                  Tested: {result.timestamp}
                </div>
              </div>
            {:else}
              <div class="text-gray-400">
                Waiting for test to run...
              </div>
            {/if}
          {/snippet}
        </Card>
      {/each}

      <!-- SSE Stream Test Result -->
      {#if testResults['SSE Stream Test']}
        {@const sseResult = testResults['SSE Stream Test']}
        <Card class="p-6 lg:col-span-2">
          {#snippet children()}
            <div class="flex items-start justify-between mb-4">
              <div>
                <h3 class="text-lg font-semibold">Server-Sent Events Test</h3>
                <p class="text-sm text-gray-600">Real-time streaming response test</p>
              </div>
              <Badge variant={sseResult.success ? 'success' : 'destructive'}>
                {#snippet children()}
                  {sseResult.success ? 'STREAM OK' : 'STREAM FAIL'}
                {/snippet}
              </Badge>
            </div>

            <div class="space-y-2 text-sm">
              <div>
                <span class="font-medium">Status:</span>
                <span class={cn(
                  "ml-2",
                  sseResult.success ? "text-green-600" : "text-red-600"
                )}>
                  {sseResult.status || 'N/A'} {sseResult.success ? '✓' : '✗'}
                </span>
              </div>

              {#if sseResult.streamSample}
                <details class="mt-2">
                  <summary class="font-medium cursor-pointer">Stream Output Sample</summary>
                  <pre class="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto max-h-60">
{sseResult.streamSample}
                  </pre>
                </details>
              {/if}
            </div>
          {/snippet}
        </Card>
      {/if}
    </div>

    <!-- Summary -->
    <Card class="mt-6 p-6">
      {#snippet children()}
        <h2 class="text-xl font-semibold mb-4">Test Summary</h2>
        {@const totalTests = Object.keys(testResults).length}
        {@const passedTests = Object.values(testResults).filter(r => r.success).length}
        
        <div class="grid grid-cols-3 gap-4 text-center">
          <div class="p-4 bg-blue-50 rounded">
            <div class="text-2xl font-bold text-blue-600">{totalTests}</div>
            <div class="text-sm text-blue-600">Total Tests</div>
          </div>
          <div class="p-4 bg-green-50 rounded">
            <div class="text-2xl font-bold text-green-600">{passedTests}</div>
            <div class="text-sm text-green-600">Passed</div>
          </div>
          <div class="p-4 bg-red-50 rounded">
            <div class="text-2xl font-bold text-red-600">{totalTests - passedTests}</div>
            <div class="text-sm text-red-600">Failed</div>
          </div>
        </div>

        {#if totalTests > 0}
          <div class="mt-4">
            <div class="w-full bg-gray-200 rounded-full h-2">
              <div 
                class="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-300"
                style="width: {(passedTests / totalTests) * 100}%"
              ></div>
            </div>
            <p class="text-center mt-2 text-sm text-gray-600">
              Success Rate: {Math.round((passedTests / totalTests) * 100)}%
            </p>
          </div>
        {/if}
      {/snippet}
    </Card>
  </div>
</div>