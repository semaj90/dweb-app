<!-- 
  Legal AI Platform Demo Page
  Complete full-stack integration showcase
  Features: CRUD operations, Go microservices integration, AI processing
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import SimpleCaseManager from '$lib/components/legal/SimpleCaseManager.svelte';
  import { legalPlatformClient } from '$lib/services/legal-platform-client';

  // Demo state
  let healthStatus = writable({
    services: {
      enhanced_rag: false,
      upload_service: false,
      database: false
    },
    loading: true,
    lastCheck: null as Date | null
  });

  let demoData = writable({
    totalCases: 0,
    aiProcessings: 0,
    vectorSearches: 0,
    uploadedFiles: 0
  });

  // System health check
  async function checkSystemHealth() {
    healthStatus.update(status => ({ ...status, loading: true }));
    
    try {
      const response = await legalPlatformClient.healthCheck();
      
      if (response.success) {
        healthStatus.update(status => ({
          services: response.data?.services || status.services,
          loading: false,
          lastCheck: new Date()
        }));
      } else {
        console.error('Health check failed:', response.error);
        healthStatus.update(status => ({ ...status, loading: false }));
      }
    } catch (error) {
      console.error('Health check error:', error);
      healthStatus.update(status => ({ ...status, loading: false }));
    }
  }

  // Demo AI chat
  let chatMessage = writable('');
  let chatResponse = writable('');
  let chatLoading = writable(false);

  async function testAIChat() {
    const message = $chatMessage.trim();
    if (!message) return;

    chatLoading.set(true);
    chatResponse.set('');
    
    try {
      const response = await legalPlatformClient.chatWithAI(message);
      
      if (response.success) {
        chatResponse.set(JSON.stringify(response.data, null, 2));
        demoData.update(data => ({ ...data, aiProcessings: data.aiProcessings + 1 }));
      } else {
        chatResponse.set(`Error: ${response.error}`);
      }
    } catch (error) {
      chatResponse.set(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      chatLoading.set(false);
    }
  }

  // Demo vector search
  let searchQuery = writable('');
  let searchResults = writable('');
  let searchLoading = writable(false);

  async function testVectorSearch() {
    const query = $searchQuery.trim();
    if (!query) return;

    searchLoading.set(true);
    searchResults.set('');
    
    try {
      const response = await legalPlatformClient.vectorSearch(query);
      
      if (response.success) {
        searchResults.set(JSON.stringify(response.data, null, 2));
        demoData.update(data => ({ ...data, vectorSearches: data.vectorSearches + 1 }));
      } else {
        searchResults.set(`Error: ${response.error}`);
      }
    } catch (error) {
      searchResults.set(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      searchLoading.set(false);
    }
  }

  // Initialize demo on mount
  onMount(async () => {
    await checkSystemHealth();
    
    // Refresh health check every 30 seconds
    const healthInterval = setInterval(checkSystemHealth, 30000);
    
    return () => clearInterval(healthInterval);
  });

  // Format service status
  function formatServiceStatus(isHealthy: boolean): string {
    return isHealthy ? '🟢 Online' : '🔴 Offline';
  }
</script>

<svelte:head>
  <title>Legal AI Platform Demo</title>
  <meta name="description" content="Complete full-stack legal AI platform demonstration with Go microservices integration">
</svelte:head>

<div class="legal-ai-platform-demo bg-gray-50 min-h-screen">
  <!-- Hero Section -->
  <div class="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div class="text-center">
        <h1 class="text-4xl font-bold sm:text-5xl md:text-6xl">
          Legal AI Platform
        </h1>
        <p class="mt-3 max-w-md mx-auto text-xl sm:text-2xl md:mt-5 md:max-w-3xl">
          Complete full-stack integration with Go microservices, PostgreSQL + pgvector, and AI processing
        </p>
        <div class="mt-5 max-w-md mx-auto sm:flex sm:justify-center md:mt-8">
          <div class="rounded-md shadow">
            <button 
              class="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-blue-600 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10"
              click={checkSystemHealth}
            >
              {$healthStatus.loading ? 'Checking...' : 'Check System Health'}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- System Status Dashboard -->
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="bg-white rounded-lg shadow-sm p-6 mb-8">
      <h2 class="text-2xl font-bold text-gray-900 mb-6">System Status</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <!-- Service Status -->
        <div class="bg-gray-50 rounded-lg p-4">
          <h3 class="text-sm font-medium text-gray-500 mb-3">Services</h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Enhanced RAG:</span>
              <span>{formatServiceStatus($healthStatus.services.enhanced_rag)}</span>
            </div>
            <div class="flex justify-between">
              <span>Upload Service:</span>
              <span>{formatServiceStatus($healthStatus.services.upload_service)}</span>
            </div>
            <div class="flex justify-between">
              <span>Database:</span>
              <span>{formatServiceStatus($healthStatus.services.database)}</span>
            </div>
          </div>
        </div>

        <!-- Demo Statistics -->
        <div class="bg-blue-50 rounded-lg p-4">
          <h3 class="text-sm font-medium text-blue-600 mb-3">Total Cases</h3>
          <div class="text-2xl font-bold text-blue-900">{$demoData.totalCases}</div>
        </div>

        <div class="bg-green-50 rounded-lg p-4">
          <h3 class="text-sm font-medium text-green-600 mb-3">AI Processings</h3>
          <div class="text-2xl font-bold text-green-900">{$demoData.aiProcessings}</div>
        </div>

        <div class="bg-purple-50 rounded-lg p-4">
          <h3 class="text-sm font-medium text-purple-600 mb-3">Vector Searches</h3>
          <div class="text-2xl font-bold text-purple-900">{$demoData.vectorSearches}</div>
        </div>
      </div>

      {#if $healthStatus.lastCheck}
        <p class="text-sm text-gray-500">
          Last health check: {$healthStatus.lastCheck.toLocaleString()}
        </p>
      {/if}
    </div>

    <!-- AI Integration Testing -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
      <!-- AI Chat Test -->
      <div class="bg-white rounded-lg shadow-sm p-6">
        <h3 class="text-xl font-semibold text-gray-900 mb-4">AI Chat Test</h3>
        <p class="text-gray-600 mb-4">Test the Enhanced RAG service integration:</p>
        
        <div class="space-y-4">
          <div>
            <label for="chat-input" class="block text-sm font-medium text-gray-700 mb-2">
              Ask the AI a legal question:
            </label>
            <input 
              type="text"
              id="chat-input"
              bind:value={$chatMessage}
              placeholder="e.g., What are the key elements of a contract?"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              keydown={(e) => e.key === 'Enter' && testAIChat()}
            />
          </div>
          
          <button 
            class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50"
            disabled={$chatLoading}
            click={testAIChat}
          >
            {$chatLoading ? 'Processing...' : 'Send to AI'}
          </button>
          
          {#if $chatResponse}
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">AI Response:</label>
              <pre class="bg-gray-100 p-3 rounded-md text-sm overflow-x-auto whitespace-pre-wrap">{$chatResponse}</pre>
            </div>
          {/if}
        </div>
      </div>

      <!-- Vector Search Test -->
      <div class="bg-white rounded-lg shadow-sm p-6">
        <h3 class="text-xl font-semibold text-gray-900 mb-4">Vector Search Test</h3>
        <p class="text-gray-600 mb-4">Test PostgreSQL + pgvector semantic search:</p>
        
        <div class="space-y-4">
          <div>
            <label for="search-input" class="block text-sm font-medium text-gray-700 mb-2">
              Search query:
            </label>
            <input 
              type="text"
              id="search-input"
              bind:value={$searchQuery}
              placeholder="e.g., contract disputes"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              keydown={(e) => e.key === 'Enter' && testVectorSearch()}
            />
          </div>
          
          <button 
            class="bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50"
            disabled={$searchLoading}
            click={testVectorSearch}
          >
            {$searchLoading ? 'Searching...' : 'Vector Search'}
          </button>
          
          {#if $searchResults}
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">Search Results:</label>
              <pre class="bg-gray-100 p-3 rounded-md text-sm overflow-x-auto whitespace-pre-wrap max-h-48">{$searchResults}</pre>
            </div>
          {/if}
        </div>
      </div>
    </div>

    <!-- Architecture Overview -->
    <div class="bg-white rounded-lg shadow-sm p-6 mb-8">
      <h2 class="text-2xl font-bold text-gray-900 mb-6">Platform Architecture</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="text-center">
          <div class="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
            <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
          </div>
          <h3 class="text-lg font-semibold text-gray-900 mb-2">SvelteKit Frontend</h3>
          <p class="text-gray-600 text-sm">Svelte 5 + TypeScript with reactive components and modern UI patterns</p>
        </div>

        <div class="text-center">
          <div class="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
            <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2"></path>
            </svg>
          </div>
          <h3 class="text-lg font-semibold text-gray-900 mb-2">Go Microservices</h3>
          <p class="text-gray-600 text-sm">Enhanced RAG (8094), Upload Service (8093), and multi-protocol API support</p>
        </div>

        <div class="text-center">
          <div class="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
            <svg class="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"></path>
            </svg>
          </div>
          <h3 class="text-lg font-semibold text-gray-900 mb-2">Database Layer</h3>
          <p class="text-gray-600 text-sm">PostgreSQL + pgvector with Drizzle ORM for type-safe operations</p>
        </div>
      </div>

      <div class="mt-8 p-4 bg-gray-50 rounded-lg">
        <h4 class="text-lg font-semibold text-gray-900 mb-2">Key Features Implemented:</h4>
        <ul class="list-disc list-inside text-gray-600 space-y-1">
          <li>Complete CRUD operations for cases, evidence, and criminal records</li>
          <li>Vector similarity search using PostgreSQL + pgvector</li>
          <li>AI processing through Enhanced RAG Go microservice</li>
          <li>Multi-protocol API support (REST/HTTP with future gRPC/QUIC)</li>
          <li>Real-time health monitoring and error handling</li>
          <li>Type-safe client-server communication</li>
          <li>Production-ready UI components with Tailwind CSS</li>
          <li>Scalable microservices architecture</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Case Management Demo -->
  <div class="bg-white">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <h2 class="text-3xl font-bold text-gray-900 mb-4">Case Management System</h2>
        <p class="text-xl text-gray-600">Complete CRUD operations with PostgreSQL + Drizzle ORM integration</p>
      </div>
      
      <SimpleCaseManager />
    </div>
  </div>

  <!-- Technology Stack -->
  <div class="bg-gray-900 text-white">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h2 class="text-3xl font-bold text-center mb-8">Technology Stack</h2>
      
      <div class="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
        <div>
          <h3 class="text-lg font-semibold mb-2">Frontend</h3>
          <ul class="text-gray-300 space-y-1">
            <li>SvelteKit 2</li>
            <li>Svelte 5</li>
            <li>TypeScript</li>
            <li>Tailwind CSS</li>
          </ul>
        </div>
        
        <div>
          <h3 class="text-lg font-semibold mb-2">Backend</h3>
          <ul class="text-gray-300 space-y-1">
            <li>Go Microservices</li>
            <li>gRPC/QUIC</li>
            <li>WebSocket</li>
            <li>REST APIs</li>
          </ul>
        </div>
        
        <div>
          <h3 class="text-lg font-semibold mb-2">Database</h3>
          <ul class="text-gray-300 space-y-1">
            <li>PostgreSQL 17</li>
            <li>pgvector</li>
            <li>Drizzle ORM</li>
            <li>Redis Cache</li>
          </ul>
        </div>
        
        <div>
          <h3 class="text-lg font-semibold mb-2">AI/ML</h3>
          <ul class="text-gray-300 space-y-1">
            <li>Ollama</li>
            <li>GPU Acceleration</li>
            <li>Vector Search</li>
            <li>RAG Pipeline</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .legal-ai-platform-demo {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }
</style>