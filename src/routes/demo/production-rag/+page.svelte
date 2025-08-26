<!--
  Production RAG System Demo
  
  Comprehensive demonstration of the production-grade RAG orchestration system
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import ProductionRAGDashboard from '$lib/components/rag/ProductionRAGDashboard.svelte';
  
  let systemReady = false;
  let isInitializing = true;
  let initError: string | null = null;

  onMount(async () => {
    await initializeSystem();
  });

  /**
   * Initialize the production RAG system
   */
  async function initializeSystem(): Promise<void> {
    try {
      console.log('[Production RAG Demo] Initializing system...');
      
      // Check if system is accessible
      const response = await fetch('/api/rag/orchestrate', { method: 'PATCH' });
      
      if (response.ok) {
        systemReady = true;
        console.log('[Production RAG Demo] ✅ System ready');
      } else {
        throw new Error(`System not ready: ${response.statusText}`);
      }
      
    } catch (error) {
      console.error('[Production RAG Demo] ❌ System initialization failed:', error);
      initError = error instanceof Error ? error.message : 'Unknown error';
    } finally {
      isInitializing = false;
    }
  }

  /**
   * Example document processing
   */
  async function runExample(): Promise<void> {
    try {
      const response = await fetch('/api/rag/orchestrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          uploadId: 'example-upload-1',
          caseId: 'demo-case-001',
          filename: 'legal-document-example.pdf',
          storageUrl: 'http://localhost:9000/deeds-storage/example.pdf'
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Example job started:', result);
      }
      
    } catch (error) {
      console.error('Example failed:', error);
    }
  }

  /**
   * Example RAG query
   */
  async function runQueryExample(): Promise<void> {
    try {
      const response = await fetch('/api/rag/orchestrate?' + new URLSearchParams({
        query: 'What are the key legal terms in this case?',
        caseId: 'demo-case-001',
        limit: '5'
      }));

      if (response.ok) {
        const result = await response.json();
        console.log('Query result:', result);
      }
      
    } catch (error) {
      console.error('Query example failed:', error);
    }
  }
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900">
  <!-- Header -->
  <header class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
            🚀 Production RAG System Demo
          </h1>
          <p class="text-gray-600 dark:text-gray-400 mt-1">
            Enterprise-grade document processing and retrieval-augmented generation
          </p>
        </div>
        
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-2">
            <div class="w-2 h-2 {systemReady ? 'bg-green-500' : 'bg-red-500'} rounded-full"></div>
            <span class="text-sm text-gray-600 dark:text-gray-400">
              {systemReady ? 'System Ready' : 'System Offline'}
            </span>
          </div>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    
    {#if isInitializing}
      <!-- Loading State -->
      <div class="flex items-center justify-center py-20">
        <div class="text-center">
          <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 class="text-xl font-medium text-gray-900 dark:text-white mb-2">
            Initializing Production RAG System
          </h2>
          <p class="text-gray-600 dark:text-gray-400">
            Starting microservices and checking health...
          </p>
        </div>
      </div>
      
    {:else if initError}
      <!-- Error State -->
      <div class="max-w-2xl mx-auto">
        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <span class="text-red-400 text-2xl">❌</span>
            </div>
            <div class="ml-4">
              <h3 class="text-lg font-medium text-red-800 dark:text-red-200">
                System Initialization Failed
              </h3>
              <p class="text-red-700 dark:text-red-300 mt-2">
                {initError}
              </p>
              <div class="mt-4">
                <button
                  on:click={initializeSystem}
                  class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Retry Initialization
                </button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Troubleshooting Guide -->
        <div class="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Troubleshooting Guide
          </h3>
          
          <div class="space-y-4 text-sm text-gray-600 dark:text-gray-400">
            <div>
              <h4 class="font-medium text-gray-900 dark:text-white">1. Start Required Services</h4>
              <p>Run the startup script: <code class="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">START-LEGAL-AI.bat</code></p>
            </div>
            
            <div>
              <h4 class="font-medium text-gray-900 dark:text-white">2. Check Service Status</h4>
              <ul class="list-disc list-inside mt-2 space-y-1">
                <li>PostgreSQL: <code>net start postgresql-x64-17</code></li>
                <li>Redis: <code>redis-cli ping</code></li>
                <li>Ollama: <code>curl http://localhost:11434/api/tags</code></li>
                <li>Enhanced RAG: <code>curl http://localhost:8094/health</code></li>
              </ul>
            </div>
            
            <div>
              <h4 class="font-medium text-gray-900 dark:text-white">3. Build Go Services</h4>
              <p>Navigate to <code>go-microservice</code> and run: <code>go run cmd/enhanced-rag/main.go</code></p>
            </div>
          </div>
        </div>
      </div>
      
    {:else if systemReady}
      <!-- System Ready - Show Dashboard -->
      <div class="space-y-8">
        
        <!-- Feature Overview -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            System Architecture Overview
          </h2>
          
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="text-center">
              <div class="text-3xl mb-2">📄</div>
              <h3 class="font-medium text-gray-900 dark:text-white">Document Processing</h3>
              <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                PDF extraction, OCR, and text analysis
              </p>
            </div>
            
            <div class="text-center">
              <div class="text-3xl mb-2">🧠</div>
              <h3 class="font-medium text-gray-900 dark:text-white">AI Embeddings</h3>
              <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Vector generation with nomic-embed-text
              </p>
            </div>
            
            <div class="text-center">
              <div class="text-3xl mb-2">🚀</div>
              <h3 class="font-medium text-gray-900 dark:text-white">GPU Acceleration</h3>
              <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                CUDA-optimized processing pipeline
              </p>
            </div>
            
            <div class="text-center">
              <div class="text-3xl mb-2">💬</div>
              <h3 class="font-medium text-gray-900 dark:text-white">Legal AI Chat</h3>
              <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Contextual legal Q&A with citations
              </p>
            </div>
          </div>
        </div>

        <!-- Quick Actions -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Quick Actions
          </h2>
          
          <div class="flex flex-wrap gap-4">
            <button
              on:click={runExample}
              class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              📤 Process Example Document
            </button>
            
            <button
              on:click={runQueryExample}
              class="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            >
              🔍 Run Example Query
            </button>
            
            <a
              href="/api/rag/orchestrate?query=What%20are%20the%20main%20legal%20principles?"
              target="_blank"
              class="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
            >
              🔗 Test API Directly
            </a>
          </div>
        </div>

        <!-- Production Dashboard -->
        <ProductionRAGDashboard />
        
        <!-- API Documentation -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            API Endpoints
          </h2>
          
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="space-y-4">
              <h3 class="font-medium text-gray-900 dark:text-white">Document Processing</h3>
              
              <div class="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <div class="flex items-center space-x-2 mb-2">
                  <span class="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">POST</span>
                  <code class="text-sm">/api/rag/orchestrate</code>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-400">
                  Start document processing pipeline
                </p>
              </div>
              
              <div class="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <div class="flex items-center space-x-2 mb-2">
                  <span class="px-2 py-1 bg-green-100 text-green-800 text-xs rounded">GET</span>
                  <code class="text-sm">/api/rag/orchestrate/status/[jobId]</code>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-400">
                  Get processing job status
                </p>
              </div>
            </div>
            
            <div class="space-y-4">
              <h3 class="font-medium text-gray-900 dark:text-white">RAG Queries</h3>
              
              <div class="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <div class="flex items-center space-x-2 mb-2">
                  <span class="px-2 py-1 bg-green-100 text-green-800 text-xs rounded">GET</span>
                  <code class="text-sm">/api/rag/orchestrate?query=...</code>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-400">
                  Query the RAG system
                </p>
              </div>
              
              <div class="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <div class="flex items-center space-x-2 mb-2">
                  <span class="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded">PATCH</span>
                  <code class="text-sm">/api/rag/orchestrate</code>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-400">
                  System health check
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Performance Metrics -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Performance Targets
          </h2>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="text-center">
              <div class="text-2xl font-bold text-green-600">< 5ms</div>
              <p class="text-sm text-gray-600 dark:text-gray-400">QUIC Latency</p>
            </div>
            
            <div class="text-center">
              <div class="text-2xl font-bold text-blue-600">150+</div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Tokens/sec</p>
            </div>
            
            <div class="text-center">
              <div class="text-2xl font-bold text-purple-600">99.9%</div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Uptime</p>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </main>
</div>

<style>
  code {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }
  
  .animate-spin {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>