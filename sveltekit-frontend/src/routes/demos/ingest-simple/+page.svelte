<script lang="ts">
  /**
   * Simple Document Ingest Demo
   * Demonstrates the working ingest integration without complex UI dependencies
   */
  
  import { onMount } from 'svelte';
  
  let serviceHealth = null;
  let serviceStatus = 'checking...';
  let ingestResult = null;
  let documentTitle = 'Test Contract';
  let documentContent = 'This is a test legal contract with indemnification clauses and termination provisions.';
  let isProcessing = false;
  
  async function checkServiceHealth() {
    try {
      const response = await fetch('/api/v1/ingest');
      if (response.ok) {
        serviceHealth = await response.json();
        serviceStatus = 'healthy';
      } else {
        serviceStatus = 'unhealthy';
      }
    } catch (error) {
      console.error('Health check failed:', error);
      serviceStatus = 'error';
    }
  }
  
  async function ingestDocument() {
    if (isProcessing) return;
    
    isProcessing = true;
    ingestResult = null;
    
    try {
      const response = await fetch('/api/v1/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: documentTitle,
          content: documentContent,
          case_id: 'DEMO-CASE-001',
          metadata: { document_type: 'contract', demo: true }
        })
      });
      
      if (response.ok) {
        ingestResult = await response.json();
      } else {
        const error = await response.json();
        ingestResult = { error: error.error || 'Ingest failed' };
      }
    } catch (error) {
      ingestResult = { error: error.message };
    } finally {
      isProcessing = false;
    }
  }
  
  onMount(() => {
    checkServiceHealth();
  });
</script>

<svelte:head>
  <title>Simple Ingest Demo</title>
</svelte:head>

<div class="container mx-auto p-6 max-w-4xl">
  <h1 class="text-3xl font-bold mb-6">Document Ingest Integration - Demo</h1>
  
  <!-- Service Status -->
  <div class="bg-white border rounded-lg p-6 mb-6">
    <h2 class="text-xl font-semibold mb-4">Service Status</h2>
    <div class="flex items-center space-x-4">
      <div class="w-3 h-3 rounded-full {serviceStatus === 'healthy' ? 'bg-green-500' : serviceStatus === 'checking...' ? 'bg-yellow-500' : 'bg-red-500'}"></div>
      <span class="font-medium">{serviceStatus}</span>
      {#if serviceHealth}
        <span class="text-sm text-gray-600">
          Port {serviceHealth.upstream?.port} • {serviceHealth.upstream?.config?.embed_model}
        </span>
      {/if}
    </div>
  </div>
  
  <!-- Ingest Form -->
  <div class="bg-white border rounded-lg p-6 mb-6">
    <h2 class="text-xl font-semibold mb-4">Test Document Ingest</h2>
    
    <div class="space-y-4">
      <div>
        <label class="block text-sm font-medium mb-2">Document Title</label>
        <input 
          type="text" 
          bind:value={documentTitle}
          class="w-full p-2 border rounded"
          disabled={isProcessing}
        />
      </div>
      
      <div>
        <label class="block text-sm font-medium mb-2">Document Content</label>
        <textarea 
          bind:value={documentContent}
          rows="4"
          class="w-full p-2 border rounded"
          disabled={isProcessing}
        ></textarea>
      </div>
      
      <button 
        onclick={ingestDocument}
        disabled={isProcessing || serviceStatus !== 'healthy'}
        class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400"
      >
        {isProcessing ? 'Processing...' : '🚀 Ingest Document'}
      </button>
    </div>
  </div>
  
  <!-- Results -->
  {#if ingestResult}
    <div class="bg-white border rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">Result</h2>
      
      {#if ingestResult.error}
        <div class="bg-red-50 border border-red-200 rounded p-4">
          <p class="text-red-700">Error: {ingestResult.error}</p>
        </div>
      {:else}
        <div class="bg-green-50 border border-green-200 rounded p-4">
          <h3 class="font-semibold text-green-800 mb-2">✅ Success!</h3>
          <div class="text-sm space-y-1">
            <p><strong>Document ID:</strong> {ingestResult.document_id}</p>
            <p><strong>Embedding ID:</strong> {ingestResult.embedding_id}</p>
            <p><strong>Processing Time:</strong> {ingestResult.process_time_ms?.toFixed(1)}ms</p>
            <p><strong>Status:</strong> {ingestResult.status}</p>
          </div>
        </div>
      {/if}
    </div>
  {/if}
  
  <!-- API Info -->
  <div class="bg-gray-50 border rounded-lg p-6 mt-6">
    <h2 class="text-xl font-semibold mb-4">Integration Details</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
      <div>
        <h3 class="font-semibold mb-2">Architecture</h3>
        <ul class="space-y-1">
          <li>• SvelteKit Frontend (port 5173)</li>
          <li>• Go Ingest Service (port 8227)</li>
          <li>• PostgreSQL + pgvector</li>
          <li>• Ollama Embeddings (768 dims)</li>
        </ul>
      </div>
      <div>
        <h3 class="font-semibold mb-2">API Endpoints</h3>
        <ul class="space-y-1">
          <li>• GET /api/v1/ingest (health)</li>
          <li>• POST /api/v1/ingest (single)</li>
          <li>• POST /api/v1/ingest/batch</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<style>
  .container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  }
</style>