/// <reference types="vite/client" />
<!-- Test page for Simple File Upload with RAG integration -->
<script lang="ts">
  import SimpleFileUpload from '$lib/components/ai/SimpleFileUpload.svelte';
  import { onMount } from 'svelte';

  interface ServiceStatus { healthy: boolean; [key: string]: unknown }
  interface SystemStatus { services?: Record<string, ServiceStatus>; [key: string]: unknown }

  let uploadResults = $state<unknown[]>([]);
  let systemStatus = $state<SystemStatus>({});

  function handleUploadComplete(result: any) {
    console.log('Upload completed:', result);
    uploadResults = [...uploadResults, result];
  }

  onMount(async () => {
    try {
      // Production-ready REST status polling with retries, timeout, backoff, and background loop
      const API_BASE = import.meta.env.VITE_API_BASE || '';
      const MAX_RETRIES = 5;
      const REQUEST_TIMEOUT_MS = 8000;
      const POLL_INTERVAL_MS = 5000;
let pollActive = $state(true);

      async function fetchStatus(attempt = 1) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
        try {
          const res = await fetch(`${API_BASE}/api/rag/status`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        signal: controller.signal,
        cache: 'no-store'
          });
        clearTimeout(timer);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const json = await res.json();
          systemStatus = json;
          return res;
        } catch (err) {
          clearTimeout(timer);
          if (attempt < MAX_RETRIES && pollActive) {
        const backoff = Math.min(1000 * 2 ** (attempt - 1), 10000) + Math.random() * 250;
        console.warn(`Status fetch failed (attempt ${attempt}):`, err);
        await new Promise(r => setTimeout(r, backoff));
        return fetchStatus(attempt + 1);
          } else {
        console.error('Giving up fetching system status:', err);
          }
        }
      }

      const first = await fetchStatus(); // initial immediate load

      // Provide a Response object for existing code below (re-used json via systemStatus)
      const response = new Response(JSON.stringify(systemStatus), {
        status: first?.status || (systemStatus ? 200 : 500),
        headers: { 'Content-Type': 'application/json' }
      });

      // Background polling loop
      (async function poll() {
        while (pollActive) {
          await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
          if (document.hidden) continue; // skip while tab hidden
          await fetchStatus(1);
        }
      })();

      // Stop polling when leaving page
      addEventListener('beforeunload', () => { pollActive = false; });
      if (response.ok) {
        systemStatus = await response.json();
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  });
</script>

<svelte:head>
  <title>Enhanced File Upload Test - Legal AI System</title>
</svelte:head>

<div class="container mx-auto p-6">
  <h1 class="text-3xl font-bold mb-6">Enhanced File Upload Test</h1>

  <!-- System Status Display (fixed) -->
  {#if systemStatus.services}
    <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
      {#each Object.entries(systemStatus.services as Record<string, ServiceStatus>) as [service, status]}
        <div class="text-center">
          <div class="font-medium capitalize">{service}</div>
          <div class="text-sm {status.healthy ? 'text-green-600' : 'text-red-600'}">
            {status.healthy ? '✓ Online' : '✗ Offline'}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="text-gray-500">Loading system status...</div>
  {/if}

  <!-- Simple File Upload Component (UnoCSS attributify) -->
  <!-- Converted UnoCSS attributify props to class to satisfy TS HTMLProps -->
  <div class="mb-8 border border-gray-200 rounded-lg p-4">
    <SimpleFileUpload uploadcomplete={handleUploadComplete} />
  </div>

  <!-- Upload Results -->
  {#if uploadResults.length > 0}
    <div class="mt-8">
      <h2 class="text-xl font-semibold mb-4">Upload Results</h2>
      <div class="space-y-4">
        {#each uploadResults as result}
          <div class="p-4 border rounded-lg bg-white shadow-sm">
            <div class="flex justify-between items-start mb-2">
              <h3 class="font-medium">{result.filename || 'Unknown file'}</h3>
              <span class="px-2 py-1 text-xs rounded {result.status === 'processed' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                {result.status}
              </span>
            </div>
            {#if result.documentId}
              <p class="text-sm text-gray-600 mb-1">Document ID: {result.documentId}</p>
            {/if}
            {#if result.size}
              <p class="text-sm text-gray-600 mb-1">Size: {(result.size / 1024).toFixed(1)} KB</p>
            {/if}
            {#if result.embeddingGenerated}
              <p class="text-sm text-green-600">✓ Embeddings generated</p>
            {/if}
            {#if result.error}
              <p class="text-sm text-red-600">Error: {result.error}</p>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Debug Information -->
  <div class="mt-8 p-4 bg-gray-50 rounded-lg">
    <h2 class="text-lg font-semibold mb-2">Debug Information</h2>
    <div class="text-sm space-y-1">
      <p><strong>PostgreSQL:</strong> localhost:5432 (Native Windows)</p>
      <p><strong>Redis:</strong> localhost:6379 (Native Windows)</p>
      <p><strong>Qdrant:</strong> localhost:6333 (Native Windows)</p>
      <p><strong>Frontend:</strong> http://localhost:5173</p>
      <p><strong>Ollama:</strong> http://localhost:11434 (Native Windows)</p>
      <p><strong>Features:</strong> OCR, Embeddings, Vector Search, Local File Storage</p>
    </div>
  </div>
</div>

<!-- Replaced raw CSS with UnoCSS utilities (no <style> needed).
  Ensure the wrapping div uses: class="mx-auto p-6 max-w-1200px"
  Add 'max-w-1200px' to safelist in uno.config if using arbitrary values.
-->
