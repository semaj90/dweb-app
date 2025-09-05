<script lang="ts">
import { Button } from 'bits-ui';
import { onMount } from 'svelte';

let response = '';
let loading = false;
let error = '';
let systemStatus = { gpu: false, ollama: false, synthesis: false };

async function checkSystemStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    systemStatus = {
      gpu: data.services.gpu === 'accelerated',
      ollama: data.services.ollama === 'healthy',
      synthesis: res.ok
    };
  } catch (e) {
    error = 'System health check failed';
  }
}

async function synthesize(type: 'correlation' | 'timeline' | 'compare' | 'merge') {
  loading = true;
  error = '';
  
  try {
    const res = await fetch('/api/evidence/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        type, 
        evidenceIds: ['EVD-001', 'EVD-002'],
        caseId: 'CASE-2024-001',
        title: `${type} synthesis test`
      })
    });
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    
    const data = await res.json();
    response = JSON.stringify(data, null, 2);
  } catch (e) {
    error = e.message;
    response = '';
  } finally {
    loading = false;
  }
}

async function testGemma3() {
  loading = true;
  try {
    const res = await fetch('/api/ai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message: 'What are the key elements of contract law?',
        model: 'gemma3-legal'
      })
    });
    const data = await res.json();
    response = data.response || data.error;
  } catch (e) {
    error = e.message;
  } finally {
    loading = false;
  }
}

onMount(checkSystemStatus);
</script>

<div class="p-6 max-w-4xl mx-auto">
  <h1 class="text-3xl font-bold mb-6">AI Assistant - Production Interface</h1>
  
  <!-- System Status -->
  <div class="bg-gray-900 p-4 rounded-lg mb-6">
    <h2 class="text-xl mb-3">System Status</h2>
    <div class="grid grid-cols-3 gap-4">
      <div class="flex items-center gap-2">
        <div class="w-3 h-3 rounded-full {systemStatus.gpu ? 'bg-green-500' : 'bg-red-500'}"></div>
        <span>GPU: {systemStatus.gpu ? 'Accelerated' : 'CPU Fallback'}</span>
      </div>
      <div class="flex items-center gap-2">
        <div class="w-3 h-3 rounded-full {systemStatus.ollama ? 'bg-green-500' : 'bg-red-500'}"></div>
        <span>Ollama: {systemStatus.ollama ? 'Active' : 'Offline'}</span>
      </div>
      <div class="flex items-center gap-2">
        <div class="w-3 h-3 rounded-full {systemStatus.synthesis ? 'bg-green-500' : 'bg-red-500'}"></div>
        <span>Synthesis: {systemStatus.synthesis ? 'Ready' : 'Error'}</span>
      </div>
    </div>
  </div>

  <!-- Evidence Synthesis Controls -->
  <div class="bg-gray-800 p-4 rounded-lg mb-6">
    <h2 class="text-xl mb-3">Evidence Synthesis</h2>
    <div class="grid grid-cols-4 gap-3">
      <Button.Root 
        class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
        onclick={() => synthesize('correlation')}
        disabled={loading}
      >
        Correlation
      </Button.Root>
      <Button.Root 
        class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
        onclick={() => synthesize('timeline')}
        disabled={loading}
      >
        Timeline
      </Button.Root>
      <Button.Root 
        class="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded"
        onclick={() => synthesize('compare')}
        disabled={loading}
      >
        Compare
      </Button.Root>
      <Button.Root 
        class="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded"
        onclick={() => synthesize('merge')}
        disabled={loading}
      >
        Merge
      </Button.Root>
    </div>
  </div>

  <!-- Gemma3 Legal Test -->
  <div class="bg-gray-800 p-4 rounded-lg mb-6">
    <h2 class="text-xl mb-3">Gemma3 Legal AI</h2>
    <Button.Root 
      class="bg-red-600 hover:bg-red-700 px-6 py-2 rounded"
      onclick={testGemma3}
      disabled={loading}
    >
      Test Legal Query
    </Button.Root>
  </div>

  <!-- Response Display -->
  <div class="bg-black p-4 rounded-lg">
    <h3 class="text-lg mb-2">Response</h3>
    {#if loading}
      <div class="text-blue-400">Processing...</div>
    {:else if error}
      <div class="text-red-400">Error: {error}</div>
    {:else if response}
      <pre class="text-green-400 whitespace-pre-wrap overflow-x-auto">{response}</pre>
    {:else}
      <div class="text-gray-500">No response yet</div>
    {/if}
  </div>

  <!-- System Actions -->
  <div class="mt-6 flex gap-3">
    <Button.Root 
      class="bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded"
      onclick={checkSystemStatus}
    >
      Refresh Status
    </Button.Root>
    <Button.Root 
      class="bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded"
      onclick={() => window.open('/api/health', '_blank')}
    >
      Health Check
    </Button.Root>
  </div>
</div>

<style>
  :global(body) {
    background: #111;
    color: #fff;
  }
</style>