<script lang="ts">
  import Button from '$lib/components/ui/MeltButton.svelte';
  import { onMount } from 'svelte';

  // Use plain reactive variables instead of undefined $state helper
  let response: string = '';
  let loading: boolean = false;
  let error: string = '';
  let systemStatus = { gpu: false, ollama: false, synthesis: false };

  async function checkSystemStatus() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      systemStatus = {
        gpu: data?.services?.gpu === 'accelerated',
        ollama: data?.services?.ollama === 'healthy',
        synthesis: res.ok
      };
    } catch (e: any) {
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
    } catch (e: any) {
      error = e?.message ?? String(e);
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
      response = data?.response ?? data?.error ?? JSON.stringify(data);
    } catch (e: any) {
      error = e?.message ?? String(e);
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
        <div
          class="w-3 h-3 rounded-full"
          class:bg-green-500={systemStatus.gpu}
          class:bg-red-500={!systemStatus.gpu}
        ></div>
        <span>GPU: {systemStatus.gpu ? 'Accelerated' : 'CPU Fallback'}</span>
      </div>
      <div class="flex items-center gap-2">
        <div
          class="w-3 h-3 rounded-full"
          class:bg-green-500={systemStatus.ollama}
          class:bg-red-500={!systemStatus.ollama}
        ></div>
        <span>Ollama: {systemStatus.ollama ? 'Active' : 'Offline'}</span>
      </div>
      <div class="flex items-center gap-2">
        <div
          class="w-3 h-3 rounded-full"
          class:bg-green-500={systemStatus.synthesis}
          class:bg-red-500={!systemStatus.synthesis}
        ></div>
        <span>Synthesis: {systemStatus.synthesis ? 'Ready' : 'Error'}</span>
      </div>
    </div>
  </div>

  <!-- Evidence Synthesis Controls -->
  <div class="bg-gray-800 p-4 rounded-lg mb-6">
    <h2 class="text-xl mb-3">Evidence Synthesis</h2>
    <div class="grid grid-cols-4 gap-3">
      <Button
        class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
        click={() => synthesize('correlation')}
        {disabled}
        disabled={loading}
      >
        Correlation
      </Button>
      <Button
        class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
        click={() => synthesize('timeline')}
        disabled={loading}
      >
        Timeline
      </Button>
      <Button
        class="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded"
        click={() => synthesize('compare')}
        disabled={loading}
      >
        Compare
      </Button>
      <Button
        class="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded"
        click={() => synthesize('merge')}
        disabled={loading}
      >
        Merge
      </Button>
    </div>
  </div>

  <!-- Gemma3 Legal Test -->
  <div class="bg-gray-800 p-4 rounded-lg mb-6">
    <h2 class="text-xl mb-3">Gemma3 Legal AI</h2>
    <Button
      class="bg-red-600 hover:bg-red-700 px-6 py-2 rounded"
      click={testGemma3}
      disabled={loading}
    >
      Test Legal Query
    </Button>
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
    <Button
      class="bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded"
      click={checkSystemStatus}
    >
      Refresh Status
    </Button>
    <Button
      class="bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded"
      click={() => window.open('/api/health', '_blank')}
    >
      Health Check
    </Button>
  </div>
</div>

<style>
  :global(body) {
    background: #111;
    color: #fff;
  }
</style>