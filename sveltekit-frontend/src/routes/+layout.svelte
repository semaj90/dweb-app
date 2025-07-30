
<script lang="ts">
import '../app.css';
import '../lib/styles/nier.css';
import '../lib/styles/theme.css';
import Navigation from '$lib/components/Navigation.svelte';
import { aiService } from '$lib/services/ai-service';
import { onMount } from 'svelte';

let llmEndpoint = '';
let llmStatus: 'Ollama' | 'vLLM' | 'offline' = 'offline';

onMount(() => {
  llmEndpoint = aiService.getCurrentLlmEndpoint?.() || '';
  if (llmEndpoint.includes('11434')) llmStatus = 'Ollama';
  else if (llmEndpoint.includes('8000')) llmStatus = 'vLLM';
  else llmStatus = 'offline';
});
</script>

<svelte:window on:keydown />

<main class="min-h-screen bg-background font-mono">
  <Navigation />
  <div class="w-full flex justify-end items-center gap-2 p-2">
	<span
	  class="ai-status-indicator {llmStatus === 'Ollama' ? 'ai-status-online' : llmStatus === 'vLLM' ? 'ai-status-processing' : 'ai-status-offline'}"
	  aria-label="LLM backend status"
	  title={llmStatus}
	></span>
	<span class="text-xs text-nier-text-muted">
	  LLM: {llmStatus}{llmEndpoint ? ` (${llmEndpoint})` : ''}
	</span>
  </div>
  <div class="container mx-auto p-4">
	<slot />
  </div>
</main>