
<script lang="ts">
// import { createBubbler } from 'svelte/legacy';
// const bubble = createBubbler();
import '../app.css';
import '../lib/styles/nier.css';
import '../lib/styles/theme.css';
import Navigation from '$lib/components/Navigation.svelte';
// Temporarily disabled YoRHa managers due to Svelte 5 compatibility issues
// import YoRHaNotificationManager from '$lib/components/yorha/YoRHaNotificationManager.svelte';
// import YoRHaDialogManager from '$lib/components/yorha/YoRHaDialogManager.svelte';
// import YoRHaModalManager from '$lib/components/yorha/YoRHaModalManager.svelte';
import { notificationStore as notificationStoreExport } from '$lib/stores/notifications';
// import { aiService } from '$lib/services/ai-service';
import { onMount } from 'svelte';
  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

let llmEndpoint = $state('');
let llmStatus: 'Ollama' | 'vLLM' | 'offline' = $state('offline');

onMount(() => {
  // Temporary disable AI service connection for development
  llmEndpoint = 'mock://localhost:5175';
  llmStatus = 'offline';

  // TODO: Re-enable when AI service is fixed
  // llmEndpoint = aiService.getCurrentLlmEndpoint?.() || '';
  // if (llmEndpoint.includes('11434')) llmStatus = 'Ollama';
  // else if (llmEndpoint.includes('8000')) llmStatus = 'vLLM';
  // else llmStatus = 'offline';

  // Dev-only sanity ping to confirm notifications are wired
  if (import.meta.env.DEV) {
    const notificationStore = notificationStoreExport as unknown as {
      info: (message: string, options?: { duration?: number; position?: 'top-right'|'top-left'|'bottom-right'|'bottom-left'|'center'; showProgress?: boolean }) => string;
    };
    notificationStore.info('YoRHa notifications connected', {
      duration: 2000,
      position: 'top-right',
      showProgress: false
    });
  }
});
</script>

<!-- <svelte:window onkeydown={bubble('keydown')} /> -->

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
	{@render children?.()}
  </div>
  
  <!-- Global UI Managers - Temporarily disabled for Svelte 5 compatibility -->
  <!-- <YoRHaNotificationManager />
  <YoRHaDialogManager />
  <YoRHaModalManager /> -->
</main>