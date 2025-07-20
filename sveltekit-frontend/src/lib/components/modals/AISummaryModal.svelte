<script lang="ts">
  import { aiService } from '$lib/services/aiService';
  import Dialog from '$lib/components/ui/dialog/Dialog.svelte';
  import Button from "$lib/components/ui/button";
  import Badge from '$lib/components/ui/Badge.svelte';
  import { Sparkles, Copy, X, AlertCircle, Check } from 'lucide-svelte';
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();
  let copied = false;

  // Use the Svelte store reactively
  $: summary = $aiService.summary;
  $: isLoading = $aiService.isLoading;
  $: error = $aiService.error;
  $: model = $aiService.model;
  $: lastSummarizedContent = $aiService.lastSummarizedContent;
  $: isOpen = isLoading || summary !== null || error !== null;

  async function copyToClipboard() {
    if (summary) {
      try {
        await navigator.clipboard.writeText(summary);
        copied = true;
        setTimeout(() => copied = false, 2000);
      } catch (err) {
        console.error('Failed to copy text:', err);}}}
  function closeModal() {
    aiService.reset();
    dispatch('close');}
</script>

<Dialog open={isOpen} on:close={closeModal} size="lg" title="AI Summary" description="AI-generated summary of your content">
  <div slot="header" class="container mx-auto px-4">
    <Sparkles class="container mx-auto px-4" />
    <span>AI Summary</span>
    {#if model}
      <Badge variant="secondary" class="container mx-auto px-4">{model}</Badge>
    {/if}
  </div>

  <div class="container mx-auto px-4">
    {#if isLoading}
      <!-- Loading State -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4"></div>
          <span class="container mx-auto px-4">Analyzing content...</span>
        </div>
      </div>
    {:else if error}
      <!-- Error State -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <AlertCircle class="container mx-auto px-4" />
          <span class="container mx-auto px-4">AI Error</span>
        </div>
        <p class="container mx-auto px-4">{error}</p>
      </div>
    {:else if summary}
      <!-- Summary Content -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <Button on:click={() => copyToClipboard()} variant="ghost" size="sm" aria-label="Copy summary to clipboard">
            <Copy class="container mx-auto px-4" />
            <span class="container mx-auto px-4">Copy</span>
          </Button>
          {#if copied}
            <span class="container mx-auto px-4"><Check class="container mx-auto px-4" />Copied!</span>
          {/if}
        </div>
        <div class="container mx-auto px-4">
          {summary}
        </div>
        {#if lastSummarizedContent}
          <div class="container mx-auto px-4">
            <span class="container mx-auto px-4">Source:</span> {lastSummarizedContent}
          </div>
        {/if}
      </div>
    {:else}
      <div class="container mx-auto px-4">No summary available.</div>
    {/if}
  </div>

  <div slot="footer" class="container mx-auto px-4">
    <Button on:click={() => closeModal()} variant="secondary" aria-label="Close summary modal">
      <X class="container mx-auto px-4" />
      <span class="container mx-auto px-4">Close</span>
    </Button>
  </div>
</Dialog>

<style>
  /* @unocss-include */
  .prose {
    max-width: none;}
</style>
