<script lang="ts">
  import { aiService } from "../../../lib/services/aiService";
  import Dialog from '../../../lib/components/ui/dialog/Dialog.svelte';
  import { Button } from "$lib/components/ui/button";
  import Badge from '../../../lib/components/ui/Badge.svelte';
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
        console.error('Failed to copy text:', err);
      }
    }
  }

  function closeModal() {
    aiService.reset();
    dispatch('close');
  }
</script>

<Dialog open={isOpen} on:close={closeModal} size="lg" title="AI Summary" description="AI-generated summary of your content">
  <div slot="header" class="mx-auto px-4 max-w-7xl">
    <Sparkles class="mx-auto px-4 max-w-7xl" />
    <span>AI Summary</span>
    {#if model}
      <Badge variant="secondary" class="mx-auto px-4 max-w-7xl">{model}</Badge>
    {/if}
  </div>

  <div class="mx-auto px-4 max-w-7xl">
    {#if isLoading}
      <!-- Loading State -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl"></div>
          <span class="mx-auto px-4 max-w-7xl">Analyzing content...</span>
        </div>
      </div>
    {:else if error}
      <!-- Error State -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <AlertCircle class="mx-auto px-4 max-w-7xl" />
          <span class="mx-auto px-4 max-w-7xl">AI Error</span>
        </div>
        <p class="mx-auto px-4 max-w-7xl">{error}</p>
      </div>
    {:else if summary}
      <!-- Summary Content -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Button onclick={() => copyToClipboard()} variant="ghost" size="sm" aria-label="Copy summary to clipboard">
            <Copy class="mx-auto px-4 max-w-7xl" />
            <span class="mx-auto px-4 max-w-7xl">Copy</span>
          </Button>
          {#if copied}
            <span class="mx-auto px-4 max-w-7xl"><Check class="mx-auto px-4 max-w-7xl" />Copied!</span>
          {/if}
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          {summary}
        </div>
        {#if lastSummarizedContent}
          <div class="mx-auto px-4 max-w-7xl">
            <span class="mx-auto px-4 max-w-7xl">Source:</span> {lastSummarizedContent}
          </div>
        {/if}
      </div>
    {:else}
      <div class="mx-auto px-4 max-w-7xl">No summary available.</div>
    {/if}
  </div>

  <div slot="footer" class="mx-auto px-4 max-w-7xl">
    <Button onclick={() => closeModal()} variant="secondary" aria-label="Close summary modal">
      <X class="mx-auto px-4 max-w-7xl" />
      <span class="mx-auto px-4 max-w-7xl">Close</span>
    </Button>
  </div>
</Dialog>

<style>
  .prose {
    max-width: none;
  }
</style>
