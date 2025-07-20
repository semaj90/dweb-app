<script lang="ts">
  import { page } from '$app/stores';
  import VisualEvidenceEditor from '$lib/components/evidence-editor/VisualEvidenceEditor.svelte';
  import { Button } from '$lib/components/ui';
  import { onMount } from 'svelte';
  
  let caseId: string | null = null;
  let readOnly = false;
  
  onMount(() => {
    // Get case ID from URL params if provided
    caseId = $page.url.searchParams.get('caseId');
    readOnly = $page.url.searchParams.get('readOnly') === 'true';
  });
  
  function toggleReadOnly() {
    readOnly = !readOnly;
}
</script>

<svelte:head>
  <title>Visual Evidence Editor - Legal AI Assistant</title>
  <meta name="description" content="Advanced visual evidence management with AI-powered tagging and analysis" />
</svelte:head>

<div class="container mx-auto px-4">
  <!-- Header -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div>
        <h1 class="container mx-auto px-4">Visual Evidence Editor</h1>
        <p class="container mx-auto px-4">
          Drag and drop evidence files for AI-powered analysis and tagging
        </p>
      </div>
      
      <div class="container mx-auto px-4">
        <Button 
          on:click={toggleReadOnly}
          variant={readOnly ? "default" : "outline"}
          size="sm"
        >
          {readOnly ? 'Enable Editing' : 'Read Only'}
        </Button>
        
        <div class="container mx-auto px-4">
          {#if caseId}
            Case: {caseId}
          {:else}
            Demo Mode
          {/if}
        </div>
      </div>
    </div>
  </div>
  
  <!-- Main Editor -->
  <div class="container mx-auto px-4">
    <VisualEvidenceEditor {caseId} {readOnly} />
  </div>
</div>

<!-- Help Overlay (initially hidden) -->
<div class="container mx-auto px-4" style="display: none;" id="help-overlay">
  <h3 class="container mx-auto px-4">Quick Start Guide</h3>
  <ul class="container mx-auto px-4">
    <li>• Drag files onto the canvas to add evidence</li>
    <li>• Files are automatically analyzed with AI</li>
    <li>• Click evidence to view details in the inspector</li>
    <li>• Use the AI assistant for search and insights</li>
    <li>• Edit metadata and tags in the inspector panel</li>
  </ul>
  <Button 
    size="sm" 
    class="container mx-auto px-4"
    on:click={() => {
      const helpOverlay = document.getElementById('help-overlay');
      if (helpOverlay) {
        helpOverlay.style.display = 'none';
}
    "
  >
    Got it!
  </Button>
</div>

<style>
  /* @unocss-include */
  .evidence-editor-page {
    height: 100vh;
    overflow: hidden;
}
</style>
