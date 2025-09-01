<script lang="ts">

  import { onMount } from 'svelte';
let diagram = $state(`graph TD\n    A[src/routes/interactive-canvas/+page.svelte] --> B[+Header.svelte]\n    A --> C[+Sidebar.svelte]\n    A --> D[Main Content Area]\n    D --> E[+FileUploadSection.svelte]\n    D --> F[+AutomateUploadSection.svelte]\n    D --> G[+AddNotesSection.svelte]\n    E --> H[+Dropdown.svelte]\n    E --> I[+Checkbox.svelte]\n    F --> H\n    F --> I\n    G --> H\n    G --> I\n    B --> J[+SearchInput.svelte]`);
  let svg = $state('');
  let container: HTMLElement = $state();
  
  onMount(async () => {
    try {
      const { default: mermaid } = await import('mermaid');
      mermaid.initialize({ startOnLoad: false });
      const { svg: renderedSvg } = await mermaid.render('ui-diagram', diagram);
      svg = renderedSvg;
    } catch (err) {
      console.error('Failed to load mermaid:', err);
      svg = '<p>Diagram failed to load</p>';
}
  });
</script>

<div class="space-y-4" bind:this={container}>
  {@html svg}
</div>

<style>
  /* @unocss-include */
.mermaid-diagram-container {
  width: 100%;
  max-width: 900px;
  margin: 2rem auto;
  background: var(--pico-background, #fff);
  border-radius: 1rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  padding: 2rem;
  overflow-x: auto;
}
</style>


