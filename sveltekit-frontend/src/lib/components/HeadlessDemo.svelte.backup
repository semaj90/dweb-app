<script lang="ts">
  import { createDialog, createSelect } from '@melt-ui/svelte';
  import { fade } from 'svelte/transition';
  
  export const title = 'Legal Case Manager';
  export let items = ['Active Cases', 'Pending Cases', 'Closed Cases'];
  
  // Melt UI Dialog
  const {
    elements: { trigger, overlay, content, title: dialogTitle, description, close },
    states: { open }
  } = createDialog();
  
  // Melt UI Select
  const {
    elements: { trigger: selectTrigger, menu, option, label },
    states: { selectedLabel, open: selectOpen }
  } = createSelect({
    defaultSelected: { value: items[0], label: items[0] }
  });
</script>

<div class="mx-auto px-4 max-w-7xl">
  <h2 class="mx-auto px-4 max-w-7xl">Headless UI Components Demo</h2>
  
  <!-- Melt UI Button -->
  <button class="mx-auto px-4 max-w-7xl">
    Primary Action Button
  </button>
  
  <!-- Melt UI Select -->
  <div class="mx-auto px-4 max-w-7xl">
    <button 
      use:selectTrigger 
      class="mx-auto px-4 max-w-7xl"
      aria-label="Case Type Filter"
    >
      {$selectedLabel || 'Select case type...'}
    </button>
    
    {#if $selectOpen}
      <div 
        use:menu 
        class="mx-auto px-4 max-w-7xl"
        transition:fade={{ duration: 150 }}
      >
        {#each items as item, index}
          <div 
            use:option
            class="mx-auto px-4 max-w-7xl"
            data-value={item}
          >
            {item}
          </div>
        {/each}
      </div>
    {/if}
  </div>
  
  <!-- Melt UI Dialog Trigger -->
  <button 
    use:trigger
    class="mx-auto px-4 max-w-7xl"
  >
    Open Case Details Dialog
  </button>
  
  <!-- Melt UI Dialog -->
  {#if $open}
    <div use:overlay class="mx-auto px-4 max-w-7xl" transition:fade={{ duration: 150 }}>
      <div use:content class="mx-auto px-4 max-w-7xl">
        <h3 use:dialogTitle class="mx-auto px-4 max-w-7xl">
          Case Management System
        </h3>
        <p use:description class="mx-auto px-4 max-w-7xl">
          This is a demo of Melt UI headless components integrated with PicoCSS styling and custom CSS variables.
        </p>
        
        <div class="mx-auto px-4 max-w-7xl">
          <button use:close class="mx-auto px-4 max-w-7xl">
            Cancel
          </button>
          <button class="mx-auto px-4 max-w-7xl">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .headless-demo {
    max-width: 500px;
    margin: 0 auto;
    padding: var(--spacing-lg);
  }

  .select-menu {
    position: absolute;
    z-index: 50;
    min-width: 200px;
    background-color: var(--color-background);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    padding: var(--spacing-xs);
    margin-top: var(--spacing-xs);
  }

  .select-option {
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background-color var(--transition-fast);
  }

  .select-option:hover {
    background-color: var(--color-surface);
  }

  .dialog-overlay {
    position: fixed;
    inset: 0;
    z-index: 50;
    background-color: rgb(0 0 0 / 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-lg);
  }

  .dialog-content {
    background-color: var(--color-background);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    padding: var(--spacing-xl);
    max-width: 500px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
  }

  .dialog-title {
    font-size: var(--font-size-xl);
    font-weight: 600;
    margin-bottom: var(--spacing-md);
    color: var(--color-text);
  }

  .dialog-description {
    color: var(--color-text-muted);
    margin-bottom: var(--spacing-lg);
    line-height: 1.6;
  }

  .dialog-actions {
    display: flex;
    gap: var(--spacing-sm);
    justify-content: flex-end;
  }
</style>