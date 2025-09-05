<script lang="ts">
  interface Props {
    items?: any;
  }
  let {
    items = ['Active Cases', 'Pending Cases', 'Closed Cases']
  } = $props();



  
  import { fade } from 'svelte/transition';
  
  export const title = 'Legal Case Manager';
    
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

<div class="space-y-4">
  <h2 class="space-y-4">Headless UI Components Demo</h2>
  
  <!-- Melt UI Button -->
  <button class="space-y-4">
    Primary Action Button
  </button>
  
  <!-- Melt UI Select -->
  <div class="space-y-4">
    <button 
      use:selectTrigger 
      class="space-y-4"
      aria-label="Case Type Filter"
    >
      {$selectedLabel || 'Select case type...'}
    </button>
    
    {#if $selectOpen}
      <div 
        use:menu 
        class="space-y-4"
        transitionfade={{ duration: 150 }}
      >
        {#each items as item, index}
          <div 
            use:option
            class="space-y-4"
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
    class="space-y-4"
  >
    Open Case Details Dialog
  </button>
  
  <!-- Melt UI Dialog -->
  {#if $open}
    <div use:overlay class="space-y-4" transitionfade={{ duration: 150 }}>
      <div use:content class="space-y-4">
        <h3 use:dialogTitle class="space-y-4">
          Case Management System
        </h3>
        <p use:description class="space-y-4">
          This is a demo of Melt UI headless components integrated with PicoCSS styling and custom CSS variables.
        </p>
        
        <div class="space-y-4">
          <button use:close class="space-y-4">
            Cancel
          </button>
          <button class="space-y-4">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .headless-demo {
    max-width: 500px;
    margin: 0 auto;
    padding: var(--spacing-lg);
}
  .select-menu {
    position: absolute
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
    cursor: pointer
    transition: background-color var(--transition-fast);
}
  .select-option:hover {
    background-color: var(--color-surface);
}
  .dialog-overlay {
    position: fixed
    inset: 0;
    z-index: 50;
    background-color: rgb(0 0 0 / 0.5);
    display: flex
    align-items: center
    justify-content: center
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
    overflow-y: auto
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
    display: flex
    gap: var(--spacing-sm);
    justify-content: flex-end;
}
</style>
