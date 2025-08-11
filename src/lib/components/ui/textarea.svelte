<script>
  export let value = '';
  export let placeholder = '';
  export let disabled = false;
  export let readonly = false;
  export let className = '';
  export let id = '';
  export let name = '';
  export let required = false;
  export let rows = 4;
  export let cols = undefined;
  export let maxlength = undefined;
  export let minlength = undefined;
  export let wrap = undefined;
  export let resize = true;
  
  function handleInput(event) {
    value = event.target.value;
  }
  
  $: resizeClass = resize ? '' : 'resize-none';
</script>

<textarea
  {id}
  {name}
  {placeholder}
  {disabled}
  {readonly}
  {required}
  {rows}
  {cols}
  {maxlength}
  {minlength}
  {wrap}
  {value}
  on:input={handleInput}
  on:change
  on:focus
  on:blur
  on:keydown
  on:keyup
  on:keypress
  class="yorha-textarea {resizeClass} {className}"
></textarea>

<style>
  .yorha-textarea {
    width: 100%;
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 2px solid var(--yorha-text-muted, #808080);
    color: var(--yorha-text-primary, #e0e0e0);
    font-family: var(--yorha-font-primary, 'JetBrains Mono', monospace);
    font-size: 14px;
    padding: 12px 16px;
    border-radius: 0;
    transition: all 0.2s ease;
    line-height: 1.4;
    
    /* Cross-browser appearance normalization */
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
  }

  .yorha-textarea:focus {
    outline: none;
    border-color: var(--yorha-secondary, #ffd700);
    box-shadow: 
      0 0 0 1px var(--yorha-secondary, #ffd700),
      inset 0 0 10px rgba(255, 215, 0, 0.1);
  }

  .yorha-textarea::placeholder {
    color: var(--yorha-text-muted, #808080);
  }

  .yorha-textarea:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: var(--yorha-bg-secondary, #1a1a1a);
  }

  .yorha-textarea:read-only {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border-color: var(--yorha-text-muted, #808080);
  }

  /* Scrollbar styling for textarea */
  .yorha-textarea::-webkit-scrollbar {
    width: 8px;
  }

  .yorha-textarea::-webkit-scrollbar-track {
    background: var(--yorha-bg-primary, #0a0a0a);
  }

  .yorha-textarea::-webkit-scrollbar-thumb {
    background: var(--yorha-secondary, #ffd700);
    border: 1px solid var(--yorha-bg-primary, #0a0a0a);
  }

  .yorha-textarea::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 215, 0, 0.8);
  }
</style>