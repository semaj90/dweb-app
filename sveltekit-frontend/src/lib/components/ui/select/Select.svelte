<script lang="ts">
  // Basic Select wrapper that uses enhanced-bits Select
  import { Select as EnhancedSelect } from '../enhanced-bits/Select.svelte';
  
  let { 
    value = $bindable(),
    options = [],
    placeholder = 'Select...',
    ...props 
  } = $props();
</script>

<EnhancedSelect {value} {options} {placeholder} {...props} />