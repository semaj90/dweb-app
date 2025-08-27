<!-- Re-export from textarea subdirectory -->
<script lang="ts">
  import Textarea from './textarea/Textarea.svelte';
  
  // Re-export all props
  let { ...props } = $props();
</script>

<Textarea {...props} />