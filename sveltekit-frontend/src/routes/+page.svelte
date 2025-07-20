<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  
  onMount(() => {
    // Redirect to the detective mode page
    goto('/detective', { replaceState: true });
  });
</script>

<svelte:head>
  <title>Detective Mode - Legal Case Management</title>
  <meta name="description" content="Google Slides-like evidence management for prosecutors using UnoCSS, Melt UI, and Bits UI" />
</svelte:head>

<!-- Loading screen while redirecting -->
<div class="container mx-auto px-4">
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4"></div>
    <p class="container mx-auto px-4">Loading Detective Mode...</p>
    <p class="container mx-auto px-4">Redirecting to the evidence board...</p>
  </div>
</div>