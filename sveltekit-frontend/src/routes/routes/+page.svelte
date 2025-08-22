<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  onMount(() => {
    // client-side fallback if server redirect isn't available in dev
    goto('/all-routes');
  });
</script>

<svelte:head>
  <title>Redirecting...</title>
</svelte:head>

<div style="padding:3rem;text-align:center;">
  <h2>Redirecting to All Routes...</h2>
  <p>If you are not redirected automatically, <a href="/all-routes">click here</a>.</p>
</div>
