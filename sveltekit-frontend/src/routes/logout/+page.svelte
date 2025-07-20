<script lang="ts">
  import { goto, invalidateAll } from '$app/navigation';
  import { onMount } from 'svelte';
  
  onMount(async () => {
    try {
      // Call logout API endpoint
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'}
      });
      
      if (response.ok) {
        // Invalidate all data and redirect to home
        await invalidateAll();
        goto('/');
      } else {
        console.error('Logout failed');
        // Redirect anyway for security
        goto('/');}
    } catch (error) {
      console.error('Logout error:', error);
      // Redirect anyway for security
      goto('/');}
  });
</script>

<svelte:head>
  <title>Logging out... - WardenNet</title>
</svelte:head>

<div class="container mx-auto px-4">
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <h2 class="container mx-auto px-4">Logging out...</h2>
      <p>Please wait while we log you out securely.</p>
      <div class="container mx-auto px-4">
        <span class="container mx-auto px-4"></span>
      </div>
    </div>
  </div>
</div>
