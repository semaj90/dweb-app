<script lang="ts">
  import '../app.css';
  import '../lib/styles/nier.css';
  import 'shadcn-svelte/styles.css';
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  
  let mounted = false;
  onMount(() => { mounted = true; });
</script>

{#if mounted}
  <div class="app-layout">
    <main class="main-content">
      <slot />
    </main>
  </div>
{/if}

<style>
  .app-layout {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  
  .main-content {
    flex: 1;
    width: 100%;
    max-width: none;
  }
</style>
