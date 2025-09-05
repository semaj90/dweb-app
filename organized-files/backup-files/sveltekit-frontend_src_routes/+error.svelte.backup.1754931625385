
<script lang="ts">
  import { page } from '$app/stores';
  import { dev } from '$app/environment';
  
  $: error = $page.error;
  $: status = $page.status;
</script>

<svelte:head>
  <title>Error {status}</title>
</svelte:head>

<div class="container mx-auto px-4">
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <h1 class="container mx-auto px-4">{status}</h1>
        <h2 class="container mx-auto px-4">
          {#if status === 404}
            Page not found
          {:else if status === 500}
            Internal server error
          {:else}
            Something went wrong
          {/if}
        </h2>
        
        {#if error?.message}
          <p class="container mx-auto px-4">{error.message}</p>
        {/if}
        
        {#if dev && error}
          <details class="container mx-auto px-4">
            <summary class="container mx-auto px-4">
              Error Details (Development)
            </summary>
            <pre class="container mx-auto px-4">
              {JSON.stringify(error, null, 2)}
            </pre>
          </details>
        {/if}
        
        <div class="container mx-auto px-4">
          <a 
            href="/" 
            class="container mx-auto px-4"
          >
            Go Home
          </a>
        </div>
      </div>
    </div>
  </div>
</div>
