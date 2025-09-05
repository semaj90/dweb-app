
<script lang="ts">
  import { page } from '$app/stores';
  import { dev } from '$app/environment';
  
  let error = $derived($page.error)
  let status = $derived($page.status)
</script>

<svelte:head>
  <title>Error {status}</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <h1 class="space-y-4">{status}</h1>
        <h2 class="space-y-4">
          {#if status === 404}
            Page not found
          {:else if status === 500}
            Internal server error
          {:else}
            Something went wrong
          {/if}
        </h2>
        
        {#if error?.message}
          <p class="space-y-4">{error.message}</p>
        {/if}
        
        {#if dev && error}
          <details class="space-y-4">
            <summary class="space-y-4">
              Error Details (Development)
            </summary>
            <pre class="space-y-4">
              {JSON.stringify(error, null, 2)}
            </pre>
          </details>
        {/if}
        
        <div class="space-y-4">
          <a 
            href="/" 
            class="space-y-4"
          >
            Go Home
          </a>
        </div>
      </div>
    </div>
  </div>
</div>
