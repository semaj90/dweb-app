
<script lang="ts">
  import { page } from '$app/stores';
  import { dev } from '$app/environment';
  
  $: error = $page.error;
  $: status = $page.status;
</script>

<svelte:head>
  <title>Error {status}</title>
</svelte:head>

<div className="${1}">
  <div className="${1}">
    <div className="${1}">
      <div className="${1}">
        <h1 className="${1}">{status}</h1>
        <h2 className="${1}">
          {#if status === 404}
            Page not found
          {:else if status === 500}
            Internal server error
          {:else}
            Something went wrong
          {/if}
        </h2>
        
        {#if error?.message}
          <p className="${1}">{error.message}</p>
        {/if}
        
        {#if dev && error}
          <details className="${1}">
            <summary className="${1}">
              Error Details (Development)
            </summary>
            <pre className="${1}">
              {JSON.stringify(error, null, 2)}
            </pre>
          </details>
        {/if}
        
        <div className="${1}">
          <a 
            href="/" 
            className="${1}"
          >
            Go Home
          </a>
        </div>
      </div>
    </div>
  </div>
</div>
