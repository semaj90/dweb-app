
<script lang="ts">
  import { page } from '$app/state';
  import { dev } from '$app/environment';
  
  let error = $derived(page.error);
  let status = $derived(page.status);
</script>

<svelte:head>
  <title>Error {status}</title>
</svelte:head>

<div class="min-h-screen flex items-center justify-center bg-gray-50">
  <div class="max-w-md w-full space-y-8 text-center">
    <div class="space-y-4">
      <h1 class="text-4xl font-bold text-gray-900">{status}</h1>
      <h2 class="text-2xl font-semibold text-gray-700">
          {#if status === 404}
            Page not found
          {:else if status === 500}
            Internal server error
          {:else}
            Something went wrong
          {/if}
      </h2>
      
      {#if error?.message}
        <p class="text-gray-600">{error.message}</p>
      {/if}
      
      {#if dev && error}
        <details class="mt-6 text-left">
          <summary class="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
            Error Details (Development)
          </summary>
          <pre class="mt-2 p-4 bg-gray-100 rounded-lg text-xs text-gray-800 overflow-auto">
            {JSON.stringify(error, null, 2)}
          </pre>
        </details>
      {/if}
      
      <div class="mt-8">
        <a 
          href="/" 
          class="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Go Home
        </a>
      </div>
    </div>
  </div>
</div>
