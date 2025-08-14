<script lang="ts">
  let message = "SvelteKit 2 + Svelte 5 Test Page";
  let count = $state(0);
  
  function increment() {
    count++;
  }
</script>

<svelte:head>
  <title>Simple Test - Legal AI System</title>
</svelte:head>

<div class="p-8 max-w-4xl mx-auto">
  <h1 class="text-3xl font-bold mb-6">{message}</h1>
  
  <div class="bg-white rounded-lg shadow p-6">
    <h2 class="text-xl font-semibold mb-4">Server Status: ✅ Working</h2>
    
    <div class="space-y-4">
      <p class="text-gray-700">
        This is a simple test page to verify that the SvelteKit server is running correctly
        with Svelte 5 features.
      </p>
      
      <div class="bg-gray-100 p-4 rounded">
        <p><strong>Svelte 5 State Test:</strong></p>
        <p>Count: <span class="font-mono">{count}</span></p>
        <button 
          onclick={increment}
          class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 mt-2"
        >
          Increment
        </button>
      </div>
      
      <div class="bg-green-50 border border-green-200 rounded p-4">
        <h3 class="font-semibold text-green-800">✅ Working Features:</h3>
        <ul class="list-disc list-inside text-green-700 mt-2">
          <li>SvelteKit 2 server running</li>
          <li>Svelte 5 runes ($state)</li>
          <li>TypeScript compilation</li>
          <li>CSS styling</li>
          <li>Component rendering</li>
        </ul>
      </div>
      
      <div class="space-y-2">
        <a href="/" class="text-blue-600 hover:underline block">← Back to Homepage</a>
        <a href="/login" class="text-blue-600 hover:underline block">Login Page</a>
        <a href="/register" class="text-blue-600 hover:underline block">Registration Page</a>
        <a href="/evidence-editor" class="text-blue-600 hover:underline block">Evidence Editor</a>
        <a href="/crud-dashboard" class="text-blue-600 hover:underline block">CRUD Dashboard</a>
      </div>
    </div>
  </div>
</div>

<style>
  /* Test styles - removed unused body selector */
</style>