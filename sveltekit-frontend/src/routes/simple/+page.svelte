<!-- Simple test homepage to verify routing works -->
<script lang="ts">
  import { goto } from '$app/navigation';
</script>

<div class="min-h-screen bg-[#EAE8E1] font-mono text-[#3D3D3D] p-8">
  <h1 class="text-3xl font-bold mb-8">Simple Navigation Test</h1>
  
  <div class="space-y-4">
    <h2 class="text-xl font-semibold">Test these navigation links:</h2>
    
    <!-- Direct links (should work) -->
    <div class="space-x-4">
      <a href="/yorha-command-center" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">YoRHa Command Center (Link)</a>
      <a href="/evidenceboard" class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">Evidence Board (Link)</a>
      <a href="/endpoints" class="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600">Endpoints (Link)</a>
      <a href="/demo/enhanced-rag-semantic" class="bg-orange-500 text-white px-4 py-2 rounded hover:bg-orange-600">Enhanced RAG Demo (Link)</a>
    </div>
    
    <!-- Programmatic navigation buttons -->
    <div class="space-x-4 mt-6">
      <button 
        onclick={() => goto('/yorha-command-center')} 
        class="bg-blue-700 text-white px-4 py-2 rounded hover:bg-blue-800"
      >
        YoRHa Command Center (Button)
      </button>
      <button 
        onclick={() => goto('/evidenceboard')} 
        class="bg-green-700 text-white px-4 py-2 rounded hover:bg-green-800"
      >
        Evidence Board (Button)
      </button>
      <button 
        onclick={() => goto('/endpoints')} 
        class="bg-purple-700 text-white px-4 py-2 rounded hover:bg-purple-800"
      >
        Endpoints (Button)
      </button>
    </div>

    <div class="mt-8 p-4 bg-gray-100 rounded">
      <h3 class="font-semibold">Debug Info:</h3>
      <p>Current URL: {typeof window !== 'undefined' ? window.location.href : 'SSR - Not available'}</p>
      <p>Time: {new Date().toLocaleString()}</p>
    </div>
  </div>
</div>