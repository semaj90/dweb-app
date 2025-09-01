<script lang="ts">
  import Context7TestDemo from '$lib/components/dev/Context7TestDemo.svelte';
</script>

<svelte:head>
  <title>Context7 Integration Test Suite - Legal AI Development</title>
  <meta name="description" content="Test suite for Context7 semantic search, agent orchestration, and MCP tools integration" />
</svelte:head>

<main class="min-h-screen bg-gray-50">
  <div class="bg-gradient-to-r from-green-600 to-blue-600 text-white py-8">
    <div class="max-w-6xl mx-auto px-6">
      <h1 class="text-3xl font-bold mb-4">
        Context7 Integration Test Suite
      </h1>
      <p class="text-xl text-green-100 max-w-4xl">
        Comprehensive testing environment for Context7 semantic search, agent orchestration, 
        MCP tools integration, and real-time workflow automation.
      </p>
      
      <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-green-700 bg-opacity-50 rounded-lg p-4">
          <h3 class="font-semibold mb-2">🔍 Semantic Search</h3>
          <p class="text-sm text-green-100">Real Context7 MCP semantic search testing</p>
        </div>
        <div class="bg-blue-700 bg-opacity-50 rounded-lg p-4">
          <h3 class="font-semibold mb-2">🤖 Agent Orchestration</h3>
          <p class="text-sm text-blue-100">Multi-agent coordination and triggers</p>
        </div>
        <div class="bg-teal-700 bg-opacity-50 rounded-lg p-4">
          <h3 class="font-semibold mb-2">📊 Audit Logging</h3>
          <p class="text-sm text-teal-100">Real-time audit results and logging</p>
        </div>
      </div>
    </div>
  </div>

  <Context7TestDemo />

  <footer class="bg-white border-t border-gray-200 py-6">
    <div class="max-w-6xl mx-auto px-6 text-center">
      <p class="text-sm text-gray-500">
        Context7 MCP Integration Test Suite - Validate real semantic search, agent orchestration, and workflow automation.
      </p>
    </div>
  </footer>
</main>