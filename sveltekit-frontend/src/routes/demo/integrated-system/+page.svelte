<script lang="ts">
  import IntegratedSystemDemo from '$lib/components/demo/IntegratedSystemDemo.svelte';
</script>

<svelte:head>
  <title>Integrated System Demo - Legal AI Platform</title>
  <meta name="description" content="Complete demonstration of XState machines, authentication, AI assistant, and production services integration" />
</svelte:head>

<main class="min-h-screen bg-gray-50 py-8">
  <div class="container mx-auto px-4">
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">
        Integrated Legal AI System Demo
      </h1>
      <p class="text-xl text-gray-600 max-w-3xl mx-auto">
        Experience the complete integration of modern authentication, session management, 
        AI assistant with Ollama cluster, and production services using XState machines 
        and Svelte 5 runes.
      </p>
    </div>
    
    <IntegratedSystemDemo />
  </div>
</main>

<style>
  main {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  }
</style>