<!-- Cache System Demo Page -->
<script lang="ts">
  import CacheDemo from '$lib/components/cache/CacheDemo.svelte';
</script>

<svelte:head>
  <title>Multi-Layer Cache System Demo</title>
  <meta name="description" content="Interactive demonstration of the comprehensive caching architecture" />
</svelte:head>

<main class="min-h-screen bg-background">
  <div class="container mx-auto py-8">
    <div class="mb-8 text-center">
      <h1 class="text-3xl font-bold mb-2">
        Multi-Layer Cache System Demo
      </h1>
      <p class="text-muted-foreground text-lg">
        Interactive demonstration of Loki.js + Redis + PostgreSQL caching architecture
      </p>
    </div>

    <CacheDemo />
  </div>
</main>