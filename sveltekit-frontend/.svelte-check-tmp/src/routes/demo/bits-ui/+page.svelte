<!-- Bits-UI Components Demo Page -->
<script lang="ts">
  import BitsUIDemo from '$lib/components/ui/BitsUIDemo.svelte';
  import { page } from '$app/stores';
  
  // Page metadata
  const title = 'Bits-UI Components Demo';
  const description = 'Modern Svelte 5 components with bits-ui primitives for Legal AI Platform';
</script>

<svelte:head>
  <title>{title} - Legal AI Platform</title>
  <meta name="description" content={description} />
</svelte:head>

<!-- Navigation Breadcrumb -->
<nav class="bg-yorha-bg-secondary border-b border-yorha-border px-6 py-3">
  <div class="max-w-6xl mx-auto">
    <div class="flex items-center space-x-2 text-sm font-mono text-yorha-text-secondary">
      <a href="/" class="hover:text-yorha-text-primary transition-colors">Home</a>
      <span>/</span>
      <a href="/demo" class="hover:text-yorha-text-primary transition-colors">Demo</a>
      <span>/</span>
      <span class="text-yorha-text-primary">Bits-UI</span>
    </div>
  </div>
</nav>

<!-- Main Demo Content -->
<BitsUIDemo />

<!-- Developer Info Footer -->
<footer class="bg-yorha-bg-secondary border-t border-yorha-border py-6 mt-8">
  <div class="max-w-6xl mx-auto px-6">
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm font-mono">
      <!-- Development Info -->
      <div class="space-y-2">
        <h3 class="font-semibold text-yorha-text-primary">Development Status</h3>
        <ul class="space-y-1 text-yorha-text-secondary">
          <li>✅ Svelte 5 Migration Complete</li>
          <li>✅ Bits-UI Integration Active</li>
          <li>✅ TypeScript Errors Resolved</li>
          <li>✅ Modern Patterns Implemented</li>
        </ul>
      </div>

      <!-- Component Library -->
      <div class="space-y-2">
        <h3 class="font-semibold text-yorha-text-primary">Available Components</h3>
        <ul class="space-y-1 text-yorha-text-secondary">
          <li>• BitsDialog - Modal component</li>
          <li>• BitsSelect - Dropdown with search</li>
          <li>• BitsInput - Enhanced input fields</li>
          <li>• Button - Svelte 5 compatible</li>
        </ul>
      </div>

      <!-- Usage Info -->
      <div class="space-y-2">
        <h3 class="font-semibold text-yorha-text-primary">Usage</h3>
        <div class="text-yorha-text-secondary">
          <p>Import components from:</p>
          <code class="block bg-yorha-bg-tertiary px-2 py-1 rounded mt-1">
            $lib/components/ui/bits
          </code>
        </div>
      </div>
    </div>
  </div>
</footer>