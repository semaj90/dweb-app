<script lang="ts">
  import CaseManagerXState from '$lib/components/legal/CaseManagerXState.svelte';
</script>

<svelte:head>
  <title>XState Legal Case Machine Test</title>
  <meta name="description" content="Testing hardened XState machine with Svelte 5 and SSR compatibility" />
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <div class="md:flex md:items-center md:justify-between">
        <div class="flex-1 min-w-0">
          <h1 class="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            XState Legal Case Machine Test
          </h1>
          <p class="mt-1 text-sm text-gray-500">
            Testing hardened XState machine with proper SSR/hydration patterns
          </p>
        </div>
        <div class="mt-4 flex md:mt-0 md:ml-4">
          <div class="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
            <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
            XState Active
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <main>
    <CaseManagerXState />
  </main>

  <!-- Footer -->
  <footer class="bg-white border-t border-gray-200 mt-12">
    <div class="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
      <div class="text-center text-sm text-gray-500">
        <p>
          XState Machine: ✅ Hardened • 
          Svelte 5: ✅ Compatible • 
          SSR: ✅ Safe • 
          Event Handlers: ✅ Fixed
        </p>
      </div>
    </div>
  </footer>
</div>