<script lang="ts">
  import HeadlessDemo from "$lib/components/HeadlessDemo.svelte";
  // $state is provided globally via src/types/svelte-helpers.d.ts
  import BitsDemo from "$lib/components/BitsDemo.svelte";
  import { onMount } from 'svelte';

  let mounted = $state(false);

  onMount(() => {
    mounted = true;
  });
</script>

<svelte:head>
  <title>Headless UI Components Demo - Legal Case Management</title>
  <meta name="description" content="Demo of , UnoCSS, Melt UI, and Bits UI integration" />
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <header class="bg-blue-600 text-white py-8">
    <div class="space-y-4">
      <h1 class="text-3xl font-bold mb-2">
        🏛️ Legal Case Management System
      </h1>
      <p class="text-blue-100 mb-6">
        Demonstrating  + UnoCSS + Melt UI + Bits UI Integration
      </p>

      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="flex items-center gap-2 bg-blue-700 rounded p-3">
          <span class="text-2xl">🎨</span>
          <span></span>
        </div>
        <div class="flex items-center gap-2 bg-blue-700 rounded p-3">
          <span class="text-2xl">⚡</span>
          <span>UnoCSS</span>
        </div>
        <div class="flex items-center gap-2 bg-blue-700 rounded p-3">
          <span class="text-2xl">🧩</span>
          <span>Melt UI</span>
        </div>
        <div class="flex items-center gap-2 bg-blue-700 rounded p-3">
          <span class="text-2xl">🔧</span>
          <span>Bits UI</span>
        </div>
      </div>
    </div>
  </header>

  {#if mounted}
    <main class="py-8">
      <div class="mx-auto px-4 max-w-7xl space-y-8">

        <!-- Styling Examples -->
        <section class="bg-white rounded-lg shadow-md p-6">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">📋  + Custom Styling</h2>
          <div class="bg-gray-50 rounded p-4">
            <div class="mb-4">
              <h3 class="text-lg font-semibold">Case Information</h3>
            </div>
            <div class="grid md:grid-cols-2 gap-4">
              <div class="space-y-2">
                <label class="block text-sm font-medium text-gray-700" for="field-1">Case Number</label>
                <input type="text" class="border rounded px-3 py-2 w-full" value="CAS-2025-001234" readonly id="field-1" />
              </div>

              <div class="space-y-2">
                <label class="block text-sm font-medium text-gray-700" for="field-2">Case Status</label>
                <select class="border rounded px-3 py-2 w-full" id="field-2">
                  <option>Active Investigation</option>
                  <option>Pending Review</option>
                  <option>Closed</option>
                </select>
              </div>

              <div class="space-y-2 md:col-span-2">
                <label class="block text-sm font-medium text-gray-700" for="field-3">Priority Level</label>
                <div class="flex gap-2">
                  <div class="px-3 py-1 bg-red-100 text-red-800 rounded text-sm">High Priority Case</div>
                  <div class="px-3 py-1 bg-yellow-100 text-yellow-800 rounded text-sm">Medium Priority Case</div>
                  <div class="px-3 py-1 bg-green-100 text-green-800 rounded text-sm">Low Priority Case</div>
                </div>
              </div>
            </div>
            <div class="flex gap-2 mt-4">
              <button class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Save Changes</button>
              <button class="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400">Cancel</button>
            </div>
          </div>
        </section>

        <!-- UnoCSS Utilities -->
        <section class="bg-white rounded-lg shadow-md p-6">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">⚡ UnoCSS Utilities</h2>
          <div class="grid gap-4">
            <div class="grid md:grid-cols-3 gap-4">
              <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
                <div class="flex justify-between items-start mb-2">
                  <h4 class="font-semibold text-gray-800">State vs. Johnson</h4>
                  <span class="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Active</span>
                </div>
                <p class="text-gray-600 text-sm mb-3">Criminal case involving financial fraud</p>
                <div class="flex gap-2">
                  <button class="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">View Details</button>
                  <button class="px-3 py-1 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300">Edit</button>
                </div>
              </div>

              <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
                <div class="flex justify-between items-start mb-2">
                  <h4 class="font-semibold text-gray-800">Miller Corp Dispute</h4>
                  <span class="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs">Pending</span>
                </div>
                <p class="text-gray-600 text-sm mb-3">Contract dispute resolution</p>
                <div class="flex gap-2">
                  <button class="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">View Details</button>
                  <button class="px-3 py-1 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300">Edit</button>
                </div>
              </div>

              <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
                <div class="flex justify-between items-start mb-2">
                  <h4 class="font-semibold text-gray-800">Estate Settlement</h4>
                  <span class="px-2 py-1 bg-gray-100 text-gray-800 rounded text-xs">Closed</span>
                </div>
                <p class="text-gray-600 text-sm mb-3">Probate and estate administration</p>
                <div class="flex gap-2">
                  <button class="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700">View Archive</button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Melt UI Demo -->
        <section class="bg-white rounded-lg shadow-md p-6">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">🧩 Melt UI Headless Components</h2>
          <div class="border rounded p-4">
            <div class="text-center text-gray-600">
              <HeadlessDemo />
            </div>
          </div>
        </section>

        <!-- Bits UI Demo -->
        <section class="bg-white rounded-lg shadow-md p-6">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">🔧 Bits UI Components</h2>
          <div class="border rounded p-4">
            <div class="text-center text-gray-600">
              <BitsDemo />
            </div>
          </div>
        </section>

        <!-- Alert Examples -->
        <section class="bg-white rounded-lg shadow-md p-6">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">🚨 Alert System</h2>
          <div class="space-y-4">
            <div class="border-l-4 border-green-500 bg-green-50 p-4">
              <div class="flex">
                <div class="ml-3">
                  <strong class="text-green-800">Success!</strong>
                  <span class="text-green-700"> Case has been successfully saved and is now active.</span>
                </div>
              </div>
            </div>

            <div class="border-l-4 border-yellow-500 bg-yellow-50 p-4">
              <div class="flex">
                <div class="ml-3">
                  <strong class="text-yellow-800">Warning!</strong>
                  <span class="text-yellow-700"> This case has a pending deadline in 3 days.</span>
                </div>
              </div>
            </div>

            <div class="border-l-4 border-red-500 bg-red-50 p-4">
              <div class="flex">
                <div class="ml-3">
                  <strong class="text-red-800">Error!</strong>
                  <span class="text-red-700"> Unable to upload evidence file. Please try again.</span>
                </div>
              </div>
            </div>

            <div class="border-l-4 border-blue-500 bg-blue-50 p-4">
              <div class="flex">
                <div class="ml-3">
                  <strong class="text-blue-800">Info:</strong>
                  <span class="text-blue-700"> New evidence has been added to this case by another team member.</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- File Upload Demo -->
        <section class="bg-white rounded-lg shadow-md p-6">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">📎 Evidence Upload</h2>
          <div class="border-2 border-dashed border-gray-300 rounded-lg p-8">
            <div class="text-center">
              <div class="mx-auto mb-4">
                <span class="text-6xl">📄</span>
                <h3 class="text-lg font-medium text-gray-900 mb-2">Drop evidence files here</h3>
                <p class="text-gray-600 mb-2">
                  Or click to browse files
                </p>
                <p class="text-sm text-gray-500">
                  Supported: PDF, DOC, IMG, VIDEO
                </p>
              </div>
            </div>

            <div class="mt-6">
              <div class="bg-gray-200 rounded-full h-2">
                <div class="bg-blue-600 h-2 rounded-full" style="width: 75%"></div>
              </div>
              <p class="text-sm text-gray-600 mt-1">Uploading evidence_report.pdf... 75%</p>
            </div>
          </div>
        </section>
      </div>
    </main>
  {:else}
    <div class="flex items-center justify-center min-h-screen">
      <div class="text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p class="text-gray-600">Loading UI components...</p>
      </div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  /* Additional custom styles if needed */
  .transition-shadow {
    transition: box-shadow 0.2s ease-in-out;
}
</style>
