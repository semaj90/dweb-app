<!--
  Simple Enhanced Actions Test Page
  Tests SuperForms + Enhanced Actions pattern
-->
<script lang="ts">
  import { enhance } from '$app/forms';
  import { page } from '$app/stores';
  import type { PageData } from './$types';

  let { data = $bindable() } = $props(); // PageData;
let isSubmitting = $state(false);

  // Enhanced form submission
  function createEnhancedSubmit() {
    return enhance(({ formData, action, cancel }) => {
      isSubmitting = true;
      
      // Add metadata to form submission
      formData.append('metadata', JSON.stringify({
        submitTime: new Date().toISOString(),
        userAgent: navigator.userAgent
      }));

      return async ({ result, update }) => {
        isSubmitting = false;
        
        if (result.type === 'success') {
          console.log('✅ Form submitted successfully:', result.data);
        } else if (result.type === 'failure') {
          console.log('❌ Form submission failed:', result.data);
        }

        await update();
      };
    });
  }
</script>

<svelte:head>
  <title>Enhanced Actions Test - Legal AI Platform</title>
</svelte:head>

<div class="container mx-auto px-4 py-8 max-w-2xl">
  <div class="mb-8">
    <h1 class="text-3xl font-bold tracking-tight mb-2">Enhanced Actions Test</h1>
    <p class="text-gray-600">Testing SuperForms + Zod + Enhanced Actions pattern</p>
  </div>

  <!-- Success Message -->
  {#if $page.form?.success}
    <div class="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
      <h3 class="text-green-800 font-semibold">Success!</h3>
      <p class="text-green-700">{$page.form.message}</p>
    </div>
  {/if}

  <!-- Form Errors -->
  {#if $page.form?.errors && Object.keys($page.form.errors).length > 0}
    <div class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
      <h3 class="text-red-800 font-semibold">Form Errors:</h3>
      {#each Object.entries($page.form.errors) as [field, error]}
        <p class="text-red-700">• {field}: {error}</p>
      {/each}
    </div>
  {/if}

  <!-- Enhanced Form -->
  <form 
    method="POST" 
    action="?/testAction"
    use:createEnhancedSubmit()
    class="space-y-6 bg-white p-6 rounded-lg shadow-md"
  >
    <!-- Title Field -->
    <div>
      <label for="title" class="block text-sm font-medium text-gray-700 mb-2">
        Title *
      </label>
      <input
        id="title"
        name="title"
        type="text"
        value={data.form.data.title}
        placeholder="Enter a title"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        class:border-red-500={$page.form?.errors?.title}
        required
      />
      {#if $page.form?.errors?.title}
        <p class="mt-1 text-sm text-red-600">{$page.form.errors.title}</p>
      {/if}
    </div>

    <!-- Description Field -->
    <div>
      <label for="description" class="block text-sm font-medium text-gray-700 mb-2">
        Description
      </label>
      <textarea
        id="description"
        name="description"
        rows="4"
        value={data.form.data.description}
        placeholder="Enter a description (optional)"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      ></textarea>
    </div>

    <!-- Priority Field -->
    <div>
      <label for="priority" class="block text-sm font-medium text-gray-700 mb-2">
        Priority
      </label>
      <select
        id="priority"
        name="priority"
        value={data.form.data.priority}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      >
        <option value="low">Low Priority</option>
        <option value="medium">Medium Priority</option>
        <option value="high">High Priority</option>
      </select>
    </div>

    <!-- Submit Button -->
    <div class="flex justify-end">
      <button
        type="submit"
        disabled={isSubmitting}
        class="px-6 py-2 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if isSubmitting}
          <span class="flex items-center">
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Submitting...
          </span>
        {:else}
          Test Enhanced Actions
        {/if}
      </button>
    </div>
  </form>

  <!-- Debug Info -->
  <div class="mt-8 p-4 bg-gray-50 rounded-lg">
    <h3 class="text-lg font-semibold mb-3">Debug Information</h3>
    <div class="space-y-2 text-sm">
      <div><strong>Form Data:</strong> {JSON.stringify(data.form.data, null, 2)}</div>
      <div><strong>Page Form:</strong> {JSON.stringify($page.form, null, 2)}</div>
      <div><strong>Is Submitting:</strong> {isSubmitting}</div>
    </div>
  </div>

  <!-- Features List -->
  <div class="mt-8 p-4 bg-blue-50 rounded-lg">
    <h3 class="text-lg font-semibold mb-3 text-blue-800">Enhanced Actions Features</h3>
    <ul class="space-y-1 text-sm text-blue-700">
      <li>✅ Progressive enhancement with JavaScript</li>
      <li>✅ Server-side validation fallback</li>
      <li>✅ Real-time form state management</li>
      <li>✅ Enhanced submission with metadata</li>
      <li>✅ Loading states and error handling</li>
      <li>✅ Form data persistence across requests</li>
    </ul>
  </div>
</div>