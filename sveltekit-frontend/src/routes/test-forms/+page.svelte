<!--
  Simple SuperForms + Zod + Enhanced Actions Test Page
-->
<script lang="ts">
  import { enhance } from '$app/forms';
  import { superForm } from 'sveltekit-superforms/client';
  import { zod } from 'sveltekit-superforms/adapters';
  import { z } from 'zod';

  // Simple test schema
  const testSchema = z.object({
    name: z.string().min(1, 'Name is required'),
    email: z.string().email('Invalid email'),
    message: z.string().min(10, 'Message must be at least 10 characters')
  });

  // Mock form data
  const data = {
    form: {
      name: '',
      email: '',
      message: ''
    }
  };

  // SuperForm setup
  const { form, errors, enhance: formEnhance, submitting } = superForm(data, {
    validators: zod(testSchema),
    onSubmit: ({ formData }) => {
      console.log('Form submitted:', Object.fromEntries(formData));
    }
  });

  // Enhanced action
  function createEnhancedAction() {
    return enhance(({ formData }) => {
      console.log('Enhanced action:', Object.fromEntries(formData));
      
      return async ({ result, update }) => {
        console.log('Result:', result);
        await update();
      };
    });
  }
</script>

<svelte:head>
  <title>SuperForms + Zod Test</title>
</svelte:head>

<div class="max-w-2xl mx-auto p-6">
  <h1 class="text-3xl font-bold mb-6 text-white">SuperForms + Zod + Enhanced Actions Test</h1>

  <div class="bg-gray-800 p-6 rounded-lg border border-gray-700">
    <h2 class="text-xl font-semibold mb-4 text-yellow-400">Contact Form</h2>
    
    <form 
      method="POST" 
      action="?/submit"
      use:createEnhancedAction()
      class="space-y-4"
    >
      <!-- Name Field -->
      <div>
        <label for="name" class="block text-sm font-medium text-gray-300 mb-1">
          Name
        </label>
        <input
          id="name"
          name="name"
          type="text"
          bind:value={$form.name}
          class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-transparent"
          class:border-red-500={$errors.name}
        />
        {#if $errors.name}
          <p class="text-red-400 text-sm mt-1">{$errors.name[0]}</p>
        {/if}
      </div>

      <!-- Email Field -->
      <div>
        <label for="email" class="block text-sm font-medium text-gray-300 mb-1">
          Email
        </label>
        <input
          id="email"
          name="email"
          type="email"
          bind:value={$form.email}
          class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-transparent"
          class:border-red-500={$errors.email}
        />
        {#if $errors.email}
          <p class="text-red-400 text-sm mt-1">{$errors.email[0]}</p>
        {/if}
      </div>

      <!-- Message Field -->
      <div>
        <label for="message" class="block text-sm font-medium text-gray-300 mb-1">
          Message
        </label>
        <textarea
          id="message"
          name="message"
          rows="4"
          bind:value={$form.message}
          class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-transparent resize-none"
          class:border-red-500={$errors.message}
        ></textarea>
        {#if $errors.message}
          <p class="text-red-400 text-sm mt-1">{$errors.message[0]}</p>
        {/if}
      </div>

      <!-- Submit Button -->
      <button 
        type="submit" 
        disabled={$submitting}
        class="w-full bg-yellow-400 text-black py-2 px-4 rounded-md font-semibold hover:bg-yellow-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {#if $submitting}
          Submitting...
        {:else}
          Submit Form
        {/if}
      </button>
    </form>

    <!-- Form Data Display -->
    <div class="mt-6 p-4 bg-gray-900 rounded border border-gray-600">
      <h3 class="text-lg font-medium text-green-400 mb-2">Form Data (Live)</h3>
      <pre class="text-sm text-gray-300">{JSON.stringify($form, null, 2)}</pre>
    </div>

    <!-- Errors Display -->
    {#if Object.keys($errors).length > 0}
      <div class="mt-4 p-4 bg-red-900 rounded border border-red-600">
        <h3 class="text-lg font-medium text-red-400 mb-2">Validation Errors</h3>
        <pre class="text-sm text-red-300">{JSON.stringify($errors, null, 2)}</pre>
      </div>
    {/if}
  </div>

  <!-- Documentation -->
  <div class="mt-8 bg-blue-900 p-6 rounded-lg border border-blue-700">
    <h3 class="text-xl font-semibold text-blue-400 mb-4">Implementation Details</h3>
    
    <div class="space-y-3 text-blue-200">
      <div>
        <h4 class="font-medium text-blue-300">✅ SuperForms Integration</h4>
        <p class="text-sm">Real-time validation with Zod schemas and reactive form state</p>
      </div>
      
      <div>
        <h4 class="font-medium text-blue-300">✅ Enhanced Actions</h4>
        <p class="text-sm">Progressive enhancement with custom form submission handling</p>
      </div>
      
      <div>
        <h4 class="font-medium text-blue-300">✅ Zod Validation</h4>
        <p class="text-sm">Type-safe validation with custom error messages</p>
      </div>
      
      <div>
        <h4 class="font-medium text-blue-300">✅ Reactive Updates</h4>
        <p class="text-sm">Live form data display and error handling</p>
      </div>
    </div>
  </div>
</div>