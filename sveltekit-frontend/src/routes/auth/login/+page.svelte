<script lang="ts">
  import { $props } from 'svelte';
  let { data } = $props();
  let { form } = $props();
</script>

<svelte:head>
  <title>Login - Legal AI Platform</title>
</svelte:head>

<div class="min-h-screen flex items-center justify-center bg-gray-900 px-4">
  <div class="w-full max-w-md">
    <div class="bg-gray-800 p-8 rounded-lg border border-gray-700">
      <h1 class="text-3xl font-bold text-center text-yellow-400 mb-8">
        Legal AI Platform
      </h1>
      
      {#if form?.error}
        <div class="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4">
          {form.error}
        </div>
      {/if}

      <form method="POST" class="space-y-6">
        <div>
          <label for="email" class="block text-sm font-medium text-gray-300 mb-2">
            Email
          </label>
          <input
            type="email"
            name="email"
            id="email"
            required
            class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
            placeholder="Enter your email"
          />
        </div>

        <div>
          <label for="password" class="block text-sm font-medium text-gray-300 mb-2">
            Password
          </label>
          <input
            type="password"
            name="password"
            id="password"
            required
            class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
            placeholder="Enter your password"
          />
        </div>

        <button
          type="submit"
          class="w-full bg-yellow-500 hover:bg-yellow-600 text-black font-semibold py-2 px-4 rounded transition-colors"
        >
          Sign In
        </button>
      </form>

      <div class="mt-6 text-center">
        <p class="text-gray-400 text-sm">
          Demo: Use any email and password to login
        </p>
      </div>
    </div>
  </div>
</div>