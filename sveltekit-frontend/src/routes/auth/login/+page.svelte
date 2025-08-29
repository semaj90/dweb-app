<!--
  Simple Login Page - Works with Existing Database
  Now with NES.css Retro Gaming Modal Option
-->
<script lang="ts">
  import { enhance } from '$app/forms';
  import NesAuthModal from '$lib/components/auth/NesAuthModal.svelte';
  
  interface Props {
    data?: any;
    form?: any;
  }
  
  let { data, form }: Props = $props();
  
  let isLoading = $state(false);
  let showPassword = $state(false);
  let isNesModalOpen = $state(false);
  
  // Auto-fill demo credentials
  function fillDemoCredentials() {
    const emailInput = document.getElementById('email') as HTMLInputElement;
    const passwordInput = document.getElementById('password') as HTMLInputElement;
    
    if (emailInput && passwordInput) {
      emailInput.value = 'admin@legal-ai.local';
      passwordInput.value = 'admin123';
    }
  }
  
  function openNesModal() {
    isNesModalOpen = true;
  }
</script>

<svelte:head>
  <title>Login - Legal AI Platform</title>
</svelte:head>

<div class="min-h-screen flex items-center justify-center bg-gray-900 px-4 py-8">
  <div class="w-full max-w-md">
    <div class="bg-gray-800 p-8 rounded-lg border border-gray-700 max-h-none overflow-visible">
      <h1 class="text-3xl font-bold text-center text-yellow-400 mb-8">
        Legal AI Platform
      </h1>
      
      <h2 class="text-xl text-center text-white mb-6">
        Sign In
      </h2>
      
      {#if form?.error}
        <div class="error-message bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4">
          {form.error}
        </div>
      {/if}

      <form 
        method="POST" 
        action="?/login"
        use:enhance={({ formData, cancel }) => {
          isLoading = true;
          return async ({ result }) => {
            isLoading = false;
            if (result.type === 'redirect') {
              // Let SvelteKit handle the redirect
            }
          };
        }}
        class="space-y-4"
      >
        <!-- Email -->
        <div>
          <label for="email" class="block text-sm font-medium text-gray-300 mb-1">
            Email Address
          </label>
          <input
            type="email"
            name="email"
            id="email"
            required
            disabled={isLoading}
            class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
            placeholder="admin@legal-ai.local"
          />
        </div>

        <!-- Password -->
        <div>
          <label for="password" class="block text-sm font-medium text-gray-300 mb-1">
            Password
          </label>
          <input
            type={showPassword ? 'text' : 'password'}
            name="password"
            id="password"
            required
            disabled={isLoading}
            class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
            placeholder="Enter your password"
          />
        </div>

        <!-- Options -->
        <div class="flex items-center justify-between">
          <div class="flex items-center">
            <input
              type="checkbox"
              id="showPassword"
              bind:checked={showPassword}
              disabled={isLoading}
              class="mr-2"
            />
            <label for="showPassword" class="text-sm text-gray-300">
              Show password
            </label>
          </div>
          
          <div class="flex items-center">
            <input
              type="checkbox"
              name="rememberMe"
              id="rememberMe"
              disabled={isLoading}
              class="mr-2"
            />
            <label for="rememberMe" class="text-sm text-gray-300">
              Remember me
            </label>
          </div>
        </div>

        <!-- Demo Button -->
        <button
          type="button"
          onclick={fillDemoCredentials}
          disabled={isLoading}
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
        >
          Fill Demo Credentials
        </button>

        <!-- Submit Button -->
        <button
          type="submit"
          disabled={isLoading}
          class="w-full bg-yellow-500 hover:bg-yellow-600 text-black font-semibold py-2 px-4 rounded transition-colors disabled:opacity-50"
        >
          {#if isLoading}
            Signing In...
          {:else}
            Sign In
          {/if}
        </button>
      </form>

      <!-- NES Modal Option - Made more prominent -->
      <div class="mt-6 text-center">
        <div class="mb-2">
          <p class="text-sm text-gray-400">Or try our retro gaming experience:</p>
        </div>
        <button
          type="button"
          onclick={openNesModal}
          class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded-lg transition-colors transform hover:scale-105 border-2 border-purple-400"
        >
          🎮 Retro Gaming Login (NES Style)
        </button>
      </div>

      <!-- Register Link -->
      <div class="mt-6 text-center">
        <p class="text-gray-400 text-sm">
          Don't have an account?
          <a href="/auth/register" class="text-yellow-400 hover:underline">Create one here</a>
        </p>
      </div>
      
      <!-- Quick Access -->
      <div class="mt-4 text-center">
        <p class="text-gray-500 text-xs">
          Demo: admin@legal-ai.local / admin123
        </p>
      </div>
    </div>
  </div>
</div>

<!-- NES Auth Modal -->
<NesAuthModal 
  bind:isOpen={isNesModalOpen}
  {form}
/>