<!--
  Simple Register Page - Svelte 5 Compatible
  Basic registration without complex UI dependencies
-->
<script lang="ts">
  import { enhance } from '$app/forms';
  
  interface Props {
    data?: any;
    form?: any;
  }
  
  let { data, form }: Props = $props();
  
  let isLoading = $state(false);
  let showPassword = $state(false);
</script>

<svelte:head>
  <title>Register - Legal AI Platform</title>
</svelte:head>

<div class="min-h-screen flex items-center justify-center bg-gray-900 px-4 py-8">
  <div class="w-full max-w-lg">
    <div class="bg-gray-800 p-8 rounded-lg border border-gray-700">
      <h1 class="text-3xl font-bold text-center text-yellow-400 mb-8">
        Legal AI Platform
      </h1>
      
      <h2 class="text-xl text-center text-white mb-6">
        Create Account
      </h2>
      
      {#if form?.error}
        <div class="error-message bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4" data-testid="error-message">
          {form.error}
        </div>
      {/if}

      <form 
        method="POST" 
        action="?/register"
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
        <!-- Personal Information -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="firstName" class="block text-sm font-medium text-gray-300 mb-1">
              First Name
            </label>
            <input
              type="text"
              name="firstName"
              id="firstName"
              required
              disabled={isLoading}
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
              placeholder="John"
            />
          </div>

          <div>
            <label for="lastName" class="block text-sm font-medium text-gray-300 mb-1">
              Last Name
            </label>
            <input
              type="text"
              name="lastName"
              id="lastName"
              required
              disabled={isLoading}
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
              placeholder="Smith"
            />
          </div>
        </div>

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
            placeholder="john.smith@prosecutor.gov"
          />
        </div>

        <!-- Professional Information -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="role" class="block text-sm font-medium text-gray-300 mb-1">
              Role
            </label>
            <select
              name="role"
              id="role"
              required
              disabled={isLoading}
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
            >
              <option value="prosecutor">Prosecutor</option>
              <option value="investigator">Investigator</option>
              <option value="analyst">Legal Analyst</option>
              <option value="admin">Administrator</option>
            </select>
          </div>

          <div>
            <label for="badgeNumber" class="block text-sm font-medium text-gray-300 mb-1">
              Badge/ID (Optional)
            </label>
            <input
              type="text"
              name="badgeNumber"
              id="badgeNumber"
              disabled={isLoading}
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
              placeholder="12345"
            />
          </div>
        </div>

        <!-- Department & Jurisdiction -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="department" class="block text-sm font-medium text-gray-300 mb-1">
              Department
            </label>
            <input
              type="text"
              name="department"
              id="department"
              required
              disabled={isLoading}
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
              placeholder="District Attorney's Office"
            />
          </div>

          <div>
            <label for="jurisdiction" class="block text-sm font-medium text-gray-300 mb-1">
              Jurisdiction
            </label>
            <input
              type="text"
              name="jurisdiction"
              id="jurisdiction"
              required
              disabled={isLoading}
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
              placeholder="Los Angeles County"
            />
          </div>
        </div>

        <!-- Password Fields -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              placeholder="Enter secure password"
            />
          </div>

          <div>
            <label for="confirmPassword" class="block text-sm font-medium text-gray-300 mb-1">
              Confirm Password
            </label>
            <input
              type={showPassword ? 'text' : 'password'}
              name="confirmPassword"
              id="confirmPassword"
              required
              disabled={isLoading}
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400"
              placeholder="Confirm your password"
            />
          </div>
        </div>

        <!-- Show password toggle -->
        <div class="flex items-center">
          <input
            type="checkbox"
            id="showPassword"
            bind:checked={showPassword}
            disabled={isLoading}
            class="mr-2"
          />
          <label for="showPassword" class="text-sm text-gray-300">
            Show passwords
          </label>
        </div>

        <!-- Terms -->
        <div class="space-y-2">
          <div class="flex items-center">
            <input
              type="checkbox"
              name="agreeToTerms"
              id="agreeToTerms"
              required
              disabled={isLoading}
              class="mr-2"
            />
            <label for="agreeToTerms" class="text-sm text-gray-300">
              I agree to the Terms of Service
            </label>
          </div>

          <div class="flex items-center">
            <input
              type="checkbox"
              name="agreeToPrivacy"
              id="agreeToPrivacy"
              required
              disabled={isLoading}
              class="mr-2"
            />
            <label for="agreeToPrivacy" class="text-sm text-gray-300">
              I agree to the Privacy Policy
            </label>
          </div>
        </div>

        <!-- Submit Button -->
        <button
          type="submit"
          disabled={isLoading}
          class="w-full bg-yellow-500 hover:bg-yellow-600 text-black font-semibold py-2 px-4 rounded transition-colors disabled:opacity-50"
        >
          {#if isLoading}
            Creating Account...
          {:else}
            Create Account
          {/if}
        </button>
      </form>

      <!-- Login Link -->
      <div class="mt-6 text-center">
        <p class="text-gray-400 text-sm">
          Already have an account?
          <a href="/auth/login" class="text-yellow-400 hover:underline">Sign in here</a>
        </p>
      </div>
    </div>
  </div>
</div>