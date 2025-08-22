<!--
  Unified Authentication Flow - Single Page App UX
  Combines login and register into seamless experience
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';

  // Authentication mode state
  let authMode: 'login' | 'register' = 'login';
  let isLoading = false;
  let error = '';
  let success = '';

  // Form data
  let formData = {
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    role: 'prosecutor',
    department: '',
    jurisdiction: '',
    badgeNumber: '',
    agreeToTerms: false,
    agreeToPrivacy: false
  };

  // Toggle between login and register
  function toggleAuthMode() {
    authMode = authMode === 'login' ? 'register' : 'login';
    error = '';
    success = '';
  }

  // Handle form submission
  async function handleSubmit(event: Event) {
    event.preventDefault();
    isLoading = true;
    error = '';
    success = '';

    try {
      const form = new FormData();

      // Add common fields
      form.append('email', formData.email);
      form.append('password', formData.password);

      // Add register-specific fields
      if (authMode === 'register') {
        form.append('confirmPassword', formData.confirmPassword);
        form.append('firstName', formData.firstName);
        form.append('lastName', formData.lastName);
        form.append('role', formData.role);
        form.append('department', formData.department);
        form.append('jurisdiction', formData.jurisdiction);
        form.append('badgeNumber', formData.badgeNumber);
        form.append('agreeToTerms', formData.agreeToTerms.toString());
        form.append('agreeToPrivacy', formData.agreeToPrivacy.toString());
      }

      const endpoint = authMode === 'login' ? '/auth/login' : '/auth/register';

      const response = await fetch(endpoint, {
        method: 'POST',
        body: form
      });

      if (response.ok) {
        success = authMode === 'login' ? 'Login successful! Redirecting...' : 'Registration successful! Redirecting...';
        // Redirect on success
        setTimeout(() => {
          window.location.href = '/dashboard';
        }, 1500);
      } else {
        const result = await response.json();
        error = result.error || `${authMode === 'login' ? 'Login' : 'Registration'} failed`;
      }
    } catch (err) {
      error = 'Network error. Please try again.';
    } finally {
      isLoading = false;
    }
  }

  // Validate form
  let isFormValid = false;

  // Use a reactive statement to compute form validity instead of $derived
  $: isFormValid = (authMode === 'login')
    ? Boolean(formData.email && formData.password)
    : Boolean(
        formData.email && formData.password && formData.confirmPassword &&
        formData.firstName && formData.lastName && formData.department &&
        formData.jurisdiction && formData.agreeToTerms && formData.agreeToPrivacy &&
        formData.password === formData.confirmPassword
      );
</script>

<svelte:head>
  <title>{authMode === 'login' ? 'Login' : 'Register'} - Legal AI Platform</title>
  <meta name="description" content="Access the Legal AI Platform - Unified authentication experience" />
</svelte:head>

<div class="min-h-screen flex items-center justify-center bg-gray-900 px-4 py-8">
  <div class="w-full max-w-md">
    <div class="bg-gray-800 p-8 rounded-lg border border-gray-700 shadow-2xl">
      <!-- Header -->
      <div class="text-center mb-8">
        <h1 class="text-3xl font-bold text-yellow-400 mb-2">
          Legal AI Platform
        </h1>
        <p class="text-gray-400 text-sm">
          Advanced evidence processing with AI-powered analysis
        </p>
      </div>

      <!-- Auth Mode Toggle -->
      <div class="flex bg-gray-700 rounded-lg p-1 mb-6">
        <button
          type="button"
          class="flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all duration-200 {authMode === 'login'
            ? 'bg-yellow-500 text-black'
            : 'text-gray-300 hover:text-white'}"
          onclick={() => authMode = 'login'}
        >
          🔐 Login
        </button>
        <button
          type="button"
          class="flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all duration-200 {authMode === 'register'
            ? 'bg-yellow-500 text-black'
            : 'text-gray-300 hover:text-white'}"
          onclick={() => authMode = 'register'}
        >
          📝 Register
        </button>
      </div>

      <!-- Error/Success Messages -->
      {#if error}
        <div class="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4" transition:slide>
          {error}
        </div>
      {/if}

      {#if success}
        <div class="bg-green-900/50 border border-green-500 text-green-200 px-4 py-3 rounded mb-4" transition:slide>
          {success}
        </div>
      {/if}

      <!-- Auth Form -->
      <form onsubmit={handleSubmit} class="space-y-4">
        <!-- Common Fields -->
        <div>
          <label for="email" class="block text-sm font-medium text-gray-300 mb-2">
            Email Address
          </label>
          <input
            type="email"
            id="email"
            bind:value={formData.email}
            required
            class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
            placeholder="Enter your email"
          />
        </div>

        <!-- Register-specific: Name fields -->
        {#if authMode === 'register'}
          <div class="grid grid-cols-2 gap-3" transition:slide={{ duration: 300, easing: cubicOut }}>
            <div>
              <label for="firstName" class="block text-sm font-medium text-gray-300 mb-2">
                First Name
              </label>
              <input
                type="text"
                id="firstName"
                bind:value={formData.firstName}
                required
                class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
                placeholder="First name"
              />
            </div>
            <div>
              <label for="lastName" class="block text-sm font-medium text-gray-300 mb-2">
                Last Name
              </label>
              <input
                type="text"
                id="lastName"
                bind:value={formData.lastName}
                required
                class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
                placeholder="Last name"
              />
            </div>
          </div>
        {/if}

        <div>
          <label for="password" class="block text-sm font-medium text-gray-300 mb-2">
            Password
          </label>
          <input
            type="password"
            id="password"
            bind:value={formData.password}
            required
            class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
            placeholder="Enter your password"
          />
        </div>

        <!-- Register-specific: Confirm Password -->
        {#if authMode === 'register'}
          <div transition:slide={{ duration: 300, easing: cubicOut }}>
            <label for="confirmPassword" class="block text-sm font-medium text-gray-300 mb-2">
              Confirm Password
            </label>
            <input
              type="password"
              id="confirmPassword"
              bind:value={formData.confirmPassword}
              required
              class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors {formData.password && formData.confirmPassword && formData.password !== formData.confirmPassword ? 'border-red-500' : ''}"
              placeholder="Confirm your password"
            />
            {#if formData.password && formData.confirmPassword && formData.password !== formData.confirmPassword}
              <p class="text-red-400 text-sm mt-1">Passwords do not match</p>
            {/if}
          </div>

          <!-- Professional Details -->
          <div class="space-y-3" transition:slide={{ duration: 300, easing: cubicOut }}>
            <div>
              <label for="role" class="block text-sm font-medium text-gray-300 mb-2">
                Role
              </label>
              <select
                id="role"
                bind:value={formData.role}
                required
                class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
              >
                <option value="prosecutor">Prosecutor</option>
                <option value="investigator">Investigator</option>
                <option value="analyst">Analyst</option>
                <option value="admin">Administrator</option>
              </select>
            </div>

            <div>
              <label for="department" class="block text-sm font-medium text-gray-300 mb-2">
                Department
              </label>
              <input
                type="text"
                id="department"
                bind:value={formData.department}
                required
                class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
                placeholder="e.g., District Attorney's Office"
              />
            </div>

            <div>
              <label for="jurisdiction" class="block text-sm font-medium text-gray-300 mb-2">
                Jurisdiction
              </label>
              <input
                type="text"
                id="jurisdiction"
                bind:value={formData.jurisdiction}
                required
                class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
                placeholder="e.g., Los Angeles County"
              />
            </div>

            <div>
              <label for="badgeNumber" class="block text-sm font-medium text-gray-300 mb-2">
                Badge Number <span class="text-gray-500">(Optional)</span>
              </label>
              <input
                type="text"
                id="badgeNumber"
                bind:value={formData.badgeNumber}
                class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-400 transition-colors"
                placeholder="Badge number"
              />
            </div>
          </div>

          <!-- Terms and Privacy -->
          <div class="space-y-3" transition:slide={{ duration: 300, easing: cubicOut }}>
            <label class="flex items-start space-x-3 cursor-pointer">
              <input
                type="checkbox"
                bind:checked={formData.agreeToTerms}
                required
                class="mt-0.5 h-4 w-4 text-yellow-500 bg-gray-700 border-gray-600 rounded focus:ring-yellow-500 focus:ring-2"
              />
              <span class="text-sm text-gray-300">
                I agree to the <a href="/legal/terms" class="text-yellow-400 hover:underline">Terms of Service</a>
              </span>
            </label>

            <label class="flex items-start space-x-3 cursor-pointer">
              <input
                type="checkbox"
                bind:checked={formData.agreeToPrivacy}
                required
                class="mt-0.5 h-4 w-4 text-yellow-500 bg-gray-700 border-gray-600 rounded focus:ring-yellow-500 focus:ring-2"
              />
              <span class="text-sm text-gray-300">
                I agree to the <a href="/legal/privacy" class="text-yellow-400 hover:underline">Privacy Policy</a>
              </span>
            </label>
          </div>
        {/if}

        <!-- Submit Button -->
        <button
          type="submit"
          disabled={!isFormValid || isLoading}
          class="w-full bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-black font-semibold py-3 px-4 rounded transition-all duration-200 mt-6"
        >
          {#if isLoading}
            <span class="flex items-center justify-center">
              <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {authMode === 'login' ? 'Signing In...' : 'Creating Account...'}
            </span>
          {:else}
            {authMode === 'login' ? '🔐 Sign In' : '📝 Create Account'}
          {/if}
        </button>
      </form>

      <!-- Demo Notice -->
      {#if authMode === 'login'}
        <div class="mt-6 text-center">
          <p class="text-gray-400 text-sm">
            Demo: Use any email and password to login
          </p>
        </div>
      {/if}

      <!-- Alternative Action -->
      <div class="mt-6 text-center">
        <p class="text-gray-400 text-sm">
          {authMode === 'login' ? "Don't have an account?" : "Already have an account?"}
          <button
            type="button"
            onclick={toggleAuthMode}
            class="text-yellow-400 hover:text-yellow-300 hover:underline ml-1 transition-colors"
          >
            {authMode === 'login' ? 'Create one here' : 'Sign in instead'}
          </button>
        </p>
      </div>
    </div>
  </div>
</div>

<style>
  /* Custom checkbox styling for better visibility */
  input[type="checkbox"] {
    accent-color: #eab308;
  }
</style>