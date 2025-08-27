<script lang="ts">
  import { page } from '$app/stores';
  import { superForm } from 'sveltekit-superforms/client';
  
  let { data, form } = $props();
  
  // Initialize SuperForm
  const { form: formData, enhance } = superForm(data.form);
  
  let isAutoLoggingIn = $state(false);
  
  // Check for registration success message
  const showRegistrationSuccess = $derived($page.url.searchParams.get('registered') === 'true');
  
  // Auto-fill demo user credentials
  function autoLoginDemo() {
    console.log('🔧 Auto-fill demo credentials clicked');
    
    // Update SuperForm data directly
    $formData.email = 'demo@legalai.gov';
    $formData.password = 'demo123456';
    
    console.log('✅ Demo credentials filled');
  }
  
  // Auto-login with demo user (skip form submission)
  async function quickDemoLogin() {
    console.log('⚡ Quick demo login clicked');
    isAutoLoggingIn = true;
    
    try {
      console.log('📡 Calling auto-login endpoint...');
      const response = await fetch('/auth/login/auto', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      const result = await response.json();
      console.log('📨 Auto-login response:', result);
      
      if (result.success) {
        console.log('✅ Auto-login successful, redirecting...');
        // Redirect to dashboard
        window.location.href = result.redirectTo || '/dashboard';
      } else {
        // Fall back to auto-fill if auto-login fails
        console.warn('⚠️ Auto-login failed, falling back to auto-fill:', result.error);
        autoLoginDemo();
      }
    } catch (error) {
      console.error('❌ Quick demo login failed:', error);
      // Fall back to auto-fill
      autoLoginDemo();
    } finally {
      isAutoLoggingIn = false;
    }
  }
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

      {#if showRegistrationSuccess}
        <div class="success-message bg-green-900/50 border border-green-500 text-green-200 px-4 py-3 rounded mb-4" data-testid="success-message">
          Account registered successfully! You can now sign in.
        </div>
      {/if}
      
      {#if form?.error}
        <div class="error-message bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4" data-testid="error-message">
          {form.error}
        </div>
      {/if}

      {#if form?.errors?.email}
        <div class="error-message bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4" data-testid="error-message">
          Invalid email or password
        </div>
      {/if}

      {#if form?.errors?.password}
        <div class="error-message bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4" data-testid="error-message">
          Invalid email or password
        </div>
      {/if}

      <form method="POST" action="?/login" use:enhance class="space-y-6">
        <div>
          <label for="email" class="block text-sm font-medium text-gray-300 mb-2">
            Email
          </label>
          <input
            type="email"
            name="email"
            id="email"
            required
            bind:value={$formData.email}
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
            bind:value={$formData.password}
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

      <!-- Quick Demo Login -->
      <div class="mt-4 space-y-2">
        <button
          type="button"
          onclick={quickDemoLogin}
          disabled={isAutoLoggingIn}
          class="w-full bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors flex items-center justify-center"
        >
          {#if isAutoLoggingIn}
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Logging in...
          {:else}
            ⚡ Quick Demo Login
          {/if}
        </button>
        
        <button
          type="button"
          onclick={autoLoginDemo}
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded transition-colors"
        >
          📝 Auto-fill Demo Credentials
        </button>
      </div>

      <div class="mt-4 text-center">
        <p class="text-gray-400 text-sm">
          Demo Account: demo@legalai.gov / demo123456
        </p>
        <div class="text-gray-500 text-xs mt-2 space-y-1">
          <p>⚡ Quick Login: Instant access (one-click)</p>
          <p>📝 Auto-fill: Fill form then click Sign In</p>
        </div>
      </div>
    </div>
  </div>
</div>