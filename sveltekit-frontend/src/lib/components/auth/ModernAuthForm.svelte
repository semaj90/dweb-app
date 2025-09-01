<script lang="ts">
  // Svelte 5 runes are used directly without imports
  import { enhance } from '$app/forms';
  // import { Dialog } from 'bits-ui';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Alert } from '$lib/components/ui/alert/index.js';
  import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator.js';
  
  interface Props {
    mode?: 'login' | 'register';
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
    onSuccess?: (user: any) => void;
  }

  let { 
    mode = $bindable('login'),
    open = $bindable(false),
    onOpenChange,
    onSuccess
  }: Props = $props();

  // Svelte 5 runes for reactive state
  let formData = $state({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: ''
  });
  
  let loading = $state(false);
  let error = $state('');
  let success = $state('');

  // Focus management for accessibility
  let emailInput = $state<HTMLInputElement>();
  let passwordInput = $state<HTMLInputElement>();

  // Derived state for form validation
  let isValid = $derived(() => {
    const hasEmail = formData.email.includes('@');
    const hasPassword = formData.password.length >= 6;
    
    if (mode === 'register') {
      const hasConfirmPassword = formData.confirmPassword === formData.password;
      const hasName = formData.firstName.trim() && formData.lastName.trim();
      return hasEmail && hasPassword && hasConfirmPassword && hasName;
    }
    
    return hasEmail && hasPassword;
  });

  // Form submission with AI-powered analytics
  async function handleSubmit(event: Event) {
    const form = event.target as HTMLFormElement;
    const formDataObj = new FormData(form);
    
    loading = true;
    error = '';
    success = '';

    try {
      // Use GPU orchestrator to log authentication events for security analysis
      const authContext = {
        mode,
        email: formData.email,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent
      };

      // Trigger AI analysis for suspicious login patterns
      const securityAnalysis = await mcpGPUOrchestrator.routeAPIRequest(
        '/api/security/analyze-login',
        authContext,
        { userId: null, securityLevel: 'authentication' }
      );

      const endpoint = mode === 'login' ? '/api/auth/login' : '/api/auth/register';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      const result = await response.json();

      if (response.ok) {
        success = result.message || `${mode === 'login' ? 'Login' : 'Registration'} successful!`;
        
        // Log successful authentication with AI context
        await mcpGPUOrchestrator.processLegalDocument(
          `Authentication success: ${mode} for ${formData.email}`,
          {
            includeRAG: false,
            includeGraph: true,
            generateSummary: false
          }
        );

        // Reset form and close dialog
        setTimeout(() => {
          formData = {
            email: '',
            password: '',
            confirmPassword: '',
            firstName: '',
            lastName: ''
          };
          open = false;
          onSuccess?.(result.user);
        }, 1000);
      } else {
        error = result.error || 'Authentication failed';
        
        // Log failed authentication attempt for security analysis
        await mcpGPUOrchestrator.routeAPIRequest(
          '/api/security/log-failed-auth',
          { ...authContext, error: result.error },
          { userId: null, securityLevel: 'high' }
        );
      }
    } catch (err) {
      error = 'Network error occurred. Please try again.';
      console.error('Auth error:', err);
    } finally {
      loading = false;
    }
  }

  // Toggle between login and register modes
  function toggleMode() {
    mode = mode === 'login' ? 'register' : 'login';
    error = '';
    success = '';
    formData.confirmPassword = '';
    formData.firstName = '';
    formData.lastName = '';
  }

  // Effect for focus management when dialog opens
  $effect(() => {
    if (open && emailInput) {
      setTimeout(() => emailInput?.focus(), 100);
    }
  });

  // Effect for handling open state changes
  $effect(() => {
    if (onOpenChange) {
      onOpenChange(open);
    }
  });
</script>

{#if open}
  <div class="fixed inset-0 z-50 bg-black/80 animate-in fade-in-0" />
  <div class="fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 animate-in fade-in-0 zoom-in-95 slide-in-from-left-1/2 slide-in-from-top-[48%] sm:rounded-lg">
    <h2 class="text-lg font-semibold leading-none tracking-tight">
      {mode === 'login' ? 'Login to Legal AI' : 'Create Legal AI Account'}
    </h2>
    
    <p class="text-sm text-muted-foreground">
        {mode === 'login' 
          ? 'Access your legal case management system with AI-powered analysis'
          : 'Join the next generation of legal professionals with AI assistance'
        }
    </p>

      <form submit={handleSubmit} class="space-y-4">
        <!-- Success Message -->
        {#if success}
          <Alert variant="default" class="border-green-200 bg-green-50 text-green-800">
            <div class="flex items-center gap-2">
              <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
              {success}
            </div>
          </Alert>
        {/if}

        <!-- Error Message -->
        {#if error}
          <Alert variant="destructive">
            <div class="flex items-center gap-2">
              <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {error}
            </div>
          </Alert>
        {/if}

        <!-- Name Fields (Register Only) -->
        {#if mode === 'register'}
          <div class="grid grid-cols-2 gap-4">
            <div class="space-y-2">
              <Label for="firstName">First Name</Label>
              <Input 
                id="firstName"
                name="firstName"
                type="text"
                placeholder="John"
                autocomplete="given-name"
                bind:value={formData.firstName}
                disabled={loading}
                required
              />
            </div>
            <div class="space-y-2">
              <Label for="lastName">Last Name</Label>
              <Input 
                id="lastName"
                name="lastName"
                type="text"
                placeholder="Doe"
                autocomplete="family-name"
                bind:value={formData.lastName}
                disabled={loading}
                required
              />
            </div>
          </div>
        {/if}

        <!-- Email Field -->
        <div class="space-y-2">
          <Label for="email">Email</Label>
          <Input 
            bind:this={emailInput}
            id="email"
            name="email"
            type="email"
            placeholder="prosecutor@example.com"
            autocomplete="email"
            bind:value={formData.email}
            disabled={loading}
            required
          />
        </div>

        <!-- Password Field -->
        <div class="space-y-2">
          <Label for="password">Password</Label>
          <Input 
            bind:this={passwordInput}
            id="password"
            name="password"
            type="password"
            placeholder="••••••••"
            autocomplete={mode === 'login' ? 'current-password' : 'new-password'}
            bind:value={formData.password}
            disabled={loading}
            required
          />
          {#if mode === 'register'}
            <p class="text-xs text-muted-foreground">Must be at least 6 characters</p>
          {/if}
        </div>

        <!-- Confirm Password (Register Only) -->
        {#if mode === 'register'}
          <div class="space-y-2">
            <Label for="confirmPassword">Confirm Password</Label>
            <Input 
              id="confirmPassword"
              name="confirmPassword"
              type="password"
              placeholder="••••••••"
              autocomplete="new-password"
              bind:value={formData.confirmPassword}
              disabled={loading}
              required
            />
            {#if formData.confirmPassword && formData.confirmPassword !== formData.password}
              <p class="text-xs text-red-500">Passwords do not match</p>
            {/if}
          </div>
        {/if}

        <!-- Submit Button -->
        <Button 
          type="submit" 
          class="w-full"
          disabled={loading || !isValid}
        >
          {#if loading}
            <svg class="mr-2 h-4 w-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Processing...
          {:else}
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          {/if}
        </Button>

        <!-- Mode Toggle -->
        <div class="text-center">
          <button 
            type="button"
            on:onclick={toggleMode}
            class="text-sm text-primary hover:underline"
            disabled={loading}
          >
            {mode === 'login' 
              ? "Don't have an account? Sign up" 
              : "Already have an account? Sign in"
            }
          </button>
        </div>
      </form>

      <!-- Demo Accounts Notice -->
      {#if mode === 'login'}
        <div class="border-t pt-4">
          <p class="text-xs text-muted-foreground text-center">
            Demo accounts: admin@prosecutor.com / password
          </p>
        </div>
      {/if}

    <button 
      type="button"
      on:onclick={() => open = false}
      class="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none"
    >
      <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
      </svg>
      <span class="sr-only">Close</span>
    </button>
  </div>
{/if}

<style>
  /* Additional component-specific styles if needed */
  :global(.animate-in) {
    animation-duration: 200ms;
    animation-fill-mode: both;
  }
  
  :global(.animate-out) {
    animation-duration: 150ms;
    animation-fill-mode: both;
  }
</style>