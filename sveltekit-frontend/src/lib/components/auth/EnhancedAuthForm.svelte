<script lang="ts">
  import { $props, $state, $derived, $effect } from 'svelte';
  import { enhance } from '$app/forms';
  import { Dialog } from 'bits-ui';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Alert } from '$lib/components/ui/alert/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { Progress } from '$lib/components/ui/progress/index.js';
  import { Checkbox } from '$lib/components/ui/checkbox/index.js';
  import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator.js';
  import { scale, fade } from 'svelte/transition';
  import { quartOut } from 'svelte/easing';
  
  interface Props {
    mode?: 'login' | 'register';
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
    onSuccess?: (user: any) => void;
    allowGuestMode?: boolean;
  }

  let { 
    mode = $bindable('login'),
    open = $bindable(false),
    onOpenChange,
    onSuccess,
    allowGuestMode = false
  }: Props = $props();

  // Enhanced Svelte 5 reactive state
  let formData = $state({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    acceptTerms: false,
    rememberMe: false
  });
  
  let formState = $state({
    loading: false,
    error: '',
    success: '',
    passwordStrength: 0,
    showPassword: false,
    showConfirmPassword: false,
    emailExists: false,
    verificationSent: false
  });

  // Form element references for focus management
  let emailInput = $state<HTMLInputElement>();
  let passwordInput = $state<HTMLInputElement>();
  let firstNameInput = $state<HTMLInputElement>();

  // Enhanced validation with security requirements
  let validation = $derived(() => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const hasValidEmail = emailRegex.test(formData.email);
    const hasStrongPassword = formData.password.length >= 8 && 
      /(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/.test(formData.password);
    
    if (mode === 'register') {
      const passwordsMatch = formData.confirmPassword === formData.password;
      const hasName = formData.firstName.trim().length >= 2 && formData.lastName.trim().length >= 2;
      const termsAccepted = formData.acceptTerms;
      
      return {
        isValid: hasValidEmail && hasStrongPassword && passwordsMatch && hasName && termsAccepted,
        hasValidEmail,
        hasStrongPassword,
        passwordsMatch,
        hasName,
        termsAccepted
      };
    }
    
    return {
      isValid: hasValidEmail && formData.password.length >= 6,
      hasValidEmail,
      hasStrongPassword: true,
      passwordsMatch: true,
      hasName: true,
      termsAccepted: true
    };
  });

  // Password strength calculation
  let passwordStrength = $derived(() => {
    const password = formData.password;
    if (!password) return 0;
    
    let strength = 0;
    if (password.length >= 8) strength += 25;
    if (/[a-z]/.test(password)) strength += 25;
    if (/[A-Z]/.test(password)) strength += 25;
    if (/\d/.test(password)) strength += 15;
    if (/[@$!%*?&]/.test(password)) strength += 10;
    
    return Math.min(strength, 100);
  });

  // Real-time email validation
  async function checkEmailExists() {
    if (!validation.hasValidEmail) return;
    
    try {
      const response = await fetch('/api/auth/check-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: formData.email })
      });
      
      const result = await response.json();
      formState.emailExists = result.exists;
    } catch (error) {
      console.error('Email check failed:', error);
    }
  }

  // Enhanced form submission with comprehensive security
  async function handleSubmit(event: Event) {
    const form = event.target as HTMLFormElement;
    
    formState.loading = true;
    formState.error = '';
    formState.success = '';

    try {
      // Security context for AI analysis
      const authContext = {
        mode,
        email: formData.email,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        ipAddress: await getClientIP(),
        passwordStrength: passwordStrength
      };

      // AI-powered security analysis
      const securityAnalysis = await mcpGPUOrchestrator.routeAPIRequest(
        '/api/security/analyze-login-attempt',
        authContext,
        { userId: null, securityLevel: 'authentication' }
      );

      if (securityAnalysis?.riskLevel === 'high') {
        formState.error = 'Security check failed. Please try again later.';
        return;
      }

      const endpoint = mode === 'login' ? '/api/auth/login' : '/api/auth/register';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({
          ...formData,
          securityContext: authContext
        })
      });

      const result = await response.json();

      if (response.ok) {
        formState.success = result.message || `${mode === 'login' ? 'Login' : 'Registration'} successful!`;
        
        if (mode === 'register' && result.requiresVerification) {
          formState.verificationSent = true;
          formState.success = 'Please check your email to verify your account.';
        }

        // Log successful authentication
        await logAuthEvent('success', authContext, result);

        // Close dialog after delay
        setTimeout(() => {
          resetForm();
          open = false;
          onSuccess?.(result.user);
        }, mode === 'register' && result.requiresVerification ? 3000 : 1500);
      } else {
        formState.error = result.error || 'Authentication failed';
        await logAuthEvent('failed', authContext, result);
      }
    } catch (err) {
      formState.error = 'Network error occurred. Please try again.';
      console.error('Auth error:', err);
    } finally {
      formState.loading = false;
    }
  }

  // Helper functions
  async function getClientIP(): Promise<string> {
    try {
      const response = await fetch('/api/client-ip');
      const data = await response.json();
      return data.ip || 'unknown';
    } catch {
      return 'unknown';
    }
  }

  async function logAuthEvent(type: 'success' | 'failed', context: any, result: any) {
    try {
      await mcpGPUOrchestrator.processLegalDocument(
        `Authentication ${type}: ${mode} for ${formData.email}`,
        {
          includeRAG: false,
          includeGraph: true,
          generateSummary: false,
          metadata: { context, result }
        }
      );
    } catch (error) {
      console.error('Failed to log auth event:', error);
    }
  }

  function resetForm() {
    formData = {
      email: '',
      password: '',
      confirmPassword: '',
      firstName: '',
      lastName: '',
      acceptTerms: false,
      rememberMe: false
    };
    formState = {
      loading: false,
      error: '',
      success: '',
      passwordStrength: 0,
      showPassword: false,
      showConfirmPassword: false,
      emailExists: false,
      verificationSent: false
    };
  }

  function toggleMode() {
    mode = mode === 'login' ? 'register' : 'login';
    formState.error = '';
    formState.success = '';
    formData.confirmPassword = '';
    formData.firstName = '';
    formData.lastName = '';
    formData.acceptTerms = false;
  }

  async function handleGuestLogin() {
    if (!allowGuestMode) return;
    
    formState.loading = true;
    try {
      const response = await fetch('/api/auth/guest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const result = await response.json();
      if (response.ok) {
        onSuccess?.(result.user);
        open = false;
      }
    } catch (error) {
      console.error('Guest login failed:', error);
    } finally {
      formState.loading = false;
    }
  }

  // Effects for enhanced UX
  $effect(() => {
    if (open && emailInput) {
      setTimeout(() => emailInput?.focus(), 100);
    }
  });

  $effect(() => {
    if (onOpenChange) {
      onOpenChange(open);
    }
  });

  // Real-time email validation effect
  $effect(() => {
    if (formData.email && mode === 'register') {
      const debounce = setTimeout(checkEmailExists, 500);
      return () => clearTimeout(debounce);
    }
  });
</script>

<Dialog.Root bind:open>
  <Dialog.Portal>
    <Dialog.Overlay 
      class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" 
    />
    <Dialog.Content 
      class="fixed left-[50%] top-[50%] z-50 grid w-full max-w-md translate-x-[-50%] translate-y-[-50%] gap-6 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg"
      onopenautofocus={(e) => {
        e.preventDefault();
        emailInput?.focus();
      }}
    >
      <!-- Header -->
      <div class="flex flex-col space-y-2 text-center sm:text-left">
        <Dialog.Title class="text-lg font-semibold leading-none tracking-tight">
          {mode === 'login' ? 'Welcome back' : 'Create your account'}
        </Dialog.Title>
        <Dialog.Description class="text-sm text-muted-foreground">
          {mode === 'login' 
            ? 'Access your legal case management system with AI-powered analysis'
            : 'Join the next generation of legal professionals with AI assistance'
          }
        </Dialog.Description>
      </div>

      <form onsubmit={handleSubmit} class="space-y-4">
        <!-- Success Message -->
        {#if formState.success}
          <Alert variant="default" class="border-green-200 bg-green-50 text-green-800">
            <div class="flex items-center gap-2">
              <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
              {formState.success}
            </div>
          </Alert>
        {/if}

        <!-- Error Message -->
        {#if formState.error}
          <Alert variant="destructive">
            <div class="flex items-center gap-2">
              <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {formState.error}
            </div>
          </Alert>
        {/if}

        <!-- Name Fields (Register Only) -->
        {#if mode === 'register'}
          <div class="grid grid-cols-2 gap-4" transition:scale={{ duration: 300, easing: quartOut }}>
            <div class="space-y-2">
              <Label for="firstName">First Name *</Label>
              <Input 
                bind:this={firstNameInput}
                id="firstName"
                name="firstName"
                type="text"
                placeholder="John"
                bind:value={formData.firstName}
                disabled={formState.loading}
                required
                class={!validation.hasName && formData.firstName ? 'border-red-500' : ''}
              />
              {#if !validation.hasName && formData.firstName}
                <p class="text-xs text-red-500">Must be at least 2 characters</p>
              {/if}
            </div>
            <div class="space-y-2">
              <Label for="lastName">Last Name *</Label>
              <Input 
                id="lastName"
                name="lastName"
                type="text"
                placeholder="Doe"
                bind:value={formData.lastName}
                disabled={formState.loading}
                required
                class={!validation.hasName && formData.lastName ? 'border-red-500' : ''}
              />
            </div>
          </div>
        {/if}

        <!-- Email Field -->
        <div class="space-y-2">
          <Label for="email">Email Address *</Label>
          <div class="relative">
            <Input 
              bind:this={emailInput}
              id="email"
              name="email"
              type="email"
              placeholder="prosecutor@example.com"
              bind:value={formData.email}
              disabled={formState.loading}
              required
              class={!validation.hasValidEmail && formData.email ? 'border-red-500' : ''}
            />
            {#if mode === 'register' && formState.emailExists}
              <Badge variant="destructive" class="absolute right-2 top-2 text-xs">
                Email exists
              </Badge>
            {/if}
          </div>
          {#if !validation.hasValidEmail && formData.email}
            <p class="text-xs text-red-500">Please enter a valid email address</p>
          {/if}
        </div>

        <!-- Password Field -->
        <div class="space-y-2">
          <Label for="password">Password *</Label>
          <div class="relative">
            <Input 
              bind:this={passwordInput}
              id="password"
              name="password"
              type={formState.showPassword ? 'text' : 'password'}
              placeholder="••••••••"
              bind:value={formData.password}
              disabled={formState.loading}
              required
              class={mode === 'register' && !validation.hasStrongPassword && formData.password ? 'border-red-500' : ''}
            />
            <button
              type="button"
              onclick={() => formState.showPassword = !formState.showPassword}
              class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700"
            >
              {#if formState.showPassword}
                <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L3 3m6.878 6.878L21 21" />
                </svg>
              {:else}
                <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              {/if}
            </button>
          </div>
          
          {#if mode === 'register' && formData.password}
            <div class="space-y-1" transition:fade={{ duration: 200 }}>
              <div class="flex items-center justify-between text-xs">
                <span>Password strength</span>
                <span class={
                  passwordStrength >= 80 ? 'text-green-600' :
                  passwordStrength >= 60 ? 'text-yellow-600' :
                  passwordStrength >= 40 ? 'text-orange-600' : 'text-red-600'
                }>
                  {passwordStrength >= 80 ? 'Strong' :
                   passwordStrength >= 60 ? 'Good' :
                   passwordStrength >= 40 ? 'Fair' : 'Weak'}
                </span>
              </div>
              <Progress value={passwordStrength} class="h-2" />
              <p class="text-xs text-muted-foreground">
                Use 8+ characters with uppercase, lowercase, numbers, and symbols
              </p>
            </div>
          {/if}
        </div>

        <!-- Confirm Password (Register Only) -->
        {#if mode === 'register'}
          <div class="space-y-2" transition:scale={{ duration: 300, easing: quartOut }}>
            <Label for="confirmPassword">Confirm Password *</Label>
            <div class="relative">
              <Input 
                id="confirmPassword"
                name="confirmPassword"
                type={formState.showConfirmPassword ? 'text' : 'password'}
                placeholder="••••••••"
                bind:value={formData.confirmPassword}
                disabled={formState.loading}
                required
                class={!validation.passwordsMatch && formData.confirmPassword ? 'border-red-500' : ''}
              />
              <button
                type="button"
                onclick={() => formState.showConfirmPassword = !formState.showConfirmPassword}
                class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700"
              >
                {#if formState.showConfirmPassword}
                  <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L3 3m6.878 6.878L21 21" />
                  </svg>
                {:else}
                  <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                {/if}
              </button>
            </div>
            {#if !validation.passwordsMatch && formData.confirmPassword}
              <p class="text-xs text-red-500">Passwords do not match</p>
            {/if}
          </div>
        {/if}

        <!-- Terms and Remember Me -->
        <div class="space-y-3">
          {#if mode === 'register'}
            <div class="flex items-center space-x-2">
              <Checkbox 
                id="terms"
                bind:checked={formData.acceptTerms}
                required
              />
              <Label for="terms" class="text-sm leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                I agree to the <a href="/terms" class="text-primary hover:underline">Terms of Service</a> 
                and <a href="/privacy" class="text-primary hover:underline">Privacy Policy</a>
              </Label>
            </div>
          {/if}
          
          {#if mode === 'login'}
            <div class="flex items-center space-x-2">
              <Checkbox 
                id="remember"
                bind:checked={formData.rememberMe}
              />
              <Label for="remember" class="text-sm leading-none">
                Remember me for 30 days
              </Label>
            </div>
          {/if}
        </div>

        <!-- Submit Button -->
        <Button 
          type="submit" 
          class="w-full"
          disabled={formState.loading || !validation.isValid}
        >
          {#if formState.loading}
            <svg class="mr-2 h-4 w-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Processing...
          {:else}
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          {/if}
        </Button>

        <!-- Guest Mode -->
        {#if allowGuestMode && mode === 'login'}
          <Button 
            type="button"
            variant="outline"
            class="w-full"
            onclick={handleGuestLogin}
            disabled={formState.loading}
          >
            Continue as Guest
          </Button>
        {/if}

        <!-- Mode Toggle -->
        <div class="text-center">
          <button 
            type="button"
            onclick={toggleMode}
            class="text-sm text-primary hover:underline"
            disabled={formState.loading}
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
          <div class="text-xs text-muted-foreground text-center space-y-1">
            <p class="font-medium">Demo Accounts:</p>
            <p>Admin: admin@prosecutor.com / password</p>
            <p>User: user@prosecutor.com / password</p>
          </div>
        </div>
      {/if}

      <!-- Close Button -->
      <Dialog.Close class="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
        <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
        <span class="sr-only">Close</span>
      </Dialog.Close>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<style>
  :global(.animate-in) {
    animation-duration: 200ms;
    animation-fill-mode: both;
  }
  
  :global(.animate-out) {
    animation-duration: 150ms;
    animation-fill-mode: both;
  }
</style>