<!--
  Enhanced Login Form - Legal AI Platform
  Using Bits UI v2 + Superforms + XState + MCP GPU Orchestrator
-->
<script lang="ts">
  import { enhance } from '$app/forms';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { superForm } from 'sveltekit-superforms/client';
  import { zod } from 'sveltekit-superforms/adapters';
  import { createActor } from 'xstate';
  import * as Form from '$lib/components/ui/form';
  import * as Card from '$lib/components/ui/card';
  import * as Alert from '$lib/components/ui/alert';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { Eye, EyeOff, Shield, Loader2, AlertCircle, Zap } from 'lucide-svelte';
  import { authMachine } from '$lib/machines/auth-machine';
  import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator';
  import { z } from 'zod';
  
  // Form schema
  const loginSchema = z.object({
    email: z.string().email('Please enter a valid email address'),
    password: z.string().min(8, 'Password must be at least 8 characters'),
    rememberMe: z.boolean().default(false),
    twoFactorCode: z.string().optional()
  });

  interface Props {
    data: any;
    redirectTo?: string;
    showRegistration?: boolean;
    enableGPUAuth?: boolean;
  }

  let { 
    data, 
    redirectTo = '/dashboard', 
    showRegistration = true,
    enableGPUAuth = true 
  }: Props = $props();

  // Form state
  let showPassword = $state(false);
  let showTwoFactor = $state(false);
  let isLoading = $state(false);
  let errorMessage = $state('');
  let successMessage = $state('');
  let gpuAuthStatus = $state<'idle' | 'processing' | 'success' | 'error'>('idle');

  // XState auth machine
  const authActor = createActor(authMachine);
  authActor.start();

  // Superform setup
  const { form, errors, enhance: formEnhance, submitting, message } = superForm(data, {
    validators: zod(loginSchema),
    resetForm: false,
    delayMs: 300,
    timeoutMs: 8000,
    onSubmit: async ({ formData, cancel }) => {
      isLoading = true;
      errorMessage = '';
      successMessage = '';

      // GPU-enhanced authentication if enabled
      if (enableGPUAuth) {
        try {
          gpuAuthStatus = 'processing';
          
          // Use MCP GPU orchestrator for enhanced security analysis
          const securityCheck = await mcpGPUOrchestrator.dispatchGPUTask({
            id: `auth_${Date.now()}`,
            type: 'security_analysis',
            priority: 'high',
            data: {
              email: formData.get('email'),
              timestamp: new Date().toISOString(),
              userAgent: navigator.userAgent,
              fingerprint: await generateDeviceFingerprint()
            },
            context: {
              action: 'login_attempt',
              enhancedSecurity: true
            },
            config: {
              useGPU: true,
              model: 'security-analysis-model',
              protocol: 'quic'
            }
          });

          if (securityCheck.riskScore > 0.8) {
            gpuAuthStatus = 'error';
            errorMessage = 'Security verification failed. Please try again.';
            cancel();
            return;
          }

          gpuAuthStatus = 'success';
        } catch (error) {
          console.warn('GPU authentication failed, proceeding with standard auth:', error);
          gpuAuthStatus = 'idle';
        }
      }

      // Send to XState machine
      authActor.send({
        type: 'START_LOGIN',
        data: {
          email: formData.get('email') as string,
          password: formData.get('password') as string,
          rememberMe: formData.get('rememberMe') === 'on',
          twoFactorCode: formData.get('twoFactorCode') as string,
          deviceInfo: {
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
          }
        }
      });
    },
    onResult: ({ result }) => {
      isLoading = false;
      
      if (result.type === 'success') {
        const data = result.data as any;
        
        if (data?.requiresTwoFactor) {
          showTwoFactor = true;
          successMessage = 'Please enter your two-factor authentication code.';
        } else if (data?.success) {
          successMessage = 'Login successful! Redirecting...';
          setTimeout(() => {
            goto(redirectTo);
          }, 1500);
        }
      } else if (result.type === 'error') {
        errorMessage = result.error?.message || 'Login failed. Please try again.';
      }
    }
  });

  // XState subscription
  authActor.subscribe((state) => {
    if (state.matches('authenticating')) {
      isLoading = true;
    } else if (state.matches('authenticated')) {
      isLoading = false;
      successMessage = 'Authentication successful!';
      setTimeout(() => goto(redirectTo), 1000);
    } else if (state.matches('error')) {
      isLoading = false;
      errorMessage = state.context.error || 'Authentication failed';
    } else if (state.matches('requiresTwoFactor')) {
      isLoading = false;
      showTwoFactor = true;
      successMessage = 'Please enter your two-factor code.';
    }
  });

  // Device fingerprinting for security
  async function generateDeviceFingerprint(): Promise<string> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillText('Device fingerprint', 2, 2);
    }
    
    const fingerprint = {
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
      screenResolution: `${screen.width}x${screen.height}`,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      canvas: canvas.toDataURL()
    };
    
    return btoa(JSON.stringify(fingerprint));
  }

  // Handle social login
  async function handleSocialLogin(provider: 'google' | 'github') {
    isLoading = true;
    
    try {
      // Redirect to OAuth provider
      window.location.href = `/auth/oauth/${provider}`;
    } catch (error) {
      errorMessage = `${provider} login failed. Please try again.`;
      isLoading = false;
    }
  }

  // Handle password visibility toggle
  function togglePasswordVisibility() {
    showPassword = !showPassword;
  }
</script>

<Card.Root class="w-full max-w-md mx-auto">
  <Card.Header class="text-center">
    <div class="flex items-center justify-center mb-4">
      <Shield class="h-8 w-8 text-primary mr-2" />
      <h1 class="text-2xl font-bold">Legal AI Platform</h1>
    </div>
    <Card.Title class="text-xl">Sign In</Card.Title>
    <Card.Description>
      Enter your credentials to access the legal AI system
    </Card.Description>
  </Card.Header>

  <Card.Content>
    <!-- GPU Authentication Status -->
    {#if enableGPUAuth && gpuAuthStatus !== 'idle'}
      <Alert.Root class="mb-4" variant={gpuAuthStatus === 'error' ? 'destructive' : 'default'}>
        <Zap class="h-4 w-4" />
        <Alert.Title>GPU-Enhanced Security</Alert.Title>
        <Alert.Description>
          {#if gpuAuthStatus === 'processing'}
            Running advanced security analysis...
          {:else if gpuAuthStatus === 'success'}
            Security verification completed successfully.
          {:else if gpuAuthStatus === 'error'}
            Enhanced security check failed. Using standard authentication.
          {/if}
        </Alert.Description>
      </Alert.Root>
    {/if}

    <!-- Error Message -->
    {#if errorMessage}
      <Alert.Root variant="destructive" class="mb-4">
        <AlertCircle class="h-4 w-4" />
        <Alert.Title>Error</Alert.Title>
        <Alert.Description>{errorMessage}</Alert.Description>
      </Alert.Root>
    {/if}

    <!-- Success Message -->
    {#if successMessage}
      <Alert.Root class="mb-4">
        <Shield class="h-4 w-4" />
        <Alert.Title>Success</Alert.Title>
        <Alert.Description>{successMessage}</Alert.Description>
      </Alert.Root>
    {/if}

    <form method="POST" action="?/login" use:formEnhance class="space-y-4">
      <input type="hidden" name="redirectTo" value={redirectTo} />

      <!-- Email Field -->
      <Form.Field {form} name="email">
        <Form.Control let:attrs>
          <Label for="email">Email Address</Label>
          <Input
            {...attrs}
            id="email"
            type="email"
            placeholder="prosecutor@legal-ai.com"
            bind:value={$form.email}
            disabled={isLoading}
            class="mt-1"
          />
        </Form.Control>
        <Form.FieldErrors />
      </Form.Field>

      <!-- Password Field -->
      <Form.Field {form} name="password">
        <Form.Control let:attrs>
          <Label for="password">Password</Label>
          <div class="relative">
            <Input
              {...attrs}
              id="password"
              type={showPassword ? 'text' : 'password'}
              placeholder="Enter your password"
              bind:value={$form.password}
              disabled={isLoading}
              class="mt-1 pr-10"
            />
            <button
              type="button"
              class="absolute inset-y-0 right-0 pr-3 flex items-center"
              onclick={togglePasswordVisibility}
              disabled={isLoading}
            >
              {#if showPassword}
                <EyeOff class="h-4 w-4 text-gray-400" />
              {:else}
                <Eye class="h-4 w-4 text-gray-400" />
              {/if}
            </button>
          </div>
        </Form.Control>
        <Form.FieldErrors />
      </Form.Field>

      <!-- Two-Factor Code (if required) -->
      {#if showTwoFactor}
        <Form.Field {form} name="twoFactorCode">
          <Form.Control let:attrs>
            <Label for="twoFactorCode">Two-Factor Authentication Code</Label>
            <Input
              {...attrs}
              id="twoFactorCode"
              type="text"
              placeholder="000000"
              bind:value={$form.twoFactorCode}
              disabled={isLoading}
              class="mt-1"
              maxlength="6"
            />
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>
      {/if}

      <!-- Remember Me -->
      <div class="flex items-center space-x-2">
        <Checkbox
          id="rememberMe"
          name="rememberMe"
          bind:checked={$form.rememberMe}
          disabled={isLoading}
        />
        <Label for="rememberMe" class="text-sm">Remember me for 30 days</Label>
      </div>

      <!-- Submit Button -->
      <Button 
        type="submit" 
        class="w-full" 
        disabled={isLoading || $submitting}
      >
        {#if isLoading || $submitting}
          <Loader2 class="mr-2 h-4 w-4 animate-spin" />
          {showTwoFactor ? 'Verifying...' : 'Signing In...'}
        {:else}
          {showTwoFactor ? 'Verify & Sign In' : 'Sign In'}
        {/if}
      </Button>

      <!-- Forgot Password Link -->
      <div class="text-center">
        <a 
          href="/auth/forgot-password" 
          class="text-sm text-primary hover:underline"
          tabindex={isLoading ? -1 : 0}
        >
          Forgot your password?
        </a>
      </div>
    </form>

    <!-- Divider -->
    <div class="relative mt-6">
      <div class="absolute inset-0 flex items-center">
        <span class="w-full border-t" />
      </div>
      <div class="relative flex justify-center text-xs uppercase">
        <span class="bg-background px-2 text-muted-foreground">Or continue with</span>
      </div>
    </div>

    <!-- Social Login Buttons -->
    <div class="grid grid-cols-2 gap-4 mt-6">
      <Button 
        variant="outline" 
        onclick={() => handleSocialLogin('google')}
        disabled={isLoading}
      >
        <svg class="mr-2 h-4 w-4" viewBox="0 0 24 24">
          <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
          <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
          <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
          <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
        </svg>
        Google
      </Button>
      
      <Button 
        variant="outline" 
        onclick={() => handleSocialLogin('github')}
        disabled={isLoading}
      >
        <svg class="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        GitHub
      </Button>
    </div>

    <!-- Registration Link -->
    {#if showRegistration}
      <div class="mt-6 text-center">
        <p class="text-sm text-muted-foreground">
          Don't have an account?
          <a 
            href="/auth/register" 
            class="text-primary hover:underline font-medium"
            tabindex={isLoading ? -1 : 0}
          >
            Create one here
          </a>
        </p>
      </div>
    {/if}
  </Card.Content>
</Card.Root>