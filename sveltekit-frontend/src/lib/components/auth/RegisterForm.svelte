<!--
  Enhanced Registration Form - Legal AI Platform
  Using Bits UI v2 + Superforms + XState + MCP GPU Orchestrator
-->
<script lang="ts">
  import { $props, $state, $derived } from 'svelte';
  import { enhance } from '$app/forms';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { superForm } from 'sveltekit-superforms/client';
  import { zod } from 'sveltekit-superforms/adapters';
  import { createActor } from 'xstate';
  import * as Form from '$lib/components/ui/form';
  import * as Card from '$lib/components/ui/card';
  import * as Alert from '$lib/components/ui/alert';
  import * as Select from '$lib/components/ui/select';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { Textarea } from '$lib/components/ui/textarea';
  import { 
    Eye, EyeOff, Shield, Loader2, AlertCircle, 
    Zap, UserPlus, Badge, Building, Scale 
  } from 'lucide-svelte';
  import { authMachine } from '$lib/machines/auth-machine';
  import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator';
  import { z } from 'zod';
  
  // Enhanced registration schema for legal professionals
  const registerSchema = z.object({
    email: z.string().email('Please enter a valid email address'),
    firstName: z.string().min(2, 'First name must be at least 2 characters'),
    lastName: z.string().min(2, 'Last name must be at least 2 characters'),
    password: z.string()
      .min(12, 'Password must be at least 12 characters')
      .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 
        'Password must include uppercase, lowercase, number, and special character'),
    confirmPassword: z.string(),
    role: z.enum(['prosecutor', 'investigator', 'analyst', 'admin']),
    department: z.string().min(2, 'Department is required'),
    jurisdiction: z.string().min(2, 'Jurisdiction is required'),
    badgeNumber: z.string().optional(),
    agreeToTerms: z.boolean().refine(val => val === true, 'You must agree to the terms'),
    agreeToPrivacy: z.boolean().refine(val => val === true, 'You must agree to privacy policy'),
    enableTwoFactor: z.boolean().default(false)
  }).refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });

  interface Props {
    data: any;
    redirectTo?: string;
    showLogin?: boolean;
    enableGPUValidation?: boolean;
  }

  let { 
    data, 
    redirectTo = '/dashboard', 
    showLogin = true,
    enableGPUValidation = true 
  }: Props = $props();

  // Form state
  let showPassword = $state(false);
  let showConfirmPassword = $state(false);
  let isLoading = $state(false);
  let errorMessage = $state('');
  let successMessage = $state('');
  let gpuValidationStatus = $state<'idle' | 'processing' | 'success' | 'warning' | 'error'>('idle');
  let securityScore = $state(0);

  // Legal role options
  const roleOptions = [
    { value: 'prosecutor', label: 'Prosecutor', icon: Scale },
    { value: 'investigator', label: 'Investigator', icon: Badge },
    { value: 'analyst', label: 'Legal Analyst', icon: Building },
    { value: 'admin', label: 'Administrator', icon: Shield }
  ];

  // XState auth machine
  const authActor = createActor(authMachine);
  authActor.start();

  // Superform setup
  const { form, errors, enhance: formEnhance, submitting, message } = superForm(data, {
    validators: zod(registerSchema),
    resetForm: false,
    delayMs: 300,
    timeoutMs: 15000,
    onSubmit: async ({ formData, cancel }) => {
      isLoading = true;
      errorMessage = '';
      successMessage = '';

      // GPU-enhanced validation if enabled
      if (enableGPUValidation) {
        try {
          gpuValidationStatus = 'processing';
          
          // Use MCP GPU orchestrator for security analysis and validation
          const validationCheck = await mcpGPUOrchestrator.dispatchGPUTask({
            id: `register_validation_${Date.now()}`,
            type: 'security_validation',
            priority: 'high',
            data: {
              email: formData.get('email'),
              firstName: formData.get('firstName'),
              lastName: formData.get('lastName'),
              role: formData.get('role'),
              department: formData.get('department'),
              jurisdiction: formData.get('jurisdiction'),
              badgeNumber: formData.get('badgeNumber'),
              timestamp: new Date().toISOString(),
              userAgent: navigator.userAgent,
              fingerprint: await generateRegistrationFingerprint()
            },
            context: {
              action: 'registration_attempt',
              enhancedValidation: true,
              legalProfessionalCheck: true
            },
            config: {
              useGPU: true,
              model: 'legal-validation-model',
              protocol: 'quic'
            }
          });

          securityScore = validationCheck.securityScore || 0;

          if (validationCheck.riskScore > 0.9) {
            gpuValidationStatus = 'error';
            errorMessage = 'Registration validation failed. Please verify your information.';
            cancel();
            return;
          } else if (validationCheck.riskScore > 0.7) {
            gpuValidationStatus = 'warning';
            // Continue but with warning
          } else {
            gpuValidationStatus = 'success';
          }

          // Additional legal professional verification
          if (validationCheck.legalVerification && !validationCheck.legalVerification.verified) {
            errorMessage = 'Unable to verify legal professional credentials. Please contact support.';
            cancel();
            return;
          }

        } catch (error) {
          console.warn('GPU validation failed, proceeding with standard validation:', error);
          gpuValidationStatus = 'idle';
        }
      }

      // Send to XState machine
      authActor.send({
        type: 'START_REGISTRATION',
        data: {
          email: formData.get('email') as string,
          firstName: formData.get('firstName') as string,
          lastName: formData.get('lastName') as string,
          password: formData.get('password') as string,
          role: formData.get('role') as string,
          department: formData.get('department') as string,
          jurisdiction: formData.get('jurisdiction') as string,
          badgeNumber: formData.get('badgeNumber') as string,
          enableTwoFactor: formData.get('enableTwoFactor') === 'on',
          deviceInfo: {
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            securityScore
          }
        }
      });
    },
    onResult: ({ result }) => {
      isLoading = false;
      
      if (result.type === 'success') {
        const data = result.data as any;
        
        if (data?.requiresVerification) {
          successMessage = 'Registration successful! Please check your email to verify your account.';
        } else if (data?.success) {
          successMessage = 'Registration successful! Redirecting to login...';
          setTimeout(() => {
            goto('/auth/login');
          }, 2000);
        }
      } else if (result.type === 'error') {
        errorMessage = result.error?.message || 'Registration failed. Please try again.';
      }
    }
  });

  // XState subscription
  authActor.subscribe((state) => {
    if (state.matches('registering')) {
      isLoading = true;
    } else if (state.matches('registered')) {
      isLoading = false;
      successMessage = 'Registration successful!';
      setTimeout(() => goto('/auth/login'), 1500);
    } else if (state.matches('error')) {
      isLoading = false;
      errorMessage = state.context.error || 'Registration failed';
    } else if (state.matches('requiresVerification')) {
      isLoading = false;
      successMessage = 'Please check your email to verify your account.';
    }
  });

  // Enhanced device fingerprinting for registration
  async function generateRegistrationFingerprint(): Promise<string> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillText('Legal AI Registration', 2, 2);
    }
    
    const fingerprint = {
      userAgent: navigator.userAgent,
      language: navigator.language,
      languages: navigator.languages,
      platform: navigator.platform,
      screenResolution: `${screen.width}x${screen.height}`,
      colorDepth: screen.colorDepth,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      canvas: canvas.toDataURL(),
      cookieEnabled: navigator.cookieEnabled,
      onlineStatus: navigator.onLine,
      doNotTrack: navigator.doNotTrack,
      hardwareConcurrency: navigator.hardwareConcurrency
    };
    
    return btoa(JSON.stringify(fingerprint));
  }

  // Password visibility toggles
  function togglePasswordVisibility() {
    showPassword = !showPassword;
  }

  function toggleConfirmPasswordVisibility() {
    showConfirmPassword = !showConfirmPassword;
  }

  // Real-time password strength checker
  let passwordStrength = $derived(calculatePasswordStrength($form.password || ''));

  function calculatePasswordStrength(password: string): { score: number; feedback: string; color: string } {
    if (!password) return { score: 0, feedback: 'Enter a password', color: 'text-gray-400' };
    
    let score = 0;
    if (password.length >= 12) score += 2;
    if (password.length >= 16) score += 1;
    if (/[a-z]/.test(password)) score += 1;
    if (/[A-Z]/.test(password)) score += 1;
    if (/\d/.test(password)) score += 1;
    if (/[@$!%*?&]/.test(password)) score += 1;
    if (password.length >= 20) score += 1;
    
    if (score < 3) return { score, feedback: 'Weak', color: 'text-red-500' };
    if (score < 5) return { score, feedback: 'Fair', color: 'text-yellow-500' };
    if (score < 7) return { score, feedback: 'Good', color: 'text-blue-500' };
    return { score, feedback: 'Excellent', color: 'text-green-500' };
  }
</script>

<Card.Root class="w-full max-w-2xl mx-auto">
  <Card.Header class="text-center">
    <div class="flex items-center justify-center mb-4">
      <Shield class="h-8 w-8 text-primary mr-2" />
      <h1 class="text-2xl font-bold">Legal AI Platform</h1>
    </div>
    <Card.Title class="text-xl flex items-center justify-center gap-2">
      <UserPlus class="h-5 w-5" />
      Create Account
    </Card.Title>
    <Card.Description>
      Register as a legal professional to access the AI-powered legal system
    </Card.Description>
  </Card.Header>

  <Card.Content>
    <!-- GPU Validation Status -->
    {#if enableGPUValidation && gpuValidationStatus !== 'idle'}
      <Alert.Root class="mb-4" variant={gpuValidationStatus === 'error' ? 'destructive' : 'default'}>
        <Zap class="h-4 w-4" />
        <Alert.Title>AI-Enhanced Validation</Alert.Title>
        <Alert.Description>
          {#if gpuValidationStatus === 'processing'}
            Running advanced credential verification...
          {:else if gpuValidationStatus === 'success'}
            Professional credentials verified successfully. Security Score: {securityScore}/100
          {:else if gpuValidationStatus === 'warning'}
            Verification completed with warnings. Please review your information.
          {:else if gpuValidationStatus === 'error'}
            Credential verification failed. Please check your information.
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

    <form method="POST" action="?/register" use:formEnhance class="space-y-4">
      <input type="hidden" name="redirectTo" value={redirectTo} />

      <!-- Personal Information -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- First Name -->
        <Form.Field {form} name="firstName">
          <Form.Control let:attrs>
            <Label for="firstName">First Name</Label>
            <Input
              {...attrs}
              id="firstName"
              type="text"
              placeholder="John"
              bind:value={$form.firstName}
              disabled={isLoading}
              class="mt-1"
            />
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>

        <!-- Last Name -->
        <Form.Field {form} name="lastName">
          <Form.Control let:attrs>
            <Label for="lastName">Last Name</Label>
            <Input
              {...attrs}
              id="lastName"
              type="text"
              placeholder="Smith"
              bind:value={$form.lastName}
              disabled={isLoading}
              class="mt-1"
            />
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>
      </div>

      <!-- Email -->
      <Form.Field {form} name="email">
        <Form.Control let:attrs>
          <Label for="email">Official Email Address</Label>
          <Input
            {...attrs}
            id="email"
            type="email"
            placeholder="john.smith@prosecutor.gov"
            bind:value={$form.email}
            disabled={isLoading}
            class="mt-1"
          />
        </Form.Control>
        <Form.FieldErrors />
      </Form.Field>

      <!-- Professional Information -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Role -->
        <Form.Field {form} name="role">
          <Form.Control let:attrs>
            <Label for="role">Professional Role</Label>
            <Select.Root bind:selected={$form.role}>
              <Select.Trigger class="mt-1">
                <Select.Value placeholder="Select your role" />
              </Select.Trigger>
              <Select.Content>
                {#each roleOptions as option}
                  <Select.Item value={option.value}>
                    <div class="flex items-center gap-2">
                      <svelte:component this={option.icon} class="h-4 w-4" />
                      {option.label}
                    </div>
                  </Select.Item>
                {/each}
              </Select.Content>
            </Select.Root>
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>

        <!-- Badge Number -->
        <Form.Field {form} name="badgeNumber">
          <Form.Control let:attrs>
            <Label for="badgeNumber">Badge/ID Number (Optional)</Label>
            <Input
              {...attrs}
              id="badgeNumber"
              type="text"
              placeholder="12345"
              bind:value={$form.badgeNumber}
              disabled={isLoading}
              class="mt-1"
            />
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>
      </div>

      <!-- Department & Jurisdiction -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Form.Field {form} name="department">
          <Form.Control let:attrs>
            <Label for="department">Department/Agency</Label>
            <Input
              {...attrs}
              id="department"
              type="text"
              placeholder="District Attorney's Office"
              bind:value={$form.department}
              disabled={isLoading}
              class="mt-1"
            />
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>

        <Form.Field {form} name="jurisdiction">
          <Form.Control let:attrs>
            <Label for="jurisdiction">Jurisdiction</Label>
            <Input
              {...attrs}
              id="jurisdiction"
              type="text"
              placeholder="Los Angeles County"
              bind:value={$form.jurisdiction}
              disabled={isLoading}
              class="mt-1"
            />
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>
      </div>

      <!-- Password Fields -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Password -->
        <Form.Field {form} name="password">
          <Form.Control let:attrs>
            <Label for="password">Password</Label>
            <div class="relative">
              <Input
                {...attrs}
                id="password"
                type={showPassword ? 'text' : 'password'}
                placeholder="Enter secure password"
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
            {#if $form.password}
              <div class="mt-2 flex items-center gap-2">
                <div class="h-2 flex-1 bg-gray-200 rounded">
                  <div 
                    class="h-full rounded transition-all duration-300"
                    class:bg-red-500={passwordStrength.score < 3}
                    class:bg-yellow-500={passwordStrength.score >= 3 && passwordStrength.score < 5}
                    class:bg-blue-500={passwordStrength.score >= 5 && passwordStrength.score < 7}
                    class:bg-green-500={passwordStrength.score >= 7}
                    style="width: {Math.min(100, (passwordStrength.score / 8) * 100)}%"
                  ></div>
                </div>
                <span class="text-sm {passwordStrength.color}">{passwordStrength.feedback}</span>
              </div>
            {/if}
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>

        <!-- Confirm Password -->
        <Form.Field {form} name="confirmPassword">
          <Form.Control let:attrs>
            <Label for="confirmPassword">Confirm Password</Label>
            <div class="relative">
              <Input
                {...attrs}
                id="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                placeholder="Confirm your password"
                bind:value={$form.confirmPassword}
                disabled={isLoading}
                class="mt-1 pr-10"
              />
              <button
                type="button"
                class="absolute inset-y-0 right-0 pr-3 flex items-center"
                onclick={toggleConfirmPasswordVisibility}
                disabled={isLoading}
              >
                {#if showConfirmPassword}
                  <EyeOff class="h-4 w-4 text-gray-400" />
                {:else}
                  <Eye class="h-4 w-4 text-gray-400" />
                {/if}
              </button>
            </div>
          </Form.Control>
          <Form.FieldErrors />
        </Form.Field>
      </div>

      <!-- Security Options -->
      <div class="space-y-3">
        <div class="flex items-center space-x-2">
          <Checkbox
            id="enableTwoFactor"
            name="enableTwoFactor"
            bind:checked={$form.enableTwoFactor}
            disabled={isLoading}
          />
          <Label for="enableTwoFactor" class="text-sm">
            Enable two-factor authentication (recommended for legal professionals)
          </Label>
        </div>
      </div>

      <!-- Terms and Privacy -->
      <div class="space-y-3">
        <div class="flex items-center space-x-2">
          <Checkbox
            id="agreeToTerms"
            name="agreeToTerms"
            bind:checked={$form.agreeToTerms}
            disabled={isLoading}
          />
          <Label for="agreeToTerms" class="text-sm">
            I agree to the <a href="/legal/terms" class="text-primary hover:underline">Terms of Service</a>
          </Label>
        </div>

        <div class="flex items-center space-x-2">
          <Checkbox
            id="agreeToPrivacy"
            name="agreeToPrivacy"
            bind:checked={$form.agreeToPrivacy}
            disabled={isLoading}
          />
          <Label for="agreeToPrivacy" class="text-sm">
            I agree to the <a href="/legal/privacy" class="text-primary hover:underline">Privacy Policy</a>
          </Label>
        </div>
      </div>

      <!-- Submit Button -->
      <Button 
        type="submit" 
        class="w-full" 
        disabled={isLoading || $submitting}
      >
        {#if isLoading || $submitting}
          <Loader2 class="mr-2 h-4 w-4 animate-spin" />
          Creating Account...
        {:else}
          <UserPlus class="mr-2 h-4 w-4" />
          Create Legal Professional Account
        {/if}
      </Button>
    </form>

    <!-- Login Link -->
    {#if showLogin}
      <div class="mt-6 text-center">
        <p class="text-sm text-muted-foreground">
          Already have an account?
          <a 
            href="/auth/login" 
            class="text-primary hover:underline font-medium"
            tabindex={isLoading ? -1 : 0}
          >
            Sign in here
          </a>
        </p>
      </div>
    {/if}
  </Card.Content>
</Card.Root>