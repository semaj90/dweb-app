<!--
  Simplified Registration Form - Svelte 5 Compatible
  Basic registration without complex dependencies
-->
<script lang="ts">
  import { $props, $state, $derived } from 'svelte';
  import { enhance } from '$app/forms';
  import { goto } from '$app/navigation';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Eye, EyeOff, Shield, Loader2, AlertCircle, UserPlus } from 'lucide-svelte';
  
  interface Props {
    data?: any;
    redirectTo?: string;
    showLogin?: boolean;
  }

  let { 
    data, 
    redirectTo = '/dashboard', 
    showLogin = true 
  }: Props = $props();

  // Form state
  let showPassword = $state(false);
  let showConfirmPassword = $state(false);
  let isLoading = $state(false);
  let errorMessage = $state('');
  let successMessage = $state('');

  // Form data
  let formData = $state({
    email: '',
    firstName: '',
    lastName: '',
    password: '',
    confirmPassword: '',
    role: 'analyst',
    department: '',
    jurisdiction: '',
    badgeNumber: '',
    agreeToTerms: false,
    agreeToPrivacy: false,
    enableTwoFactor: false
  });

  // Role options
  const roleOptions = [
    { value: 'prosecutor', label: 'Prosecutor' },
    { value: 'investigator', label: 'Investigator' },
    { value: 'analyst', label: 'Legal Analyst' },
    { value: 'admin', label: 'Administrator' }
  ];

  // Form validation
  function validateForm(): boolean {
    if (!formData.email || !formData.password || !formData.firstName || !formData.lastName) {
      errorMessage = 'Please fill in all required fields';
      return false;
    }

    if (formData.password !== formData.confirmPassword) {
      errorMessage = 'Passwords do not match';
      return false;
    }

    if (formData.password.length < 8) {
      errorMessage = 'Password must be at least 8 characters';
      return false;
    }

    if (!formData.agreeToTerms || !formData.agreeToPrivacy) {
      errorMessage = 'You must agree to the terms and privacy policy';
      return false;
    }

    return true;
  }

  // Password visibility toggles
  function togglePasswordVisibility() {
    showPassword = !showPassword;
  }

  function toggleConfirmPasswordVisibility() {
    showConfirmPassword = !showConfirmPassword;
  }

  // Password strength checker
  let passwordStrength = $derived(calculatePasswordStrength(formData.password));

  function calculatePasswordStrength(password: string): { score: number; feedback: string; color: string } {
    if (!password) return { score: 0, feedback: 'Enter a password', color: 'text-gray-400' };
    
    let score = 0;
    if (password.length >= 8) score += 2;
    if (password.length >= 12) score += 1;
    if (/[a-z]/.test(password)) score += 1;
    if (/[A-Z]/.test(password)) score += 1;
    if (/\d/.test(password)) score += 1;
    if (/[@$!%*?&]/.test(password)) score += 1;
    
    if (score < 3) return { score, feedback: 'Weak', color: 'text-red-500' };
    if (score < 5) return { score, feedback: 'Fair', color: 'text-yellow-500' };
    if (score < 7) return { score, feedback: 'Good', color: 'text-blue-500' };
    return { score, feedback: 'Excellent', color: 'text-green-500' };
  }
</script>

<div class="w-full max-w-2xl mx-auto">
  <div class="bg-card p-8 rounded-lg border border-border">
    <div class="text-center mb-6">
      <div class="flex items-center justify-center mb-4">
        <Shield class="h-8 w-8 text-primary mr-2" />
        <h1 class="text-2xl font-bold">Legal AI Platform</h1>
      </div>
      <h2 class="text-xl flex items-center justify-center gap-2">
        <UserPlus class="h-5 w-5" />
        Create Account
      </h2>
      <p class="text-muted-foreground mt-2">
        Register as a legal professional to access the AI-powered legal system
      </p>
    </div>

    <!-- Error Message -->
    {#if errorMessage}
      <div class="bg-destructive/15 border border-destructive text-destructive-foreground px-4 py-3 rounded mb-4 flex items-center gap-2">
        <AlertCircle class="h-4 w-4" />
        <span>{errorMessage}</span>
      </div>
    {/if}

    <!-- Success Message -->
    {#if successMessage}
      <div class="bg-green-500/15 border border-green-500 text-green-700 px-4 py-3 rounded mb-4 flex items-center gap-2">
        <Shield class="h-4 w-4" />
        <span>{successMessage}</span>
      </div>
    {/if}

    <form 
      method="POST" 
      action="?/register" 
      use:enhance={({ formData, cancel }) => {
        if (!validateForm()) {
          cancel();
          return;
        }
        
        isLoading = true;
        errorMessage = '';
        successMessage = '';

        return async ({ result }) => {
          isLoading = false;
          
          if (result.type === 'success') {
            successMessage = 'Registration successful! Redirecting to dashboard...';
            setTimeout(() => {
              goto('/dashboard');
            }, 2000);
          } else if (result.type === 'failure') {
            errorMessage = result.data?.form?.errors?.email?.[0] || 'Registration failed. Please try again.';
          } else if (result.type === 'error') {
            errorMessage = 'An error occurred during registration. Please try again.';
          }
        };
      }}
      class="space-y-4"
    >
      <input type="hidden" name="redirectTo" value={redirectTo} />

      <!-- Personal Information -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- First Name -->
        <div>
          <Label for="firstName">First Name</Label>
          <Input
            id="firstName"
            name="firstName"
            type="text"
            placeholder="John"
            bind:value={formData.firstName}
            disabled={isLoading}
            required
            class="mt-1"
          />
        </div>

        <!-- Last Name -->
        <div>
          <Label for="lastName">Last Name</Label>
          <Input
            id="lastName"
            name="lastName"
            type="text"
            placeholder="Smith"
            bind:value={formData.lastName}
            disabled={isLoading}
            required
            class="mt-1"
          />
        </div>
      </div>

      <!-- Email -->
      <div>
        <Label for="email">Official Email Address</Label>
        <Input
          id="email"
          name="email"
          type="email"
          placeholder="john.smith@prosecutor.gov"
          bind:value={formData.email}
          disabled={isLoading}
          required
          class="mt-1"
        />
      </div>

      <!-- Professional Information -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Role -->
        <div>
          <Label for="role">Professional Role</Label>
          <select
            id="role"
            name="role"
            bind:value={formData.role}
            disabled={isLoading}
            required
            class="mt-1 w-full px-3 py-2 bg-input border border-border rounded text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          >
            {#each roleOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
        </div>

        <!-- Badge Number -->
        <div>
          <Label for="badgeNumber">Badge/ID Number (Optional)</Label>
          <Input
            id="badgeNumber"
            name="badgeNumber"
            type="text"
            placeholder="12345"
            bind:value={formData.badgeNumber}
            disabled={isLoading}
            class="mt-1"
          />
        </div>
      </div>

      <!-- Department & Jurisdiction -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <Label for="department">Department/Agency</Label>
          <Input
            id="department"
            name="department"
            type="text"
            placeholder="District Attorney's Office"
            bind:value={formData.department}
            disabled={isLoading}
            required
            class="mt-1"
          />
        </div>

        <div>
          <Label for="jurisdiction">Jurisdiction</Label>
          <Input
            id="jurisdiction"
            name="jurisdiction"
            type="text"
            placeholder="Los Angeles County"
            bind:value={formData.jurisdiction}
            disabled={isLoading}
            required
            class="mt-1"
          />
        </div>
      </div>

      <!-- Password Fields -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Password -->
        <div>
          <Label for="password">Password</Label>
          <div class="relative">
            <Input
              id="password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              placeholder="Enter secure password"
              bind:value={formData.password}
              disabled={isLoading}
              required
              class="mt-1 pr-10"
            />
            <button
              type="button"
              class="absolute inset-y-0 right-0 pr-3 flex items-center"
              onclick={togglePasswordVisibility}
              disabled={isLoading}
            >
              {#if showPassword}
                <EyeOff class="h-4 w-4 text-muted-foreground" />
              {:else}
                <Eye class="h-4 w-4 text-muted-foreground" />
              {/if}
            </button>
          </div>
          {#if formData.password}
            <div class="mt-2 flex items-center gap-2">
              <div class="h-2 flex-1 bg-muted rounded">
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
        </div>

        <!-- Confirm Password -->
        <div>
          <Label for="confirmPassword">Confirm Password</Label>
          <div class="relative">
            <Input
              id="confirmPassword"
              name="confirmPassword"
              type={showConfirmPassword ? 'text' : 'password'}
              placeholder="Confirm your password"
              bind:value={formData.confirmPassword}
              disabled={isLoading}
              required
              class="mt-1 pr-10"
            />
            <button
              type="button"
              class="absolute inset-y-0 right-0 pr-3 flex items-center"
              onclick={toggleConfirmPasswordVisibility}
              disabled={isLoading}
            >
              {#if showConfirmPassword}
                <EyeOff class="h-4 w-4 text-muted-foreground" />
              {:else}
                <Eye class="h-4 w-4 text-muted-foreground" />
              {/if}
            </button>
          </div>
        </div>
      </div>

      <!-- Security Options -->
      <div class="space-y-3">
        <div class="flex items-center space-x-2">
          <input
            type="checkbox"
            id="enableTwoFactor"
            name="enableTwoFactor"
            bind:checked={formData.enableTwoFactor}
            disabled={isLoading}
            class="rounded border-border text-primary focus:ring-primary"
          />
          <Label for="enableTwoFactor" class="text-sm">
            Enable two-factor authentication (recommended for legal professionals)
          </Label>
        </div>
      </div>

      <!-- Terms and Privacy -->
      <div class="space-y-3">
        <div class="flex items-center space-x-2">
          <input
            type="checkbox"
            id="agreeToTerms"
            name="agreeToTerms"
            bind:checked={formData.agreeToTerms}
            disabled={isLoading}
            required
            class="rounded border-border text-primary focus:ring-primary"
          />
          <Label for="agreeToTerms" class="text-sm">
            I agree to the <a href="/legal/terms" class="text-primary hover:underline">Terms of Service</a>
          </Label>
        </div>

        <div class="flex items-center space-x-2">
          <input
            type="checkbox"
            id="agreeToPrivacy"
            name="agreeToPrivacy"
            bind:checked={formData.agreeToPrivacy}
            disabled={isLoading}
            required
            class="rounded border-border text-primary focus:ring-primary"
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
        disabled={isLoading}
      >
        {#if isLoading}
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
  </div>
</div>