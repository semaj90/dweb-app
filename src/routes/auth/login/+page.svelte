<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { enhance } from '$app/forms';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import type { ActionData } from './$types';

  interface Props {
    form?: ActionData;
  }

  let { form }: Props = $props();

  let isSubmitting = $state(false);
  let formData = $state({
    email: '',
    password: '',
    rememberMe: false
  });

  // Check for registration success message
  let showRegistrationSuccess = $state(false);
  $effect(() => {
    if ($page.url.searchParams.get('registered') === 'true') {
      showRegistrationSuccess = true;
    }
  });

  // Demo credentials
  const demoCredentials = [
    { email: 'admin@legal.ai', password: 'admin123', role: 'Administrator' },
    { email: 'prosecutor@legal.ai', password: 'prosecutor123', role: 'Prosecutor' },
    { email: 'detective@legal.ai', password: 'detective123', role: 'Detective' },
    { email: 'user@legal.ai', password: 'user123', role: 'General User' }
  ];

  function handleSubmit() {
    isSubmitting = true;
  }

  function goBack() {
    goto('/yorha-demo');
  }

  function fillDemoCredentials(email: string, password: string) {
    formData.email = email;
    formData.password = password;
  }
</script>

<svelte:head>
  <title>User Login - Legal AI System</title>
  <meta name="description" content="Sign in to the Legal AI Case Management System" />
</svelte:head>

<div class="auth-container">
  <div class="auth-background"></div>
  
  <div class="auth-content">
    <!-- Back Button -->
    <Button variant="ghost" class="back-button" onclick={goBack}>
      ← Back to Demo Navigation
    </Button>

    <!-- Registration Success Alert -->
    {#if showRegistrationSuccess}
      <div class="alert alert-success">
        ✅ Registration successful! You can now sign in with your credentials.
      </div>
    {/if}

    <Card class="auth-card">
      <CardHeader class="auth-header">
        <CardTitle class="auth-title">User Login</CardTitle>
        <CardDescription class="auth-description">
          Sign in to access the Legal AI Case Management System
        </CardDescription>
      </CardHeader>

      <CardContent class="auth-form-content">
        <form method="POST" use:enhance={() => {
          handleSubmit();
          return async ({ result, update }) => {
            isSubmitting = false;
            if (result.type === 'success') {
              goto('/cases');
            }
            await update();
          };
        }}>
          <div class="form-grid">
            <div class="input-group">
              <Label for="email" class="yorha-label">Email Address</Label>
              <Input 
                id="email" 
                name="email" 
                type="email" 
                required 
                class="yorha-input"
                bind:value={formData.email}
                placeholder="your.email@example.com"
              />
              {#if form?.fieldErrors?.email}
                <span class="error-message">{form.fieldErrors.email}</span>
              {/if}
            </div>

            <div class="input-group">
              <Label for="password" class="yorha-label">Password</Label>
              <Input 
                id="password" 
                name="password" 
                type="password" 
                required 
                class="yorha-input"
                bind:value={formData.password}
                placeholder="Enter your password"
              />
              {#if form?.fieldErrors?.password}
                <span class="error-message">{form.fieldErrors.password}</span>
              {/if}
            </div>

            <div class="checkbox-group">
              <label class="checkbox-label">
                <input 
                  type="checkbox" 
                  name="rememberMe"
                  class="yorha-checkbox"
                  bind:checked={formData.rememberMe}
                />
                <span class="checkbox-text">Remember me for 30 days</span>
              </label>
            </div>
          </div>

          <!-- Error Display -->
          {#if form?.message}
            <div class="alert {form.type === 'error' ? 'alert-error' : 'alert-success'}">
              {form.message}
            </div>
          {/if}

          <!-- Submit Button -->
          <div class="form-actions">
            <Button 
              type="submit" 
              class="login-button" 
              disabled={isSubmitting}
              size="lg"
            >
              {#if isSubmitting}
                <span class="loading-spinner"></span>
                Signing In...
              {:else}
                Sign In
              {/if}
            </Button>

            <div class="auth-links">
              <a href="/auth/forgot-password" class="auth-link">Forgot your password?</a>
              <p class="link-text">
                Don't have an account? 
                <a href="/auth/register" class="auth-link">Create one here</a>
              </p>
            </div>
          </div>
        </form>
      </CardContent>
    </Card>

    <!-- Demo Credentials -->
    <Card class="demo-card">
      <CardHeader>
        <CardTitle class="demo-title">Demo Credentials</CardTitle>
        <CardDescription>Click any credential set to auto-fill the form</CardDescription>
      </CardHeader>
      <CardContent>
        <div class="demo-grid">
          {#each demoCredentials as cred, index}
            <Button
              variant="outline"
              size="sm"
              class="demo-credential-button"
              onclick={() => fillDemoCredentials(cred.email, cred.password)}
            >
              <div class="cred-info">
                <span class="cred-role">{cred.role}</span>
                <span class="cred-email">{cred.email}</span>
                <span class="cred-password">Password: {cred.password}</span>
              </div>
            </Button>
          {/each}
        </div>
      </CardContent>
    </Card>

    <!-- System Information -->
    <div class="system-info">
      <div class="info-grid">
        <div class="info-item">
          <span class="info-icon">🔒</span>
          <span class="info-text">Secure Authentication</span>
        </div>
        <div class="info-item">
          <span class="info-icon">🤖</span>
          <span class="info-text">AI-Powered Legal Tools</span>
        </div>
        <div class="info-item">
          <span class="info-icon">⚡</span>
          <span class="info-text">GPU-Accelerated Processing</span>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .auth-container {
    min-height: 100vh;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    font-family: 'Rajdhani', 'Roboto Mono', monospace;
  }

  .auth-background {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    z-index: -2;
  }

  .auth-background::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 80%, rgba(214, 158, 46, 0.05) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(68, 200, 245, 0.05) 0%, transparent 50%);
    z-index: -1;
  }

  .auth-content {
    width: 100%;
    max-width: 500px;
    position: relative;
    z-index: 10;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .back-button {
    color: #a0aec0;
    border: 1px solid #4a5568;
    background: rgba(26, 32, 44, 0.8);
    backdrop-filter: blur(10px);
    align-self: flex-start;
  }

  .back-button:hover {
    color: #d69e2e;
    border-color: #d69e2e;
  }

  /* Auth Card */
  :global(.auth-card) {
    background: rgba(45, 55, 72, 0.9);
    border: 1px solid rgba(214, 158, 46, 0.3);
    border-radius: 0;
    backdrop-filter: blur(15px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
  }

  .auth-header {
    text-align: center;
    padding: 2rem 2rem 1rem;
    border-bottom: 1px solid rgba(214, 158, 46, 0.2);
  }

  :global(.auth-title) {
    color: #d69e2e;
    font-size: 2rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(214, 158, 46, 0.3);
  }

  :global(.auth-description) {
    color: #a0aec0;
    font-size: 1.125rem;
    line-height: 1.6;
  }

  .auth-form-content {
    padding: 2rem;
  }

  /* Form Layout */
  .form-grid {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  /* YoRHa Form Elements */
  :global(.yorha-label) {
    color: #e2e8f0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.875rem;
  }

  :global(.yorha-input) {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    border-radius: 0;
    padding: 0.75rem 1rem;
    font-family: inherit;
    transition: all 0.3s ease;
  }

  :global(.yorha-input:focus) {
    border-color: #d69e2e;
    box-shadow: 0 0 0 2px rgba(214, 158, 46, 0.2);
  }

  :global(.yorha-input::placeholder) {
    color: #6b7280;
  }

  .checkbox-group {
    margin-top: 0.5rem;
  }

  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    cursor: pointer;
  }

  .yorha-checkbox {
    width: 18px;
    height: 18px;
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    border-radius: 0;
    transition: all 0.3s ease;
  }

  .yorha-checkbox:checked {
    background: #d69e2e;
    border-color: #d69e2e;
  }

  .checkbox-text {
    color: #a0aec0;
    font-size: 0.875rem;
    user-select: none;
  }

  /* Alerts */
  .alert {
    padding: 1rem;
    border-radius: 0;
    margin-bottom: 1rem;
    font-weight: 500;
  }

  .alert-error {
    background: rgba(245, 101, 101, 0.1);
    border: 1px solid #f56565;
    color: #f56565;
  }

  .alert-success {
    background: rgba(104, 211, 145, 0.1);
    border: 1px solid #68d391;
    color: #68d391;
  }

  /* Error Messages */
  .error-message {
    color: #f56565;
    font-size: 0.875rem;
    font-weight: 500;
  }

  /* Form Actions */
  .form-actions {
    margin-top: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  :global(.login-button) {
    width: 100%;
    background: linear-gradient(135deg, #d69e2e 0%, #ed8936 100%);
    border: none;
    color: #1a202c;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 1rem 2rem;
    min-height: 3rem;
    border-radius: 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }

  :global(.login-button:hover) {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(214, 158, 46, 0.3);
  }

  :global(.login-button:disabled) {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .loading-spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid #1a202c;
    border-radius: 50%;
    border-top-color: transparent;
    animation: spin 1s linear infinite;
    margin-right: 0.5rem;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .auth-links {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  .link-text {
    color: #a0aec0;
    font-size: 0.95rem;
    margin: 0;
  }

  .auth-link {
    color: #d69e2e;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.3s ease;
  }

  .auth-link:hover {
    color: #ed8936;
    text-decoration: underline;
  }

  /* Demo Credentials Card */
  :global(.demo-card) {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid rgba(74, 85, 104, 0.5);
    border-radius: 0;
    backdrop-filter: blur(10px);
  }

  :global(.demo-title) {
    color: #68d391;
    font-size: 1.25rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .demo-grid {
    display: grid;
    gap: 0.75rem;
  }

  :global(.demo-credential-button) {
    background: rgba(45, 55, 72, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    border-radius: 0;
    padding: 1rem;
    text-align: left;
    transition: all 0.3s ease;
  }

  :global(.demo-credential-button:hover) {
    border-color: #68d391;
    background: rgba(104, 211, 145, 0.1);
  }

  .cred-info {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .cred-role {
    font-weight: 600;
    color: #68d391;
    text-transform: uppercase;
    font-size: 0.875rem;
  }

  .cred-email {
    color: #a0aec0;
    font-size: 0.875rem;
  }

  .cred-password {
    color: #6b7280;
    font-size: 0.75rem;
    font-family: monospace;
  }

  /* System Information */
  .system-info {
    padding: 1.5rem;
    background: rgba(26, 32, 44, 0.6);
    border: 1px solid rgba(74, 85, 104, 0.5);
    backdrop-filter: blur(10px);
  }

  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .info-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .info-icon {
    font-size: 1.25rem;
  }

  .info-text {
    color: #a0aec0;
    font-size: 0.875rem;
    font-weight: 500;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .auth-container {
      padding: 1rem;
    }

    .auth-content {
      max-width: 100%;
    }

    .auth-form-content {
      padding: 1.5rem;
    }

    .info-grid {
      grid-template-columns: 1fr;
    }
  }
</style>