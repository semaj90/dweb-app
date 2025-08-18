<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { enhance } from '$app/forms';
  import { goto } from '$app/navigation';
  import type { ActionData } from './$types';

  interface Props {
    form?: ActionData;
  }

  let { form }: Props = $props();

  let isSubmitting = $state(false);
  let formData = $state({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    role: 'user'
  });

  const userRoles = [
    { value: 'user', label: 'General User' },
    { value: 'detective', label: 'Detective' },
    { value: 'prosecutor', label: 'Prosecutor' },
    { value: 'admin', label: 'Administrator' }
  ];

  function handleSubmit() {
    isSubmitting = true;
  }

  function goBack() {
    goto('/yorha-demo');
  }
</script>

<svelte:head>
  <title>User Registration - Legal AI System</title>
  <meta name="description" content="Register for the Legal AI Case Management System" />
</svelte:head>

<div class="auth-container">
  <div class="auth-background"></div>
  
  <div class="auth-content">
    <!-- Back Button -->
    <Button variant="ghost" class="back-button" onclick={goBack}>
      ← Back to Demo Navigation
    </Button>

    <Card class="auth-card">
      <CardHeader class="auth-header">
        <CardTitle class="auth-title">User Registration</CardTitle>
        <CardDescription class="auth-description">
          Create your account for the Legal AI Case Management System
        </CardDescription>
      </CardHeader>

      <CardContent class="auth-form-content">
        <form method="POST" use:enhance={() => {
          handleSubmit();
          return async ({ result, update }) => {
            isSubmitting = false;
            if (result.type === 'success') {
              goto('/auth/login?registered=true');
            }
            await update();
          };
        }}>
          <div class="form-grid">
            <!-- Personal Information -->
            <div class="form-section">
              <h3 class="section-title">Personal Information</h3>
              
              <div class="input-group">
                <Label for="firstName" class="yorha-label">First Name</Label>
                <Input 
                  id="firstName" 
                  name="firstName" 
                  type="text" 
                  required 
                  class="yorha-input"
                  bind:value={formData.firstName}
                  placeholder="Enter your first name"
                />
              </div>

              <div class="input-group">
                <Label for="lastName" class="yorha-label">Last Name</Label>
                <Input 
                  id="lastName" 
                  name="lastName" 
                  type="text" 
                  required 
                  class="yorha-input"
                  bind:value={formData.lastName}
                  placeholder="Enter your last name"
                />
              </div>
            </div>

            <!-- Account Information -->
            <div class="form-section">
              <h3 class="section-title">Account Information</h3>
              
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
                <Label for="role" class="yorha-label">User Role</Label>
                <select 
                  id="role" 
                  name="role" 
                  class="yorha-select"
                  bind:value={formData.role}
                  required
                >
                  {#each userRoles as role}
                    <option value={role.value}>{role.label}</option>
                  {/each}
                </select>
              </div>
            </div>

            <!-- Security -->
            <div class="form-section">
              <h3 class="section-title">Security</h3>
              
              <div class="input-group">
                <Label for="password" class="yorha-label">Password</Label>
                <Input 
                  id="password" 
                  name="password" 
                  type="password" 
                  required 
                  class="yorha-input"
                  bind:value={formData.password}
                  placeholder="Enter a secure password"
                />
                {#if form?.fieldErrors?.password}
                  <span class="error-message">{form.fieldErrors.password}</span>
                {/if}
              </div>

              <div class="input-group">
                <Label for="confirmPassword" class="yorha-label">Confirm Password</Label>
                <Input 
                  id="confirmPassword" 
                  name="confirmPassword" 
                  type="password" 
                  required 
                  class="yorha-input"
                  bind:value={formData.confirmPassword}
                  placeholder="Confirm your password"
                />
                {#if form?.fieldErrors?.confirmPassword}
                  <span class="error-message">{form.fieldErrors.confirmPassword}</span>
                {/if}
              </div>
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
              class="register-button" 
              disabled={isSubmitting}
              size="lg"
            >
              {#if isSubmitting}
                <span class="loading-spinner"></span>
                Creating Account...
              {:else}
                Create Account
              {/if}
            </Button>

            <div class="auth-links">
              <p class="link-text">
                Already have an account? 
                <a href="/auth/login" class="auth-link">Sign in here</a>
              </p>
            </div>
          </div>
        </form>
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
    max-width: 600px;
    position: relative;
    z-index: 10;
  }

  .back-button {
    margin-bottom: 1.5rem;
    color: #a0aec0;
    border: 1px solid #4a5568;
    background: rgba(26, 32, 44, 0.8);
    backdrop-filter: blur(10px);
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
    gap: 2rem;
  }

  .form-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .section-title {
    color: #d69e2e;
    font-size: 1.25rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(214, 158, 46, 0.2);
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

  .yorha-select {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    border-radius: 0;
    padding: 0.75rem 1rem;
    font-family: inherit;
    transition: all 0.3s ease;
  }

  .yorha-select:focus {
    border-color: #d69e2e;
    box-shadow: 0 0 0 2px rgba(214, 158, 46, 0.2);
    outline: none;
  }

  .yorha-select option {
    background: #1a202c;
    color: #e2e8f0;
  }

  /* Error Messages */
  .error-message {
    color: #f56565;
    font-size: 0.875rem;
    font-weight: 500;
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

  /* Form Actions */
  .form-actions {
    margin-top: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  :global(.register-button) {
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

  :global(.register-button:hover) {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(214, 158, 46, 0.3);
  }

  :global(.register-button:disabled) {
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
    text-align: center;
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

  /* System Information */
  .system-info {
    margin-top: 2rem;
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