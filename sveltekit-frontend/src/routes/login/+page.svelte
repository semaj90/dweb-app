<script lang="ts">
  import { $props, $state } from 'svelte';
  import { goto } from '$app/navigation';
  import { browser } from '$app/environment';
  import { authService } from '$lib/stores/auth.svelte.js';
  import ModernAuthForm from '$lib/components/auth/ModernAuthForm.svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();

  // Modern authentication state management
  let showAuthDialog = $state(true);
  let authMode = $state<'login' | 'register'>('login');

  // Demo users for quick testing
  interface DemoUser {
    email: string;
    password: string;
    name: string;
    role: string;
  }

  const demoUsers: DemoUser[] = [
    { email: 'admin@prosecutor.com', password: 'password', name: 'Demo Admin', role: 'admin' },
    { email: 'prosecutor@legal.ai', password: 'password123', name: 'John Prosecutor', role: 'prosecutor' },
    { email: 'detective@legal.ai', password: 'password123', name: 'Jane Detective', role: 'investigator' },
    { email: 'analyst@legal.ai', password: 'password123', name: 'Legal Analyst', role: 'analyst' }
  ];

  // Quick demo login function using modern auth service
  async function quickLogin(demoUser: DemoUser) {
    const result = await authService.login(demoUser.email, demoUser.password);
    if (result.success) {
      await goto('/dashboard');
    }
  }

  // Handle successful authentication
  function handleAuthSuccess(user: any) {
    console.log('Authentication successful:', user);
    showAuthDialog = false;
    
    // Record authentication activity in session
    const { recordActivity } = import('$lib/stores/sessionManager.svelte.js').then(module => {
      module.recordActivity('/login', 'authentication_success', 'modern_auth_form');
    });
    
    goto('/dashboard');
  }

  // Check if already authenticated
  if (browser && authService.state.isAuthenticated) {
    goto('/dashboard');
  }
</script>

<svelte:head>
  <title>Admin Login - Legal Case Management</title>
</svelte:head>

<div class="login-container">
  <div class="login-card">
    <div class="login-header">
      <div class="logo">⚖️</div>
      <h1>Legal Case Management</h1>
      <p>AI-Powered Legal Analysis System</p>
    </div>

    <!-- Modern Authentication Dialog Component -->
    <ModernAuthForm 
      bind:mode={authMode}
      open={showAuthDialog} onopenchange={(open) => showAuthDialog = open}
      onsuccess={handleAuthSuccess}
    />

    <div class="login-footer">
      <div class="divider">
        <span>or use demo accounts</span>
      </div>

      <div class="demo-users">
        {#each demoUsers as demoUser}
          <Button
            variant="outline"
            onclick={() => quickLogin(demoUser)}
            class="demo-user-button w-full justify-start"
            title="Click to login as {demoUser.name}"
          >
            <div class="demo-user-info">
              <span class="demo-name">{demoUser.name}</span>
              <span class="demo-role">{demoUser.role}</span>
            </div>
            <span class="demo-email">{demoUser.email}</span>
          </Button>
        {/each}
      </div>

      <div class="auth-mode-toggle">
        <Button 
          variant="ghost" 
          onclick={() => { authMode = authMode === 'login' ? 'register' : 'login'; showAuthDialog = true; }}
        >
          {authMode === 'login' ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
        </Button>
      </div>

      <p class="login-links">
        <a href="/">← Back to Home</a>
        <a href="/dashboard">Go to Dashboard</a>
      </p>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .login-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 2rem;
  }

  .login-card {
    background: white;
    border-radius: 1rem;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 400px;
    padding: 2rem;
  }

  .login-header {
    text-align: center;
    margin-bottom: 2rem;
  }

  .logo {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .login-header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
  }

  .login-header p {
    color: #6b7280;
    font-size: 0.875rem;
  }

  .login-footer {
    margin-top: 2rem;
  }

  .divider {
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
  }

  .divider::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: #e5e7eb;
  }

  .divider span {
    background: white;
    color: #6b7280;
    padding: 0 1rem;
    font-size: 0.875rem;
  }

  .demo-users {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }

  :global(.demo-user-button) {
    text-align: left !important;
    height: auto !important;
    padding: 1rem !important;
    justify-content: flex-start !important;
  }

  :global(.demo-user-button:hover) {
    transform: translateY(-1px);
  }

  .auth-mode-toggle {
    text-align: center;
    margin: 1rem 0;
  }

  .demo-user-info {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
  }

  .demo-name {
    font-weight: 600;
    color: #1f2937;
  }

  .demo-role {
    font-size: 0.75rem;
    background: #dbeafe;
    color: #1e40af;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
    text-transform: capitalize;
  }

  .demo-email {
    font-size: 0.75rem;
    color: #6b7280;
    font-family: monospace;
  }


  .login-links {
    display: flex;
    justify-content: space-between;
    margin-top: 1rem;
  }

  .login-links a {
    color: #3b82f6;
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .login-links a:hover {
    text-decoration: underline;
  }

  @media (max-width: 480px) {
    .login-container {
      padding: 1rem;
    }

    .login-card {
      padding: 1.5rem;
    }
  }
</style>
