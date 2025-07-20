<script lang="ts">
  interface RegisterForm {
    error?: string;
    name?: string;
    email?: string;
    role?: string;
    [key: string]: any;}
  import { enhance } from '$app/forms';
  import type { ActionData } from './$types';
  export let form: RegisterForm = {};

  let loading = false;
  let passwordStrength = 0;
  let password = '';
  let confirmPassword = '';

  function checkPasswordStrength(pwd: string) {
    let strength = 0;
    if (pwd.length >= 8) strength++;
    if (/[a-z]/.test(pwd)) strength++;
    if (/[A-Z]/.test(pwd)) strength++;
    if (/[0-9]/.test(pwd)) strength++;
    if (/[^a-zA-Z0-9]/.test(pwd)) strength++;
    return strength;}
  $: passwordStrength = checkPasswordStrength(password);
  $: passwordsMatch = password === confirmPassword && confirmPassword !== '';
  $: strengthText = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'][passwordStrength] || 'Very Weak';
  $: strengthColor = ['var(--del-color)', 'var(--color-orange)', 'var(--color-amber-500)', 'var(--ins-color)', 'var(--ins-color)'][passwordStrength] || 'var(--del-color)';
</script>

<svelte:head>
  <title>Legal Case Management - Register</title>
  <meta name="description" content="Create your account for the Legal Case Management System" />
</svelte:head>

<main class="container mx-auto px-4">
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 1L3 5V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V5L12 1Z"/>
          </svg>
        </div>
        <div>
          <h1>Join Legal Case Management</h1>
          <p class="container mx-auto px-4">Start Your Journey</p>
        </div>
      </div>
      
      <p class="container mx-auto px-4">
        Create your account to access powerful case management tools, 
        evidence tracking, and streamlined legal workflows designed for modern legal professionals.
      </p>
      
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <span class="container mx-auto px-4">✓</span>
          <span>Complete case management system</span>
        </div>
        <div class="container mx-auto px-4">
          <span class="container mx-auto px-4">✓</span>
          <span>Secure evidence handling</span>
        </div>
        <div class="container mx-auto px-4">
          <span class="container mx-auto px-4">✓</span>
          <span>Advanced analytics & reporting</span>
        </div>
      </div>
    </div>
    
    <div class="container mx-auto px-4">
      <article>
        <header>
          <h2>Create Your Account</h2>
        </header>
        
        <form method="POST" use:enhance={(({ formElement, formData, action, cancel, submitter }) => {
          loading = true;
          return async ({ result, update }) => {
            loading = false;
            await update();
          };
        })}>
          
          {#if form?.error}
            <div class="container mx-auto px-4" role="alert">
              <strong>Error:</strong> {form?.error}
            </div>
          {/if}

          <label for="name">
            Full Name
            <input
              type="text"
              id="name"
              name="name"
              value={form?.name || ''}
              placeholder="John Doe"
              required
              autocomplete="name"
              aria-required="true"
            />
          </label>
          
          <label for="email">
            Email Address
            <input
              type="email"
              id="email"
              name="email"
              value={form?.email || ''}
              placeholder="prosecutor@example.com"
              required
              autocomplete="email"
              aria-required="true"
            />
          </label>
          
          <label for="password">
            Password
            <input
              type="password"
              id="password"
              name="password"
              bind:value={password}
              placeholder="••••••••"
              required
              autocomplete="new-password"
              minlength="8"
              aria-required="true"
            />
            {#if password}
              <small>
                Password Strength: <span style="color: {strengthColor};">{strengthText}</span>
                <progress value={passwordStrength} max="5" style="margin-top: 0.5rem;"></progress>
              </small>
            {/if}
            <small>Must be at least 8 characters with mixed case, numbers & symbols</small>
          </label>
          
          <label for="confirmPassword">
            Confirm Password
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              bind:value={confirmPassword}
              placeholder="••••••••"
              required
              autocomplete="new-password"
              aria-required="true"
            />
            {#if confirmPassword && !passwordsMatch}
              <small style="color: var(--del-color);">Passwords do not match</small>
            {:else if confirmPassword && passwordsMatch}
              <small style="color: var(--ins-color);">Passwords match ✓</small>
            {/if}
          </label>
          
          <label for="role">
            Role
            <select id="role" name="role" required aria-required="true">
              <option value="">Select your role...</option>
              <option value="prosecutor" selected={form?.role === 'prosecutor'}>Prosecutor</option>
              <option value="investigator" selected={form?.role === 'investigator'}>Investigator</option>
              <option value="admin" selected={form?.role === 'admin'}>Administrator</option>
              <option value="analyst" selected={form?.role === 'analyst'}>Legal Analyst</option>
            </select>
          </label>
          
          <label>
            <input type="checkbox" name="terms" required />
            I agree to the <a href="/terms">Terms of Service</a> and <a href="/privacy">Privacy Policy</a>
          </label>
          
          <button type="submit" class="container mx-auto px-4" disabled={loading} aria-busy={loading}>
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>
        
        <footer>
          Already have an account? <a href="/login">Sign in here</a>
        </footer>
      </article>
    </div>
  </div>
</main>

<style>
  /* @unocss-include */
  .auth-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    min-height: 80vh;
    align-items: center;
    padding: 2rem 0;}
  .auth-info {
    padding: 2rem;}
  .auth-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;}
  .brand-icon {
    color: var(--harvard-crimson);
    flex-shrink: 0;}
  .auth-header h1 {
    font-size: 2.5rem;
    margin: 0;
    line-height: 1.2;}
  .auth-subtitle {
    color: var(--text-muted);
    margin: 0;
    font-size: 1.1rem;}
  .auth-description {
    font-size: 1.1rem;
    line-height: 1.6;
    margin-bottom: 2rem;
    color: var(--text-muted);}
  .feature-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;}
  .feature-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;}
  .feature-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    background: var(--ins-color);
    color: white;
    font-size: 0.875rem;
    font-weight: bold;
    flex-shrink: 0;}
  .auth-form {
    padding: 1rem;}
  .auth-form article {
    margin: 0;
    max-width: 400px;}
  .auth-form h2 {
    text-align: center;
    margin-bottom: 1.5rem;}
  .error-alert {
    padding: 1rem;
    margin-bottom: 1rem;
    background: var(--del-background-color);
    border: 1px solid var(--del-color);
    border-radius: var(--border-radius);
    color: var(--del-color);}
  @media (max-width: 768px) {
    .auth-layout {
      grid-template-columns: 1fr;
      gap: 2rem;}
    .auth-info {
      padding: 1rem;
      text-align: center;}
    .auth-header {
      justify-content: center;}
    .auth-header h1 {
      font-size: 2rem;}}
</style>