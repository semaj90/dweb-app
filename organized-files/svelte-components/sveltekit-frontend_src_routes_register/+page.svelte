<script lang="ts">
  import { superForm } from 'sveltekit-superforms';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();

  const { form, errors, enhance, submitting, message } = superForm(data.form, {
    resetForm: true,
    scrollToError: 'smooth',
    errorSelector: '[data-invalid]',
    onUpdated: ({ form: f }) => {
      if (f.message) {
        console.log('Registration message:', f.message);
      }
    }
  });

  // Role options for the select dropdown
  const roleOptions = [
    { value: 'prosecutor', label: '⚖️ Prosecutor', desc: 'Lead legal proceedings and case strategy' },
    { value: 'investigator', label: '🔍 Investigator', desc: 'Gather evidence and investigate cases' },
    { value: 'admin', label: '👑 Administrator', desc: 'System management and oversight' },
    { value: 'analyst', label: '📊 Legal Analyst', desc: 'Research and analyze legal precedents' }
  ] as const;

  // Password strength calculation
  let passwordStrength = $derived.by(() => {
    if (!$form.password) return 0;
    let strength = 0;
    if ($form.password.length >= 8) strength++;
    if (/[a-z]/.test($form.password)) strength++;
    if (/[A-Z]/.test($form.password)) strength++;
    if (/[0-9]/.test($form.password)) strength++;
    if (/[^a-zA-Z0-9]/.test($form.password)) strength++;
    return strength;
  });

  let passwordsMatch = $derived($form.password === $form.confirmPassword && $form.confirmPassword !== '')

  let strengthText = $derived(['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'][passwordStrength] || 'Very Weak')
</script>

<svelte:head>
  <title>Register - Legal Case Management</title>
  <meta name="description" content="Create your account for the Legal Case Management System" />
</svelte:head>

<div class="register-container">
  <div class="register-card">
    <div class="register-header">
      <div class="logo">⚖️</div>
      <h1>Join Legal Case Management</h1>
      <p>Create your professional account</p>
    </div>

    <form method="POST" use:enhance class="register-form">
      {#if $message}
        <div class="error-message" role="alert">
          {$message}
        </div>
      {/if}

      <div class="form-group">
        <label for="name">Full Name</label>
        <input
          id="name"
          name="name"
          type="text"
          bind:value={$form.name}
          data-invalid={$errors.name}
          placeholder="John Doe"
          required
          class="form-input"
          class:error={$errors.name}
          autocomplete="name"
        />
        {#if $errors.name}
          <div class="field-error">{$errors.name}</div>
        {/if}
      </div>

      <div class="form-group">
        <label for="email">Email Address</label>
        <input
          id="email"
          name="email"
          type="email"
          bind:value={$form.email}
          data-invalid={$errors.email}
          placeholder="prosecutor@example.com"
          required
          class="form-input"
          class:error={$errors.email}
          autocomplete="email"
        />
        {#if $errors.email}
          <div class="field-error">{$errors.email}</div>
        {/if}
      </div>

      <div class="form-group">
        <label for="password">Password</label>
        <input
          id="password"
          name="password"
          type="password"
          bind:value={$form.password}
          data-invalid={$errors.password}
          placeholder="••••••••"
          required
          class="form-input"
          class:error={$errors.password}
          autocomplete="new-password"
          minlength="8"
        />
        {#if $form.password}
          <div class="password-strength">
            <div class="strength-meter">
              <div
                class="strength-fill"
                style="width: {(passwordStrength / 5) * 100}%; background-color: {
                  passwordStrength <= 1 ? '#ef4444' :
                  passwordStrength <= 2 ? '#f97316' :
                  passwordStrength <= 3 ? '#eab308' :
                  passwordStrength <= 4 ? '#22c55e' : '#16a34a'
                };"
              ></div>
            </div>
            <span class="strength-text" style="color: {
              passwordStrength <= 1 ? '#ef4444' :
              passwordStrength <= 2 ? '#f97316' :
              passwordStrength <= 3 ? '#eab308' :
              passwordStrength <= 4 ? '#22c55e' : '#16a34a'
            };">
              {strengthText}
            </span>
          </div>
          <small class="form-help">
            Use 8+ characters with uppercase, lowercase, numbers & symbols
          </small>
        {/if}
        {#if $errors.password}
          <div class="field-error">{$errors.password}</div>
        {/if}
      </div>

      <div class="form-group">
        <label for="confirmPassword">Confirm Password</label>
        <input
          id="confirmPassword"
          name="confirmPassword"
          type="password"
          bind:value={$form.confirmPassword}
          data-invalid={$errors.confirmPassword}
          placeholder="••••••••"
          required
          class="form-input"
          class:error={$errors.confirmPassword}
          autocomplete="new-password"
        />
        {#if $form.confirmPassword && !passwordsMatch}
          <div class="field-error">Passwords do not match</div>
        {:else if $form.confirmPassword && passwordsMatch}
          <div class="field-success">Passwords match ✓</div>
        {/if}
        {#if $errors.confirmPassword}
          <div class="field-error">{$errors.confirmPassword}</div>
        {/if}
      </div>

      <div class="form-group">
        <label for="role">Role</label>
        <select
          id="role"
          name="role"
          bind:value={$form.role}
          data-invalid={$errors.role}
          required
          class="form-select"
          class:error={$errors.role}
        >
          <option value="">Select your role...</option>
          {#each roleOptions as option}
            <option value={option.value}>
              {option.label}
            </option>
          {/each}
        </select>
        {#if $form.role}
          <small class="form-help">
            {roleOptions.find(opt => opt.value === $form.role)?.desc}
          </small>
        {/if}
        {#if $errors.role}
          <div class="field-error">{$errors.role}</div>
        {/if}
      </div>

      <div class="form-group">
        <label class="checkbox-label">
          <input
            type="checkbox"
            name="terms"
            bind:checked={$form.terms}
            data-invalid={$errors.terms}
            required
            class="form-checkbox"
          />
          <span class="checkbox-text">
            I agree to the <a href="/terms" target="_blank">Terms of Service</a>
            and <a href="/privacy" target="_blank">Privacy Policy</a>
          </span>
        </label>
        {#if $errors.terms}
          <div class="field-error">{$errors.terms}</div>
        {/if}
      </div>

      <button
        type="submit"
        disabled={$submitting}
        class="register-button"
      >
        {#if $submitting}
          <div class="spinner"></div>
          Creating Account...
        {:else}
          Create Account
        {/if}
      </button>
    </form>

    <div class="register-footer">
      <p class="login-link">
        Already have an account?
        <a href="/login">Sign in here</a>
      </p>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .register-container {
    min-height: 100vh;
    display: flex
    align-items: center
    justify-content: center
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 2rem;
  }

  .register-card {
    background: white
    border-radius: 1rem;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 450px;
    padding: 2rem;
  }

  .register-header {
    text-align: center
    margin-bottom: 2rem;
  }

  .logo {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .register-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
  }

  .register-header p {
    color: #6b7280;
    font-size: 0.875rem;
  }

  .register-form {
    display: flex
    flex-direction: column
    gap: 1.5rem;
  }

  .form-group {
    display: flex
    flex-direction: column
  }

  .form-group label {
    font-weight: 500;
    color: #374151;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
  }

  .form-input, .form-select {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 0.5rem;
    font-size: 1rem;
    transition: all 0.2s ease;
  }

  .form-input:focus, .form-select:focus {
    outline: none
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  .form-input.error, .form-select.error {
    border-color: #dc2626;
    box-shadow: 0 0 0 1px #dc2626;
  }

  .checkbox-label {
    display: flex
    align-items: flex-start;
    gap: 0.5rem;
    cursor: pointer
  }

  .form-checkbox {
    margin-top: 0.125rem;
  }

  .checkbox-text {
    font-size: 0.875rem;
    line-height: 1.4;
  }

  .checkbox-text a {
    color: #3b82f6;
    text-decoration: underline
  }

  .password-strength {
    margin-top: 0.5rem;
    display: flex
    align-items: center
    gap: 0.5rem;
  }

  .strength-meter {
    flex: 1;
    height: 4px;
    background: #e5e7eb;
    border-radius: 2px;
    overflow: hidden
  }

  .strength-fill {
    height: 100%;
    border-radius: 2px;
    transition: all 0.3s ease;
  }

  .strength-text {
    font-size: 0.75rem;
    font-weight: 500;
    min-width: 80px;
    text-align: right
  }

  .form-help {
    color: #6b7280;
    font-size: 0.75rem;
    margin-top: 0.25rem;
  }

  .field-error {
    color: #dc2626;
    font-size: 0.75rem;
    margin-top: 0.25rem;
  }

  .field-success {
    color: #16a34a;
    font-size: 0.75rem;
    margin-top: 0.25rem;
  }

  .error-message {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
    padding: 0.75rem;
    border-radius: 0.5rem;
    font-size: 0.875rem;
  }

  .register-button {
    width: 100%;
    background: #3b82f6;
    color: white
    border: none
    padding: 0.875rem;
    border-radius: 0.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer
    transition: background-color 0.2s ease;
    display: flex
    align-items: center
    justify-content: center
    gap: 0.5rem;
    margin-top: 1rem;
  }

  .register-button:hover:not(:disabled) {
    background: #2563eb;
  }

  .register-button:disabled {
    background: #9ca3af;
    cursor: not-allowed;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .register-footer {
    margin-top: 2rem;
    text-align: center
  }

  .login-link {
    font-size: 0.875rem;
    color: #6b7280;
  }

  .login-link a {
    color: #3b82f6;
    text-decoration: none
    font-weight: 500;
  }

  .login-link a:hover {
    text-decoration: underline
  }

  @media (max-width: 480px) {
    .register-container {
      padding: 1rem;
    }

    .register-card {
      padding: 1.5rem;
    }
  }
</style>