<!--
  NES.css Retro Gaming Authentication Modal
  Integrates with existing SvelteKit auth system
-->
<script lang="ts">
  import { enhance } from '$app/forms';
  import { createEventDispatcher } from 'svelte';
  
  interface Props {
    isOpen?: boolean;
    form?: any;
  }
  
  let { isOpen = $bindable(false), form }: Props = $props();
  
  const dispatch = createEventDispatcher();
  
  let activeForm = $state<'signin' | 'signup'>('signin');
  let isLoading = $state(false);
  let showPassword = $state(false);
let modalElement = $state<HTMLDialogElement;
  
  // Watch for modal state changes
  $effect(() >(> {
    if (modalElement) {
      if (isOpen) {
        modalElement.showModal());
      } else {
        modalElement.close();
      }
    }
  });
  
  function switchForm(formType: 'signin' | 'signup') {
    activeForm = formType;
  }
  
  function closeModal() {
    isOpen = false;
    dispatch('close');
  }
  
  function fillDemoCredentials() {
    const emailInput = document.getElementById('nes_signin_email') as HTMLInputElement;
    const passwordInput = document.getElementById('nes_signin_password') as HTMLInputElement;
    
    if (emailInput && passwordInput) {
      emailInput.value = 'admin@legal-ai.local';
      passwordInput.value = 'admin123';
    }
  }
  
  function fillDemoSignupCredentials() {
    const firstNameInput = document.getElementById('nes_signup_firstname') as HTMLInputElement;
    const lastNameInput = document.getElementById('nes_signup_lastname') as HTMLInputElement;
    const emailInput = document.getElementById('nes_signup_email') as HTMLInputElement;
    const passwordInput = document.getElementById('nes_signup_password') as HTMLInputElement;
    const roleSelect = document.getElementById('nes_signup_role') as HTMLSelectElement;
    
    if (firstNameInput) firstNameInput.value = 'Demo';
    if (lastNameInput) lastNameInput.value = 'User';
    if (emailInput) emailInput.value = 'demo@legal-ai.local';
    if (passwordInput) passwordInput.value = 'demo123';
    if (roleSelect) roleSelect.value = 'attorney';
  }
  
  // Handle escape key
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      closeModal();
    }
  }
</script>

<svelte:head>
  <!-- Import nes.css and font -->
  <link href="https://unpkg.com/nes.css@latest/css/nes.min.css" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/css?family=Press+Start+2P" rel="stylesheet">
</svelte:head>

<!-- The Dialog Modal -->
<dialog 
  bind:this={modalElement}
  class="nes-dialog"
  id="nes-auth-modal"
  keydown={handleKeydown}
  aria-labelledby="auth-modal-title"
  aria-modal="true"
>
  <div class="modal-content">
    <!-- Modal title for accessibility -->
    <h2 id="auth-modal-title" class="modal-title">Authentication</h2>
    
    <!-- Tab buttons to switch between forms -->
    <div class="auth-tabs" role="tablist">
      <button 
        class="tab-btn {activeForm === 'signin' ? 'is-active' : ''}" 
        on:onclick={() => switchForm('signin')}
        disabled={isLoading}
        role="tab"
        aria-selected={activeForm === 'signin'}
        aria-controls="signin-panel"
        id="signin-tab"
      >
        Sign In
      </button>
      <button 
        class="tab-btn {activeForm === 'signup' ? 'is-active' : ''}" 
        on:onclick={() => switchForm('signup')}
        disabled={isLoading}
        role="tab"
        aria-selected={activeForm === 'signup'}
        aria-controls="signup-panel"
        id="signup-tab"
      >
        Sign Up
      </button>
    </div>

    <!-- Error Display -->
    {#if form?.error}
      <div class="nes-container is-dark with-title" style="margin-bottom: 1rem;">
        <p class="title">Error</p>
        <p>{form.error}</p>
      </div>
    {/if}

    <!-- Sign In Form -->
    {#if activeForm === 'signin'}
      <div 
        id="signin-panel"
        role="tabpanel"
        aria-labelledby="signin-tab"
        aria-hidden={activeForm !== 'signin'}
      >
        <form 
          method="POST" 
          action="/auth/login?/login"
          use:enhance={({ formData, cancel }) => {
            isLoading = true;
            return async ({ result }) => {
              isLoading = false;
              if (result.type === 'redirect') {
                closeModal();
              }
            };
          }}
          class="auth-form"
        >
          <div class="nes-field">
            <label for="nes_signin_email">Email</label>
            <input 
              type="email" 
              id="nes_signin_email" 
              name="email"
              class="nes-input" 
              placeholder="admin@legal-ai.local"
              required
              disabled={isLoading}
            />
          </div>
          
          <div class="nes-field" style="margin-top: 1rem;">
            <label for="nes_signin_password">Password</label>
            <input 
              type={showPassword ? 'text' : 'password'}
              id="nes_signin_password" 
              name="password"
              class="nes-input" 
              placeholder="Enter your password"
              required
              disabled={isLoading}
            />
          </div>

          <!-- Remember me checkbox -->
          <div style="margin-top: 1rem;">
            <label>
              <input 
                type="checkbox" 
                name="rememberMe"
                class="nes-checkbox" 
                disabled={isLoading}
              />
              <span>Remember me</span>
            </label>
          </div>

          <!-- Show password checkbox -->
          <div style="margin-top: 0.5rem;">
            <label>
              <input 
                type="checkbox" 
                class="nes-checkbox" 
                bind:checked={showPassword}
                disabled={isLoading}
              />
              <span>Show password</span>
            </label>
          </div>

          <!-- Demo credentials button -->
          <div style="margin-top: 1rem;">
            <button 
              type="button"
              class="nes-btn is-warning"
              on:onclick={fillDemoCredentials}
              disabled={isLoading}
              style="width: 100%;"
            >
              🎯 Fill Demo Login (admin@legal-ai.local)
            </button>
          </div>

          <div class="form-actions">
            <button 
              type="button"
              class="nes-btn"
              on:onclick={closeModal}
              disabled={isLoading}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              class="nes-btn is-primary"
              disabled={isLoading}
            >
              {#if isLoading}
                Signing In...
              {:else}
                Sign In
              {/if}
            </button>
          </div>
        </form>
      </div>
    {/if}

    <!-- Sign Up Form -->
    {#if activeForm === 'signup'}
      <div 
        id="signup-panel"
        role="tabpanel"
        aria-labelledby="signup-tab"
        aria-hidden={activeForm !== 'signup'}
      >
        <form 
          method="POST" 
          action="/auth/register?/register"
          use:enhance={({ formData, cancel }) => {
            isLoading = true;
            return async ({ result }) => {
              isLoading = false;
              if (result.type === 'redirect') {
                closeModal();
              }
            };
          }}
          class="auth-form"
        >
          <div class="nes-field">
            <label for="nes_signup_firstname">First Name</label>
            <input 
              type="text" 
              id="nes_signup_firstname" 
              name="firstName"
              class="nes-input" 
              placeholder="John"
              required
              disabled={isLoading}
            />
          </div>

          <div class="nes-field" style="margin-top: 1rem;">
            <label for="nes_signup_lastname">Last Name</label>
            <input 
              type="text" 
              id="nes_signup_lastname" 
              name="lastName"
              class="nes-input" 
              placeholder="Doe"
              required
              disabled={isLoading}
            />
          </div>
          
          <div class="nes-field" style="margin-top: 1rem;">
            <label for="nes_signup_email">Email</label>
            <input 
              type="email" 
              id="nes_signup_email" 
              name="email"
              class="nes-input" 
              placeholder="john@example.com"
              required
              disabled={isLoading}
            />
          </div>
          
          <div class="nes-field" style="margin-top: 1rem;">
            <label for="nes_signup_password">Password</label>
            <input 
              type={showPassword ? 'text' : 'password'}
              id="nes_signup_password" 
              name="password"
              class="nes-input" 
              placeholder="Choose a secure password"
              required
              disabled={isLoading}
            />
          </div>

          <div class="nes-field" style="margin-top: 1rem;">
            <label for="nes_signup_role">Role</label>
            <div class="nes-select">
              <select name="role" id="nes_signup_role" required disabled={isLoading}>
                <option value="">Choose your role</option>
                <option value="attorney">Attorney</option>
                <option value="paralegal">Paralegal</option>
                <option value="investigator">Investigator</option>
                <option value="admin">Administrator</option>
              </select>
            </div>
          </div>

          <!-- Show password checkbox -->
          <div style="margin-top: 1rem;">
            <label>
              <input 
                type="checkbox" 
                class="nes-checkbox" 
                bind:checked={showPassword}
                disabled={isLoading}
              />
              <span>Show password</span>
            </label>
          </div>

          <!-- Demo credentials button for Sign Up -->
          <div style="margin-top: 1rem;">
            <button 
              type="button"
              class="nes-btn is-warning"
              on:onclick={fillDemoSignupCredentials}
              disabled={isLoading}
              style="width: 100%;"
            >
              🎯 Fill Demo Registration Data
            </button>
          </div>

          <div class="form-actions">
            <button 
              type="button"
              class="nes-btn"
              on:onclick={closeModal}
              disabled={isLoading}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              class="nes-btn is-success"
              disabled={isLoading}
            >
              {#if isLoading}
                Creating...
              {:else}
                Create Account
              {/if}
            </button>
          </div>
        </form>
      </div>
    {/if}
  </div>
</dialog>

<style>
  /* Basic page styling for retro NES theme */
  :global(#nes-auth-modal) {
    max-width: 90vw;
    width: 500px;
    font-family: 'Press Start 2P', cursive;
    background-color: #212529;
    color: #fff;
    border: 4px solid #fff;
  }

  /* Style for the dialog backdrop */
  :global(#nes-auth-modal::backdrop) {
    background-color: rgba(0, 0, 0, 0.8);
  }

  .modal-content {
    padding: 1rem;
  }

  .modal-title {
    font-size: 1rem;
    margin-bottom: 1rem;
    text-align: center;
    color: #fff;
  }

  /* Custom styles for the auth modal */
  .auth-tabs {
    display: flex;
    border-bottom: 4px solid #212529;
    margin-bottom: 1.5rem;
  }

  .tab-btn {
    flex-grow: 1;
    padding: 1rem;
    background-color: #929292; /* Inactive tab color */
    border: none;
    font-family: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    border-top: 4px solid #212529;
    border-left: 4px solid #212529;
    border-right: 4px solid #212529;
    color: #000;
  }

  .tab-btn.is-active {
    background-color: #fff; /* Active tab color */
    border-bottom: 4px solid #fff;
    margin-bottom: -4px;
  }

  .tab-btn:disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }

  .auth-form {
    text-align: left;
  }

  .form-actions {
    text-align: right;
    margin-top: 1.5rem;
    display: flex;
    gap: 1rem;
    justify-content: flex-end;
  }

  /* NES.css overrides for dark theme */
  :global(#nes-auth-modal .nes-input) {
    background-color: #333;
    border-color: #666;
    color: #fff;
  }

  :global(#nes-auth-modal .nes-input:focus) {
    border-color: #0084ff;
    box-shadow: 0 0 0 3px rgba(0, 132, 255, 0.2);
  }

  :global(#nes-auth-modal .nes-select select) {
    background-color: #333;
    border-color: #666;
    color: #fff;
  }

  :global(#nes-auth-modal .nes-container.is-dark) {
    background-color: #d32f2f;
    border-color: #b71c1c;
  }

  :global(#nes-auth-modal label) {
    color: #fff;
    font-size: 0.7rem;
    margin-bottom: 0.5rem;
    display: block;
  }

  :global(#nes-auth-modal .nes-checkbox + span) {
    font-size: 0.7rem;
  }

  /* Demo button styling */
  :global(#nes-auth-modal .nes-btn.is-warning) {
    background-color: #f39c12;
    border-color: #e67e22;
    animation: glow 2s ease-in-out infinite alternate;
  }

  :global(#nes-auth-modal .nes-btn.is-warning:hover) {
    background-color: #e67e22;
    transform: scale(1.02);
  }

  @keyframes glow {
    from {
      box-shadow: 0 0 5px #f39c12;
    }
    to {
      box-shadow: 0 0 10px #f39c12, 0 0 15px #f39c12;
    }
  }

  /* Responsive adjustments */
  @media (max-width: 640px) {
    :global(#nes-auth-modal) {
      width: 95vw;
      margin: 0;
    }

    .form-actions {
      flex-direction: column;
      gap: 0.5rem;
    }

    .tab-btn {
      font-size: 0.6rem;
      padding: 0.75rem;
    }
  }
</style>