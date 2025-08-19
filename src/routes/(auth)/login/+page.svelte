<!-- YoRHa Login Page -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  
  let email = '';
  let password = '';
  let rememberMe = false;
  let isLoading = false;
  let error = '';
  let showPassword = false;
  let typewriterText = '';
  
  const fullText = 'ACCESS YORHA NETWORK...';
  
  // YoRHa colors
  const colors = {
    bg: '#D4D3A7',
    text: '#454138',
    accent: '#BAA68C',
    border: '#8B8680',
    highlight: '#CDC8B0',
    error: '#8B4513',
    success: '#6B7353'
  };
  
  onMount(() => {
    // Typewriter effect
    let index = 0;
    const interval = setInterval(() => {
      if (index < fullText.length) {
        typewriterText = fullText.substring(0, index + 1);
        index++;
      } else {
        clearInterval(interval);
      }
    }, 50);
    
    return () => clearInterval(interval);
  });
  
  async function handleLogin() {
    if (!email || !password) {
      error = 'Please enter email and password';
      return;
    }
    
    isLoading = true;
    error = '';
    
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email,
          password
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.success) {
        // Redirect to dashboard or previous page
        const redirectTo = $page.url.searchParams.get('redirectTo') || '/dashboard';
        goto(redirectTo);
      } else {
        error = data.error || 'Login failed';
      }
    } catch (err) {
      error = 'Network error. Please try again.';
    } finally {
      isLoading = false;
    }
  }
  
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      handleLogin();
    }
  }
</script>

<div class="login-container" style="background-color: {colors.bg}">
  <!-- Scan lines overlay -->
  <div class="scan-lines"></div>
  
  <div class="login-box" style="background-color: {colors.highlight}; border-color: {colors.border}">
    <!-- Header -->
    <div class="login-header">
      <div class="icon-container" style="border-color: {colors.border}">
        <span class="icon">🔐</span>
      </div>
      <h1 class="title" style="color: {colors.text}">
        {typewriterText}<span class="cursor">_</span>
      </h1>
      <p class="subtitle" style="color: {colors.text}">YoRHa Network Interface v13.2.7</p>
    </div>
    
    <!-- Login Form -->
    <div class="login-form">
      {#if error}
        <div class="error-message" style="background-color: {colors.error}; color: {colors.bg}">
          <span class="error-icon">⚠️</span>
          {error}
        </div>
      {/if}
      
      <div class="input-group">
        <label for="email" style="color: {colors.text}">EMAIL / UNIT ID</label>
        <div class="input-wrapper" style="border-color: {colors.border}; background-color: {colors.bg}">
          <span class="input-icon">📧</span>
          <input 
            id="email"
            type="email"
            bind:value={email}
            onkeydown={handleKeydown}
            placeholder="unit@yorha.net"
            disabled={isLoading}
            style="color: {colors.text}"
          />
        </div>
      </div>
      
      <div class="input-group">
        <label for="password" style="color: {colors.text}">ACCESS CODE</label>
        <div class="input-wrapper" style="border-color: {colors.border}; background-color: {colors.bg}">
          <span class="input-icon">🔑</span>
          <input 
            id="password"
            type={showPassword ? 'text' : 'password'}
            bind:value={password}
            onkeydown={handleKeydown}
            placeholder="••••••••"
            disabled={isLoading}
            style="color: {colors.text}"
          />
          <button 
            type="button"
            class="toggle-password"
            onclick={() => showPassword = !showPassword}
            style="color: {colors.text}"
          >
            {showPassword ? '👁️' : '👁️‍🗨️'}
          </button>
        </div>
      </div>
      
      <div class="options">
        <label class="remember-me" style="color: {colors.text}">
          <input type="checkbox" bind:checked={rememberMe} />
          <span>Remember Unit</span>
        </label>
        <a href="/forgot-password" class="forgot-link" style="color: {colors.text}">
          Forgot Access Code?
        </a>
      </div>
      
      <button 
        class="login-button"
        onclick={handleLogin}
        disabled={isLoading}
        style="background-color: {colors.text}; color: {colors.bg}"
      >
        {#if isLoading}
          <span class="loading">AUTHENTICATING...</span>
        {:else}
          ACCESS SYSTEM
        {/if}
      </button>
      
      <div class="divider" style="border-color: {colors.border}">
        <span style="background-color: {colors.highlight}; color: {colors.text}">OR</span>
      </div>
      
      <a 
        href="/register" 
        class="register-link"
        style="border-color: {colors.border}; color: {colors.text}"
      >
        REGISTER NEW UNIT
      </a>
    </div>
    
    <!-- Footer -->
    <div class="login-footer" style="color: {colors.text}">
      <p>FOR THE GLORY OF MANKIND</p>
    </div>
  </div>
</div>

<style>
  .login-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    position: relative;
  }
  
  .scan-lines {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    opacity: 0.05;
    background-image: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0, 0, 0, 0.3) 2px,
      rgba(0, 0, 0, 0.3) 4px
    );
    z-index: 1;
  }
  
  .login-box {
    width: 100%;
    max-width: 400px;
    border: 2px solid;
    padding: 2rem;
    position: relative;
    z-index: 2;
  }
  
  .login-header {
    text-align: center;
    margin-bottom: 2rem;
  }
  
  .icon-container {
    display: inline-block;
    padding: 1rem;
    border: 2px solid;
    margin-bottom: 1rem;
  }
  
  .icon {
    font-size: 2rem;
  }
  
  .title {
    font-size: 1.5rem;
    font-weight: bold;
    font-family: monospace;
    margin: 0 0 0.5rem 0;
  }
  
  .cursor {
    animation: blink 1s infinite;
  }
  
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
  
  .subtitle {
    font-size: 0.75rem;
    font-family: monospace;
    opacity: 0.7;
  }
  
  .login-form {
    margin-bottom: 2rem;
  }
  
  .error-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }
  
  .error-icon {
    font-size: 1rem;
  }
  
  .input-group {
    margin-bottom: 1rem;
  }
  
  .input-group label {
    display: block;
    font-size: 0.75rem;
    font-family: monospace;
    margin-bottom: 0.5rem;
    letter-spacing: 0.1em;
  }
  
  .input-wrapper {
    display: flex;
    align-items: center;
    border: 1px solid;
    padding: 0;
    transition: all 0.2s;
  }
  
  .input-wrapper:focus-within {
    transform: translate(1px, 1px);
  }
  
  .input-icon {
    padding: 0.75rem;
    font-size: 1rem;
  }
  
  .input-wrapper input {
    flex: 1;
    padding: 0.75rem 0;
    border: none;
    background: transparent;
    outline: none;
    font-family: monospace;
    font-size: 0.875rem;
  }
  
  .toggle-password {
    padding: 0.75rem;
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 1rem;
  }
  
  .options {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }
  
  .remember-me {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    cursor: pointer;
  }
  
  .forgot-link {
    font-size: 0.875rem;
    text-decoration: underline;
  }
  
  .login-button {
    width: 100%;
    padding: 1rem;
    border: none;
    font-family: monospace;
    font-weight: bold;
    font-size: 0.875rem;
    letter-spacing: 0.1em;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .login-button:hover:not(:disabled) {
    transform: translate(1px, 1px);
  }
  
  .login-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .loading {
    animation: pulse 1s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .divider {
    position: relative;
    text-align: center;
    margin: 1.5rem 0;
    border-top: 1px solid;
  }
  
  .divider span {
    position: absolute;
    top: -0.75rem;
    left: 50%;
    transform: translateX(-50%);
    padding: 0 1rem;
    font-size: 0.75rem;
    font-family: monospace;
  }
  
  .register-link {
    display: block;
    width: 100%;
    padding: 1rem;
    border: 1px solid;
    text-align: center;
    text-decoration: none;
    font-family: monospace;
    font-size: 0.875rem;
    transition: all 0.2s;
  }
  
  .register-link:hover {
    transform: translate(1px, 1px);
  }
  
  .login-footer {
    text-align: center;
    font-size: 0.75rem;
    font-family: monospace;
    opacity: 0.7;
  }
  
  .login-footer p {
    margin: 0;
  }
</style>