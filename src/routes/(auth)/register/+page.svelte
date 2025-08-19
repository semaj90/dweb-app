<!-- YoRHa Registration Page -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  
  let currentStep = 1;
  let isLoading = false;
  let error = '';
  let typewriterText = '';
  
  const fullText = 'INITIALIZE NEW UNIT...';
  
  // Form data
  let formData = {
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    unitType: 'combat',
    acceptTerms: false
  };
  
  // Password strength
  $: passwordStrength = getPasswordStrength(formData.password);
  
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
  
  function getPasswordStrength(password: string) {
    if (!password) return { level: 0, text: 'ENTER PASSWORD', color: colors.text };
    if (password.length < 6) return { level: 1, text: 'WEAK', color: colors.error };
    if (password.length < 10) return { level: 2, text: 'MEDIUM', color: colors.accent };
    if (password.length >= 10 && /[A-Z]/.test(password) && /[0-9]/.test(password) && /[^A-Za-z0-9]/.test(password)) {
      return { level: 3, text: 'STRONG', color: colors.success };
    }
    return { level: 2, text: 'MEDIUM', color: colors.accent };
  }
  
  function validateStep1() {
    error = '';
    
    if (!formData.name || formData.name.length < 2) {
      error = 'Unit name must be at least 2 characters';
      return false;
    }
    
    if (!formData.email || !formData.email.includes('@')) {
      error = 'Valid email address required';
      return false;
    }
    
    return true;
  }
  
  function validateStep2() {
    error = '';
    
    if (!formData.password || formData.password.length < 8) {
      error = 'Password must be at least 8 characters';
      return false;
    }
    
    if (formData.password !== formData.confirmPassword) {
      error = 'Passwords do not match';
      return false;
    }
    
    if (!formData.acceptTerms) {
      error = 'You must accept the YoRHa protocols';
      return false;
    }
    
    return true;
  }
  
  function nextStep() {
    if (currentStep === 1 && validateStep1()) {
      currentStep = 2;
    } else if (currentStep === 2 && validateStep2()) {
      handleRegister();
    }
  }
  
  function previousStep() {
    if (currentStep > 1) {
      currentStep--;
      error = '';
    }
  }
  
  async function handleRegister() {
    isLoading = true;
    error = '';
    
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          unitType: formData.unitType
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.success) {
        // Show success message briefly then redirect
        currentStep = 3;
        setTimeout(() => {
          goto('/dashboard');
        }, 3000);
      } else {
        error = data.error || 'Registration failed';
      }
    } catch (err) {
      error = 'Network error. Please try again.';
    } finally {
      isLoading = false;
    }
  }
  
  const unitTypes = [
    { value: 'combat', label: 'COMBAT', icon: '⚔️', description: 'Frontline combat operations' },
    { value: 'scanner', label: 'SCANNER', icon: '📡', description: 'Reconnaissance and analysis' },
    { value: 'support', label: 'SUPPORT', icon: '🛡️', description: 'Tactical support and healing' },
    { value: 'operator', label: 'OPERATOR', icon: '🎮', description: 'Remote operations control' },
    { value: 'healer', label: 'HEALER', icon: '💚', description: 'Medical and repair duties' }
  ];
</script>

<div class="register-container" style="background-color: {colors.bg}">
  <!-- Scan lines overlay -->
  <div class="scan-lines"></div>
  
  <div class="register-box" style="background-color: {colors.highlight}; border-color: {colors.border}">
    <!-- Header -->
    <div class="register-header">
      <div class="icon-container" style="border-color: {colors.border}">
        <span class="icon">📝</span>
      </div>
      <h1 class="title" style="color: {colors.text}">
        {typewriterText}<span class="cursor">_</span>
      </h1>
      <p class="subtitle" style="color: {colors.text}">YoRHa Unit Registration Protocol</p>
    </div>
    
    <!-- Progress Indicator -->
    {#if currentStep < 3}
      <div class="progress-indicator">
        {#each [1, 2] as step}
          <div class="progress-step" class:active={step <= currentStep}>
            <div 
              class="step-circle"
              style="border-color: {step <= currentStep ? colors.text : colors.border};
                     background-color: {step < currentStep ? colors.text : 'transparent'};
                     color: {step < currentStep ? colors.bg : colors.text}"
            >
              {#if step < currentStep}
                ✓
              {:else}
                {step}
              {/if}
            </div>
            {#if step < 2}
              <div 
                class="step-line"
                style="background-color: {step < currentStep ? colors.text : colors.border}"
              ></div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
    
    <!-- Registration Form -->
    <div class="register-form">
      {#if error}
        <div class="error-message" style="background-color: {colors.error}; color: {colors.bg}">
          <span class="error-icon">⚠️</span>
          {error}
        </div>
      {/if}
      
      {#if currentStep === 1}
        <!-- Step 1: Basic Information -->
        <div class="step-content">
          <h2 class="step-title" style="color: {colors.text}">BASIC INFORMATION</h2>
          
          <div class="input-group">
            <label for="name" style="color: {colors.text}">UNIT NAME</label>
            <div class="input-wrapper" style="border-color: {colors.border}; background-color: {colors.bg}">
              <span class="input-icon">👤</span>
              <input 
                id="name"
                type="text"
                bind:value={formData.name}
                placeholder="Unit 2B"
                disabled={isLoading}
                style="color: {colors.text}"
              />
            </div>
          </div>
          
          <div class="input-group">
            <label for="email" style="color: {colors.text}">NETWORK EMAIL</label>
            <div class="input-wrapper" style="border-color: {colors.border}; background-color: {colors.bg}">
              <span class="input-icon">📧</span>
              <input 
                id="email"
                type="email"
                bind:value={formData.email}
                placeholder="unit@yorha.net"
                disabled={isLoading}
                style="color: {colors.text}"
              />
            </div>
          </div>
          
          <div class="input-group">
            <label style="color: {colors.text}">UNIT TYPE</label>
            <div class="unit-types">
              {#each unitTypes as type}
                <button
                  type="button"
                  class="unit-type-btn"
                  class:selected={formData.unitType === type.value}
                  onclick={() => formData.unitType = type.value}
                  style="border-color: {colors.border};
                         background-color: {formData.unitType === type.value ? colors.text : colors.bg};
                         color: {formData.unitType === type.value ? colors.bg : colors.text}"
                >
                  <span class="type-icon">{type.icon}</span>
                  <span class="type-label">{type.label}</span>
                  <span class="type-desc">{type.description}</span>
                </button>
              {/each}
            </div>
          </div>
        </div>
        
      {:else if currentStep === 2}
        <!-- Step 2: Security Setup -->
        <div class="step-content">
          <h2 class="step-title" style="color: {colors.text}">SECURITY CONFIGURATION</h2>
          
          <div class="input-group">
            <label for="password" style="color: {colors.text}">ACCESS CODE</label>
            <div class="input-wrapper" style="border-color: {colors.border}; background-color: {colors.bg}">
              <span class="input-icon">🔑</span>
              <input 
                id="password"
                type="password"
                bind:value={formData.password}
                placeholder="••••••••"
                disabled={isLoading}
                style="color: {colors.text}"
              />
            </div>
            
            {#if formData.password}
              <div class="password-strength" style="border-color: {colors.border}; background-color: {colors.bg}">
                <div class="strength-header">
                  <span style="color: {colors.text}">SECURITY LEVEL</span>
                  <span style="color: {passwordStrength.color}">{passwordStrength.text}</span>
                </div>
                <div class="strength-bars">
                  {#each [1, 2, 3] as level}
                    <div 
                      class="strength-bar"
                      style="background-color: {level <= passwordStrength.level ? passwordStrength.color : colors.border};
                             opacity: {level <= passwordStrength.level ? 1 : 0.3}"
                    ></div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
          
          <div class="input-group">
            <label for="confirmPassword" style="color: {colors.text}">CONFIRM ACCESS CODE</label>
            <div class="input-wrapper" style="border-color: {colors.border}; background-color: {colors.bg}">
              <span class="input-icon">🛡️</span>
              <input 
                id="confirmPassword"
                type="password"
                bind:value={formData.confirmPassword}
                placeholder="••••••••"
                disabled={isLoading}
                style="color: {colors.text}"
              />
            </div>
          </div>
          
          <div class="terms-group">
            <label class="checkbox-label" style="color: {colors.text}">
              <input 
                type="checkbox" 
                bind:checked={formData.acceptTerms}
                disabled={isLoading}
              />
              <span>
                I accept the YoRHa protocols and agree to fight for the glory of mankind
              </span>
            </label>
          </div>
        </div>
        
      {:else if currentStep === 3}
        <!-- Step 3: Success -->
        <div class="success-content">
          <div class="success-icon" style="color: {colors.success}">✅</div>
          <h2 class="success-title" style="color: {colors.text}">REGISTRATION COMPLETE</h2>
          <p class="success-message" style="color: {colors.text}">
            Welcome to YoRHa Network. Your unit has been successfully registered and activated.
          </p>
          <div class="unit-details-card" style="border-color: {colors.border}; background-color: {colors.bg}">
            <div class="detail-row">
              <span style="color: {colors.text}">UNIT NAME:</span>
              <span style="color: {colors.text}">{formData.name}</span>
            </div>
            <div class="detail-row">
              <span style="color: {colors.text}">EMAIL:</span>
              <span style="color: {colors.text}">{formData.email}</span>
            </div>
            <div class="detail-row">
              <span style="color: {colors.text}">TYPE:</span>
              <span style="color: {colors.text}">{formData.unitType.toUpperCase()}</span>
            </div>
            <div class="detail-row">
              <span style="color: {colors.text}">STATUS:</span>
              <span style="color: {colors.success}">ACTIVE</span>
            </div>
          </div>
          <p class="redirect-message" style="color: {colors.text}">
            Redirecting to command center...
          </p>
        </div>
      {/if}
      
      {#if currentStep < 3}
        <div class="form-actions">
          {#if currentStep > 1}
            <button 
              class="back-button"
              onclick={previousStep}
              disabled={isLoading}
              style="border-color: {colors.border}; color: {colors.text}"
            >
              ← BACK
            </button>
          {/if}
          
          <button 
            class="continue-button"
            onclick={nextStep}
            disabled={isLoading}
            style="background-color: {colors.text}; color: {colors.bg}"
          >
            {#if isLoading}
              <span class="loading">PROCESSING...</span>
            {:else if currentStep === 2}
              COMPLETE REGISTRATION
            {:else}
              CONTINUE →
            {/if}
          </button>
        </div>
      {/if}
      
      {#if currentStep === 1}
        <div class="login-link" style="color: {colors.text}">
          Already have an account? 
          <a href="/login" style="color: {colors.text}">Access Terminal</a>
        </div>
      {/if}
    </div>
    
    <!-- Footer -->
    <div class="register-footer" style="color: {colors.text}">
      <p>FOR THE GLORY OF MANKIND</p>
    </div>
  </div>
</div>

<style>
  .register-container {
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
  
  .register-box {
    width: 100%;
    max-width: 500px;
    border: 2px solid;
    padding: 2rem;
    position: relative;
    z-index: 2;
  }
  
  .register-header {
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
  
  .progress-indicator {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 2rem;
  }
  
  .progress-step {
    display: flex;
    align-items: center;
  }
  
  .step-circle {
    width: 32px;
    height: 32px;
    border: 2px solid;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: monospace;
    font-size: 0.875rem;
    font-weight: bold;
  }
  
  .step-line {
    width: 60px;
    height: 2px;
    margin: 0 0.5rem;
  }
  
  .register-form {
    margin-bottom: 1rem;
  }
  
  .error-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }
  
  .step-title {
    font-size: 1rem;
    font-family: monospace;
    margin: 0 0 1.5rem 0;
    text-align: center;
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
  
  .unit-types {
    display: grid;
    gap: 0.5rem;
  }
  
  .unit-type-btn {
    display: grid;
    grid-template-columns: auto 1fr;
    grid-template-rows: auto auto;
    gap: 0.25rem 0.5rem;
    padding: 0.75rem;
    border: 1px solid;
    background: transparent;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
  }
  
  .unit-type-btn:hover {
    transform: translate(1px, 1px);
  }
  
  .type-icon {
    grid-row: span 2;
    font-size: 1.5rem;
    align-self: center;
  }
  
  .type-label {
    font-family: monospace;
    font-weight: bold;
    font-size: 0.875rem;
  }
  
  .type-desc {
    font-size: 0.75rem;
    opacity: 0.7;
  }
  
  .password-strength {
    margin-top: 0.5rem;
    padding: 0.75rem;
    border: 1px solid;
  }
  
  .strength-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.75rem;
    font-family: monospace;
  }
  
  .strength-bars {
    display: flex;
    gap: 0.25rem;
  }
  
  .strength-bar {
    flex: 1;
    height: 4px;
  }
  
  .terms-group {
    margin-top: 1.5rem;
  }
  
  .checkbox-label {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    font-size: 0.875rem;
    cursor: pointer;
    line-height: 1.4;
  }
  
  .checkbox-label input {
    margin-top: 0.2rem;
  }
  
  .success-content {
    text-align: center;
    padding: 2rem 0;
  }
  
  .success-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }
  
  .success-title {
    font-size: 1.25rem;
    font-family: monospace;
    margin: 0 0 1rem 0;
  }
  
  .success-message {
    margin-bottom: 1.5rem;
    line-height: 1.5;
  }
  
  .unit-details-card {
    border: 1px solid;
    padding: 1rem;
    margin-bottom: 1.5rem;
  }
  
  .detail-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    font-family: monospace;
    font-size: 0.875rem;
  }
  
  .redirect-message {
    font-size: 0.875rem;
    font-style: italic;
    animation: pulse 1s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .form-actions {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
  }
  
  .back-button,
  .continue-button {
    flex: 1;
    padding: 1rem;
    border: 1px solid;
    font-family: monospace;
    font-weight: bold;
    font-size: 0.875rem;
    letter-spacing: 0.1em;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .back-button {
    background: transparent;
  }
  
  .back-button:hover:not(:disabled),
  .continue-button:hover:not(:disabled) {
    transform: translate(1px, 1px);
  }
  
  .back-button:disabled,
  .continue-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .login-link {
    text-align: center;
    margin-top: 1.5rem;
    font-size: 0.875rem;
  }
  
  .login-link a {
    text-decoration: underline;
  }
  
  .register-footer {
    text-align: center;
    font-size: 0.75rem;
    font-family: monospace;
    opacity: 0.7;
    margin-top: 1rem;
  }
  
  .register-footer p {
    margin: 0;
  }
</style>