<!--
  NES.css Authentication Demo Page
  Shows the retro gaming modal in action
-->
<script lang="ts">
  import NesAuthModal from '$lib/components/auth/NesAuthModal.svelte';
  import { page } from '$app/stores';
  
  interface Props {
    data?: any;
    form?: any;
  }
  
  let { data, form }: Props = $props();
  
  let isAuthModalOpen = $state(false);
  
  function openAuthModal() {
    isAuthModalOpen = true;
  }
  
  function closeAuthModal() {
    isAuthModalOpen = false;
  }
</script>

<svelte:head>
  <title>NES.css Authentication Demo - YoRHa Legal AI</title>
  <meta name="description" content="Retro gaming authentication interface for the YoRHa Legal AI Platform" />
  <!-- Import nes.css and font for page styling -->
  <link href="https://unpkg.com/nes.css@latest/css/nes.min.css" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/css?family=Press+Start+2P" rel="stylesheet">
</svelte:head>

<div class="nes-container">
  <!-- Header -->
  <div class="header-section">
    <h1 class="nes-text is-primary">YoRHa Legal AI Platform</h1>
    <p class="nes-text">Retro Gaming Authentication Interface</p>
  </div>

  <!-- Demo Section -->
  <div class="demo-section">
    <div class="nes-container with-title">
      <p class="title">Authentication Demo</p>
      <p>Experience our retro NES-style authentication modal that integrates with the existing SvelteKit authentication system.</p>
      
      <div class="feature-list">
        <ul class="nes-list is-circle">
          <li>Retro NES.css styling with 8-bit aesthetics</li>
          <li>Tabbed interface for Sign In / Sign Up</li>
          <li>Integration with existing auth backend</li>
          <li>Demo credentials pre-fill functionality</li>
          <li>Responsive design for all devices</li>
          <li>Real form validation and error handling</li>
        </ul>
      </div>

      <div class="demo-actions">
        <button 
          type="button" 
          class="nes-btn is-primary"
          on:onclick={openAuthModal}
        >
          🎮 Launch Auth Modal
        </button>
      </div>
    </div>

    <!-- Demo Credentials Info -->
    <div class="nes-container is-dark with-title">
      <p class="title">Demo Credentials</p>
      <div class="credentials-info">
        <p><strong>Email:</strong> admin@legal-ai.local</p>
        <p><strong>Password:</strong> admin123</p>
        <p class="note">Click "Fill Demo Credentials" in the modal to auto-populate these values.</p>
      </div>
    </div>

    <!-- Features Grid -->
    <div class="features-grid">
      <div class="nes-container with-title">
        <p class="title">Sign In Features</p>
        <ul class="nes-list is-disc">
          <li>Email/password authentication</li>
          <li>Remember me option</li>
          <li>Show/hide password toggle</li>
          <li>Demo credentials button</li>
          <li>Real-time validation</li>
        </ul>
      </div>

      <div class="nes-container with-title">
        <p class="title">Sign Up Features</p>
        <ul class="nes-list is-disc">
          <li>Full name collection</li>
          <li>Role selection dropdown</li>
          <li>Email validation</li>
          <li>Password requirements</li>
          <li>Profile data integration</li>
        </ul>
      </div>
    </div>

    <!-- Tech Stack -->
    <div class="nes-container with-title">
      <p class="title">Tech Stack Integration</p>
      <div class="tech-badges">
        <span class="nes-badge">
          <span class="is-success">SvelteKit</span>
        </span>
        <span class="nes-badge">
          <span class="is-warning">NES.css</span>
        </span>
        <span class="nes-badge">
          <span class="is-primary">PostgreSQL</span>
        </span>
        <span class="nes-badge">
          <span class="is-error">bcryptjs</span>
        </span>
        <span class="nes-badge">
          <span class="is-success">TypeScript</span>
        </span>
      </div>
    </div>

    <!-- Navigation -->
    <div class="nav-section">
      <a href="/auth/login" class="nes-btn">Standard Login</a>
      <a href="/auth/register" class="nes-btn">Standard Register</a>
      <a href="/yorha/dashboard" class="nes-btn is-warning">YoRHa Dashboard</a>
    </div>
  </div>
</div>

<!-- The NES Auth Modal Component -->
<NesAuthModal 
  bind:isOpen={isAuthModalOpen}
  {form}
  close={closeAuthModal}
/>

<style>
  /* Global NES styling for the page */
  :global(body) {
    background-color: #212529;
    color: #fff;
    font-family: 'Press Start 2P', cursive;
    min-height: 100vh;
    margin: 0;
    padding: 1rem;
  }

  .header-section {
    text-align: center;
    margin-bottom: 2rem;
  }

  .header-section h1 {
    margin-bottom: 1rem;
  }

  .demo-section {
    display: flex;
    flex-direction: column;
    gap: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  .feature-list {
    margin: 1rem 0;
  }

  .demo-actions {
    text-align: center;
    margin-top: 1.5rem;
  }

  .credentials-info {
    line-height: 1.8;
  }

  .credentials-info .note {
    font-size: 0.6rem;
    margin-top: 1rem;
    opacity: 0.8;
  }

  .features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .tech-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1rem;
  }

  .nav-section {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 2rem;
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .features-grid {
      grid-template-columns: 1fr;
    }

    .nav-section {
      flex-direction: column;
      align-items: center;
    }
    
    :global(body) {
      padding: 0.5rem;
    }
  }

  /* NES.css customizations for dark theme */
  :global(.nes-container) {
    margin-bottom: 1rem;
  }

  :global(.nes-container.is-dark) {
    background-color: #333;
    border-color: #666;
  }

  :global(.nes-list li) {
    margin-bottom: 0.5rem;
  }

  /* Animation for the launch button */
  .demo-actions .nes-btn {
    transition: transform 0.1s ease;
    font-size: 0.8rem;
  }

  .demo-actions .nes-btn:hover {
    transform: scale(1.05);
  }

  .demo-actions .nes-btn:active {
    transform: scale(0.95);
  }
</style>