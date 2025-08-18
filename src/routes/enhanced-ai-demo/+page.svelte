<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';

  onMount(() => {
    // Redirect to the actual enhanced legal AI demo
    goto('/enhanced-legal-ai-demo', { replaceState: true });
  });
</script>

<svelte:head>
  <title>Redirecting to Enhanced Legal AI Demo</title>
</svelte:head>

<div class="redirect-page">
  <div class="redirect-content">
    <div class="redirect-icon">🚀</div>
    <div class="redirect-title">REDIRECTING TO ENHANCED AI DEMO</div>
    <div class="redirect-message">
      You are being redirected to the Enhanced Legal AI Demo...
    </div>
    <div class="redirect-fallback">
      <a href="/enhanced-legal-ai-demo" class="redirect-link">
        Click here if not redirected automatically
      </a>
    </div>
  </div>
</div>

<style>
  .redirect-page {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .redirect-content {
    text-align: center;
    padding: 3rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    border: 2px solid #ffbf00;
    border-radius: 8px;
  }

  .redirect-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    animation: pulse 1.5s ease-in-out infinite;
  }

  .redirect-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ffbf00;
  }

  .redirect-message {
    font-size: 1rem;
    margin-bottom: 2rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  .redirect-fallback {
    margin-top: 2rem;
  }

  .redirect-link {
    color: #00ff41;
    text-decoration: none;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 2px solid #00ff41;
    padding: 1rem 2rem;
    border-radius: 4px;
    transition: all 0.3s ease;
  }

  .redirect-link:hover {
    background: #00ff41;
    color: #0a0a0a;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
</style>