<script lang="ts">
  import { onMount } from 'svelte';
  import { user } from '$lib/stores/user';

  let loading = true;

  onMount(() => {
    // Simple loading simulation
    setTimeout(() => {
      loading = false;
    }, 1000);
  });
</script>

<svelte:head>
  <title>Detective Mode - Legal Case Management</title>
  <meta name="description" content="Professional evidence management for legal professionals" />
</svelte:head>

<div class="home-container">
  {#if loading}
    <div class="loading-screen">
      <div class="spinner"></div>
      <h1>Detective Mode</h1>
      <p>Loading Legal Case Management System...</p>
    </div>
  {:else}
    <div class="hero-section">
      <div class="hero-content">
        <h1 class="hero-title">Legal Case Management</h1>
        <p class="hero-subtitle">Professional evidence management and case organization</p>

        <div class="action-buttons">
          {#if $user}
            <a href="/dashboard" class="btn btn-primary">
              <span class="btn-icon">📊</span>
              Go to Dashboard
            </a>
            <a href="/cases" class="btn btn-secondary">
              <span class="btn-icon">📁</span>
              View Cases
            </a>
          {:else}
            <a href="/login" class="btn btn-primary">
              <span class="btn-icon">🔐</span>
              Login
            </a>
            <a href="/register" class="btn btn-secondary">
              <span class="btn-icon">👤</span>
              Register
            </a>
          {/if}
        </div>
      </div>
    </div>

    <div class="features-section">
      <div class="features-grid">
        <div class="feature-card">
          <div class="feature-icon">📁</div>
          <h3>Case Management</h3>
          <p>Organize and track legal cases with comprehensive case files</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🔍</div>
          <h3>Evidence Board</h3>
          <p>Visual evidence management with drag-and-drop interface</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">📊</div>
          <h3>Analytics</h3>
          <p>Track case progress and analyze patterns</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🤖</div>
          <h3>AI Assistant</h3>
          <p>AI-powered legal research and document analysis</p>
        </div>
      </div>
    </div>

    <!-- All Routes List (dynamic) -->
    <RoutesList />
<script lang="ts">
  import RoutesList from "./RoutesList.svelte";
</script>
  {/if}
</div>

<style>
  .home-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .loading-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
    text-align: center;
  }

  .spinner {
    width: 50px;
    height: 50px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 2rem;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .hero-section {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
    padding: 2rem;
  }

  .hero-content {
    max-width: 600px;
  }

  .hero-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #1f2937, #dc2626);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .hero-subtitle {
    font-size: 1.25rem;
    color: #6b7280;
    margin-bottom: 2rem;
  }

  .action-buttons {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    text-decoration: none;
    font-weight: 600;
    transition: all 0.2s ease;
  }

  .btn-primary {
    background: #3b82f6;
    color: white;
  }

  .btn-primary:hover {
    background: #2563eb;
    transform: translateY(-1px);
  }

  .btn-secondary {
    background: #f3f4f6;
    color: #1f2937;
    border: 2px solid #e5e7eb;
  }

  .btn-secondary:hover {
    background: #e5e7eb;
    transform: translateY(-1px);
  }

  .features-section {
    padding: 4rem 2rem;
    background: #f9fafb;
  }

  .features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  .feature-card {
    background: white;
    padding: 2rem;
    border-radius: 0.75rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
  }

  .feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .feature-card h3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #1f2937;
  }

  .feature-card p {
    color: #6b7280;
    line-height: 1.5;
  }

  @media (max-width: 768px) {
    .hero-title {
      font-size: 2rem;
    }

    .action-buttons {
      flex-direction: column;
      align-items: center;
    }

    .btn {
      width: 100%;
      max-width: 200px;
    }
  }
</style>