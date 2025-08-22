<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { multiLibraryStartup, type StartupStatus } from '$lib/services/multi-library-startup';
  
  let startupStatus: StartupStatus | null = $state(null);
  let showStartupLog = $state(false);

  onMount(async () => {
    if (!browser) return;
    
    console.log('🚀 Initializing YoRHa Legal AI Platform...');
    
    try {
      // Initialize multi-library integration on app startup
      startupStatus = await multiLibraryStartup.initialize();
      
      if (startupStatus.initialized) {
        console.log('✅ YoRHa Legal AI Platform Ready');
        
        // Show brief startup notification
        showStartupLog = true;
        setTimeout(() => {
          showStartupLog = false;
        }, 4000);
      }
    } catch (error) {
      console.error('❌ Platform initialization failed:', error);
    }
  });
</script>

<!-- Multi-Library Startup Notification -->
{#if showStartupLog && startupStatus}
  <div class="startup-notification">
    <div class="startup-content">
      <h3>🚀 YoRHa Legal AI Platform</h3>
      <p>Multi-Library Integration Complete</p>
      <div class="startup-services">
        {#each Object.entries(startupStatus.services) as [service, status]}
          <span class="service-status" class:ready={status} class:failed={!status}>
            {status ? '✅' : '❌'} {service.toUpperCase()}
          </span>
        {/each}
      </div>
      <p class="startup-time">Initialized in {startupStatus.initTime}ms</p>
    </div>
  </div>
{/if}

<div class="app">
  <header>
    <h1>YoRHa Legal AI</h1>
    <nav class="main-nav">
      <a href="/">Home</a>
      <a href="/yorha-command-center">YoRHa Command Center</a>
      <a href="/demo/enhanced-rag-semantic">Enhanced RAG Demo</a>
      <a href="/endpoints">Endpoints</a>
      {#if startupStatus?.initialized}
        <span class="status-indicator ready">🟢 INTEGRATED</span>
      {:else}
        <span class="status-indicator loading">🟡 LOADING</span>
      {/if}
    </nav>
  </header>
  <main>
    <slot />
  </main>
</div>

<style>
  /* Startup Notification Styles */
  .startup-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
    border: 2px solid #ffd700;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    animation: slideIn 0.5s ease-out;
    max-width: 400px;
  }
  
  .startup-content h3 {
    margin: 0 0 0.5rem 0;
    color: #ffd700;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  
  .startup-content p {
    margin: 0 0 1rem 0;
    color: #e0e0e0;
    font-size: 0.9rem;
  }
  
  .startup-services {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
    margin: 1rem 0;
  }
  
  .service-status {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    border: 1px solid;
    font-family: 'JetBrains Mono', monospace;
  }
  
  .service-status.ready {
    color: #00ff41;
    border-color: #00ff41;
    background: rgba(0, 255, 65, 0.1);
  }
  
  .service-status.failed {
    color: #ff0041;
    border-color: #ff0041;
    background: rgba(255, 0, 65, 0.1);
  }
  
  .startup-time {
    font-size: 0.8rem !important;
    color: #b0b0b0 !important;
    text-align: right;
    margin: 0.5rem 0 0 0 !important;
  }
  
  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  /* Main App Styles */
  .app {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
  }
  
  header {
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
    color: #ffd700;
    padding: 1rem;
    border-bottom: 2px solid #ffd700;
    box-shadow: 0 2px 10px rgba(255, 215, 0, 0.2);
  }
  
  header h1 {
    margin: 0;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    text-shadow: 0 0 10px currentColor;
  }
  
  .main-nav {
    margin-top: 1rem;
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
  }
  
  .main-nav a {
    color: #ffd700;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border: 1px solid #ffd700;
    border-radius: 4px;
    transition: all 0.2s ease;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  
  .main-nav a:hover {
    background: #ffd700;
    color: #1a1a1a;
    box-shadow: 0 0 8px rgba(255, 215, 0, 0.5);
    transform: translateY(-1px);
  }
  
  .status-indicator {
    font-size: 0.8rem;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
  }
  
  .status-indicator.ready {
    background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%);
    color: #000000;
    box-shadow: 0 0 8px rgba(0, 255, 65, 0.5);
  }
  
  .status-indicator.loading {
    background: linear-gradient(135deg, #ffd700 0%, #cc8800 100%);
    color: #000000;
    box-shadow: 0 0 8px rgba(255, 215, 0, 0.5);
    animation: pulse 2s ease-in-out infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
  
  main {
    padding: 1rem;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.1) 0%, rgba(26, 26, 26, 0.1) 100%);
  }
</style>