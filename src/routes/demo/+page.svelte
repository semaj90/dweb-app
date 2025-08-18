<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { allRoutes } from '$lib/data/routes-config';

  // State management
  let isLoading = $state(true);
  let demoRoutes = $state([]);
  let selectedCategory = $state('all');

  // Filter demo routes
  let demoCategories = {
    'all': 'All Demos',
    'ai': 'AI Demonstrations',
    'enhanced': 'Enhanced Features',
    'integration': 'System Integration',
    'ui': 'UI Components',
    'experimental': 'Experimental'
  };

  let filteredDemos = $derived(() => {
    let demos = demoRoutes;
    if (selectedCategory !== 'all') {
      demos = demos.filter(demo => 
        demo.tags.includes(selectedCategory) || 
        demo.category === selectedCategory
      );
    }
    return demos;
  });

  function navigateHome() {
    goto('/');
  }

  function navigateToDemo(route) {
    goto(route);
  }

  function getStatusColor(status) {
    switch (status) {
      case 'active': return '#00ff41';
      case 'experimental': return '#ffbf00';
      case 'beta': return '#4ecdc4';
      default: return '#888';
    }
  }

  onMount(() => {
    // Get all demo routes from the routes config
    demoRoutes = allRoutes.filter(route => 
      route.category === 'demo' || route.route.includes('/demo/')
    );
    
    setTimeout(() => {
      isLoading = false;
    }, 800);
  });
</script>

<svelte:head>
  <title>Demo Overview - YoRHa Legal AI</title>
  <meta name="description" content="Overview of all AI demonstrations and capabilities">
</svelte:head>

{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">🎯</div>
      <div class="loading-text">LOADING DEMONSTRATION OVERVIEW...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <div class="demo-interface">
    <header class="demo-header">
      <button class="back-button" onclick={navigateHome}>
        ← COMMAND CENTER
      </button>
      <div class="header-title">
        <h1>🎯 AI DEMONSTRATION CENTER</h1>
        <div class="header-subtitle">Overview of AI Capabilities and Technology Showcases</div>
      </div>
      <div class="header-stats">
        <div class="stat-item">
          <div class="stat-value">{demoRoutes.length}</div>
          <div class="stat-label">DEMOS</div>
        </div>
      </div>
    </header>

    <section class="category-filters">
      {#each Object.entries(demoCategories) as [key, label]}
        <button
          class="category-btn {selectedCategory === key ? 'active' : ''}"
          onclick={() => selectedCategory = key}
        >
          {label}
        </button>
      {/each}
    </section>

    <main class="demos-grid">
      {#each filteredDemos as demo (demo.id)}
        <div class="demo-card {demo.status}" onclick={() => navigateToDemo(demo.route)}>
          <div class="card-header">
            <div class="card-icon">{demo.icon}</div>
            <div class="status-badge" style="color: {getStatusColor(demo.status)}">
              {demo.status.toUpperCase()}
            </div>
          </div>

          <div class="card-content">
            <h3 class="card-title">{demo.label}</h3>
            <p class="card-description">{demo.description}</p>
            
            <div class="card-tags">
              {#each demo.tags.slice(0, 3) as tag}
                <span class="tag">{tag}</span>
              {/each}
              {#if demo.tags.length > 3}
                <span class="tag-more">+{demo.tags.length - 3}</span>
              {/if}
            </div>
          </div>

          <div class="card-footer">
            <button class="launch-btn">LAUNCH DEMO</button>
            <div class="route-path">{demo.route}</div>
          </div>
        </div>
      {/each}
    </main>

    {#if filteredDemos.length === 0}
      <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-text">NO DEMOS IN THIS CATEGORY</div>
        <button class="reset-btn" onclick={() => selectedCategory = 'all'}>
          VIEW ALL DEMOS
        </button>
      </div>
    {/if}

    <footer class="demo-footer">
      <div class="footer-info">
        Demo Center v2.1.5 | {demoRoutes.length} Available Demonstrations
      </div>
    </footer>
  </div>
{/if}

<style>
  :global(:root) {
    --yorha-primary: #c4b49a;
    --yorha-accent-warm: #ffbf00;
    --yorha-accent-cool: #4ecdc4;
    --yorha-success: #00ff41;
    --yorha-light: #ffffff;
    --yorha-muted: #f0f0f0;
    --yorha-dark: #1a1a1a;
    --yorha-bg: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
  }

  .loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    background: var(--yorha-bg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    color: var(--yorha-light);
  }

  .loading-content {
    text-align: center;
  }

  .loading-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    color: var(--yorha-accent-warm);
  }

  .loading-text {
    font-size: 1.2rem;
    color: var(--yorha-muted);
    letter-spacing: 2px;
    margin-bottom: 2rem;
  }

  .loading-bar {
    width: 300px;
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
    margin: 0 auto;
  }

  .loading-progress {
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-accent-warm), var(--yorha-success));
    animation: loading 2s ease-in-out infinite;
  }

  .demo-interface {
    min-height: 100vh;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
  }

  .demo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2rem;
    border-bottom: 2px solid var(--yorha-accent-warm);
    background: rgba(26, 26, 26, 0.9);
  }

  .back-button {
    background: transparent;
    border: 2px solid var(--yorha-accent-cool);
    color: var(--yorha-accent-cool);
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  .header-title h1 {
    margin: 0;
    font-size: 2rem;
    background: linear-gradient(45deg, var(--yorha-accent-warm), var(--yorha-success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-transform: uppercase;
  }

  .header-subtitle {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    margin-top: 0.5rem;
  }

  .header-stats {
    text-align: center;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--yorha-accent-warm);
    line-height: 1;
  }

  .stat-label {
    font-size: 0.7rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    margin-top: 0.3rem;
  }

  .category-filters {
    display: flex;
    gap: 1rem;
    padding: 2rem;
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
    flex-wrap: wrap;
    justify-content: center;
  }

  .category-btn {
    background: transparent;
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: var(--yorha-accent-cool);
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .category-btn:hover {
    border-color: var(--yorha-accent-cool);
    background: rgba(78, 205, 196, 0.1);
  }

  .category-btn.active {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  .demos-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 2rem;
    padding: 2rem;
  }

  .demo-card {
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .demo-card:hover {
    border-color: var(--yorha-accent-warm);
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(255, 191, 0, 0.2);
  }

  .demo-card.experimental {
    border-color: rgba(255, 191, 0, 0.6);
  }

  .demo-card.beta {
    border-color: rgba(78, 205, 196, 0.6);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 1.5rem 1rem;
  }

  .card-icon {
    font-size: 2.5rem;
  }

  .status-badge {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.3rem 0.6rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    border: 1px solid currentColor;
  }

  .card-content {
    padding: 0 1.5rem 1rem;
  }

  .card-title {
    font-size: 1.3rem;
    color: var(--yorha-light);
    margin: 0 0 0.8rem;
    text-transform: uppercase;
  }

  .card-description {
    color: var(--yorha-muted);
    line-height: 1.5;
    margin: 0 0 1rem;
  }

  .card-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .tag {
    background: rgba(78, 205, 196, 0.2);
    color: var(--yorha-accent-cool);
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.75rem;
    text-transform: uppercase;
    border: 1px solid var(--yorha-accent-cool);
  }

  .tag-more {
    background: rgba(255, 255, 255, 0.1);
    color: var(--yorha-muted);
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.75rem;
    border: 1px solid var(--yorha-muted);
  }

  .card-footer {
    padding: 1rem 1.5rem 1.5rem;
    border-top: 1px solid rgba(255, 191, 0, 0.2);
  }

  .launch-btn {
    width: 100%;
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
    padding: 1rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-bottom: 0.8rem;
  }

  .launch-btn:hover {
    background: var(--yorha-accent-warm);
  }

  .route-path {
    font-size: 0.8rem;
    color: var(--yorha-muted);
    text-align: center;
    font-family: 'Courier New', monospace;
  }

  .empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--yorha-muted);
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    opacity: 0.5;
  }

  .empty-text {
    font-size: 1.5rem;
    margin-bottom: 2rem;
    text-transform: uppercase;
  }

  .reset-btn {
    background: var(--yorha-accent-warm);
    color: var(--yorha-dark);
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    cursor: pointer;
  }

  .demo-footer {
    background: rgba(26, 26, 26, 0.9);
    border-top: 1px solid rgba(255, 191, 0, 0.3);
    padding: 2rem;
    text-align: center;
  }

  .footer-info {
    color: var(--yorha-muted);
    font-size: 0.9rem;
  }

  @keyframes loading {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(0%); }
    100% { transform: translateX(100%); }
  }

  @media (max-width: 768px) {
    .demo-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .category-filters {
      justify-content: center;
    }

    .demos-grid {
      grid-template-columns: 1fr;
      padding: 1rem;
    }
  }
</style>