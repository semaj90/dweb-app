<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  let isLoading = $state(true);
  let mcpStatus = $state({
    server: 'disconnected',
    capabilities: [],
    activeTools: 0,
    lastSync: null
  });

  function navigateHome() {
    goto('/');
  }

  onMount(() => {
    // Simulate loading MCP status
    setTimeout(() => {
      mcpStatus = {
        server: 'connected',
        capabilities: ['filesystem', 'search', 'context7'],
        activeTools: 12,
        lastSync: new Date()
      };
      isLoading = false;
    }, 800);
  });
</script>

<svelte:head>
  <title>MCP Tools - YoRHa Legal AI</title>
</svelte:head>

{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">🔧</div>
      <div class="loading-text">INITIALIZING MCP TOOLS...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <div class="mcp-interface">
    <header class="mcp-header">
      <button class="back-button" onclick={navigateHome}>
        ← COMMAND CENTER
      </button>
      <div class="header-title">
        <h1>🔧 MCP DEVELOPMENT TOOLS</h1>
        <div class="header-subtitle">Model Context Protocol Development Tools</div>
      </div>
      <div class="connection-status {mcpStatus.server}">
        STATUS: {mcpStatus.server.toUpperCase()}
      </div>
    </header>

    <main class="mcp-content">
      <div class="tools-grid">
        <section class="tool-section">
          <h2>📁 FILESYSTEM TOOLS</h2>
          <div class="tool-buttons">
            <button class="tool-btn">LIST DIRECTORY</button>
            <button class="tool-btn">READ FILE</button>
            <button class="tool-btn">SEARCH FILES</button>
            <button class="tool-btn">WRITE FILE</button>
          </div>
        </section>

        <section class="tool-section">
          <h2>🔍 SEARCH TOOLS</h2>
          <div class="tool-buttons">
            <button class="tool-btn">SEMANTIC SEARCH</button>
            <button class="tool-btn">VECTOR SEARCH</button>
            <button class="tool-btn">FULL TEXT SEARCH</button>
            <button class="tool-btn">FUZZY SEARCH</button>
          </div>
        </section>

        <section class="tool-section">
          <h2>🧠 CONTEXT7 TOOLS</h2>
          <div class="tool-buttons">
            <button class="tool-btn">GET LIBRARY DOCS</button>
            <button class="tool-btn">RESOLVE LIBRARY ID</button>
            <button class="tool-btn">CONTEXT ANALYSIS</button>
            <button class="tool-btn">CODE COMPLETION</button>
          </div>
        </section>

        <section class="tool-section">
          <h2>⚙️ SYSTEM TOOLS</h2>
          <div class="tool-buttons">
            <button class="tool-btn">HEALTH CHECK</button>
            <button class="tool-btn">PERFORMANCE METRICS</button>
            <button class="tool-btn">LOG VIEWER</button>
            <button class="tool-btn">CONFIG MANAGER</button>
          </div>
        </section>
      </div>

      <div class="status-panel">
        <h3>🔌 MCP SERVER STATUS</h3>
        <div class="status-info">
          <div class="status-item">
            <span class="label">Server:</span>
            <span class="value {mcpStatus.server}">{mcpStatus.server}</span>
          </div>
          <div class="status-item">
            <span class="label">Active Tools:</span>
            <span class="value">{mcpStatus.activeTools}</span>
          </div>
          <div class="status-item">
            <span class="label">Last Sync:</span>
            <span class="value">{mcpStatus.lastSync?.toLocaleTimeString() || 'Never'}</span>
          </div>
        </div>
      </div>
    </main>

    <footer class="mcp-footer">
      <div class="footer-info">
        MCP Development Tools v1.0.0 | Model Context Protocol Interface
      </div>
    </footer>
  </div>
{/if}

<style>
  :global(:root) {
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

  .mcp-interface {
    min-height: 100vh;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
  }

  .mcp-header {
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

  .connection-status {
    padding: 0.8rem 1.5rem;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .connection-status.connected {
    background: rgba(0, 255, 65, 0.2);
    color: var(--yorha-success);
    border: 1px solid var(--yorha-success);
  }

  .connection-status.disconnected {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
  }

  .mcp-content {
    padding: 2rem;
  }

  .tools-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .tool-section {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    padding: 2rem;
  }

  .tool-section h2 {
    font-size: 1.3rem;
    color: var(--yorha-accent-warm);
    margin: 0 0 1.5rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
    padding-bottom: 0.5rem;
  }

  .tool-buttons {
    display: grid;
    gap: 1rem;
  }

  .tool-btn {
    background: rgba(42, 42, 42, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: var(--yorha-accent-cool);
    padding: 1rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .tool-btn:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    transform: translateY(-2px);
  }

  .status-panel {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    padding: 2rem;
    max-width: 600px;
    margin: 0 auto;
  }

  .status-panel h3 {
    font-size: 1.2rem;
    color: var(--yorha-accent-warm);
    margin: 0 0 1.5rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: center;
  }

  .status-info {
    display: grid;
    gap: 1rem;
  }

  .status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 1rem;
    background: rgba(42, 42, 42, 0.6);
    border-radius: 4px;
  }

  .status-item .label {
    color: var(--yorha-muted);
    font-weight: 600;
    text-transform: uppercase;
  }

  .status-item .value {
    color: var(--yorha-light);
  }

  .status-item .value.connected {
    color: var(--yorha-success);
  }

  .mcp-footer {
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
    .mcp-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .tools-grid {
      grid-template-columns: 1fr;
    }

    .tool-section {
      padding: 1.5rem;
    }
  }
</style>