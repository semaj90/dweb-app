<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  let isLoading = $state(true);
  let settings = $state({
    ai: {
      defaultModel: 'gemma3-legal',
      temperature: 0.7,
      maxTokens: 2000,
      enableStreaming: true
    },
    database: {
      connectionTimeout: 30,
      maxConnections: 10,
      enableQueryLogging: false
    },
    security: {
      enableTwoFactor: false,
      sessionTimeout: 120,
      enableAuditLog: true
    },
    ui: {
      theme: 'yorha',
      enableAnimations: true,
      compactMode: false
    }
  });

  function navigateHome() {
    goto('/');
  }

  function saveSettings() {
    console.log('Saving settings:', settings);
    // In a real app, this would make an API call
  }

  function resetSettings() {
    settings = {
      ai: {
        defaultModel: 'gemma3-legal',
        temperature: 0.7,
        maxTokens: 2000,
        enableStreaming: true
      },
      database: {
        connectionTimeout: 30,
        maxConnections: 10,
        enableQueryLogging: false
      },
      security: {
        enableTwoFactor: false,
        sessionTimeout: 120,
        enableAuditLog: true
      },
      ui: {
        theme: 'yorha',
        enableAnimations: true,
        compactMode: false
      }
    };
  }

  onMount(() => {
    setTimeout(() => {
      isLoading = false;
    }, 800);
  });
</script>

<svelte:head>
  <title>System Settings - YoRHa Legal AI</title>
</svelte:head>

{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">⚙️</div>
      <div class="loading-text">LOADING SYSTEM SETTINGS...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <div class="settings-interface">
    <header class="settings-header">
      <button class="back-button" onclick={navigateHome}>
        ← COMMAND CENTER
      </button>
      <div class="header-title">
        <h1>⚙️ SYSTEM SETTINGS</h1>
        <div class="header-subtitle">System Configuration and Preferences</div>
      </div>
    </header>

    <main class="settings-content">
      <div class="settings-grid">
        <!-- AI Settings -->
        <section class="settings-section">
          <h2>🤖 AI CONFIGURATION</h2>
          <div class="settings-group">
            <label class="setting-label">DEFAULT MODEL</label>
            <select bind:value={settings.ai.defaultModel} class="setting-select">
              <option value="gemma3-legal">Gemma 3 Legal</option>
              <option value="llama3-8b">Llama 3 8B</option>
              <option value="mixtral-8x7b">Mixtral 8x7B</option>
            </select>
          </div>
          <div class="settings-group">
            <label class="setting-label">TEMPERATURE: {settings.ai.temperature}</label>
            <input type="range" min="0" max="1" step="0.1" bind:value={settings.ai.temperature} class="setting-slider" />
          </div>
          <div class="settings-group">
            <label class="setting-label">MAX TOKENS</label>
            <input type="number" bind:value={settings.ai.maxTokens} class="setting-input" min="100" max="4000" />
          </div>
          <div class="settings-group">
            <label class="setting-checkbox">
              <input type="checkbox" bind:checked={settings.ai.enableStreaming} />
              <span class="checkmark"></span>
              ENABLE STREAMING RESPONSES
            </label>
          </div>
        </section>

        <!-- Database Settings -->
        <section class="settings-section">
          <h2>🗄️ DATABASE CONFIGURATION</h2>
          <div class="settings-group">
            <label class="setting-label">CONNECTION TIMEOUT (seconds)</label>
            <input type="number" bind:value={settings.database.connectionTimeout} class="setting-input" min="10" max="300" />
          </div>
          <div class="settings-group">
            <label class="setting-label">MAX CONNECTIONS</label>
            <input type="number" bind:value={settings.database.maxConnections} class="setting-input" min="1" max="50" />
          </div>
          <div class="settings-group">
            <label class="setting-checkbox">
              <input type="checkbox" bind:checked={settings.database.enableQueryLogging} />
              <span class="checkmark"></span>
              ENABLE QUERY LOGGING
            </label>
          </div>
        </section>

        <!-- Security Settings -->
        <section class="settings-section">
          <h2>🛡️ SECURITY CONFIGURATION</h2>
          <div class="settings-group">
            <label class="setting-checkbox">
              <input type="checkbox" bind:checked={settings.security.enableTwoFactor} />
              <span class="checkmark"></span>
              ENABLE TWO-FACTOR AUTHENTICATION
            </label>
          </div>
          <div class="settings-group">
            <label class="setting-label">SESSION TIMEOUT (minutes)</label>
            <input type="number" bind:value={settings.security.sessionTimeout} class="setting-input" min="30" max="480" />
          </div>
          <div class="settings-group">
            <label class="setting-checkbox">
              <input type="checkbox" bind:checked={settings.security.enableAuditLog} />
              <span class="checkmark"></span>
              ENABLE AUDIT LOGGING
            </label>
          </div>
        </section>

        <!-- UI Settings -->
        <section class="settings-section">
          <h2>🎨 USER INTERFACE</h2>
          <div class="settings-group">
            <label class="setting-label">THEME</label>
            <select bind:value={settings.ui.theme} class="setting-select">
              <option value="yorha">YoRHa Dark</option>
              <option value="light">Light Mode</option>
              <option value="dark">Dark Mode</option>
            </select>
          </div>
          <div class="settings-group">
            <label class="setting-checkbox">
              <input type="checkbox" bind:checked={settings.ui.enableAnimations} />
              <span class="checkmark"></span>
              ENABLE ANIMATIONS
            </label>
          </div>
          <div class="settings-group">
            <label class="setting-checkbox">
              <input type="checkbox" bind:checked={settings.ui.compactMode} />
              <span class="checkmark"></span>
              COMPACT MODE
            </label>
          </div>
        </section>
      </div>
    </main>

    <footer class="settings-footer">
      <div class="footer-actions">
        <button class="footer-btn primary" onclick={saveSettings}>SAVE SETTINGS</button>
        <button class="footer-btn secondary" onclick={resetSettings}>RESET TO DEFAULTS</button>
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

  .settings-interface {
    min-height: 100vh;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
  }

  .settings-header {
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

  .settings-content {
    padding: 2rem;
  }

  .settings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 2rem;
    max-width: 1400px;
    margin: 0 auto;
  }

  .settings-section {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    padding: 2rem;
  }

  .settings-section h2 {
    font-size: 1.3rem;
    color: var(--yorha-accent-warm);
    margin: 0 0 1.5rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
    padding-bottom: 0.5rem;
  }

  .settings-group {
    margin-bottom: 1.5rem;
  }

  .setting-label {
    display: block;
    font-size: 0.9rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
    font-weight: 600;
  }

  .setting-input,
  .setting-select {
    width: 100%;
    background: rgba(42, 42, 42, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: var(--yorha-light);
    padding: 0.8rem 1rem;
    font-family: inherit;
    font-size: 1rem;
    border-radius: 4px;
    transition: all 0.3s ease;
  }

  .setting-input:focus,
  .setting-select:focus {
    outline: none;
    border-color: var(--yorha-accent-cool);
    box-shadow: 0 0 10px rgba(78, 205, 196, 0.3);
  }

  .setting-slider {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
    outline: none;
    cursor: pointer;
  }

  .setting-slider::-webkit-slider-thumb {
    appearance: none;
    width: 18px;
    height: 18px;
    background: var(--yorha-accent-warm);
    border-radius: 50%;
    cursor: pointer;
  }

  .setting-checkbox {
    display: flex;
    align-items: center;
    gap: 1rem;
    cursor: pointer;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--yorha-light);
  }

  .setting-checkbox input[type="checkbox"] {
    display: none;
  }

  .checkmark {
    width: 20px;
    height: 20px;
    border: 2px solid var(--yorha-accent-cool);
    border-radius: 3px;
    position: relative;
    transition: all 0.3s ease;
  }

  .setting-checkbox input[type="checkbox"]:checked + .checkmark {
    background: var(--yorha-accent-cool);
  }

  .setting-checkbox input[type="checkbox"]:checked + .checkmark::after {
    content: '✓';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: var(--yorha-dark);
    font-weight: bold;
    font-size: 14px;
  }

  .settings-footer {
    background: rgba(26, 26, 26, 0.9);
    border-top: 2px solid var(--yorha-accent-warm);
    padding: 2rem;
    text-align: center;
  }

  .footer-actions {
    display: flex;
    gap: 2rem;
    justify-content: center;
  }

  .footer-btn {
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
    font-size: 1rem;
  }

  .footer-btn.primary {
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
  }

  .footer-btn.primary:hover {
    background: var(--yorha-accent-warm);
    transform: scale(1.05);
  }

  .footer-btn.secondary {
    background: transparent;
    color: var(--yorha-accent-cool);
    border: 2px solid var(--yorha-accent-cool);
  }

  .footer-btn.secondary:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  @keyframes loading {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(0%); }
    100% { transform: translateX(100%); }
  }

  @media (max-width: 768px) {
    .settings-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .settings-grid {
      grid-template-columns: 1fr;
      gap: 1rem;
    }

    .settings-section {
      padding: 1.5rem;
    }

    .footer-actions {
      flex-direction: column;
      gap: 1rem;
    }
  }
</style>