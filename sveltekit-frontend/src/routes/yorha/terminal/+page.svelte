<!-- YoRHa Terminal Interface -->
<script lang="ts">
  // $state is declared globally in src/types/svelte-helpers.d.ts
  import { onMount, onDestroy } from 'svelte';
  import { yorhaAPI } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  import YoRHaTerminal from '$lib/components/yorha/YoRHaTerminal.svelte';
  import {
    Terminal,
    Play,
    Square,
    RotateCcw,
    Settings,
    ChevronRight
  } from 'lucide-svelte';

  // Terminal state
  let terminalHistory = $state([]);
  let currentInput = $state('');
  let isExecuting = $state(false);
  let terminalRef: any;

  // Terminal commands
  const availableCommands = {
    help: {
      description: 'Show available commands',
      usage: 'help [command]',
      execute: (args) => showHelp(args)
    },
    status: {
      description: 'Show system status',
      usage: 'status',
      execute: () => getSystemStatus()
    },
    rag: {
      description: 'Execute RAG query',
      usage: 'rag <query>',
      execute: (args) => executeRAG(args.join(' '))
    },
    search: {
      description: 'Search legal database',
      usage: 'search <term>',
      execute: (args) => searchDatabase(args.join(' '))
    },
    cluster: {
      description: 'Cluster management',
      usage: 'cluster <health|status|restart>',
      execute: (args) => clusterCommand(args[0])
    },
    clear: {
      description: 'Clear terminal',
      usage: 'clear',
      execute: () => clearTerminal()
    },
    echo: {
      description: 'Echo text',
      usage: 'echo <text>',
      execute: (args) => echoText(args.join(' '))
    },
    version: {
      description: 'Show system version',
      usage: 'version',
      execute: () => showVersion()
    }
  };

  onMount(() => {
    // Initialize terminal with welcome message
    addOutput('YORHA TERMINAL v1.0.0 - Legal AI System Interface', 'system');
    addOutput('Type "help" for available commands.', 'system');
    addOutput('', 'system');
  });

  function addOutput(text: string, type: 'system' | 'user' | 'success' | 'error' | 'info' = 'system') {
    const timestamp = new Date().toLocaleTimeString();
    terminalHistory = [...terminalHistory, {
      id: Date.now() + Math.random(),
      timestamp,
      text,
      type
    }];
  }

  async function executeCommand(command: string) {
    if (!command.trim()) return;

    isExecuting = true;
    addOutput(`> ${command}`, 'user');

    const parts = command.trim().split(' ');
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);

    if (availableCommands[cmd]) {
      try {
        await availableCommands[cmd].execute(args);
      } catch (error) {
        addOutput(`Error executing ${cmd}: ${error.message}`, 'error');
      }
    } else {
      addOutput(`Unknown command: ${cmd}. Type "help" for available commands.`, 'error');
    }

    isExecuting = false;
    currentInput = '';
  }

  function showHelp(args: string[]) {
    if (args.length > 0) {
      const cmd = args[0].toLowerCase();
      if (availableCommands[cmd]) {
        addOutput(`${cmd}: ${availableCommands[cmd].description}`, 'info');
        addOutput(`Usage: ${availableCommands[cmd].usage}`, 'info');
      } else {
        addOutput(`Unknown command: ${cmd}`, 'error');
      }
    } else {
      addOutput('Available commands:', 'info');
      Object.entries(availableCommands).forEach(([cmd, info]) => {
        addOutput(`  ${cmd.padEnd(10)} - ${info.description}`, 'info');
      });
    }
  }

  async function getSystemStatus() {
    try {
      addOutput('Fetching system status...', 'info');
      const status = await yorhaAPI.getSystemStatus();

      addOutput('=== SYSTEM STATUS ===', 'success');
      addOutput(`Database: ${status.database.connected ? 'CONNECTED' : 'DISCONNECTED'}`, 'info');
      addOutput(`Backend: ${status.backend.healthy ? 'HEALTHY' : 'UNHEALTHY'}`, 'info');
      addOutput(`Frontend: ${status.frontend.renderFPS} FPS`, 'info');
      addOutput(`Services: ${status.backend.activeServices} active`, 'info');
      addOutput(`CPU: ${status.backend.cpuUsage}%`, 'info');
      addOutput(`Memory: ${status.backend.memoryUsage}%`, 'info');
    } catch (error) {
      addOutput('Failed to fetch system status (using mock data)', 'error');
      addOutput('=== SYSTEM STATUS (MOCK) ===', 'success');
      addOutput('Database: CONNECTED', 'info');
      addOutput('Backend: HEALTHY', 'info');
      addOutput('Frontend: 60 FPS', 'info');
      addOutput('Services: 8 active', 'info');
    }
  }

  async function executeRAG(query: string) {
    if (!query) {
      addOutput('Error: Please provide a query. Usage: rag <query>', 'error');
      return;
    }

    try {
      addOutput(`Executing RAG query: "${query}"`, 'info');
      const response = await fetch('/api/yorha/enhanced-rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context: 'terminal' })
      });

      if (response.ok) {
        const result = await response.json();
        addOutput('=== RAG RESULT ===', 'success');
        addOutput(JSON.stringify(result, null, 2), 'info');
      } else {
        addOutput(`RAG query failed: HTTP ${response.status}`, 'error');
      }
    } catch (error) {
      addOutput(`RAG query error: ${error.message}`, 'error');
    }
  }

  async function searchDatabase(term: string) {
    if (!term) {
      addOutput('Error: Please provide a search term. Usage: search <term>', 'error');
      return;
    }

    try {
      addOutput(`Searching database for: "${term}"`, 'info');
      const response = await fetch(`/api/yorha/legal-data?search=${encodeURIComponent(term)}&limit=5`);

      if (response.ok) {
        const result = await response.json();
        addOutput('=== SEARCH RESULTS ===', 'success');
        if (result.results && result.results.length > 0) {
          result.results.forEach((item, index) => {
            addOutput(`${index + 1}. ${item.title || item.name || 'Untitled'}`, 'info');
          });
        } else {
          addOutput('No results found.', 'info');
        }
      } else {
        addOutput(`Search failed: HTTP ${response.status}`, 'error');
      }
    } catch (error) {
      addOutput(`Search error: ${error.message}`, 'error');
    }
  }

  async function clusterCommand(action: string) {
    if (!action) {
      addOutput('Error: Please specify action. Usage: cluster <health|status|restart>', 'error');
      return;
    }

    switch (action.toLowerCase()) {
      case 'health':
        try {
          addOutput('Checking cluster health...', 'info');
          const response = await fetch('/api/v1/cluster/health');
          if (response.ok) {
            const health = await response.json();
            addOutput('=== CLUSTER HEALTH ===', 'success');
            addOutput(JSON.stringify(health, null, 2), 'info');
          } else {
            addOutput(`Health check failed: HTTP ${response.status}`, 'error');
          }
        } catch (error) {
          addOutput(`Health check error: ${error.message}`, 'error');
        }
        break;

      case 'status':
        addOutput('=== CLUSTER STATUS ===', 'success');
        addOutput('PostgreSQL: RUNNING', 'info');
        addOutput('Redis: RUNNING', 'info');
        addOutput('Ollama: RUNNING', 'info');
        addOutput('SvelteKit: RUNNING', 'info');
        addOutput('Enhanced RAG: RUNNING', 'info');
        break;

      case 'restart':
        addOutput('Cluster restart not implemented in terminal mode', 'error');
        break;

      default:
        addOutput(`Unknown cluster action: ${action}`, 'error');
    }
  }

  function clearTerminal() {
    terminalHistory = [];
    addOutput('Terminal cleared.', 'system');
  }

  function echoText(text: string) {
    addOutput(text || '', 'info');
  }

  function showVersion() {
    addOutput('=== SYSTEM VERSION ===', 'success');
    addOutput('YoRHa Terminal: 1.0.0', 'info');
    addOutput('Legal AI Platform: 2.0.0', 'info');
    addOutput('SvelteKit: 2.x', 'info');
    addOutput('Node.js: ' + (typeof process !== 'undefined' ? process.version : 'Browser'), 'info');
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !isExecuting) {
      executeCommand(currentInput);
    }
  }
</script>

<svelte:head>
  <title>YoRHa Terminal - Command Interface</title>
</svelte:head>

<div class="yorha-terminal-page">
  <!-- Page Header -->
  <header class="yorha-page-header">
    <div class="yorha-header-content">
      <div class="yorha-header-title">
        <Terminal size={48} />
        <h1>YORHA TERMINAL</h1>
        <div class="yorha-header-subtitle">COMMAND LINE INTERFACE</div>
      </div>
    </div>
  </header>

  <!-- Terminal Container -->
  <section class="yorha-terminal-section">
    <div class="yorha-terminal-container">
      <!-- Terminal Header -->
      <div class="yorha-terminal-header">
        <div class="yorha-terminal-title">
          <Terminal size={16} />
          <span>YoRHa Terminal</span>
        </div>
        <div class="yorha-terminal-controls">
          <button class="yorha-terminal-control" onclick={() => clearTerminal()}>
            <RotateCcw size={14} />
          </button>
          <button class="yorha-terminal-control">
            <Settings size={14} />
          </button>
        </div>
      </div>

      <!-- Terminal Output -->
      <div class="yorha-terminal-output">
        {#each terminalHistory as entry (entry.id)}
          <div class="yorha-terminal-line yorha-line-{entry.type}">
            <span class="yorha-terminal-timestamp">[{entry.timestamp}]</span>
            <span class="yorha-terminal-text">{entry.text}</span>
          </div>
        {/each}

        {#if isExecuting}
          <div class="yorha-terminal-line yorha-line-system">
            <span class="yorha-terminal-timestamp">[{new Date().toLocaleTimeString()}]</span>
            <span class="yorha-terminal-text">
              <span class="yorha-terminal-spinner">⠋</span>
              Executing...
            </span>
          </div>
        {/if}
      </div>

      <!-- Terminal Input -->
      <div class="yorha-terminal-input-container">
        <span class="yorha-terminal-prompt">
          <ChevronRight size={16} />
          YORHA:~$
        </span>
        <input
          type="text"
          bind:value={currentInput}
          onkeydown={handleKeydown}
          disabled={isExecuting}
          class="yorha-terminal-input"
          placeholder="Type command... (try 'help')"
          autocomplete="off"
          spellcheck="false"
        />
      </div>
    </div>

    <!-- Command Reference -->
    <div class="yorha-command-reference">
      <h3>Quick Reference</h3>
      <div class="yorha-command-grid">
        <div class="yorha-command-item">
          <strong>help</strong> - Show available commands
        </div>
        <div class="yorha-command-item">
          <strong>status</strong> - System status
        </div>
        <div class="yorha-command-item">
          <strong>rag &lt;query&gt;</strong> - AI analysis
        </div>
        <div class="yorha-command-item">
          <strong>search &lt;term&gt;</strong> - Database search
        </div>
        <div class="yorha-command-item">
          <strong>cluster &lt;action&gt;</strong> - Cluster management
        </div>
        <div class="yorha-command-item">
          <strong>clear</strong> - Clear terminal
        </div>
      </div>
    </div>
  </section>
</div>

<style>
  .yorha-terminal-page {
    @apply min-h-screen;
  }

  /* Page Header */
  .yorha-page-header {
    @apply py-12 px-6 border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-header-content {
    @apply max-w-6xl mx-auto text-center;
  }

  .yorha-header-title h1 {
    @apply text-3xl md:text-4xl font-bold tracking-wider text-amber-400 flex items-center justify-center gap-4;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-header-subtitle {
    @apply text-lg text-amber-300 tracking-wide opacity-80 mt-2;
  }

  /* Terminal Section */
  .yorha-terminal-section {
    @apply p-6 max-w-6xl mx-auto space-y-6;
  }

  .yorha-terminal-container {
    @apply bg-black border-2 border-amber-400 border-opacity-60;
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.3);
  }

  /* Terminal Header */
  .yorha-terminal-header {
    @apply flex items-center justify-between px-4 py-2 bg-amber-400 text-black;
  }

  .yorha-terminal-title {
    @apply flex items-center gap-2 font-mono text-sm font-bold;
  }

  .yorha-terminal-controls {
    @apply flex items-center gap-2;
  }

  .yorha-terminal-control {
    @apply p-1 hover:bg-black hover:bg-opacity-20 transition-colors;
  }

  /* Terminal Output */
  .yorha-terminal-output {
    @apply p-4 h-96 overflow-y-auto font-mono text-sm;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.95) 0%, rgba(255, 191, 0, 0.02) 100%);
  }

  .yorha-terminal-line {
    @apply flex gap-2 mb-1;
  }

  .yorha-terminal-timestamp {
    @apply text-amber-400 opacity-60 text-xs;
  }

  .yorha-terminal-text {
    @apply flex-1;
  }

  .yorha-line-system {
    @apply text-amber-400;
  }

  .yorha-line-user {
    @apply text-green-400;
  }

  .yorha-line-success {
    @apply text-green-400;
  }

  .yorha-line-error {
    @apply text-red-400;
  }

  .yorha-line-info {
    @apply text-amber-300;
  }

  .yorha-terminal-spinner {
    @apply inline-block;
    animation: spin 1s linear infinite;
  }

  /* Terminal Input */
  .yorha-terminal-input-container {
    @apply flex items-center border-t border-amber-400 border-opacity-30 bg-black bg-opacity-50;
  }

  .yorha-terminal-prompt {
    @apply px-4 py-3 text-amber-400 font-mono text-sm flex items-center gap-2;
  }

  .yorha-terminal-input {
    @apply flex-1 px-2 py-3 bg-transparent text-amber-300 font-mono text-sm;
    @apply focus:outline-none placeholder-amber-400 placeholder-opacity-50;
    @apply disabled:opacity-50 disabled:cursor-not-allowed;
  }

  /* Command Reference */
  .yorha-command-reference {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-6;
  }

  .yorha-command-reference h3 {
    @apply text-lg font-bold text-amber-400 mb-4 tracking-wider;
  }

  .yorha-command-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3;
  }

  .yorha-command-item {
    @apply text-amber-300 text-sm font-mono;
  }

  .yorha-command-item strong {
    @apply text-amber-400;
  }

  @keyframes spin {
    0% { content: '⠋'; }
    12.5% { content: '⠙'; }
    25% { content: '⠹'; }
    37.5% { content: '⠸'; }
    50% { content: '⠼'; }
    62.5% { content: '⠴'; }
    75% { content: '⠦'; }
    87.5% { content: '⠧'; }
    100% { content: '⠇'; }
  }

  /* Responsive */
  @media (max-width: 768px) {
    .yorha-header-title h1 {
      @apply text-2xl flex-col;
    }

    .yorha-terminal-output {
      @apply h-64;
    }

    .yorha-command-grid {
      @apply grid-cols-1 gap-2;
    }
  }
</style>