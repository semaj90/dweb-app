<script lang="ts">
  import '../app.css';
import '../lib/styles/modern-yorha-theme.css';
import { onMount, setContext } from 'svelte';
import Navigation from '$lib/components/Navigation.svelte';

// Simplified state management for compatibility
let theme = $state<'dark' | 'light' >('dark');
let sidebarOpen = $state(false);

// App context for global state
setContext('app', {
  get theme() { return theme; },
  set theme(value) { theme = value; },
  get sidebarOpen() { return sidebarOpen; },
  set sidebarOpen(value) { sidebarOpen = value; },
  toggleSidebar: () => { sidebarOpen = !sidebarOpen; },
  toggleTheme: () => { theme = theme === 'dark' ? 'light' : 'dark'; }
});

onMount(() => {
  // Modern YoRHa UI initialization
  console.log('Modern Dark YoRHa Legal AI Interface initialized');

  // Set theme on document
  document.documentElement.setAttribute('data-theme', theme);
});

// Reactive theme updates - temporarily simplified
// $effect(() => {
//   document.documentElement.setAttribute('data-theme', theme);
// });
</script>

<div class="app-layout golden-grid-holy-grail yorha-bg-primary">
  <header class="app-header" style="grid-area: header;">
    <Navigation {sidebarOpen} />
  </header>

  <aside
    class="app-sidebar transition-transform duration-300 {!sidebarOpen ? 'sidebar-hidden' : ''}"
    style="grid-area: sidebar;"
  >
    <!-- Sidebar content will go here -->
    <nav class="sidebar-nav p-golden-lg">
      <div class="space-y-golden">
        <a href="/" class="yorha-nav-item">Home</a>
        <a href="/cases" class="yorha-nav-item">Cases</a>
        <a href="/evidence" class="yorha-nav-item">Evidence</a>
        <a href="/yorha" class="yorha-nav-item">YoRHa Interface</a>
      </div>
    </nav>
  </aside>

  <main class="app-main" style="grid-area: main;">
    <div class="container p-golden-xl">
  {@/* TODO: manual review – previously {@render children?.()} */}
    </div>
  </main>

  <footer class="app-footer" style="grid-area: footer;">
    <div class="container p-golden-md golden-flex-between">
      <span class="yorha-text-muted">Legal AI Platform © 2024</span>
      <div class="golden-flex">
        <span class="yorha-status-indicator yorha-status-online"></span>
        <span class="yorha-text-muted">System Online</span>
      </div>
    </div>
  </footer>
</div>

<style>
  .app-layout {
    min-height: 100vh;
  }

  .app-header {
    background: var(--yorha-bg-secondary);
    border-bottom: 1px solid var(--yorha-border-primary);
    box-shadow: var(--yorha-shadow-sm);
  }

  .app-sidebar {
    background: var(--yorha-bg-secondary);
    border-right: 1px solid var(--yorha-border-primary);
    overflow-y: auto;
  }

  .app-main {
    background: var(--yorha-bg-primary);
    overflow-y: auto;
    position: relative;
  }

  .app-footer {
    background: var(--yorha-bg-tertiary);
    border-top: 1px solid var(--yorha-border-primary);
    font-size: var(--text-sm);
  }

  .sidebar-nav {
    height: 100%;
  }

  @media (max-width: 768px) {
    .app-layout {
      grid-template-areas:
        "header"
        "main"
        "footer";
      grid-template-columns: 1fr;
    }

    .app-sidebar {
      position: fixed;
      top: 0;
      left: 0;
      bottom: 0;
      width: 16rem;
      z-index: 50;
      transform: translateX(-100%);
    }

    .app-sidebar:not(.sidebar-hidden) {
      transform: translateX(0);
    }

    .app-main {
      padding-top: 0;
    }
  }
</style>