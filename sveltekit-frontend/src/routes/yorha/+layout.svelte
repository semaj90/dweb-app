<!-- YoRHa Interface Layout -->
<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { yorhaAPI } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  import { onMount, onDestroy } from 'svelte';
  import {
    Home,
    Terminal,
    Monitor,
    Settings,
    Database,
    Activity,
    Cpu,
    Search,
    FileText,
    Bot,
    ChevronRight,
    ChevronLeft
  } from 'lucide-svelte';

  // System status and navigation
  let systemStatus = $state({
    connected: false,
    services: 0,
    errors: 0
  });

  let sidebarOpen = $state(false);
  let currentPath = $state('');

  // Navigation structure
  const navItems = [
    {
      path: '/yorha',
      label: 'Command Center',
      icon: Terminal,
      description: 'Main YoRHa interface hub'
    },
    {
      path: '/yorha/dashboard',
      label: 'System Dashboard',
      icon: Monitor,
      description: 'Live system monitoring'
    },
    {
      path: '/yorha/components',
      label: 'UI Components',
      icon: Bot,
      description: '3D UI component gallery'
    },
    {
      path: '/yorha/api-test',
      label: 'API Testing',
      icon: Cpu,
      description: 'Live API integration tests'
    },
    {
      path: '/yorha/terminal',
      label: 'Terminal',
      icon: Terminal,
      description: 'YoRHa command terminal'
    },
    {
      path: '/yorha/data-grid',
      label: 'Data Grid',
      icon: Database,
      description: 'Advanced data visualization'
    },
    {
      path: '/yorha/search',
      label: 'Vector Search',
      icon: Search,
      description: 'Semantic search interface'
    },
    {
      path: '/yorha/chat',
      label: 'AI Chat',
      icon: Bot,
      description: 'Enhanced AI conversation'
    }
  ];

  // System status monitoring
  onMount(async () => {
    currentPath = $page.url.pathname;

    // Initialize YoRHa API and check system status
    try {
      const status = await yorhaAPI.getSystemStatus();
      systemStatus = {
        connected: true,
        services: Object.keys(status).length,
        errors: 0
      };
    } catch (error) {
      console.warn('YoRHa API not available:', error);
      systemStatus.connected = false;
    }

    // Subscribe to route changes
    const unsubscribe = page.subscribe(($page) => {
      currentPath = $page.url.pathname;
    });

    return unsubscribe;
  });

  onDestroy(() => {
    yorhaAPI.dispose();
  });

  function navigateTo(path: string) {
    goto(path);
    sidebarOpen = false;
  }

  function isActivePath(path: string): boolean {
    return currentPath === path || (path !== '/yorha' && currentPath.startsWith(path));
  }
</script>

<svelte:head>
  <title>YoRHa Interface - Legal AI System</title>
  <meta name="description" content="YoRHa-themed interface for Legal AI system access and control." />
</svelte:head>

<div class="yorha-layout">
  <!-- Top Navigation Bar -->
  <header class="yorha-header">
    <div class="yorha-header-content">
      <!-- Logo and Title -->
      <div class="yorha-brand">
        <button
          class="yorha-menu-toggle"
          onclick={() => sidebarOpen = !sidebarOpen}
          aria-label="Toggle sidebar"
        >
          <Terminal size={20} />
        </button>

        <h1 class="yorha-brand-title">
          <span class="yorha-brand-icon">⬢</span>
          YoRHa SYSTEM
        </h1>
      </div>

      <!-- System Status -->
      <div class="yorha-status-bar">
        <div class="yorha-status-item" class:yorha-status-connected={systemStatus.connected}>
          <Activity size={16} />
          <span>{systemStatus.connected ? 'CONNECTED' : 'OFFLINE'}</span>
        </div>

        <div class="yorha-status-item">
          <Monitor size={16} />
          <span>{systemStatus.services} SERVICES</span>
        </div>

        {#if systemStatus.errors > 0}
          <div class="yorha-status-item yorha-status-error">
            <span>⚠ {systemStatus.errors} ERRORS</span>
          </div>
        {/if}
      </div>

      <!-- Quick Actions -->
      <div class="yorha-quick-actions">
        <button class="yorha-quick-btn" onclick={() => goto('/')}>
          <Home size={16} />
          <span>MAIN</span>
        </button>
        <button class="yorha-quick-btn" onclick={() => goto('/demos')}>
          <FileText size={16} />
          <span>DEMOS</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Sidebar Navigation -->
  <aside class="yorha-sidebar" class:yorha-sidebar-open={sidebarOpen}>
    <nav class="yorha-nav">
      <div class="yorha-nav-header">
        <h2>NAVIGATION</h2>
        <button
          class="yorha-sidebar-close"
          onclick={() => sidebarOpen = false}
        >
          <ChevronLeft size={20} />
        </button>
      </div>

      <ul class="yorha-nav-list">
        {#each navItems as item}
          <li class="yorha-nav-item">
            <button
              class="yorha-nav-link"
              class:yorha-nav-active={isActivePath(item.path)}
              onclick={() => navigateTo(item.path)}
            >
              <svelte:component this={item.icon} size={18} />
              <div class="yorha-nav-content">
                <span class="yorha-nav-label">{item.label}</span>
                <span class="yorha-nav-desc">{item.description}</span>
              </div>
              <ChevronRight size={16} class="yorha-nav-arrow" />
            </button>
          </li>
        {/each}
      </ul>
    </nav>
  </aside>

  <!-- Main Content Area -->
  <main class="yorha-main" class:yorha-main-sidebar-open={sidebarOpen}>
    <slot />
  </main>

  <!-- Sidebar Overlay -->
  {#if sidebarOpen}
    <div
      class="yorha-overlay"
      onclick={() => sidebarOpen = false}
    ></div>
  {/if}
</div>

<style>
  .yorha-layout {
    @apply min-h-screen bg-black text-amber-400 font-mono;
    font-family: 'Courier New', monospace;
    background-image:
      radial-gradient(circle at 20% 50%, rgba(255, 191, 0, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 191, 0, 0.03) 0%, transparent 50%);
  }

  /* Header */
  .yorha-header {
    @apply fixed top-0 left-0 right-0 z-40 bg-black border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.95) 0%, rgba(255, 191, 0, 0.05) 100%);
    backdrop-filter: blur(8px);
  }

  .yorha-header-content {
    @apply flex items-center justify-between px-6 py-4;
  }

  .yorha-brand {
    @apply flex items-center gap-4;
  }

  .yorha-menu-toggle {
    @apply p-2 text-amber-400 hover:text-amber-300 transition-colors;
    @apply border border-amber-400 border-opacity-30 hover:border-opacity-60;
  }

  .yorha-brand-title {
    @apply text-xl font-bold tracking-wider flex items-center gap-2;
  }

  .yorha-brand-icon {
    @apply text-amber-400 text-2xl;
  }

  .yorha-status-bar {
    @apply flex items-center gap-6;
  }

  .yorha-status-item {
    @apply flex items-center gap-2 text-xs text-amber-400 opacity-60;
  }

  .yorha-status-connected {
    @apply text-green-400 opacity-100;
  }

  .yorha-status-error {
    @apply text-red-400 opacity-100;
  }

  .yorha-quick-actions {
    @apply flex items-center gap-2;
  }

  .yorha-quick-btn {
    @apply px-3 py-2 bg-amber-400 text-black text-xs font-mono tracking-wider;
    @apply hover:bg-amber-300 transition-colors flex items-center gap-2;
  }

  /* Sidebar */
  .yorha-sidebar {
    @apply fixed top-[73px] left-0 bottom-0 w-80 bg-gray-900 border-r border-amber-400 border-opacity-30;
    @apply transform -translate-x-full transition-transform duration-300 z-30;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.95) 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-sidebar-open {
    @apply translate-x-0;
  }

  .yorha-nav {
    @apply h-full flex flex-col;
  }

  .yorha-nav-header {
    @apply flex items-center justify-between p-6 border-b border-amber-400 border-opacity-30;
  }

  .yorha-nav-header h2 {
    @apply text-lg font-bold text-amber-400 tracking-wider;
  }

  .yorha-sidebar-close {
    @apply p-2 text-amber-400 hover:text-amber-300 transition-colors;
  }

  .yorha-nav-list {
    @apply flex-1 py-4;
  }

  .yorha-nav-item {
    @apply border-b border-amber-400 border-opacity-10;
  }

  .yorha-nav-link {
    @apply w-full p-4 text-left flex items-center gap-4;
    @apply hover:bg-amber-400 hover:bg-opacity-10 transition-colors;
    @apply text-amber-300 hover:text-amber-400;
  }

  .yorha-nav-active {
    @apply bg-amber-400 bg-opacity-20 text-amber-400 border-r-2 border-amber-400;
  }

  .yorha-nav-content {
    @apply flex-1 min-w-0;
  }

  .yorha-nav-label {
    @apply block font-semibold text-sm;
  }

  .yorha-nav-desc {
    @apply block text-xs opacity-60 truncate;
  }

  .yorha-nav-arrow {
    @apply opacity-40 transition-opacity;
  }

  .yorha-nav-link:hover .yorha-nav-arrow {
    @apply opacity-100;
  }

  /* Main Content */
  .yorha-main {
    @apply pt-[73px] min-h-screen transition-all duration-300;
  }

  .yorha-main-sidebar-open {
    @apply lg:pl-80;
  }

  /* Overlay */
  .yorha-overlay {
    @apply fixed inset-0 bg-black bg-opacity-50 z-20;
  }

  /* Responsive */
  @media (max-width: 1024px) {
    .yorha-main-sidebar-open {
      @apply pl-0;
    }
  }

  @media (max-width: 768px) {
    .yorha-header-content {
      @apply px-4 py-3;
    }

    .yorha-brand-title {
      @apply text-lg;
    }

    .yorha-status-bar {
      @apply gap-4;
    }

    .yorha-quick-actions {
      @apply hidden;
    }

    .yorha-sidebar {
      @apply w-full;
    }
  }
</style>