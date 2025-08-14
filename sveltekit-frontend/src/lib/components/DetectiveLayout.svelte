<!-- YoRHa Detective Interface Layout -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';

  interface NavItem {
    label: string;
    href: string;
    icon: string;
    active?: boolean;
  }

  let navItems: NavItem[] = [
    { label: 'COMMAND CENTER', href: '/', icon: '🏠' },
    { label: 'ACTIVE CASES', href: '/cases', icon: '📁' },
    { label: 'EVIDENCE', href: '/evidence', icon: '📋' },
    { label: 'PERSONS OF INTEREST', href: '/persons', icon: '👥' },
    { label: 'ANALYSIS', href: '/analysis', icon: '📊' },
    { label: 'GLOBAL SEARCH', href: '/search', icon: '🔍' },
    { label: 'TERMINAL', href: '/terminal', icon: '💻' },
    { label: 'SYSTEM CONFIG', href: '/config', icon: '⚙️' }
  ];

  // Mark current page as active
  $: navItems = navItems.map(item => ({
    ...item,
    active: $page.url.pathname === item.href
  }));

  function handleNavigation(href: string) {
    goto(href);
  }
</script>

<div class="min-h-screen bg-stone-900 text-stone-100 font-mono flex">
  <!-- Left Sidebar Navigation -->
  <aside class="w-64 bg-stone-800 border-r border-stone-600 flex flex-col">
    <!-- YoRHa Detective Header -->
    <div class="p-4 border-b border-stone-600">
      <div class="text-center">
        <h1 class="text-xl font-bold tracking-wider text-stone-100">YORHA</h1>
        <h2 class="text-lg tracking-wider text-stone-100">DETECTIVE</h2>
        <p class="text-xs text-stone-400 mt-1">Investigation Interface</p>
      </div>
    </div>

    <!-- Navigation Menu -->
    <nav class="flex-1 p-4">
      <ul class="space-y-1">
        {#each navItems as item}
          <li>
            <button
              on:click={() => handleNavigation(item.href)}
              class="w-full flex items-center gap-3 px-3 py-2 text-left text-sm transition-colors
                     {item.active 
                       ? 'bg-stone-700 text-stone-100 border-l-2 border-blue-500' 
                       : 'text-stone-300 hover:bg-stone-700 hover:text-stone-100'}"
            >
              <span class="text-xs">{item.icon}</span>
              <span class="tracking-wider">{item.label}</span>
              {#if item.active}
                <span class="ml-auto text-xs">▶</span>
              {/if}
            </button>
          </li>
        {/each}
      </ul>
    </nav>

    <!-- Bottom Status -->
    <div class="p-4 border-t border-stone-600">
      <div class="flex items-center gap-2 text-xs text-stone-400">
        <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
        <span>Online</span>
      </div>
      <div class="text-xs text-stone-500 mt-1">
        System: Operational
      </div>
      <div class="text-xs text-stone-500">
        {new Date().toLocaleTimeString()}
      </div>
    </div>
  </aside>

  <!-- Main Content Area -->
  <main class="flex-1 p-6 overflow-auto">
    <slot />
  </main>
</div>

<style>
  :global(body) {
    background-color: #1c1917;
    color: #e7e5e4;
    font-family: 'Courier New', monospace;
  }
</style>