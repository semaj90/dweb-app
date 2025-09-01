<!-- YoRHa Navigation Sidebar Component -->
<script lang="ts">
  import type { Props } from "$lib/types/global";
  import { page } from "$app/stores";
  import { goto } from "$app/navigation";
  import { onMount } from 'svelte';
  
  // Props
  let { routes, collapsed, currentPage } = $props<{
    routes: Array<{
      id: string;
      label: string;
      route: string;
    }>;
    currentPage: string;
  }>();

  // Navigation state
  let expandedSections = $state(new Set(['main', 'demo', 'admin']););
  let searchQuery = $state('');
  let filteredRoutes = $state(routes);
  let hoveredRoute = $state<string | null>(null);

  // Organize routes into sections
  let organizedRoutes = $derived(() => {
    const sections = {
      main: routes.filter(r => 
        ['command-center', 'cases', 'evidence', 'persons', 'ai-assistant', 'search', 'documents', 'reports', 'memory', 'chat'].includes(r.id)
      ),
      demo: routes.filter(r => r.id.startsWith('demo-') || r.id === 'enhanced-rag' || r.id === 'context7'),
      admin: routes.filter(r => 
        ['dev-tools', 'security', 'settings', 'profile', 'help'].includes(r.id)
      )
    };
    return sections;
  });

  // Filter routes based on search
  $effect(() => {
    if (searchQuery.trim()) {
      filteredRoutes = routes.filter(route => 
        route.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
        route.id.toLowerCase().includes(searchQuery.toLowerCase())
      );
    } else {
      filteredRoutes = routes;
    }
  });

  function navigateToRoute(route: string) {
    if (route !== $page.url.pathname) {
      goto(route);
    }
  }

  function toggleSection(sectionId: string) {
    if (expandedSections.has(sectionId)) {
      expandedSections.delete(sectionId);
    } else {
      expandedSections.add(sectionId);
    }
    expandedSections = new Set(expandedSections);
  }

  function isRouteActive(route: string): boolean {
    return $page.url.pathname === route;
  }

  function getSectionIcon(sectionId: string): string {
    switch (sectionId) {
      case 'main': return '⚡';
      case 'demo': return '🎯';
      case 'admin': return '⚙️';
    }
  }

  function getSectionTitle(sectionId: string): string {
    switch (sectionId) {
      case 'main': return 'CORE OPERATIONS';
      case 'demo': return 'AI DEMONSTRATIONS';
      case 'admin': return 'SYSTEM ADMINISTRATION';
    }
  }

  // Auto-scroll to active route
  onMount(() => {
    const activeElement = document.querySelector('.nav-item.active');
    if (activeElement) {
      activeElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
</script>

<!-- Navigation Container -->
<div class="yorha-3d-panel nes-legal-container h-full flex flex-col">
  
  <!-- Search Bar (when not collapsed) -->
  {#if !collapsed}
    <div class="nes-search-section neural-sprite-loading">
      <div class="relative">
        <input
          bind:value={searchQuery}
          placeholder="Search operations..."
          class="nes-legal-priority-medium yorha-3d-button w-full"
        />
        <div class="neural-sprite-active absolute right-3 top-2.5">
          🔍
        </div>
      </div>
      
      {#if searchQuery}
        <div class="nes-legal-priority-low mt-2 text-xs">
          Found {filteredRoutes.length} operation{filteredRoutes.length !== 1 ? 's' : ''}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Navigation Content -->
  <div class="nes-nav-content neural-sprite-active flex-1 overflow-y-auto custom-scrollbar">
    
    {#if searchQuery}
      <!-- Search Results -->
      <div class="search-results p-2">
        {#each filteredRoutes as route}
          <button
            class="nes-nav-item nes-legal-priority-medium yorha-3d-button w-full text-left {isRouteActive(route.route) ? 'nes-legal-priority-high neural-sprite-active' : ''}"
            on:onclick={() => navigateToRoute(route.route)}
            on:on:mouseenter={() => hoveredRoute = route.id}
            on:on:mouseleave={() => hoveredRoute = null}
          >
            <div class="flex items-center gap-3">
              <span class="neural-sprite-active text-lg">{route.icon}</span>
              {#if !collapsed}
                <span class="nes-legal-title text-sm font-medium">{route.label}</span>
              {/if}
            </div>
          </button>
        {/each}
      </div>
    {:else}
      <!-- Organized Navigation -->
      <div class="organized-nav p-2">
        {#each Object.entries(organizedRoutes) as [sectionId, sectionRoutes]}
          <div class="nav-section mb-4">
            
            <!-- Section Header -->
            <button
              class="section-header w-full flex items-center justify-between p-2 rounded hover:bg-yorha-accent-warm/10 transition-colors duration-200"
              on:onclick={() => toggleSection(sectionId)}
            >
              <div class="flex items-center gap-2">
                <span class="text-lg">{getSectionIcon(sectionId)}</span>
                {#if !collapsed}
                  <span class="text-xs font-bold text-yorha-accent-warm uppercase tracking-wider">
                    {getSectionTitle(sectionId)}
                  </span>
                {/if}
              </div>
              {#if !collapsed}
                <span class="text-yorha-muted transition-transform duration-200 {expandedSections.has(sectionId) ? 'rotate-90' : ''}">
                  ▶
                </span>
              {/if}
            </button>

            <!-- Section Routes -->
            {#if expandedSections.has(sectionId) || collapsed}
              <div class="section-routes ml-2 mt-1">
                {#each sectionRoutes as route}
                  <button
                    class="nav-item w-full text-left p-3 mb-1 rounded border border-transparent hover:border-yorha-accent-warm/50 hover:bg-yorha-accent-warm/10 transition-all duration-200 {isRouteActive(route.route) ? 'active bg-yorha-accent-warm/20 border-yorha-accent-warm text-yorha-accent-warm' : 'text-yorha-light'}"
                    on:onclick={() => navigateToRoute(route.route)}
                    on:on:mouseenter={() => hoveredRoute = route.id}
                    on:on:mouseleave={() => hoveredRoute = null}
                    title={collapsed ? route.label : ''}
                  >
                    <div class="flex items-center gap-3">
                      <span class="text-lg">{route.icon}</span>
                      {#if !collapsed}
                        <div class="flex flex-col">
                          <span class="text-sm font-medium">{route.label}</span>
                          {#if hoveredRoute === route.id}
                            <span class="text-xs text-yorha-muted mt-1">
                              {route.route}
                            </span>
                          {/if}
                        </div>
                      {/if}
                    </div>
                    
                    <!-- Active indicator -->
                    {#if isRouteActive(route.route)}
                      <div class="absolute right-2 top-1/2 transform -translate-y-1/2">
                        <div class="w-2 h-2 bg-yorha-accent-warm rounded-full animate-pulse"></div>
                      </div>
                    {/if}
                  </button>
                {/each}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Footer Info (when not collapsed) -->
  {#if !collapsed}
    <div class="nav-footer p-4 border-t border-yorha-accent-warm/30 bg-yorha-darker">
      <div class="text-xs text-yorha-muted space-y-1">
        <div class="flex items-center justify-between">
          <span>Active Routes:</span>
          <span class="text-yorha-accent-warm">{routes.length}</span>
        </div>
        <div class="flex items-center justify-between">
          <span>Current:</span>
          <span class="text-yorha-accent-warm font-mono text-[10px]">
            {$page.url.pathname}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span>System:</span>
          <span class="text-green-400">OPERATIONAL</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .yorha-navigation {
    --yorha-primary: #c4b49a;
    --yorha-secondary: #b5a48a;
    --yorha-accent-warm: #4a4a4a;
    --yorha-accent-cool: #6b6b6b;
    --yorha-light: #ffffff;
    --yorha-muted: #f0f0f0;
    --yorha-dark: #aca08a;
    --yorha-darker: #b8ad98;
  }

  .nav-item {
    position: relative;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
  }

  .nav-item.active {
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
  }

  .nav-item:hover {
    transform: translateX(2px);
  }

  .section-header {
    font-family: 'JetBrains Mono', monospace;
  }

  /* Custom Scrollbar */
  .custom-scrollbar::-webkit-scrollbar {
    width: 8px;
  }

  .custom-scrollbar::-webkit-scrollbar-track {
    background: var(--yorha-darker);
    border-radius: 4px;
  }

  .custom-scrollbar::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--yorha-accent-warm) 0%, #b8941f 100%);
    border-radius: 4px;
    border: 1px solid var(--yorha-accent-warm);
  }

  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, var(--yorha-accent-cool) 0%, #6bb8e6 100%);
  }

  /* Smooth animations */
  .nav-item,
  .section-header {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* Glow effect for active items */
  .nav-item.active::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.1), transparent);
    border-radius: inherit;
    opacity: 0;
    animation: glow 2s ease-in-out infinite alternate;
  }

  @keyframes glow {
    from { opacity: 0.3; }
    to { opacity: 0.7; }
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .nav-item {
      padding: 12px;
    }
    
    .section-header {
      padding: 8px;
    }
  }
</style>



