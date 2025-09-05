<script lang="ts">
  import type { PageData } from './$types';
  
  let { data } = $props<{ data: PageData }>();
  
  // Component state
  let selectedCategory = $state('all');
  let searchTerm = $state('');
  let viewMode = $state<'grid' | 'list'>('grid');
  let showEmptyRoutes = $state(false);
  
  // Derived data
  let filteredRoutes = $derived(() => {
    let routes = data.routes;
    
    // Filter by category
    if (selectedCategory !== 'all') {
      routes = routes.filter(route => route.category === selectedCategory);
    }
    
    // Filter by search term
    if (searchTerm.trim()) {
      const search = searchTerm.toLowerCase();
      routes = routes.filter(route => 
        route.path.toLowerCase().includes(search) || 
        route.description.toLowerCase().includes(search)
      );
    }
    
    // Filter empty routes
    if (!showEmptyRoutes) {
      routes = routes.filter(route => route.status !== 'empty');
    }
    
    return routes;
  });
  
  let categoryStats = $derived(() => {
    const stats: Record<string, { total: number; functional: number; icon: string }> = {};
    
    Object.entries(data.routesByCategory).forEach(([category, routes]) => {
      const categoryInfo = Object.values(data.categories).find(cat => cat.name === category);
      stats[category] = {
        total: routes.length,
        functional: routes.filter(r => r.status === 'functional').length,
        icon: categoryInfo?.icon || '📄'
      };
    });
    
    return stats;
  });

  // Helper functions
  function getStatusIcon(status: string) {
    switch (status) {
      case 'functional': return '✅';
      case 'api': return '🔌';
      case 'layout': return '📐';
      case 'empty': return '📂';
      default: return '❓';
    }
  }
  
  function getStatusColor(status: string) {
    switch (status) {
      case 'functional': return 'text-green-400';
      case 'api': return 'text-blue-400';
      case 'layout': return 'text-purple-400';
      case 'empty': return 'text-gray-500';
      default: return 'text-amber-400';
    }
  }
  
  function getPriorityColor(priority: string) {
    switch (priority) {
      case 'high': return 'border-red-400 text-red-400';
      case 'medium': return 'border-yellow-400 text-yellow-400';
      case 'low': return 'border-gray-400 text-gray-400';
      default: return 'border-amber-400 text-amber-400';
    }
  }

  function handleRouteClick(route: any) {
    if (route.status === 'functional') {
      window.location.href = route.path;
    }
  }
</script>

<svelte:head>
  <title>All Routes - YoRHa Legal AI System</title>
  <meta name="description" content="Complete directory of all routes in the Legal AI system" />
</svelte:head>

<div class="yorha-routes-container">
  <!-- Header Section -->
  <div class="yorha-routes-header">
    <h1 class="yorha-routes-title">SYSTEM ROUTE DIRECTORY</h1>
    <div class="yorha-routes-subtitle">
      Complete navigation map for Legal AI Platform v13.0
    </div>
    
    <!-- Statistics Grid -->
    <div class="yorha-stats-grid">
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">{data.stats.total}</div>
        <div class="yorha-stat-label">TOTAL ROUTES</div>
      </div>
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">{data.stats.functional}</div>
        <div class="yorha-stat-label">FUNCTIONAL</div>
      </div>
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">{data.stats.api}</div>
        <div class="yorha-stat-label">API ENDPOINTS</div>
      </div>
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">{data.stats.categories}</div>
        <div class="yorha-stat-label">CATEGORIES</div>
      </div>
    </div>
  </div>

  <!-- Controls Section -->
  <div class="yorha-controls-section">
    <div class="yorha-controls-row">
      <!-- Search -->
      <div class="yorha-search-container">
        <input
          bind:value={searchTerm}
          type="text"
          placeholder="SEARCH ROUTES..."
          class="yorha-search-input"
        />
      </div>
      
      <!-- Category Filter -->
      <select bind:value={selectedCategory} class="yorha-select">
        <option value="all">ALL CATEGORIES</option>
        {#each Object.entries(categoryStats) as [category, stats]}
          <option value={category}>
            {stats.icon} {category.toUpperCase()} ({stats.functional}/{stats.total})
          </option>
        {/each}
      </select>
      
      <!-- View Mode -->
      <div class="yorha-toggle-group">
        <button
          class="yorha-toggle-btn"
          class:active={viewMode === 'grid'}
          onclick={() => viewMode = 'grid'}
        >
          GRID
        </button>
        <button
          class="yorha-toggle-btn"
          class:active={viewMode === 'list'}
          onclick={() => viewMode = 'list'}
        >
          LIST
        </button>
      </div>
      
      <!-- Show Empty Toggle -->
      <label class="yorha-checkbox-label">
        <input
          type="checkbox"
          bind:checked={showEmptyRoutes}
          class="yorha-checkbox"
        />
        SHOW EMPTY ROUTES
      </label>
    </div>
  </div>

  <!-- Routes Display -->
  <div class="yorha-routes-content">
    {#if filteredRoutes.length === 0}
      <div class="yorha-no-results">
        <div class="yorha-no-results-icon">🔍</div>
        <div class="yorha-no-results-text">NO ROUTES FOUND</div>
        <div class="yorha-no-results-desc">
          Try adjusting your search criteria or category filter
        </div>
      </div>
    {:else}
      <div class="yorha-routes-grid" class:list-view={viewMode === 'list'}>
        {#each filteredRoutes as route}
          <div 
            class="yorha-route-card {getStatusColor(route.status)} {getPriorityColor(route.priority)}"
            class:clickable={route.status === 'functional'}
            onclick={() => handleRouteClick(route)}
            role={route.status === 'functional' ? 'button' : 'article'}
            tabindex={route.status === 'functional' ? 0 : -1}
          >
            <div class="yorha-route-header">
              <div class="yorha-route-status">
                {getStatusIcon(route.status)}
              </div>
              <div class="yorha-route-path">
                {route.path}
              </div>
              <div class="yorha-route-priority">
                {route.priority.toUpperCase()}
              </div>
            </div>
            
            <div class="yorha-route-info">
              <div class="yorha-route-category">
                {categoryStats[route.category]?.icon} {route.category}
              </div>
              <div class="yorha-route-description">
                {route.description}
              </div>
            </div>
            
            {#if route.isParametric}
              <div class="yorha-route-badge">PARAMETRIC</div>
            {/if}
            
            {#if route.status === 'functional'}
              <div class="yorha-route-action">
                NAVIGATE →
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Category Overview -->
  <div class="yorha-category-overview">
    <h2 class="yorha-section-title">CATEGORY OVERVIEW</h2>
    <div class="yorha-category-grid">
      {#each Object.entries(categoryStats) as [category, stats]}
        <div class="yorha-category-card">
          <div class="yorha-category-icon">{stats.icon}</div>
          <div class="yorha-category-name">{category}</div>
          <div class="yorha-category-stats">
            {stats.functional}/{stats.total} functional
          </div>
          <div class="yorha-category-bar">
            <div 
              class="yorha-category-fill"
              style="width: {(stats.functional / stats.total * 100)}%"
            ></div>
          </div>
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .yorha-routes-container {
    @apply min-h-screen bg-black text-amber-400 font-mono p-6;
    background-image:
      radial-gradient(circle at 25% 25%, rgba(255, 191, 0, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 75% 75%, rgba(255, 215, 0, 0.05) 0%, transparent 50%);
  }

  .yorha-routes-header {
    @apply text-center mb-8 pb-6 border-b border-amber-400;
  }

  .yorha-routes-title {
    @apply text-4xl font-bold mb-2 tracking-wider;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-routes-subtitle {
    @apply text-lg text-amber-300 mb-6;
  }

  .yorha-stats-grid {
    @apply grid grid-cols-2 md:grid-cols-4 gap-4 mt-6;
  }

  .yorha-stat-item {
    @apply text-center bg-gray-900 p-4 border border-amber-400 border-opacity-30;
  }

  .yorha-stat-value {
    @apply text-3xl font-bold text-amber-300;
    text-shadow: 0 0 10px rgba(255, 191, 0, 0.3);
  }

  .yorha-stat-label {
    @apply text-xs text-amber-500 uppercase tracking-wide mt-1;
  }

  .yorha-controls-section {
    @apply mb-8 bg-gray-900 p-6 border border-amber-400;
  }

  .yorha-controls-row {
    @apply flex flex-wrap gap-4 items-center justify-between;
  }

  .yorha-search-container {
    @apply flex-1 min-w-0;
  }

  .yorha-search-input {
    @apply w-full bg-black border border-amber-400 text-amber-400 px-4 py-2;
    @apply focus:outline-none focus:ring-2 focus:ring-amber-400 focus:bg-amber-400 focus:text-black;
    @apply placeholder:text-amber-600;
  }

  .yorha-select {
    @apply bg-black border border-amber-400 text-amber-400 px-4 py-2;
    @apply focus:outline-none focus:ring-2 focus:ring-amber-400;
  }

  .yorha-toggle-group {
    @apply flex border border-amber-400;
  }

  .yorha-toggle-btn {
    @apply bg-black border-r border-amber-400 text-amber-400 px-4 py-2;
    @apply hover:bg-amber-400 hover:text-black transition-all;
    @apply focus:outline-none focus:ring-2 focus:ring-amber-400;
  }

  .yorha-toggle-btn:last-child {
    @apply border-r-0;
  }

  .yorha-toggle-btn.active {
    @apply bg-amber-400 text-black;
  }

  .yorha-checkbox-label {
    @apply flex items-center gap-2 text-amber-400 cursor-pointer;
  }

  .yorha-checkbox {
    @apply w-4 h-4 bg-black border border-amber-400;
  }

  .yorha-routes-content {
    @apply mb-12;
  }

  .yorha-routes-grid {
    @apply grid gap-4;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  }

  .yorha-routes-grid.list-view {
    @apply grid-cols-1;
  }

  .yorha-route-card {
    @apply bg-gray-900 border-2 p-4 transition-all duration-200;
    @apply hover:border-opacity-100 hover:shadow-lg;
    box-shadow: 0 0 15px rgba(255, 191, 0, 0.1);
  }

  .yorha-route-card.clickable {
    @apply cursor-pointer;
  }

  .yorha-route-card.clickable:hover {
    @apply bg-amber-400 bg-opacity-5;
    box-shadow: 0 0 25px rgba(255, 191, 0, 0.3);
  }

  .yorha-route-header {
    @apply flex items-center justify-between mb-3;
  }

  .yorha-route-status {
    @apply text-xl;
  }

  .yorha-route-path {
    @apply font-bold text-lg flex-1 mx-3;
  }

  .yorha-route-priority {
    @apply text-xs px-2 py-1 border opacity-70;
  }

  .yorha-route-info {
    @apply mb-3;
  }

  .yorha-route-category {
    @apply text-sm text-amber-500 mb-1;
  }

  .yorha-route-description {
    @apply text-sm text-amber-300;
  }

  .yorha-route-badge {
    @apply inline-block text-xs px-2 py-1 bg-blue-600 text-blue-100 mb-2;
  }

  .yorha-route-action {
    @apply text-sm text-green-400 font-bold text-right;
  }

  .yorha-no-results {
    @apply text-center py-12;
  }

  .yorha-no-results-icon {
    @apply text-6xl mb-4 opacity-50;
  }

  .yorha-no-results-text {
    @apply text-xl font-bold mb-2;
  }

  .yorha-no-results-desc {
    @apply text-amber-500;
  }

  .yorha-category-overview {
    @apply bg-gray-900 p-6 border border-amber-400;
  }

  .yorha-section-title {
    @apply text-2xl font-bold mb-6 text-center;
  }

  .yorha-category-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4;
  }

  .yorha-category-card {
    @apply bg-black border border-amber-400 border-opacity-30 p-4;
    @apply hover:border-opacity-100 transition-all;
  }

  .yorha-category-icon {
    @apply text-2xl mb-2;
  }

  .yorha-category-name {
    @apply font-bold mb-1;
  }

  .yorha-category-stats {
    @apply text-sm text-amber-500 mb-2;
  }

  .yorha-category-bar {
    @apply w-full bg-gray-800 h-2;
  }

  .yorha-category-fill {
    @apply h-full bg-amber-400 transition-all duration-1000;
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .yorha-routes-title {
      @apply text-2xl;
    }

    .yorha-controls-row {
      @apply flex-col items-stretch;
    }

    .yorha-routes-grid {
      grid-template-columns: 1fr;
    }

    .yorha-stats-grid {
      @apply grid-cols-2;
    }
  }
</style>