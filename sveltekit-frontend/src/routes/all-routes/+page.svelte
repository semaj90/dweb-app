<!--
  All Routes Index - Dynamic Route Discovery and Navigation
  YoRHa-themed comprehensive route listing
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import Button from '$lib/components/ui/button/Button.svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';

  // Import route configuration
  import { allRoutes, routeCategories } from '$lib/data/routes-config';

  // Discover all route modules
  const modules = import.meta.glob('/src/routes/**/+page.svelte');

  // State management
  let searchValue = '';
  let categoryValue = 'all';
  let sortValue: 'name' | 'category' | 'status' = 'name';

  // Build discovered routes from file system
  let discoveredRoutes: string[] = [];
  let allAvailableRoutes: any[] = [];
  let filteredRoutes: any[] = [];

  // Update discovered routes
  $: discoveredRoutes = Object.keys(modules)
    .map((path) => {
      let route = path
        .replace('/src/routes', '')
        .replace('/+page.svelte', '')
        .replace('/+layout.svelte', '');

      // Handle root route
      if (route === '') route = '/';

      // Handle dynamic routes
      route = route.replace(/\[([^\]]+)\]/g, ':$1');

      return route;
    })
    .filter((r, i, arr) => arr.indexOf(r) === i) // unique
    .filter(r => r !== '/+error') // filter out error pages
    .sort();

  // Combine configured routes with discovered routes
  $: allAvailableRoutes = [
    ...allRoutes.map(route => ({
      ...route,
      type: 'configured',
      available: discoveredRoutes.includes(route.route)
    })),
    ...discoveredRoutes
      .filter(route => !allRoutes.some(r => r.route === route))
      .map(route => ({
        id: route.replace(/[\/\:]/g, '-').slice(1) || 'home',
        label: formatRouteLabel(route),
        route,
        icon: inferRouteIcon(route),
        description: `Discovered route: ${route}`,
        category: inferRouteCategory(route),
        status: 'discovered',
        tags: [],
        type: 'discovered',
        available: true
      }))
  ];

  // Filter and sort routes
  $: filteredRoutes = allAvailableRoutes
    .filter(route => {
      // Search filter
      if (searchValue) {
        const searchLower = searchValue.toLowerCase();
        return (
          route.label.toLowerCase().includes(searchLower) ||
          route.route.toLowerCase().includes(searchLower) ||
          route.description.toLowerCase().includes(searchLower) ||
          (route.tags && route.tags.some((tag: string) => tag.toLowerCase().includes(searchLower)))
        );
      }
      return true;
    })
    .filter(route => {
      // Category filter
      if (categoryValue === 'all') return true;
      if (categoryValue === 'available') return route.available;
      if (categoryValue === 'configured') return route.type === 'configured';
      if (categoryValue === 'discovered') return route.type === 'discovered';
      return route.category === categoryValue;
    })
    .sort((a, b) => {
      switch (sortValue) {
        case 'category':
          return a.category.localeCompare(b.category) || a.label.localeCompare(b.label);
        case 'status':
          return a.status.localeCompare(b.status) || a.label.localeCompare(b.label);
        default:
          return a.label.localeCompare(b.label);
      }
    });

  function formatRouteLabel(route: string): string {
    if (route === '/') return 'Home';

    return route
      .split('/')
      .filter(Boolean)
      .map(segment =>
        segment
          .replace(/^:/, '') // Remove parameter prefix
          .split('-')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ')
      )
      .join(' → ');
  }

  function inferRouteIcon(route: string): string {
    if (route === '/') return '🏠';
    if (route.includes('demo')) return '🎯';
    if (route.includes('dev')) return '🔧';
    if (route.includes('ai')) return '🤖';
    if (route.includes('legal')) return '⚖️';
    if (route.includes('admin')) return '⚙️';
    if (route.includes('case')) return '📁';
    if (route.includes('evidence')) return '🔍';
    if (route.includes('chat')) return '💬';
    if (route.includes('search')) return '🔎';
    if (route.includes('upload')) return '📤';
    if (route.includes('report')) return '📊';
    if (route.includes('setting')) return '⚙️';
    if (route.includes('profile')) return '👤';
    if (route.includes('help')) return '❓';
    if (route.includes('security')) return '🛡️';
    return '📄';
  }

  function inferRouteCategory(route: string): string {
    if (route.includes('/demo/')) return 'demo';
    if (route.includes('/dev/')) return 'dev';
    if (route.includes('/ai/') || route.includes('/chat/')) return 'ai';
    if (route.includes('/legal/')) return 'legal';
    if (route.includes('/admin/') || route.includes('/settings/')) return 'admin';
    return 'main';
  }

  async function navigateToRoute(route: string) {
    try {
      await goto(route);
    } catch (error) {
      console.error('Navigation failed:', error);
      alert(`Failed to navigate to ${route}`);
    }
  }

  function getRouteStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-300 border-green-500/30';
      case 'experimental': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case 'beta': return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'deprecated': return 'bg-red-500/20 text-red-300 border-red-500/30';
      case 'discovered': return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  }

  onMount(() => {
    console.log('Discovered routes:', discoveredRoutes);
    console.log('All available routes:', allAvailableRoutes);
  });
</script>

<svelte:head>
  <title>YoRHa Routes Index - Legal AI System</title>
</svelte:head>

<div class="min-h-screen bg-yorha-bg-primary text-yorha-text-primary p-6">
  <div class="max-w-7xl mx-auto space-y-6">
    <!-- Header -->
    <div class="text-center border-b border-yorha-text-muted pb-6">
      <h1 class="text-5xl font-bold text-yorha-secondary mb-4">
        📚 YoRHa Routes Index
      </h1>
      <p class="text-yorha-text-secondary text-xl mb-2">
        Complete navigation system for the Legal AI Platform
      </p>
      <p class="text-yorha-text-muted">
        {allAvailableRoutes.length} total routes • {filteredRoutes.length} displayed
      </p>
    </div>

    <!-- Controls -->
    <Card class="p-6">
      <div class="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <!-- Search -->
        <div class="flex-1 max-w-md">
          <label for="search-routes" class="block text-sm font-medium text-yorha-text-accent mb-2">
            Search Routes
          </label>
          <input
            id="search-routes"
            bind:value={searchValue}
            placeholder="Search by name, path, or description..."
            class="w-full bg-yorha-bg-secondary border border-yorha-text-muted p-3 rounded text-yorha-text-primary placeholder-yorha-text-muted focus:border-yorha-secondary focus:ring-2 focus:ring-yorha-secondary/20"
          />
        </div>

        <!-- Category Filter -->
        <div>
          <label for="category-filter" class="block text-sm font-medium text-yorha-text-accent mb-2">
            Category
          </label>
          <select
            id="category-filter"
            bind:value={categoryValue}
            class="bg-yorha-bg-secondary border border-yorha-text-muted p-3 rounded text-yorha-text-primary focus:border-yorha-secondary focus:ring-2 focus:ring-yorha-secondary/20"
          >
            <option value="all">All Categories</option>
            <option value="available">Available Only</option>
            <option value="configured">Configured</option>
            <option value="discovered">Discovered</option>
            <option value="main">Main</option>
            <option value="demo">Demo</option>
            <option value="ai">AI</option>
            <option value="legal">Legal</option>
            <option value="dev">Development</option>
            <option value="admin">Admin</option>
          </select>
        </div>

        <!-- Sort -->
        <div>
          <label for="sort-by" class="block text-sm font-medium text-yorha-text-accent mb-2">
            Sort By
          </label>
          <select
            id="sort-by"
            bind:value={sortValue}
            class="bg-yorha-bg-secondary border border-yorha-text-muted p-3 rounded text-yorha-text-primary focus:border-yorha-secondary focus:ring-2 focus:ring-yorha-secondary/20"
          >
            <option value="name">Name</option>
            <option value="category">Category</option>
            <option value="status">Status</option>
          </select>
        </div>
      </div>
    </Card>

    <!-- Statistics -->
    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
      {#each Object.entries(routeCategories) as [category, info]}
        {@const count = allAvailableRoutes.filter(r => r.category === category).length}
        {#if count > 0}
          <Card class="p-4 text-center">
            <div class="text-2xl mb-2" style="color: {info.color}">{info.icon}</div>
            <div class="text-lg font-semibold text-yorha-text-primary">{count}</div>
            <div class="text-sm text-yorha-text-secondary">{info.label}</div>
          </Card>
        {/if}
      {/each}
    </div>

    <!-- Routes Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {#each filteredRoutes as route}
        <Card class="p-4 hover:bg-yorha-bg-secondary/50 transition-colors">
          <div class="flex items-start justify-between mb-3">
            <div class="flex items-center gap-2">
              <span class="text-xl">{route.icon}</span>
              <span class="text-sm font-medium text-yorha-text-accent uppercase tracking-wide">
                {route.category}
              </span>
            </div>

            <div class="flex gap-1">
              {#if route.available}
                <Badge variant="outline" class="text-xs {getRouteStatusColor(route.status)}">
                  {route.status}
                </Badge>
              {:else}
                <Badge variant="destructive" class="text-xs">
                  Missing
                </Badge>
              {/if}

              {#if route.type === 'discovered'}
                <Badge variant="secondary" class="text-xs">
                  Discovered
                </Badge>
              {/if}
            </div>
          </div>

          <h3 class="text-lg font-semibold text-yorha-text-primary mb-2 leading-tight">
            {route.label}
          </h3>

          <code class="block text-xs text-yorha-accent bg-yorha-bg-primary p-2 rounded mb-3 font-mono">
            {route.route}
          </code>

          <p class="text-sm text-yorha-text-secondary mb-4 line-clamp-2">
            {route.description}
          </p>

          {#if route.tags && route.tags.length > 0}
            <div class="flex flex-wrap gap-1 mb-4">
              {#each route.tags.slice(0, 3) as tag}
                <span class="text-xs bg-yorha-bg-tertiary text-yorha-text-muted px-2 py-1 rounded">
                  {tag}
                </span>
              {/each}
              {#if route.tags.length > 3}
                <span class="text-xs text-yorha-text-muted">
                  +{route.tags.length - 3}
                </span>
              {/if}
            </div>
          {/if}

          <Button
            onclick={() => navigateToRoute(route.route)}
            disabled={!route.available}
            class="w-full {route.available
              ? 'bg-yorha-secondary text-yorha-bg-primary hover:bg-yorha-secondary-dark'
              : 'bg-yorha-bg-tertiary text-yorha-text-muted cursor-not-allowed'
            }"
            size="sm"
          >
            {route.available ? 'Navigate' : 'Unavailable'}
          </Button>
        </Card>
      {/each}
    </div>

    {#if filteredRoutes.length === 0}
      <Card class="p-8 text-center">
        <div class="text-4xl mb-4">🔍</div>
        <h3 class="text-xl font-semibold text-yorha-text-primary mb-2">No Routes Found</h3>
        <p class="text-yorha-text-secondary">
          Try adjusting your search or filter criteria.
        </p>
      </Card>
    {/if}

    <!-- Footer Info -->
    <Card class="p-6">
      <h2 class="text-xl font-semibold text-yorha-accent mb-4">System Information</h2>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <h3 class="font-semibold text-yorha-text-primary mb-2">Route Discovery</h3>
          <ul class="text-sm text-yorha-text-secondary space-y-1">
            <li>• Configured routes: {allRoutes.length}</li>
            <li>• Discovered routes: {discoveredRoutes.length}</li>
            <li>• Available routes: {allAvailableRoutes.filter(r => r.available).length}</li>
            <li>• Missing routes: {allAvailableRoutes.filter(r => !r.available).length}</li>
          </ul>
        </div>

        <div>
          <h3 class="font-semibold text-yorha-text-primary mb-2">Navigation</h3>
          <ul class="text-sm text-yorha-text-secondary space-y-1">
            <li>• Current path: <code class="text-yorha-accent">{$page.url.pathname}</code></li>
            <li>• Route ID: <code class="text-yorha-accent">{$page.route?.id ?? 'unknown'}</code></li>
            <li>• Page params: <code class="text-yorha-accent">{JSON.stringify($page.params)}</code></li>
          </ul>
        </div>

        <div>
          <h3 class="font-semibold text-yorha-text-primary mb-2">Quick Actions</h3>
          <div class="space-y-2">
            <Button
              size="sm"
              variant="outline"
              onclick={() => goto('/')}
              class="w-full border-yorha-accent text-yorha-accent hover:bg-yorha-accent hover:text-yorha-bg-primary"
            >
              🏠 Go Home
            </Button>
            <Button
              size="sm"
              variant="outline"
              onclick={() => goto('/dev/dynamic-routing-test')}
              class="w-full border-yorha-accent text-yorha-accent hover:bg-yorha-accent hover:text-yorha-bg-primary"
            >
              🛣️ Routing Test
            </Button>
          </div>
        </div>
      </div>
    </Card>
  </div>
</div>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>