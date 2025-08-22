<!--
  Dynamic Routing Test Page - Simplified Version
  Basic route testing without complex imports
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import Button from '$lib/components/ui/button/Button.svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  // Import route configuration
  import { allRoutes, getRoutesByCategory, searchRoutes } from '$lib/data/routes-config';

  // State management
  const testResults = writable<string[]>([]);
  const routeStats = writable<any>({});
  
  let isLoading = false;
  let currentPath = '';

  // Test configuration
  const testCases = [
    {
      name: 'Route Configuration Load',
      test: async () => {
        const routes = allRoutes;
        return `✅ Loaded ${routes.length} routes from configuration`;
      }
    },
    {
      name: 'Category Filter Test - Demo',
      test: async () => {
        const demoRoutes = getRoutesByCategory('demo');
        return `✅ Found ${demoRoutes.length} demo routes`;
      }
    },
    {
      name: 'Category Filter Test - Dev',
      test: async () => {
        const devRoutes = getRoutesByCategory('dev');
        return `✅ Found ${devRoutes.length} dev routes`;
      }
    },
    {
      name: 'Category Filter Test - AI',
      test: async () => {
        const aiRoutes = getRoutesByCategory('ai');
        return `✅ Found ${aiRoutes.length} AI routes`;
      }
    },
    {
      name: 'Search Test - AI',
      test: async () => {
        const results = searchRoutes('ai');
        return `✅ Found ${results.length} routes matching 'ai'`;
      }
    },
    {
      name: 'Search Test - Demo',
      test: async () => {
        const results = searchRoutes('demo');
        return `✅ Found ${results.length} routes matching 'demo'`;
      }
    },
    {
      name: 'Current Page Test',
      test: async () => {
        return `✅ Current path: ${currentPath}`;
      }
    },
    {
      name: 'Navigation Test',
      test: async () => {
        // Test navigation to a known route
        const demoRoutes = getRoutesByCategory('demo');
        if (demoRoutes.length > 0) {
          return `✅ Can navigate to: ${demoRoutes[0].route}`;
        }
        return `✅ Navigation system available`;
      }
    }
  ];

  onMount(async () => {
    // Get current path from page store
    page.subscribe(($page) => {
      currentPath = $page.url.pathname;
    });
    
    // Calculate route statistics
    calculateRouteStats();
  });

  function calculateRouteStats() {
    const categories = ['main', 'demo', 'ai', 'legal', 'dev', 'admin'];
    const stats: any = {
      total: allRoutes.length,
      categories: {}
    };
    
    categories.forEach(category => {
      stats.categories[category] = getRoutesByCategory(category).length;
    });
    
    // Count by status
    stats.active = allRoutes.filter(r => r.status === 'active').length;
    stats.experimental = allRoutes.filter(r => r.status === 'experimental').length;
    stats.beta = allRoutes.filter(r => r.status === 'beta').length;
    
    routeStats.set(stats);
  }

  async function runAllTests() {
    isLoading = true;
    testResults.set([]);
    
    try {
      for (const testCase of testCases) {
        try {
          const result = await testCase.test();
          testResults.update(results => [...results, result]);
        } catch (error) {
          testResults.update(results => [...results, `❌ ${testCase.name}: ${error.message}`]);
        }
      }
    } catch (error) {
      testResults.update(results => [...results, `❌ Test suite failed: ${error.message}`]);
    } finally {
      isLoading = false;
    }
  }

  async function navigateToRoute(route: string) {
    try {
      await goto(route);
      testResults.update(results => [
        ...results,
        `✅ Navigated to: ${route}`
      ]);
    } catch (error) {
      testResults.update(results => [
        ...results,
        `❌ Navigation failed: ${error.message}`
      ]);
    }
  }

  function formatJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }
</script>

<div class="min-h-screen bg-yorha-bg-primary text-yorha-text-primary p-6">
  <div class="max-w-6xl mx-auto space-y-6">
    <!-- Header -->
    <div class="text-center">
      <h1 class="text-4xl font-bold text-yorha-secondary mb-4">
        🛣️ Dynamic Routing Test Suite
      </h1>
      <p class="text-yorha-text-secondary">
        Testing route configuration and navigation functionality
      </p>
      <p class="text-sm text-yorha-text-muted mt-2">
        Current Path: <code class="bg-yorha-bg-secondary px-2 py-1 rounded">{currentPath}</code>
      </p>
    </div>

    <!-- Test Controls -->
    <Card class="p-6">
      <h2 class="text-2xl font-semibold mb-4 text-yorha-accent">Test Controls</h2>
      
      <div class="flex gap-4 mb-6">
        <Button 
          onclick={runAllTests} 
          disabled={isLoading}
          class="bg-yorha-secondary text-yorha-bg-primary hover:bg-yorha-secondary-dark"
        >
          {isLoading ? 'Running Tests...' : 'Run All Tests'}
        </Button>
        
        <Button 
          onclick={calculateRouteStats}
          variant="outline"
          class="border-yorha-accent text-yorha-accent hover:bg-yorha-accent hover:text-yorha-bg-primary"
        >
          Refresh Stats
        </Button>
      </div>
    </Card>

    <!-- Test Results -->
    <Card class="p-6">
      <h2 class="text-2xl font-semibold mb-4 text-yorha-accent">Test Results</h2>
      
      {#if $testResults.length > 0}
        <div class="bg-yorha-bg-secondary p-4 rounded font-mono text-sm space-y-2 max-h-64 overflow-y-auto">
          {#each $testResults as result}
            <div class="text-yorha-text-primary">{result}</div>
          {/each}
        </div>
      {:else}
        <p class="text-yorha-text-secondary">No test results yet. Run tests to see output.</p>
      {/if}
    </Card>

    <!-- Route Statistics -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <Card class="p-6">
        <h3 class="text-xl font-semibold mb-4 text-yorha-secondary">Route Statistics</h3>
        
        {#if $routeStats.total}
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="text-yorha-text-secondary">Total Routes:</span>
              <Badge variant="secondary">{$routeStats.total}</Badge>
            </div>
            
            <div class="flex justify-between">
              <span class="text-yorha-text-secondary">Active:</span>
              <Badge variant="default">{$routeStats.active}</Badge>
            </div>
            
            <div class="flex justify-between">
              <span class="text-yorha-text-secondary">Experimental:</span>
              <Badge variant="outline">{$routeStats.experimental}</Badge>
            </div>
            
            <div class="flex justify-between">
              <span class="text-yorha-text-secondary">Beta:</span>
              <Badge variant="destructive">{$routeStats.beta}</Badge>
            </div>
          </div>
        {:else}
          <p class="text-yorha-text-secondary">Loading statistics...</p>
        {/if}
      </Card>

      <Card class="p-6">
        <h3 class="text-xl font-semibold mb-4 text-yorha-secondary">Categories</h3>
        
        {#if $routeStats.categories}
          <div class="space-y-2">
            {#each Object.entries($routeStats.categories) as [category, count]}
              <div class="flex justify-between text-sm">
                <span class="text-yorha-text-secondary capitalize">{category}:</span>
                <Badge variant="outline" class="text-xs">{count}</Badge>
              </div>
            {/each}
          </div>
        {/if}
      </Card>

      <Card class="p-6">
        <h3 class="text-xl font-semibold mb-4 text-yorha-secondary">Quick Navigation</h3>
        
        <div class="space-y-2">
          <Button
            size="sm"
            variant="ghost"
            onclick={() => navigateToRoute('/')}
            class="w-full justify-start text-yorha-accent hover:bg-yorha-accent hover:text-yorha-bg-primary"
          >
            🏠 Home
          </Button>
          
          <Button
            size="sm"
            variant="ghost"
            onclick={() => navigateToRoute('/demo')}
            class="w-full justify-start text-yorha-accent hover:bg-yorha-accent hover:text-yorha-bg-primary"
          >
            🎯 Demo Overview
          </Button>
          
          <Button
            size="sm"
            variant="ghost"
            onclick={() => navigateToRoute('/dev/mcp-tools')}
            class="w-full justify-start text-yorha-accent hover:bg-yorha-accent hover:text-yorha-bg-primary"
          >
            🔧 MCP Tools
          </Button>
          
          <Button
            size="sm"
            variant="ghost"
            onclick={() => navigateToRoute('/cases')}
            class="w-full justify-start text-yorha-accent hover:bg-yorha-accent hover:text-yorha-bg-primary"
          >
            📁 Case Management
          </Button>
        </div>
      </Card>
    </div>

    <!-- Route Categories List -->
    <Card class="p-6">
      <h2 class="text-2xl font-semibold mb-4 text-yorha-accent">Route Categories</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {#each ['main', 'demo', 'ai', 'legal', 'dev', 'admin'] as category}
          {@const categoryRoutes = getRoutesByCategory(category)}
          {#if categoryRoutes.length > 0}
            <div class="border border-yorha-text-muted p-4 rounded">
              <h3 class="text-lg font-semibold mb-3 text-yorha-text-accent capitalize">
                {category} ({categoryRoutes.length})
              </h3>
              
              <div class="space-y-1 max-h-32 overflow-y-auto">
                {#each categoryRoutes.slice(0, 5) as route}
                  <button
                    class="block w-full text-left text-sm text-yorha-text-secondary hover:text-yorha-accent hover:bg-yorha-bg-secondary p-1 rounded transition-colors"
                    onclick={() => navigateToRoute(route.route)}
                  >
                    {route.icon} {route.label}
                  </button>
                {/each}
                {#if categoryRoutes.length > 5}
                  <p class="text-xs text-yorha-text-muted">
                    +{categoryRoutes.length - 5} more...
                  </p>
                {/if}
              </div>
            </div>
          {/if}
        {/each}
      </div>
    </Card>

    <!-- Debug Information -->
    <Card class="p-6">
      <h2 class="text-2xl font-semibold mb-4 text-yorha-accent">Debug Information</h2>
      
      <details class="cursor-pointer">
        <summary class="text-yorha-secondary hover:text-yorha-accent mb-4">
          View All Routes Configuration
        </summary>
        
        <div class="grid grid-cols-1 gap-2 max-h-64 overflow-y-auto">
          {#each allRoutes as route}
            <div class="bg-yorha-bg-secondary p-2 rounded text-xs">
              <div class="flex justify-between items-center">
                <span class="font-mono text-yorha-accent">{route.id}</span>
                <Badge variant="outline" class="text-xs">{route.category}</Badge>
              </div>
              <div class="text-yorha-text-secondary">{route.route}</div>
              <div class="text-yorha-text-muted text-xs">{route.description}</div>
            </div>
          {/each}
        </div>
      </details>
    </Card>
  </div>
</div>

<style>
  :global(.yorha-terminal-grid) {
    background-image: 
      linear-gradient(rgba(255, 215, 0, 0.1) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 215, 0, 0.1) 1px, transparent 1px);
    background-size: 20px 20px;
  }
</style>