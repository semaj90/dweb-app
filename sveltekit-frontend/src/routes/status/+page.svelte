<!--
  Comprehensive System Status Page
  Shows integration status of all components: GPU Cache, Gaming, PostgreSQL, APIs, etc.
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { Button } from 'bits-ui';
  import GPUCacheIntegrationDemo from '$lib/components/ui/gaming/demo/GPUCacheIntegrationDemo.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';

  // System status state
  let systemHealth = $state<any>(null);
  let integrationTests = $state<Record<string, { status: 'success' | 'warning' | 'error'; message: string; details?: any }>>({});
  let isLoading = $state(true);
  let lastUpdated = $state<string>('');

  onMount(async () => {
    if (!browser) return;
    await loadSystemStatus();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadSystemStatus, 30000);
    return () => clearInterval(interval);
  });

  async function loadSystemStatus() {
    try {
      isLoading = true;

      // Load health data
      const healthResponse = await fetch('/api/health');
      if (healthResponse.ok) {
        systemHealth = await healthResponse.json();
      }

      // Test GPU cache integration
      await testGPUCacheIntegration();

      // Test gaming components
      await testGamingComponents();

      // Test PostgreSQL integration
      await testPostgreSQLIntegration();

      // Test API endpoints
      await testAPIEndpoints();

      lastUpdated = new Date().toLocaleTimeString();
    } catch (error) {
      console.error('Failed to load system status:', error);
    } finally {
      isLoading = false;
    }
  }

  async function testGPUCacheIntegration() {
    try {
      // Check CSS custom properties
      const computedStyle = getComputedStyle(document.documentElement);
      const gpuVars = [
        '--gpu-cache-bg-primary',
        '--gpu-cache-accent-primary',
        '--nes-prg-rom-color',
        '--gpu-cache-state-idle'
      ];
      
      const loadedVars = gpuVars.filter(varName => 
        computedStyle.getPropertyValue(varName).trim() !== ''
      );

      if (loadedVars.length === gpuVars.length) {
        integrationTests['gpu-cache'] = {
          status: 'success',
          message: 'GPU cache CSS integration fully loaded',
          details: { loadedVars: loadedVars.length, totalVars: gpuVars.length }
        };
      } else {
        integrationTests['gpu-cache'] = {
          status: 'warning',
          message: `GPU cache CSS partially loaded: ${loadedVars.length}/${gpuVars.length} variables`,
          details: { loadedVars, missingVars: gpuVars.filter(v => !loadedVars.includes(v)) }
        };
      }
    } catch (error) {
      integrationTests['gpu-cache'] = {
        status: 'error',
        message: `GPU cache integration error: ${error}`
      };
    }
  }

  async function testGamingComponents() {
    try {
      // Test gaming constants availability
      const { NES_COLOR_PALETTE, N64_TEXTURE_PRESETS } = await import('$lib/components/ui/gaming/constants/gaming-constants.js');
      
      if (NES_COLOR_PALETTE && N64_TEXTURE_PRESETS) {
        integrationTests['gaming'] = {
          status: 'success',
          message: 'Gaming components and constants loaded successfully',
          details: {
            nesColors: Object.keys(NES_COLOR_PALETTE).length,
            n64Presets: Object.keys(N64_TEXTURE_PRESETS).length
          }
        };
      } else {
        integrationTests['gaming'] = {
          status: 'error',
          message: 'Gaming constants not properly loaded'
        };
      }
    } catch (error) {
      integrationTests['gaming'] = {
        status: 'error',
        message: `Gaming components error: ${error}`
      };
    }
  }

  async function testPostgreSQLIntegration() {
    try {
      const response = await fetch('/api/v1/health');
      if (response.ok) {
        const data = await response.json();
        const pgStatus = data.services?.databases?.postgres?.status;
        
        if (pgStatus === 'healthy') {
          integrationTests['postgresql'] = {
            status: 'success',
            message: 'PostgreSQL + pgvector connected and healthy',
            details: { host: data.services.databases.postgres.host, port: data.services.databases.postgres.port }
          };
        } else {
          integrationTests['postgresql'] = {
            status: 'error',
            message: 'PostgreSQL connection failed or unhealthy'
          };
        }
      } else {
        integrationTests['postgresql'] = {
          status: 'error',
          message: 'Unable to check PostgreSQL status'
        };
      }
    } catch (error) {
      integrationTests['postgresql'] = {
        status: 'error',
        message: `PostgreSQL test error: ${error}`
      };
    }
  }

  async function testAPIEndpoints() {
    try {
      const endpoints = [
        '/api/v1/vector/search',
        '/api/v1/rag',
        '/api/v1/gpu-cache',
        '/api/v1/cluster'
      ];
let successCount = $state(0);
      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint, { method: 'HEAD' });
          if (response.status !== 404) successCount++;
        } catch (e) {
          // Endpoint might not exist yet, that's ok
        }
      }

      integrationTests['api-endpoints'] = {
        status: successCount >= endpoints.length / 2 ? 'success' : 'warning',
        message: `API endpoints: ${successCount}/${endpoints.length} accessible`,
        details: { endpoints, successCount }
      };
    } catch (error) {
      integrationTests['api-endpoints'] = {
        status: 'error',
        message: `API endpoints test error: ${error}`
      };
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'success': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'success': return '✅';
      case 'warning': return '⚠️';
      case 'error': return '❌';
      default: return '❓';
    }
  }

  function getBadgeVariant(status: string) {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'destructive';
      default: return 'secondary';
    }
  }
</script>

<svelte:head>
  <title>System Status - YoRHa Legal AI Platform</title>
  <meta name="description" content="Real-time system status and integration monitoring for the Legal AI platform" />
</svelte:head>

<div class="status-page min-h-screen bg-gradient-to-br from-gray-900 to-black p-6">
  <!-- Header -->
  <header class="mb-8">
    <div class="flex items-center justify-between mb-4">
      <h1 class="text-4xl font-bold text-white">
        🎯 System Status
      </h1>
      <div class="flex items-center gap-4">
        <span class="text-gray-400">Last updated: {lastUpdated}</span>
        <Button.Root
          on:on:on:click={loadSystemStatus}
          disabled={isLoading}
          class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
        >
          {isLoading ? '🔄' : '🔃'} Refresh
        </Button.Root>
      </div>
    </div>
    
    {#if systemHealth?.overall}
      <div class="overall-status p-6 bg-gray-800 rounded-lg border border-gray-700 mb-6">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-2xl font-bold text-white mb-2">Overall System Health</h2>
            <p class="text-gray-300">
              {systemHealth.overall.healthyServices} of {systemHealth.overall.totalServices} services healthy
            </p>
          </div>
          <div class="text-right">
            <Badge variant={getBadgeVariant(systemHealth.overall.status)} class="text-lg px-4 py-2 mb-2">
              {systemHealth.overall.status.toUpperCase()}
            </Badge>
            <div class="text-2xl font-mono text-white">
              {systemHealth.overall.healthScore}%
            </div>
          </div>
        </div>
      </div>
    {/if}
  </header>

  <!-- Integration Tests Results -->
  <section class="mb-12">
    <h2 class="text-2xl font-bold text-white mb-6">Integration Test Results</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {#each Object.entries(integrationTests) as [testName, result]}
        <Card class="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle class="flex items-center justify-between text-white">
              <span class="capitalize">{testName.replace('-', ' ')}</span>
              <span class="text-xl">{getStatusIcon(result.status)}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p class="text-sm {getStatusColor(result.status)} mb-2">
              {result.message}
            </p>
            {#if result.details}
              <details class="text-xs text-gray-400">
                <summary class="cursor-pointer">Details</summary>
                <pre class="mt-2 p-2 bg-gray-900 rounded text-xs overflow-auto">
{JSON.stringify(result.details, null, 2)}
                </pre>
              </details>
            {/if}
          </CardContent>
        </Card>
      {/each}
    </div>
  </section>

  <!-- System Services Status -->
  {#if systemHealth?.services}
    <section class="mb-12">
      <h2 class="text-2xl font-bold text-white mb-6">System Services</h2>
      
      <!-- Databases -->
      <Card class="mb-6 bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle class="text-white">Database Services</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {#each Object.entries(systemHealth.services.databases) as [name, service]}
              <div class="service-card p-3 bg-gray-900 rounded border border-gray-600">
                <div class="flex items-center justify-between mb-1">
                  <h4 class="font-semibold text-white capitalize">{name}</h4>
                  <span class="text-sm">
                    {service.status === 'healthy' ? '🟢' : '🔴'}
                  </span>
                </div>
                <p class="text-xs text-gray-400">
                  {service.host}:{service.port}
                </p>
                <Badge variant={service.status === 'healthy' ? 'success' : 'destructive'} class="text-xs">
                  {service.status}
                </Badge>
              </div>
            {/each}
          </div>
        </CardContent>
      </Card>

      <!-- AI Services -->
      <Card class="mb-6 bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle class="text-white">AI/ML Services</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {#each Object.entries(systemHealth.services.aiServices) as [name, service]}
              <div class="service-card p-3 bg-gray-900 rounded border border-gray-600">
                <div class="flex items-center justify-between mb-1">
                  <h4 class="font-semibold text-white capitalize">{name.replace('Service', '')}</h4>
                  <span class="text-sm">
                    {service.status === 'healthy' ? '🟢' : '🔴'}
                  </span>
                </div>
                <p class="text-xs text-gray-400">
                  {service.host}:{service.port}
                </p>
                <Badge variant={service.status === 'healthy' ? 'success' : 'destructive'} class="text-xs">
                  {service.status}
                </Badge>
              </div>
            {/each}
          </div>
        </CardContent>
      </Card>

      <!-- GPU Services -->
      <Card class="mb-6 bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle class="text-white">GPU Acceleration Services</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each Object.entries(systemHealth.services.gpuServices) as [name, service]}
              <div class="service-card p-3 bg-gray-900 rounded border border-gray-600">
                <div class="flex items-center justify-between mb-1">
                  <h4 class="font-semibold text-white capitalize">{name}</h4>
                  <span class="text-sm">
                    {service.status === 'healthy' || service.status === 'ready' ? '🟢' : '🔴'}
                  </span>
                </div>
                {#if service.host}
                  <p class="text-xs text-gray-400">{service.host}:{service.port}</p>
                {/if}
                {#if service.vram}
                  <p class="text-xs text-gray-400">VRAM: {service.vram}</p>
                {/if}
                <Badge variant={service.status === 'healthy' || service.status === 'ready' ? 'success' : 'destructive'} class="text-xs">
                  {service.status}
                </Badge>
              </div>
            {/each}
          </div>
        </CardContent>
      </Card>
    </section>
  {/if}

  <!-- Performance Metrics -->
  {#if systemHealth?.performance}
    <section class="mb-12">
      <h2 class="text-2xl font-bold text-white mb-6">Performance Metrics</h2>
      <Card class="bg-gray-800 border-gray-700">
        <CardContent class="p-6">
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="metric-group">
              <h4 class="font-semibold text-white mb-2">System Uptime</h4>
              <p class="text-2xl font-mono text-green-400">
                {Math.floor(systemHealth.performance.systemUptime / 3600)}h {Math.floor((systemHealth.performance.systemUptime % 3600) / 60)}m
              </p>
            </div>
            <div class="metric-group">
              <h4 class="font-semibold text-white mb-2">Memory Usage</h4>
              <p class="text-2xl font-mono text-blue-400">
                {systemHealth.performance.memoryUsage.heapUsed} / {systemHealth.performance.memoryUsage.heapTotal} MB
              </p>
            </div>
            <div class="metric-group">
              <h4 class="font-semibold text-white mb-2">CPU Usage</h4>
              <p class="text-2xl font-mono text-yellow-400">
                {systemHealth.performance.cpuUsage.user}ms user / {systemHealth.performance.cpuUsage.system}ms system
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  {/if}

  <!-- Live Demo Section -->
  <section class="mb-12">
    <h2 class="text-2xl font-bold text-white mb-6">GPU Cache Integration Demo</h2>
    <Card class="bg-gray-800 border-gray-700">
      <CardContent class="p-6">
        <GPUCacheIntegrationDemo 
          showProgressionDemo={true}
          enableRealTimeMetrics={true}
          debugMode={false}
        />
      </CardContent>
    </Card>
  </section>

  <!-- Architecture Summary -->
  {#if systemHealth?.architecture}
    <section>
      <h2 class="text-2xl font-bold text-white mb-6">Platform Architecture</h2>
      <Card class="bg-gray-800 border-gray-700">
        <CardContent class="p-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 class="font-semibold text-white mb-4">Platform Information</h4>
              <div class="space-y-2 text-gray-300">
                <p><strong>Platform:</strong> {systemHealth.architecture.platform}</p>
                <p><strong>Version:</strong> {systemHealth.architecture.version}</p>
                <p><strong>GPU Architecture:</strong> {systemHealth.architecture.gpuArchitecture}</p>
                <p><strong>Microservices:</strong> {systemHealth.architecture.microservices}</p>
              </div>
            </div>
            <div>
              <h4 class="font-semibold text-white mb-4">Features & Protocols</h4>
              <div class="space-y-2">
                <div>
                  <h5 class="text-gray-400 mb-1">Protocols:</h5>
                  <div class="flex flex-wrap gap-2">
                    {#each systemHealth.architecture.protocols as protocol}
                      <Badge variant="outline" class="text-xs">{protocol}</Badge>
                    {/each}
                  </div>
                </div>
                <div>
                  <h5 class="text-gray-400 mb-1">Features:</h5>
                  <div class="flex flex-wrap gap-2">
                    {#each systemHealth.architecture.features as feature}
                      <Badge variant="secondary" class="text-xs">{feature}</Badge>
                    {/each}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  {/if}
</div>

<style>
  .status-page {
    font-family: 'Inter', sans-serif;
  }

  .service-card {
    transition: all 0.2s ease;
  }

  .service-card:hover {
    border-color: rgba(59, 130, 246, 0.5);
    transform: translateY(-1px);
  }

  .metric-group {
    padding: 1rem;
    background: rgba(31, 41, 55, 0.5);
    border-radius: 0.5rem;
    border: 1px solid rgba(75, 85, 99, 0.3);
  }

  /* Use GPU cache CSS variables */
  :global(.status-page) {
    background: var(--gpu-cache-bg-primary, #000000);
  }

  :global(.service-card) {
    background: var(--gpu-cache-bg-secondary, #111827);
    border-color: var(--gpu-cache-border-primary, #374151);
  }

  :global(.metric-group) {
    background: var(--gpu-cache-bg-tertiary, #1f2937);
    border-color: var(--gpu-cache-border-secondary, #4b5563);
  }
</style>