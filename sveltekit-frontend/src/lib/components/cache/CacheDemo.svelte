<!-- Multi-Layer Cache System Demo Component -->
<!-- Demonstrates Loki.js + Redis + PostgreSQL caching with real-time statistics -->

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { Button } from 'bits-ui';
  import { Input } from 'bits-ui';
  import { Card, CardContent, CardHeader, CardTitle } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Progress } from 'bits-ui';
  import { Tabs, TabsContent, TabsList, TabsTrigger } from 'bits-ui';
  import { 
    Database, 
    Zap, 
    BarChart3, 
    Trash2, 
    RefreshCw, 
    CheckCircle, 
    XCircle,
    Clock,
    HardDrive,
    Activity
  } from 'lucide-svelte';

  // State management
  const cacheStats = writable<any>(null);
  const healthStatus = writable<any>(null);
  const isLoading = writable(false);
  const testResults = writable<unknown[]>([]);

  // Form state
  let cacheKey = $state('');
  let cacheValue = $state('');
  let selectedTTL = $state('300'); // 5 minutes default
  let selectedPriority = $state('medium');
  let selectedTags = $state('');

  // Demo data
  const ttlOptions = [
    { value: '60', label: '1 minute' },
    { value: '300', label: '5 minutes' },
    { value: '900', label: '15 minutes' },
    { value: '3600', label: '1 hour' },
    { value: '86400', label: '24 hours' }
  ];

  const priorityOptions = [
    { value: 'low', label: 'Low Priority' },
    { value: 'medium', label: 'Medium Priority' },
    { value: 'high', label: 'High Priority' }
  ];

  // ============================================================================
  // CACHE OPERATIONS
  // ============================================================================

  async function loadCacheStats() {
    try {
      const response = await fetch('/api/cache?action=stats');
      const data = await response.json();
      if (data.success) {
        cacheStats.set(data.stats);
      }
    } catch (error) {
      console.error('Failed to load cache stats:', error);
    }
  }

  async function loadHealthStatus() {
    try {
      const response = await fetch('/api/cache?action=health');
      const data = await response.json();
      if (data.success) {
        healthStatus.set(data.health);
      }
    } catch (error) {
      console.error('Failed to load health status:', error);
    }
  }

  async function setCacheValue() {
    if (!cacheKey.trim() || !cacheValue.trim()) {
      addTestResult('error', 'Key and value are required');
      return;
    }

    isLoading.set(true);
    try {
      const tags = selectedTags.split(',').map(t => t.trim()).filter(t => t);
      
      const response = await fetch('/api/cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key: cacheKey,
          value: cacheValue,
          options: {
            ttl: parseInt(selectedTTL) * 1000, // Convert to milliseconds
            priority: selectedPriority,
            tags: tags.length > 0 ? tags : undefined
          }
        })
      });

      const data = await response.json();
      if (data.success) {
        addTestResult('success', `Cached "${cacheKey}" successfully`);
        cacheKey = '';
        cacheValue = '';
      } else {
        addTestResult('error', `Failed to cache: ${data.error}`);
      }
    } catch (error) {
      addTestResult('error', `Cache error: ${error}`);
    } finally {
      isLoading.set(false);
      await refreshStats();
    }
  }

  async function getCacheValue() {
    if (!cacheKey.trim()) {
      addTestResult('error', 'Key is required');
      return;
    }

    isLoading.set(true);
    try {
      const response = await fetch(`/api/cache?action=get&key=${encodeURIComponent(cacheKey)}`);
      const data = await response.json();
      
      if (data.success) {
        if (data.cached) {
          addTestResult('success', `Retrieved "${cacheKey}": ${JSON.stringify(data.value)}`);
        } else {
          addTestResult('warning', `Key "${cacheKey}" not found in cache`);
        }
      } else {
        addTestResult('error', `Failed to retrieve: ${data.error}`);
      }
    } catch (error) {
      addTestResult('error', `Retrieval error: ${error}`);
    } finally {
      isLoading.set(false);
      await refreshStats();
    }
  }

  async function deleteCacheValue() {
    if (!cacheKey.trim()) {
      addTestResult('error', 'Key is required');
      return;
    }

    isLoading.set(true);
    try {
      const response = await fetch(`/api/cache?key=${encodeURIComponent(cacheKey)}`, {
        method: 'DELETE'
      });

      const data = await response.json();
      if (data.success) {
        addTestResult('success', `Deleted "${cacheKey}" successfully`);
        cacheKey = '';
      } else {
        addTestResult('error', `Failed to delete: ${data.error}`);
      }
    } catch (error) {
      addTestResult('error', `Delete error: ${error}`);
    } finally {
      isLoading.set(false);
      await refreshStats();
    }
  }

  async function clearCache() {
    if (!confirm('Are you sure you want to clear all cache?')) return;

    isLoading.set(true);
    try {
      const response = await fetch('/api/cache?action=clear', {
        method: 'DELETE'
      });

      const data = await response.json();
      if (data.success) {
        addTestResult('success', 'Cache cleared successfully');
      } else {
        addTestResult('error', `Failed to clear cache: ${data.error}`);
      }
    } catch (error) {
      addTestResult('error', `Clear error: ${error}`);
    } finally {
      isLoading.set(false);
      await refreshStats();
    }
  }

  // ============================================================================
  // DEMO OPERATIONS
  // ============================================================================

  async function runPerformanceTest() {
    isLoading.set(true);
    addTestResult('info', 'Starting performance test...');

    try {
      const testData = [];
      const testSize = 100;

      // Generate test data
      for (let i = 0; i < testSize; i++) {
        testData.push({
          operation: 'set',
          key: `perf_test_${i}`,
          value: `Test data ${i} - ${Math.random().toString(36)}`,
          options: {
            ttl: 300000, // 5 minutes
            priority: i % 3 === 0 ? 'high' : 'medium',
            tags: [`test`, `batch-${Math.floor(i / 10)}`]
          }
        });
      }

      const startTime = Date.now();

      // Execute batch operation
      const response = await fetch('/api/cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operations: testData })
      });

      const result = await response.json();
      const endTime = Date.now();
      const duration = endTime - startTime;

      if (result.success) {
        addTestResult('success', 
          `Performance test completed: ${result.summary.successful}/${testSize} operations in ${duration}ms`
        );
        addTestResult('info', 
          `Average: ${(duration / testSize).toFixed(2)}ms per operation`
        );
      } else {
        addTestResult('error', 'Performance test failed');
      }
    } catch (error) {
      addTestResult('error', `Performance test error: ${error}`);
    } finally {
      isLoading.set(false);
      await refreshStats();
    }
  }

  async function testCacheHitMiss() {
    isLoading.set(true);
    addTestResult('info', 'Testing cache hit/miss patterns...');

    try {
      const testKey = `hit_miss_test_${Date.now()}`;
      
      // Test cache miss
      let response = await fetch(`/api/cache?action=get&key=${testKey}`);
      let data = await response.json();
      
      if (!data.cached) {
        addTestResult('success', '✓ Cache miss test passed');
      } else {
        addTestResult('warning', '⚠ Unexpected cache hit');
      }

      // Set value
      response = await fetch('/api/cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key: testKey,
          value: 'test_data_for_hit_test',
          options: { ttl: 60000 }
        })
      });

      // Test cache hit
      response = await fetch(`/api/cache?action=get&key=${testKey}`);
      data = await response.json();
      
      if (data.cached && data.value === 'test_data_for_hit_test') {
        addTestResult('success', '✓ Cache hit test passed');
      } else {
        addTestResult('error', '✗ Cache hit test failed');
      }

      // Clean up
      await fetch(`/api/cache?key=${testKey}`, { method: 'DELETE' });
      
    } catch (error) {
      addTestResult('error', `Hit/miss test error: ${error}`);
    } finally {
      isLoading.set(false);
      await refreshStats();
    }
  }

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  function addTestResult(type: 'success' | 'error' | 'warning' | 'info', message: string) {
    testResults.update(results => [
      {
        type,
        message,
        timestamp: new Date().toLocaleTimeString()
      },
      ...results.slice(0, 49) // Keep last 50 results
    ]);
  }

  async function refreshStats() {
    await Promise.all([
      loadCacheStats(),
      loadHealthStatus()
    ]);
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatPercentage(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(() => {
    refreshStats();
    
    // Auto-refresh stats every 10 seconds
    const interval = setInterval(refreshStats, 10000);
    
    return () => clearInterval(interval);
  });
</script>

<!-- Main Demo Interface -->
<div class="cache-demo space-y-6">
  <!-- Header -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Database size={24} />
        Multi-Layer Cache System Demo
        <Badge variant="secondary">Loki.js + Redis + PostgreSQL</Badge>
      </CardTitle>
    </CardHeader>
    <CardContent>
      <p class="text-muted-foreground">
        Interactive demonstration of the comprehensive caching architecture with real-time statistics and performance testing.
      </p>
    </CardContent>
  </Card>

  <!-- Main Content Tabs -->
  <Tabs value="operations" class="w-full">
    <TabsList class="grid w-full grid-cols-4">
      <TabsTrigger value="operations">Cache Operations</TabsTrigger>
      <TabsTrigger value="statistics">Statistics</TabsTrigger>
      <TabsTrigger value="health">Health Monitor</TabsTrigger>
      <TabsTrigger value="testing">Performance Tests</TabsTrigger>
    </TabsList>

    <!-- Cache Operations Tab -->
    <TabsContent value="operations" class="space-y-4">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Cache Operations Form -->
        <Card>
          <CardHeader>
            <CardTitle>Cache Operations</CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Cache Key</label>
              <Input
                bind:value={cacheKey}
                placeholder="Enter cache key"
                class="w-full"
              />
            </div>

            <div>
              <label class="block text-sm font-medium mb-2">Cache Value</label>
              <Input
                bind:value={cacheValue}
                placeholder="Enter cache value"
                class="w-full"
              />
            </div>

            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium mb-2">TTL</label>
                <select bind:value={selectedTTL} class="w-full p-2 border rounded">
                  {#each ttlOptions as option}
                    <option value={option.value}>{option.label}</option>
                  {/each}
                </select>
              </div>

              <div>
                <label class="block text-sm font-medium mb-2">Priority</label>
                <select bind:value={selectedPriority} class="w-full p-2 border rounded">
                  {#each priorityOptions as option}
                    <option value={option.value}>{option.label}</option>
                  {/each}
                </select>
              </div>
            </div>

            <div>
              <label class="block text-sm font-medium mb-2">Tags (comma-separated)</label>
              <Input
                bind:value={selectedTags}
                placeholder="tag1, tag2, tag3"
                class="w-full"
              />
            </div>

            <div class="flex flex-wrap gap-2">
              <Button on:on:click={setCacheValue} disabled={$isLoading}>
                <Database class="mr-2" size={16} />
                Set Value
              </Button>
              
              <Button variant="outline" on:on:click={getCacheValue} disabled={$isLoading}>
                <RefreshCw class="mr-2" size={16} />
                Get Value
              </Button>
              
              <Button variant="destructive" on:on:click={deleteCacheValue} disabled={$isLoading}>
                <Trash2 class="mr-2" size={16} />
                Delete
              </Button>
              
              <Button variant="destructive" on:on:click={clearCache} disabled={$isLoading}>
                <XCircle class="mr-2" size={16} />
                Clear All
              </Button>
            </div>
          </CardContent>
        </Card>

        <!-- Test Results -->
        <Card>
          <CardHeader>
            <CardTitle class="flex items-center justify-between">
              Test Results
              <Button 
                variant="ghost" 
                size="sm" 
                on:on:click={() => testResults.set([])}
              >
                Clear
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-2 max-h-80 overflow-y-auto">
              {#each $testResults as result}
                <div class="flex items-start gap-2 p-2 rounded border-l-4 
                           {result.type === 'success' ? 'border-green-500 bg-green-50' : 
                            result.type === 'error' ? 'border-red-500 bg-red-50' : 
                            result.type === 'warning' ? 'border-yellow-500 bg-yellow-50' : 
                            'border-blue-500 bg-blue-50'}">
                  <div class="text-xs text-muted-foreground min-w-fit">
                    {result.timestamp}
                  </div>
                  <div class="text-sm flex-1">
                    {result.message}
                  </div>
                </div>
              {/each}
              
              {#if $testResults.length === 0}
                <div class="text-center text-muted-foreground py-8">
                  No test results yet. Try some cache operations!
                </div>
              {/if}
            </div>
          </CardContent>
        </Card>
      </div>
    </TabsContent>

    <!-- Statistics Tab -->
    <TabsContent value="statistics" class="space-y-4">
      {#if $cacheStats}
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <!-- Service Stats -->
          <Card>
            <CardHeader class="pb-2">
              <CardTitle class="text-sm">Service Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span class="text-sm">Requests:</span>
                  <span class="font-mono">{$cacheStats.service.requests}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-sm">Hits:</span>
                  <span class="font-mono text-green-600">{$cacheStats.service.hits}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-sm">Misses:</span>
                  <span class="font-mono text-red-600">{$cacheStats.service.misses}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-sm">Hit Rate:</span>
                  <span class="font-mono">{formatPercentage($cacheStats.service.hitRate)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <!-- Overall Performance -->
          {#if $cacheStats.layers}
            <Card>
              <CardHeader class="pb-2">
                <CardTitle class="text-sm">Overall Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div class="space-y-2">
                  <div class="flex justify-between">
                    <span class="text-sm">Total Requests:</span>
                    <span class="font-mono">{$cacheStats.layers.overall.totalRequests}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-sm">Hit Rate:</span>
                    <span class="font-mono">{formatPercentage($cacheStats.layers.overall.totalHitRate)}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-sm">Avg Response:</span>
                    <span class="font-mono">{$cacheStats.layers.overall.avgResponseTime.toFixed(2)}ms</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-sm">Healthy Layers:</span>
                    <span class="font-mono">{$cacheStats.layers.overall.healthyLayers}/{$cacheStats.layers.overall.totalLayers}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <!-- Layer Statistics -->
            {#each $cacheStats.layers.layers as layer}
              <Card>
                <CardHeader class="pb-2">
                  <CardTitle class="text-sm capitalize">{layer.layer} Layer</CardTitle>
                </CardHeader>
                <CardContent>
                  <div class="space-y-2">
                    <div class="flex justify-between">
                      <span class="text-sm">Items:</span>
                      <span class="font-mono">{layer.itemCount.toLocaleString()}</span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-sm">Hit Rate:</span>
                      <span class="font-mono">{formatPercentage(layer.hitRate)}</span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-sm">Memory:</span>
                      <span class="font-mono text-xs">{formatBytes(layer.memoryUsage)}</span>
                    </div>
                    <Progress value={layer.hitRate * 100} class="h-2" />
                  </div>
                </CardContent>
              </Card>
            {/each}
          {/if}
        </div>
      {:else}
        <Card>
          <CardContent class="py-8">
            <div class="text-center text-muted-foreground">
              <Activity class="mx-auto mb-2" size={48} />
              Loading cache statistics...
            </div>
          </CardContent>
        </Card>
      {/if}
    </TabsContent>

    <!-- Health Monitor Tab -->
    <TabsContent value="health" class="space-y-4">
      {#if $healthStatus}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Overall Health -->
          <Card>
            <CardHeader>
              <CardTitle class="flex items-center gap-2">
                {#if $healthStatus.healthy}
                  <CheckCircle class="text-green-500" size={20} />
                  System Healthy
                {:else}
                  <XCircle class="text-red-500" size={20} />
                  System Issues Detected
                {/if}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span>Service Health:</span>
                  <Badge variant={$healthStatus.service ? 'default' : 'destructive'}>
                    {$healthStatus.service ? 'Healthy' : 'Unhealthy'}
                  </Badge>
                </div>
                
                {#if $healthStatus.layers}
                  <h4 class="font-medium">Cache Layers:</h4>
                  {#each Object.entries($healthStatus.layers.layers) as [layerName, isHealthy]}
                    <div class="flex items-center justify-between">
                      <span class="capitalize">{layerName}:</span>
                      <Badge variant={isHealthy ? 'default' : 'destructive'}>
                        {isHealthy ? 'Healthy' : 'Unhealthy'}
                      </Badge>
                    </div>
                  {/each}
                {/if}
              </div>
            </CardContent>
          </Card>

          <!-- Issues & Recommendations -->
          <Card>
            <CardHeader>
              <CardTitle>Issues & Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              {#if $healthStatus.layers?.issues && $healthStatus.layers.issues.length > 0}
                <div class="space-y-2">
                  {#each $healthStatus.layers.issues as issue}
                    <div class="p-2 bg-red-50 border border-red-200 rounded text-sm">
                      ⚠ {issue}
                    </div>
                  {/each}
                </div>
              {:else}
                <div class="text-center text-green-600 py-4">
                  <CheckCircle class="mx-auto mb-2" size={32} />
                  All systems operating normally
                </div>
              {/if}
            </CardContent>
          </Card>
        </div>
      {:else}
        <Card>
          <CardContent class="py-8">
            <div class="text-center text-muted-foreground">
              <HardDrive class="mx-auto mb-2" size={48} />
              Loading health status...
            </div>
          </CardContent>
        </Card>
      {/if}
    </TabsContent>

    <!-- Performance Tests Tab -->
    <TabsContent value="testing" class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Test Controls -->
        <Card>
          <CardHeader>
            <CardTitle>Performance Tests</CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <div class="space-y-2">
              <Button 
                on:on:click={runPerformanceTest} 
                disabled={$isLoading}
                class="w-full"
              >
                <Zap class="mr-2" size={16} />
                Run Batch Performance Test
              </Button>
              <p class="text-sm text-muted-foreground">
                Tests batch operations with 100 cache entries
              </p>
            </div>

            <div class="space-y-2">
              <Button 
                variant="outline"
                on:on:click={testCacheHitMiss} 
                disabled={$isLoading}
                class="w-full"
              >
                <BarChart3 class="mr-2" size={16} />
                Test Cache Hit/Miss
              </Button>
              <p class="text-sm text-muted-foreground">
                Validates cache hit and miss behavior
              </p>
            </div>

            <div class="space-y-2">
              <Button 
                variant="outline"
                on:on:click={refreshStats} 
                disabled={$isLoading}
                class="w-full"
              >
                <RefreshCw class="mr-2" size={16} />
                Refresh Statistics
              </Button>
              <p class="text-sm text-muted-foreground">
                Updates all cache statistics and health status
              </p>
            </div>
          </CardContent>
        </Card>

        <!-- Test Results Display -->
        <Card>
          <CardHeader>
            <CardTitle>Test Information</CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <div class="p-4 bg-blue-50 rounded">
              <h4 class="font-medium text-blue-900 mb-2">Cache Layer Architecture</h4>
              <ul class="text-sm text-blue-800 space-y-1">
                <li><strong>L1:</strong> Memory Cache (fastest, volatile)</li>
                <li><strong>L2:</strong> Loki.js (persistent, document-based)</li>
                <li><strong>L3:</strong> Redis (distributed, high-speed)</li>
                <li><strong>L4:</strong> PostgreSQL (persistent, SQL-based)</li>
              </ul>
            </div>

            <div class="p-4 bg-green-50 rounded">
              <h4 class="font-medium text-green-900 mb-2">Features Demonstrated</h4>
              <ul class="text-sm text-green-800 space-y-1">
                <li>✓ Multi-layer cache retrieval</li>
                <li>✓ Write-through caching</li>
                <li>✓ Automatic failover</li>
                <li>✓ Performance monitoring</li>
                <li>✓ Health checking</li>
                <li>✓ Batch operations</li>
              </ul>
            </div>

            {#if $isLoading}
              <div class="text-center py-4">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                <p class="text-sm text-muted-foreground mt-2">Running tests...</p>
              </div>
            {/if}
          </CardContent>
        </Card>
      </div>
    </TabsContent>
  </Tabs>
</div>

<style>
  .cache-demo {
    @apply max-w-7xl mx-auto p-4;
  }
</style>

