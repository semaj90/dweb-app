<!-- Unified GPU/WASM Integration Demo Component -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Button from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';

  // System status and results
  const systemHealth = writable(null);
  const activeOperations = writable([]);
  const results = writable([]);
  const metrics = writable([]);
let isLoading = $state(false);
let selectedOperation = $state('processDocument');
let testInput = $state('');
let errorMessage = $state('');

  // Demo data for different operations
  const demoInputs = {
    processDocument: `LEGAL CONTRACT AGREEMENT

This Service Agreement is entered into between Company A and Company B.

TERMS AND CONDITIONS:
1. Service Period: 12 months from execution
2. Payment Terms: Net 30 days
3. Deliverables: As specified in Schedule A
4. Termination: Either party may terminate with 60 days notice

Both parties acknowledge they have read and agree to these terms.`,
    
    performInference: JSON.stringify([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    
    processCanvas: JSON.stringify({
      width: 64,
      height: 64,
      data: Array(64 * 64 * 4).fill(0).map((_, i) => {
        const pixel = Math.floor(i / 4);
        const component = i % 4;
        if (component === 3) return 255; // Alpha
        return (pixel % 256); // RGB pattern
      }),
      format: 'RGBA'
    }),
    
    matmul: JSON.stringify({
      a: [1, 2, 3, 4, 5, 6],
      b: [7, 8, 9, 10, 11, 12],
      m: 2,
      n: 3,
      k: 3
    }),
    
    attention: JSON.stringify({
      query: Array(64).fill(0).map(() => Math.random()),
      key: Array(64).fill(0).map(() => Math.random()),
      value: Array(64).fill(0).map(() => Math.random()),
      seq_len: 8,
      dim: 8
    })
  };

  onMount(() => {
    updateSystemHealth();
    updateMetrics();
    testInput = demoInputs[selectedOperation];
    
    // Auto-refresh system status
    const interval = setInterval(() => {
      updateSystemHealth();
      updateMetrics();
    }, 5000);
    
    return () => clearInterval(interval);
  });

  async function updateSystemHealth() {
    try {
      const response = await fetch('/api/v1/orchestrator?endpoint=health');
      const data = await response.json();
      if (data.success) {
        systemHealth.set(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch system health:', error);
    }
  }

  async function updateMetrics() {
    try {
      const response = await fetch('/api/v1/orchestrator?endpoint=metrics');
      const data = await response.json();
      if (data.success) {
        metrics.set(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  }

  async function executeOperation() {
    if (!testInput.trim()) return;
    
    isLoading = true;
    errorMessage = '';
    
    try {
      let requestData;
      
      switch (selectedOperation) {
        case 'processDocument':
          requestData = {
            operation: 'processDocument',
            data: {
              document: testInput,
              analysisType: 'comprehensive'
            },
            options: {
              priority: 'HIGH',
              maxTokens: 1024
            }
          };
          break;
          
        case 'performInference':
          requestData = {
            operation: 'performInference',
            data: {
              input: JSON.parse(testInput)
            },
            options: {
              priority: 'HIGH',
              modelType: 'transformer'
            }
          };
          break;
          
        case 'processCanvas':
          const canvasData = JSON.parse(testInput);
          requestData = {
            operation: 'processCanvas',
            data: {
              canvasState: canvasData
            },
            options: {
              priority: 'NORMAL',
              targetBitDepth: 24
            }
          };
          break;
          
        case 'matmul':
          const matrixData = JSON.parse(testInput);
          requestData = {
            operation: 'matmul',
            data: matrixData,
            options: {
              priority: 'HIGH'
            }
          };
          break;
          
        case 'attention':
          const attentionData = JSON.parse(testInput);
          requestData = {
            operation: 'attention',
            data: attentionData,
            options: {
              priority: 'HIGH'
            }
          };
          break;
      }
      
      const response = await fetch('/api/v1/orchestrator', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });
      
      const result = await response.json();
      
      if (result.success) {
        results.update(prev => [
          {
            id: Date.now(),
            operation: selectedOperation,
            timestamp: new Date(),
            data: result.data,
            metadata: result.metadata,
            processingTime: result.totalProcessingTime
          },
          ...prev.slice(0, 9) // Keep last 10 results
        ]);
      } else {
        errorMessage = result.error || 'Operation failed';
      }
      
    } catch (error) {
      errorMessage = `Error: ${error.message}`;
      console.error('Operation failed:', error);
    } finally {
      isLoading = false;
      updateSystemHealth();
      updateMetrics();
    }
  }

  function onOperationChange() {
    testInput = demoInputs[selectedOperation];
    errorMessage = '';
  }

  function getHealthColor(status) {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  }

  function getServiceColor(status) {
    switch (status) {
      case 'online': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'offline': return 'text-red-600';
      default: return 'text-gray-600';
    }
  }
</script>

<div class="unified-integration-demo p-6 max-w-7xl mx-auto">
  <div class="mb-8">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">
      Unified GPU/WASM Integration System
    </h1>
    <p class="text-gray-600">
      Complete integration of WASM modules, GPU acceleration, QUIC services, and neural processing
    </p>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
    <!-- System Health Card -->
    <Card class="lg:col-span-1">
      <CardHeader>
        <CardTitle>System Health</CardTitle>
      </CardHeader>
      <CardContent>
        {#if $systemHealth}
          <div class="space-y-3">
            <div class="flex justify-between items-center">
              <span class="text-sm font-medium">Overall Status:</span>
              <span class="text-sm font-semibold {getHealthColor($systemHealth.overall)}">
                {$systemHealth.overall.toUpperCase()}
              </span>
            </div>
            
            <div class="border-t pt-3">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Services</h4>
              <div class="space-y-1">
                {#each Object.entries($systemHealth.services) as [service, status]}
                  <div class="flex justify-between items-center text-xs">
                    <span class="capitalize">{service}:</span>
                    <span class="{getServiceColor(status)}">{status}</span>
                  </div>
                {/each}
              </div>
            </div>
            
            <div class="border-t pt-3">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Performance</h4>
              <div class="space-y-1 text-xs">
                <div class="flex justify-between">
                  <span>Avg Latency:</span>
                  <span>{$systemHealth.performance.averageLatency.toFixed(0)}ms</span>
                </div>
                <div class="flex justify-between">
                  <span>Throughput:</span>
                  <span>{$systemHealth.performance.throughput.toFixed(1)}/s</span>
                </div>
              </div>
            </div>
          </div>
        {:else}
          <div class="text-center text-gray-500">Loading system health...</div>
        {/if}
      </CardContent>
    </Card>

    <!-- Operation Controls Card -->
    <Card class="lg:col-span-2">
      <CardHeader>
        <CardTitle>Execute Operations</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-4">
          <!-- Operation Selection -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Operation Type
            </label>
            <select 
              bind:value={selectedOperation}
              change={onOperationChange}
              class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="processDocument">Legal Document Processing</option>
              <option value="performInference">Neural Inference</option>
              <option value="processCanvas">Canvas State Processing</option>
              <option value="matmul">Matrix Multiplication</option>
              <option value="attention">Attention Mechanism</option>
            </select>
          </div>

          <!-- Input Data -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Input Data
            </label>
            <textarea 
              bind:value={testInput}
              rows="8"
              class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
              placeholder="Enter test data..."
            />
          </div>

          <!-- Error Display -->
          {#if errorMessage}
            <div class="bg-red-50 border border-red-200 rounded-md p-3">
              <p class="text-red-600 text-sm">{errorMessage}</p>
            </div>
          {/if}

          <!-- Execute Button -->
          <Button 
            on:on:click={executeOperation}
            disabled={isLoading || !testInput.trim()}
            class="w-full"
          >
            {#if isLoading}
              <span class="inline-flex items-center">
                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </span>
            {:else}
              Execute {selectedOperation}
            {/if}
          </Button>
        </div>
      </CardContent>
    </Card>
  </div>

  <!-- Results and Metrics -->
  <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
    <!-- Results Card -->
    <Card>
      <CardHeader>
        <CardTitle>Operation Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-4 max-h-96 overflow-y-auto">
          {#if $results.length > 0}
            {#each $results as result (result.id)}
              <div class="border border-gray-200 rounded-lg p-4">
                <div class="flex justify-between items-start mb-2">
                  <div>
                    <h4 class="font-medium text-gray-900 capitalize">
                      {result.operation.replace(/([A-Z])/g, ' $1').trim()}
                    </h4>
                    <p class="text-xs text-gray-500">
                      {result.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                  <div class="text-right">
                    <p class="text-xs text-gray-600">
                      {result.processingTime}ms
                    </p>
                    {#if result.metadata?.servicesUsed}
                      <p class="text-xs text-blue-600">
                        {result.metadata.servicesUsed.join(', ')}
                      </p>
                    {/if}
                  </div>
                </div>
                
                <div class="bg-gray-50 rounded p-2 text-xs font-mono">
                  {#if result.data?.success !== undefined}
                    <p class="text-{result.data.success ? 'green' : 'red'}-600 mb-1">
                      Status: {result.data.success ? 'Success' : 'Failed'}
                    </p>
                  {/if}
                  
                  {#if result.metadata?.performance}
                    <p>Latency: {result.metadata.performance.latency}ms</p>
                    <p>Throughput: {result.metadata.performance.throughput.toFixed(2)}/s</p>
                    <p>Resource Usage: {result.metadata.performance.resourceUsage.toFixed(2)}</p>
                  {/if}
                  
                  {#if result.metadata?.fallbacksTriggered?.length > 0}
                    <p class="text-yellow-600">
                      Fallbacks: {result.metadata.fallbacksTriggered.join(' → ')}
                    </p>
                  {/if}
                </div>
              </div>
            {/each}
          {:else}
            <div class="text-center text-gray-500 py-8">
              No results yet. Execute an operation to see results here.
            </div>
          {/if}
        </div>
      </CardContent>
    </Card>

    <!-- Performance Metrics Card -->
    <Card>
      <CardHeader>
        <CardTitle>Performance Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        {#if $metrics?.metrics?.length > 0}
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <div class="bg-blue-50 rounded-lg p-3">
                <h4 class="text-sm font-medium text-blue-800">Total Operations</h4>
                <p class="text-2xl font-bold text-blue-600">{$metrics.count}</p>
              </div>
              
              {#if $metrics.latestMetric}
                <div class="bg-green-50 rounded-lg p-3">
                  <h4 class="text-sm font-medium text-green-800">Latest Latency</h4>
                  <p class="text-2xl font-bold text-green-600">
                    {$metrics.latestMetric.latency}ms
                  </p>
                </div>
              {/if}
            </div>
            
            <!-- Recent Metrics Chart (simplified) -->
            <div>
              <h4 class="text-sm font-medium text-gray-700 mb-2">Recent Performance</h4>
              <div class="space-y-1">
                {#each $metrics.metrics.slice(-10) as metric, i}
                  <div class="flex items-center space-x-2 text-xs">
                    <span class="w-16 text-gray-500">
                      {metric.timestamp ? new Date(metric.timestamp).toLocaleTimeString() : 'N/A'}
                    </span>
                    <div class="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        class="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style="width: {Math.min(100, (metric.latency / 1000) * 100)}%"
                      ></div>
                    </div>
                    <span class="w-12 text-right">{metric.latency}ms</span>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        {:else}
          <div class="text-center text-gray-500 py-8">
            No metrics available yet. Execute some operations to see performance data.
          </div>
        {/if}
      </CardContent>
    </Card>
  </div>
</div>

<style>
  .unified-integration-demo {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
    min-height: 100vh;
    background: #f8fafc;
  }
</style>