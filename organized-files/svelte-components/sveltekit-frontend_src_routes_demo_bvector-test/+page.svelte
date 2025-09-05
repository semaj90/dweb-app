<!-- BVector Store Integration Test Interface -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { BVectorIntegrationTestSuite, runBVectorIntegrationTest } from '$lib/tests/bvector-integration-test';
  import type { TestResult, TestConfig } from '$lib/tests/bvector-integration-test';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/Card';
  import { Badge } from '$lib/components/ui/badge';
  import { Progress } from '$lib/components/ui/progress';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import {
    Play,
    Square,
    CheckCircle,
    XCircle,
    Clock,
    Cpu,
    Database,
    Zap,
    Brain,
    Settings,
    RefreshCw,
    Download,
    AlertTriangle
  } from 'lucide-svelte';

  // Test state
  let isRunning = $state(false);
  let currentTest = $state('');
  let progress = $state(0);
  let results = $state<TestResult[]>([]);
  let summary = $state<any>(null);
  let testLogs = $state<string[]>([]);

  // Test configuration
  let testConfig = $state<TestConfig>({
    skipGpuTests: false,
    skipGoBindings: false,
    mockWorkers: false,
    verbose: true
  });

  // Real-time test status
  let embeddingWorkerStatus = $state<'unknown' | 'available' | 'unavailable'>('unknown');
  let gpuStatus = $state<'unknown' | 'available' | 'unavailable'>('unknown');
  let goBinaryStatus = $state<Record<string, boolean>>({});
  let systemResources = $state<any>({});

  onMount(async () => {
    await checkSystemStatus();
  });

  async function checkSystemStatus() {
    try {
      // Check embedding worker availability
      const workerTest = new Worker('/static/workers/embedding-worker.js');
      workerTest.terminate();
      embeddingWorkerStatus = 'available';
    } catch {
      embeddingWorkerStatus = 'unavailable';
    }

    // Check GPU availability
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      if (gl && gl.getExtension('OES_texture_float')) {
        gpuStatus = 'available';
      } else {
        gpuStatus = 'unavailable';
      }
    } catch {
      gpuStatus = 'unavailable';
    }

    // Get system resource info
    if ('navigator' in globalThis && 'hardwareConcurrency' in navigator) {
      systemResources.cpuCores = navigator.hardwareConcurrency;
    }
    
    if ('navigator' in globalThis && 'deviceMemory' in navigator) {
      systemResources.memoryGB = (navigator as any).deviceMemory;
    }
  }

  async function runTests() {
    if (isRunning) return;
    
    isRunning = true;
    currentTest = 'Initializing...';
    progress = 0;
    results = [];
    summary = null;
    testLogs = [];

    try {
      addLog('🚀 Starting BVector Store Integration Tests');
      addLog(`Configuration: GPU=${!testConfig.skipGpuTests}, Go=${!testConfig.skipGoBindings}, Workers=${!testConfig.mockWorkers}`);
      
      const testSuite = new BVectorIntegrationTestSuite(testConfig);
      
      // Mock progress updates during test execution
      const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 10, 95);
      }, 500);

      const testResults = await testSuite.runFullIntegrationTest();
      
      clearInterval(progressInterval);
      progress = 100;
      
      results = testResults.results;
      summary = testResults.summary;
      
      addLog(`✅ Tests completed: ${summary.passed}/${summary.totalTests} passed in ${summary.totalDuration}ms`);
      
      if (summary.failed > 0) {
        addLog(`❌ ${summary.failed} tests failed - check detailed results below`);
      }

    } catch (error) {
      addLog(`❌ Test execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      isRunning = false;
      currentTest = '';
    }
  }

  function addLog(message: string) {
    testLogs = [...testLogs, `[${new Date().toLocaleTimeString()}] ${message}`];
  }

  function stopTests() {
    if (isRunning) {
      addLog('⏹️ Test execution stopped by user');
      isRunning = false;
    }
  }

  function exportResults() {
    const exportData = {
      timestamp: new Date().toISOString(),
      configuration: testConfig,
      systemInfo: {
        embeddingWorkerStatus,
        gpuStatus,
        goBinaryStatus,
        systemResources
      },
      summary,
      results: results.map(result => ({
        ...result,
        details: result.details ? Object.keys(result.details).reduce((acc, key) => {
          acc[key] = typeof result.details![key] === 'object' 
            ? JSON.stringify(result.details![key]) 
            : result.details![key];
          return acc;
        }, {} as Record<string, any>) : undefined
      })),
      logs: testLogs
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bvector-test-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function getTestIcon(result: TestResult) {
    return result.success ? CheckCircle : XCircle;
  }

  function getTestBadgeVariant(result: TestResult): 'default' | 'secondary' | 'destructive' {
    return result.success ? 'default' : 'destructive';
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'available': return CheckCircle;
      case 'unavailable': return XCircle;
      default: return Clock;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'available': return 'text-green-600';
      case 'unavailable': return 'text-red-600';
      default: return 'text-yellow-600';
    }
  }

  // Reactive computations
  const canRunTests = $derived(!isRunning);
  const hasResults = $derived(results.length > 0);
  const successRate = $derived(summary ? Math.round((summary.passed / summary.totalTests) * 100) : 0);
</script>

<div class="max-w-7xl mx-auto p-6 space-y-6">
  <!-- Page Header -->
  <div class="flex items-center justify-between">
    <div>
      <h1 class="text-3xl font-bold text-gray-900">BVector Store Integration Tests</h1>
      <p class="text-gray-600 mt-2">
        Comprehensive testing of embedding workers, Go binaries, GPU cache, and reinforcement learning
      </p>
    </div>
    
    <div class="flex items-center gap-3">
      <Button
        on:click={checkSystemStatus}
        variant="outline"
        size="sm"
        disabled={isRunning}
      >
        <RefreshCw class="w-4 h-4 mr-2" />
        Refresh Status
      </Button>
      
      {#if hasResults}
        <Button
          on:click={exportResults}
          variant="outline"
          size="sm"
        >
          <Download class="w-4 h-4 mr-2" />
          Export Results
        </Button>
      {/if}
    </div>
  </div>

  <!-- System Status Cards -->
  <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
    <Card>
      <CardHeader class="pb-3">
        <CardTitle class="flex items-center gap-2 text-sm">
          <Brain class="w-4 h-4" />
          Embedding Worker
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="flex items-center gap-2">
          <svelte:component 
            this={getStatusIcon(embeddingWorkerStatus)} 
            class="w-5 h-5 {getStatusColor(embeddingWorkerStatus)}"
          />
          <span class="text-sm font-medium capitalize">{embeddingWorkerStatus}</span>
        </div>
      </CardContent>
    </Card>

    <Card>
      <CardHeader class="pb-3">
        <CardTitle class="flex items-center gap-2 text-sm">
          <Zap class="w-4 h-4" />
          GPU Acceleration
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="flex items-center gap-2">
          <svelte:component 
            this={getStatusIcon(gpuStatus)} 
            class="w-5 h-5 {getStatusColor(gpuStatus)}"
          />
          <span class="text-sm font-medium capitalize">{gpuStatus}</span>
        </div>
      </CardContent>
    </Card>

    <Card>
      <CardHeader class="pb-3">
        <CardTitle class="flex items-center gap-2 text-sm">
          <Database class="w-4 h-4" />
          Go Binaries
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="text-sm">
          {Object.keys(goBinaryStatus).length > 0 
            ? `${Object.values(goBinaryStatus).filter(Boolean).length}/${Object.keys(goBinaryStatus).length} Available`
            : 'Not Checked'
          }
        </div>
      </CardContent>
    </Card>

    <Card>
      <CardHeader class="pb-3">
        <CardTitle class="flex items-center gap-2 text-sm">
          <Cpu class="w-4 h-4" />
          System Resources
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="text-sm space-y-1">
          {#if systemResources.cpuCores}
            <div>CPU: {systemResources.cpuCores} cores</div>
          {/if}
          {#if systemResources.memoryGB}
            <div>RAM: {systemResources.memoryGB}GB</div>
          {/if}
        </div>
      </CardContent>
    </Card>
  </div>

  <!-- Test Configuration -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Settings class="w-5 h-5" />
        Test Configuration
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <label class="flex items-center space-x-2">
          <input 
            type="checkbox" 
            bind:checked={testConfig.skipGpuTests}
            disabled={isRunning}
            class="rounded"
          >
          <span class="text-sm">Skip GPU Tests</span>
        </label>

        <label class="flex items-center space-x-2">
          <input 
            type="checkbox" 
            bind:checked={testConfig.skipGoBindings}
            disabled={isRunning}
            class="rounded"
          >
          <span class="text-sm">Skip Go Binaries</span>
        </label>

        <label class="flex items-center space-x-2">
          <input 
            type="checkbox" 
            bind:checked={testConfig.mockWorkers}
            disabled={isRunning}
            class="rounded"
          >
          <span class="text-sm">Mock Workers</span>
        </label>
      </div>
    </CardContent>
  </Card>

  <!-- Test Execution -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- Test Controls -->
    <Card>
      <CardHeader>
        <CardTitle>Test Execution</CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="flex gap-3">
          <Button
            on:click={runTests}
            disabled={!canRunTests}
            class="flex-1"
          >
            <Play class="w-4 h-4 mr-2" />
            {isRunning ? 'Running Tests...' : 'Run Integration Tests'}
          </Button>

          {#if isRunning}
            <Button
              on:click={stopTests}
              variant="outline"
            >
              <Square class="w-4 h-4 mr-2" />
              Stop
            </Button>
          {/if}
        </div>

        {#if isRunning}
          <div class="space-y-2">
            <div class="flex items-center justify-between text-sm">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} class="w-full" />
            {#if currentTest}
              <p class="text-sm text-gray-600">{currentTest}</p>
            {/if}
          </div>
        {/if}

        {#if summary}
          <div class="border rounded-lg p-4 bg-gray-50">
            <h4 class="font-semibold mb-3">Test Summary</h4>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="text-gray-600">Total Tests:</span>
                <span class="font-medium ml-2">{summary.totalTests}</span>
              </div>
              <div>
                <span class="text-gray-600">Success Rate:</span>
                <span class="font-medium ml-2 {successRate >= 80 ? 'text-green-600' : successRate >= 60 ? 'text-yellow-600' : 'text-red-600'}">
                  {successRate}%
                </span>
              </div>
              <div>
                <span class="text-gray-600">Passed:</span>
                <span class="font-medium ml-2 text-green-600">{summary.passed}</span>
              </div>
              <div>
                <span class="text-gray-600">Failed:</span>
                <span class="font-medium ml-2 text-red-600">{summary.failed}</span>
              </div>
              <div>
                <span class="text-gray-600">Duration:</span>
                <span class="font-medium ml-2">{summary.totalDuration}ms</span>
              </div>
            </div>
          </div>
        {/if}
      </CardContent>
    </Card>

    <!-- Test Logs -->
    <Card>
      <CardHeader>
        <CardTitle>Test Logs</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea class="h-64 w-full rounded-md border p-4">
          {#if testLogs.length === 0}
            <p class="text-gray-500 text-sm">No logs yet. Run tests to see real-time output.</p>
          {:else}
            <div class="space-y-1">
              {#each testLogs as log}
                <div class="text-xs font-mono text-gray-700">{log}</div>
              {/each}
            </div>
          {/if}
        </ScrollArea>
      </CardContent>
    </Card>
  </div>

  <!-- Detailed Results -->
  {#if hasResults}
    <Card>
      <CardHeader>
        <CardTitle>Detailed Test Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-4">
          {#each results as result}
            <div class="flex items-start gap-4 p-4 border rounded-lg">
              <svelte:component 
                this={getTestIcon(result)} 
                class="w-5 h-5 mt-0.5 {result.success ? 'text-green-600' : 'text-red-600'}"
              />
              
              <div class="flex-1">
                <div class="flex items-center gap-3 mb-2">
                  <h4 class="font-medium">{result.testName.replace(/_/g, ' ')}</h4>
                  <Badge variant={getTestBadgeVariant(result)}>
                    {result.success ? 'PASS' : 'FAIL'}
                  </Badge>
                  <span class="text-sm text-gray-500">{result.duration}ms</span>
                </div>

                {#if result.error}
                  <div class="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-md mb-3">
                    <AlertTriangle class="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
                    <p class="text-sm text-red-800">{result.error}</p>
                  </div>
                {/if}

                {#if result.details}
                  <div class="bg-gray-50 rounded-md p-3">
                    <h5 class="text-sm font-medium mb-2">Test Details</h5>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                      {#each Object.entries(result.details) as [key, value]}
                        <div class="flex justify-between">
                          <span class="text-gray-600">{key.replace(/([A-Z])/g, ' $1').toLowerCase()}:</span>
                          <span class="font-mono text-right">
                            {typeof value === 'object' ? JSON.stringify(value) : value}
                          </span>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}
</div>