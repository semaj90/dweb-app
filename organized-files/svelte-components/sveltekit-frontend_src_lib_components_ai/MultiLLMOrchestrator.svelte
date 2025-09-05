<!--
Multi-LLM Orchestration Component
Provides UI for managing multiple AI workers and orchestrating parallel processing
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { derived, writable } from 'svelte/store';
  import { Badge } from '$lib/components/ui/badge';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Progress } from '$lib/components/ui/progress';
  import { 
    Play, 
    Pause, 
    Square, 
    RefreshCw, 
    Settings,
    Cpu,
    Brain,
    Zap,
    Database,
    Globe,
    Activity,
    Clock,
    CheckCircle,
    AlertCircle,
    X
  } from 'lucide-svelte';
  
  import { aiWorkerManager } from '$lib/services/ai-worker-manager.js';
  import type { 
    AITask, 
    AIResponse, 
    WorkerStatus, 
    ProcessingMetrics,
    WorkerPool 
  } from '$lib/types/ai-worker.js';

  interface Props {
    autoStart?: boolean;
    showMetrics?: boolean;
    maxConcurrentTasks?: number;
    enabledProviders?: string[];
  }

  let { 
    autoStart = true,
    showMetrics = true,
    maxConcurrentTasks = 3,
    enabledProviders = ['ollama', 'autogen', 'crewai']
  } = $props();

  // Component state
  let isInitialized = $state(false);
  let isProcessing = $state(false);
  let workerStatus = $state<WorkerStatus | null>(null);
  let workerPool = $state<WorkerPool | null>(null);
  let processingMetrics = $state<ProcessingMetrics[]>([]);
  let activeTasks = $state<Map<string, AITask>>(new Map());
  let completedTasks = $state<Map<string, AIResponse>>(new Map());
  let taskErrors = $state<Map<string, Error>>(new Map());

  // UI state
  let selectedTask = $state<string | null>(null);
  let showSettings = $state(false);
  let statusRefreshInterval: number | null = null;

  // Provider configurations
  let providerConfigs = $state([
    {
      id: 'ollama',
      name: 'Ollama',
      icon: Cpu,
      endpoint: 'http://localhost:11434',
      enabled: true,
      status: 'unknown',
      models: ['gemma3-legal', 'llama3:8b-instruct', 'nomic-embed-text']
    },
    
    {
      id: 'autogen',
      name: 'AutoGen',
      icon: Brain,
      endpoint: 'http://localhost:8001',
      enabled: true,
      status: 'unknown',
      models: ['autogen-agents']
    },
    {
      id: 'crewai',
      name: 'CrewAI',
      icon: Database,
      endpoint: 'http://localhost:8002',
      enabled: true,
      status: 'unknown',
      models: ['crewai-agents']
    }
  ]);

  // Derived stores
  let totalTasks = $derived(activeTasks.size + completedTasks.size + taskErrors.size)
  let successRate = $derived(totalTasks > 0 ? Math.round((completedTasks.size / totalTasks) * 100) : 0);
  let averageResponseTime = $derived(processingMetrics.length === 0 ? 0 : 
    Math.round(processingMetrics.reduce((sum, m) => sum + (m.processingTime || 0), 0) / processingMetrics.length)
  );

  onMount(async () => {
    if (autoStart) {
      await initializeOrchestrator();
    }
    
    // Set up event handlers
    aiWorkerManager.onTaskComplete = handleTaskComplete;
    aiWorkerManager.onTaskError = handleTaskError;
    aiWorkerManager.onStatusUpdate = handleStatusUpdate;
    
    // Start status monitoring
    startStatusMonitoring();
  });

  onDestroy(() => {
    if (statusRefreshInterval) {
      clearInterval(statusRefreshInterval);
    }
  });

  async function initializeOrchestrator() {
    try {
      isProcessing = true;
      
      await aiWorkerManager.initialize();
      
      isInitialized = true;
      await refreshStatus();
      
      console.log('Multi-LLM Orchestrator initialized successfully');
    } catch (error) {
      console.error('Failed to initialize orchestrator:', error);
    } finally {
      isProcessing = false;
    }
  }

  async function refreshStatus() {
    try {
      [workerStatus, workerPool] = await Promise.all([
        aiWorkerManager.getStatus(),
        Promise.resolve(aiWorkerManager.getWorkerPoolStatus())
      ]);
      
      processingMetrics = aiWorkerManager.getMetrics();
      
      // Check provider health
      await checkProviderHealth();
    } catch (error) {
      console.error('Failed to refresh status:', error);
    }
  }

  async function checkProviderHealth() {
    for (const provider of providerConfigs) {
      try {
        const response = await fetch(`${provider.endpoint}/health`, { 
          method: 'GET',
          signal: AbortSignal.timeout(2000)
        });
        provider.status = response.ok ? 'online' : 'offline';
      } catch {
        provider.status = 'offline';
      }
    }
  }

  function startStatusMonitoring() {
    statusRefreshInterval = setInterval(refreshStatus, 5000);
  }

  function handleTaskComplete(taskId: string, response: AIResponse) {
    activeTasks.delete(taskId);
    completedTasks.set(taskId, response);
    
    // Trigger reactivity
    activeTasks = new Map(activeTasks);
    completedTasks = new Map(completedTasks);
  }

  function handleTaskError(taskId: string, error: Error) {
    activeTasks.delete(taskId);
    taskErrors.set(taskId, error);
    
    // Trigger reactivity
    activeTasks = new Map(activeTasks);
    taskErrors = new Map(taskErrors);
  }

  function handleStatusUpdate(status: WorkerStatus) {
    workerStatus = status;
  }

  async function submitTestTask(providerId: string) {
    const testTask: AITask = {
      taskId: crypto.randomUUID(),
      type: 'generate',
      providerId,
      model: providerConfigs.find(p => p.id === providerId)?.models[0] || 'default',
      prompt: 'Hello! Please respond with a brief test message to verify the connection.',
      timestamp: Date.now(),
      priority: 'medium',
      temperature: 0.1,
      maxTokens: 50
    };

    try {
      activeTasks.set(testTask.taskId, testTask);
      activeTasks = new Map(activeTasks);
      
      await aiWorkerManager.submitTask(testTask);
      console.log(`Test task submitted to ${providerId}`);
    } catch (error) {
      console.error(`Failed to submit test task to ${providerId}:`, error);
      activeTasks.delete(testTask.taskId);
      taskErrors.set(testTask.taskId, error as Error);
    }
  }

  async function cancelTask(taskId: string) {
    try {
      await aiWorkerManager.cancelTask(taskId);
      activeTasks.delete(taskId);
      activeTasks = new Map(activeTasks);
    } catch (error) {
      console.error('Failed to cancel task:', error);
    }
  }

  async function clearCompletedTasks() {
    completedTasks.clear();
    taskErrors.clear();
    completedTasks = new Map();
    taskErrors = new Map();
  }

  function getProviderIcon(providerId: string) {
    return providerConfigs.find(p => p.id === providerId)?.icon || Globe;
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'online': return 'text-green-500';
      case 'offline': return 'text-red-500';
      case 'unknown': return 'text-gray-400';
      default: return 'text-gray-400';
    }
  }

  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }
</script>

<div class="w-full space-y-6">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        Multi-LLM Orchestrator
      </h2>
      <p class="text-gray-600 dark:text-gray-400">
        Manage and monitor multiple AI processing workers
      </p>
    </div>
    
    <div class="flex items-center gap-2">
      <Button
        variant="outline"
        size="sm"
        onclick={refreshStatus}
        disabled={isProcessing}
      >
        <RefreshCw class="h-4 w-4 mr-2 {isProcessing ? 'animate-spin' : ''}" />
        Refresh
      </Button>
      
      <Button
        variant="outline"
        size="sm"
        onclick={() => showSettings = !showSettings}
      >
        <Settings class="h-4 w-4" />
      </Button>
      
      {#if !isInitialized}
        <Button onclick={initializeOrchestrator} disabled={isProcessing}>
          <Play class="h-4 w-4 mr-2" />
          Initialize
        </Button>
      {/if}
    </div>
  </div>

  <!-- Status Overview -->
  {#if isInitialized && workerStatus}
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <Card>
        <CardContent class="p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Active Tasks</p>
              <p class="text-2xl font-bold">{workerStatus.activeRequests}</p>
            </div>
            <Activity class="h-8 w-8 text-blue-500" />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent class="p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Queue Length</p>
              <p class="text-2xl font-bold">{workerStatus.queueLength}</p>
            </div>
            <Clock class="h-8 w-8 text-yellow-500" />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent class="p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Success Rate</p>
              <p class="text-2xl font-bold">{successRate}%</p>
            </div>
            <CheckCircle class="h-8 w-8 text-green-500" />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent class="p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Avg Response</p>
              <p class="text-2xl font-bold">{formatDuration(averageResponseTime)}</p>
            </div>
            <Zap class="h-8 w-8 text-purple-500" />
          </div>
        </CardContent>
      </Card>
    </div>
  {/if}

  <!-- Provider Status -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Database class="h-5 w-5" />
        AI Providers
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {#each providerConfigs as provider}
          <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-2">
                <svelte:component this={provider.icon} class="h-5 w-5 text-blue-500" />
                <span class="font-medium">{provider.name}</span>
              </div>
              <Badge 
                class="px-2 py-1 text-xs {provider.status === 'online' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}"
              >
                {provider.status}
              </Badge>
            </div>
            
            <div class="space-y-2">
              <p class="text-xs text-gray-500">{provider.endpoint}</p>
              <div class="flex flex-wrap gap-1">
                {#each provider.models.slice(0, 2) as model}
                  <Badge variant="outline" class="text-xs">
                    {model.split(':')[0]}
                  </Badge>
                {/each}
                {#if provider.models.length > 2}
                  <Badge variant="outline" class="text-xs">
                    +{provider.models.length - 2}
                  </Badge>
                {/if}
              </div>
              
              <Button
                variant="outline"
                size="sm"
                class="w-full mt-2"
                onclick={() => submitTestTask(provider.id)}
                disabled={provider.status !== 'online'}
              >
                Test Connection
              </Button>
            </div>
          </div>
        {/each}
      </div>
    </CardContent>
  </Card>

  <!-- Active Tasks -->
  {#if activeTasks.size > 0}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center justify-between">
          <span class="flex items-center gap-2">
            <Activity class="h-5 w-5" />
            Active Tasks ({activeTasks.size})
          </span>
          <Button variant="outline" size="sm" onclick={clearCompletedTasks}>
            Clear Completed
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-3">
          {#each Array.from(activeTasks.entries()) as [taskId, task]}
            <div class="flex items-center justify-between p-3 border rounded-lg">
              <div class="flex items-center gap-3">
                <svelte:component 
                  this={getProviderIcon(task.providerId)} 
                  class="h-4 w-4 text-blue-500" 
                />
                <div>
                  <p class="font-medium text-sm">{task.type} - {task.model}</p>
                  <p class="text-xs text-gray-500 truncate max-w-md">
                    {task.prompt.substring(0, 100)}...
                  </p>
                </div>
              </div>
              
              <div class="flex items-center gap-2">
                <Badge variant="outline" class="text-xs">
                  {task.priority}
                </Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  onclick={() => cancelTask(taskId)}
                >
                  <X class="h-4 w-4" />
                </Button>
              </div>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Recent Results -->
  {#if completedTasks.size > 0 || taskErrors.size > 0}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <CheckCircle class="h-5 w-5" />
          Recent Results
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-3 max-h-96 overflow-y-auto">
          {#each Array.from(completedTasks.entries()) as [taskId, response]}
            <div class="flex items-start justify-between p-3 border border-green-200 rounded-lg bg-green-50 dark:bg-green-900/20">
              <div class="flex items-start gap-3">
                <CheckCircle class="h-4 w-4 text-green-500 mt-0.5" />
                <div class="flex-1">
                  <p class="font-medium text-sm">{response.providerId} - {response.model}</p>
                  <p class="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {response.content.substring(0, 150)}...
                  </p>
                  <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
                    <span>{response.tokensUsed} tokens</span>
                    <span>{formatDuration(response.responseTime)}</span>
                  </div>
                </div>
              </div>
            </div>
          {/each}
          
          {#each Array.from(taskErrors.entries()) as [taskId, error]}
            <div class="flex items-start justify-between p-3 border border-red-200 rounded-lg bg-red-50 dark:bg-red-900/20">
              <div class="flex items-start gap-3">
                <AlertCircle class="h-4 w-4 text-red-500 mt-0.5" />
                <div class="flex-1">
                  <p class="font-medium text-sm text-red-700 dark:text-red-400">
                    Task Failed
                  </p>
                  <p class="text-xs text-red-600 dark:text-red-400 mt-1">
                    {error.message}
                  </p>
                </div>
              </div>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Worker Pool Status -->
  {#if showMetrics && workerPool}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <Cpu class="h-5 w-5" />
          Worker Pool Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">Worker Distribution</p>
            <p class="text-lg font-medium">{workerPool.taskDistribution}</p>
          </div>
          <div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">Active Workers</p>
            <p class="text-lg font-medium">{workerPool.workers.length} / {workerPool.maxWorkers}</p>
          </div>
          <div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">Total Processed</p>
            <p class="text-lg font-medium">{workerPool.totalTasks}</p>
          </div>
        </div>
        
        {#if workerPool.currentLoad.length > 0}
          <div class="mt-4">
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">Worker Load Distribution</p>
            <div class="space-y-2">
              {#each workerPool.currentLoad as load, index}
                <div class="flex items-center gap-3">
                  <span class="text-sm font-medium w-16">Worker {index + 1}</span>
                  <div class="flex-1">
                    <Progress value={load} max={maxConcurrentTasks} class="h-2" />
                  </div>
                  <span class="text-sm text-gray-500">{load}/{maxConcurrentTasks}</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  /* @unocss-include */
</style>