<!--
Multi-LLM Orchestrator Demo Page
Showcases the service worker-based AI orchestration system
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { Textarea } from '$lib/components/ui/textarea';
  import { 
    Cpu, 
    Brain, 
    Zap, 
    Database,
    Play,
    Pause,
    RotateCcw,
    Settings,
    Activity,
    Users,
    Workflow
  } from 'lucide-svelte';
  
  import MultiLLMOrchestrator from '$lib/components/ai/MultiLLMOrchestrator.svelte';
  import LLMSelector from '$lib/components/ai/LLMSelector.svelte';
  import { aiWorkerManager, createGenerationTask, createAnalysisTask } from '$lib/services/ai-worker-manager.js';
  import type { AITask, LLMModel } from '$lib/types/ai-worker.js';

  // Demo state
  let selectedModel: LLMModel | undefined = $state();
  let userPrompt = $state('Analyze the following legal document for key terms, potential issues, and recommendations...');
  let isProcessing = $state(false);
  let demoResults = $state<Array<{ task: AITask; response?: unknown; error?: string }>>([]);
  
  // Demo scenarios
  const demoScenarios = [
    {
      name: 'Legal Document Analysis',
      description: 'Parallel analysis across multiple AI models',
      prompt: 'Analyze this contract for potential legal issues, key terms, and compliance requirements.',
      tasks: [
        { provider: 'ollama', model: 'gemma3-legal', focus: 'Legal compliance analysis' },
        { provider: 'vllm', model: 'vllm-gemma3-legal', focus: 'Risk assessment' },
        { provider: 'autogen', model: 'autogen-agents', focus: 'Multi-agent legal review' }
      ]
    },
    {
      name: 'Evidence Processing',
      description: 'Multi-stage evidence analysis pipeline',
      prompt: 'Process and categorize evidence files for case preparation.',
      tasks: [
        { provider: 'ollama', model: 'nomic-embed-text', focus: 'Text embedding generation' },
        { provider: 'ollama', model: 'gemma3-legal', focus: 'Content classification' },
        { provider: 'crewai', model: 'crewai-agents', focus: 'Evidence correlation' }
      ]
    },
    {
      name: 'Case Research',
      description: 'Comprehensive legal research workflow',
      prompt: 'Research relevant case law and statutes for this legal matter.',
      tasks: [
        { provider: 'autogen', model: 'autogen-agents', focus: 'Legal research coordination' },
        { provider: 'crewai', model: 'crewai-agents', focus: 'Case law analysis' },
        { provider: 'ollama', model: 'gemma3-legal', focus: 'Statute interpretation' }
      ]
    }
  ];

  onMount(() => {
    // Initialize with default model if available
    if (!selectedModel) {
      selectedModel = {
        id: 'gemma3-legal',
        name: 'gemma3:legal',
        displayName: 'Gemma3 Legal Specialist',
        provider: 'ollama',
        size: '7.3GB',
        specialization: 'legal',
        status: 'online',
        performance: {
          tokensPerSecond: 25,
          memoryUsage: '6.8GB',
          responseTime: 1200
        },
        capabilities: ['legal-analysis', 'case-research', 'document-review'],
        endpoint: 'http://localhost:11434'
      };
    }
  });

  async function runDemoScenario(scenario: any) {
    if (!selectedModel) return;

    isProcessing = true;
    demoResults = [];

    try {
      // Create tasks for the scenario
      const tasks = scenario.tasks.map((taskConfig: any) => 
        createAnalysisTask(
          `${scenario.prompt}\n\nFocus: ${taskConfig.focus}`,
          taskConfig.focus,
          taskConfig.model,
          taskConfig.provider,
          {
            priority: 'high',
            temperature: 0.1,
            maxTokens: 512
          }
        )
      );

      // Submit all tasks in parallel
      const taskPromises = tasks.map(async (task) => {
        try {
          demoResults = [...demoResults, { task }];
          
          const taskId = await aiWorkerManager.submitTask(task);
          const result = await aiWorkerManager.waitForTask(taskId);
          
          // Update result
          const index = demoResults.findIndex(r => r.task.taskId === task.taskId);
          if (index >= 0) {
            demoResults[index] = { task, response: result };
          }
          
          return result;
        } catch (error) {
          console.error('Task failed:', error);
          const index = demoResults.findIndex(r => r.task.taskId === task.taskId);
          if (index >= 0) {
            demoResults[index] = { task, error: (error as Error).message };
          }
        }
      });

      await Promise.all(taskPromises);
      console.log(`Demo scenario "${scenario.name}" completed`);
    } catch (error) {
      console.error('Demo scenario failed:', error);
    } finally {
      isProcessing = false;
    }
  }

  async function submitCustomTask() {
    if (!selectedModel || !userPrompt.trim()) return;

    isProcessing = true;

    try {
      const task = createGenerationTask(
        userPrompt,
        selectedModel.name,
        selectedModel.provider,
        {
          priority: 'high',
          temperature: 0.1,
          maxTokens: 1024
        }
      );

      demoResults = [{ task }];
      
      const taskId = await aiWorkerManager.submitTask(task);
      const result = await aiWorkerManager.waitForTask(taskId);
      
      demoResults = [{ task, response: result }];
      console.log('Custom task completed:', result);
    } catch (error) {
      console.error('Custom task failed:', error);
      demoResults = [{ task: demoResults[0]?.task, error: (error as Error).message }];
    } finally {
      isProcessing = false;
    }
  }

  function clearResults() {
    demoResults = [];
  }

  function getProviderIcon(providerId: string) {
    switch (providerId) {
      case 'ollama': return Cpu;
      case 'vllm': return Zap;
      case 'autogen': return Brain;
      case 'crewai': return Database;
      default: return Activity;
    }
  }

  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }
</script>

<svelte:head>
  <title>Multi-LLM Orchestrator - Legal AI System</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
  
  <!-- Header Section -->
  <div class="bg-white dark:bg-gray-800 shadow-sm border-b">
    <div class="max-w-7xl mx-auto px-6 py-8">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Multi-LLM Orchestrator
          </h1>
          <p class="text-xl text-gray-600 dark:text-gray-300 mt-2">
            Parallel AI processing with service workers and multiple LLM providers
          </p>
          <div class="flex items-center space-x-4 mt-4">
            <Badge class="bg-blue-500 text-white flex items-center gap-1">
              <Workflow class="h-3 w-3" />
              Service Workers
            </Badge>
            <Badge class="bg-green-500 text-white flex items-center gap-1">
              <Users class="h-3 w-3" />
              Multi-Agent
            </Badge>
            <Badge class="bg-purple-500 text-white flex items-center gap-1">
              <Zap class="h-3 w-3" />
              Parallel Processing
            </Badge>
            <Badge class="bg-orange-500 text-white flex items-center gap-1">
              <Brain class="h-3 w-3" />
              AI Orchestration
            </Badge>
          </div>
        </div>
        
        <div class="text-right">
          <p class="text-sm text-gray-500 dark:text-gray-400">Phase 2 Implementation</p>
          <p class="text-lg font-semibold text-gray-800 dark:text-gray-200">Service Worker Multi-threading</p>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-6 py-8 space-y-8">
    
    <!-- Quick Demo Section -->
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <Play class="h-5 w-5" />
          Quick Demo Scenarios
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {#each demoScenarios as scenario}
            <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <h3 class="font-semibold mb-2">{scenario.name}</h3>
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {scenario.description}
              </p>
              <div class="space-y-2 mb-4">
                {#each scenario.tasks as task}
                  {@const SvelteComponent = getProviderIcon(task.provider)}
                  <div class="flex items-center gap-2 text-xs">
                    <SvelteComponent 
                      class="h-3 w-3 text-blue-500" 
                    />
                    <span class="text-gray-600 dark:text-gray-400">{task.focus}</span>
                  </div>
                {/each}
              </div>
              <Button
                variant="outline"
                size="sm"
                class="w-full"
                on:on:click={() => runDemoScenario(scenario)}
                disabled={isProcessing}
              >
                {#if isProcessing}
                  <Pause class="h-4 w-4 mr-2" />
                  Processing...
                {:else}
                  <Play class="h-4 w-4 mr-2" />
                  Run Demo
                {/if}
              </Button>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>

    <!-- Custom Task Section -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Settings class="h-5 w-5" />
            Custom Task
          </CardTitle>
        </CardHeader>
        <CardContent class="space-y-4">
          <div>
            <label class="block text-sm font-medium mb-2">Select AI Model</label>
            <LLMSelector 
              bind:selectedModel={selectedModel}
              showMetrics={true}
              filterBy="all"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium mb-2">Task Prompt</label>
            <Textarea
              bind:value={userPrompt}
              placeholder="Enter your AI task prompt..."
              rows={4}
              class="w-full"
            />
          </div>
          
          <div class="flex gap-2">
            <Button
              on:on:click={submitCustomTask}
              disabled={isProcessing || !selectedModel}
              class="flex-1"
            >
              {#if isProcessing}
                <Pause class="h-4 w-4 mr-2" />
                Processing...
              {:else}
                <Play class="h-4 w-4 mr-2" />
                Submit Task
              {/if}
            </Button>
            
            <Button variant="outline" on:on:click={clearResults}>
              <RotateCcw class="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      <!-- Results Section -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center justify-between">
            <span class="flex items-center gap-2">
              <Activity class="h-5 w-5" />
              Task Results ({demoResults.length})
            </span>
            {#if demoResults.length > 0}
              <Button variant="ghost" size="sm" on:on:click={clearResults}>
                Clear
              </Button>
            {/if}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {#if demoResults.length === 0}
            <div class="text-center py-8 text-gray-500 dark:text-gray-400">
              <Activity class="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No results yet. Run a demo scenario or submit a custom task.</p>
            </div>
          {:else}
            <div class="space-y-3 max-h-96 overflow-y-auto">
              {#each demoResults as result}
                {@const SvelteComponent_1 = getProviderIcon(result.task.providerId)}
                <div class="border rounded-lg p-3 {result.error ? 'border-red-200 bg-red-50 dark:bg-red-900/20' : result.response ? 'border-green-200 bg-green-50 dark:bg-green-900/20' : 'border-yellow-200 bg-yellow-50 dark:bg-yellow-900/20'}">
                  <div class="flex items-start justify-between mb-2">
                    <div class="flex items-center gap-2">
                      <SvelteComponent_1 
                        class="h-4 w-4 text-blue-500" 
                      />
                      <span class="font-medium text-sm">
                        {result.task.providerId} - {result.task.model}
                      </span>
                      <Badge variant="outline" class="text-xs">
                        {result.task.type}
                      </Badge>
                    </div>
                    
                    {#if result.response}
                      <Badge class="bg-green-100 text-green-800 text-xs">
                        Completed
                      </Badge>
                    {:else if result.error}
                      <Badge class="bg-red-100 text-red-800 text-xs">
                        Failed
                      </Badge>
                    {:else}
                      <Badge class="bg-yellow-100 text-yellow-800 text-xs">
                        Processing
                      </Badge>
                    {/if}
                  </div>
                  
                  <p class="text-xs text-gray-600 dark:text-gray-400 mb-2">
                    {result.task.prompt.substring(0, 100)}...
                  </p>
                  
                  {#if result.response}
                    <div class="mt-2 p-2 bg-white dark:bg-gray-800 rounded text-xs">
                      <p class="font-medium mb-1">Response:</p>
                      <p class="text-gray-700 dark:text-gray-300">
                        {result.response.response?.content || 'Task completed successfully'}
                      </p>
                      {#if result.response.metrics}
                        <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
                          <span>Processing: {formatDuration(result.response.metrics.processingTime || 0)}</span>
                          <span>Tokens: {result.response.metrics.tokensProcessed || 0}</span>
                        </div>
                      {/if}
                    </div>
                  {:else if result.error}
                    <div class="mt-2 p-2 bg-red-100 dark:bg-red-900/30 rounded text-xs">
                      <p class="font-medium text-red-700 dark:text-red-400 mb-1">Error:</p>
                      <p class="text-red-600 dark:text-red-400">{result.error}</p>
                    </div>
                  {:else}
                    <div class="mt-2 flex items-center gap-2 text-xs text-gray-500">
                      <div class="animate-spin h-3 w-3 border border-gray-300 border-t-blue-500 rounded-full"></div>
                      <span>Processing task...</span>
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        </CardContent>
      </Card>
    </div>

    <!-- Main Orchestrator Component -->
    <MultiLLMOrchestrator 
      autoStart={true}
      showMetrics={true}
      maxConcurrenttasks={3}
      enabledProviders={['ollama', 'vllm', 'autogen', 'crewai']}
    />

    <!-- Architecture Information -->
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <Workflow class="h-5 w-5" />
          Architecture Overview
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 class="font-semibold mb-3">Service Worker Features</h3>
            <ul class="space-y-2 text-sm">
              <li class="flex items-center gap-2">
                <Zap class="h-4 w-4 text-green-500" />
                <span>Parallel task processing across multiple workers</span>
              </li>
              <li class="flex items-center gap-2">
                <Cpu class="h-4 w-4 text-blue-500" />
                <span>Intelligent load balancing and task distribution</span>
              </li>
              <li class="flex items-center gap-2">
                <RotateCcw class="h-4 w-4 text-purple-500" />
                <span>Automatic retry logic with exponential backoff</span>
              </li>
              <li class="flex items-center gap-2">
                <Activity class="h-4 w-4 text-orange-500" />
                <span>Real-time status monitoring and metrics</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h3 class="font-semibold mb-3">Supported Providers</h3>
            <ul class="space-y-2 text-sm">
              <li class="flex items-center gap-2">
                <Cpu class="h-4 w-4 text-blue-500" />
                <span><strong>Ollama:</strong> Local LLM hosting (Gemma3, Llama3)</span>
              </li>
              <li class="flex items-center gap-2">
                <Brain class="h-4 w-4 text-green-500" />
                <span><strong>AutoGen:</strong> Multi-agent conversation system</span>
              </li>
              <li class="flex items-center gap-2">
                <Database class="h-4 w-4 text-purple-500" />
                <span><strong>CrewAI:</strong> Role-based agent coordination</span>
              </li>
            </ul>
          </div>
        </div>
        
        <div class="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <p class="text-sm text-blue-800 dark:text-blue-300">
            <strong>Phase 2 Complete:</strong> Service worker infrastructure enables true multi-threading for AI tasks, 
            allowing parallel processing across different LLM providers while maintaining responsive UI interactions.
            Next phase will implement AutoGen and CrewAI agent coordination.
          </p>
        </div>
      </CardContent>
    </Card>
  </div>
</div>

<style>
  /* @unocss-include */
</style>