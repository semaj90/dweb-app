<!--
Agent Orchestrator Component
Manages AutoGen and CrewAI multi-agent workflows
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '$lib/components/ui/select';
  import {
    Users,
    Brain,
    Database,
    Play,
    Pause,
    Square,
    RefreshCw,
    MessageSquare,
    FileText,
    Gavel,
    Search,
    Shield,
    Clock,
    CheckCircle,
    AlertCircle,
    Activity,
    Settings,
    Download
  } from 'lucide-svelte';

  import { autoGenService, analyzeCaseWithAgents, reviewEvidenceWithAgents, researchLegalPrecedents } from '$lib/services/autogen-service.js';
  import { crewAIService, analyzeLegalCaseWithCrew, analyzeContractWithCrew } from '$lib/services/crewai-service.js';
  import type {
    AutoGenConversation,
    AutoGenMessage
  } from '$lib/services/autogen-service.js';
  import type {
    CrewExecution,
    CrewTaskResult
  } from '$lib/services/crewai-service.js';

  interface Props {
    defaultWorkflow?: string;
    showAdvancedControls?: boolean;
    autoStartServices?: boolean;
  }

  let {
    defaultWorkflow = 'case_analysis',
    showAdvancedControls = true,
    autoStartServices = true
  } = $props();

  // Component state
  let selectedWorkflow = $state(defaultWorkflow);
  let selectedProvider = $state<'autogen' | 'crewai'>('autogen');
  let inputText = $state('');
  let isProcessing = $state(false);
  let serviceStatus = $state({ autogen: false, crewai: false });

  // Execution state
  let activeConversation = $state<AutoGenConversation | null>(null);
  let activeExecution = $state<CrewExecution | null>(null);
  let conversationMessages = $state<AutoGenMessage[]>([]);
  let executionResults = $state<CrewTaskResult[]>([]);

  // Monitoring
  let statusCheckInterval: number | null = null;
  let executionProgress = $state(0);
  let lastUpdate = $state<string>('');

  // Available workflows
  const workflows = [
    {
      id: 'case_analysis',
      name: 'Legal Case Analysis',
      description: 'Comprehensive case analysis with multiple legal experts',
      icon: Gavel,
      providers: ['autogen', 'crewai'],
      estimatedTime: '2-3 minutes'
    },
    {
      id: 'evidence_review',
      name: 'Evidence Review',
      description: 'Forensic evidence analysis and admissibility assessment',
      icon: Shield,
      providers: ['autogen', 'crewai'],
      estimatedTime: '1-2 minutes'
    },
    {
      id: 'legal_research',
      name: 'Legal Research',
      description: 'Precedent research and statute analysis',
      icon: Search,
      providers: ['autogen'],
      estimatedTime: '2-4 minutes'
    },
    {
      id: 'contract_analysis',
      name: 'Contract Analysis',
      description: 'Contract review, risk assessment, and negotiation strategy',
      icon: FileText,
      providers: ['crewai'],
      estimatedTime: '1-2 minutes'
    }
  ];

  onMount(async () => {
    if (autoStartServices) {
      await checkServiceStatus();
      startStatusMonitoring();
    }
  });

  onDestroy(() => {
    if (statusCheckInterval) {
      clearInterval(statusCheckInterval);
    }
  });

  async function checkServiceStatus() {
    try {
      const [autogenHealthy, crewaiHealthy] = await Promise.all([
        autoGenService.healthCheck(),
        crewAIService.healthCheck()
      ]);

      serviceStatus = { autogen: autogenHealthy, crewai: crewaiHealthy };
    } catch (error) {
      console.error('Failed to check service status:', error);
    }
  }

  function startStatusMonitoring() {
    statusCheckInterval = setInterval(checkServiceStatus, 10000); // Every 10 seconds
  }

  async function executeWorkflow() {
    if (!inputText.trim() || isProcessing) return;

    const workflow = workflows.find(w => w.id === selectedWorkflow);
    if (!workflow || !workflow.providers.includes(selectedProvider)) {
      console.error('Invalid workflow or provider combination');
      return;
    }

    isProcessing = true;
    executionProgress = 0;
    lastUpdate = 'Starting workflow...';

    try {
      if (selectedProvider === 'autogen') {
        await executeAutoGenWorkflow();
      } else {
        await executeCrewAIWorkflow();
      }
    } catch (error) {
      console.error('Workflow execution failed:', error);
      lastUpdate = `Error: ${(error as Error).message}`;
    } finally {
      isProcessing = false;
      executionProgress = 100;
    }
  }

  async function executeAutoGenWorkflow() {
    lastUpdate = 'Initializing AutoGen agents...';
    executionProgress = 10;

    switch (selectedWorkflow) {
      case 'case_analysis':
        activeConversation = null;
        conversationMessages = [];
        lastUpdate = 'Analyzing case with legal experts...';

        const caseResult = await analyzeCaseWithAgents(inputText, [], 'federal');

        // Simulate conversation for demo purposes
        conversationMessages = [
          {
            id: '1',
            sender: 'prosecutor',
            recipient: 'legal_researcher',
            content: 'Please research precedents for this case type.',
            timestamp: Date.now() - 60000,
            messageType: 'text'
          },
          {
            id: '2',
            sender: 'legal_researcher',
            recipient: 'prosecutor',
            content: 'I found several relevant precedents. The strongest cases support prosecution.',
            timestamp: Date.now() - 30000,
            messageType: 'text'
          },
          {
            id: '3',
            sender: 'coordinator',
            recipient: 'all',
            content: caseResult.content,
            timestamp: Date.now(),
            messageType: 'text'
          }
        ];

        lastUpdate = 'Case analysis completed';
        executionProgress = 100;
        break;

      case 'evidence_review':
        lastUpdate = 'Reviewing evidence with forensic experts...';
        executionProgress = 30;

        const evidenceResult = await reviewEvidenceWithAgents(inputText, 'digital', []);

        conversationMessages = [
          {
            id: '1',
            sender: 'evidence_analyst',
            recipient: 'prosecutor',
            content: evidenceResult.content,
            timestamp: Date.now(),
            messageType: 'text'
          }
        ];

        lastUpdate = 'Evidence review completed';
        break;

      case 'legal_research':
        lastUpdate = 'Researching legal precedents...';
        executionProgress = 40;

        const researchResult = await researchLegalPrecedents(inputText, 'federal', 'criminal');

        conversationMessages = [
          {
            id: '1',
            sender: 'legal_researcher',
            recipient: 'coordinator',
            content: researchResult.content,
            timestamp: Date.now(),
            messageType: 'text'
          }
        ];

        lastUpdate = 'Legal research completed';
        break;
    }
  }

  async function executeCrewAIWorkflow() {
    lastUpdate = 'Assembling CrewAI team...';
    executionProgress = 10;

    switch (selectedWorkflow) {
      case 'case_analysis':
        activeExecution = null;
        executionResults = [];
        lastUpdate = 'Legal investigation crew analyzing case...';

        const caseResult = await analyzeLegalCaseWithCrew(inputText, [], 'federal');

        // Simulate crew execution results
        executionResults = [
          {
            taskId: 'initial-investigation',
            agentId: 'case-investigator',
            output: 'Initial investigation completed. Key evidence identified and timeline established.',
            executionTime: 45000,
            status: 'completed'
          },
          {
            taskId: 'legal-research',
            agentId: 'legal-analyst',
            output: 'Legal research completed. Found 5 relevant precedents and applicable statutes.',
            executionTime: 60000,
            status: 'completed'
          },
          {
            taskId: 'evidence-analysis',
            agentId: 'evidence-specialist',
            output: 'Evidence analysis completed. All evidence meets admissibility standards.',
            executionTime: 30000,
            status: 'completed'
          },
          {
            taskId: 'final-report',
            agentId: 'report-writer',
            output: caseResult.content,
            executionTime: 25000,
            status: 'completed'
          }
        ];

        lastUpdate = 'Legal investigation completed';
        executionProgress = 100;
        break;

      case 'contract_analysis':
        lastUpdate = 'Contract analysis crew reviewing document...';
        executionProgress = 30;

        const contractResult = await analyzeContractWithCrew(inputText, 'commercial', 'general');

        executionResults = [
          {
            taskId: 'contract-review',
            agentId: 'contract-reviewer',
            output: 'Contract review completed. Identified 3 high-risk clauses and 2 missing provisions.',
            executionTime: 40000,
            status: 'completed'
          },
          {
            taskId: 'compliance-check',
            agentId: 'compliance-officer',
            output: 'Compliance analysis completed. Contract meets regulatory requirements with minor updates needed.',
            executionTime: 35000,
            status: 'completed'
          },
          {
            taskId: 'negotiation-strategy',
            agentId: 'negotiation-advisor',
            output: contractResult.content,
            executionTime: 20000,
            status: 'completed'
          }
        ];

        lastUpdate = 'Contract analysis completed';
        break;
    }
  }

  async function cancelExecution() {
    if (activeConversation) {
      try {
        await autoGenService.terminateConversation(activeConversation.id);
        activeConversation = null;
      } catch (error) {
        console.error('Failed to cancel AutoGen conversation:', error);
      }
    }

    if (activeExecution) {
      try {
        await crewAIService.cancelExecution(activeExecution.id);
        activeExecution = null;
      } catch (error) {
        console.error('Failed to cancel CrewAI execution:', error);
      }
    }

    isProcessing = false;
    lastUpdate = 'Execution cancelled';
  }

  function clearResults() {
    conversationMessages = [];
    executionResults = [];
    activeConversation = null;
    activeExecution = null;
    executionProgress = 0;
    lastUpdate = '';
  }

  function formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);

    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  }

  function downloadResults() {
    const results = selectedProvider === 'autogen'
      ? conversationMessages: executionResults

    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedWorkflow}_${selectedProvider}_results.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function getWorkflowIcon(workflowId: string) {
    return workflows.find(w => w.id === workflowId)?.icon || Activity;
  }

  function getServiceStatusColor(status: boolean) {
    return status ? 'text-green-500' : 'text-red-500';
  }
</script>

<div class="w-full space-y-6">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        Agent Orchestrator
      </h2>
      <p class="text-gray-600 dark:text-gray-400">
        Multi-agent AI workflows with AutoGen and CrewAI
      </p>
    </div>

    <div class="flex items-center gap-2">
      <Badge class="flex items-center gap-1 {getServiceStatusColor(serviceStatus.autogen)}">
        <Brain class="h-3 w-3" />
        AutoGen {serviceStatus.autogen ? 'Online' : 'Offline'}
      </Badge>
      <Badge class="flex items-center gap-1 {getServiceStatusColor(serviceStatus.crewai)}">
        <Database class="h-3 w-3" />
        CrewAI {serviceStatus.crewai ? 'Online' : 'Offline'}
      </Badge>

      <Button
        variant="outline"
        size="sm"
        on:click={checkServiceStatus}
      >
        <RefreshCw class="h-4 w-4" />
      </Button>
    </div>
  </div>

  <!-- Workflow Configuration -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Settings class="h-5 w-5" />
        Workflow Configuration
      </CardTitle>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium mb-2">Workflow Type</label>
          <Select bind:value={selectedWorkflow}>
            <SelectTrigger>
              <SelectValue placeholder="Select workflow..." />
            </SelectTrigger>
            <SelectContent>
              {#each workflows as workflow}
                <SelectItem value={workflow.id}>
                  <div class="flex items-center gap-2">
                    <svelte:component this={workflow.icon} class="h-4 w-4" />
                    {workflow.name}
                  </div>
                </SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>

        <div>
          <label class="block text-sm font-medium mb-2">AI Provider</label>
          <Select bind:value={selectedProvider}>
            <SelectTrigger>
              <SelectValue placeholder="Select provider..." />
            </SelectTrigger>
            <SelectContent>
              {#each workflows.find(w => w.id === selectedWorkflow)?.providers || [] as provider}
                <SelectItem value={provider}>
                  <div class="flex items-center gap-2">
                    {#if provider === 'autogen'}
                      <Brain class="h-4 w-4" />
                      AutoGen
                    {:else}
                      <Database class="h-4 w-4" />
                      CrewAI
                    {/if}
                  </div>
                </SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>
      </div>

      {#if selectedWorkflow}
        {@const workflow = workflows.find(w => w.id === selectedWorkflow)}
        <div class="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div class="flex items-start gap-3">
            <svelte:component this={workflow?.icon || Activity} class="h-5 w-5 text-blue-500 mt-0.5" />
            <div>
              <p class="font-medium text-blue-800 dark:text-blue-300">{workflow?.name}</p>
              <p class="text-sm text-blue-600 dark:text-blue-400">{workflow?.description}</p>
              <p class="text-xs text-blue-500 dark:text-blue-400 mt-1">
                Estimated time: {workflow?.estimatedTime}
              </p>
            </div>
          </div>
        </div>
      {/if}

      <div>
        <label class="block text-sm font-medium mb-2">Input</label>
        <Textarea
          bind:value={inputText}
          placeholder="Enter your legal case description, evidence details, or contract text..."
          rows={4}
          class="w-full"
        />
      </div>

      <div class="flex gap-2">
        <Button
          on:click={executeWorkflow}
          disabled={isProcessing || !inputText.trim() || (!serviceStatus.autogen && selectedProvider === 'autogen') || (!serviceStatus.crewai && selectedProvider === 'crewai')}
          class="flex-1"
        >
          {#if isProcessing}
            <Pause class="h-4 w-4 mr-2" />
            Processing...
          {:else}
            <Play class="h-4 w-4 mr-2" />
            Execute Workflow
          {/if}
        </Button>

        {#if isProcessing}
          <Button variant="outline" on:click={cancelExecution}>
            <Square class="h-4 w-4" />
          </Button>
        {/if}

        {#if conversationMessages.length > 0 || executionResults.length > 0}
          <Button variant="outline" on:click={clearResults}>
            Clear
          </Button>
          <Button variant="outline" on:click={downloadResults}>
            <Download class="h-4 w-4" />
          </Button>
        {/if}
      </div>
    </CardContent>
  </Card>

  <!-- Execution Status -->
  {#if isProcessing || lastUpdate}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <Activity class="h-5 w-5" />
          Execution Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-sm font-medium">Progress</span>
            <span class="text-sm text-gray-500">{executionProgress}%</span>
          </div>

          <div class="w-full bg-gray-200 rounded-full h-2">
            <div
              class="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
              style="width: {executionProgress}%"
            ></div>
          </div>

          <div class="flex items-center gap-2 text-sm">
            {#if isProcessing}
              <div class="animate-spin h-4 w-4 border border-gray-300 border-t-blue-500 rounded-full"></div>
            {:else}
              <CheckCircle class="h-4 w-4 text-green-500" />
            {/if}
            <span class="text-gray-700 dark:text-gray-300">{lastUpdate}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Results Display -->
  {#if selectedProvider === 'autogen' && conversationMessages.length > 0}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <MessageSquare class="h-5 w-5" />
          AutoGen Conversation ({conversationMessages.length} messages)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-3 max-h-96 overflow-y-auto">
          {#each conversationMessages as message}
            <div class="flex items-start gap-3 p-3 border rounded-lg">
              <div class="flex-shrink-0">
                <div class="h-8 w-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                  <span class="text-xs font-medium text-blue-600 dark:text-blue-400">
                    {message.sender.substring(0, 2).toUpperCase()}
                  </span>
                </div>
              </div>

              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 mb-1">
                  <span class="font-medium text-sm">{message.sender}</span>
                  {#if message.recipient !== 'all'}
                    <span class="text-xs text-gray-500">→ {message.recipient}</span>
                  {/if}
                  <span class="text-xs text-gray-400">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p class="text-sm text-gray-700 dark:text-gray-300">{message.content}</p>
              </div>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}

  {#if selectedProvider === 'crewai' && executionResults.length > 0}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <Users class="h-5 w-5" />
          CrewAI Execution Results ({executionResults.length} tasks)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-3">
          {#each executionResults as result}
            <div class="border rounded-lg p-4">
              <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                  <span class="font-medium text-sm">{result.taskId}</span>
                  <Badge variant="outline" class="text-xs">
                    {result.agentId}
                  </Badge>
                </div>

                <div class="flex items-center gap-2">
                  {#if result.status === 'completed'}
                    <CheckCircle class="h-4 w-4 text-green-500" />
                  {:else if result.status === 'failed'}
                    <AlertCircle class="h-4 w-4 text-red-500" />
                  {:else}
                    <Clock class="h-4 w-4 text-yellow-500" />
                  {/if}
                  <span class="text-xs text-gray-500">
                    {formatDuration(result.executionTime)}
                  </span>
                </div>
              </div>

              <p class="text-sm text-gray-700 dark:text-gray-300">{result.output}</p>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Workflow Templates -->
  {#if showAdvancedControls}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <FileText class="h-5 w-5" />
          Quick Start Templates
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Button
            variant="outline"
            class="h-auto p-4 justify-start"
            on:click={() => {
              selectedWorkflow = 'case_analysis';
              selectedProvider = 'autogen';
              inputText = 'John Smith was accused of embezzling $50,000 from his employer over a 6-month period. Evidence includes suspicious bank transfers, altered financial records, and witness testimony from colleagues who noticed unusual behavior.';
            }}
          >
            <div class="text-left">
              <p class="font-medium">Criminal Case Analysis</p>
              <p class="text-xs text-gray-500">AutoGen multi-agent analysis</p>
            </div>
          </Button>

          <Button
            variant="outline"
            class="h-auto p-4 justify-start"
            on:click={() => {
              selectedWorkflow = 'contract_analysis';
              selectedProvider = 'crewai';
              inputText = 'Software licensing agreement between TechCorp and ClientCorp for enterprise SaaS platform. Contract includes liability limitations, data processing clauses, and termination provisions. Review for compliance and negotiation opportunities.';
            }}
          >
            <div class="text-left">
              <p class="font-medium">Contract Review</p>
              <p class="text-xs text-gray-500">CrewAI specialized team</p>
            </div>
          </Button>

          <Button
            variant="outline"
            class="h-auto p-4 justify-start"
            on:click={() => {
              selectedWorkflow = 'evidence_review';
              selectedProvider = 'autogen';
              inputText = 'Digital evidence package includes: smartphone data extraction, email communications, cloud storage files, and network logs. Chain of custody maintained by certified technician. Need admissibility assessment for federal court.';
            }}
          >
            <div class="text-left">
              <p class="font-medium">Digital Evidence Review</p>
              <p class="text-xs text-gray-500">Forensic analysis workflow</p>
            </div>
          </Button>

          <Button
            variant="outline"
            class="h-auto p-4 justify-start"
            on:click={() => {
              selectedWorkflow = 'legal_research';
              selectedProvider = 'autogen';
              inputText = 'Research precedents for cryptocurrency fraud cases involving privacy coins. Focus on 4th Amendment protections, blockchain analysis admissibility, and international cooperation in digital asset recovery.';
            }}
          >
            <div class="text-left">
              <p class="font-medium">Legal Research</p>
              <p class="text-xs text-gray-500">Precedent and statute analysis</p>
            </div>
          </Button>
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  /* @unocss-include */
</style>