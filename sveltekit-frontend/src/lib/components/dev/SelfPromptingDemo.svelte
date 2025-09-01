<script lang="ts">

  import { onMount } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import { 
    copilotOrchestrator, 
    generateMCPPrompt, 
    commonMCPQueries, 
    formatMCPResponse,
    type OrchestrationOptions,
    type AgentResult 
  } from '$lib/utils/mcp-helpers';

  // Agent orchestration state
  const activeAgents = writable<Set<string>>(new Set());
  const agentResults = writable<AgentResult[]>([]);
  const orchestrationLog = writable<Array<{
    timestamp: string;
    phase: string;
    agent?: string;
    prompt: string;
    result?: unknown;
    selfPrompt?: string;
  }>>([]);
  const isRunning = writable(false);
  const currentPhase = writable<string>('idle');

  // Workflow configuration
  let selectedWorkflow = $state('legal-evidence-analysis');
  let initialPrompt = $state('Analyze the evidence for case CASE-2024-001 and build a comprehensive legal argument');
  let enableSelfPrompting = $state(true);
  let enableMultiAgent = $state(true);
  let maxIterations = $state(3);
  let currentIteration = $state(0);

  // Progress tracking
  const progress = derived([currentPhase, isRunning], ([$phase, $running]) => {
    if (!$running) return 0;
    const phases = ['semantic-search', 'memory-analysis', 'agent-coordination', 'self-prompting', 'synthesis'];
    return ((phases.indexOf($phase) + 1) / phases.length) * 100;
  });

  // Available workflows
  const workflows = [
    {
      id: 'legal-evidence-analysis',
      name: 'Legal Evidence Analysis',
      description: 'Multi-agent analysis of evidence with case building and recommendation generation',
      agents: ['autogen', 'crewai', 'claude'],
      options: {
        useSemanticSearch: true,
        useMemory: true,
        useMultiAgent: true,
        useCodebase: true,
        synthesizeOutputs: true
      }
    },
    {
      id: 'development-guidance',
      name: 'Development Guidance',
      description: 'Context7 MCP integration for stack analysis and best practices',
      agents: ['context7', 'copilot', 'claude'],
      options: {
        useSemanticSearch: true,
        useCodebase: true,
        synthesizeOutputs: true
      }
    },
    {
      id: 'self-improving-workflow',
      name: 'Self-Improving Workflow',
      description: 'Agents analyze their own outputs and generate follow-up prompts',
      agents: ['autogen', 'crewai', 'vllm', 'claude'],
      options: {
        useMultiAgent: true,
        synthesizeOutputs: true,
        logErrors: true
      }
    }
  ];

  // Agent communication visualization
  const agentCommunications = writable<Array<{
    from: string;
    to: string;
    message: string;
    timestamp: string;
    type: 'prompt' | 'result' | 'self-prompt';
  }>>([]);

  // Real-time agent status
  const agentStatus = writable<Record<string, {
    status: 'idle' | 'processing' | 'completed' | 'error';
    currentTask: string;
    progress: number;
    lastUpdate: string;
  }>>({});

  /**
   * Execute the self-prompting orchestration workflow
   */
  async function executeWorkflow() {
    if ($isRunning) return;

    isRunning.set(true);
    currentIteration = 0;
    agentResults.set([]);
    orchestrationLog.set([]);
    agentCommunications.set([]);
    
    const selectedWorkflowConfig = workflows.find(w => w.id === selectedWorkflow);
    if (!selectedWorkflowConfig) {
      console.error('Selected workflow not found');
      isRunning.set(false);
      return;
    }

    try {
      await runOrchestrationLoop(selectedWorkflowConfig, initialPrompt);
    } catch (error) {
      console.error('Workflow execution error:', error);
      addLogEntry('error', '', `Workflow failed: ${error}`, {});
    } finally {
      isRunning.set(false);
      currentPhase.set('completed');
    }
  }

  /**
   * Main orchestration loop with self-prompting
   */
  async function runOrchestrationLoop(workflow: any, prompt: string) {
    let currentPrompt = prompt;
    
    for (let i = 0; i < maxIterations; i++) {
      currentIteration = i + 1;
      addLogEntry('iteration-start', '', `Starting iteration ${i + 1}/${maxIterations}`, {});

      // Phase 1: Semantic Search and Memory Analysis
      currentPhase.set('semantic-search');
      const searchResults = await executeSemanticSearch(currentPrompt);
      
      // Phase 2: Memory Graph Reading
      currentPhase.set('memory-analysis');
      const memoryResults = await executeMemoryAnalysis(currentPrompt);

      // Phase 3: Multi-Agent Coordination
      currentPhase.set('agent-coordination');
      const orchestrationOptions: OrchestrationOptions = {
        ...workflow.options,
        agents: workflow.agents,
        context: { 
          iteration: i + 1,
          previousResults: i > 0 ? $agentResults : [],
          searchResults,
          memoryResults
        }
      };

      const results = await copilotOrchestrator(currentPrompt, orchestrationOptions);
      
      // Update agent results store
      if (results.agentResults) {
        agentResults.update(prev => [...prev, ...results.agentResults]);
      }

      // Phase 4: Self-Prompting Generation
      currentPhase.set('self-prompting');
      if (enableSelfPrompting && results.selfPrompt) {
        const selfPromptResult = await generateSelfPrompt(results, workflow.agents);
        
        // Use the self-generated prompt for the next iteration
        if (selfPromptResult.nextPrompt && i < maxIterations - 1) {
          currentPrompt = selfPromptResult.nextPrompt;
          addLogEntry('self-prompt-generated', '', 'Generated new prompt for next iteration', {
            nextPrompt: currentPrompt,
            reasoning: selfPromptResult.reasoning
          });
        }
      }

      // Phase 5: Result Synthesis and Analysis
      currentPhase.set('synthesis');
      await synthesizeIterationResults(results, i + 1);

      // Break early if agents decide the task is complete
      if (await checkTaskCompletion(results)) {
        addLogEntry('task-completion', '', 'Agents determined task is complete', results);
        break;
      }

      // Simulate processing delay for demo
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Final synthesis
    await generateFinalReport();
  }

  /**
   * Execute semantic search with Context7 MCP integration
   */
  async function executeSemanticSearch(prompt: string) {
    addLogEntry('semantic-search', 'context7', `Executing semantic search for: ${prompt}`, {});
    
    updateAgentStatus('context7', 'processing', 'Performing semantic search');
    
    // Simulate Context7 semantic search
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const searchResults = {
      query: prompt,
      results: [
        { 
          document: 'Legal Precedent DB',
          relevance: 0.89,
          snippet: 'Evidence analysis protocols for criminal cases...',
          caseId: 'PRECEDENT-2023-045'
        },
        {
          document: 'Prosecution Guidelines',
          relevance: 0.76,
          snippet: 'Multi-agent coordination in evidence processing...',
          caseId: 'GUIDELINE-PROC-001'
        }
      ]
    };

    updateAgentStatus('context7', 'completed', 'Semantic search completed');
    addLogEntry('semantic-search', 'context7', 'Semantic search completed', searchResults);
    
    return searchResults;
  }

  /**
   * Execute memory graph analysis
   */
  async function executeMemoryAnalysis(prompt: string) {
    addLogEntry('memory-analysis', 'memory-server', `Reading memory graph for: ${prompt}`, {});
    
    updateAgentStatus('memory-server', 'processing', 'Analyzing memory graph');
    
    await new Promise(resolve => setTimeout(resolve, 600));
    
    const memoryResults = {
      nodes: [
        {
          id: 'case-CASE-2024-001',
          type: 'legal-case',
          relations: ['evidence-items', 'legal-precedents', 'witness-statements'],
          strength: 0.92
        },
        {
          id: 'evidence-chain',
          type: 'evidence-network',
          relations: ['physical-evidence', 'digital-evidence', 'testimonial-evidence'],
          strength: 0.85
        }
      ],
      insights: [
        'Strong evidence chain established for case CASE-2024-001',
        'Multiple precedent cases support prosecution strategy',
        'Key witness testimony patterns identified'
      ]
    };

    updateAgentStatus('memory-server', 'completed', 'Memory analysis completed');
    addLogEntry('memory-analysis', 'memory-server', 'Memory graph analysis completed', memoryResults);
    
    return memoryResults;
  }

  /**
   * Generate self-prompting suggestions
   */
  async function generateSelfPrompt(results: any, agents: string[]) {
    addLogEntry('self-prompting', 'meta-agent', 'Analyzing results for self-prompting', {});
    
    // Simulate meta-analysis by the orchestration system
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const analysis = {
      strengths: [
        'Strong legal precedent identification',
        'Comprehensive evidence mapping',
        'Multi-agent coordination successful'
      ],
      weaknesses: [
        'Need deeper witness testimony analysis',
        'Missing timeline correlation analysis',
        'Insufficient defendant background research'
      ],
      nextFocus: 'Investigate temporal relationships between evidence items and witness statements'
    };

    const nextPrompt = `Based on previous analysis, focus on: ${analysis.nextFocus}. 
    Specifically examine witness testimony timing relative to evidence collection, 
    and identify any inconsistencies or corroborating patterns that strengthen the case.`;

    const selfPromptResult = {
      analysis,
      nextPrompt,
      reasoning: 'Previous iteration showed strong evidence foundation but lacked temporal analysis. Next iteration should focus on timeline correlation to strengthen legal argument.',
      confidence: 0.87
    };

    addLogEntry('self-prompting', 'meta-agent', 'Self-prompt generated', selfPromptResult);
    
    return selfPromptResult;
  }

  /**
   * Synthesize results from each iteration
   */
  async function synthesizeIterationResults(results: any, iteration: number) {
    addLogEntry('synthesis', 'synthesizer', `Synthesizing iteration ${iteration} results`, {});
    
    const synthesis = {
      iteration,
      keyFindings: [
        `Iteration ${iteration}: Enhanced evidence correlation discovered`,
        'Agent coordination efficiency improved by 15%',
        'Legal argument strength increased through precedent analysis'
      ],
      agentPerformance: workflow.agents.map(agent => ({
        agent,
        performance: Math.random() * 0.3 + 0.7, // Simulate performance metric
        contribution: `Agent ${agent} contributed specialized analysis`
      })),
      nextSteps: iteration < maxIterations ? ['Continue with refined focus', 'Apply self-prompting insights'] : ['Prepare final report']
    };

    addLogEntry('synthesis', 'synthesizer', 'Iteration synthesis completed', synthesis);
    
    return synthesis;
  }

  /**
   * Check if task is complete based on agent consensus
   */
  async function checkTaskCompletion(results: any): Promise<boolean> {
    // Simple heuristic: if we have sufficient high-confidence results
    const agentConfidence = results.agentResults?.map((r: any) => r.confidence || 0.8) || [];
    const avgConfidence = agentConfidence.reduce((a: number, b: number) => a + b, 0) / agentConfidence.length;
    
    return avgConfidence > 0.85 && currentIteration >= 2;
  }

  /**
   * Generate final comprehensive report
   */
  async function generateFinalReport() {
    currentPhase.set('final-report');
    addLogEntry('final-report', 'orchestrator', 'Generating comprehensive final report', {});
    
    const finalReport = {
      summary: 'Multi-agent legal evidence analysis completed successfully',
      totalIterations: currentIteration,
      agentContributions: $agentResults.length,
      keyInsights: [
        'Evidence chain integrity verified through multi-agent analysis',
        'Legal precedents strongly support prosecution strategy',
        'Witness testimony patterns show high consistency',
        'Self-prompting improved analysis depth by 40%'
      ],
      recommendations: [
        'Proceed with prosecution based on strong evidence foundation',
        'Focus on temporal evidence correlation in court presentation',
        'Leverage identified precedent cases for legal arguments',
        'Consider additional witness testimony for case strengthening'
      ],
      confidenceScore: 0.91,
      generatedAt: new Date().toISOString()
    };

    addLogEntry('final-report', 'orchestrator', 'Final report generated', finalReport);
  }

  /**
   * Context7 MCP tool integration demonstration
   */
  async function demonstrateContext7Integration() {
    const mcpQueries = [
      commonMCPQueries.analyzeSvelteKit(),
      commonMCPQueries.performanceBestPractices(),
      commonMCPQueries.securityBestPractices(),
      commonMCPQueries.aiChatIntegration()
    ];

    for (const query of mcpQueries) {
      const prompt = generateMCPPrompt(query);
      addLogEntry('context7-demo', 'context7', `Executing: ${prompt}`, {});
      
      // Simulate MCP tool execution
      await new Promise(resolve => setTimeout(resolve, 300));
      
      const mockResult = {
        tool: query.tool,
        result: `Context7 analysis completed for ${query.component || query.area || query.feature}`,
        recommendations: [
          'Follow SvelteKit best practices for legal applications',
          'Implement proper security measures for sensitive data',
          'Optimize performance for large document processing'
        ]
      };

      addLogEntry('context7-demo', 'context7', `Result: ${formatMCPResponse(mockResult)}`, mockResult);
    }
  }

  /**
   * Utility functions
   */
  function addLogEntry(phase: string, agent: string, prompt: string, result: any) {
    orchestrationLog.update(log => [...log, {
      timestamp: new Date().toISOString(),
      phase,
      agent,
      prompt,
      result,
      selfPrompt: result.selfPrompt
    }]);
  }

  function updateAgentStatus(agent: string, status: 'idle' | 'processing' | 'completed' | 'error', task: string) {
    agentStatus.update(statusMap => ({
      ...statusMap,
      [agent]: {
        status,
        currentTask: task,
        progress: status === 'completed' ? 100 : status === 'processing' ? 50 : 0,
        lastUpdate: new Date().toISOString()
      }
    }));
  }

  function addAgentCommunication(from: string, to: string, message: string, type: 'prompt' | 'result' | 'self-prompt') {
    agentCommunications.update(comms => [...comms, {
      from,
      to,
      message,
      timestamp: new Date().toISOString(),
      type
    }]);
  }

  function clearLogs() {
    orchestrationLog.set([]);
    agentResults.set([]);
    agentCommunications.set([]);
    agentStatus.set({});
    currentPhase.set('idle');
  }

  function stopWorkflow() {
    isRunning.set(false);
    currentPhase.set('stopped');
  }

  // Initialize demo data
  onMount(() => {
    // Add some demo agent status
    updateAgentStatus('autogen', 'idle', 'Ready for legal analysis');
    updateAgentStatus('crewai', 'idle', 'Awaiting workflow coordination');
    updateAgentStatus('vllm', 'idle', 'High-performance inference ready');
    updateAgentStatus('claude', 'idle', 'Ready for legal reasoning');
    updateAgentStatus('context7', 'idle', 'MCP tools available');
  });
</script>

<div class="space-y-6 p-6 max-w-7xl mx-auto">
  <!-- Header -->
  <div class="border-b border-gray-200 pb-4">
    <h2 class="text-3xl font-bold text-gray-900">Self-Prompting Agent Orchestration Demo</h2>
    <p class="text-gray-600 mt-2">
      Comprehensive demonstration of multi-agent coordination with self-prompting loops,
      Context7 MCP integration, and legal AI workflow automation.
    </p>
  </div>

  <!-- Progress Indicator -->
  {#if $isRunning}
    <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-medium text-blue-900">Phase: {$currentPhase}</span>
        <span class="text-sm text-blue-700">Iteration {currentIteration}/{maxIterations}</span>
      </div>
      <div class="w-full bg-blue-200 rounded-full h-2">
        <div 
          class="bg-blue-600 h-2 rounded-full transition-all duration-500"
          style="width: {$progress}%"
        ></div>
      </div>
    </div>
  {/if}

  <!-- Workflow Configuration -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold mb-4">Workflow Configuration</h3>
      
      <div class="space-y-4">
        <div>
          <label for="workflow" class="block text-sm font-medium mb-2">Select Workflow</label>
          <select
            id="workflow"
            bind:value={selectedWorkflow}
            disabled={$isRunning}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {#each workflows as workflow}
              <option value={workflow.id}>{workflow.name}</option>
            {/each}
          </select>
          <p class="text-sm text-gray-600 mt-1">
            {workflows.find(w => w.id === selectedWorkflow)?.description}
          </p>
        </div>

        <div>
          <label for="prompt" class="block text-sm font-medium mb-2">Initial Prompt</label>
          <textarea
            id="prompt"
            bind:value={initialPrompt}
            disabled={$isRunning}
            rows="3"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            placeholder="Enter the initial task for the agent orchestration..."
          ></textarea>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="flex items-center">
              <input
                type="checkbox"
                bind:checked={enableSelfPrompting}
                disabled={$isRunning}
                class="mr-2"
              />
              <span class="text-sm">Enable Self-Prompting</span>
            </label>
          </div>
          <div>
            <label class="flex items-center">
              <input
                type="checkbox"
                bind:checked={enableMultiAgent}
                disabled={$isRunning}
                class="mr-2"
              />
              <span class="text-sm">Multi-Agent Coordination</span>
            </label>
          </div>
        </div>

        <div>
          <label for="iterations" class="block text-sm font-medium mb-2">Max Iterations</label>
          <input
            id="iterations"
            type="number"
            bind:value={maxIterations}
            disabled={$isRunning}
            min="1"
            max="10"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
        </div>
      </div>

      <div class="flex gap-3 mt-6">
        <button
          on:onclick={executeWorkflow}
          disabled={$isRunning}
          class="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {#if $isRunning}
            <span class="flex items-center justify-center gap-2">
              <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Running Workflow...
            </span>
          {:else}
            Execute Workflow
          {/if}
        </button>
        
        {#if $isRunning}
          <button
            on:onclick={stopWorkflow}
            class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            Stop
          </button>
        {/if}
        
        <button
          on:onclick={clearLogs}
          disabled={$isRunning}
          class="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          Clear
        </button>
      </div>
    </div>

    <!-- Agent Status Panel -->
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold mb-4">Agent Status</h3>
      
      <div class="space-y-3">
        {#each Object.entries($agentStatus) as [agent, status]}
          <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-3 h-3 rounded-full {
                status.status === 'processing' ? 'bg-blue-500 animate-pulse' :
                status.status === 'completed' ? 'bg-green-500' :
                status.status === 'error' ? 'bg-red-500' :
                'bg-gray-400'
              }"></div>
              <div>
                <div class="font-medium">{agent}</div>
                <div class="text-sm text-gray-600">{status.currentTask}</div>
              </div>
            </div>
            <div class="text-right">
              <div class="text-sm font-medium">{status.status}</div>
              <div class="text-xs text-gray-500">{status.progress}%</div>
            </div>
          </div>
        {/each}
      </div>

      <div class="mt-4 pt-4 border-t border-gray-200">
        <button
          on:onclick={demonstrateContext7Integration}
          disabled={$isRunning}
          class="w-full bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
        >
          Demo Context7 MCP Integration
        </button>
      </div>
    </div>
  </div>

  <!-- Real-time Orchestration Log -->
  <div class="bg-white border border-gray-200 rounded-lg p-6">
    <h3 class="text-lg font-semibold mb-4">Real-time Orchestration Log</h3>
    
    <div class="max-h-96 overflow-y-auto space-y-3">
      {#each $orchestrationLog as entry}
        <div class="border-l-4 pl-4 py-2 {
          entry.phase === 'error' ? 'border-red-500 bg-red-50' :
          entry.phase === 'final-report' ? 'border-green-500 bg-green-50' :
          entry.phase === 'self-prompting' ? 'border-purple-500 bg-purple-50' :
          'border-blue-500 bg-blue-50'
        }">
          <div class="flex items-center justify-between mb-1">
            <span class="text-sm font-medium">
              {entry.phase} {entry.agent ? `(${entry.agent})` : ''}
            </span>
            <span class="text-xs text-gray-500">
              {new Date(entry.timestamp).toLocaleTimeString()}
            </span>
          </div>
          <div class="text-sm text-gray-700 mb-1">{entry.prompt}</div>
          {#if entry.result && Object.keys(entry.result).length > 0}
            <details class="text-xs">
              <summary class="cursor-pointer text-blue-600 hover:text-blue-800">View Details</summary>
              <pre class="mt-2 p-2 bg-white rounded text-gray-600 overflow-x-auto">{JSON.stringify(entry.result, null, 2)}</pre>
            </details>
          {/if}
        </div>
      {/each}
      
      {#if $orchestrationLog.length === 0}
        <div class="text-center text-gray-500 py-8">
          No orchestration activity yet. Click "Execute Workflow" to begin.
        </div>
      {/if}
    </div>
  </div>

  <!-- Agent Communication Visualization -->
  {#if $agentCommunications.length > 0}
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold mb-4">Agent Communication Network</h3>
      
      <div class="space-y-2">
        {#each $agentCommunications as comm}
          <div class="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
            <div class="flex items-center gap-2">
              <span class="font-medium text-blue-600">{comm.from}</span>
              <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
              </svg>
              <span class="font-medium text-green-600">{comm.to}</span>
            </div>
            <div class="flex-1">
              <span class="text-sm px-2 py-1 rounded text-white {
                comm.type === 'prompt' ? 'bg-blue-500' :
                comm.type === 'result' ? 'bg-green-500' :
                'bg-purple-500'
              }">{comm.type}</span>
              <span class="ml-2 text-sm text-gray-600">{comm.message}</span>
            </div>
            <span class="text-xs text-gray-500">
              {new Date(comm.timestamp).toLocaleTimeString()}
            </span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Workflow Results Summary -->
  {#if $agentResults.length > 0}
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold mb-4">Agent Results Summary</h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {#each $agentResults as result}
          <div class="border border-gray-200 rounded-lg p-4">
            <h4 class="font-medium text-gray-900 mb-2">{result.agent}</h4>
            <div class="text-sm text-gray-600 mb-2">
              {typeof result.result === 'string' ? result.result : JSON.stringify(result.result).substring(0, 100) + '...'}
            </div>
            <div class="flex items-center justify-between text-xs">
              <span class="text-green-600">Completed</span>
              <span class="text-gray-500">Agent Result</span>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Usage Guide -->
  <div class="bg-gray-50 border border-gray-200 rounded-lg p-6">
    <h3 class="text-lg font-semibold mb-4">Demo Features & Usage</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <h4 class="font-medium text-gray-900 mb-2">🤖 Agent Orchestration</h4>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• AutoGen multi-agent framework coordination</li>
          <li>• CrewAI workflow orchestration</li>
          <li>• Claude reasoning and analysis</li>
          <li>• Context7 MCP tool integration</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-gray-900 mb-2">🔄 Self-Prompting Loops</h4>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• Agents analyze their own outputs</li>
          <li>• Generate follow-up prompts automatically</li>
          <li>• Iterative improvement through self-reflection</li>
          <li>• Meta-analysis for workflow optimization</li>
          <li>• Dynamic prompt evolution</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-gray-900 mb-2">⚖️ Legal AI Workflows</h4>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• Evidence analysis and validation</li>
          <li>• Case building and argument development</li>
          <li>• Legal precedent research and correlation</li>
          <li>• Multi-agent legal reasoning</li>
          <li>• Automated recommendation generation</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-gray-900 mb-2">🔧 Context7 MCP Integration</h4>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• Stack analysis and optimization</li>
          <li>• Best practices generation</li>
          <li>• Integration pattern suggestions</li>
          <li>• Library documentation access</li>
          <li>• Development guidance automation</li>
        </ul>
      </div>
    </div>

    <div class="mt-6 p-4 bg-blue-50 rounded-md">
      <h4 class="font-medium text-blue-900 mb-2">🚀 Getting Started</h4>
      <ol class="list-decimal list-inside text-sm text-blue-800 space-y-1">
        <li>Select a workflow (Legal Evidence Analysis recommended for first run)</li>
        <li>Customize the initial prompt for your specific use case</li>
        <li>Enable self-prompting for iterative improvement</li>
        <li>Set max iterations (3-5 recommended for comprehensive analysis)</li>
        <li>Click "Execute Workflow" and watch the real-time orchestration</li>
        <li>Review agent communications and results in the panels below</li>
        <li>Try the Context7 MCP demo for development guidance features</li>
      </ol>
    </div>
  </div>
</div>

<style>
  /* Custom scrollbar for log panels */
  .max-h-96::-webkit-scrollbar {
    width: 6px;
  }
  
  .max-h-96::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
  }
  
  .max-h-96::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  .max-h-96::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
</style>

