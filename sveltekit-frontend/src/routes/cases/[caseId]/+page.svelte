<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { useMachine } from '@xstate/svelte';
  import { legalCaseMachine, legalCaseSelectors } from '$lib/state/legal-case-machine.js';
  import type { LegalCaseContext } from '$lib/state/legal-case-machine.js';
  import Button from '$lib/components/ui/Button.svelte';
  import Card from '$lib/components/ui/Card.svelte';

  // Extract caseId from route params
  const caseId = $derived($page.params.caseId);
  
  // Initialize XState machine with route-based caseId
  const { state, send } = useMachine(legalCaseMachine, {
    context: {
      ...legalCaseMachine.context,
      caseId: caseId
    }
  });

  // Reactive selectors using the hardened machine
  const isLoading = $derived(legalCaseSelectors.isLoading($state));
  const hasError = $derived(legalCaseSelectors.hasError($state));
  const currentCase = $derived(legalCaseSelectors.getCurrentCase($state));
  const evidence = $derived(legalCaseSelectors.getEvidence($state));
  const aiSummary = $derived(legalCaseSelectors.getAISummary($state));
  const similarCases = $derived(legalCaseSelectors.getSimilarCases($state));
  const activeTab = $derived(legalCaseSelectors.getActiveTab($state));
  let workflowStage = $state(legalCaseSelectors.getWorkflowStage($state));
  const nextActions = $derived(legalCaseSelectors.getNextActions($state));
  const canStartAIAnalysis = $derived(legalCaseSelectors.canStartAIAnalysis($state));
  const stats = $derived(legalCaseSelectors.getStats($state));
  const currentState = $derived($state.value);

  // Auto-load case when route changes
  $effect(() => {
    if (caseId && caseId !== $state.context.caseId) {
      send({ type: 'LOAD_CASE', caseId });
    }
  });

  // Interval-based re-validation for AI progress
  let revalidationInterval: NodeJS.Timeout | null = null;

  onMount(() => {
    // Set up auto-revalidation for AI analysis progress
    if (currentCase && aiSummary) {
      revalidationInterval = setInterval(() => {
        if (legalCaseSelectors.isInState('aiAnalysis.analyzing')($state)) {
          // Simulate progress updates during AI analysis
          const currentProgress = $state.context.aiAnalysisProgress;
          if (currentProgress < 100) {
            send({ type: 'AI_ANALYSIS_PROGRESS', progress: Math.min(currentProgress + 10, 95) });
          }
        }
      }, 1000);
    }

    return () => {
      if (revalidationInterval) {
        clearInterval(revalidationInterval);
      }
    };
  });

  // Event handlers
  function handleAddEvidence(files: FileList) {
    if (files.length > 0) {
      const fileArray = Array.from(files);
      send({ type: 'ADD_EVIDENCE', files: fileArray });
    }
  }

  function handleStartAIAnalysis() {
    send({ type: 'START_AI_ANALYSIS' });
  }

  function handleFindSimilarCases() {
    send({ type: 'FIND_SIMILAR_CASES' });
  }

  function handleTabSwitch(tab: LegalCaseContext['activeTab']) {
    send({ type: 'SWITCH_TAB', tab });
  }

  function handleWorkflowStageChange(stage: LegalCaseContext['workflowStage']) {
    send({ type: 'SET_WORKFLOW_STAGE', stage });
  }

  function handleRetry() {
    send({ type: 'RETRY' });
  }

  function handleDismissError() {
    send({ type: 'DISMISS_ERROR' });
  }

  function handleRefresh() {
    send({ type: 'REFRESH' });
  }

  // File upload handler
  let fileInput: HTMLInputElement;

  function triggerFileUpload() {
    fileInput?.click();
  }

  function onFileChange(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files) {
      handleAddEvidence(target.files);
    }
  }
</script>

<svelte:head>
  <title>{currentCase?.title || `Case ${caseId}`} - Legal AI Platform</title>
  <meta name="description" content="AI-powered legal case management with XState workflow orchestration" />
</svelte:head>

<div class="case-detail-page min-h-screen bg-gray-50">
  
  <!-- Breadcrumb Navigation -->
  <nav class="bg-white shadow-sm border-b">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center space-x-4 py-4">
        <a href="/cases" class="text-sm text-gray-500 hover:text-gray-700">Cases</a>
        <svg class="w-4 h-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 111.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
        </svg>
        <span class="text-sm font-medium text-gray-900">
          {currentCase?.title || `Case ${caseId}`}
        </span>
      </div>
    </div>
  </nav>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    
    <!-- Error State -->
    {#if hasError}
      <Card class="mb-6 border-red-200 bg-red-50">
        <div class="p-6">
          <div class="flex items-start">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3 flex-1">
              <h3 class="text-sm font-medium text-red-800">Case Loading Error</h3>
              <p class="mt-2 text-sm text-red-700">{$state.context.error}</p>
              <div class="mt-4 flex space-x-3">
                <Button size="sm" variant="outline" on:click={handleRetry}>
                  Retry Loading
                </Button>
                <Button size="sm" variant="ghost" on:click={handleDismissError}>
                  Dismiss
                </Button>
              </div>
            </div>
          </div>
        </div>
      </Card>
    {/if}

    <!-- Loading State -->
    {#if isLoading && !currentCase}
      <Card>
        <div class="p-12">
          <div class="flex flex-col items-center">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
            <h3 class="text-lg font-medium text-gray-900">Loading Case</h3>
            <p class="text-sm text-gray-500">Please wait while we fetch case details...</p>
          </div>
        </div>
      </Card>
    {/if}

    <!-- Case Content -->
    {#if currentCase}
      <div class="space-y-6">
        
        <!-- Case Header with Actions -->
        <Card>
          <div class="p-6">
            <div class="lg:flex lg:items-center lg:justify-between">
              <div class="flex-1 min-w-0">
                <div class="flex items-center">
                  <h1 class="text-3xl font-bold text-gray-900 truncate">
                    {currentCase.title}
                  </h1>
                  {#if isLoading}
                    <div class="ml-3 animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                  {/if}
                </div>
                <div class="mt-1 flex flex-col sm:flex-row sm:flex-wrap sm:mt-0 sm:space-x-6">
                  <div class="mt-2 flex items-center text-sm text-gray-500">
                    <svg class="flex-shrink-0 mr-1.5 h-4 w-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clip-rule="evenodd" />
                    </svg>
                    Case #{currentCase.caseNumber}
                  </div>
                  <div class="mt-2 flex items-center text-sm text-gray-500">
                    <svg class="flex-shrink-0 mr-1.5 h-4 w-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" clip-rule="evenodd" />
                    </svg>
                    {workflowStage}
                  </div>
                  <div class="mt-2 flex items-center text-sm text-gray-500">
                    <svg class="flex-shrink-0 mr-1.5 h-4 w-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd" />
                    </svg>
                    State: {currentState}
                  </div>
                </div>
              </div>
              <div class="mt-5 flex lg:mt-0 lg:ml-4">
                <span class="sm:ml-3">
                  <Button on:click={handleRefresh} variant="outline">
                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Refresh
                  </Button>
                </span>
                <span class="ml-3">
                  <select 
                    bind:value={workflowStage}
                    change={(e) => handleWorkflowStageChange(e.target.value)}
                    class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                  >
                    <option value="investigation">Investigation</option>
                    <option value="analysis">Analysis</option>
                    <option value="preparation">Preparation</option>
                    <option value="review">Review</option>
                    <option value="closed">Closed</option>
                  </select>
                </span>
              </div>
            </div>
            
            {#if currentCase.description}
              <div class="mt-6">
                <p class="text-gray-700">{currentCase.description}</p>
              </div>
            {/if}

            <!-- Stats Dashboard -->
            <div class="mt-8 border-t border-gray-200 pt-6">
              <dl class="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
                <div class="border-l-4 border-blue-500 pl-4">
                  <dt class="text-sm font-medium text-gray-500">Evidence Items</dt>
                  <dd class="text-2xl font-semibold text-gray-900">{stats.totalEvidence}</dd>
                </div>
                <div class="border-l-4 border-green-500 pl-4">
                  <dt class="text-sm font-medium text-gray-500">Processed</dt>
                  <dd class="text-2xl font-semibold text-gray-900">{stats.processedEvidence}</dd>
                </div>
                <div class="border-l-4 border-purple-500 pl-4">
                  <dt class="text-sm font-medium text-gray-500">AI Confidence</dt>
                  <dd class="text-2xl font-semibold text-gray-900">{stats.averageConfidence}%</dd>
                </div>
                <div class="border-l-4 border-orange-500 pl-4">
                  <dt class="text-sm font-medium text-gray-500">Processing Time</dt>
                  <dd class="text-2xl font-semibold text-gray-900">{stats.processingTime}ms</dd>
                </div>
              </dl>
            </div>
          </div>
        </Card>

        <!-- Navigation Tabs -->
        <div class="border-b border-gray-200">
          <nav class="flex space-x-8">
            {#each [
              { id: 'overview', label: 'Overview', icon: '📋' },
              { id: 'evidence', label: 'Evidence', icon: '📁', badge: evidence.length },
              { id: 'analysis', label: 'AI Analysis', icon: '🤖', badge: aiSummary ? '✓' : null },
              { id: 'search', label: 'Search', icon: '🔍' }
            ] as tab}
              <button
                class="group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm {activeTab === tab.id ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}"
                click={() => handleTabSwitch(tab.id)}
              >
                <span class="mr-2">{tab.icon}</span>
                {tab.label}
                {#if tab.badge}
                  <span class="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {activeTab === tab.id ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'}">
                    {tab.badge}
                  </span>
                {/if}
              </button>
            {/each}
          </nav>
        </div>

        <!-- Tab Content -->
        <div class="tab-content">
          
          <!-- Overview Tab -->
          {#if activeTab === 'overview'}
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div class="lg:col-span-2">
                <Card>
                  <div class="p-6">
                    <h3 class="text-lg font-medium text-gray-900 mb-4">Next Actions</h3>
                    <div class="space-y-3">
                      {#each nextActions as action, index}
                        <div class="flex items-center">
                          <div class="flex-shrink-0">
                            <span class="inline-flex items-center justify-center h-8 w-8 rounded-full bg-blue-500 text-white text-sm font-medium">
                              {index + 1}
                            </span>
                          </div>
                          <div class="ml-4">
                            <p class="text-sm font-medium text-gray-900">{action}</p>
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                </Card>
              </div>
              
              <div class="space-y-6">
                <!-- Quick Actions -->
                <Card>
                  <div class="p-6">
                    <h3 class="text-lg font-medium text-gray-900 mb-4">Quick Actions</h3>
                    <div class="space-y-3">
                      <Button 
                        on:click={triggerFileUpload}
                        class="w-full justify-center"
                        variant="outline"
                      >
                        📎 Upload Evidence
                      </Button>
                      <Button 
                        on:click={handleStartAIAnalysis}
                        disabled={!canStartAIAnalysis}
                        class="w-full justify-center"
                      >
                        🤖 Start AI Analysis
                      </Button>
                      <Button 
                        on:click={handleFindSimilarCases}
                        class="w-full justify-center"
                        variant="outline"
                      >
                        🔍 Find Similar Cases
                      </Button>
                    </div>
                  </div>
                </Card>
              </div>
            </div>
          {/if}

          <!-- Evidence Tab -->
          {#if activeTab === 'evidence'}
            <div class="space-y-6">
              <!-- Evidence Upload -->
              <Card>
                <div class="p-6">
                  <h3 class="text-lg font-medium text-gray-900 mb-4">Upload Evidence</h3>
                  <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
                    <input
                      type="file"
                      multiple
                      bind:this={fileInput}
                      change={onFileChange}
                      class="hidden"
                    />
                    <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                      <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                    <p class="mt-2 text-sm text-gray-600">
                      <Button on:click={triggerFileUpload} variant="outline">
                        Choose files
                      </Button>
                      or drag and drop
                    </p>
                    <p class="text-xs text-gray-500">PNG, JPG, PDF, DOCX up to 10MB each</p>
                  </div>
                </div>
              </Card>

              <!-- Evidence List -->
              {#if evidence.length > 0}
                <Card>
                  <div class="p-6">
                    <h3 class="text-lg font-medium text-gray-900 mb-4">Evidence Items ({evidence.length})</h3>
                    <div class="space-y-4">
                      {#each evidence as item}
                        <div class="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow">
                          <div class="flex items-start justify-between">
                            <div class="flex-1">
                              <h4 class="text-sm font-medium text-gray-900">{item.title}</h4>
                              <p class="text-sm text-gray-500">{item.type}</p>
                              {#if item.aiSummary}
                                <div class="mt-2 p-3 bg-blue-50 rounded-md">
                                  <p class="text-sm text-blue-800">{item.aiSummary}</p>
                                </div>
                              {/if}
                            </div>
                            <div class="ml-4 flex-shrink-0 flex space-x-2">
                              <Button size="sm" variant="outline">View</Button>
                              <Button size="sm" on:click={() => send({ type: 'SELECT_EVIDENCE', evidence: item })}>
                                Select
                              </Button>
                            </div>
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                </Card>
              {:else}
                <Card>
                  <div class="p-12 text-center">
                    <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <h3 class="mt-2 text-sm font-medium text-gray-900">No evidence uploaded</h3>
                    <p class="mt-1 text-sm text-gray-500">Get started by uploading your first piece of evidence.</p>
                    <div class="mt-6">
                      <Button on:click={triggerFileUpload}>
                        Upload Evidence
                      </Button>
                    </div>
                  </div>
                </Card>
              {/if}
            </div>
          {/if}

          <!-- Analysis Tab -->
          {#if activeTab === 'analysis'}
            <div class="space-y-6">
              <Card>
                <div class="p-6">
                  <div class="flex items-center justify-between mb-6">
                    <h3 class="text-lg font-medium text-gray-900">AI Analysis</h3>
                    {#if legalCaseSelectors.isInState('aiAnalysis.analyzing')($state)}
                      <div class="flex items-center text-sm text-blue-600">
                        <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                        Analyzing... ({$state.context.aiAnalysisProgress}%)
                      </div>
                    {/if}
                  </div>

                  <div class="flex space-x-4 mb-6">
                    <Button 
                      on:click={handleStartAIAnalysis}
                      disabled={!canStartAIAnalysis || legalCaseSelectors.isInState('aiAnalysis.analyzing')($state)}
                    >
                      {legalCaseSelectors.isInState('aiAnalysis.analyzing')($state) ? 'Analyzing...' : 'Start AI Analysis'}
                    </Button>
                    <Button 
                      variant="outline"
                      on:click={handleFindSimilarCases}
                      disabled={isLoading}
                    >
                      Find Similar Cases
                    </Button>
                  </div>

                  {#if aiSummary}
                    <div class="border border-gray-200 rounded-lg p-4 bg-gray-50">
                      <h4 class="font-medium text-gray-900 mb-2">AI Summary</h4>
                      <p class="text-gray-700">{aiSummary}</p>
                    </div>
                  {/if}

                  {#if similarCases.length > 0}
                    <div class="mt-6">
                      <h4 class="font-medium text-gray-900 mb-4">Similar Cases</h4>
                      <div class="space-y-3">
                        {#each similarCases as similarCase}
                          <div class="border border-gray-200 rounded-lg p-4 flex justify-between items-center hover:shadow-sm transition-shadow">
                            <div>
                              <h5 class="font-medium text-gray-900">{similarCase.title}</h5>
                              <p class="text-sm text-gray-500">Similarity: {similarCase.similarity}%</p>
                            </div>
                            <Button size="sm" variant="outline">
                              View Case
                            </Button>
                          </div>
                        {/each}
                      </div>
                    </div>
                  {/if}
                </div>
              </Card>
            </div>
          {/if}

          <!-- Search Tab -->
          {#if activeTab === 'search'}
            <Card>
              <div class="p-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Search Related Cases</h3>
                <div class="flex space-x-4">
                  <input
                    type="text"
                    placeholder="Search for similar cases, precedents, or legal documents..."
                    class="flex-1 min-w-0 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  />
                  <Button>Search</Button>
                </div>
              </div>
            </Card>
          {/if}
        </div>
      </div>
    {/if}

    <!-- Debug Panel (development only) -->
    {#if import.meta.env.DEV}
      <Card class="mt-8 bg-gray-900 text-gray-100">
        <div class="p-4">
          <h4 class="text-sm font-semibold mb-3 text-white">🔧 XState Debug Panel</h4>
          <div class="grid grid-cols-2 gap-4 text-xs font-mono">
            <div>
              <p class="text-gray-400">Route Params:</p>
              <p class="text-green-400">caseId: {caseId}</p>
              <p class="text-gray-400 mt-2">Machine State:</p>
              <p class="text-blue-400">{currentState}</p>
            </div>
            <div>
              <p class="text-gray-400">Status:</p>
              <p class="text-yellow-400">Loading: {isLoading}</p>
              <p class="text-red-400">Error: {hasError}</p>
              <p class="text-green-400">Case Loaded: {!!currentCase}</p>
              <p class="text-purple-400">Evidence: {evidence.length}</p>
              <p class="text-orange-400">AI Summary: {aiSummary ? 'Available' : 'None'}</p>
            </div>
          </div>
        </div>
      </Card>
    {/if}
  </div>
</div>

<style>
  .animate-spin {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>