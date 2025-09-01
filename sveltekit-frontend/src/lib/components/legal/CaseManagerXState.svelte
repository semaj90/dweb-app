/// <reference types="vite/client" />
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { useMachine } from '@xstate/svelte';
  import { legalCaseMachine, legalCaseSelectors } from '$lib/state/legal-case-machine.js';
  import type { LegalCaseContext, LegalCaseEvents } from '$lib/state/legal-case-machine.js';
  import Button from '$lib/components/ui/Button.svelte';
  import Card from '$lib/components/ui/Card.svelte';

  // Get caseId from route params
  let caseId = $state<string | null>(null);
  
  // Initialize XState machine with proper service binding
  const { state, send } = useMachine(legalCaseMachine, {
    context: {
      ...legalCaseMachine.context,
      caseId: caseId
    }
  });

  // Reactive selectors using the hardened machine
  let isLoading = $derived(legalCaseSelectors.isLoading($state));
  let hasError = $derived(legalCaseSelectors.hasError($state));
  let currentCase = $derived(legalCaseSelectors.getCurrentCase($state));
  let evidence = $derived(legalCaseSelectors.getEvidence($state));
  let aiSummary = $derived(legalCaseSelectors.getAISummary($state));
  let similarCases = $derived(legalCaseSelectors.getSimilarCases($state));
  let activeTab = $derived(legalCaseSelectors.getActiveTab($state));
  let workflowStage = $derived(legalCaseSelectors.getWorkflowStage($state));
  let nextActions = $derived(legalCaseSelectors.getNextActions($state));
  let canStartAIAnalysis = $derived(legalCaseSelectors.canStartAIAnalysis($state));
  let stats = $derived(legalCaseSelectors.getStats($state));

  // Form state
  let newCaseTitle = $state('');
  let newCaseDescription = $state('');
  let newCaseNumber = $state('');

  // Handle route changes and load case
  $effect(() => {
    const routeCaseId = $page.params.caseId;
    if (routeCaseId && routeCaseId !== caseId) {
      caseId = routeCaseId;
      send({ type: 'LOAD_CASE', caseId: routeCaseId });
    }
  });

  // Machine event handlers using proper XState patterns
  function handleCreateCase() {
    if (!newCaseTitle || !newCaseDescription || !newCaseNumber) {
      return;
    }

    // Update form data first, then create case
    send({
      type: 'UPDATE_CASE_FORM',
      data: {
        title: newCaseTitle,
        description: newCaseDescription,
        caseNumber: newCaseNumber,
        status: 'active'
      }
    });

    send({ type: 'CREATE_CASE', caseData: {
      title: newCaseTitle,
      description: newCaseDescription,
      caseNumber: newCaseNumber,
      status: 'active'
    }});

    // Clear form
    newCaseTitle = '';
    newCaseDescription = '';
    newCaseNumber = '';
  }

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

  // File upload handler
let fileInput = $state<HTMLInputElement;

  function triggerFileUpload() {
    fileInput?.click();
  }

  function onFileChange(event: Event) {
    const target >(event.target as HTMLInputElement);
    if (target.files) {
      handleAddEvidence(target.files);
    }
  }
</script>

<!-- Legal Case Manager Component with XState -->
<div class="case-manager-xstate p-6 max-w-7xl mx-auto">
  
  <!-- Error State -->
  {#if hasError}
    <Card class="mb-6 border-red-200 bg-red-50">
      <div class="p-4">
        <h3 class="text-lg font-semibold text-red-800 mb-2">Error</h3>
        <p class="text-red-600 mb-4">{$state.context.error}</p>
        <div class="flex gap-2">
          <Button variant="outline" size="sm" on:on:click={handleRetry}>
            Retry
          </Button>
          <Button variant="ghost" size="sm" on:on:click={handleDismissError}>
            Dismiss
          </Button>
        </div>
      </div>
    </Card>
  {/if}

  <!-- Loading State -->
  {#if isLoading}
    <div class="flex items-center justify-center py-12">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      <span class="ml-3 text-gray-600">Loading...</span>
    </div>
  {/if}

  <!-- Case Creation Form (when no case is loaded) -->
  {#if !currentCase && !isLoading}
    <Card class="mb-6">
      <div class="p-6">
        <h2 class="text-2xl font-bold mb-4">Create New Case</h2>
        <div class="space-y-4">
          <div>
            <label for="case-title" class="block text-sm font-medium mb-2">Case Title</label>
            <input
              id="case-title"
              type="text"
              bind:value={newCaseTitle}
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter case title..."
            />
          </div>
          
          <div>
            <label for="case-number" class="block text-sm font-medium mb-2">Case Number</label>
            <input
              id="case-number"
              type="text"
              bind:value={newCaseNumber}
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter case number..."
            />
          </div>
          
          <div>
            <label for="case-description" class="block text-sm font-medium mb-2">Description</label>
            <textarea
              id="case-description"
              bind:value={newCaseDescription}
              rows="4"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Describe the case..."
            ></textarea>
          </div>
          
          <Button on:on:click={handleCreateCase} class="w-full">
            Create Case
          </Button>
        </div>
      </div>
    </Card>
  {/if}

  <!-- Case Management Interface (when case is loaded) -->
  {#if currentCase && !isLoading}
    <div class="space-y-6">
      
      <!-- Case Header -->
      <Card>
        <div class="p-6">
          <div class="flex justify-between items-start mb-4">
            <div>
              <h1 class="text-3xl font-bold text-gray-900">{currentCase.title}</h1>
              <p class="text-sm text-gray-500">Case #{currentCase.caseNumber}</p>
            </div>
            <div class="flex items-center gap-2">
              <select 
                bind:value={workflowStage}
                change={(e) => handleWorkflowStageChange(e.target.value)}
                class="px-3 py-1 border border-gray-300 rounded-md text-sm"
              >
                <option value="investigation">Investigation</option>
                <option value="analysis">Analysis</option>
                <option value="preparation">Preparation</option>
                <option value="review">Review</option>
                <option value="closed">Closed</option>
              </select>
            </div>
          </div>
          
          <p class="text-gray-700 mb-4">{currentCase.description}</p>
          
          <!-- Stats -->
          <div class="grid grid-cols-4 gap-4">
            <div class="text-center">
              <div class="text-2xl font-bold text-blue-600">{stats.totalEvidence}</div>
              <div class="text-sm text-gray-500">Evidence Items</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-green-600">{stats.processedEvidence}</div>
              <div class="text-sm text-gray-500">Processed</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-purple-600">{stats.averageConfidence}%</div>
              <div class="text-sm text-gray-500">Avg Confidence</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-orange-600">{stats.processingTime}ms</div>
              <div class="text-sm text-gray-500">Processing Time</div>
            </div>
          </div>
        </div>
      </Card>

      <!-- Navigation Tabs -->
      <div class="border-b border-gray-200">
        <nav class="flex space-x-8">
          {#each ['overview', 'evidence', 'analysis', 'search'] as tab}
            <button
              class="py-2 px-1 border-b-2 font-medium text-sm {activeTab === tab ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}"
              on:onclick={() => handleTabSwitch(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          {/each}
        </nav>
      </div>

      <!-- Tab Content -->
      <div class="tab-content">
        
        <!-- Overview Tab -->
        {#if activeTab === 'overview'}
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold mb-4">Next Actions</h3>
              <ul class="space-y-2">
                {#each nextActions as action}
                  <li class="flex items-center">
                    <span class="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
                    {action}
                  </li>
                {/each}
              </ul>
            </div>
          </Card>
        {/if}

        <!-- Evidence Tab -->
        {#if activeTab === 'evidence'}
          <div class="space-y-4">
            <!-- Evidence Upload -->
            <Card>
              <div class="p-6">
                <h3 class="text-lg font-semibold mb-4">Upload Evidence</h3>
                <input
                  type="file"
                  multiple
                  bind:this={fileInput}
                  change={onFileChange}
                  class="hidden"
                />
                <Button on:on:click={triggerFileUpload}>
                  Choose Files
                </Button>
              </div>
            </Card>

            <!-- Evidence List -->
            {#if evidence.length > 0}
              <Card>
                <div class="p-6">
                  <h3 class="text-lg font-semibold mb-4">Evidence Items</h3>
                  <div class="space-y-3">
                    {#each evidence as item}
                      <div class="border border-gray-200 rounded-lg p-4">
                        <div class="flex justify-between items-start">
                          <div>
                            <h4 class="font-medium">{item.title}</h4>
                            <p class="text-sm text-gray-500">{item.type}</p>
                          </div>
                          <div class="flex gap-2">
                            <Button size="sm" variant="outline">View</Button>
                            <Button size="sm" on:on:click={() => send({ type: 'SELECT_EVIDENCE', evidence: item })}>
                              Select
                            </Button>
                          </div>
                        </div>
                        {#if item.aiSummary}
                          <div class="mt-3 p-3 bg-blue-50 rounded-md">
                            <p class="text-sm">{item.aiSummary}</p>
                          </div>
                        {/if}
                      </div>
                    {/each}
                  </div>
                </div>
              </Card>
            {/if}
          </div>
        {/if}

        <!-- Analysis Tab -->
        {#if activeTab === 'analysis'}
          <div class="space-y-4">
            <Card>
              <div class="p-6">
                <h3 class="text-lg font-semibold mb-4">AI Analysis</h3>
                <div class="flex gap-3 mb-4">
                  <Button 
                    on:on:click={handleStartAIAnalysis}
                    disabled={!canStartAIAnalysis}
                  >
                    Start AI Analysis
                  </Button>
                  <Button 
                    variant="outline"
                    on:on:click={handleFindSimilarCases}
                  >
                    Find Similar Cases
                  </Button>
                </div>

                {#if aiSummary}
                  <div class="border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium mb-2">AI Summary</h4>
                    <p class="text-gray-700">{aiSummary}</p>
                  </div>
                {/if}

                {#if similarCases.length > 0}
                  <div class="mt-6">
                    <h4 class="font-medium mb-3">Similar Cases</h4>
                    <div class="space-y-2">
                      {#each similarCases as similarCase}
                        <div class="border border-gray-200 rounded-lg p-3 flex justify-between items-center">
                          <div>
                            <h5 class="font-medium">{similarCase.title}</h5>
                            <p class="text-sm text-gray-500">Similarity: {similarCase.similarity}%</p>
                          </div>
                          <Button size="sm" variant="outline">View</Button>
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
              <h3 class="text-lg font-semibold mb-4">Search Cases</h3>
              <div class="flex gap-3">
                <input
                  type="text"
                  placeholder="Enter search query..."
                  class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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
    <Card class="mt-8 bg-gray-50">
      <div class="p-4">
        <h4 class="text-sm font-semibold mb-2">XState Debug</h4>
        <div class="text-xs space-y-1">
          <p>Current State: <span class="font-mono">{$state.value}</span></p>
          <p>Loading: {isLoading}</p>
          <p>Error: {hasError}</p>
          <p>Evidence Count: {evidence.length}</p>
          <p>AI Summary: {aiSummary ? 'Available' : 'None'}</p>
        </div>
      </div>
    </Card>
  {/if}
</div>

<style>
  .case-manager-xstate {
    min-height: 100vh;
  }
  
  .animate-spin {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>