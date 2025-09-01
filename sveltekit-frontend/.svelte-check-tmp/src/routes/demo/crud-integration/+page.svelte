<!--
  CRUD Integration Demo - PostgreSQL + Drizzle ORM + MCP + XState
  Demonstrates full stack CRUD operations with real database saves
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { createMachine, interpret } from 'xstate';
  import { caseManagementMachine } from '$lib/machines/caseManagementMachine';
  import type { CaseData, EvidenceData } from '$lib/mcp/cases.mcp';

  // Demo state
  let isLoading = $state(false);
  let cases: CaseData[] = $state([]);
  let selectedCase: CaseData | null = $state(null);
  let error: string | null = $state(null);
  let successMessage: string | null = $state(null);

  // Form data
  let newCaseForm = $state({
    title: '',
    description: '',
    priority: 'medium' as const,
    status: 'open' as const
  });

  let newEvidenceForm = $state({
    content: '',
    evidenceType: 'document' as const,
    source: '',
    tags: ''
  });

  // Demo user ID for testing
  const DEMO_USER_ID = 'demo-user-123';

  // API Helper Functions
  async function loadCases() {
    try {
      isLoading = true;
      error = null;
      
      console.log('🔄 Loading cases from API...');
      const response = await fetch(`/api/v1/cases?userId=${DEMO_USER_ID}&limit=10`);
      const result = await response.json();
      
      if (result.success) {
        cases = result.data.cases;
        console.log(`✅ Loaded ${cases.length} cases`);
      } else {
        error = result.error || 'Failed to load cases';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load cases';
      console.error('❌ Error loading cases:', err);
    } finally {
      isLoading = false;
    }
  }

  async function createCase() {
    try {
      isLoading = true;
      error = null;
      successMessage = null;
      
      const caseData = {
        ...newCaseForm,
        userId: DEMO_USER_ID
      };

      console.log('🆕 Creating case:', caseData.title);
      const response = await fetch('/api/v1/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(caseData)
      });

      const result = await response.json();
      
      if (result.success) {
        successMessage = `Case "${result.data.case.title}" created successfully!`;
        // Reset form
        newCaseForm = { title: '', description: '', priority: 'medium', status: 'open' };
        // Reload cases
        await loadCases();
        console.log(`✅ Case created: ${result.data.caseId}`);
      } else {
        error = result.error || 'Failed to create case';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to create case';
      console.error('❌ Error creating case:', err);
    } finally {
      isLoading = false;
    }
  }

  async function loadCaseDetails(caseId: string) {
    try {
      isLoading = true;
      error = null;
      
      console.log(`🔍 Loading case details: ${caseId}`);
      const response = await fetch(`/api/v1/cases/${caseId}`);
      const result = await response.json();
      
      if (result.success) {
        selectedCase = result.data.case;
        console.log(`✅ Case loaded: ${selectedCase?.title}`);
      } else {
        error = result.error || 'Failed to load case details';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load case details';
      console.error('❌ Error loading case details:', err);
    } finally {
      isLoading = false;
    }
  }

  async function addEvidence() {
    if (!selectedCase) return;
    
    try {
      isLoading = true;
      error = null;
      successMessage = null;
      
      const evidenceData = {
        ...newEvidenceForm,
        tags: newEvidenceForm.tags.split(',').map(t => t.trim()).filter(Boolean)
      };

      console.log('📋 Adding evidence to case:', selectedCase.id);
      const response = await fetch(`/api/v1/cases/${selectedCase.id}/evidence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(evidenceData)
      });

      const result = await response.json();
      
      if (result.success) {
        successMessage = `Evidence added successfully!`;
        // Reset form
        newEvidenceForm = { content: '', evidenceType: 'document', source: '', tags: '' };
        // Reload case details to show new evidence
        await loadCaseDetails(selectedCase.id);
        console.log(`✅ Evidence added: ${result.data.evidenceId}`);
      } else {
        error = result.error || 'Failed to add evidence';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to add evidence';
      console.error('❌ Error adding evidence:', err);
    } finally {
      isLoading = false;
    }
  }

  async function updateCaseStatus(caseId: string, newStatus: string) {
    try {
      isLoading = true;
      error = null;
      
      console.log(`📝 Updating case status: ${caseId} -> ${newStatus}`);
      const response = await fetch(`/api/v1/cases/${caseId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      });

      const result = await response.json();
      
      if (result.success) {
        successMessage = `Case status updated to ${newStatus}`;
        await loadCases(); // Refresh the list
        if (selectedCase && selectedCase.id === caseId) {
          await loadCaseDetails(caseId); // Refresh details if this case is selected
        }
        console.log(`✅ Case status updated`);
      } else {
        error = result.error || 'Failed to update case status';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to update case status';
      console.error('❌ Error updating case status:', err);
    } finally {
      isLoading = false;
    }
  }

  // Load cases on component mount
  onMount(() => {
    loadCases();
  });

  // Clear messages after 5 seconds
  $effect(() => {
    if (successMessage || error) {
      setTimeout(() => {
        successMessage = null;
        error = null;
      }, 5000);
    }
  });
</script>

<svelte:head>
  <title>CRUD Integration Demo - PostgreSQL + Drizzle ORM</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8 text-center">
      <h1 class="text-4xl font-bold text-slate-900 mb-2">
        🚀 CRUD Integration Demo
      </h1>
      <p class="text-slate-600 text-lg">
        PostgreSQL + pgvector + Drizzle ORM + MCP Tools + XState Integration
      </p>
      <div class="mt-4 flex justify-center gap-2">
        <span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
          ✅ Database Connected
        </span>
        <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
          ✅ MCP Tools Active
        </span>
        <span class="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium">
          ✅ REST API Ready
        </span>
      </div>
    </div>

    <!-- Status Messages -->
    {#if error}
      <div class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div class="flex items-center">
          <span class="text-red-500 mr-2">❌</span>
          <span class="text-red-800">{error}</span>
        </div>
      </div>
    {/if}

    {#if successMessage}
      <div class="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
        <div class="flex items-center">
          <span class="text-green-500 mr-2">✅</span>
          <span class="text-green-800">{successMessage}</span>
        </div>
      </div>
    {/if}

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- Create New Case Form -->
      <div class="bg-white rounded-xl shadow-lg p-6">
        <h2 class="text-2xl font-semibold text-slate-900 mb-4 flex items-center">
          <span class="mr-2">🆕</span>
          Create New Case
        </h2>
        
        <form on:submit|preventDefault={createCase}>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-slate-700 mb-1">
                Case Title *
              </label>
              <input 
                bind:value={newCaseForm.title}
                type="text" 
                required
                class="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter case title..."
              />
            </div>
            
            <div>
              <label class="block text-sm font-medium text-slate-700 mb-1">
                Description
              </label>
              <textarea 
                bind:value={newCaseForm.description}
                rows="3"
                class="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter case description..."
              ></textarea>
            </div>
            
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium text-slate-700 mb-1">
                  Priority
                </label>
                <select 
                  bind:value={newCaseForm.priority}
                  class="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              
              <div>
                <label class="block text-sm font-medium text-slate-700 mb-1">
                  Status
                </label>
                <select 
                  bind:value={newCaseForm.status}
                  class="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="open">Open</option>
                  <option value="active">Active</option>
                  <option value="closed">Closed</option>
                </select>
              </div>
            </div>
          </div>
          
          <button 
            type="submit" 
            disabled={isLoading || !newCaseForm.title.trim()}
            class="mt-6 w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {#if isLoading}
              <span class="mr-2">⏳</span> Creating Case...
            {:else}
              <span class="mr-2">💾</span> Save to Database
            {/if}
          </button>
        </form>
      </div>

      <!-- Cases List -->
      <div class="bg-white rounded-xl shadow-lg p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-2xl font-semibold text-slate-900 flex items-center">
            <span class="mr-2">📋</span>
            Active Cases ({cases.length})
          </h2>
          <button 
            click={loadCases}
            disabled={isLoading}
            class="px-3 py-1 bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 disabled:opacity-50"
          >
            🔄 Refresh
          </button>
        </div>
        
        {#if isLoading && cases.length === 0}
          <div class="text-center py-8 text-slate-500">
            <span class="text-2xl">⏳</span>
            <p>Loading cases from database...</p>
          </div>
        {:else if cases.length === 0}
          <div class="text-center py-8 text-slate-500">
            <span class="text-4xl">📝</span>
            <p>No cases found. Create your first case above!</p>
          </div>
        {:else}
          <div class="space-y-3 max-h-96 overflow-y-auto">
            {#each cases as case (case.id)}
              <div class="border border-slate-200 rounded-lg p-4 hover:bg-slate-50 transition-colors">
                <div class="flex items-start justify-between">
                  <div class="flex-1">
                    <h3 class="font-medium text-slate-900">{case.title}</h3>
                    {#if case.description}
                      <p class="text-sm text-slate-600 mt-1">{case.description.substring(0, 100)}{case.description.length > 100 ? '...' : ''}</p>
                    {/if}
                    <div class="flex items-center gap-2 mt-2">
                      <span class="px-2 py-1 bg-{case.priority === 'high' ? 'red' : case.priority === 'medium' ? 'yellow' : 'green'}-100 text-{case.priority === 'high' ? 'red' : case.priority === 'medium' ? 'yellow' : 'green'}-800 rounded text-xs">
                        {case.priority}
                      </span>
                      <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                        {case.status}
                      </span>
                    </div>
                  </div>
                  <div class="flex gap-2 ml-4">
                    <button 
                      click={() => loadCaseDetails(case.id)}
                      class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs hover:bg-blue-200"
                    >
                      👁️ View
                    </button>
                    <select 
                      value={case.status}
                      change={(e) => updateCaseStatus(case.id, (e.target as HTMLSelectElement).value)}
                      class="text-xs border border-slate-300 rounded px-1 py-1"
                    >
                      <option value="open">Open</option>
                      <option value="active">Active</option>
                      <option value="closed">Closed</option>
                      <option value="archived">Archive</option>
                    </select>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Case Details & Evidence -->
    {#if selectedCase}
      <div class="mt-8 bg-white rounded-xl shadow-lg p-6">
        <h2 class="text-2xl font-semibold text-slate-900 mb-4 flex items-center">
          <span class="mr-2">🔍</span>
          Case Details: {selectedCase.title}
        </h2>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <!-- Case Information -->
          <div>
            <h3 class="text-lg font-medium text-slate-900 mb-3">Case Information</h3>
            <div class="space-y-2 text-sm">
              <p><strong>Case ID:</strong> {selectedCase.id}</p>
              <p><strong>Title:</strong> {selectedCase.title}</p>
              <p><strong>Description:</strong> {selectedCase.description || 'No description'}</p>
              <p><strong>Priority:</strong> {selectedCase.priority}</p>
              <p><strong>Status:</strong> {selectedCase.status}</p>
              <p><strong>Created:</strong> {new Date(selectedCase.createdAt || Date.now()).toLocaleString()}</p>
            </div>
          </div>
          
          <!-- Add Evidence Form -->
          <div>
            <h3 class="text-lg font-medium text-slate-900 mb-3">Add Evidence</h3>
            <form on:submit|preventDefault={addEvidence}>
              <div class="space-y-3">
                <div>
                  <label class="block text-xs font-medium text-slate-700 mb-1">Evidence Content *</label>
                  <textarea 
                    bind:value={newEvidenceForm.content}
                    rows="3"
                    required
                    class="w-full px-2 py-1 text-sm border border-slate-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                    placeholder="Enter evidence details..."
                  ></textarea>
                </div>
                
                <div class="grid grid-cols-2 gap-2">
                  <div>
                    <label class="block text-xs font-medium text-slate-700 mb-1">Type</label>
                    <select 
                      bind:value={newEvidenceForm.evidenceType}
                      class="w-full px-2 py-1 text-sm border border-slate-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="document">Document</option>
                      <option value="testimony">Testimony</option>
                      <option value="physical">Physical</option>
                      <option value="digital">Digital</option>
                    </select>
                  </div>
                  
                  <div>
                    <label class="block text-xs font-medium text-slate-700 mb-1">Source</label>
                    <input 
                      bind:value={newEvidenceForm.source}
                      type="text" 
                      class="w-full px-2 py-1 text-sm border border-slate-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                      placeholder="Evidence source..."
                    />
                  </div>
                </div>
                
                <div>
                  <label class="block text-xs font-medium text-slate-700 mb-1">Tags (comma-separated)</label>
                  <input 
                    bind:value={newEvidenceForm.tags}
                    type="text" 
                    class="w-full px-2 py-1 text-sm border border-slate-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                    placeholder="tag1, tag2, tag3..."
                  />
                </div>
              </div>
              
              <button 
                type="submit" 
                disabled={isLoading || !newEvidenceForm.content.trim()}
                class="mt-4 w-full bg-green-600 text-white py-1 px-3 text-sm rounded-md hover:bg-green-700 disabled:opacity-50"
              >
                {#if isLoading}
                  ⏳ Adding Evidence...
                {:else}
                  📋 Add Evidence
                {/if}
              </button>
            </form>
          </div>
        </div>
        
        <!-- Evidence List -->
        {#if selectedCase.evidence && selectedCase.evidence.length > 0}
          <div class="mt-6">
            <h3 class="text-lg font-medium text-slate-900 mb-3">
              Evidence ({selectedCase.evidence.length} items)
            </h3>
            <div class="space-y-2">
              {#each selectedCase.evidence as evidence (evidence.id)}
                <div class="border border-slate-200 rounded p-3">
                  <div class="flex items-start justify-between">
                    <div>
                      <p class="text-sm text-slate-900">{evidence.content}</p>
                      <div class="flex items-center gap-2 mt-1">
                        <span class="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs">
                          {evidence.evidenceType}
                        </span>
                        {#if evidence.source}
                          <span class="text-xs text-slate-500">Source: {evidence.source}</span>
                        {/if}
                      </div>
                    </div>
                    <span class="text-xs text-slate-500 ml-4">
                      {new Date(evidence.createdAt || Date.now()).toLocaleString()}
                    </span>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* Custom scrollbar for evidence list */
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