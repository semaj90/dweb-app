<!-- Example: Modular Case Management with API Integration -->
<script lang="ts">
  import { ModularDialog } from '../modular-dialog';
  import { ModularCommand } from '../modular-command';
  import { Button } from 'bits-ui';
  import { Search, Plus, FileText, Users, Calendar } from 'lucide-svelte';
  import { reactiveApiClient } from '$lib/services/api-client';
  import type { Case, Evidence } from '$lib/types/api';
  
  // Component state
  let showCaseDialog = $state(false);
  let showCommandPalette = $state(false);
  let selectedCaseId = $state<string>('');
  let selectedCase = $state<Case | null>(null);
  let caseEvidence = $state<Evidence[]>([]);

  // Search and data management
  function handleCaseSelect(item: Case, type: string) {
    if (type === 'cases') {
      selectedCaseId = item.id;
      selectedCase = item;
      showCaseDialog = true;
    }
  }

  function handleCommandSearch(item: any, type: string) {
    console.log(`Selected ${type}:`, item);
    
    switch (type) {
      case 'cases':
        handleCaseSelect(item, type);
        break;
      case 'evidence':
        // Navigate to evidence view
        break;
      case 'documents':
        // Open document viewer
        break;
      case 'people':
        // Show person profile
        break;
    }
  }

  // Keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    if (event.metaKey || event.ctrlKey) {
      if (event.key === 'k') {
        event.preventDefault();
        showCommandPalette = true;
      }
    }
  }

  // Data refresh handlers
  async function refreshCaseData() {
    if (selectedCaseId) {
      const updatedCase = await reactiveApiClient.fetchCase(selectedCaseId, false);
      if (updatedCase) {
        selectedCase = updatedCase;
      }
      
      const updatedEvidence = await reactiveApiClient.fetchEvidence(selectedCaseId, false);
      caseEvidence = updatedEvidence;
    }
  }

  // Form submission handlers
  async function createNewCase() {
    // Show create case dialog or form
    console.log('Create new case');
  }

  async function updateCase(caseData: Partial<Case>) {
    if (!selectedCase) return;
    
    try {
      await reactiveApiClient.updateCase({
        id: selectedCase.id,
        ...caseData
      });
      await refreshCaseData();
    } catch (error) {
      console.error('Failed to update case:', error);
    }
  }
</script>

<svelte:window keydown={handleKeydown} />

<div class="p-6 max-w-6xl mx-auto">
  <!-- Header -->
  <div class="flex items-center justify-between mb-8">
    <div>
      <h1 class="text-3xl font-bold font-mono text-yorha-text-primary">Case Management</h1>
      <p class="text-yorha-text-secondary font-mono mt-2">
        Search and manage legal cases with AI-powered tools
      </p>
    </div>
    
    <div class="flex items-center gap-3">
      <!-- Global Search Button -->
      <button
        on:onclick={() => showCommandPalette = true}
        class="flex items-center gap-2 px-4 py-2 bg-yorha-bg-secondary border border-yorha-border rounded-md hover:bg-yorha-bg-hover transition-colors font-mono"
      >
        <Search class="h-4 w-4" />
        <span>Search</span>
        <kbd class="px-2 py-1 text-xs bg-yorha-bg-primary border border-yorha-border rounded">⌘K</kbd>
      </button>
      
      <!-- Create Case Button -->
      <button
        on:onclick={createNewCase}
        class="flex items-center gap-2 px-4 py-2 bg-yorha-accent text-yorha-text-accent border border-yorha-accent rounded-md hover:bg-yorha-accent/80 transition-colors font-mono"
      >
        <Plus class="h-4 w-4" />
        New Case
      </button>
    </div>
  </div>

  <!-- Quick Actions -->
  <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
    <button
      on:onclick={() => showCommandPalette = true}
      class="p-4 bg-yorha-bg-secondary border border-yorha-border rounded-md hover:bg-yorha-bg-hover transition-colors text-left"
    >
      <Search class="h-6 w-6 text-yorha-accent mb-2" />
      <h3 class="font-mono font-semibold text-yorha-text-primary">Global Search</h3>
      <p class="text-sm text-yorha-text-secondary font-mono">Find cases, evidence, documents</p>
    </button>
    
    <button
      on:onclick={createNewCase}
      class="p-4 bg-yorha-bg-secondary border border-yorha-border rounded-md hover:bg-yorha-bg-hover transition-colors text-left"
    >
      <Plus class="h-6 w-6 text-yorha-accent mb-2" />
      <h3 class="font-mono font-semibold text-yorha-text-primary">New Case</h3>
      <p class="text-sm text-yorha-text-secondary font-mono">Create a new legal case</p>
    </button>
    
    <div class="p-4 bg-yorha-bg-secondary border border-yorha-border rounded-md">
      <FileText class="h-6 w-6 text-yorha-accent mb-2" />
      <h3 class="font-mono font-semibold text-yorha-text-primary">Evidence</h3>
      <p class="text-sm text-yorha-text-secondary font-mono">Manage case evidence</p>
    </div>
    
    <div class="p-4 bg-yorha-bg-secondary border border-yorha-border rounded-md">
      <Calendar class="h-6 w-6 text-yorha-accent mb-2" />
      <h3 class="font-mono font-semibold text-yorha-text-primary">Timeline</h3>
      <p class="text-sm text-yorha-text-secondary font-mono">Case timeline view</p>
    </div>
  </div>

  <!-- Demo Instructions -->
  <div class="bg-yorha-bg-secondary border border-yorha-border rounded-md p-6 font-mono">
    <h2 class="text-lg font-semibold text-yorha-text-primary mb-4">Modular Components Demo</h2>
    
    <div class="space-y-4 text-sm text-yorha-text-secondary">
      <div>
        <h3 class="font-semibold text-yorha-text-primary mb-2">Command Palette (Global Search)</h3>
        <ul class="list-disc list-inside space-y-1 ml-4">
          <li>Press <kbd class="bg-yorha-bg-primary px-1 py-0.5 rounded">Ctrl+K</kbd> or <kbd class="bg-yorha-bg-primary px-1 py-0.5 rounded">⌘K</kbd> to open</li>
          <li>Search across cases, evidence, documents, and people</li>
          <li>Real-time API integration with PostgreSQL + pgvector</li>
          <li>Vector similarity search for semantic matching</li>
        </ul>
      </div>
      
      <div>
        <h3 class="font-semibold text-yorha-text-primary mb-2">Modular Dialog System</h3>
        <ul class="list-disc list-inside space-y-1 ml-4">
          <li>Auto-loading data based on entity ID</li>
          <li>Reactive updates through API client</li>
          <li>Built-in error handling and retry logic</li>
          <li>Customizable loading and error states</li>
        </ul>
      </div>
      
      <div>
        <h3 class="font-semibold text-yorha-text-primary mb-2">API Integration Features</h3>
        <ul class="list-disc list-inside space-y-1 ml-4">
          <li>PostgreSQL with pgvector similarity search</li>
          <li>Drizzle ORM for type-safe database operations</li>
          <li>Reactive data stores with caching</li>
          <li>Automatic retry and error handling</li>
          <li>Real-time updates via WebSocket (planned)</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<!-- Modular Command Palette -->
<ModularCommand
  bind:open={showCommandPalette}
  placeholder="Search cases, evidence, documents, people..."
  searchTypes={['cases', 'evidence', 'documents', 'people']}
  searchLimit={10}
  minQueryLength={2}
  debounceMs={300}
  onSelect={handleCommandSearch}
  class="max-w-2xl mx-auto mt-4"
/>

<!-- Modular Case Details Dialog -->
<ModularDialog
  bind:open={showCaseDialog}
  title={selectedCase?.title || 'Case Details'}
  size="xl"
  entityType="case"
  entityId={selectedCaseId}
  autoLoad={true}
  cacheData={true}
  class="case-details-dialog"
>
  {#snippet children({ data: caseData, refresh })}
    {#if caseData}
      <div class="space-y-6">
        <!-- Case Overview -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm font-semibold text-yorha-text-primary font-mono">Status</label>
            <div class="mt-1 px-3 py-2 bg-yorha-bg-primary border border-yorha-border rounded-md font-mono">
              {caseData.status}
            </div>
          </div>
          <div>
            <label class="text-sm font-semibold text-yorha-text-primary font-mono">Priority</label>
            <div class="mt-1 px-3 py-2 bg-yorha-bg-primary border border-yorha-border rounded-md font-mono">
              {caseData.priority}
            </div>
          </div>
        </div>

        <!-- Case Description -->
        {#if caseData.description}
          <div>
            <label class="text-sm font-semibold text-yorha-text-primary font-mono">Description</label>
            <div class="mt-1 px-3 py-2 bg-yorha-bg-primary border border-yorha-border rounded-md font-mono text-sm">
              {caseData.description}
            </div>
          </div>
        {/if}

        <!-- Quick Actions -->
        <div class="flex flex-wrap gap-2">
          <button
            on:onclick={() => updateCase({ status: 'investigating' })}
            class="px-3 py-1 text-sm bg-yorha-accent text-yorha-text-accent border border-yorha-accent rounded-md hover:bg-yorha-accent/80 transition-colors font-mono"
          >
            Start Investigation
          </button>
          <button
            on:onclick={refresh}
            class="px-3 py-1 text-sm bg-yorha-bg-secondary border border-yorha-border rounded-md hover:bg-yorha-bg-hover transition-colors font-mono"
          >
            Refresh Data
          </button>
        </div>
      </div>
    {:else}
      <div class="text-center py-8">
        <p class="font-mono text-yorha-text-secondary">No case data available</p>
      </div>
    {/if}
  {/snippet}

  {#snippet footer({ close })}
    <div class="flex justify-end gap-2">
      <button
        on:onclick={close}
        class="px-4 py-2 text-sm bg-yorha-bg-secondary border border-yorha-border rounded-md hover:bg-yorha-bg-hover transition-colors font-mono"
      >
        Close
      </button>
    </div>
  {/snippet}
</ModularDialog>