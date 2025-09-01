<script lang="ts">
  import { page } from '$app/stores';
  import { goto, invalidateAll } from '$app/navigation';
  import { enhance } from '$app/forms';
  import { superForm } from 'sveltekit-superforms';
  import { zodClient } from 'sveltekit-superforms/adapters';
  import { z } from 'zod';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Textarea } from '$lib/components/ui/textarea/index.js';
  import * as Select from '$lib/components/ui/select/index.js';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import * as AlertDialog from '$lib/components/ui/alert-dialog/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import * as Card from '$lib/components/ui/card/index.js';
  import { Separator } from '$lib/components/ui/separator/index.js';
  import * as Tabs from '$lib/components/ui/tabs/index.js';
  import { toast } from 'svelte-sonner';
  import { Plus, Search, Filter, Edit2, Trash2, FileText, Eye, AlertCircle } from 'lucide-svelte';
  import { cn } from '$lib/utils.js';
  import type { PageData, ActionData } from './$types.js';
  
  // Feedback Integration
  import FeedbackIntegration from '$lib/components/feedback/FeedbackIntegration.svelte';

  // Zod schemas for validation
  const createCaseSchema = z.object({
    title: z.string().min(1, 'Case title is required').max(500, 'Case title too long'),
    description: z.string().optional(),
    priority: z.enum(['low', 'medium', 'high', 'critical']).default('medium'),
    status: z.enum(['open', 'investigating', 'pending', 'closed', 'archived']).default('open'),
    incidentDate: z.string().optional(),
    location: z.string().optional(),
    jurisdiction: z.string().optional()
  });

  const addEvidenceSchema = z.object({
    caseId: z.string().min(1, 'Case ID is required'),
    title: z.string().min(1, 'Evidence title is required').max(255, 'Title too long'),
    description: z.string().optional(),
    evidenceType: z.enum(['document', 'photo', 'video', 'audio', 'physical', 'digital', 'testimony']).default('document'),
    tags: z.string().optional()
  });

  // Props from load function
  let { data }: { data: PageData } = $props();
  let form: ActionData = $state(null);

  // Superforms for type-safe form handling
  const createCaseForm = superForm(data.createCaseForm, {
    validators: zodClient(createCaseSchema),
    onUpdated: ({ form }) => {
      if (form.valid) {
        toast.success('Case created successfully');
        createCaseDialogOpen = false;
        invalidateAll();
        
        // Track successful case creation for feedback
        if (caseCreationFeedback) {
          caseCreationFeedback.markCompleted({
            success: true,
            caseTitle: form.data.title,
            casePriority: form.data.priority,
            caseStatus: form.data.status
          });
        }
      }
    },
    onError: ({ result }) => {
      toast.error(result.error.message || 'Failed to create case');
      
      // Track failed case creation for feedback
      if (caseCreationFeedback) {
        caseCreationFeedback.markFailed({
          errorType: 'case_creation_error',
          errorMessage: result.error.message
        });
      }
    }
  });
  
  const addEvidenceForm = superForm(data.addEvidenceForm, {
    validators: zodClient(addEvidenceSchema),
    onUpdated: ({ form }) => {
      if (form.valid) {
        toast.success('Evidence added successfully');
        addEvidenceDialogOpen = false;
        invalidateAll();
      }
    },
    onError: ({ result }) => {
      toast.error(result.error.message || 'Failed to add evidence');
    }
  });

  const { form: createFormData, enhance: createEnhance } = createCaseForm;
  const { form: evidenceFormData, enhance: evidenceEnhance } = addEvidenceForm;

  // UI state
  let createCaseDialogOpen = $state(false);
  let addEvidenceDialogOpen = $state(false);
  let deleteEvidenceDialogOpen = $state(false);
  let evidenceToDelete = $state(null);

  // Search and filter state
  let searchQuery = $state('');
  let statusFilter = $state('all');
  let priorityFilter = $state('all');
  let sortBy = $state('createdAt');
  let sortOrder = $state('desc');

  // Vector search state
  let useVectorSearch = $state(false);
  let vectorSearchResults = $state([]);
  let isSearching = $state(false);

  // Feedback integration references
  let pageFeedback: any;
  let searchFeedback: any;
  let caseCreationFeedback: any;

  // Filtered and sorted cases
  let filteredCases = $derived(() => {
    let filtered = data.userCases || [];
    
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(c => 
        c.title?.toLowerCase().includes(query) ||
        c.description?.toLowerCase().includes(query) ||
        c.location?.toLowerCase().includes(query) ||
        c.jurisdiction?.toLowerCase().includes(query)
      );
    }
    
    if (statusFilter !== 'all') {
      filtered = filtered.filter(c => c.status === statusFilter);
    }
    
    if (priorityFilter !== 'all') {
      filtered = filtered.filter(c => c.priority === priorityFilter);
    }
    
    // Sort cases
    filtered.sort((a, b) => {
      const aVal = a[sortBy as keyof typeof a];
      const bVal = b[sortBy as keyof typeof b];
      const comparison = aVal > bVal ? 1 : -1;
      return sortOrder === 'desc' ? -comparison : comparison;
    });
    
    return filtered;
  });

  // Priority colors
  const priorityColors = {
    low: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300',
    high: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300',
    critical: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
  };

  // Status colors
  const statusColors = {
    open: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
    investigating: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300',
    pending: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300',
    closed: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300',
    archived: 'bg-gray-100 text-gray-600 dark:bg-gray-900 dark:text-gray-400'
  };

  // Vector search function
  async function performVectorSearch() {
    if (!searchQuery.trim()) return;
    
    // Track search interaction for feedback
    const searchInteractionId = searchFeedback?.triggerFeedback({
      query: searchQuery,
      searchType: 'vector_search',
      legalDomain: 'case_management',
      searchStartTime: Date.now()
    });
    
    isSearching = true;
    const searchStartTime = Date.now();
    
    try {
      const response = await fetch('/api/cases/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          useVectorSearch: true,
          limit: 10,
          threshold: 0.7
        })
      });
      
      if (response.ok) {
        const results = await response.json();
        vectorSearchResults = results.data || [];
        toast.success(`Found ${vectorSearchResults.length} similar cases using AI search`);
        
        // Track successful search for feedback
        if (searchInteractionId && searchFeedback) {
          searchFeedback.markCompleted({
            success: true,
            resultCount: vectorSearchResults.length,
            searchTime: Date.now() - searchStartTime,
            relevanceScore: vectorSearchResults.length > 0 ? 0.8 : 0.3 // Estimated relevance
          });
        }
      } else {
        throw new Error('Search failed');
      }
    } catch (error) {
      toast.error('Vector search failed');
      
      // Track failed search for feedback
      if (searchInteractionId && searchFeedback) {
        searchFeedback.markFailed({
          errorType: 'search_error',
          errorMessage: error.message,
          searchTime: Date.now() - searchStartTime
        });
      }
    } finally {
      isSearching = false;
    }
  }

  // View case details
  function viewCase(caseItem: any) {
    goto(`/cases?view=${caseItem.id}`);
  }

  // Delete evidence handler
  function confirmDeleteEvidence(evidence: any) {
    evidenceToDelete = evidence;
    deleteEvidenceDialogOpen = true;
  }

  async function deleteEvidence() {
    if (!evidenceToDelete) return;
    
    const formData = new FormData();
    formData.append('evidenceId', evidenceToDelete.id);

    const response = await fetch('/cases?/deleteEvidence', {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      toast.success('Evidence deleted successfully');
      deleteEvidenceDialogOpen = false;
      evidenceToDelete = null;
      await invalidateAll();
    } else {
      toast.error('Failed to delete evidence');
    }
  }

  // Set evidence form case ID when dialog opens
  $effect(() => {
    if (addEvidenceDialogOpen && data.activeCase?.id) {
      $evidenceFormData.caseId = data.activeCase.id;
    }
  });
</script>

<svelte:head>
  <title>Legal Cases - YoRHa Detective Interface</title>
  <meta name="description" content="Manage legal cases with AI-powered vector search and PostgreSQL storage" />
</svelte:head>

<div class="container mx-auto py-6 px-4 max-w-7xl">
  <div class="flex flex-col space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold tracking-tight text-foreground">Legal Cases</h1>
        <p class="text-muted-foreground">
          Manage cases with AI-powered search and PostgreSQL vector storage
        </p>
      </div>
      <Button on:on:click={() => createCaseDialogOpen = true} class="gap-2">
        <Plus class="h-4 w-4" />
        New Case
      </Button>
    </div>

    <!-- Stats Overview -->
    {#if data.caseStats}
    <div class="grid gap-4 md:grid-cols-4">
      <Card.Root>
        <Card.Header class="flex flex-row items-center justify-between space-y-0 pb-2">
          <Card.Title class="text-sm font-medium">Total Cases</Card.Title>
          <FileText class="h-4 w-4 text-muted-foreground" />
        </Card.Header>
        <Card.Content>
          <div class="text-2xl font-bold">{data.caseStats.total}</div>
        </Card.Content>
      </Card.Root>
      <Card.Root>
        <Card.Header class="flex flex-row items-center justify-between space-y-0 pb-2">
          <Card.Title class="text-sm font-medium">Open Cases</Card.Title>
          <Eye class="h-4 w-4 text-muted-foreground" />
        </Card.Header>
        <Card.Content>
          <div class="text-2xl font-bold text-blue-600">{data.caseStats.open}</div>
        </Card.Content>
      </Card.Root>
      <Card.Root>
        <Card.Header class="flex flex-row items-center justify-between space-y-0 pb-2">
          <Card.Title class="text-sm font-medium">High Priority</Card.Title>
          <AlertCircle class="h-4 w-4 text-red-500" />
        </Card.Header>
        <Card.Content>
          <div class="text-2xl font-bold text-red-600">{data.caseStats.highPriority}</div>
        </Card.Content>
      </Card.Root>
      <Card.Root>
        <Card.Header class="flex flex-row items-center justify-between space-y-0 pb-2">
          <Card.Title class="text-sm font-medium">Closed Cases</Card.Title>
          <div class="h-4 w-4 rounded-full bg-gray-500"></div>
        </Card.Header>
        <Card.Content>
          <div class="text-2xl font-bold text-gray-600">{data.caseStats.closed}</div>
        </Card.Content>
      </Card.Root>
    </div>
    {/if}

    <!-- Filters and Search -->
    <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
      <div class="flex flex-col gap-4 md:flex-row md:items-center">
        <div class="relative">
          <Search class="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            bind:value={searchQuery}
            placeholder="Search cases with AI vector search..."
            class="pl-8 w-full md:w-[400px]"
          />
        </div>
        
        <div class="flex gap-2 items-center">
          <Button
            variant="outline"
            size="sm"
            on:on:click={performVectorSearch}
            disabled={!searchQuery.trim() || isSearching}
            class="gap-2"
          >
            {#if isSearching}
              <div class="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"></div>
            {:else}
              <Search class="h-4 w-4" />
            {/if}
            AI Search
          </Button>
          
          <SelectRoot bind:selected={statusFilter}>
            <SelectTrigger class="w-[140px]">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="open">Open</SelectItem>
              <SelectItem value="investigating">Investigating</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="closed">Closed</SelectItem>
              <SelectItem value="archived">Archived</SelectItem>
            </SelectContent>
          </SelectRoot>

          <SelectRoot bind:selected={priorityFilter}>
            <SelectTrigger class="w-[140px]">
              <SelectValue placeholder="Priority" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Priority</SelectItem>
              <SelectItem value="low">Low</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
            </SelectContent>
          </SelectRoot>
        </div>
      </div>
    </div>

    <!-- Cases Grid or Detail View -->
    {#if data.activeCase}
      <!-- Case Detail View -->
      <div class="space-y-6">
        <div class="flex items-center justify-between">
          <Button variant="outline" on:on:click={() => goto('/cases')}>
            ← Back to Cases
          </Button>
          <div class="flex gap-2">
            <Button variant="outline" size="sm" on:on:click={() => addEvidenceDialogOpen = true}>
              <Plus class="h-4 w-4 mr-2" />
              Add Evidence
            </Button>
          </div>
        </div>

        <Card.Root>
          <Card.Header>
            <div class="flex items-start justify-between">
              <div class="space-y-2">
                <Card.Title class="text-2xl">{data.activeCase.title}</Card.Title>
                <div class="flex gap-2">
                  <Badge class={cn(priorityColors[data.activeCase.priority])}>
                    {data.activeCase.priority}
                  </Badge>
                  <Badge class={cn(statusColors[data.activeCase.status])}>
                    {data.activeCase.status}
                  </Badge>
                  {#if data.activeCase.location}
                    <Badge variant="secondary">📍 {data.activeCase.location}</Badge>
                  {/if}
                  {#if data.activeCase.jurisdiction}
                    <Badge variant="outline">⚖️ {data.activeCase.jurisdiction}</Badge>
                  {/if}
                </div>
              </div>
            </div>
          </Card.Header>
          <Card.Content>
            <p class="text-muted-foreground mb-4">{data.activeCase.description}</p>
            {#if data.activeCase.incidentDate}
              <div class="text-sm text-muted-foreground">
                <strong>Incident Date:</strong> {new Date(data.activeCase.incidentDate).toLocaleDateString()}
              </div>
            {/if}
          </Card.Content>
        </Card.Root>

        <!-- Evidence Section -->
        <div class="space-y-4">
          <h2 class="text-xl font-semibold">Evidence ({data.caseEvidence.length})</h2>
          
          {#if data.caseEvidence.length === 0}
            <Card.Root>
              <Card.Content class="flex flex-col items-center justify-center py-12">
                <FileText class="h-12 w-12 text-muted-foreground mb-4" />
                <p class="text-muted-foreground mb-4">No evidence has been added to this case yet.</p>
                <Button on:on:click={() => addEvidenceDialogOpen = true}>
                  <Plus class="h-4 w-4 mr-2" />
                  Add First Evidence
                </Button>
              </Card.Content>
            </Card.Root>
          {:else}
            <div class="grid gap-4">
              {#each data.caseEvidence as evidence}
                <Card.Root>
                  <Card.Header>
                    <div class="flex items-center justify-between">
                      <Card.Title class="text-lg">{evidence.title}</Card.Title>
                      <div class="flex gap-2">
                        <Badge variant="secondary">{evidence.evidenceType}</Badge>
                        <Button variant="ghost" size="sm">
                          <Edit2 class="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          on:on:click={() => confirmDeleteEvidence(evidence)}
                        >
                          <Trash2 class="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </Card.Header>
                  <Card.Content>
                    <p class="text-sm text-muted-foreground mb-2">{evidence.description}</p>
                    <div class="flex justify-between items-center text-xs text-muted-foreground">
                      <span>Collected: {new Date(evidence.collectedAt).toLocaleDateString()}</span>
                      {#if evidence.tags}
                        <span>Tags: {evidence.tags}</span>
                      {/if}
                    </div>
                  </Card.Content>
                </Card.Root>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    {:else}
      <!-- Cases Grid View -->
      <Tabs.Root value="all-cases" class="w-full">
        <Tabs.List class="grid w-full grid-cols-3">
          <Tabs.Trigger value="all-cases">All Cases ({filteredCases.length})</Tabs.Trigger>
          <Tabs.Trigger value="vector-search" disabled={!vectorSearchResults.length}>
            AI Search Results ({vectorSearchResults.length})
          </Tabs.Trigger>
          <Tabs.Trigger value="analytics">Analytics</Tabs.Trigger>
        </Tabs.List>
        
        <Tabs.Content value="all-cases" class="space-y-4">
          <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {#each filteredCases as caseItem}
              <Card.Root class="cursor-pointer transition-colors hover:bg-muted/50" on:on:click={() => viewCase(caseItem)}>
                <Card.Header>
                  <div class="flex items-start justify-between">
                    <Card.Title class="text-lg line-clamp-2">{caseItem.title}</Card.Title>
                    <div class="flex flex-col gap-1">
                      <Badge class={cn(priorityColors[caseItem.priority], 'text-xs')}>
                        {caseItem.priority}
                      </Badge>
                      <Badge class={cn(statusColors[caseItem.status], 'text-xs')}>
                        {caseItem.status}
                      </Badge>
                    </div>
                  </div>
                </Card.Header>
                <Card.Content>
                  <p class="text-sm text-muted-foreground line-clamp-3 mb-4">
                    {caseItem.description || 'No description provided'}
                  </p>
                  <div class="space-y-2">
                    {#if caseItem.location}
                      <div class="text-xs text-muted-foreground">📍 {caseItem.location}</div>
                    {/if}
                    {#if caseItem.jurisdiction}
                      <div class="text-xs text-muted-foreground">⚖️ {caseItem.jurisdiction}</div>
                    {/if}
                    <div class="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Created: {new Date(caseItem.createdAt).toLocaleDateString()}</span>
                      {#if caseItem.updatedAt}
                        <span>Updated: {new Date(caseItem.updatedAt).toLocaleDateString()}</span>
                      {/if}
                    </div>
                  </div>
                </Card.Content>
              </Card.Root>
            {:else}
              <div class="col-span-full">
                <Card.Root>
                  <Card.Content class="flex flex-col items-center justify-center py-12">
                    <FileText class="h-12 w-12 text-muted-foreground mb-4" />
                    <p class="text-muted-foreground mb-4">
                      {searchQuery.trim() ? 'No cases found matching your search.' : 'No cases found.'}
                    </p>
                    <Button on:on:click={() => createCaseDialogOpen = true}>
                      <Plus class="h-4 w-4 mr-2" />
                      Create Your First Case
                    </Button>
                  </Card.Content>
                </Card.Root>
              </div>
            {/each}
          </div>
        </Tabs.Content>
        
        <Tabs.Content value="vector-search" class="space-y-4">
          {#if vectorSearchResults.length > 0}
            <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {#each vectorSearchResults as result}
                <Card.Root class="cursor-pointer transition-colors hover:bg-muted/50" on:on:click={() => viewCase(result)}>
                  <Card.Header>
                    <div class="flex items-start justify-between">
                      <Card.Title class="text-lg line-clamp-2">{result.title}</Card.Title>
                      <Badge variant="secondary" class="text-xs">
                        {Math.round(result.similarity * 100)}% match
                      </Badge>
                    </div>
                  </Card.Header>
                  <Card.Content>
                    <p class="text-sm text-muted-foreground line-clamp-3 mb-2">
                      {result.description || 'No description provided'}
                    </p>
                    <div class="text-xs text-green-600 mb-2">
                      🤖 AI-powered similarity: {(result.similarity * 100).toFixed(1)}%
                    </div>
                  </Card.Content>
                </Card.Root>
              {/each}
            </div>
          {:else}
            <Card.Root>
              <Card.Content class="flex flex-col items-center justify-center py-12">
                <Search class="h-12 w-12 text-muted-foreground mb-4" />
                <p class="text-muted-foreground mb-4">Use the AI Search button to find similar cases</p>
                <p class="text-sm text-muted-foreground text-center max-w-md">
                  Vector search uses AI to find semantically similar cases based on content, not just keywords.
                </p>
              </Card.Content>
            </Card.Root>
          {/if}
        </Tabs.Content>
        
        <Tabs.Content value="analytics" class="space-y-4">
          <div class="grid gap-4 md:grid-cols-2">
            <Card.Root>
              <Card.Header>
                <Card.Title>Case Distribution</Card.Title>
              </Card.Header>
              <Card.Content>
                <div class="space-y-2">
                  {#if data.caseStats}
                    <div class="flex justify-between">
                      <span>Open Cases</span>
                      <span class="font-semibold">{data.caseStats.open}</span>
                    </div>
                    <div class="flex justify-between">
                      <span>Investigating</span>
                      <span class="font-semibold">{data.caseStats.investigating || 0}</span>
                    </div>
                    <div class="flex justify-between">
                      <span>Pending</span>
                      <span class="font-semibold">{data.caseStats.pending || 0}</span>
                    </div>
                    <div class="flex justify-between">
                      <span>Closed</span>
                      <span class="font-semibold">{data.caseStats.closed}</span>
                    </div>
                  {/if}
                </div>
              </Card.Content>
            </Card.Root>
            
            <Card.Root>
              <Card.Header>
                <Card.Title>Priority Breakdown</Card.Title>
              </Card.Header>
              <Card.Content>
                <div class="space-y-2">
                  {#if data.caseStats}
                    <div class="flex justify-between">
                      <span class="text-red-600">Critical</span>
                      <span class="font-semibold">{data.caseStats.critical || 0}</span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-orange-600">High</span>
                      <span class="font-semibold">{data.caseStats.highPriority}</span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-yellow-600">Medium</span>
                      <span class="font-semibold">{data.caseStats.medium || 0}</span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-green-600">Low</span>
                      <span class="font-semibold">{data.caseStats.low || 0}</span>
                    </div>
                  {/if}
                </div>
              </Card.Content>
            </Card.Root>
          </div>
        </Tabs.Content>
      </Tabs.Root>
    {/if}
  </div>
</div>

<!-- Create Case Dialog -->
<Dialog.Root bind:open={createCaseDialogOpen}>
  <Dialog.Content class="sm:max-w-[600px]">
    <Dialog.Header>
      <Dialog.Title>Create New Case</Dialog.Title>
      <Dialog.Description>
        Add a new legal case with AI-powered categorization and PostgreSQL storage.
      </Dialog.Description>
    </Dialog.Header>
    <form method="POST" action="?/createCase" use:createEnhance>
      <div class="grid gap-4 py-4">
        <div class="grid gap-2">
          <Label for="case-title">Case Title *</Label>
          <Input
            id="case-title"
            name="title"
            bind:value={$createFormData.title}
            placeholder="Enter case title..."
            required
          />
        </div>
        <div class="grid gap-2">
          <Label for="case-description">Description</Label>
          <Textarea
            id="case-description"
            name="description"
            bind:value={$createFormData.description}
            placeholder="Describe the case details..."
            class="min-h-[100px]"
          />
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div class="grid gap-2">
            <Label for="case-priority">Priority</Label>
            <SelectRoot bind:selected={$createFormData.priority}>
              <SelectTrigger>
                <SelectValue placeholder="Select priority" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
              </SelectContent>
            </SelectRoot>
            <input type="hidden" name="priority" value={$createFormData.priority} />
          </div>
          <div class="grid gap-2">
            <Label for="case-status">Status</Label>
            <SelectRoot bind:selected={$createFormData.status}>
              <SelectTrigger>
                <SelectValue placeholder="Select status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="open">Open</SelectItem>
                <SelectItem value="investigating">Investigating</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </SelectRoot>
            <input type="hidden" name="status" value={$createFormData.status} />
          </div>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div class="grid gap-2">
            <Label for="case-location">Location</Label>
            <Input
              id="case-location"
              name="location"
              bind:value={$createFormData.location}
              placeholder="Case location..."
            />
          </div>
          <div class="grid gap-2">
            <Label for="case-jurisdiction">Jurisdiction</Label>
            <Input
              id="case-jurisdiction"
              name="jurisdiction"
              bind:value={$createFormData.jurisdiction}
              placeholder="Legal jurisdiction..."
            />
          </div>
        </div>
        <div class="grid gap-2">
          <Label for="incident-date">Incident Date</Label>
          <Input
            id="incident-date"
            name="incidentDate"
            type="datetime-local"
            bind:value={$createFormData.incidentDate}
          />
        </div>
      </div>
      <Dialog.Footer>
        <Button variant="outline" type="button" on:on:click={() => createCaseDialogOpen = false}>
          Cancel
        </Button>
        <Button type="submit" disabled={!$createFormData.title?.trim()}>
          Create Case
        </Button>
      </Dialog.Footer>
    </form>
  </Dialog.Content>
</Dialog.Root>

<!-- Add Evidence Dialog -->
<Dialog.Root bind:open={addEvidenceDialogOpen}>
  <Dialog.Content class="sm:max-w-[525px]">
    <Dialog.Header>
      <Dialog.Title>Add Evidence</Dialog.Title>
      <Dialog.Description>
        Add new evidence with AI-powered analysis and vector indexing.
      </Dialog.Description>
    </Dialog.Header>
    <form method="POST" action="?/addEvidence" use:evidenceEnhance>
      <input type="hidden" name="caseId" value={$evidenceFormData.caseId} />
      <div class="grid gap-4 py-4">
        <div class="grid gap-2">
          <Label for="evidence-title">Evidence Title *</Label>
          <Input
            id="evidence-title"
            name="title"
            bind:value={$evidenceFormData.title}
            placeholder="Enter evidence title..."
            required
          />
        </div>
        <div class="grid gap-2">
          <Label for="evidence-description">Description</Label>
          <Textarea
            id="evidence-description"
            name="description"
            bind:value={$evidenceFormData.description}
            placeholder="Describe the evidence..."
            class="min-h-[80px]"
          />
        </div>
        <div class="grid gap-2">
          <Label for="evidence-type">Evidence Type</Label>
          <SelectRoot bind:selected={$evidenceFormData.evidenceType}>
            <SelectTrigger>
              <SelectValue placeholder="Select type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="document">Document</SelectItem>
              <SelectItem value="photo">Photo</SelectItem>
              <SelectItem value="video">Video</SelectItem>
              <SelectItem value="audio">Audio</SelectItem>
              <SelectItem value="physical">Physical Evidence</SelectItem>
              <SelectItem value="digital">Digital Evidence</SelectItem>
              <SelectItem value="testimony">Testimony</SelectItem>
            </SelectContent>
          </SelectRoot>
          <input type="hidden" name="evidenceType" value={$evidenceFormData.evidenceType} />
        </div>
        <div class="grid gap-2">
          <Label for="evidence-tags">Tags</Label>
          <Input
            id="evidence-tags"
            name="tags"
            bind:value={$evidenceFormData.tags}
            placeholder="Comma-separated tags..."
          />
        </div>
      </div>
      <Dialog.Footer>
        <Button variant="outline" type="button" on:on:click={() => addEvidenceDialogOpen = false}>
          Cancel
        </Button>
        <Button type="submit" disabled={!$evidenceFormData.title?.trim()}>
          Add Evidence
        </Button>
      </Dialog.Footer>
    </form>
  </Dialog.Content>
</Dialog.Root>

<!-- Delete Evidence Confirmation Dialog -->
<AlertDialog.Root bind:open={deleteEvidenceDialogOpen}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title>Are you absolutely sure?</AlertDialog.Title>
      <AlertDialog.Description>
        This action cannot be undone. This will permanently delete the evidence
        "{evidenceToDelete?.title}" from the case and remove it from the vector index.
      </AlertDialog.Description>
    </AlertDialog.Header>
    <AlertDialog.Footer>
      <Button variant="outline" on:on:click={() => deleteEvidenceDialogOpen = false}>
        Cancel
      </Button>
      <Button variant="destructive" on:on:click={deleteEvidence}>
        Delete Evidence
      </Button>
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>

<!-- Feedback Integration Components -->
<FeedbackIntegration
  bind:this={pageFeedback}
  interactionType="page_visit"
  ratingType="ui_experience"
  priority="low"
  context={{ 
    page: 'cases',
    totalCases: filteredCases.length,
    hasActiveFilters: searchQuery.trim() || statusFilter !== 'all' || priorityFilter !== 'all'
  }}
  trackOnMount={true}
  let:feedback
/>

<FeedbackIntegration
  bind:this={searchFeedback}
  interactionType="ai_search"
  ratingType="search_relevance"
  priority="medium"
  context={{ component: 'VectorSearch', legalDomain: 'case_management' }}
  let:feedback
/>

<FeedbackIntegration
  bind:this={caseCreationFeedback}
  interactionType="case_creation"
  ratingType="ui_experience"
  priority="high"
  context={{ component: 'CreateCaseDialog' }}
  let:feedback
/>

<!-- Tailwind CSS will handle all styling through bits-ui components -->
