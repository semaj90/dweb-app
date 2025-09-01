<!-- 
  Legal Case Manager Component
  Bits UI v2 + Svelte 5 implementation for comprehensive case management
  Features: CRUD operations, real-time updates, AI integration
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { UiButton as Button, UiInput as Input, UiLabel as Label, UiCard as Card, UiCardContent as CardContent, UiCardHeader as CardHeader, UiCardTitle as CardTitle, UiBadge as Badge } from '$lib/components/ui';
  import { DialogStandard as Dialog } from '$lib/components/ui';
  import SmartTextarea from '$lib/components/ui/SmartTextarea.svelte';
  import SelectStandard from '$lib/components/ui/select/SelectStandard.svelte';
  import { legalPlatformClient, type CaseData, type ApiResponse } from '$lib/services/legal-platform-client';

  // Component state
  let cases = writable<CaseData[]>([]);
  let loading = writable(false);
  let error = writable('');
  let searchQuery = writable('');
  let selectedCase = writable<CaseData | null>(null);
  let isCreateDialogOpen = writable(false);
  let isEditDialogOpen = writable(false);

  // Form state
  let formData = writable<Partial<CaseData>>({
    title: '',
    description: '',
    priority: 'medium',
    status: 'open',
    location: '',
    incidentDate: ''
  });

  // Priority options
  const priorityOptions = [
    { value: 'low', label: 'Low', class: 'bg-green-100 text-green-800' },
    { value: 'medium', label: 'Medium', class: 'bg-yellow-100 text-yellow-800' },
    { value: 'high', label: 'High', class: 'bg-orange-100 text-orange-800' },
    { value: 'urgent', label: 'Urgent', class: 'bg-red-100 text-red-800' }
  ];

  // Status options
  const statusOptions = [
    { value: 'open', label: 'Open', class: 'bg-blue-100 text-blue-800' },
    { value: 'investigating', label: 'Investigating', class: 'bg-purple-100 text-purple-800' },
    { value: 'trial', label: 'Trial', class: 'bg-indigo-100 text-indigo-800' },
    { value: 'closed', label: 'Closed', class: 'bg-gray-100 text-gray-800' },
    { value: 'dismissed', label: 'Dismissed', class: 'bg-slate-100 text-slate-800' }
  ];

  // Load cases on component mount
  onMount(async () => {
    await loadCases();
  });

  // Load all cases
  async function loadCases() {
    loading.set(true);
    error.set('');
    
    try {
      const response = await legalPlatformClient.getAllCases();
      
      if (response.success && response.data) {
        cases.set(response.data);
      } else {
        error.set(response.error || 'Failed to load cases');
      }
    } catch (err) {
      error.set(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      loading.set(false);
    }
  }

  // Search cases
  async function searchCases(query: string) {
    if (!query.trim()) {
      await loadCases();
      return;
    }

    loading.set(true);
    error.set('');
    
    try {
      const response = await legalPlatformClient.searchCases(query);
      
      if (response.success && response.data) {
        cases.set(response.data);
      } else {
        error.set(response.error || 'Search failed');
      }
    } catch (err) {
      error.set(err instanceof Error ? err.message : 'Search error');
    } finally {
      loading.set(false);
    }
  }

  // Create new case
  async function createCase() {
    const data = $formData;
    
    if (!data.title?.trim()) {
      error.set('Case title is required');
      return;
    }

    loading.set(true);
    error.set('');
    
    try {
      const response = await legalPlatformClient.createCase(data as CaseData);
      
      if (response.success) {
        isCreateDialogOpen.set(false);
        formData.set({
          title: '',
          description: '',
          priority: 'medium',
          status: 'open',
          location: '',
          incidentDate: ''
        });
        await loadCases();
      } else {
        error.set(response.error || 'Failed to create case');
      }
    } catch (err) {
      error.set(err instanceof Error ? err.message : 'Creation failed');
    } finally {
      loading.set(false);
    }
  }

  // Update existing case
  async function updateCase() {
    const data = $formData;
    const selected = $selectedCase;
    
    if (!selected?.id) return;

    loading.set(true);
    error.set('');
    
    try {
      const response = await legalPlatformClient.updateCase(selected.id, data);
      
      if (response.success) {
        isEditDialogOpen.set(false);
        selectedCase.set(null);
        await loadCases();
      } else {
        error.set(response.error || 'Failed to update case');
      }
    } catch (err) {
      error.set(err instanceof Error ? err.message : 'Update failed');
    } finally {
      loading.set(false);
    }
  }

  // Delete case
  async function deleteCase(caseId: string) {
    if (!confirm('Are you sure you want to delete this case?')) return;

    loading.set(true);
    error.set('');
    
    try {
      const response = await legalPlatformClient.deleteCase(caseId);
      
      if (response.success) {
        await loadCases();
      } else {
        error.set(response.error || 'Failed to delete case');
      }
    } catch (err) {
      error.set(err instanceof Error ? err.message : 'Deletion failed');
    } finally {
      loading.set(false);
    }
  }

  // Open edit dialog
  function openEditDialog(caseData: CaseData) {
    selectedCase.set(caseData);
    formData.set({
      title: caseData.title,
      description: caseData.description || '',
      priority: caseData.priority || 'medium',
      status: caseData.status || 'open',
      location: caseData.location || '',
      incidentDate: caseData.incidentDate || ''
    });
    isEditDialogOpen.set(true);
  }

  // Get priority badge class
  function getPriorityClass(priority: string) {
    const option = priorityOptions.find(p => p.value === priority);
    return option?.class || 'bg-gray-100 text-gray-800';
  }

  // Get status badge class
  function getStatusClass(status: string) {
    const option = statusOptions.find(s => s.value === status);
    return option?.class || 'bg-gray-100 text-gray-800';
  }

  // Reactive search
  $: if ($searchQuery) {
    const debounceTimer = setTimeout(() => {
      searchCases($searchQuery);
    }, 300);
    
    return () => clearTimeout(debounceTimer);
  }
</script>

<div class="legal-case-manager p-6 max-w-7xl mx-auto">
  <!-- Header -->
  <div class="flex justify-between items-center mb-6">
    <div>
      <h1 class="text-3xl font-bold text-gray-900">Case Management</h1>
      <p class="text-gray-600">Manage legal cases with AI-powered assistance</p>
    </div>
    
    <Dialog bind:open={$isCreateDialogOpen}>
      <DialogTrigger asChild let:builder>
        <Button builders={[builder]} class="bg-blue-600 hover:bg-blue-700">
          Create New Case
        </Button>
      </DialogTrigger>
      <DialogContent class="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Create New Case</DialogTitle>
        </DialogHeader>
        
        <form on:submit|preventDefault={createCase} class="space-y-4">
          <div class="space-y-2">
            <Label for="title">Case Title *</Label>
            <Input 
              id="title" 
              bind:value={$formData.title} 
              placeholder="Enter case title"
              required
            />
          </div>
          
          <div class="space-y-2">
            <Label for="description">Description</Label>
            <Textarea 
              id="description" 
              bind:value={$formData.description} 
              placeholder="Case description"
              rows={3}
            />
          </div>
          
          <div class="grid grid-cols-2 gap-4">
            <div class="space-y-2">
              <Label for="priority">Priority</Label>
              <Select bind:value={$formData.priority}>
                <SelectTrigger>
                  <SelectValue placeholder="Select priority" />
                </SelectTrigger>
                <SelectContent>
                  {#each priorityOptions as option}
                    <SelectItem value={option.value}>{option.label}</SelectItem>
                  {/each}
                </SelectContent>
              </Select>
            </div>
            
            <div class="space-y-2">
              <Label for="status">Status</Label>
              <Select bind:value={$formData.status}>
                <SelectTrigger>
                  <SelectValue placeholder="Select status" />
                </SelectTrigger>
                <SelectContent>
                  {#each statusOptions as option}
                    <SelectItem value={option.value}>{option.label}</SelectItem>
                  {/each}
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div class="grid grid-cols-2 gap-4">
            <div class="space-y-2">
              <Label for="location">Location</Label>
              <Input 
                id="location" 
                bind:value={$formData.location} 
                placeholder="Incident location"
              />
            </div>
            
            <div class="space-y-2">
              <Label for="incidentDate">Incident Date</Label>
              <Input 
                id="incidentDate" 
                type="date"
                bind:value={$formData.incidentDate}
              />
            </div>
          </div>
          
          <div class="flex justify-end space-x-2 pt-4">
            <Button 
              type="button" 
              variant="outline"
              on:on:click={() => isCreateDialogOpen.set(false)}
            >
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={$loading}
              class="bg-blue-600 hover:bg-blue-700"
            >
              {$loading ? 'Creating...' : 'Create Case'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  </div>

  <!-- Search -->
  <div class="mb-6">
    <Input 
      bind:value={$searchQuery}
      placeholder="Search cases by title, description, or case number..."
      class="max-w-md"
    />
  </div>

  <!-- Error Alert -->
  {#if $error}
    <Alert class="mb-6 border-red-200 bg-red-50">
      <AlertDescription class="text-red-800">
        {$error}
      </AlertDescription>
    </Alert>
  {/if}

  <!-- Loading State -->
  {#if $loading}
    <div class="flex justify-center items-center py-12">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    </div>
  {/if}

  <!-- Cases Grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    {#each $cases as caseData (caseData.id)}
      <Card class="hover:shadow-lg transition-shadow">
        <CardHeader class="pb-3">
          <div class="flex justify-between items-start">
            <div class="flex-1">
              <CardTitle class="text-lg line-clamp-2">{caseData.title}</CardTitle>
              <p class="text-sm text-gray-500 mt-1">#{caseData.id?.slice(-8)}</p>
            </div>
            <div class="flex space-x-2">
              <Badge class={getPriorityClass(caseData.priority || 'medium')}>
                {caseData.priority?.toUpperCase()}
              </Badge>
              <Badge class={getStatusClass(caseData.status || 'open')}>
                {caseData.status?.toUpperCase()}
              </Badge>
            </div>
          </div>
        </CardHeader>
        
        <CardContent class="pt-0">
          {#if caseData.description}
            <p class="text-sm text-gray-600 mb-3 line-clamp-2">
              {caseData.description}
            </p>
          {/if}
          
          {#if caseData.location}
            <p class="text-xs text-gray-500 mb-2">📍 {caseData.location}</p>
          {/if}
          
          {#if caseData.incidentDate}
            <p class="text-xs text-gray-500 mb-3">
              📅 {new Date(caseData.incidentDate).toLocaleDateString()}
            </p>
          {/if}
          
          <Separator class="mb-3" />
          
          <div class="flex justify-between items-center">
            <p class="text-xs text-gray-400">
              Created {new Date(caseData.createdAt || Date.now()).toLocaleDateString()}
            </p>
            
            <div class="flex space-x-2">
              <Button 
                size="sm" 
                variant="outline"
                on:on:click={() => openEditDialog(caseData)}
              >
                Edit
              </Button>
              <Button 
                size="sm" 
                variant="destructive"
                on:on:click={() => deleteCase(caseData.id!)}
              >
                Delete
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    {/each}
  </div>

  <!-- Empty State -->
  {#if !$loading && $cases.length === 0}
    <div class="text-center py-12">
      <div class="mx-auto max-w-md">
        <h3 class="text-lg font-medium text-gray-900 mb-2">No cases found</h3>
        <p class="text-gray-500 mb-6">
          {$searchQuery ? 'No cases match your search criteria.' : 'Get started by creating your first case.'}
        </p>
        
        {#if !$searchQuery}
          <Button 
            class="bg-blue-600 hover:bg-blue-700"
            on:on:click={() => isCreateDialogOpen.set(true)}
          >
            Create First Case
          </Button>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Edit Dialog -->
  <Dialog bind:open={$isEditDialogOpen}>
    <DialogContent class="max-w-2xl">
      <DialogHeader>
        <DialogTitle>Edit Case</DialogTitle>
      </DialogHeader>
      
      <form on:submit|preventDefault={updateCase} class="space-y-4">
        <div class="space-y-2">
          <Label for="edit-title">Case Title *</Label>
          <Input 
            id="edit-title" 
            bind:value={$formData.title} 
            placeholder="Enter case title"
            required
          />
        </div>
        
        <div class="space-y-2">
          <Label for="edit-description">Description</Label>
          <Textarea 
            id="edit-description" 
            bind:value={$formData.description} 
            placeholder="Case description"
            rows={3}
          />
        </div>
        
        <div class="grid grid-cols-2 gap-4">
          <div class="space-y-2">
            <Label for="edit-priority">Priority</Label>
            <Select bind:value={$formData.priority}>
              <SelectTrigger>
                <SelectValue placeholder="Select priority" />
              </SelectTrigger>
              <SelectContent>
                {#each priorityOptions as option}
                  <SelectItem value={option.value}>{option.label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
          </div>
          
          <div class="space-y-2">
            <Label for="edit-status">Status</Label>
            <Select bind:value={$formData.status}>
              <SelectTrigger>
                <SelectValue placeholder="Select status" />
              </SelectTrigger>
              <SelectContent>
                {#each statusOptions as option}
                  <SelectItem value={option.value}>{option.label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
          </div>
        </div>
        
        <div class="grid grid-cols-2 gap-4">
          <div class="space-y-2">
            <Label for="edit-location">Location</Label>
            <Input 
              id="edit-location" 
              bind:value={$formData.location} 
              placeholder="Incident location"
            />
          </div>
          
          <div class="space-y-2">
            <Label for="edit-incidentDate">Incident Date</Label>
            <Input 
              id="edit-incidentDate" 
              type="date"
              bind:value={$formData.incidentDate}
            />
          </div>
        </div>
        
        <div class="flex justify-end space-x-2 pt-4">
          <Button 
            type="button" 
            variant="outline"
            on:on:click={() => isEditDialogOpen.set(false)}
          >
            Cancel
          </Button>
          <Button 
            type="submit" 
            disabled={$loading}
            class="bg-blue-600 hover:bg-blue-700"
          >
            {$loading ? 'Updating...' : 'Update Case'}
          </Button>
        </div>
      </form>
    </DialogContent>
  </Dialog>
</div>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>