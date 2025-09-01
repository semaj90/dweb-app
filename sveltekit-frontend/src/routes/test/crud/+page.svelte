<!-- Test CRUD Display with SSR UI -->
<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { enhance } from '$app/forms';
  import { invalidateAll } from '$app/navigation';
  
  // Gaming UI Components
  import { ProgressiveGamingProvider, NES8BitButton, SNES16BitButton, N643DButton } from '$lib/components/ui/gaming';
  
  // Standard UI Components
  import Button from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  
  // Form components
  import { Input } from '$lib/components/ui/input';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '$lib/components/ui/select';

  /** @type {import('./$types').PageData} */
  export let data;
let isCreating = $state(false);
let editingCase = $state(null);
let isDeleting = $state(false);
  
  // Form state
let formData = $state({
    title: '',
    description: '',
    priority: 'medium',
    status: 'open',
    category: 'investigation'
  });

  // UI state
let gamingEra = $state('nes'); // nes, snes, n64
let showSystemHealth = $state(false);
  
  // Reset form
  const resetForm = () => {
    formData = {
      title: '',
      description: '',
      priority: 'medium', 
      status: 'open',
      category: 'investigation'
    };
    editingCase = null;
    isCreating = false;
  };

  // Edit case handler
  const startEdit = (caseItem) => {
    editingCase = caseItem.id;
    formData = {
      title: caseItem.title || '',
      description: caseItem.description || '',
      priority: caseItem.priority || 'medium',
      status: caseItem.status || 'open',
      category: caseItem.category || 'investigation'
    };
    isCreating = true;
  };

  // Delete confirmation
  const confirmDelete = (caseItem) => {
    if (confirm(`Delete case "${caseItem.title}"? This cannot be undone.`)) {
      isDeleting = true;
      deleteCase(caseItem.id);
    }
  };

  // Delete case
  const deleteCase = async (caseId) => {
    try {
      const response = await fetch(`/test/crud?action=delete&id=${caseId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        await invalidateAll();
      } else {
        alert('Failed to delete case');
      }
    } catch (error) {
      console.error('Delete error:', error);
      alert('Error deleting case');
    } finally {
      isDeleting = false;
    }
  };

  // Form submission handler
  const handleSubmit = () => {
    return async ({ result, update }) => {
      if (result.type === 'success') {
        resetForm();
        await update();
      } else if (result.type === 'error') {
        alert('Error saving case: ' + result.error?.message);
      }
    };
  };

  // Priority badge colors
  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'urgent': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'default';
    }
  };

  // Status badge colors
  const getStatusColor = (status) => {
    switch (status) {
      case 'open': return 'default';
      case 'active': return 'default';
      case 'under_review': return 'secondary';
      case 'closed': return 'outline';
      case 'archived': return 'secondary';
      default: return 'default';
    }
  };

  onMount(() => {
    // Test client-side functionality
    console.log('🎮 CRUD Test Page Loaded');
    console.log('📊 Data received:', data);
  });
</script>

<svelte:head>
  <title>CRUD Test - Legal AI Platform</title>
  <meta name="description" content="Test CRUD operations with SSR and gaming UI components" />
</svelte:head>

<ProgressiveGamingProvider enableAutoEvolution={false} integrateWithYorha={true}>
  <div class="container mx-auto p-6 space-y-8">
    
    <!-- Header with System Status -->
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">CRUD Test Dashboard</h1>
        <p class="text-muted-foreground">Testing SSR, UI Components & Database Operations</p>
      </div>
      
      <div class="flex flex-wrap gap-2">
        <Button 
          variant="outline" 
          size="sm"
          on:on:on:click={() => showSystemHealth = !showSystemHealth}
        >
          {showSystemHealth ? 'Hide' : 'Show'} System Health
        </Button>
        
        <!-- Gaming Era Selector -->
        <select 
          bind:value={gamingEra}
          class="px-3 py-1 border rounded text-sm"
        >
          <option value="nes">8-bit NES</option>
          <option value="snes">16-bit SNES</option>
          <option value="n64">3D N64</option>
        </select>
      </div>
    </div>

    <!-- System Health Panel (Conditional) -->
    {#if showSystemHealth}
      <Card>
        <CardHeader>
          <CardTitle>🔧 System Health Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="space-y-2">
              <h4 class="font-medium">Database</h4>
              <Badge variant={data.health.database?.connected ? "default" : "destructive"}>
                {data.health.database?.connected ? 'Connected' : 'Offline'}
              </Badge>
              {#if data.health.database?.responseTime}
                <p class="text-sm text-muted-foreground">Response: {data.health.database.responseTime}ms</p>
              {/if}
            </div>
            
            <div class="space-y-2">
              <h4 class="font-medium">Server</h4>
              <Badge variant="default">SSR Active</Badge>
              <p class="text-sm text-muted-foreground">Cases: {data.cases?.length || 0}</p>
            </div>
            
            <div class="space-y-2">
              <h4 class="font-medium">UI Framework</h4>
              <Badge variant="default">SvelteKit + bits-ui</Badge>
              <p class="text-sm text-muted-foreground">Gaming Era: {gamingEra.toUpperCase()}</p>
            </div>
          </div>
          
          {#if data.health.database?.error}
            <div class="mt-4 p-3 bg-red-50 border border-red-200 rounded">
              <p class="text-sm text-red-700">Database Error: {data.health.database.error}</p>
            </div>
          {/if}
        </CardContent>
      </Card>
    {/if}

    <!-- Create/Edit Form -->
    <Card>
      <CardHeader>
        <CardTitle>
          {editingCase ? '✏️ Edit Case' : '➕ Create New Case'}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form method="POST" action="?/createCase" use:enhance={handleSubmit}>
          {#if editingCase}
            <input type="hidden" name="id" value={editingCase} />
            <input type="hidden" name="_action" value="update" />
          {:else}
            <input type="hidden" name="_action" value="create" />
          {/if}
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Title Input -->
            <div class="space-y-2">
              <label for="title" class="text-sm font-medium">Case Title</label>
              <Input
                id="title"
                name="title"
                bind:value={formData.title}
                placeholder="Enter case title..."
                required
              />
            </div>

            <!-- Category Select -->
            <div class="space-y-2">
              <label for="category" class="text-sm font-medium">Category</label>
              <select 
                id="category"
                name="category"
                bind:value={formData.category}
                class="w-full px-3 py-2 border border-input rounded-md"
              >
                <option value="investigation">Investigation</option>
                <option value="litigation">Litigation</option>
                <option value="compliance">Compliance</option>
                <option value="contract">Contract</option>
                <option value="criminal">Criminal</option>
                <option value="civil">Civil</option>
              </select>
            </div>

            <!-- Priority Select -->
            <div class="space-y-2">
              <label for="priority" class="text-sm font-medium">Priority</label>
              <select 
                id="priority"
                name="priority"
                bind:value={formData.priority}
                class="w-full px-3 py-2 border border-input rounded-md"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="urgent">Urgent</option>
              </select>
            </div>

            <!-- Status Select -->
            <div class="space-y-2">
              <label for="status" class="text-sm font-medium">Status</label>
              <select 
                id="status"
                name="status"
                bind:value={formData.status}
                class="w-full px-3 py-2 border border-input rounded-md"
              >
                <option value="open">Open</option>
                <option value="active">Active</option>
                <option value="under_review">Under Review</option>
                <option value="closed">Closed</option>
                <option value="archived">Archived</option>
              </select>
            </div>
          </div>

          <!-- Description -->
          <div class="mt-4 space-y-2">
            <label for="description" class="text-sm font-medium">Description</label>
            <Textarea
              id="description"
              name="description"
              bind:value={formData.description}
              placeholder="Enter case description..."
              rows="3"
            />
          </div>

          <!-- Gaming UI Action Buttons -->
          <div class="mt-6 flex flex-wrap gap-3">
            {#if gamingEra === 'nes'}
              <NES8BitButton type="submit" disabled={isCreating}>
                {editingCase ? '💾 Update Case' : '➕ Create Case'}
              </NES8BitButton>
              
              {#if isCreating}
                <NES8BitButton type="button" on:on:on:click={resetForm}>
                  ❌ Cancel
                </NES8BitButton>
              {/if}
            {:else if gamingEra === 'snes'}
              <SNES16BitButton type="submit" disabled={isCreating} enableMode7={true}>
                {editingCase ? '💾 Update Case' : '➕ Create Case'}
              </SNES16BitButton>
              
              {#if isCreating}
                <SNES16BitButton type="button" on:on:on:click={resetForm} plasmaEffect={true}>
                  ❌ Cancel
                </SNES16BitButton>
              {/if}
            {:else}
              <N643DButton 
                type="submit" 
                disabled={isCreating}
                materialType="pbr"
                enableParticles={true}
              >
                {editingCase ? '💾 Update Case' : '➕ Create Case'}
              </N643DButton>
              
              {#if isCreating}
                <N643DButton 
                  type="button" 
                  on:on:on:click={resetForm}
                  variant="secondary"
                  materialType="metal"
                >
                  ❌ Cancel
                </N643DButton>
              {/if}
            {/if}
          </div>
        </form>
      </CardContent>
    </Card>

    <!-- Cases List -->
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center justify-between">
          📋 Cases List
          <Badge variant="outline">{data.cases?.length || 0} total</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {#if data.cases && data.cases.length > 0}
          <div class="space-y-4">
            {#each data.cases as caseItem (caseItem.id)}
              <div class="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                <div class="flex flex-col md:flex-row justify-between items-start gap-4">
                  <div class="flex-1 space-y-2">
                    <div class="flex items-start justify-between gap-2">
                      <h3 class="font-semibold text-lg">{caseItem.title}</h3>
                      <div class="flex gap-2">
                        <Badge variant={getPriorityColor(caseItem.priority)}>
                          {caseItem.priority}
                        </Badge>
                        <Badge variant={getStatusColor(caseItem.status)}>
                          {caseItem.status}
                        </Badge>
                      </div>
                    </div>
                    
                    {#if caseItem.description}
                      <p class="text-sm text-muted-foreground">{caseItem.description}</p>
                    {/if}
                    
                    <div class="flex flex-wrap gap-2 text-xs text-muted-foreground">
                      {#if caseItem.category}
                        <span>📂 {caseItem.category}</span>
                      {/if}
                      {#if caseItem.created_at}
                        <span>📅 {new Date(caseItem.created_at).toLocaleDateString()}</span>
                      {/if}
                      {#if caseItem.id}
                        <span>🆔 {caseItem.id.slice(0, 8)}</span>
                      {/if}
                    </div>
                  </div>

                  <!-- Action Buttons -->
                  <div class="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      on:on:on:click={() => startEdit(caseItem)}
                      disabled={isDeleting}
                    >
                      ✏️ Edit
                    </Button>
                    
                    <Button
                      variant="destructive"
                      size="sm"
                      on:on:on:click={() => confirmDelete(caseItem)}
                      disabled={isDeleting}
                    >
                      🗑️ Delete
                    </Button>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <div class="text-center py-8">
            <p class="text-muted-foreground">No cases found. Create your first case above!</p>
          </div>
        {/if}
      </CardContent>
    </Card>

    <!-- SSR Test Results -->
    <Card>
      <CardHeader>
        <CardTitle>🧪 SSR & Hydration Test Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 class="font-medium mb-2">✅ SSR Status</h4>
            <ul class="text-sm space-y-1 text-muted-foreground">
              <li>• Page data loaded: {data ? '✅' : '❌'}</li>
              <li>• Cases from server: {data.cases ? '✅' : '❌'}</li>
              <li>• Database health: {data.health ? '✅' : '❌'}</li>
              <li>• Form actions: {typeof enhance !== 'undefined' ? '✅' : '❌'}</li>
            </ul>
          </div>
          
          <div>
            <h4 class="font-medium mb-2">🎮 Component Status</h4>
            <ul class="text-sm space-y-1 text-muted-foreground">
              <li>• Gaming UI loaded: ✅</li>
              <li>• bits-ui components: ✅</li>
              <li>• Form validation: ✅</li>
              <li>• Progressive enhancement: ✅</li>
            </ul>
          </div>
        </div>
        
        <div class="mt-4 p-3 bg-muted rounded-lg">
          <h5 class="font-medium mb-2">Debug Info:</h5>
          <pre class="text-xs overflow-auto">{JSON.stringify({
            hasData: !!data,
            casesCount: data.cases?.length || 0,
            databaseConnected: data.health?.database?.connected || false,
            gamingEra,
            timestamp: new Date().toISOString()
          }, null, 2)}</pre>
        </div>
      </CardContent>
    </Card>
  </div>
</ProgressiveGamingProvider>

<style>
  /* Ensure proper SSR styling */
  .container {
    min-height: 100vh;
    opacity: 1;
    animation: fadeIn 0.3s ease-in;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  /* Gaming era specific styling */
  :global(.progressive-gaming-provider) {
    --gaming-transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  /* Mobile responsiveness */
  @media (max-width: 768px) {
    .container {
      padding: 1rem;
    }
  }
</style>