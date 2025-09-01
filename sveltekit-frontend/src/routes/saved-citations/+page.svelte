<script lang="ts">
  import { run } from 'svelte/legacy';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { Button } from '$lib/components/ui/button';
  import { Card as CardRoot, CardContent, CardHeader } from '$lib/components/ui/card';
  import DialogContent from '$lib/components/ui/dialog/DialogContent.svelte';
  import DialogDescription from '$lib/components/ui/dialog/DialogDescription.svelte';
  import DialogFooter from '$lib/components/ui/dialog/DialogFooter.svelte';
  import DialogHeader from '$lib/components/ui/dialog/DialogHeader.svelte';
  import DialogRoot from '$lib/components/ui/dialog/DialogRoot.svelte';
  import DialogTitle from '$lib/components/ui/dialog/DialogTitle.svelte';
  import DropdownMenuContent from '$lib/components/ui/dropdown-menu/DropdownMenuContent.svelte';
  import DropdownMenuItem from '$lib/components/ui/dropdown-menu/DropdownMenuItem.svelte';
  import DropdownMenuRoot from '$lib/components/ui/dropdown-menu/DropdownMenuRoot.svelte';
  import DropdownMenuSeparator from '$lib/components/ui/dropdown-menu/DropdownMenuSeparator.svelte';
  import DropdownMenuTrigger from '$lib/components/ui/dropdown-menu/DropdownMenuTrigger.svelte';
  import Input from '$lib/components/ui/Input.svelte';
  import { Copy, Edit, MoreVertical, Plus, Search, Star, Tag, Trash2 } from 'lucide-svelte';
  import { onMount } from 'svelte';

  import type { Citation } from '$lib/types/api';

  let editingCitation: Citation | null = $state(null);
  let searchQuery = $state('');
  let selectedCategory = $state('all');
  let showAddDialog = $state(false);
  let filteredCitations: Citation[] = $state([]);
  let savedCitations: Citation[] = $state([]);

  // Initialize with sample data
  onMount(() => {
    savedCitations = [
      {
        id: '1',
        title: 'Miranda Rights',
        content: 'The defendant must be clearly informed of their rights...',
        source: 'Miranda v. Arizona, 384 U.S. 436 (1966)',
        category: 'case-law',
        tags: ['constitutional', 'police-procedure', 'rights'],
        createdAt: new Date(),
        updatedAt: new Date(),
        isFavorite: true,
        notes: 'Landmark case establishing Miranda warnings',
      },
    ];
  });

  // New citation form
  let newCitation = $state({
    title: '',
    content: '',
    source: '',
    category: 'general',
    tags: '',
    notes: '',
  });

  // Categories for filtering
  const categories = [
    { value: 'all', label: 'All Categories' },
    { value: 'general', label: 'General' },
    { value: 'constitutional', label: 'Constitutional Law' },
    { value: 'case-law', label: 'Case Law' },
    { value: 'statutes', label: 'Statutes' },
    { value: 'evidence', label: 'Evidence' },
    { value: 'report-citations', label: 'From Reports' },
  ];

  // Reactive filtering with Fuse.js-like search
  run(() => {
    filteredCitations = savedCitations.filter((citation) => {
      const matchesSearch =
        searchQuery === '' ||
        citation.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        citation.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        citation.source.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (citation.notes && citation.notes.toLowerCase().includes(searchQuery.toLowerCase())) ||
        citation.tags.some((tag: string) => tag.toLowerCase().includes(searchQuery.toLowerCase()));

      const matchesCategory = selectedCategory === 'all' || citation.category === selectedCategory;

      return matchesSearch && matchesCategory;
    });
  });
  async function saveCitation() {
    try {
      const citation = {
        ...newCitation,
        tags: newCitation.tags
          .split(',')
          .map((tag) => tag.trim())
          .filter((tag) => tag.length > 0),
        id: crypto.randomUUID(),
        isFavorite: false,
        savedAt: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      // In a real app, this would POST to /api/user/saved-citations
      savedCitations = [...savedCitations, citation];

      // Reset form
      newCitation = {
        title: '',
        content: '',
        source: '',
        category: 'general',
        tags: '',
        notes: '',
      };

      showAddDialog = false;
    } catch (error) {
      console.error('Error saving citation:', error);
    }
  }
  async function deleteCitation(citationId: string) {
    try {
      // In a real app, this would DELETE /api/user/saved-citations/{id}
      savedCitations = savedCitations.filter((c) => c.id !== citationId);
    } catch (error) {
      console.error('Error deleting citation:', error);
    }
  }
  async function toggleFavorite(citation: any) {
    try {
      citation.isFavorite = !citation.isFavorite;
      // In a real app, this would PATCH /api/user/saved-citations/{id}
      savedCitations = [...savedCitations];
    } catch (error) {
      console.error('Error updating citation:', error);
    }
  }
  function copyCitation(citation: any) {
    const citationText = `${citation.content}\n\nSource: ${citation.source}`;
    navigator.clipboard.writeText(citationText);
  }
  function editCitation(citation: any) {
    editingCitation = { ...citation };
    editingCitation.tags = citation.tags.join(', ');
  }
  async function updateCitation() {
    try {
      const updated = {
        ...editingCitation,
        tags: Array.isArray(editingCitation.tags)
          ? editingCitation.tags
          : (editingCitation.tags as any as string)
              ?.split(',')
              .map((tag: string) => tag.trim())
              .filter((tag: string) => tag.length > 0),
      };

      const index = savedCitations.findIndex((c) => c.id === updated.id);
      if (index >= 0) {
        savedCitations[index] = updated;
        savedCitations = [...savedCitations];
      }
      editingCitation = null;
    } catch (error) {
      console.error('Error updating citation:', error);
    }
  }
  // Stats
  let totalCitations = $derived(savedCitations.length);
  let favoriteCitations = $derived(savedCitations.filter((c) => c.isFavorite).length);
  let categoryCounts = $derived(
    savedCitations.reduce((acc, citation) => {
      acc[citation.category] = (acc[citation.category] || 0) + 1;
      return acc;
    }, {})
  );
</script>

<svelte:head>
  <title>Saved Citations - Legal AI Assistant</title>
</svelte:head>

<div class="space-y-4">
  <!-- Header -->
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <h1 class="space-y-4">Saved Citations</h1>
        <p class="space-y-4">Manage your collection of legal citations and references</p>
      </div>

      <div class="space-y-4">
        <div class="space-y-4">
          <span class="space-y-4">{totalCitations}</span>
          <span class="space-y-4">Total</span>
        </div>
        <div class="space-y-4">
          <span class="space-y-4">{favoriteCitations}</span>
          <span class="space-y-4">Favorites</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Toolbar -->
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <Search class="space-y-4" />
        <Input
          type="text"
          placeholder="Search citations..."
          bind:value={searchQuery}
          class="space-y-4" />
      </div>

      <select bind:value={selectedCategory} class="space-y-4">
        {#each categories as category}
          <option value={category.value}>{category.label}</option>
        {/each}
      </select>
    </div>

    <div class="space-y-4">
      <Button on:on:click={() => (showAddDialog = true)}>
        <Plus class="space-y-4" />
        Add Citation
      </Button>
    </div>
  </div>

  <!-- Citations Grid -->
  <div class="space-y-4">
    {#each filteredCitations as citation (citation.id)}
      <CardRoot class="citation-card">
        <CardHeader class="citation-header">
          <div class="space-y-4">
            <h3 class="space-y-4">{citation.title}</h3>

            <DropdownMenuRoot>
              {#snippet children({ trigger, menu })}
                <DropdownMenuTrigger {trigger}>
                  <Button variant="ghost" size="sm">
                    <MoreVertical class="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent {menu}>
                  <DropdownMenuItem on:on:click={() => toggleFavorite(citation)}>
                    <Star class="w-4 h-4 mr-2" />
                    {citation.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
                  </DropdownMenuItem>
                  <DropdownMenuItem on:on:click={() => copyCitation(citation)}>
                    <Copy class="w-4 h-4 mr-2" />
                    Copy citation
                  </DropdownMenuItem>
                  <DropdownMenuItem on:on:click={() => editCitation(citation)}>
                    <Edit class="w-4 h-4 mr-2" />
                    Edit
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    on:on:click={() => deleteCitation(citation.id)}
                    class="text-destructive">
                    <Trash2 class="w-4 h-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              {/snippet}
            </DropdownMenuRoot>
          </div>

          <div class="space-y-4">
            <Badge variant="secondary" class="mr-2">
              {citation.category}
            </Badge>
            {#if citation.isFavorite}
              <Badge variant="secondary" class="mr-2">
                <Star class="space-y-4" />
                Favorite
              </Badge>
            {/if}
          </div>
        </CardHeader>

        <CardContent class="citation-content">
          <p class="space-y-4">{citation.content}</p>
          <p class="space-y-4">Source: {citation.source}</p>

          {#if citation.notes}
            <div class="space-y-4">
              <p>{citation.notes}</p>
            </div>
          {/if}

          {#if citation.tags.length > 0}
            <div class="space-y-4">
              {#each citation.tags as tag}
                <Badge variant="secondary" class="mr-2">
                  <Tag class="w-3 h-3 mr-1" />
                  {tag}
                </Badge>
              {/each}
            </div>
          {/if}

          <div class="space-y-4">
            <span class="space-y-4">
              Saved {new Date(citation.savedAt).toLocaleDateString()}
            </span>

            {#if citation.contextData?.caseId}
              <Badge variant="secondary" class="mr-2">
                Case: {citation.contextData.caseId}
              </Badge>
            {/if}
          </div>
        </CardContent>
      </CardRoot>
    {/each}

    {#if filteredCitations.length === 0}
      <div class="space-y-4">
        <div class="space-y-4">
          {#if searchQuery || selectedCategory !== 'all'}
            <h3 class="space-y-4">No citations found</h3>
            <p class="space-y-4">No citations match your current search criteria.</p>
            <Button
              variant="secondary"
              on:on:click={() => {
                searchQuery = '';
                selectedCategory = 'all';
              }}>
              Clear filters
            </Button>
          {:else}
            <h3 class="space-y-4">No saved citations</h3>
            <p class="space-y-4">
              You haven't saved any citations yet. Start by adding citations from reports or create
              new ones.
            </p>
            <Button on:on:click={() => (showAddDialog = true)}>
              <Plus class="space-y-4" />
              Add your first citation
            </Button>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

<!-- Add Citation Dialog -->
<DialogRoot open={showAddDialog} openchange={(open) => showAddDialog = open}>
  <DialogContent class="sm:max-w-[425px]" overlay={{}} content={{}} openState={showAddDialog}>
    <DialogHeader>
      <DialogTitle title="Add New Citation" />
      <DialogDescription description="Create a new citation to save for future reference." />
    </DialogHeader>

    <div class="space-y-4">
      <div class="space-y-4">
        <label for="title">Title</label>
        <Input id="title" bind:value={newCitation.title} placeholder="Citation title" />
      </div>

      <div class="space-y-4">
        <label for="content">Content</label>
        <textarea
          id="content"
          bind:value={newCitation.content}
          placeholder="Citation text or quote"
          rows="4"></textarea>
      </div>

      <div class="space-y-4">
        <label for="source">Source</label>
        <Input id="source" bind:value={newCitation.source} placeholder="Source reference" />
      </div>

      <div class="space-y-4">
        <div class="space-y-4">
          <label for="category">Category</label>
          <select id="category" bind:value={newCitation.category}>
            {#each categories.slice(1) as category}
              <option value={category.value}>{category.label}</option>
            {/each}
          </select>
        </div>

        <div class="space-y-4">
          <label for="tags">Tags</label>
          <Input id="tags" bind:value={newCitation.tags} placeholder="tag1, tag2, tag3" />
        </div>
      </div>

      <div class="space-y-4">
        <label for="notes">Notes (optional)</label>
        <textarea
          id="notes"
          bind:value={newCitation.notes}
          placeholder="Personal notes about this citation"
          rows="4"></textarea>
      </div>
    </div>

    <DialogFooter>
      <Button variant="secondary" on:on:click={() => (showAddDialog = false)}>Cancel</Button>
      <Button on:on:click={() => saveCitation()} disabled={!newCitation.title || !newCitation.content}>
        Save Citation
      </Button>
    </DialogFooter>
  </DialogContent>
</DialogRoot>

<!-- Edit Citation Dialog -->
{#if editingCitation}
  <DialogRoot open={true} openchange={() => (editingCitation = null)}>
    <DialogContent class="sm:max-w-[425px]" overlay={{}} content={{}} openState={true}>
      <DialogHeader>
        <DialogTitle title="Edit Citation" />
      </DialogHeader>

      <div class="space-y-4">
        <div class="space-y-4">
          <label for="edit-title">Title</label>
          <Input id="edit-title" bind:value={editingCitation.title} />
        </div>

        <div class="space-y-4">
          <label for="edit-content">Content</label>
          <textarea id="edit-content" bind:value={editingCitation.content} rows="4"></textarea>
        </div>

        <div class="space-y-4">
          <label for="edit-source">Source</label>
          <Input id="edit-source" bind:value={editingCitation.source} />
        </div>

        <div class="space-y-4">
          <div class="space-y-4">
            <label for="edit-category">Category</label>
            <select id="edit-category" bind:value={editingCitation.category}>
              {#each categories.slice(1) as category}
                <option value={category.value}>{category.label}</option>
              {/each}
            </select>
          </div>

          <div class="space-y-4">
            <label for="edit-tags">Tags</label>
            <Input id="edit-tags" bind:value={editingCitation.tags} />
          </div>
        </div>

        <div class="space-y-4">
          <label for="edit-notes">Notes</label>
          <textarea id="edit-notes" bind:value={editingCitation.notes} rows="4"></textarea>
        </div>
      </div>

      <DialogFooter>
        <Button variant="secondary" on:on:click={() => (editingCitation = null)}>Cancel</Button>
        <Button on:on:click={() => updateCitation()}>Update Citation</Button>
      </DialogFooter>
    </DialogContent>
  </DialogRoot>
{/if}

<style>
  /* @unocss-include */
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
</style>
