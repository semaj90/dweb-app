<script lang="ts">
  import Badge from "$lib/components/ui/Badge.svelte";
  import { Button } from "$lib/components/ui/button";
  import CardRoot from "$lib/components/ui/Card.svelte";
  import CardContent from "$lib/components/ui/CardContent.svelte";
  import CardHeader from "$lib/components/ui/CardHeader.svelte";
  import DialogContent from "$lib/components/ui/dialog/DialogContent.svelte";
  import DialogDescription from "$lib/components/ui/dialog/DialogDescription.svelte";
  import DialogFooter from "$lib/components/ui/dialog/DialogFooter.svelte";
  import DialogHeader from "$lib/components/ui/dialog/DialogHeader.svelte";
  import DialogRoot from "$lib/components/ui/dialog/DialogRoot.svelte";
  import DialogTitle from "$lib/components/ui/dialog/DialogTitle.svelte";
  import DropdownMenuContent from "$lib/components/ui/dropdown-menu/DropdownMenuContent.svelte";
  import DropdownMenuItem from "$lib/components/ui/dropdown-menu/DropdownMenuItem.svelte";
  import DropdownMenuRoot from "$lib/components/ui/dropdown-menu/DropdownMenuRoot.svelte";
  import DropdownMenuSeparator from "$lib/components/ui/dropdown-menu/DropdownMenuSeparator.svelte";
  import DropdownMenuTrigger from "$lib/components/ui/dropdown-menu/DropdownMenuTrigger.svelte";
  import Input from "$lib/components/ui/Input.svelte";
  import {
    Copy,
    Edit,
    MoreVertical,
    Plus,
    Search,
    Star,
    Tag,
    Trash2,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  import type { Citation } from "$lib/types/api";
  
  let editingCitation: Citation | null = null;
  let searchQuery = "";
  let selectedCategory = "all";
  let showAddDialog = false;
  let filteredCitations: Citation[] = [];
  let savedCitations: Citation[] = [];

  // Initialize with sample data
  onMount(() => {
    savedCitations = [
      {
        id: "1",
        title: "Miranda Rights",
        content: "The defendant must be clearly informed of their rights...",
        source: "Miranda v. Arizona, 384 U.S. 436 (1966)",
        category: "case-law",
        tags: ["constitutional", "police-procedure", "rights"],
        createdAt: new Date(),
        updatedAt: new Date(),
        isFavorite: true,
        notes: "Landmark case establishing Miranda warnings"
}
    ];
  });

  // New citation form
  let newCitation = {
    title: "",
    content: "",
    source: "",
    category: "general",
    tags: "",
    notes: "",
  };

  // Categories for filtering
  const categories = [
    { value: "all", label: "All Categories" },
    { value: "general", label: "General" },
    { value: "constitutional", label: "Constitutional Law" },
    { value: "case-law", label: "Case Law" },
    { value: "statutes", label: "Statutes" },
    { value: "evidence", label: "Evidence" },
    { value: "report-citations", label: "From Reports" },
  ];

  // Reactive filtering with Fuse.js-like search
  $: {
    filteredCitations = savedCitations.filter((citation) => {
      const matchesSearch =
        searchQuery === "" ||
        citation.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        citation.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        citation.source.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (citation.notes &&
          citation.notes.toLowerCase().includes(searchQuery.toLowerCase())) ||
        citation.tags.some((tag: string) =>
          tag.toLowerCase().includes(searchQuery.toLowerCase())
        );

      const matchesCategory =
        selectedCategory === "all" || citation.category === selectedCategory;

      return matchesSearch && matchesCategory;
    });
}
  async function saveCitation() {
    try {
      const citation = {
        ...newCitation,
        tags: newCitation.tags
          .split(",")
          .map((tag) => tag.trim())
          .filter((tag) => tag.length > 0),
        id: crypto.randomUUID(),
        isFavorite: false,
        savedAt: new Date(),
      };

      // In a real app, this would POST to /api/user/saved-citations
      savedCitations = [...savedCitations, citation];

      // Reset form
      newCitation = {
        title: "",
        content: "",
        source: "",
        category: "general",
        tags: "",
        notes: "",
      };

      showAddDialog = false;
    } catch (error) {
      console.error("Error saving citation:", error);
}}
  async function deleteCitation(citationId: string) {
    try {
      // In a real app, this would DELETE /api/user/saved-citations/{id}
      savedCitations = savedCitations.filter((c) => c.id !== citationId);
    } catch (error) {
      console.error("Error deleting citation:", error);
}}
  async function toggleFavorite(citation: any) {
    try {
      citation.isFavorite = !citation.isFavorite;
      // In a real app, this would PATCH /api/user/saved-citations/{id}
      savedCitations = [...savedCitations];
    } catch (error) {
      console.error("Error updating citation:", error);
}}
  function copyCitation(citation: any) {
    const citationText = `${citation.content}\n\nSource: ${citation.source}`;
    navigator.clipboard.writeText(citationText);
}
  function editCitation(citation: any) {
    editingCitation = { ...citation };
    editingCitation.tags = citation.tags.join(", ");
}
  async function updateCitation() {
    try {
      const updated = {
        ...editingCitation,
        tags: editingCitation.tags
          .split(",")
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
      console.error("Error updating citation:", error);
}}
  // Stats
  $: totalCitations = savedCitations.length;
  $: favoriteCitations = savedCitations.filter((c) => c.isFavorite).length;
  $: categoryCounts = savedCitations.reduce((acc, citation) => {
    acc[citation.category] = (acc[citation.category] || 0) + 1;
    return acc;
  }, {});
</script>

<svelte:head>
  <title>Saved Citations - Legal AI Assistant</title>
</svelte:head>

<div class="container mx-auto px-4">
  <!-- Header -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <h1 class="container mx-auto px-4">Saved Citations</h1>
        <p class="container mx-auto px-4">
          Manage your collection of legal citations and references
        </p>
      </div>

      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <span class="container mx-auto px-4">{totalCitations}</span>
          <span class="container mx-auto px-4">Total</span>
        </div>
        <div class="container mx-auto px-4">
          <span class="container mx-auto px-4">{favoriteCitations}</span>
          <span class="container mx-auto px-4">Favorites</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Toolbar -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <Search class="container mx-auto px-4" />
        <Input
          type="text"
          placeholder="Search citations..."
          bind:value={searchQuery}
          class="container mx-auto px-4"
        />
      </div>

      <select bind:value={selectedCategory} class="container mx-auto px-4">
        {#each categories as category}
          <option value={category.value}>{category.label}</option>
        {/each}
      </select>
    </div>

    <div class="container mx-auto px-4">
      <Button on:click={() => (showAddDialog = true)}>
        <Plus class="container mx-auto px-4" />
        Add Citation
      </Button>
    </div>
  </div>

  <!-- Citations Grid -->
  <div class="container mx-auto px-4">
    {#each filteredCitations as citation (citation.id)}
      <CardRoot class="citation-card">
        <CardHeader class="citation-header">
          <div class="container mx-auto px-4">
            <h3 class="container mx-auto px-4">{citation.title}</h3>

            <DropdownMenuRoot let:trigger let:menu>
              <DropdownMenuTrigger {trigger}>
                <Button variant="ghost" size="sm">
                  <MoreVertical class="container mx-auto px-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent {menu}>
                <DropdownMenuItem on:click={() => toggleFavorite(citation)}>
                  <Star class="container mx-auto px-4" />
                  {citation.isFavorite
                    ? "Remove from favorites"
                    : "Add to favorites"}
                </DropdownMenuItem>
                <DropdownMenuItem on:click={() => copyCitation(citation)}>
                  <Copy class="container mx-auto px-4" />
                  Copy citation
                </DropdownMenuItem>
                <DropdownMenuItem on:click={() => editCitation(citation)}>
                  <Edit class="container mx-auto px-4" />
                  Edit
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  on:click={() => deleteCitation(citation.id)}
                  class="container mx-auto px-4"
                >
                  <Trash2 class="container mx-auto px-4" />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenuRoot>
          </div>

          <div class="container mx-auto px-4">
            <Badge variant="secondary" class="container mx-auto px-4">
              {citation.category}
            </Badge>
            {#if citation.isFavorite}
              <Badge variant="secondary" class="container mx-auto px-4">
                <Star class="container mx-auto px-4" />
                Favorite
              </Badge>
            {/if}
          </div>
        </CardHeader>

        <CardContent class="citation-content">
          <p class="container mx-auto px-4">{citation.content}</p>
          <p class="container mx-auto px-4">Source: {citation.source}</p>

          {#if citation.notes}
            <div class="container mx-auto px-4">
              <p>{citation.notes}</p>
            </div>
          {/if}

          {#if citation.tags.length > 0}
            <div class="container mx-auto px-4">
              {#each citation.tags as tag}
                <Badge variant="secondary" class="container mx-auto px-4">
                  <Tag class="container mx-auto px-4" />
                  {tag}
                </Badge>
              {/each}
            </div>
          {/if}

          <div class="container mx-auto px-4">
            <span class="container mx-auto px-4">
              Saved {new Date(citation.savedAt).toLocaleDateString()}
            </span>

            {#if citation.contextData?.caseId}
              <Badge variant="secondary" class="container mx-auto px-4">
                Case: {citation.contextData.caseId}
              </Badge>
            {/if}
          </div>
        </CardContent>
      </CardRoot>
    {/each}

    {#if filteredCitations.length === 0}
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          {#if searchQuery || selectedCategory !== "all"}
            <h3 class="container mx-auto px-4">No citations found</h3>
            <p class="container mx-auto px-4">
              No citations match your current search criteria.
            </p>
            <Button
              variant="secondary"
              on:click={() => {
                searchQuery = "";
                selectedCategory = "all";
              }}
            >
              Clear filters
            </Button>
          {:else}
            <h3 class="container mx-auto px-4">No saved citations</h3>
            <p class="container mx-auto px-4">
              You haven't saved any citations yet. Start by adding citations
              from reports or create new ones.
            </p>
            <Button on:click={() => (showAddDialog = true)}>
              <Plus class="container mx-auto px-4" />
              Add your first citation
            </Button>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

<!-- Add Citation Dialog -->
<DialogRoot bind:open={showAddDialog}>
  <DialogContent
    overlay={() => {"
    content={() => {"
    openState={{ subscribe: () => () => {} "
    class="container mx-auto px-4"
  >
    <DialogHeader>
      <DialogTitle title="Add New Citation" />
      <DialogDescription
        description="Create a new citation to save for future reference."
      />
    </DialogHeader>

    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <label for="title">Title</label>
        <Input
          id="title"
          bind:value={newCitation.title}
          placeholder="Citation title"
        />
      </div>

      <div class="container mx-auto px-4">
        <label for="content">Content</label>
        <textarea
          id="content"
          bind:value={newCitation.content}
          placeholder="Citation text or quote"
          rows="4"
        ></textarea>
      </div>

      <div class="container mx-auto px-4">
        <label for="source">Source</label>
        <Input
          id="source"
          bind:value={newCitation.source}
          placeholder="Source reference"
        />
      </div>

      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <label for="category">Category</label>
          <select id="category" bind:value={newCitation.category}>
            {#each categories.slice(1) as category}
              <option value={category.value}>{category.label}</option>
            {/each}
          </select>
        </div>

        <div class="container mx-auto px-4">
          <label for="tags">Tags</label>
          <Input
            id="tags"
            bind:value={newCitation.tags}
            placeholder="tag1, tag2, tag3"
          />
        </div>
      </div>

      <div class="container mx-auto px-4">
        <label for="notes">Notes (optional)</label>
        <textarea
          id="notes"
          bind:value={newCitation.notes}
          placeholder="Personal notes about this citation"
          rows="4"
        ></textarea>
      </div>
    </div>

    <DialogFooter>
      <Button variant="secondary" on:click={() => (showAddDialog = false)}
        >Cancel</Button
      >
      <Button
        on:click={() => saveCitation()}
        disabled={!newCitation.title || !newCitation.content}
      >
        Save Citation
      </Button>
    </DialogFooter>
  </DialogContent>
</DialogRoot>

<!-- Edit Citation Dialog -->
{#if editingCitation}
  <DialogRoot open={true} onOpenChange={() => (editingCitation = null)}>
    <DialogContent
      overlay={() => {"
      content={() => {"
      openState={{ subscribe: () => () => {} "
      class="container mx-auto px-4"
    >
      <DialogHeader>
        <DialogTitle title="Edit Citation" />
      </DialogHeader>

      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <label for="edit-title">Title</label>
          <Input id="edit-title" bind:value={editingCitation.title} />
        </div>

        <div class="container mx-auto px-4">
          <label for="edit-content">Content</label>
          <textarea
            id="edit-content"
            bind:value={editingCitation.content}
            rows="4"
          ></textarea>
        </div>

        <div class="container mx-auto px-4">
          <label for="edit-source">Source</label>
          <Input id="edit-source" bind:value={editingCitation.source} />
        </div>

        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">
            <label for="edit-category">Category</label>
            <select id="edit-category" bind:value={editingCitation.category}>
              {#each categories.slice(1) as category}
                <option value={category.value}>{category.label}</option>
              {/each}
            </select>
          </div>

          <div class="container mx-auto px-4">
            <label for="edit-tags">Tags</label>
            <Input id="edit-tags" bind:value={editingCitation.tags} />
          </div>
        </div>

        <div class="container mx-auto px-4">
          <label for="edit-notes">Notes</label>
          <textarea id="edit-notes" bind:value={editingCitation.notes} rows="4"
          ></textarea>
        </div>
      </div>

      <DialogFooter>
        <Button variant="secondary" on:click={() => (editingCitation = null)}
          >Cancel</Button
        >
        <Button on:click={() => updateCitation()}>Update Citation</Button>
      </DialogFooter>
    </DialogContent>
  </DialogRoot>
{/if}

<style>
  /* @unocss-include */
  .saved-citations-page {
    min-height: 100vh;
    background: #f8fafc;
}
  .page-header {
    background: white;
    border-bottom: 1px solid #e5e7eb;
    padding: 32px 24px;
}
  .header-content {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
  .page-title {
    font-size: 28px;
    font-weight: 700;
    color: #1f2937;
    margin: 0 0 8px 0;
}
  .page-subtitle {
    font-size: 16px;
    color: #6b7280;
    margin: 0;
}
  .header-stats {
    display: flex;
    gap: 24px;
}
  .stat {
    text-align: center;
}
  .stat-number {
    display: block;
    font-size: 24px;
    font-weight: 700;
    color: #1f2937;
}
  .stat-label {
    font-size: 12px;
    color: #6b7280;
    text-transform: uppercase;
    font-weight: 500;
}
  .toolbar {
    background: white;
    border-bottom: 1px solid #e5e7eb;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
  .toolbar-left {
    display: flex;
    gap: 16px;
    align-items: center;
}
  .toolbar-right {
    display: flex;
    gap: 8px;
}
  .search-container {
    position: relative;
    width: 300px;
}
  :global(.search-input) {
    padding-left: 40px !important;
}
  .category-filter {
    padding: 8px 12px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: white;
    font-size: 14px;
    min-width: 150px;
}
  .citations-grid {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 20px;
}
  :global(.citation-card) {
    transition: all 0.2s ease;
    height: fit-content;
}
  :global(.citation-card:hover) {
    box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}
  :global(.citation-header) {
    padding-bottom: 12px !important;
}
  .citation-title-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}
  .citation-title {
    font-size: 16px;
    font-weight: 600;
    color: #1f2937;
    margin: 0;
    flex: 1;
    padding-right: 8px;
}
  .citation-meta {
    display: flex;
    gap: 8px;
    align-items: center;
}
  :global(.category-badge),
  :global(.favorite-badge),
  :global(.context-badge) {
    font-size: 10px !important;
    padding: 2px 6px !important;
    height: auto !important;
}
  :global(.favorite-badge) {
    background: #fef3c7 !important;
    color: #92400e !important;
}
  :global(.citation-content) {
    padding-top: 0 !important;
}
  .citation-text {
    font-size: 14px;
    color: #374151;
    line-height: 1.5;
    margin: 0 0 12px 0;
}
  .citation-source {
    font-size: 12px;
    color: #6b7280;
    font-style: italic;
    margin: 0 0 12px 0;
}
  .citation-notes {
    background: #f3f4f6;
    padding: 12px;
    border-radius: 6px;
    margin: 12px 0;
}
  .citation-notes p {
    font-size: 12px;
    color: #4b5563;
    margin: 0;
    line-height: 1.4;
}
  .citation-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 12px 0;
}
  :global(.tag) {
    font-size: 10px !important;
    padding: 2px 6px !important;
    height: auto !important;
    background: #e0e7ff !important;
    color: #3730a3 !important;
}
  .citation-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #f3f4f6;
}
  .saved-date {
    font-size: 11px;
    color: #9ca3af;
}
  .empty-state {
    grid-column: 1 / -1;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 400px;
}
  .empty-content {
    text-align: center;
    max-width: 400px;
}
  .empty-title {
    font-size: 18px;
    font-weight: 600;
    color: #374151;
    margin: 0 0 8px 0;
}
  .empty-message {
    font-size: 14px;
    color: #6b7280;
    margin: 0 0 16px 0;
    line-height: 1.5;
}
  .citation-form {
    display: flex;
    flex-direction: column;
    gap: 16px;
}
  .form-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
  .form-field label {
    font-size: 14px;
    font-weight: 500;
    color: #374151;
}
  .form-field textarea {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 14px;
    resize: vertical;
    min-height: 80px;
}
  .form-field select {
    padding: 8px 12px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 14px;
}
  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
</style>
