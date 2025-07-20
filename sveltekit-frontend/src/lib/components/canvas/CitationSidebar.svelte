<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import type { Citation } from "$lib/types/api";
  import { Copy, Search, Star, Tag, Trash2 } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";
  import Badge from '$lib/components/ui/Badge.svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Input from '$lib/components/ui/Input.svelte';

  export let citations: Citation[] = [];

  const dispatch = createEventDispatcher();

  let searchQuery = "";
  let selectedCategory = "all";
  let filteredCitations: Citation[] = [];

  // Categories for filtering
  const categories = [
    { value: "all", label: "All Citations" },
    { value: "general", label: "General" },
    { value: "report-citations", label: "From Reports" },
    { value: "statutes", label: "Statutes" },
    { value: "case-law", label: "Case Law" },
    { value: "evidence", label: "Evidence" },
  ];

  // Reactive filtering
  $: {
    filteredCitations = citations.filter((citation) => {
      const matchesSearch =
        searchQuery === "" ||
        citation.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        citation.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        citation.source.toLowerCase().includes(searchQuery.toLowerCase()) ||
        citation.tags.some((tag) =>
          tag.toLowerCase().includes(searchQuery.toLowerCase())
        );

      const matchesCategory =
        selectedCategory === "all" || citation.category === selectedCategory;

      return matchesSearch && matchesCategory;
    });
}
  function selectCitation(citation: Citation) {
    dispatch("citationSelected", citation);
}
  function deleteCitation(citation: Citation) {
    dispatch("deleteCitation", citation);
}
  function copyCitation(citation: Citation) {
    const citationText = `${citation.content}\n\nSource: ${citation.source}`;
    navigator.clipboard.writeText(citationText);
}
  function toggleFavorite(citation: Citation) {
    // Update favorite status - in real app, this would update the database
    citation.isFavorite = !citation.isFavorite;
    dispatch("updateCitation", citation);
}
  // Drag and drop functionality
  function handleDragStart(event: DragEvent, citation: Citation) {
    if (event.dataTransfer) {
      event.dataTransfer.setData("text/plain", citation.content);
      event.dataTransfer.setData("application/json", JSON.stringify(citation));
      event.dataTransfer.effectAllowed = "copy";
}}
</script>

<div class="container mx-auto px-4">
  <div class="container mx-auto px-4">
    <h2 class="container mx-auto px-4">Saved Citations</h2>
    <p class="container mx-auto px-4">
      {filteredCitations.length} of {citations.length} citations
    </p>
  </div>

  <!-- Search and Filters -->
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

  <!-- Citations List -->
  <div class="container mx-auto px-4">
    {#each filteredCitations as citation (citation.id)}
      <Card class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">
            <h3 class="container mx-auto px-4">{citation.title}</h3>
            <div class="container mx-auto px-4">
              <Button
                variant="ghost"
                size="sm"
                on:click={() => toggleFavorite(citation)}
                class="container mx-auto px-4"
              >
                <Star class="container mx-auto px-4" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                on:click={() => copyCitation(citation)}
                title="Copy citation"
              >
                <Copy class="container mx-auto px-4" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                on:click={() => deleteCitation(citation)}
                title="Delete citation"
                class="container mx-auto px-4"
              >
                <Trash2 class="container mx-auto px-4" />
              </Button>
            </div>
          </div>

          <div class="container mx-auto px-4">
            <p class="container mx-auto px-4">{citation.content}</p>
            <p class="container mx-auto px-4">Source: {citation.source}</p>

            {#if citation.notes}
              <p class="container mx-auto px-4">Notes: {citation.notes}</p>
            {/if}
          </div>

          <!-- Tags -->
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

          <!-- Drag Handle -->
          <div
            class="container mx-auto px-4"
            draggable={true}
            role="button"
            tabindex={0}
            on:dragstart={(e) => handleDragStart(e, citation)}
            on:keydown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                // For keyboard users, trigger a click event instead
                e.currentTarget.click();
}
            "
            title="Drag to insert into report"
          >
            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4"></div>
              <div class="container mx-auto px-4"></div>
              <div class="container mx-auto px-4"></div>
            </div>
            <span class="container mx-auto px-4">Drag to report</span>
          </div>

          <div class="container mx-auto px-4">
            <span class="container mx-auto px-4">
              Saved {new Date(citation.savedAt).toLocaleDateString()}
            </span>
            <Badge variant="secondary" class="container mx-auto px-4">
              {citation.category}
            </Badge>
          </div>
        </div>
      </Card>
    {/each}

    {#if filteredCitations.length === 0}
      <div class="container mx-auto px-4">
        {#if searchQuery || selectedCategory !== "all"}
          <p class="container mx-auto px-4">No citations match your search criteria.</p>
          <Button
            variant="secondary"
            size="sm"
            on:click={() => {
              searchQuery = "";
              selectedCategory = "all";
            "
          >
            Clear filters
          </Button>
        {:else}
          <p class="container mx-auto px-4">No saved citations yet.</p>
          <p class="container mx-auto px-4">
            Right-click on text in reports to save citations, or add them from
            the search results.
          </p>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* @unocss-include */
  .citation-sidebar {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: white;
}
  .sidebar-header {
    padding: 24px 24px 16px;
    border-bottom: 1px solid #e5e7eb;
}
  .sidebar-title {
    font-size: 18px;
    font-weight: 600;
    color: #1f2937;
    margin: 0 0 4px 0;
}
  .sidebar-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin: 0;
}
  .search-section {
    padding: 16px 24px;
    border-bottom: 1px solid #e5e7eb;
    background: #fafafa;
}
  .search-input-container {
    position: relative;
    margin-bottom: 12px;
}
  :global(.search-input) {
    padding-left: 40px !important;
}
  .category-select {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: white;
    font-size: 14px;
    color: #374151;
    outline: none;
}
  .category-select:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
  .citations-list {
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px;
}
  :global(.citation-card) {
    margin-bottom: 16px;
    transition: box-shadow 0.2s ease;
    cursor: pointer;
}
  :global(.citation-card:hover) {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
  :global(.citation-content) {
    padding: 16px !important;
}
  .citation-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}
  .citation-title {
    font-size: 14px;
    font-weight: 600;
    color: #1f2937;
    margin: 0;
    flex: 1;
    padding-right: 8px;
}
  .citation-actions {
    display: flex;
    gap: 4px;
    opacity: 0;
    transition: opacity 0.2s ease;
}
  :global(.citation-card:hover .citation-actions) {
    opacity: 1;
}
  :global(.favorite-btn.favorited) {
    color: #f59e0b !important;
}
  :global(.delete-btn:hover) {
    color: #dc2626 !important;
}
  .citation-body {
    margin-bottom: 12px;
}
  .citation-text {
    font-size: 13px;
    color: #374151;
    line-height: 1.5;
    margin: 0 0 8px 0;
}
  .citation-source {
    font-size: 12px;
    color: #6b7280;
    font-style: italic;
    margin: 0 0 8px 0;
}
  .citation-notes {
    font-size: 12px;
    color: #4b5563;
    background: #f3f4f6;
    padding: 8px;
    border-radius: 4px;
    margin: 8px 0 0 0;
}
  .citation-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
}
  :global(.tag) {
    font-size: 11px !important;
    padding: 2px 6px !important;
    height: auto !important;
}
  .drag-handle {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 4px;
    cursor: grab;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
  .drag-handle:hover {
    background: #e2e8f0;
    border-color: #94a3b8;
}
  .drag-handle:active {
    cursor: grabbing;
}
  .drag-indicator {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
  .drag-line {
    width: 12px;
    height: 2px;
    background: #94a3b8;
    border-radius: 1px;
}
  .drag-text {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
}
  .citation-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: #9ca3af;
}
  .saved-date {
    font-size: 11px;
    color: #9ca3af;
}
  :global(.category-badge) {
    font-size: 10px !important;
    padding: 2px 6px !important;
    height: auto !important;
}
  .empty-state {
    text-align: center;
    padding: 48px 16px;
}
  .empty-message {
    font-size: 14px;
    color: #6b7280;
    margin: 0 0 8px 0;
}
  .empty-submessage {
    font-size: 12px;
    color: #9ca3af;
    margin: 0 0 16px 0;
    line-height: 1.4;
}
</style>
