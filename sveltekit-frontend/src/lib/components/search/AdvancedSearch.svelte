<script lang="ts">

  import { createCombobox, melt } from 'melt';
  // Global Svelte runes are declared in src/types/svelte-helpers.d.ts
  import { fly } from 'svelte/transition';
  import Fuse from "fuse.js";
  import { Search, X, Tag, Calendar, FileType } from 'lucide-svelte';
  import type { Evidence } from '$lib/stores/report';

  export let items: Evidence[] = [];
  export let onResults: (results: Evidence[]) => void = () => {};
  export let onSelect: (item: Evidence) => void = () => {};
  export let placeholder = 'Search evidence...';
  export let maxResults = 10;
  export let showFilters = true;
  export let showTags = true;
;
let searchValue = $state('');
let fuse = $state<Fuse<Evidence>;
  let searchResults: Evidence[] >([]);
let allTags = $state<string[] >([]);
let selectedTags = $state<string[] >([]);
let selectedTypes = $state<string[] >([]);
let dateRange = $state<{ start?: Date; end?: Date } >({});

  // Fuse.js configuration for fuzzy search
  const fuseOptions = {
    keys: [
      { name: 'title', weight: 0.4 },
      { name: 'description', weight: 0.3 },
      { name: 'tags', weight: 0.2 },
      { name: 'metadata.format', weight: 0.1 }
    ],
    threshold: 0.3,
    distance: 100,
    minMatchCharLength: 2,
    includeScore: true,
    includeMatches: true
  };

  // Combobox for search input
  const {
    elements: { input, menu, option },
    states: { open, inputValue },
    helpers: { isSelected }
  } = createCombobox({
    forceVisible: true,
    defaultSelected: undefined
  });

  // Initialize Fuse when items change
  $: if (items.length > 0) {
    fuse = new Fuse(items, fuseOptions);
    allTags = [...new Set(items.flatMap(item => item.tags || []))];
  }

  // Perform search when input changes
  $: if (fuse && searchValue) {
    const fuseResults = fuse.search(searchValue);
    searchResults = fuseResults
      .map(result => result.item)
      .slice(0, maxResults);
  } else {
    searchResults = items.slice(0, maxResults);
  }

  // Apply filters (wrap in derived function)
  let filteredResults = $derived(() => searchResults.filter((item: Evidence) => {
    // Type filter
    if (selectedTypes.length > 0 && !selectedTypes.includes(item.type)) {
      return false;
    }
    // Tag filter
    if (selectedTags.length > 0) {
      const itemTags = item.tags || [];
      if (!selectedTags.some(tag => itemTags.includes(tag))) {
        return false;
      }
    }
    // Date range filter
    if (dateRange.start || dateRange.end) {
      const itemDate = new Date(item.createdAt);
      if (dateRange.start && itemDate < dateRange.start) return false;
      if (dateRange.end && itemDate > dateRange.end) return false;
    }
    return true;
  }));

  // Update results when filters change
  $: onResults(filteredResults);

  // Sync input value from combobox state (reactive)
  $: searchValue = $inputValue;

  // Handle item selection
  const handleSelect = (item: Evidence) => {
    onSelect(item);
    inputValue.set('');
  };

  // Clear search
  const clearSearch = () => {
    inputValue.set('');
    selectedTags = [];
    selectedTypes = [];
    dateRange = {};
  };

  // Toggle tag filter
  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      selectedTags = selectedTags.filter(t => t !== tag);
    } else {
      selectedTags = [...selectedTags, tag];
    }
  };

  // Toggle type filter
  const toggleType = (type: string) => {
    if (selectedTypes.includes(type)) {
      selectedTypes = selectedTypes.filter(t => t !== type);
    } else {
      selectedTypes = [...selectedTypes, type];
    }
  };

  // Evidence types
  const evidenceTypes = ['document', 'image', 'video', 'audio', 'link'];

  // Highlight search matches
  const highlightMatches = (text: string, searchTerm: string): string => {
    if (!searchTerm) return text;
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  };
</script>

<div class="advanced-search">
  <!-- Search Input -->
  <div class="search-input-container">
    <div class="search-input-wrapper">
      <Search size={20} class="search-icon" />
      <input
        <!-- <!-- use:melt={$input}
        class="search-input"
        type="text"
        {placeholder}
        autocomplete="off"
      />
      {#if searchValue}
        <button
          class="clear-button"
          on:onclick={() => clearSearch()}
          title="Clear search"
        >
          <X size={16} />
        </button>
      {/if}
    </div>

    <!-- Results dropdown -->
    {#if $open && searchResults.length > 0}
      <div
        <!-- <!-- use:melt={$menu}
        class="search-results"
        transitifly={{ y: -5, duration: 150 }}
      >
        {#each filteredResults as item (item.id)}
          <button
            <!-- <!-- use:melt={$option({ value: item.id, label: item.title })}
            class="search-result-item"
            class:highlighted={$isSelected(item.id)}
            on:onclick={() => handleSelect(item)}
          >
            <div class="result-icon">
              {#if item.type === 'document'}
                <FileType size={16} />
              {:else if item.type === 'image'}
                <img src={item.url} alt="" class="result-thumbnail" />
              {:else}
                <div class="result-type-badge {item.type}">{item.type[0].toUpperCase()}</div>
              {/if}
            </div>

            <div class="result-content">
              <div class="result-title">
                {@html highlightMatches(item.title, searchValue)}
              </div>
              {#if item.description}
                <div class="result-description">
                  {@html highlightMatches(item.description.slice(0, 100), searchValue)}
                  {#if item.description.length > 100}...{/if}
                </div>
              {/if}

              {#if item.tags && item.tags.length > 0}
                <div class="result-tags">
                  {#each item.tags.slice(0, 3) as tag}
                    <span class="result-tag">{tag}</span>
                  {/each}
                </div>
              {/if}
            </div>

            <div class="result-meta">
              <span class="result-type">{item.type}</span>
              <span class="result-date">
                {new Date(item.createdAt).toLocaleDateString()}
              </span>
            </div>
          </button>
        {/each}

        {#if filteredResults.length === 0}
          <div class="no-results">
            <Search size={24} />
            <p>No results found</p>
            <small>Try adjusting your search terms or filters</small>
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Filters -->
  {#if showFilters}
    <div class="search-filters">
      <!-- Type filters -->
      <div class="filter-group">
        <label class="filter-label">
          <FileType size={14} />
          Type
        </label>
        <div class="filter-options">
          {#each evidenceTypes as type}
            <button
              class="filter-chip"
              class:active={selectedTypes.includes(type)}
              on:onclick={() => toggleType(type)}
            >
              {type}
            </button>
          {/each}
        </div>
      </div>

      <!-- Tag filters -->
      {#if showTags && allTags.length > 0}
        <div class="filter-group">
          <label class="filter-label">
            <Tag size={14} />
            Tags
          </label>
          <div class="filter-options">
            {#each allTags.slice(0, 10) as tag}
              <button
                class="filter-chip"
                class:active={selectedTags.includes(tag)}
                on:onclick={() => toggleTag(tag)}
              >
                {tag}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Date range -->
      <div class="filter-group">
        <label class="filter-label">
          <Calendar size={14} />
          Date Range
        </label>
        <div class="date-range-inputs">
          <input
            type="date"
            class="date-input"
            bind:value={dateRange.start}
            placeholder="Start date"
          />
          <span>to</span>
          <input
            type="date"
            class="date-input"
            bind:value={dateRange.end}
            placeholder="End date"
          />
        </div>
      </div>
    </div>
  {/if}

  <!-- Active filters summary -->
  {#if selectedTags.length > 0 || selectedTypes.length > 0 || dateRange.start || dateRange.end}
    <div class="active-filters">
      <span class="active-filters-label">Active filters:</span>

      {#each selectedTypes as type}
        <span class="active-filter">
          {type}
          <button on:onclick={() => toggleType(type)}>
            <X size={12} />
          </button>
        </span>
      {/each}

      {#each selectedTags as tag}
        <span class="active-filter">
          #{tag}
          <button on:onclick={() => toggleTag(tag)}>
            <X size={12} />
          </button>
        </span>
      {/each}

      {#if dateRange.start || dateRange.end}
        <span class="active-filter">
          {dateRange.start?.toLocaleDateString() || '...'} - {dateRange.end?.toLocaleDateString() || '...'}
          <button on:onclick={() => dateRange = {}}>
            <X size={12} />
          </button>
        </span>
      {/if}

      <button class="clear-all-filters" on:onclick={() => clearSearch()}>
        Clear all
      </button>
    </div>
  {/if}
</div>

<style>
.advanced-search {
  position: relative;
  width: 100%;
}

.search-input-container {
  position: relative;
}

.search-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 0.75rem;
  color: #6b7280;
  z-index: 1;
}

.search-input {
  width: 100%;
  padding: 0.75rem 2.5rem 0.75rem 2.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  background: #ffffff;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.search-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.clear-button {
  position: absolute;
  right: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 1.5rem;
  height: 1.5rem;
  border: none;
  background: none;
  color: #6b7280;
  border-radius: 0.25rem;
  cursor: pointer;
  transition: color 0.15s ease;
}

.clear-button:hover {
  color: #374151;
}

.search-results {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: #ffffff;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  max-height: 20rem;
  overflow-y: auto;
  z-index: 50;
  margin-top: 0.25rem;
}

.search-result-item {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  width: 100%;
  padding: 0.75rem;
  border: none;
  background: none;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.15s ease;
  border-bottom: 1px solid #f1f5f9;
}

.search-result-item:hover,
.search-result-item.highlighted {
  background: #f8fafc;
}

.search-result-item:last-child {
  border-bottom: none;
}

.result-icon {
  flex-shrink: 0;
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.result-thumbnail {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 0.25rem;
}

.result-type-badge {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
  color: white;
  border-radius: 0.25rem;
}

.result-type-badge.document { background: #3b82f6; }
.result-type-badge.video { background: #8b5cf6; }
.result-type-badge.audio { background: #f59e0b; }
.result-type-badge.link { background: #06b6d4; }

.result-content {
  flex: 1;
  min-width: 0;
}

.result-title {
  font-weight: 500;
  color: #111827;
  margin-bottom: 0.25rem;
  word-break: break-word;
}

.result-description {
  font-size: 0.875rem;
  color: #6b7280;
  line-height: 1.4;
  margin-bottom: 0.25rem;
}

.result-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}

.result-tag {
  font-size: 0.75rem;
  background: #eff6ff;
  color: #3b82f6;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
}

.result-meta {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.25rem;
  font-size: 0.75rem;
  color: #9ca3af;
}

.result-type {
  text-transform: capitalize;
  font-weight: 500;
}

.no-results {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  color: #6b7280;
  text-align: center;
}

.no-results p {
  margin: 0.5rem 0 0.25rem;
  font-weight: 500;
}

.no-results small {
  font-size: 0.75rem;
  opacity: 0.8;
}

.search-filters {
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.filter-label {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}

.filter-options {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.filter-chip {
  padding: 0.375rem 0.75rem;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #374151;
  border-radius: 1rem;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.15s ease;
}

.filter-chip:hover {
  border-color: #3b82f6;
  background: #f8fafc;
}

.filter-chip.active {
  background: #3b82f6;
  color: white;
  border-color: #3b82f6;
}

.date-range-inputs {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.date-input {
  padding: 0.375rem 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 0.875rem;
}

.active-filters {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  padding: 0.75rem;
  background: #f8fafc;
  border-radius: 0.5rem;
}

.active-filters-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}

.active-filter {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  background: #3b82f6;
  color: white;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 500;
}

.active-filter button {
  display: flex;
  align-items: center;
  border: none;
  background: rgba(255, 255, 255, 0.3);
  color: inherit;
  border-radius: 50%;
  padding: 0.125rem;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.active-filter button:hover {
  background: rgba(255, 255, 255, 0.5);
}

.clear-all-filters {
  padding: 0.25rem 0.5rem;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #6b7280;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.15s ease;
}

.clear-all-filters:hover {
  border-color: #ef4444;
  color: #ef4444;
}

/* Search highlighting */
:global(mark) {
  background: #fef08a;
  color: #713f12;
  padding: 0.125rem 0.25rem;
  border-radius: 0.125rem;
}
</style>

<!-- TODO: migrate export lets to $props(); CommonProps assumed. -->
