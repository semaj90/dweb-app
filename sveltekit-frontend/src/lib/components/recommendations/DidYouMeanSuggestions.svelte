<!-- "Did You Mean?" Suggestions Component - SSR compatible with Bits-UI v2 -->
<script lang="ts">
  import { $props, $state, $effect } from 'svelte';
  import { createCombobox, melt } from 'melt';
  import { Check, ChevronDown, Search, FileText, User, Folder, Tag } from 'lucide-svelte';
  import { fly } from 'svelte/transition';
  
  interface Suggestion {
    label: string;
    entityId: string;
    type: 'PERSON' | 'DOCUMENT' | 'CASE' | 'EVIDENCE' | 'TAG';
    score: number;
    description: string;
    icon: string;
    tags: string[];
  }

  interface Props {
    query?: string;
    placeholder?: string;
    contextType?: string;
    onSelect?: (suggestion: Suggestion) => void;
    onSearch?: (query: string) => void;
  }

  let { 
    query = $bindable(''), 
    placeholder = 'Search legal documents, cases, people...', 
    contextType = 'GENERAL',
    onSelect,
    onSearch 
  }: Props = $props();

  // Svelte 5 reactive state
  let suggestions = $state<Suggestion[]>([]);
  let loading = $state(false);
  let correctedQuery = $state('');
  let debounceTimer = $state<NodeJS.Timeout | null>(null);

  // Melt-UI combobox builder
  const {
    elements: { menu, input, option, label },
    states: { open, inputValue, selected },
    helpers: { isSelected }
  } = createCombobox<Suggestion>({
    forceVisible: true,
  });

  // Sync input with query prop
  $effect(() => {
    inputValue.set(query);
  });

  $effect(() => {
    query = $inputValue;
  });

  // Debounced search effect
  $effect(() => {
    if (query.length >= 2) {
      if (debounceTimer) clearTimeout(debounceTimer);
      
      debounceTimer = setTimeout(async () => {
        await performSearch(query);
      }, 300);
    } else {
      suggestions = [];
    }

    return () => {
      if (debounceTimer) clearTimeout(debounceTimer);
    };
  });

  async function performSearch(searchQuery: string) {
    if (!searchQuery || searchQuery.length < 2) return;
    
    loading = true;
    
    try {
      const response = await fetch(`/api/suggest?q=${encodeURIComponent(searchQuery)}&context=${contextType}&limit=8`);
      if (!response.ok) throw new Error('Search failed');
      
      const data = await response.json();
      suggestions = data.suggestions || [];
      correctedQuery = data.correctedQuery || '';
      
      onSearch?.(searchQuery);
    } catch (error) {
      console.error('Search error:', error);
      suggestions = [];
    } finally {
      loading = false;
    }
  }

  function handleSelection(suggestion: Suggestion) {
    query = suggestion.label;
    onSelect?.(suggestion);
    open.set(false);
  }

  function getIconComponent(iconName: string, type: string) {
    switch (type) {
      case 'PERSON': return User;
      case 'DOCUMENT': return FileText;
      case 'CASE': return Folder;
      case 'TAG': return Tag;
      default: return Search;
    }
  }

  function getTypeColor(type: string): string {
    switch (type) {
      case 'PERSON': return 'bg-blue-100 text-blue-700';
      case 'DOCUMENT': return 'bg-green-100 text-green-700';
      case 'CASE': return 'bg-purple-100 text-purple-700';
      case 'EVIDENCE': return 'bg-orange-100 text-orange-700';
      case 'TAG': return 'bg-gray-100 text-gray-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  }
</script>

<div class="relative w-full">
  <!-- Search Input -->
  <div class="relative">
    <input
      use:melt={$input}
      class="w-full px-4 py-3 pl-10 pr-10 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      {placeholder}
      autocomplete="off"
    />
    <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
    {#if loading}
      <div class="absolute right-3 top-1/2 transform -translate-y-1/2">
        <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
      </div>
    {:else}
      <ChevronDown class="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
    {/if}
  </div>

  <!-- Corrected Query Hint -->
  {#if correctedQuery && correctedQuery !== query}
    <div class="mt-2 text-sm text-gray-600">
      Did you mean: <button class="text-blue-600 hover:underline" onclick={() => query = correctedQuery}>
        {correctedQuery}
      </button>?
    </div>
  {/if}

  <!-- Suggestions Dropdown -->
  {#if $open && suggestions.length > 0}
    <div
      use:melt={$menu}
      class="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-auto"
      transition:fly={{ y: -5, duration: 150 }}
    >
      {#each suggestions as suggestion, index (suggestion.entityId)}
        <button
          use:melt={$option({ value: suggestion, label: suggestion.label })}
          class="w-full px-4 py-3 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none border-b border-gray-100 last:border-b-0"
          onclick={() => handleSelection(suggestion)}
        >
          <div class="flex items-center gap-3">
            <!-- Icon -->
            <div class="p-1.5 {getTypeColor(suggestion.type)} rounded-md">
              <svelte:component 
                this={getIconComponent(suggestion.icon, suggestion.type)} 
                class="w-3.5 h-3.5" 
              />
            </div>
            
            <!-- Content -->
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between">
                <span class="font-medium text-gray-900 truncate">{suggestion.label}</span>
                <span class="text-xs text-gray-500 ml-2">
                  {Math.round(suggestion.score * 100)}%
                </span>
              </div>
              
              {#if suggestion.description}
                <p class="text-xs text-gray-600 truncate mt-0.5">{suggestion.description}</p>
              {/if}
              
              <!-- Tags -->
              {#if suggestion.tags.length > 0}
                <div class="flex gap-1 mt-1">
                  {#each suggestion.tags.slice(0, 3) as tag}
                    <span class="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                      {tag}
                    </span>
                  {/each}
                </div>
              {/if}
            </div>
          </div>
        </button>
      {/each}
    </div>
  {/if}

  <!-- No Results -->
  {#if $open && !loading && suggestions.length === 0 && query.length >= 2}
    <div class="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg p-4">
      <div class="text-center text-gray-500">
        <Search class="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p class="text-sm">No suggestions found for "{query}"</p>
        <p class="text-xs mt-1">Try a different search term or check spelling</p>
      </div>
    </div>
  {/if}
</div>

<style>
  /* Ensure proper z-index stacking */
  :global(.melt-dialog-overlay) {
    z-index: 50;
  }
  
  :global(.melt-dialog-content) {
    z-index: 51;
  }
  
  /* Custom scrollbar for suggestions */
  .suggestions-scroll::-webkit-scrollbar {
    width: 6px;
  }
  
  .suggestions-scroll::-webkit-scrollbar-track {
    background: #f1f1f1;
  }
  
  .suggestions-scroll::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }
  
  .suggestions-scroll::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
</style>