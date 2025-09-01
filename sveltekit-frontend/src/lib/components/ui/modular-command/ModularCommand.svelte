<!-- Modular API-Integrated Command Palette -->
<script lang="ts">
  import { Command } from 'bits-ui';
  import { Search, FileText, Users, Calendar, Gavel, Loader2 } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { createEventDispatcher, onMount } from 'svelte';
  import { reactiveApiClient } from '$lib/services/api-client';
  import type { CommandSearchRequest, CommandSearchResponse, Case, Evidence, User, LegalDocument } from '$lib/types/api';

  interface Props {
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
    placeholder?: string;
    class?: string;
    
    // Search configuration
    searchTypes?: ('cases' | 'evidence' | 'documents' | 'people')[];
    searchLimit?: number;
    minQueryLength?: number;
    debounceMs?: number;
    includeVectorSearch?: boolean;
    
    // Data filtering
    userId?: string;
    caseId?: string;
    filters?: Record<string, any>;
    
    // Event handlers
    onSelect?: (item: any, type: string) => void;
    onSearchChange?: (query: string) => void;
  }

  let {
    open = $bindable(false),
    onOpenChange,
    placeholder = 'Search cases, evidence, documents...',
    class: className = '',
    
    searchTypes = ['cases', 'evidence', 'documents', 'people'],
    searchLimit = 10,
    minQueryLength = 2,
    debounceMs = 300,
    includeVectorSearch = true,
    
    userId,
    caseId,
    filters = {},
    
    onSelect,
    onSearchChange
  }: Props = $props();

  const dispatch = createEventDispatcher();

  // Search state
  let query = $state('');
  let isSearching = $state(false);
  let searchResults = $state<CommandSearchResponse['results']>({
    cases: [],
    evidence: [],
    documents: [],
    people: []
  });
  let totalResults = $state(0);

  // Debounced search
let searchTimeout = $state<number | null >(null);

  const iconMap = {
    cases: Gavel,
    evidence: FileText,
    documents: FileText,
    people: Users
  };

  const labelMap = {
    cases: 'Cases',
    evidence: 'Evidence',
    documents: 'Documents',
    people: 'People'
  };

  async function performSearch(searchQuery: string) {
    if (searchQuery.length < minQueryLength) {
      searchResults = { cases: [], evidence: [], documents: [], people: [] };
      totalResults = 0;
      return;
    }

    isSearching = true;

    try {
      const searchParams: CommandSearchRequest = {
        query: searchQuery,
        types: searchTypes,
        limit: searchLimit,
        userId
      };

      const response = await reactiveApiClient.commandSearch(searchParams);
      
      if (response.success && response.data) {
        searchResults = response.data.results || { cases: [], evidence: [], documents: [], people: [] };
        totalResults = response.data.totalResults || 0;
      }
    } catch (error) {
      console.error('Command search failed:', error);
      searchResults = { cases: [], evidence: [], documents: [], people: [] };
      totalResults = 0;
    } finally {
      isSearching = false;
    }
  }

  function handleQueryChange(newQuery: string) {
    query = newQuery;
    onSearchChange?.(newQuery);

    // Clear previous timeout
    if (searchTimeout) {
      clearTimeout(searchTimeout);
    }

    // Debounce search
    if (newQuery.trim()) {
      searchTimeout = setTimeout(() => {
        performSearch(newQuery.trim());
      }, debounceMs) as any;
    } else {
      searchResults = { cases: [], evidence: [], documents: [], people: [] };
      totalResults = 0;
    }
  }

  function handleSelect(item: any, type: string) {
    onSelect?.(item, type);
    dispatch('select', { item, type });
    open = false;
  }

  function handleOpenChange(newOpen: boolean) {
    open = newOpen;
    onOpenChange?.(newOpen);
    
    if (!newOpen) {
      // Clear search when closed
      query = '';
      searchResults = { cases: [], evidence: [], documents: [], people: [] };
      totalResults = 0;
    }
  }

  function formatResultText(item: any, type: string): string {
    switch (type) {
      case 'cases':
        return `${item.title} #${item.caseNumber || item.id?.slice(-6)} - ${item.status}`;
      case 'evidence':
        return `${item.title} (${item.evidenceType}) - Case: ${item.caseId?.slice(-6)}`;
      case 'documents':
        return `${item.title} - ${item.documentType}`;
      case 'people':
        return `${item.name} (${item.role}) - ${item.email}`;
      default:
        return item.title || item.name || 'Unknown';
    }
  }

  function formatResultDescription(item: any, type: string): string {
    switch (type) {
      case 'cases':
        return item.description || `Priority: ${item.priority} | Created: ${new Date(item.createdAt).toLocaleDateString()}`;
      case 'evidence':
        return item.description || `Collected: ${item.collectedAt ? new Date(item.collectedAt).toLocaleDateString() : 'Unknown'}`;
      case 'documents':
        return item.metadata?.summary || `Created: ${new Date(item.createdAt).toLocaleDateString()}`;
      case 'people':
        return `Department: ${item.department || 'Unknown'} | Role: ${item.role}`;
      default:
        return '';
    }
  }

  // Cleanup timeout on unmount
  onMount(() => {
    return () => {
      if (searchTimeout) {
        clearTimeout(searchTimeout);
      }
    };
  });
</script>

<Command.Root
  bind:open
  onOpenChange={handleOpenChange}
  className={cn(
    'modular-command-palette',
    'flex h-full w-full flex-col overflow-hidden rounded-md bg-popover text-popover-foreground',
    className
  )}
>
  <div class="flex items-center border-b px-3 modular-command-header">
    <Search class="mr-2 h-4 w-4 shrink-0 opacity-50" />
    <Command.Input
      bind:value={query}
      input={(e) => handleQueryChange(e.currentTarget.value)}
      {placeholder}
      class="flex h-11 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50 font-mono"
    />
    {#if isSearching}
      <Loader2 class="ml-2 h-4 w-4 animate-spin opacity-50" />
    {/if}
  </div>

  <Command.List class="max-h-[400px] overflow-y-auto overflow-x-hidden modular-command-list">
    {#if query.length < minQueryLength}
      <div class="py-6 text-center text-sm text-muted-foreground font-mono">
        Type {minQueryLength}+ characters to search...
      </div>
    {:else if isSearching}
      <div class="py-6 text-center text-sm text-muted-foreground font-mono flex items-center justify-center gap-2">
        <Loader2 class="h-4 w-4 animate-spin" />
        Searching...
      </div>
    {:else if totalResults === 0 && query.length >= minQueryLength}
      <Command.Empty class="py-6 text-center text-sm text-muted-foreground font-mono">
        No results found for "{query}"
      </Command.Empty>
    {:else}
      {#each searchTypes as type}
        {#if searchResults[type]?.length > 0}
          <Command.Group class="modular-command-group">
            <Command.GroupHeading class="px-2 py-1.5 text-xs font-medium text-muted-foreground font-mono uppercase tracking-wider flex items-center gap-2">
              <svelte:component this={iconMap[type]} class="h-3 w-3" />
              {labelMap[type]} ({searchResults[type].length})
            </Command.GroupHeading>
            
            {#each searchResults[type] as item}
              <Command.Item
                value={formatResultText(item, type)}
                onSelect={() => handleSelect(item, type)}
                class="relative flex cursor-default select-none items-center rounded-sm px-2 py-2 text-sm outline-none aria-selected:bg-accent aria-selected:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 modular-command-item font-mono"
              >
                <div class="flex items-start gap-3 w-full">
                  <svelte:component this={iconMap[type]} class="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
                  <div class="flex flex-col gap-1 min-w-0 flex-1">
                    <div class="font-medium text-sm truncate">
                      {formatResultText(item, type)}
                      {#if item.similarity !== undefined}
                        <span class="text-xs text-muted-foreground ml-2">
                          ({Math.round(item.similarity * 100)}% match)
                        </span>
                      {/if}
                    </div>
                    <div class="text-xs text-muted-foreground truncate">
                      {formatResultDescription(item, type)}
                    </div>
                  </div>
                </div>
              </Command.Item>
            {/each}
          </Command.Group>
        {/if}
      {/each}

      {#if totalResults > searchLimit}
        <div class="px-2 py-2 text-xs text-muted-foreground text-center font-mono border-t">
          Showing {searchLimit} of {totalResults} results
        </div>
      {/if}
    {/if}
  </Command.List>
</Command.Root>

<style>
  :global(.modular-command-palette) {
    @apply bg-yorha-bg-primary border border-yorha-border shadow-xl;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  }

  :global(.modular-command-header) {
    @apply border-yorha-border bg-yorha-bg-secondary;
  }

  :global(.modular-command-list) {
    @apply bg-yorha-bg-primary;
  }

  :global(.modular-command-group) {
    @apply border-yorha-border;
  }

  :global(.modular-command-item) {
    @apply hover:bg-yorha-bg-hover text-yorha-text-primary;
    @apply transition-colors duration-150;
  }

  :global(.modular-command-item[aria-selected="true"]) {
    @apply bg-yorha-accent text-yorha-text-accent;
  }
</style>