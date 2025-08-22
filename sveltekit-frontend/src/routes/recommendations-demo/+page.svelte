<!-- Legal AI Recommendations Demo - SSR with Svelte 5 + Melt-UI + Bits-UI v2 -->
<script lang="ts">
  import { $props, $state } from 'svelte';
  import { page } from '$app/stores';
  import FOAFModal from '$lib/components/recommendations/FOAFModal.svelte';
  import DidYouMeanSuggestions from '$lib/components/recommendations/DidYouMeanSuggestions.svelte';
  import { Users, Search, Sparkles, Database, Brain, Network } from 'lucide-svelte';

  let { data } = $props();
  
  // Svelte 5 runes for component state
  let foafModalOpen = $state(false);
  let searchQuery = $state('');
  let selectedContext = $state('GENERAL');

  const contextOptions = [
    { value: 'GENERAL', label: 'General', icon: Search },
    { value: 'PERSON', label: 'People', icon: Users },
    { value: 'CASE', label: 'Cases', icon: Database },
    { value: 'DOCUMENT', label: 'Documents', icon: Brain },
  ];

  function handleSuggestionSelect(suggestion: any) {
    console.log('Selected suggestion:', suggestion);
    // Navigate to the selected entity or perform action
  }

  function handleSearch(query: string) {
    console.log('Search performed:', query);
  }
</script>

<svelte:head>
  <title>{data.meta.title}</title>
  <meta name="description" content={data.meta.description}" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
  <!-- Header -->
  <header class="bg-white border-b border-gray-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Network class="w-8 h-8 text-blue-600" />
            Legal AI Recommendations
          </h1>
          <p class="text-gray-600 mt-2">
            Friend-of-a-friend network analysis and intelligent search suggestions
          </p>
        </div>
        <div class="flex items-center gap-4">
          <span class="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
            ✅ SSR Ready
          </span>
          <span class="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full">
            🎯 Svelte 5 + Melt-UI
          </span>
        </div>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
    <!-- Demo Controls -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
      <!-- FOAF Section -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div class="flex items-center gap-3 mb-4">
          <Users class="w-6 h-6 text-blue-600" />
          <h2 class="text-xl font-semibold text-gray-900">Professional Network</h2>
        </div>
        
        <p class="text-gray-600 mb-6">
          Discover legal professionals in your extended network through friend-of-a-friend analysis.
        </p>

        {#if data.foafData}
          <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <p class="text-blue-800 text-sm">{data.foafData.summary}</p>
            <p class="text-blue-600 text-xs mt-1">
              Found {data.foafData.totalFound} professionals
            </p>
          </div>
        {/if}

        <button
          class="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors flex items-center justify-center gap-2"
          onclick={() => foafModalOpen = true}
        >
          <Users class="w-4 h-4" />
          Show Network Recommendations
        </button>
      </div>

      <!-- Search Suggestions Section -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div class="flex items-center gap-3 mb-4">
          <Sparkles class="w-6 h-6 text-purple-600" />
          <h2 class="text-xl font-semibold text-gray-900">Intelligent Search</h2>
        </div>
        
        <p class="text-gray-600 mb-6">
          Get intelligent suggestions with "did you mean?" functionality and fuzzy search.
        </p>

        <!-- Context Selection -->
        <div class="mb-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">Search Context</label>
          <select 
            bind:value={selectedContext}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            {#each contextOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
        </div>

        <!-- Search Component -->
        <DidYouMeanSuggestions 
          bind:query={searchQuery}
          contexttype={selectedContext}
          placeholder="Search for legal documents, cases, people..."
          onselect={handleSuggestionSelect}
          onsearch={handleSearch}
        />
      </div>
    </div>

    <!-- SSR Data Preview -->
    <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
        <Database class="w-5 h-5 text-green-600" />
        Server-Side Rendered Data
      </h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- FOAF Preview -->
        <div>
          <h4 class="font-medium text-gray-800 mb-3">FOAF Network Preview</h4>
          {#if data.foafData?.people}
            <div class="space-y-2">
              {#each data.foafData.people.slice(0, 3) as person}
                <div class="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                    <Users class="w-4 h-4 text-blue-600" />
                  </div>
                  <div>
                    <p class="font-medium text-gray-900 text-sm">{person.name}</p>
                    <p class="text-xs text-gray-600">{person.role} • {person.specialization}</p>
                  </div>
                  <div class="ml-auto text-xs text-blue-600 font-medium">
                    {Math.round(person.confidence * 100)}%
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <p class="text-gray-500 text-sm">No FOAF data loaded</p>
          {/if}
        </div>

        <!-- Initial Suggestions Preview -->
        <div>
          <h4 class="font-medium text-gray-800 mb-3">Initial Suggestions</h4>
          {#if data.initialSuggestions.length > 0}
            <div class="space-y-2">
              {#each data.initialSuggestions.slice(0, 4) as suggestion}
                <div class="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <div class="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                    <Search class="w-4 h-4 text-purple-600" />
                  </div>
                  <div class="flex-1">
                    <p class="font-medium text-gray-900 text-sm">{suggestion.label}</p>
                    <p class="text-xs text-gray-600">{suggestion.type} • {suggestion.description}</p>
                  </div>
                  <div class="text-xs text-purple-600 font-medium">
                    {Math.round(suggestion.score * 100)}%
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <p class="text-gray-500 text-sm">No initial suggestions loaded</p>
          {/if}
        </div>
      </div>
    </div>

    <!-- Technical Details -->
    <div class="mt-12 bg-gray-50 rounded-xl p-6">
      <h3 class="text-lg font-semibold text-gray-900 mb-4">Technical Implementation</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="bg-white p-4 rounded-lg border border-gray-200">
          <h4 class="font-medium text-gray-800 mb-2">Frontend</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• SvelteKit 2 SSR</li>
            <li>• Svelte 5 runes</li>
            <li>• Melt-UI + Bits-UI v2</li>
            <li>• Tailwind CSS</li>
          </ul>
        </div>
        <div class="bg-white p-4 rounded-lg border border-gray-200">
          <h4 class="font-medium text-gray-800 mb-2">Backend</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Go gRPC service</li>
            <li>• PostgreSQL + pgvector</li>
            <li>• Redis caching</li>
            <li>• Dynamic port allocation</li>
          </ul>
        </div>
        <div class="bg-white p-4 rounded-lg border border-gray-200">
          <h4 font-medium text-gray-800 mb-2">AI/ML</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• LangChain.js</li>
            <li>• Legal-BERT ready</li>
            <li>• Ollama integration</li>
            <li>• Fuse.js fuzzy search</li>
          </ul>
        </div>
        <div class="bg-white p-4 rounded-lg border border-gray-200">
          <h4 class="font-medium text-gray-800 mb-2">Graph DB</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Memgraph ready</li>
            <li>• Cypher queries</li>
            <li>• FOAF algorithms</li>
            <li>• Network analysis</li>
          </ul>
        </div>
      </div>
    </div>
  </main>
</div>

<!-- FOAF Modal -->
<FOAFModal
  bind:open={foafModalOpen}
  personid={data.personId}
  onclose={() => foafModalOpen = false}
/>

<style>
  /* Additional styles for enhanced UX */
  :global(.animate-pulse) {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: .5; }
  }
</style>