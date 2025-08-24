<!--
  Legal Search Demo Page
  Demonstrates the LegalSearchCombobox component with various configurations
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import LegalSearchCombobox from '$lib/components/search/LegalSearchCombobox.svelte';
  import { Search, Zap, Database, Users, FileText, Scale } from 'lucide-svelte';
  
  // Demo state
  let selectedResult: any = null;
  let searchHistory: any[] = [];
  let demoConfig = {
    enableVectorSearch: true,
    aiSuggestions: true,
    categories: ['cases', 'evidence', 'precedents'],
    maxResults: 20,
    similarityThreshold: 0.7
  };
  
  let demoSearches = [
    "Miranda rights violation",
    "DNA evidence chain of custody", 
    "Constitutional search and seizure",
    "Witness testimony credibility",
    "Probable cause determination",
    "Fourth Amendment exceptions"
  ];
  
  let currentDemo = 0;
  
  // Handle search results
  function handleSearchSelect(event) {
    selectedResult = event.detail;
    searchHistory = [event.detail, ...searchHistory.slice(0, 4)];
    console.log('Selected result:', event.detail);
  }
  
  function handleSearchQuery(event) {
    console.log('Search performed:', event.detail);
  }
  
  function handleClear() {
    selectedResult = null;
    console.log('Search cleared');
  }
  
  // Demo functions
  function runDemoSearch() {
    const searchText = demoSearches[currentDemo];
    // Simulate typing the search
    const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement;
    if (searchInput) {
      searchInput.value = searchText;
      searchInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
    currentDemo = (currentDemo + 1) % demoSearches.length;
  }
  
  function toggleVectorSearch() {
    demoConfig.enableVectorSearch = !demoConfig.enableVectorSearch;
    demoConfig = { ...demoConfig };
  }
  
  function toggleAISuggestions() {
    demoConfig.aiSuggestions = !demoConfig.aiSuggestions;
    demoConfig = { ...demoConfig };
  }
  
  onMount(() => {
    console.log('Legal Search Demo loaded');
  });
</script>

<svelte:head>
  <title>Legal Search Demo - LegalSearchCombobox Component</title>
  <meta name="description" content="Interactive demo of the LegalSearchCombobox component with vector search and AI suggestions" />
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white border-b border-gray-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-900 flex items-center">
            <Search class="mr-3 h-8 w-8 text-blue-600" />
            Legal Search Component Demo
          </h1>
          <p class="mt-2 text-gray-600">
            Interactive demonstration of the LegalSearchCombobox with vector search and AI integration
          </p>
        </div>
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-2 text-sm">
            <div class={`h-3 w-3 rounded-full ${demoConfig.enableVectorSearch ? 'bg-green-400' : 'bg-gray-300'}`}></div>
            <span class="text-gray-600">Vector Search</span>
          </div>
          <div class="flex items-center space-x-2 text-sm">
            <div class={`h-3 w-3 rounded-full ${demoConfig.aiSuggestions ? 'bg-purple-400' : 'bg-gray-300'}`}></div>
            <span class="text-gray-600">AI Suggestions</span>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Main Demo Area -->
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
      
      <!-- Search Component Demo -->
      <div class="lg:col-span-2">
        <div class="bg-white rounded-xl shadow-lg p-8">
          <h2 class="text-2xl font-semibold text-gray-900 mb-6">
            Legal Search Component
          </h2>
          
          <!-- Search Component -->
          <div class="mb-8">
            <LegalSearchCombobox
              placeholder="Search cases, evidence, precedents, statutes..."
              categories={demoConfig.categories}
              enableVectorSearch={demoConfig.enableVectorSearch}
              aiSuggestions={demoConfig.aiSuggestions}
              maxResults={demoConfig.maxResults}
              similarityThreshold={demoConfig.similarityThreshold}
              className="w-full"
              on:select={handleSearchSelect}
              on:search={handleSearchQuery}
              on:clear={handleClear}
            />
          </div>
          
          <!-- Demo Controls -->
          <div class="border-t border-gray-200 pt-6">
            <h3 class="text-lg font-medium text-gray-900 mb-4">Demo Controls</h3>
            <div class="flex flex-wrap gap-3">
              <button
                class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                on:click={runDemoSearch}
              >
                <Zap class="mr-2 h-4 w-4" />
                Try Demo Search
              </button>
              
              <button
                class={`inline-flex items-center px-4 py-2 border text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                  demoConfig.enableVectorSearch 
                    ? 'border-green-500 text-green-700 bg-green-50 hover:bg-green-100' 
                    : 'border-gray-300 text-gray-700 bg-white hover:bg-gray-50'
                }`}
                on:click={toggleVectorSearch}
              >
                <Database class="mr-2 h-4 w-4" />
                Vector Search: {demoConfig.enableVectorSearch ? 'ON' : 'OFF'}
              </button>
              
              <button
                class={`inline-flex items-center px-4 py-2 border text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                  demoConfig.aiSuggestions 
                    ? 'border-purple-500 text-purple-700 bg-purple-50 hover:bg-purple-100'
                    : 'border-gray-300 text-gray-700 bg-white hover:bg-gray-50'
                }`}
                on:click={toggleAISuggestions}
              >
                <Zap class="mr-2 h-4 w-4" />
                AI Suggestions: {demoConfig.aiSuggestions ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>
        </div>
        
        <!-- Search Result Display -->
        {#if selectedResult}
          <div class="mt-8 bg-white rounded-xl shadow-lg p-8">
            <h3 class="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <FileText class="mr-2 h-5 w-5" />
              Selected Result
            </h3>
            
            <div class="bg-gray-50 rounded-lg p-6">
              <div class="flex items-start justify-between mb-4">
                <div>
                  <h4 class="text-lg font-medium text-gray-900">{selectedResult.title}</h4>
                  <div class="flex items-center space-x-3 mt-2">
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {selectedResult.type}
                    </span>
                    {#if selectedResult.score}
                      <span class="text-sm text-gray-500">
                        Relevance: {Math.round(selectedResult.score * 100)}%
                      </span>
                    {/if}
                  </div>
                </div>
              </div>
              
              {#if selectedResult.content}
                <p class="text-gray-700 mb-4">{selectedResult.content}</p>
              {/if}
              
              {#if selectedResult.metadata}
                <div class="space-y-2">
                  {#if selectedResult.metadata.date}
                    <div class="text-sm text-gray-600">
                      <strong>Date:</strong> {new Date(selectedResult.metadata.date).toLocaleDateString()}
                    </div>
                  {/if}
                  {#if selectedResult.metadata.status}
                    <div class="text-sm text-gray-600">
                      <strong>Status:</strong> {selectedResult.metadata.status}
                    </div>
                  {/if}
                  {#if selectedResult.metadata.jurisdiction}
                    <div class="text-sm text-gray-600">
                      <strong>Jurisdiction:</strong> {selectedResult.metadata.jurisdiction}
                    </div>
                  {/if}
                  {#if selectedResult.metadata.tags && selectedResult.metadata.tags.length > 0}
                    <div class="text-sm text-gray-600">
                      <strong>Tags:</strong> 
                      <div class="flex flex-wrap gap-1 mt-1">
                        {#each selectedResult.metadata.tags as tag}
                          <span class="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-200 text-gray-800">
                            {tag}
                          </span>
                        {/each}
                      </div>
                    </div>
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
      
      <!-- Sidebar -->
      <div class="space-y-8">
        
        <!-- Configuration Panel -->
        <div class="bg-white rounded-xl shadow-lg p-6">
          <h3 class="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Zap class="mr-2 h-5 w-5" />
            Configuration
          </h3>
          
          <div class="space-y-4">
            <div>
              <label class="text-sm font-medium text-gray-700">Search Categories</label>
              <div class="mt-2 space-y-2">
                {#each ['cases', 'evidence', 'precedents', 'statutes', 'criminals', 'documents'] as category}
                  <label class="flex items-center">
                    <input
                      type="checkbox"
                      class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      checked={demoConfig.categories.includes(category)}
                      on:change={(e) => {
                        if (e.target.checked) {
                          demoConfig.categories = [...demoConfig.categories, category];
                        } else {
                          demoConfig.categories = demoConfig.categories.filter(c => c !== category);
                        }
                      }}
                    />
                    <span class="ml-2 text-sm text-gray-700 capitalize">{category}</span>
                  </label>
                {/each}
              </div>
            </div>
            
            <div>
              <label class="text-sm font-medium text-gray-700">Max Results</label>
              <input
                type="range"
                min="5"
                max="50"
                step="5"
                class="w-full mt-2"
                bind:value={demoConfig.maxResults}
              />
              <div class="text-xs text-gray-500 mt-1">{demoConfig.maxResults} results</div>
            </div>
            
            <div>
              <label class="text-sm font-medium text-gray-700">Similarity Threshold</label>
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.1"
                class="w-full mt-2"
                bind:value={demoConfig.similarityThreshold}
              />
              <div class="text-xs text-gray-500 mt-1">{demoConfig.similarityThreshold}</div>
            </div>
          </div>
        </div>
        
        <!-- Search History -->
        {#if searchHistory.length > 0}
          <div class="bg-white rounded-xl shadow-lg p-6">
            <h3 class="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Users class="mr-2 h-5 w-5" />
              Recent Searches
            </h3>
            
            <div class="space-y-3">
              {#each searchHistory as result}
                <div class="p-3 bg-gray-50 rounded-lg">
                  <div class="flex items-center justify-between">
                    <h4 class="font-medium text-gray-900 text-sm truncate">{result.title}</h4>
                    <span class="text-xs text-gray-500 ml-2">{result.type}</span>
                  </div>
                  {#if result.score}
                    <div class="text-xs text-gray-500 mt-1">
                      Score: {Math.round(result.score * 100)}%
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          </div>
        {/if}
        
        <!-- Demo Example Searches -->
        <div class="bg-white rounded-xl shadow-lg p-6">
          <h3 class="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Scale class="mr-2 h-5 w-5" />
            Example Searches
          </h3>
          
          <div class="space-y-2">
            {#each demoSearches as searchTerm, index}
              <button
                class={`w-full text-left p-3 rounded-lg text-sm transition-colors ${
                  index === currentDemo 
                    ? 'bg-blue-50 text-blue-700 border border-blue-200'
                    : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                }`}
                on:click={() => {
                  const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement;
                  if (searchInput) {
                    searchInput.value = searchTerm;
                    searchInput.dispatchEvent(new Event('input', { bubbles: true }));
                  }
                }}
              >
                {searchTerm}
              </button>
            {/each}
          </div>
        </div>
        
      </div>
    </div>
  </div>
</div>