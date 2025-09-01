<!--
  Real-Time Legal Search Demo Page
  Showcasing WebSocket/NATS integration with Svelte 5 + bits-ui
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  
  // Real-time search components
  import RealTimeLegalSearch from '$lib/components/search/RealTimeLegalSearch.svelte';
  import { useRealTimeSearch } from '$lib/services/real-time-search.js';
  
  // UI Components
  import { Button } from '$lib/components/ui/button/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
  import { Tabs, TabsContent, TabsList, TabsTrigger } from '$lib/components/ui/tabs/index.js';
  
  // Icons
  import { 
    Zap, 
    Wifi, 
    TrendingUp, 
    Activity,
    Search,
    Database,
    MessageSquare,
    Settings
  } from 'lucide-svelte';

  // Real-time search service
  const { state, isReady, hasResults, searchStatus, search } = useRealTimeSearch();

  // Demo state
  let selectedResult: any = $state(null);
  let testQueries = $state([
    'Fourth Amendment search and seizure',
    'Miranda rights violations',
    'DNA evidence admissibility',
    'Constitutional due process',
    'Criminal liability standards',
    'Evidence chain of custody',
    'Federal court precedents',
    'Search warrant requirements'
  ]);
  
  let demoMetrics = $state({
    totalSearches: 0,
    avgResponseTime: 0,
    realTimeConnections: 0,
    vectorSimilarityQueries: 0
  });

  // Handle search result selection
  function handleSearchSelect(event: CustomEvent) {
    selectedResult = event.detail.result;
    console.log('🔍 Selected result:', selectedResult);
    
    demoMetrics.totalSearches++;
  }

  // Test specific search
  async function testSearch(query: string) {
    console.log('🧪 Testing search:', query);
    try {
      await search(query, {
        categories: ['cases', 'evidence', 'precedents'],
        vectorSearch: true,
        streamResults: true,
        includeAI: true
      });
      demoMetrics.vectorSimilarityQueries++;
    } catch (error) {
      console.error('❌ Test search failed:', error);
    }
  }

  // Component lifecycle
  onMount(() => {
    console.log('🚀 Real-Time Search Demo page loaded');
  });
</script>

<svelte:head>
  <title>Real-Time Legal Search Demo - YoRHa Legal AI</title>
  <meta name="description" content="Demonstration of real-time legal search with WebSocket streaming, vector similarity, and AI enhancement" />
</svelte:head>

<div class="container mx-auto px-4 py-8">
  <!-- Page Header -->
  <div class="mb-8 text-center">
    <h1 class="text-4xl font-bold text-gray-900 mb-4">
      <span class="flex items-center justify-center gap-2">
        <Zap class="w-10 h-10 text-blue-500" />
        Real-Time Legal Search
      </span>
    </h1>
    <p class="text-lg text-gray-600 max-w-3xl mx-auto">
      Experience the next generation of legal research with real-time WebSocket streaming, 
      vector similarity search, and AI-powered result enhancement.
    </p>
  </div>

  <!-- Connection Status Banner -->
  <div class="mb-6">
    {#if $state.connectionStatus === 'connected'}
      <div class="flex items-center justify-center gap-2 bg-green-50 border border-green-200 rounded-lg p-3">
        <Wifi class="w-5 h-5 text-green-500" />
        <span class="text-green-700 font-medium">Real-time connection established</span>
        <Badge variant="secondary" class="bg-green-100 text-green-800">
          Enhanced RAG Active
        </Badge>
      </div>
    {:else if $state.connectionStatus === 'connecting'}
      <div class="flex items-center justify-center gap-2 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
        <Activity class="w-5 h-5 text-yellow-500 animate-pulse" />
        <span class="text-yellow-700 font-medium">Connecting to real-time services...</span>
      </div>
    {:else}
      <div class="flex items-center justify-center gap-2 bg-blue-50 border border-blue-200 rounded-lg p-3">
        <Database class="w-5 h-5 text-blue-500" />
        <span class="text-blue-700 font-medium">Using HTTP fallback mode</span>
        <Badge variant="secondary" class="bg-blue-100 text-blue-800">
          Standard Search
        </Badge>
      </div>
    {/if}
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
    <!-- Main Search Interface -->
    <div class="lg:col-span-2">
      <Card class="mb-6">
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Search class="w-5 h-5" />
            Legal AI Search Interface
          </CardTitle>
          <CardDescription>
            Search legal cases, evidence, precedents, and statutes with real-time streaming results
          </CardDescription>
        </CardHeader>
        <CardContent>
          <!-- Real-Time Search Component -->
          <RealTimeLegalSearch 
            placeholder="Search cases, precedents, evidence, statutes..."
            categories={['cases', 'evidence', 'precedents', 'statutes', 'documents']}
            enableRealTime={true}
            enableVectorSearch={true}
            enableAI={true}
            maxResults={15}
            autoSearch={true}
            select={handleSearchSelect}
          />
        </CardContent>
      </Card>

      <!-- Selected Result Details -->
      {#if selectedResult}
        <Card>
          <CardHeader>
            <CardTitle>Selected Result Details</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-4">
              <!-- Result Header -->
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <h3 class="text-lg font-semibold text-gray-900">
                    {selectedResult.title}
                  </h3>
                  <div class="flex items-center gap-2 mt-1">
                    <Badge variant={selectedResult.realTime ? 'default' : 'secondary'}>
                      {selectedResult.type}
                    </Badge>
                    {#if selectedResult.realTime}
                      <Badge variant="outline" class="text-blue-600 border-blue-200">
                        <TrendingUp class="w-3 h-3 mr-1" />
                        Real-time
                      </Badge>
                    {/if}
                    <span class="text-sm text-gray-500">
                      Relevance: {(selectedResult.score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <!-- Content -->
              <div class="prose prose-sm max-w-none">
                <p class="text-gray-700">{selectedResult.content}</p>
              </div>

              <!-- Metadata -->
              <div class="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 class="font-medium text-gray-900 mb-2">Metadata</h4>
                  <div class="space-y-1 text-sm text-gray-600">
                    {#if selectedResult.metadata.jurisdiction}
                      <div><span class="font-medium">Jurisdiction:</span> {selectedResult.metadata.jurisdiction}</div>
                    {/if}
                    {#if selectedResult.metadata.status}
                      <div><span class="font-medium">Status:</span> {selectedResult.metadata.status}</div>
                    {/if}
                    {#if selectedResult.metadata.date}
                      <div><span class="font-medium">Date:</span> {new Date(selectedResult.metadata.date).toLocaleDateString()}</div>
                    {/if}
                  </div>
                </div>
                <div>
                  <h4 class="font-medium text-gray-900 mb-2">Search Details</h4>
                  <div class="space-y-1 text-sm text-gray-600">
                    <div><span class="font-medium">Source:</span> {selectedResult.realTime ? 'Real-time stream' : 'Database query'}</div>
                    {#if selectedResult.similarity}
                      <div><span class="font-medium">Vector similarity:</span> {(selectedResult.similarity * 100).toFixed(1)}%</div>
                    {/if}
                  </div>
                </div>
              </div>

              <!-- Highlights -->
              {#if selectedResult.highlights && selectedResult.highlights.length > 0}
                <div>
                  <h4 class="font-medium text-gray-900 mb-2">Highlights</h4>
                  <div class="space-y-2">
                    {#each selectedResult.highlights as highlight}
                      <div class="p-2 bg-yellow-50 border border-yellow-200 rounded text-sm">
                        ...{highlight}...
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}
            </div>
          </CardContent>
        </Card>
      {/if}
    </div>

    <!-- Sidebar -->
    <div class="space-y-6">
      <!-- Quick Test Searches -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Settings class="w-4 h-4" />
            Quick Tests
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-2">
            {#each testQueries as query}
              <Button 
                variant="outline" 
                size="sm" 
                class="w-full justify-start text-left h-auto p-2"
                on:on:click={() => testSearch(query)}
              >
                <div class="text-xs truncate">{query}</div>
              </Button>
            {/each}
          </div>
        </CardContent>
      </Card>

      <!-- Real-Time Status -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Activity class="w-4 h-4" />
            System Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-3 text-sm">
            <div class="flex justify-between">
              <span>Connection Status:</span>
              <Badge variant={$state.connectionStatus === 'connected' ? 'default' : 'secondary'}>
                {$state.connectionStatus}
              </Badge>
            </div>
            
            <div class="flex justify-between">
              <span>Search Status:</span>
              <Badge variant="outline">
                {$searchStatus}
              </Badge>
            </div>

            <div class="flex justify-between">
              <span>Active Results:</span>
              <span class="font-medium">{$state.results.length}</span>
            </div>

            {#if $state.searchMetrics.totalQueries > 0}
              <div class="flex justify-between">
                <span>Total Queries:</span>
                <span class="font-medium">{$state.searchMetrics.totalQueries}</span>
              </div>
              
              <div class="flex justify-between">
                <span>Avg Response Time:</span>
                <span class="font-medium">{$state.searchMetrics.averageResponseTime}ms</span>
              </div>
            {/if}
          </div>
        </CardContent>
      </Card>

      <!-- Search Features -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Zap class="w-4 h-4" />
            Features Active
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-2 text-sm">
            <div class="flex items-center justify-between">
              <span>Real-time Streaming</span>
              <div class="w-2 h-2 rounded-full bg-green-400"></div>
            </div>
            
            <div class="flex items-center justify-between">
              <span>Vector Similarity</span>
              <div class="w-2 h-2 rounded-full bg-blue-400"></div>
            </div>
            
            <div class="flex items-center justify-between">
              <span>AI Enhancement</span>
              <div class="w-2 h-2 rounded-full bg-purple-400"></div>
            </div>
            
            <div class="flex items-center justify-between">
              <span>WebSocket Connection</span>
              <div class="w-2 h-2 rounded-full {$state.connectionStatus === 'connected' ? 'bg-green-400' : 'bg-gray-400'}"></div>
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- API Endpoints -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <MessageSquare class="w-4 h-4" />
            API Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-2 text-xs">
            <div class="flex justify-between items-center">
              <span>Enhanced RAG (8094)</span>
              <Badge variant="outline" class="text-green-600">
                Active
              </Badge>
            </div>
            
            <div class="flex justify-between items-center">
              <span>Upload Service (8093)</span>
              <Badge variant="outline" class="text-green-600">
                Active
              </Badge>
            </div>
            
            <div class="flex justify-between items-center">
              <span>WebSocket Stream</span>
              <Badge variant="outline" class="{$state.connectionStatus === 'connected' ? 'text-green-600' : 'text-gray-500'}">
                {$state.connectionStatus === 'connected' ? 'Connected' : 'Pending'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  </div>

  <!-- Technical Information -->
  <div class="mt-12">
    <Tabs value="overview" class="w-full">
      <TabsList class="grid w-full grid-cols-4">
        <TabsTrigger value="overview">Overview</TabsTrigger>
        <TabsTrigger value="architecture">Architecture</TabsTrigger>
        <TabsTrigger value="features">Features</TabsTrigger>
        <TabsTrigger value="performance">Performance</TabsTrigger>
      </TabsList>
      
      <TabsContent value="overview" class="mt-6">
        <Card>
          <CardHeader>
            <CardTitle>Real-Time Legal Search Overview</CardTitle>
            <CardDescription>
              Advanced legal research platform with streaming results and AI enhancement
            </CardDescription>
          </CardHeader>
          <CardContent class="prose prose-sm max-w-none">
            <p>
              This demonstration showcases a production-ready real-time legal search system built with:
            </p>
            <ul>
              <li><strong>Svelte 5</strong> with enhanced reactivity and performance</li>
              <li><strong>SvelteKit 2</strong> for full-stack application architecture</li>
              <li><strong>bits-ui v2</strong> for accessible, composable UI components</li>
              <li><strong>WebSocket streaming</strong> for real-time result delivery</li>
              <li><strong>Vector similarity search</strong> using enhanced embeddings</li>
              <li><strong>AI-powered enhancement</strong> with semantic analysis</li>
            </ul>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="architecture" class="mt-6">
        <Card>
          <CardHeader>
            <CardTitle>System Architecture</CardTitle>
          </CardHeader>
          <CardContent class="prose prose-sm max-w-none">
            <h3>Service Integration</h3>
            <ul>
              <li><strong>Enhanced RAG Service (8094):</strong> AI processing and vector search</li>
              <li><strong>Upload Service (8093):</strong> Document processing and metadata</li>
              <li><strong>WebSocket Streaming:</strong> Real-time result delivery</li>
              <li><strong>NATS Messaging:</strong> Distributed event system (planned)</li>
            </ul>
            
            <h3>Data Flow</h3>
            <ol>
              <li>User types query in search interface</li>
              <li>Debounced search triggers WebSocket message</li>
              <li>Enhanced RAG service processes query with vector similarity</li>
              <li>Results stream back in real-time chunks</li>
              <li>AI enhancement adds semantic analysis to top results</li>
              <li>UI updates reactively with Svelte 5 stores</li>
            </ol>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="features" class="mt-6">
        <Card>
          <CardHeader>
            <CardTitle>Advanced Features</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 class="font-semibold mb-3">Real-Time Features</h4>
                <ul class="space-y-2 text-sm">
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-green-400"></div>
                    WebSocket streaming results
                  </li>
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-blue-400"></div>
                    Live search suggestions
                  </li>
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-purple-400"></div>
                    Progressive result enhancement
                  </li>
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-yellow-400"></div>
                    Connection status monitoring
                  </li>
                </ul>
              </div>
              
              <div>
                <h4 class="font-semibold mb-3">AI Enhancement</h4>
                <ul class="space-y-2 text-sm">
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-indigo-400"></div>
                    Vector similarity scoring
                  </li>
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-pink-400"></div>
                    Semantic result analysis
                  </li>
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-teal-400"></div>
                    Legal concept extraction
                  </li>
                  <li class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-orange-400"></div>
                    Context-aware suggestions
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="performance" class="mt-6">
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="text-center p-4 bg-gray-50 rounded-lg">
                <div class="text-2xl font-bold text-blue-600">
                  {$state.searchMetrics.totalQueries || 0}
                </div>
                <div class="text-sm text-gray-600">Total Queries</div>
              </div>
              
              <div class="text-center p-4 bg-gray-50 rounded-lg">
                <div class="text-2xl font-bold text-green-600">
                  {$state.searchMetrics.averageResponseTime || 0}ms
                </div>
                <div class="text-sm text-gray-600">Avg Response</div>
              </div>
              
              <div class="text-center p-4 bg-gray-50 rounded-lg">
                <div class="text-2xl font-bold text-purple-600">
                  {$state.results.length}
                </div>
                <div class="text-sm text-gray-600">Live Results</div>
              </div>
              
              <div class="text-center p-4 bg-gray-50 rounded-lg">
                <div class="text-2xl font-bold text-orange-600">
                  {$state.connectionStatus === 'connected' ? '100%' : '0%'}
                </div>
                <div class="text-sm text-gray-600">Uptime</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  </div>
</div>

<style>
  :global(.prose ul li) {
    @apply mb-1;
  }
  
  :global(.prose ol li) {
    @apply mb-1;
  }
</style>