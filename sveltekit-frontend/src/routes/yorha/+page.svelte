<!-- YoRHa Main Interface Page -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { yorhaAPI } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  import YoRHaCommandCenter from '$lib/components/yorha/YoRHaCommandCenter.svelte';
  import { onMount } from 'svelte';
  import {
    Play,
    Terminal,
    Settings,
    Monitor,
    ChevronRight,
    Gamepad2,
    Activity,
    Cpu,
    Database,
    Search,
    FileText,
    Bot,
    Zap
  } from 'lucide-svelte';
  import { debounce, withAbort } from '$lib/yorha/constants';
  import YoRHaNavCard from '$lib/components/yorha/YoRHaNavCard.svelte';
  import { ensureLocalIndex, localSearch, isLocalIndexReady, getLocalDocumentCount, wasLoadedFromCache, mergeResults } from '$lib/yorha/localSearch';
  import { initHybridLayer, reRankWithPgVector } from '$lib/yorha/hybridSearchManager';

  // System data for command center
  let systemData = $state({
    activeCases: 12,
    evidenceItems: 234,
    personsOfInterest: 8,
    aiQueries: 1847,
    systemLoad: 45,
    gpuUtilization: 78,
    memoryUsage: 62,
    networkLatency: 23
  });

  // API response states
  let ragResult = $state(null);
  let searchResults = $state([]);
  let isLoading = $state(false);
  let activeSection = $state('dashboard');
  let layoutData = $state(null);
  let searchMode = $state<'local' | 'hybrid' | 'remote'>('hybrid');
  let localIndexReady = $state(false);
  let localIndexCount = $state(0);
  let localLoadedFromCache = $state(false);

  onMount(() => {
    // Fire and forget async initialization
    (async () => {
      try {
        const layout = await yorhaAPI.loadLayout('/api/yorha/layout');
        layoutData = layout;
        yorhaAPI.startDataStreams();
      } catch (error) {
        console.warn('YoRHa layout not available:', error);
      }
    })();

    // Update system data periodically
    const interval = setInterval(() => {
      systemData = {
        ...systemData,
        systemLoad: Math.max(20, Math.min(90, systemData.systemLoad + (Math.random() - 0.5) * 10)),
        gpuUtilization: Math.max(30, Math.min(95, systemData.gpuUtilization + (Math.random() - 0.5) * 8)),
        memoryUsage: Math.max(40, Math.min(85, systemData.memoryUsage + (Math.random() - 0.5) * 6)),
        networkLatency: Math.max(10, Math.min(100, systemData.networkLatency + (Math.random() - 0.5) * 5))
      };
    }, 3000);

    return () => clearInterval(interval);
  });

  function navigateTo(path: string) {
    goto(path);
  }

  // API integration functions
  async function performRAGQuery(query: string = 'Legal case precedent analysis') {
    isLoading = true;
    ragResult = null;
    const { promise, abort } = withAbort(async (signal) => {
      const response = await fetch('/api/yorha/enhanced-rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context: 'legal_analysis' }),
        signal
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
    try {
      const data = await promise;
      ragResult = data;
      systemData.aiQueries += 1;
      activeSection = 'rag-results';
    } catch (e) {
      if ((e as any).name !== 'AbortError') console.error('RAG query failed', e);
    } finally {
      isLoading = false;
    }
    return () => abort();
  }

  async function performSemanticSearch(searchTerm: string = 'contract liability') {
    isLoading = true;
    searchResults = [];
    let localResults: any[] = [];
    if (isLocalIndexReady() && (searchMode === 'local' || searchMode === 'hybrid')) {
      localResults = localSearch(searchTerm, 50);
      if (searchMode === 'local') {
        searchResults = localResults as any;
        activeSection = 'search-results';
        isLoading = false;
        return;
      }
    }
    const { promise, abort } = withAbort(async (signal) => {
      if (searchMode === 'local') return { results: [] }; // guard
      const response = await fetch(`/api/yorha/legal-data?search=${encodeURIComponent(searchTerm)}&limit=25`, { signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
    try {
      const data = await promise;
      const remote = (data.results || []).map((item: any, index: number) => ({
        id: item.id || index + 1,
        title: item.title || item.name || `Document ${index + 1}`,
        type: item.type || 'Legal Document',
        relevance: Math.round((item.relevance || Math.random()) * 100),
        status: item.status || 'active',
        metadata: item
      }));
      searchResults = searchMode === 'hybrid' ? mergeResults(localResults, remote) : remote;
      activeSection = 'search-results';
    } catch (e) {
      if ((e as any).name !== 'AbortError') console.error('Search failed', e);
    } finally {
      isLoading = false;
    }
    return () => abort();
  }

  async function checkClusterHealth() {
    isLoading = true;
    const { promise, abort } = withAbort(async (signal) => {
      const response = await fetch('/api/v1/cluster/health', { signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
    try {
      const healthData = await promise;
      if (healthData.services) {
        systemData.systemLoad = healthData.cpu_usage || systemData.systemLoad;
        systemData.memoryUsage = healthData.memory_usage || systemData.memoryUsage;
      }
      activeSection = 'system-health';
    } catch (e) {
      if ((e as any).name !== 'AbortError') console.error('Health check failed', e);
    } finally {
      isLoading = false;
    }
    return () => abort();
  }

  // Debounced helper for potential future search input
  const debouncedSearch = debounce((q: string) => performSemanticSearch(q), 400);

  // Build local index lazily after mount (non-blocking)
  onMount(() => {
    // Restore mode
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem('yorha-search-mode');
      if (saved === 'local' || saved === 'hybrid' || saved === 'remote') searchMode = saved;
    }
    ensureLocalIndex().then(() => {
      localIndexReady = true;
      localIndexCount = getLocalDocumentCount();
      localLoadedFromCache = wasLoadedFromCache();
      console.info('[YoRHa] Local search index ready:', localIndexCount, 'docs', 'cached=', localLoadedFromCache);
    });
    initHybridLayer({ refreshIntervalMs: 300000 });
  });

  $effect(() => {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('yorha-search-mode', searchMode);
    }
  });
</script>

<svelte:head>
  <title>YoRHa Interface - Command Center</title>
</svelte:head>

<div class="yorha-page">
  <!-- Hero Section -->
  <section class="yorha-hero">
    <div class="yorha-hero-content">
      <div class="yorha-hero-title">
        <Terminal size={64} class="yorha-hero-icon" />
        <h1>YoRHa COMMAND CENTER</h1>
        <div class="yorha-hero-subtitle">LEGAL AI SYSTEM INTERFACE</div>
      </div>

      <div class="yorha-hero-status">
        <div class="yorha-status-indicator yorha-status-online">
          <Activity size={20} />
          <span>SYSTEM OPERATIONAL</span>
        </div>
        <div class="yorha-hero-stats">
          <span>AI QUERIES: {systemData.aiQueries}</span>
          <span>CASES: {systemData.activeCases}</span>
          <span>EVIDENCE: {systemData.evidenceItems}</span>
        </div>
      </div>
    </div>
  </section>

  <!-- Quick Actions -->
  <section class="yorha-actions">
    <div class="yorha-actions-grid">
      <button
        class="yorha-action-card yorha-action-primary"
        onclick={() => performRAGQuery()}
        disabled={isLoading}
      >
        <Cpu size={32} />
        <h3>ENHANCED RAG</h3>
        <p>AI-powered legal analysis</p>
        {#if isLoading}
          <div class="yorha-loading-indicator"></div>
        {/if}
      </button>

      <button
        class="yorha-action-card yorha-action-search"
        onclick={() => performSemanticSearch()}
        disabled={isLoading}
      >
        <Search size={32} />
        <h3>VECTOR SEARCH</h3>
        <p>Semantic document retrieval</p>
      </button>

      <button
        class="yorha-action-card yorha-action-health"
        onclick={() => checkClusterHealth()}
        disabled={isLoading}
      >
        <Monitor size={32} />
        <h3>SYSTEM HEALTH</h3>
        <p>Cluster monitoring</p>
      </button>

      <button
        class="yorha-action-card yorha-action-database"
        onclick={() => performSemanticSearch('database evidence')}
        disabled={isLoading}
      >
        <Database size={32} />
        <h3>DATABASE</h3>
        <p>Direct data access</p>
      </button>
    </div>
  </section>

  <!-- YoRHa Command Center Integration -->
  <section class="yorha-dashboard">
    <YoRHaCommandCenter {systemData} />
  </section>

  <!-- Interface Navigation -->
  <section class="yorha-navigation">
    <h2 class="yorha-section-title">
      <Bot size={24} />
      INTERFACE MODULES
    </h2>

    <div class="yorha-nav-grid" role="grid" aria-label="YoRHa interface modules">
      <YoRHaNavCard title="SYSTEM DASHBOARD" description="Real-time monitoring and analytics" path="/yorha/dashboard" icon={Monitor} ariaLabel="Open System Dashboard">
        <ChevronRight size={16} slot="trailing" />
      </YoRHaNavCard>
      <YoRHaNavCard title="3D COMPONENTS" description="Interactive UI component gallery" path="/yorha/components" icon={Gamepad2} ariaLabel="Open 3D Components">
        <ChevronRight size={16} slot="trailing" />
      </YoRHaNavCard>
      <YoRHaNavCard title="API TESTING" description="Live API integration suite" path="/yorha/api-test" icon={Zap} ariaLabel="Open API Testing">
        <ChevronRight size={16} slot="trailing" />
      </YoRHaNavCard>
      <YoRHaNavCard title="TERMINAL" description="Command-line interface" path="/yorha/terminal" icon={Terminal} ariaLabel="Open Terminal">
        <ChevronRight size={16} slot="trailing" />
      </YoRHaNavCard>
      <YoRHaNavCard title="DATA GRID" description="Advanced data visualization" path="/yorha/data-grid" icon={Database} ariaLabel="Open Data Grid">
        <ChevronRight size={16} slot="trailing" />
      </YoRHaNavCard>
      <YoRHaNavCard title="AI CHAT" description="Enhanced conversation interface" path="/yorha/chat" icon={Bot} ariaLabel="Open AI Chat">
        <ChevronRight size={16} slot="trailing" />
      </YoRHaNavCard>
    </div>
  </section>

  <!-- Results Display -->
  <section class="yorha-search-box" aria-label="Semantic Search">
    <div class="yorha-search-inner max-w-6xl mx-auto">
      <label for="yorha-search" class="sr-only">Search legal documents</label>
      <input id="yorha-search" type="search" placeholder="Search legal documents..." class="yorha-search-input" oninput={(e) => debouncedSearch((e.target as HTMLInputElement).value)} />
      <div class="yorha-search-meta">
        <fieldset class="yorha-search-modes" aria-label="Search Mode">
          <legend class="sr-only">Search Mode</legend>
          <label><input type="radio" name="search-mode" value="local" bind:group={searchMode} /> Local</label>
          <label><input type="radio" name="search-mode" value="hybrid" bind:group={searchMode} /> Hybrid</label>
          <label><input type="radio" name="search-mode" value="remote" bind:group={searchMode} /> Remote</label>
        </fieldset>
        <div class="yorha-index-status" aria-live="polite">
          {#if !localIndexReady}
            <span class="yorha-index-loading">Building local index…</span>
          {:else}
            <span class="yorha-index-ready">Local index: {localIndexCount} docs ({localLoadedFromCache ? 'cached' : 'fresh'})</span>
          {/if}
        </div>
      </div>
    </div>
  </section>
  {#if ragResult}
    <section class="yorha-results">
      <h2 class="yorha-section-title">RAG QUERY RESULTS</h2>
      <div class="yorha-results-content">
        <pre class="yorha-json-display">{JSON.stringify(ragResult, null, 2)}</pre>
      </div>
    </section>
  {/if}

  {#if searchResults.length > 0}
    <section class="yorha-results">
      <h2 class="yorha-section-title">SEARCH RESULTS ({searchResults.length})</h2>
      <div class="yorha-search-results">
        {#each searchResults as result}
          <div class="yorha-result-item">
            <div class="yorha-result-header">
              <h4>{result.title}</h4>
              <span class="yorha-result-relevance">{result.relevance}%</span>
            </div>
            <div class="yorha-result-meta">
              <span class="yorha-result-type">{result.type}</span>
              <span class="yorha-result-status yorha-status-{result.status}">{result.status}</span>
            </div>
          </div>
        {/each}
      </div>
    </section>
  {/if}
</div>

<style lang="postcss">
  .yorha-page {
    @apply space-y-8 pb-16;
  }

  /* Hero Section */
  .yorha-hero {
    @apply py-16 px-6 border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, transparent 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-hero-content {
    @apply max-w-6xl mx-auto text-center space-y-8;
  }

  .yorha-hero-title h1 {
    @apply text-4xl md:text-6xl font-bold tracking-wider text-amber-400;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-hero-icon {
    @apply text-amber-400 mx-auto mb-4;
  }

  .yorha-hero-subtitle {
    @apply text-xl text-amber-300 tracking-wide opacity-80;
  }

  .yorha-hero-status {
    @apply flex flex-col md:flex-row items-center justify-center gap-6;
  }

  .yorha-status-indicator {
    @apply flex items-center gap-2 px-4 py-2 border;
  }

  .yorha-status-online {
    @apply border-green-400 text-green-400;
  }

  .yorha-hero-stats {
    @apply flex gap-6 text-sm text-amber-400 opacity-60;
  }

  /* Quick Actions */
  .yorha-actions {
    @apply px-6;
  }

  .yorha-actions-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto;
  }

  .yorha-action-card {
    @apply bg-gray-900 border-2 p-6 text-center space-y-4 transition-all duration-300;
    @apply hover:border-opacity-80 hover:bg-opacity-20 disabled:opacity-50;
  }

  .yorha-action-primary {
    @apply border-blue-400 text-blue-400 hover:bg-blue-400;
  }

  .yorha-action-search {
    @apply border-green-400 text-green-400 hover:bg-green-400;
  }

  .yorha-action-health {
    @apply border-purple-400 text-purple-400 hover:bg-purple-400;
  }

  .yorha-action-database {
    @apply border-orange-400 text-orange-400 hover:bg-orange-400;
  }

  .yorha-action-card h3 {
    @apply font-bold tracking-wider;
  }

  .yorha-action-card p {
    @apply text-xs opacity-80;
  }

  .yorha-loading-indicator {
    @apply w-6 h-6 border-2 border-current border-t-transparent rounded-full mx-auto;
    animation: spin 1s linear infinite;
  }

  /* Dashboard */
  .yorha-dashboard {
    @apply px-6;
  }

  /* Navigation */
  .yorha-navigation {
    @apply px-6 space-y-6;
  }

  .yorha-section-title {
    @apply text-2xl font-bold text-amber-400 tracking-wider flex items-center gap-3 max-w-6xl mx-auto;
  }

  .yorha-nav-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto;
  }

  .yorha-nav-card {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-6 cursor-pointer;
    @apply hover:border-opacity-60 transition-all duration-300 hover:bg-amber-900 hover:bg-opacity-10;
    @apply space-y-4;
  }

  /* Navigation card inner elements now live inside YoRHaNavCard component; keep styles global */
  :global(.yorha-nav-header h3) {
    @apply text-lg font-bold text-amber-400 tracking-wider;
  }
  :global(.yorha-nav-card p) {
    @apply text-amber-300 text-sm;
  }
  :global(.yorha-nav-footer span) {
    @apply text-xs text-amber-400 opacity-60 font-mono;
  }

  .yorha-nav-footer {
    @apply flex items-center justify-between pt-4 border-t border-amber-400 border-opacity-20;
  }

  /* Removed duplicate .yorha-nav-footer span (global style applied above) */

  /* Results */
  .yorha-results {
    @apply px-6 space-y-4;
  }

  .yorha-results-content {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-6 max-w-6xl mx-auto;
  }

  .yorha-json-display {
    @apply text-amber-300 text-xs font-mono whitespace-pre-wrap max-h-96 overflow-y-auto;
  }

  .yorha-search-box {
    @apply px-6 mt-4;
  }
  .yorha-search-inner {
    @apply space-y-3;
  }
  .yorha-search-input {
    @apply w-full bg-gray-900 border border-amber-400 border-opacity-30 px-4 py-2 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-amber-400;
  }
  .yorha-search-meta {
    @apply flex flex-col md:flex-row md:items-center md:justify-between gap-3 text-xs text-amber-300;
  }
  .yorha-search-modes {
    @apply flex gap-4;
  }
  .yorha-search-modes label {
    @apply flex items-center gap-1 cursor-pointer;
  }
  .yorha-index-loading { @apply animate-pulse text-amber-400; }
  .yorha-index-ready { @apply text-amber-400; }

  .yorha-search-results {
    @apply space-y-4 max-w-6xl mx-auto;
  }

  .yorha-result-item {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-4;
  }

  .yorha-result-header {
    @apply flex items-center justify-between mb-2;
  }

  .yorha-result-header h4 {
    @apply font-semibold text-amber-400;
  }

  .yorha-result-relevance {
    @apply text-xs bg-amber-400 text-black px-2 py-1 font-mono;
  }

  .yorha-result-meta {
    @apply flex gap-4 text-xs;
  }

  .yorha-result-type {
    @apply text-amber-300 opacity-80;
  }

  .yorha-result-status {
    @apply px-2 py-1 text-xs font-mono;
  }

  .yorha-status-active {
    @apply bg-green-400 text-black;
  }

  .yorha-status-pending {
    @apply bg-yellow-400 text-black;
  }

  .yorha-status-inactive {
    @apply bg-gray-400 text-black;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Responsive */
  @media (max-width: 768px) {
    .yorha-hero-title h1 {
      @apply text-3xl;
    }

    .yorha-hero-stats {
      @apply flex-col gap-2;
    }

    .yorha-actions-grid {
      @apply grid-cols-1 gap-4;
    }

    .yorha-nav-grid {
      @apply grid-cols-1 gap-4;
    }
  }
</style>