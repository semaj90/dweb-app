<!-- YoRHa Interface Home Page -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import {
    Play,
    Terminal,
    Settings,
    Monitor,
    ChevronRight,
    Home,
    Gamepad2,
    Activity,
    Cpu,
    Database,
    Search,
    FileText,
    Bot
  } from 'lucide-svelte';
  import YoRHaCommandCenter from '$lib/components/yorha/YoRHaCommandCenter.svelte';
  // tsserver sometimes reports Svelte components as having no default export — silence for now
  // @ts-ignore: Svelte component typing mismatch
  import YoRHaTable from '$lib/components/yorha/YoRHaTable.svelte';
  import { state } from 'svelte/reactivity';

  // System data for command center (use any to avoid strict prop typing noise in quick pass)
  let systemData = state({
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
  let ragResult = state<any>(null);
  let searchResults = state<any[]>([]);
  let isLoading = state<boolean>(false);
  let activeSection = state<string>('dashboard');

  // Table configuration for results
  const tableColumns = [
    { key: 'id', title: 'ID', sortable: true, width: '80px' },
    { key: 'title', title: 'Title', sortable: true },
    { key: 'type', title: 'Type', sortable: true, width: '120px' },
    { key: 'relevance', title: 'Relevance', sortable: true, width: '100px', type: 'number' },
    { key: 'status', title: 'Status', type: 'status', width: '120px' },
    { key: 'actions', title: 'Actions', type: 'action', width: '150px' }
  ];

  function navigateTo(path: string) {
    goto(path);
  }

  // API integration functions
  async function performRAGQuery(query: string = "Legal case precedent analysis") {
    isLoading.value = true;
    ragResult.value = null;

    try {
      const response = await fetch('/api/yorha/enhanced-rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context: 'legal_analysis' })
      });

      if (response.ok) {
        ragResult.value = await response.json();
        systemData.value.aiQueries += 1;
        activeSection.value = 'rag-results';
      }
    } catch (error) {
      console.error('RAG query failed:', error);
    } finally {
      isLoading.value = false;
    }
  }

  async function performSemanticSearch(searchTerm: string = "contract liability") {
    isLoading.value = true;
    searchResults.value = [];

    try {
      const response = await fetch(`/api/yorha/legal-data?search=${encodeURIComponent(searchTerm)}&limit=10`);

      if (response.ok) {
        const data = await response.json();
        const results = Array.isArray(data?.results) ? data.results : [];
        searchResults.value = results.map((item: any, index: number) => ({
          id: (item && (item.id ?? item._id)) || index + 1,
          title: (item && (item.title ?? item.name)) || `Document ${index + 1}`,
          type: (item && item.type) || 'Legal Document',
          relevance: Math.round(((item && (item.relevance ?? item.score)) ?? Math.random()) * 100),
          status: (item && item.status) || 'active',
          metadata: item
        }));
        activeSection.value = 'search-results';
      }
    } catch (error) {
      console.error('Semantic search failed:', error);
    } finally {
      isLoading.value = false;
    }
  }
</script>

<svelte:head>
  <title>YoRHa Interface - Legal AI Command Center</title>
  <meta name="description" content="YoRHa-themed interface for Legal AI system access and demonstrations." />
</svelte:head>

<div class="yorha-home-grid">
  <header class="yorha-header">
    <div class="header-left">
      <Home class="header-icon" />
      <h1>YoRHa Command Interface</h1>
    </div>
    <div class="header-right">
      <button class="yorha-btn-icon" on:click={() => navigateTo('/settings')}>
        <Settings />
      </button>
      <button class="yorha-btn-icon" on:click={() => navigateTo('/profile')}>
        <Gamepad2 />
      </button>
    </div>
  </header>

  <aside class="yorha-sidebar">
    <nav>
      <ul>
        <li>
          <button class="yorha-btn-sidebar" class:active={activeSection === 'dashboard'} on:click={() => activeSection.value = 'dashboard'}>
            <Monitor />
            <span>Dashboard</span>
          </button>
        </li>
        <li>
          <button class="yorha-btn-sidebar" class:active={activeSection === 'rag-results'} on:click={() => performRAGQuery()}>
            <Bot />
            <span>RAG Analysis</span>
          </button>
        </li>
        <li>
          <button class="yorha-btn-sidebar" class:active={activeSection === 'search-results'} on:click={() => performSemanticSearch()}>
            <Search />
            <span>Semantic Search</span>
          </button>
        </li>
        <li>
          <button class="yorha-btn-sidebar" on:click={() => navigateTo('/documents')}>
            <FileText />
            <span>Documents</span>
          </button>
        </li>
        <li>
          <button class="yorha-btn-sidebar" on:click={() => navigateTo('/yorha-terminal')}>
            <Terminal />
            <span>Terminal</span>
          </button>
        </li>
      </ul>
    </nav>
  </aside>

  <main class="yorha-main-content">
    {#if isLoading.value}
      <div class="loading-overlay">
        <div class="spinner"></div>
        <p>Processing...</p>
      </div>
    {/if}

    {#if activeSection.value === 'dashboard'}
      <section id="dashboard">
        <h2 class="section-title">System Dashboard</h2>
        <YoRHaCommandCenter bind:systemData={systemData.value} />
      </section>
    {/if}

    {#if activeSection.value === 'rag-results' && ragResult.value}
      <section id="rag-results">
        <h2 class="section-title">RAG Analysis Results</h2>
        <div class="rag-summary">
          <h3>Summary</h3>
          <p>{ragResult.value.summary}</p>
        </div>
        <div class="rag-details">
          <h3>Key Points</h3>
          <ul>
            {#each ragResult.value.keyPoints as point}
              <li>{point}</li>
            {/each}
          </ul>
        </div>
        <YoRHaTable title="Cited Sources" columns={tableColumns} data={ragResult.value.sources} />
      </section>
    {/if}

    {#if activeSection.value === 'search-results' && searchResults.value.length > 0}
      <section id="search-results">
        <h2 class="section-title">Semantic Search Results</h2>
        <YoRHaTable title="Found Documents" columns={tableColumns} bind:data={searchResults.value} />
      </section>
    {/if}
  </main>
</div>

<style>
/* ... existing styles ... */
.yorha-home-grid {
    @apply min-h-screen bg-black text-amber-400 font-mono;
    font-family: 'Courier New', monospace;
    background-image:
      radial-gradient(circle at 20% 50%, rgba(255, 191, 0, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 191, 0, 0.03) 0%, transparent 50%);
  }

  /* Header */
  .yorha-header {
    @apply py-16 px-6 border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, transparent 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-header-content {
    @apply max-w-6xl mx-auto;
  }

  .yorha-nav {
    @apply mb-8;
  }

  .yorha-nav-btn {
    @apply px-4 py-2 bg-amber-400 text-black font-mono text-sm tracking-wider;
    @apply hover:bg-amber-300 transition-colors flex items-center gap-2;
  }

  .yorha-title-section {
    @apply text-center space-y-6;
  }

  .yorha-main-title {
    @apply text-6xl md:text-8xl font-bold tracking-wider flex items-center justify-center gap-4;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-title-icon {
    @apply text-amber-400;
  }

  .yorha-subtitle {
    @apply text-2xl text-amber-300 tracking-wide opacity-80;
  }

  .yorha-status {
    @apply flex items-center justify-center gap-2 text-green-400 font-bold;
  }

  /* Main Section */
  .yorha-main-section {
    @apply py-16 px-6;
  }

  .yorha-interface-grid {
    @apply grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl mx-auto;
  }

  .yorha-interface-card {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-8 cursor-pointer;
    @apply hover:border-opacity-60 transition-all duration-300 hover:bg-amber-900 hover:bg-opacity-10;
    @apply space-y-6;
  }

  .yorha-card-primary {
    @apply border-amber-400 border-opacity-60;
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.2);
  }

  .yorha-card-header {
    @apply flex items-center gap-4;
  }

  .yorha-card-icon {
    @apply text-amber-400;
  }

  .yorha-card-title {
    @apply text-xl font-bold text-amber-400 tracking-wider;
  }

  .yorha-card-description {
    @apply text-amber-300 leading-relaxed;
  }

  .yorha-card-stats {
    @apply flex gap-4 text-xs text-amber-400 opacity-60;
  }

  .yorha-card-footer {
    @apply flex items-center justify-between pt-4 border-t border-amber-400 border-opacity-20;
  }

  .yorha-card-path {
    @apply text-xs text-amber-400 opacity-60 font-mono;
  }

  .yorha-card-arrow {
    @apply text-amber-300 opacity-60;
  }

  /* Footer */
  .yorha-footer {
    @apply border-t border-amber-400 border-opacity-30 bg-gray-900 bg-opacity-50 px-6 py-12;
  }

  .yorha-footer-content {
    @apply max-w-6xl mx-auto;
  }

  .yorha-system-info h3 {
    @apply text-xl font-bold text-amber-400 mb-6 tracking-wider;
  }

  .yorha-info-grid {
    @apply grid grid-cols-1 md:grid-cols-2 gap-4;
  }

  .yorha-info-item {
    @apply text-amber-300 border-l-2 border-amber-400 border-opacity-30 pl-4;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .yorha-main-title {
      @apply text-4xl;
    }

    .yorha-interface-grid {
      @apply grid-cols-1 gap-6;
    }

    .yorha-info-grid {
      @apply grid-cols-1 gap-3;
    }
  }

  /* API Integration Section */
  .yorha-api-section {
    @apply py-12 px-6 border-b border-amber-400 border-opacity-20;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-api-controls {
    @apply max-w-6xl mx-auto space-y-8;
  }

  .yorha-section-title {
    @apply text-2xl font-bold text-amber-400 mb-6 tracking-wider flex items-center gap-3;
  }

  .yorha-api-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8;
  }

  .yorha-api-btn {
    @apply bg-black border-2 border-amber-400 text-amber-400 px-6 py-4 font-mono text-sm tracking-wider;
    @apply hover:bg-amber-400 hover:text-black transition-all duration-300 flex items-center justify-center gap-3;
    @apply disabled:opacity-50 disabled:cursor-not-allowed;
  }

  .yorha-api-rag {
    @apply border-blue-400 text-blue-400 hover:bg-blue-400;
  }

  .yorha-api-search {
    @apply border-green-400 text-green-400 hover:bg-green-400;
  }

  .yorha-api-health {
    @apply border-purple-400 text-purple-400 hover:bg-purple-400;
  }

  .yorha-api-database {
    @apply border-orange-400 text-orange-400 hover:bg-orange-400;
  }

  .yorha-spinner {
    @apply w-4 h-4 border-2 border-current border-t-transparent rounded-full;
    animation: spin 1s linear infinite;
  }

  /* Results Display */
  .yorha-results-section {
    @apply mt-8 bg-gray-900 border border-amber-400 border-opacity-30 p-6;
  }

  .yorha-results-title {
    @apply text-lg font-bold text-amber-400 mb-4 tracking-wider;
  }

  .yorha-results-content {
    @apply bg-black border border-amber-400 border-opacity-20 p-4 rounded;
  }

  .yorha-json-display {
    @apply text-amber-300 text-xs font-mono whitespace-pre-wrap max-h-96 overflow-y-auto;
  }

  .yorha-table-wrapper {
    @apply bg-black border border-amber-400 border-opacity-30 rounded;
  }

  /* Dashboard Section */
  .yorha-dashboard-section {
    @apply py-8 px-6 bg-gray-900 bg-opacity-50;
  }

  /* Enhanced Card Styles */
  .yorha-api-live {
    @apply border-blue-400 border-opacity-60;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
  }

  .yorha-api-db {
    @apply border-orange-400 border-opacity-60;
    box-shadow: 0 0 20px rgba(251, 146, 60, 0.2);
  }

  /* Hover animations */
  .yorha-interface-card:hover .yorha-card-arrow {
    @apply text-amber-400 opacity-100;
    transform: translateX(4px);
  }

  .yorha-interface-card:hover .yorha-card-icon {
    text-shadow: 0 0 10px rgba(255, 191, 0, 0.5);
  }

  .yorha-api-btn:hover {
    text-shadow: 0 0 10px currentColor;
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.3);
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .yorha-btn-sidebar.active {
  background-color: var(--yorha-accent);
  color: var(--yorha-bg-primary);
  box-shadow: 0 0 15px var(--yorha-accent);
}
</style>