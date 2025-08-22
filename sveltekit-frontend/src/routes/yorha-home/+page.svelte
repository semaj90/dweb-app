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

  // System data for command center (use any to avoid strict prop typing noise in quick pass)
  let systemData: any = {
    activeCases: 12,
    evidenceItems: 234,
    personsOfInterest: 8,
    aiQueries: 1847,
    systemLoad: 45,
    gpuUtilization: 78,
    memoryUsage: 62,
    networkLatency: 23
  };

  // API response states
  let ragResult: any = null;
  let searchResults: any[] = [];
  let isLoading: boolean = false;
  let activeSection: string = 'dashboard';

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
    isLoading = true;
    ragResult = null;

    try {
      const response = await fetch('/api/yorha/enhanced-rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context: 'legal_analysis' })
      });

      if (response.ok) {
        ragResult = await response.json();
        systemData.aiQueries += 1;
        activeSection = 'rag-results';
      }
    } catch (error) {
      console.error('RAG query failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function performSemanticSearch(searchTerm: string = "contract liability") {
    isLoading = true;
    searchResults = [];

    try {
      const response = await fetch(`/api/yorha/legal-data?search=${encodeURIComponent(searchTerm)}&limit=10`);

      if (response.ok) {
        const data = await response.json();
        const results = Array.isArray(data?.results) ? data.results : [];
        searchResults = results.map((item: any, index: number) => ({
          id: (item && (item.id ?? item._id)) || index + 1,
          title: (item && (item.title ?? item.name)) || `Document ${index + 1}`,
          type: (item && item.type) || 'Legal Document',
          relevance: Math.round(((item && (item.relevance ?? item.score)) ?? Math.random()) * 100),
          status: (item && item.status) || 'active',
          metadata: item
        }));
        activeSection = 'search-results';
      }
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function checkClusterHealth() {
    isLoading = true;

    try {
      const response = await fetch('/api/v1/cluster/health');
      if (response.ok) {
        const healthData = await response.json();
        console.log('Cluster health:', healthData);
        // Update system data based on health response
        if (healthData.services) {
          systemData.systemLoad = healthData.cpu_usage || systemData.systemLoad;
          systemData.memoryUsage = healthData.memory_usage || systemData.memoryUsage;
        }
        activeSection = 'system-health';
      }
    } catch (error) {
      console.error('Health check failed:', error);
    } finally {
      isLoading = false;
    }
  }

  function handleTableAction({ row }: { row: any }) {
    console.log('Table action for:', row);
    // Navigate to detailed view or perform action (safe id extraction)
    const id = (row && (row.id ?? row._id)) ?? 'unknown';
    goto(`/evidence/${id}`);
  }
</script>

<svelte:head>
  <title>YoRHa Interface - Legal AI Command Center</title>
  <meta name="description" content="YoRHa-themed interface for Legal AI system access and demonstrations." />
</svelte:head>

<div class="yorha-interface">
  <!-- Header -->
  <section class="yorha-header">
    <div class="yorha-header-content">
      <div class="yorha-nav">
        <button
          class="yorha-nav-btn"
          onclick={() => navigateTo('/')}
        >
          <Home size={16} />
          MAIN SYSTEM
        </button>
      </div>

      <div class="yorha-title-section">
        <h1 class="yorha-main-title">
          <Terminal size={48} class="yorha-title-icon" />
          YoRHa INTERFACE
        </h1>
        <div class="yorha-subtitle">LEGAL AI COMMAND CENTER</div>
        <div class="yorha-status">
          <Activity size={16} />
          SYSTEM OPERATIONAL
        </div>
      </div>
    </div>
  </section>

  <!-- API Integration Controls -->
  <section class="yorha-api-section">
    <div class="yorha-api-controls">
      <h2 class="yorha-section-title">
        <Bot size={24} />
        LIVE API INTEGRATION
      </h2>

      <!-- API Action Buttons -->
      <div class="yorha-api-grid">
        <button
          class="yorha-api-btn yorha-api-rag"
          onclick={() => performRAGQuery()}
          disabled={isLoading}
        >
          <Cpu size={20} />
          RAG QUERY
          {#if isLoading && activeSection === 'rag-results'}
            <div class="yorha-spinner"></div>
          {/if}
        </button>

        <button
          class="yorha-api-btn yorha-api-search"
          onclick={() => performSemanticSearch()}
          disabled={isLoading}
        >
          <Search size={20} />
          SEMANTIC SEARCH
          {#if isLoading && activeSection === 'search-results'}
            <div class="yorha-spinner"></div>
          {/if}
        </button>

        <button
          class="yorha-api-btn yorha-api-health"
          onclick={() => checkClusterHealth()}
          disabled={isLoading}
        >
          <Monitor size={20} />
          CLUSTER HEALTH
          {#if isLoading && activeSection === 'system-health'}
            <div class="yorha-spinner"></div>
          {/if}
        </button>

        <button
          class="yorha-api-btn yorha-api-database"
          onclick={() => performSemanticSearch('database evidence')}
          disabled={isLoading}
        >
          <Database size={20} />
          DATABASE QUERY
        </button>
      </div>

      <!-- Live API Results Display -->
      {#if ragResult}
        <div class="yorha-results-section">
          <h3 class="yorha-results-title">RAG QUERY RESULTS</h3>
          <div class="yorha-results-content">
            <pre class="yorha-json-display">{JSON.stringify(ragResult, null, 2)}</pre>
          </div>
        </div>
      {/if}

      {#if searchResults.length > 0}
        <div class="yorha-results-section">
          <h3 class="yorha-results-title">SEARCH RESULTS ({searchResults.length})</h3>
          <div class="yorha-table-wrapper">
            <YoRHaTable
              columns={tableColumns}
              data={searchResults}
              selectable={true}
              pagination={true}
              pageSize={5}
              glitchEffect={true}
            >
              <svelte:fragment slot="actions" let:row>
                <button class="yorha-action-btn-sm" onclick={() => handleTableAction({ row })}>
                  VIEW
                </button>
                <button class="yorha-action-btn-sm" onclick={() => console.log('Edit:', row)}>
                  EDIT
                </button>
              </svelte:fragment>
            </YoRHaTable>
          </div>
        </div>
      {/if}
    </div>
  </section>

  <!-- YoRHa Command Center Integration -->
  {#if activeSection === 'dashboard' || activeSection === 'system-health'}
    <section class="yorha-dashboard-section">
      <YoRHaCommandCenter {systemData} />
    </section>
  {/if}

  <!-- Main Interface Options -->
  <section class="yorha-main-section">
    <div class="yorha-interface-grid">
      <!-- YoRHa Interface -->
      <div class="yorha-interface-card yorha-card-primary" onclick={() => navigateTo('/yorha')}>
        <h2 class="yorha-card-title">YORHA INTERFACE</h2>
        <p class="yorha-card-description">
          Complete YoRHa-themed interface with 3D components, dashboard, and API testing suite
        </p>
        <!-- removed per-file $state/$derived imports; runes provided globally -->
      </div>
        <div class="yorha-card-stats">
          <span>3D UI COMPONENTS</span>
          <span>LIVE MONITORING</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/yorha</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>

      <!-- Demo Center -->
      <div class="yorha-interface-card" onclick={() => navigateTo('/demos')}>
        <div class="yorha-card-header">
          <Gamepad2 size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">DEMO CENTER</h2>
        </div>
        <p class="yorha-card-description">
          Access comprehensive interactive demonstrations of all Legal AI system capabilities
        </p>
        <div class="yorha-card-stats">
          <span>27+ INTERACTIVE DEMOS</span>
          <span>8 CATEGORIES</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/demos</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>

      <!-- Live RAG API -->
      <div class="yorha-interface-card yorha-api-live" onclick={() => performRAGQuery('Live legal analysis')}>
        <div class="yorha-card-header">
          <Cpu size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">LIVE RAG API</h2>
        </div>
        <p class="yorha-card-description">
          Real-time RAG queries to your Enhanced AI service running on port 8094
        </p>
        <div class="yorha-card-stats">
          <span>REAL-TIME AI</span>
          <span>GPU ACCELERATED</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/api/yorha/enhanced-rag</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>

      <!-- YoRHa Dashboard -->
      <div class="yorha-interface-card" onclick={() => {activeSection = 'dashboard'; systemData = {...systemData, aiQueries: systemData.aiQueries + 1};}}>
        <div class="yorha-card-header">
          <Monitor size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">COMMAND CENTER</h2>
        </div>
        <p class="yorha-card-description">
          YoRHa command center with live system metrics and health monitoring
        </p>
        <div class="yorha-card-stats">
          <span>SYSTEM MONITORING</span>
          <span>REAL-TIME DATA</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">INTEGRATED</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>

      <!-- Live Database API -->
      <div class="yorha-interface-card yorha-api-db" onclick={() => performSemanticSearch('legal precedents')}>
        <div class="yorha-card-header">
          <Database size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">DATABASE API</h2>
        </div>
        <p class="yorha-card-description">
          Query legal database with semantic search and vector similarity
        </p>
        <div class="yorha-card-stats">
          <span>VECTOR SEARCH</span>
          <span>PGVECTOR</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/api/yorha/legal-data</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>
    </div>
  </section>

  <!-- Footer Info -->
  <section class="yorha-footer">
    <div class="yorha-footer-content">
      <div class="yorha-system-info">
        <h3>SYSTEM INFORMATION</h3>
        <div class="yorha-info-grid">
          <div class="yorha-info-item">
            <strong>Frontend:</strong> SvelteKit 2 + Svelte 5
          </div>
          <div class="yorha-info-item">
            <strong>UI Framework:</strong> bits-ui + melt-ui + shadcn-svelte
          </div>
          <div class="yorha-info-item">
            <strong>Theme:</strong> YoRHa Cyberpunk Interface
          </div>
          <div class="yorha-info-item">
            <strong>Status:</strong> Production Ready
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

<style>
  .yorha-interface {
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
</style>