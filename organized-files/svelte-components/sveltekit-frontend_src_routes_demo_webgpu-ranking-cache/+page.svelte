<!--
  WebGPU Ranking Cache Demo
  Showcases client-side WebAssembly ranking cache with GPU acceleration
  Features: Concurrency, real-time performance monitoring, NES memory integration
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { integratedCacheSystem, type SearchContext } from '$lib/services/integrated-webgpu-cache-system';
  import CachePerformanceMonitor from '$lib/components/dashboard/CachePerformanceMonitor.svelte';

  // Demo state
  let searchQuery = $state('contract liability insurance coverage');
  let searchResults = $state<any[]>([]);
  let isSearching = $state(false);
  let lastSearchTime = $state<number>(0);
  
  // Performance metrics
  let systemMetrics = $state<any>({});
  let operationHistory = $state<any[]>([]);
  
  // Demo configuration
  let enableWebGPU = $state(true);
  let enableWASM = $state(true);
  let enableNESMemory = $state(true);
  let compressionLevel = $state<'none' | 'low' | 'high'>('high');
  let maxResults = $state(10);
  let similarityThreshold = $state(0.7);

  // Demo data
  const sampleDocuments = [
    {
      id: '1',
      type: 'contract' as const,
      priority: 100,
      size: 15420,
      confidenceLevel: 0.95,
      riskLevel: 'medium' as const,
      lastAccessed: Date.now() - 3600000,
      bankId: 1,
      compressed: false,
      metadata: {
        description: 'Commercial liability insurance contract with comprehensive coverage terms',
        vectorEmbedding: new Float32Array(384).map(() => Math.random() - 0.5),
        sourceUrl: 'https://example.com/contracts/liability-001',
        keywords: ['liability', 'insurance', 'coverage', 'commercial', 'terms']
      }
    },
    {
      id: '2',
      type: 'evidence' as const,
      priority: 85,
      size: 8760,
      confidenceLevel: 0.87,
      riskLevel: 'high' as const,
      lastAccessed: Date.now() - 1800000,
      bankId: 2,
      compressed: true,
      metadata: {
        description: 'Email correspondence regarding coverage disputes and claim denials',
        vectorEmbedding: new Float32Array(384).map(() => Math.random() - 0.5),
        sourceUrl: 'https://example.com/evidence/emails-dispute',
        keywords: ['coverage', 'dispute', 'claim', 'denial', 'correspondence']
      }
    },
    {
      id: '3',
      type: 'brief' as const,
      priority: 92,
      size: 25600,
      confidenceLevel: 0.91,
      riskLevel: 'low' as const,
      lastAccessed: Date.now() - 7200000,
      bankId: 3,
      compressed: false,
      metadata: {
        description: 'Legal brief analyzing precedent cases for insurance coverage interpretation',
        vectorEmbedding: new Float32Array(384).map(() => Math.random() - 0.5),
        sourceUrl: 'https://example.com/briefs/coverage-precedent',
        keywords: ['precedent', 'coverage', 'interpretation', 'legal', 'analysis']
      }
    }
  ];

  // Monitoring intervals
  let metricsInterval: number | null = null;
  let demoInterval: number | null = null;

  /**
   * Initialize demo and load sample data
   */
  onMount(async () => {
    try {
      console.log('🎮 Initializing WebGPU Ranking Cache Demo');
      
      // Load sample documents into cache
      for (const doc of sampleDocuments) {
        await integratedCacheSystem.storeLegalDocument(doc, {
          enableRanking: true,
          useGPUAcceleration: enableWebGPU,
          priority: 'medium'
        });
      }
      
      // Start metrics monitoring
      startMetricsMonitoring();
      
      // Start demo automation
      startDemoAutomation();
      
      console.log('✅ Demo initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize demo:', error);
    }
  });

  onDestroy(() => {
    if (metricsInterval) clearInterval(metricsInterval);
    if (demoInterval) clearInterval(demoInterval);
  });

  /**
   * Start real-time metrics monitoring
   */
  function startMetricsMonitoring(): void {
    metricsInterval = setInterval(() => {
      systemMetrics = integratedCacheSystem.getSystemMetrics();
    }, 1000) as any;
  }

  /**
   * Start demo automation (optional background activity)
   */
  function startDemoAutomization(): void {
    const demoQueries = [
      'liability coverage limits',
      'insurance claim procedures',
      'contract termination clauses',
      'coverage exclusions analysis',
      'legal precedent research'
    ];

    let queryIndex = 0;
    demoInterval = setInterval(async () => {
      if (!isSearching) {
        const query = demoQueries[queryIndex % demoQueries.length];
        await performBackgroundSearch(query);
        queryIndex++;
      }
    }, 15000) as any; // Every 15 seconds
  }

  /**
   * Perform enhanced search with full GPU acceleration
   */
  async function performSearch(): Promise<void> {
    if (!searchQuery.trim() || isSearching) return;

    isSearching = true;
    const startTime = performance.now();

    try {
      const searchContext: SearchContext = {
        query: searchQuery.trim(),
        documentTypes: ['contract', 'evidence', 'brief', 'citation'],
        userPreferences: {},
        sessionContext: {},
        similarityThreshold: similarityThreshold,
        maxResults: maxResults,
        useSemanticSearch: true,
        enableRanking: true
      };

      const result = await integratedCacheSystem.performEnhancedSearch(searchContext);
      
      if (result.success && result.found && result.results) {
        searchResults = result.results.map((r, index) => ({
          ...r,
          rank: index + 1,
          relevanceScore: r.score,
          cacheHit: result.protocol === 'webgpu',
          processingTime: result.processingTime
        }));
      } else {
        searchResults = [];
      }

      lastSearchTime = performance.now() - startTime;

      // Add to operation history
      operationHistory = [
        {
          timestamp: new Date().toISOString(),
          operation: 'search',
          query: searchQuery,
          results: searchResults.length,
          processingTime: lastSearchTime,
          cacheHit: result.protocol === 'webgpu',
          success: result.success
        },
        ...operationHistory
      ].slice(0, 20); // Keep last 20 operations

    } catch (error) {
      console.error('Search failed:', error);
      searchResults = [];
      
      operationHistory = [
        {
          timestamp: new Date().toISOString(),
          operation: 'search',
          query: searchQuery,
          results: 0,
          processingTime: performance.now() - startTime,
          cacheHit: false,
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        },
        ...operationHistory
      ].slice(0, 20);
      
    } finally {
      isSearching = false;
    }
  }

  /**
   * Background search for demo automation
   */
  async function performBackgroundSearch(query: string): Promise<void> {
    const searchContext: SearchContext = {
      query: query,
      documentTypes: ['contract', 'evidence', 'brief'],
      userPreferences: {},
      sessionContext: {},
      similarityThreshold: 0.6,
      maxResults: 5,
      useSemanticSearch: true,
      enableRanking: true
    };

    try {
      const result = await integratedCacheSystem.performEnhancedSearch(searchContext);
      
      operationHistory = [
        {
          timestamp: new Date().toISOString(),
          operation: 'background_search',
          query: query,
          results: result.results?.length || 0,
          processingTime: result.processingTime,
          cacheHit: result.protocol === 'webgpu',
          success: result.success,
          background: true
        },
        ...operationHistory
      ].slice(0, 20);

    } catch (error) {
      console.warn('Background search failed:', error);
    }
  }

  /**
   * Clear all cache data
   */
  async function clearCache(): Promise<void> {
    try {
      await integratedCacheSystem.destroy();
      searchResults = [];
      operationHistory = [];
      systemMetrics = {};
      console.log('🧹 Cache cleared successfully');
    } catch (error) {
      console.error('Failed to clear cache:', error);
    }
  }

  /**
   * Reload sample data
   */
  async function reloadSampleData(): Promise<void> {
    try {
      for (const doc of sampleDocuments) {
        await integratedCacheSystem.storeLegalDocument(doc, {
          enableRanking: true,
          useGPUAcceleration: enableWebGPU,
          priority: 'medium'
        });
      }
      console.log('📁 Sample data reloaded');
    } catch (error) {
      console.error('Failed to reload sample data:', error);
    }
  }

  /**
   * Handle demo configuration changes
   */
  function updateConfiguration(): void {
    console.log('⚙️ Configuration updated:', {
      enableWebGPU,
      enableWASM,
      enableNESMemory,
      compressionLevel
    });
    
    // Configuration changes would typically require reinitialization
    // For demo purposes, we'll just log the changes
  }

  // Reactive statements
  $: healthColor = systemMetrics.systemHealth === 'excellent' ? '#28a745' :
                  systemMetrics.systemHealth === 'good' ? '#ffc107' :
                  systemMetrics.systemHealth === 'fair' ? '#fd7e14' : '#dc3545';

  $: webgpuUtilizationColor = (systemMetrics.webgpuUtilization || 0) > 0.8 ? '#dc3545' :
                             (systemMetrics.webgpuUtilization || 0) > 0.6 ? '#ffc107' : '#28a745';
</script>

<div class="webgpu-demo-container">
  
  <!-- Demo Header -->
  <div class="demo-header">
    <h1 class="demo-title">🚀 WebGPU Ranking Cache Demo</h1>
    <p class="demo-subtitle">
      Client-side WebAssembly ranking cache with GPU acceleration, NES memory integration, and real-time performance monitoring
    </p>
    
    <div class="system-status">
      <div class="status-indicator" style="color: {healthColor}">
        System Health: {systemMetrics.systemHealth?.toUpperCase() || 'INITIALIZING'}
      </div>
      <div class="status-metric">
        WebGPU: {enableWebGPU ? '🟢 ENABLED' : '🔴 DISABLED'}
      </div>
      <div class="status-metric">
        WASM Workers: {enableWASM ? '🟢 ENABLED' : '🔴 DISABLED'}
      </div>
      <div class="status-metric">
        NES Memory: {enableNESMemory ? '🟢 ENABLED' : '🔴 DISABLED'}
      </div>
    </div>
  </div>

  <!-- Configuration Panel -->
  <div class="config-panel">
    <h3>Demo Configuration</h3>
    <div class="config-grid">
      
      <div class="config-group">
        <label class="config-label">
          <input type="checkbox" bind:checked={enableWebGPU} on:change={updateConfiguration} />
          WebGPU Acceleration
        </label>
      </div>

      <div class="config-group">
        <label class="config-label">
          <input type="checkbox" bind:checked={enableWASM} on:change={updateConfiguration} />
          WebAssembly Workers
        </label>
      </div>

      <div class="config-group">
        <label class="config-label">
          <input type="checkbox" bind:checked={enableNESMemory} on:change={updateConfiguration} />
          NES Memory Bridge
        </label>
      </div>

      <div class="config-group">
        <label class="config-label">Compression Level</label>
        <select bind:value={compressionLevel} on:change={updateConfiguration} class="config-select">
          <option value="none">None</option>
          <option value="low">Low</option>
          <option value="high">High</option>
        </select>
      </div>

      <div class="config-group">
        <label class="config-label">Max Results</label>
        <input type="range" min="5" max="50" bind:value={maxResults} class="config-range" />
        <span class="config-value">{maxResults}</span>
      </div>

      <div class="config-group">
        <label class="config-label">Similarity Threshold</label>
        <input type="range" min="0.1" max="1.0" step="0.1" bind:value={similarityThreshold} class="config-range" />
        <span class="config-value">{similarityThreshold.toFixed(1)}</span>
      </div>
    </div>
  </div>

  <!-- Search Interface -->
  <div class="search-section">
    <h3>Enhanced Legal Document Search</h3>
    <div class="search-interface">
      <div class="search-input-container">
        <input 
          type="text" 
          bind:value={searchQuery}
          placeholder="Enter legal search query (e.g., 'contract liability insurance coverage')"
          class="search-input"
          disabled={isSearching}
          on:keydown={(e) => e.key === 'Enter' && performSearch()}
        />
        <button 
          class="search-button"
          disabled={isSearching || !searchQuery.trim()}
          on:click={performSearch}
        >
          {isSearching ? '🔄 SEARCHING...' : '🔍 SEARCH'}
        </button>
      </div>
      
      {#if lastSearchTime > 0}
        <div class="search-stats">
          Last search: {lastSearchTime.toFixed(2)}ms | Results: {searchResults.length}
        </div>
      {/if}
    </div>

    <!-- Search Results -->
    {#if searchResults.length > 0}
      <div class="results-section">
        <h4>Search Results ({searchResults.length})</h4>
        <div class="results-grid">
          {#each searchResults as result (result.docId)}
            <div class="result-card rank-{result.rank <= 3 ? result.rank : 'other'}">
              <div class="result-header">
                <div class="result-rank">#{result.rank}</div>
                <div class="result-score">{(result.relevanceScore * 100).toFixed(1)}%</div>
                <div class="result-cache-status">
                  {result.cacheHit ? '🔥 CACHE HIT' : '⚡ COMPUTED'}
                </div>
              </div>
              
              <div class="result-content">
                <div class="result-summary">{result.summary}</div>
                {#if result.url}
                  <div class="result-url">
                    <a href={result.url} target="_blank" rel="noopener">{result.url}</a>
                  </div>
                {/if}
              </div>
              
              <div class="result-footer">
                <div class="result-flags">Flags: 0x{result.flags.toString(16).padStart(2, '0')}</div>
                <div class="result-timing">{result.processingTime?.toFixed(2)}ms</div>
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <!-- Performance Dashboard -->
  <div class="dashboard-section">
    <CachePerformanceMonitor 
      updateInterval={1000}
      showRealTimeCharts={true}
      showHistoricalData={true}
      enablePerformanceTuning={true}
      darkTheme={true}
      showAdvancedMetrics={true}
    />
  </div>

  <!-- Operation History -->
  <div class="history-section">
    <h3>Recent Operations</h3>
    <div class="history-container">
      {#each operationHistory as operation (operation.timestamp)}
        <div class="operation-item" class:background={operation.background} class:success={operation.success} class:error={operation.error}>
          <div class="operation-time">{new Date(operation.timestamp).toLocaleTimeString()}</div>
          <div class="operation-type">{operation.operation.toUpperCase()}</div>
          <div class="operation-query">{operation.query}</div>
          <div class="operation-stats">
            {operation.results} results | {operation.processingTime.toFixed(1)}ms
            {operation.cacheHit ? '| 🔥 CACHE HIT' : ''}
          </div>
          {#if operation.error}
            <div class="operation-error">Error: {operation.error}</div>
          {/if}
        </div>
      {/each}
    </div>
  </div>

  <!-- Demo Actions -->
  <div class="demo-actions">
    <button class="action-button clear" on:click={clearCache}>
      🧹 Clear Cache
    </button>
    <button class="action-button reload" on:click={reloadSampleData}>
      📁 Reload Sample Data
    </button>
    <button class="action-button optimize" on:click={() => console.log('Manual optimization would be triggered')}>
      ⚙️ Optimize System
    </button>
  </div>
</div>

<style>
  .webgpu-demo-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    font-family: 'Inter', sans-serif;
    background: #f8f9fa;
    min-height: 100vh;
  }

  /* Demo Header */
  .demo-header {
    text-align: center
    margin-bottom: 30px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white
    border-radius: 12px;
  }

  .demo-title {
    margin: 0 0 10px 0;
    font-size: 2.5rem;
    font-weight: 700;
  }

  .demo-subtitle {
    margin: 0 0 20px 0;
    font-size: 1.1rem;
    opacity: 0.9;
  }

  .system-status {
    display: flex
    justify-content: center
    gap: 20px;
    flex-wrap: wrap
  }

  .status-indicator {
    font-weight: 700;
    font-size: 1.1rem;
  }

  .status-metric {
    font-size: 0.9rem;
    opacity: 0.9;
  }

  /* Configuration Panel */
  .config-panel {
    background: white
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }

  .config-panel h3 {
    margin: 0 0 15px 0;
    color: #333;
    font-weight: 600;
  }

  .config-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
  }

  .config-group {
    display: flex
    flex-direction: column
    gap: 5px;
  }

  .config-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: #555;
    cursor: pointer
  }

  .config-label input[type="checkbox"] {
    margin-right: 8px;
  }

  .config-select {
    padding: 6px 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 0.9rem;
  }

  .config-range {
    width: 100%;
  }

  .config-value {
    font-weight: 600;
    color: #007bff;
  }

  /* Search Section */
  .search-section {
    background: white
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }

  .search-section h3 {
    margin: 0 0 15px 0;
    color: #333;
    font-weight: 600;
  }

  .search-input-container {
    display: flex
    gap: 10px;
    margin-bottom: 10px;
  }

  .search-input {
    flex: 1;
    padding: 12px 16px;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.2s ease;
  }

  .search-input:focus {
    outline: none
    border-color: #007bff;
  }

  .search-input:disabled {
    background: #f8f9fa;
    opacity: 0.7;
  }

  .search-button {
    padding: 12px 24px;
    background: #007bff;
    color: white
    border: none
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer
    transition: all 0.2s ease;
  }

  .search-button:hover:not(:disabled) {
    background: #0056b3;
    transform: translateY(-1px);
  }

  .search-button:disabled {
    background: #6c757d;
    cursor: not-allowed;
  }

  .search-stats {
    font-size: 0.9rem;
    color: #666;
    text-align: center
  }

  /* Results Section */
  .results-section {
    margin-top: 20px;
  }

  .results-section h4 {
    margin: 0 0 15px 0;
    color: #333;
    font-weight: 600;
  }

  .results-grid {
    display: grid
    gap: 15px;
  }

  .result-card {
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 15px;
    background: white
    transition: all 0.2s ease;
  }

  .result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .result-card.rank-1 {
    border-left: 4px solid #28a745;
  }

  .result-card.rank-2 {
    border-left: 4px solid #ffc107;
  }

  .result-card.rank-3 {
    border-left: 4px solid #fd7e14;
  }

  .result-header {
    display: flex
    justify-content: space-between;
    align-items: center
    margin-bottom: 10px;
  }

  .result-rank {
    font-weight: 700;
    color: #007bff;
    font-size: 1.1rem;
  }

  .result-score {
    font-weight: 600;
    color: #28a745;
  }

  .result-cache-status {
    font-size: 0.8rem;
    font-weight: 600;
    color: #666;
  }

  .result-content {
    margin-bottom: 10px;
  }

  .result-summary {
    margin-bottom: 5px;
    color: #333;
    line-height: 1.4;
  }

  .result-url {
    font-size: 0.8rem;
  }

  .result-url a {
    color: #007bff;
    text-decoration: none
  }

  .result-url a:hover {
    text-decoration: underline
  }

  .result-footer {
    display: flex
    justify-content: space-between;
    align-items: center
    font-size: 0.8rem;
    color: #666;
    padding-top: 10px;
    border-top: 1px solid #f8f9fa;
  }

  /* Dashboard Section */
  .dashboard-section {
    margin-bottom: 20px;
  }

  /* History Section */
  .history-section {
    background: white
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }

  .history-section h3 {
    margin: 0 0 15px 0;
    color: #333;
    font-weight: 600;
  }

  .history-container {
    max-height: 300px;
    overflow-y: auto
    display: flex
    flex-direction: column
    gap: 8px;
  }

  .operation-item {
    display: grid
    grid-template-columns: auto auto 1fr auto;
    gap: 10px;
    align-items: center
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9rem;
    border-left: 3px solid #dee2e6;
  }

  .operation-item.success {
    border-left-color: #28a745;
    background: rgba(40, 167, 69, 0.05);
  }

  .operation-item.error {
    border-left-color: #dc3545;
    background: rgba(220, 53, 69, 0.05);
  }

  .operation-item.background {
    opacity: 0.7;
    font-style: italic
  }

  .operation-time {
    font-family: 'Courier New', monospace;
    color: #666;
  }

  .operation-type {
    font-weight: 600;
    color: #007bff;
  }

  .operation-query {
    color: #333;
    overflow: hidden
    text-overflow: ellipsis
    white-space: nowrap
  }

  .operation-stats {
    color: #666;
    text-align: right
  }

  .operation-error {
    grid-column: 1 / -1;
    color: #dc3545;
    font-size: 0.8rem;
    margin-top: 4px;
  }

  /* Demo Actions */
  .demo-actions {
    display: flex
    gap: 12px;
    justify-content: center
    flex-wrap: wrap
  }

  .action-button {
    padding: 10px 20px;
    border: none
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer
    transition: all 0.2s ease;
  }

  .action-button.clear {
    background: #dc3545;
    color: white
  }

  .action-button.reload {
    background: #28a745;
    color: white
  }

  .action-button.optimize {
    background: #ffc107;
    color: #212529;
  }

  .action-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .system-status {
      flex-direction: column
      gap: 10px;
    }

    .config-grid {
      grid-template-columns: 1fr;
    }

    .search-input-container {
      flex-direction: column
    }

    .operation-item {
      grid-template-columns: 1fr;
      gap: 5px;
    }

    .operation-stats {
      text-align: left
    }

    .demo-actions {
      flex-direction: column
    }
  }

  /* Scrollbar Styling */
  .history-container::-webkit-scrollbar {
    width: 6px;
  }

  .history-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }

  .history-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }

  .history-container::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
</style>