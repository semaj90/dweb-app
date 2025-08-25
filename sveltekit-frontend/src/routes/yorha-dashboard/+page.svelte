//-- YoRHa Legal AI Dashboard - Complete Integration
<script lang="ts">
  import { onMount } from 'svelte';
  import YoRHaDataGrid from '$lib/components/yorha/YoRHaDataGrid.svelte';
  import YoRHaForm from '$lib/components/yorha/YoRHaForm.svelte';
  import YoRHaTerminal from '$lib/components/yorha/YoRHaTerminal.svelte';
  import YoRHaModal from '$lib/components/yorha/YoRHaModal.svelte';
  import YoRHaNotification from '$lib/components/yorha/YoRHaNotification.svelte';
  import { documentsColumns, casesColumns, evidenceColumns, documentFormFields, caseFormFields, evidenceFormFields, debounce, withAbort } from '$lib/yorha/constants';

  // Domain interfaces
  interface RagAnalysis {
    confidenceScore: number;
    legalComplexity: string;
    riskLevel: string;
  }

  interface RagResult {
    title?: string;
    yorha_type?: string;
    yorha_confidence?: number;
    content?: string;
    summary?: string;
    yorha_analysis?: {
      relevanceScore?: number;
      legalWeight?: number;
      riskFactor?: number;
      actionRequired?: string;
    };
  }

  interface RagRecommendation {
    title: string;
    priority: string;
    description: string;
    actionItems: string[];
    estimatedTime?: string;
    yorha_confidence?: number;
  }

  interface Pagination {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  }

  interface NotificationItem {
    id: number;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    timestamp: Date;
  }

  // State management
  let activeTab = 'documents';
  let isLoading = false;
  let selectedData: any[] = [];
  let searchQuery = '';
  let modalOpen = false;
  let modalType: 'create' | 'edit' = 'create';
  let modalData: any = null;
  let notifications: NotificationItem[] = [];
  let terminalActive = false;

  // Data stores
  let documentsData: any[] = [];
  let casesData: any[] = [];
  let evidenceData: any[] = [];
  let pagination: Pagination = {
    page: 1,
    limit: 25,
    total: 0,
    totalPages: 0
  };

  // Enhanced RAG state
  let ragResults: RagResult[] = [];
  let ragAnalysis: RagAnalysis | null = null;
  let ragRecommendations: RagRecommendation[] = [];

  // Reactive selection based on activeTab
    $: currentData = activeTab === 'documents'
      ? documentsData
      : activeTab === 'cases'
        ? casesData
        : evidenceData;

    $: currentColumns = activeTab === 'documents'
      ? documentsColumns
      : activeTab === 'cases'
        ? casesColumns
        : evidenceColumns;

    $: currentFormFields = activeTab === 'documents'
      ? documentFormFields
      : activeTab === 'cases'
        ? caseFormFields
        : evidenceFormFields;

  // Grid configurations (centralized constants)
  import { documentsColumns, casesColumns, evidenceColumns, documentFormFields, caseFormFields, evidenceFormFields, debounce, withAbort } from '$lib/yorha/constants';

  const triggerSearch = debounce(() => loadData(), 400);

  // Lifecycle
  onMount(() => {
    loadData();
  });

  // Data loading functions
  async function loadData() {
    isLoading = true;
    try {
      await Promise.all([
        loadDocuments(),
        loadCases(),
        loadEvidence()
      ]);
      addNotification('success', 'Data loaded successfully');
    } catch (error) {
      console.error('Error loading data:', error);
      addNotification('error', 'Failed to load data');
    } finally {
      isLoading = false;
    }
  }

  async function loadDocuments() {
    const { promise } = withAbort(async (signal) => {
      const url = `/api/yorha/legal-data?type=documents&page=${pagination.page}&limit=${pagination.limit}&search=${encodeURIComponent(searchQuery)}`;
      const response = await fetch(url, { signal, headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
    try { const result = await promise; if (result.success) { documentsData = result.data; pagination = result.pagination; } } catch (e) { if ((e as any).name !== 'AbortError') console.error('Documents load failed', e); }
  }

  async function loadCases() {
    const { promise } = withAbort(async (signal) => {
      const url = `/api/yorha/legal-data?type=cases&page=${pagination.page}&limit=${pagination.limit}&search=${encodeURIComponent(searchQuery)}`;
      const response = await fetch(url, { signal, headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
    try { const result = await promise; if (result.success) { casesData = result.data; } } catch (e) { if ((e as any).name !== 'AbortError') console.error('Cases load failed', e); }
  }

  async function loadEvidence() {
    const { promise } = withAbort(async (signal) => {
      const url = `/api/yorha/legal-data?type=evidence&page=${pagination.page}&limit=${pagination.limit}&search=${encodeURIComponent(searchQuery)}`;
      const response = await fetch(url, { signal, headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
    try { const result = await promise; if (result.success) { evidenceData = result.data; } } catch (e) { if ((e as any).name !== 'AbortError') console.error('Evidence load failed', e); }
  }

  // Enhanced RAG functions
  async function performEnhancedAnalysis(query: string) {
    if (!query.trim()) return;

    isLoading = true;
    try {
      const response = await fetch('/api/yorha/enhanced-rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          dataType: activeTab,
          analysisType: 'comprehensive',
          includeRecommendations: true,
          includeMetadata: true
        })
      });

      const result = await response.json();

      if (result.success) {
        ragResults = result.results;
        ragAnalysis = result.analysis;
        ragRecommendations = result.recommendations;
        addNotification('success', `Enhanced analysis completed: ${result.results.length} results found`);
      } else {
        addNotification('error', result.error || 'Enhanced analysis failed');
      }
    } catch (error) {
      console.error('Enhanced RAG error:', error);
      addNotification('error', 'Enhanced analysis failed');
    } finally {
      isLoading = false;
    }
  }

  // CRUD operations
  async function createItem(dataType: string, data: any) {
    try {
      const response = await fetch('/api/yorha/legal-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataType, data })
      });

      const result = await response.json();

      if (result.success) {
        await loadData();
        addNotification('success', `${dataType} created successfully`);
        closeModal();
      } else {
        addNotification('error', result.error || 'Failed to create item');
      }
    } catch (error) {
      console.error('Create error:', error);
      addNotification('error', 'Failed to create item');
    }
  }

  async function updateItem(dataType: string, id: string, data: any) {
    try {
      const response = await fetch('/api/yorha/legal-data', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataType, id, data })
      });

      const result = await response.json();

      if (result.success) {
        await loadData();
        addNotification('success', `${dataType} updated successfully`);
        closeModal();
      } else {
        addNotification('error', result.error || 'Failed to update item');
      }
    } catch (error) {
      console.error('Update error:', error);
      addNotification('error', 'Failed to update item');
    }
  }

  async function deleteItem(dataType: string, id: string) {
    if (!confirm('Are you sure you want to delete this item?')) return;

    try {
      const response = await fetch('/api/yorha/legal-data', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataType, id })
      });

      const result = await response.json();

      if (result.success) {
        await loadData();
        addNotification('success', `${dataType} deleted successfully`);
      } else {
        addNotification('error', result.error || 'Failed to delete item');
      }
    } catch (error) {
      console.error('Delete error:', error);
      addNotification('error', 'Failed to delete item');
    }
  }

  // Modal management
  function openCreateModal() {
    modalType = 'create';
    modalData = null;
    modalOpen = true;
  }

  function openEditModal(item: any) {
    modalType = 'edit';
    modalData = item;
    modalOpen = true;
  }

  function closeModal() {
    modalOpen = false;
    modalData = null;
  }

  // Notification management
  function addNotification(type: 'success' | 'error' | 'warning' | 'info', message: string) {
    const notification = {
      id: Date.now(),
      type,
      message,
      timestamp: new Date()
    };
    notifications = [...notifications, notification];

    // Auto-remove after 5 seconds
    setTimeout(() => {
      notifications = notifications.filter(n => n.id !== notification.id);
    }, 5000);
  }

  // Terminal command handler
  function handleTerminalCommand(command: string) {
    const parts = command.toLowerCase().split(' ');
    const cmd = parts[0];

    switch (cmd) {
      case 'analyze':
        if (parts[1]) {
          performEnhancedAnalysis(parts.slice(1).join(' '));
        }
        break;
      case 'search':
        if (parts[1]) {
          searchQuery = parts.slice(1).join(' ');
          loadData();
        }
        break;
      case 'tab':
        if (parts[1] && ['documents', 'cases', 'evidence'].includes(parts[1])) {
          activeTab = parts[1];
        }
        break;
      case 'create':
        openCreateModal();
        break;
      case 'clear':
        ragResults = [];
        ragAnalysis = null;
        ragRecommendations = [];
        break;
    }
  }

</script>

<svelte:head>
  <title>YoRHa Legal AI Dashboard</title>
</svelte:head>

<div class="yorha-dashboard">
  <!-- Header -->
  <header class="dashboard-header">
    <div class="header-content">
      <h1 class="dashboard-title">YoRHa Legal AI Command Center</h1>
      <div class="header-stats">
        <div class="stat">
          <span class="stat-label">Documents</span>
          <span class="stat-value">{documentsData.length}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Cases</span>
          <span class="stat-value">{casesData.length}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Evidence</span>
          <span class="stat-value">{evidenceData.length}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Status</span>
          <span class="stat-value status-{isLoading ? 'loading' : 'ready'}">
            {isLoading ? 'PROCESSING' : 'READY'}
          </span>
        </div>
      </div>
    </div>

    <!-- Controls -->
    <div class="header-controls">
      <div class="search-container">
  <input type="search" class="search-input" placeholder="Search legal data..." aria-label="Search legal data" bind:value={searchQuery} oninput={() => triggerSearch()} onkeydown={(e) => e.key === 'Enter' && loadData()} />
        <button class="search-btn" onclick={() => loadData()}>SEARCH</button>
      </div>

      <button class="analyze-btn" onclick={() => performEnhancedAnalysis(searchQuery)}>
        ENHANCED ANALYSIS
      </button>

      <button class="terminal-btn" onclick={() => terminalActive = !terminalActive}>
        TERMINAL
      </button>
    </div>
  </header>

  <!-- Tab Navigation -->
  <nav class="tab-navigation">
    <button
      class="tab-btn {activeTab === 'documents' ? 'active' : ''}"
      onclick={() => activeTab = 'documents'}
    >
      DOCUMENTS
    </button>
    <button
      class="tab-btn {activeTab === 'cases' ? 'active' : ''}"
      onclick={() => activeTab = 'cases'}
    >
      CASES
    </button>
    <button
      class="tab-btn {activeTab === 'evidence' ? 'active' : ''}"
      onclick={() => activeTab = 'evidence'}
    >
      EVIDENCE
    </button>
  </nav>

  <!-- Main Content -->
  <main class="dashboard-main">
    <div class="content-wrapper">
      <!-- Data Grid -->
      <section class="data-section">
        <div class="section-header">
          <h2 class="section-title">{activeTab.toUpperCase()} MANAGEMENT</h2>
          <button class="create-btn" onclick={openCreateModal}>
            CREATE NEW {activeTab.toUpperCase().slice(0, -1)}
          </button>
        </div>

        <YoRHaDataGrid
          columns={currentColumns}
          data={currentData}
          loading={isLoading}
          selectable={true}
          multiSelect={true}
          sortable={true}
          filterable={true}
          resizable={true}
          maxHeight={500}
          glitchEffect={false}
        />
      </section>

      <!-- Enhanced RAG Results -->
      {#if ragResults.length > 0}
        <section class="rag-section">
          <div class="section-header">
            <h2 class="section-title">ENHANCED RAG ANALYSIS RESULTS</h2>
            <div class="rag-stats">
              {#if ragAnalysis}
                <span class="rag-stat">Confidence: {(ragAnalysis.confidenceScore * 100).toFixed(1)}%</span>
                <span class="rag-stat">Complexity: {ragAnalysis.legalComplexity}</span>
                <span class="rag-stat">Risk: {ragAnalysis.riskLevel}</span>
              {/if}
            </div>
          </div>

          <div class="rag-results">
            {#each ragResults as result}
              <div class="rag-result">
                <div class="result-header">
                  <h3 class="result-title">{result.title || 'Analysis Result'}</h3>
                  <div class="result-meta">
                    <span class="result-type">{result.yorha_type}</span>
                    <span class="result-confidence">
                      {(result.yorha_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div class="result-content">
                  {result.content || result.summary || 'No content available'}
                </div>
                {#if result.yorha_analysis}
                  <div class="result-analysis">
                    <span class="analysis-item">Relevance: {(result.yorha_analysis.relevanceScore * 100).toFixed(1)}%</span>
                    <span class="analysis-item">Legal Weight: {(result.yorha_analysis.legalWeight * 100).toFixed(1)}%</span>
                    <span class="analysis-item">Risk: {(result.yorha_analysis.riskFactor * 100).toFixed(1)}%</span>
                    <span class="analysis-item">Action: {result.yorha_analysis.actionRequired}</span>
                  </div>
                {/if}
              </div>
            {/each}
          </div>

          <!-- Recommendations -->
        {#if ragRecommendations.length > 0}
          <div class="recommendations">
            <h3 class="recommendations-title">RECOMMENDATIONS</h3>
            {#each ragRecommendations as rec}
              <div class="recommendation">
                <div class="rec-header">
                  <h4 class="rec-title">{rec.title}</h4>
                  <span class={"rec-priority " + (rec.priority ? 'priority-' + rec.priority.toLowerCase() : '')}>
                    {rec.priority}
                  </span>
                </div>
                <p class="rec-description">{rec.description}</p>
                <div class="rec-actions">
                  {#each rec.actionItems as action}
                    <span class="rec-action">{action}</span>
                  {/each}
                </div>
                <div class="rec-meta">
                  {#if rec.estimatedTime}<span class="rec-time">Est. Time: {rec.estimatedTime}</span>{/if}
                  {#if rec.yorha_confidence !== undefined}<span class="rec-confidence">Confidence: {(rec.yorha_confidence * 100).toFixed(1)}%</span>{/if}
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </section>
    {/if}
    </div>

    <!-- Terminal Panel -->
    {#if terminalActive}
      <aside class="terminal-panel">
        <YoRHaTerminal
          title="YoRHa Legal AI Terminal"
          isActive={terminalActive}
          oncommand={handleTerminalCommand}
        />
      </aside>
    {/if}
  </main>
  {#if modalOpen}
    <YoRHaModal
      open={modalOpen}
      title={(modalType === 'create' ? 'Create' : 'Edit') + ' ' + activeTab.toUpperCase().slice(0, -1)}
    >
      <YoRHaForm
        title={(modalType === 'create' ? 'Create New' : 'Edit') + ' ' + activeTab.toUpperCase().slice(0, -1)}
        fields={currentFormFields}
        submitLabel={modalType === 'create' ? 'Create' : 'Update'}
        onsubmit={(data) => {
          if (modalType === 'create') {
            createItem(activeTab, data);
          } else {
            updateItem(activeTab, modalData?.id, data);
          }
        }}
        oncancel={closeModal}
      />
    </YoRHaModal>
  {/if}
    </YoRHaModal>
  {/if}

  <!-- Notifications -->
  <div class="notifications" role="region" aria-live="polite" aria-label="Notifications">
    {#each notifications as notification}
      <YoRHaNotification
        type={notification.type}
        message={notification.message}
        onclose={() => notifications = notifications.filter(n => n.id !== notification.id)}
      />
    {/each}
  </div>
</div>

<style>
  .yorha-dashboard {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
  }

  .dashboard-header {
    background: linear-gradient(45deg, #ffbf00, #ffd700);
    color: #000;
    padding: 16px 24px;
    border-bottom: 3px solid #ffbf00;
  }

  .header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .dashboard-title {
    font-size: 24px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0;
  }

  .header-stats {
    display: flex;
    gap: 24px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.8;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 700;
  }

  .status-loading {
    color: #ff6b35;
  }

  .status-ready {
    color: #00ff41;
  }

  .header-controls {
    display: flex;
    gap: 16px;
    align-items: center;
  }

  .search-container {
    display: flex;
    gap: 8px;
  }

  .search-input {
    background: #000;
    border: 2px solid #ffbf00;
    color: #e0e0e0;
    padding: 8px 12px;
    font-family: inherit;
    font-size: 12px;
    width: 300px;
  }

  .search-input:focus {
    outline: none;
    border-color: #ffd700;
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
  }

  .search-btn,
  .analyze-btn,
  .terminal-btn {
    background: #000;
    border: 2px solid #ffbf00;
    color: #ffbf00;
    padding: 8px 16px;
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .search-btn:hover,
  .analyze-btn:hover,
  .terminal-btn:hover {
    background: #ffbf00;
    color: #000;
    transform: translateY(-1px);
  }

  .tab-navigation {
    background: #1a1a1a;
    border-bottom: 2px solid #ffbf00;
    display: flex;
    padding: 0 24px;
  }

  .tab-btn {
    background: transparent;
    border: none;
    color: #808080;
    padding: 16px 24px;
    font-family: inherit;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.2s ease;
  }

  .tab-btn:hover {
    color: #ffbf00;
    border-bottom-color: #ffbf00;
  }

  .tab-btn.active {
    color: #ffbf00;
    border-bottom-color: #ffbf00;
    background: rgba(255, 191, 0, 0.1);
  }

  .dashboard-main {
    display: flex;
    min-height: calc(100vh - 120px);
  }

  .content-wrapper {
    flex: 1;
    padding: 24px;
  }

  .terminal-panel {
    width: 400px;
    border-left: 2px solid #ffbf00;
    background: #0a0a0a;
  }

  .data-section,
  .rag-section {
    margin-bottom: 32px;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #333;
  }

  .section-title {
    font-size: 18px;
    font-weight: 700;
    color: #ffbf00;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0;
  }

  .create-btn {
    background: #00ff41;
    border: 2px solid #00ff41;
    color: #000;
    padding: 8px 16px;
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .create-btn:hover {
    background: transparent;
    color: #00ff41;
    transform: translateY(-1px);
  }

  .action-buttons {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    background: #1a1a1a;
    border: 1px solid #ffbf00;
    color: #ffbf00;
    padding: 4px 8px;
    font-family: inherit;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    background: #ffbf00;
    color: #000;
  }

  .action-btn.delete {
    border-color: #ff0041;
    color: #ff0041;
  }

  .action-btn.delete:hover {
    background: #ff0041;
    color: #fff;
  }

  .action-btn.analyze {
    border-color: #00ff41;
    color: #00ff41;
  }

  .action-btn.analyze:hover {
    background: #00ff41;
    color: #000;
  }

  .rag-stats {
    display: flex;
    gap: 16px;
  }

  .rag-stat {
    font-size: 12px;
    color: #ffbf00;
    text-transform: uppercase;
  }

  .rag-results {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 24px;
  }

  .rag-result {
    background: #1a1a1a;
    border: 1px solid #333;
    padding: 16px;
    border-left: 4px solid #ffbf00;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .result-title {
    font-size: 16px;
    font-weight: 600;
    color: #ffbf00;
    margin: 0;
  }

  .result-meta {
    display: flex;
    gap: 12px;
  }

  .result-type,
  .result-confidence {
    font-size: 12px;
    color: #808080;
    text-transform: uppercase;
  }

  .result-content {
    color: #e0e0e0;
    line-height: 1.5;
    margin-bottom: 12px;
  }

  .result-analysis {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }

  .analysis-item {
    font-size: 11px;
    color: #00ff41;
    text-transform: uppercase;
  }

  .recommendations {
    background: #0f0f0f;
    border: 1px solid #ffbf00;
    padding: 16px;
  }

  .recommendations-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffbf00;
    text-transform: uppercase;
    margin: 0 0 16px 0;
  }

  .recommendation {
    background: #1a1a1a;
    border: 1px solid #333;
    padding: 12px;
    margin-bottom: 12px;
    border-left: 4px solid #00ff41;
  }

  .rec-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .rec-title {
    font-size: 14px;
    font-weight: 600;
    color: #00ff41;
    margin: 0;
  }

  .rec-priority {
    font-size: 10px;
    padding: 2px 6px;
    text-transform: uppercase;
    font-weight: 600;
  }

  .priority-high {
    background: #ff0041;
    color: #fff;
  }

  .priority-medium {
    background: #ffaa00;
    color: #000;
  }

  .priority-low {
    background: #00ff41;
    color: #000;
  }

  .rec-description {
    color: #e0e0e0;
    margin: 0 0 12px 0;
    line-height: 1.4;
  }

  .rec-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 8px;
  }

  .rec-action {
    font-size: 11px;
    background: #333;
    color: #ffbf00;
    padding: 2px 6px;
    text-transform: uppercase;
  }

  .rec-meta {
    display: flex;
    gap: 16px;
  }

  .rec-time,
  .rec-confidence {
    font-size: 11px;
    color: #808080;
    text-transform: uppercase;
  }

  .notifications {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  @media (max-width: 1200px) {
    .dashboard-main {
      flex-direction: column;
    }

    .terminal-panel {
      width: 100%;
      border-left: none;
      border-top: 2px solid #ffbf00;
    }
  }

  @media (max-width: 768px) {
    .header-content {
      flex-direction: column;
      gap: 16px;
      align-items: flex-start;
    }

    .header-controls {
      flex-direction: column;
      width: 100%;
    }

    .search-input {
      width: 100%;
    }

    .tab-navigation {
      overflow-x: auto;
    }

    .dashboard-main {
      padding: 16px;
    }
  }
</style>
