<script lang="ts">
  import { onMount } from 'svelte';
  import CanvasBoard from '$lib/components/yorha/CanvasBoard.svelte';
  import YoRHaTable from '$lib/components/yorha/YoRHaTable.svelte';
  import YoRHaDataGrid from '$lib/components/yorha/YoRHaDataGrid.svelte';
  import {
    yorhaTablePresets,
    yorhaGridPresets,
    tableManager,
    statusFormatters
  } from '$lib/components/yorha/index.ts';

  // Sample data for legal cases
  const sampleCases = [
    {
      id: 1,
      case_number: 'CASE-2024-001',
      title: 'Corporate Fraud Investigation',
      status: 'active',
      priority: 'high',
      assigned_to: 'Det. Anderson',
      created_at: '2024-08-01',
      description: 'Investigation into suspected financial misconduct at TechCorp Industries.'
    },
    {
      id: 2,
      case_number: 'CASE-2024-002',
      title: 'Cybersecurity Breach Analysis',
      status: 'pending',
      priority: 'critical',
      assigned_to: 'Analyst Johnson',
      created_at: '2024-08-05',
      description: 'Analysis of data breach affecting 50,000 customer records.'
    },
    {
      id: 3,
      case_number: 'CASE-2024-003',
      title: 'Employment Discrimination Case',
      status: 'closed',
      priority: 'medium',
      assigned_to: 'Prosecutor Smith',
      created_at: '2024-07-15',
      description: 'Workplace discrimination allegations against MegaCorp Ltd.'
    },
    {
      id: 4,
      case_number: 'CASE-2024-004',
      title: 'Intellectual Property Theft',
      status: 'active',
      priority: 'high',
      assigned_to: 'Det. Williams',
      created_at: '2024-08-08',
      description: 'Investigation into stolen trade secrets and patent violations.'
    },
    {
      id: 5,
      case_number: 'CASE-2024-005',
      title: 'Tax Evasion Investigation',
      status: 'inactive',
      priority: 'low',
      assigned_to: 'Analyst Davis',
      created_at: '2024-07-20',
      description: 'Review of tax compliance issues for multiple entities.'
    }
  ];

  // Sample evidence data
  const sampleEvidence = [
    {
      id: 1,
      evidence_id: 'EVD-2024-001',
      filename: 'financial_records.pdf',
      type: 'document',
      ai_processed: true,
      confidence_score: 0.92,
      relevance_score: 0.87,
      tags: 'financial, fraud, analysis',
      status: 'processed',
      case_id: 'CASE-2024-001',
      size: 2048576
    },
    {
      id: 2,
      evidence_id: 'EVD-2024-002',
      filename: 'security_logs.json',
      type: 'digital',
      ai_processed: true,
      confidence_score: 0.88,
      relevance_score: 0.94,
      tags: 'cybersecurity, breach, logs',
      status: 'processing',
      case_id: 'CASE-2024-002',
      size: 5242880
    },
    {
      id: 3,
      evidence_id: 'EVD-2024-003',
      filename: 'witness_statement.docx',
      type: 'document',
      ai_processed: false,
      confidence_score: 0.0,
      relevance_score: 0.0,
      tags: 'witness, statement, testimony',
      status: 'pending',
      case_id: 'CASE-2024-003',
      size: 1024000
    }
  ];

  // Component state
  let tableLoading = $state(false);
  let gridLoading = $state(false);
  let selectedDemo = $state('table');
  let glitchEffect = $state(false);
  let showEvidenceBoard = $state(false);

  // Get table presets
  const casesTableConfig = yorhaTablePresets.legalCases;
  const evidenceGridConfig = yorhaGridPresets.evidenceAnalysis;

  // Initialize table managers
  onMount(() => {
    tableManager.createTable('demo-cases', {
      pageSize: 10,
      sortColumn: 'created_at',
      sortDirection: 'desc'
    });

    tableManager.createTable('demo-evidence', {
      pageSize: 20,
      sortColumn: 'evidence_id',
      sortDirection: 'asc'
    });
  });

  function toggleLoading(type: 'table' | 'grid') {
    if (type === 'table') {
      tableLoading = true;
      setTimeout(() => tableLoading = false, 2000);
    } else {
      gridLoading = true;
      setTimeout(() => gridLoading = false, 2000);
    }
  }

  function triggerGlitch() {
    glitchEffect = true;
    setTimeout(() => glitchEffect = false, 2000);
  }

  // Mock AI analysis function
  function mockAIAnalysis(row: any) {
    tableManager.addNotification({
      type: 'info',
      title: 'AI Analysis',
      message: `Starting AI analysis for ${row.case_number || row.evidence_id}...`,
      duration: 3000
    });

    setTimeout(() => {
      tableManager.addNotification({
        type: 'success',
        title: 'Analysis Complete',
        message: `AI analysis completed with 94% confidence`,
        duration: 5000
      });
    }, 3000);
  }
</script>

<svelte:head>
  <title>YoRHa Dashboard - Legal AI System</title>
</svelte:head>

<div class="yorha-demo-container">
  <div class="yorha-demo-header">
    <h1 class="yorha-demo-title">YoRHa LEGAL AI SYSTEM</h1>
    <div class="yorha-demo-subtitle">Advanced Case Management & Evidence Analysis</div>

    <div class="yorha-demo-controls">
      <button
        class="yorha-control-btn"
        class:active={selectedDemo === 'table'}
        onclick={() => selectedDemo = 'table'}
      >
        CASES TABLE
      </button>
      <button
        class="yorha-control-btn"
        class:active={selectedDemo === 'grid'}
        onclick={() => selectedDemo = 'grid'}
      >
        EVIDENCE GRID
      </button>
      <button
        class="yorha-control-btn yorha-action"
        onclick={() => toggleLoading(selectedDemo as 'table' | 'grid')}
      >
        TEST LOADING
      </button>
      <button
        class="yorha-control-btn yorha-evidence-action"
        onclick={() => showEvidenceBoard = true}
      >
        EVIDENCE BOARD
      </button>
      <button
        class="yorha-control-btn yorha-warning"
        onclick={triggerGlitch}
      >
        GLITCH EFFECT
      </button>
    </div>
  </div>

  {#if selectedDemo === 'table'}
    <div class="yorha-demo-section">
      <h2 class="yorha-section-title">ACTIVE LEGAL CASES</h2>
      <div class="yorha-section-description">
        Case management system with AI-powered analysis and tracking
      </div>

      <YoRHaTable
        columns={casesTableConfig.columns}
        data={sampleCases}
        loading={tableLoading}
        selectable={true}
        sortable={true}
        pagination={true}
        pageSize={3}
        hover={true}
        striped={true}
        bordered={true}
        glitchEffect={glitchEffect}
        className="demo-table"
      >
        <svelte:fragment slot="actions" let:row>
          <button
            class="yorha-action-btn-sm"
            onclick={() => mockAIAnalysis(row)}
          >
            ANALYZE
          </button>
          <button class="yorha-action-btn-sm">VIEW</button>
          <button class="yorha-action-btn-sm">EDIT</button>
        </svelte:fragment>
      </YoRHaTable>
    </div>

    <!-- Additional table demos -->
    <div class="yorha-demo-section">
      <h2 class="yorha-section-title">EVIDENCE MANAGEMENT</h2>
      <div class="yorha-section-description">
        Compact evidence tracking with processing status
      </div>

      <YoRHaTable
        columns={[
          { key: 'evidence_id', title: 'ID', sortable: true, width: '120px' },
          { key: 'filename', title: 'FILENAME', sortable: true, width: '200px' },
          { key: 'type', title: 'TYPE', sortable: true, type: 'status', width: '100px' },
          { key: 'status', title: 'STATUS', sortable: true, type: 'status', width: '100px' },
          { key: 'case_id', title: 'CASE', sortable: true, width: '120px' },
          { key: 'actions', title: 'ACTIONS', type: 'action', width: '150px' }
        ]}
        data={sampleEvidence}
        loading={false}
        selectable={true}
        sortable={true}
        pagination={false}
        dense={true}
        hover={true}
        striped={true}
        bordered={false}
        className="demo-compact-table"
      >
        <svelte:fragment slot="actions" let:row>
          <button class="yorha-action-btn-sm">PROCESS</button>
          <button class="yorha-action-btn-sm">DOWNLOAD</button>
        </svelte:fragment>
      </YoRHaTable>
    </div>
  {:else}
    <div class="yorha-demo-section">
      <h2 class="yorha-section-title">EVIDENCE ANALYSIS GRID</h2>
      <div class="yorha-section-description">
        Advanced data grid with AI processing metrics and filtering
      </div>

      <YoRHaDataGrid
        columns={evidenceGridConfig.columns}
        data={sampleEvidence}
        loading={gridLoading}
        editable={true}
        selectable={true}
        multiSelect={true}
        sortable={true}
        filterable={true}
        resizable={true}
        virtualScroll={false}
        rowHeight={50}
        maxHeight={400}
        glitchEffect={glitchEffect}
        className="demo-grid"
      >
        <svelte:fragment slot="actions" let:row>
          <button
            class="yorha-action-btn"
            onclick={() => mockAIAnalysis(row)}
          >
            ANALYZE
          </button>
          <button class="yorha-action-btn">VIEW</button>
          <button class="yorha-action-btn yorha-danger">DELETE</button>
        </svelte:fragment>
      </YoRHaDataGrid>
    </div>
  {/if}

  <!-- Statistics and Info Panel -->
  <div class="yorha-demo-stats">
    <h3 class="yorha-stats-title">SYSTEM STATISTICS</h3>
    <div class="yorha-stats-grid">
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">{sampleCases.length}</div>
        <div class="yorha-stat-label">ACTIVE CASES</div>
      </div>
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">{sampleEvidence.length}</div>
        <div class="yorha-stat-label">EVIDENCE ITEMS</div>
      </div>
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">
          {sampleEvidence.filter(e => e.ai_processed).length}
        </div>
        <div class="yorha-stat-label">AI PROCESSED</div>
      </div>
      <div class="yorha-stat-item">
        <div class="yorha-stat-value">
          {Math.round(sampleEvidence.reduce((avg, e) => avg + e.confidence_score, 0) / sampleEvidence.length * 100)}%
        </div>
        <div class="yorha-stat-label">AVG CONFIDENCE</div>
      </div>
    </div>
  </div>

  <!-- Quick Access Panel -->
  <div class="yorha-demo-features">
    <h3 class="yorha-features-title">QUICK ACCESS</h3>
    <div class="yorha-features-grid">
      <div class="yorha-feature">
        <div class="yorha-feature-icon">⚖️</div>
        <div class="yorha-feature-title">CASE ANALYSIS</div>
        <div class="yorha-feature-desc">AI-powered case review and insights</div>
      </div>
      <div class="yorha-feature">
        <div class="yorha-feature-icon">🔍</div>
        <div class="yorha-feature-title">EVIDENCE SEARCH</div>
        <div class="yorha-feature-desc">Advanced filtering and semantic search</div>
      </div>
      <div class="yorha-feature">
        <div class="yorha-feature-icon">📊</div>
        <div class="yorha-feature-title">ANALYTICS</div>
        <div class="yorha-feature-desc">Performance metrics and reporting</div>
      </div>
      <div class="yorha-feature">
        <div class="yorha-feature-icon">🤖</div>
        <div class="yorha-feature-title">AI ASSISTANT</div>
        <div class="yorha-feature-desc">Intelligent legal research support</div>
      </div>
      <div class="yorha-feature">
        <div class="yorha-feature-icon">🗂️</div>
        <div class="yorha-feature-title">DOCUMENT MGMT</div>
        <div class="yorha-feature-desc">Automated processing and classification</div>
      </div>
      <div class="yorha-feature">
        <div class="yorha-feature-icon">⚙️</div>
        <div class="yorha-feature-title">SYSTEM CONFIG</div>
        <div class="yorha-feature-desc">Platform settings and integrations</div>
      </div>
    </div>
  </div>
</div>

{#if showEvidenceBoard}
  <CanvasBoard on:close={() => showEvidenceBoard = false} />
{/if}

<style>
  .yorha-demo-container {
    @apply min-h-screen bg-black text-amber-400 font-mono p-6;
    background-image:
      radial-gradient(circle at 25% 25%, rgba(255, 191, 0, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 75% 75%, rgba(255, 215, 0, 0.05) 0%, transparent 50%);
  }

  .yorha-demo-header {
    @apply text-center mb-8 pb-6 border-b border-amber-400;
  }

  .yorha-demo-title {
    @apply text-4xl font-bold mb-2 tracking-wider;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-demo-subtitle {
    @apply text-lg text-amber-300 mb-6;
  }

  .yorha-demo-controls {
    @apply flex justify-center gap-4 flex-wrap;
  }

  .yorha-control-btn {
    @apply bg-black border border-amber-400 text-amber-400 px-4 py-2 font-bold;
    @apply hover:bg-amber-400 hover:text-black transition-all duration-200;
    @apply focus:outline-none focus:ring-2 focus:ring-amber-400;
  }

  .yorha-control-btn.active {
    @apply bg-amber-400 text-black;
    box-shadow: 0 0 15px rgba(255, 191, 0, 0.6);
  }

  .yorha-control-btn.yorha-action {
    @apply border-blue-400 text-blue-400 hover:bg-blue-400 hover:text-black;
  }

  .yorha-control-btn.yorha-warning {
    @apply border-red-400 text-red-400 hover:bg-red-400 hover:text-black;
  }

  .yorha-demo-section {
    @apply mb-12;
  }

  .yorha-section-title {
    @apply text-2xl font-bold mb-2 text-amber-300;
  }

  .yorha-section-description {
    @apply text-sm text-amber-500 mb-6;
  }

  :global(.demo-table),
  :global(.demo-compact-table),
  :global(.demo-grid) {
    @apply mb-4;
    box-shadow: 0 0 25px rgba(255, 191, 0, 0.3);
  }

  .yorha-demo-stats {
    @apply mb-8 bg-gray-900 p-6 border border-amber-400;
  }

  .yorha-stats-title {
    @apply text-xl font-bold mb-4 text-center;
  }

  .yorha-stats-grid {
    @apply grid grid-cols-2 md:grid-cols-4 gap-4;
  }

  .yorha-stat-item {
    @apply text-center;
  }

  .yorha-stat-value {
    @apply text-2xl font-bold text-amber-300;
  }

  .yorha-stat-label {
    @apply text-xs text-amber-500 uppercase tracking-wide;
  }

  .yorha-demo-features {
    @apply bg-gray-900 p-6 border border-amber-400;
  }

  .yorha-features-title {
    @apply text-xl font-bold mb-6 text-center;
  }

  .yorha-features-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6;
  }

  .yorha-feature {
    @apply text-center p-4 border border-amber-400 border-opacity-30;
    @apply hover:border-opacity-100 hover:bg-amber-400 hover:bg-opacity-5 transition-all;
  }

  .yorha-feature-icon {
    @apply text-2xl mb-2;
  }

  .yorha-feature-title {
    @apply font-bold text-amber-300 mb-2;
  }

  .yorha-feature-desc {
    @apply text-sm text-amber-500;
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .yorha-demo-title {
      @apply text-2xl;
    }

    .yorha-demo-controls {
      @apply gap-2;
    }

    .yorha-control-btn {
      @apply text-sm px-3 py-1;
    }
  }
</style>
