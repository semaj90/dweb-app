<!-- YoRHa Demo Page - Fully Working Integration Demo -->
<script lang="ts">
  import { onMount } from 'svelte';
  import YoRHaDataGrid from '$lib/components/yorha/YoRHaDataGrid.svelte';
  import YoRHaForm from '$lib/components/yorha/YoRHaForm.svelte';
  import YoRHaTerminal from '$lib/components/yorha/YoRHaTerminal.svelte';
  import YoRHaModal from '$lib/components/yorha/YoRHaModal.svelte';

  // Demo state
  let currentView = $state('grid');
  let modalOpen = $state(false);
  let terminalActive = $state(true);
  let notifications = $state<any[]>([]);

  // Sample data for demonstration
  let demoData = $state([
    {
      id: 'DOC-000001',
      yorha_id: 'YORHA-DOC-000001',
      title: 'Contract Analysis - YoRHa Legal Division',
      documentType: 'contract',
      status: 'ANALYZED',
      priority: 'HIGH',
      confidence: 0.94,
      processed: true,
      timestamp: new Date('2024-08-15'),
      content: 'Legal contract analysis using YoRHa AI systems',
      classification: 'CONFIDENTIAL',
      assignedTo: 'Unit 2B',
      jurisdiction: 'YoRHa Command'
    },
    {
      id: 'DOC-000002',
      yorha_id: 'YORHA-DOC-000002',
      title: 'Evidence Evaluation - Android Units',
      documentType: 'evidence',
      status: 'PENDING',
      priority: 'MEDIUM',
      confidence: 0.87,
      processed: false,
      timestamp: new Date('2024-08-14'),
      content: 'Digital evidence from android unit investigation',
      classification: 'RESTRICTED',
      assignedTo: 'Unit 9S',
      jurisdiction: 'YoRHa Command'
    },
    {
      id: 'DOC-000003',
      yorha_id: 'YORHA-DOC-000003',
      title: 'Legal Precedent - Machine Litigation',
      documentType: 'precedent',
      status: 'ARCHIVED',
      priority: 'LOW',
      confidence: 0.92,
      processed: true,
      timestamp: new Date('2024-08-13'),
      content: 'Historical legal precedent for machine rights cases',
      classification: 'PUBLIC',
      assignedTo: 'Unit A2',
      jurisdiction: 'Global Legal Network'
    },
    {
      id: 'DOC-000004',
      yorha_id: 'YORHA-DOC-000004',
      title: 'Regulatory Compliance - AI Ethics',
      documentType: 'regulation',
      status: 'ACTIVE',
      priority: 'CRITICAL',
      confidence: 0.98,
      processed: true,
      timestamp: new Date('2024-08-15'),
      content: 'AI ethics compliance regulations for YoRHa operations',
      classification: 'TOP_SECRET',
      assignedTo: 'Commander White',
      jurisdiction: 'YoRHa Command'
    },
    {
      id: 'DOC-000005',
      yorha_id: 'YORHA-DOC-000005',
      title: 'Case Brief - Android Rights Violation',
      documentType: 'brief',
      status: 'IN_REVIEW',
      priority: 'HIGH',
      confidence: 0.89,
      processed: true,
      timestamp: new Date('2024-08-14'),
      content: 'Legal brief concerning alleged android rights violations',
      classification: 'CONFIDENTIAL',
      assignedTo: 'Unit 2B',
      jurisdiction: 'YoRHa Legal Division'
    }
  ]);

  // Grid configuration
  const columns = [
    { key: 'yorha_id', title: 'YORHA ID', sortable: true, width: 160 },
    { key: 'title', title: 'DOCUMENT TITLE', sortable: true, filterable: true, width: 300 },
    { key: 'documentType', title: 'TYPE', sortable: true, type: 'status', width: 120 },
    { key: 'priority', title: 'PRIORITY', sortable: true, type: 'status', width: 100 },
    { key: 'status', title: 'STATUS', sortable: true, type: 'status', width: 120 },
    { key: 'confidence', title: 'CONFIDENCE', sortable: true, type: 'number', width: 120,
      formatter: (value) => `${(value * 100).toFixed(1)}%` },
    { key: 'assignedTo', title: 'ASSIGNED TO', sortable: true, width: 140 },
    { key: 'timestamp', title: 'PROCESSED', sortable: true, type: 'date', width: 140 },
    { key: 'actions', title: 'ACTIONS', type: 'action', width: 200 }
  ];

  // Form configuration
  const formFields = [
    { id: 'title', label: 'Document Title', type: 'text', required: true },
    { id: 'content', label: 'Content Analysis', type: 'textarea', required: true },
    { id: 'documentType', label: 'Document Type', type: 'select', required: true,
      options: [
        { value: 'contract', label: 'Contract' },
        { value: 'evidence', label: 'Evidence' },
        { value: 'precedent', label: 'Legal Precedent' },
        { value: 'regulation', label: 'Regulation' },
        { value: 'brief', label: 'Legal Brief' }
      ]
    },
    { id: 'priority', label: 'Priority Level', type: 'select', required: true,
      options: [
        { value: 'LOW', label: 'Low Priority' },
        { value: 'MEDIUM', label: 'Medium Priority' },
        { value: 'HIGH', label: 'High Priority' },
        { value: 'CRITICAL', label: 'Critical Priority' }
      ]
    },
    { id: 'assignedTo', label: 'Assigned To', type: 'select',
      options: [
        { value: 'Unit 2B', label: 'Unit 2B - Combat Specialist' },
        { value: 'Unit 9S', label: 'Unit 9S - Scanner Unit' },
        { value: 'Unit A2', label: 'Unit A2 - Attacker Unit' },
        { value: 'Commander White', label: 'Commander White' }
      ]
    },
    { id: 'classification', label: 'Security Classification', type: 'select',
      options: [
        { value: 'PUBLIC', label: 'Public' },
        { value: 'RESTRICTED', label: 'Restricted' },
        { value: 'CONFIDENTIAL', label: 'Confidential' },
        { value: 'TOP_SECRET', label: 'Top Secret' }
      ]
    }
  ];

  // Enhanced RAG simulation
  let ragResults = $state<any[]>([]);
  let ragAnalysis = $state<any>(null);

  // Functions
  function handleGridAction(action: string, row: any) {
    switch (action) {
      case 'edit':
        openEditModal(row);
        break;
      case 'delete':
        deleteDocument(row.id);
        break;
      case 'analyze':
        performAnalysis(row);
        break;
      case 'view':
        viewDocument(row);
        break;
    }
  }

  function openEditModal(row: any) {
    modalOpen = true;
    addNotification('info', `Editing document: ${row.title}`);
  }

  function deleteDocument(id: string) {
    if (confirm('Confirm deletion of classified document?')) {
      demoData = demoData.filter(item => item.id !== id);
      addNotification('success', 'Document securely deleted from YoRHa archives');
    }
  }

  function performAnalysis(row: any) {
    addNotification('info', `Initiating enhanced AI analysis for ${row.yorha_id}`);
    
    // Simulate enhanced analysis
    setTimeout(() => {
      ragResults = [
        {
          id: 'ANALYSIS-001',
          title: `Enhanced Analysis: ${row.title}`,
          confidence: 0.95,
          summary: `AI analysis of ${row.title} has been completed. The document shows high legal relevance with critical classification requirements.`,
          keyTerms: ['legal analysis', 'yorha protocol', 'android rights', 'machine litigation'],
          riskLevel: row.priority === 'CRITICAL' ? 'HIGH' : 'MEDIUM',
          recommendations: [
            'Immediate review by commanding officer required',
            'Cross-reference with existing legal precedents',
            'Update security classification if necessary'
          ],
          processingTime: Math.random() * 2000 + 500,
          yorhaAnalysis: {
            relevanceScore: 0.94,
            legalWeight: 0.89,
            riskFactor: row.priority === 'CRITICAL' ? 0.85 : 0.45,
            actionRequired: row.priority === 'CRITICAL' ? 'URGENT' : 'REVIEW'
          }
        }
      ];

      ragAnalysis = {
        confidenceScore: 0.95,
        legalComplexity: 'HIGH',
        riskLevel: row.priority === 'CRITICAL' ? 'HIGH' : 'MEDIUM',
        processingTime: Math.random() * 2000 + 500,
        aiModel: 'YoRHa-Legal-Enhanced-4.0',
        timestamp: new Date()
      };

      addNotification('success', `Enhanced analysis completed for ${row.yorha_id}`);
    }, 2000);
  }

  function viewDocument(row: any) {
    addNotification('info', `Accessing classified document: ${row.yorha_id}`);
    currentView = 'document';
  }

  function createDocument(data: any) {
    const newDoc = {
      id: `DOC-${String(demoData.length + 1).padStart(6, '0')}`,
      yorha_id: `YORHA-DOC-${String(demoData.length + 1).padStart(6, '0')}`,
      ...data,
      confidence: Math.random() * 0.3 + 0.7, // 0.7 - 1.0
      processed: false,
      timestamp: new Date(),
      status: 'PENDING'
    };

    demoData = [...demoData, newDoc];
    modalOpen = false;
    addNotification('success', `New document created: ${newDoc.yorha_id}`);
  }

  function handleTerminalCommand(command: string) {
    const parts = command.toLowerCase().split(' ');
    const cmd = parts[0];

    switch (cmd) {
      case 'status':
        addNotification('info', `YoRHa Legal AI System Status: OPERATIONAL | Documents: ${demoData.length} | Analysis Engine: ACTIVE`);
        break;
      case 'analyze':
        if (parts[1]) {
          const docId = parts[1].toUpperCase();
          const doc = demoData.find(d => d.yorha_id.includes(docId) || d.id.includes(docId));
          if (doc) {
            performAnalysis(doc);
          } else {
            addNotification('error', `Document not found: ${docId}`);
          }
        } else {
          addNotification('info', 'Usage: analyze <document_id>');
        }
        break;
      case 'create':
        modalOpen = true;
        addNotification('info', 'Document creation interface activated');
        break;
      case 'list':
        addNotification('info', `Active documents: ${demoData.map(d => d.yorha_id).join(', ')}`);
        break;
      case 'classify':
        if (parts[1] && parts[2]) {
          const docId = parts[1].toUpperCase();
          const classification = parts[2].toUpperCase();
          const doc = demoData.find(d => d.yorha_id.includes(docId));
          if (doc) {
            doc.classification = classification;
            addNotification('success', `Document ${docId} reclassified as ${classification}`);
          }
        }
        break;
      case 'help':
        addNotification('info', 'Available commands: status, analyze <id>, create, list, classify <id> <level>, clear, help');
        break;
      case 'clear':
        ragResults = [];
        ragAnalysis = null;
        addNotification('info', 'Analysis results cleared');
        break;
      default:
        addNotification('warning', `Unknown command: ${command}. Type 'help' for available commands.`);
    }
  }

  function addNotification(type: 'success' | 'error' | 'warning' | 'info', message: string) {
    const notification = {
      id: Date.now(),
      type,
      message,
      timestamp: new Date()
    };
    notifications = [...notifications, notification];
    
    setTimeout(() => {
      notifications = notifications.filter(n => n.id !== notification.id);
    }, 5000);
  }

  onMount(() => {
    addNotification('success', 'YoRHa Legal AI System initialized successfully');
    addNotification('info', 'All components operational - Enhanced RAG engine active');
  });
</script>

<svelte:head>
  <title>YoRHa Legal AI Demo - Complete Integration</title>
</svelte:head>

<div class="yorha-demo">
  <!-- Header -->
  <header class="demo-header">
    <div class="header-content">
      <h1 class="demo-title">YoRHa Legal AI Demonstration</h1>
      <div class="system-status">
        <span class="status-indicator active">SYSTEM: OPERATIONAL</span>
        <span class="status-indicator">DOCS: {demoData.length}</span>
        <span class="status-indicator">AI: ENHANCED</span>
      </div>
    </div>
    
    <nav class="demo-nav">
      <button 
        class="nav-btn {currentView === 'grid' ? 'active' : ''}"
        onclick={() => currentView = 'grid'}
      >
        DATA GRID
      </button>
      <button 
        class="nav-btn {currentView === 'analysis' ? 'active' : ''}"
        onclick={() => currentView = 'analysis'}
      >
        AI ANALYSIS
      </button>
      <button 
        class="nav-btn {currentView === 'form' ? 'active' : ''}"
        onclick={() => currentView = 'form'}
      >
        DOCUMENT ENTRY
      </button>
      <button 
        class="nav-btn"
        onclick={() => modalOpen = true}
      >
        CREATE NEW
      </button>
    </nav>
  </header>

  <!-- Main Content -->
  <main class="demo-main">
    <div class="content-area">
      {#if currentView === 'grid'}
        <section class="grid-section">
          <div class="section-header">
            <h2 class="section-title">LEGAL DOCUMENT MANAGEMENT GRID</h2>
            <div class="grid-controls">
              <span class="record-count">{demoData.length} CLASSIFIED DOCUMENTS</span>
            </div>
          </div>
          
          <YoRHaDataGrid
            columns={columns}
            data={demoData}
            selectable={true}
            multiSelect={true}
            sortable={true}
            filterable={true}
            resizable={true}
            maxHeight={500}
            glitchEffect={false}

          />
        </section>

      {:else if currentView === 'analysis'}
        <section class="analysis-section">
          <div class="section-header">
            <h2 class="section-title">ENHANCED AI ANALYSIS RESULTS</h2>
            {#if ragAnalysis}
              <div class="analysis-stats">
                <span class="stat">Confidence: {(ragAnalysis.confidenceScore * 100).toFixed(1)}%</span>
                <span class="stat">Complexity: {ragAnalysis.legalComplexity}</span>
                <span class="stat">Risk: {ragAnalysis.riskLevel}</span>
              </div>
            {/if}
          </div>

          {#if ragResults.length > 0}
            <div class="analysis-results">
              {#each ragResults as result}
                <div class="analysis-result">
                  <div class="result-header">
                    <h3 class="result-title">{result.title}</h3>
                    <div class="result-meta">
                      <span class="confidence">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                      <span class="risk-level">Risk: {result.riskLevel}</span>
                    </div>
                  </div>
                  
                  <div class="result-content">
                    <p class="summary">{result.summary}</p>
                    
                    <div class="key-terms">
                      <h4>Key Legal Terms:</h4>
                      <div class="terms">
                        {#each result.keyTerms as term}
                          <span class="term">{term}</span>
                        {/each}
                      </div>
                    </div>

                    <div class="recommendations">
                      <h4>AI Recommendations:</h4>
                      <ul>
                        {#each result.recommendations as rec}
                          <li>{rec}</li>
                        {/each}
                      </ul>
                    </div>

                    {#if result.yorhaAnalysis}
                      <div class="detailed-analysis">
                        <h4>Detailed Analysis Metrics:</h4>
                        <div class="metrics">
                          <div class="metric">
                            <span class="metric-label">Relevance Score:</span>
                            <span class="metric-value">{(result.yorhaAnalysis.relevanceScore * 100).toFixed(1)}%</span>
                          </div>
                          <div class="metric">
                            <span class="metric-label">Legal Weight:</span>
                            <span class="metric-value">{(result.yorhaAnalysis.legalWeight * 100).toFixed(1)}%</span>
                          </div>
                          <div class="metric">
                            <span class="metric-label">Risk Factor:</span>
                            <span class="metric-value">{(result.yorhaAnalysis.riskFactor * 100).toFixed(1)}%</span>
                          </div>
                          <div class="metric">
                            <span class="metric-label">Action Required:</span>
                            <span class="metric-value action-{result.yorhaAnalysis.actionRequired.toLowerCase()}">{result.yorhaAnalysis.actionRequired}</span>
                          </div>
                        </div>
                      </div>
                    {/if}
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="no-analysis">
              <p>No analysis results available. Select a document and run analysis to see results here.</p>
              <button class="demo-btn" onclick={() => currentView = 'grid'}>
                Go to Document Grid
              </button>
            </div>
          {/if}
        </section>

      {:else if currentView === 'form'}
        <section class="form-section">
          <div class="section-header">
            <h2 class="section-title">DOCUMENT CREATION INTERFACE</h2>
          </div>
          
          <YoRHaForm
            title="Create New Legal Document"
            subtitle="Enter classified legal document information"
            fields={formFields}
            submitLabel="CREATE DOCUMENT"
            cancelLabel="ABORT"
            on:submit={createDocument}
            on:cancel={() => currentView = 'grid'}
          />
        </section>
      {/if}
    </div>

    <!-- Terminal Panel -->
    <aside class="terminal-section">
      <YoRHaTerminal
        title="YoRHa Legal AI Command Terminal"
        isActive={terminalActive}
        onCommand={handleTerminalCommand}
      />
    </aside>
  </main>

  <!-- Modal -->
  {#if modalOpen}
    <YoRHaModal
      isOpen={modalOpen}
      title="Create New Legal Document"
      onClose={() => modalOpen = false}
    >
      <YoRHaForm
        title="Document Creation Interface"
        subtitle="Enter classified document information"
        fields={formFields}
        submitLabel="CREATE"
        cancelLabel="CANCEL"
        on:submit={createDocument}
        on:cancel={() => modalOpen = false}
      />
    </YoRHaModal>
  {/if}

  <!-- Notifications -->
  <div class="notifications">
    {#each notifications as notification}
      <div class="notification notification-{notification.type}">
        <div class="notification-content">
          <span class="notification-type">[{notification.type.toUpperCase()}]</span>
          <span class="notification-message">{notification.message}</span>
        </div>
        <button 
          class="notification-close"
          onclick={() => notifications = notifications.filter(n => n.id !== notification.id)}
        >
          ✕
        </button>
      </div>
    {/each}
  </div>
</div>

<style>
  .yorha-demo {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
  }

  .demo-header {
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

  .demo-title {
    font-size: 24px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0;
  }

  .system-status {
    display: flex;
    gap: 16px;
  }

  .status-indicator {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 8px;
    border: 1px solid #000;
    background: rgba(0, 0, 0, 0.2);
  }

  .status-indicator.active {
    color: #00ff41;
    border-color: #00ff41;
    background: rgba(0, 255, 65, 0.1);
  }

  .demo-nav {
    display: flex;
    gap: 8px;
  }

  .nav-btn {
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

  .nav-btn:hover,
  .nav-btn.active {
    background: #ffbf00;
    color: #000;
  }

  .demo-main {
    display: flex;
    min-height: calc(100vh - 120px);
  }

  .content-area {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  .terminal-section {
    width: 400px;
    border-left: 2px solid #ffbf00;
    background: #0a0a0a;
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

  .grid-controls,
  .analysis-stats {
    display: flex;
    gap: 16px;
    align-items: center;
  }

  .record-count,
  .stat {
    font-size: 12px;
    color: #ffbf00;
    text-transform: uppercase;
    font-weight: 600;
  }

  .grid-actions {
    display: flex;
    gap: 4px;
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

  .analysis-results {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .analysis-result {
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

  .confidence,
  .risk-level {
    font-size: 12px;
    color: #808080;
    text-transform: uppercase;
  }

  .result-content {
    color: #e0e0e0;
  }

  .summary {
    line-height: 1.5;
    margin-bottom: 16px;
  }

  .key-terms h4,
  .recommendations h4,
  .detailed-analysis h4 {
    font-size: 14px;
    color: #ffbf00;
    margin: 0 0 8px 0;
    text-transform: uppercase;
  }

  .terms {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
  }

  .term {
    background: #333;
    color: #ffbf00;
    padding: 4px 8px;
    font-size: 11px;
    text-transform: uppercase;
  }

  .recommendations ul {
    margin: 0 0 16px 0;
    padding-left: 20px;
  }

  .recommendations li {
    margin-bottom: 4px;
    line-height: 1.4;
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
  }

  .metric {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #333;
  }

  .metric-label {
    font-size: 12px;
    color: #808080;
  }

  .metric-value {
    font-size: 12px;
    font-weight: 600;
    color: #ffbf00;
  }

  .action-urgent {
    color: #ff0041 !important;
  }

  .action-review {
    color: #ffaa00 !important;
  }

  .no-analysis {
    text-align: center;
    padding: 40px;
    color: #808080;
  }

  .demo-btn {
    background: #ffbf00;
    border: 2px solid #ffbf00;
    color: #000;
    padding: 12px 24px;
    font-family: inherit;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    cursor: pointer;
    margin-top: 16px;
  }

  .notifications {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-width: 400px;
  }

  .notification {
    background: #1a1a1a;
    border: 1px solid #333;
    padding: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-left: 4px solid #ffbf00;
  }

  .notification-success {
    border-left-color: #00ff41;
  }

  .notification-error {
    border-left-color: #ff0041;
  }

  .notification-warning {
    border-left-color: #ffaa00;
  }

  .notification-info {
    border-left-color: #00aaff;
  }

  .notification-content {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .notification-type {
    font-size: 10px;
    font-weight: 700;
    color: #ffbf00;
  }

  .notification-message {
    font-size: 12px;
    color: #e0e0e0;
  }

  .notification-close {
    background: none;
    border: none;
    color: #808080;
    cursor: pointer;
    font-size: 14px;
    padding: 0;
    width: 16px;
    height: 16px;
  }

  .notification-close:hover {
    color: #ffbf00;
  }

  @media (max-width: 1200px) {
    .demo-main {
      flex-direction: column;
    }

    .terminal-section {
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

    .demo-nav {
      flex-wrap: wrap;
    }

    .content-area {
      padding: 16px;
    }

    .metrics {
      grid-template-columns: 1fr;
    }
  }
</style>