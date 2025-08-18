<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';

  // State management
  let isLoading = $state(true);
  let evidenceFiles = $state([]);
  let analysisResults = $state([]);
  let selectedFile = $state(null);
  let processingStatus = $state('idle');
  let uploadProgress = $state(0);

  // Mock evidence data
  let mockEvidence = [
    {
      id: 'evidence-001',
      name: 'contract_signed.pdf',
      type: 'document',
      size: '2.3 MB',
      status: 'analyzed',
      timestamp: '2024-08-18T10:30:00Z',
      analysis: {
        confidence: 0.94,
        entities: ['Person: John Smith', 'Date: 2024-01-15', 'Amount: $50,000'],
        summary: 'Legal contract document with digital signatures detected'
      }
    },
    {
      id: 'evidence-002', 
      name: 'email_thread.eml',
      type: 'email',
      size: '156 KB',
      status: 'processing',
      timestamp: '2024-08-18T11:15:00Z',
      analysis: null
    },
    {
      id: 'evidence-003',
      name: 'financial_records.xlsx',
      type: 'spreadsheet', 
      size: '4.7 MB',
      status: 'pending',
      timestamp: '2024-08-18T12:00:00Z',
      analysis: null
    }
  ];

  // Navigation function
  function navigateHome() {
    goto('/');
  }

  function selectFile(file) {
    selectedFile = file;
  }

  function getStatusColor(status) {
    switch (status) {
      case 'analyzed': return '#00ff41';
      case 'processing': return '#ffbf00';
      case 'pending': return '#ff6b6b';
      default: return '#888';
    }
  }

  function getStatusIcon(status) {
    switch (status) {
      case 'analyzed': return '✅';
      case 'processing': return '⚙️';
      case 'pending': return '⏳';
      default: return '❓';
    }
  }

  function getTypeIcon(type) {
    switch (type) {
      case 'document': return '📄';
      case 'email': return '📧';
      case 'spreadsheet': return '📊';
      case 'image': return '🖼️';
      default: return '📁';
    }
  }

  // Initialize component
  onMount(() => {
    evidenceFiles = mockEvidence;
    setTimeout(() => {
      isLoading = false;
    }, 800);

    // Simulate processing updates
    const processingInterval = setInterval(() => {
      evidenceFiles = evidenceFiles.map(file => {
        if (file.status === 'processing') {
          return {
            ...file,
            status: 'analyzed',
            analysis: {
              confidence: 0.87,
              entities: ['Person: Jane Doe', 'Subject: Meeting Schedule'],
              summary: 'Email communication analysis completed'
            }
          };
        }
        return file;
      });
    }, 3000);

    return () => clearInterval(processingInterval);
  });
</script>

<svelte:head>
  <title>Evidence Analysis - YoRHa Legal AI</title>
  <meta name="description" content="Digital evidence processing with OCR and AI analysis">
</svelte:head>

<!-- Loading Screen -->
{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">🔍</div>
      <div class="loading-text">INITIALIZING EVIDENCE ANALYSIS SYSTEM...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <!-- Main Interface -->
  <div class="evidence-interface">
    
    <!-- Header -->
    <header class="evidence-header">
      <div class="header-left">
        <button class="back-button" onclick={navigateHome}>
          ← COMMAND CENTER
        </button>
        <div class="header-title">
          <h1>🔍 EVIDENCE ANALYSIS SYSTEM</h1>
          <div class="header-subtitle">Digital Evidence Processing with OCR and AI</div>
        </div>
      </div>
      
      <div class="header-stats">
        <div class="stat-item">
          <div class="stat-value">{evidenceFiles.length}</div>
          <div class="stat-label">TOTAL FILES</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{evidenceFiles.filter(f => f.status === 'analyzed').length}</div>
          <div class="stat-label">ANALYZED</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{evidenceFiles.filter(f => f.status === 'processing').length}</div>
          <div class="stat-label">PROCESSING</div>
        </div>
      </div>
    </header>

    <!-- Main Content Grid -->
    <div class="evidence-content">
      
      <!-- Evidence Files List -->
      <section class="evidence-files">
        <h2 class="section-title">📁 EVIDENCE FILES</h2>
        
        <div class="upload-zone">
          <div class="upload-area">
            <div class="upload-icon">📤</div>
            <div class="upload-text">
              <div class="upload-title">DROP FILES TO ANALYZE</div>
              <div class="upload-subtitle">Supports PDF, DOC, XLS, EML, TXT, Images</div>
            </div>
            <button class="upload-button">BROWSE FILES</button>
          </div>
        </div>

        <div class="files-list">
          {#each evidenceFiles as file (file.id)}
            <div 
              class="file-card {selectedFile?.id === file.id ? 'selected' : ''}"
              onclick={() => selectFile(file)}
            >
              <div class="file-icon">{getTypeIcon(file.type)}</div>
              
              <div class="file-info">
                <div class="file-name">{file.name}</div>
                <div class="file-meta">
                  <span class="file-size">{file.size}</span>
                  <span class="file-timestamp">{new Date(file.timestamp).toLocaleString()}</span>
                </div>
              </div>
              
              <div class="file-status">
                <div class="status-badge" style="color: {getStatusColor(file.status)}">
                  {getStatusIcon(file.status)} {file.status.toUpperCase()}
                </div>
              </div>
              
              <div class="file-actions">
                <button class="action-btn">VIEW</button>
                <button class="action-btn">ANALYZE</button>
              </div>
            </div>
          {/each}
        </div>
      </section>

      <!-- Analysis Panel -->
      <section class="analysis-panel">
        <h2 class="section-title">🧠 AI ANALYSIS</h2>
        
        {#if selectedFile}
          <div class="analysis-content">
            <div class="file-details">
              <h3>{selectedFile.name}</h3>
              <div class="file-metadata">
                <div class="meta-item">
                  <span class="meta-label">TYPE:</span>
                  <span class="meta-value">{selectedFile.type.toUpperCase()}</span>
                </div>
                <div class="meta-item">
                  <span class="meta-label">SIZE:</span>
                  <span class="meta-value">{selectedFile.size}</span>
                </div>
                <div class="meta-item">
                  <span class="meta-label">STATUS:</span>
                  <span class="meta-value" style="color: {getStatusColor(selectedFile.status)}">
                    {selectedFile.status.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            {#if selectedFile.analysis}
              <div class="analysis-results">
                <div class="confidence-score">
                  <div class="confidence-label">CONFIDENCE SCORE</div>
                  <div class="confidence-value">{Math.round(selectedFile.analysis.confidence * 100)}%</div>
                  <div class="confidence-bar">
                    <div 
                      class="confidence-fill" 
                      style="width: {selectedFile.analysis.confidence * 100}%"
                    ></div>
                  </div>
                </div>

                <div class="entities-section">
                  <h4>EXTRACTED ENTITIES</h4>
                  <div class="entities-list">
                    {#each selectedFile.analysis.entities as entity}
                      <div class="entity-tag">{entity}</div>
                    {/each}
                  </div>
                </div>

                <div class="summary-section">
                  <h4>AI SUMMARY</h4>
                  <div class="summary-text">{selectedFile.analysis.summary}</div>
                </div>
              </div>
            {:else if selectedFile.status === 'processing'}
              <div class="processing-indicator">
                <div class="processing-spinner">⚙️</div>
                <div class="processing-text">AI ANALYSIS IN PROGRESS...</div>
                <div class="processing-stages">
                  <div class="stage active">OCR EXTRACTION</div>
                  <div class="stage">ENTITY RECOGNITION</div>
                  <div class="stage">SEMANTIC ANALYSIS</div>
                  <div class="stage">CONFIDENCE SCORING</div>
                </div>
              </div>
            {:else}
              <div class="no-analysis">
                <div class="no-analysis-icon">⏳</div>
                <div class="no-analysis-text">ANALYSIS PENDING</div>
                <button class="start-analysis-btn">START AI ANALYSIS</button>
              </div>
            {/if}
          </div>
        {:else}
          <div class="no-selection">
            <div class="no-selection-icon">📋</div>
            <div class="no-selection-text">SELECT A FILE TO VIEW ANALYSIS</div>
          </div>
        {/if}
      </section>
    </div>

    <!-- Footer Actions -->
    <footer class="evidence-footer">
      <div class="footer-actions">
        <button class="footer-btn primary">BATCH ANALYZE ALL</button>
        <button class="footer-btn secondary">EXPORT RESULTS</button>
        <button class="footer-btn secondary">GENERATE REPORT</button>
      </div>
      
      <div class="footer-info">
        <div class="system-status">
          <span class="status-label">AI ENGINE:</span>
          <span class="status-value active">OPERATIONAL</span>
        </div>
        <div class="timestamp">LAST UPDATE: {new Date().toLocaleString()}</div>
      </div>
    </footer>
  </div>
{/if}

<style>
  /* === GLOBAL VARIABLES === */
  :global(:root) {
    --yorha-primary: #c4b49a;
    --yorha-secondary: #b5a48a;
    --yorha-accent-warm: #ffbf00;
    --yorha-accent-cool: #4ecdc4;
    --yorha-success: #00ff41;
    --yorha-warning: #ffbf00;
    --yorha-error: #ff6b6b;
    --yorha-light: #ffffff;
    --yorha-muted: #f0f0f0;
    --yorha-dark: #1a1a1a;
    --yorha-darker: #0a0a0a;
    --yorha-bg: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
  }

  /* === LOADING SCREEN === */
  .loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    background: var(--yorha-bg);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    font-family: 'JetBrains Mono', monospace;
    color: var(--yorha-light);
  }

  .loading-content {
    text-align: center;
    animation: fadeInUp 0.8s ease-out;
  }

  .loading-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    color: var(--yorha-accent-warm);
    animation: pulse 2s ease-in-out infinite;
  }

  .loading-text {
    font-size: 1.2rem;
    color: var(--yorha-muted);
    letter-spacing: 2px;
    margin-bottom: 2rem;
  }

  .loading-bar {
    width: 300px;
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
    margin: 0 auto;
  }

  .loading-progress {
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-accent-warm), var(--yorha-success));
    animation: loading 2s ease-in-out infinite;
  }

  /* === MAIN INTERFACE === */
  .evidence-interface {
    min-height: 100vh;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
    animation: fadeIn 0.5s ease-out;
  }

  /* === HEADER === */
  .evidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2rem;
    border-bottom: 2px solid var(--yorha-accent-warm);
    background: rgba(26, 26, 26, 0.9);
    backdrop-filter: blur(10px);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 2rem;
  }

  .back-button {
    background: transparent;
    border: 2px solid var(--yorha-accent-cool);
    color: var(--yorha-accent-cool);
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    transform: translateX(-5px);
  }

  .header-title h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(45deg, var(--yorha-accent-warm), var(--yorha-success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .header-subtitle {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    margin-top: 0.5rem;
    letter-spacing: 1px;
  }

  .header-stats {
    display: flex;
    gap: 2rem;
  }

  .stat-item {
    text-align: center;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--yorha-accent-warm);
    line-height: 1;
  }

  .stat-label {
    font-size: 0.7rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
  }

  /* === MAIN CONTENT === */
  .evidence-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    padding: 2rem;
    min-height: calc(100vh - 250px);
  }

  /* === EVIDENCE FILES SECTION === */
  .evidence-files {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
  }

  .section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--yorha-accent-warm);
    margin: 0 0 1.5rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
  }

  .upload-zone {
    margin-bottom: 2rem;
  }

  .upload-area {
    border: 2px dashed rgba(78, 205, 196, 0.5);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
  }

  .upload-area:hover {
    border-color: var(--yorha-accent-cool);
    background: rgba(78, 205, 196, 0.05);
  }

  .upload-icon {
    font-size: 3rem;
    color: var(--yorha-accent-cool);
    margin-bottom: 1rem;
  }

  .upload-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--yorha-light);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .upload-subtitle {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    margin-bottom: 1.5rem;
  }

  .upload-button {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .upload-button:hover {
    background: var(--yorha-success);
    transform: scale(1.05);
  }

  .files-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .file-card {
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    gap: 1rem;
    align-items: center;
    background: rgba(42, 42, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .file-card:hover {
    border-color: var(--yorha-accent-warm);
    background: rgba(42, 42, 42, 0.8);
    transform: translateY(-2px);
  }

  .file-card.selected {
    border-color: var(--yorha-success);
    background: rgba(0, 255, 65, 0.1);
  }

  .file-icon {
    font-size: 2rem;
  }

  .file-info {
    min-width: 0;
  }

  .file-name {
    font-weight: 600;
    color: var(--yorha-light);
    margin-bottom: 0.3rem;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
  }

  .file-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: var(--yorha-muted);
  }

  .status-badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 0.3rem 0.6rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    border: 1px solid currentColor;
    text-transform: uppercase;
  }

  .file-actions {
    display: flex;
    gap: 0.5rem;
  }

  .action-btn {
    background: transparent;
    border: 1px solid var(--yorha-accent-cool);
    color: var(--yorha-accent-cool);
    padding: 0.4rem 0.8rem;
    font-family: inherit;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 3px;
  }

  .action-btn:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  /* === ANALYSIS PANEL === */
  .analysis-panel {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
  }

  .analysis-content {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .file-details h3 {
    font-size: 1.2rem;
    color: var(--yorha-light);
    margin: 0 0 1rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .file-metadata {
    display: grid;
    gap: 0.5rem;
  }

  .meta-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
  }

  .meta-label {
    color: var(--yorha-muted);
    font-weight: 600;
  }

  .meta-value {
    color: var(--yorha-light);
  }

  .confidence-score {
    padding: 1rem;
    background: rgba(42, 42, 42, 0.6);
    border-radius: 6px;
    border: 1px solid rgba(0, 255, 65, 0.3);
  }

  .confidence-label {
    font-size: 0.8rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }

  .confidence-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--yorha-success);
    margin-bottom: 0.5rem;
  }

  .confidence-bar {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
  }

  .confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-success), var(--yorha-accent-warm));
    transition: width 0.5s ease;
  }

  .entities-section h4,
  .summary-section h4 {
    font-size: 1rem;
    color: var(--yorha-accent-warm);
    margin: 0 0 1rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .entities-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .entity-tag {
    background: rgba(78, 205, 196, 0.2);
    color: var(--yorha-accent-cool);
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    border: 1px solid var(--yorha-accent-cool);
  }

  .summary-text {
    color: var(--yorha-light);
    line-height: 1.6;
    padding: 1rem;
    background: rgba(42, 42, 42, 0.4);
    border-radius: 6px;
    border-left: 3px solid var(--yorha-accent-warm);
  }

  .processing-indicator {
    text-align: center;
    padding: 2rem;
  }

  .processing-spinner {
    font-size: 3rem;
    animation: spin 2s linear infinite;
    margin-bottom: 1rem;
  }

  .processing-text {
    font-size: 1.1rem;
    color: var(--yorha-warning);
    margin-bottom: 2rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .processing-stages {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .stage {
    padding: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    color: var(--yorha-muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .stage.active {
    border-color: var(--yorha-warning);
    background: rgba(255, 191, 0, 0.1);
    color: var(--yorha-warning);
  }

  .no-analysis,
  .no-selection {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--yorha-muted);
  }

  .no-analysis-icon,
  .no-selection-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
  }

  .no-analysis-text,
  .no-selection-text {
    font-size: 1.1rem;
    margin-bottom: 2rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .start-analysis-btn {
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .start-analysis-btn:hover {
    background: var(--yorha-accent-warm);
    transform: scale(1.05);
  }

  /* === FOOTER === */
  .evidence-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2rem;
    border-top: 2px solid var(--yorha-accent-warm);
    background: rgba(26, 26, 26, 0.9);
  }

  .footer-actions {
    display: flex;
    gap: 1rem;
  }

  .footer-btn {
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .footer-btn.primary {
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
  }

  .footer-btn.primary:hover {
    background: var(--yorha-accent-warm);
    transform: scale(1.05);
  }

  .footer-btn.secondary {
    background: transparent;
    color: var(--yorha-accent-cool);
    border: 2px solid var(--yorha-accent-cool);
  }

  .footer-btn.secondary:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  .footer-info {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
    font-size: 0.8rem;
  }

  .system-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .status-label {
    color: var(--yorha-muted);
  }

  .status-value.active {
    color: var(--yorha-success);
    font-weight: 600;
  }

  .timestamp {
    color: var(--yorha-muted);
  }

  /* === ANIMATIONS === */
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(50px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes loading {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(0%); }
    100% { transform: translateX(100%); }
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.1); }
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  /* === RESPONSIVE DESIGN === */
  @media (max-width: 1200px) {
    .evidence-content {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 768px) {
    .evidence-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .header-left {
      flex-direction: column;
      gap: 1rem;
    }

    .header-stats {
      gap: 1rem;
    }

    .file-card {
      grid-template-columns: auto 1fr;
      grid-template-rows: auto auto;
      gap: 0.5rem;
    }

    .file-actions {
      grid-column: 1 / -1;
      justify-content: center;
    }

    .footer-actions {
      flex-direction: column;
      gap: 0.5rem;
    }
  }
</style>