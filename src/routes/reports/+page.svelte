<!--
  Report Generation Interface
  AI-powered report creation with legal document analysis
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  
  // Component state
  let reportType = 'case-summary';
  let selectedDocuments: string[] = [];
  let generatedReport = '';
  let isGenerating = false;
  let reportProgress = 0;

  // Report types
  const reportTypes = [
    { value: 'case-summary', label: 'Case Summary', icon: '📋' },
    { value: 'evidence-analysis', label: 'Evidence Analysis', icon: '🔍' },
    { value: 'legal-research', label: 'Legal Research', icon: '📚' },
    { value: 'compliance-audit', label: 'Compliance Audit', icon: '✅' },
    { value: 'risk-assessment', label: 'Risk Assessment', icon: '⚠️' }
  ];

  // Mock documents
  const availableDocuments = [
    { id: '1', name: 'Contract Agreement.pdf', type: 'contract', size: '2.4 MB' },
    { id: '2', name: 'Evidence Photo 1.jpg', type: 'evidence', size: '5.2 MB' },
    { id: '3', name: 'Witness Statement.docx', type: 'statement', size: '1.1 MB' },
    { id: '4', name: 'Legal Precedent.pdf', type: 'precedent', size: '800 KB' },
    { id: '5', name: 'Case Notes.txt', type: 'notes', size: '45 KB' }
  ];

  // Generate report function
  async function generateReport() {
    if (selectedDocuments.length === 0) {
      alert('Please select at least one document');
      return;
    }

    isGenerating = true;
    reportProgress = 0;
    generatedReport = '';

    // Simulate report generation progress
    const progressInterval = setInterval(() => {
      reportProgress += Math.random() * 10;
      if (reportProgress >= 100) {
        reportProgress = 100;
        clearInterval(progressInterval);
        
        // Mock generated report
        generatedReport = `
# ${reportTypes.find(rt => rt.value === reportType)?.label || 'Report'}

## Executive Summary
This AI-generated report analyzes ${selectedDocuments.length} document(s) to provide insights and recommendations based on the selected criteria.

## Key Findings
1. **Document Analysis**: All ${selectedDocuments.length} documents have been processed using advanced AI models
2. **Compliance Status**: Documents appear to meet standard legal requirements
3. **Risk Assessment**: Low to moderate risk levels identified
4. **Recommendations**: Further review recommended for critical sections

## Detailed Analysis
${selectedDocuments.map((docId, index) => {
  const doc = availableDocuments.find(d => d.id === docId);
  return `### Document ${index + 1}: ${doc?.name}
- Type: ${doc?.type}
- Size: ${doc?.size}
- Status: Analyzed ✅
- Key findings from this document...`;
}).join('\n\n')}

## Conclusions
Based on the comprehensive analysis of the selected documents, this report provides actionable insights for legal decision-making. The AI analysis indicates strong compliance with regulatory requirements.

---
*Report generated on ${new Date().toLocaleDateString()} using YoRHa Legal AI Platform*
        `.trim();
        
        isGenerating = false;
      }
    }, 200);
  }

  function toggleDocument(docId: string) {
    if (selectedDocuments.includes(docId)) {
      selectedDocuments = selectedDocuments.filter(id => id !== docId);
    } else {
      selectedDocuments = [...selectedDocuments, docId];
    }
  }

  function downloadReport() {
    if (!generatedReport) return;
    
    const blob = new Blob([generatedReport], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${reportType}-report-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="reports-container">
  <div class="reports-header">
    <h1 class="page-title">Report Generation</h1>
    <p class="page-subtitle">AI-powered legal document analysis and reporting</p>
  </div>

  <div class="reports-content">
    <!-- Report Configuration -->
    <div class="config-section">
      <div class="section-header">
        <h2>Report Configuration</h2>
        <span class="section-icon">⚙️</span>
      </div>

      <div class="config-grid">
        <div class="config-group">
          <label for="report-type">Report Type</label>
          <select id="report-type" bind:value={reportType} class="config-select">
            {#each reportTypes as type}
              <option value={type.value}>
                {type.icon} {type.label}
              </option>
            {/each}
          </select>
        </div>

        <div class="config-group full-width">
          <label>Select Documents</label>
          <div class="document-list">
            {#each availableDocuments as doc}
              <div 
                class="document-item" 
                class:selected={selectedDocuments.includes(doc.id)}
                on:click={() => toggleDocument(doc.id)}
                role="button"
                tabindex="0"
                on:keydown={(e) => e.key === 'Enter' && toggleDocument(doc.id)}
              >
                <div class="document-info">
                  <span class="document-name">{doc.name}</span>
                  <span class="document-meta">{doc.type} • {doc.size}</span>
                </div>
                <div class="document-checkbox">
                  {#if selectedDocuments.includes(doc.id)}
                    <span class="check-icon">✓</span>
                  {:else}
                    <span class="check-icon">○</span>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>

      <div class="config-actions">
        <button 
          class="generate-btn" 
          on:click={generateReport}
          disabled={isGenerating || selectedDocuments.length === 0}
        >
          {#if isGenerating}
            <span class="btn-icon">🔄</span>
            Generating...
          {:else}
            <span class="btn-icon">📊</span>
            Generate Report
          {/if}
        </button>
      </div>
    </div>

    <!-- Progress -->
    {#if isGenerating}
      <div class="progress-section">
        <div class="progress-header">
          <h3>Processing Documents</h3>
          <span class="progress-percent">{Math.round(reportProgress)}%</span>
        </div>
        <div class="progress-bar">
          <div 
            class="progress-fill" 
            style="width: {reportProgress}%"
          ></div>
        </div>
        <div class="progress-status">
          Analyzing {selectedDocuments.length} document(s) with AI models...
        </div>
      </div>
    {/if}

    <!-- Generated Report -->
    {#if generatedReport && !isGenerating}
      <div class="report-section">
        <div class="section-header">
          <h2>Generated Report</h2>
          <div class="report-actions">
            <button class="action-btn" on:click={downloadReport}>
              <span class="btn-icon">📥</span>
              Download
            </button>
          </div>
        </div>

        <div class="report-content">
          <pre class="report-text">{generatedReport}</pre>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .reports-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
  }

  .reports-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .page-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--yorha-accent-warm, #ffbf00);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .page-subtitle {
    font-size: 1.1rem;
    color: var(--yorha-text-secondary, #a0a0a0);
    margin: 0;
  }

  .config-section,
  .progress-section,
  .report-section {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 2px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--yorha-border, #333);
  }

  .section-header h2 {
    color: var(--yorha-text-primary, #ffffff);
    margin: 0;
    font-size: 1.5rem;
  }

  .section-icon {
    font-size: 1.5rem;
  }

  .config-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
  }

  .config-group.full-width {
    grid-column: 1 / -1;
  }

  .config-group label {
    display: block;
    font-weight: 600;
    color: var(--yorha-text-primary, #ffffff);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    font-size: 0.9rem;
    letter-spacing: 1px;
  }

  .config-select {
    width: 100%;
    padding: 0.75rem;
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 4px;
    color: var(--yorha-text-primary, #ffffff);
    font-family: inherit;
  }

  .config-select:focus {
    outline: none;
    border-color: var(--yorha-accent-warm, #ffbf00);
  }

  .document-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .document-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .document-item:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    background: rgba(255, 191, 0, 0.05);
  }

  .document-item.selected {
    border-color: var(--yorha-success, #00ff41);
    background: rgba(0, 255, 65, 0.1);
  }

  .document-info {
    display: flex;
    flex-direction: column;
  }

  .document-name {
    font-weight: 600;
    color: var(--yorha-text-primary, #ffffff);
    margin-bottom: 0.25rem;
  }

  .document-meta {
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
    text-transform: uppercase;
  }

  .document-checkbox {
    display: flex;
    align-items: center;
  }

  .check-icon {
    font-size: 1.2rem;
    color: var(--yorha-success, #00ff41);
  }

  .config-actions {
    text-align: center;
  }

  .generate-btn {
    background: linear-gradient(45deg, var(--yorha-accent-warm, #ffbf00), var(--yorha-success, #00ff41));
    border: none;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: inherit;
    color: var(--yorha-bg-primary, #0a0a0a);
  }

  .generate-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .generate-btn:not(:disabled):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(255, 191, 0, 0.3);
  }

  .btn-icon {
    margin-right: 0.5rem;
  }

  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .progress-header h3 {
    color: var(--yorha-text-primary, #ffffff);
    margin: 0;
  }

  .progress-percent {
    color: var(--yorha-accent-warm, #ffbf00);
    font-weight: 600;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background: var(--yorha-bg-primary, #0a0a0a);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-accent-warm, #ffbf00), var(--yorha-success, #00ff41));
    transition: width 0.3s ease;
  }

  .progress-status {
    color: var(--yorha-text-secondary, #a0a0a0);
    font-size: 0.9rem;
  }

  .report-actions {
    display: flex;
    gap: 1rem;
  }

  .action-btn {
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    padding: 0.5rem 1rem;
    color: var(--yorha-text-primary, #ffffff);
    font-family: inherit;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    background: rgba(255, 191, 0, 0.05);
  }

  .report-content {
    max-height: 600px;
    overflow-y: auto;
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 4px;
  }

  .report-text {
    padding: 2rem;
    margin: 0;
    font-family: inherit;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--yorha-text-primary, #ffffff);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  @media (max-width: 768px) {
    .reports-container {
      padding: 1rem;
    }

    .config-grid {
      grid-template-columns: 1fr;
      gap: 1rem;
    }

    .page-title {
      font-size: 2rem;
    }
  }
</style>