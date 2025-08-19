<!--
  Legal Document Processing Interface
  Advanced document analysis with AI-powered legal insights
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  
  // Component state
  let uploadedFiles: File[] = [];
  let processingFiles: Map<string, { progress: number; status: string }> = new Map();
  let analyzedDocuments: any[] = [];
  let searchQuery = '';
  let filterType = 'all';
  let sortBy = 'date';

  // Document types
  const documentTypes = [
    { value: 'all', label: 'All Documents', icon: '📄' },
    { value: 'contract', label: 'Contracts', icon: '📝' },
    { value: 'statute', label: 'Statutes', icon: '⚖️' },
    { value: 'precedent', label: 'Precedents', icon: '📚' },
    { value: 'evidence', label: 'Evidence', icon: '🔍' },
    { value: 'correspondence', label: 'Correspondence', icon: '✉️' }
  ];

  // Mock processed documents
  let mockDocuments = [
    {
      id: '1',
      name: 'Service Agreement.pdf',
      type: 'contract',
      uploadDate: new Date('2024-01-15'),
      size: '2.4 MB',
      status: 'processed',
      analysis: {
        keyTerms: ['liability', 'termination', 'intellectual property'],
        entities: ['ABC Corp', 'John Smith', 'California'],
        riskLevel: 'medium',
        compliance: 95,
        summary: 'Standard service agreement with standard liability clauses'
      }
    },
    {
      id: '2',
      name: 'Case Law Research.docx',
      type: 'precedent',
      uploadDate: new Date('2024-01-20'),
      size: '1.8 MB',
      status: 'processed',
      analysis: {
        keyTerms: ['due process', 'constitutional', 'judicial review'],
        entities: ['Supreme Court', 'District Court', 'United States'],
        riskLevel: 'low',
        compliance: 98,
        summary: 'Comprehensive case law analysis with strong precedential value'
      }
    }
  ];

  // File upload handling
  function handleFileUpload(event: Event) {
    const target = event.target as HTMLInputElement;
    const files = target.files;
    
    if (files) {
      const newFiles = Array.from(files);
      uploadedFiles = [...uploadedFiles, ...newFiles];
      
      // Start processing simulation for each file
      newFiles.forEach(file => {
        processDocument(file);
      });
    }
  }

  // Drag and drop handling
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    const files = event.dataTransfer?.files;
    
    if (files) {
      const newFiles = Array.from(files);
      uploadedFiles = [...uploadedFiles, ...newFiles];
      
      newFiles.forEach(file => {
        processDocument(file);
      });
    }
  }

  // Process document simulation
  async function processDocument(file: File) {
    const fileId = `${file.name}-${Date.now()}`;
    
    // Initialize processing state
    processingFiles.set(fileId, { progress: 0, status: 'analyzing' });
    processingFiles = processingFiles;

    // Simulate processing steps
    const steps = [
      { progress: 20, status: 'extracting text' },
      { progress: 40, status: 'analyzing content' },
      { progress: 60, status: 'identifying entities' },
      { progress: 80, status: 'assessing risks' },
      { progress: 100, status: 'complete' }
    ];

    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 800));
      processingFiles.set(fileId, step);
      processingFiles = processingFiles;
    }

    // Add to analyzed documents
    const mockAnalysis = {
      id: fileId,
      name: file.name,
      type: getDocumentType(file.name),
      uploadDate: new Date(),
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      status: 'processed',
      analysis: {
        keyTerms: generateMockKeyTerms(),
        entities: generateMockEntities(),
        riskLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        compliance: Math.floor(Math.random() * 20) + 80,
        summary: generateMockSummary(file.name)
      }
    };

    analyzedDocuments = [mockAnalysis, ...analyzedDocuments];
    processingFiles.delete(fileId);
    processingFiles = processingFiles;
  }

  function getDocumentType(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase();
    const name = filename.toLowerCase();
    
    if (name.includes('contract') || name.includes('agreement')) return 'contract';
    if (name.includes('statute') || name.includes('law')) return 'statute';
    if (name.includes('case') || name.includes('precedent')) return 'precedent';
    if (name.includes('evidence') || name.includes('exhibit')) return 'evidence';
    if (name.includes('letter') || name.includes('correspondence')) return 'correspondence';
    
    return 'contract'; // default
  }

  function generateMockKeyTerms(): string[] {
    const terms = [
      'liability', 'indemnification', 'termination', 'breach', 'damages',
      'intellectual property', 'confidential', 'force majeure', 'governing law',
      'jurisdiction', 'arbitration', 'warranty', 'disclaimer', 'assignment'
    ];
    return terms.sort(() => 0.5 - Math.random()).slice(0, 3 + Math.floor(Math.random() * 3));
  }

  function generateMockEntities(): string[] {
    const entities = [
      'ABC Corporation', 'XYZ Inc', 'John Smith', 'Jane Doe',
      'California', 'New York', 'Delaware', 'Supreme Court',
      'District Court', 'Securities Commission'
    ];
    return entities.sort(() => 0.5 - Math.random()).slice(0, 2 + Math.floor(Math.random() * 3));
  }

  function generateMockSummary(filename: string): string {
    const summaries = [
      `Analysis of ${filename} reveals standard legal provisions with moderate complexity`,
      `Document review shows compliance with regulatory requirements and industry standards`,
      `Legal assessment indicates potential areas requiring attention and review`,
      `Comprehensive analysis reveals key legal considerations and risk factors`
    ];
    return summaries[Math.floor(Math.random() * summaries.length)];
  }

  // Filtering and sorting
  $: filteredDocuments = [...mockDocuments, ...analyzedDocuments]
    .filter(doc => {
      const matchesSearch = !searchQuery || 
        doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.analysis.summary.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesFilter = filterType === 'all' || doc.type === filterType;
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'type':
          return a.type.localeCompare(b.type);
        case 'date':
        default:
          return b.uploadDate.getTime() - a.uploadDate.getTime();
      }
    });

  function downloadDocument(doc: any) {
    // Mock download functionality
    console.log('Downloading document:', doc.name);
  }

  function viewDocument(doc: any) {
    // Mock view functionality
    console.log('Viewing document:', doc.name);
  }

  function deleteDocument(docId: string) {
    analyzedDocuments = analyzedDocuments.filter(doc => doc.id !== docId);
    mockDocuments = mockDocuments.filter(doc => doc.id !== docId);
  }
</script>

<div class="documents-container">
  <div class="documents-header">
    <h1 class="page-title">Legal Document Processing</h1>
    <p class="page-subtitle">AI-powered document analysis and legal insights</p>
  </div>

  <!-- Upload Section -->
  <div class="upload-section">
    <div class="upload-area" 
         on:dragover={handleDragOver} 
         on:drop={handleDrop}
         role="button" 
         tabindex="0"
    >
      <div class="upload-icon">📁</div>
      <div class="upload-text">
        <h3>Upload Legal Documents</h3>
        <p>Drag & drop files here, or click to browse</p>
        <p class="upload-formats">Supports PDF, DOC, DOCX, TXT</p>
      </div>
      <input 
        type="file" 
        multiple 
        accept=".pdf,.doc,.docx,.txt" 
        onchange={handleFileUpload}
        class="upload-input"
      />
    </div>

    <!-- Processing Status -->
    {#if processingFiles.size > 0}
      <div class="processing-section">
        <h3>Processing Documents</h3>
        {#each Array.from(processingFiles.entries()) as [fileId, processing]}
          <div class="processing-item">
            <div class="processing-info">
              <span class="processing-name">{fileId.split('-')[0]}</span>
              <span class="processing-status">{processing.status}</span>
            </div>
            <div class="processing-progress">
              <div class="progress-bar">
                <div class="progress-fill" style="width: {processing.progress}%"></div>
              </div>
              <span class="progress-percent">{processing.progress}%</span>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Filters and Search -->
  <div class="filters-section">
    <div class="search-group">
      <input 
        type="text" 
        placeholder="Search documents..." 
        bind:value={searchQuery}
        class="search-input"
      />
    </div>

    <div class="filter-group">
      <select bind:value={filterType} class="filter-select">
        {#each documentTypes as type}
          <option value={type.value}>
            {type.icon} {type.label}
          </option>
        {/each}
      </select>
    </div>

    <div class="sort-group">
      <select bind:value={sortBy} class="sort-select">
        <option value="date">Sort by Date</option>
        <option value="name">Sort by Name</option>
        <option value="type">Sort by Type</option>
      </select>
    </div>
  </div>

  <!-- Documents Grid -->
  <div class="documents-grid">
    {#each filteredDocuments as doc}
      <div class="document-card">
        <div class="document-header">
          <div class="document-info">
            <h3 class="document-name">{doc.name}</h3>
            <div class="document-meta">
              <span class="document-type">
                {documentTypes.find(t => t.value === doc.type)?.icon || '📄'} 
                {documentTypes.find(t => t.value === doc.type)?.label || doc.type}
              </span>
              <span class="document-size">{doc.size}</span>
              <span class="document-date">{doc.uploadDate.toLocaleDateString()}</span>
            </div>
          </div>
          
          <div class="document-actions">
            <button class="action-btn" onclick={() => viewDocument(doc)} title="View">
              👁️
            </button>
            <button class="action-btn" onclick={() => downloadDocument(doc)} title="Download">
              📥
            </button>
            <button class="action-btn danger" onclick={() => deleteDocument(doc.id)} title="Delete">
              🗑️
            </button>
          </div>
        </div>

        <div class="document-analysis">
          <div class="analysis-section">
            <h4>Key Terms</h4>
            <div class="key-terms">
              {#each doc.analysis.keyTerms as term}
                <span class="key-term">{term}</span>
              {/each}
            </div>
          </div>

          <div class="analysis-section">
            <h4>Entities</h4>
            <div class="entities">
              {#each doc.analysis.entities as entity}
                <span class="entity">{entity}</span>
              {/each}
            </div>
          </div>

          <div class="analysis-section">
            <h4>Risk Assessment</h4>
            <div class="risk-info">
              <span class="risk-level risk-{doc.analysis.riskLevel}">
                {doc.analysis.riskLevel.toUpperCase()}
              </span>
              <span class="compliance">
                Compliance: {doc.analysis.compliance}%
              </span>
            </div>
          </div>

          <div class="analysis-section full-width">
            <h4>Summary</h4>
            <p class="summary">{doc.analysis.summary}</p>
          </div>
        </div>
      </div>
    {/each}

    {#if filteredDocuments.length === 0}
      <div class="empty-state">
        <div class="empty-icon">📄</div>
        <h3>No documents found</h3>
        <p>Upload some documents or adjust your search filters</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .documents-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
  }

  .documents-header {
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

  .upload-section {
    margin-bottom: 3rem;
  }

  .upload-area {
    border: 2px dashed var(--yorha-border, #333);
    border-radius: 8px;
    padding: 3rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    background: var(--yorha-bg-secondary, #1a1a1a);
  }

  .upload-area:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    background: rgba(255, 191, 0, 0.05);
  }

  .upload-input {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
  }

  .upload-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .upload-text h3 {
    color: var(--yorha-text-primary, #ffffff);
    margin-bottom: 0.5rem;
    font-size: 1.2rem;
  }

  .upload-text p {
    color: var(--yorha-text-secondary, #a0a0a0);
    margin: 0.25rem 0;
  }

  .upload-formats {
    font-size: 0.9rem;
    font-style: italic;
  }

  .processing-section {
    margin-top: 2rem;
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 1.5rem;
  }

  .processing-section h3 {
    color: var(--yorha-text-primary, #ffffff);
    margin-bottom: 1rem;
  }

  .processing-item {
    margin-bottom: 1rem;
  }

  .processing-info {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }

  .processing-name {
    color: var(--yorha-text-primary, #ffffff);
    font-weight: 600;
  }

  .processing-status {
    color: var(--yorha-accent-warm, #ffbf00);
    font-size: 0.9rem;
    text-transform: capitalize;
  }

  .processing-progress {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .progress-bar {
    flex: 1;
    height: 6px;
    background: var(--yorha-bg-primary, #0a0a0a);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-accent-warm, #ffbf00), var(--yorha-success, #00ff41));
    transition: width 0.3s ease;
  }

  .progress-percent {
    color: var(--yorha-text-secondary, #a0a0a0);
    font-size: 0.8rem;
    min-width: 35px;
  }

  .filters-section {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .search-input,
  .filter-select,
  .sort-select {
    padding: 0.75rem;
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 4px;
    color: var(--yorha-text-primary, #ffffff);
    font-family: inherit;
  }

  .search-input:focus,
  .filter-select:focus,
  .sort-select:focus {
    outline: none;
    border-color: var(--yorha-accent-warm, #ffbf00);
  }

  .documents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 2rem;
  }

  .document-card {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 1.5rem;
    transition: all 0.2s ease;
  }

  .document-card:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    box-shadow: 0 4px 8px rgba(255, 191, 0, 0.1);
  }

  .document-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--yorha-border, #333);
  }

  .document-name {
    color: var(--yorha-text-primary, #ffffff);
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
    font-weight: 600;
  }

  .document-meta {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .document-actions {
    display: flex;
    gap: 0.5rem;
  }

  .action-btn {
    background: transparent;
    border: 1px solid var(--yorha-border, #333);
    padding: 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.9rem;
  }

  .action-btn:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    background: rgba(255, 191, 0, 0.1);
  }

  .action-btn.danger:hover {
    border-color: var(--yorha-danger, #ff4757);
    background: rgba(255, 71, 87, 0.1);
  }

  .document-analysis {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .analysis-section {
    margin-bottom: 1rem;
  }

  .analysis-section.full-width {
    grid-column: 1 / -1;
  }

  .analysis-section h4 {
    color: var(--yorha-text-primary, #ffffff);
    margin: 0 0 0.5rem 0;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .key-terms,
  .entities {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }

  .key-term,
  .entity {
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.7rem;
    color: var(--yorha-text-primary, #ffffff);
  }

  .risk-info {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .risk-level {
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    text-align: center;
  }

  .risk-low {
    background: rgba(0, 255, 65, 0.2);
    color: var(--yorha-success, #00ff41);
    border: 1px solid var(--yorha-success, #00ff41);
  }

  .risk-medium {
    background: rgba(255, 191, 0, 0.2);
    color: var(--yorha-accent-warm, #ffbf00);
    border: 1px solid var(--yorha-accent-warm, #ffbf00);
  }

  .risk-high {
    background: rgba(255, 71, 87, 0.2);
    color: var(--yorha-danger, #ff4757);
    border: 1px solid var(--yorha-danger, #ff4757);
  }

  .compliance {
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .summary {
    color: var(--yorha-text-secondary, #a0a0a0);
    font-size: 0.8rem;
    line-height: 1.4;
    margin: 0;
  }

  .empty-state {
    grid-column: 1 / -1;
    text-align: center;
    padding: 4rem 2rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .empty-state h3 {
    margin-bottom: 0.5rem;
  }

  @media (max-width: 768px) {
    .documents-container {
      padding: 1rem;
    }

    .filters-section {
      grid-template-columns: 1fr;
    }

    .documents-grid {
      grid-template-columns: 1fr;
    }

    .document-analysis {
      grid-template-columns: 1fr;
    }

    .page-title {
      font-size: 2rem;
    }
  }
</style>