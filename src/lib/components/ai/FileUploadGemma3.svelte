<!--
  File Upload with LLVM-Quality WebAssembly Gemma3 Integration
  Complete file processing pipeline with PostgreSQL and pgvector storage
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { gemma3UploadPgVectorService } from '$lib/services/gemma3-upload-pgvector';

  // Component props
  export let userId: string;
  export let caseId: string = '';
  export let maxFileSize: number = 50 * 1024 * 1024; // 50MB default
  export let allowedTypes: string[] = ['text/*', 'application/pdf', 'image/*', '.docx', '.doc'];
  export let enableBatchProcessing: boolean = true;
  export let autoProcess: boolean = false;

  // Component state
  const uploadState = writable({
    dragActive: false,
    processing: false,
    uploadedFiles: [] as File[],
    processedResults: [] as any[],
    currentProgress: {
      current: 0,
      total: 0,
      fileName: ''
    },
    error: null as string | null,
    searchQuery: '',
    searchResults: [] as any[]
  });

  // Processing options
  let processingOptions = {
    extractText: true,
    generateEmbeddings: true,
    performLegalAnalysis: true,
    storeInDatabase: true,
    analysisType: 'comprehensive' as 'comprehensive' | 'quick' | 'risk-focused' | 'legal-precedent',
    enableImageAnalysis: true,
    useWebAssembly: true
  };

  let fileInput: HTMLInputElement;
  let dropZone: HTMLDivElement;
  let processingStats: any = null;

  // Component lifecycle
  onMount(async () => {
    console.log('🚀 FileUpload Gemma3 component mounted');
    
    // Set up drag and drop handlers
    setupDragAndDrop();
    
    // Load initial processing stats
    await loadProcessingStats();
  });

  // Drag and drop setup
  function setupDragAndDrop() {
    if (!dropZone) return;

    dropZone.addEventListener('dragenter', handleDragEnter);
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
  }

  function handleDragEnter(e: DragEvent) {
    e.preventDefault();
    uploadState.update(state => ({ ...state, dragActive: true }));
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    // Only deactivate if leaving the drop zone entirely
    if (!dropZone.contains(e.relatedTarget as Node)) {
      uploadState.update(state => ({ ...state, dragActive: false }));
    }
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    uploadState.update(state => ({ ...state, dragActive: false }));
    
    const files = Array.from(e.dataTransfer?.files || []);
    handleFiles(files);
  }

  // File handling
  function handleFileInput(e: Event) {
    const target = e.target as HTMLInputElement;
    const files = Array.from(target.files || []);
    handleFiles(files);
  }

  function handleFiles(files: File[]) {
    // Validate files
    const validFiles = files.filter(file => validateFile(file));
    
    if (validFiles.length === 0) {
      uploadState.update(state => ({
        ...state,
        error: 'No valid files selected'
      }));
      return;
    }

    uploadState.update(state => ({
      ...state,
      uploadedFiles: [...state.uploadedFiles, ...validFiles],
      error: null
    }));

    console.log(`📁 Added ${validFiles.length} files for processing`);

    // Auto-process if enabled
    if (autoProcess) {
      processFiles();
    }
  }

  function validateFile(file: File): boolean {
    // Check file size
    if (file.size > maxFileSize) {
      console.warn(`❌ File ${file.name} exceeds size limit (${maxFileSize} bytes)`);
      return false;
    }

    // Check file type
    const isAllowed = allowedTypes.some(type => {
      if (type.startsWith('.')) {
        return file.name.toLowerCase().endsWith(type.toLowerCase());
      } else if (type.endsWith('/*')) {
        return file.type.startsWith(type.slice(0, -1));
      } else {
        return file.type === type;
      }
    });

    if (!isAllowed) {
      console.warn(`❌ File ${file.name} has unsupported type (${file.type})`);
      return false;
    }

    return true;
  }

  // File processing
  async function processFiles() {
    const files = $uploadState.uploadedFiles;
    if (files.length === 0) {
      uploadState.update(state => ({
        ...state,
        error: 'No files to process'
      }));
      return;
    }

    uploadState.update(state => ({
      ...state,
      processing: true,
      error: null,
      processedResults: [],
      currentProgress: { current: 0, total: files.length, fileName: '' }
    }));

    try {
      console.log(`🚀 Starting processing of ${files.length} files with LLVM WebAssembly`);

      if (enableBatchProcessing && files.length > 1) {
        // Batch processing
        const results = await gemma3UploadPgVectorService.processBatchUpload(
          files,
          {
            ...processingOptions,
            userId,
            caseId: caseId || undefined
          },
          3 // Max concurrency
        );

        uploadState.update(state => ({
          ...state,
          processing: false,
          processedResults: results,
          uploadedFiles: [] // Clear after processing
        }));

        console.log(`✅ Batch processing completed: ${results.length} files processed`);

      } else {
        // Process files individually with progress updates
        const results = [];

        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          
          uploadState.update(state => ({
            ...state,
            currentProgress: {
              current: i + 1,
              total: files.length,
              fileName: file.name
            }
          }));

          console.log(`📄 Processing file ${i + 1}/${files.length}: ${file.name}`);

          try {
            const result = await gemma3UploadPgVectorService.processUploadedFile(
              file,
              {
                ...processingOptions,
                userId,
                caseId: caseId || undefined
              }
            );

            results.push(result);
            console.log(`✅ Processed ${file.name} using ${result.method}`);

          } catch (error) {
            console.error(`❌ Failed to process ${file.name}:`, error);
            results.push({
              originalName: file.name,
              error: error instanceof Error ? error.message : 'Processing failed',
              success: false
            });
          }
        }

        uploadState.update(state => ({
          ...state,
          processing: false,
          processedResults: results,
          uploadedFiles: [] // Clear after processing
        }));

        console.log(`✅ Individual processing completed: ${results.length} files processed`);
      }

      // Reload processing stats
      await loadProcessingStats();

    } catch (error) {
      console.error('❌ File processing failed:', error);
      uploadState.update(state => ({
        ...state,
        processing: false,
        error: `Processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    }
  }

  // Search functionality
  async function searchDocuments() {
    const query = $uploadState.searchQuery.trim();
    if (!query) {
      uploadState.update(state => ({
        ...state,
        error: 'Please enter a search query'
      }));
      return;
    }

    uploadState.update(state => ({ ...state, processing: true, error: null }));

    try {
      console.log(`🔍 Searching documents: "${query}"`);

      const results = await gemma3UploadPgVectorService.searchDocuments(query, {
        threshold: 0.6,
        maxResults: 10,
        userId,
        caseId: caseId || undefined,
        includeAnalysis: true
      });

      uploadState.update(state => ({
        ...state,
        processing: false,
        searchResults: results
      }));

      console.log(`✅ Search completed: ${results.length} results found`);

    } catch (error) {
      console.error('❌ Document search failed:', error);
      uploadState.update(state => ({
        ...state,
        processing: false,
        error: `Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    }
  }

  // Stats loading
  async function loadProcessingStats() {
    try {
      processingStats = gemma3UploadPgVectorService.getProcessingStats();
      console.log('📊 Processing stats loaded:', processingStats);
    } catch (error) {
      console.error('❌ Failed to load processing stats:', error);
    }
  }

  // Utility functions
  function removeFile(index: number) {
    uploadState.update(state => ({
      ...state,
      uploadedFiles: state.uploadedFiles.filter((_, i) => i !== index)
    }));
  }

  function clearResults() {
    uploadState.update(state => ({
      ...state,
      processedResults: [],
      searchResults: [],
      error: null
    }));
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatProcessingTime(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }
</script>

<div class="file-upload-gemma3">
  <!-- Header -->
  <div class="upload-header">
    <h3>📁 LLVM-Quality File Processing with pgvector</h3>
    <p>Upload documents for AI analysis and vector storage</p>
  </div>

  <!-- Processing Options -->
  <div class="processing-options">
    <h4>🔧 Processing Options</h4>
    <div class="options-grid">
      <label class="option-item">
        <input type="checkbox" bind:checked={processingOptions.extractText} />
        📄 Extract Text
      </label>
      <label class="option-item">
        <input type="checkbox" bind:checked={processingOptions.generateEmbeddings} />
        🧠 Generate Embeddings
      </label>
      <label class="option-item">
        <input type="checkbox" bind:checked={processingOptions.performLegalAnalysis} />
        ⚖️ Legal AI Analysis
      </label>
      <label class="option-item">
        <input type="checkbox" bind:checked={processingOptions.storeInDatabase} />
        💾 Store in Database
      </label>
      <label class="option-item">
        <input type="checkbox" bind:checked={processingOptions.enableImageAnalysis} />
        🖼️ Image Analysis
      </label>
      <label class="option-item">
        <input type="checkbox" bind:checked={processingOptions.useWebAssembly} />
        🚀 Use WebAssembly
      </label>
    </div>

    <div class="analysis-type">
      <label for="analysis-type">Analysis Type:</label>
      <select id="analysis-type" bind:value={processingOptions.analysisType}>
        <option value="comprehensive">Comprehensive</option>
        <option value="quick">Quick</option>
        <option value="risk-focused">Risk Focused</option>
        <option value="legal-precedent">Legal Precedent</option>
      </select>
    </div>
  </div>

  <!-- File Upload Zone -->
  <div 
    class="drop-zone"
    class:drag-active={$uploadState.dragActive}
    bind:this={dropZone}
  >
    <div class="drop-zone-content">
      <div class="upload-icon">📁</div>
      <h4>Drop files here or click to upload</h4>
      <p>Supports: PDF, Text, Images, Word documents</p>
      <p>Max size: {formatFileSize(maxFileSize)}</p>
      
      <button class="upload-button" on:click={() => fileInput.click()}>
        Choose Files
      </button>
    </div>

    <input
      type="file"
      bind:this={fileInput}
      on:change={handleFileInput}
      multiple={enableBatchProcessing}
      accept={allowedTypes.join(',')}
      style="display: none"
    />
  </div>

  <!-- Uploaded Files List -->
  {#if $uploadState.uploadedFiles.length > 0}
    <div class="files-list">
      <h4>📋 Files Ready for Processing ({$uploadState.uploadedFiles.length})</h4>
      
      {#each $uploadState.uploadedFiles as file, index}
        <div class="file-item">
          <div class="file-info">
            <div class="file-name">{file.name}</div>
            <div class="file-details">
              {file.type} • {formatFileSize(file.size)}
            </div>
          </div>
          <button class="remove-button" on:click={() => removeFile(index)}>
            ❌
          </button>
        </div>
      {/each}

      <div class="files-actions">
        <button 
          class="process-button"
          disabled={$uploadState.processing}
          on:click={processFiles}
        >
          {#if $uploadState.processing}
            <div class="spinner small"></div>
            Processing...
          {:else}
            🚀 Process Files with LLVM
          {/if}
        </button>
        
        <button class="clear-button" on:click={() => uploadState.update(s => ({ ...s, uploadedFiles: [] }))}>
          🗑️ Clear List
        </button>
      </div>
    </div>
  {/if}

  <!-- Processing Progress -->
  {#if $uploadState.processing && $uploadState.currentProgress.total > 0}
    <div class="progress-panel">
      <h4>⚡ Processing Files with LLVM WebAssembly</h4>
      
      <div class="progress-bar">
        <div 
          class="progress-fill" 
          style="width: {($uploadState.currentProgress.current / $uploadState.currentProgress.total) * 100}%"
        ></div>
      </div>
      
      <div class="progress-info">
        <span>
          {$uploadState.currentProgress.current} / {$uploadState.currentProgress.total}
        </span>
        <span class="current-file">
          {$uploadState.currentProgress.fileName}
        </span>
      </div>
    </div>
  {/if}

  <!-- Processing Results -->
  {#if $uploadState.processedResults.length > 0}
    <div class="results-panel">
      <div class="results-header">
        <h4>📊 Processing Results</h4>
        <button class="clear-button" on:click={clearResults}>
          🗑️ Clear Results
        </button>
      </div>

      <div class="results-summary">
        <div class="summary-stat">
          <span class="stat-value">{$uploadState.processedResults.filter(r => !r.error).length}</span>
          <span class="stat-label">Successful</span>
        </div>
        <div class="summary-stat">
          <span class="stat-value">{$uploadState.processedResults.filter(r => r.error).length}</span>
          <span class="stat-label">Failed</span>
        </div>
        <div class="summary-stat">
          <span class="stat-value">
            {$uploadState.processedResults.filter(r => r.method === 'webassembly').length}
          </span>
          <span class="stat-label">WebAssembly</span>
        </div>
      </div>

      <div class="results-list">
        {#each $uploadState.processedResults as result}
          <div class="result-item" class:error={result.error}>
            <div class="result-header">
              <div class="result-title">{result.originalName}</div>
              <div class="result-badges">
                {#if result.method}
                  <span class="method-badge method-{result.method}">{result.method.toUpperCase()}</span>
                {/if}
                {#if result.processingTime}
                  <span class="time-badge">{formatProcessingTime(result.processingTime)}</span>
                {/if}
                {#if result.databaseId}
                  <span class="db-badge">STORED</span>
                {/if}
              </div>
            </div>

            {#if result.error}
              <div class="result-error">❌ {result.error}</div>
            {:else}
              <div class="result-details">
                {#if result.extractedText}
                  <div class="detail-item">
                    📄 Text extracted: {result.extractedText.length} characters
                  </div>
                {/if}
                
                {#if result.embeddings}
                  <div class="detail-item">
                    🧠 Vector embedding: {result.embeddings.dimensions} dimensions ({result.embeddings.model})
                  </div>
                {/if}
                
                {#if result.analysis}
                  <div class="detail-item">
                    ⚖️ Analysis: {result.analysis.keyFindings?.length || 0} findings, 
                    {result.analysis.legalRisks?.length || 0} risks identified
                  </div>
                {/if}

                {#if result.imageAnalysis}
                  <div class="detail-item">
                    🖼️ Image analysis: {result.imageAnalysis.vertexCount} vertices, 
                    {result.imageAnalysis.detectedObjects?.length || 0} objects detected
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Document Search -->
  <div class="search-panel">
    <h4>🔍 Search Processed Documents</h4>
    
    <div class="search-input-group">
      <input
        type="text"
        bind:value={$uploadState.searchQuery}
        placeholder="Search by content, legal terms, or concepts..."
        class="search-input"
        on:keypress={(e) => e.key === 'Enter' && searchDocuments()}
      />
      <button 
        class="search-button"
        disabled={$uploadState.processing}
        on:click={searchDocuments}
      >
        {#if $uploadState.processing}
          <div class="spinner small"></div>
        {:else}
          🔍 Search
        {/if}
      </button>
    </div>

    {#if $uploadState.searchResults.length > 0}
      <div class="search-results">
        <h5>📋 Search Results ({$uploadState.searchResults.length})</h5>
        
        {#each $uploadState.searchResults as result}
          <div class="search-result-item">
            <div class="result-header">
              <span class="result-title">{result.title}</span>
              <span class="similarity-score">
                {(result.similarity * 100).toFixed(1)}% similar
              </span>
            </div>
            
            <div class="result-content">
              {result.content || 'No content preview available'}
            </div>
            
            <div class="result-metadata">
              <span class="metadata-item">📄 {result.fileName}</span>
              <span class="metadata-item">📊 {formatFileSize(result.fileSize)}</span>
              <span class="metadata-item">🧠 {result.embedding.model}</span>
              {#if result.caseId}
                <span class="metadata-item">📁 Case: {result.caseId}</span>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Processing Statistics -->
  {#if processingStats}
    <div class="stats-panel">
      <h4>📈 Processing Statistics</h4>
      
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value">{processingStats.filesProcessed}</div>
          <div class="stat-label">Files Processed</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{processingStats.averageProcessingTime.toFixed(0)}ms</div>
          <div class="stat-label">Avg Processing Time</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{processingStats.webassemblySuccessRate.toFixed(1)}%</div>
          <div class="stat-label">WebAssembly Success</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{processingStats.vectorEmbeddings}</div>
          <div class="stat-label">Vector Embeddings</div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Error Display -->
  {#if $uploadState.error}
    <div class="error-panel">
      <h4>❌ Error</h4>
      <p>{$uploadState.error}</p>
    </div>
  {/if}
</div>

<style>
  .file-upload-gemma3 {
    max-width: 1000px;
    margin: 0 auto;
    padding: 1.5rem;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    background: #1a1a1a;
    color: #e0e0e0;
    border-radius: 12px;
    border: 1px solid #333;
  }

  .upload-header {
    text-align: center;
    margin-bottom: 2rem;
  }

  .upload-header h3 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg, #00ff88, #0088ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .processing-options {
    margin-bottom: 2rem;
    padding: 1rem;
    background: #222;
    border-radius: 8px;
    border: 1px solid #333;
  }

  .options-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
  }

  .option-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }

  .option-item input[type="checkbox"] {
    margin-right: 0.5rem;
  }

  .analysis-type {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1rem;
  }

  .analysis-type select {
    padding: 0.5rem;
    background: #333;
    border: 1px solid #444;
    border-radius: 4px;
    color: #ccc;
  }

  .drop-zone {
    border: 2px dashed #444;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-bottom: 2rem;
    background: #1f1f1f;
  }

  .drop-zone.drag-active {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
    transform: scale(1.02);
  }

  .drop-zone:hover {
    border-color: #0088ff;
    background: #222;
  }

  .upload-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .upload-button {
    padding: 0.75rem 1.5rem;
    background: linear-gradient(45deg, #00ff88, #0088ff);
    border: none;
    border-radius: 8px;
    color: #000;
    font-weight: bold;
    cursor: pointer;
    margin-top: 1rem;
    transition: transform 0.2s;
  }

  .upload-button:hover {
    transform: translateY(-2px);
  }

  .files-list {
    background: #222;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 2rem;
  }

  .file-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: #2a2a2a;
    border-radius: 6px;
    margin-bottom: 0.5rem;
  }

  .file-info {
    flex: 1;
  }

  .file-name {
    font-weight: bold;
    color: #00ff88;
  }

  .file-details {
    font-size: 0.9rem;
    color: #888;
  }

  .remove-button {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1rem;
  }

  .files-actions {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
  }

  .process-button, .clear-button, .search-button {
    padding: 0.75rem 1rem;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: bold;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: all 0.2s;
  }

  .process-button {
    background: linear-gradient(45deg, #00ff88, #0088ff);
    color: #000;
  }

  .clear-button {
    background: #666;
    color: #fff;
  }

  .search-button {
    background: #0088ff;
    color: #fff;
  }

  .process-button:hover, .search-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
  }

  .process-button:disabled, .search-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid #333;
    border-top: 2px solid #00ff88;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .spinner.small {
    width: 12px;
    height: 12px;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .progress-panel {
    background: #222;
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background: #333;
    border-radius: 4px;
    overflow: hidden;
    margin: 1rem 0;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(45deg, #00ff88, #0088ff);
    transition: width 0.3s ease;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
  }

  .current-file {
    color: #00ff88;
  }

  .results-panel, .search-panel, .stats-panel {
    background: #222;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .results-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .summary-stat {
    text-align: center;
    padding: 1rem;
    background: #2a2a2a;
    border-radius: 6px;
  }

  .stat-value {
    display: block;
    font-size: 1.5rem;
    font-weight: bold;
    color: #00ff88;
  }

  .stat-label {
    font-size: 0.9rem;
    color: #888;
  }

  .result-item {
    background: #2a2a2a;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  .result-item.error {
    border-left: 4px solid #ff4444;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .result-title {
    font-weight: bold;
    color: #00ff88;
  }

  .result-badges {
    display: flex;
    gap: 0.5rem;
  }

  .method-badge, .time-badge, .db-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: bold;
  }

  .method-webassembly {
    background: #00ff88;
    color: #000;
  }

  .method-ollama {
    background: #ff8800;
    color: #000;
  }

  .time-badge {
    background: #333;
    color: #ccc;
  }

  .db-badge {
    background: #0088ff;
    color: #000;
  }

  .result-error {
    color: #ff4444;
    font-weight: bold;
  }

  .result-details {
    margin-top: 0.5rem;
  }

  .detail-item {
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
    color: #ccc;
  }

  .search-input-group {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .search-input {
    flex: 1;
    padding: 0.75rem;
    background: #333;
    border: 1px solid #444;
    border-radius: 6px;
    color: #ccc;
    font-size: 1rem;
  }

  .search-input:focus {
    outline: none;
    border-color: #00ff88;
  }

  .search-results {
    margin-top: 1rem;
  }

  .search-result-item {
    background: #2a2a2a;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  .similarity-score {
    background: #0088ff;
    color: #000;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: bold;
  }

  .result-content {
    margin: 0.5rem 0;
    color: #ccc;
  }

  .result-metadata {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
  }

  .metadata-item {
    background: #333;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
  }

  .stat-card {
    text-align: center;
    padding: 1.5rem;
    background: #2a2a2a;
    border-radius: 6px;
  }

  .stat-card .stat-value {
    font-size: 2rem;
    font-weight: bold;
    color: #00ff88;
  }

  .stat-card .stat-label {
    margin-top: 0.5rem;
    color: #888;
    font-size: 0.9rem;
  }

  .error-panel {
    background: rgba(255, 68, 68, 0.1);
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
  }

  .error-panel h4 {
    color: #ff4444;
    margin-bottom: 0.5rem;
  }
</style>