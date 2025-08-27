<!--
  LLVM-Quality WebAssembly Gemma3 Integration Demo
  Complete showcase of WASM inference, Loki caching, vertex analysis, and legal AI processing
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { gemma3LokiIntegration } from '$lib/services/gemma3-loki-integration';
  import { vertexBufferImageAnalyzer } from '$lib/services/vertex-buffer-image-analyzer';
  import { fuseLazySearch } from '$lib/services/fuse-lazy-search-indexeddb';
  import { lokiEvidenceService } from '$lib/utils/loki-evidence';

  // Demo state management
  const demoState = writable({
    initialized: false,
    activeTab: 'legal-analysis',
    processing: false,
    results: null as any,
    performance: {
      totalTime: 0,
      cacheHits: 0,
      webassemblyInferences: 0,
      vertexAnalyses: 0
    },
    error: null as string | null
  });

  // Sample legal documents for demo
  const sampleDocuments = [
    {
      title: "Contract Amendment - Technology Services",
      content: `This Amendment to the Technology Services Agreement ("Amendment") is made effective as of January 15, 2025, between TechCorp Solutions Inc., a Delaware corporation ("Provider"), and Legal AI Systems LLC, a California limited liability company ("Client").

WHEREAS, the parties entered into a Technology Services Agreement dated December 1, 2024 ("Agreement");

WHEREAS, the parties wish to modify certain terms of the Agreement;

NOW, THEREFORE, the parties agree as follows:

1. SCOPE OF SERVICES: The Provider shall deliver enhanced AI inference services using LLVM-quality WebAssembly technology, including but not limited to:
   a) Real-time legal document analysis
   b) Vector-based similarity search
   c) GPU-accelerated processing capabilities

2. PERFORMANCE STANDARDS: Provider warrants that the services shall achieve:
   a) Sub-50ms response times for standard queries
   b) 99.9% uptime availability
   c) GDPR and CCPA compliance for data processing

3. LIABILITY: Provider's aggregate liability shall not exceed the total fees paid in the preceding 12 months. Client acknowledges that AI predictions are probabilistic and should be verified by qualified legal counsel.

4. INDEMNIFICATION: Each party shall indemnify the other against third-party claims arising from breach of this Amendment, provided such breach is material and not cured within 30 days of written notice.

This Amendment supersedes any conflicting terms in the original Agreement. All other terms remain in full force and effect.`,
      caseId: "DEMO-001",
      analysisType: "comprehensive" as const
    },
    {
      title: "Evidence Analysis - Digital Forensics Report",
      content: `DIGITAL FORENSICS EXAMINATION REPORT

Case Number: 2025-CF-0142
Examiner: Dr. Sarah Chen, GCFA, EnCE
Date: January 20, 2025

EXECUTIVE SUMMARY:
This report details the forensic examination of digital evidence seized pursuant to Search Warrant SW-2025-0089. The examination revealed significant findings regarding unauthorized access to computer systems.

EVIDENCE EXAMINED:
1. Dell OptiPlex 7090 Desktop Computer (Serial: ABC123456)
2. Samsung SSD 970 EVO 1TB (Serial: DEF789012)
3. USB Flash Drive 64GB Sandisk (Serial: GHI345678)

KEY FINDINGS:
1. UNAUTHORIZED ACCESS: Log files indicate unauthorized remote access attempts on December 15-18, 2024, originating from IP addresses registered in Eastern Europe.

2. DATA EXFILTRATION: Forensic timeline analysis shows 2.3GB of sensitive data was copied to external storage on December 17, 2024, at 23:47 UTC.

3. ANTI-FORENSICS: Evidence of CCleaner and DBAN usage suggests attempts to destroy digital evidence on December 19, 2024.

4. MALWARE DETECTION: Custom keylogger identified in System32 directory with creation timestamp December 14, 2024, 14:32 UTC.

LEGAL IMPLICATIONS:
- Computer Fraud and Abuse Act violations (18 U.S.C. § 1030)
- Wire fraud potential under 18 U.S.C. § 1343
- Identity theft considerations under 18 U.S.C. § 1028

CHAIN OF CUSTODY maintained per FBI guidelines. All procedures followed NIST SP 800-86 standards.

Respectfully submitted,
Dr. Sarah Chen
Certified Digital Forensics Examiner`,
      caseId: "DEMO-002",
      analysisType: "risk-focused" as const
    }
  ];

  let selectedDocument = sampleDocuments[0];
  let customContent = '';
  let searchQuery = '';
  let searchResults: any[] = [];
  let performanceStats: any = null;

  // Component lifecycle
  onMount(async () => {
    try {
      console.log('🚀 Initializing LLVM-Quality WebAssembly Gemma3 Demo');
      
      // Initialize all services
      await Promise.all([
        gemma3LokiIntegration.initialize?.() || Promise.resolve(),
        vertexBufferImageAnalyzer.initialize(),
        fuseLazySearch.initialize(),
        // lokiEvidenceService is initialized automatically
      ]);

      demoState.update(state => ({
        ...state,
        initialized: true
      }));

      console.log('✅ LLVM Gemma3 Demo initialized successfully');

    } catch (error) {
      console.error('❌ Demo initialization failed:', error);
      demoState.update(state => ({
        ...state,
        error: `Initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    }
  });

  // Demo functions
  async function runLegalAnalysis() {
    demoState.update(state => ({ ...state, processing: true, error: null, results: null }));

    try {
      const startTime = performance.now();
      
      const content = customContent.trim() || selectedDocument.content;
      const title = customContent.trim() ? "Custom Legal Document" : selectedDocument.title;

      console.log(`🔍 Analyzing: ${title}`);

      const result = await gemma3LokiIntegration.analyzeLegalDocument({
        content,
        title,
        caseId: selectedDocument.caseId,
        analysisType: selectedDocument.analysisType,
        useCache: true,
        storeResults: true,
        userId: 'demo-user'
      });

      const totalTime = performance.now() - startTime;

      demoState.update(state => ({
        ...state,
        processing: false,
        results: result,
        performance: {
          ...state.performance,
          totalTime,
          webassemblyInferences: state.performance.webassemblyInferences + (result.analysis.method === 'webassembly' ? 1 : 0),
          cacheHits: state.performance.cacheHits + (result.caching.cached ? 1 : 0)
        }
      }));

      console.log(`✅ Analysis completed in ${totalTime.toFixed(2)}ms using ${result.analysis.method}`);

    } catch (error) {
      console.error('❌ Legal analysis failed:', error);
      demoState.update(state => ({
        ...state,
        processing: false,
        error: `Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    }
  }

  async function runBatchAnalysis() {
    demoState.update(state => ({ ...state, processing: true, error: null, results: null }));

    try {
      console.log('📦 Starting batch analysis of all sample documents');

      const response = await fetch('/api/ai/gemma3-loki?action=batch-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documents: sampleDocuments,
          analysisType: 'comprehensive',
          useCache: true,
          storeResults: true,
          userId: 'demo-user',
          maxConcurrency: 3
        })
      });

      if (!response.ok) {
        throw new Error(`Batch analysis failed: ${response.status}`);
      }

      const result = await response.json();

      demoState.update(state => ({
        ...state,
        processing: false,
        results: result.batchAnalysis,
        performance: {
          ...state.performance,
          totalTime: result.batchAnalysis.performance.totalTime,
          webassemblyInferences: state.performance.webassemblyInferences + result.batchAnalysis.successful
        }
      }));

      console.log(`✅ Batch analysis completed: ${result.batchAnalysis.successful}/${result.batchAnalysis.total} successful`);

    } catch (error) {
      console.error('❌ Batch analysis failed:', error);
      demoState.update(state => ({
        ...state,
        processing: false,
        error: `Batch analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    }
  }

  async function runSemanticSearch() {
    if (!searchQuery.trim()) {
      demoState.update(state => ({ ...state, error: 'Please enter a search query' }));
      return;
    }

    demoState.update(state => ({ ...state, processing: true, error: null }));

    try {
      console.log(`🔍 Performing semantic search: "${searchQuery}"`);

      const results = await gemma3LokiIntegration.searchLegalContent(searchQuery, {
        includeEmbeddings: true,
        maxResults: 10
      });

      searchResults = results;

      demoState.update(state => ({ ...state, processing: false }));

      console.log(`✅ Search completed: ${results.length} results found`);

    } catch (error) {
      console.error('❌ Semantic search failed:', error);
      demoState.update(state => ({
        ...state,
        processing: false,
        error: `Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    }
  }

  async function loadPerformanceStats() {
    try {
      const stats = gemma3LokiIntegration.getPerformanceStats();
      performanceStats = stats;
      console.log('📊 Performance stats loaded:', stats);
    } catch (error) {
      console.error('❌ Failed to load performance stats:', error);
    }
  }

  function selectDocument(doc: typeof sampleDocuments[0]) {
    selectedDocument = doc;
    customContent = '';
  }

  function setActiveTab(tab: string) {
    demoState.update(state => ({ ...state, activeTab: tab }));
  }
</script>

<div class="llvm-gemma3-demo">
  <div class="demo-header">
    <h2>🚀 LLVM-Quality WebAssembly Gemma3 Integration</h2>
    <p>High-performance legal AI with intelligent caching, vertex analysis, and evidence processing</p>
    
    {#if !$demoState.initialized}
      <div class="loading">
        <div class="spinner"></div>
        <span>Initializing LLVM WebAssembly engines...</span>
      </div>
    {/if}
  </div>

  {#if $demoState.initialized}
    <!-- Tab Navigation -->
    <div class="tab-nav">
      <button 
        class="tab-button"
        class:active={$demoState.activeTab === 'legal-analysis'}
        on:click={() => setActiveTab('legal-analysis')}
      >
        📄 Legal Analysis
      </button>
      <button 
        class="tab-button"
        class:active={$demoState.activeTab === 'batch-processing'}
        on:click={() => setActiveTab('batch-processing')}
      >
        📦 Batch Processing
      </button>
      <button 
        class="tab-button"
        class:active={$demoState.activeTab === 'semantic-search'}
        on:click={() => setActiveTab('semantic-search')}
      >
        🔍 Semantic Search
      </button>
      <button 
        class="tab-button"
        class:active={$demoState.activeTab === 'performance'}
        on:click={() => setActiveTab('performance')}
      >
        📊 Performance
      </button>
    </div>

    <!-- Legal Analysis Tab -->
    {#if $demoState.activeTab === 'legal-analysis'}
      <div class="tab-content">
        <div class="analysis-controls">
          <h3>🎯 Single Document Analysis</h3>
          
          <!-- Document Selection -->
          <div class="document-selector">
            <h4>Sample Documents:</h4>
            <div class="document-buttons">
              {#each sampleDocuments as doc}
                <button 
                  class="document-button"
                  class:selected={selectedDocument === doc}
                  on:click={() => selectDocument(doc)}
                >
                  {doc.title}
                  <span class="analysis-type">{doc.analysisType}</span>
                </button>
              {/each}
            </div>
          </div>

          <!-- Custom Content -->
          <div class="custom-content">
            <h4>Or Enter Custom Legal Text:</h4>
            <textarea
              bind:value={customContent}
              placeholder="Enter your legal document text here..."
              rows="8"
              class="custom-textarea"
            ></textarea>
          </div>

          <!-- Analysis Button -->
          <button 
            class="analyze-button"
            disabled={$demoState.processing}
            on:click={runLegalAnalysis}
          >
            {#if $demoState.processing}
              <div class="spinner small"></div>
              Analyzing with LLVM WebAssembly...
            {:else}
              🚀 Run LLVM Analysis
            {/if}
          </button>
        </div>

        <!-- Results Display -->
        {#if $demoState.results && !$demoState.results.total}
          <div class="results-panel">
            <h3>📋 Analysis Results</h3>
            
            <div class="performance-badge">
              <span class="method-badge method-{$demoState.results.analysis.method}">
                {$demoState.results.analysis.method.toUpperCase()}
              </span>
              <span class="time-badge">
                {$demoState.results.analysis.processingTime.toFixed(2)}ms
              </span>
              {#if $demoState.results.caching.cached}
                <span class="cache-badge">CACHED</span>
              {/if}
            </div>

            <div class="analysis-summary">
              <h4>📄 Summary</h4>
              <p>{$demoState.results.analysis.summary}</p>
            </div>

            <div class="key-findings">
              <h4>🔍 Key Findings</h4>
              <ul>
                {#each $demoState.results.analysis.keyFindings as finding}
                  <li>{finding}</li>
                {/each}
              </ul>
            </div>

            <div class="legal-risks">
              <h4>⚠️ Legal Risks</h4>
              <div class="risks-grid">
                {#each $demoState.results.analysis.legalRisks as risk}
                  <div class="risk-card risk-{risk.severity}">
                    <div class="risk-type">{risk.type}</div>
                    <div class="risk-severity">{risk.severity.toUpperCase()}</div>
                    <div class="risk-description">{risk.description}</div>
                    <div class="risk-recommendation">💡 {risk.recommendation}</div>
                  </div>
                {/each}
              </div>
            </div>

            <div class="entities-terms">
              <div class="entities">
                <h4>🏢 Legal Entities</h4>
                <div class="tag-list">
                  {#each $demoState.results.analysis.entities as entity}
                    <span class="entity-tag">{entity}</span>
                  {/each}
                </div>
              </div>
              
              <div class="key-terms">
                <h4>🔑 Key Terms</h4>
                <div class="tag-list">
                  {#each $demoState.results.analysis.keyTerms as term}
                    <span class="term-tag">{term}</span>
                  {/each}
                </div>
              </div>
            </div>

            <div class="embeddings-info">
              <h4>🧠 Vector Embeddings</h4>
              <p>
                Generated {$demoState.results.embeddings.dimensions}-dimensional embedding 
                using {$demoState.results.embeddings.model} 
                in {$demoState.results.embeddings.processingTime.toFixed(2)}ms
              </p>
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Batch Processing Tab -->
    {#if $demoState.activeTab === 'batch-processing'}
      <div class="tab-content">
        <div class="batch-controls">
          <h3>📦 Batch Document Processing</h3>
          <p>Process multiple legal documents simultaneously with LLVM-quality performance</p>
          
          <div class="batch-info">
            <div class="document-count">
              📄 {sampleDocuments.length} sample documents ready for processing
            </div>
            <div class="processing-options">
              <strong>Features:</strong>
              <ul>
                <li>✅ Parallel WebAssembly inference</li>
                <li>✅ Intelligent Loki caching</li>
                <li>✅ Evidence database storage</li>
                <li>✅ Semantic indexing</li>
              </ul>
            </div>
          </div>

          <button 
            class="analyze-button"
            disabled={$demoState.processing}
            on:click={runBatchAnalysis}
          >
            {#if $demoState.processing}
              <div class="spinner small"></div>
              Processing batch with LLVM...
            {:else}
              🚀 Run Batch Analysis
            {/if}
          </button>
        </div>

        {#if $demoState.results && $demoState.results.total}
          <div class="batch-results">
            <h3>📊 Batch Processing Results</h3>
            
            <div class="batch-summary">
              <div class="stat-card">
                <div class="stat-value">{$demoState.results.successful}</div>
                <div class="stat-label">Successful</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">{$demoState.results.failed}</div>
                <div class="stat-label">Failed</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">{$demoState.results.performance.totalTime.toFixed(0)}ms</div>
                <div class="stat-label">Total Time</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">{$demoState.results.performance.averageTimePerDocument.toFixed(0)}ms</div>
                <div class="stat-label">Avg/Document</div>
              </div>
            </div>

            <div class="batch-results-list">
              {#each $demoState.results.results as result}
                <div class="batch-result-item">
                  <div class="result-header">
                    <span class="document-title">{result.analysis.summary.substring(0, 60)}...</span>
                    <span class="method-badge method-{result.analysis.method}">
                      {result.analysis.method.toUpperCase()}
                    </span>
                  </div>
                  <div class="result-stats">
                    <span>⏱️ {result.analysis.processingTime.toFixed(2)}ms</span>
                    <span>🎯 {(result.analysis.confidence * 100).toFixed(1)}% confidence</span>
                    <span>⚠️ {result.analysis.legalRisks.length} risks identified</span>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Semantic Search Tab -->
    {#if $demoState.activeTab === 'semantic-search'}
      <div class="tab-content">
        <div class="search-controls">
          <h3>🔍 Semantic Legal Search</h3>
          <p>Search through cached analyses and evidence using vector embeddings</p>
          
          <div class="search-input-group">
            <input
              type="text"
              bind:value={searchQuery}
              placeholder="Enter search query (e.g., 'contract liability', 'digital forensics', 'indemnification clause')"
              class="search-input"
              on:keypress={(e) => e.key === 'Enter' && runSemanticSearch()}
            />
            <button 
              class="search-button"
              disabled={$demoState.processing}
              on:click={runSemanticSearch}
            >
              {#if $demoState.processing}
                <div class="spinner small"></div>
              {:else}
                🔍 Search
              {/if}
            </button>
          </div>
        </div>

        {#if searchResults.length > 0}
          <div class="search-results">
            <h4>📋 Search Results ({searchResults.length})</h4>
            
            {#each searchResults as result}
              <div class="search-result-item">
                <div class="result-header">
                  <span class="result-title">{result.title || 'Cached Analysis'}</span>
                  <span class="result-source">Source: {result.source}</span>
                  {#if result.similarity}
                    <span class="similarity-score">{(result.similarity * 100).toFixed(1)}% similar</span>
                  {/if}
                </div>
                <div class="result-content">
                  {result.summary || result.description || 'No description available'}
                </div>
                {#if result.metadata}
                  <div class="result-metadata">
                    {#if result.metadata.caseId}
                      <span class="metadata-tag">Case: {result.metadata.caseId}</span>
                    {/if}
                    {#if result.metadata.confidence}
                      <span class="metadata-tag">Confidence: {(result.metadata.confidence * 100).toFixed(1)}%</span>
                    {/if}
                    {#if result.metadata.riskLevel}
                      <span class="metadata-tag risk-{result.metadata.riskLevel}">Risk: {result.metadata.riskLevel}</span>
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {:else if searchQuery}
          <div class="no-results">
            <p>No results found for "{searchQuery}". Try analyzing some documents first.</p>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Performance Tab -->
    {#if $demoState.activeTab === 'performance'}
      <div class="tab-content">
        <div class="performance-controls">
          <h3>📊 Performance Metrics</h3>
          <button class="refresh-button" on:click={loadPerformanceStats}>
            🔄 Refresh Stats
          </button>
        </div>

        <div class="performance-overview">
          <div class="perf-stat-card">
            <div class="perf-stat-value">{$demoState.performance.webassemblyInferences}</div>
            <div class="perf-stat-label">WebAssembly Inferences</div>
          </div>
          <div class="perf-stat-card">
            <div class="perf-stat-value">{$demoState.performance.cacheHits}</div>
            <div class="perf-stat-label">Cache Hits</div>
          </div>
          <div class="perf-stat-card">
            <div class="perf-stat-value">{$demoState.performance.totalTime.toFixed(0)}ms</div>
            <div class="perf-stat-label">Total Processing Time</div>
          </div>
        </div>

        {#if performanceStats}
          <div class="detailed-performance">
            <h4>🔧 System Performance Details</h4>
            
            <div class="performance-grid">
              <div class="perf-section">
                <h5>🚀 LLVM WebAssembly Engine</h5>
                <ul>
                  <li>Total Requests: {performanceStats.totalRequests}</li>
                  <li>WebAssembly Success Rate: {performanceStats.webassemblySuccessRate?.toFixed(1)}%</li>
                  <li>Average Processing: {performanceStats.averageProcessingTime.toFixed(2)}ms</li>
                  <li>Ollama Fallbacks: {performanceStats.ollamaFallbacks}</li>
                </ul>
              </div>

              <div class="perf-section">
                <h5>💾 Loki Cache Performance</h5>
                <ul>
                  <li>Cache Hit Rate: {performanceStats.cacheHitRate?.toFixed(1)}%</li>
                  <li>Total Entries: {performanceStats.cacheStats?.totalEntries || 0}</li>
                  <li>Memory Usage: {performanceStats.cacheStats?.memoryUsage || 0} MB</li>
                </ul>
              </div>

              <div class="perf-section">
                <h5>🔍 Search & Evidence</h5>
                <ul>
                  <li>Evidence Documents: {performanceStats.evidenceStats?.total || 0}</li>
                  <li>Indexed Items: {performanceStats.searchStats?.totalItems || 0}</li>
                  <li>Items with Embeddings: {performanceStats.searchStats?.itemsWithEmbeddings || 0}</li>
                </ul>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Error Display -->
    {#if $demoState.error}
      <div class="error-panel">
        <h4>❌ Error</h4>
        <p>{$demoState.error}</p>
      </div>
    {/if}
  {/if}
</div>

<style>
  .llvm-gemma3-demo {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    color: #e0e0e0;
    border-radius: 12px;
    border: 1px solid #333;
  }

  .demo-header {
    text-align: center;
    margin-bottom: 2rem;
  }

  .demo-header h2 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg, #00ff88, #0088ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .demo-header p {
    color: #888;
    font-size: 1rem;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 2rem;
    background: rgba(0, 255, 136, 0.1);
    border-radius: 8px;
    border: 1px solid rgba(0, 255, 136, 0.3);
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid #333;
    border-top: 2px solid #00ff88;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .spinner.small {
    width: 16px;
    height: 16px;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .tab-nav {
    display: flex;
    gap: 1px;
    margin-bottom: 2rem;
    background: #333;
    border-radius: 8px;
    padding: 2px;
  }

  .tab-button {
    flex: 1;
    padding: 0.75rem 1rem;
    background: #222;
    border: none;
    color: #888;
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.2s;
  }

  .tab-button:hover {
    background: #333;
    color: #ccc;
  }

  .tab-button.active {
    background: linear-gradient(45deg, #00ff88, #0088ff);
    color: #000;
    font-weight: bold;
  }

  .tab-content {
    background: #1a1a1a;
    border-radius: 8px;
    padding: 1.5rem;
    border: 1px solid #333;
  }

  .document-selector {
    margin-bottom: 1.5rem;
  }

  .document-buttons {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
  }

  .document-button {
    padding: 1rem;
    background: #222;
    border: 1px solid #444;
    border-radius: 8px;
    color: #ccc;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
  }

  .document-button:hover {
    border-color: #00ff88;
    background: #2a2a2a;
  }

  .document-button.selected {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
  }

  .analysis-type {
    display: block;
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.25rem;
  }

  .custom-textarea {
    width: 100%;
    min-height: 120px;
    padding: 1rem;
    background: #222;
    border: 1px solid #444;
    border-radius: 8px;
    color: #ccc;
    font-family: 'Monaco', monospace;
    font-size: 0.9rem;
    resize: vertical;
  }

  .custom-textarea:focus {
    outline: none;
    border-color: #00ff88;
  }

  .analyze-button, .search-button, .refresh-button {
    padding: 0.75rem 1.5rem;
    background: linear-gradient(45deg, #00ff88, #0088ff);
    border: none;
    border-radius: 8px;
    color: #000;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .analyze-button:hover, .search-button:hover, .refresh-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
  }

  .analyze-button:disabled, .search-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .results-panel {
    margin-top: 2rem;
    padding: 1.5rem;
    background: #1f1f1f;
    border-radius: 8px;
    border: 1px solid #333;
  }

  .performance-badge {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .method-badge {
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

  .time-badge, .cache-badge {
    padding: 0.25rem 0.5rem;
    background: #333;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  .cache-badge {
    background: #0088ff;
    color: #000;
    font-weight: bold;
  }

  .risks-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    margin-top: 0.5rem;
  }

  .risk-card {
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid;
  }

  .risk-low {
    background: rgba(0, 255, 136, 0.1);
    border-color: #00ff88;
  }

  .risk-medium {
    background: rgba(255, 136, 0, 0.1);
    border-color: #ff8800;
  }

  .risk-high {
    background: rgba(255, 68, 68, 0.1);
    border-color: #ff4444;
  }

  .risk-critical {
    background: rgba(255, 0, 0, 0.1);
    border-color: #ff0000;
  }

  .risk-severity {
    font-weight: bold;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
  }

  .entities-terms {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 1.5rem;
  }

  .tag-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .entity-tag, .term-tag {
    padding: 0.25rem 0.5rem;
    background: #333;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  .entity-tag {
    border: 1px solid #00ff88;
  }

  .term-tag {
    border: 1px solid #0088ff;
  }

  .search-input-group {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .search-input {
    flex: 1;
    padding: 0.75rem;
    background: #222;
    border: 1px solid #444;
    border-radius: 8px;
    color: #ccc;
    font-size: 1rem;
  }

  .search-input:focus {
    outline: none;
    border-color: #00ff88;
  }

  .search-result-item {
    padding: 1rem;
    background: #222;
    border-radius: 8px;
    border: 1px solid #333;
    margin-bottom: 1rem;
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

  .result-source {
    background: #333;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  .similarity-score {
    background: #0088ff;
    color: #000;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: bold;
  }

  .result-metadata {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .metadata-tag {
    padding: 0.25rem 0.5rem;
    background: #333;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  .performance-overview {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .perf-stat-card, .stat-card {
    text-align: center;
    padding: 1.5rem;
    background: #222;
    border-radius: 8px;
    border: 1px solid #333;
  }

  .perf-stat-value, .stat-value {
    font-size: 2rem;
    font-weight: bold;
    color: #00ff88;
  }

  .perf-stat-label, .stat-label {
    margin-top: 0.5rem;
    color: #888;
    font-size: 0.9rem;
  }

  .performance-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
  }

  .perf-section {
    background: #222;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid #333;
  }

  .perf-section h5 {
    margin-bottom: 1rem;
    color: #00ff88;
  }

  .perf-section ul {
    list-style: none;
    padding: 0;
  }

  .perf-section li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #333;
  }

  .error-panel {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(255, 68, 68, 0.1);
    border: 1px solid #ff4444;
    border-radius: 8px;
  }

  .error-panel h4 {
    color: #ff4444;
    margin-bottom: 0.5rem;
  }

  .batch-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .batch-result-item {
    background: #222;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #333;
    margin-bottom: 0.5rem;
  }

  .result-stats {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #888;
  }

  .no-results {
    text-align: center;
    padding: 2rem;
    color: #888;
  }
</style>