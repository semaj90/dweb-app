<!-- CUDA gRPC Streaming Demo Page -->
<script lang="ts">
    import { onMount } from 'svelte';
    import { page } from '$app/stores';
    import { LegalCudaGrpcService } from '$lib/services/legal-cuda-grpc-client';
    import type { CudaResponse, DocumentProgress, SearchResults } from '$lib/wasm/legal_grpc_client';
    
    // State management
    let cudaService: LegalCudaGrpcService;
    let isConnected = false;
    let loading = true;
    let error = '';
    
    // Streaming states
    let activeSession = '';
    let streamingActive = false;
    let responses: any[] = [];
    
    // Form inputs
    let textInput = '';
    let documentContent = '';
    let searchQuery = 'contract termination clause';
    let selectedCollection = 'legal_documents';
    
    // Performance metrics
    let lastResponseTime = 0;
    let totalResponses = 0;
    let avgProcessingTime = 0;
    
    // Demo data
    const sampleLegalText = `
TERMINATION CLAUSE

Either party may terminate this Agreement upon thirty (30) days written notice 
to the other party. In the event of termination, all obligations under this 
Agreement shall cease, except for those provisions which by their nature should 
survive termination, including but not limited to confidentiality, 
indemnification, and limitation of liability clauses.

Upon termination, each party shall return or destroy all confidential information 
received from the other party. The terminating party shall be liable for all 
costs and expenses incurred up to the date of termination.
    `.trim();
    
    onMount(async () => {
        try {
            // Initialize CUDA gRPC service
            cudaService = new LegalCudaGrpcService('http://localhost:8080');
            
            // Wait for connection
            await new Promise(resolve => setTimeout(resolve, 1000));
            isConnected = await cudaService.isConnected();
            
            if (!isConnected) {
                error = 'Failed to connect to CUDA gRPC service. Ensure the server is running.';
            }
        } catch (err) {
            console.error('Failed to initialize CUDA service:', err);
            error = 'CUDA gRPC service initialization failed. Check console for details.';
        } finally {
            loading = false;
        }
    });
    
    // Start bidirectional streaming session
    async function startStreamingSession() {
        if (!cudaService || streamingActive) return;
        
        try {
            streamingActive = true;
            activeSession = `session_${Date.now()}`;
            responses = [];
            error = '';
            
            await cudaService.startEmbeddingStream(
                activeSession,
                handleStreamResponse,
                handleStreamError,
                handleStreamComplete
            );
            
            console.log(`🚀 Started streaming session: ${activeSession}`);
        } catch (err) {
            console.error('Failed to start streaming session:', err);
            error = 'Failed to start streaming session';
            streamingActive = false;
        }
    }
    
    // Send text for CUDA embedding processing
    async function sendTextForEmbedding() {
        if (!cudaService || !activeSession || !textInput.trim()) return;
        
        try {
            const success = await cudaService.sendTextForEmbedding(
                activeSession, 
                textInput, 
                true // is_final
            );
            
            if (success) {
                console.log('📝 Text sent for CUDA processing');
                textInput = '';
            } else {
                error = 'Failed to send text for processing';
            }
        } catch (err) {
            console.error('Failed to send text:', err);
            error = 'Text processing failed';
        }
    }
    
    // Perform document analysis
    async function processDocument() {
        if (!cudaService || !documentContent.trim()) return;
        
        try {
            error = '';
            const documentId = `doc_${Date.now()}`;
            
            await cudaService.processDocument(
                documentId,
                documentContent,
                'contract',
                handleDocumentProgress
            );
            
            console.log('📄 Document processing started');
        } catch (err) {
            console.error('Document processing failed:', err);
            error = 'Document processing failed';
        }
    }
    
    // Perform semantic search
    async function performSemanticSearch() {
        if (!cudaService || !searchQuery.trim()) return;
        
        try {
            error = '';
            
            await cudaService.searchSemantic(
                searchQuery,
                selectedCollection,
                10,
                handleSearchResults
            );
            
            console.log('🔍 Semantic search started');
        } catch (err) {
            console.error('Semantic search failed:', err);
            error = 'Semantic search failed';
        }
    }
    
    // Event handlers
    function handleStreamResponse(response: CudaResponse) {
        console.log('📡 Stream response:', response);
        
        const parsedResponse = typeof response === 'string' ? JSON.parse(response) : response;
        responses = [...responses, {
            type: 'stream',
            timestamp: new Date(),
            data: parsedResponse
        }];
        
        // Update performance metrics
        totalResponses++;
        if (parsedResponse.performance?.processing_time_us) {
            lastResponseTime = parsedResponse.performance.processing_time_us;
            avgProcessingTime = ((avgProcessingTime * (totalResponses - 1)) + lastResponseTime) / totalResponses;
        }
    }
    
    function handleStreamError(errorMsg: string) {
        console.error('❌ Stream error:', errorMsg);
        error = errorMsg;
        streamingActive = false;
    }
    
    function handleStreamComplete() {
        console.log('✅ Stream completed');
        streamingActive = false;
    }
    
    function handleDocumentProgress(progress: DocumentProgress) {
        console.log('📊 Document progress:', progress);
        responses = [...responses, {
            type: 'document',
            timestamp: new Date(),
            data: progress
        }];
    }
    
    function handleSearchResults(results: SearchResults) {
        console.log('🎯 Search results:', results);
        responses = [...responses, {
            type: 'search',
            timestamp: new Date(),
            data: results
        }];
    }
    
    // Stop streaming session
    async function stopStreamingSession() {
        if (!cudaService || !activeSession) return;
        
        try {
            await cudaService.closeStream(activeSession);
            streamingActive = false;
            activeSession = '';
            console.log('🛑 Streaming session stopped');
        } catch (err) {
            console.error('Failed to stop streaming session:', err);
        }
    }
    
    // Load sample text
    function loadSampleText() {
        textInput = sampleLegalText;
    }
    
    // Clear responses
    function clearResponses() {
        responses = [];
        totalResponses = 0;
        avgProcessingTime = 0;
        lastResponseTime = 0;
    }
</script>

<svelte:head>
    <title>CUDA gRPC Streaming - Legal AI</title>
    <meta name="description" content="Real-time CUDA-accelerated legal document processing via gRPC streaming" />
</svelte:head>

<div class="cuda-streaming-page">
    <header class="page-header">
        <h1>🚀 CUDA gRPC Streaming</h1>
        <p>Real-time legal document processing with GPU acceleration</p>
    </header>
    
    {#if loading}
        <div class="loading-state">
            <div class="spinner"></div>
            <p>Initializing CUDA gRPC service...</p>
        </div>
    {:else if error}
        <div class="error-state">
            <h3>⚠️ Connection Error</h3>
            <p>{error}</p>
            <button onclick={() => window.location.reload()}>🔄 Retry</button>
        </div>
    {:else}
        <!-- Status Panel -->
        <section class="status-panel">
            <div class="status-grid">
                <div class="status-item">
                    <span class="status-label">Connection</span>
                    <span class="status-value" class:connected={isConnected}>
                        {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                    </span>
                </div>
                <div class="status-item">
                    <span class="status-label">Session</span>
                    <span class="status-value">{activeSession || 'None'}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Streaming</span>
                    <span class="status-value" class:active={streamingActive}>
                        {streamingActive ? '🔴 Active' : '⚪ Idle'}
                    </span>
                </div>
                <div class="status-item">
                    <span class="status-label">Responses</span>
                    <span class="status-value">{totalResponses}</span>
                </div>
            </div>
        </section>
        
        <!-- Performance Metrics -->
        {#if totalResponses > 0}
            <section class="metrics-panel">
                <h3>📊 Performance Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <span class="metric-label">Last Response Time</span>
                        <span class="metric-value">{(lastResponseTime / 1000).toFixed(2)}ms</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Average Response Time</span>
                        <span class="metric-value">{(avgProcessingTime / 1000).toFixed(2)}ms</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Processed</span>
                        <span class="metric-value">{totalResponses}</span>
                    </div>
                </div>
            </section>
        {/if}
        
        <!-- Control Panel -->
        <section class="control-panel">
            <div class="control-section">
                <h3>🎛️ Streaming Controls</h3>
                <div class="button-group">
                    <button 
                        onclick={startStreamingSession}
                        disabled={!isConnected || streamingActive}
                        class="primary"
                    >
                        🚀 Start Stream
                    </button>
                    
                    <button 
                        onclick={stopStreamingSession}
                        disabled={!streamingActive}
                        class="secondary"
                    >
                        🛑 Stop Stream
                    </button>
                    
                    <button 
                        onclick={clearResponses}
                        class="tertiary"
                    >
                        🗑️ Clear
                    </button>
                </div>
            </div>
        </section>
        
        <!-- Input Sections -->
        <div class="input-grid">
            <!-- Text Processing -->
            <section class="input-section">
                <h3>📝 Text Embedding</h3>
                <div class="textarea-container">
                    <textarea 
                        bind:value={textInput}
                        placeholder="Enter legal text for CUDA processing..."
                        rows="6"
                        disabled={!activeSession}
                    ></textarea>
                    <div class="textarea-actions">
                        <button onclick={loadSampleText} class="small">
                            📋 Load Sample
                        </button>
                        <button 
                            onclick={sendTextForEmbedding}
                            disabled={!activeSession || !textInput.trim()}
                            class="primary small"
                        >
                            ⚡ Process with CUDA
                        </button>
                    </div>
                </div>
            </section>
            
            <!-- Document Analysis -->
            <section class="input-section">
                <h3>📄 Document Analysis</h3>
                <div class="textarea-container">
                    <textarea 
                        bind:value={documentContent}
                        placeholder="Enter document content for comprehensive analysis..."
                        rows="6"
                    ></textarea>
                    <div class="textarea-actions">
                        <select bind:value={selectedCollection}>
                            <option value="legal_documents">Legal Documents</option>
                            <option value="contracts">Contracts</option>
                            <option value="cases">Cases</option>
                            <option value="statutes">Statutes</option>
                        </select>
                        <button 
                            onclick={processDocument}
                            disabled={!isConnected || !documentContent.trim()}
                            class="primary small"
                        >
                            🔬 Analyze Document
                        </button>
                    </div>
                </div>
            </section>
        </div>
        
        <!-- Search Section -->
        <section class="search-section">
            <h3>🔍 Semantic Search</h3>
            <div class="search-controls">
                <input 
                    type="text" 
                    bind:value={searchQuery}
                    placeholder="Enter search query..."
                    class="search-input"
                />
                <select bind:value={selectedCollection} class="collection-select">
                    <option value="legal_documents">Legal Documents</option>
                    <option value="contracts">Contracts</option>
                    <option value="cases">Cases</option>
                    <option value="statutes">Statutes</option>
                </select>
                <button 
                    onclick={performSemanticSearch}
                    disabled={!isConnected || !searchQuery.trim()}
                    class="primary"
                >
                    🚀 Search
                </button>
            </div>
        </section>
        
        <!-- Real-time Responses -->
        <section class="responses-section">
            <h3>🔄 Real-time Responses</h3>
            <div class="responses-container">
                {#if responses.length === 0}
                    <div class="empty-state">
                        <p>No responses yet. Start a streaming session and send some text for processing.</p>
                    </div>
                {:else}
                    {#each responses as response, i (i)}
                        <div class="response-item" class:stream={response.type === 'stream'} 
                             class:document={response.type === 'document'} 
                             class:search={response.type === 'search'}>
                            <div class="response-header">
                                <span class="response-type">
                                    {#if response.type === 'stream'}🔄 Stream
                                    {:else if response.type === 'document'}📄 Document  
                                    {:else if response.type === 'search'}🔍 Search{/if}
                                </span>
                                <span class="response-timestamp">
                                    {response.timestamp.toLocaleTimeString()}
                                </span>
                            </div>
                            <div class="response-content">
                                <pre>{JSON.stringify(response.data, null, 2)}</pre>
                            </div>
                        </div>
                    {/each}
                {/if}
            </div>
        </section>
    {/if}
</div>

<style>
    .cuda-streaming-page {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    .page-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .page-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .page-header p {
        font-size: 1.1rem;
        color: #6b7280;
    }
    
    .loading-state {
        text-align: center;
        padding: 4rem;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e5e7eb;
        border-left: 4px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .error-state {
        text-align: center;
        padding: 2rem;
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .status-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    
    .status-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    .status-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }
    
    .status-value {
        font-weight: 600;
        font-size: 1rem;
    }
    
    .status-value.connected {
        color: #059669;
    }
    
    .status-value.active {
        color: #dc2626;
    }
    
    .metrics-panel {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric {
        text-align: center;
    }
    
    .metric-label {
        display: block;
        font-size: 0.875rem;
        color: #0369a1;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-weight: 700;
        font-size: 1.125rem;
        color: #0c4a6e;
    }
    
    .control-panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .button-group {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .input-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .input-section {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    .textarea-container {
        margin-top: 1rem;
    }
    
    .textarea-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.75rem;
        gap: 0.75rem;
    }
    
    .search-section {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .search-controls {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        align-items: center;
    }
    
    .search-input {
        flex: 1;
        padding: 0.75rem;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 1rem;
    }
    
    .collection-select {
        padding: 0.75rem;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 1rem;
        min-width: 150px;
    }
    
    .responses-section {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    .responses-container {
        max-height: 600px;
        overflow-y: auto;
        margin-top: 1rem;
    }
    
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #6b7280;
    }
    
    .response-item {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin-bottom: 1rem;
        overflow: hidden;
    }
    
    .response-item.stream {
        border-left: 4px solid #3b82f6;
    }
    
    .response-item.document {
        border-left: 4px solid #10b981;
    }
    
    .response-item.search {
        border-left: 4px solid #f59e0b;
    }
    
    .response-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #f9fafb;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .response-type {
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .response-timestamp {
        font-size: 0.75rem;
        color: #6b7280;
    }
    
    .response-content {
        padding: 1rem;
        background: #fafafa;
    }
    
    .response-content pre {
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.8rem;
        line-height: 1.4;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0;
        color: #374151;
    }
    
    /* Button styles */
    button {
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
        border: none;
        font-size: 0.925rem;
        transition: all 0.2s;
    }
    
    button.primary {
        background: #3b82f6;
        color: white;
    }
    
    button.primary:hover:not(:disabled) {
        background: #2563eb;
    }
    
    button.secondary {
        background: #6b7280;
        color: white;
    }
    
    button.secondary:hover:not(:disabled) {
        background: #4b5563;
    }
    
    button.tertiary {
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
    }
    
    button.tertiary:hover:not(:disabled) {
        background: #e5e7eb;
    }
    
    button.small {
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
    }
    
    button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    textarea {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 1rem;
        font-family: inherit;
        resize: vertical;
    }
    
    h3 {
        margin: 0 0 1rem 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
    }
    
    @media (max-width: 768px) {
        .cuda-streaming-page {
            padding: 1rem;
        }
        
        .input-grid {
            grid-template-columns: 1fr;
        }
        
        .status-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .search-controls {
            flex-direction: column;
            align-items: stretch;
        }
        
        .button-group {
            flex-direction: column;
        }
        
        .textarea-actions {
            flex-direction: column;
            align-items: stretch;
        }
    }
</style>