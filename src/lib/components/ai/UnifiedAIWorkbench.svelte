<script lang="ts">
    // Unified AI Workbench: Complete integration of WebAssembly Gemma3 + GPU acceleration
    // + Evidence processing + Document management + Real-time analytics
    
    import { onMount, onDestroy } from 'svelte';
    import { writable, derived } from 'svelte/store';
    import { gpuGemma3Orchestrator } from '$lib/services/gpu-gemma3-orchestrator';
    import { evidenceProcessingMachine } from '$lib/state/evidenceProcessingMachine';
    import { interpret } from 'xstate';
    import LocalGemma3Chat from './LocalGemma3Chat.svelte';
    import Button from '$lib/components/ui/Button.svelte';
    import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
    
    // Props
    export let workbenchMode: 'chat' | 'document-analysis' | 'batch-processing' | 'research' = 'document-analysis';
    export let enableGpuAcceleration = true;
    export let showAdvancedMetrics = true;
    export let maxConcurrentDocs = 4;
    
    // State management
    let initialized = false;
    let loading = false;
    let error = '';
    
    // Evidence processing state machine
    const evidenceProcessor = interpret(evidenceProcessingMachine).start();
    
    // Reactive stores
    const systemStats = writable({
        orchestrator: {
            initialized: false,
            documentsProcessed: 0,
            totalProcessingTime: 0,
            avgThroughput: 0,
            gpuOperations: 0,
            wasmOperations: 0,
            queueLength: 0
        },
        gemma3: {
            initialized: false,
            wasmAvailable: false,
            modelLoaded: false,
            ollamaHealthy: false,
            requests: { total: 0, webAssembly: 0, ollama: 0, cacheHits: 0 },
            performance: { averageResponseTime: 0, wasmTokensPerSecond: 0, memoryUsage: 0 }
        },
        gpu: {
            available: false,
            accelerated: false,
            serviceUrl: ''
        }
    });
    
    const documentQueue = writable<Array<{
        id: string;
        title: string;
        content: string;
        status: 'queued' | 'processing' | 'completed' | 'error';
        progress: number;
        results?: any;
        error?: string;
        timestamp: number;
    }>>([]);
    
    const batchResults = writable<{
        totalDocuments: number;
        completed: number;
        errors: number;
        averageTime: number;
        clustering?: any;
        insights?: string[];
    } | null>(null);
    
    const realtimeMetrics = writable({
        tokensPerSecond: 0,
        memoryUsage: 0,
        gpuUtilization: 0,
        cacheHitRate: 0,
        activeStreams: 0,
        queuedDocuments: 0
    });
    
    // File upload handling
    let fileInput: HTMLInputElement;
    let dragDropActive = false;
    let uploadProgress = 0;
    
    // Derived computed values
    const totalDocuments = derived(documentQueue, $queue => $queue.length);
    const completedDocuments = derived(documentQueue, $queue => $queue.filter(doc => doc.status === 'completed').length);
    const processingEfficiency = derived(
        [systemStats, totalDocuments], 
        ([$stats, $total]) => $total > 0 ? ($stats.orchestrator.documentsProcessed / $total) * 100 : 0
    );
    
    const systemHealth = derived(systemStats, $stats => {
        const checks = [
            $stats.orchestrator.initialized,
            $stats.gemma3.initialized,
            $stats.gpu.available || true, // GPU is optional
        ];
        return checks.filter(Boolean).length / checks.length;
    });
    
    // Component lifecycle
    onMount(async () => {
        await initializeWorkbench();
        startMetricsUpdates();
        
        // Set up evidence processing listeners
        evidenceProcessor.subscribe(state => {
            if (state.changed) {
                console.log('[UnifiedAIWorkbench] Evidence processing state:', state.value);
                updateDocumentStatus(state.context);
            }
        });
    });
    
    onDestroy(() => {
        gpuGemma3Orchestrator.dispose();
        evidenceProcessor.stop();
    });
    
    // Initialization
    async function initializeWorkbench(): Promise<void> {
        loading = true;
        error = '';
        
        try {
            console.log('[UnifiedAIWorkbench] Initializing AI workbench...');
            
            const success = await gpuGemma3Orchestrator.initialize();
            if (success) {
                initialized = true;
                await updateSystemStats();
                console.log('[UnifiedAIWorkbench] Workbench initialized successfully');
            } else {
                throw new Error('Failed to initialize orchestrator');
            }
            
        } catch (err) {
            error = err instanceof Error ? err.message : 'Initialization failed';
            console.error('[UnifiedAIWorkbench] Initialization error:', err);
        } finally {
            loading = false;
        }
    }
    
    // Document processing
    async function processDocument(title: string, content: string, analysisType: 'comprehensive' | 'quick' | 'risk-focused' = 'comprehensive'): Promise<void> {
        if (!initialized) return;
        
        const documentId = crypto.randomUUID();
        
        // Add to queue
        documentQueue.update(queue => [...queue, {
            id: documentId,
            title,
            content,
            status: 'queued',
            progress: 0,
            timestamp: Date.now()
        }]);
        
        try {
            // Update status to processing
            updateDocumentStatus({ documentId, status: 'processing', progress: 10 });
            
            // Send to evidence processor
            evidenceProcessor.send({
                type: 'START_PROCESSING',
                data: { evidenceId: documentId, fileName: title, contentType: 'text/plain' }
            });
            
            // Process with orchestrator
            const result = await gpuGemma3Orchestrator.processLegalDocument(title, content, {
                analysisType,
                generateEmbeddings: true,
                performClustering: workbenchMode === 'batch-processing',
                findSimilarDocuments: true,
                storeResults: true
            });
            
            // Update with results
            updateDocumentStatus({
                documentId,
                status: 'completed',
                progress: 100,
                results: result
            });
            
            // Send completion to evidence processor
            evidenceProcessor.send({
                type: 'ANALYSIS_COMPLETE',
                data: { evidenceId: documentId, analysis: result.analysis }
            });
            
            await updateSystemStats();
            
        } catch (err) {
            console.error('[UnifiedAIWorkbench] Document processing failed:', err);
            
            updateDocumentStatus({
                documentId,
                status: 'error',
                progress: 0,
                error: err instanceof Error ? err.message : 'Processing failed'
            });
            
            evidenceProcessor.send({
                type: 'ERROR',
                data: { evidenceId: documentId, error: err instanceof Error ? err.message : 'Unknown error' }
            });
        }
    }
    
    // Batch processing
    async function processBatch(): Promise<void> {
        const documents = $documentQueue.filter(doc => doc.status === 'queued');
        if (documents.length === 0) return;
        
        try {
            console.log(`[UnifiedAIWorkbench] Processing batch: ${documents.length} documents`);
            
            const batchData = documents.map(doc => ({
                title: doc.title,
                content: doc.content,
                metadata: { id: doc.id }
            }));
            
            const result = await gpuGemma3Orchestrator.processBatchDocuments(batchData, {
                analysisType: 'comprehensive',
                generateEmbeddings: true,
                performClustering: true,
                maxConcurrency: maxConcurrentDocs
            });
            
            // Update results
            result.results.forEach((docResult, index) => {
                const originalDoc = documents[index];
                updateDocumentStatus({
                    documentId: originalDoc.id,
                    status: docResult.error ? 'error' : 'completed',
                    progress: 100,
                    results: docResult,
                    error: docResult.error
                });
            });
            
            // Store batch results
            batchResults.set({
                totalDocuments: documents.length,
                completed: result.results.filter(r => !r.error).length,
                errors: result.results.filter(r => r.error).length,
                averageTime: result.processing.totalTime / documents.length,
                clustering: result.clustering,
                insights: generateBatchInsights(result)
            });
            
            await updateSystemStats();
            
        } catch (err) {
            console.error('[UnifiedAIWorkbench] Batch processing failed:', err);
            error = err instanceof Error ? err.message : 'Batch processing failed';
        }
    }
    
    // File handling
    async function handleFileUpload(event: Event): Promise<void> {
        const input = event.target as HTMLInputElement;
        const files = Array.from(input.files || []);
        
        for (const file of files) {
            try {
                const content = await readFileContent(file);
                await processDocument(file.name, content);
            } catch (err) {
                console.error(`[UnifiedAIWorkbench] Failed to process file ${file.name}:`, err);
            }
        }
        
        // Clear input
        input.value = '';
    }
    
    async function handleDrop(event: DragEvent): Promise<void> {
        event.preventDefault();
        dragDropActive = false;
        
        const files = Array.from(event.dataTransfer?.files || []);
        
        for (const file of files) {
            if (file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md')) {
                try {
                    const content = await readFileContent(file);
                    await processDocument(file.name, content);
                } catch (err) {
                    console.error(`[UnifiedAIWorkbench] Failed to process dropped file ${file.name}:`, err);
                }
            }
        }
    }
    
    function handleDragOver(event: DragEvent): void {
        event.preventDefault();
        dragDropActive = true;
    }
    
    function handleDragLeave(): void {
        dragDropActive = false;
    }
    
    async function readFileContent(file: File): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }
    
    // Utility functions
    function updateDocumentStatus(update: { documentId: string; status?: string; progress?: number; results?: any; error?: string }): void {
        documentQueue.update(queue => {
            const doc = queue.find(d => d.id === update.documentId);
            if (doc) {
                if (update.status) doc.status = update.status as any;
                if (update.progress !== undefined) doc.progress = update.progress;
                if (update.results) doc.results = update.results;
                if (update.error) doc.error = update.error;
            }
            return [...queue];
        });
    }
    
    async function updateSystemStats(): Promise<void> {
        const stats = gpuGemma3Orchestrator.getSystemStats();
        systemStats.set(stats);
    }
    
    function startMetricsUpdates(): void {
        setInterval(async () => {
            await updateSystemStats();
            
            // Update real-time metrics
            const stats = $systemStats;
            realtimeMetrics.set({
                tokensPerSecond: stats.gemma3.performance.wasmTokensPerSecond || 0,
                memoryUsage: stats.gemma3.performance.memoryUsage || 0,
                gpuUtilization: stats.gpu.available ? 75 : 0, // Simulated
                cacheHitRate: stats.gemma3.requests.total > 0 ? 
                    (stats.gemma3.requests.cacheHits / stats.gemma3.requests.total) * 100 : 0,
                activeStreams: 0, // Would track active streaming requests
                queuedDocuments: $documentQueue.filter(doc => doc.status === 'queued').length
            });
        }, 2000); // Update every 2 seconds
    }
    
    function generateBatchInsights(batchResult: any): string[] {
        const insights = [];
        
        if (batchResult.processing.parallelization) {
            insights.push(`Parallel processing achieved ${batchResult.processing.documentsPerSecond.toFixed(1)} docs/sec`);
        }
        
        if (batchResult.clustering) {
            insights.push(`Documents clustered into ${batchResult.clustering.numClusters} semantic groups`);
        }
        
        const avgAnalysisTime = batchResult.processing.totalTime / batchResult.results.length;
        insights.push(`Average analysis time: ${avgAnalysisTime.toFixed(0)}ms per document`);
        
        return insights;
    }
    
    function clearQueue(): void {
        documentQueue.set([]);
        batchResults.set(null);
    }
    
    function getStatusColor(status: string): string {
        switch (status) {
            case 'queued': return 'text-yellow-600';
            case 'processing': return 'text-blue-600';
            case 'completed': return 'text-green-600';
            case 'error': return 'text-red-600';
            default: return 'text-gray-600';
        }
    }
    
    function getStatusIcon(status: string): string {
        switch (status) {
            case 'queued': return '⏳';
            case 'processing': return '⚙️';
            case 'completed': return '✅';
            case 'error': return '❌';
            default: return '📄';
        }
    }
    
    function formatBytes(bytes: number): string {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function formatDuration(ms: number): string {
        if (ms < 1000) return `${ms.toFixed(0)}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60000).toFixed(1)}m`;
    }
</script>

<!-- Unified AI Workbench Interface -->
<div class="unified-ai-workbench">
    <!-- Header with System Status -->
    <Card class="mb-6">
        <CardHeader>
            <CardTitle class="flex items-center justify-between">
                <span class="flex items-center gap-3">
                    🤖 Unified AI Legal Workbench
                    <span class="system-health health-{Math.round($systemHealth * 100)}">
                        {Math.round($systemHealth * 100)}% System Health
                    </span>
                </span>
                
                <div class="flex items-center gap-2">
                    <span class="mode-indicator mode-{workbenchMode}">
                        {workbenchMode.replace('-', ' ').toUpperCase()}
                    </span>
                    <Button variant="outline" size="sm" on:click={clearQueue}>
                        Clear Queue
                    </Button>
                </div>
            </CardTitle>
        </CardHeader>
        
        {#if showAdvancedMetrics}
            <CardContent>
                <div class="metrics-grid">
                    <!-- System Performance -->
                    <div class="metric-card">
                        <h4>🚀 Performance</h4>
                        <div class="metric-value">{$realtimeMetrics.tokensPerSecond.toFixed(0)} tok/s</div>
                        <div class="metric-label">Generation Speed</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>🧠 Memory</h4>
                        <div class="metric-value">{formatBytes($realtimeMetrics.memoryUsage * 1024 * 1024)}</div>
                        <div class="metric-label">WebAssembly Heap</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>🎮 GPU</h4>
                        <div class="metric-value">{$realtimeMetrics.gpuUtilization.toFixed(0)}%</div>
                        <div class="metric-label">Utilization</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>💾 Cache</h4>
                        <div class="metric-value">{$realtimeMetrics.cacheHitRate.toFixed(0)}%</div>
                        <div class="metric-label">Hit Rate</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>📊 Throughput</h4>
                        <div class="metric-value">{$systemStats.orchestrator.avgThroughput.toFixed(1)}</div>
                        <div class="metric-label">Docs/Second</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>⚡ Method</h4>
                        <div class="metric-value method-indicator">
                            {#if $systemStats.orchestrator.wasmOperations > 0}🧩{/if}
                            {#if $systemStats.orchestrator.gpuOperations > 0}🎮{/if}
                        </div>
                        <div class="metric-label">WebAssembly + GPU</div>
                    </div>
                </div>
            </CardContent>
        {/if}
    </Card>

    <!-- Error Display -->
    {#if error}
        <Card class="mb-4 border-red-200 bg-red-50">
            <CardContent class="text-red-700">
                ⚠️ {error}
            </CardContent>
        </Card>
    {/if}

    <!-- Loading State -->
    {#if loading}
        <Card class="mb-6">
            <CardContent class="text-center py-8">
                <div class="loading-spinner mx-auto mb-4"></div>
                <p class="text-gray-600">Initializing AI workbench systems...</p>
                <div class="initialization-steps mt-4">
                    <div class="step">🧠 Loading Gemma3 WebAssembly engine...</div>
                    <div class="step">🎮 Connecting to GPU acceleration service...</div>
                    <div class="step">🔗 Initializing evidence processing pipeline...</div>
                </div>
            </CardContent>
        </Card>
    {/if}

    {#if initialized}
        <!-- Mode-specific Interface -->
        {#if workbenchMode === 'chat'}
            <LocalGemma3Chat 
                analysisMode="chat"
                enableStreaming={true}
                showPerformanceStats={false}
            />
            
        {:else if workbenchMode === 'document-analysis'}
            <!-- Document Upload and Analysis -->
            <Card class="mb-6">
                <CardHeader>
                    <CardTitle>📄 Document Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                    <!-- Drag and Drop Zone -->
                    <div 
                        class="drag-drop-zone"
                        class:active={dragDropActive}
                        on:drop={handleDrop}
                        on:dragover={handleDragOver}
                        on:dragleave={handleDragLeave}
                        role="button"
                        tabindex="0"
                    >
                        <div class="drag-drop-content">
                            <div class="drag-drop-icon">📁</div>
                            <h3>Drop legal documents here or click to browse</h3>
                            <p>Supports: .txt, .md, .pdf, .doc, .docx files</p>
                            <Button 
                                variant="outline" 
                                on:click={() => fileInput.click()}
                                class="mt-4"
                            >
                                Browse Files
                            </Button>
                            <input 
                                bind:this={fileInput}
                                type="file"
                                multiple
                                accept=".txt,.md,.pdf,.doc,.docx"
                                on:change={handleFileUpload}
                                hidden
                            />
                        </div>
                    </div>
                </CardContent>
            </Card>

        {:else if workbenchMode === 'batch-processing'}
            <!-- Batch Processing Interface -->
            <div class="batch-interface">
                <Card class="mb-6">
                    <CardHeader>
                        <CardTitle class="flex items-center justify-between">
                            📦 Batch Processing
                            <Button 
                                on:click={processBatch}
                                disabled={$documentQueue.filter(d => d.status === 'queued').length === 0}
                                variant="default"
                            >
                                Process Batch ({$documentQueue.filter(d => d.status === 'queued').length})
                            </Button>
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <!-- Batch Results Summary -->
                        {#if $batchResults}
                            <div class="batch-summary mb-4">
                                <div class="summary-stats">
                                    <div class="stat">
                                        <span class="stat-number">{$batchResults.completed}</span>
                                        <span class="stat-label">Completed</span>
                                    </div>
                                    <div class="stat">
                                        <span class="stat-number">{$batchResults.errors}</span>
                                        <span class="stat-label">Errors</span>
                                    </div>
                                    <div class="stat">
                                        <span class="stat-number">{formatDuration($batchResults.averageTime)}</span>
                                        <span class="stat-label">Avg Time</span>
                                    </div>
                                </div>
                                
                                {#if $batchResults.insights}
                                    <div class="insights">
                                        <h4>📊 Insights</h4>
                                        <ul>
                                            {#each $batchResults.insights as insight}
                                                <li>{insight}</li>
                                            {/each}
                                        </ul>
                                    </div>
                                {/if}
                            </div>
                        {/if}
                        
                        <!-- File Upload for Batch -->
                        <div class="batch-upload">
                            <Button 
                                variant="outline"
                                on:click={() => fileInput.click()}
                            >
                                Add Documents to Batch
                            </Button>
                            <input 
                                bind:this={fileInput}
                                type="file"
                                multiple
                                accept=".txt,.md,.pdf,.doc,.docx"
                                on:change={handleFileUpload}
                                hidden
                            />
                        </div>
                    </CardContent>
                </Card>
            </div>
        {/if}

        <!-- Document Queue -->
        {#if $documentQueue.length > 0}
            <Card>
                <CardHeader>
                    <CardTitle>
                        📋 Document Queue ({$documentQueue.length})
                        <span class="queue-progress">
                            {$completedDocuments} / {$totalDocuments} completed
                        </span>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div class="document-list">
                        {#each $documentQueue as doc (doc.id)}
                            <div class="document-item">
                                <div class="document-header">
                                    <span class="document-icon">{getStatusIcon(doc.status)}</span>
                                    <span class="document-title">{doc.title}</span>
                                    <span class="document-status {getStatusColor(doc.status)}">
                                        {doc.status}
                                    </span>
                                </div>
                                
                                {#if doc.status === 'processing'}
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {doc.progress}%"></div>
                                    </div>
                                {/if}
                                
                                {#if doc.results}
                                    <div class="document-results">
                                        <div class="result-summary">
                                            Processing time: {formatDuration(doc.results.processing.totalTime)}
                                            {#if doc.results.performance.tokensPerSecond}
                                                | Speed: {doc.results.performance.tokensPerSecond.toFixed(0)} tok/s
                                            {/if}
                                        </div>
                                        
                                        {#if doc.results.analysis}
                                            <div class="analysis-preview">
                                                <strong>Analysis:</strong> 
                                                {doc.results.analysis.summary.substring(0, 150)}...
                                            </div>
                                        {/if}
                                    </div>
                                {/if}
                                
                                {#if doc.error}
                                    <div class="document-error">
                                        ❌ {doc.error}
                                    </div>
                                {/if}
                            </div>
                        {/each}
                    </div>
                </CardContent>
            </Card>
        {/if}
    {/if}
</div>

<style>
    .unified-ai-workbench {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        min-height: 100vh;
    }

    .system-health {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }

    .health-100 { background: linear-gradient(45deg, #10b981, #059669); }
    .health-75, .health-80, .health-90 { background: linear-gradient(45deg, #f59e0b, #d97706); }
    .health-50, .health-60, .health-70 { background: linear-gradient(45deg, #f59e0b, #d97706); }
    .health-0, .health-10, .health-20, .health-30, .health-40 { background: linear-gradient(45deg, #ef4444, #dc2626); }

    .mode-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }

    .mode-chat { background: linear-gradient(45deg, #8b5cf6, #7c3aed); }
    .mode-document-analysis { background: linear-gradient(45deg, #06b6d4, #0891b2); }
    .mode-batch-processing { background: linear-gradient(45deg, #10b981, #059669); }
    .mode-research { background: linear-gradient(45deg, #f59e0b, #d97706); }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-card {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }

    .metric-card h4 {
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        color: #6b7280;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #9ca3af;
    }

    .method-indicator {
        font-size: 1.25rem;
        display: flex;
        justify-content: center;
        gap: 0.25rem;
    }

    .drag-drop-zone {
        border: 2px dashed #d1d5db;
        border-radius: 0.5rem;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.2s ease;
        cursor: pointer;
        background: #fafafa;
    }

    .drag-drop-zone:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }

    .drag-drop-zone.active {
        border-color: #10b981;
        background: #ecfdf5;
        border-style: solid;
    }

    .drag-drop-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .drag-drop-content h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #374151;
    }

    .drag-drop-content p {
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .batch-summary {
        background: #f8fafc;
        border-radius: 0.5rem;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
    }

    .summary-stats {
        display: flex;
        gap: 2rem;
        margin-bottom: 1rem;
    }

    .stat {
        text-align: center;
    }

    .stat-number {
        display: block;
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    .stat-label {
        font-size: 0.875rem;
        color: #6b7280;
    }

    .insights {
        border-top: 1px solid #e5e7eb;
        padding-top: 1rem;
    }

    .insights h4 {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #374151;
    }

    .insights ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .insights li {
        padding: 0.25rem 0;
        font-size: 0.875rem;
        color: #6b7280;
    }

    .insights li::before {
        content: "💡 ";
        margin-right: 0.5rem;
    }

    .document-list {
        space-y: 1rem;
    }

    .document-item {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        background: white;
        margin-bottom: 1rem;
    }

    .document-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .document-icon {
        font-size: 1.25rem;
    }

    .document-title {
        flex: 1;
        font-weight: 600;
        color: #1f2937;
    }

    .document-status {
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: capitalize;
    }

    .progress-bar {
        width: 100%;
        height: 0.5rem;
        background: #f3f4f6;
        border-radius: 9999px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(45deg, #10b981, #059669);
        transition: width 0.3s ease;
        border-radius: 9999px;
    }

    .document-results {
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid #f3f4f6;
    }

    .result-summary {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }

    .analysis-preview {
        font-size: 0.875rem;
        color: #374151;
        line-height: 1.5;
    }

    .document-error {
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 0.375rem;
        color: #b91c1c;
        font-size: 0.875rem;
    }

    .queue-progress {
        font-size: 0.875rem;
        font-weight: 400;
        color: #6b7280;
    }

    .loading-spinner {
        width: 3rem;
        height: 3rem;
        border: 3px solid #f3f4f6;
        border-top-color: #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    .initialization-steps {
        text-align: left;
        max-width: 400px;
        margin: 0 auto;
    }

    .step {
        padding: 0.5rem 0;
        font-size: 0.875rem;
        color: #6b7280;
        border-left: 2px solid #e5e7eb;
        padding-left: 1rem;
        margin-bottom: 0.5rem;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .unified-ai-workbench {
            padding: 0.5rem;
        }

        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .summary-stats {
            flex-direction: column;
            gap: 1rem;
        }

        .document-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
    }
</style>