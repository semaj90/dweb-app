<!--
  Document Detail Modal - Cache-First Hybrid Architecture
  
  Implements the complete "Graph on a Texture" concept with:
  1. Cache-first strategy using IndexedDB
  2. Server-side processing for complex analysis
  3. Real-time graph updates and interactions
  4. Vector similarity integration
  5. GPU acceleration for document analysis
-->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import { db } from '$lib/db/dexie-integration.js';
  import { logger } from '$lib/logging/structured-logger.js';
  import { unifiedStore } from '$lib/storage/unified-dimensional-store.js';
  
  // Component props
  let { documentId = $bindable() } = $props(); // string;
  let { isOpen = $bindable() } = $props(); // false;
  let { onClose = $bindable() } = $props(); // (() => void) | null = null;
  
  const dispatch = createEventDispatcher<{
    nodeSelect: { nodeId: string; documentId: string };
    graphUpdate: { connections: any[] };
    cacheUpdate: { documentId: string; data: any };
  }>();

  // ========================================================================
  // STATE MANAGEMENT
  // ========================================================================
  
  interface DocumentDetail {
    id: string;
    title: string;
    content: string;
    document_type: string;
    metadata: Record<string, any>;
    created_at: string;
    updated_at: string;
    has_embedding: boolean;
    content_hash?: string;
  }

  interface RelatedDocument {
    id: string;
    title: string;
    content: string;
    documentType: string;
    similarity: number;
    metadata: Record<string, any>;
  }

  interface GraphConnection {
    type: string;
    targetId: string;
    targetTitle: string;
    relationship_strength: number;
    connection_type: string;
  }

  // Component state
let document = $state<DocumentDetail | null >(null);
let relatedDocuments = $state<RelatedDocument[] >([]);
let graphConnections = $state<GraphConnection[] >([]);
let caseAssociations = $state<any[] >([]);
let gpuAnalysis = $state<any >(null);
let metadata = $state<any >(null);
let loading = $state(false);
let error = $state<string | null >(null);
let cacheHit = $state(false);
let serverResponseTime = $state(0);
let activeTab = $state('document');
let enableGPUAnalysis = $state(false);

  // ========================================================================
  // CACHE-FIRST DATA LOADING STRATEGY
  // ========================================================================

  /**
   * Cache-first document loading with fallback to server
   * Implements the hybrid architecture pattern
   */
  async function loadDocumentData(docId: string, forceRefresh = false): Promise<void> {
    if (!docId) return;
    
    const startTime = performance.now();
    loading = true;
    error = null;
    cacheHit = false;

    try {
      // Step 1: Check IndexedDB cache first (Fast Path)
      if (!forceRefresh) {
        await logger.logPerformance({
          operation: 'cache_lookup_start',
          documentId: docId,
          startTime: Date.now()
        });

        const cachedData = await db.getCache(`document_detail_${docId}`);
        
        if (cachedData) {
          // Cache hit - populate UI immediately
          populateUIFromCache(cachedData);
          cacheHit = true;
          
          await logger.logPerformance({
            operation: 'cache_hit',
            documentId: docId,
            processingTime: performance.now() - startTime,
            cacheSize: JSON.stringify(cachedData).length
          });
          
          // Still fetch fresh data in background for next time
          loadFromServerInBackground(docId);
          return;
        }
      }

      // Step 2: Fetch from server (Slow Path)
      await loadFromServer(docId, forceRefresh);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      error = errorMessage;
      
      await logger.logError({
        error: errorMessage,
        context: 'document_detail_load',
        documentId: docId,
        severity: 'medium',
        category: 'ui'
      });
      
    } finally {
      loading = false;
      serverResponseTime = performance.now() - startTime;
    }
  }

  /**
   * Load document data from server with comprehensive analysis
   */
  async function loadFromServer(docId: string, forceRefresh = false): Promise<void> {
    const serverStartTime = performance.now();
    
    await logger.logAPIRequest({
      requestId: crypto.randomUUID(),
      method: 'GET',
      endpoint: `/api/document/${docId}`,
      userAgent: navigator.userAgent,
      ipAddress: 'client',
      headers: { 'Cache-Control': forceRefresh ? 'no-cache' : 'max-age=300' }
    });

    const url = `/api/document/${docId}${enableGPUAnalysis ? '?gpu=true' : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Server returned error');
    }

    // Populate UI from server response
    document = data.document;
    relatedDocuments = data.related_documents || [];
    graphConnections = data.graph_connections || [];
    caseAssociations = data.case_associations || [];
    gpuAnalysis = data.gpu_analysis;
    metadata = data.enhanced_metadata;

    // Update unified store for cross-component sync
    await unifiedStore.updateDocumentCache(docId, {
      document: document,
      relatedDocuments,
      graphConnections,
      timestamp: Date.now()
    });

    // Cache for next time with intelligent TTL
    const cacheData = {
      document,
      relatedDocuments,
      graphConnections,
      caseAssociations,
      gpuAnalysis,
      metadata,
      cached_at: Date.now(),
      cache_source: 'server'
    };

    const ttl = data.cache_instructions?.cache_duration || 5 * 60 * 1000; // 5 minutes
    await db.setCache(`document_detail_${docId}`, cacheData, ttl);

    const serverTime = performance.now() - serverStartTime;
    
    await logger.logAPIResponse({
      requestId: crypto.randomUUID(),
      statusCode: 200,
      responseSize: JSON.stringify(data).length,
      processingTime: serverTime,
      success: true
    });

    // Emit events for graph updates
    dispatch('cacheUpdate', { documentId: docId, data: cacheData });
    if (graphConnections.length > 0) {
      dispatch('graphUpdate', { connections: graphConnections });
    }
  }

  /**
   * Background server refresh for cache warming
   */
  async function loadFromServerInBackground(docId: string): Promise<void> {
    try {
      // Use a shorter timeout for background requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const url = `/api/document/${docId}?background=true`;
      const response = await fetch(url, { 
        signal: controller.signal,
        headers: { 'X-Background-Request': 'true' }
      });
      
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // Update cache for next time
          const cacheData = {
            document: data.document,
            relatedDocuments: data.related_documents || [],
            graphConnections: data.graph_connections || [],
            caseAssociations: data.case_associations || [],
            gpuAnalysis: data.gpu_analysis,
            metadata: data.enhanced_metadata,
            cached_at: Date.now(),
            cache_source: 'background_refresh'
          };

          await db.setCache(`document_detail_${docId}`, cacheData, 5 * 60 * 1000);
        }
      }
    } catch (err) {
      // Silent fail for background requests
      console.debug('Background refresh failed:', err);
    }
  }

  /**
   * Populate UI from cached data
   */
  function populateUIFromCache(cachedData: any): void {
    document = cachedData.document;
    relatedDocuments = cachedData.relatedDocuments || [];
    graphConnections = cachedData.graphConnections || [];
    caseAssociations = cachedData.caseAssociations || [];
    gpuAnalysis = cachedData.gpuAnalysis;
    metadata = cachedData.metadata;
  }

  // ========================================================================
  // INTERACTION HANDLERS
  // ========================================================================

  /**
   * Handle related document click with cache-first strategy
   */
  async function handleRelatedDocumentClick(relatedDoc: RelatedDocument): Promise<void> {
    await logger.logUserAction({
      action: 'click',
      target: 'related_document',
      documentId: relatedDoc.id,
      metadata: { similarity: relatedDoc.similarity, source_document: documentId }
    });

    // Emit node selection event for graph updates
    dispatch('nodeSelect', { 
      nodeId: relatedDoc.id, 
      documentId: relatedDoc.id 
    });

    // Load the clicked document
    await loadDocumentData(relatedDoc.id);
  }

  /**
   * Handle graph connection click
   */
  async function handleGraphConnectionClick(connection: GraphConnection): Promise<void> {
    await logger.logUserAction({
      action: 'click',
      target: 'graph_connection',
      documentId: connection.targetId,
      metadata: { 
        connection_type: connection.connection_type,
        strength: connection.relationship_strength,
        source_document: documentId 
      }
    });

    // For now, show connection details (in production, might navigate to target)
    alert(`Graph Connection: ${connection.type}\nTarget: ${connection.targetTitle}\nStrength: ${connection.relationship_strength}`);
  }

  /**
   * Toggle GPU analysis and refresh
   */
  async function toggleGPUAnalysis(): Promise<void> {
    enableGPUAnalysis = !enableGPUAnalysis;
    
    if (enableGPUAnalysis && !gpuAnalysis) {
      await loadDocumentData(documentId, true); // Force refresh with GPU
    }
  }

  /**
   * Close modal and cleanup
   */
  function handleClose(): void {
    isOpen = false;
    if (onClose) onClose();
    
    // Reset state
    document = null;
    relatedDocuments = [];
    graphConnections = [];
    caseAssociations = [];
    gpuAnalysis = null;
    metadata = null;
    error = null;
    activeTab = 'document';
  }

  // ========================================================================
  // LIFECYCLE
  // ========================================================================

  onMount(() => {
    if (documentId) {
      loadDocumentData(documentId);
    }
  });

  // Reactive loading when documentId changes
  $: if (documentId && isOpen) {
    loadDocumentData(documentId);
  }

  // Helper functions
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatTimestamp(timestamp: string): string {
    return new Date(timestamp).toLocaleString();
  }

  function getSimilarityColor(similarity: number): string {
    if (similarity >= 0.9) return 'text-green-600 bg-green-50';
    if (similarity >= 0.7) return 'text-blue-600 bg-blue-50';
    if (similarity >= 0.5) return 'text-yellow-600 bg-yellow-50';
    return 'text-gray-600 bg-gray-50';
  }

  function getStrengthColor(strength: number): string {
    if (strength >= 0.8) return 'border-green-500 bg-green-50';
    if (strength >= 0.6) return 'border-blue-500 bg-blue-50';
    if (strength >= 0.4) return 'border-yellow-500 bg-yellow-50';
    return 'border-gray-500 bg-gray-50';
  }
</script>

<!-- Modal Overlay -->
{#if isOpen}
  <div 
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    transitifade={{ duration: 200 }}
    on:click|self={handleClose}
    role="dialog"
    aria-modal="true"
    aria-labelledby="document-detail-title"
  >
    <!-- Modal Content -->
    <div 
      class="bg-white rounded-lg shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden"
      transitifly={{ y: 50, duration: 300, easing: cubicOut }}
    >
      
      <!-- Header -->
      <div class="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
        <div class="flex-1">
          <h2 id="document-detail-title" class="text-2xl font-bold text-gray-900">
            {document?.title || 'Document Details'}
          </h2>
          <div class="flex items-center gap-4 mt-2 text-sm text-gray-600">
            <span class="flex items-center gap-1">
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
                <path fill-rule="evenodd" d="M4 5a2 2 0 012-2v1a1 1 0 102 0V3a2 2 0 012 0v1a1 1 0 102 0V3a2 2 0 012 2v1a1 1 0 102 0V5a2 2 0 012 2v6a2 2 0 01-2 2H6a2 2 0 01-2-2V5z" clip-rule="evenodd"/>
              </svg>
              {document?.document_type || 'Unknown'}
            </span>
            
            {#if cacheHit}
              <span class="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                ⚡ Cache Hit
              </span>
            {/if}
            
            {#if serverResponseTime > 0}
              <span class="text-xs text-gray-500">
                {serverResponseTime.toFixed(0)}ms
              </span>
            {/if}
          </div>
        </div>
        
        <!-- Controls -->
        <div class="flex items-center gap-2">
          <button
            type="button"
            on:onclick={toggleGPUAnalysis}
            class="px-3 py-1.5 text-sm border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            class:bg-blue-50={enableGPUAnalysis}
            class:text-blue-700={enableGPUAnalysis}
            class:border-blue-300={enableGPUAnalysis}
          >
            🚀 GPU Analysis
          </button>
          
          <button
            type="button"
            on:onclick={() => loadDocumentData(documentId, true)}
            class="px-3 py-1.5 text-sm border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          >
            {loading ? '⟳' : '↻'} Refresh
          </button>
          
          <button
            type="button"
            on:onclick={handleClose}
            class="text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-md p-1"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- Loading State -->
      {#if loading}
        <div class="flex items-center justify-center p-12">
          <div class="flex items-center gap-3">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span class="text-gray-600">
              {cacheHit ? 'Refreshing in background...' : 'Loading document details...'}
            </span>
          </div>
        </div>
      {/if}

      <!-- Error State -->
      {#if error}
        <div class="p-6">
          <div class="bg-red-50 border border-red-200 rounded-md p-4">
            <div class="flex items-center gap-2">
              <svg class="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
              </svg>
              <h3 class="text-sm font-medium text-red-800">Error Loading Document</h3>
            </div>
            <p class="mt-2 text-sm text-red-700">{error}</p>
            <button
              type="button"
              on:onclick={() => loadDocumentData(documentId, true)}
              class="mt-3 text-sm text-red-800 underline hover:text-red-900"
            >
              Try again
            </button>
          </div>
        </div>
      {/if}

      <!-- Content -->
      {#if document && !loading}
        <div class="flex-1 overflow-hidden">
          
          <!-- Tab Navigation -->
          <div class="border-b border-gray-200 bg-gray-50">
            <nav class="flex space-x-8 px-6" aria-label="Tabs">
              {#each [
                { id: 'document', label: 'Document', count: null },
                { id: 'related', label: 'Related', count: relatedDocuments.length },
                { id: 'connections', label: 'Connections', count: graphConnections.length },
                { id: 'cases', label: 'Cases', count: caseAssociations.length },
                { id: 'analysis', label: 'Analysis', count: gpuAnalysis ? 1 : 0 }
              ] as tab}
                <button
                  type="button"
                  on:onclick={() => activeTab = tab.id}
                  class="py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap focus:outline-none focus:ring-2 focus:ring-blue-500"
                  class:border-blue-500={activeTab === tab.id}
                  class:text-blue-600={activeTab === tab.id}
                  class:border-transparent={activeTab !== tab.id}
                  class:text-gray-500={activeTab !== tab.id}
                  class:hover:text-gray-700={activeTab !== tab.id}
                  class:hover:border-gray-300={activeTab !== tab.id}
                >
                  {tab.label}
                  {#if tab.count !== null}
                    <span class="ml-2 py-0.5 px-2 text-xs rounded-full"
                          class:bg-blue-100={activeTab === tab.id}
                          class:text-blue-800={activeTab === tab.id}
                          class:bg-gray-100={activeTab !== tab.id}
                          class:text-gray-600={activeTab !== tab.id}>
                      {tab.count}
                    </span>
                  {/if}
                </button>
              {/each}
            </nav>
          </div>

          <!-- Tab Content -->
          <div class="flex-1 overflow-y-auto max-h-[60vh] p-6">
            
            <!-- Document Tab -->
            {#if activeTab === 'document'}
              <div class="space-y-6">
                <!-- Document Info -->
                <div class="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                  <div>
                    <dt class="text-sm font-medium text-gray-500">Document Type</dt>
                    <dd class="mt-1 text-sm text-gray-900 font-semibold">{document.document_type}</dd>
                  </div>
                  <div>
                    <dt class="text-sm font-medium text-gray-500">Created</dt>
                    <dd class="mt-1 text-sm text-gray-900">{formatTimestamp(document.created_at)}</dd>
                  </div>
                  <div>
                    <dt class="text-sm font-medium text-gray-500">Content Length</dt>
                    <dd class="mt-1 text-sm text-gray-900">{document.content?.length || 0} characters</dd>
                  </div>
                  <div>
                    <dt class="text-sm font-medium text-gray-500">Vector Embedding</dt>
                    <dd class="mt-1 text-sm text-gray-900">
                      <span class="px-2 py-1 text-xs rounded-full"
                            class:bg-green-100={document.has_embedding}
                            class:text-green-800={document.has_embedding}
                            class:bg-gray-100={!document.has_embedding}
                            class:text-gray-800={!document.has_embedding}>
                        {document.has_embedding ? '✓ Available' : '✗ Not Available'}
                      </span>
                    </dd>
                  </div>
                </div>

                <!-- Content -->
                <div>
                  <h3 class="text-lg font-semibold text-gray-900 mb-3">Document Content</h3>
                  <div class="bg-white border border-gray-200 rounded-lg p-4 max-h-64 overflow-y-auto">
                    <pre class="text-sm text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">{document.content}</pre>
                  </div>
                </div>

                <!-- Metadata -->
                {#if document.metadata && Object.keys(document.metadata).length > 0}
                  <div>
                    <h3 class="text-lg font-semibold text-gray-900 mb-3">Metadata</h3>
                    <div class="bg-white border border-gray-200 rounded-lg p-4">
                      <pre class="text-sm text-gray-600 font-mono">{JSON.stringify(document.metadata, null, 2)}</pre>
                    </div>
                  </div>
                {/if}
              </div>
            {/if}

            <!-- Related Documents Tab -->
            {#if activeTab === 'related'}
              <div class="space-y-4">
                <h3 class="text-lg font-semibold text-gray-900">Related Documents</h3>
                
                {#if relatedDocuments.length === 0}
                  <div class="text-center py-8 text-gray-500">
                    <svg class="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                    </svg>
                    No related documents found
                  </div>
                {:else}
                  <div class="space-y-3">
                    {#each relatedDocuments as relatedDoc (relatedDoc.id)}
                      <button
                        type="button"
                        on:onclick={() => handleRelatedDocumentClick(relatedDoc)}
                        class="w-full p-4 text-left border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
                      >
                        <div class="flex items-start justify-between">
                          <div class="flex-1">
                            <h4 class="font-semibold text-gray-900">{relatedDoc.title}</h4>
                            <p class="mt-1 text-sm text-gray-600 line-clamp-2">{relatedDoc.content}</p>
                            <div class="mt-2 flex items-center gap-2">
                              <span class="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                                {relatedDoc.documentType}
                              </span>
                              <span class="text-xs text-gray-500">
                                Source: {relatedDoc.metadata?.source || 'Unknown'}
                              </span>
                            </div>
                          </div>
                          <div class="ml-4">
                            <span class="px-2 py-1 text-xs font-medium rounded-full {getSimilarityColor(relatedDoc.similarity)}">
                              {(relatedDoc.similarity * 100).toFixed(1)}% similar
                            </span>
                          </div>
                        </div>
                      </button>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}

            <!-- Graph Connections Tab -->
            {#if activeTab === 'connections'}
              <div class="space-y-4">
                <h3 class="text-lg font-semibold text-gray-900">Graph Connections</h3>
                
                {#if graphConnections.length === 0}
                  <div class="text-center py-8 text-gray-500">
                    <svg class="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                    No graph connections found
                  </div>
                {:else}
                  <div class="grid gap-4">
                    {#each graphConnections as connection (connection.targetId)}
                      <button
                        type="button"
                        on:onclick={() => handleGraphConnectionClick(connection)}
                        class="p-4 border-l-4 rounded-r-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors {getStrengthColor(connection.relationship_strength)}"
                      >
                        <div class="flex items-start justify-between">
                          <div class="flex-1">
                            <div class="flex items-center gap-2">
                              <span class="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded-full">
                                {connection.type}
                              </span>
                              <span class="text-xs text-gray-500">
                                {connection.connection_type.replace('_', ' ').toUpperCase()}
                              </span>
                            </div>
                            <h4 class="mt-2 font-semibold text-gray-900">{connection.targetTitle}</h4>
                            <p class="mt-1 text-sm text-gray-600">Target ID: {connection.targetId}</p>
                          </div>
                          <div class="ml-4 text-right">
                            <div class="text-sm font-semibold text-gray-900">
                              {(connection.relationship_strength * 100).toFixed(0)}%
                            </div>
                            <div class="text-xs text-gray-500">strength</div>
                          </div>
                        </div>
                      </button>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}

            <!-- Case Associations Tab -->
            {#if activeTab === 'cases'}
              <div class="space-y-4">
                <h3 class="text-lg font-semibold text-gray-900">Case Associations</h3>
                
                {#if caseAssociations.length === 0}
                  <div class="text-center py-8 text-gray-500">
                    <svg class="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                    </svg>
                    No case associations found
                  </div>
                {:else}
                  <div class="space-y-3">
                    {#each caseAssociations as caseAssoc (caseAssoc.id)}
                      <div class="p-4 border border-gray-200 rounded-lg">
                        <div class="flex items-start justify-between">
                          <div>
                            <h4 class="font-semibold text-gray-900">{caseAssoc.title}</h4>
                            <div class="mt-2 flex items-center gap-4">
                              <span class="px-2 py-1 text-xs font-medium rounded-full"
                                    class:bg-green-100={caseAssoc.status === 'active'}
                                    class:text-green-800={caseAssoc.status === 'active'}
                                    class:bg-gray-100={caseAssoc.status !== 'active'}
                                    class:text-gray-800={caseAssoc.status !== 'active'}>
                                {caseAssoc.status}
                              </span>
                              <span class="text-xs text-gray-500">
                                Created: {formatTimestamp(caseAssoc.created_at)}
                              </span>
                            </div>
                          </div>
                          {#if caseAssoc.priority}
                            <span class="px-2 py-1 text-xs font-medium rounded-full"
                                  class:bg-red-100={caseAssoc.priority === 'high'}
                                  class:text-red-800={caseAssoc.priority === 'high'}
                                  class:bg-yellow-100={caseAssoc.priority === 'medium'}
                                  class:text-yellow-800={caseAssoc.priority === 'medium'}
                                  class:bg-green-100={caseAssoc.priority === 'low'}
                                  class:text-green-800={caseAssoc.priority === 'low'}>
                              {caseAssoc.priority} priority
                            </span>
                          {/if}
                        </div>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}

            <!-- GPU Analysis Tab -->
            {#if activeTab === 'analysis'}
              <div class="space-y-4">
                <div class="flex items-center justify-between">
                  <h3 class="text-lg font-semibold text-gray-900">GPU Analysis</h3>
                  {#if !enableGPUAnalysis}
                    <button
                      type="button"
                      on:onclick={toggleGPUAnalysis}
                      class="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      🚀 Enable GPU Analysis
                    </button>
                  {/if}
                </div>
                
                {#if !enableGPUAnalysis}
                  <div class="text-center py-8 text-gray-500">
                    <svg class="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                    GPU analysis is disabled
                    <p class="mt-2 text-sm">Enable GPU acceleration for advanced legal document analysis</p>
                  </div>
                {:else if !gpuAnalysis}
                  <div class="text-center py-8 text-gray-500">
                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    Processing with GPU acceleration...
                  </div>
                {:else}
                  <div class="space-y-4">
                    <!-- GPU Analysis Results -->
                    <div class="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                      <h4 class="font-semibold text-blue-900 mb-2">RTX 3060 Ti Analysis</h4>
                      <div class="grid grid-cols-2 gap-4">
                        <div>
                          <dt class="text-sm font-medium text-blue-700">Confidence Score</dt>
                          <dd class="mt-1 text-lg font-bold text-blue-900">
                            {(gpuAnalysis.confidence * 100).toFixed(1)}%
                          </dd>
                        </div>
                        <div>
                          <dt class="text-sm font-medium text-blue-700">Processing Time</dt>
                          <dd class="mt-1 text-lg font-bold text-blue-900">
                            {gpuAnalysis.processingTime}ms
                          </dd>
                        </div>
                      </div>
                    </div>

                    {#if gpuAnalysis.legalAnalysis}
                      <div class="p-4 bg-white border border-gray-200 rounded-lg">
                        <h4 class="font-semibold text-gray-900 mb-2">Legal Analysis</h4>
                        <pre class="text-sm text-gray-700 whitespace-pre-wrap">{JSON.stringify(gpuAnalysis.legalAnalysis, null, 2)}</pre>
                      </div>
                    {/if}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>