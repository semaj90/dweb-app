<!--
  WebGPU Legal Document Graph Demo
  
  Complete demonstration of the "tricubic tensor" legal document system:
  - WebGPU-accelerated 3D graph visualization
  - Dimensional tensor stores for GPU memory management
  - Client-side IndexedDB persistence with Dexie.js
  - Server synchronization with conflict resolution
  - Interactive exploration of legal document networks
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import LegalDocumentGraphViewer from '$lib/components/visualization/LegalDocumentGraphViewer.svelte';
  import { syncService, syncStatus, offlineCapabilities } from '$lib/services/client-server-sync';
  import { legalDB, LegalDBUtils, type GraphVisualizationData, type DocumentCache } from '$lib/db/client-db';
  import type { VectorSearchResult } from '$lib/server/db/enhanced-vector-operations';

  // ============================================================================
  // REACTIVE STATE
  // ============================================================================

  const demoState = writable({
    currentStep: 1,
    maxSteps: 6,
    isGeneratingData: false,
    dataGenerated: false,
    selectedGraphId: 'demo-legal-network',
    searchQuery: '',
    searchResults: [] as VectorSearchResult[],
    databaseStats: {
      totalRecords: 0,
      storageUsed: 'Unknown',
      tables: [] as Array<{ name: string; count: number }>
    }
  });

  const webgpuSupported = writable(false);
  const graphViewer = writable<LegalDocumentGraphViewer | null>(null);

  // Derived stores
  const canProceed = derived(
    [demoState, webgpuSupported],
    ([$demoState, $webgpuSupported]) => 
      $webgpuSupported && !$demoState.isGeneratingData
  );

  const stepProgress = derived(
    demoState,
    $demoState => ($demoState.currentStep / $demoState.maxSteps) * 100
  );

  // ============================================================================
  // DEMO STEPS
  // ============================================================================

  const steps = [
    {
      title: 'System Initialization',
      description: 'Initialize WebGPU, IndexedDB, and sync services',
      component: 'initialization'
    },
    {
      title: 'Generate Sample Data',
      description: 'Create sample legal documents and relationships',
      component: 'data-generation'
    },
    {
      title: 'Client-Side Persistence',
      description: 'Store data in IndexedDB with Dexie.js',
      component: 'client-storage'
    },
    {
      title: 'Graph Visualization',
      description: 'Render 3D legal document network with WebGPU',
      component: 'webgpu-visualization'
    },
    {
      title: 'Vector Search Integration',
      description: 'Hybrid client-server vector search',
      component: 'vector-search'
    },
    {
      title: 'Server Synchronization',
      description: 'Sync data with PostgreSQL backend',
      component: 'server-sync'
    }
  ];

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  onMount(async () => {
    await checkWebGPUSupport();
    await updateDatabaseStats();
    
    // Update stats periodically
    setInterval(updateDatabaseStats, 10000);
  });

  async function checkWebGPUSupport(): Promise<void> {
    try {
      if (!navigator.gpu) {
        throw new Error('WebGPU not available');
      }
      
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No WebGPU adapter found');
      }

      $webgpuSupported = true;
      console.log('[Demo] WebGPU support confirmed');
    } catch (error) {
      console.error('[Demo] WebGPU check failed:', error);
      $webgpuSupported = false;
    }
  }

  async function updateDatabaseStats(): Promise<void> {
    try {
      const stats = await LegalDBUtils.getStorageStats();
      demoState.update(state => ({
        ...state,
        databaseStats: stats
      }));
    } catch (error) {
      console.warn('[Demo] Failed to update database stats:', error);
    }
  }

  // ============================================================================
  // DEMO ACTIONS
  // ============================================================================

  async function generateSampleData(): Promise<void> {
    demoState.update(state => ({ ...state, isGeneratingData: true }));

    try {
      // Generate sample documents
      const sampleDocs: Omit<DocumentCache, 'id'>[] = [
        {
          documentId: 'contract-001',
          hash: LegalDBUtils.createHash('contract-001-content'),
          title: 'Software License Agreement',
          content: 'This Software License Agreement governs the use of proprietary software...',
          contentType: 'text',
          fileSize: 15420,
          lastAccessed: new Date(),
          metadata: {
            aiSummary: 'Standard software licensing terms with indemnification clauses',
            keyTerms: ['license', 'indemnification', 'intellectual property', 'termination'],
            legalEntities: { parties: ['Licensor Corp', 'Licensee LLC'] },
            riskLevel: 'medium',
            jurisdiction: 'California',
            documentType: 'contract'
          }
        },
        {
          documentId: 'case-001',
          hash: LegalDBUtils.createHash('case-001-content'),
          title: 'Smith v. TechCorp Data Breach Litigation',
          content: 'Class action lawsuit regarding unauthorized data disclosure affecting 2.3M users...',
          contentType: 'text',
          fileSize: 28940,
          lastAccessed: new Date(),
          metadata: {
            aiSummary: 'Major data breach class action with significant privacy implications',
            keyTerms: ['data breach', 'privacy', 'class action', 'damages'],
            legalEntities: { 
              parties: ['John Smith', 'TechCorp Inc.'],
              court: 'US District Court Northern District California'
            },
            riskLevel: 'high',
            jurisdiction: 'Federal',
            documentType: 'litigation'
          }
        },
        {
          documentId: 'precedent-001',
          hash: LegalDBUtils.createHash('precedent-001-content'),
          title: 'Carpenter v. United States (2018)',
          content: 'Supreme Court decision on digital privacy and Fourth Amendment protections...',
          contentType: 'text',
          fileSize: 45620,
          lastAccessed: new Date(),
          metadata: {
            aiSummary: 'Landmark decision establishing digital privacy protections under Fourth Amendment',
            keyTerms: ['fourth amendment', 'digital privacy', 'cell phone', 'warrant'],
            legalEntities: {
              parties: ['Timothy Carpenter', 'United States'],
              court: 'Supreme Court of the United States'
            },
            riskLevel: 'low',
            jurisdiction: 'Federal',
            documentType: 'precedent'
          }
        }
      ];

      // Store in IndexedDB
      for (const doc of sampleDocs) {
        await legalDB.documentCache.put(doc);
      }

      // Generate sample graph data
      const graphData: Omit<GraphVisualizationData, 'id'> = {
        graphId: $demoState.selectedGraphId,
        graphType: 'legal-entities',
        nodes: [
          {
            id: 'contract-001',
            label: 'Software License Agreement',
            type: 'document',
            position: { x: 0, y: 0, z: 0 },
            size: 8,
            color: '#3b82f6',
            metadata: { importance: 0.7, category: 'contract' }
          },
          {
            id: 'case-001',
            label: 'Smith v. TechCorp',
            type: 'case',
            position: { x: 50, y: 25, z: 15 },
            size: 12,
            color: '#ef4444',
            metadata: { importance: 0.9, category: 'litigation' }
          },
          {
            id: 'precedent-001',
            label: 'Carpenter v. United States',
            type: 'precedent',
            position: { x: -30, y: 40, z: -20 },
            size: 15,
            color: '#10b981',
            metadata: { importance: 1.0, category: 'supreme_court' }
          },
          {
            id: 'entity-techcorp',
            label: 'TechCorp Inc.',
            type: 'entity',
            position: { x: 25, y: -15, z: 10 },
            size: 10,
            color: '#f59e0b',
            metadata: { importance: 0.8, category: 'corporation' }
          },
          {
            id: 'concept-privacy',
            label: 'Digital Privacy Rights',
            type: 'concept',
            position: { x: -10, y: 20, z: 30 },
            size: 6,
            color: '#8b5cf6',
            metadata: { importance: 0.6, category: 'legal_concept' }
          }
        ],
        edges: [
          {
            id: 'edge-1',
            source: 'contract-001',
            target: 'entity-techcorp',
            weight: 0.8,
            type: 'references',
            color: '#6b7280',
            metadata: { relationship: 'party_to_contract' }
          },
          {
            id: 'edge-2',
            source: 'case-001',
            target: 'entity-techcorp',
            weight: 1.0,
            type: 'involves',
            color: '#dc2626',
            metadata: { relationship: 'defendant' }
          },
          {
            id: 'edge-3',
            source: 'case-001',
            target: 'concept-privacy',
            weight: 0.9,
            type: 'concerns',
            color: '#7c3aed',
            metadata: { relationship: 'legal_issue' }
          },
          {
            id: 'edge-4',
            source: 'precedent-001',
            target: 'concept-privacy',
            weight: 0.95,
            type: 'establishes',
            color: '#059669',
            metadata: { relationship: 'legal_precedent' }
          },
          {
            id: 'edge-5',
            source: 'case-001',
            target: 'precedent-001',
            weight: 0.7,
            type: 'cites',
            color: '#0891b2',
            metadata: { relationship: 'citation' }
          }
        ],
        layout: {
          algorithm: 'force-directed-3d',
          parameters: {
            repulsion: 100,
            attraction: 2,
            damping: 0.95,
            iterations: 1000
          },
          dimensions: 3
        },
        cameraPosition: { x: 0, y: 0, z: 100 },
        createdAt: new Date(),
        lastAccessed: new Date(),
        computationTime: 250
      };

      await legalDB.graphVisualizationData.put(graphData);

      await updateDatabaseStats();
      
      demoState.update(state => ({
        ...state,
        dataGenerated: true,
        currentStep: Math.max(state.currentStep, 2)
      }));

      console.log('[Demo] Sample data generated successfully');
    } catch (error) {
      console.error('[Demo] Failed to generate sample data:', error);
    } finally {
      demoState.update(state => ({ ...state, isGeneratingData: false }));
    }
  }

  async function performVectorSearch(): Promise<void> {
    const query = $demoState.searchQuery.trim();
    if (!query) return;

    try {
      const results = await syncService.performHybridVectorSearch(query, {
        limit: 10,
        lodLevel: 2,
        filterOptions: {
          evidenceType: ['document', 'case', 'precedent']
        }
      });

      demoState.update(state => ({
        ...state,
        searchResults: results,
        currentStep: Math.max(state.currentStep, 5)
      }));

      console.log('[Demo] Vector search completed:', results.length, 'results');
    } catch (error) {
      console.error('[Demo] Vector search failed:', error);
    }
  }

  async function syncToServer(): Promise<void> {
    try {
      // Sync documents
      const docs = await legalDB.documentCache.toArray();
      for (const doc of docs.slice(0, 3)) { // Limit for demo
        await syncService.syncDocumentToServer(doc.documentId);
      }

      // Sync graph visualization
      await syncService.syncGraphVisualization($demoState.selectedGraphId);

      // Force sync all pending operations
      await syncService.forceSyncAll();

      demoState.update(state => ({
        ...state,
        currentStep: Math.max(state.currentStep, 6)
      }));

      console.log('[Demo] Server synchronization completed');
    } catch (error) {
      console.error('[Demo] Server sync failed:', error);
    }
  }

  function nextStep(): void {
    demoState.update(state => ({
      ...state,
      currentStep: Math.min(state.currentStep + 1, state.maxSteps)
    }));
  }

  function prevStep(): void {
    demoState.update(state => ({
      ...state,
      currentStep: Math.max(state.currentStep - 1, 1)
    }));
  }
</script>

<!-- ============================================================================ -->
<!-- PAGE TEMPLATE -->
<!-- ============================================================================ -->

<div class="demo-page">
  <header class="demo-header">
    <h1>WebGPU Legal Document Graph Demonstration</h1>
    <p class="subtitle">
      Complete "Tricubic Tensor" implementation with dimensional tensor stores, 
      WebGPU visualization, and client-server synchronization
    </p>
    
    <!-- Progress Bar -->
    <div class="progress-container">
      <div class="progress-bar" style="width: {$stepProgress}%"></div>
      <span class="progress-text">Step {$demoState.currentStep} of {$demoState.maxSteps}</span>
    </div>
  </header>

  <!-- WebGPU Support Check -->
  {#if !$webgpuSupported}
    <div class="warning-banner">
      <h3>⚠️ WebGPU Not Supported</h3>
      <p>
        This demo requires WebGPU support. Please use Chrome 113+ or Firefox 110+ with WebGPU enabled.
        <br>
        Chrome: Enable <code>chrome://flags/#enable-unsafe-webgpu</code>
        <br>
        Firefox: Enable <code>about:config → dom.webgpu.enabled</code>
      </p>
    </div>
  {/if}

  <!-- Main Content -->
  <main class="demo-content">
    <!-- Step Navigation -->
    <nav class="step-navigation">
      {#each steps as step, index}
        <button 
          class="step-button"
          class:active={$demoState.currentStep === index + 1}
          class:completed={$demoState.currentStep > index + 1}
          disabled={!$canProceed}
          click={() => demoState.update(s => ({ ...s, currentStep: index + 1 }))}
        >
          <span class="step-number">{index + 1}</span>
          <span class="step-title">{step.title}</span>
        </button>
      {/each}
    </nav>

    <!-- Step Content -->
    <section class="step-content">
      {#if $demoState.currentStep === 1}
        <!-- Step 1: System Initialization -->
        <div class="step-panel">
          <h2>System Initialization</h2>
          <p>Check system capabilities and initialize core services.</p>
          
          <div class="capability-grid">
            <div class="capability-item">
              <span class="capability-label">WebGPU Support:</span>
              <span class="capability-status" class:supported={$webgpuSupported}>
                {$webgpuSupported ? '✅ Available' : '❌ Not Available'}
              </span>
            </div>
            
            <div class="capability-item">
              <span class="capability-label">IndexedDB:</span>
              <span class="capability-status supported">✅ Available</span>
            </div>
            
            <div class="capability-item">
              <span class="capability-label">Online Status:</span>
              <span class="capability-status" class:supported={$syncStatus.isOnline}>
                {$syncStatus.isOnline ? '✅ Online' : '❌ Offline'}
              </span>
            </div>
            
            <div class="capability-item">
              <span class="capability-label">Graph Visualization:</span>
              <span class="capability-status" class:supported={$offlineCapabilities.graphVisualizationAvailable}>
                {$offlineCapabilities.graphVisualizationAvailable ? '✅ Available' : '❌ Not Available'}
              </span>
            </div>
          </div>

          <button click={nextStep} disabled={!$webgpuSupported} class="primary-button">
            Initialize System
          </button>
        </div>

      {:else if $demoState.currentStep === 2}
        <!-- Step 2: Generate Sample Data -->
        <div class="step-panel">
          <h2>Generate Sample Legal Data</h2>
          <p>Create sample legal documents, cases, and precedents with relationships.</p>
          
          <div class="data-preview">
            <h3>Sample Data Includes:</h3>
            <ul>
              <li>📄 Software License Agreement (Contract)</li>
              <li>⚖️ Smith v. TechCorp Data Breach Case</li>
              <li>🏛️ Carpenter v. United States Precedent</li>
              <li>🏢 Legal Entities and Relationships</li>
              <li>🔗 Citation Network and Connections</li>
            </ul>
          </div>

          <div class="action-buttons">
            <button 
              click={generateSampleData} 
              disabled={$demoState.isGeneratingData || $demoState.dataGenerated}
              class="primary-button"
            >
              {$demoState.isGeneratingData ? 'Generating...' : 
               $demoState.dataGenerated ? 'Data Generated ✅' : 'Generate Sample Data'}
            </button>
            
            {#if $demoState.dataGenerated}
              <button click={nextStep} class="secondary-button">
                Continue to Storage
              </button>
            {/if}
          </div>
        </div>

      {:else if $demoState.currentStep === 3}
        <!-- Step 3: Client-Side Persistence -->
        <div class="step-panel">
          <h2>Client-Side Persistence</h2>
          <p>Data stored in IndexedDB using Dexie.js with intelligent caching.</p>
          
          <div class="storage-stats">
            <h3>Database Statistics</h3>
            <div class="stats-grid">
              <div class="stat-item">
                <span class="stat-label">Total Records:</span>
                <span class="stat-value">{$demoState.databaseStats.totalRecords}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">Storage Used:</span>
                <span class="stat-value">{$demoState.databaseStats.storageUsed}</span>
              </div>
            </div>
            
            <div class="table-breakdown">
              <h4>Table Breakdown:</h4>
              {#each $demoState.databaseStats.tables as table}
                <div class="table-stat">
                  <span class="table-name">{table.name}:</span>
                  <span class="table-count">{table.count} records</span>
                </div>
              {/each}
            </div>
          </div>

          <div class="action-buttons">
            <button click={async () => { await updateDatabaseStats(); }} class="secondary-button">
              Refresh Stats
            </button>
            
            <button click={nextStep} class="primary-button">
              Launch Visualization
            </button>
          </div>
        </div>

      {:else if $demoState.currentStep === 4}
        <!-- Step 4: Graph Visualization -->
        <div class="step-panel">
          <h2>WebGPU 3D Graph Visualization</h2>
          <p>Interactive exploration of the legal document network using dimensional tensor stores.</p>
          
          <div class="visualization-container">
            {#if $webgpuSupported && $demoState.dataGenerated}
              <LegalDocumentGraphViewer
                bind:this={$graphViewer}
                graphId={$demoState.selectedGraphId}
                width={800}
                height={500}
                enablePhysics={true}
                enableStreaming={true}
                maxNodes={1000}
                class="demo-graph-viewer"
              />
            {:else}
              <div class="visualization-placeholder">
                <p>Visualization requires WebGPU support and sample data.</p>
              </div>
            {/if}
          </div>

          <div class="visualization-controls">
            <button click={() => $graphViewer?.resetCamera()} class="control-button">
              🎯 Reset Camera
            </button>
            
            <button click={() => $graphViewer?.togglePhysics()} class="control-button">
              ⚡ Toggle Physics
            </button>
            
            <button click={() => $graphViewer?.focusOnNode('case-001')} class="control-button">
              🔍 Focus on Case
            </button>
          </div>

          <button click={nextStep} class="primary-button">
            Test Vector Search
          </button>
        </div>

      {:else if $demoState.currentStep === 5}
        <!-- Step 5: Vector Search Integration -->
        <div class="step-panel">
          <h2>Hybrid Vector Search</h2>
          <p>Combine client-side cache with server-side vector search using PostgreSQL pgvector.</p>
          
          <div class="search-interface">
            <div class="search-input-group">
              <input
                type="text"
                bind:value={$demoState.searchQuery}
                placeholder="Search legal documents... (e.g., 'data breach privacy')"
                class="search-input"
                keydown={(e) => e.key === 'Enter' && performVectorSearch()}
              />
              <button click={performVectorSearch} class="search-button">
                🔍 Search
              </button>
            </div>
            
            <div class="search-options">
              <label class="option-label">
                <input type="checkbox" checked /> Use client cache
              </label>
              <label class="option-label">
                <input type="checkbox" checked /> Hybrid search
              </label>
              <label class="option-label">
                <input type="checkbox" checked /> Fallback to client
              </label>
            </div>
          </div>

          {#if $demoState.searchResults.length > 0}
            <div class="search-results">
              <h3>Search Results ({$demoState.searchResults.length})</h3>
              {#each $demoState.searchResults as result}
                <div class="result-item">
                  <div class="result-header">
                    <span class="result-title">{result.metadata?.title || result.id}</span>
                    <span class="result-similarity">
                      {(result.similarity * 100).toFixed(1)}% match
                    </span>
                  </div>
                  <div class="result-content">
                    {result.content.substring(0, 200)}...
                  </div>
                  <div class="result-meta">
                    <span class="result-type">{result.sourceType}</span>
                    <span class="result-lod">LOD {result.ragLevel}</span>
                  </div>
                </div>
              {/each}
            </div>
          {/if}

          <button click={nextStep} class="primary-button">
            Test Server Sync
          </button>
        </div>

      {:else if $demoState.currentStep === 6}
        <!-- Step 6: Server Synchronization -->
        <div class="step-panel">
          <h2>Server Synchronization</h2>
          <p>Synchronize client data with PostgreSQL backend using conflict resolution.</p>
          
          <div class="sync-status">
            <h3>Sync Status</h3>
            <div class="status-grid">
              <div class="status-item">
                <span class="status-label">Connection:</span>
                <span class="status-value" class:line={$syncStatus.isOnline}>
                  {$syncStatus.isOnline ? 'Online' : 'Offline'}
                </span>
              </div>
              
              <div class="status-item">
                <span class="status-label">Pending Operations:</span>
                <span class="status-value">{$syncStatus.pendingOperations}</span>
              </div>
              
              <div class="status-item">
                <span class="status-label">Last Sync:</span>
                <span class="status-value">
                  {$syncStatus.lastSync ? $syncStatus.lastSync.toLocaleTimeString() : 'Never'}
                </span>
              </div>
              
              <div class="status-item">
                <span class="status-label">Conflicts:</span>
                <span class="status-value">{$syncStatus.conflictCount}</span>
              </div>
            </div>

            {#if $syncStatus.currentOperation}
              <div class="current-operation">
                <span class="operation-label">Current:</span>
                <span class="operation-text">{$syncStatus.currentOperation}</span>
                <div class="operation-progress" style="width: {$syncStatus.syncProgress * 100}%"></div>
              </div>
            {/if}
          </div>

          <div class="action-buttons">
            <button 
              click={syncToServer}
              disabled={!$syncStatus.isOnline}
              class="primary-button"
            >
              Sync to Server
            </button>
            
            <button click={async () => { await syncService.clearCache(); }} class="secondary-button">
              Clear Cache
            </button>
          </div>

          <div class="completion-message">
            <h3>🎉 Demo Complete!</h3>
            <p>
              You've successfully demonstrated the complete "Tricubic Tensor" legal document system
              with WebGPU visualization, dimensional tensor stores, and client-server synchronization.
            </p>
          </div>
        </div>
      {/if}
    </section>

    <!-- Navigation Controls -->
    <footer class="demo-navigation">
      <button 
        click={prevStep} 
        disabled={$demoState.currentStep === 1}
        class="nav-button prev"
      >
        ← Previous
      </button>
      
      <span class="step-indicator">
        {$demoState.currentStep} / {$demoState.maxSteps}
      </span>
      
      <button 
        click={nextStep}
        disabled={$demoState.currentStep === $demoState.maxSteps}
        class="nav-button next"
      >
        Next →
      </button>
    </footer>
  </main>
</div>

<!-- ============================================================================ -->
<!-- COMPONENT STYLES -->
<!-- ============================================================================ -->

<style>
  .demo-page {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    font-family: 'Inter', sans-serif;
  }

  .demo-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .demo-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.5rem;
  }

  .subtitle {
    font-size: 1.1rem;
    color: #6b7280;
    margin-bottom: 2rem;
    line-height: 1.6;
  }

  .progress-container {
    position: relative;
    width: 100%;
    height: 8px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 0.9rem;
    color: #6b7280;
    font-weight: 500;
  }

  .warning-banner {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .warning-banner h3 {
    color: #b45309;
    margin-bottom: 0.5rem;
  }

  .warning-banner code {
    background: rgba(0, 0, 0, 0.1);
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
  }

  .demo-content {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 2rem;
    min-height: 600px;
  }

  .step-navigation {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .step-button {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 1rem;
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    background: white;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
  }

  .step-button:hover {
    border-color: #3b82f6;
    background: #f8fafc;
  }

  .step-button.active {
    border-color: #3b82f6;
    background: #eff6ff;
  }

  .step-button.completed {
    border-color: #10b981;
    background: #f0fdf4;
  }

  .step-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: #e5e7eb;
    color: #6b7280;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .step-button.active .step-number {
    background: #3b82f6;
    color: white;
  }

  .step-button.completed .step-number {
    background: #10b981;
    color: white;
  }

  .step-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
  }

  .step-content {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
  }

  .step-panel {
    padding: 2rem;
  }

  .step-panel h2 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
  }

  .step-panel p {
    color: #6b7280;
    margin-bottom: 2rem;
    line-height: 1.6;
  }

  .capability-grid,
  .stats-grid,
  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .capability-item,
  .stat-item,
  .status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
  }

  .capability-status.supported,
  .status-value.online {
    color: #059669;
    font-weight: 600;
  }

  .data-preview {
    background: #f0f9ff;
    border: 1px solid #0ea5e9;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .data-preview h3 {
    color: #0c4a6e;
    margin-bottom: 1rem;
  }

  .data-preview ul {
    margin: 0;
    padding-left: 1.5rem;
  }

  .data-preview li {
    color: #075985;
    margin-bottom: 0.5rem;
  }

  .visualization-container {
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .visualization-placeholder {
    height: 500px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f9fafb;
    color: #6b7280;
  }

  .visualization-controls {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 2rem;
  }

  .control-button {
    padding: 0.5rem 1rem;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .control-button:hover {
    background: #e5e7eb;
  }

  .search-interface {
    margin-bottom: 2rem;
  }

  .search-input-group {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .search-input {
    flex: 1;
    padding: 0.75rem;
    border: 2px solid #e5e7eb;
    border-radius: 6px;
    font-size: 1rem;
  }

  .search-input:focus {
    outline: none;
    border-color: #3b82f6;
  }

  .search-button {
    padding: 0.75rem 1.5rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
  }

  .search-options {
    display: flex;
    gap: 1rem;
  }

  .option-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    color: #6b7280;
  }

  .search-results {
    margin-top: 2rem;
  }

  .search-results h3 {
    margin-bottom: 1rem;
    color: #1f2937;
  }

  .result-item {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 0.5rem;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .result-title {
    font-weight: 600;
    color: #1f2937;
  }

  .result-similarity {
    background: #eff6ff;
    color: #1d4ed8;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
  }

  .result-content {
    color: #4b5563;
    margin-bottom: 0.5rem;
    line-height: 1.5;
  }

  .result-meta {
    display: flex;
    gap: 0.5rem;
    font-size: 0.8rem;
  }

  .result-type,
  .result-lod {
    background: #f3f4f6;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    color: #6b7280;
  }

  .current-operation {
    position: relative;
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 1rem;
    margin-top: 1rem;
  }

  .operation-progress {
    position: absolute;
    bottom: 0;
    left: 0;
    height: 3px;
    background: #f59e0b;
    transition: width 0.3s ease;
  }

  .completion-message {
    background: #f0fdf4;
    border: 1px solid #22c55e;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    margin-top: 2rem;
  }

  .completion-message h3 {
    color: #166534;
    margin-bottom: 1rem;
  }

  .completion-message p {
    color: #15803d;
    margin: 0;
  }

  .action-buttons {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .primary-button {
    padding: 0.75rem 1.5rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }

  .primary-button:hover:not(:disabled) {
    background: #2563eb;
  }

  .primary-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .secondary-button {
    padding: 0.75rem 1.5rem;
    background: white;
    color: #374151;
    border: 2px solid #e5e7eb;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .secondary-button:hover {
    border-color: #d1d5db;
    background: #f9fafb;
  }

  .demo-navigation {
    grid-column: 1 / -1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem;
    border-top: 1px solid #e5e7eb;
    margin-top: 2rem;
  }

  .nav-button {
    padding: 0.5rem 1rem;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.2s;
  }

  .nav-button:hover:not(:disabled) {
    background: #e5e7eb;
  }

  .nav-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .step-indicator {
    font-weight: 600;
    color: #6b7280;
  }

  /* Responsive Design */
  @media (max-width: 1024px) {
    .demo-content {
      grid-template-columns: 1fr;
    }
    
    .step-navigation {
      flex-direction: row;
      overflow-x: auto;
    }
    
    .step-button {
      min-width: 150px;
    }
  }

  @media (max-width: 768px) {
    .demo-page {
      padding: 1rem;
    }
    
    .demo-header h1 {
      font-size: 2rem;
    }
    
    .capability-grid,
    .stats-grid,
    .status-grid {
      grid-template-columns: 1fr;
    }
    
    .action-buttons {
      flex-direction: column;
    }
    
    .search-input-group {
      flex-direction: column;
    }
  }
</style>