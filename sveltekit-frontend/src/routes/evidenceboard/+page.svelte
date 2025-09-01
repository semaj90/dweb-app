<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  // Using simple text icons to avoid Svelte 5 compatibility issues with Lucide
  // import { 
  //   Upload,
  //   FileText,
  //   Image,
  //   Search,
  //   Filter,
  //   MoreVertical,
  //   Eye,
  //   Download,
  //   Trash2,
  //   Brain,
  //   Zap,
  //   Target,
  //   Plus,
  //   Minus,
  //   Link,
  //   Library,
  //   BarChart3,
  //   X
  // } from 'lucide-svelte';

  // Evidence management stores
  let evidenceItems = $state([]);
  let filteredEvidence = $state([]);
  let searchQuery = $state('');
  let selectedFilter = $state('all');
  let isUploading = $state(false);
  let uploadProgress = $state(0);
  let processingStatus = $state<'loading' | 'processing' | 'success' | 'error'>('loading');
  let showUploadModal = $state(false);
  let zoomLevel = $state(100);
  let isConnected = $state(true);
  let showAIAssistant = $state(false);

  // Context7 integration state
  let context7Enabled = $state(true);
  let semanticSearchResults = $state([]);
  let ragEnhanced = $state(true);

  // File upload state
  let uploadedFiles = $state([]);
  let dragOver = $state(false);

  // Computed properties using Svelte 5 runes
  let totalEvidence = $derived(evidenceItems.length);
  let processingCount = $derived(
    evidenceItems.filter((item) => item.status === 'processing').length
  );
  let readyCount = $derived(evidenceItems.filter((item) => item.status === 'ready').length);

  onMount(async () => {
    await loadExistingEvidence();
    startRealTimeUpdates();
  });

  async function loadExistingEvidence() {
    try {
      const response = await fetch('/api/evidence/list');
      if (response.ok) {
        const data = await response.json();
        evidenceItems = data.evidence || [];
        filterEvidence();
      }
    } catch (error) {
      console.error('Failed to load evidence:', error);
    }
  }

  function filterEvidence() {
    let filtered = evidenceItems;

    if (searchQuery.trim()) {
      filtered = filtered.filter(
        (item) =>
          item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.summary?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }

    if (selectedFilter !== 'all') {
      filtered = filtered.filter((item) => item.type === selectedFilter);
    }

    filteredEvidence = filtered;
  }

  function startRealTimeUpdates() {
    setInterval(() => {
      evidenceItems = evidenceItems.map((item) => {
        if (item.status === 'processing' && Math.random() > 0.7) {
          return {
            ...item,
            status: 'ready',
            prosecutionScore: Math.random() * 0.4 + 0.6,
            summary: `AI-generated summary for ${item.filename}`,
            entities: ['entity1', 'entity2', 'entity3'],
          };
        }
        return item;
      });
    }, 3000);
  }

  function handleFileUpload(event: DragEvent | Event) {
    const files = event.type === 'drop' 
      ? (event as DragEvent).dataTransfer?.files
      : (event.target as HTMLInputElement).files;
    
    if (!files) return;

    Array.from(files).forEach(file => {
      const newEvidence = {
        id: crypto.randomUUID(),
        filename: file.name,
        status: 'processing',
        type: file.type.startsWith('image/') ? 'image' : 'document',
        size: file.size,
        uploadDate: Date.now(),
        tags: []
      };
      evidenceItems = [...evidenceItems, newEvidence];
    });

    showUploadModal = false;
    filterEvidence();
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    dragOver = true;
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    dragOver = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    dragOver = false;
    handleFileUpload(event);
  }

  $effect(() => {
    filterEvidence();
  });
</script>

<svelte:head>
  <title>EVIDENCE BOARD - YoRHa Detective Interface</title>
</svelte:head>

<!-- YoRHa Detective Interface -->
<div class="yorha-interface">
  <!-- Left Sidebar -->
  <aside class="yorha-sidebar">
    <div class="yorha-logo">
      <div class="yorha-title">YORHA</div>
      <div class="yorha-subtitle">DETECTIVE</div>
      <div class="yorha-subtext">Investigation Interface</div>
    </div>

    <nav class="yorha-nav">
      <div class="nav-section">
        <button class="nav-item" on:onclick={() => goto('/')}>
          <span class="nav-icon">⌂</span>
          COMMAND CENTER
        </button>
        
        <button class="nav-item active-section">
          <span class="nav-text">ACTIVE CASES</span>
          <span class="nav-count">3</span>
        </button>
        
        <div class="nav-subsection">
          <button class="nav-subitem">⚬ ACTIVE</button>
          <button class="nav-subitem">⧖ PENDING</button>
          <button class="nav-subitem">⨯ CLOSED</button>
        </div>
        
        <button class="nav-item evidence-active">
          <span class="nav-icon">📁</span>
          EVIDENCE
        </button>
        
        <div class="nav-subsection">
          <button class="nav-subitem active">📚 LIBRARY</button>
          <button class="nav-subitem">⚏ BOARD</button>
        </div>
        
        <button class="nav-item" on:onclick={() => goto('/cases')}>
          <span class="nav-icon">👤</span>
          PERSONS OF INTEREST
        </button>
        
        <button class="nav-item" on:onclick={() => goto('/demo/ai-dashboard')}>
          <span class="nav-icon">📊</span>
          ANALYSIS
        </button>
        
        <button class="nav-item" on:onclick={() => goto('/semantic-search-demo')}>
          <span class="nav-icon">🔍</span>
          GLOBAL SEARCH
        </button>
        
        <button class="nav-item" on:onclick={() => goto('/dev/mcp-tools')}>
          <span class="nav-icon">></span>
          TERMINAL
        </button>
      </div>
      
      <div class="nav-section bottom-section">
        <button class="nav-item" on:onclick={() => goto('/admin')}>
          <span class="nav-icon">⚙</span>
          SYSTEM CONFIG
        </button>
      </div>
    </nav>

    <div class="yorha-status">
      <div class="status-item">Online</div>
      <div class="status-time">12:32</div>
      <div class="status-text">System: Operational</div>
    </div>
  </aside>

  <!-- Main Evidence Board Area -->
  <main class="yorha-main">
    <!-- Top Header -->
    <header class="evidence-header">
      <div class="header-left">
        <button class="header-icon" on:onclick={() => goto('/')}>⌂</button>
        <h1 class="evidence-title">EVIDENCE BOARD</h1>
        <div class="evidence-subtitle">Corporate Espionage Investigation</div>
      </div>
      
      <div class="header-right">
        <div class="case-info">
          Case: <span class="case-id">CORPORATE ESPIONAGE INV</span>
        </div>
        <button class="header-btn" on:onclick={() => goto('/legal/documents')}>
          <span class="icon">📚</span>
          LIBRARY
        </button>
        <button class="header-btn" on:onclick={() => goto('/demo/ai-dashboard')}>
          <span class="icon">📊</span>
          ANALYSIS
        </button>
        <button class="header-btn ai-assistant" on:onclick={() => {
          showAIAssistant = true;
        }}>
          <span class="icon">🧠</span>
          AI ASSISTANT
        </button>
      </div>
    </header>

    <!-- Toolbar -->
    <div class="evidence-toolbar">
      <div class="toolbar-left">
        <div class="zoom-controls">
          <button class="tool-btn" on:onclick={() => zoomLevel = Math.max(25, zoomLevel - 25)}>
            <span class="icon">➖</span>
          </button>
          <span class="zoom-display">{zoomLevel}%</span>
          <button class="tool-btn" on:onclick={() => zoomLevel = Math.min(200, zoomLevel + 25)}>
            <span class="icon">➕</span>
          </button>
        </div>
        
        <button class="tool-btn" on:onclick={() => goto('/demo/ai-integration')}>
          <span class="icon">🔗</span>
          CONNECT
        </button>
        
        <button class="tool-btn primary" on:onclick={() => showUploadModal = true}>
          <span class="icon">➕</span>
          ADD EVIDENCE
        </button>
        
        <button class="tool-btn" on:onclick={() => goto('/legal/documents')}>
          <span class="icon">📚</span>
          LIBRARY ({totalEvidence})
        </button>
      </div>
      
      <div class="toolbar-right">
        <div class="connection-status {isConnected ? 'connected' : 'disconnected'}">
          ● Connected
        </div>
      </div>
    </div>

    <!-- Evidence Grid Area -->
    <div class="evidence-grid-container" 
         role="region"
         aria-label="Evidence drop zone"
         ondragover={handleDragOver}
         ondragleave={handleDragLeave}
         ondrop={handleDrop}
         class:drag-over={dragOver}
         style="zoom: {zoomLevel}%">
      
      <div class="grid-background"></div>
      
      {#if filteredEvidence.length === 0}
        <div class="empty-state">
          <div class="empty-icon">📁</div>
          <div class="empty-title">No Evidence Files</div>
          <div class="empty-subtitle">Upload evidence files to investigate</div>
          <div class="empty-hint">Drag files here to begin analysis</div>
        </div>
      {:else}
        {#each filteredEvidence as evidence (evidence.id)}
          <div class="evidence-node" 
               style="left: {Math.random() * 70}%; top: {Math.random() * 60 + 20}%;">
            <div class="evidence-icon">
              {#if evidence.type === 'image'}
                <Image class="w-6 h-6" />
              {:else}
                <FileText class="w-6 h-6" />
              {/if}
            </div>
            <div class="evidence-label">{evidence.filename}</div>
            <div class="evidence-status status-{evidence.status}">
              {evidence.status.toUpperCase()}
            </div>
          </div>
        {/each}
      {/if}
    </div>
  </main>
</div>

<!-- Upload Evidence Modal -->
{#if showUploadModal}
  <div class="yorha-modal-backdrop">
    <div class="yorha-modal">
      <div class="modal-header">
        <div class="modal-icon">📤</div>
        <div class="modal-title">UPLOAD EVIDENCE</div>
        <button class="modal-close" on:onclick={() => showUploadModal = false}>
          <span class="icon">✕</span>
        </button>
      </div>
      
      <div class="modal-content">
        <div class="modal-subtitle">Upload evidence files to the case database</div>
        
        <div class="upload-zone" 
             ondragover={handleDragOver}
             ondragleave={handleDragLeave}
             ondrop={handleDrop}
             class:drag-over={dragOver}
             role="region"
             aria-label="Evidence file drop zone">
          <div class="upload-icon">📤</div>
          <div class="upload-text">Drop file or click to browse</div>
          <div class="upload-hint">Images, videos, or PDF documents</div>
          
          <input type="file" 
                 multiple 
                 accept="image/*,video/*,.pdf,.doc,.docx,.txt"
                 change={handleFileUpload}
                 class="file-input">
        </div>
      </div>
      
      <div class="modal-actions">
        <button class="modal-btn cancel" on:onclick={() => showUploadModal = false}>
          CANCEL
        </button>
        <button class="modal-btn primary">
          UPLOAD
        </button>
      </div>
    </div>
  </div>
{/if}


<!-- AI Assistant Modal -->
{#if showAIAssistant}
  <div class="yorha-modal-backdrop">
    <div class="yorha-modal ai-modal">
      <div class="modal-header">
        <div class="modal-icon">🤖</div>
        <div class="modal-title">AI ASSISTANT</div>
        <button class="modal-close" on:onclick={() => showAIAssistant = false}>
          <span class="icon">✕</span>
        </button>
      </div>
      
      <div class="modal-content ai-content">
        <div class="modal-subtitle">YoRHa Legal AI Analysis Engine</div>
        
        <div class="ai-status">
          <div class="ai-indicator active"></div>
          <span>Neural Network Status: ACTIVE</span>
        </div>
        
        <div class="ai-features">
          <div class="ai-feature">
            <span class="icon">🧠</span>
            <span>Evidence Pattern Analysis</span>
          </div>
          <div class="ai-feature">
            <span class="icon">🎯</span>
            <span>Case Correlation Engine</span>
          </div>
          <div class="ai-feature">
            <span class="icon">⚡</span>
            <span>Real-time Legal Advisory</span>
          </div>
        </div>
        
        <div class="ai-query-box">
          <textarea 
            placeholder="Enter legal query or case analysis request..." 
            class="ai-textarea"
            rows="4"></textarea>
        </div>
        
        <div class="ai-quick-actions">
          <button class="ai-action-btn">Analyze Current Case</button>
          <button class="ai-action-btn">Evidence Summary</button>
          <button class="ai-action-btn">Legal Precedent Search</button>
        </div>
      </div>
      
      <div class="modal-actions">
        <button class="modal-btn cancel" on:onclick={() => showAIAssistant = false}>
          CLOSE
        </button>
        <button class="modal-btn primary ai-execute">
          <span class="icon">🧠</span>
          EXECUTE QUERY
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  /* YoRHa Interface Base Styles */
  .yorha-interface {
    display: flex;
    height: 95vh;
    background: #2a2a2a;
    color: #d4af37;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 11px;
    overflow: hidden;
    margin: 10px;
    border: 1px solid #3a3a3a;
  }

  /* Sidebar Styles */
  .yorha-sidebar {
    width: 180px;
    background: #1a1a1a;
    border-right: 1px solid #3a3a3a;
    display: flex;
    flex-direction: column;
  }

  .yorha-logo {
    padding: 15px 12px;
    border-bottom: 1px solid #3a3a3a;
  }

  .yorha-title {
    font-size: 18px;
    font-weight: bold;
    color: #d4af37;
    line-height: 1;
  }

  .yorha-subtitle {
    font-size: 18px;
    font-weight: bold;
    color: #d4af37;
    line-height: 1;
  }

  .yorha-subtext {
    font-size: 10px;
    color: #888;
    margin-top: 5px;
  }

  .yorha-nav {
    flex: 1;
    padding: 15px 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .nav-section {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .nav-item {
    display: flex;
    align-items: center;
    padding: 6px 12px;
    background: none;
    border: none;
    color: #888;
    text-align: left;
    font-family: inherit;
    font-size: 10px;
    cursor: pointer;
    transition: all 0.2s;
    justify-content: space-between;
  }

  .nav-item:hover {
    background: #2a2a2a;
    color: #d4af37;
  }

  .nav-item.active-section {
    background: #2a2a2a;
    color: #d4af37;
  }

  .nav-item.evidence-active {
    background: #1a2a1a;
    color: #d4af37;
    border-left: 3px solid #d4af37;
  }

  .nav-subsection {
    background: #0f0f0f;
    padding: 5px 0;
  }

  .nav-subitem {
    display: block;
    width: 100%;
    padding: 6px 25px;
    background: none;
    border: none;
    color: #666;
    text-align: left;
    font-family: inherit;
    font-size: 10px;
    cursor: pointer;
    transition: color 0.2s;
  }

  .nav-subitem:hover {
    color: #d4af37;
  }

  .nav-subitem.active {
    color: #d4af37;
    background: #2a1a0a;
  }

  .nav-icon {
    margin-right: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    width: 14px;
    height: 14px;
    line-height: 1;
  }

  .nav-count {
    font-size: 10px;
    background: #d4af37;
    color: #000;
    padding: 1px 6px;
    border-radius: 2px;
  }

  .yorha-status {
    padding: 15px;
    border-top: 1px solid #3a3a3a;
    font-size: 10px;
    color: #666;
  }

  .status-item {
    color: #d4af37;
  }

  .status-time {
    font-weight: bold;
  }

  /* Main Content Styles */
  .yorha-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #2a2a2a;
  }

  .evidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 15px;
    border-bottom: 1px solid #3a3a3a;
    background: #2a2a2a;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 15px;
  }

  .header-icon {
    background: none;
    border: 1px solid #555;
    color: #d4af37;
    padding: 6px 8px;
    font-family: inherit;
    cursor: pointer;
  }

  .evidence-title {
    font-size: 24px;
    font-weight: bold;
    color: #d4af37;
    margin: 0;
  }

  .evidence-subtitle {
    font-size: 12px;
    color: #888;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 15px;
  }

  .case-info {
    font-size: 11px;
    color: #888;
  }

  .case-id {
    color: #d4af37;
    background: #1a1a1a;
    padding: 2px 8px;
    font-family: inherit;
  }

  .header-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    background: #3a3a3a;
    border: 1px solid #555;
    color: #d4af37;
    padding: 6px 12px;
    font-family: inherit;
    font-size: 10px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .header-btn:hover {
    background: #4a4a4a;
  }

  /* Toolbar Styles */
  .evidence-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 15px;
    background: #242424;
    border-bottom: 1px solid #3a3a3a;
  }

  .toolbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .zoom-controls {
    display: flex;
    align-items: center;
    gap: 5px;
    background: #1a1a1a;
    border: 1px solid #555;
    padding: 4px;
  }

  .zoom-display {
    padding: 0 8px;
    font-size: 10px;
    color: #d4af37;
    min-width: 35px;
    text-align: center;
  }

  .tool-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    background: #3a3a3a;
    border: 1px solid #555;
    color: #d4af37;
    padding: 6px 10px;
    font-family: inherit;
    font-size: 10px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .tool-btn:hover {
    background: #4a4a4a;
  }

  .tool-btn.primary {
    background: #d4af37;
    color: #000;
    border-color: #d4af37;
  }

  .tool-btn.primary:hover {
    background: #e4bf47;
  }

  .toolbar-right {
    font-size: 10px;
  }

  .connection-status {
    padding: 4px 8px;
    border: 1px solid #555;
  }

  .connection-status.connected {
    color: #4ade80;
    border-color: #4ade80;
  }

  .connection-status.disconnected {
    color: #ef4444;
    border-color: #ef4444;
  }

  /* Evidence Grid Styles */
  .evidence-grid-container {
    flex: 1;
    position: relative;
    background: #2a2a2a;
    overflow: hidden;
    transition: zoom 0.2s;
  }

  .evidence-grid-container.drag-over {
    background: #2a3a2a;
    border: 2px dashed #d4af37;
  }

  .grid-background {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
      linear-gradient(rgba(212, 175, 55, 0.1) 1px, transparent 1px),
      linear-gradient(90deg, rgba(212, 175, 55, 0.1) 1px, transparent 1px);
    background-size: 20px 20px;
  }

  .empty-state {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    color: #666;
  }

  .empty-icon {
    font-size: 48px;
    margin-bottom: 15px;
  }

  .empty-title {
    font-size: 16px;
    color: #888;
    margin-bottom: 5px;
  }

  .empty-subtitle {
    font-size: 12px;
    margin-bottom: 5px;
  }

  .empty-hint {
    font-size: 10px;
    color: #555;
  }

  .evidence-node {
    position: absolute;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px;
    background: #1a1a1a;
    border: 1px solid #555;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 100px;
  }

  .evidence-node:hover {
    background: #2a2a2a;
    border-color: #d4af37;
    transform: scale(1.05);
  }

  .evidence-icon {
    color: #d4af37;
    margin-bottom: 5px;
  }

  .evidence-label {
    font-size: 10px;
    color: #ccc;
    text-align: center;
    margin-bottom: 5px;
    word-break: break-word;
  }

  .evidence-status {
    font-size: 8px;
    padding: 2px 6px;
    border-radius: 2px;
  }

  .status-processing {
    background: #fbbf24;
    color: #000;
  }

  .status-ready {
    background: #4ade80;
    color: #000;
  }

  .status-error {
    background: #ef4444;
    color: #fff;
  }

  /* Modal Styles */
  .yorha-modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
  }

  .yorha-modal {
    background: #2a2a2a;
    border: 2px solid #d4af37;
    min-width: 400px;
    max-width: 500px;
  }

  .modal-header {
    display: flex;
    align-items: center;
    padding: 15px 20px;
    border-bottom: 1px solid #555;
    background: #1a1a1a;
  }

  .modal-icon {
    font-size: 20px;
    margin-right: 12px;
    color: #d4af37;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
  }

  .modal-title {
    flex: 1;
    font-size: 14px;
    font-weight: bold;
    color: #d4af37;
  }

  .modal-close {
    background: none;
    border: none;
    color: #888;
    cursor: pointer;
    padding: 4px;
  }

  .modal-close:hover {
    color: #d4af37;
  }

  .modal-content {
    padding: 20px;
  }

  .modal-subtitle {
    font-size: 12px;
    color: #888;
    margin-bottom: 20px;
  }

  .upload-zone {
    border: 2px dashed #555;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }

  .upload-zone:hover,
  .upload-zone.drag-over {
    border-color: #d4af37;
    background: rgba(212, 175, 55, 0.1);
  }

  .upload-icon {
    font-size: 48px;
    color: #666;
    margin-bottom: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 60px;
    height: 60px;
    line-height: 1;
  }

  .upload-text {
    font-size: 14px;
    color: #ccc;
    margin-bottom: 5px;
  }

  .upload-hint {
    font-size: 10px;
    color: #666;
  }

  .file-input {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    opacity: 0;
    cursor: pointer;
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    padding: 15px 20px;
    border-top: 1px solid #555;
    background: #1a1a1a;
  }

  .modal-btn {
    padding: 8px 16px;
    font-family: inherit;
    font-size: 11px;
    cursor: pointer;
    border: 1px solid #555;
    background: #3a3a3a;
    color: #d4af37;
    transition: all 0.2s;
  }

  .modal-btn:hover {
    background: #4a4a4a;
  }

  .modal-btn.primary {
    background: #d4af37;
    color: #000;
    border-color: #d4af37;
  }

  .modal-btn.primary:hover {
    background: #e4bf47;
  }

  .modal-btn.cancel {
    background: #555;
    color: #ccc;
  }

  /* AI Assistant Specific Styles */
  .header-btn.ai-assistant {
    background: linear-gradient(135deg, #d4af37 0%, #f4cf47 100%);
    color: #000;
    border-color: #d4af37;
    font-weight: bold;
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
    animation: ai-pulse 2s infinite;
  }

  .header-btn.ai-assistant:hover {
    background: linear-gradient(135deg, #f4cf47 0%, #d4af37 100%);
    box-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
  }

  @keyframes ai-pulse {
    0% { box-shadow: 0 0 10px rgba(212, 175, 55, 0.3); }
    50% { box-shadow: 0 0 20px rgba(212, 175, 55, 0.6); }
    100% { box-shadow: 0 0 10px rgba(212, 175, 55, 0.3); }
  }

  .ai-modal {
    min-width: 500px;
    max-width: 600px;
  }

  .ai-content {
    padding: 20px;
  }

  .ai-status {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 15px 0;
    padding: 10px;
    background: #1a1a1a;
    border: 1px solid #d4af37;
  }

  .ai-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
  }

  .ai-indicator.active {
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
    animation: ai-blink 1.5s infinite;
  }

  @keyframes ai-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .ai-features {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    margin: 15px 0;
  }

  .ai-feature {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    background: #242424;
    border: 1px solid #555;
    font-size: 11px;
  }

  .ai-query-box {
    margin: 15px 0;
  }

  .ai-textarea {
    width: 100%;
    background: #1a1a1a;
    border: 1px solid #d4af37;
    color: #d4af37;
    padding: 10px;
    font-family: inherit;
    font-size: 11px;
    resize: vertical;
  }

  .ai-textarea:focus {
    outline: none;
    box-shadow: 0 0 8px rgba(212, 175, 55, 0.3);
  }

  .ai-quick-actions {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 8px;
    margin: 15px 0;
  }

  .ai-action-btn {
    padding: 8px 12px;
    background: #3a3a3a;
    border: 1px solid #555;
    color: #d4af37;
    font-family: inherit;
    font-size: 10px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .ai-action-btn:hover {
    background: #4a4a4a;
    border-color: #d4af37;
  }

  .modal-btn.ai-execute {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  /* Icon styling */
  .icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    width: 18px;
    height: 18px;
    line-height: 1;
  }
</style>