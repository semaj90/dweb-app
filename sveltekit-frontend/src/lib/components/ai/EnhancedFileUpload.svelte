<!-- Enhanced File Upload with Real OCR, Embeddings, and Database Integration -->
<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { createUploadMachine } from '$lib/machines/uploadMachine';
  import type { ProcessingPipeline } from '$lib/types/upload';
  import { toast } from '$lib/utils/toast';
  import { Check, FileText, Loader2, Search, Upload, X } from 'lucide-svelte';
  import { onMount } from 'svelte';
  import { createActor } from 'xstate';

  // Props interface
  interface Props extends CommonProps {
    onUploadComplete?: (doc: any) => void;
    accept?: string;
    maxSize?: number;
    enableOCR?: boolean;
    enableEmbedding?: boolean;
    enableRAG?: boolean;
    class?: string;
  }

  // Svelte 5 props with defaults
  let {
    onUploadComplete = () => {},
    accept = '.pdf,.docx,.txt,.jpg,.png,.tiff',
    maxSize = 50 * 1024 * 1024, // 50MB
    enableOCR = true,
    enableEmbedding = true,
    enableRAG = true,
    class = '',
  }: Props = $props();

  // State variables
  let files = $state<File[]>([]);
  let fileStates = $state<Map<string, any>>(new Map());
  let searchQuery = $state('');
  let searchResults = $state<any[]>([]);
  let isSearching = $state(false);
  let systemStatus = $state<any>({});

  // === MCP INTEGRATION LAYER ===
  const MCP_ENDPOINTS = {
    process: '/rag/process',
    status: '/rag/status',
    search: '/rag/search',
  } as const;

  let statusSocket: WebSocket | null = null;

  function connectStatusSocket() {
    try {
      const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/rag/ws/uploads`;
      statusSocket = new WebSocket(wsUrl);
      statusSocket.onopen=() => console.debug('[UploadWS] connected');
      statusSocket.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data || '{}');
          if (!msg.filename) return;
          const entryId = Array.from(fileStates.keys()).find(
            (id) => fileStates.get(id)?.name === msg.filename
          );
          if (entryId) {
            const current = fileStates.get(entryId);
            fileStates.set(entryId, {
              ...current,
              progress: typeof msg.progress === 'number' ? msg.progress : current.progress,
              status: msg.status || current.status,
              error: msg.error || current.error,
              documentId: msg.documentId || current.documentId,
            });
            fileStates = new Map(fileStates);
          }
        } catch (e) {
          console.warn('[UploadWS] parse failed', e);
        }
      }
      statusSocket.onerror = (e) => console.warn('[UploadWS] error', e);
      statusSocket.onclose = () => {
        console.debug('[UploadWS] closed – retrying in 5s');
        setTimeout(() => connectStatusSocket(), 5000);
      }
    } catch (e) {
      console.warn('[UploadWS] init failed', e);
    }
  }

  async function checkSystemStatus() {
    const status = { ocr: false, embeddings: false, search: false, storage: false }
    try {
      const aggregate = await fetch(MCP_ENDPOINTS.status).catch(() => null);
      if (aggregate?.ok) {
        const json = await aggregate.json().catch(() => ({}));
        status.ocr = !!json.ocr?.healthy || !!json.ocr;
        status.embeddings = !!json.embeddings?.healthy || !!json.embeddings;
        status.search = !!json.search?.healthy || !!json.search;
        status.storage = !!json.storage?.healthy || !!json.storage;
      } else {
        const probes: [keyof typeof status, string][] = [
          ['ocr', '/rag/ocr/health'],
          ['embeddings', '/rag/embeddings/health'],
          ['search', MCP_ENDPOINTS.search],
          ['storage', MCP_ENDPOINTS.process],
        ];
        await Promise.all(
          probes.map(async ([k, url]) => {
            try {
              const r = await fetch(url);
              if (r.ok) status[k] = true;
            } catch {
              /* ignore */
            }
          })
        );
      }
    } catch (e) {
      console.warn('System status check failed:', e);
    }
    systemStatus = status;
  }

  // Initialize basic pipeline config (placeholder until prop wiring)
  const basePipeline: ProcessingPipeline = {
    gpu: {
      enabled: enableOCR,
      webgpuEnabled: false,
      accelerateOCR: true,
      accelerateEmbedding: true,
    },
    rag: {
      enabled: enableRAG,
      extractText: true,
      generateEmbeddings: true,
      storeVectors: true,
      updateIndex: true,
    },
    ocr: { enabled: enableOCR, engines: ['tesseract'], languages: ['eng'] },
    yolo: { enabled: false },
  } as any;
  const uploadMachineActor = createActor(createUploadMachine(basePipeline));
  uploadMachineActor.start();

  // File upload handler with real RAG processing
  async function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;

    const incoming: File[] = Array.from(input.files);

    // Validate files
    const validFiles = incoming.filter(file => {
      if (file.size > maxSize) {
        toast.error(`File ${file.name} is too large. Max size: ${formatFileSize(maxSize)}`);
        return false;
      }
      return true;
    });

    if (validFiles.length === 0) return;

    // Add files to state
    files = [...files, ...validFiles];

    // Process files through RAG pipeline
    try {
      const formData = new FormData();
      validFiles.forEach(file => formData.append('files', file));
      formData.append('enableOCR', enableOCR.toString());
      formData.append('enableEmbedding', enableEmbedding.toString());
      formData.append('enableRAG', enableRAG.toString());

      // Update file states to show processing
      validFiles.forEach(file => {
        const fileId = `${file.name}-${Date.now()}`;
        fileStates.set(fileId, {
          name: file.name,
          size: file.size,
          progress: 0,
          status: 'uploading'
        });
      });
      fileStates = new Map(fileStates);

      const response = await fetch('/api/rag/process', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        toast.success(`Successfully processed ${result.successfulUploads} of ${result.totalFiles} files`);

        // Update file states with results
        result.results.forEach((fileResult: any) => {
          const fileId = Array.from(fileStates.keys()).find(
            id => fileStates.get(id)?.name === fileResult.filename
          );
          if (fileId) {
            fileStates.set(fileId, {
              ...fileStates.get(fileId),
              progress: fileResult.status === 'processed' ? 100 : -1,
              status: fileResult.status,
              documentId: fileResult.documentId,
              error: fileResult.error
            });
          }
        });
        fileStates = new Map(fileStates);

        // Call completion callback
        onUploadComplete(result);
      } else {
        const error = await response.json();
        toast.error(`Upload failed: ${error.error}`);

        // Mark all files as failed
        validFiles.forEach(file => {
          const fileId = Array.from(fileStates.keys()).find(
            id => fileStates.get(id)?.name === file.name
          );
          if (fileId) {
            fileStates.set(fileId, {
              ...fileStates.get(fileId),
              progress: -1,
              status: 'error',
              error: error.error
            });
          }
        });
        fileStates = new Map(fileStates);
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(`Upload failed: ${error.message}`);
    }

    // Clear input
    input.value = '';
  }

  // Update file state
  function updateFileState(fileId: string, updates: any) {
    const current = fileStates.get(fileId);
    if (current) {
      fileStates.set(fileId, { ...current, ...updates });
      fileStates = new Map(fileStates);
    }
  }

  // Semantic search with real API
  async function handleSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    try {
      const searchResponse = await fetch(MCP_ENDPOINTS.search, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          searchType: 'hybrid',
          limit: 10,
          threshold: 0.7,
        }),
      });

      if (searchResponse.ok) {
        const result = await searchResponse.json();
        if (result.success) {
          searchResults = result.results || [];
          toast.success(`Found ${searchResults.length} results`);
        } else {
          throw new Error(result.error || 'Search failed');
        }
      } else {
        throw new Error('Search service unavailable');
      }
    } catch (error) {
      console.error('Search failed:', error);
      toast.error(`Search failed: ${error.message}`);
    } finally {
      isSearching = false;
    }
  }

  // Helper functions
  function formatFileSize(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  function getStatusColor(progress: number): string {
    if (progress === -1) return 'text-red-500';
    if (progress === 100) return 'text-green-500';
    return 'text-blue-500';
  }

  function getProgressColor(progress: number): string {
    if (progress === -1) return 'bg-red-500';
    if (progress === 100) return 'bg-green-500';
    return 'bg-blue-500';
  }

  // Auto-search effect
  $effect(() => {
    if (searchQuery.length > 2) {
      const timer = setTimeout(handleSearch, 300);
      return () => clearTimeout(timer);
    }
  });

  // Mount lifecycle: connect WebSocket + initial status
  onMount(async () => {
    connectStatusSocket();
    await checkSystemStatus();
  });

  const machineState = $state<any>(uploadMachineActor.getSnapshot());
  uploadMachineActor.subscribe((sn) => { machineState.value = sn; });
  function getEntries() { return machineState.value?.context?.files || []; }
</script>

<div class="enhanced-file-upload {className}">
  <!-- System Status -->
  <div class="system-status mb-4 grid grid-cols-2 md:grid-cols-4 gap-2">
    <div
      class="status-item {systemStatus.ocr
        ? 'bg-green-50 border-green-200'
        : 'bg-red-50 border-red-200'}">
      <span class="text-xs font-medium {systemStatus.ocr ? 'text-green-800' : 'text-red-800'}">
        OCR {systemStatus.ocr ? '✓' : '✗'}
      </span>
    </div>
    <div
      class="status-item {systemStatus.embeddings
        ? 'bg-green-50 border-green-200'
        : 'bg-red-50 border-red-200'}">
      <span
        class="text-xs font-medium {systemStatus.embeddings ? 'text-green-800' : 'text-red-800'}">
        Embeddings {systemStatus.embeddings ? '✓' : '✗'}
      </span>
    </div>
    <div
      class="status-item {systemStatus.search
        ? 'bg-green-50 border-green-200'
        : 'bg-red-50 border-red-200'}">
      <span class="text-xs font-medium {systemStatus.search ? 'text-green-800' : 'text-red-800'}">
        Search {systemStatus.search ? '✓' : '✗'}
      </span>
    </div>
    <div
      class="status-item {systemStatus.storage
        ? 'bg-green-50 border-green-200'
        : 'bg-red-50 border-red-200'}">
      <span class="text-xs font-medium {systemStatus.storage ? 'text-green-800' : 'text-red-800'}">
        Storage {systemStatus.storage ? '✓' : '✗'}
      </span>
    </div>
  </div>

  <!-- Upload Area -->
  <div class="upload-area">
    <input
      type="file"
      {accept}
      multiple
      onchange={handleFileUpload}
      class="hidden"
      id="file-input" />
    <label for="file-input" class="upload-label">
      <Upload class="w-12 h-12 mb-4 text-gray-400" />
      <p class="text-lg font-medium">Drop files here or click to upload</p>
      <p class="text-sm text-gray-500 mt-2">
        Supports: PDF, DOCX, TXT, Images with real OCR processing
      </p>
      <p class="text-xs text-gray-400 mt-1">
        Max size: {formatFileSize(maxSize)}
      </p>
    </label>
  </div>

  <!-- File Processing Status -->
  {#if getEntries().length > 0}
    <div class="file-list mt-6">
      <h3 class="text-lg font-semibold mb-4">Processing Files ({getEntries().length})</h3>
      {#each getEntries() as state (state.id)}
        <div class="file-item">
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center space-x-3">
              <FileText class="w-5 h-5 text-blue-500" />
              <div>
                <p class="font-medium">{state.file.name}</p>
                <p class="text-sm {getStatusColor(state.progress)}">{state.status}</p>
              </div>
            </div>
            <div class="flex items-center space-x-2">
              {#if state.status === 'complete'}
                <Check class="w-5 h-5 text-green-500" />
              {:else if state.status === 'error'}
                <X class="w-5 h-5 text-red-500" />
              {:else}
                <Loader2 class="w-5 h-5 animate-spin text-blue-500" />
              {/if}
              <span class="text-sm font-medium {getStatusColor(state.progress)}">
                {state.status === 'error' ? 'Error' : `${state.progress}%`}
              </span>
            </div>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div
              class="{getProgressColor(state.progress)} h-2 rounded-full transition-all duration-300"
              style="width: {Math.max(0, state.progress)}%">
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Semantic Search -->
  {#if enableEmbedding && systemStatus.search}
    <div class="search-section mt-8">
      <h3 class="text-lg font-semibold mb-4">Semantic Document Search</h3>
      <div class="flex space-x-2">
        <input
          type="text"
          bind:value={searchQuery}
          placeholder="Search uploaded documents with AI..."
          class="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" />
        <button
          onclick={handleSearch}
          disabled={isSearching || !searchQuery.trim()}
          class="px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600 flex items-center gap-2">
          {#if isSearching}
            <Loader2 class="w-4 h-4 animate-spin" />
          {:else}
            <Search class="w-4 h-4" />
          {/if}
          Search
        </button>
      </div>

      <!-- Search Results -->
      {#if searchResults.length > 0}
        <div class="search-results mt-4">
          <h4 class="font-medium mb-2">Results ({searchResults.length})</h4>
          <div class="space-y-3">
            {#each searchResults as result}
              <div class="result-item p-4 border rounded-lg hover:bg-gray-50">
                <div class="flex items-start justify-between">
                  <div class="flex-1">
                    <p class="font-medium">{result.filename}</p>
                    <p class="text-sm text-gray-600 mt-1">
                      {result.excerpt}
                    </p>
                    <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
                      <span>Similarity: {(result.similarity * 100).toFixed(1)}%</span>
                      <span>Type: {result.searchType}</span>
                      {#if result.matchedBy}
                        <span>Matched by: {result.matchedBy.join(', ')}</span>
                      {/if}
                    </div>
                  </div>
                  <button class="text-sm text-blue-500 hover:text-blue-700 ml-4">
                    View Details
                  </button>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .enhanced-file-upload {
    padding: 1.5rem;
    background: #fff;
    border-radius: 0.5rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  }
  .system-status .status-item {
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    border: 1px solid #e5e7eb;
    text-align: center;
  }
  .upload-area {
    border: 2px dashed #d1d5db;
    border-radius: 0.5rem;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .upload-area:hover {
    border-color: #60a5fa;
  }
  .upload-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  .file-item {
    padding: 1rem;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    margin-bottom: 0.75rem;
    background: #f9fafb;
  }
  .search-section {
    border-top: 1px solid #e5e7eb;
    padding-top: 1.5rem;
  }
  .result-item {
    cursor: pointer;
    transition: background-color 0.15s;
  }
  .result-item:hover {
    background: #f9fafb;
  }
  .hidden {
    display: none;
  }
</style>
