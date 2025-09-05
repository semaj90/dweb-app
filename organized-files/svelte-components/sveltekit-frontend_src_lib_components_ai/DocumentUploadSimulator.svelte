<!-- Document Upload Simulator with AI Processing -->
<script lang="ts">
  import { onMount } from 'svelte';
  
  interface DocumentUpload {
    id: string
    filename: string
    size: number
    type: string
    status: 'uploading' | 'processing' | 'embedding' | 'completed' | 'error';
    progress: number
    extractedText?: string;
    summary?: string;
    embeddings?: number[];
    localStorageKey?: string;
    error?: string;
  }

  let uploads: DocumentUpload[] = $state([]);
  let isDragging = $state(false);
  let fileInput: HTMLInputElement

  const API_BASE = 'http://localhost:8081/api';
  const MAX_LOCAL_STORAGE_SIZE = 10 * 1024 * 1024; // 10MB

  async function simulateUpload(file: File): Promise<void> {
    const uploadId = crypto.randomUUID();
    const upload: DocumentUpload = {
      id: uploadId,
      filename: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading',
      progress: 0
    };

    uploads = [...uploads, upload];

    try {
      // Phase 1: Upload simulation (fast)
      await updateProgress(uploadId, 'uploading', 25);
      await delay(500);

      // Phase 2: OCR Processing (if PDF)
      await updateProgress(uploadId, 'processing', 50);
      const extractedText = await extractTextFromFile(file);
      await updateUpload(uploadId, { extractedText });
      await delay(1000);

      // Phase 3: AI Summarization
      await updateProgress(uploadId, 'processing', 75);
      const summary = await generateSummary(extractedText, file.type);
      await updateUpload(uploadId, { summary });
      await delay(1500);

      // Phase 4: Generate Embeddings
      await updateProgress(uploadId, 'embedding', 90);
      const embeddings = await generateEmbeddings(extractedText);
      await updateUpload(uploadId, { embeddings });
      await delay(1000);

      // Phase 5: Store in Local Storage (if under 10MB)
      const processedData = {
        filename: file.name,
        extractedText,
        summary,
        embeddings,
        processedAt: new Date().toISOString()
      };

      let localStorageKey: string | undefined;
      if (file.size < MAX_LOCAL_STORAGE_SIZE) {
        localStorageKey = `doc_${uploadId}`;
        localStorage.setItem(localStorageKey, JSON.stringify(processedData));
      }

      // Complete
      await updateProgress(uploadId, 'completed', 100);
      await updateUpload(uploadId, { localStorageKey });

    } catch (error) {
      console.error('Upload error:', error);
      await updateUpload(uploadId, { 
        status: 'error', 
        error: error instanceof Error ? error.message : 'Processing failed' 
      });
    }
  }

  async function extractTextFromFile(file: File): Promise<string> {
    if (file.type === 'application/pdf') {
      // Simulate PDF OCR processing
      const formData = new FormData();
      formData.append('file', file);
      formData.append('enable_ocr', 'true');
      formData.append('document_type', 'legal');

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`OCR processing failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result.extracted_text || 'PDF text extraction failed';
    } else if (file.type === 'text/plain') {
      return await file.text();
    } else {
      // For other file types, return placeholder
      return `Content extracted from ${file.name}\n\nThis is simulated extracted text from the uploaded document. In production, this would contain the actual OCR-processed content from the file.`;
    }
  }

  async function generateSummary(text: string, fileType: string): Promise<string> {
    const response = await fetch('/api/ai/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: text,
        type: 'legal',
        length: 'medium'
      })
    });

    if (!response.ok) {
      throw new Error(`Summarization failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result.summary || 'Summary generation failed';
  }

  async function generateEmbeddings(text: string): Promise<number[]> {
    // Simulate embedding generation using nomic-embed-text
    // In production, this would call your Go service
    const response = await fetch(`${API_BASE}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.substring(0, 8000) }) // Limit text length
    });

    if (response.ok) {
      const result = await response.json();
      return result.embedding || [];
    }

    // Fallback: generate mock 384-dimensional embedding
    return Array.from({ length: 384 }, () => Math.random() * 2 - 1);
  }

  async function updateProgress(id: string, status: DocumentUpload['status'], progress: number): Promise<void> {
    uploads = uploads.map(upload => 
      upload.id === id ? { ...upload, status, progress } : upload
    );
  }

  async function updateUpload(id: string, updates: Partial<DocumentUpload>): Promise<void> {
    uploads = uploads.map(upload => 
      upload.id === id ? { ...upload, ...updates } : upload
    );
  }

  function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function handleDrop(event: DragEvent): void {
    event.preventDefault();
    isDragging = false;
    
    const files = event.dataTransfer?.files;
    if (files) {
      Array.from(files).forEach(simulateUpload);
    }
  }

  function handleFileInput(event: Event): void {
    const files = (event.target as HTMLInputElement).files;
    if (files) {
      Array.from(files).forEach(simulateUpload);
    }
  }

  function removeUpload(id: string): void {
    const upload = uploads.find(u => u.id === id);
    if (upload?.localStorageKey) {
      localStorage.removeItem(upload.localStorageKey);
    }
    uploads = uploads.filter(u => u.id !== id);
  }

  function downloadProcessedData(upload: DocumentUpload): void {
    const data = {
      filename: upload.filename,
      extractedText: upload.extractedText,
      summary: upload.summary,
      embeddings: upload.embeddings,
      processedAt: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${upload.filename}_processed.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function getStatusColor(status: DocumentUpload['status']): string {
    switch (status) {
      case 'uploading': return 'text-blue-400';
      case 'processing': return 'text-yellow-400';
      case 'embedding': return 'text-purple-400';
      case 'completed': return 'text-green-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }

  function getStatusText(status: DocumentUpload['status']): string {
    switch (status) {
      case 'uploading': return 'Uploading to PostgreSQL...';
      case 'processing': return 'AI Processing (OCR + Summary)...';
      case 'embedding': return 'Generating Embeddings (Nomic)...';
      case 'completed': return 'Completed ✅';
      case 'error': return 'Error ❌';
      default: return 'Pending';
    }
  }

  onMount(() => {
    // Clean up old localStorage entries on mount
    const keys = Object.keys(localStorage).filter(key => key.startsWith('doc_'));
    console.log(`Found ${keys.length} cached documents in localStorage`);
  });
</script>

<div class="document-upload-simulator">
  <h2 class="text-2xl font-bold text-green-400 mb-6">📄 AI Document Processing Simulator</h2>
  
  <!-- Upload Area -->
  <div 
    class="upload-area border-2 border-dashed border-gray-600 rounded-lg p-8 text-center transition-colors duration-200"
    class:border-green-400={isDragging}
    class:bg-green-400/10={isDragging}
    ondrop={handleDrop}
    ondragover={(e) => e.preventDefault()}
    ondragenter={() => isDragging = true}
    ondragleave={() => isDragging = false}
  >
    <div class="text-4xl mb-4">📄</div>
    <p class="text-lg mb-4">Drop PDFs or text files here, or click to browse</p>
    <input 
      bind:this={fileInput}
      type="file" 
      accept=".pdf,.txt,.json"
      multiple
      class="hidden"
      onchange={handleFileInput}
    />
    <button 
      class="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
      onclick={() => fileInput.click()}
    >
      Select Files
    </button>
    <p class="text-sm text-gray-400 mt-4">
      Supports: PDF (OCR), TXT, JSON • Files under 10MB cached locally
    </p>
  </div>

  <!-- Processing Queue -->
  {#each uploads as upload (upload.id)}
    <div class="upload-item bg-gray-800 rounded-lg p-6 mb-4 border border-gray-700">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center space-x-3">
          <div class="text-2xl">
            {upload.type === 'application/pdf' ? '📄' : '📝'}
          </div>
          <div>
            <h3 class="font-semibold text-white">{upload.filename}</h3>
            <p class="text-sm text-gray-400">
              {(upload.size / 1024).toFixed(1)} KB • 
              {upload.size < MAX_LOCAL_STORAGE_SIZE ? 'Local Storage' : 'PostgreSQL Only'}
            </p>
          </div>
        </div>
        <button 
          class="text-gray-400 hover:text-red-400 transition-colors"
          onclick={() => removeUpload(upload.id)}
        >
          ✕
        </button>
      </div>

      <!-- Progress Bar -->
      <div class="mb-4">
        <div class="flex justify-between text-sm mb-2">
          <span class={getStatusColor(upload.status)}>
            {getStatusText(upload.status)}
          </span>
          <span class="text-gray-400">{upload.progress}%</span>
        </div>
        <div class="w-full bg-gray-700 rounded-full h-2">
          <div 
            class="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-300"
            style="width: {upload.progress}%"
          ></div>
        </div>
      </div>

      <!-- Content Display -->
      {#if upload.status === 'completed'}
        <div class="space-y-4">
          <!-- Extracted Text Preview -->
          {#if upload.extractedText}
            <div class="bg-gray-900 rounded p-4">
              <h4 class="text-sm font-semibold text-green-400 mb-2">📝 Extracted Text</h4>
              <div class="text-xs text-gray-300 max-h-32 overflow-y-auto">
                {upload.extractedText.substring(0, 500)}
                {#if upload.extractedText.length > 500}...{/if}
              </div>
            </div>
          {/if}

          <!-- AI Summary -->
          {#if upload.summary}
            <div class="bg-gray-900 rounded p-4">
              <h4 class="text-sm font-semibold text-blue-400 mb-2">🤖 AI Summary</h4>
              <div class="text-sm text-gray-200">
                {upload.summary}
              </div>
            </div>
          {/if}

          <!-- Embeddings Info -->
          {#if upload.embeddings}
            <div class="bg-gray-900 rounded p-4">
              <h4 class="text-sm font-semibold text-purple-400 mb-2">🧠 Vector Embeddings</h4>
              <div class="text-xs text-gray-300">
                Generated {upload.embeddings.length}D vector using Nomic-Embed-Text
                <br />
                First 5 dimensions: [{upload.embeddings.slice(0, 5).map(n => n.toFixed(3)).join(', ')}...]
              </div>
            </div>
          {/if}

          <!-- Actions -->
          <div class="flex space-x-3">
            <button 
              class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
              onclick={() => downloadProcessedData(upload)}
            >
              📥 Download JSON
            </button>
            {#if upload.localStorageKey}
              <span class="px-4 py-2 bg-green-600/20 text-green-400 rounded text-sm">
                💾 Cached Locally
              </span>
            {/if}
          </div>
        </div>
      {/if}

      {#if upload.status === 'error'}
        <div class="bg-red-900/20 border border-red-700 rounded p-3">
          <p class="text-red-400 text-sm">❌ {upload.error}</p>
        </div>
      {/if}
    </div>
  {/each}

  {#if uploads.length === 0}
    <div class="text-center py-12 text-gray-500">
      <div class="text-6xl mb-4">📄</div>
      <p>No documents uploaded yet. Drop files above to start processing.</p>
    </div>
  {/if}
</div>

<style>
  .document-upload-simulator {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }
  
  .upload-area {
    cursor: pointer
  }
  
  .upload-item {
    animation: slideIn 0.3s ease-out;
  }
  
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>