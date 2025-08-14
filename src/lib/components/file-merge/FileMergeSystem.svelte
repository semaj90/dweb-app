<!-- File Merge System UI Component -->
<!-- Native Windows Implementation -->

<script lang="ts">
  import { onMount } from 'svelte';
  import { createFileMergeSystem, fileMergeStore, type FileMetadata, type MergeOperation } from '$lib/services/file-merge-system.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '$lib/components/ui/Card/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';

  // Props
  interface Props {
    userId: string;
    caseId?: string;
  }

  let { userId, caseId }: Props = $props();

  // State
  let fileMergeSystem: ReturnType<typeof createFileMergeSystem>;
  let files = $state<FileMetadata[]>([]);
  let selectedFiles = $state<string[]>([]);
  let searchQuery = $state('');
  let searchResults = $state<Array<FileMetadata & { similarity: number }>>([]);
  let mergeType = $state<MergeOperation['mergeType']>('concatenate');
  let targetFilename = $state('');
  let uploadFiles = $state<FileList | null>(null);
  let dragOver = $state(false);
  let activeTab = $state('upload');

  // Reactive store values
  const store = $derived($fileMergeStore);

  onMount(async () => {
    fileMergeSystem = createFileMergeSystem();
    await loadFiles();
  });

  async function loadFiles() {
    try {
      // In a real implementation, you'd have an API endpoint to get files
      // For now, this is a placeholder
      files = [];
    } catch (error) {
      console.error('Failed to load files:', error);
    }
  }

  async function handleFileUpload(fileList: FileList) {
    for (const file of Array.from(fileList)) {
      try {
        const metadata = await fileMergeSystem.uploadFile(file, {
          userId,
          caseId,
          tags: {
            uploadSource: 'web-ui',
            originalSize: file.size,
            uploadedAt: new Date().toISOString()
          }
        });
        
        files = [...files, metadata];
      } catch (error) {
        console.error('Upload failed:', error);
      }
    }
  }

  async function handleMergeFiles() {
    if (selectedFiles.length < 2) {
      alert('Please select at least 2 files to merge');
      return;
    }

    if (!targetFilename.trim()) {
      alert('Please enter a target filename');
      return;
    }

    try {
      const operation = await fileMergeSystem.mergeFiles(
        selectedFiles,
        targetFilename,
        mergeType,
        { userId, caseId, tags: { mergedAt: new Date().toISOString() } }
      );

      if (operation.result) {
        files = [...files, operation.result.metadata];
        selectedFiles = [];
        targetFilename = '';
      }
    } catch (error) {
      console.error('Merge failed:', error);
    }
  }

  async function handleSearch() {
    if (!searchQuery.trim()) {
      searchResults = [];
      return;
    }

    try {
      searchResults = await fileMergeSystem.similaritySearch(searchQuery, {
        limit: 20,
        threshold: 0.7,
        caseId
      });
    } catch (error) {
      console.error('Search failed:', error);
      searchResults = [];
    }
  }

  async function downloadFile(fileId: string) {
    try {
      const stream = await fileMergeSystem.downloadFile(fileId);
      // Handle download in browser
      const response = new Response(stream);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      const metadata = files.find(f => f.id === fileId);
      const a = document.createElement('a');
      a.href = url;
      a.download = metadata?.originalPath || 'download';
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
    }
  }

  async function deleteFile(fileId: string) {
    if (!confirm('Are you sure you want to delete this file?')) return;

    try {
      await fileMergeSystem.deleteFile(fileId);
      files = files.filter(f => f.id !== fileId);
      selectedFiles = selectedFiles.filter(id => id !== fileId);
    } catch (error) {
      console.error('Delete failed:', error);
    }
  }

  function formatFileSize(bytes: number): string {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + ' ' + sizes[i];
  }

  function formatDate(date: Date): string {
    return new Date(date).toLocaleString();
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    dragOver = true;
  }

  function handleDragLeave() {
    dragOver = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    dragOver = false;
    if (event.dataTransfer?.files) {
      handleFileUpload(event.dataTransfer.files);
    }
  }

  function toggleFileSelection(fileId: string) {
    if (selectedFiles.includes(fileId)) {
      selectedFiles = selectedFiles.filter(id => id !== fileId);
    } else {
      selectedFiles = [...selectedFiles, fileId];
    }
  }
</script>

<div class="p-6 max-w-7xl mx-auto space-y-6">
  <h1 class="text-3xl font-bold text-gray-900 dark:text-white">
    File Merge System
  </h1>

  <!-- Tab Navigation -->
  <div class="border-b border-gray-200 dark:border-gray-700">
    <nav class="flex space-x-8">
      <button
        class="py-2 px-1 border-b-2 font-medium text-sm"
        class:border-blue-500={activeTab === 'upload'}
        class:text-blue-600={activeTab === 'upload'}
        class:border-transparent={activeTab !== 'upload'}
        class:text-gray-500={activeTab !== 'upload'}
        onclick={() => activeTab = 'upload'}
      >
        Upload & Manage
      </button>
      <button
        class="py-2 px-1 border-b-2 font-medium text-sm"
        class:border-blue-500={activeTab === 'merge'}
        class:text-blue-600={activeTab === 'merge'}
        class:border-transparent={activeTab !== 'merge'}
        class:text-gray-500={activeTab !== 'merge'}
        onclick={() => activeTab = 'merge'}
      >
        Merge Files
      </button>
      <button
        class="py-2 px-1 border-b-2 font-medium text-sm"
        class:border-blue-500={activeTab === 'search'}
        class:text-blue-600={activeTab === 'search'}
        class:border-transparent={activeTab !== 'search'}
        class:text-gray-500={activeTab !== 'search'}
        onclick={() => activeTab = 'search'}
      >
        Similarity Search
      </button>
      <button
        class="py-2 px-1 border-b-2 font-medium text-sm"
        class:border-blue-500={activeTab === 'operations'}
        class:text-blue-600={activeTab === 'operations'}
        class:border-transparent={activeTab !== 'operations'}
        class:text-gray-500={activeTab !== 'operations'}
        onclick={() => activeTab = 'operations'}
      >
        Operations
      </button>
    </nav>
  </div>

  <!-- Upload & Manage Tab -->
  {#if activeTab === 'upload'}
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Upload Section -->
      <Card class="lg:col-span-1">
        <CardHeader>
          <CardTitle>Upload Files</CardTitle>
          <CardDescription>
            Upload documents to MinIO storage with automatic vectorization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div 
            class="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center transition-colors"
            class:border-blue-500={dragOver}
            class:bg-blue-50={dragOver}
            ondragover={handleDragOver}
            ondragleave={handleDragLeave}
            ondrop={handleDrop}
          >
            {#if store.uploading}
              <div class="space-y-2">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p class="text-sm text-gray-600">Uploading...</p>
                <div class="w-full bg-gray-200 rounded-full h-2">
                  <div class="bg-blue-600 h-2 rounded-full" style="width: {store.progress}%"></div>
                </div>
              </div>
            {:else}
              <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
              <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
                Drag and drop files here, or 
                <label class="text-blue-600 hover:text-blue-500 cursor-pointer">
                  browse
                  <input 
                    type="file" 
                    multiple 
                    class="hidden" 
                    bind:files={uploadFiles}
                    onchange={() => uploadFiles && handleFileUpload(uploadFiles)}
                  />
                </label>
              </p>
              <p class="text-xs text-gray-500">Supports: PDF, DOCX, TXT, Images</p>
            {/if}
          </div>

          {#if store.error}
            <div class="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
              <p class="text-sm text-red-600">{store.error}</p>
            </div>
          {/if}
        </CardContent>
      </Card>

      <!-- Files List -->
      <Card class="lg:col-span-2">
        <CardHeader>
          <CardTitle>Files ({files.length})</CardTitle>
          <CardDescription>
            Manage your uploaded documents
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div class="space-y-3 max-h-96 overflow-y-auto">
            {#each files as file (file.id)}
              <div class="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800">
                <div class="flex items-center space-x-3">
                  <input 
                    type="checkbox" 
                    checked={selectedFiles.includes(file.id)}
                    onchange={() => toggleFileSelection(file.id)}
                    class="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                  />
                  <div>
                    <p class="font-medium text-sm">{file.originalPath}</p>
                    <div class="flex items-center space-x-2 text-xs text-gray-500">
                      <Badge variant="outline">{file.mimeType}</Badge>
                      <span>{formatFileSize(file.size)}</span>
                      <span>{formatDate(file.uploadedAt)}</span>
                      {#if file.embedding}
                        <Badge variant="default">Vectorized</Badge>
                      {/if}
                    </div>
                  </div>
                </div>
                <div class="flex items-center space-x-2">
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onclick={() => downloadFile(file.id)}
                  >
                    Download
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onclick={() => deleteFile(file.id)}
                    class="text-red-600 hover:text-red-700"
                  >
                    Delete
                  </Button>
                </div>
              </div>
            {:else}
              <div class="text-center py-8 text-gray-500">
                No files uploaded yet
              </div>
            {/each}
          </div>
        </CardContent>
      </Card>
    </div>
  {/if}

  <!-- Merge Files Tab -->
  {#if activeTab === 'merge'}
    <Card>
      <CardHeader>
        <CardTitle>Merge Selected Files</CardTitle>
        <CardDescription>
          Combine multiple files into a single document with various merge strategies
        </CardDescription>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium mb-2">Target Filename</label>
            <Input 
              bind:value={targetFilename}
              placeholder="merged-document.pdf"
              class="w-full"
            />
          </div>
          <div>
            <label class="block text-sm font-medium mb-2">Merge Type</label>
            <select bind:value={mergeType} class="w-full p-2 border border-gray-300 rounded-md">
              <option value="concatenate">Concatenate</option>
              <option value="overlay">Overlay</option>
              <option value="archive">Archive (ZIP)</option>
              <option value="legal-discovery">Legal Discovery Package</option>
            </select>
          </div>
        </div>

        <div>
          <p class="text-sm font-medium mb-2">
            Selected Files ({selectedFiles.length})
          </p>
          <div class="space-y-2 max-h-40 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded p-3">
            {#each selectedFiles as fileId}
              {@const file = files.find(f => f.id === fileId)}
              {#if file}
                <div class="flex items-center justify-between text-sm">
                  <span>{file.originalPath}</span>
                  <Badge variant="outline">{formatFileSize(file.size)}</Badge>
                </div>
              {/if}
            {:else}
              <p class="text-gray-500 text-sm">No files selected for merging</p>
            {/each}
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button 
          onclick={handleMergeFiles}
          disabled={selectedFiles.length < 2 || !targetFilename.trim()}
          class="w-full"
        >
          Merge Files
        </Button>
      </CardFooter>
    </Card>
  {/if}

  <!-- Similarity Search Tab -->
  {#if activeTab === 'search'}
    <Card>
      <CardHeader>
        <CardTitle>Document Similarity Search</CardTitle>
        <CardDescription>
          Search through vectorized documents using semantic similarity
        </CardDescription>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="flex space-x-2">
          <Input 
            bind:value={searchQuery}
            placeholder="Enter search query (e.g., 'contract liability terms')"
            class="flex-1"
            onkeydown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button onclick={handleSearch}>Search</Button>
        </div>

        <div class="space-y-3 max-h-96 overflow-y-auto">
          {#each searchResults as result (result.id)}
            <div class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <h4 class="font-medium text-sm">{result.originalPath}</h4>
                  <div class="flex items-center space-x-2 mt-1">
                    <Badge variant="default">
                      {Math.round(result.similarity * 100)}% match
                    </Badge>
                    <Badge variant="outline">{result.mimeType}</Badge>
                    <span class="text-xs text-gray-500">
                      {formatFileSize(result.size)}
                    </span>
                  </div>
                  {#if result.tags}
                    <div class="mt-2 flex flex-wrap gap-1">
                      {#each Object.entries(result.tags) as [key, value]}
                        <Badge variant="outline" class="text-xs">
                          {key}: {String(value).substring(0, 20)}
                        </Badge>
                      {/each}
                    </div>
                  {/if}
                </div>
                <Button 
                  variant="ghost" 
                  size="sm"
                  onclick={() => downloadFile(result.id)}
                >
                  Download
                </Button>
              </div>
            </div>
          {:else}
            {#if searchQuery}
              <div class="text-center py-8 text-gray-500">
                No similar documents found
              </div>
            {:else}
              <div class="text-center py-8 text-gray-500">
                Enter a search query to find similar documents
              </div>
            {/if}
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Operations Tab -->
  {#if activeTab === 'operations'}
    <Card>
      <CardHeader>
        <CardTitle>Merge Operations</CardTitle>
        <CardDescription>
          Track file merge operations and their status
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div class="space-y-3 max-h-96 overflow-y-auto">
          {#each store.operations as operation (operation.id)}
            <div class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <h4 class="font-medium text-sm">{operation.targetFilename}</h4>
                  <div class="flex items-center space-x-2 mt-1">
                    <Badge 
                      variant={operation.status === 'completed' ? 'default' : 
                              operation.status === 'failed' ? 'destructive' : 'secondary'}
                    >
                      {operation.status}
                    </Badge>
                    <Badge variant="outline">{operation.mergeType}</Badge>
                    <span class="text-xs text-gray-500">
                      {operation.sourceFiles.length} files
                    </span>
                  </div>
                  <p class="text-xs text-gray-500 mt-1">
                    Created: {formatDate(operation.createdAt)}
                  </p>
                  {#if operation.status === 'processing'}
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                      <div class="bg-blue-600 h-2 rounded-full" style="width: {operation.progress}%"></div>
                    </div>
                  {/if}
                </div>
                {#if operation.result}
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onclick={() => downloadFile(operation.result.fileId)}
                  >
                    Download Result
                  </Button>
                {/if}
              </div>
            </div>
          {:else}
            <div class="text-center py-8 text-gray-500">
              No merge operations yet
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  /* Custom styles for drag and drop */
  .drag-over {
    @apply border-blue-500 bg-blue-50;
  }
</style>