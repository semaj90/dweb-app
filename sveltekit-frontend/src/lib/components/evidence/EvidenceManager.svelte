<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Badge } from '$lib/components/ui/badge';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Upload, Search, Tag, FileText, Image, Video, Music, Download, Eye, Calendar, User } from 'lucide-svelte';

  interface Evidence {
    id: string;
    title: string;
    description?: string;
    file_name: string;
    file_type?: string;
    file_size?: number;
    uploaded_at: string;
    tags?: string[];
    case_id: string;
    uploaded_by?: string;
  }

  // Props
  export let caseId: string;
  export let onEvidenceSelect: (evidence: Evidence) => void = () => {};
  export let allowUpload: boolean = true;

  // State
  let evidence: Evidence[] = [];
  let loading = false;
  let uploadLoading = false;
  let searchQuery = '';
  let selectedTags: string[] = [];
  let filteredEvidence: Evidence[] = [];

  // Simple modal state (replaces melt dialog usage)
  let showUpload = false;

  // File upload state
  let fileInput: HTMLInputElement | undefined;
  let uploadFiles: FileList | null = null;
  let uploadTitle = '';
  let uploadDescription = '';
  let dragActive = false;

  // Available tag filters
  const availableTags = [
    'pdf', 'document', 'image', 'video', 'audio', 'contract',
    'legal-document', 'invoice', 'financial', 'email', 'communication',
    'photograph', 'evidence', 'scanned', 'multi-page', 'encrypted',
    'digitally-signed', 'large-file', 'small-file', 'needs-review'
  ];

  // Load evidence for case
  async function loadEvidence() {
    if (!caseId) return;

    loading = true;
    try {
      const response = await fetch(`/api/cases/${caseId}/evidence`);
      if (response.ok) {
        const data = await response.json();
        evidence = data.evidence || [];
      } else {
        console.error('Failed to load evidence:', response.statusText);
      }
    } catch (error) {
      console.error('Failed to load evidence:', error);
    } finally {
      loading = false;
    }
  }

  // Upload evidence
  async function uploadEvidence() {
    if (!uploadFiles || uploadFiles.length === 0) return;

    const formData = new FormData();
    formData.append('case_id', caseId);
    formData.append('title', uploadTitle);
    formData.append('description', uploadDescription);

    for (let i = 0; i < uploadFiles.length; i++) {
      formData.append('files', uploadFiles[i]);
    }

    try {
      uploadLoading = true;
      const response = await fetch('/api/evidence/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();

        // Trigger AI auto-tagging in background
        if (result.evidenceIds) {
          fetch('/api/ai/auto-tag-evidence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              caseId,
              evidenceIds: result.evidenceIds
            })
          }).catch(console.error); // Don't block upload on auto-tag failure
        }

        // Reset form and reload
        resetUploadForm();
        showUpload = false;
        await loadEvidence();
      } else {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed. Please try again.');
    } finally {
      uploadLoading = false;
    }
  }

  function resetUploadForm() {
    uploadTitle = '';
    uploadDescription = '';
    uploadFiles = null;
    if (fileInput) fileInput.value = '';
  }

  // Filter evidence based on search and tags
  $: filteredEvidence = evidence.filter((item) => {
    const matchesSearch = !searchQuery ||
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.file_name.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesTags = selectedTags.length === 0 ||
      selectedTags.some(tag => item.tags?.includes(tag));

    return matchesSearch && matchesTags;
  });

  // Get file type icon
  function getFileIcon(fileType: string | null) {
    if (!fileType) return FileText;
    if (fileType.includes('image')) return Image;
    if (fileType.includes('video')) return Video;
    if (fileType.includes('audio')) return Music;
    return FileText;
  }

  // Format file size
  function formatFileSize(bytes: number | null): string {
    if (!bytes) return 'Unknown';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  // Toggle tag filter
  function toggleTag(tag: string) {
    if (selectedTags.includes(tag)) {
      selectedTags = selectedTags.filter(t => t !== tag);
    } else {
      selectedTags = [...selectedTags, tag];
    }
  }

  // Handle drag and drop
  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dragActive = true;
  }

  function handleDragLeave() {
    dragActive = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragActive = false;
    uploadFiles = e.dataTransfer?.files || null;
  }

  // Evidence actions
  function downloadEvidence(item: Evidence) {
    window.open(`/api/evidence/${item.id}/download`, '_blank');
  }

  function viewEvidence(item: Evidence) {
    onEvidenceSelect(item);
  }

  // Clear all filters
  function clearFilters() {
    searchQuery = '';
    selectedTags = [];
  }

  // Load evidence on mount and when caseId changes
  onMount(loadEvidence);

  $: if (caseId) {
    loadEvidence();
  }
</script>

<div class="evidence-manager space-y-6">
  <!-- Header -->
  <div class="flex justify-between items-center">
    <div>
      <h2 class="text-2xl font-bold text-gray-900 dark:text-white">Case Evidence</h2>
      <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
        {filteredEvidence.length} of {evidence.length} items
      </p>
    </div>
    {#if allowUpload}
      <Button on:click={() => (showUpload = true)} class="gap-2">
        <Upload class="w-4 h-4" />
        Upload Evidence
      </Button>
    {/if}
  </div>

  <!-- Search and Filters -->
  <Card>
    <CardHeader>
      {#if allowUpload}
        <Button on:click={() => (showUpload = true)} class="gap-2">
          <Upload class="w-4 h-4" />
          Upload Evidence
        </Button>
      {/if}
    </CardHeader>
    <CardContent>
      <!-- Search Input -->
      <div class="flex gap-2">
        <div class="relative flex-1">
          <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            bind:value={searchQuery}
            placeholder="Search evidence by title, description, or filename..."
            class="pl-10"
          />
        </div>
        {#if searchQuery || selectedTags.length > 0}
          <Button on:click={clearFilters} variant="outline" size="sm">
            Clear Filters
          </Button>
        {/if}
      </div>

      <!-- Tag Filters -->
      <div class="flex flex-wrap gap-2 mt-3">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Filter by tags:</span>
        {#each availableTags as tag}
          <Badge
            variant={selectedTags.includes(tag) ? 'default' : 'outline'}
            class="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            on:click={() => toggleTag(tag)}
          >
            <Tag class="w-3 h-3 mr-1" />
            {tag}
          </Badge>
        {/each}
      </div>
    </CardContent>
  </Card>

  <!-- Evidence Grid -->
  {#if loading}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {#each Array(6) as _, i}
        <Card class="animate-pulse">
          <CardHeader class="pb-3">
            <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
          </CardHeader>
          <CardContent class="space-y-3">
            <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
            <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
            <div class="flex gap-2">
              <div class="h-8 bg-gray-200 dark:bg-gray-700 rounded flex-1"></div>
              <div class="h-8 bg-gray-200 dark:bg-gray-700 rounded flex-1"></div>
            </div>
          </CardContent>
        </Card>
      {/each}
    </div>
  {:else}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {#each filteredEvidence as item (item.id)}
        <Card class="hover:shadow-md transition-all duration-200 border-l-4 border-l-blue-500">
          <CardHeader class="pb-3">
            <CardTitle class="flex items-center gap-2 text-lg">
              <svelte:component this={getFileIcon(item.file_type)} class="w-5 h-5 text-blue-600" />
              <span class="truncate">{item.title}</span>
            </CardTitle>
          </CardHeader>
          <CardContent class="space-y-3">
            <!-- File Name -->
            <div class="text-sm text-gray-600 dark:text-gray-400 font-mono bg-gray-50 dark:bg-gray-800 p-2 rounded text-xs">
              📁 {item.file_name}
            </div>

            {#if item.description}
              <p class="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">{item.description}</p>
            {/if}

            <!-- File Info -->
            <div class="text-xs text-gray-500 dark:text-gray-500 space-y-1 bg-gray-50 dark:bg-gray-800 p-3 rounded">
              <div class="flex items-center gap-2">
                <FileText class="w-3 h-3" />
                <span>Type: {item.file_type || 'Unknown'}</span>
              </div>
              <div class="flex items-center gap-2">
                <span>📊</span>
                <span>Size: {formatFileSize(item.file_size)}</span>
              </div>
              <div class="flex items-center gap-2">
                <Calendar class="w-3 h-3" />
                <span>Uploaded: {new Date(item.uploaded_at).toLocaleDateString()}</span>
              </div>
              {#if item.uploaded_by}
                <div class="flex items-center gap-2">
                  <User class="w-3 h-3" />
                  <span>By: {item.uploaded_by}</span>
                </div>
              {/if}
            </div>

            <!-- Tags -->
            {#if item.tags && item.tags.length > 0}
              <div class="flex flex-wrap gap-1">
                {#each item.tags as tag}
                  <Badge variant="secondary" class="text-xs">
                    <Tag class="w-3 h-3 mr-1" />
                    {tag}
                  </Badge>
                {/each}
              </div>
            {/if}

            <!-- Actions -->
            <div class="flex gap-2 pt-2">
              <Button
                on:click={() => viewEvidence(item)}
                size="sm"
                variant="default"
                class="flex-1 gap-2"
              >
                <Eye class="w-4 h-4" />
                View
              </Button>
              <Button
                on:click={() => downloadEvidence(item)}
                size="sm"
                variant="outline"
                class="gap-2"
              >
                <Download class="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      {/each}
    </div>
  {/if}

  {#if !loading && filteredEvidence.length === 0}
    <div class="text-center py-12">
      {#if evidence.length === 0}
        <FileText class="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No evidence yet</h3>
        <p class="text-gray-600 dark:text-gray-400 mb-4">Upload the first piece of evidence to get started.</p>
        {#if allowUpload}
          <Button on:click={() => (showUpload = true)} variant="outline" class="gap-2">
            <Upload class="w-4 h-4" />
            Upload First Evidence
          </Button>
        {/if}
      {:else}
        <Search class="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No matches found</h3>
        <p class="text-gray-600 dark:text-gray-400 mb-4">Try adjusting your search or filters.</p>
        <Button on:click={clearFilters} variant="outline">Clear Filters</Button>
        {#if allowUpload}
          <div class="mt-4">
            <Button on:click={() => (showUpload = true)} variant="outline" class="gap-2">
              <Upload class="w-4 h-4" />
              Upload Evidence
            </Button>
          </div>
        {/if}
      {/if}
    </div>
  {/if}

  <!-- Upload Modal -->
  {#if showUpload}
    <div class="fixed inset-0 z-50 bg-black/50" on:click={() => { if (!uploadLoading) showUpload = false; }} />
    <div
      class="fixed left-[50%] top-[50%] z-50 max-h-[85vh] w-[90vw] max-w-[600px] -translate-x-1/2 -translate-y-1/2 rounded-lg bg-white dark:bg-gray-800 p-6 shadow-xl border border-gray-200 dark:border-gray-700"
      on:click|stopPropagation
    >
      <h2 class="text-xl font-semibold mb-6 text-gray-900 dark:text-white">Upload Evidence</h2>

      <div class="space-y-6">
        <!-- File Drop Zone -->
        <div
          class="border-2 border-dashed rounded-md p-6 text-center"
          on:dragover|preventDefault={handleDragOver}
          on:dragleave={handleDragLeave}
          on:drop={handleDrop}
          class:opacity-80={dragActive}
        >
          <Upload class="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Drag and drop files here, or click to browse
          </p>
          <input
            bind:this={fileInput}
            type="file"
            multiple
            accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.mp4,.mp3,.zip"
            on:change={(e: Event) => {
              const target = e.target as HTMLInputElement;
              uploadFiles = target?.files || null;
            }}
            class="hidden"
          />
          <Button
            on:click={() => fileInput && fileInput.click()}
            variant="outline"
            size="sm"
            class="mt-2"
          >
            Browse Files
          </Button>
        </div>

        <!-- Selected Files Display -->
        {#if uploadFiles && uploadFiles.length > 0}
          <div class="mt-3 space-y-2">
            <p class="text-sm font-medium text-gray-700 dark:text-gray-300">
              Selected files ({uploadFiles.length}):
            </p>
            {#each Array.from(uploadFiles) as file}
              <div class="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <div class="flex items-center gap-2">
                  <svelte:component this={getFileIcon(file.type)} class="w-4 h-4 text-blue-600" />
                  <span class="text-sm text-gray-900 dark:text-white truncate">{file.name}</span>
                </div>
                <span class="text-xs text-gray-500">{formatFileSize(file.size)}</span>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Title -->
        <div>
          <label class="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">Title</label>
          <Input
            bind:value={uploadTitle}
            placeholder="Enter a title for this evidence..."
            required
          />
        </div>

        <!-- Description -->
        <div>
          <label class="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">Description (Optional)</label>
          <textarea
            bind:value={uploadDescription}
            placeholder="Provide additional context or description..."
            class="w-full min-h-[100px] rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-sm text-gray-900 dark:text-white placeholder-gray-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          ></textarea>
        </div>

        <!-- Actions -->
        <div class="flex gap-3 pt-4">
          <Button
            on:click={uploadEvidence}
            disabled={!uploadFiles || uploadFiles.length === 0 || !uploadTitle.trim() || uploadLoading}
            class="flex-1 gap-2"
          >
            {#if uploadLoading}
              <div class="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
              Uploading...
            {:else}
              <Upload class="w-4 h-4" />
              Upload Evidence
            {/if}
          </Button>
          <Button
            on:click={() => { if (!uploadLoading) { resetUploadForm(); showUpload = false; } }}
            variant="outline"
            class="flex-1"
            disabled={uploadLoading}
          >
            Cancel
          </Button>
        </div>
      </div>
    </div>
  {/if}
</div>
