<script lang="ts">
  import { onMount } from 'svelte';
  import { $state, $derived, $effect } from 'svelte';
  import { writable } from 'svelte/store';
  import { page } from "$app/state";
  // Use modular components for file upload
  import { Button, Card, Badge, Input } from '$lib/components/ui/modular';
  import FileUploadSection from '$lib/components/FileUploadSection.svelte';
  import GoldenRatioLoader from '$lib/components/ui/enhanced-bits/GoldenRatioLoader.svelte';
  import {
    Upload,
    FileText,
    Image,
    Search,
    Filter,
    MoreVertical,
    Eye,
    Download,
    Trash2,
    Brain,
    Zap,
    Target,
  } from 'lucide-svelte';

  // Evidence management stores
  let evidenceItems = $state([]);
  let filteredEvidence = $state([]);
  let searchQuery = $state('');
  let selectedFilter = $state('all');
  let isUploading = $state(false);
  let uploadProgress = $state(0);
  let processingStatus = $state<'loading' | 'processing' | 'success' | 'error'>('loading');

  // Context7 integration state
  let context7Enabled = $state(true);
  let semanticSearchResults = $state([]);
  let ragEnhanced = $state(true);

  // File upload state (handled by FileUploadSection component)
  let uploadedFiles = $state([]);

  // ...existing code...

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
      let response: Response;
        try {
          response = await fetch('/api/evidence/list');
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
        } catch (error) {
          console.error('Fetch failed:', error);
          throw error;
        }
      if (response.ok) {
        const data = await response.json();
        evidenceItems = data.evidence || [];
        filterEvidence();
      }
    } catch (error) {
      console.error('Failed to load evidence:', error);
    }
  }


  async function triggerContext7Analysis(newEvidence: EvidenceItem[]) {
    try {
      for (const item of newEvidence) {
        const response = await fetch('/api/context7/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            evidenceId: item.id,
            content: item.summary,
            type: 'legal_evidence',
          }),
        });

        if (response.ok) {
          const analysis = await response.json();

          // Update evidence item with Context7 analysis
          const index = evidenceItems.findIndex((e) => e.id === item.id);
          if (index !== -1) {
            evidenceItems[index].context7Analysis = analysis;
            evidenceItems = [...evidenceItems];
          }
        }
      }
    } catch (error) {
      console.error('Context7 analysis failed:', error);
    }
  }

  async function performSemanticSearch(query: string) {
    if (!query.trim()) {
      semanticSearchResults = [];
      return;
    }

    try {
      const response = await fetch('/api/evidence/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          useSemanticSearch: true,
          includeContext7: context7Enabled,
          maxResults: 10,
        }),
      });

      if (response.ok) {
        const results = await response.json();
        semanticSearchResults = results.matches || [];
      }
    } catch (error) {
      console.error('Semantic search failed:', error);
    }
  }

  function filterEvidence() {
    let filtered = evidenceItems;

    // Apply search filter
    if (searchQuery.trim()) {
      filtered = filtered.filter(
        (item) =>
          item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.summary?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }

    // Apply type filter
    if (selectedFilter !== 'all') {
      filtered = filtered.filter((item) => item.type === selectedFilter);
    }

    filteredEvidence = filtered;
  }

  // Watch for search and filter changes
  $effect(() => {
    filterEvidence();
  });

  // Debounced semantic search
  let searchTimeout: ReturnType<typeof setTimeout>;
  $effect(() => {
    if (searchQuery) {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        performSemanticSearch(searchQuery);
      }, 500);
    }
  });

  function startRealTimeUpdates() {
    // Simulate real-time processing updates
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

  async function deleteEvidence(evidenceId: string) {
    try {
      let response: Response;
        try {
          response = await fetch(`/api/evidence/${evidenceId}`, {
        method: 'DELETE',
      });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
        } catch (error) {
          console.error('Fetch failed:', error);
          throw error;
        }

      if (response.ok) {
        evidenceItems = evidenceItems.filter((item) => item.id !== evidenceId);
        filterEvidence();
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  }

  function getFileIcon(type: string) {
    switch (type) {
      case 'pdf':
        return FileText;
      case 'image':
        return Image;
      default:
        return FileText;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'ready':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  function formatFileSize(bytes: number) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + ' ' + sizes[i];
  }

  // Evidence item shape (lightweight client-side typing)
  interface EvidenceItem {
    id: string;
    filename: string;
    status: string;
    type: string;
    size: number;
    uploadDate: string | number | Date;
    summary?: string;
    prosecutionScore?: number;
    tags: string[];
    summaryType?: string | null;
    context7Analysis?: unknown; // future structured type
  }

  function getSummaryTypeVariant(summaryType: string) {
    switch (summaryType) {
      case 'key_points':
        return { label: 'Key Points', color: 'bg-indigo-100 text-indigo-700 border-indigo-200' };
      case 'narrative':
        return { label: 'Narrative', color: 'bg-emerald-100 text-emerald-700 border-emerald-200' };
      case 'prosecutorial':
        return { label: 'Prosecutorial', color: 'bg-rose-100 text-rose-700 border-rose-200' };
      default:
        return { label: summaryType, color: 'bg-slate-100 text-slate-700 border-slate-200' };
    }
  }

  // Handler functions for FileUploadSection integration
  async function handleFileUpload(data: { files: File[]; tags: string[] }) {
    isUploading = true;
    processingStatus = 'processing';

    try {
      const formData = new FormData();
      data.files.forEach((file) => {
        formData.append('files', file);
      });
      
      // Add tags if provided
      if (data.tags.length > 0) {
        formData.append('tags', JSON.stringify(data.tags));
      }

      formData.append('context7Enabled', 'true');
      formData.append('ragEnhanced', 'true');
      formData.append('extractEntities', 'true');
      formData.append('generateSummary', 'true');
      formData.append('summaryType', 'narrative');

      const response = await fetch('/api/evidence/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        evidenceItems = [...evidenceItems, ...result.evidence];
        filterEvidence();
        processingStatus = 'success';
        
        if (context7Enabled) {
          await triggerContext7Analysis(result.evidence);
        }
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      processingStatus = 'error';
    } finally {
      isUploading = false;
    }
  }

  function handleFilesChanged(files: any[]) {
    // Files changed in the upload component
    console.log('Files changed:', files);
  }

  function handleUploadError(error: string) {
    console.error('Upload error:', error);
    processingStatus = 'error';
  }
</script>

<svelte:head>
  <title>Evidence Board - Legal AI Suite</title>
  <meta
    name="description"
    content="Upload, analyze, and manage legal evidence with AI-powered insights" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold text-slate-900 mb-2">📋 Evidence Board</h1>
      <p class="text-lg text-slate-600">
        Upload, analyze, and manage legal evidence with Context7 AI integration
      </p>
    </div>

    <!-- Stats Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      <Card variant="default" class="text-center">
        {#snippet header()}
          <div class="text-sm font-medium text-muted-foreground">Total Evidence</div>
        {/snippet}
        <div class="text-2xl font-bold text-blue-600">{totalEvidence}</div>
      </Card>

      <Card variant="default" class="text-center">
        {#snippet header()}
          <div class="text-sm font-medium text-muted-foreground">Processing</div>
        {/snippet}
        <div class="text-2xl font-bold text-yellow-600">{processingCount}</div>
      </Card>

      <Card variant="default" class="text-center">
        {#snippet header()}
          <div class="text-sm font-medium text-muted-foreground">Ready</div>
        {/snippet}
        <div class="text-2xl font-bold text-green-600">{readyCount}</div>
      </Card>

      <Card variant="default" class="text-center">
        {#snippet header()}
          <div class="text-sm font-medium text-muted-foreground">Context7 AI</div>
        {/snippet}
        <div class="flex items-center justify-center space-x-2">
          <div class="w-3 h-3 {context7Enabled ? 'bg-green-500' : 'bg-gray-400'} rounded-full">
          </div>
          <span class="text-sm font-medium">{context7Enabled ? 'Enabled' : 'Disabled'}</span>
        </div>
      </Card>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Upload Section -->
      <div class="lg:col-span-2">
        <FileUploadSection
          reportId="evidenceboard"
          acceptedTypes={['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.txt', '.doc', '.docx']}
          maxFileSize={50 * 1024 * 1024}
          maxFiles={10}
          multiple={true}
          onupload={handleFileUpload}
          onfilesChanged={handleFilesChanged}
          onerror={handleUploadError}
        />

        <!-- Evidence Grid -->
        <Card variant="default" class="mt-6">
          {#snippet header()}
            <div class="space-y-4">
              <h3 class="text-lg font-semibold">Evidence Collection ({filteredEvidence.length})</h3>

              <!-- Search and Filter -->
              <div class="flex flex-col sm:flex-row gap-4">
                <div class="flex-1">
                  <Input bind:value={searchQuery} placeholder="Search evidence..." variant="default" />
                </div>
                <select
                  bind:value={selectedFilter}
                  class="px-3 py-2 border border-border rounded-md bg-background text-foreground">
                  <option value="all">All Types</option>
                  <option value="pdf">PDFs</option>
                  <option value="image">Images</option>
                  <option value="document">Documents</option>
                </select>
              </div>
            </div>
          {/snippet}
            {#if filteredEvidence.length === 0}
              <div class="text-center py-12">
                <FileText class="h-16 w-16 text-slate-300 mx-auto mb-4" />
                <p class="text-slate-500">
                  No evidence files found. Upload some files to get started.
                </p>
              </div>
            {:else}
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                {#each filteredEvidence as item (item.id)}
                  <div
                    class="border border-slate-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div class="flex items-start justify-between mb-3">
                      <div class="flex items-center space-x-2">
                        {#if getFileIcon(item.type) === FileText}
                          <FileText class="h-5 w-5 text-slate-500" />
                        {:else if getFileIcon(item.type) === Image}
                          <Image class="h-5 w-5 text-slate-500" />
                        {:else}
                          <FileText class="h-5 w-5 text-slate-500" />
                        {/if}
                        <h4 class="font-medium text-slate-900 truncate flex-1">
                          {item.filename}
                        </h4>
                      </div>
                      <button class="text-slate-400 hover:text-slate-600">
                        <MoreVertical class="h-4 w-4" />
                      </button>
                    </div>

                    <div class="space-y-2 mb-3">
                      <div class="flex flex-wrap gap-2 items-center">
                        <Badge size="sm" variant="outline">
                          {item.status}
                        </Badge>
                        {#if item.summaryType}
                          <span class={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${getSummaryTypeVariant(item.summaryType).color}`}>
                            {getSummaryTypeVariant(item.summaryType).label}
                          </span>
                        {/if}
                      </div>
                      <p class="text-xs text-slate-500">
                        {formatFileSize(item.size)} • Uploaded {new Date(
                          item.uploadDate
                        ).toLocaleDateString()}
                      </p>
                    </div>

                    {#if item.summary}
                      <p class="text-sm text-slate-700 mb-3 line-clamp-2">
                        {item.summary}
                      </p>
                    {/if}

                    {#if item.prosecutionScore}
                      <div class="mb-3">
                        <div class="flex justify-between text-xs text-slate-600 mb-1">
                          <span>Prosecution Relevance</span>
                          <span>{(item.prosecutionScore * 100).toFixed(0)}%</span>
                        </div>
                        <div class="w-full bg-slate-200 rounded-full h-2">
                          <div
                            class="h-2 rounded-full bg-gradient-to-r from-green-400 to-blue-500"
                            style:width="{item.prosecutionScore * 100}%">
                          </div>
                        </div>
                      </div>
                    {/if}

                    {#if item.tags.length > 0}
                      <div class="flex flex-wrap gap-1 mb-3">
                        {#each item.tags.slice(0, 3) as tag}
                          <Badge variant="outline" size="sm">{tag}</Badge>
                        {/each}
                      </div>
                    {/if}

                    <!-- Actions -->
                    <div class="flex space-x-2">
                      <Button size="sm" variant="outline" class="flex-1">
                        <Eye class="h-3 w-3 mr-1" />
                        View
                      </Button>
                      <Button size="sm" variant="outline">
                        <Download class="h-3 w-3" />
                      </Button>
                      <Button size="sm" variant="outline" onclick={() => deleteEvidence(item.id)}>
                        <Trash2 class="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                {/each}
              </div>
            {/if}
        </Card>
      </div>

      <!-- AI Insights Panel -->
      <div class="space-y-6">
        <!-- Context7 Analysis -->
        <Card variant="default">
          {#snippet header()}
            <div class="flex items-center space-x-2">
              <Brain class="h-5 w-5" />
              <span class="text-lg font-semibold">AI Insights</span>
            </div>
          {/snippet}
            {#if context7Enabled}
              <div class="space-y-4">
                <div class="flex items-center space-x-2">
                  <Zap class="h-4 w-4 text-blue-500" />
                  <span class="text-sm font-medium">Context7 Analysis Active</span>
                </div>

                <div class="text-sm text-slate-600">
                  AI is continuously analyzing your evidence for:
                </div>

                <ul class="text-sm text-slate-600 space-y-1">
                  <li>• Legal entity extraction</li>
                  <li>• Case law connections</li>
                  <li>• Prosecution relevance scoring</li>
                  <li>• Semantic relationship mapping</li>
                </ul>
              </div>
            {:else}
              <div class="text-center py-6">
                <Target class="h-12 w-12 text-slate-300 mx-auto mb-2" />
                <p class="text-sm text-slate-500">Enable Context7 for AI insights</p>
              </div>
            {/if}
        </Card>

        <!-- Semantic Search Results -->
        {#if semanticSearchResults.length > 0}
          <Card variant="default">
            {#snippet header()}
              <div class="flex items-center space-x-2">
                <Search class="h-5 w-5" />
                <span class="text-lg font-semibold">Semantic Search</span>
              </div>
            {/snippet}
              <div class="space-y-3">
                {#each semanticSearchResults.slice(0, 5) as result}
                  <div class="p-3 bg-slate-50 rounded-md">
                    <h5 class="font-medium text-sm text-slate-900 mb-1">
                      {result.filename}
                    </h5>
                    <p class="text-xs text-slate-600 mb-2">
                      Similarity: {(result.similarity * 100).toFixed(0)}%
                    </p>
                    <p class="text-xs text-slate-700 line-clamp-2">
                      {result.content}
                    </p>
                  </div>
                {/each}
              </div>
          </Card>
        {/if}

        <!-- Quick Actions -->
        <Card variant="default">
          {#snippet header()}
            <div class="text-lg font-semibold">Quick Actions</div>
          {/snippet}
            <div class="space-y-3">
              <Button class="w-full justify-start">
                <Brain class="h-4 w-4 mr-2" />
                Generate Case Summary
              </Button>
              <Button variant="outline" class="w-full justify-start">
                <Search class="h-4 w-4 mr-2" />
                Find Similar Cases
              </Button>
              <Button variant="outline" class="w-full justify-start">
                <Download class="h-4 w-4 mr-2" />
                Export Evidence Report
              </Button>
            </div>
        </Card>
      </div>
    </div>
  </div>
</div>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>

