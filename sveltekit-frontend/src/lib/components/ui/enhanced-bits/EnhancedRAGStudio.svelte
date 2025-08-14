<!-- EnhancedRAG:Studio UI - Complete RAG Management Dashboard -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Card, CardHeader, CardTitle, CardContent } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Badge } from '$lib/components/ui/badge';
  import {
    Upload,
    Globe,
    Search,
    Database,
    Activity,
    FileText,
    Brain,
    TrendingUp,
    Settings,
    Play,
    Pause,
    RefreshCw
  } from 'lucide-svelte';
  // State management
  let activeTab = $state<'upload' | 'crawl' | 'search' | 'logs' | 'settings'>('search');
  let isLoading = $state(false);
  let serviceStatus = $state<any>({});
  let searchResults = $state<any[]>([]);
  let recentLogs = $state<any[]>([]);
  let embeddings = $state<any[]>([]);
  let patches = $state<any[]>([]);
  let rlMetrics = $state<any>({});

  // Form states
  let searchQuery = $state('');
  let uploadFile = $state<File | null>(null);
  let crawlUrl = $state('');
  let feedbackScore = $state(0);

  // Real-time updates
  let statusInterval: number | null = null;
  let logsInterval: number | null = null;

  onMount(async () => {
    await loadServiceStatus();
    await loadRecentLogs();
    await loadEmbeddings();
    startRealTimeUpdates();

    return () => {
      if (statusInterval) clearInterval(statusInterval);
      if (logsInterval) clearInterval(logsInterval);
    };
  });

  function startRealTimeUpdates() {
    // Update service status every 30 seconds
    statusInterval = setInterval(async () => {
      await loadServiceStatus();
    }, 30000);

    // Update logs every 10 seconds
    logsInterval = setInterval(async () => {
      await loadRecentLogs();
    }, 10000);
  }

  async function loadServiceStatus() {
    try {
      const response = await fetch('/api/rag?action=status');
      const data = await response.json();
      serviceStatus = data;
    } catch (error) {
      console.error('Failed to load service status:', error);
    }
  }

  async function loadRecentLogs() {
    try {
      const response = await fetch('/api/logs');
      const data = await response.json();
      recentLogs = data.logs || [];
    } catch (error) {
      console.error('Failed to load logs:', error);
    }
  }

  async function loadEmbeddings() {
    try {
      const response = await fetch('/api/embeddings');
      const data = await response.json();
      embeddings = data.embeddings || [];
    } catch (error) {
      console.error('Failed to load embeddings:', error);
    }
  }

  async function handleSearch() {
    if (!searchQuery.trim()) return;

    isLoading = true;
    try {
      const response = await fetch('/api/rag?action=search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          options: { topK: 10, threshold: 0.7 }
        })
      });

      const data = await response.json();
      searchResults = data.results || [];

      // Log search activity
      await logActivity('search', { query: searchQuery, results: data.results.length });

    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function handleUpload() {
    if (!uploadFile) return;

    isLoading = true;
    try {
      const formData = new FormData();
      formData.append('file', uploadFile);

      const response = await fetch('/api/rag?action=upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        await logActivity('upload', { filename: uploadFile.name, chunks: data.document.chunks });
        uploadFile = null;
        await loadEmbeddings();
      }

    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function handleCrawl() {
    if (!crawlUrl.trim()) return;

    isLoading = true;
    try {
      const response = await fetch('/api/rag?action=crawl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: crawlUrl,
          options: { maxPages: 1 }
        })
      });

      const data = await response.json();

      if (data.success) {
        await logActivity('crawl', { url: crawlUrl, chunks: data.document.chunks });
        crawlUrl = '';
        await loadEmbeddings();
      }

    } catch (error) {
      console.error('Crawl failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function submitFeedback(resultId: string, score: number) {
    try {
      await fetch('/api/rl-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resultId,
          score,
          query: searchQuery,
          timestamp: Date.now()
        })
      });

      await logActivity('feedback', { resultId, score });

    } catch (error) {
      console.error('Feedback submission failed:', error);
    }
  }

  async function logActivity(action: string, metadata: any) {
    try {
      await fetch('/api/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          metadata,
          timestamp: new Date().toISOString()
        })
      });
    } catch (error) {
      console.error('Failed to log activity:', error);
    }
  }

  function formatTimestamp(timestamp: string) {
    return new Date(timestamp).toLocaleString();
  }

  function getStatusColor(healthy: boolean) {
    return healthy ? 'bg-green-500' : 'bg-red-500';
  }
</script>

<div class="w-full min-h-screen bg-gray-50 p-6">
  <!-- Header -->
  <div class="mb-8">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">EnhancedRAG:Studio</h1>
    <p class="text-gray-600">Complete RAG management and monitoring dashboard</p>
  </div>

  <!-- Service Status Bar -->
  <Card class="mb-6">
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Activity class="w-5 h-5" />
        Service Status
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div class="flex gap-4">
        <div class="flex items-center gap-2">
          <div class="w-3 h-3 rounded-full {getStatusColor(serviceStatus.services?.redis)}"></div>
          <span>Redis Vector DB</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="w-3 h-3 rounded-full {getStatusColor(serviceStatus.services?.ingestion)}"></div>
          <span>Document Ingestion</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="w-3 h-3 rounded-full {getStatusColor(serviceStatus.services?.ai)}"></div>
          <span>AI Service</span>
        </div>
        {#if serviceStatus.indexStats}
          <div class="ml-auto text-sm text-gray-600">
            Documents: {serviceStatus.indexStats.num_docs || 0}
          </div>
        {/if}
      </div>
    </CardContent>
  </Card>

  <!-- Navigation Tabs -->
  <div class="flex gap-2 mb-6">
    <Button
      variant={activeTab === 'search' ? 'default' : 'outline'}
      on:click={() => activeTab = 'search'}
      class="flex items-center gap-2"
    >
      <Search class="w-4 h-4" />
      Search
    </Button>
    <Button
      variant={activeTab === 'upload' ? 'default' : 'outline'}
      on:click={() => activeTab = 'upload'}
      class="flex items-center gap-2"
    >
      <Upload class="w-4 h-4" />
      Upload
    </Button>
    <Button
      variant={activeTab === 'crawl' ? 'default' : 'outline'}
      on:click={() => activeTab = 'crawl'}
      class="flex items-center gap-2"
    >
      <Globe class="w-4 h-4" />
      Crawl
    </Button>
    <Button
      variant={activeTab === 'logs' ? 'default' : 'outline'}
      on:click={() => activeTab = 'logs'}
      class="flex items-center gap-2"
    >
      <FileText class="w-4 h-4" />
      Logs
    </Button>
    <Button
      variant={activeTab === 'settings' ? 'default' : 'outline'}
      on:click={() => activeTab = 'settings'}
      class="flex items-center gap-2"
    >
      <Settings class="w-4 h-4" />
      Settings
    </Button>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Main Content Area -->
    <div class="lg:col-span-2">
      {#if activeTab === 'search'}
        <Card>
          <CardHeader>
            <CardTitle class="flex items-center gap-2">
              <Search class="w-5 h-5" />
              Semantic Search
            </CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <div class="flex gap-2">
              <Input
                bind:value={searchQuery}
                placeholder="Enter your search query..."
                class="flex-1"
                onkeydown={(e) => e.key === 'Enter' && handleSearch()}
              />
              <Button on:click={handleSearch} disabled={isLoading || !searchQuery.trim()}>
                {#if isLoading}
                  <RefreshCw class="w-4 h-4 animate-spin" />
                {:else}
                  <Search class="w-4 h-4" />
                {/if}
                Search
              </Button>
            </div>

            <!-- Search Results -->
            {#if searchResults.length > 0}
              <div class="space-y-3">
                <h3 class="font-semibold">Results ({searchResults.length})</h3>
                {#each searchResults as result}
                  <div class="border rounded-lg p-4 space-y-2">
                    <div class="flex justify-between items-start">
                      <div class="flex-1">
                        <div class="font-medium">{result.metadata.title || 'Untitled'}</div>
                        <div class="text-sm text-gray-600 mb-2">{result.content}</div>
                        <div class="flex gap-2">
                          <Badge variant="outline">Score: {result.score.toFixed(3)}</Badge>
                          <Badge variant="outline">{result.metadata.type}</Badge>
                        </div>
                      </div>
                      <div class="flex gap-1 ml-4">
                        <Button
                          size="sm"
                          variant="outline"
                          on:click={() => submitFeedback(result.id, 1)}
                        >
                          👍
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          on:click={() => submitFeedback(result.id, -1)}
                        >
                          👎
                        </Button>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            {/if}
          </CardContent>
        </Card>

      {:else if activeTab === 'upload'}
        <Card>
          <CardHeader>
            <CardTitle class="flex items-center gap-2">
              <Upload class="w-5 h-5" />
              Document Upload
            </CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Select PDF Document</label>
              <input
                type="file"
                accept=".pdf"
                onchange={(e) => uploadFile = e.target.files?.[0] || null}
                class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>
            <Button on:click={handleUpload} disabled={isLoading || !uploadFile}>
              {#if isLoading}
                <RefreshCw class="w-4 h-4 animate-spin mr-2" />
              {:else}
                <Upload class="w-4 h-4 mr-2" />
              {/if}
              Upload & Process
            </Button>
          </CardContent>
        </Card>

      {:else if activeTab === 'crawl'}
        <Card>
          <CardHeader>
            <CardTitle class="flex items-center gap-2">
              <Globe class="w-5 h-5" />
              Web Crawler
            </CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Website URL</label>
              <Input
                bind:value={crawlUrl}
                placeholder="https://example.com"
                type="url"
              />
            </div>
            <Button on:click={handleCrawl} disabled={isLoading || !crawlUrl.trim()}>
              {#if isLoading}
                <RefreshCw class="w-4 h-4 animate-spin mr-2" />
              {:else}
                <Globe class="w-4 h-4 mr-2" />
              {/if}
              Crawl & Process
            </Button>
          </CardContent>
        </Card>

      {:else if activeTab === 'logs'}
        <Card>
          <CardHeader>
            <CardTitle class="flex items-center gap-2">
              <FileText class="w-5 h-5" />
              Activity Logs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-2 max-h-96 overflow-y-auto">
              {#each recentLogs as log}
                <div class="border-l-4 border-blue-500 pl-4 py-2">
                  <div class="flex justify-between items-start">
                    <div>
                      <div class="font-medium capitalize">{log.action}</div>
                      <div class="text-sm text-gray-600">{JSON.stringify(log.metadata)}</div>
                    </div>
                    <div class="text-xs text-gray-500">
                      {formatTimestamp(log.timestamp)}
                    </div>
                  </div>
                </div>
              {/each}
            </div>
          </CardContent>
        </Card>

      {:else if activeTab === 'settings'}
        <Card>
          <CardHeader>
            <CardTitle class="flex items-center gap-2">
              <Settings class="w-5 h-5" />
              System Settings
            </CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Search Threshold</label>
              <Input type="number" step="0.1" min="0" max="1" value="0.7" />
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Max Results</label>
              <Input type="number" min="1" max="50" value="10" />
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Cache TTL (seconds)</label>
              <Input type="number" min="60" max="86400" value="7200" />
            </div>
            <Button>Save Settings</Button>
          </CardContent>
        </Card>
      {/if}
    </div>

    <!-- Sidebar - Metrics & Monitoring -->
    <div class="space-y-6">
      <!-- Embeddings Overview -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Database class="w-5 h-5" />
            Vector Database
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-sm">Total Documents:</span>
              <span class="font-medium">{embeddings.length}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm">Index Size:</span>
              <span class="font-medium">{serviceStatus.indexStats?.num_docs || 0}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm">Memory Usage:</span>
              <span class="font-medium">~</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- RL Metrics -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Brain class="w-5 h-5" />
            RL Feedback
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-sm">Positive Feedback:</span>
              <span class="font-medium text-green-600">{rlMetrics.positive || 0}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm">Negative Feedback:</span>
              <span class="font-medium text-red-600">{rlMetrics.negative || 0}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm">Avg Score:</span>
              <span class="font-medium">{rlMetrics.avgScore || 'N/A'}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Performance Metrics -->
      <Card>
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <TrendingUp class="w-5 h-5" />
            Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-sm">Avg Response Time:</span>
              <span class="font-medium">~ms</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm">Cache Hit Rate:</span>
              <span class="font-medium">~%</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm">Queries/Hour:</span>
              <span class="font-medium">~</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  </div>
</div>

<style>
  /* Custom scrollbar for logs */
  .overflow-y-auto::-webkit-scrollbar {
    width: 6px;
  }

  .overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }

  .overflow-y-auto::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }

  .overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
</style>
