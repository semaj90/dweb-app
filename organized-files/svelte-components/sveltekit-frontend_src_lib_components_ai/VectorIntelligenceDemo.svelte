<script lang="ts">
  import { Search, Database, Brain, FileText, AlertCircle, CheckCircle2, Loader2, Star, Clock } from "lucide-svelte";
  import { Button } from "$lib/components/ui/button";
  import Input from "$lib/components/ui/Input.svelte";
  import Badge from "$lib/components/ui/Badge.svelte";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/Card";

  type SearchResult = {
    id: string
    title: string
    content: string
    similarity: number
    documentType: 'deed' | 'contract' | 'evidence' | 'case_law';
    metadata?: {
      caseId?: string;
      uploadDate?: string;
      tags?: string[];
    };
  };

  type SearchMetrics = {
    totalDocuments: number
    searchTime: number
    vectorDimensions: number
    similarityThreshold: number
  };

  // Modern Svelte 5 runes
  let query = $state("");
  let isSearching = $state(false);
  let results = $state<SearchResult[]>([]);
  let metrics = $state<SearchMetrics | null>(null);
  let error = $state<string | null>(null);
  let selectedResult = $state<SearchResult | null>(null);

  // Derived state for UI feedback
  const hasResults = $derived(results.length > 0)
  const showMetrics = $derived(metrics !== null)
  const searchButtonDisabled = $derived(isSearching || query.trim().length === 0);

  // Vector intelligence search function
  async function performSemanticSearch() {
    if (!query.trim() || isSearching) return;

    isSearching = true;
    error = null;
    const startTime = performance.now();

    try {
      const response = await fetch('/api/semantic-search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim()
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || `Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      const searchTime = performance.now() - startTime;

      // Align with the simpler API response
      results = data.results.map((r: any) => ({
        id: r.id,
        title: `Document ${r.id}`, // API doesn't provide title, create one
        content: r.content,
        similarity: r.similarity,
        documentType: 'deed' // Mock type
      }));

      metrics = {
        totalDocuments: data.results.length,
        searchTime: Math.round(searchTime),
        vectorDimensions: 384, // Assuming this value, as API doesn't provide it
        similarityThreshold: 0.0 // API doesn't use a threshold input
      };

    } catch (err) {
      error = err instanceof Error ? err.message : 'Search failed';
      results = [];
      metrics = null;
    } finally {
      isSearching = false;
    }
  }

  // Handle form submission
  function handleSubmit(event: SubmitEvent) {
    event.preventDefault();
    performSemanticSearch();
  }

  // Handle Enter key in search input
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !searchButtonDisabled) {
      performSemanticSearch();
    }
  }

  // Format similarity score as percentage
  function formatSimilarity(score: number): string {
    return `${Math.round(score * 100)}%`;
  }

  // Get document type icon and color
  function getDocumentTypeStyle(type: SearchResult['documentType']) {
    switch (type) {
      case 'deed':
        return { icon: FileText, color: 'bg-blue-100 text-blue-800' };
      case 'contract':
        return { icon: FileText, color: 'bg-green-100 text-green-800' };
      case 'evidence':
        return { icon: Database, color: 'bg-orange-100 text-orange-800' };
      case 'case_law':
        return { icon: Brain, color: 'bg-purple-100 text-purple-800' };
      default:
        return { icon: FileText, color: 'bg-gray-100 text-gray-800' };
    }
  }

  // Demo placeholder results for development
  const demoResults: SearchResult[] = [
    {
      id: "demo-1",
      title: "Property Deed - 123 Main Street",
      content: "This warranty deed transfers ownership of the property located at 123 Main Street from John Smith to Jane Doe. The property includes all fixtures and improvements...",
      similarity: 0.92,
      documentType: 'deed',
      metadata: {
        caseId: "CASE-2024-001",
        uploadDate: "2024-01-15",
        tags: ["property", "transfer", "warranty"]
      }
    },
    {
      id: "demo-2",
      title: "Employment Contract - Tech Corp",
      content: "This employment agreement establishes the terms of employment between Tech Corp and the employee. The position includes responsibilities for software development...",
      similarity: 0.87,
      documentType: 'contract',
      metadata: {
        caseId: "CASE-2024-002",
        uploadDate: "2024-01-10",
        tags: ["employment", "technology", "intellectual property"]
      }
    }
  ];
</script>

<!-- Vector Intelligence Demo Component -->
<div class="max-w-4xl mx-auto p-6 space-y-6">
  <!-- Header Section -->
  <div class="text-center space-y-2">
    <div class="flex items-center justify-center gap-2 text-2xl font-bold">
      <Brain class="h-6 w-6 text-purple-600" />
      Vector Intelligence Demo
    </div>
    <p class="text-muted-foreground">
      Semantic search powered by pgvector and AI embeddings for legal document analysis
    </p>
  </div>

  <!-- Search Interface -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Search class="h-5 w-5" />
        Semantic Document Search
      </CardTitle>
    </CardHeader>
    <CardContent class="space-y-4">
      <form onsubmit={handleSubmit} class="flex gap-2">
        <Input
          bind:value={query}
          onkeydown={handleKeydown}
          placeholder="Search legal documents using natural language..."
          class="flex-1"
          disabled={isSearching}
        />
        <Button
          type="submit"
          disabled={searchButtonDisabled}
          class="min-w-[100px]"
        >
          {#if isSearching}
            <Loader2 class="h-4 w-4 animate-spin mr-2" />
            Searching
          {:else}
            <Search class="h-4 w-4 mr-2" />
            Search
          {/if}
        </Button>
      </form>

      <!-- Example queries -->
      <div class="flex flex-wrap gap-2">
        <span class="text-sm text-muted-foreground">Try:</span>
        {#each ["property ownership transfer", "contract liability clauses", "employment agreements", "intellectual property rights"] as example}
          <Button
            variant="outline"
            size="sm"
            on:click={() => { query = example; }}
            disabled={isSearching}
          >
            {example}
          </Button>
        {/each}
      </div>
    </CardContent>
  </Card>

  <!-- Error Display -->
  {#if error}
    <div class="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-2">
      <AlertCircle class="h-4 w-4 text-red-600" />
      <div class="text-red-800">{error}</div>
    </div>
  {/if}

  <!-- Search Metrics -->
  {#if showMetrics}
    <Card>
      <CardContent class="pt-6">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div class="text-2xl font-bold">{metrics.totalDocuments}</div>
            <div class="text-sm text-muted-foreground">Documents</div>
          </div>
          <div>
            <div class="text-2xl font-bold">{metrics.searchTime}ms</div>
            <div class="text-sm text-muted-foreground">Search Time</div>
          </div>
          <div>
            <div class="text-2xl font-bold">{metrics.vectorDimensions}D</div>
            <div class="text-sm text-muted-foreground">Vector Space</div>
          </div>
          <div>
            <div class="text-2xl font-bold">{Math.round(metrics.similarityThreshold * 100)}%</div>
            <div class="text-sm text-muted-foreground">Threshold</div>
          </div>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Search Results -->
  {#if hasResults}
    <div class="space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-semibold">Search Results</h3>
        <Badge variant="secondary">{results.length} found</Badge>
      </div>

      <div class="grid gap-4">
        {#each results as result (result.id)}
          {@const typeStyle = getDocumentTypeStyle(result.documentType)}
          <Card
            class="hover:shadow-md transition-shadow cursor-pointer border-l-4 border-l-purple-500"
            on:click={() => selectedResult = result}
          >
            <CardContent class="pt-6">
              <div class="space-y-3">
                <!-- Document header -->
                <div class="flex items-start justify-between gap-4">
                  <div class="flex-1">
                    <div class="flex items-center gap-2 mb-1">
                      <typeStyle.icon class="h-4 w-4" />
                      <h4 class="font-semibold line-clamp-1">{result.title}</h4>
                    </div>
                    <p class="text-sm text-muted-foreground line-clamp-2">
                      {result.content}
                    </p>
                  </div>
                  <div class="flex flex-col items-end gap-2">
                    <div class="flex items-center gap-1">
                      <Star class="h-3 w-3 text-yellow-500" />
                      <span class="text-sm font-mono">{formatSimilarity(result.similarity)}</span>
                    </div>
                    <Badge class={typeStyle.color}>
                      {result.documentType.replace('_', ' ')}
                    </Badge>
                  </div>
                </div>

                <!-- Metadata -->
                {#if result.metadata}
                  <div class="flex items-center gap-4 text-xs text-muted-foreground">
                    {#if result.metadata.caseId}
                      <span>Case: {result.metadata.caseId}</span>
                    {/if}
                    {#if result.metadata.uploadDate}
                      <span class="flex items-center gap-1">
                        <Clock class="h-3 w-3" />
                        {result.metadata.uploadDate}
                      </span>
                    {/if}
                  </div>
                  {#if result.metadata.tags}
                    <div class="flex flex-wrap gap-1">
                      {#each result.metadata.tags as tag}
                        <Badge variant="outline" class="text-xs">{tag}</Badge>
                      {/each}
                    </div>
                  {/if}
                {/if}
              </div>
            </CardContent>
          </Card>
        {/each}
      </div>
    </div>
  {:else if !isSearching && query.trim().length > 0}
    <!-- No results state -->
    <Card>
      <CardContent class="pt-6 text-center space-y-2">
        <Search class="h-12 w-12 text-muted-foreground mx-auto" />
        <h3 class="font-semibold">No results found</h3>
        <p class="text-sm text-muted-foreground">
          Try adjusting your search terms or using different keywords
        </p>
      </CardContent>
    </Card>
  {/if}

  <!-- Demo Notice -->
  {#if !hasResults && !isSearching && query.trim().length === 0}
    <Card class="border-dashed">
      <CardContent class="pt-6 text-center space-y-4">
        <div class="flex justify-center">
          <Database class="h-16 w-16 text-muted-foreground" />
        </div>
        <div class="space-y-2">
          <h3 class="font-semibold">Vector Intelligence Ready</h3>
          <p class="text-sm text-muted-foreground max-w-md mx-auto">
            Enter a search query to find semantically similar legal documents using AI-powered vector embeddings
          </p>
        </div>
        <div class="flex justify-center">
          <Button
            variant="outline"
            on:click={() => { results = demoResults; metrics = { totalDocuments: 1250, searchTime: 45, vectorDimensions: 384, similarityThreshold: 0.7 }; }}
          >
            Load Demo Results
          </Button>
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<!-- Selected Result Modal (future enhancement) -->
{#if selectedResult}
  <div class="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
    <Card class="max-w-2xl w-full max-h-[80vh] overflow-auto">
      <CardHeader>
        <CardTitle class="flex items-center justify-between">
          {selectedResult.title}
          <Button variant="ghost" size="sm" on:click={() => selectedResult = null}>
            ×
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <p class="text-sm">{selectedResult.content}</p>
        <div class="flex items-center gap-2">
          <Badge>Similarity: {formatSimilarity(selectedResult.similarity)}</Badge>
          <Badge variant="outline">{selectedResult.documentType}</Badge>
        </div>
      </CardContent>
    </Card>
  </div>
{/if}