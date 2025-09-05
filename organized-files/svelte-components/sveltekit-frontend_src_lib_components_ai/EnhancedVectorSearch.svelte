<!-- Enhanced Vector Search Interface with Ranking, Analytics, and Real-time Results -->
<script lang="ts">
  import {
    Badge,
    Button,
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    Checkbox,
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    Input,
    Progress,
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
    Slider,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
  } from "bits-ui";
  import {
    BarChart3,
    Brain,
    ChevronDown,
    ChevronUp,
    Clock,
    Download,
    Eye,
    Filter,
    Loader2,
    Search,
    Share2,
    Target,
    TrendingUp,
    Zap,
  } from "lucide-svelte";
  import { createEventDispatcher, onMount } from "svelte";
  import { derived, get, writable } from "svelte/store";

  // Props
  let {
    caseId = "",
    userId = "",
    maxResults = 20,
    enableAnalytics = true,
    enableFilters = true,
    showPreview = true,
    class: className = "",
  } = $props();

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    search: { query: string results: SearchResult[] };
    select: { result: SearchResult };
    filter: { filters: SearchFilters };
    analytics: { event: string data: any };
  }>();

  // Types
  interface SearchResult {
    id: string
    documentId: string
    chunkId?: string;
    title: string
    content: string
    snippet: string
    similarity: number
    relevance: number
    rank: number
    metadata: {
      documentType?: string;
      jurisdiction?: string;
      tags?: string[];
      createdAt?: string;
      fileSize?: number;
      pageNumber?: number;
      section?: string;
    };
    highlights: string[];
    aiSummary?: string;
    entities?: Array<{
      text: string
      type: string
      confidence: number
    }>;
  }

  interface SearchFilters {
    documentTypes: string[];
    jurisdictions: string[];
    dateRange: {
      from?: Date;
      to?: Date;
    };
    similarityThreshold: number
    maxResults: number
    tags: string[];
    sortBy: "relevance" | "similarity" | "date" | "title";
    sortOrder: "asc" | "desc";
  }

  interface SearchAnalytics {
    totalSearches: number
    averageResultCount: number
    topQueries: Array<{ query: string count: number }>;
    averageSimilarity: number
    responseTime: number
    clickThroughRate: number
    commonFilters: Record<string, number>;
    performanceMetrics: {
      vectorSearchTime: number
      rankingTime: number
      totalTime: number
    };
  }

  // State management
  const searchQuery = writable("");
  const searchResults = writable<SearchResult[]>([]);
  const isSearching = writable(false);
  const searchFilters = writable<SearchFilters>({
    documentTypes: [],
    jurisdictions: [],
    dateRange: {},
    similarityThreshold: 0.7,
    maxResults: maxResults,
    tags: [],
    sortBy: "relevance",
    sortOrder: "desc",
  });
  const searchAnalytics = writable<SearchAnalytics>({
    totalSearches: 0,
    averageResultCount: 0,
    topQueries: [],
    averageSimilarity: 0,
    responseTime: 0,
    clickThroughRate: 0,
    commonFilters: {},
    performanceMetrics: {
      vectorSearchTime: 0,
      rankingTime: 0,
      totalTime: 0,
    },
  });

  const showFilters = writable(false);
  const showAnalytics = writable(false);
  const selectedResult = writable<SearchResult | null>(null);
  const searchHistory = writable<string[]>([]);

  // Derived state
  const hasResults = derived(searchResults, ($results) => $results.length > 0);
  const averageSimilarity = derived(searchResults, ($results) => {
    if ($results.length === 0) return 0;
    return (
      $results.reduce((acc, result) => acc + result.similarity, 0) /
      $results.length
    );
  });
  const topDocumentTypes = derived(searchResults, ($results) => {
    const types = new Map<string, number>();
    $results.forEach((result) => {
      const type = result.metadata.documentType || "unknown";
      types.set(type, (types.get(type) || 0) + 1);
    });
    return Array.from(types.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
  });

  // Available options
  const documentTypes = [
    { value: "contract", label: "Contract" },
    { value: "motion", label: "Motion" },
    { value: "brief", label: "Brief" },
    { value: "evidence", label: "Evidence" },
    { value: "correspondence", label: "Correspondence" },
    { value: "statute", label: "Statute" },
    { value: "regulation", label: "Regulation" },
    { value: "case_law", label: "Case Law" },
    { value: "other", label: "Other" },
  ];

  const jurisdictions = [
    { value: "federal", label: "Federal" },
    { value: "state", label: "State" },
    { value: "local", label: "Local" },
    { value: "international", label: "International" },
  ];

  const sortOptions = [
    { value: "relevance", label: "Relevance" },
    { value: "similarity", label: "Similarity" },
    { value: "date", label: "Date" },
    { value: "title", label: "Title" },
  ];

  // ============================================================================
  // SEARCH FUNCTIONALITY
  // ============================================================================

  async function performSearch(query?: string) {
    const searchTerm = query || get(searchQuery);
    if (!searchTerm.trim()) return;

    isSearching.set(true);
    const startTime = Date.now();

    try {
      const filters = get(searchFilters);

      // Build search request
      const searchRequest = {
        query: searchTerm,
        caseId: caseId || undefined,
        filters: {
          documentTypes: filters.documentTypes,
          jurisdictions: filters.jurisdictions,
          dateRange: filters.dateRange,
          tags: filters.tags,
          similarityThreshold: filters.similarityThreshold,
          maxResults: filters.maxResults,
        },
        sortBy: filters.sortBy,
        sortOrder: filters.sortOrder,
        includeAnalytics: enableAnalytics,
        generateSnippets: true,
        highlightTerms: true,
      };

      // Make API call
      const response = await fetch("/api/search/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(searchRequest),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();

      // Process results
      const results: SearchResult[] = data.results.map(
        (result: any, index: number) => ({
          ...result,
          rank: index + 1,
          highlights: result.highlights || [],
          snippet: result.snippet || result.content.substring(0, 200) + "...",
        })
      );

      searchResults.set(results);

      // Update search history
      searchHistory.update((history) => {
        const newHistory = [
          searchTerm,
          ...history.filter((h) => h !== searchTerm),
        ];
        return newHistory.slice(0, 10); // Keep last 10 searches
      });

      // Update analytics
      if (enableAnalytics && data.analytics) {
        searchAnalytics.update((analytics) => ({
          ...analytics,
          totalSearches: analytics.totalSearches + 1,
          averageResultCount: Math.round(
            (analytics.averageResultCount + results.length) / 2
          ),
          responseTime: Date.now() - startTime,
          performanceMetrics:
            data.analytics.performanceMetrics || analytics.performanceMetrics,
          averageSimilarity: get(averageSimilarity),
        }));
      }

      // Dispatch events
      dispatch("search", { query: searchTerm, results });
      dispatch("analytics", {
        event: "search_performed",
        data: {
          query: searchTerm,
          resultCount: results.length,
          responseTime: Date.now() - startTime,
        },
      });
    } catch (error) {
      console.error("Search error:", error);
      searchResults.set([]);
    } finally {
      isSearching.set(false);
    }
  }

  function handleResultClick(result: SearchResult) {
    selectedResult.set(result);

    // Track click analytics
    if (enableAnalytics) {
      dispatch("analytics", {
        event: "result_clicked",
        data: {
          resultId: result.id,
          rank: result.rank,
          query: get(searchQuery),
        },
      });
    }

    dispatch("select", { result });
  }

  function applySorting(
    results: SearchResult[],
    sortBy: string,
    sortOrder: string
  ): SearchResult[] {
    return [...results].sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case "similarity":
          comparison = b.similarity - a.similarity;
          break;
        case "date":
          const dateA = new Date(a.metadata.createdAt || 0);
          const dateB = new Date(b.metadata.createdAt || 0);
          comparison = dateB.getTime() - dateA.getTime();
          break;
        case "title":
          comparison = a.title.localeCompare(b.title);
          break;
        case "relevance":
        default:
          comparison = b.relevance - a.relevance;
          break;
      }

      return sortOrder === "asc" ? -comparison: comparison
    });
  }

  // ============================================================================
  // FILTER MANAGEMENT
  // ============================================================================

  function applyFilters() {
    dispatch("filter", { filters: get(searchFilters) });
    if (get(searchQuery).trim()) {
      performSearch();
    }
  }

  function resetFilters() {
    searchFilters.set({
      documentTypes: [],
      jurisdictions: [],
      dateRange: {},
      similarityThreshold: 0.7,
      maxResults: maxResults,
      tags: [],
      sortBy: "relevance",
      sortOrder: "desc",
    });
    applyFilters();
  }

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  function formatSimilarity(similarity: number): string {
    return `${Math.round(similarity * 100)}%`;
  }

  function formatDate(dateString?: string): string {
    if (!dateString) return "Unknown";
    return new Date(dateString).toLocaleDateString();
  }

  function formatFileSize(bytes?: number): string {
    if (!bytes) return "Unknown";
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round((bytes / Math.pow(1024, i)) * 100) / 100} ${sizes[i]}`;
  }

  function getDocumentTypeColor(type?: string): string {
    const colors = {
      contract: "blue",
      motion: "green",
      brief: "purple",
      evidence: "red",
      correspondence: "yellow",
      statute: "indigo",
      regulation: "pink",
      case_law: "gray",
      other: "slate",
    };
    return colors[type as keyof typeof colors] || "gray";
  }

  function highlightText(text: string, highlights: string[]): string {
    let highlightedText = text;
    highlights.forEach((highlight) => {
      const regex = new RegExp(`(${highlight})`, "gi");
      highlightedText = highlightedText.replace(
        regex,
        '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
      );
    });
    return highlightedText;
  }

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(() => {
    // Load search history from localStorage
    const savedHistory = localStorage.getItem("vector-search-history");
    if (savedHistory) {
      try {
        searchHistory.set(JSON.parse(savedHistory));
      } catch (e) {
        console.warn("Failed to load search history");
      }
    }

    // Load analytics if enabled
    if (enableAnalytics) {
      loadAnalytics();
    }

    // Auto-save search history
    searchHistory.subscribe((history) => {
      localStorage.setItem("vector-search-history", JSON.stringify(history));
    });
  });

  async function loadAnalytics() {
    try {
      const response = await fetch("/api/search/analytics");
      if (response.ok) {
        const data = await response.json();
        searchAnalytics.set(data);
      }
    } catch (error) {
      console.warn("Failed to load analytics:", error);
    }
  }
</script>

<!-- Main Search Interface -->
<div class="enhanced-vector-search {className}">
  <!-- Search Header -->
  <div class="search-header">
    <div class="search-input-container">
      <div class="relative">
        <Search class="search-icon" size={20} />
        <Input
          bind:value={$searchQuery}
          placeholder="Search legal documents with AI-powered semantic search..."
          class="search-input"
          on:keydown={(e) => e.key === "Enter" && performSearch()}
          disabled={$isSearching}
        />
        {#if $isSearching}
          <Loader2 class="loading-icon animate-spin" size={20} />
        {/if}
      </div>

      <div class="search-actions">
        <Button
          on:click={() => performSearch()}
          disabled={$isSearching || !$searchQuery.trim()}
          class="search-button"
        >
          {#if $isSearching}
            <Loader2 class="mr-2 animate-spin" size={16} />
            Searching...
          {:else}
            <Search class="mr-2" size={16} />
            Search
          {/if}
        </Button>

        {#if enableFilters}
          <Button
            variant="outline"
            on:click={() => showFilters.update((s) => !s)}
            class="filter-button"
          >
            <Filter class="mr-2" size={16} />
            Filters
            {#if $showFilters}
              <ChevronUp class="ml-2" size={16} />
            {:else}
              <ChevronDown class="ml-2" size={16} />
            {/if}
          </Button>
        {/if}

        {#if enableAnalytics}
          <Button
            variant="outline"
            on:click={() => showAnalytics.update((s) => !s)}
          >
            <BarChart3 class="mr-2" size={16} />
            Analytics
          </Button>
        {/if}
      </div>
    </div>

    <!-- Search History -->
    {#if $searchHistory.length > 0}
      <div class="search-history">
        <p class="history-label">Recent searches:</p>
        <div class="history-tags">
          {#each $searchHistory.slice(0, 5) as historyItem}
            <Button
              variant="ghost"
              size="sm"
              on:click={() => {
                searchQuery.set(historyItem);
                performSearch(historyItem);
              }}
              class="history-tag"
            >
              <Clock class="mr-1" size={12} />
              {historyItem}
            </Button>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <!-- Advanced Filters -->
  {#if $showFilters && enableFilters}
    <Card class="filters-panel">
      <CardHeader>
        <CardTitle class="flex items-center justify-between">
          <span>Advanced Filters</span>
          <Button variant="ghost" size="sm" on:click={resetFilters}>
            Reset
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="filter-grid">
          <!-- Document Types -->
          <div class="filter-group">
            <label class="filter-label">Document Types</label>
            <div class="checkbox-group">
              {#each documentTypes as type}
                <Checkbox
                  bind:checked={
                    $searchFilters.documentTypes.includes(type.value
                  }
                  on:change={() => {
                    searchFilters.update((f) => {
                      if (f.documentTypes.includes(type.value)) {
                        f.documentTypes = f.documentTypes.filter(
                          (t) => t !== type.value
                        );
                      } else {
                        f.documentTypes = [...f.documentTypes, type.value];
                      }
                      return f;
                    });
                  }}
                >
                  {type.label}
                </Checkbox>
              {/each}
            </div>
          </div>

          <!-- Jurisdictions -->
          <div class="filter-group">
            <label class="filter-label">Jurisdictions</label>
            <div class="checkbox-group">
              {#each jurisdictions as jurisdiction}
                <Checkbox
                  bind:checked={
                    $searchFilters.jurisdictions.includes(jurisdiction.value
                  }
                  on:change={() => {
                    searchFilters.update((f) => {
                      if (f.jurisdictions.includes(jurisdiction.value)) {
                        f.jurisdictions = f.jurisdictions.filter(
                          (j) => j !== jurisdiction.value
                        );
                      } else {
                        f.jurisdictions = [
                          ...f.jurisdictions,
                          jurisdiction.value,
                        ];
                      }
                      return f;
                    });
                  }}
                >
                  {jurisdiction.label}
                </Checkbox>
              {/each}
            </div>
          </div>

          <!-- Similarity Threshold -->
          <div class="filter-group">
            <label class="filter-label">
              Similarity Threshold: {formatSimilarity(
                $searchFilters.similarityThreshold
              )}
            </label>
            <Slider
              bind:value={$searchFilters.similarityThreshold}
              min={0.1}
              max={1.0}
              step={0.05}
              class="similarity-slider"
            />
          </div>

          <!-- Sort Options -->
          <div class="filter-group">
            <label class="filter-label">Sort By</label>
            <Select bind:value={$searchFilters.sortBy}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {#each sortOptions as option}
                  <SelectItem value={option.value}>{option.label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
          </div>
        </div>

        <Button on:click={applyFilters} class="w-full">Apply Filters</Button>
      </CardContent>
    </Card>
  {/if}

  <!-- Search Results -->
  {#if $hasResults}
    <div class="search-results">
      <!-- Results Header -->
      <div class="results-header">
        <div class="results-meta">
          <h3 class="results-title">
            Search Results ({$searchResults.length})
          </h3>
          <div class="results-stats">
            Average Similarity: {formatSimilarity($averageSimilarity)}
          </div>
        </div>

        <!-- Quick Stats -->
        {#if $topDocumentTypes.length > 0}
          <div class="quick-stats">
            <p class="stats-label">Document Types:</p>
            <div class="stats-badges">
              {#each $topDocumentTypes as [type, count]}
                <Badge variant={getDocumentTypeColor(type)}>
                  {documentTypes.find((t) => t.value === type)?.label || type}: {count}
                </Badge>
              {/each}
            </div>
          </div>
        {/if}
      </div>

      <!-- Results List -->
      <div class="results-list">
        {#each $searchResults as result (result.id)}
          <Card class="result-item" on:click={() => handleResultClick(result)}>
            <CardContent class="result-content">
              <!-- Result Header -->
              <div class="result-header">
                <div class="result-title-section">
                  <h4 class="result-title">{result.title}</h4>
                  <div class="result-meta">
                    <Badge
                      variant={getDocumentTypeColor(
                        result.metadata.documentType
                      )}
                    >
                      {documentTypes.find(
                        (t) => t.value === result.metadata.documentType
                      )?.label || "Document"}
                    </Badge>
                    <span class="result-date"
                      >{formatDate(result.metadata.createdAt)}</span
                    >
                    {#if result.metadata.fileSize}
                      <span class="result-size"
                        >{formatFileSize(result.metadata.fileSize)}</span
                      >
                    {/if}
                  </div>
                </div>

                <div class="result-metrics">
                  <div class="metric">
                    <Target size={14} />
                    <span class="metric-label">Similarity</span>
                    <span class="metric-value"
                      >{formatSimilarity(result.similarity)}</span
                    >
                  </div>
                  <div class="metric">
                    <TrendingUp size={14} />
                    <span class="metric-label">Rank</span>
                    <span class="metric-value">#{result.rank}</span>
                  </div>
                </div>
              </div>

              <!-- Result Content -->
              <div class="result-snippet">
                {@html highlightText(result.snippet, result.highlights)}
              </div>

              <!-- Result Tags -->
              {#if result.metadata.tags && result.metadata.tags.length > 0}
                <div class="result-tags">
                  {#each result.metadata.tags as tag}
                    <Badge variant="outline" class="tag-badge">{tag}</Badge>
                  {/each}
                </div>
              {/if}

              <!-- Result Actions -->
              <div class="result-actions">
                <Button variant="ghost" size="sm">
                  <Eye class="mr-1" size={14} />
                  View
                </Button>
                <Button variant="ghost" size="sm">
                  <Download class="mr-1" size={14} />
                  Download
                </Button>
                <Button variant="ghost" size="sm">
                  <Share2 class="mr-1" size={14} />
                  Share
                </Button>
              </div>
            </CardContent>
          </Card>
        {/each}
      </div>
    </div>
  {:else if $searchQuery.trim() && !$isSearching}
    <!-- No Results -->
    <div class="no-results">
      <div class="no-results-content">
        <Search class="no-results-icon" size={48} />
        <h3 class="no-results-title">No results found</h3>
        <p class="no-results-description">
          Try adjusting your search terms or filters
        </p>
        <Button variant="outline" on:click={resetFilters}>Reset Filters</Button>
      </div>
    </div>
  {/if}

  <!-- Analytics Panel -->
  {#if $showAnalytics && enableAnalytics}
    <Dialog bind:open={$showAnalytics}>
      <DialogContent class="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Search Analytics</DialogTitle>
        </DialogHeader>

        <Tabs value="overview" class="analytics-tabs">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="queries">Top Queries</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" class="analytics-overview">
            <div class="analytics-grid">
              <Card class="metric-card">
                <CardContent class="metric-content">
                  <div class="metric-icon">
                    <Search size={24} />
                  </div>
                  <div class="metric-info">
                    <p class="metric-label">Total Searches</p>
                    <p class="metric-value">{$searchAnalytics.totalSearches}</p>
                  </div>
                </CardContent>
              </Card>

              <Card class="metric-card">
                <CardContent class="metric-content">
                  <div class="metric-icon">
                    <Target size={24} />
                  </div>
                  <div class="metric-info">
                    <p class="metric-label">Avg. Results</p>
                    <p class="metric-value">
                      {$searchAnalytics.averageResultCount}
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card class="metric-card">
                <CardContent class="metric-content">
                  <div class="metric-icon">
                    <Zap size={24} />
                  </div>
                  <div class="metric-info">
                    <p class="metric-label">Avg. Response</p>
                    <p class="metric-value">
                      {$searchAnalytics.responseTime}ms
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card class="metric-card">
                <CardContent class="metric-content">
                  <div class="metric-icon">
                    <Brain size={24} />
                  </div>
                  <div class="metric-info">
                    <p class="metric-label">Avg. Similarity</p>
                    <p class="metric-value">
                      {formatSimilarity($searchAnalytics.averageSimilarity)}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="performance" class="analytics-performance">
            <div class="performance-metrics">
              <Card>
                <CardHeader>
                  <CardTitle>Performance Breakdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <div class="performance-bars">
                    <div class="performance-item">
                      <span class="performance-label">Vector Search</span>
                      <Progress
                        value={($searchAnalytics.performanceMetrics
                          .vectorSearchTime /
                          $searchAnalytics.performanceMetrics.totalTime) *
                          100}
                        class="performance-bar"
                      />
                      <span class="performance-value"
                        >{$searchAnalytics.performanceMetrics
                          .vectorSearchTime}ms</span
                      >
                    </div>
                    <div class="performance-item">
                      <span class="performance-label">Ranking</span>
                      <Progress
                        value={($searchAnalytics.performanceMetrics
                          .rankingTime /
                          $searchAnalytics.performanceMetrics.totalTime) *
                          100}
                        class="performance-bar"
                      />
                      <span class="performance-value"
                        >{$searchAnalytics.performanceMetrics
                          .rankingTime}ms</span
                      >
                    </div>
                    <div class="performance-item">
                      <span class="performance-label">Total</span>
                      <Progress value={100} class="performance-bar" />
                      <span class="performance-value"
                        >{$searchAnalytics.performanceMetrics.totalTime}ms</span
                      >
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="queries" class="analytics-queries">
            {#if $searchAnalytics.topQueries.length > 0}
              <Card>
                <CardHeader>
                  <CardTitle>Most Popular Queries</CardTitle>
                </CardHeader>
                <CardContent>
                  <div class="top-queries-list">
                    {#each $searchAnalytics.topQueries as { query, count }}
                      <div class="query-item">
                        <span class="query-text">{query}</span>
                        <Badge variant="secondary">{count} searches</Badge>
                      </div>
                    {/each}
                  </div>
                </CardContent>
              </Card>
            {:else}
              <div class="no-analytics">
                <p>
                  No query data available yet. Perform some searches to see
                  analytics.
                </p>
              </div>
            {/if}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  {/if}
</div>

<style>
  .enhanced-vector-search {
    display: flex
    flex-direction: column
    gap: 1.5rem;
  }

  .search-header {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .search-input-container {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  @media (min-width: 1024px) {
    .search-input-container {
      flex-direction: row
    }
  }

  .search-input {
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    height: 3rem;
    font-size: 1rem;
  }

  .search-icon {
    position: absolute
    left: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--muted-foreground);
  }

  .loading-icon {
    position: absolute
    right: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--primary);
  }

  .search-actions {
    display: flex
    gap: 0.5rem;
  }

  @media (min-width: 1024px) {
    .search-actions {
      flex-shrink: 0;
    }
  }

  .search-button,
  .filter-button {
    height: 3rem;
  }

  .search-button {
    padding-left: 1.5rem;
    padding-right: 1.5rem;
  }

  .search-history {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  @media (min-width: 640px) {
    .search-history {
      flex-direction: row
      align-items: center
    }
  }

  .history-label {
    font-size: 0.875rem;
    color: var(--muted-foreground);
  }

  .history-tags {
    display: flex
    flex-wrap: wrap
    gap: 0.5rem;
  }

  .history-tag {
    height: 1.75rem;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
    font-size: 0.75rem;
  }

  .filters-panel {
    border: 2px dashed;
    border-color: rgba(107, 114, 128, 0.25);
  }

  .filter-grid {
    display: grid
    grid-template-columns: 1fr;
    gap: 1rem;
  }

  @media (min-width: 768px) {
    .filter-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (min-width: 1024px) {
    .filter-grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }

  .filter-group {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  .filter-label {
    font-size: 0.875rem;
    font-weight: 500;
  }

  .checkbox-group {
    display: flex
    flex-direction: column
    gap: 0.25rem;
  }

  .similarity-slider {
    width: 100%;
  }

  .search-results {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .results-header {
    display: flex
    flex-direction: column
    gap: 0.75rem;
  }

  .results-meta {
    display: flex
    align-items: center
    justify-content: space-between;
  }

  .results-title {
    font-size: 1.25rem;
    font-weight: 600;
  }

  .results-stats {
    font-size: 0.875rem;
    color: var(--muted-foreground);
  }

  .quick-stats {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  @media (min-width: 640px) {
    .quick-stats {
      flex-direction: row
      align-items: center
    }
  }

  .stats-label {
    font-size: 0.875rem;
    font-weight: 500;
  }

  .stats-badges {
    display: flex
    flex-wrap: wrap
    gap: 0.5rem;
  }

  .results-list {
    display: flex
    flex-direction: column
    gap: 0.75rem;
  }

  .result-item {
    cursor: pointer
    transition: box-shadow 0.2s;
  }

  .result-item:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }

  .result-content {
    display: flex
    flex-direction: column
    gap: 0.75rem;
  }

  .result-header {
    display: flex
    align-items: flex-start;
    justify-content: space-between;
  }

  .result-title-section {
    flex: 1;
    min-width: 0;
  }

  .result-title {
    font-weight: 500;
    font-size: 1.125rem;
    overflow: hidden
    text-overflow: ellipsis
    white-space: nowrap
  }

  .result-meta {
    display: flex
    align-items: center
    gap: 0.5rem;
    margin-top: 0.25rem;
    font-size: 0.875rem;
    color: var(--muted-foreground);
  }

  .result-date,
  .result-size {
    font-size: 0.75rem;
  }

  .result-metrics {
    display: flex
    flex-direction: column
    gap: 0.5rem;
    text-align: right
  }

  .metric {
    display: flex
    align-items: center
    gap: 0.25rem;
    font-size: 0.75rem;
  }

  .metric-label {
    color: var(--muted-foreground);
  }

  .metric-value {
    font-weight: 500;
  }

  .result-snippet {
    font-size: 0.875rem;
    line-height: 1.625;
  }

  .result-tags {
    display: flex
    flex-wrap: wrap
    gap: 0.25rem;
  }

  .tag-badge {
    font-size: 0.75rem;
  }

  .result-actions {
    display: flex
    gap: 0.5rem;
  }

  .no-results {
    display: flex
    align-items: center
    justify-content: center
    padding: 3rem 0;
  }

  .no-results-content {
    text-align: center
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .no-results-icon {
    margin: 0 auto;
    color: var(--muted-foreground);
  }

  .no-results-title {
    font-size: 1.125rem;
    font-weight: 500;
  }

  .no-results-description {
    color: var(--muted-foreground);
  }

  .analytics-tabs {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .analytics-overview {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .analytics-grid {
    display: grid
    grid-template-columns: 1fr;
    gap: 1rem;
  }

  @media (min-width: 640px) {
    .analytics-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (min-width: 1024px) {
    .analytics-grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }

  .metric-card {
    padding: 1rem;
  }

  .metric-content {
    display: flex
    align-items: center
    gap: 0.75rem;
  }

  .metric-icon {
    padding: 0.5rem;
    background-color: rgba(var(--primary-rgb), 0.1);
    border-radius: 0.5rem;
  }

  .metric-info {
    display: flex
    flex-direction: column
    gap: 0.25rem;
  }

  .metric-label {
    font-size: 0.875rem;
    color: var(--muted-foreground);
  }

  .metric-value {
    font-size: 1.25rem;
    font-weight: 600;
  }

  .performance-metrics {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .performance-bars {
    display: flex
    flex-direction: column
    gap: 0.75rem;
  }

  .performance-item {
    display: flex
    align-items: center
    gap: 0.75rem;
  }

  .performance-label {
    width: 6rem;
    font-size: 0.875rem;
  }

  .performance-bar {
    flex: 1;
  }

  .performance-value {
    width: 4rem;
    font-size: 0.875rem;
    font-family: monospace
    text-align: right
  }

  .top-queries-list {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  .query-item {
    display: flex
    align-items: center
    justify-content: space-between;
    padding: 0.5rem;
    border-radius: 0.375rem;
    border: 1px solid var(--border);
  }

  .query-text {
    font-family: monospace
    font-size: 0.875rem;
  }

  .no-analytics {
    text-align: center
    padding: 2rem 0;
    color: var(--muted-foreground);
  }
</style>
