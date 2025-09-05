<script lang="ts">
  import { browser } from "$app/environment";
  import { goto } from "$app/navigation";
  import { page } from "$app/stores";
  import { Tooltip } from "$lib/components/ui";
  import { Button } from "$lib/components/ui/button";
  import { notifications } from "$lib/stores/notification";
  import {
    AlertCircle,
    ArrowRight,
    Clock,
    Database,
    Eye,
    FileText,
    Filter,
    Loader2,
    Search,
    Star,
    Users,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  interface SearchResult {
    id: string
    type: string
    title: string
    description?: string;
    content?: string;
    url?: string;
    relevance?: number;
    score?: number;
    metadata?: Record<string, unknown>;
}
  interface SearchResults {
    query: string
    results: SearchResult[];
    total: number
    page: number
    hasMore: boolean
    executionTime?: number;
    source?: string;
}
  // Search state
  let searchQuery = "";
  let searchResults: SearchResults | null = null;
  let isSearching = false;
  let searchError: string | null = null;
  let searchHistory: string[] = [];

  // Filters
  let selectedType = "";
  let selectedCaseId = "";
  let threshold = 0.7;
  let limit = 20;
  let showAdvancedFilters = false;

  // Recent searches (simulated)
  let recentSearches = [
    "contract fraud investigation",
    "witness testimony evidence",
    "financial records analysis",
    "crime scene documentation",
    "legal precedent research",
  ];

  // Popular searches (simulated)
  let popularSearches = [
    "fraud cases",
    "evidence analysis",
    "witness statements",
    "legal research",
    "case documentation",
  ];

  onMount(() => {
    // Load search query from URL if present
    const urlQuery = $page.url.searchParams.get("q");
    if (urlQuery) {
      searchQuery = urlQuery;
      performSearch();
}
    // Load search history from localStorage
    if (browser) {
      const savedHistory = localStorage.getItem("searchHistory");
      if (savedHistory) {
        searchHistory = JSON.parse(savedHistory);
}
}
  });

  async function performSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    searchError = null;
    searchResults = null;

    try {
      // Update URL with search query
      const url = new URL(window.location.href);
      url.searchParams.set("q", searchQuery);
      if (selectedType) url.searchParams.set("type", selectedType);
      if (selectedCaseId) url.searchParams.set("caseId", selectedCaseId);
      url.searchParams.set("threshold", threshold.toString());
      url.searchParams.set("limit", limit.toString());

      window.history.replaceState({}, "", url);

      // Perform search
      const searchParams = new URLSearchParams({
        q: searchQuery,
        limit: limit.toString(),
        threshold: threshold.toString(),
      });

      if (selectedType) searchParams.set("type", selectedType);
      if (selectedCaseId) searchParams.set("caseId", selectedCaseId);

      const response = await fetch(`/api/search?${searchParams}`);
      const result = await response.json();

      if (response.ok) {
        searchResults = result.data;

        // Add to search history
        if (!searchHistory.includes(searchQuery)) {
          searchHistory = [searchQuery, ...searchHistory.slice(0, 9)]; // Keep last 10
          if (browser) {
            localStorage.setItem(
              "searchHistory",
              JSON.stringify(searchHistory)
            );
}
}
      } else {
        throw new Error(result.error || "Search failed");
}
    } catch (error) {
      console.error("Search error:", error);
      searchError = error instanceof Error ? error.message : "Search failed";
      notifications.add({
        type: "error",
        title: "Search Error",
        message: searchError,
      });
    } finally {
      isSearching = false;
}
}
  function handleSearchSubmit(e: Event) {
    e.preventDefault();
    performSearch();
}
  function selectQuickSearch(query: string) {
    searchQuery = query;
    performSearch();
}
  function clearFilters() {
    selectedType = "";
    selectedCaseId = "";
    threshold = 0.7;
    limit = 20;
    showAdvancedFilters = false;
}
  function formatScore(score: number): string {
    return `${(score * 100).toFixed(1)}%`;
}
  function getResultIcon(type: string) {
    switch (type) {
      case "case":
        return Database;
      case "evidence":
        return FileText;
      case "participant":
        return Users;
      default:
        return FileText;
}
}
  function handleResultClick(result: SearchResult) {
    // Navigate to the appropriate page based on result type
    switch (result.type) {
      case "case":
        goto(`/cases/${result.id}`);
        break;
      case "evidence":
        goto(`/evidence/${result.id}`);
        break;
      default:
        // Handle other types or show details modal
        break;
}
}
</script>

<svelte:head>
  <title>Search - Legal Case Management</title>
  <meta
    name="description"
    content="Advanced search across cases, evidence, and participant data"
  />
</svelte:head>

<div class="space-y-4">
  <!-- Header -->
  <div
    class="space-y-4"
  >
    <h1 class="space-y-4">
      <Search class="space-y-4" />
      Advanced Search
    </h1>
    <p class="space-y-4">
      Search across cases, evidence, and participant data using AI-powered
      semantic search
    </p>
  </div>

  <div class="space-y-4">
    <!-- Main Search Panel -->
    <div class="space-y-4">
      <!-- Search Form -->
      <div class="space-y-4">
        <form onsubmit={handleSearchSubmit} class="space-y-4">
          <!-- Main Search Input -->
          <div class="space-y-4">
            <Search
              class="space-y-4"
            />
            <input
              type="text"
              bind:value={searchQuery}
              placeholder="Search cases, evidence, participants..."
              class="space-y-4"
              required
            />
          </div>

          <!-- Search Actions -->
          <div class="space-y-4">
            <Button type="submit" disabled={isSearching || !searchQuery.trim()}>
              {#if isSearching}
                <Loader2 class="space-y-4" />
                Searching...
              {:else}
                <Search class="space-y-4" />
                Search
              {/if}
            </Button>

            <Tooltip content="Show/hide advanced search filters">
              <Button
                variant="outline"
                type="button"
                onclick={() => (showAdvancedFilters = !showAdvancedFilters)}
              >
                <Filter class="space-y-4" />
                Filters
              </Button>
            </Tooltip>

            {#if selectedType || selectedCaseId || threshold !== 0.7 || limit !== 20}
              <Tooltip content="Clear all filters">
                <Button
                  variant="outline"
                  type="button"
                  onclick={() => clearFilters()}
                >
                  Clear Filters
                </Button>
              </Tooltip>
            {/if}
          </div>

          <!-- Advanced Filters -->
          {#if showAdvancedFilters}
            <div class="space-y-4">
              <h3 class="space-y-4">Advanced Filters</h3>

              <div class="space-y-4">
                <div>
                  <label
                    for="type-filter"
                    class="space-y-4"
                  >
                    Content Type
                  </label>
                  <select
                    id="type-filter"
                    bind:value={selectedType}
                    class="space-y-4"
                  >
                    <option value="">All Types</option>
                    <option value="case">Cases</option>
                    <option value="evidence">Evidence</option>
                    <option value="participant">Participants</option>
                  </select>
                </div>

                <div>
                  <label
                    for="case-filter"
                    class="space-y-4"
                  >
                    Specific Case
                  </label>
                  <input
                    id="case-filter"
                    type="text"
                    bind:value={selectedCaseId}
                    placeholder="Case ID (optional)"
                    class="space-y-4"
                  />
                </div>

                <div>
                  <label
                    for="threshold-filter"
                    class="space-y-4"
                  >
                    Relevance Threshold
                  </label>
                  <select
                    id="threshold-filter"
                    bind:value={threshold}
                    class="space-y-4"
                  >
                    <option value={0.5}>50% - Very Broad</option>
                    <option value={0.6}>60% - Broad</option>
                    <option value={0.7}>70% - Balanced</option>
                    <option value={0.8}>80% - Precise</option>
                    <option value={0.9}>90% - Very Precise</option>
                  </select>
                </div>

                <div>
                  <label
                    for="limit-filter"
                    class="space-y-4"
                  >
                    Max Results
                  </label>
                  <select
                    id="limit-filter"
                    bind:value={limit}
                    class="space-y-4"
                  >
                    <option value={10}>10 results</option>
                    <option value={20}>20 results</option>
                    <option value={50}>50 results</option>
                    <option value={100}>100 results</option>
                  </select>
                </div>
              </div>
            </div>
          {/if}
        </form>
      </div>

      <!-- Search Error -->
      {#if searchError}
        <div class="space-y-4">
          <div class="space-y-4">
            <AlertCircle class="space-y-4" />
            <div>
              <h3 class="space-y-4">Search Error</h3>
              <p class="space-y-4">{searchError}</p>
            </div>
          </div>
        </div>
      {/if}

      <!-- Search Results -->
      {#if searchResults}
        <div class="space-y-4">
          <!-- Results Header -->
          <div class="space-y-4">
            <div class="space-y-4">
              <h2 class="space-y-4">
                Search Results
              </h2>
              <div class="space-y-4">
                <span>{searchResults.results.length} results</span>
                <span>in {searchResults.executionTime}ms</span>
                <span class="space-y-4">via {searchResults.source}</span>
              </div>
            </div>
          </div>

          <!-- Results List -->
          {#if searchResults.results.length > 0}
            <div class="space-y-4">
              {#each searchResults.results as result}
                <div
                  class="space-y-4"
                  onclick={() => handleResultClick(result)}
                  onkeydown={(e) =>
                    e.key === "Enter" && handleResultClick(result)}
                  role="button"
                  tabindex={0}
                >
                  <div class="space-y-4">
                    <div class="space-y-4">
                      <svelte:component
                        this={getResultIcon(result.type)}
                        class="space-y-4"
                      />
                    </div>

                    <div class="space-y-4">
                      <div class="space-y-4">
                        <h3 class="space-y-4">
                          {result.title || "Untitled"}
                        </h3>
                        <div class="space-y-4">
                          <span
                            class="space-y-4"
                          >
                            {result.type}
                          </span>
                          <span
                            class="space-y-4"
                          >
                            {formatScore(result.score ?? 0)} match
                          </span>
                        </div>
                      </div>

                      <p class="space-y-4">
                        {result.content}
                      </p>

                      {#if result.metadata && Object.keys(result.metadata).length > 0}
                        <div class="space-y-4">
                          {#each Object.entries(result.metadata) as [key, value]}
                            {#if value && key !== "type"}
                              <span
                                class="space-y-4"
                              >
                                {key}: {value}
                              </span>
                            {/if}
                          {/each}
                        </div>
                      {/if}
                    </div>

                    <div class="space-y-4">
                      <ArrowRight class="space-y-4" />
                    </div>
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="space-y-4">
              <Search class="space-y-4" />
              <h3 class="space-y-4">
                No results found
              </h3>
              <p class="space-y-4">
                Try adjusting your search terms or filters
              </p>
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- Sidebar -->
    <div class="space-y-4">
      <!-- Quick Search Suggestions -->
      {#if !searchResults}
        <div class="space-y-4">
          <h3 class="space-y-4">
            <Star class="space-y-4" />
            Popular Searches
          </h3>
          <div class="space-y-4">
            {#each popularSearches as search}
              <button
                type="button"
                class="space-y-4"
                onclick={() => selectQuickSearch(search)}
              >
                {search}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Search History -->
      {#if searchHistory.length > 0}
        <div class="space-y-4">
          <h3 class="space-y-4">
            <Clock class="space-y-4" />
            Recent Searches
          </h3>
          <div class="space-y-4">
            {#each searchHistory.slice(0, 5) as search}
              <button
                type="button"
                class="space-y-4"
                onclick={() => selectQuickSearch(search)}
              >
                {search}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Search Tips -->
      <div class="space-y-4">
        <h3 class="space-y-4">Search Tips</h3>
        <ul class="space-y-4">
          <li>• Use natural language queries</li>
          <li>• Be specific for better results</li>
          <li>• Try synonyms if no results</li>
          <li>• Use filters to narrow results</li>
          <li>• Higher thresholds = more precise</li>
        </ul>
      </div>

      <!-- Quick Actions -->
      <div class="space-y-4">
        <h3 class="space-y-4">Quick Actions</h3>
        <div class="space-y-4">
          <a href="/cases" class="space-y-4">
            <Button variant="outline" class="space-y-4">
              <Database class="space-y-4" />
              Browse Cases
            </Button>
          </a>
          <a href="/evidence" class="space-y-4">
            <Button variant="outline" class="space-y-4">
              <FileText class="space-y-4" />
              Browse Evidence
            </Button>
          </a>
          <a href="/ai-assistant" class="space-y-4">
            <Button variant="outline" class="space-y-4">
              <Eye class="space-y-4" />
              AI Assistant
            </Button>
          </a>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical
    line-clamp: 2;
    overflow: hidden
}
</style>
