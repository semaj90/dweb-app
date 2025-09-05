<script lang="ts">
  import { Badge } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
  } from "$lib/components/ui/card";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
  } from "$lib/components/ui/select";
  import { Slider } from "$lib/components/ui/slider";
  import { Switch } from "$lib/components/ui/switch";
  import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
  } from "$lib/components/ui/tooltip";
  import {
    Brain,
    Briefcase,
    FileText,
    Info,
    Scale,
    Search,
    Sparkles,
  } from "lucide-svelte";

  let query = $state("");
  let documentType = $state<"document" | "evidence" | "case">("document");
  let threshold = $state([0.7]);
  let limit = $state(10);
  let personalized = $state(false);
  let includeExplanation = $state(true);
  let searching = $state(false);
  let results = $state<any[]>([]);
  let error = $state<string | null>(null);

  // Document type icons
  const typeIcons = {
    document: FileText,
    evidence: Briefcase,
    case: Scale,
  };

  async function performSearch() {
    if (!query.trim()) return;

    searching = true;
    error = null;
    results = [];

    try {
      const response = await fetch("/api/vector/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          options: {
            documentType,
            threshold: threshold[0],
            limit,
            personalized,
            includeExplanation,
            userId: "demo-user", // In production, get from auth
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Search failed");
      }

      const data = await response.json();
      results = data.results || [];
    } catch (err) {
      error = err instanceof Error ? err.message : "Search failed";
    } finally {
      searching = false;
    }
  }

  function getRankingColor(factor: string, value: number): string {
    if (value >= 0.8) return "text-green-600";
    if (value >= 0.6) return "text-yellow-600";
    return "text-gray-600";
  }

  function formatScore(score: number): string {
    return (score * 100).toFixed(1) + "%";
  }
</script>

<div class="container mx-auto py-8 max-w-6xl">
  <div class="mb-8">
    <h1 class="text-4xl font-bold mb-2 flex items-center gap-2">
      <Brain class="h-10 w-10" />
      AI-Powered Vector Search & Ranking
    </h1>
    <p class="text-lg text-muted-foreground">
      Semantic search with multi-factor ranking using nomic-embed-text
    </p>
  </div>

  <!-- Search Configuration -->
  <Card class="mb-8">
    <CardHeader>
      <CardTitle>Search Configuration</CardTitle>
      <CardDescription>
        Configure your semantic search parameters
      </CardDescription>
    </CardHeader>
    <CardContent class="space-y-6">
      <!-- Search Query -->
      <div class="space-y-2">
        <Label for="query">Search Query</Label>
        <div class="flex gap-2">
          <Input
            id="query"
            bind:value={query}
            placeholder="Enter your legal search query..."
            class="flex-1"
            on:keydown={(e) => e.key === "Enter" && performSearch()}
          />
          <Button
            on:click={performSearch}
            disabled={searching || !query.trim()}
          >
            {#if searching}
              <div
                class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"
              ></div>
              Searching...
            {:else}
              <Search class="mr-2 h-4 w-4" />
              Search
            {/if}
          </Button>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Document Type -->
        <div class="space-y-2">
          <Label for="type">Document Type</Label>
          <Select bind:value={documentType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {#each Object.entries(typeIcons) as [value, Icon]}
                <SelectItem {value}>
                  <div class="flex items-center gap-2">
                    <Icon class="h-4 w-4" />
                    <span class="capitalize">{value}</span>
                  </div>
                </SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>

        <!-- Result Limit -->
        <div class="space-y-2">
          <Label for="limit">Result Limit</Label>
          <Select bind:value={limit}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={5}>5 results</SelectItem>
              <SelectItem value={10}>10 results</SelectItem>
              <SelectItem value={20}>20 results</SelectItem>
              <SelectItem value={50}>50 results</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <!-- Similarity Threshold -->
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <Label for="threshold">Similarity Threshold</Label>
          <span class="text-sm text-muted-foreground"
            >{formatScore(threshold[0])}</span
          >
        </div>
        <Slider
          id="threshold"
          bind:value={threshold}
          min={0.3}
          max={0.95}
          step={0.05}
          class="py-2"
        />
        <div class="flex justify-between text-xs text-muted-foreground">
          <span>Broad (30%)</span>
          <span>Focused (95%)</span>
        </div>
      </div>

      <!-- Options -->
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <div class="space-y-0.5">
            <Label for="personalized" class="text-base font-medium">
              Personalized Results
            </Label>
            <p class="text-sm text-muted-foreground">
              Rank results based on your search history
            </p>
          </div>
          <Switch id="personalized" bind:checked={personalized} />
        </div>

        <div class="flex items-center justify-between">
          <div class="space-y-0.5">
            <Label for="explanation" class="text-base font-medium">
              Include Explanations
            </Label>
            <p class="text-sm text-muted-foreground">
              Show why each result was ranked
            </p>
          </div>
          <Switch id="explanation" bind:checked={includeExplanation} />
        </div>
      </div>
    </CardContent>
  </Card>

  <!-- Search Results -->
  {#if error}
    <Card class="border-destructive">
      <CardContent class="pt-6">
        <p class="text-destructive">{error}</p>
      </CardContent>
    </Card>
  {:else if results.length > 0}
    <div class="space-y-4">
      <h2 class="text-2xl font-semibold flex items-center gap-2">
        <Sparkles class="h-6 w-6" />
        Search Results
      </h2>

      {#each results as result, index}
        <Card class="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div class="flex items-start justify-between">
              <div class="flex-1">
                <CardTitle class="text-lg flex items-center gap-2">
                  {@const Icon = typeIcons[documentType]}
                  <Icon class="h-5 w-5" />
                  Result #{index + 1}
                </CardTitle>
                <CardDescription class="mt-1">
                  {result.metadata?.filename ||
                    result.metadata?.title ||
                    "Untitled"}
                </CardDescription>
              </div>
              <Badge variant="secondary" class="text-lg">
                {formatScore(result.score)}
              </Badge>
            </div>
          </CardHeader>
          <CardContent class="space-y-4">
            <!-- Content Preview -->
            <div class="bg-muted p-4 rounded-lg">
              <p class="text-sm line-clamp-3">
                {result.content}
              </p>
            </div>

            <!-- Ranking Factors -->
            <div class="space-y-2">
              <div class="flex items-center gap-2 text-sm font-medium">
                <Info class="h-4 w-4" />
                Ranking Factors
              </div>
              <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                <TooltipProvider>
                  {#each Object.entries(result.rankingFactors) as [factor, value]}
                    <Tooltip>
                      <TooltipTrigger>
                        <div class="text-center p-2 bg-muted rounded">
                          <div class="text-xs text-muted-foreground capitalize">
                            {factor.replace(/([A-Z])/g, " $1").trim()}
                          </div>
                          <div
                            class={`font-semibold ${getRankingColor(factor, value)}`}
                          >
                            {formatScore(value)}
                          </div>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p class="max-w-xs">
                          {#if factor === "vectorSimilarity"}
                            Semantic similarity between query and content
                          {:else if factor === "documentRecency"}
                            How recently the document was created/updated
                          {:else if factor === "userPreference"}
                            Based on your search history and interactions
                          {:else if factor === "contextRelevance"}
                            Quality of metadata and AI analysis
                          {:else if factor === "entityOverlap"}
                            Shared entities between query and document
                          {/if}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  {/each}
                </TooltipProvider>
              </div>
            </div>

            <!-- Explanation -->
            {#if result.explanation}
              <div class="bg-blue-50 dark:bg-blue-950 p-3 rounded-lg">
                <p class="text-sm text-blue-800 dark:text-blue-200">
                  <strong>Why this result:</strong>
                  {result.explanation}
                </p>
              </div>
            {/if}

            <!-- Metadata -->
            {#if result.metadata && Object.keys(result.metadata).length > 0}
              <div class="flex flex-wrap gap-2">
                {#each Object.entries(result.metadata).slice(0, 5) as [key, value]}
                  {#if typeof value === "string" || typeof value === "number"}
                    <Badge variant="outline" class="text-xs">
                      {key}: {value}
                    </Badge>
                  {/if}
                {/each}
              </div>
            {/if}
          </CardContent>
        </Card>
      {/each}
    </div>
  {:else if searching}
    <Card>
      <CardContent class="py-12">
        <div class="flex flex-col items-center justify-center space-y-4">
          <div
            class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"
          ></div>
          <p class="text-muted-foreground">Searching through vectors...</p>
        </div>
      </CardContent>
    </Card>
  {:else}
    <Card>
      <CardContent class="py-12 text-center">
        <Search class="mx-auto h-12 w-12 text-muted-foreground mb-4" />
        <p class="text-muted-foreground">
          Enter a search query to find semantically similar documents
        </p>
      </CardContent>
    </Card>
  {/if}
</div>
