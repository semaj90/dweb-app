<script lang="ts">
  import { Badge, Button, Input } from "$lib/components/ui";
  import Fuse from "fuse.js";
  import { createEventDispatcher, onMount } from "svelte";

  const dispatch = createEventDispatcher();

  export let selectedNode: unknown = null;

  // AI processing state
  let isProcessing = false;
  let processingStatus = "";

  // Search functionality
  let searchQuery = "";
  let searchResults: unknown[] = [];
  let fuse: Fuse<any> | null = null;

  // All evidence items for search (would be populated from store)
  let allEvidence: unknown[] = [];

  // AI suggestions and insights
  let aiInsights: {
    connections: unknown[];
    similarEvidence: unknown[];
    timeline: unknown[];
    suggestedActions: unknown[];
  } = {
    connections: [],
    similarEvidence: [],
    timeline: [],
    suggestedActions: [],
  };

  onMount(() => {
    // Initialize Fuse.js for fuzzy search
    setupSearch();
  });

  function setupSearch() {
    // TODO: Load all evidence from store/API
    // For now, using mock data
    allEvidence = [];

    if (allEvidence.length > 0) {
      fuse = new Fuse(allEvidence, {
        keys: ["name", "tags", "title", "people", "locations", "summary"],
        threshold: 0.4,
        includeScore: true,
      });
    }
  }

  async function reprocessWithAI() {
    if (!selectedNode || isProcessing) return;

    isProcessing = true;
    processingStatus = "Analyzing with AI...";

    try {
      const response = await fetch("/api/ai/tag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: selectedNode.content,
          fileName: selectedNode.name,
          fileType: selectedNode.type,
        }),
      });

      if (response.ok) {
        const newTags = await response.json();
        selectedNode.aiTags = newTags;
        dispatch("tagsUpdate", newTags);
        processingStatus = "Analysis complete!";

        // Generate insights
        await generateInsights();
      } else {
        processingStatus = "Analysis failed";
      }
    } catch (error) {
      console.error("AI reprocessing failed:", error);
      processingStatus = "Analysis failed";
    } finally {
      isProcessing = false;
      setTimeout(() => (processingStatus = ""), 3000);
    }
  }

  async function generateInsights() {
    if (!selectedNode?.aiTags) return;

    try {
      // Mock insights generation - in real app, this would call an insights API
      aiInsights = {
        connections: generateConnections(),
        similarEvidence: findSimilarEvidence(),
        timeline: generateTimeline(),
        suggestedActions: generateSuggestedActions(),
      };
    } catch (error) {
      console.error("Failed to generate insights:", error);
    }
  }

  function generateConnections(): unknown[] {
    if (!selectedNode?.aiTags) return [];

    const connections: unknown[] = [];
    const { people, locations, organizations } = selectedNode.aiTags;

    // Mock connections based on shared entities
    if (people?.length > 0) {
      connections.push({
        type: "person",
        entity: people[0],
        strength: "high",
        description: `Connected through person: ${people[0]}`,
      });
    }

    if (locations?.length > 0) {
      connections.push({
        type: "location",
        entity: locations[0],
        strength: "medium",
        description: `Connected through location: ${locations[0]}`,
      });
    }

    return connections;
  }

  function findSimilarEvidence(): unknown[] {
    // In real app, this would use vector similarity search
    return [
      {
        name: "Similar Evidence 1",
        similarity: 0.85,
        reason: "Similar tags and content",
      },
      {
        name: "Similar Evidence 2",
        similarity: 0.72,
        reason: "Shared people and location",
      },
    ];
  }

  function generateTimeline(): unknown[] {
    if (!selectedNode?.aiTags?.dates) return [];

    return selectedNode.aiTags.dates.map((date: string) => ({
      date,
      event: `Evidence related to ${date}`,
      type: "evidence",
    }));
  }

  function generateSuggestedActions(): unknown[] {
    const actions: unknown[] = [];

    if (selectedNode?.aiTags?.legalRelevance === "high") {
      actions.push({
        action: "Prioritize Review",
        reason: "High legal relevance detected",
        priority: "high",
      });
    }

    if (selectedNode?.aiTags?.people?.length > 0) {
      actions.push({
        action: "Cross-reference People",
        reason: "Multiple people identified",
        priority: "medium",
      });
    }

    return actions;
  }

  function performSearch() {
    if (!fuse || !searchQuery.trim()) {
      searchResults = [];
      return;
    }

    const results = fuse.search(searchQuery);
    searchResults = results.map((result) => ({
      ...result.item,
      score: result.score,
    }));
  }

  function clearSearch() {
    searchQuery = "";
    searchResults = [];
  }

  // Reactive search
  $: if (searchQuery) {
    performSearch();
  } else {
    searchResults = [];
  }

  // Auto-generate insights when node changes
  $: if (selectedNode?.aiTags) {
    generateInsights();
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  <h2 class="mx-auto px-4 max-w-7xl">AI Assistant</h2>

  <!-- Search Section -->
  <div class="mx-auto px-4 max-w-7xl">
    <h3 class="mx-auto px-4 max-w-7xl">Evidence Search</h3>
    <div class="mx-auto px-4 max-w-7xl">
      <Input
        bind:value={searchQuery}
        placeholder="Search evidence, tags, people..."
        class="mx-auto px-4 max-w-7xl"
      />
      {#if searchQuery}
        <Button onclick={clearSearch} variant="outline" size="sm">Clear</Button
        >
      {/if}
    </div>

    <!-- Search Results -->
    {#if searchResults.length > 0}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          {searchResults.length} results found
        </div>
        {#each searchResults.slice(0, 5) as result}
          <div
            class="mx-auto px-4 max-w-7xl"
          >
            <div class="mx-auto px-4 max-w-7xl">{result.name}</div>
            <div class="mx-auto px-4 max-w-7xl">
              Score: {(1 - result.score).toFixed(2)}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  {#if selectedNode}
    <!-- AI Analysis Status -->
    {#if processingStatus}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">{processingStatus}</div>
      </div>
    {/if}

    <!-- AI Tags Section -->
    {#if selectedNode.aiTags}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <h3 class="mx-auto px-4 max-w-7xl">AI Analysis</h3>
          <Button
            onclick={reprocessWithAI}
            disabled={isProcessing}
            variant="outline"
            size="sm"
          >
            {isProcessing ? "Processing..." : "Re-analyze"}
          </Button>
        </div>

        <!-- Summary -->
        {#if selectedNode.aiTags.summary}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">Summary</div>
            <div class="mx-auto px-4 max-w-7xl">
              {selectedNode.aiTags.summary}
            </div>
          </div>
        {/if}

        <!-- Auto Tags -->
        {#if selectedNode.aiTags.tags?.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">Auto Tags</div>
            <div class="mx-auto px-4 max-w-7xl">
              {#each selectedNode.aiTags.tags as tag}
                <Badge class="mx-auto px-4 max-w-7xl">{tag}</Badge>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Key Facts -->
        {#if selectedNode.aiTags.keyFacts?.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">Key Facts</div>
            <ul class="mx-auto px-4 max-w-7xl">
              {#each selectedNode.aiTags.keyFacts as fact}
                <li class="mx-auto px-4 max-w-7xl">
                  <span class="mx-auto px-4 max-w-7xl">•</span>
                  <span>{fact}</span>
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
    {:else}
      <!-- No AI analysis yet -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">🤖</div>
        <div class="mx-auto px-4 max-w-7xl">No AI analysis available</div>
        <Button onclick={reprocessWithAI} disabled={isProcessing}>
          {isProcessing ? "Processing..." : "Analyze with AI"}
        </Button>
      </div>
    {/if}

    <!-- AI Insights -->
    {#if aiInsights.connections.length > 0 || aiInsights.similarEvidence.length > 0}
      <div class="mx-auto px-4 max-w-7xl">
        <h3 class="mx-auto px-4 max-w-7xl">AI Insights</h3>

        <!-- Connections -->
        {#if aiInsights.connections.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">Connections</div>
            {#each aiInsights.connections as connection}
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">{connection.entity}</div>
                <div class="mx-auto px-4 max-w-7xl">
                  {connection.description}
                </div>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Similar Evidence -->
        {#if aiInsights.similarEvidence.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">Similar Evidence</div>
            {#each aiInsights.similarEvidence as similar}
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">{similar.name}</div>
                <div class="mx-auto px-4 max-w-7xl">{similar.reason}</div>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Suggested Actions -->
        {#if aiInsights.suggestedActions.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">Suggested Actions</div>
            {#each aiInsights.suggestedActions as action}
              <div
                class="mx-auto px-4 max-w-7xl"
              >
                <div>
                  <div class="mx-auto px-4 max-w-7xl">{action.action}</div>
                  <div class="mx-auto px-4 max-w-7xl">{action.reason}</div>
                </div>
                <Badge
                  class={action.priority === "high"
                    ? "bg-red-100 text-red-800"
                    : "bg-yellow-100 text-yellow-800"}
                >
                  {action.priority}
                </Badge>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  {:else}
    <!-- No selection state -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">🤖</div>
      <div class="mx-auto px-4 max-w-7xl">AI Assistant Ready</div>
      <div class="mx-auto px-4 max-w-7xl">
        Select evidence to get AI insights and analysis
      </div>
    </div>
  {/if}
</div>

<style>
  .ai-assistant-panel {
    min-height: 100%;
  }
</style>
