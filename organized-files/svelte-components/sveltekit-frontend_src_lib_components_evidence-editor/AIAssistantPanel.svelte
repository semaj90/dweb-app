<script lang="ts">
  interface Props {
    ontagsUpdate?: (event?: any) => void;
  }
  let {
    selectedNode = null
  } = $props();



  import Badge from "$lib/components/ui/Badge.svelte";
import Button from "$lib/components/ui/Button.svelte";
import Input from "$lib/components/ui/Input.svelte";
  import Fuse from "fuse.js";
  
  
  
  // AI processing state
  let isProcessing = false;
  let processingStatus = "";

  // Search functionality
  let searchQuery = "";
  let searchResults: any[] = [];
  let fuse: Fuse<any> | null = null;

  // All evidence items for search (would be populated from store)
  let allEvidence: any[] = [];

  // AI suggestions and insights
  let aiInsights: {
    connections: any[];
    similarEvidence: any[];
    timeline: any[];
    suggestedActions: any[];
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
        ontagsUpdate?.();
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
  function generateConnections(): any[] {
    if (!selectedNode?.aiTags) return [];

    const connections: any[] = [];
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
  function findSimilarEvidence(): any[] {
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
  function generateTimeline(): any[] {
    if (!selectedNode?.aiTags?.dates) return [];

    return selectedNode.aiTags.dates.map((date: string) => ({
      date,
      event: `Evidence related to ${date}`,
      type: "evidence",
    }));
}
  function generateSuggestedActions(): any[] {
    const actions: any[] = [];

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
  $effect(() => { if (searchQuery) {
    performSearch();
  } else {
    searchResults = [];
}
  // Auto-generate insights when node changes
  $effect(() => { if (selectedNode?.aiTags) {
    generateInsights();
}
</script>

<div class="space-y-4">
  <h2 class="space-y-4">AI Assistant</h2>

  <!-- Search Section -->
  <div class="space-y-4">
    <h3 class="space-y-4">Evidence Search</h3>
    <div class="space-y-4">
      <Input
        bind:value={searchQuery}
        placeholder="Search evidence, tags, people..."
        class="space-y-4"
      />
      {#if searchQuery}
        <Button onclick={clearSearch} variant="outline" size="sm">Clear</Button>
      {/if}
    </div>

    <!-- Search Results -->
    {#if searchResults.length > 0}
      <div class="space-y-4">
        <div class="space-y-4">
          {searchResults.length} results found
        </div>
        {#each searchResults.slice(0, 5) as result}
          <div
            class="space-y-4"
          >
            <div class="space-y-4">{result.name}</div>
            <div class="space-y-4">
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
      <div class="space-y-4">
        <div class="space-y-4">{processingStatus}</div>
      </div>
    {/if}

    <!-- AI Tags Section -->
    {#if selectedNode.aiTags}
      <div class="space-y-4">
        <div class="space-y-4">
          <h3 class="space-y-4">AI Analysis</h3>
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
          <div class="space-y-4">
            <div class="space-y-4">Summary</div>
            <div class="space-y-4">
              {selectedNode.aiTags.summary}
            </div>
          </div>
        {/if}

        <!-- Auto Tags -->
        {#if selectedNode.aiTags.tags?.length > 0}
          <div class="space-y-4">
            <div class="space-y-4">Auto Tags</div>
            <div class="space-y-4">
              {#each selectedNode.aiTags.tags as tag}
                <Badge>{tag}</Badge>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Key Facts -->
        {#if selectedNode.aiTags.keyFacts?.length > 0}
          <div class="space-y-4">
            <div class="space-y-4">Key Facts</div>
            <ul class="space-y-4">
              {#each selectedNode.aiTags.keyFacts as fact}
                <li class="space-y-4">
                  <span class="space-y-4">•</span>
                  <span>{fact}</span>
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
    {:else}
      <!-- No AI analysis yet -->
      <div class="space-y-4">
        <div class="space-y-4">🤖</div>
        <div class="space-y-4">No AI analysis available</div>
        <Button onclick={reprocessWithAI} disabled={isProcessing}>
          {isProcessing ? "Processing..." : "Analyze with AI"}
        </Button>
      </div>
    {/if}

    <!-- AI Insights -->
    {#if aiInsights.connections.length > 0 || aiInsights.similarEvidence.length > 0}
      <div class="space-y-4">
        <h3 class="space-y-4">AI Insights</h3>

        <!-- Connections -->
        {#if aiInsights.connections.length > 0}
          <div class="space-y-4">
            <div class="space-y-4">Connections</div>
            {#each aiInsights.connections as connection}
              <div class="space-y-4">
                <div class="space-y-4">{connection.entity}</div>
                <div class="space-y-4">
                  {connection.description}
                </div>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Similar Evidence -->
        {#if aiInsights.similarEvidence.length > 0}
          <div class="space-y-4">
            <div class="space-y-4">Similar Evidence</div>
            {#each aiInsights.similarEvidence as similar}
              <div class="space-y-4">
                <div class="space-y-4">{similar.name}</div>
                <div class="space-y-4">{similar.reason}</div>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Suggested Actions -->
        {#if aiInsights.suggestedActions.length > 0}
          <div class="space-y-4">
            <div class="space-y-4">Suggested Actions</div>
            {#each aiInsights.suggestedActions as action}
              <div
                class="space-y-4"
              >
                <div>
                  <div class="space-y-4">{action.action}</div>
                  <div class="space-y-4">{action.reason}</div>
                </div>
                <Badge>{action.priority}</Badge>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  {:else}
    <!-- No selection state -->
    <div class="space-y-4">
      <div class="space-y-4">🤖</div>
      <div class="space-y-4">AI Assistant Ready</div>
      <div class="space-y-4">
        Select evidence to get AI insights and analysis
      </div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
</style>
