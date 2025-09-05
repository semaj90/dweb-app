<!-- RAG System Demo Page -->
<script lang="ts">
  import AskAI from "$lib/components/ai/AskAI.svelte";
  import {
    AlertTriangle,
    Brain,
    CheckCircle,
    Database,
    Search,
    Zap,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  interface SystemStatus {
    database: boolean
    qdrant: boolean
    embeddings: boolean
    vectorSearch: boolean
}
  let systemStatus: SystemStatus = {
    database: false,
    qdrant: false,
    embeddings: false,
    vectorSearch: false,
  };

  let isLoadingStatus = true;
  let testQuery = "";
  interface TestResults {
    error?: string;
    data?: unknown;
    results?: Array<{
      title: string
      content: string
      score: number
      source: string
      type: string
    }>;
    executionTime?: number;
    source?: string;
}
  let testResults: TestResults | null = null;
  let isTestingSearch = false;

  // Demo queries
  const demoQueries = [
    "What are the most common types of evidence in fraud cases?",
    "Explain the legal requirements for search warrants",
    "How should digital evidence be preserved?",
    "What are the key elements of a criminal investigation?",
    "Summarize the chain of custody procedures",
  ];

  onMount(async () => {
    await checkSystemStatus();
  });

  async function checkSystemStatus() {
    isLoadingStatus = true;

    try {
      // Check database connection
      const dbResponse = await fetch("/api/search/cases?limit=1");
      systemStatus.database = dbResponse.ok;

      // Check Qdrant
      const qdrantResponse = await fetch("/api/qdrant");
      systemStatus.qdrant = qdrantResponse.ok;

      // Check embeddings
      const embeddingResponse = await fetch("/api/embeddings");
      systemStatus.embeddings = embeddingResponse.ok;

      // Check vector search
      const vectorResponse = await fetch("/api/search/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: "test",
          options: { limit: 1 },
        }),
      });
      systemStatus.vectorSearch = vectorResponse.ok;
    } catch (error) {
      console.error("Status check failed:", error);
    } finally {
      isLoadingStatus = false;
}}
  async function testVectorSearch() {
    if (!testQuery.trim()) return;

    isTestingSearch = true;
    testResults = null;

    try {
      const response = await fetch("/api/search/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: testQuery,
          options: {
            limit: 5,
            threshold: 0.5,
            searchType: "hybrid",
          },
        }),
      });

      if (response.ok) {
        const data = await response.json();
        testResults = data.data;
      } else {
        const error = await response.json();
        testResults = { error: error.error };
}
    } catch (error) {
      testResults = { error: "Network error" };
    } finally {
      isTestingSearch = false;
}}
  function handleAIResponse(event: CustomEvent) {
    console.log("AI Response:", event.detail);
}
  function handleReferenceClick(event: CustomEvent) {
    console.log("Reference clicked:", event.detail);
}
  function getStatusIcon(status: boolean) {
    return status ? CheckCircle: AlertTriangle
}
  function getStatusColor(status: boolean) {
    return status ? "text-green-600" : "text-red-600";
}
</script>

<svelte:head>
  <title>RAG System Demo - AI Legal Assistant</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <!-- Header -->
    <div class="space-y-4">
      <h1 class="space-y-4">RAG System Demo</h1>
      <p class="space-y-4">
        Production-ready Retrieval-Augmented Generation system with PostgreSQL +
        pgvector, Qdrant vector database, and intelligent legal assistant
        capabilities.
      </p>
    </div>

    <!-- System Status -->
    <div class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-4">
          <h2 class="space-y-4">System Status</h2>
          <button
            onclick={() => checkSystemStatus()}
            class="space-y-4"
            disabled={isLoadingStatus}
          >
            {isLoadingStatus ? "Checking..." : "Refresh"}
          </button>
        </div>

        <div class="space-y-4">
          <div class="space-y-4">
            <Database class="space-y-4" />
            <div>
              <p class="space-y-4">Database</p>
              <p class="space-y-4">
                {systemStatus.database ? "Connected" : "Offline"}
              </p>
            </div>
          </div>

          <div class="space-y-4">
            <Zap class="space-y-4" />
            <div>
              <p class="space-y-4">Qdrant</p>
              <p class="space-y-4">
                {systemStatus.qdrant ? "Ready" : "Unavailable"}
              </p>
            </div>
          </div>

          <div class="space-y-4">
            <Brain class="space-y-4" />
            <div>
              <p class="space-y-4">Embeddings</p>
              <p class="space-y-4">
                {systemStatus.embeddings ? "Active" : "Disabled"}
              </p>
            </div>
          </div>

          <div class="space-y-4">
            <Search
              class="space-y-4"
            />
            <div>
              <p class="space-y-4">Vector Search</p>
              <p class="space-y-4">
                {systemStatus.vectorSearch ? "Operational" : "Error"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Demo Area -->
    <div class="space-y-4">
      <!-- AI Assistant -->
      <div class="space-y-4">
        <div class="space-y-4">
          <div class="space-y-4">
            <h3 class="space-y-4">
              AI Legal Assistant
            </h3>
            <p class="space-y-4">
              Ask questions about legal procedures, cases, and evidence. The AI
              uses vector search to find relevant information.
            </p>
          </div>

          <div class="space-y-4">
            <AskAI
              caseId=""
              evidenceIds={[]}
              placeholder="Ask about legal procedures, cases, or evidence..."
              showReferences={true}
              enableVoiceInput={true}
              maxHeight="500px"
              onresponse={handleAIResponse}
              on:referenceClicked={handleReferenceClick}
            />
          </div>
        </div>

        <!-- Demo Queries -->
        <div class="space-y-4">
          <h4 class="space-y-4">
            Try these sample questions:
          </h4>
          <div class="space-y-4">
            {#each demoQueries as query}
              <button
                class="space-y-4"
                onclick={() => (testQuery = query)}
              >
                "{query}"
              </button>
            {/each}
          </div>
        </div>
      </div>

      <!-- Vector Search Test -->
      <div class="space-y-4">
        <div class="space-y-4">
          <div class="space-y-4">
            <h3 class="space-y-4">
              Vector Search Test
            </h3>
            <p class="space-y-4">
              Test the vector similarity search directly to see raw results.
            </p>
          </div>

          <div class="space-y-4">
            <div class="space-y-4">
              <input
                bind:value={testQuery}
                placeholder="Enter search query..."
                class="space-y-4"
              />
              <button
                onclick={() => testVectorSearch()}
                disabled={!testQuery.trim() || isTestingSearch}
                class="space-y-4"
              >
                {isTestingSearch ? "Searching..." : "Search"}
              </button>
            </div>

            {#if testResults}
              <div class="space-y-4">
                {#if testResults.error}
                  <div class="space-y-4">
                    <strong>Error:</strong>
                    {testResults.error}
                  </div>
                {:else}
                  <div class="space-y-4">
                    <div class="space-y-4">
                      Found {testResults.results?.length || 0} results in {testResults.executionTime ||
                        0}ms (Source: {testResults.source || "unknown"})
                    </div>

                    {#if testResults.results && testResults.results.length > 0}
                      {#each testResults.results as result}
                        <div
                          class="space-y-4"
                        >
                          <div class="space-y-4">
                            <h5 class="space-y-4">
                              {result.title}
                            </h5>
                            <span class="space-y-4">
                              {Math.round(result.score * 100)}% match
                            </span>
                          </div>
                          <p class="space-y-4">
                            {result.content.substring(0, 200)}...
                          </p>
                          <div class="space-y-4">
                            <span>Type: {result.type}</span>
                            <span>Source: {result.source}</span>
                          </div>
                        </div>
                      {/each}
                    {:else}
                      <p class="space-y-4">No results found.</p>
                    {/if}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </div>

        <!-- System Information -->
        <div class="space-y-4">
          <h4 class="space-y-4">System Information</h4>
          <div class="space-y-4">
            <div class="space-y-4">
              <span>Vector Database:</span>
              <span>PostgreSQL + pgvector / Qdrant</span>
            </div>
            <div class="space-y-4">
              <span>Embedding Model:</span>
              <span>OpenAI text-embedding-ada-002</span>
            </div>
            <div class="space-y-4">
              <span>LLM:</span>
              <span>GPT-3.5-turbo / Ollama (Local)</span>
            </div>
            <div class="space-y-4">
              <span>Search Types:</span>
              <span>Similarity, Hybrid, Semantic</span>
            </div>
            <div class="space-y-4">
              <span>Caching:</span>
              <span>Redis + IndexedDB</span>
            </div>
          </div>
        </div>

        <!-- Quick Actions -->
        <div class="space-y-4">
          <h4 class="space-y-4">Quick Actions</h4>
          <div class="space-y-4">
            <a
              href="/api/embeddings"
              target="_blank"
              class="space-y-4"
            >
              → View Embeddings API Status
            </a>
            <a
              href="/api/qdrant"
              target="_blank"
              class="space-y-4"
            >
              → View Qdrant Collection Status
            </a>
            <button
              class="space-y-4"
              onclick={() => window.open("/cases", "_blank")}
            >
              → Browse Case Database
            </button>
            <button
              class="space-y-4"
              onclick={() => window.open("/evidence", "_blank")}
            >
              → Browse Evidence Collection
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Setup Instructions -->
    <div class="space-y-4">
      <h3 class="space-y-4">
        Setup Instructions
      </h3>
      <div class="space-y-4">
        <p>
          <strong>1. Start the services:</strong> <code>npm run db:start</code>
        </p>
        <p>
          <strong>2. Initialize vector search:</strong>
          <code>npm run vector:init</code>
        </p>
        <p>
          <strong>3. Sync existing data:</strong>
          <code>npm run vector:sync</code>
        </p>
        <p>
          <strong>4. Configure environment:</strong> Set OpenAI API key in
          <code>.env</code>
        </p>
      </div>
    </div>
  </div>
</div>
