<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { onMount } from "svelte";

  let status = {
    docker: { status: "checking", message: "" },
    postgres: { status: "checking", message: "" },
    ollama: { status: "checking", message: "" },
    vector: { status: "checking", message: "" },
  };

  let testResults: {
    embedding: any;
    search: any;
    analysis: any;
  } = {
    embedding: null,
    search: null,
    analysis: null,
  };

  let searchQuery = "legal contract breach of agreement";
  let documentText = `This is a sample legal document regarding a breach of contract. 
The defendant failed to deliver goods as specified in the agreement dated January 1, 2024. 
The contract clearly states that delivery must be completed within 30 days of order placement.`;

  let isLoading = false;

  async function checkServices() {
    // Check Vector/Ollama service
    try {
      const response = await fetch("/api/vector");
      const data = await response.json();

      if (data.success) {
        status.ollama = {
          status: "healthy",
          message: `Connected! Models: ${data.ollama.models.join(", ")}`,
        };
        status.vector = {
          status: "healthy",
          message: `Embedding model: ${data.embedding.model} (${data.embedding.dimension}D)`,
        };
      } else {
        status.ollama = { status: "error", message: data.error };
        status.vector = { status: "error", message: "Not configured" };
}
    } catch (error) {
      status.ollama = { status: "error", message: "Cannot connect to Ollama" };
      status.vector = { status: "error", message: "Service unavailable" };
}
}
  async function testEmbedding() {
    isLoading = true;
    try {
      const response = await fetch("/api/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "test" }),
      });

      const data = await response.json();
      testResults.embedding = data;
    } catch (error) {
      testResults.embedding = {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
}
    isLoading = false;
}
  async function testSearch() {
    isLoading = true;
    try {
      // First store the document
      const storeResponse = await fetch("/api/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "store",
          documentId: "test-doc-1",
          documentType: "case",
          text: documentText,
          metadata: { title: "Test Document" },
        }),
      });

      const storeData = await storeResponse.json();

      // Then search for it
      const searchResponse = await fetch("/api/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "search",
          query: searchQuery,
          limit: 5,
        }),
      });

      const searchData = await searchResponse.json();
      testResults.search = { store: storeData, search: searchData };
    } catch (error) {
      testResults.search = {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
}
    isLoading = false;
}
  async function testAnalysis() {
    isLoading = true;
    try {
      const response = await fetch("/api/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "analyze",
          text: documentText,
          analysisType: "legal_issues",
        }),
      });

      const data = await response.json();
      testResults.analysis = data;
    } catch (error) {
      testResults.analysis = {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
}
    isLoading = false;
}
  onMount(() => {
    checkServices();
  });
</script>

<div class="container mx-auto px-4">
  <h1 class="container mx-auto px-4">AI Integration Test Dashboard</h1>

  <!-- Service Status -->
  <div class="container mx-auto px-4">
    <h2 class="container mx-auto px-4">Service Status</h2>
    <div class="container mx-auto px-4">
      {#each Object.entries(status) as [service, info]}
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">
            <span class="container mx-auto px-4">{service}</span>
            <span
              class={`text-sm ${
                info.status === "healthy"
                  ? "text-green-600"
                  : info.status === "error"
                    ? "text-red-600"
                    : "text-yellow-600"
              }`}
            >
              {info.status}
            </span>
          </div>
          {#if info.message}
            <p class="container mx-auto px-4">{info.message}</p>
          {/if}
        </div>
      {/each}
    </div>
  </div>

  <!-- Test Controls -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <h3 class="container mx-auto px-4">Test Vector Embeddings</h3>
      <Button on:click={() => testEmbedding()} disabled={isLoading}>
        Test Embedding Generation
      </Button>
      {#if testResults.embedding}
        <pre class="container mx-auto px-4">
{JSON.stringify(testResults.embedding, null, 2)}
                </pre>
      {/if}
    </div>

    <div class="container mx-auto px-4">
      <h3 class="container mx-auto px-4">Test Semantic Search</h3>
      <div class="container mx-auto px-4">
        <div>
          <label class="container mx-auto px-4">Document Text</label>
          <textarea
            bind:value={documentText}
            class="container mx-auto px-4"
            rows="4"
          ></textarea>
        </div>
        <div>
          <label class="container mx-auto px-4">Search Query</label>
          <input
            type="text"
            bind:value={searchQuery}
            class="container mx-auto px-4"
          />
        </div>
        <Button on:click={() => testSearch()} disabled={isLoading}>
          Test Store & Search
        </Button>
      </div>
      {#if testResults.search}
        <pre class="container mx-auto px-4">
{JSON.stringify(testResults.search, null, 2)}
                </pre>
      {/if}
    </div>

    <div class="container mx-auto px-4">
      <h3 class="container mx-auto px-4">Test AI Analysis</h3>
      <Button on:click={() => testAnalysis()} disabled={isLoading}>
        Analyze Document for Legal Issues
      </Button>
      {#if testResults.analysis}
        <div class="container mx-auto px-4">
          {#if testResults.analysis.success}
            <p class="container mx-auto px-4">{testResults.analysis.analysis}</p>
          {:else}
            <p class="container mx-auto px-4">Error: {testResults.analysis.error}</p>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  <div class="container mx-auto px-4">
    <h3 class="container mx-auto px-4">Quick Start Commands</h3>
    <code class="container mx-auto px-4">
      # Start all services with Ollama<br />
      .\setup-complete-with-ollama.ps1<br /><br />

      # Check Docker services<br />
      docker ps<br /><br />

      # View Ollama models<br />
      docker exec prosecutor_ollama ollama list
    </code>
  </div>
</div>

<style>
  /* @unocss-include */
  :global(.container) {
    font-family:
      -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
</style>
