<!-- Enhanced RAG Demo with Local LLM Support -->
<script lang="ts">
  import AskAI from "$lib/components/ai/AskAI.svelte";
  import { aiService } from "$lib/services/ai-service";
  import { tauriLLM, type LocalModel } from "$lib/services/tauri-llm";
  import {
    AlertTriangle,
    Brain,
    CheckCircle,
    Cloud,
    Cpu,
    Database,
    Search,
    Shield,
    Zap,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  interface SystemStatus {
    database: boolean
    qdrant: boolean
    embeddings: boolean
    vectorSearch: boolean
    tauriLLM: boolean
    localModels: LocalModel[];
}
  let systemStatus: SystemStatus = {
    database: false,
    qdrant: false,
    embeddings: false,
    vectorSearch: false,
    tauriLLM: false,
    localModels: [],
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
  let selectedProvider: "auto" | "local" | "cloud" = "auto";
  let legalAnalysisText = "";

  interface AnalysisResults {
    error?: string;
    summary?: string;
    riskAssessment?: string;
    classification?: {
      category: string
      confidence: number
    };
    keyEntities: Array<{
      text: string
      type: string
      confidence: number
    }>;
}
  let analysisResults: AnalysisResults | null = null;
  let isAnalyzing = false;

  // Demo queries optimized for legal domain
  const legalDemoQueries = [
    "What are the key elements required to establish a breach of contract?",
    "Explain the difference between criminal and civil liability",
    "What constitutes admissible evidence in federal court?",
    "How does attorney-client privilege protect confidential communications?",
    "What are the requirements for a valid search warrant?",
    "Explain the burden of proof in criminal vs civil cases",
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

      // Check Tauri LLM capabilities
      await tauriLLM.initialize();
      systemStatus.tauriLLM = tauriLLM.isAvailable();
      systemStatus.localModels = tauriLLM.getAvailableModels();
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
            provider: selectedProvider,
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
  async function analyzeLegalDocument() {
    if (!legalAnalysisText.trim()) return;

    isAnalyzing = true;
    analysisResults = null;

    try {
      // Use enhanced AI service for legal analysis
      await aiService.initialize();

      if (systemStatus.tauriLLM) {
        // Use local legal analysis
        const result = await aiService.analyzeLegalDocument(legalAnalysisText);
        analysisResults = {
          ...result,
          keyEntities: Array.isArray(result.keyEntities) && typeof result.keyEntities[0] === 'string' 
            ? result.keyEntities.map((entity: string) => ({ text: entity, type: 'entity', confidence: 1.0 }))
            : result.keyEntities || []
        };
      } else {
        // Fallback to cloud analysis
        const response = await fetch("/api/ai/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: `Analyze this legal document and provide classification, key points, and risk assessment: ${legalAnalysisText}`,
            options: {
              provider: "auto",
              legalContext: true,
              maxTokens: 800,
            },
          }),
        });

        if (response.ok) {
          const data = await response.json();
          analysisResults = {
            classification: { category: "document", confidence: 0.8 },
            summary: data.data.answer,
            keyEntities: ["Legal Document"].map(entity => ({ text: entity, type: "document", confidence: 1.0 })),
            riskAssessment: "Analysis completed using cloud AI",
          };
}}
    } catch (error) {
      analysisResults = {
        error: error instanceof Error ? error.message : "Unknown error",
        keyEntities: [], // Required property
        classification: { category: "error", confidence: 0 },
        summary: "Analysis failed",
        riskAssessment: "Unable to assess",
        similarity: 0
      };
    } finally {
      isAnalyzing = false;
}}
  async function loadLocalModel(modelId: string) {
    try {
      await tauriLLM.loadModel(modelId);
      await checkSystemStatus(); // Refresh status
    } catch (error) {
      console.error("Failed to load model:", error);
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
  function getProviderIcon(provider: string) {
    switch (provider) {
      case "local":
        return Cpu;
      case "cloud":
        return Cloud;
      default:
        return Brain;
}}
</script>

<svelte:head>
  <title>Enhanced RAG Demo - Local AI + Cloud Integration</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <!-- Header -->
    <div class="space-y-4">
      <h1 class="space-y-4">
        Enhanced RAG System Demo
      </h1>
      <p class="space-y-4">
        Production-ready Retrieval-Augmented Generation with <strong
          >Local Rust LLMs</strong
        >, Legal-BERT models, PostgreSQL + pgvector, Qdrant, and intelligent
        fallback mechanisms.
      </p>
      <div class="space-y-4">
        <span
          class="space-y-4"
        >
          <Shield class="space-y-4" />
          Privacy-First Local AI
        </span>
        <span
          class="space-y-4"
        >
          <Cpu class="space-y-4" />
          Rust + Legal-BERT
        </span>
        <span
          class="space-y-4"
        >
          <Cloud class="space-y-4" />
          Cloud Fallback
        </span>
      </div>
    </div>

    <!-- System Status -->
    <div class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-4">
          <h2 class="space-y-4">
            Enhanced System Status
          </h2>
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

          <div
            class="space-y-4"
          >
            <Cpu class="space-y-4" />
            <div>
              <p class="space-y-4">Local LLM</p>
              <p class="space-y-4">
                {systemStatus.tauriLLM ? "Available" : "Not Available"}
              </p>
            </div>
          </div>
        </div>

        <!-- Local Models Status -->
        {#if systemStatus.tauriLLM && systemStatus.localModels.length > 0}
          <div class="space-y-4">
            <h3 class="space-y-4">
              Local AI Models
            </h3>
            <div class="space-y-4">
              {#each systemStatus.localModels as model}
                <div class="space-y-4">
                  <div class="space-y-4">
                    <h4 class="space-y-4">{model.name}</h4>
                    <span
                      class="space-y-4"
                    >
                      {model.isLoaded ? "Loaded" : "Available"}
                    </span>
                  </div>
                  <div class="space-y-4">
                    <p><span class="space-y-4">Type:</span> {model.type}</p>
                    <p>
                      <span class="space-y-4">Domain:</span>
                      {model.domain}
                    </p>
                    <p>
                      <span class="space-y-4">Architecture:</span>
                      {model.architecture}
                    </p>
                    {#if model.dimensions}
                      <p>
                        <span class="space-y-4">Dimensions:</span>
                        {model.dimensions}
                      </p>
                    {/if}
                  </div>
                  {#if !model.isLoaded}
                    <button
                      onclick={() => loadLocalModel(model.id)}
                      class="space-y-4"
                    >
                      Load Model
                    </button>
                  {/if}
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    </div>

    <!-- Main Demo Area -->
    <div class="space-y-4">
      <!-- AI Assistant -->
      <div class="space-y-4">
        <div class="space-y-4">
          <div class="space-y-4">
            <h3 class="space-y-4">
              Enhanced AI Legal Assistant
            </h3>
            <p class="space-y-4">
              Ask questions using local Legal-BERT models or cloud LLMs with
              automatic fallback.
            </p>
          </div>

          <div class="space-y-4">
            <!-- Provider Selection -->
            <div class="space-y-4">
              <label
                for="ai-provider-select"
                class="space-y-4"
                >AI Provider</label
              >
              <div class="space-y-4">
                <button
                  class="space-y-4"
                  onclick={() => (selectedProvider = "auto")}
                >
                  <Brain class="space-y-4" />
                  Auto
                </button>
                <button
                  class="space-y-4"
                  onclick={() => (selectedProvider = "local")}
                  disabled={!systemStatus.tauriLLM}
                >
                  <Cpu class="space-y-4" />
                  Local Only
                </button>
                <button
                  class="space-y-4"
                  onclick={() => (selectedProvider = "cloud")}
                >
                  <Cloud class="space-y-4" />
                  Cloud Only
                </button>
              </div>
            </div>

            <AskAI
              caseId=""
              evidenceIds={[]}
              placeholder="Ask about legal procedures, cases, or evidence..."
              showReferences={true}
              enableVoiceInput={true}
              maxHeight="400px"
              onresponse={handleAIResponse}
              on:referenceClicked={handleReferenceClick}
            />
          </div>
        </div>

        <!-- Legal Demo Queries -->
        <div class="space-y-4">
          <h4 class="space-y-4">
            Legal Domain Sample Questions:
          </h4>
          <div class="space-y-4">
            {#each legalDemoQueries as query}
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

      <!-- Legal Document Analysis & Vector Search -->
      <div class="space-y-4">
        <!-- Document Analysis -->
        <div class="space-y-4">
          <div class="space-y-4">
            <h3 class="space-y-4">
              Legal Document Analysis
            </h3>
            <p class="space-y-4">
              Analyze legal documents using local Legal-BERT models for
              classification and risk assessment.
            </p>
          </div>

          <div class="space-y-4">
            <textarea
              bind:value={legalAnalysisText}
              placeholder="Paste a legal document excerpt for analysis..."
              class="space-y-4"
            ></textarea>

            <button
              onclick={() => analyzeLegalDocument()}
              disabled={!legalAnalysisText.trim() || isAnalyzing}
              class="space-y-4"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze Document"}
            </button>

            {#if analysisResults}
              <div class="space-y-4">
                {#if analysisResults.error}
                  <div class="space-y-4">
                    <strong>Error:</strong>
                    {analysisResults.error}
                  </div>
                {:else}
                  <div class="space-y-4">
                    {#if analysisResults.classification}
                      <div>
                        <h5 class="space-y-4">
                          Classification:
                        </h5>
                        <p class="space-y-4">
                          {analysisResults.classification.category}
                          ({Math.round(
                            analysisResults.classification.confidence * 100
                          )}% confidence)
                        </p>
                      </div>
                    {/if}

                    {#if analysisResults && analysisResults.summary}
                      <div>
                        <h5 class="space-y-4">Summary:</h5>
                        <p class="space-y-4">
                          {analysisResults.summary}
                        </p>
                      </div>
                    {/if}

                    {#if analysisResults.keyEntities}
                      <div>
                        <h5 class="space-y-4">Key Entities:</h5>
                        <div class="space-y-4">
                          {#each analysisResults.keyEntities.slice(0, 5) as entity}
                            <span
                              class="space-y-4"
                            >
                              {entity}
                            </span>
                          {/each}
                        </div>
                      </div>
                    {/if}

                    {#if analysisResults && analysisResults.riskAssessment}
                      <div>
                        <h5 class="space-y-4">
                          Risk Assessment:
                        </h5>
                        <p class="space-y-4">
                          {analysisResults.riskAssessment}
                        </p>
                      </div>
                    {/if}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </div>

        <!-- Vector Search Test -->
        <div class="space-y-4">
          <div class="space-y-4">
            <h3 class="space-y-4">
              Vector Search Test
            </h3>
            <p class="space-y-4">
              Test vector similarity search with hybrid local/cloud embedding
              generation.
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
                            <div class="space-y-4">
                              <span class="space-y-4">
                                {Math.round(result.score * 100)}% match
                              </span>
                              <svelte:component
                                this={getProviderIcon(result.source)}
                                class="space-y-4"
                              />
                            </div>
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
      </div>
    </div>

    <!-- Architecture Information -->
    <div class="space-y-4">
      <h3 class="space-y-4">
        Enhanced RAG Architecture
      </h3>
      <div class="space-y-4">
        <div>
          <h4 class="space-y-4">Local AI (Rust)</h4>
          <ul class="space-y-4">
            <li>• Legal-BERT embeddings</li>
            <li>• Document classification</li>
            <li>• Privacy-first processing</li>
            <li>• Offline capabilities</li>
          </ul>
        </div>
        <div>
          <h4 class="space-y-4">Vector Databases</h4>
          <ul class="space-y-4">
            <li>• PostgreSQL + pgvector</li>
            <li>• Qdrant for advanced search</li>
            <li>• HNSW indexing</li>
            <li>• Metadata filtering</li>
          </ul>
        </div>
        <div>
          <h4 class="space-y-4">Cloud Fallbacks</h4>
          <ul class="space-y-4">
            <li>• OpenAI embeddings</li>
            <li>• GPT-3.5/4 chat</li>
            <li>• Ollama local LLMs</li>
            <li>• Intelligent routing</li>
          </ul>
        </div>
        <div>
          <h4 class="space-y-4">Performance</h4>
          <ul class="space-y-4">
            <li>• Redis caching</li>
            <li>• IndexedDB storage</li>
            <li>• Batch processing</li>
            <li>• Memory optimization</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Setup Instructions -->
    <div class="space-y-4">
      <h3 class="space-y-4">
        Setup Instructions
      </h3>
      <div class="space-y-4">
        <p><strong>1. Start services:</strong> <code>npm run db:start</code></p>
        <p>
          <strong>2. Initialize vector search:</strong>
          <code>npm run vector:init</code>
        </p>
        <p>
          <strong>3. Set up Tauri (optional):</strong> See
          <code>TAURI_RUST_SETUP.md</code>
        </p>
        <p>
          <strong>4. Configure environment:</strong> Set API keys in
          <code>.env</code>
        </p>
        <p>
          <strong>5. Test local models:</strong> Load legal-BERT models for offline
          AI
        </p>
      </div>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  code {
    font-family: "Courier New", monospace;
    background: #f5f5f5;
    padding: 2px 4px;
    border-radius: 3px;
}
</style>
