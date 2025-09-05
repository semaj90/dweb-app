<script lang="ts">
  import { Button, Dialog, Select } from "bits-ui";
  import { onMount } from "svelte";

  // Enhanced AI Types
  interface DocumentRequest {
    content: string
    document_type: string
    practice_area?: string;
    jurisdiction: string
    metadata?: Record<string, any>;
    use_gpu?: boolean;
  }

  interface DocumentResponse {
    success: boolean
    message: string
    processed_content?: string;
    summary?: string;
    keywords?: string[];
    legal_entities?: LegalEntity[];
    sentiment?: number;
    confidence?: number;
    processing_time?: string;
    cached_result?: boolean;
  }

  interface LegalEntity {
    name: string
    type: string
    confidence: number
    start_pos: number
    end_pos: number
  }

  interface VectorSearchRequest {
    query: string
    limit?: number;
    filters?: Record<string, any>;
    use_gpu?: boolean;
    model?: string;
  }

  interface VectorSearchResponse {
    results: VectorResult[];
    total: number
    query: string
    took: string
  }

  interface VectorResult {
    id: string
    content: string
    score: number
    metadata: Record<string, any>;
  }

  // Component state
  let serviceStatus = {
    healthy: false,
    loading: true,
    services: {} as Record<string, string>,
    version: "",
    config: {} as Record<string, any>,
  };

  let documentContent = "";
  let selectedDocumentType = "contract";
  let selectedJurisdiction = "US";
  let selectedPracticeArea = "commercial";
  let useGPU = true;
  let processing = false;
  let processResult: DocumentResponse | null = null;

  let searchQuery = "";
  let searchLimit = 10;
  let searching = false;
  let searchResults: VectorSearchResponse | null = null;

  let showProcessDialog = false;
  let showSearchDialog = false;

  // Enhanced configuration
  // Enhanced configuration
  const API_BASE = "/api"; // Use SvelteKit API routes

  const documentTypes = [
    { value: "contract", label: "Contract" },
    { value: "litigation", label: "Litigation" },
    { value: "patent", label: "Patent" },
    { value: "regulatory", label: "Regulatory" },
    { value: "general", label: "General Legal" },
  ];

  const jurisdictions = [
    { value: "US", label: "United States" },
    { value: "CA", label: "Canada" },
    { value: "UK", label: "United Kingdom" },
    { value: "EU", label: "European Union" },
    { value: "INTL", label: "International" },
  ];

  const practiceAreas = [
    { value: "commercial", label: "Commercial Law" },
    { value: "ip", label: "Intellectual Property" },
    { value: "constitutional", label: "Constitutional Law" },
    { value: "criminal", label: "Criminal Law" },
    { value: "corporate", label: "Corporate Law" },
    { value: "employment", label: "Employment Law" },
  ];

  // Enhanced service functions
  async function checkServiceHealth() {
    try {
      serviceStatus.loading = true;
      const response = await fetch(`${API_BASE}/health`);

      if (response.ok) {
        const health = await response.json();
        serviceStatus = {
          healthy: true,
          loading: false,
          services: health.services || {},
          version: health.version || "",
          config: health.config || {},
        };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error("Health check failed:", error);
      serviceStatus = {
        healthy: false,
        loading: false,
        services: {},
        version: "",
        config: {},
      };
    }
  }

  async function processDocument() {
    if (!documentContent.trim()) {
      alert("Please enter document content");
      return;
    }

    try {
      processing = true;
      processResult = null;

      const request: DocumentRequest = {
        content: documentContent,
        document_type: selectedDocumentType,
        practice_area: selectedPracticeArea,
        jurisdiction: selectedJurisdiction,
        use_gpu: useGPU,
        metadata: {
          timestamp: new Date().toISOString(),
          user_id: "demo-user",
          session_id: "demo-session",
        },
      };

      const response = await fetch(`${API_BASE}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      processResult = await response.json();
      showProcessDialog = true;
    } catch (error) {
      console.error("Document processing failed:", error);
      alert(`Processing failed: ${error.message}`);
    } finally {
      processing = false;
    }
  }

  async function performVectorSearch() {
    if (!searchQuery.trim()) {
      alert("Please enter a search query");
      return;
    }

    try {
      searching = true;
      searchResults = null;

      const request: VectorSearchRequest = {
        query: searchQuery,
        limit: searchLimit,
        use_gpu: useGPU,
        model: "gemma3-legal",
        filters: {
          jurisdiction: selectedJurisdiction,
          practice_area: selectedPracticeArea,
        },
      };

      const response = await fetch(`${API_BASE}/vector-search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      searchResults = await response.json();
      showSearchDialog = true;
    } catch (error) {
      console.error("Vector search failed:", error);
      alert(`Search failed: ${error.message}`);
    } finally {
      searching = false;
    }
  }

  function getSentimentColor(sentiment: number): string {
    if (sentiment > 0.7) return "text-green-600";
    if (sentiment > 0.5) return "text-blue-600";
    if (sentiment > 0.3) return "text-yellow-600";
    return "text-red-600";
  }

  function getSentimentLabel(sentiment: number): string {
    if (sentiment > 0.7) return "Positive";
    if (sentiment > 0.5) return "Neutral-Positive";
    if (sentiment > 0.3) return "Neutral-Negative";
    return "Negative";
  }

  onMount(() => {
    checkServiceHealth();
    // Refresh health status every 30 seconds
    const interval = setInterval(checkServiceHealth, 30000);
    return () => clearInterval(interval);
  });
</script>

<!-- Enhanced Legal AI Interface -->
<div class="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
  <div class="max-w-6xl mx-auto">
    <!-- Header -->
    <header class="mb-8">
      <h1 class="text-4xl font-bold text-slate-800 mb-2">
        🏛️ Enhanced Legal AI System
      </h1>
      <p class="text-slate-600 text-lg">
        Gemma3-Legal GGUF • NVIDIA CUDA • Redis-Native • Advanced RAG
      </p>

      <!-- Service Status -->
      <div
        class="mt-4 p-4 rounded-lg border-2 {serviceStatus.healthy
          ? 'bg-green-50 border-green-200'
          : 'bg-red-50 border-red-200'}"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div
              class="w-3 h-3 rounded-full {serviceStatus.healthy
                ? 'bg-green-500'
                : 'bg-red-500'}"
            ></div>
            <span
              class="font-semibold {serviceStatus.healthy
                ? 'text-green-800'
                : 'text-red-800'}"
            >
              {serviceStatus.healthy ? "System Online" : "System Offline"}
            </span>
            {#if serviceStatus.version}
              <span class="text-sm text-slate-600"
                >v{serviceStatus.version}</span
              >
            {/if}
          </div>

          {#if serviceStatus.healthy}
            <div class="flex gap-4 text-sm">
              <span class="text-slate-600">
                Redis: <span
                  class="font-mono {serviceStatus.services.redis === 'connected'
                    ? 'text-green-600'
                    : 'text-red-600'}"
                >
                  {serviceStatus.services.redis || "unknown"}
                </span>
              </span>
              <span class="text-slate-600">
                GPU: <span
                  class="font-mono {serviceStatus.services.gpu === 'true'
                    ? 'text-green-600'
                    : 'text-blue-600'}"
                >
                  {serviceStatus.services.gpu === "true"
                    ? "enabled"
                    : "disabled"}
                </span>
              </span>
            </div>
          {/if}
        </div>
      </div>
    </header>

    <!-- Main Interface -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- Document Processing -->
      <div class="bg-white rounded-xl shadow-lg p-6">
        <h2
          class="text-2xl font-bold text-slate-800 mb-4 flex items-center gap-2"
        >
          📄 Document Processing
        </h2>

        <!-- Configuration -->
        <div class="grid grid-cols-2 gap-4 mb-4">
          <Select.Root bind:selected={selectedDocumentType}>
            <Select.Trigger
              class="h-10 px-3 rounded-lg border border-slate-300 bg-white"
            >
              <Select.Value placeholder="Document Type" />
            </Select.Trigger>
            <Select.Content
              class="bg-white border border-slate-300 rounded-lg shadow-lg z-50"
            >
              {#each documentTypes as type}
                <Select.Item
                  value={type.value}
                  class="px-3 py-2 hover:bg-slate-100 cursor-pointer"
                >
                  {type.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>

          <Select.Root bind:selected={selectedJurisdiction}>
            <Select.Trigger
              class="h-10 px-3 rounded-lg border border-slate-300 bg-white"
            >
              <Select.Value placeholder="Jurisdiction" />
            </Select.Trigger>
            <Select.Content
              class="bg-white border border-slate-300 rounded-lg shadow-lg z-50"
            >
              {#each jurisdictions as jurisdiction}
                <Select.Item
                  value={jurisdiction.value}
                  class="px-3 py-2 hover:bg-slate-100 cursor-pointer"
                >
                  {jurisdiction.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
        </div>

        <div class="grid grid-cols-2 gap-4 mb-4">
          <Select.Root bind:selected={selectedPracticeArea}>
            <Select.Trigger
              class="h-10 px-3 rounded-lg border border-slate-300 bg-white"
            >
              <Select.Value placeholder="Practice Area" />
            </Select.Trigger>
            <Select.Content
              class="bg-white border border-slate-300 rounded-lg shadow-lg z-50"
            >
              {#each practiceAreas as area}
                <Select.Item
                  value={area.value}
                  class="px-3 py-2 hover:bg-slate-100 cursor-pointer"
                >
                  {area.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>

          <label class="flex items-center gap-2 px-3 py-2">
            <input type="checkbox" bind:checked={useGPU} class="rounded" />
            <span class="text-sm font-medium">CUDA GPU Acceleration</span>
          </label>
        </div>

        <!-- Document Input -->
        <textarea
          bind:value={documentContent}
          placeholder="Enter legal document content for analysis..."
          class="w-full h-32 p-3 border border-slate-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        ></textarea>

        <!-- Process Button -->
        <Button.Root
          on:click={processDocument}
          disabled={processing || !serviceStatus.healthy}
          class="w-full mt-4 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white font-semibold rounded-lg transition-colors"
        >
          {#if processing}
            <div class="flex items-center justify-center gap-2">
              <div
                class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"
              ></div>
              Processing...
            </div>
          {:else}
            🚀 Process Document
          {/if}
        </Button.Root>
      </div>

      <!-- Vector Search -->
      <div class="bg-white rounded-xl shadow-lg p-6">
        <h2
          class="text-2xl font-bold text-slate-800 mb-4 flex items-center gap-2"
        >
          🔍 Vector Search
        </h2>

        <!-- Search Configuration -->
        <div class="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label class="block text-sm font-medium text-slate-700 mb-1"
              >Search Limit</label
            >
            <input
              type="number"
              bind:value={searchLimit}
              min="1"
              max="50"
              class="w-full h-10 px-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <div class="flex items-end">
            <label class="flex items-center gap-2 px-3 py-2">
              <input type="checkbox" bind:checked={useGPU} class="rounded" />
              <span class="text-sm font-medium">GPU Acceleration</span>
            </label>
          </div>
        </div>

        <!-- Search Input -->
        <textarea
          bind:value={searchQuery}
          placeholder="Enter legal search query (e.g., 'contract liability terms', 'patent infringement')"
          class="w-full h-32 p-3 border border-slate-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        ></textarea>

        <!-- Search Button -->
        <Button.Root
          on:click={performVectorSearch}
          disabled={searching || !serviceStatus.healthy}
          class="w-full mt-4 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-slate-400 text-white font-semibold rounded-lg transition-colors"
        >
          {#if searching}
            <div class="flex items-center justify-center gap-2">
              <div
                class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"
              ></div>
              Searching...
            </div>
          {:else}
            🔍 Vector Search
          {/if}
        </Button.Root>
      </div>
    </div>
  </div>
</div>

<!-- Process Results Dialog -->
<Dialog.Root bind:open={showProcessDialog}>
  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 bg-black/50 z-40" />
    <Dialog.Content
      class="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white rounded-xl shadow-2xl p-6 max-w-4xl max-h-[80vh] overflow-y-auto z-50"
    >
      {#if processResult}
        <Dialog.Title class="text-2xl font-bold text-slate-800 mb-4">
          📊 Document Analysis Results
        </Dialog.Title>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Summary -->
          <div class="space-y-4">
            <div>
              <h3 class="font-semibold text-slate-700 mb-2">Summary</h3>
              <p class="text-slate-600 bg-slate-50 p-3 rounded-lg">
                {processResult.summary}
              </p>
            </div>

            <!-- Keywords -->
            {#if processResult.keywords}
              <div>
                <h3 class="font-semibold text-slate-700 mb-2">Keywords</h3>
                <div class="flex flex-wrap gap-2">
                  {#each processResult.keywords as keyword}
                    <span
                      class="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                    >
                      {keyword}
                    </span>
                  {/each}
                </div>
              </div>
            {/if}

            <!-- Sentiment -->
            {#if processResult.sentiment !== undefined}
              <div>
                <h3 class="font-semibold text-slate-700 mb-2">
                  Sentiment Analysis
                </h3>
                <div class="flex items-center gap-2">
                  <span
                    class="{getSentimentColor(
                      processResult.sentiment
                    )} font-semibold"
                  >
                    {getSentimentLabel(processResult.sentiment)}
                  </span>
                  <span class="text-slate-500">
                    ({(processResult.sentiment * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            {/if}
          </div>

          <!-- Legal Entities -->
          {#if processResult.legal_entities && processResult.legal_entities.length > 0}
            <div>
              <h3 class="font-semibold text-slate-700 mb-2">Legal Entities</h3>
              <div class="space-y-2 max-h-64 overflow-y-auto">
                {#each processResult.legal_entities as entity}
                  <div class="bg-slate-50 p-3 rounded-lg">
                    <div class="flex justify-between items-start mb-1">
                      <span class="font-medium text-slate-800"
                        >{entity.name}</span
                      >
                      <span class="text-xs text-slate-500"
                        >{(entity.confidence * 100).toFixed(1)}%</span
                      >
                    </div>
                    <span class="text-sm text-slate-600 capitalize"
                      >{entity.type}</span
                    >
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>

        <!-- Metadata -->
        <div class="mt-6 pt-4 border-t border-slate-200">
          <div class="flex justify-between text-sm text-slate-500">
            <span>Processing Time: {processResult.processing_time}</span>
            <span
              >Confidence: {processResult.confidence
                ? (processResult.confidence * 100).toFixed(1) + "%"
                : "N/A"}</span
            >
            <span>{processResult.cached_result ? "📋 Cached" : "🔥 Fresh"}</span
            >
          </div>
        </div>
      {/if}

      <Dialog.Close
        class="mt-6 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors"
      >
        Close
      </Dialog.Close>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<!-- Search Results Dialog -->
<Dialog.Root bind:open={showSearchDialog}>
  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 bg-black/50 z-40" />
    <Dialog.Content
      class="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white rounded-xl shadow-2xl p-6 max-w-4xl max-h-[80vh] overflow-y-auto z-50"
    >
      {#if searchResults}
        <Dialog.Title class="text-2xl font-bold text-slate-800 mb-4">
          🔍 Vector Search Results
        </Dialog.Title>

        <div class="mb-4 text-sm text-slate-600">
          Found {searchResults.total} results for "{searchResults.query}" in {searchResults.took}
        </div>

        <div class="space-y-4">
          {#each searchResults.results as result}
            <div
              class="border border-slate-200 rounded-lg p-4 hover:bg-slate-50 transition-colors"
            >
              <div class="flex justify-between items-start mb-2">
                <span class="font-medium text-slate-800">{result.id}</span>
                <span class="text-sm font-semibold text-green-600">
                  {(result.score * 100).toFixed(1)}% match
                </span>
              </div>
              <p class="text-slate-600 mb-3">{result.content}</p>
              <div class="flex flex-wrap gap-2 text-xs">
                {#each Object.entries(result.metadata) as [key, value]}
                  <span class="px-2 py-1 bg-slate-100 text-slate-700 rounded">
                    {key}: {value}
                  </span>
                {/each}
              </div>
            </div>
          {/each}
        </div>
      {/if}

      <Dialog.Close
        class="mt-6 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors"
      >
        Close
      </Dialog.Close>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<style>
  /* UnoCSS will handle most styling, but we can add custom styles here if needed */
  .animate-spin {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>
