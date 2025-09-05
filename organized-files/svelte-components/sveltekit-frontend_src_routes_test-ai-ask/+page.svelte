<script lang="ts">
  import { browser } from "$app/environment";
  import { onMount } from "svelte";
  import AIChatInterface from "$lib/components/ai/AIChatInterface.svelte";
  import {
    aiStore,
    conversation,
    settings,
    status,
  } from "$lib/stores/ai-store";

  // Page state
  let isPageReady = false;
  let manualTestQuery = "What are the key elements of a valid contract?";
  let manualTestResponse: any = null;
  let manualTestLoading = false;
  let manualTestError: string | null = null;
  let healthCheckResults: any = null;
  let selectedExample = "";

  // Sample queries for testing
  const sampleQueries = [
    "What are the key elements of a valid contract?",
    "Explain the difference between civil and criminal law",
    "What is the statute of limitations for personal injury cases?",
    "How does the discovery process work in litigation?",
    "What are the requirements for a valid will?",
    "What constitutes probable cause for a search warrant?",
    "How does attorney-client privilege work?",
    "What are the stages of a criminal trial?",
  ];

  // Missing variables that are referenced in template
  let query = "";
  let isLoading = false;
  let error: string | null = null;
  let response: any = null;
  let exampleQueries = sampleQueries; // Alias for template

  // Initialize page
  onMount(async () => {
    if (browser) {
      // Wait for AI store to initialize
      await new Promise((resolve) => setTimeout(resolve, 200));
      isPageReady = true;
      // Auto-check health on load
      checkHealthEndpoints();
}
  });

  // Select example query
  function selectQuery(query: string) {
    manualTestQuery = query;
    selectedExample = query;
}
  // Manual API test function (for debugging)
  async function testAPIDirectly() {
    if (!manualTestQuery.trim()) return;

    manualTestLoading = true;
    manualTestError = null;
    manualTestResponse = null;

    try {
      const res = await fetch("/api/ai/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: manualTestQuery,
          context: [],
          includeHistory: false,
          maxSources: 3,
          searchThreshold: 0.7,
          useCache: false,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}: ${res.statusText}`);
}
      if (!data.success) {
        throw new Error(data.error || "API request failed");
}
      manualTestResponse = data.data;
    } catch (error) {
      console.error("Manual API test failed:", error);
      manualTestError =
        error instanceof Error ? error.message : "Unknown error occurred";
    } finally {
      manualTestLoading = false;
}}
  // Check health endpoints
  async function checkHealthEndpoints() {
    try {
      console.log("🔍 Checking AI health endpoints...");

      // Check local AI health
      const localRes = await fetch("/api/ai/health/local");
      const localHealth = await localRes.json();
      console.log("Local AI Health:", localHealth);

      // Check cloud AI health
      let cloudHealth = null;
      try {
        const cloudRes = await fetch("/api/ai/health/cloud");
        cloudHealth = await cloudRes.json();
        console.log("Cloud AI Health:", cloudHealth);
      } catch (error) {
        console.log("Cloud AI Health: Not available");
}
      healthCheckResults = {
        local: localHealth,
        cloud: cloudHealth,
        timestamp: new Date(),
      };
    } catch (error) {
      console.error("Health check failed:", error);
      manualTestError =
        "Health check failed. Make sure the development server is running.";
}}
  // Clear conversation
  function clearConversation() {
    if (confirm("Clear conversation history?")) {
      conversation.set({
        id: `conversation-${Date.now()}`,
        messages: [],
        isActive: false,
        lastUpdated: Date.now(),
      });
      manualTestResponse = null;
      manualTestError = null;
}}
  // Export conversation
  function exportConversation() {
    const data = {
      conversation: $conversation,
      settings: $settings,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ai-conversation-${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
  // Missing functions that are referenced in template
  function testGemma3() {
    console.log("Testing Gemma3...");
    // Add your test logic here
}
</script>

<svelte:head>
  <title>Gemma3 Q4_K_M Integration Test - Legal AI Assistant</title>
  <meta
    name="description"
    content="Test page for Gemma3 Q4_K_M GGUF local LLM integration with SvelteKit SSR hydration"
  />
</svelte:head>

<div class="space-y-4">
  <header class="space-y-4">
    <h1>🤖 Gemma3 Q4_K_M Integration Test</h1>
    <p class="space-y-4">
      Testing SSR-safe AI store, Ollama service integration, and shared UI
      components
    </p>

    <div class="space-y-4">
      <div class="space-y-4">
        <span class="space-y-4">Page Ready:</span>
        <span class="space-y-4" class:ready={isPageReady}
          >{isPageReady ? "✅" : "⏳"}</span
        >
      </div>
      <div class="space-y-4">
        <span class="space-y-4">AI Initializing:</span>
        <span class="space-y-4" class:loading={$status.isInitializing}
          >{$status.isInitializing ? "⏳" : "✅"}</span
        >
      </div>
      <div class="space-y-4">
        <span class="space-y-4">Local AI:</span>
        <span class="space-y-4" class:ready={$status.localModelAvailable}
          >{$status.localModelAvailable ? "✅" : "❌"}</span
        >
      </div>
      <div class="space-y-4">
        <span class="space-y-4">Cloud AI:</span>
        <span class="space-y-4" class:ready={$status.cloudModelAvailable}
          >{$status.cloudModelAvailable ? "✅" : "❌"}</span
        >
      </div>
    </div>
  </header>

  <main class="space-y-4">
    {#if !isPageReady}
      <div class="space-y-4">
        <div class="space-y-4"></div>
        <p>Initializing test environment...</p>
      </div>
    {:else}
      <!-- Main AI Chat Interface -->
      <section class="space-y-4">
        <h2>🗨️ Interactive AI Chat</h2>
        <p class="space-y-4">
          Test the full AI chat experience with SSR-safe state management and
          real-time interactions.
        </p>

        <div class="space-y-4">
          <!-- AIChatInterface component placeholder -->
          <div class="bg-gray-100 rounded-lg p-8 text-center">
            <p class="text-gray-600">AI Chat Interface (Component not implemented yet)</p>
            <p class="text-sm text-gray-500 mt-2">Height: 400px, Visible: true</p>
          </div>
        </div>

        <!-- Example Queries -->
        <div class="space-y-4">
          <h3>📝 Quick Test Queries</h3>
          <div class="space-y-4">
            {#each exampleQueries as query}
              <button
                type="button"
                class="space-y-4"
                onclick={() => aiStore.sendMessage(query)}
                disabled={$status.isLoading}
              >
                {query}
              </button>
            {/each}
          </div>
        </div>
      </section>

      <!-- Manual API Testing -->
      <section class="space-y-4">
        <h2>🔧 Manual API Testing</h2>
        <p class="space-y-4">
          Direct API endpoint testing for debugging and validation.
        </p>

        <div class="space-y-4">
          <div class="space-y-4">
            <label for="manual-query">Test Query:</label>
            <textarea
              id="manual-query"
              bind:value={manualTestQuery}
              placeholder="Enter your test query..."
              rows={1}
            ></textarea>
          </div>

          <div class="space-y-4">
            <button
              type="button"
              class="space-y-4"
              onclick={() => testAPIDirectly()}
              disabled={manualTestLoading}
            >
              {manualTestLoading ? "⏳ Testing..." : "🚀 Test API"}
            </button>

            <button
              type="button"
              class="space-y-4"
              onclick={() => checkHealthEndpoints()}
            >
              🏥 Check Health
            </button>
          </div>
        </div>

        {#if manualTestError}
          <div class="space-y-4">
            <h4>❌ Error</h4>
            <pre>{manualTestError}</pre>
          </div>
        {/if}

        {#if manualTestResponse}
          <div class="space-y-4">
            <h4>✅ Response</h4>
            <div class="space-y-4">
              <div class="space-y-4">
                <span class="space-y-4">Provider:</span>
                <span class="space-y-4">{manualTestResponse.provider}</span>
              </div>
              <div class="space-y-4">
                <span class="space-y-4">Model:</span>
                <span class="space-y-4">{manualTestResponse.model}</span>
              </div>
              <div class="space-y-4">
                <span class="space-y-4">Confidence:</span>
                <span class="space-y-4"
                  >{Math.round(manualTestResponse.confidence * 100)}%</span
                >
              </div>
              <div class="space-y-4">
                <span class="space-y-4">Execution Time:</span>
                <span class="space-y-4">{manualTestResponse.executionTime}ms</span>
              </div>
              <div class="space-y-4">
                <span class="space-y-4">From Cache:</span>
                <span class="space-y-4"
                  >{manualTestResponse.fromCache ? "Yes" : "No"}</span
                >
              </div>
            </div>

            <div class="space-y-4">
              <h5>Answer:</h5>
              <p>{manualTestResponse.answer}</p>
            </div>

            {#if manualTestResponse.sources && manualTestResponse.sources.length > 0}
              <div class="space-y-4">
                <h5>Sources ({manualTestResponse.sources.length}):</h5>
                {#each manualTestResponse.sources as source}
                  <div class="space-y-4">
                    <div class="space-y-4">
                      <span class="space-y-4">{source.title}</span>
                      <span class="space-y-4"
                        >{Math.round(source.score * 100)}%</span
                      >
                    </div>
                    <p class="space-y-4">{source.content}</p>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        {/if}
      </section>

      <!-- Store State Debug -->
      <section class="space-y-4">
        <h2>🐛 Store State Debug</h2>
        <p class="space-y-4">
          Real-time view of AI store state for debugging and validation.
        </p>

        <div class="space-y-4">
          <div class="space-y-4">
            <h4>🔧 Status Store</h4>
            <pre>{JSON.stringify($status, null, 2)}</pre>
          </div>

          <div class="space-y-4">
            <h4>⚙️ Settings Store</h4>
            <pre>{JSON.stringify($settings, null, 2)}</pre>
          </div>

          <div class="space-y-4">
            <h4>💬 Conversation Store</h4>
            <div class="space-y-4">
              <p><strong>ID:</strong> {$conversation.id || "None"}</p>
              <p><strong>Messages:</strong> {$conversation.messages.length}</p>
              <p><strong>Active:</strong> {$conversation.isActive}</p>
              <p>
                <strong>Last Updated:</strong>
                {$conversation.lastUpdated
                  ? new Date($conversation.lastUpdated).toLocaleString()
                  : "Never"}
              </p>
            </div>
          </div>
        </div>

        <div class="space-y-4">
          <button
            type="button"
            class="space-y-4"
            onclick={() => aiStore.clearConversation()}
          >
            🗑️ Clear Conversation
          </button>
          <button
            type="button"
            class="space-y-4"
            onclick={() => aiStore.reset()}
          >
            🔄 Reset All Stores
          </button>
          <button
            type="button"
            class="space-y-4"
            onclick={() => aiStore.initialize()}
          >
            🔄 Reinitialize AI
          </button>
        </div>
      </section>
      <!-- Instructions -->
      <section class="space-y-4">
        <h2>🔧 Setup Instructions</h2>
        <p class="space-y-4">
          Quick setup guide for local LLM integration with Ollama and llama.cpp.
        </p>

        <div class="space-y-4">
          <div class="space-y-4">
            <h4>📦 1. Start Local LLM Services</h4>
            <div class="space-y-4">
              <code>npm run llm:start</code>
            </div>
            <p>This starts both Ollama and llama.cpp with the Gemma3 model.</p>
          </div>

          <div class="space-y-4">
            <h4>🧪 2. Test Integration</h4>
            <div class="space-y-4">
              <code>npm run llm:test</code>
            </div>
            <p>Run comprehensive tests to verify all services are working.</p>
          </div>

          <div class="space-y-4">
            <h4>🌐 3. Start Web Application</h4>
            <div class="space-y-4">
              <code>npm run dev</code>
            </div>
            <p>Start the SvelteKit development server on port 5173.</p>
          </div>

          <div class="space-y-4">
            <h4>🏥 4. Check Service Health</h4>
            <button
              type="button"
              class="space-y-4"
              onclick={() => checkHealthEndpoints()}
            >
              Check Health Status
            </button>
            <p>Verify that all local LLM services are running correctly.</p>
          </div>
        </div>

        <div class="space-y-4">
          <h4>🛠️ Troubleshooting</h4>
          <ul>
            <li>
              <strong>Ollama not starting:</strong> Check if port 11434 is already
              in use
            </li>
            <li>
              <strong>Gemma model not found:</strong> Ensure the mo16.gguf file is
              in gemma3Q4_K_M directory
            </li>
            <li>
              <strong>Generation failing:</strong> Try loading the model
              manually: <code>ollama create gemma3-legal -f Modelfile</code>
            </li>
            <li>
              <strong>Services not connecting:</strong> Run
              <code>npm run llm:test:verbose</code> for detailed diagnostics
            </li>
          </ul>
        </div>
      </section>
    {/if}
  </main>
</div>

<div class="space-y-4">
  <div
    class="space-y-4"
  >
    <h1 class="space-y-4">🤖 Gemma3 Local LLM Test</h1>
    <p class="space-y-4">
      Test the AI assistant with local Gemma3 inference for legal queries
    </p>
  </div>

  <!-- Sample Queries -->
  <div class="space-y-4">
    <h2 class="space-y-4">
      📝 Sample Legal Queries
    </h2>
    <div class="space-y-4">
      {#each sampleQueries as sampleQuery}
        <button
          class="space-y-4"
          onclick={() => selectQuery(sampleQuery)}
        >
          <span class="space-y-4">{sampleQuery}</span>
        </button>
      {/each}
    </div>
  </div>

  <!-- Query Input -->
  <div class="space-y-4">
    <label for="query" class="space-y-4">
      💬 Ask the Legal AI Assistant
    </label>
    <textarea
      id="query"
      bind:value={query}
      placeholder="Enter your legal question here..."
      class="space-y-4"
      rows={1}
    ></textarea>

    <button
      onclick={() => testGemma3()}
      disabled={isLoading || !query.trim()}
      class="space-y-4"
    >
      {#if isLoading}
        <span class="space-y-4">
          <svg
            class="space-y-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              class="space-y-4"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              stroke-width="4"
            ></circle>
            <path
              class="space-y-4"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          Processing with Gemma3...
        </span>
      {:else}
        🚀 Ask Gemma3
      {/if}
    </button>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-4">
          <svg
            class="space-y-4"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fill-rule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clip-rule="evenodd"
            />
          </svg>
        </div>
        <div class="space-y-4">
          <h3 class="space-y-4">Error</h3>
          <p class="space-y-4">{error}</p>
        </div>
      </div>
    </div>
  {/if}

  <!-- Response Display -->
  {#if response}
    <div class="space-y-4">
      <!-- Response Header -->
      <div class="space-y-4">
        <h2 class="space-y-4">
          ✅ AI Response
        </h2>
        <div class="space-y-4">
          <div class="space-y-4">
            <span class="space-y-4">Provider:</span>
            <span
              class="space-y-4"
            >
              {response.provider === "local"
                ? "🏠 Local"
                : response.provider === "cloud"
                  ? "☁️ Cloud"
                  : "🔧 Hybrid"}
            </span>
          </div>
          <div class="space-y-4">
            <span class="space-y-4">Model:</span>
            <span class="space-y-4">{response.model}</span>
          </div>
          <div class="space-y-4">
            <span class="space-y-4">Confidence:</span>
            <span class="space-y-4"
              >{(response.confidence * 100).toFixed(1)}%</span
            >
          </div>
          <div class="space-y-4">
            <span class="space-y-4">Time:</span>
            <span class="space-y-4">{response.executionTime}ms</span>
          </div>
        </div>
      </div>

      <!-- Answer -->
      <div class="space-y-4">
        <h3 class="space-y-4">📋 Answer</h3>
        <div class="space-y-4">
          {@html response.answer
            .replace(/\n/g, "<br>")
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")}
        </div>
      </div>

      <!-- Sources -->
      {#if response.sources && response.sources.length > 0}
        <div class="space-y-4">
          <h3 class="space-y-4">
            📚 Sources ({response.sources.length})
          </h3>
          <div class="space-y-4">
            {#each response.sources as source, index}
              <div class="space-y-4">
                <div class="space-y-4">
                  <h4 class="space-y-4">{source.title}</h4>
                  <span
                    class="space-y-4"
                  >
                    {(source.score * 100).toFixed(1)}% match
                  </span>
                </div>
                <p class="space-y-4">{source.content}</p>
                <span
                  class="space-y-4"
                >
                  {source.type}
                </span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Instructions -->
  <div class="space-y-4">
    <h3 class="space-y-4">
      🔧 Testing Instructions
    </h3>
    <ul class="space-y-4">
      <li>• This page tests the Gemma3 local LLM integration</li>
      <li>
        • Queries are processed using local AI (no data sent to external
        services)
      </li>
      <li>
        • If Gemma3 is not available, the system will fall back to template
        responses
      </li>
      <li>• Check the browser console for detailed debug information</li>
      <li>• Use the sample queries above or create your own legal questions</li>
    </ul>
  </div>
</div>

<style>
  /* @unocss-include */
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .ready {
    color: #22c55e;
  }
  .loading {
    color: #f59e0b;
  }
</style>
