<script lang="ts">
  import { ollama } from "$lib/ai/ollama";
  import DocumentAnalysis from "$lib/components/DocumentAnalysis.svelte";
  import LegalChat from "$lib/components/LegalChat.svelte";
  import { onMount } from "svelte";

  let activeTab = $state<"chat" | "analysis" | "search">("chat");
  let isOllamaConnected = $state(false);
  let availableModels = $state<string[]>([]);

  onMount(() => {
    checkOllamaConnection();
    const interval = setInterval(checkOllamaConnection, 30000);
    return () => clearInterval(interval);
  });

  async function checkOllamaConnection() {
    try {
      const connected = await ollama.healthCheck();
      isOllamaConnected = connected;
      if (connected) {
        const models = await ollama.listModels();
        availableModels = models.map((m: any) => m.name);
      }
    } catch (err) {
      console.error("Ollama connection error:", err);
      isOllamaConnected = false;
    }
  }

  function handleAnalysisComplete(result: any) {
    console.log("Analysis complete:", result);
  }

  function handleChatMessage(message: any) {
    console.log("New message:", message);
  }
</script>

<div class="container mx-auto p-6">
  <div class="flex items-center justify-between mb-6">
    <div class="flex gap-4">
      <button
        on:click={() => (activeTab = "chat")}
        class="py-2 px-3 border-b-2 font-medium text-sm transition-colors"
        class:border-blue-500={activeTab === "chat"}
        class:text-blue-600={activeTab === "chat"}
      >
        Legal Chat
      </button>
      <button
        on:click={() => (activeTab = "analysis")}
        class="py-2 px-3 border-b-2 font-medium text-sm transition-colors"
        class:border-blue-500={activeTab === "analysis"}
        class:text-blue-600={activeTab === "analysis"}
      >
        Document Analysis
      </button>
      <button
        on:click={() => (activeTab = "search")}
        class="py-2 px-3 border-b-2 font-medium text-sm transition-colors"
        class:border-blue-500={activeTab === "search"}
        class:text-blue-600={activeTab === "search"}
      >
        Knowledge Search
      </button>
    </div>
    <div class="text-xs text-gray-500">
      Status: {isOllamaConnected ? "Ollama connected" : "Ollama not connected"}
    </div>
  </div>

  {#if !isOllamaConnected}
    <div
      class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 mb-8"
    >
      <h3 class="font-semibold text-red-800 dark:text-red-300 mb-2">
        Ollama Service Not Connected
      </h3>
      <p class="text-sm mb-2">
        Please ensure Ollama is running with GPU support:
      </p>
      <ol class="list-decimal ml-5 text-sm space-y-1">
        <li>
          Navigate to:
          C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app\\local-models
        </li>
        <li>Run: .\\RUN-GPU-SETUP.bat</li>
        <li>Wait for "Ollama is running with GPU acceleration!"</li>
      </ol>
      <button
        on:click={checkOllamaConnection}
        class="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
        >Retry Connection</button
      >
    </div>
  {/if}

  {#if activeTab === "chat"}
    <div class="grid gap-6 md:grid-cols-3">
      <div class="md:col-span-2">
        <LegalChat
          systemPrompt="You are an expert legal AI assistant. Provide accurate, professional legal guidance based on current laws and precedents."
          onMessage={handleChatMessage}
        />
      </div>
      <div class="space-y-4">
        <div>
          <h3 class="font-semibold mb-2">Quick Actions</h3>
          <div class="space-y-2">
            <button
              class="w-full text-left px-4 py-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >Draft Contract</button
            >
            <button
              class="w-full text-left px-4 py-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >Legal Research</button
            >
            <button
              class="w-full text-left px-4 py-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >Risk Analysis</button
            >
          </div>
        </div>
        <div>
          <h3 class="font-semibold mb-2">Available Models</h3>
          <ul class="text-sm space-y-1">
            {#each availableModels as model}
              <li class="px-2 py-1 rounded bg-gray-50 dark:bg-gray-800">
                {model}
              </li>
            {/each}
          </ul>
        </div>
      </div>
    </div>
  {:else if activeTab === "analysis"}
    <DocumentAnalysis
      onAnalysisComplete={handleAnalysisComplete}
      maxSizeMB={20}
    />
  {:else if activeTab === "search"}
    <div class="p-6 rounded-lg border dark:border-gray-700">
      <h3 class="font-semibold mb-2">Knowledge Base Search</h3>
      <p class="text-sm text-gray-600 dark:text-gray-300">
        Search through your legal documents using semantic search powered by
        embeddings.
      </p>
    </div>
  {/if}

  <footer class="mt-10 text-xs text-gray-500">
    Powered by Ollama + pgvector + LangChain • GPU Accelerated • Local LLM •
    Privacy First
  </footer>
</div>
