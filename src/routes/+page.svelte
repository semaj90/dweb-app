<script lang="ts">
  import { onMount } from 'svelte';
  import LegalChat from '$lib/components/LegalChat.svelte';
  import DocumentAnalysis from '$lib/components/DocumentAnalysis.svelte';
  import { ollama } from '$lib/ai/ollama';
  
  // Svelte 5 state
  let activeTab = $state<'chat' | 'analysis' | 'search'>('chat');
  let isOllamaConnected = $state(false);
  let availableModels = $state<string[]>([]);
  
  onMount(async () => {
    // Check Ollama connection
    checkOllamaConnection();
    
    // Set up periodic health check
    const interval = setInterval(checkOllamaConnection, 30000);
    
    return () => clearInterval(interval);
  });
  
  async function checkOllamaConnection() {
    try {
      const connected = await ollama.healthCheck();
      isOllamaConnected = connected;
      
      if (connected) {
        const models = await ollama.listModels();
        availableModels = models.map(m => m.name);
      }
    } catch (error) {
      console.error('Ollama connection error:', error);
      isOllamaConnected = false;
    }
  }
  
  function handleAnalysisComplete(result: any) {
    console.log('Analysis complete:', result);
    // You could store this in a database or state management
  }
  
  function handleChatMessage(message: any) {
    console.log('New message:', message);
    // Store in database if needed
  }
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900">
  <!-- Header -->
  <header class="bg-white dark:bg-gray-800 shadow-sm border-b dark:border-gray-700">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center h-16">
        <div class="flex items-center gap-3">
          <div class="i-carbon-scales text-2xl text-blue-600"></div>
          <h1 class="text-xl font-bold text-gray-900 dark:text-white">
            Legal AI Assistant
          </h1>
        </div>
        
        <div class="flex items-center gap-4">
          <!-- Connection Status -->
          <div class="flex items-center gap-2 text-sm">
            {#if isOllamaConnected}
              <div class="i-carbon-dot-mark text-green-500"></div>
              <span class="text-gray-600 dark:text-gray-400">
                Ollama Connected ({availableModels.length} models)
              </span>
            {:else}
              <div class="i-carbon-dot-mark text-red-500"></div>
              <span class="text-gray-600 dark:text-gray-400">
                Ollama Disconnected
              </span>
            {/if}
          </div>
          
          <!-- GPU Status -->
          <div class="flex items-center gap-2 text-sm">
            <div class="i-carbon-chip text-blue-500"></div>
            <span class="text-gray-600 dark:text-gray-400">GPU Enabled</span>
          </div>
        </div>
      </div>
    </div>
  </header>
  
  <!-- Navigation Tabs -->
  <div class="bg-white dark:bg-gray-800 border-b dark:border-gray-700">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <nav class="flex gap-8" aria-label="Tabs">
        <button
          onclick={() => activeTab = 'chat'}
          class="py-4 px-1 border-b-2 font-medium text-sm transition-colors
                 {activeTab === 'chat' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'}"
        >
          <div class="flex items-center gap-2">
            <div class="i-carbon-chat-bot"></div>
            Legal Chat
          </div>
        </button>
        
        <button
          onclick={() => activeTab = 'analysis'}
          class="py-4 px-1 border-b-2 font-medium text-sm transition-colors
                 {activeTab === 'analysis' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'}"
        >
          <div class="flex items-center gap-2">
            <div class="i-carbon-document-tasks"></div>
            Document Analysis
          </div>
        </button>
        
        <button
          onclick={() => activeTab = 'search'}
          class="py-4 px-1 border-b-2 font-medium text-sm transition-colors
                 {activeTab === 'search' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'}"
        >
          <div class="flex items-center gap-2">
            <div class="i-carbon-search"></div>
            Knowledge Search
          </div>
        </button>
      </nav>
    </div>
  </div>
  
  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if !isOllamaConnected}
      <!-- Connection Error State -->
      <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 
                  rounded-lg p-6 mb-8">
        <div class="flex items-start gap-3">
          <div class="i-carbon-warning text-red-600 text-xl mt-0.5"></div>
          <div>
            <h3 class="font-semibold text-red-900 dark:text-red-300">
              Ollama Service Not Connected
            </h3>
            <p class="text-red-700 dark:text-red-400 mt-1">
              Please ensure Ollama is running with GPU support:
            </p>
            <ol class="list-decimal list-inside mt-2 space-y-1 text-sm text-red-600 dark:text-red-400">
              <li>Navigate to: <code class="bg-red-100 dark:bg-red-900 px-1 rounded">
                C:\Users\james\Desktop\deeds-web\deeds-web-app\local-models
              </code></li>
              <li>Run: <code class="bg-red-100 dark:bg-red-900 px-1 rounded">
                .\RUN-GPU-SETUP.bat
              </code></li>
              <li>Wait for "Ollama is running with GPU acceleration!"</li>
            </ol>
            <button 
              onclick={checkOllamaConnection}
              class="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    {/if}
    
    <!-- Tab Content -->
    <div class="mt-6">
      {#if activeTab === 'chat'}
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Chat Interface -->
          <div class="lg:col-span-2">
            <LegalChat 
              systemPrompt="You are an expert legal AI assistant. Provide accurate, 
                           professional legal guidance based on current laws and precedents."
              onMessage={handleChatMessage}
            />
          </div>
          
          <!-- Quick Actions -->
          <div class="space-y-4">
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h3 class="font-semibold mb-4 text-gray-900 dark:text-white">
                Quick Actions
              </h3>
              <div class="space-y-2">
                <button class="w-full text-left px-4 py-3 bg-gray-50 dark:bg-gray-700 
                             rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 
                             transition-colors">
                  <div class="flex items-center gap-3">
                    <div class="i-carbon-document-blank text-blue-600"></div>
                    <span class="text-sm">Draft Contract</span>
                  </div>
                </button>
                <button class="w-full text-left px-4 py-3 bg-gray-50 dark:bg-gray-700 
                             rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 
                             transition-colors">
                  <div class="flex items-center gap-3">
                    <div class="i-carbon-rule text-green-600"></div>
                    <span class="text-sm">Legal Research</span>
                  </div>
                </button>
                <button class="w-full text-left px-4 py-3 bg-gray-50 dark:bg-gray-700 
                             rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 
                             transition-colors">
                  <div class="flex items-center gap-3">
                    <div class="i-carbon-warning-alt text-amber-600"></div>
                    <span class="text-sm">Risk Analysis</span>
                  </div>
                </button>
              </div>
            </div>
            
            <!-- Model Info -->
            <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 class="font-medium text-blue-900 dark:text-blue-300 mb-2">
                Available Models
              </h4>
              <ul class="space-y-1 text-sm text-blue-700 dark:text-blue-400">
                {#each availableModels as model}
                  <li class="flex items-center gap-2">
                    <div class="i-carbon-checkmark-filled text-xs"></div>
                    {model}
                  </li>
                {/each}
              </ul>
            </div>
          </div>
        </div>
        
      {:else if activeTab === 'analysis'}
        <DocumentAnalysis 
          onAnalysisComplete={handleAnalysisComplete}
          maxSizeMB={20}
        />
        
      {:else if activeTab === 'search'}
        <!-- Knowledge Search Interface -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 class="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
            Knowledge Base Search
          </h2>
          <p class="text-gray-600 dark:text-gray-400">
            Search through your legal documents using semantic search powered by embeddings.
          </p>
          <!-- Add search interface here -->
        </div>
      {/if}
    </div>
  </main>
  
  <!-- Footer -->
  <footer class="mt-16 bg-white dark:bg-gray-800 border-t dark:border-gray-700">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div class="flex justify-between items-center text-sm text-gray-500 dark:text-gray-400">
        <div>
          Powered by Ollama + pgvector + LangChain
        </div>
        <div class="flex items-center gap-4">
          <span>GPU Accelerated</span>
          <span>•</span>
          <span>Local LLM</span>
          <span>•</span>
          <span>Privacy First</span>
        </div>
      </div>
    </div>
  </footer>
</div>
