<!-- SvelteKit 2 + Svelte 5 Streaming Demo with Provider Selection -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';

  // Reactive state
  let provider = $state('ollama');
  let model = $state('gemma3:latest-legal');
  let prompt = $state('Analyze the legal implications of this evidence...');
  let output = $state('');
  let isStreaming = $state(false);
  let streamStats = $state<any>(null);
  
  // Provider configurations
  const providers = {
    ollama: {
      name: 'Ollama (Local)',
      models: ['gemma3:latest-legal', 'llama3', 'gemma2:9b'],
      description: 'Local Ollama instance with legal models'
    },
    llamacpp: {
      name: 'llama.cpp (Native)',
      models: ['ggml-model', 'llama-7b-chat'],
      description: 'Native llama.cpp server'
    },
    wasm: {
      name: 'WebAssembly (Browser)',
      models: ['wasm-llm-small'],
      description: 'Browser-based WASM model'
    },
    legal_agent: {
      name: 'Legal Orchestrator',
      models: ['legal-analyst', 'legal-researcher'],
      description: 'Multi-agent legal AI system'
    }
  };

  const samplePrompts = [
    'Summarize the key evidence in this criminal case...',
    'What are the jurisdiction requirements for this contract dispute?',
    'Analyze the chain of custody for this digital evidence...',
    'Draft a brief motion summary based on these facts...',
    'Identify potential procedural issues in this filing...'
  ];

  async function startStream() {
    if (isStreaming) return;
    
    isStreaming = true;
    output = '';
    streamStats = null;

    try {
      const response = await fetch('/api/ai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          prompt,
          model,
          streaming: true,
          enableCompression: true,
          enableSIMD: true,
          temperature: 0.1,
          maxTokens: 2048
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const data = JSON.parse(line);
            
            if (data.type === 'token' && data.token) {
              output += data.token;
            } else if (data.type === 'stream_complete') {
              streamStats = data.stats;
            } else if (data.type === 'error') {
              throw new Error(data.error);
            }
          } catch (parseError) {
            console.warn('Parse error:', parseError, 'Line:', line);
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      output += `\n\n❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    } finally {
      isStreaming = false;
    }
  }

  function stopStream() {
    isStreaming = false;
  }

  function clearOutput() {
    output = '';
    streamStats = null;
  }

  function useSamplePrompt(sample: string) {
    prompt = sample;
  }

  onMount(() => {
    // Auto-focus prompt textarea
    const textarea = document.querySelector('textarea');
    textarea?.focus();
  });
</script>

<div class="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-6">
  <div class="max-w-6xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
        🚀 Legal AI Streaming Demo
      </h1>
      <p class="text-slate-300 text-lg">
        Multi-provider streaming with SIMD optimization and compression
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Controls Panel -->
      <div class="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h2 class="text-xl font-semibold mb-4 text-blue-300">⚙️ Configuration</h2>
        
        <!-- Provider Selection -->
        <div class="mb-4">
          <label class="block text-sm font-medium text-slate-300 mb-2">Provider</label>
          <select 
            bind:value={provider}
            class="w-full p-2 rounded bg-slate-700 border border-slate-600 text-white focus:border-blue-400 focus:outline-none"
          >
            {#each Object.entries(providers) as [key, config]}
              <option value={key}>{config.name}</option>
            {/each}
          </select>
          <p class="text-xs text-slate-400 mt-1">
            {providers[provider as keyof typeof providers]?.description}
          </p>
        </div>

        <!-- Model Selection -->
        <div class="mb-4">
          <label class="block text-sm font-medium text-slate-300 mb-2">Model</label>
          <select 
            bind:value={model}
            class="w-full p-2 rounded bg-slate-700 border border-slate-600 text-white focus:border-blue-400 focus:outline-none"
          >
            {#each providers[provider as keyof typeof providers]?.models || [] as modelOption}
              <option value={modelOption}>{modelOption}</option>
            {/each}
          </select>
        </div>

        <!-- Sample Prompts -->
        <div class="mb-4">
          <label class="block text-sm font-medium text-slate-300 mb-2">Sample Prompts</label>
          <div class="space-y-1">
            {#each samplePrompts as sample}
              <button 
                onclick={() => useSamplePrompt(sample)}
                class="block w-full text-left p-2 text-xs rounded bg-slate-700/50 hover:bg-slate-700 transition-colors text-slate-300 hover:text-white"
              >
                {sample.slice(0, 50)}...
              </button>
            {/each}
          </div>
        </div>

        <!-- Stream Stats -->
        {#if streamStats}
          <div class="bg-slate-700/50 rounded p-3 border border-slate-600">
            <h3 class="text-sm font-medium text-blue-300 mb-2">📊 Stream Statistics</h3>
            <div class="text-xs space-y-1 text-slate-300">
              <div>Tokens: {streamStats.totalTokens}</div>
              <div>Time: {streamStats.responseTime}ms</div>
              <div>Speed: {streamStats.tokensPerSecond} tok/s</div>
              {#if streamStats.optimizationStats}
                <div>Compression: {streamStats.optimizationStats.compressionRate?.toFixed(1)}%</div>
                <div>SIMD: {streamStats.optimizationStats.simdProcessingRate?.toFixed(1)}%</div>
              {/if}
            </div>
          </div>
        {/if}
      </div>

      <!-- Input Panel -->
      <div class="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h2 class="text-xl font-semibold mb-4 text-green-300">💬 Prompt Input</h2>
        
        <textarea 
          bind:value={prompt}
          placeholder="Enter your legal analysis prompt..."
          rows="12"
          class="w-full p-3 rounded bg-slate-700 border border-slate-600 text-white placeholder-slate-400 focus:border-green-400 focus:outline-none resize-none"
        ></textarea>

        <div class="flex gap-2 mt-4">
          <button 
            onclick={startStream}
            disabled={isStreaming || !prompt.trim()}
            class="flex-1 bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 disabled:from-slate-600 disabled:to-slate-600 text-white font-semibold py-2 px-4 rounded transition-all duration-200 disabled:cursor-not-allowed"
          >
            {isStreaming ? '⏸️ Streaming...' : '🚀 Start Stream'}
          </button>
          
          {#if isStreaming}
            <button 
              onclick={stopStream}
              class="bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded transition-colors"
            >
              ⏹️ Stop
            </button>
          {/if}
          
          <button 
            onclick={clearOutput}
            class="bg-slate-600 hover:bg-slate-500 text-white font-semibold py-2 px-4 rounded transition-colors"
          >
            🗑️ Clear
          </button>
        </div>
      </div>

      <!-- Output Panel -->
      <div class="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h2 class="text-xl font-semibold mb-4 text-purple-300">📜 AI Response</h2>
        
        <div class="bg-slate-900/50 rounded p-4 border border-slate-600 min-h-[300px] max-h-[500px] overflow-y-auto">
          {#if isStreaming && !output}
            <div class="flex items-center justify-center py-8">
              <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
              <span class="ml-2 text-slate-400">Waiting for response...</span>
            </div>
          {:else if output}
            <pre class="whitespace-pre-wrap text-sm text-slate-100 leading-relaxed font-mono">{output}</pre>
            {#if isStreaming}
              <span class="inline-block w-2 h-4 bg-blue-400 animate-pulse ml-1"></span>
            {/if}
          {:else}
            <div class="text-center py-8 text-slate-400">
              <div class="text-4xl mb-2">🤖</div>
              <p>AI response will appear here...</p>
            </div>
          {/if}
        </div>

        {#if isStreaming}
          <div class="mt-2 text-xs text-slate-400">
            ⚡ Streaming with SIMD optimization and compression enabled
          </div>
        {/if}
      </div>
    </div>

    <!-- Footer -->
    <div class="mt-8 text-center text-slate-400 text-sm">
      <p>🔧 Native Windows • 🚀 CUDA GPU • 📡 Redis Streams • 🗄️ PostgreSQL • 🔍 Qdrant Vector DB</p>
    </div>
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    background: #0f172a;
  }
  
  /* Custom scrollbar */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 8px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: rgba(51, 65, 85, 0.3);
    border-radius: 4px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background: rgba(148, 163, 184, 0.5);
    border-radius: 4px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: rgba(148, 163, 184, 0.7);
  }
</style>