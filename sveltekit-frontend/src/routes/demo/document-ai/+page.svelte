<!-- AI Document Processing Demo Page -->
<script lang="ts">
  import DocumentUploadSimulator from '$lib/components/ai/DocumentUploadSimulator.svelte';
  import DemoNavigation from '$lib/components/navigation/DemoNavigation.svelte';
  import { onMount } from 'svelte';

  let serviceStatus = $state({
    ollama: 'checking',
    postgresql: 'checking', 
    summarizer: 'checking',
    embeddings: 'checking'
  });

  onMount(async () => {
    await checkServices();
  });

  async function checkServices() {
    // Check Ollama service
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      serviceStatus.ollama = response.ok ? 'healthy' : 'unhealthy';
    } catch {
      serviceStatus.ollama = 'unhealthy';
    }

    // Check AI summarizer service
    try {
      const response = await fetch('/api/ai/summarize');
      serviceStatus.summarizer = response.ok ? 'healthy' : 'unhealthy';
    } catch {
      serviceStatus.summarizer = 'unhealthy';
    }

    // Check PostgreSQL via AI service
    try {
      const response = await fetch('http://localhost:8081/api/health');
      if (response.ok) {
        const health = await response.json();
        serviceStatus.postgresql = health.services?.postgresql || 'unknown';
        serviceStatus.embeddings = health.services?.ollama || 'unknown';
      }
    } catch {
      serviceStatus.postgresql = 'unhealthy';
      serviceStatus.embeddings = 'unhealthy';
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'unhealthy': return 'text-red-400';
      case 'checking': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }

  function getStatusIcon(status: string): string {
    switch (status) {
      case 'healthy': return '✅';
      case 'unhealthy': return '❌';
      case 'checking': return '⏳';
      default: return '❓';
    }
  }
</script>

<svelte:head>
  <title>AI Document Processing Demo - Legal AI</title>
  <meta name="description" content="AI-powered document processing with OCR, summarization, and embeddings" />
</svelte:head>

<div class="demo-page min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
  <div class="container mx-auto px-4 py-8">
    
    <!-- Header -->
    <div class="text-center mb-12">
      <h1 class="text-4xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent mb-4">
        🤖 AI Document Processing Demo
      </h1>
      <p class="text-xl text-gray-300 max-w-3xl mx-auto">
        Upload PDFs and documents to experience AI-powered OCR, summarization, semantic embeddings, 
        and PostgreSQL storage with local caching for files under 10MB.
      </p>
    </div>

    <!-- Service Status Dashboard -->
    <div class="bg-gray-800/50 rounded-lg p-6 mb-8 border border-gray-700">
      <h2 class="text-xl font-semibold mb-4 text-green-400">🔧 System Status</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-gray-700/50 rounded p-4 text-center">
          <div class="text-2xl mb-2">{getStatusIcon(serviceStatus.ollama)}</div>
          <div class="text-sm font-medium">Ollama LLM</div>
          <div class={`text-xs ${getStatusColor(serviceStatus.ollama)}`}>
            {serviceStatus.ollama}
          </div>
        </div>
        
        <div class="bg-gray-700/50 rounded p-4 text-center">
          <div class="text-2xl mb-2">{getStatusIcon(serviceStatus.postgresql)}</div>
          <div class="text-sm font-medium">PostgreSQL</div>
          <div class={`text-xs ${getStatusColor(serviceStatus.postgresql)}`}>
            {serviceStatus.postgresql}
          </div>
        </div>

        <div class="bg-gray-700/50 rounded p-4 text-center">
          <div class="text-2xl mb-2">{getStatusIcon(serviceStatus.summarizer)}</div>
          <div class="text-sm font-medium">AI Summarizer</div>
          <div class={`text-xs ${getStatusColor(serviceStatus.summarizer)}`}>
            {serviceStatus.summarizer}
          </div>
        </div>

        <div class="bg-gray-700/50 rounded p-4 text-center">
          <div class="text-2xl mb-2">{getStatusIcon(serviceStatus.embeddings)}</div>
          <div class="text-sm font-medium">Embeddings</div>
          <div class={`text-xs ${getStatusColor(serviceStatus.embeddings)}`}>
            {serviceStatus.embeddings}
          </div>
        </div>
      </div>
      
      <button 
        class="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
        onclick={checkServices}
      >
        🔄 Refresh Status
      </button>
    </div>

    <!-- Processing Pipeline Info -->
    <div class="bg-gray-800/50 rounded-lg p-6 mb-8 border border-gray-700">
      <h2 class="text-xl font-semibold mb-4 text-blue-400">⚡ Processing Pipeline</h2>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="text-center">
          <div class="text-3xl mb-2">📄</div>
          <div class="text-sm font-medium text-blue-300">1. Upload & OCR</div>
          <div class="text-xs text-gray-400">PDF → Text extraction</div>
        </div>
        
        <div class="text-center">
          <div class="text-3xl mb-2">🤖</div>
          <div class="text-sm font-medium text-yellow-300">2. AI Summary</div>
          <div class="text-xs text-gray-400">Ollama + Go-Llama</div>
        </div>

        <div class="text-center">
          <div class="text-3xl mb-2">🧠</div>
          <div class="text-sm font-medium text-purple-300">3. Embeddings</div>
          <div class="text-xs text-gray-400">Nomic-Embed-Text</div>
        </div>

        <div class="text-center">
          <div class="text-3xl mb-2">💾</div>
          <div class="text-sm font-medium text-green-300">4. Storage</div>
          <div class="text-xs text-gray-400">PostgreSQL + Local</div>
        </div>
      </div>
    </div>

    <!-- Features -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div class="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 class="text-lg font-semibold text-green-400 mb-3">📝 OCR Processing</h3>
        <ul class="text-sm text-gray-300 space-y-2">
          <li>• PDF text extraction</li>
          <li>• Multi-format support</li>
          <li>• High accuracy OCR</li>
          <li>• GPU acceleration</li>
        </ul>
      </div>

      <div class="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 class="text-lg font-semibold text-blue-400 mb-3">🤖 AI Summarization</h3>
        <ul class="text-sm text-gray-300 space-y-2">
          <li>• Local Ollama models</li>
          <li>• Legal context awareness</li>
          <li>• Confidence scoring</li>
          <li>• Go-Llama integration</li>
        </ul>
      </div>

      <div class="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 class="text-lg font-semibold text-purple-400 mb-3">🧠 Vector Embeddings</h3>
        <ul class="text-sm text-gray-300 space-y-2">
          <li>• 384D Nomic embeddings</li>
          <li>• Semantic search ready</li>
          <li>• PostgreSQL pgvector</li>
          <li>• Neo4j graph integration</li>
        </ul>
      </div>
    </div>

    <!-- Main Upload Component -->
    <DocumentUploadSimulator />

    <!-- Footer Info -->
    <div class="mt-12 text-center text-gray-400">
      <p class="text-sm">
        🚀 Powered by: Ollama • Go-Llama • Nomic-Embed • PostgreSQL • Neo4j • SvelteKit 2
      </p>
      <p class="text-xs mt-2">
        Files under 10MB are cached locally for instant access. Larger files stored in PostgreSQL.
      </p>
    </div>
  </div>
</div>

<!-- Demo Navigation -->
<DemoNavigation />

<style>
  .demo-page {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .container {
    max-width: 1200px;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
  }

  .animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
</style>