<script lang="ts">
  import SelfPromptingDemo from '$lib/components/dev/SelfPromptingDemo.svelte';
</script>

<svelte:head>
  <title>Self-Prompting Agent Orchestration Demo - Legal AI Development</title>
  <meta name="description" content="Comprehensive demonstration of multi-agent coordination with self-prompting loops and Context7 MCP integration" />
</svelte:head>

<main class="min-h-screen bg-gray-50">
  <div class="max-w-full">
    <div class="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-12">
      <div class="max-w-7xl mx-auto px-6">
        <h1 class="text-4xl font-bold mb-4">
          Self-Prompting Agent Orchestration
        </h1>
        <p class="text-xl text-blue-100 max-w-4xl">
          Experience advanced AI agent coordination with self-prompting loops, 
          semantic search integration, memory graph analysis, and automated legal workflow processing.
        </p>
        
        <div class="mt-8 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div class="bg-blue-700 bg-opacity-50 rounded-lg p-4">
            <h3 class="font-semibold mb-2">🤖 Multi-Agent</h3>
            <p class="text-sm text-blue-100">AutoGen, CrewAI, vLLM, Claude coordination</p>
          </div>
          <div class="bg-purple-700 bg-opacity-50 rounded-lg p-4">
            <h3 class="font-semibold mb-2">🔄 Self-Prompting</h3>
            <p class="text-sm text-purple-100">Agents generate their own follow-up prompts</p>
          </div>
          <div class="bg-indigo-700 bg-opacity-50 rounded-lg p-4">
            <h3 class="font-semibold mb-2">⚖️ Legal AI</h3>
            <p class="text-sm text-indigo-100">Evidence analysis and case building workflows</p>
          </div>
          <div class="bg-teal-700 bg-opacity-50 rounded-lg p-4">
            <h3 class="font-semibold mb-2">🔧 Context7</h3>
            <p class="text-sm text-teal-100">MCP tools for development guidance</p>
          </div>
        </div>
      </div>
    </div>

    <SelfPromptingDemo />

    <footer class="bg-white border-t border-gray-200 py-8">
      <div class="max-w-7xl mx-auto px-6">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 class="font-semibold text-gray-900 mb-3">Architecture Overview</h3>
            <ul class="text-sm text-gray-600 space-y-1">
              <li>• SvelteKit 2 + Svelte 5 runes</li>
              <li>• TypeScript for type safety</li>
              <li>• UnoCSS for styling optimization</li>
              <li>• Drizzle ORM with PostgreSQL</li>
              <li>• Context7 MCP server integration</li>
            </ul>
          </div>
          
          <div>
            <h3 class="font-semibold text-gray-900 mb-3">AI Integration</h3>
            <ul class="text-sm text-gray-600 space-y-1">
              <li>• vLLM for high-performance inference</li>
              <li>• Ollama for local LLM hosting</li>
              <li>• AutoGen multi-agent framework</li>
              <li>• CrewAI workflow orchestration</li>
              <li>• Claude API for reasoning tasks</li>
            </ul>
          </div>
          
          <div>
            <h3 class="font-semibold text-gray-900 mb-3">Legal AI Features</h3>
            <ul class="text-sm text-gray-600 space-y-1">
              <li>• Evidence processing and validation</li>
              <li>• Case building and argument synthesis</li>
              <li>• Legal precedent research automation</li>
              <li>• Multi-agent collaborative analysis</li>
              <li>• Automated report generation</li>
            </ul>
          </div>
        </div>
        
        <div class="mt-8 pt-8 border-t border-gray-200 text-center">
          <p class="text-sm text-gray-500">
            Advanced AI-powered legal workflow automation with self-prompting agent orchestration.
            Built with production-ready SvelteKit architecture and Context7 MCP integration.
          </p>
        </div>
      </div>
    </footer>
  </div>
</main>

<style>
  /* Ensure full-width layout */
  :global(body) {
    margin: 0;
    padding: 0;
  }
</style>