<script lang="ts">
  import MCPToolsDemo from '$lib/components/dev/MCPToolsDemo.svelte';
</script>

<svelte:head>
  <title>Context7 MCP Tools - Legal AI Development</title>
  <meta name="description" content="Test and explore Context7 MCP tools for stack analysis and best practices" />
</svelte:head>

<main class="min-h-screen bg-gray-50">
  <div class="max-w-7xl mx-auto py-8">
    <div class="mb-8 text-center">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">
        Context7 MCP Tools Integration
      </h1>
      <p class="text-lg text-gray-600 max-w-3xl mx-auto">
        Comprehensive toolset for SvelteKit legal AI development with stack analysis, 
        best practices generation, and intelligent integration suggestions.
      </p>
    </div>

    <MCPToolsDemo />

    <div class="mt-12 bg-white border border-gray-200 rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-4">Integration Status</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div class="space-y-3">
          <h3 class="font-medium text-gray-900">🔧 MCP Server</h3>
          <ul class="space-y-1 text-sm text-gray-600">
            <li>✅ Context7 server configured</li>
            <li>✅ vLLM endpoint integration</li>
            <li>✅ Stack analysis tools</li>
            <li>✅ Best practices generation</li>
            <li>✅ Integration suggestions</li>
          </ul>
        </div>

        <div class="space-y-3">
          <h3 class="font-medium text-gray-900">🤖 AI Systems</h3>
          <ul class="space-y-1 text-sm text-gray-600">
            <li>✅ Autogen multi-agent framework</li>
            <li>✅ CrewAI workflow orchestration</li>
            <li>✅ vLLM high-performance serving</li>
            <li>✅ Ollama GPU optimization</li>
            <li>✅ Gemma3 Legal model integration</li>
          </ul>
        </div>

        <div class="space-y-3">
          <h3 class="font-medium text-gray-900">📚 Documentation</h3>
          <ul class="space-y-1 text-sm text-gray-600">
            <li>✅ SvelteKit patterns and routing</li>
            <li>✅ Drizzle ORM best practices</li>
            <li>✅ UnoCSS utility optimization</li>
            <li>✅ Bits UI component integration</li>
            <li>✅ Legal AI security guidelines</li>
          </ul>
        </div>
      </div>

      <div class="mt-6 p-4 bg-blue-50 rounded-md">
        <h4 class="font-medium text-blue-900 mb-2">Available Tools</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <strong>analyze-stack:</strong> Context-aware analysis for any stack component
          </div>
          <div>
            <strong>generate-best-practices:</strong> Security, performance, and UI/UX guidelines
          </div>
          <div>
            <strong>suggest-integration:</strong> Integration patterns for new features
          </div>
          <div>
            <strong>resolve-library-id:</strong> Find correct library documentation IDs
          </div>
          <div>
            <strong>get-library-docs:</strong> Access specific documentation topics
          </div>
        </div>
      </div>
    </div>

    <div class="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold text-yellow-800 mb-2">Next Steps</h3>
      <ol class="list-decimal list-inside space-y-2 text-yellow-700">
        <li>Run <code class="bg-yellow-100 px-2 py-1 rounded">TEST-MCP-SERVER.bat</code> to verify all integrations</li>
        <li>Start Ollama server: <code class="bg-yellow-100 px-2 py-1 rounded">ollama serve</code></li>
        <li>Configure Claude Desktop with <code class="bg-yellow-100 px-2 py-1 rounded">SETUP-CLAUDE-MCP-CONTEXT7.bat</code></li>
        <li>Test MCP tools in Claude Desktop or VS Code with Context7 integration</li>
        <li>Use tools for legal AI development guidance and best practices</li>
      </ol>
    </div>
  </div>
</main>