// Context7 MCP Integration Demo Component
// Demonstrates production Context7 MCP workflow with Copilot orchestration

<script lang="ts">
  import { writable } from 'svelte/store';
  import { copilotOrchestrator } from '$lib/utils/mcp-helpers';
  import { resolveLibraryId, getLibraryDocs, semanticSearch } from '$lib/ai/mcp-helpers';
  import { getEnhancedContext, copilotSelfPrompt } from '$lib/utils/copilot-self-prompt';
  import type { PageData } from './$types';

  interface Props {
    data: PageData
  }
  let {
    data
  } = $props();

let query = $state('');
let results = writable([]);
let loading = writable(false);
let orchestrationResult = writable(data.orchestrationResult);
let enhancedContext = writable(null);

// Display SSR-loaded Copilot context
let copilotArchitecture = $derived(data.copilotContext?.architecture)
let legalContext = $derived(data.copilotContext?.legalContext)
let libraryData = $derived(data.libraryData)

// Demo: Full Context7 MCP workflow with Copilot architecture integration
async function runFullWorkflow() {
  loading.set(true);

  try {
    // Step 1: Resolve SvelteKit library ID using Context7
    const libId = await resolveLibraryId('SvelteKit');
    console.log('Resolved library ID:', libId);

    // Step 2: Get documentation for routing topic
    const docs = await getLibraryDocs(libId, 'routing');
    console.log('Library docs:', docs.substring(0, 200) + '...');

    // Step 3: Perform semantic search
    const searchResults = await semanticSearch(query || 'SvelteKit legal AI integration');
    console.log('Search results:', searchResults);

    // Step 4: Get enhanced context with Redis cache
    const context = await getEnhancedContext(query || 'legal workflow automation');
    enhancedContext.set(context);

    // Step 5: Run full Copilot orchestration with architecture context
    const orchestration = await copilotOrchestrator(
      query || 'Analyze legal AI workflow with SvelteKit',
      {
        useSemanticSearch: true,
        useMemory: true,
        useCodebase: true,
        useMultiAgent: true,
        logErrors: true,
        synthesizeOutputs: true,
        context: {
          libId,
          docs: docs.substring(0, 500),
          copilotArchitecture,
          legalContext
        }
      }
    );

    orchestrationResult.set(orchestration);
    results.set([
      { type: 'orchestration', data: orchestration.selfPrompt },
      { type: 'library', data: { libId, docs: docs.substring(0, 300) } },
      { type: 'search', data: searchResults }
    ]);

  } catch (error) {
    console.error('Workflow error:', error);
    results.set([{ type: 'error', data: error.message }]);
  } finally {
    loading.set(false);
  }
}

// Enhanced: Run Copilot self-prompt with architecture context
async function runCopilotSelfPrompt() {
  loading.set(true);

  try {
    const result = await copilotSelfPrompt(
      query || 'Implement legal AI case management with SvelteKit',
      {
        useSemanticSearch: true,
        useMemory: true,
        useMultiAgent: true,
        useAutonomousEngineering: true,
        enableSelfSynthesis: true,
        context: {
          projectPath: '/legal-ai-cms',
          platform: 'webapp',
          urgency: 'medium',
          includeTests: true
        }
      }
    );

    orchestrationResult.set(result);
    console.log('Copilot self-prompt result:', result);

  } catch (error) {
    console.error('Copilot self-prompt error:', error);
  } finally {
    loading.set(false);
  }
}

// Demo: Quick library resolution
async function quickResolve() {
  try {
    const libId = await resolveLibraryId(query);
    const docs = await getLibraryDocs(libId, 'overview');
    results.set([{ type: 'library', data: { id: libId, docs } }]);
  } catch (error) {
    results.set([{ type: 'error', data: error.message }]);
  }
}

// Demo: Quick semantic search
async function quickSearch() {
  try {
    const searchResults = await semanticSearch(query);
    results.set([{ type: 'search', data: searchResults }]);
  } catch (error) {
    results.set([{ type: 'error', data: error.message }]);
  }
}
</script>

<main class="container mx-auto p-6 max-w-6xl">
  <h1 class="text-3xl font-bold mb-6">Context7 MCP + Copilot Architecture Demo</h1>

  <!-- SSR Data Display -->
  <section class="bg-blue-50 p-4 rounded-lg mb-6">
    <h2 class="text-xl font-semibold mb-3">SSR-Loaded Data</h2>
    {#if libraryData}
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 class="font-medium">Library: {libraryData.library}</h3>
          <p class="text-sm text-gray-600">ID: {libraryData.libId}</p>
          <p class="text-sm text-gray-600">Topic: {libraryData.topic}</p>
        </div>
        <div>
          <h3 class="font-medium">Documentation Preview</h3>
          <p class="text-xs text-gray-700 bg-white p-2 rounded">
            {libraryData.docs?.substring(0, 200)}...
          </p>
        </div>
      </div>
    {/if}
  </section>

  <!-- Copilot Architecture Context -->
  <section class="bg-green-50 p-4 rounded-lg mb-6">
    <h2 class="text-xl font-semibold mb-3">Copilot Architecture Context</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      {#if copilotArchitecture}
        <div>
          <h3 class="font-medium">Architecture Summary</h3>
          <pre class="text-xs bg-white p-2 rounded overflow-auto max-h-40">{copilotArchitecture}</pre>
        </div>
      {/if}
      {#if legalContext}
        <div>
          <h3 class="font-medium">Legal Context</h3>
          <pre class="text-xs bg-white p-2 rounded overflow-auto max-h-40">{legalContext}</pre>
        </div>
      {/if}
    </div>
  </section>

  <!-- Interactive Demo -->
  <section class="bg-gray-50 p-4 rounded-lg mb-6">
    <h2 class="text-xl font-semibold mb-3">Interactive Workflow</h2>

    <input
      type="text"
      bind:value={query}
      placeholder="Enter your legal AI query..."
      class="w-full p-3 border rounded-lg mb-4"
    />

    <div class="flex gap-3 mb-4">
      <button
        onclick={runFullWorkflow}
        disabled={$loading}
        class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {$loading ? 'Running...' : 'Run Full Workflow'}
      </button>

      <button
        onclick={runCopilotSelfPrompt}
        disabled={$loading}
        class="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50"
      >
        Run Copilot Self-Prompt
      </button>

      <a
        href="?orchestrate=true&prompt={encodeURIComponent(query || 'Analyze legal AI workflow')}"
        class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 inline-block text-center"
      >
        Orchestrate Analysis
      </a>
    </div>
  </section>
</main>
