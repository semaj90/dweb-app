<script lang="ts">
  import { onMount } from 'svelte';
  import type { AIResponse } from '$lib/types/ai';

  let cases: any[] = $state([]);
  let evidence: any[] = $state([]);
  let loading = $state(true);
  let aiResponse: AIResponse | null = $state(null);
  let chatMessage = $state('');
  let selectedCase = $state<string | null>(null);

  async function loadCases() {
    try {
      const response = await fetch('/api/cases');
      const data = await response.json();
      cases = data.cases || [];
    } catch (error) {
      console.error('Failed to load cases:', error);
    }
  }

  async function loadEvidence(caseId?: string) {
    try {
      const url = caseId ? `/api/evidence?caseId=${caseId}` : '/api/evidence';
      const response = await fetch(url);
      const data = await response.json();
      evidence = data.evidence || [];
    } catch (error) {
      console.error('Failed to load evidence:', error);
    }
  }

  async function sendAIMessage() {
    if (!chatMessage.trim()) return;

    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: chatMessage,
          caseId: selectedCase,
          context: 'dashboard'
        })
      });

      const data = await response.json();
      aiResponse = data.aiMetadata;
      chatMessage = '';
    } catch (error) {
      console.error('AI request failed:', error);
    }
  }

  function selectCase(caseId: string) {
    selectedCase = caseId;
    loadEvidence(caseId);
  }

  onMount(async () => {
    await Promise.all([loadCases(), loadEvidence()]);
    loading = false;
  });
</script>

<svelte:head>
  <title>Legal AI Dashboard - YoRHa Legal System</title>
</svelte:head>

<div class="min-h-screen p-6 bg-gradient-to-br from-gray-900 via-gray-800 to-black">
  <!-- Header -->
  <div class="mb-8">
    <h1 class="text-4xl font-mono font-bold gradient-text-primary mb-2">
      Legal AI Dashboard
    </h1>
    <p class="text-gray-400 font-mono">
      Production Legal Case Management System with AI Integration
    </p>
  </div>

  {#if loading}
    <div class="flex items-center justify-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-400"></div>
    </div>
  {:else}
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Cases Panel -->
      <div class="lg:col-span-1">
        <div class="yorha-card p-6 h-fit">
          <h2 class="text-xl font-mono font-semibold text-white mb-4">Active Cases</h2>
          <div class="space-y-3">
            {#each cases as caseItem}
              <button
                class="w-full text-left p-3 rounded border transition-all duration-200 {selectedCase === caseItem.id ? 'border-yellow-400 bg-yellow-400/10' : 'border-gray-600 hover:border-gray-500'}"
                onclick={() => selectCase(caseItem.id)}
              >
                <div class="font-mono text-sm text-yellow-400">{caseItem.id}</div>
                <div class="font-semibold text-white text-sm">{caseItem.title}</div>
                <div class="text-xs text-gray-400 mt-1">
                  Status: {caseItem.status} • Priority: {caseItem.priority}
                </div>
                <div class="text-xs text-gray-500 mt-1">
                  Evidence: {caseItem.evidenceCount} • Docs: {caseItem.documentsCount}
                </div>
              </button>
            {/each}
          </div>
        </div>
      </div>

      <!-- Main Content -->
      <div class="lg:col-span-2 space-y-6">
        <!-- AI Chat Interface -->
        <div class="yorha-card p-6">
          <h2 class="text-xl font-mono font-semibold text-white mb-4">
            Legal AI Assistant
          </h2>

          <div class="flex gap-3 mb-4">
            <input
              bind:value={chatMessage}
              placeholder="Ask about legal matters, case analysis, or precedents..."
              class="flex-1 yorha-input px-4 py-2 rounded"
              onkeydown={(e) => e.key === 'Enter' && sendAIMessage()}
            />
            <button
              onclick={sendAIMessage}
              class="yorha-btn yorha-btn-primary px-6 py-2 rounded"
              disabled={!chatMessage.trim()}
            >
              Send
            </button>
          </div>

          {#if aiResponse}
            <div class="mt-4 p-4 bg-gray-800/50 rounded border border-gray-700">
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div class="text-gray-400">Confidence</div>
                  <div class="text-yellow-400 font-mono">
                    {(aiResponse.confidence ?? 0 * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div class="text-gray-400">Processing</div>
                  <div class="text-yellow-400 font-mono">
                    {aiResponse.processingTime?.toFixed(0)}ms
                  </div>
                </div>
                <div>
                  <div class="text-gray-400">GPU Accelerated</div>
                  <div class="text-yellow-400 font-mono">
                    {aiResponse.gpuProcessed ? 'Yes' : 'No'}
                  </div>
                </div>
                <div>
                  <div class="text-gray-400">Legal Risk</div>
                  <div class="text-yellow-400 font-mono capitalize">
                    {aiResponse.legalRisk}
                  </div>
                </div>
              </div>

              {#if aiResponse.keyTerms && aiResponse.keyTerms.length > 0}
                <div class="mt-3">
                  <div class="text-gray-400 text-sm mb-2">Key Legal Terms:</div>
                  <div class="flex flex-wrap gap-2">
                    {#each aiResponse.keyTerms as term}
                      <span class="px-2 py-1 bg-yellow-400/20 text-yellow-400 rounded text-xs font-mono">
                        {term}
                      </span>
                    {/each}
                  </div>
                </div>
              {/if}
            </div>
          {/if}
        </div>

        <!-- Evidence Panel -->
        <div class="yorha-card p-6">
          <h2 class="text-xl font-mono font-semibold text-white mb-4">
            Evidence & Documents
            {#if selectedCase}
              <span class="text-sm text-gray-400">for {selectedCase}</span>
            {/if}
          </h2>

          <div class="space-y-3">
            {#each evidence as item}
              <div class="p-4 bg-gray-800/30 rounded border border-gray-700 hover:border-gray-600 transition-colors">
                <div class="flex justify-between items-start mb-2">
                  <div>
                    <div class="font-semibold text-white">{item.title}</div>
                    <div class="text-sm text-gray-400">{item.type} • {item.fileSize}</div>
                  </div>
                  <div class="text-xs text-gray-500">
                    {new Date(item.uploadedAt).toLocaleDateString()}
                  </div>
                </div>

                <div class="text-sm text-gray-300 mb-2 line-clamp-2">
                  {item.content}
                </div>

                <div class="flex flex-wrap gap-1">
                  {#each item.tags as tag}
                    <span class="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs">
                      {tag}
                    </span>
                  {/each}
                </div>
              </div>
            {/each}

            {#if evidence.length === 0}
              <div class="text-center py-8 text-gray-500">
                {selectedCase ? 'No evidence found for this case' : 'Select a case to view evidence'}
              </div>
            {/if}
          </div>
        </div>
      </div>
    </div>

    <!-- System Status -->
    <div class="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="yorha-card p-4 text-center">
        <div class="text-2xl font-mono font-bold text-yellow-400">{cases.length}</div>
        <div class="text-sm text-gray-400">Active Cases</div>
      </div>
      <div class="yorha-card p-4 text-center">
        <div class="text-2xl font-mono font-bold text-green-400">{evidence.length}</div>
        <div class="text-sm text-gray-400">Evidence Items</div>
      </div>
      <div class="yorha-card p-4 text-center">
        <div class="text-2xl font-mono font-bold text-blue-400">Gemma3</div>
        <div class="text-sm text-gray-400">AI Model</div>
      </div>
      <div class="yorha-card p-4 text-center">
        <div class="text-2xl font-mono font-bold text-purple-400">Ready</div>
        <div class="text-sm text-gray-400">System Status</div>
      </div>
    </div>
  {/if}
</div>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical
    overflow: hidden
  }
</style>