<!-- AI Summary Demo Page -->
<!-- File: sveltekit-frontend/src/routes/ai-summary/+page.svelte -->

<script lang="ts">
  import { onMount } from 'svelte';
  
  let caseData = null;
  let summary = '';
  let isGenerating = false;
  let summaryType = 'prosecution';
  let confidence = 0;
  let ragScore = 0;
  
  const generateSummary = async () => {
    isGenerating = true;
    
    try {
      const res = await fetch('/api/ai/generate-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          caseId: 'demo-case',
          summaryType,
          includeEvidence: true,
          prompt: `Generate ${summaryType} summary with legal analysis`
        })
      });
      
      const result = await res.json();
      
      if (result.success) {
        summary = result.summary;
        confidence = result.metadata?.confidence || 0.85;
        ragScore = result.metadata?.ragScore || 0.82;
      } else {
        summary = `API Error: ${result.error}`;
      }
    } catch (error) {
      summary = `Connection Error: ${error.message}`;
    }
    
    isGenerating = false;
  };

  const loadCaseDemo = async () => {
    caseData = {
      id: 'demo-case',
      title: 'State v. Digital Evidence Analysis',
      evidence: [
        { id: '1', type: 'digital', title: 'Email Communications' },
        { id: '2', type: 'document', title: 'Financial Records' },
        { id: '3', type: 'photo', title: 'Crime Scene Photos' }
      ],
      status: 'active'
    };
  };

  onMount(() => {
    loadCaseDemo();
  });
</script>

<div class="max-w-4xl mx-auto p-6">
  <h1 class="text-3xl font-bold mb-6">AI Summary Generator</h1>
  
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <!-- Case Info -->
    <div class="md:col-span-1">
      <div class="bg-white border rounded-lg p-4">
        <h3 class="font-semibold mb-3">Case Information</h3>
        {#if caseData}
          <div class="space-y-2">
            <p><strong>ID:</strong> {caseData.id}</p>
            <p><strong>Title:</strong> {caseData.title}</p>
            <p><strong>Evidence:</strong> {caseData.evidence.length} items</p>
            <p><strong>Status:</strong> {caseData.status}</p>
          </div>
          
          <div class="mt-4">
            <label for="summary-type" class="block text-sm font-medium mb-2">Summary Type</label>
            <select id="summary-type" bind:value={summaryType} class="w-full p-2 border rounded">
              <option value="prosecution">Prosecution Strategy</option>
              <option value="evidence">Evidence Analysis</option>
              <option value="timeline">Timeline Summary</option>
              <option value="overview">Case Overview</option>
            </select>
          </div>
          
          <button 
            onclick={generateSummary}
            disabled={isGenerating}
            class="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
          >
            {isGenerating ? 'Generating...' : 'Generate AI Summary'}
          </button>
        {:else}
          <p class="text-gray-500">Loading case data...</p>
        {/if}
      </div>
      
      <!-- Metrics -->
      <div class="mt-4 bg-gray-50 p-4 rounded-lg">
        <h4 class="font-medium mb-2">AI Metrics</h4>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span>Confidence:</span>
            <span class="font-mono">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <div class="flex justify-between">
            <span>RAG Score:</span>
            <span class="font-mono">{(ragScore * 100).toFixed(1)}%</span>
          </div>
          <div class="flex justify-between">
            <span>Model:</span>
            <span class="font-mono">gemma3-legal</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Summary Output -->
    <div class="md:col-span-2">
      <div class="bg-white border rounded-lg p-4 h-96">
        <h3 class="font-semibold mb-3">Generated Summary</h3>
        <div class="h-80 overflow-y-auto">
          {#if isGenerating}
            <div class="flex items-center space-x-2">
              <div class="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
              <span>Generating AI summary with Gemma3 Legal...</span>
            </div>
          {:else if summary}
            <div class="prose prose-sm max-w-none">
              <pre class="whitespace-pre-wrap">{summary}</pre>
            </div>
          {:else}
            <div class="text-gray-500">
              Click "Generate AI Summary" to create a {summaryType} summary using local Gemma3 Legal model.
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>
