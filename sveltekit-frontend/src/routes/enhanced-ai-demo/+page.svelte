<!-- ====================================================================== -->
<!-- ENHANCED RAG STUDIO - WORKING VERSION WITH SHADCN COMPONENTS -->
<!-- Using Svelte 5 compatible components only -->
<!-- ====================================================================== -->

<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import Card from '$lib/components/ui/Card.svelte';
  import CardContent from '$lib/components/ui/CardContent.svelte';
  import CardHeader from '$lib/components/ui/CardHeader.svelte';
  import CardTitle from '$lib/components/ui/CardTitle.svelte';
  import { Badge } from '$lib/components/ui/badge';
  import Input from '$lib/components/ui/Input.svelte';

  let systemInitialized = $state(false);
  let testQuery = $state('');
  let selectedModel = $state('claude-3-5-sonnet');
  let processingStatus = $state('idle');
  let analysisResults = $state(null);

  const modelOptions = [
    { value: 'claude-3-5-sonnet', label: 'Claude 3.5 Sonnet' },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
    { value: 'gemma-3-27b', label: 'Local Gemma 3 27B' }
  ];

  onMount(async () => {
    // Simulate system initialization
    for (let i = 0; i <= 100; i += 20) {
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    systemInitialized = true;
  });

  function handleAnalyze() {
    if (!testQuery.trim()) return;

    processingStatus = 'processing';
    analysisResults = null;

    // Simulate AI processing
    setTimeout(() => {
      analysisResults = {
        confidence: 92,
        entities: [
          { text: 'Legal Entity', type: 'Organization', confidence: 0.95 },
          { text: 'Contract Terms', type: 'Legal Concept', confidence: 0.88 },
          { text: 'Liability Clause', type: 'Legal Provision', confidence: 0.91 }
        ],
        recommendations: [
          'Review section 4.2 for potential liability issues',
          'Consider adding indemnification clause',
          'Verify compliance with local regulations'
        ],
        processingTime: '2.3s',
        model: selectedModel
      };
      processingStatus = 'completed';
    }, 2500);
  }

  function handleClear() {
    testQuery = '';
    analysisResults = null;
    processingStatus = 'idle';
  }
</script>

<svelte:head>
  <title>Enhanced RAG Studio - Legal AI System</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
  <!-- Header -->
  <header class="border-b border-slate-700 bg-slate-800/50 backdrop-blur-sm">
    <div class="container mx-auto px-4 py-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Enhanced RAG Studio
          </h1>
          <p class="text-slate-400 mt-2">AI-Powered Legal Document Analysis System</p>
        </div>
        <div class="flex items-center gap-4">
          <Badge class="bg-green-900 text-green-400 border-green-400">
            {systemInitialized ? '✅ Online' : '⏳ Initializing...'}
          </Badge>
          <Badge class="bg-blue-900 text-blue-400 border-blue-400">
            SvelteKit 2 + Svelte 5
          </Badge>
        </div>
      </div>
    </div>
  </header>

  <main class="container mx-auto px-4 py-8 space-y-8">
    <!-- System Status Overview -->
    <Card>
      <CardHeader>
        <CardTitle class="text-green-400">✅ System Status: Operational</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
          <div>
            <div class="text-2xl font-bold text-green-400 mb-2">✅</div>
            <div class="text-sm font-medium">SvelteKit 2</div>
            <div class="text-xs text-slate-400">Working</div>
          </div>
          <div>
            <div class="text-2xl font-bold text-green-400 mb-2">✅</div>
            <div class="text-sm font-medium">Svelte 5 Runes</div>
            <div class="text-xs text-slate-400">Working</div>
          </div>
          <div>
            <div class="text-2xl font-bold text-green-400 mb-2">✅</div>
            <div class="text-sm font-medium">Shadcn UI</div>
            <div class="text-xs text-slate-400">Compatible</div>
          </div>
          <div>
            <div class="text-2xl font-bold text-blue-400 mb-2">🚀</div>
            <div class="text-sm font-medium">Enhanced RAG</div>
            <div class="text-xs text-slate-400">{systemInitialized ? 'Ready' : 'Loading...'}</div>
          </div>
        </div>
      </CardContent>
    </Card>

    <!-- AI Query Interface -->
    <Card>
      <CardHeader>
        <CardTitle class="text-blue-400">AI Legal Analysis Interface</CardTitle>
      </CardHeader>
      <CardContent class="space-y-6">
        <!-- Input Section -->
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium mb-2">Legal Query or Document Content</label>
            <Input
              bind:value={testQuery}
              placeholder="Enter legal question, paste contract text, or describe evidence to analyze..."
              class="w-full h-24 bg-slate-800 border-slate-600 text-white placeholder-slate-400 resize-none"
            />
          </div>

          <div class="flex items-center gap-4">
            <div class="flex-1">
              <label class="block text-sm font-medium mb-2">AI Model</label>
              <select
                bind:value={selectedModel}
                class="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-white"
              >
                {#each modelOptions as option}
                  <option value={option.value}>{option.label}</option>
                {/each}
              </select>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="flex gap-4">
            <Button
              on:click={handleAnalyze}
              disabled={!testQuery.trim() || processingStatus === 'processing' || !systemInitialized}
              class="bg-blue-600 hover:bg-blue-700 text-white"
            >
              {processingStatus === 'processing' ? 'Analyzing...' : 'Analyze with AI'}
            </Button>

            <Button
              variant="outline"
              on:click={handleClear}
              disabled={processingStatus === 'processing'}
            >
              Clear
            </Button>
          </div>
        </div>

        <!-- Processing Status -->
        {#if processingStatus === 'processing'}
          <div class="p-4 bg-slate-800 border border-blue-500/30 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-6 h-6 animate-spin rounded-full border-2 border-blue-400 border-t-transparent"></div>
              <div>
                <div class="text-blue-400 font-semibold">AI Analysis in Progress</div>
                <div class="text-sm text-slate-400">Processing with {selectedModel}...</div>
              </div>
            </div>
          </div>
        {/if}

        <!-- Analysis Results -->
        {#if analysisResults}
          <div class="space-y-4">
            <div class="p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
              <div class="flex items-center justify-between mb-3">
                <div class="text-green-400 font-semibold">Analysis Complete</div>
                <Badge class="bg-green-900 text-green-400 border-green-400">
                  {analysisResults.confidence}% Confidence
                </Badge>
              </div>

              <div class="text-sm text-slate-300 mb-3">
                Processed in {analysisResults.processingTime} using {analysisResults.model}
              </div>

              <!-- Entities -->
              <div class="mb-4">
                <h4 class="font-semibold text-white mb-2">Detected Entities</h4>
                <div class="flex flex-wrap gap-2">
                  {#each analysisResults.entities as entity}
                    <Badge class="bg-purple-900 text-purple-400 border-purple-400">
                      {entity.text} ({Math.round(entity.confidence * 100)}%)
                    </Badge>
                  {/each}
                </div>
              </div>

              <!-- Recommendations -->
              <div>
                <h4 class="font-semibold text-white mb-2">AI Recommendations</h4>
                <ul class="space-y-2">
                  {#each analysisResults.recommendations as rec}
                    <li class="flex items-start gap-2 text-sm text-slate-300">
                      <span class="text-yellow-400 mt-1">•</span>
                      {rec}
                    </li>
                  {/each}
                </ul>
              </div>
            </div>
          </div>
        {/if}
      </CardContent>
    </Card>

    <!-- Architecture & Features -->
    <Card>
      <CardHeader>
        <CardTitle class="text-purple-400">Enhanced RAG Multi-Agent AI Architecture</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
          <div>
            <h4 class="font-semibold text-blue-400 mb-3">Frontend Stack</h4>
            <ul class="space-y-2 text-slate-300">
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-green-400 rounded-full"></span>
                SvelteKit 2 + Svelte 5 Runes
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-green-400 rounded-full"></span>
                Shadcn/UI Components
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-green-400 rounded-full"></span>
                UnoCSS + Tailwind
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-green-400 rounded-full"></span>
                TypeScript Integration
              </li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-purple-400 mb-3">AI & ML Pipeline</h4>
            <ul class="space-y-2 text-slate-300">
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-yellow-400 rounded-full"></span>
                Multi-Agent Orchestration
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-yellow-400 rounded-full"></span>
                Enhanced RAG with PageRank
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-yellow-400 rounded-full"></span>
                Self-Organizing Maps (SOM)
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-yellow-400 rounded-full"></span>
                Context7 MCP Integration
              </li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-green-400 mb-3">Backend Services</h4>
            <ul class="space-y-2 text-slate-300">
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-blue-400 rounded-full"></span>
                Node.js Clustering
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-blue-400 rounded-full"></span>
                7-Layer Caching Architecture
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-blue-400 rounded-full"></span>
                Real-time WebSocket Events
              </li>
              <li class="flex items-center gap-2">
                <span class="w-2 h-2 bg-blue-400 rounded-full"></span>
                Docker Orchestration
              </li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  </main>
</div>