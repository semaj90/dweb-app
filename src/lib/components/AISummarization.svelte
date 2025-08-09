<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
<script>
// @ts-nocheck
  import { Button } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Textarea } from 'bits-ui';
  import { Progress } from 'bits-ui';
  import { Tabs } from 'bits-ui';
  import { createEventDispatcher } from 'svelte';
  import { fly, fade } from 'svelte/transition';
  import { langchain } from '$lib/ai/langchain.js';
  import { multiLayerCache } from '$lib/cache/multi-layer-cache.js';
  
  // Props
  let { 
    documentContent = '',
    documentTitle = '',
    documentId = '',
    autoSummarize = false,
    showAnalysisTools = true,
    class: className = ''
  } = $props();

  // State
  let isProcessing = $state(false);
  let processingStage = $state('');
  let processingProgress = $state(0);
  let summaryResult = $state<{
    summary: string;
    keyPoints: string[];
    entities: Array<{ type: string; value: string; confidence: number }>;
    risks: Array<{ type: string; severity: string; description: string }>;
    confidence: number;
    processingTime: number;
  } | null>(null);
  
  let analysisMode = $state<'summary' | 'analysis' | 'comparison' | 'extraction'>('summary');
  let customPrompt = $state('');
  let customResult = $state('');
  let comparisonDocument = $state('');
  let extractionTemplate = $state('');

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    summarized: { result: typeof summaryResult };
    analyzed: { type: string; result: any };
    error: { error: string };
  }>();

  /**
   * Generate AI summary with comprehensive analysis
   */
  async function generateSummary() {
    if (!documentContent.trim() || isProcessing) return;

    isProcessing = true;
    processingStage = 'Initializing AI analysis...';
    processingProgress = 10;
    
    const startTime = Date.now();

    try {
      // Check cache first
      const cacheKey = `summary:${documentId || hashContent(documentContent)}`;
      const cached = await multiLayerCache.get(cacheKey);
      
      if (cached) {
        summaryResult = cached;
        dispatch('summarized', { result: cached });
        isProcessing = false;
        return;
      }

      // Stage 1: Generate main summary
      processingStage = 'Generating document summary...';
      processingProgress = 25;
      
      const summary = await langchain.summarizeDocument(documentContent, {
        type: 'map_reduce',
        maxTokens: 1000
      });

      // Stage 2: Extract key points
      processingStage = 'Extracting key points...';
      processingProgress = 50;
      
      const keyPointsExtraction = `
        Extract the 5-10 most important key points from this legal document.
        Focus on:
        1. Main legal obligations and rights
        2. Important dates and deadlines
        3. Financial terms and amounts
        4. Key parties involved
        5. Critical clauses or conditions
        
        Return as a JSON array of strings.
      `;
      
      const keyPointsResult = await langchain.extractInfo(documentContent, keyPointsExtraction);
      const keyPoints = Array.isArray(keyPointsResult) ? keyPointsResult : [];

      // Stage 3: Entity extraction
      processingStage = 'Identifying entities...';
      processingProgress = 70;
      
      const entitiesExtraction = `
        Extract all legal entities from this document including:
        - Person names (parties, witnesses, attorneys)
        - Organization names (companies, courts, agencies)
        - Legal terms and concepts
        - Dates and deadlines
        - Monetary amounts
        - Locations and jurisdictions
        
        Return as JSON array: [{"type": "person", "value": "John Doe", "confidence": 0.95}]
      `;
      
      const entitiesResult = await langchain.extractInfo(documentContent, entitiesExtraction);
      const entities = Array.isArray(entitiesResult) ? entitiesResult : [];

      // Stage 4: Risk analysis
      processingStage = 'Analyzing legal risks...';
      processingProgress = 85;
      
      const riskAnalysis = `
        Analyze this legal document for potential risks and concerns:
        1. Compliance risks
        2. Financial risks
        3. Operational risks
        4. Legal liability risks
        5. Contractual risks
        
        Return as JSON array: [{"type": "compliance", "severity": "high", "description": "Missing required disclosure"}]
      `;
      
      const risksResult = await langchain.extractInfo(documentContent, riskAnalysis);
      const risks = Array.isArray(risksResult) ? risksResult : [];

      // Stage 5: Calculate confidence score
      processingStage = 'Finalizing analysis...';
      processingProgress = 95;
      
      const confidence = calculateConfidenceScore(summary, keyPoints, entities, risks);
      const processingTime = Date.now() - startTime;

      // Compile final result
      const result = {
        summary,
        keyPoints,
        entities,
        risks,
        confidence,
        processingTime
      };

      summaryResult = result;
      processingProgress = 100;

      // Cache the result
      await multiLayerCache.set(cacheKey, result, {
        type: 'analysis',
        ttl: 3600,
        tags: ['summary', 'analysis', 'legal']
      });

      dispatch('summarized', { result });

    } catch (error) {
      console.error('Summarization failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Summarization failed';
      dispatch('error', { error: errorMessage });
    } finally {
      isProcessing = false;
      processingStage = '';
      processingProgress = 0;
    }
  }

  /**
   * Perform custom analysis with user prompt
   */
  async function performCustomAnalysis() {
    if (!customPrompt.trim() || !documentContent.trim() || isProcessing) return;

    isProcessing = true;
    processingStage = 'Processing custom analysis...';

    try {
      const result = await langchain.extractInfo(documentContent, customPrompt);
      customResult = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
      
      dispatch('analyzed', { type: 'custom', result: customResult });

    } catch (error) {
      console.error('Custom analysis failed:', error);
      dispatch('error', { error: error instanceof Error ? error.message : 'Analysis failed' });
    } finally {
      isProcessing = false;
      processingStage = '';
    }
  }

  /**
   * Compare with another document
   */
  async function performComparison() {
    if (!comparisonDocument.trim() || !documentContent.trim() || isProcessing) return;

    isProcessing = true;
    processingStage = 'Comparing documents...';

    try {
      const comparison = await langchain.compareDocuments(
        documentContent,
        comparisonDocument
      );
      
      dispatch('analyzed', { type: 'comparison', result: comparison });

    } catch (error) {
      console.error('Document comparison failed:', error);
      dispatch('error', { error: error instanceof Error ? error.message : 'Comparison failed' });
    } finally {
      isProcessing = false;
      processingStage = '';
    }
  }

  /**
   * Extract specific information using template
   */
  async function performExtraction() {
    if (!extractionTemplate.trim() || !documentContent.trim() || isProcessing) return;

    isProcessing = true;
    processingStage = 'Extracting information...';

    try {
      const extraction = await langchain.extractInfo(documentContent, extractionTemplate);
      
      dispatch('analyzed', { type: 'extraction', result: extraction });

    } catch (error) {
      console.error('Information extraction failed:', error);
      dispatch('error', { error: error instanceof Error ? error.message : 'Extraction failed' });
    } finally {
      isProcessing = false;
      processingStage = '';
    }
  }

  /**
   * Clear all results
   */
  function clearResults() {
    summaryResult = null;
    customResult = '';
    customPrompt = '';
    comparisonDocument = '';
    extractionTemplate = '';
  }

  /**
   * Auto-summarize on mount if enabled
   */
  $effect(() => {
    if (autoSummarize && documentContent && !summaryResult && !isProcessing) {
      generateSummary();
    }
  });

  // Helper functions
  function calculateConfidenceScore(
    summary: string, 
    keyPoints: string[], 
    entities: any[], 
    risks: any[]
  ): number {
    let score = 0.5; // Base score
    
    if (summary && summary.length > 100) score += 0.15;
    if (keyPoints && keyPoints.length >= 3) score += 0.15;
    if (entities && entities.length >= 5) score += 0.1;
    if (risks && risks.length > 0) score += 0.1;
    
    return Math.min(score, 1.0);
  }

  function hashContent(content: string): string {
    return btoa(content.substring(0, 100)).replace(/[/+]/g, '_').substring(0, 20);
  }

  function formatProcessingTime(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function getRiskSeverityColor(severity: string): string {
    switch (severity.toLowerCase()) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  function getEntityTypeColor(type: string): string {
    const colors: Record<string, string> = {
      person: 'bg-blue-100 text-blue-800',
      organization: 'bg-purple-100 text-purple-800',
      location: 'bg-green-100 text-green-800',
      date: 'bg-orange-100 text-orange-800',
      money: 'bg-yellow-100 text-yellow-800',
      legal: 'bg-indigo-100 text-indigo-800'
    };
    return colors[type.toLowerCase()] || 'bg-gray-100 text-gray-800';
  }

  // Pre-defined analysis templates
  const analysisTemplates = {
    contractAnalysis: `
      Analyze this contract and provide:
      1. Contract type and purpose
      2. Key parties and their roles
      3. Main obligations for each party
      4. Payment terms and amounts
      5. Important dates and deadlines
      6. Termination conditions
      7. Risk factors and concerns
    `,
    complianceCheck: `
      Review this document for compliance issues:
      1. Regulatory compliance requirements
      2. Industry standard adherence
      3. Legal requirement fulfillment
      4. Missing required clauses or disclosures
      5. Potential compliance risks
    `,
    riskAssessment: `
      Conduct a comprehensive risk assessment:
      1. Financial risks and exposure
      2. Legal liability risks
      3. Operational risks
      4. Compliance risks
      5. Reputational risks
      6. Mitigation recommendations
    `
  };
</script>

<div class={`ai-summarization ${className}`}>
  <Card.Root class="overflow-hidden">
    <!-- Header -->
    <div class="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-b">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-2xl font-bold text-gray-900">AI Document Analysis</h2>
          {#if documentTitle}
            <p class="text-sm text-gray-600 mt-1">{documentTitle}</p>
          {/if}
        </div>
        <div class="flex space-x-2">
          <Button.Root
            on:click={generateSummary}
            disabled={!documentContent.trim() || isProcessing}
            class="bg-blue-600 hover:bg-blue-700"
          >
            {isProcessing ? 'Processing...' : 'Analyze Document'}
          </Button.Root>
          {#if summaryResult || customResult}
            <Button.Root
              variant="outline"
              on:click={clearResults}
              class="border-gray-300"
            >
              Clear Results
            </Button.Root>
          {/if}
        </div>
      </div>

      <!-- Processing Progress -->
      {#if isProcessing}
        <div class="mt-4" transition:fly={{ y: -20 }}>
          <div class="flex items-center justify-between mb-2">
            <p class="text-sm text-gray-700">{processingStage}</p>
            <p class="text-sm text-gray-500">{processingProgress}%</p>
          </div>
          <Progress.Root value={processingProgress} max={100} class="h-2">
            <Progress.Indicator class="bg-blue-600 h-full rounded-full transition-all duration-500" />
          </Progress.Root>
        </div>
      {/if}
    </div>

    <div class="p-6">
      {#if showAnalysisTools}
        <!-- Analysis Mode Tabs -->
        <Tabs.Root bind:value={analysisMode} class="mb-6">
          <Tabs.List class="grid w-full grid-cols-4">
            <Tabs.Trigger value="summary">Summary</Tabs.Trigger>
            <Tabs.Trigger value="analysis">Custom Analysis</Tabs.Trigger>
            <Tabs.Trigger value="comparison">Comparison</Tabs.Trigger>
            <Tabs.Trigger value="extraction">Extraction</Tabs.Trigger>
          </Tabs.List>

          <!-- Summary Tab -->
          <Tabs.Content value="summary" class="mt-6">
            {#if summaryResult}
              <div class="space-y-6" transition:fade>
                <!-- Summary Overview -->
                <div class="bg-gray-50 rounded-lg p-4">
                  <div class="flex items-center justify-between mb-3">
                    <h3 class="text-lg font-semibold text-gray-900">Document Summary</h3>
                    <div class="flex items-center space-x-3">
                      <Badge.Root class="bg-green-100 text-green-800">
                        {Math.round(summaryResult.confidence * 100)}% confidence
                      </Badge.Root>
                      <span class="text-sm text-gray-500">
                        {formatProcessingTime(summaryResult.processingTime)}
                      </span>
                    </div>
                  </div>
                  <p class="text-gray-700 leading-relaxed">{summaryResult.summary}</p>
                </div>

                <!-- Key Points -->
                {#if summaryResult.keyPoints.length > 0}
                  <div>
                    <h3 class="text-lg font-semibold text-gray-900 mb-3">Key Points</h3>
                    <ul class="space-y-2">
                      {#each summaryResult.keyPoints as point}
                        <li class="flex items-start">
                          <div class="flex-shrink-0 h-2 w-2 bg-blue-600 rounded-full mt-2 mr-3"></div>
                          <span class="text-gray-700">{point}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                {/if}

                <!-- Entities -->
                {#if summaryResult.entities.length > 0}
                  <div>
                    <h3 class="text-lg font-semibold text-gray-900 mb-3">Identified Entities</h3>
                    <div class="flex flex-wrap gap-2">
                      {#each summaryResult.entities as entity}
                        <Badge.Root class={getEntityTypeColor(entity.type)}>
                          {entity.value}
                          <span class="ml-1 text-xs opacity-75">
                            ({Math.round(entity.confidence * 100)}%)
                          </span>
                        </Badge.Root>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Risks -->
                {#if summaryResult.risks.length > 0}
                  <div>
                    <h3 class="text-lg font-semibold text-gray-900 mb-3">Risk Analysis</h3>
                    <div class="space-y-3">
                      {#each summaryResult.risks as risk}
                        <div class="border border-gray-200 rounded-lg p-4">
                          <div class="flex items-start justify-between">
                            <div class="flex-1">
                              <div class="flex items-center space-x-2 mb-2">
                                <Badge.Root class={getRiskSeverityColor(risk.severity)}>
                                  {risk.severity} risk
                                </Badge.Root>
                                <span class="text-sm font-medium text-gray-900">
                                  {risk.type}
                                </span>
                              </div>
                              <p class="text-sm text-gray-700">{risk.description}</p>
                            </div>
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>
            {:else if !isProcessing}
              <div class="text-center py-8 text-gray-500">
                <svg class="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.462-.881-6.065-2.325" />
                </svg>
                <p>Click "Analyze Document" to generate an AI summary</p>
              </div>
            {/if}
          </Tabs.Content>

          <!-- Custom Analysis Tab -->
          <Tabs.Content value="analysis" class="mt-6">
            <div class="space-y-4">
              <!-- Template Selection -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">
                  Analysis Templates
                </label>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto"
                    on:click={() => customPrompt = analysisTemplates.contractAnalysis}
                  >
                    <div>
                      <div class="font-medium">Contract Analysis</div>
                      <div class="text-sm text-gray-500">Analyze contract terms and obligations</div>
                    </div>
                  </Button.Root>
                  
                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto"
                    on:click={() => customPrompt = analysisTemplates.complianceCheck}
                  >
                    <div>
                      <div class="font-medium">Compliance Check</div>
                      <div class="text-sm text-gray-500">Review regulatory compliance</div>
                    </div>
                  </Button.Root>
                  
                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto"
                    on:click={() => customPrompt = analysisTemplates.riskAssessment}
                  >
                    <div>
                      <div class="font-medium">Risk Assessment</div>
                      <div class="text-sm text-gray-500">Comprehensive risk analysis</div>
                    </div>
                  </Button.Root>
                </div>
              </div>

              <!-- Custom Prompt -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">
                  Custom Analysis Prompt
                </label>
                <Textarea.Root
                  bind:value={customPrompt}
                  placeholder="Enter your custom analysis prompt here..."
                  class="min-h-[120px]"
                />
              </div>

              <Button.Root
                on:click={performCustomAnalysis}
                disabled={!customPrompt.trim() || isProcessing}
                class="w-full"
              >
                Run Analysis
              </Button.Root>

              {#if customResult}
                <div class="mt-6 bg-gray-50 rounded-lg p-4">
                  <h3 class="text-lg font-semibold text-gray-900 mb-3">Analysis Result</h3>
                  <pre class="whitespace-pre-wrap text-sm text-gray-700 font-mono">{customResult}</pre>
                </div>
              {/if}
            </div>
          </Tabs.Content>

          <!-- Comparison Tab -->
          <Tabs.Content value="comparison" class="mt-6">
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">
                  Document to Compare
                </label>
                <Textarea.Root
                  bind:value={comparisonDocument}
                  placeholder="Paste the second document content here for comparison..."
                  class="min-h-[200px]"
                />
              </div>

              <Button.Root
                on:click={performComparison}
                disabled={!comparisonDocument.trim() || isProcessing}
                class="w-full"
              >
                Compare Documents
              </Button.Root>
            </div>
          </Tabs.Content>

          <!-- Extraction Tab -->
          <Tabs.Content value="extraction" class="mt-6">
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">
                  Information to Extract
                </label>
                <Textarea.Root
                  bind:value={extractionTemplate}
                  placeholder="Specify what information you want to extract (e.g., 'Extract all dates, monetary amounts, and party names')"
                  class="min-h-[120px]"
                />
              </div>

              <Button.Root
                on:click={performExtraction}
                disabled={!extractionTemplate.trim() || isProcessing}
                class="w-full"
              >
                Extract Information
              </Button.Root>
            </div>
          </Tabs.Content>
        </Tabs.Root>
      {:else if summaryResult}
        <!-- Simple summary view when tools are hidden -->
        <div class="space-y-4">
          <div class="bg-gray-50 rounded-lg p-4">
            <h3 class="text-lg font-semibold text-gray-900 mb-3">Summary</h3>
            <p class="text-gray-700">{summaryResult.summary}</p>
          </div>
          
          {#if summaryResult.keyPoints.length > 0}
            <div>
              <h3 class="text-lg font-semibold text-gray-900 mb-3">Key Points</h3>
              <ul class="space-y-1">
                {#each summaryResult.keyPoints as point}
                  <li class="text-gray-700">• {point}</li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </Card.Root>
</div>

<style>
  .ai-summarization {
    @apply max-w-5xl mx-auto;
  }
</style>