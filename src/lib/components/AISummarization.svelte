<script lang="ts">
  // @ts-nocheck
  import { Button } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Textarea } from 'bits-ui';
  import { Progress } from 'bits-ui';
  import { Tabs } from 'bits-ui';
  import { createEventDispatcher } from 'svelte';
  import { fly, fade } from 'svelte/transition';
  import { browser } from '$app/environment';

  // Props
  export let documentContent: string = '';
  export let documentTitle: string = '';
  export let documentId: string = '';
  export let autoSummarize: boolean = false;
  export let showAnalysisTools: boolean = true;
  export let className: string = '';

  // State
  let isProcessing = $state(false);
  let processingStage = $state('');
  let processingProgress = $state(0);
  let summaryResult = $state<{
    summary: string;
    keyPoints: string[];
    entities: Array<any>;
    risks: Array<any>;
    confidence: number;
    processingTime: number;
  } | null>(null);

  let analysisMode = $state('summary');
  let customPrompt = $state('');
  let customResult = $state('');
  let comparisonDocument = $state('');
  let extractionTemplate = $state('');

  // Event dispatcher
  const dispatch = createEventDispatcher();

  /**
   * Generate AI summary with comprehensive analysis using pgai extension
   */
  async function generateSummary() {
    if (!documentContent.trim() || isProcessing || !browser) return;

    isProcessing = true;
    processingStage = 'Initializing pgai analysis...';
    processingProgress = 10;

    const startTime = Date.now();

    try {
      // Check if document needs to be saved first
      let docId = documentId;
      if (!docId) {
        processingStage = 'Creating document entry...';
        processingProgress = 15;
        
        // Create document via RAG API
        const createResponse = await fetch('/api/documents', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: documentTitle || 'Untitled Document',
            content: documentContent
          })
        });

        if (!createResponse.ok) {
          throw new Error('Failed to create document entry');
        }

        const createResult = await createResponse.json();
        docId = createResult.data.id;
      }

      // Stage 1: Process document with pgai using local Gemma3 models
      processingStage = 'Processing with Gemma3 models...';
      processingProgress = 30;

      const processResponse = await fetch('/api/rag?action=process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documentId: docId
        })
      });

      if (!processResponse.ok) {
        throw new Error('pgai processing failed');
      }

      const processResult = await processResponse.json();
      
      if (!processResult.success) {
        throw new Error(processResult.error || 'Processing failed');
      }

      // Stage 2: Format results for UI
      processingStage = 'Formatting analysis results...';
      processingProgress = 85;

      const pgaiData = processResult.data;
      
      // Extract structured data from pgai response
      const summary = pgaiData.summary?.summary || 'Summary generation failed';
      const keyPoints = Array.isArray(pgaiData.summary?.key_points) 
        ? pgaiData.summary.key_points 
        : [];
      const entities = formatEntitiesFromPgai(pgaiData.summary?.entities || {});
      const risks = formatRisksFromPgai(pgaiData.summary?.legal_issues || []);
      
      const confidence = calculatePgaiConfidence(pgaiData);
      const processingTime = pgaiData.processing_time_ms || (Date.now() - startTime);

      // Stage 3: Compile final result
      processingStage = 'Finalizing analysis...';
      processingProgress = 95;

      const result = {
        summary,
        keyPoints,
        entities,
        risks,
        confidence,
        processingTime,
        pgaiMetadata: {
          documentId: docId,
          chunksCreated: pgaiData.chunks_created || 0,
          model: 'gemma3-legal',
          riskLevel: pgaiData.summary?.risk_level || 'unknown'
        }
      };

      summaryResult = result;
      processingProgress = 100;

      dispatch('summarized', { result, documentId: docId });

    } catch (error) {
      console.error('pgai Summarization failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Summarization failed';
      dispatch('error', { error: errorMessage });
    } finally {
      isProcessing = false;
      processingStage = '';
      processingProgress = 0;
    }
  }

  /**
   * Perform custom analysis with user prompt using pgai
   */
  async function performCustomAnalysis() {
    if (!customPrompt.trim() || !documentContent.trim() || isProcessing || !browser) return;

    isProcessing = true;
    processingStage = 'Processing custom analysis with Gemma3...';

    try {
      const response = await fetch('/api/rag?action=custom-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: documentContent,
          prompt: customPrompt,
          model: 'gemma3-legal'
        })
      });

      if (!response.ok) {
        throw new Error('Custom analysis request failed');
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Analysis failed');
      }

      customResult = typeof result.data === 'string' ? result.data : JSON.stringify(result.data, null, 2);

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
   * Compare with another document using semantic search
   */
  async function performComparison() {
    if (!comparisonDocument.trim() || !documentContent.trim() || isProcessing || !browser) return;

    isProcessing = true;
    processingStage = 'Comparing documents with semantic analysis...';

    try {
      const response = await fetch('/api/rag?action=document-comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document1: documentContent,
          document2: comparisonDocument,
          model: 'gemma3-legal'
        })
      });

      if (!response.ok) {
        throw new Error('Document comparison failed');
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Comparison failed');
      }

      dispatch('analyzed', { type: 'comparison', result: result.data });

    } catch (error) {
      console.error('Document comparison failed:', error);
      dispatch('error', { error: error instanceof Error ? error.message : 'Comparison failed' });
    } finally {
      isProcessing = false;
      processingStage = '';
    }
  }

  /**
   * Extract specific information using pgai and Gemma3
   */
  async function performExtraction() {
    if (!extractionTemplate.trim() || !documentContent.trim() || isProcessing || !browser) return;

    isProcessing = true;
    processingStage = 'Extracting information with Gemma3...';

    try {
      const response = await fetch('/api/rag?action=extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: documentContent,
          extractionPrompt: extractionTemplate,
          model: 'gemma3-legal'
        })
      });

      if (!response.ok) {
        throw new Error('Information extraction failed');
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Extraction failed');
      }

      dispatch('analyzed', { type: 'extraction', result: result.data });

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

  // New pgai-specific helper functions
  function calculatePgaiConfidence(pgaiData: any): number {
    let score = 0.6; // Higher base score for pgai

    if (pgaiData.summary?.summary && pgaiData.summary.summary.length > 100) score += 0.1;
    if (pgaiData.summary?.key_points && pgaiData.summary.key_points.length >= 3) score += 0.1;
    if (pgaiData.summary?.entities && Object.keys(pgaiData.summary.entities).length > 0) score += 0.1;
    if (pgaiData.chunks_created && pgaiData.chunks_created > 0) score += 0.05;
    if (pgaiData.processing_time_ms && pgaiData.processing_time_ms < 30000) score += 0.05;

    return Math.min(score, 1.0);
  }

  function formatEntitiesFromPgai(entities: any): any[] {
    const formatted = [];
    
    if (entities.persons && Array.isArray(entities.persons)) {
      entities.persons.forEach(person => {
        formatted.push({ type: 'person', value: person, confidence: 0.9 });
      });
    }
    
    if (entities.organizations && Array.isArray(entities.organizations)) {
      entities.organizations.forEach(org => {
        formatted.push({ type: 'organization', value: org, confidence: 0.9 });
      });
    }
    
    if (entities.dates && Array.isArray(entities.dates)) {
      entities.dates.forEach(date => {
        formatted.push({ type: 'date', value: date, confidence: 0.95 });
      });
    }
    
    if (entities.locations && Array.isArray(entities.locations)) {
      entities.locations.forEach(location => {
        formatted.push({ type: 'location', value: location, confidence: 0.85 });
      });
    }

    return formatted;
  }

  function formatRisksFromPgai(legalIssues: string[]): any[] {
    return legalIssues.map(issue => ({
      type: 'legal',
      severity: 'medium', // Default severity, could be enhanced
      description: issue
    }));
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

<div class="ai-summarization {className}">
  <Card.Root class="w-full max-w-5xl mx-auto shadow-lg">
    <Card.Content class="p-6">
      <div class="flex items-center justify-between mb-6">
        <div>
          <h2 class="text-2xl font-bold text-gray-900">🤖 AI Document Analysis (pgai + Gemma3)</h2>
          {#if documentTitle}
            <p class="text-sm font-normal text-gray-600 mt-1">{documentTitle}</p>
          {/if}
        </div>

        <div class="flex gap-3">
          <Button.Root
            onclick={generateSummary}
            disabled={!documentContent.trim() || isProcessing}
            class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md"
          >
            {isProcessing ? 'Processing...' : 'Analyze Document'}
          </Button.Root>

          {#if summaryResult || customResult}
            <Button.Root
              variant="outline"
              onclick={clearResults}
              class="border-gray-300 px-4 py-2 rounded-md"
            >
              Clear Results
            </Button.Root>
          {/if}
        </div>
      </div>

      {#if isProcessing}
        <div class="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200" transition:fade>
          <div class="flex items-center gap-3">
            <div class="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-blue-800">{processingStage}</p>
              <div class="w-full bg-blue-200 rounded-full h-2 mt-2">
                <div 
                  class="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                  style="width: {processingProgress}%"
                ></div>
              </div>
              <p class="text-xs text-blue-600 mt-1">{processingProgress}%</p>
            </div>
          </div>
        </div>
      {/if}

      {#if showAnalysisTools}
        <Tabs.Root bind:value={analysisMode} class="w-full">
          <Tabs.List class="grid w-full grid-cols-4">
            <Tabs.Trigger value="summary" class="text-sm">Summary</Tabs.Trigger>
            <Tabs.Trigger value="custom" class="text-sm">Custom Analysis</Tabs.Trigger>
            <Tabs.Trigger value="comparison" class="text-sm">Comparison</Tabs.Trigger>
            <Tabs.Trigger value="extraction" class="text-sm">Extraction</Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="summary" class="mt-6">
            {#if summaryResult}
              <div class="space-y-6" transition:fly={{ y: 20, duration: 300 }}>
                <div class="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border border-blue-200">
                  <div class="flex items-center justify-between mb-4">
                    <h3 class="text-xl font-semibold text-gray-900">Document Summary</h3>
                    <div class="flex gap-3">
                      <Badge variant="secondary" class="bg-green-100 text-green-800">
                        {Math.round(summaryResult.confidence * 100)}% confidence
                      </Badge>
                      <Badge variant="secondary" class="bg-blue-100 text-blue-800">
                        {formatProcessingTime(summaryResult.processingTime)}
                      </Badge>

                      {#if summaryResult.pgaiMetadata}
                        <Badge variant="secondary" class="bg-purple-100 text-purple-800">
                          Model: {summaryResult.pgaiMetadata.model}
                        </Badge>
                        <Badge variant="secondary" class="bg-yellow-100 text-yellow-800">
                          Chunks: {summaryResult.pgaiMetadata.chunksCreated}
                        </Badge>
                        <Badge variant="secondary" class="bg-red-100 text-red-800">
                          Risk: {summaryResult.pgaiMetadata.riskLevel}
                        </Badge>
                      {/if}
                    </div>
                  </div>

                  <p class="text-gray-700 leading-relaxed">{summaryResult.summary}</p>
                </div>

                {#if summaryResult.keyPoints.length > 0}
                  <div class="bg-white rounded-lg p-6 border border-gray-200">
                    <h4 class="text-lg font-semibold text-gray-900 mb-4">Key Points</h4>
                    <ul class="space-y-2">
                      {#each summaryResult.keyPoints as point}
                        <li class="flex items-start gap-3">
                          <span class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span>
                          <span class="text-gray-700">{point}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                {/if}

                {#if summaryResult.entities.length > 0}
                  <div class="bg-white rounded-lg p-6 border border-gray-200">
                    <h4 class="text-lg font-semibold text-gray-900 mb-4">Identified Entities</h4>
                    <div class="flex flex-wrap gap-2">
                      {#each summaryResult.entities as entity}
                        <Badge class={getEntityTypeColor(entity.type)}>
                          {entity.value}
                          {#if entity.confidence}
                            <span class="ml-1 text-xs opacity-75">
                              ({Math.round(entity.confidence * 100)}%)
                            </span>
                          {/if}
                        </Badge>
                      {/each}
                    </div>
                  </div>
                {/if}

                {#if summaryResult.risks.length > 0}
                  <div class="bg-white rounded-lg p-6 border border-gray-200">
                    <h4 class="text-lg font-semibold text-gray-900 mb-4">Risk Analysis</h4>
                    <div class="space-y-3">
                      {#each summaryResult.risks as risk}
                        <div class="flex items-start gap-4 p-3 bg-gray-50 rounded-lg">
                          <div class="flex-shrink-0">
                            <Badge class={getRiskSeverityColor(risk.severity)}>
                              {risk.severity} risk
                            </Badge>
                            <Badge variant="outline" class="ml-2">
                              {risk.type}
                            </Badge>
                          </div>
                          <p class="text-gray-700 text-sm">{risk.description}</p>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>
            {:else if !isProcessing}
              <div class="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                <div class="w-16 h-16 mx-auto mb-4 bg-blue-100 rounded-full flex items-center justify-center">
                  <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                  </svg>
                </div>
                <p class="text-gray-600">Click "Analyze Document" to generate an AI summary</p>
              </div>
            {/if}
          </Tabs.Content>

          <Tabs.Content value="custom" class="mt-6">
            <div class="space-y-6">
              <div class="bg-white rounded-lg p-6 border border-gray-200">
                <h4 class="text-lg font-semibold text-gray-900 mb-4">Analysis Templates</h4>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto flex flex-col items-start space-y-2"
                    onclick={() => customPrompt = analysisTemplates.contractAnalysis}
                  >
                    <span class="font-medium">Contract Analysis</span>
                    <span class="text-sm text-gray-600">Analyze contract terms and obligations</span>
                  </Button.Root>

                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto flex flex-col items-start space-y-2"
                    onclick={() => customPrompt = analysisTemplates.complianceCheck}
                  >
                    <span class="font-medium">Compliance Check</span>
                    <span class="text-sm text-gray-600">Review regulatory compliance</span>
                  </Button.Root>

                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto flex flex-col items-start space-y-2"
                    onclick={() => customPrompt = analysisTemplates.riskAssessment}
                  >
                    <span class="font-medium">Risk Assessment</span>
                    <span class="text-sm text-gray-600">Comprehensive risk analysis</span>
                  </Button.Root>
                </div>
              </div>

              <div class="bg-white rounded-lg p-6 border border-gray-200">
                <h4 class="text-lg font-semibold text-gray-900 mb-4">Custom Analysis Prompt</h4>
                <Textarea.Root
                  bind:value={customPrompt}
                  placeholder="Enter your custom analysis prompt here..."
                  class="min-h-[120px] w-full"
                />
              </div>

              <Button.Root
                onclick={performCustomAnalysis}
                disabled={!customPrompt.trim() || isProcessing}
                class="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md"
              >
                Run Analysis
              </Button.Root>

              {#if customResult}
                <div class="bg-white rounded-lg p-6 border border-gray-200">
                  <h4 class="text-lg font-semibold text-gray-900 mb-4">Analysis Result</h4>
                  <pre class="whitespace-pre-wrap text-sm text-gray-700 bg-gray-50 p-4 rounded-lg">{customResult}</pre>
                </div>
              {/if}
            </div>
          </Tabs.Content>

          <Tabs.Content value="comparison" class="mt-6">
            <div class="space-y-6">
              <div class="bg-white rounded-lg p-6 border border-gray-200">
                <h4 class="text-lg font-semibold text-gray-900 mb-4">Document to Compare</h4>
                <Textarea.Root
                  bind:value={comparisonDocument}
                  placeholder="Paste the second document content here for comparison..."
                  class="min-h-[200px] w-full"
                />
              </div>

              <Button.Root
                onclick={performComparison}
                disabled={!comparisonDocument.trim() || isProcessing}
                class="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md"
              >
                Compare Documents
              </Button.Root>
            </div>
          </Tabs.Content>

          <Tabs.Content value="extraction" class="mt-6">
            <div class="space-y-6">
              <div class="bg-white rounded-lg p-6 border border-gray-200">
                <h4 class="text-lg font-semibold text-gray-900 mb-4">Information to Extract</h4>
                <Textarea.Root
                  bind:value={extractionTemplate}
                  placeholder="Specify what information you want to extract (e.g., 'Extract all dates, monetary amounts, and party names')"
                  class="min-h-[120px] w-full"
                />
              </div>

              <Button.Root
                onclick={performExtraction}
                disabled={!extractionTemplate.trim() || isProcessing}
                class="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md"
              >
                Extract Information
              </Button.Root>
            </div>
          </Tabs.Content>
        </Tabs.Root>

      {:else if summaryResult}
        <div class="space-y-4">
          <div class="bg-white rounded-lg p-6 border border-gray-200">
            <h3 class="text-lg font-semibold text-gray-900 mb-3">Summary</h3>
            <p class="text-gray-700 leading-relaxed">{summaryResult.summary}</p>
          </div>

          {#if summaryResult.keyPoints.length > 0}
            <div class="bg-white rounded-lg p-6 border border-gray-200">
              <h4 class="text-lg font-semibold text-gray-900 mb-3">Key Points</h4>
              <ul class="space-y-1">
                {#each summaryResult.keyPoints as point}
                  <li class="text-gray-700">• {point}</li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
      {/if}
    </Card.Content>
  </Card.Root>
</div>

<style lang="postcss">
  .ai-summarization {
    @apply max-w-5xl mx-auto;
  }
</style>