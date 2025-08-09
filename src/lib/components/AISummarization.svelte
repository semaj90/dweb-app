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
  import { langchain } from '$lib/ai/langchain';
  import { multiLayerCache } from '$lib/cache/multi-layer-cache';

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








          AI Document Analysis
          {#if documentTitle}
            {documentTitle}
          {/if}


          <Button.Root
            on:click={generateSummary}
            disabled={!documentContent.trim() || isProcessing}
            class="bg-blue-600 hover:bg-blue-700"
          >
            {isProcessing ? 'Processing...' : 'Analyze Document'}

          {#if summaryResult || customResult}
            <Button.Root
              variant="outline"
              on:click={clearResults}
              class="border-gray-300"
            >
              Clear Results

          {/if}




      {#if isProcessing}


            {processingStage}
            {processingProgress}%





      {/if}



      {#if showAnalysisTools}



            Summary
            Custom Analysis
            Comparison
            Extraction




            {#if summaryResult}




                    Document Summary


                        {Math.round(summaryResult.confidence * 100)}% confidence


                        {formatProcessingTime(summaryResult.processingTime)}



                  {summaryResult.summary}



                {#if summaryResult.keyPoints.length > 0}

                    Key Points

                      {#each summaryResult.keyPoints as point}


                          {point}

                      {/each}


                {/if}


                {#if summaryResult.entities.length > 0}

                    Identified Entities

                      {#each summaryResult.entities as entity}

                          {entity.value}

                            ({Math.round(entity.confidence * 100)}%)


                      {/each}


                {/if}


                {#if summaryResult.risks.length > 0}

                    Risk Analysis

                      {#each summaryResult.risks as risk}





                                  {risk.severity} risk


                                  {risk.type}


                              {risk.description}



                      {/each}


                {/if}

            {:else if !isProcessing}




                Click "Analyze Document" to generate an AI summary

            {/if}








                  Analysis Templates


                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto"
                    on:click={() => customPrompt = analysisTemplates.contractAnalysis}
                  >

                      Contract Analysis
                      Analyze contract terms and obligations



                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto"
                    on:click={() => customPrompt = analysisTemplates.complianceCheck}
                  >

                      Compliance Check
                      Review regulatory compliance



                  <Button.Root
                    variant="outline"
                    class="text-left p-3 h-auto"
                    on:click={() => customPrompt = analysisTemplates.riskAssessment}
                  >

                      Risk Assessment
                      Comprehensive risk analysis








                  Custom Analysis Prompt

                <Textarea.Root
                  bind:value={customPrompt}
                  placeholder="Enter your custom analysis prompt here..."
                  class="min-h-[120px]"
                />


              <Button.Root
                on:click={performCustomAnalysis}
                disabled={!customPrompt.trim() || isProcessing}
                class="w-full"
              >
                Run Analysis


              {#if customResult}

                  Analysis Result
                  {customResult}

              {/if}








                  Document to Compare

                <Textarea.Root
                  bind:value={comparisonDocument}
                  placeholder="Paste the second document content here for comparison..."
                  class="min-h-[200px]"
                />


              <Button.Root
                on:click={performComparison}
                disabled={!comparisonDocument.trim() || isProcessing}
                class="w-full"
              >
                Compare Documents









                  Information to Extract

                <Textarea.Root
                  bind:value={extractionTemplate}
                  placeholder="Specify what information you want to extract (e.g., 'Extract all dates, monetary amounts, and party names')"
                  class="min-h-[120px]"
                />


              <Button.Root
                on:click={performExtraction}
                disabled={!extractionTemplate.trim() || isProcessing}
                class="w-full"
              >
                Extract Information




      {:else if summaryResult}



            Summary
            {summaryResult.summary}


          {#if summaryResult.keyPoints.length > 0}

              Key Points

                {#each summaryResult.keyPoints as point}
                  • {point}
                {/each}


          {/if}

      {/if}
</script>

<style lang="postcss">
  .ai-summarization {
    @apply max-w-5xl mx-auto;
  }
</style>
