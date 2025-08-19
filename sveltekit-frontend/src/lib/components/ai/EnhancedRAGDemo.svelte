<!-- Enhanced RAG Demo Component with Semantic Analysis -->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    semanticAnalyzer,
    isAnalyzingStore,
    semanticAnalysisStore,
    ragResponseStore,
    ragQueryStore,
    type SemanticAnalysisResult,
    type RAGQuery,
    type RAGResponse,
  } from '$lib/services/enhanced-rag-semantic-analyzer';
  import { Button } from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';

  // Reactive state using runes
  let sampleLegalText = $state(`
MEMORANDUM OF UNDERSTANDING

This Memorandum of Understanding ("MOU") is entered into on January 15, 2024, between TechCorp Inc., a Delaware corporation ("Company"), and John Smith, Esq., individually ("Consultant").

WHEREAS, Company desires to engage Consultant to provide legal advisory services regarding intellectual property matters and contract negotiations;

WHEREAS, Consultant agrees to provide such services pursuant to the terms and conditions set forth herein;

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

1. SERVICES. Consultant shall provide legal advisory services to Company, including but not limited to:
   a) Review and analysis of intellectual property portfolios
   b) Contract negotiation and drafting
   c) Legal research and compliance advisory

2. COMPENSATION. Company shall pay Consultant $350 per hour for services rendered, payable within 30 days of receipt of invoice.

3. CONFIDENTIALITY. Consultant acknowledges that during the course of engagement, Consultant may have access to confidential and proprietary information of Company.

4. LIABILITY. Company's total liability under this MOU shall not exceed $50,000 in aggregate.

5. BREACH. In the event of breach by either party, the non-breaching party may terminate this MOU upon written notice.

This MOU shall be governed by Delaware law and shall remain in effect until December 31, 2024, unless terminated earlier in accordance with its terms.

IN WITNESS WHEREOF, the parties have executed this MOU as of the date first written above.
    `);

  let queryText = $state('What are the liability limitations in this contract?');
  let isAnalyzing = $state(false);
  let analysisResult = $state<SemanticAnalysisResult | null>(null);
  let ragResponse = $state<RAGResponse | null>(null);
  let activeTab = $state<'analyze' | 'query'>('analyze');

  // Advanced search filters
  let useSemanticExpansion = $state(true);
  let confidenceThreshold = $state(0.7);
  let selectedEntityTypes = $state<string[]>(['LEGAL_CONCEPT', 'PERSON', 'ORGANIZATION', 'MONEY']);

  // Subscribe to stores
  $effect(() => {
    isAnalyzing = $isAnalyzingStore;
    analysisResult = $semanticAnalysisStore;
    ragResponse = $ragResponseStore;
  });

  /**
   * Analyze the sample legal document
   */
  async function analyzeDocument() {
    if (!sampleLegalText.trim()) return;

    isAnalyzingStore.set(true);

    try {
      const result = await semanticAnalyzer.analyzeDocument(sampleLegalText, `doc_${Date.now()}`);

      semanticAnalysisStore.set(result);
      console.log('Semantic Analysis Result:', result);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      isAnalyzingStore.set(false);
    }
  }

  /**
   * Perform enhanced RAG query
   */
  async function performRAGQuery() {
    if (!queryText.trim()) return;

    isAnalyzingStore.set(true);

    try {
      const query: RAGQuery = {
        query: queryText,
        filters: {
          entityTypes: selectedEntityTypes,
          confidenceThreshold,
        },
        semantic: {
          useEmbeddings: true,
          expandConcepts: useSemanticExpansion,
          includeRelated: true,
        },
      };

      ragQueryStore.set(query);

      const response = await semanticAnalyzer.enhancedQuery(query);
      ragResponseStore.set(response);

      console.log('RAG Query Response:', response);
    } catch (error) {
      console.error('RAG query failed:', error);
      alert(`RAG query failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      isAnalyzingStore.set(false);
    }
  }

  /**
   * Format entity type for display
   */
  function formatEntityType(type: string): string {
    return type
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }

  /**
   * Get entity type color for UI
   */
  function getEntityColor(type: string): string {
    const colors = {
      PERSON: 'bg-blue-100 text-blue-800',
      ORGANIZATION: 'bg-green-100 text-green-800',
      MONEY: 'bg-yellow-100 text-yellow-800',
      DATE: 'bg-purple-100 text-purple-800',
      LEGAL_CONCEPT: 'bg-red-100 text-red-800',
      CASE_REF: 'bg-indigo-100 text-indigo-800',
      STATUTE: 'bg-gray-100 text-gray-800',
    };
    return colors[type as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  }

  onMount(() => {
    console.log('Enhanced RAG Demo loaded');
  });
</script>

<div class="enhanced-rag-demo p-6 max-w-6xl mx-auto space-y-6">
  <!-- Header -->
  <div class="header">
    <h1 class="text-3xl font-bold text-gray-900">Enhanced RAG System with Semantic Analysis</h1>
    <p class="text-gray-600 mt-2">
      Demonstrate advanced semantic analysis, entity extraction, and intelligent querying for legal
      documents
    </p>
  </div>

  <!-- Tab Navigation -->
  <div class="tabs flex border-b">
    <button
      class="tab px-4 py-2 font-medium {activeTab === 'analyze'
        ? 'border-b-2 border-blue-500 text-blue-600'
        : 'text-gray-500 hover:text-gray-700'}"
      onclick={() => (activeTab = 'analyze')}>
      Document Analysis
    </button>
    <button
      class="tab px-4 py-2 font-medium {activeTab === 'query'
        ? 'border-b-2 border-blue-500 text-blue-600'
        : 'text-gray-500 hover:text-gray-700'}"
      onclick={() => (activeTab = 'query')}>
      Enhanced RAG Query
    </button>
  </div>

  <!-- Document Analysis Tab -->
  {#if activeTab === 'analyze'}
    <div class="analysis-tab space-y-6">
      <!-- Input Section -->
      <Card>
        <CardHeader>
          <CardTitle>Legal Document Input</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-4">
            <div>
              <label for="legal-text" class="block text-sm font-medium text-gray-700 mb-2">
                Sample Legal Text (MOU)
              </label>
              <textarea
                id="legal-text"
                bind:value={sampleLegalText}
                rows="12"
                class="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                placeholder="Enter legal document text for analysis..."
                disabled={isAnalyzing}></textarea>
            </div>

            <div class="flex justify-between items-center">
              <div class="text-sm text-gray-500">
                {sampleLegalText.length} characters, ~{Math.ceil(
                  sampleLegalText.split(/\s+/).length
                )} words
              </div>
              <Button
                on:click={analyzeDocument}
                disabled={isAnalyzing || !sampleLegalText.trim()}
                class="px-6">
                {isAnalyzing ? 'Analyzing...' : 'Analyze Document'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Analysis Results -->
      {#if analysisResult}
        <div class="analysis-results grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Entities -->
          <Card>
            <CardHeader>
              <CardTitle>Extracted Entities ({analysisResult.entities.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="space-y-3">
                {#each analysisResult.entities as entity}
                  <div class="entity-item p-3 border rounded-lg">
                    <div class="flex items-start justify-between">
                      <div class="flex-1">
                        <div class="font-semibold text-gray-900">{entity.text}</div>
                        <div class="flex items-center gap-2 mt-1">
                          <span
                            class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getEntityColor(
                              entity.type
                            )}">
                            {formatEntityType(entity.type)}
                          </span>
                          <span class="text-xs text-gray-500">
                            {Math.round(entity.confidence * 100)}% confidence
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            </CardContent>
          </Card>

          <!-- Concepts -->
          <Card>
            <CardHeader>
              <CardTitle>Legal Concepts ({analysisResult.concepts.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="space-y-3">
                {#each analysisResult.concepts as concept}
                  <div class="concept-item p-3 border rounded-lg">
                    <div class="font-semibold text-gray-900 mb-1">{concept.concept}</div>
                    <div class="text-sm text-gray-600 mb-2">
                      Category: {concept.legalCategory}
                    </div>
                    <div class="flex flex-wrap gap-1">
                      {#each concept.relatedConcepts.slice(0, 4) as related}
                        <span
                          class="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-50 text-blue-700">
                          {related}
                        </span>
                      {/each}
                    </div>
                    <div class="mt-2 text-xs text-gray-500">
                      Confidence: {Math.round(concept.confidenceScore * 100)}%
                    </div>
                  </div>
                {/each}
              </div>
            </CardContent>
          </Card>

          <!-- Metrics -->
          <Card>
            <CardHeader>
              <CardTitle>Analysis Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="grid grid-cols-2 gap-4">
                <div class="metric">
                  <div class="text-2xl font-bold text-blue-600">
                    {Math.round(analysisResult.legalRelevanceScore * 100)}%
                  </div>
                  <div class="text-sm text-gray-600">Legal Relevance</div>
                </div>
                <div class="metric">
                  <div class="text-2xl font-bold text-green-600">
                    {analysisResult.complexityIndex}/10
                  </div>
                  <div class="text-sm text-gray-600">Complexity Index</div>
                </div>
                <div class="metric">
                  <div class="text-2xl font-bold text-purple-600">
                    {analysisResult.sentimentScore > 0 ? '+' : ''}{Math.round(
                      analysisResult.sentimentScore * 100
                    )}
                  </div>
                  <div class="text-sm text-gray-600">Sentiment Score</div>
                </div>
                <div class="metric">
                  <div class="text-2xl font-bold text-orange-600">
                    {Math.round(analysisResult.processingTime)}ms
                  </div>
                  <div class="text-sm text-gray-600">Processing Time</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <!-- Embeddings Preview -->
          <Card>
            <CardHeader>
              <CardTitle>Vector Embeddings</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="space-y-2">
                <div class="text-sm text-gray-600">
                  Generated {analysisResult.summaryEmbedding.length}D vector embedding
                </div>
                <div class="embedding-preview p-2 bg-gray-50 rounded text-xs font-mono">
                  [{analysisResult.summaryEmbedding
                    .slice(0, 10)
                    .map((x) => x.toFixed(4))
                    .join(', ')}, ...]
                </div>
                <div class="text-xs text-gray-500">
                  First 10 dimensions shown. Full embedding stored in vector database.
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      {/if}
    </div>
  {/if}

  <!-- RAG Query Tab -->
  {#if activeTab === 'query'}
    <div class="query-tab space-y-6">
      <!-- Query Configuration -->
      <Card>
        <CardHeader>
          <CardTitle>Enhanced RAG Query</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-4">
            <div>
              <label for="query-text" class="block text-sm font-medium text-gray-700 mb-2">
                Query Text
              </label>
              <input
                id="query-text"
                type="text"
                bind:value={queryText}
                class="w-full p-3 border border-gray-300 rounded-lg"
                placeholder="Ask about legal concepts, entities, or document content..."
                disabled={isAnalyzing} />
            </div>

            <!-- Query Options -->
            <div class="query-options grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label class="flex items-center space-x-2">
                  <input type="checkbox" bind:checked={useSemanticExpansion} class="rounded" />
                  <span class="text-sm">Semantic Expansion</span>
                </label>
              </div>

              <div>
                <label for="confidence" class="block text-sm text-gray-600 mb-1">
                  Confidence Threshold: {confidenceThreshold}
                </label>
                <input
                  id="confidence"
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  bind:value={confidenceThreshold}
                  class="w-full" />
              </div>

              <div>
                <label class="block text-sm text-gray-600 mb-1">Entity Types</label>
                <select
                  multiple
                  bind:value={selectedEntityTypes}
                  class="w-full p-1 border border-gray-300 rounded text-sm">
                  <option value="LEGAL_CONCEPT">Legal Concepts</option>
                  <option value="PERSON">Persons</option>
                  <option value="ORGANIZATION">Organizations</option>
                  <option value="MONEY">Money</option>
                  <option value="DATE">Dates</option>
                </select>
              </div>
            </div>

            <div class="flex justify-end">
              <Button
                on:click={performRAGQuery}
                disabled={isAnalyzing || !queryText.trim()}
                class="px-6">
                {isAnalyzing ? 'Querying...' : 'Execute RAG Query'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Query Results -->
      {#if ragResponse}
        <Card>
          <CardHeader>
            <CardTitle>Query Results ({ragResponse.totalFound} found)</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-4">
              <!-- Query Summary -->
              <div class="query-summary p-3 bg-blue-50 rounded-lg">
                <div class="text-sm text-blue-800">
                  <strong>Query:</strong>
                  {ragResponse.query}
                </div>
                <div class="text-xs text-blue-600 mt-1">
                  Processing time: {Math.round(ragResponse.processingTime)}ms
                </div>
                {#if ragResponse.semanticExpansions && ragResponse.semanticExpansions.length > 0}
                  <div class="text-xs text-blue-600 mt-1">
                    <strong>Semantic expansions:</strong>
                    {ragResponse.semanticExpansions.join(', ')}
                  </div>
                {/if}
              </div>

              <!-- Results -->
              <div class="results space-y-3">
                {#each ragResponse.results as result}
                  <div class="result-item p-4 border border-gray-200 rounded-lg">
                    <div class="flex justify-between items-start mb-2">
                      <h4 class="font-medium text-gray-900">{result.title}</h4>
                      <span class="text-sm text-blue-600 font-medium">
                        {Math.round(result.relevanceScore * 100)}% match
                      </span>
                    </div>
                    <p class="text-sm text-gray-600 mb-3">{result.excerpt}</p>

                    {#if result.entities && result.entities.length > 0}
                      <div class="entities flex flex-wrap gap-1">
                        {#each result.entities.slice(0, 5) as entity}
                          <span
                            class="inline-flex items-center px-2 py-1 rounded text-xs {getEntityColor(
                              entity.type
                            )}">
                            {entity.text}
                          </span>
                        {/each}
                      </div>
                    {/if}
                  </div>
                {/each}

                {#if ragResponse.results.length === 0}
                  <div class="no-results p-4 text-center text-gray-500">
                    No results found. Try adjusting your query or lowering the confidence threshold.
                  </div>
                {/if}
              </div>
            </div>
          </CardContent>
        </Card>
      {/if}
    </div>
  {/if}
</div>

<style>
  .enhanced-rag-demo {
    font-family:
      system-ui,
      -apple-system,
      sans-serif;
  }

  .tab {
    transition: all 0.2s ease;
  }

  .entity-item,
  .concept-item,
  .result-item {
    transition: all 0.2s ease;
  }

  .entity-item:hover,
  .concept-item:hover,
  .result-item:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .metric {
    text-align: center;
    padding: 1rem;
    border-radius: 0.5rem;
    background: #f9fafb;
  }

  .embedding-preview {
    max-width: 100%;
    overflow-x: auto;
    white-space: nowrap;
  }

  @media (max-width: 768px) {
    .query-options {
      grid-template-columns: 1fr;
    }

    .analysis-results {
      grid-template-columns: 1fr;
    }
  }
</style>
