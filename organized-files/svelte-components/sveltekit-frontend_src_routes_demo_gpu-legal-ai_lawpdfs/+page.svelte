<script lang="ts">
  import { page } from '$app/stores';
  import { enhance } from '$app/forms';
  import { invalidateAll } from '$app/navigation';
  import { writable } from 'svelte/store';
  import AiSetupBanner from '$lib/components/ai/AiSetupBanner.svelte';

  // Svelte 5 reactive state
  let uploadProgress = $state(0);
  let isProcessing = $state(false);
  let ocrResults = $state<any[]>([]);
  let processingStage = $state('');
  let errorMessage = $state('');
  let uploadedFiles = $state<File[]>([]);
  let jsonOutput = $state<any>(null);
  let simdResults = $state<any>(null);
  let clusteringResults = $state<any>(null);
  let ragRecommendations = $state<any[]>([]);

  // AI readiness state for gating actions
  let aiReady = $state<boolean | null>(null);
  let aiStatus: any = $state(null);

  async function checkAiReady() {
    try {
      const res = await fetch('/api/gpu/validate-setup');
      const data = await res.json();
      aiStatus = data;
      aiReady = Boolean(data?.ok);
    } catch (e) {
      aiReady = false;
    }
  }

  $effect(() => {
    // Check once on load
    checkAiReady();
  });

  // File upload handling
  let fileInput: HTMLInputElement

  const handleFileUpload = async (event: Event) => {
    const target = event.target as HTMLInputElement;
    const files = Array.from(target.files || []);

    if (files.length === 0) return;

    uploadedFiles = files;
    isProcessing = true;
    errorMessage = '';
    uploadProgress = 0;
    processingStage = 'Uploading files...';

    try {
      // Process each file through the pipeline
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        uploadProgress = ((i + 1) / files.length) * 100;

        // Stage 1: Upload and OCR
        processingStage = `Processing ${file.name} - OCR extraction...`;
        const ocrResult = await performOCR(file);
        ocrResults = [...ocrResults, ocrResult];

        // Stage 2: Convert to JSON
        processingStage = `Processing ${file.name} - JSON conversion...`;
        const jsonResult = await convertToJSON(ocrResult);
        jsonOutput = jsonResult;

        // Stage 3: SIMD parsing via Go microservice
        processingStage = `Processing ${file.name} - SIMD parsing...`;
        const simdResult = await sendToGoMicroservice(jsonResult);
        simdResults = simdResult;

        // Stage 4: Enhanced RAG processing
        processingStage = `Processing ${file.name} - RAG enhancement...`;
        const ragResult = await enhancedRAGProcessing(simdResult);
        ragRecommendations = [...ragRecommendations, ragResult];

        // Stage 5: SOM/K-means clustering
        processingStage = `Processing ${file.name} - Clustering analysis...`;
        const clusterResult = await performClustering(ragResult);
        clusteringResults = clusterResult;
      }

      processingStage = 'Processing complete!';

    } catch (error) {
      console.error('Processing error:', error);
      errorMessage = error.message || 'An error occurred during processing';
    } finally {
      isProcessing = false;
    }
  };

  const performOCR = async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/ocr/extract', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('OCR processing failed');
    }

    return await response.json();
  };

  const convertToJSON = async (ocrData: any): Promise<any> => {
    const response = await fetch('/api/convert/to-json', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ ocrData })
    });

    if (!response.ok) {
      throw new Error('JSON conversion failed');
    }

    return await response.json();
  };

  const sendToGoMicroservice = async (jsonData: any): Promise<any> => {
    const response = await fetch('http://localhost:8084/api/simd/parse', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(jsonData)
    });

    if (!response.ok) {
      throw new Error('SIMD parsing failed');
    }

    return await response.json();
  };

  const enhancedRAGProcessing = async (simdData: any): Promise<any> => {
    const response = await fetch('/api/rag/enhanced-process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(simdData)
    });

    if (!response.ok) {
      throw new Error('Enhanced RAG processing failed');
    }

    return await response.json();
  };

  const performClustering = async (ragData: any): Promise<any> => {
    const response = await fetch('/api/clustering/som-kmeans', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(ragData)
    });

    if (!response.ok) {
      throw new Error('Clustering analysis failed');
    }

    return await response.json();
  };

  const generatePlaywrightTests = async () => {
    processingStage = 'Generating Playwright tests...';

    try {
      const response = await fetch('/api/testing/generate-playwright', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          testResults: {
            ocrResults,
            simdResults,
            clusteringResults,
            ragRecommendations
          },
          filename: 'todosom81125.txt'
        })
      });

      if (!response.ok) {
        throw new Error('Playwright test generation failed');
      }

      const result = await response.json();
      processingStage = `Tests generated: ${result.filename}`;

    } catch (error) {
      console.error('Test generation error:', error);
      errorMessage = error.message || 'Failed to generate tests';
    }
  };
</script>

<div class="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black text-white p-6">
  <div class="max-w-6xl mx-auto">
    <div class="mb-8 text-center">
      <h1 class="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
        Legal PDF Processing Pipeline
      </h1>
      <p class="text-lg text-gray-300">
        Upload � OCR � JSON � SIMD � Enhanced RAG � SOM/K-means Clustering
      </p>
    </div>

    <!-- Upload Section -->
    <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-blue-500/30 p-6 mb-6">
      <h2 class="text-2xl font-semibold mb-4 text-blue-400">=� PDF Upload</h2>

      <div class="border-2 border-dashed border-blue-500/50 rounded-lg p-8 text-center">
        <input
          bind:this={fileInput}
          type="file"
          accept=".pdf"
          multiple
          onchange={handleFileUpload}
          class="hidden"
          id="pdf-upload"
        />

        <label
          for="pdf-upload"
          class="cursor-pointer inline-flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          Choose Legal PDF Files
        </label>

        {#if uploadedFiles.length > 0}
          <div class="mt-4">
            <p class="text-sm text-gray-400">Selected files:</p>
            {#each uploadedFiles as file}
              <div class="text-sm text-blue-300">{file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)</div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Processing Status -->
    {#if isProcessing}
      <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-yellow-500/30 p-6 mb-6">
        <h3 class="text-xl font-semibold mb-4 text-yellow-400">= Processing Status</h3>

        <div class="mb-4">
          <div class="flex justify-between text-sm mb-2">
            <span>{processingStage}</span>
            <span>{uploadProgress.toFixed(1)}%</span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-2">
            <div
              class="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
              style="width: {uploadProgress}%"
            ></div>
          </div>
        </div>
      </div>
    {/if}

    <!-- Error Display -->
    {#if errorMessage}
      <div class="bg-red-900/50 backdrop-blur-sm rounded-lg border border-red-500/30 p-6 mb-6">
        <h3 class="text-xl font-semibold mb-2 text-red-400">L Error</h3>
        <p class="text-red-300">{errorMessage}</p>
      </div>
    {/if}

    <!-- Results Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- OCR Results -->
      {#if ocrResults.length > 0}
        <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-green-500/30 p-6">
          <h3 class="text-xl font-semibold mb-4 text-green-400">=� OCR Results</h3>
          <div class="space-y-3 max-h-64 overflow-y-auto">
            {#each ocrResults as result, i}
              <div class="bg-gray-700/50 rounded p-3">
                <div class="text-sm font-medium text-gray-300 mb-2">Document {i + 1}</div>
                <div class="text-xs text-gray-400 truncate">{result.text || 'Processing...'}</div>
                <div class="text-xs text-green-300 mt-1">
                  Confidence: {result.confidence || 0}% | Pages: {result.pages || 0}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- JSON Output -->
      {#if jsonOutput}
        <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-blue-500/30 p-6">
          <h3 class="text-xl font-semibold mb-4 text-blue-400">= JSON Structure</h3>
          <pre class="bg-gray-900 rounded p-3 text-xs overflow-auto max-h-64 text-gray-300">
{JSON.stringify(jsonOutput, null, 2)}
          </pre>
        </div>
      {/if}

      <!-- SIMD Results -->
      {#if simdResults}
        <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-purple-500/30 p-6">
          <h3 class="text-xl font-semibold mb-4 text-purple-400">� SIMD Processing</h3>
          <div class="space-y-2">
            <div class="text-sm">
              <span class="text-gray-400">Processing Time:</span>
              <span class="text-purple-300">{simdResults.processingTime || 0}ms</span>
            </div>
            <div class="text-sm">
              <span class="text-gray-400">Vectors Generated:</span>
              <span class="text-purple-300">{simdResults.vectorCount || 0}</span>
            </div>
            <div class="text-sm">
              <span class="text-gray-400">Legal Concepts:</span>
              <span class="text-purple-300">{simdResults.conceptsExtracted || 0}</span>
            </div>
          </div>
        </div>
      {/if}

      <!-- RAG Recommendations -->
      {#if ragRecommendations.length > 0}
        <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-orange-500/30 p-6">
          <h3 class="text-xl font-semibold mb-4 text-orange-400">🎯 RAG Recommendations</h3>
          <div class="space-y-3 max-h-64 overflow-y-auto">
            {#each ragRecommendations as rec}
              <div class="bg-gray-700/50 rounded p-3">
                <div class="text-sm font-medium text-orange-300">{rec.title || 'Recommendation'}</div>
                <div class="text-xs text-gray-400 mt-1">{rec.summary || 'Processing...'}</div>
                <div class="text-xs text-orange-300 mt-1">
                  Relevance: {rec.relevance || 0}% | Confidence: {rec.confidence || 0}%
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- Clustering Visualization -->
    {#if clusteringResults}
      <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-cyan-500/30 p-6 mt-6">
        <h3 class="text-xl font-semibold mb-4 text-cyan-400">>� SOM/K-means Clustering</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="bg-gray-700/50 rounded p-4">
            <div class="text-sm font-medium text-cyan-300">K-means Clusters</div>
            <div class="text-2xl font-bold text-white">{clusteringResults.kmeansClusters || 0}</div>
          </div>
          <div class="bg-gray-700/50 rounded p-4">
            <div class="text-sm font-medium text-cyan-300">SOM Grid Size</div>
            <div class="text-2xl font-bold text-white">{clusteringResults.somGridSize || '0x0'}</div>
          </div>
          <div class="bg-gray-700/50 rounded p-4">
            <div class="text-sm font-medium text-cyan-300">Cluster Accuracy</div>
            <div class="text-2xl font-bold text-white">{clusteringResults.accuracy || 0}%</div>
          </div>
        </div>

        <!-- "Did you mean" suggestions -->
        {#if clusteringResults?.suggestions}
          <div class="mt-4">
            <h4 class="text-lg font-semibold text-cyan-300 mb-2">=� "Did you mean" Suggestions</h4>
            <div class="flex flex-wrap gap-2">
              {#each clusteringResults.suggestions as suggestion}
                <span class="bg-cyan-900/50 text-cyan-300 px-3 py-1 rounded-full text-sm">
                  {suggestion}
                </span>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Test Generation -->
    <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-indigo-500/30 p-6 mt-6">
      <h3 class="text-xl font-semibold mb-4 text-indigo-400">>� Test Generation</h3>
      <p class="text-gray-300 mb-4">Generate comprehensive Playwright tests for this processing pipeline</p>

      <button
        on:click={generatePlaywrightTests}
        disabled={!ocrResults.length || isProcessing}
        class="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
      >
        Generate todosom81125.txt
      </button>
    </div>

    <!-- pgai Extension Test -->
    <div class="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-pink-500/30 p-6 mt-6">
      <h3 class="text-xl font-semibold mb-4 text-pink-400">> PostgreSQL pgai Extension</h3>
      <p class="text-gray-300 mb-4">Test AI summarization capabilities with the uploaded documents</p>
      {#if aiReady === false}
        <AiSetupBanner {data}={aiStatus} autoFetch={false} />
      {/if}

      <button
        on:click={async () => {
          if (!jsonOutput) return;
          if (aiReady === false) return;

          processingStage = 'Testing pgai extension...';

          try {
            const response = await fetch('/api/pgai/summarize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ document: jsonOutput })
            });

            if (!response.ok) throw new Error('pgai test failed');

            const result = await response.json();
            processingStage = `pgai summary generated: ${result.summary?.length || 0} chars`;

          } catch (error) {
            errorMessage = `pgai test error: ${error.message}`;
          }
        }}
  disabled={!jsonOutput || isProcessing || aiReady === false}
        class="px-6 py-3 bg-pink-600 hover:bg-pink-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
      >
        Test pgai Summarization
      </button>
    </div>
  </div>
</div>

<style>
  :global(html) {
    background: #0f172a;
  }
</style>