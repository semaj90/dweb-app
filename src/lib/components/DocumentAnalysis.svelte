<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
  import { onMount } from 'svelte';
  import { Button, Progress } from 'bits-ui';
  import { langchain } from '$lib/ai/langchain';
  import { ollama, MODELS } from '$lib/ai/ollama';
  import type { Document } from '$lib/ai/types';
  
  // Svelte 5 state
  let file = $state<File | null>(null);
  let isAnalyzing = $state(false);
  let progress = $state(0);
  let analysisResult = $state<any>(null);
  let extractedText = $state('');
  let documentChunks = $state<Document[]>([]);
  let error = $state<string | null>(null);
  
  // Props
  let {
    onAnalysisComplete = null,
    allowedTypes = ['.pdf', '.txt', '.docx', '.doc'],
    maxSizeMB = 10
  }: {
    onAnalysisComplete?: ((result: any) => void) | null;
    allowedTypes?: string[];
    maxSizeMB?: number;
  } = $props();
  
  // Reactive
  const canAnalyze = $derived(file !== null && !isAnalyzing);
  const fileSizeMB = $derived(file ? file.size / (1024 * 1024) : 0);
  const isValidSize = $derived(fileSizeMB <= maxSizeMB);
  
  async function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    const selectedFile = input.files?.[0];
    
    if (!selectedFile) return;
    
    // Check file type
    const extension = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
    if (!allowedTypes.includes(extension)) {
      error = `File type not allowed. Accepted types: ${allowedTypes.join(', ')}`;
      return;
    }
    
    // Check file size
    if (!isValidSize) {
      error = `File too large. Maximum size: ${maxSizeMB}MB`;
      return;
    }
    
    file = selectedFile;
    error = null;
    analysisResult = null;
  }
  
  async function extractTextFromFile(): Promise<string> {
    if (!file) throw new Error('No file selected');
    
    // For demonstration, we'll read text files directly
    // In production, you'd use proper PDF/DOCX parsing libraries
    if (file.type === 'text/plain') {
      return await file.text();
    }
    
    // For other types, you'd typically send to a backend service
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/extract-text', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Failed to extract text from file');
    }
    
    const data = await response.json();
    return data.text;
  }
  
  async function analyzeDocument() {
    if (!file || isAnalyzing) return;
    
    isAnalyzing = true;
    progress = 0;
    error = null;
    
    try {
      // Step 1: Extract text (20%)
      progress = 10;
      extractedText = await extractTextFromFile();
      progress = 20;
      
      // Step 2: Generate embeddings and store chunks (40%)
      progress = 30;
      documentChunks = await langchain.ingestDocument(extractedText, {
        filename: file.name,
        uploadedAt: new Date(),
        type: 'legal_document'
      });
      progress = 40;
      
      // Step 3: Analyze contract (60%)
      progress = 50;
      const analysis = await langchain.analyzeContract(extractedText);
      progress = 60;
      
      // Step 4: Generate summary (80%)
      progress = 70;
      const summary = await langchain.summarizeDocument(extractedText);
      progress = 80;
      
      // Step 5: Extract key information (100%)
      progress = 90;
      const keyInfo = await langchain.extractInfo(extractedText, `
        Extract:
        - Document type
        - Parties involved
        - Key dates
        - Monetary amounts
        - Important clauses
      `);
      
      analysisResult = {
        ...analysis,
        summary,
        keyInfo,
        metadata: {
          filename: file.name,
          size: fileSizeMB.toFixed(2) + ' MB',
          analyzedAt: new Date(),
          chunks: documentChunks.length
        }
      };
      
      progress = 100;
      
      // Callback
      if (onAnalysisComplete) {
        onAnalysisComplete(analysisResult);
      }
      
    } catch (err) {
      error = err instanceof Error ? err.message : 'Analysis failed';
      console.error('Document analysis error:', err);
    } finally {
      isAnalyzing = false;
    }
  }
  
  function resetAnalysis() {
    file = null;
    analysisResult = null;
    extractedText = '';
    documentChunks = [];
    progress = 0;
    error = null;
  }
</script>

<div class="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
  <h2 class="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
    Legal Document Analysis
  </h2>
  
  {#if !analysisResult}
    <!-- File Upload Section -->
    <div class="space-y-4">
      <div class="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8
                  hover:border-blue-500 dark:hover:border-blue-400 transition-colors">
        <input
          type="file"
          accept={allowedTypes.join(',')}
          onchange={handleFileSelect}
          class="hidden"
          id="file-input"
        />
        <label for="file-input" class="cursor-pointer">
          <div class="text-center">
            <div class="i-carbon-document-add text-4xl text-gray-400 dark:text-gray-500 mx-auto mb-4"></div>
            <p class="text-lg font-medium text-gray-700 dark:text-gray-300">
              {file ? file.name : 'Click to upload legal document'}
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Supported formats: {allowedTypes.join(', ')} (max {maxSizeMB}MB)
            </p>
          </div>
        </label>
      </div>
      
      {#if file}
        <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <div class="i-carbon-document text-2xl text-blue-600"></div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">{file.name}</p>
                <p class="text-sm text-gray-500 dark:text-gray-400">
                  {fileSizeMB.toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              onclick={() => file = null}
              class="text-red-600 hover:text-red-700 dark:text-red-400"
            >
              <div class="i-carbon-close text-xl"></div>
            </button>
          </div>
        </div>
      {/if}
      
      {#if error}
        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 
                    rounded-lg p-4 text-red-700 dark:text-red-400">
          <div class="flex items-center gap-2">
            <div class="i-carbon-warning"></div>
            <span>{error}</span>
          </div>
        </div>
      {/if}
      
      <Button
        onclick={analyzeDocument}
        disabled={!canAnalyze}
        class="w-full py-3 bg-blue-600 text-white rounded-lg font-medium
               hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
               transition-colors duration-200"
      >
        {#if isAnalyzing}
          <div class="flex items-center justify-center gap-2">
            <div class="i-carbon-analytics animate-spin"></div>
            <span>Analyzing... {progress}%</span>
          </div>
        {:else}
          <div class="flex items-center justify-center gap-2">
            <div class="i-carbon-analytics"></div>
            <span>Analyze Document</span>
          </div>
        {/if}
      </Button>
      
      {#if isAnalyzing}
        <Progress value={progress} class="h-2" />
      {/if}
    </div>
  {:else}
    <!-- Analysis Results -->
    <div class="space-y-6">
      <!-- Summary Section -->
      <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2 text-blue-900 dark:text-blue-300">
          <div class="i-carbon-summary-kpi"></div>
          Document Summary
        </h3>
        <p class="text-gray-700 dark:text-gray-300 leading-relaxed">
          {analysisResult.summary}
        </p>
      </div>
      
      <!-- Key Information -->
      {#if analysisResult.keyInfo}
        <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
            <div class="i-carbon-list-checked"></div>
            Key Information
          </h3>
          <div class="space-y-2">
            {#each Object.entries(analysisResult.keyInfo) as [key, value]}
              <div class="flex justify-between py-2 border-b dark:border-gray-700 last:border-0">
                <span class="font-medium text-gray-600 dark:text-gray-400">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                </span>
                <span class="text-gray-900 dark:text-white">
                  {Array.isArray(value) ? value.join(', ') : value}
                </span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
      
      <!-- Key Terms -->
      {#if analysisResult.keyTerms?.length > 0}
        <div>
          <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
            <div class="i-carbon-term"></div>
            Key Terms & Conditions
          </h3>
          <ul class="space-y-2">
            {#each analysisResult.keyTerms as term}
              <li class="flex items-start gap-2">
                <div class="i-carbon-checkmark-filled text-green-600 mt-1"></div>
                <span class="text-gray-700 dark:text-gray-300">{term}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
      
      <!-- Risks -->
      {#if analysisResult.risks?.length > 0}
        <div>
          <h3 class="text-lg font-semibold mb-3 flex items-center gap-2 text-amber-700 dark:text-amber-400">
            <div class="i-carbon-warning-alt"></div>
            Potential Risks
          </h3>
          <ul class="space-y-2">
            {#each analysisResult.risks as risk}
              <li class="flex items-start gap-2">
                <div class="i-carbon-warning text-amber-600 mt-1"></div>
                <span class="text-gray-700 dark:text-gray-300">{risk}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
      
      <!-- Recommendations -->
      {#if analysisResult.recommendations?.length > 0}
        <div>
          <h3 class="text-lg font-semibold mb-3 flex items-center gap-2 text-blue-700 dark:text-blue-400">
            <div class="i-carbon-idea"></div>
            Recommendations
          </h3>
          <ul class="space-y-2">
            {#each analysisResult.recommendations as rec}
              <li class="flex items-start gap-2">
                <div class="i-carbon-arrow-right text-blue-600 mt-1"></div>
                <span class="text-gray-700 dark:text-gray-300">{rec}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
      
      <!-- Metadata -->
      <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 text-sm">
        <div class="grid grid-cols-2 gap-4">
          <div>
            <span class="text-gray-500 dark:text-gray-400">File:</span>
            <span class="ml-2 font-medium">{analysisResult.metadata.filename}</span>
          </div>
          <div>
            <span class="text-gray-500 dark:text-gray-400">Size:</span>
            <span class="ml-2 font-medium">{analysisResult.metadata.size}</span>
          </div>
          <div>
            <span class="text-gray-500 dark:text-gray-400">Chunks:</span>
            <span class="ml-2 font-medium">{analysisResult.metadata.chunks}</span>
          </div>
          <div>
            <span class="text-gray-500 dark:text-gray-400">Analyzed:</span>
            <span class="ml-2 font-medium">
              {new Date(analysisResult.metadata.analyzedAt).toLocaleString()}
            </span>
          </div>
        </div>
      </div>
      
      <!-- Actions -->
      <div class="flex gap-3">
        <Button
          onclick={resetAnalysis}
          class="flex-1 py-2 bg-gray-600 text-white rounded-lg
                 hover:bg-gray-700 transition-colors"
        >
          Analyze Another Document
        </Button>
        <Button
          onclick={() => navigator.clipboard.writeText(JSON.stringify(analysisResult, null, 2))}
          class="py-2 px-4 bg-green-600 text-white rounded-lg
                 hover:bg-green-700 transition-colors"
        >
          <div class="i-carbon-copy"></div>
        </Button>
      </div>
    </div>
  {/if}
</div>
