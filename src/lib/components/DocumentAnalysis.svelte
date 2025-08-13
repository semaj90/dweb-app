<script lang="ts">
  import { onMount } from 'svelte';
  import { Button, Progress } from 'bits-ui';
  import { langchain } from '$lib/ai/langchain';
  <script lang="ts">
    import { Button } from 'bits-ui';
    import { langchain } from '$lib/ai/langchain';

    // Svelte 5 state
    let file = $state<File | null>(null);
    <script lang="ts">
      import { Button } from 'bits-ui';
      import { langchain } from '$lib/ai/langchain';

      let file = $state<File | null>(null);
      let isAnalyzing = $state(false);
      let progress = $state(0);
      let analysisResult = $state<any | null>(null);
      let extractedText = $state('');
      let documentChunks = $state<any[]>([]);
      let error = $state<string | null>(null);

      export let onAnalysisComplete: ((result: any) => void) | null = null;
      export let allowedTypes: string[] = ['.pdf', '.txt', '.docx', '.doc'];
      export let maxSizeMB: number = 10;

      const canAnalyze = $derived(file !== null && !isAnalyzing);
      const fileSizeMB = $derived(file ? file.size / (1024 * 1024) : 0);

      async function handleFileSelect(event: Event) {
        const input = event.target as HTMLInputElement;
        const selectedFile = input.files?.[0];
        if (!selectedFile) return;
        const extension = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
        if (!allowedTypes.includes(extension)) {
          error = `File type not allowed. Accepted types: ${allowedTypes.join(', ')}`;
          return;
        }
        if (selectedFile.size / (1024 * 1024) > maxSizeMB) {
          error = `File too large. Maximum size: ${maxSizeMB}MB`;
          return;
        }
        file = selectedFile;
        error = null;
        analysisResult = null;
      }

      async function extractTextFromFile(): Promise<string> {
        if (!file) throw new Error('No file selected');
        if (file.type === 'text/plain') return await file.text();
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/extract-text', { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Failed to extract text from file');
        const data = await response.json();
        return data.text;
      }

      async function analyzeDocument() {
        if (!file || isAnalyzing) return;
        isAnalyzing = true;
        progress = 0;
        error = null;
        try {
          progress = 10;
          extractedText = await extractTextFromFile();
          progress = 30;
          documentChunks = await langchain.ingestDocument(extractedText, { filename: file.name, uploadedAt: new Date(), type: 'legal_document' });
          progress = 50;
          const analysis = await langchain.analyzeContract(extractedText);
          progress = 70;
          const summary = await langchain.summarizeDocument(extractedText);
          progress = 85;
          const keyInfo = await langchain.extractInfo(extractedText, `Extract document type, parties, key dates, monetary amounts, important clauses as JSON`);
          analysisResult = { ...analysis, summary, keyInfo, metadata: { filename: file.name, size: fileSizeMB.toFixed(2) + ' MB', analyzedAt: new Date(), chunks: documentChunks.length } };
          progress = 100;
          onAnalysisComplete?.(analysisResult);
        } catch (err: any) {
          error = err?.message || 'Analysis failed';
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

    <h2>Legal Document Analysis</h2>

    {#if !analysisResult}
      <div class="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 hover:border-blue-500 dark:hover:border-blue-400 transition-colors">
        <input type="file" accept={allowedTypes.join(',')} onchange={handleFileSelect} class="hidden" id="file-input" />
        <label for="file-input" class="cursor-pointer block text-center">
          <div class="font-medium">{file ? file.name : 'Click to upload legal document'}</div>
          <div class="text-sm text-gray-500 mt-1">Supported formats: {allowedTypes.join(', ')} (max {maxSizeMB}MB)</div>
        </label>

        {#if file}
          <div class="mt-4 flex items-center justify-between">
            <div>
              <div class="font-medium">{file.name}</div>
              <div class="text-sm text-gray-500">{fileSizeMB.toFixed(2)} MB</div>
            </div>
            <button onclick={() => (file = null)} class="text-red-600 hover:text-red-700 dark:text-red-400">Remove</button>
          </div>
        {/if}

        {#if error}
          <div class="mt-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-400">{error}</div>
        {/if}

        <Button onclick={analyzeDocument} disabled={!canAnalyze} class="w-full mt-4 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200">
          {#if isAnalyzing}
            Analyzing... {progress}%
          {:else}
            Analyze Document
          {/if}
        </Button>
      </div>
    {:else}
      <div class="space-y-4">
        <h3 class="font-semibold">Document Summary</h3>
        <p>{analysisResult.summary}</p>

        {#if analysisResult.keyInfo}
          <div>
            <h4 class="font-semibold">Key Information</h4>
            <ul class="list-disc list-inside">
              {#each Object.entries(analysisResult.keyInfo) as [key, value]}
                <li>
                  <strong>{key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}:</strong>
                  {Array.isArray(value) ? value.join(', ') : value}
                </li>
              {/each}
            </ul>
          </div>
        {/if}

        <div class="flex items-center gap-4 text-sm text-gray-600">
          <div>File: {analysisResult.metadata.filename}</div>
          <div>Size: {analysisResult.metadata.size}</div>
          <div>Chunks: {analysisResult.metadata.chunks}</div>
          <div>Analyzed: {new Date(analysisResult.metadata.analyzedAt).toLocaleString()}</div>
        </div>

        <div class="flex items-center gap-2">
          <Button onclick={resetAnalysis} class="py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700">Analyze Another Document</Button>
          <Button onclick={() => navigator.clipboard.writeText(JSON.stringify(analysisResult, null, 2))} class="py-2 px-4 bg-green-600 text-white rounded-lg hover:bg-green-700">Copy JSON</Button>
        </div>
      </div>
    {/if}
    </div>

  {:else}
