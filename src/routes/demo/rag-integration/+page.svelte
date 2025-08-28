&lt;script lang="ts"&gt;
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Button from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';

  interface UploadStatus {
    status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
    progress: number;
    message: string;
    fileId?: string;
    jobId?: string;
  }

  interface RAGQuery {
    query: string;
    context: string[];
    embedding?: number[];
    similarity_scores?: number[];
  }

  interface RAGResponse {
    answer: string;
    sources: Array&lt;{
      id: string;
      title: string;
      excerpt: string;
      similarity: number;
      caseId?: string;
    }&gt;;
    processingTime: number;
    model: string;
    tokensUsed: number;
  }

  let fileInput: HTMLInputElement;
  let uploadStatus = writable&lt;UploadStatus&gt;({
    status: 'idle',
    progress: 0,
    message: 'Ready to upload'
  });

  let ragQuery = '';
  let ragResponse: RAGResponse | null = null;
  let isQuerying = false;

  // Demo case ID for organization
  let demoCase = {
    id: crypto.randomUUID(),
    title: 'RAG Integration Demo Case',
    created: new Date().toISOString()
  };

  onMount(async () =&gt; {
    // Initialize demo case in database
    try {
      await fetch('/api/v1/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(demoCase)
      });
    } catch (error) {
      console.warn('Demo case creation failed:', error);
    }
  });

  async function handleFileUpload() {
    if (!fileInput.files?.length) return;

    const file = fileInput.files[0];
    
    uploadStatus.update(s =&gt; ({
      ...s,
      status: 'uploading',
      progress: 10,
      message: `Uploading ${file.name}...`
    }));

    try {
      // Step 1: Get pre-signed URL from MinIO
      const presignedResponse = await fetch('/api/v1/upload/presigned', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: file.name,
          contentType: file.type,
          caseId: demoCase.id
        })
      });

      if (!presignedResponse.ok) throw new Error('Pre-signed URL failed');
      
      const { uploadUrl, fileId } = await presignedResponse.json();

      uploadStatus.update(s =&gt; ({
        ...s,
        progress: 25,
        message: 'Uploading to MinIO...',
        fileId
      }));

      // Step 2: Direct upload to MinIO
      const uploadResponse = await fetch(uploadUrl, {
        method: 'PUT',
        body: file
      });

      if (!uploadResponse.ok) throw new Error('MinIO upload failed');

      uploadStatus.update(s =&gt; ({
        ...s,
        status: 'processing',
        progress: 50,
        message: 'Starting ingestion pipeline...'
      }));

      // Step 3: Trigger ingestion job via Redis
      const jobResponse = await fetch('/api/v1/jobs/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fileId,
          caseId: demoCase.id,
          filename: file.name,
          contentType: file.type
        })
      });

      if (!jobResponse.ok) throw new Error('Job creation failed');
      
      const { jobId } = await jobResponse.json();

      uploadStatus.update(s =&gt; ({
        ...s,
        progress: 75,
        message: 'Processing document...',
        jobId
      }));

      // Step 4: Poll job status
      await pollJobStatus(jobId);

    } catch (error) {
      console.error('Upload failed:', error);
      uploadStatus.update(s =&gt; ({
        ...s,
        status: 'error',
        message: `Error: ${error.message}`
      }));
    }
  }

  async function pollJobStatus(jobId: string) {
    const maxAttempts = 30; // 30 seconds max
    let attempts = 0;

    const poll = async () =&gt; {
      if (attempts &gt;= maxAttempts) {
        uploadStatus.update(s =&gt; ({
          ...s,
          status: 'error',
          message: 'Processing timeout'
        }));
        return;
      }

      try {
        const response = await fetch(`/api/v1/jobs/${jobId}`);
        if (!response.ok) throw new Error('Job status check failed');
        
        const job = await response.json();
        
        if (job.status === 'completed') {
          uploadStatus.update(s =&gt; ({
            ...s,
            status: 'completed',
            progress: 100,
            message: `Document processed! ${job.extractedText?.length || 0} characters extracted, ${job.embeddingCount || 0} embeddings created.`
          }));
        } else if (job.status === 'failed') {
          uploadStatus.update(s =&gt; ({
            ...s,
            status: 'error',
            message: `Processing failed: ${job.error || 'Unknown error'}`
          }));
        } else {
          // Still processing
          const progressMap = {
            'ocr': 80,
            'nlp': 85,
            'embedding': 90,
            'storage': 95
          };
          
          uploadStatus.update(s =&gt; ({
            ...s,
            progress: progressMap[job.currentStep] || 75,
            message: `Processing: ${job.currentStep}...`
          }));
          
          attempts++;
          setTimeout(poll, 1000);
        }
      } catch (error) {
        console.error('Job polling error:', error);
        attempts++;
        setTimeout(poll, 1000);
      }
    };

    poll();
  }

  async function performRAGQuery() {
    if (!ragQuery.trim()) return;
    
    isQuerying = true;
    ragResponse = null;

    try {
      const response = await fetch('/api/v1/ai/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          caseId: demoCase.id,
          maxResults: 5,
          minSimilarity: 0.7
        })
      });

      if (!response.ok) throw new Error('RAG query failed');
      
      ragResponse = await response.json();
      
    } catch (error) {
      console.error('RAG query error:', error);
      ragResponse = {
        answer: `Error: ${error.message}`,
        sources: [],
        processingTime: 0,
        model: 'error',
        tokensUsed: 0
      };
    } finally {
      isQuerying = false;
    }
  }
&lt;/script&gt;

&lt;svelte:head&gt;
  &lt;title&gt;RAG Integration Demo | Legal AI Platform&lt;/title&gt;
&lt;/svelte:head&gt;

&lt;div class="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-8"&gt;
  &lt;div class="max-w-6xl mx-auto space-y-8"&gt;
    
    &lt;!-- Header --&gt;
    &lt;div class="text-center"&gt;
      &lt;h1 class="text-4xl font-bold text-slate-800 mb-4"&gt;
        🧠 RAG Integration Demo
      &lt;/h1&gt;
      &lt;p class="text-xl text-slate-600"&gt;
        Full-Stack AI Pipeline: PostgreSQL + pgvector + MinIO + Redis + Gemma3
      &lt;/p&gt;
      &lt;div class="mt-4 p-4 bg-blue-100 rounded-lg"&gt;
        &lt;p class="text-sm font-mono text-blue-800"&gt;
          Demo Case: {demoCase.title} ({demoCase.id.slice(0, 8)}...)
        &lt;/p&gt;
      &lt;/div&gt;
    &lt;/div&gt;

    &lt;div class="grid grid-cols-1 lg:grid-cols-2 gap-8"&gt;
      
      &lt;!-- File Upload Section --&gt;
      &lt;Card&gt;
        &lt;CardHeader&gt;
          &lt;CardTitle class="flex items-center gap-2"&gt;
            📄 Document Ingestion Pipeline
          &lt;/CardTitle&gt;
        &lt;/CardHeader&gt;
        &lt;CardContent class="space-y-4"&gt;
          
          &lt;div class="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center"&gt;
            &lt;input
              bind:this={fileInput}
              type="file"
              accept=".pdf,.doc,.docx,.txt"
              class="hidden"
              on:change={handleFileUpload}
            /&gt;
            
            &lt;Button
              variant="outline"
              on:click={() =&gt; fileInput.click()}
              disabled={$uploadStatus.status === 'uploading' || $uploadStatus.status === 'processing'}
            &gt;
              Choose Document
            &lt;/Button&gt;
            
            &lt;p class="text-sm text-slate-500 mt-2"&gt;
              Supported: PDF, DOC, DOCX, TXT
            &lt;/p&gt;
          &lt;/div&gt;

          {#if $uploadStatus.status !== 'idle'}
            &lt;div class="space-y-2"&gt;
              &lt;div class="flex justify-between text-sm"&gt;
                &lt;span class="capitalize"&gt;{$uploadStatus.status}&lt;/span&gt;
                &lt;span&gt;{$uploadStatus.progress}%&lt;/span&gt;
              &lt;/div&gt;
              
              &lt;div class="w-full bg-slate-200 rounded-full h-2"&gt;
                &lt;div 
                  class="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style="width: {$uploadStatus.progress}%"
                &gt;&lt;/div&gt;
              &lt;/div&gt;
              
              &lt;p class="text-sm text-slate-600"&gt;{$uploadStatus.message}&lt;/p&gt;
              
              {#if $uploadStatus.fileId}
                &lt;p class="text-xs font-mono text-slate-500"&gt;
                  File ID: {$uploadStatus.fileId.slice(0, 16)}...
                &lt;/p&gt;
              {/if}
            &lt;/div&gt;
          {/if}

          &lt;div class="bg-slate-50 rounded-lg p-4 text-xs text-slate-600"&gt;
            &lt;h4 class="font-semibold mb-2"&gt;Pipeline Steps:&lt;/h4&gt;
            &lt;ol class="space-y-1 list-decimal list-inside"&gt;
              &lt;li&gt;MinIO pre-signed URL generation&lt;/li&gt;
              &lt;li&gt;Direct browser → MinIO upload&lt;/li&gt;
              &lt;li&gt;Redis job queue trigger&lt;/li&gt;
              &lt;li&gt;OCR text extraction (Tesseract)&lt;/li&gt;
              &lt;li&gt;NLP entity extraction (Legal-BERT)&lt;/li&gt;
              &lt;li&gt;Embedding generation (nomic-embed)&lt;/li&gt;
              &lt;li&gt;PostgreSQL + pgvector storage&lt;/li&gt;
              &lt;li&gt;Qdrant vector indexing&lt;/li&gt;
            &lt;/ol&gt;
          &lt;/div&gt;
        &lt;/CardContent&gt;
      &lt;/Card&gt;

      &lt;!-- RAG Query Section --&gt;
      &lt;Card&gt;
        &lt;CardHeader&gt;
          &lt;CardTitle class="flex items-center gap-2"&gt;
            🤖 RAG Query Interface
          &lt;/CardTitle&gt;
        &lt;/CardHeader&gt;
        &lt;CardContent class="space-y-4"&gt;
          
          &lt;div class="space-y-2"&gt;
            &lt;textarea
              bind:value={ragQuery}
              placeholder="Ask a question about your uploaded documents..."
              class="w-full p-3 border border-slate-300 rounded-lg resize-none"
              rows="4"
              disabled={isQuerying}
            &gt;&lt;/textarea&gt;
            
            &lt;Button
              on:click={performRAGQuery}
              disabled={!ragQuery.trim() || isQuerying}
              class="w-full"
            &gt;
              {#if isQuerying}
                🤔 Thinking...
              {:else}
                💡 Ask Gemma3
              {/if}
            &lt;/Button&gt;
          &lt;/div&gt;

          {#if ragResponse}
            &lt;div class="bg-white border border-slate-200 rounded-lg p-4 space-y-4"&gt;
              
              &lt;!-- Answer --&gt;
              &lt;div&gt;
                &lt;h4 class="font-semibold text-slate-700 mb-2"&gt;Answer:&lt;/h4&gt;
                &lt;p class="text-slate-800 leading-relaxed"&gt;{ragResponse.answer}&lt;/p&gt;
              &lt;/div&gt;

              &lt;!-- Sources --&gt;
              {#if ragResponse.sources.length &gt; 0}
                &lt;div&gt;
                  &lt;h4 class="font-semibold text-slate-700 mb-2"&gt;Sources:&lt;/h4&gt;
                  &lt;div class="space-y-2"&gt;
                    {#each ragResponse.sources as source}
                      &lt;div class="bg-slate-50 rounded p-3 text-sm"&gt;
                        &lt;div class="flex justify-between items-start mb-1"&gt;
                          &lt;h5 class="font-semibold"&gt;{source.title}&lt;/h5&gt;
                          &lt;span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs"&gt;
                            {Math.round(source.similarity * 100)}%
                          &lt;/span&gt;
                        &lt;/div&gt;
                        &lt;p class="text-slate-600"&gt;{source.excerpt}&lt;/p&gt;
                      &lt;/div&gt;
                    {/each}
                  &lt;/div&gt;
                &lt;/div&gt;
              {/if}

              &lt;!-- Metadata --&gt;
              &lt;div class="pt-2 border-t border-slate-200 text-xs text-slate-500"&gt;
                &lt;div class="flex justify-between"&gt;
                  &lt;span&gt;Model: {ragResponse.model}&lt;/span&gt;
                  &lt;span&gt;Tokens: {ragResponse.tokensUsed}&lt;/span&gt;
                  &lt;span&gt;Time: {ragResponse.processingTime}ms&lt;/span&gt;
                &lt;/div&gt;
              &lt;/div&gt;
            &lt;/div&gt;
          {/if}

          &lt;div class="bg-slate-50 rounded-lg p-4 text-xs text-slate-600"&gt;
            &lt;h4 class="font-semibold mb-2"&gt;RAG Process:&lt;/h4&gt;
            &lt;ol class="space-y-1 list-decimal list-inside"&gt;
              &lt;li&gt;Query embedding generation (nomic)&lt;/li&gt;
              &lt;li&gt;Vector similarity search (pgvector + Qdrant)&lt;/li&gt;
              &lt;li&gt;Context assembly from top matches&lt;/li&gt;
              &lt;li&gt;Prompt construction for Gemma3&lt;/li&gt;
              &lt;li&gt;Local LLM inference (Ollama)&lt;/li&gt;
              &lt;li&gt;Response streaming to frontend&lt;/li&gt;
            &lt;/ol&gt;
          &lt;/div&gt;
        &lt;/CardContent&gt;
      &lt;/Card&gt;
    &lt;/div&gt;

    &lt;!-- Architecture Diagram --&gt;
    &lt;Card&gt;
      &lt;CardHeader&gt;
        &lt;CardTitle&gt;🏗️ System Architecture Flow&lt;/CardTitle&gt;
      &lt;/CardHeader&gt;
      &lt;CardContent&gt;
        &lt;div class="bg-slate-50 rounded-lg p-6 font-mono text-sm"&gt;
          &lt;div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-center"&gt;
            
            &lt;div class="bg-blue-100 p-4 rounded"&gt;
              &lt;div class="font-bold text-blue-800"&gt;Frontend&lt;/div&gt;
              &lt;div class="text-blue-600 mt-2"&gt;
                SvelteKit 2&lt;br/&gt;
                bits-ui&lt;br/&gt;
                UnoCSS
              &lt;/div&gt;
            &lt;/div&gt;

            &lt;div class="bg-green-100 p-4 rounded"&gt;
              &lt;div class="font-bold text-green-800"&gt;Storage&lt;/div&gt;
              &lt;div class="text-green-600 mt-2"&gt;
                PostgreSQL&lt;br/&gt;
                pgvector&lt;br/&gt;
                MinIO&lt;br/&gt;
                Redis
              &lt;/div&gt;
            &lt;/div&gt;

            &lt;div class="bg-yellow-100 p-4 rounded"&gt;
              &lt;div class="font-bold text-yellow-800"&gt;Processing&lt;/div&gt;
              &lt;div class="text-yellow-600 mt-2"&gt;
                OCR (Tesseract)&lt;br/&gt;
                NLP (Legal-BERT)&lt;br/&gt;
                Embeddings&lt;br/&gt;
                Job Queue
              &lt;/div&gt;
            &lt;/div&gt;

            &lt;div class="bg-purple-100 p-4 rounded"&gt;
              &lt;div class="font-bold text-purple-800"&gt;AI/RAG&lt;/div&gt;
              &lt;div class="text-purple-600 mt-2"&gt;
                Gemma3 LLM&lt;br/&gt;
                Vector Search&lt;br/&gt;
                Context Assembly&lt;br/&gt;
                Response Stream
              &lt;/div&gt;
            &lt;/div&gt;
          &lt;/div&gt;
        &lt;/div&gt;
      &lt;/CardContent&gt;
    &lt;/Card&gt;
  &lt;/div&gt;
&lt;/div&gt;