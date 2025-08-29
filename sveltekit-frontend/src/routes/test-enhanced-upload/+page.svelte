<!--
  Test page for EnhancedLegalUpload - Preserves OCR + LegalBERT + RAG Flow
  Tests the new Superforms-powered upload component with enhanced processing
-->
<script lang="ts">
  import EnhancedLegalUpload from '$lib/components/upload/EnhancedLegalUpload.svelte';
  import { onMount } from 'svelte';
  import type { PageData } from './$types.js';

  // Mock page data for Superforms (you'd normally get this from +page.server.ts)
  let data: PageData = $state({
    form: {
      caseId: 'CASE-' + Date.now(),
      file: undefined,
      title: '',
      description: '',
      evidenceType: 'documents',
      tags: '',
      enableAiAnalysis: true,
      enableOcr: true,
      enableEmbeddings: true,
      isAdmissible: false
    }
  } as any);

  let uploadResults = $state<any[]>([]);
  let systemStatus = $state<any>({});
  let componentErrors = $state<string[]>([]);

  function handleUploadComplete(result: any) {
    console.log('✅ Enhanced upload completed:', result);
    uploadResults = [...uploadResults, result];
  }

  function handleUploadError(error: string) {
    console.error('❌ Enhanced upload error:', error);
    componentErrors = [...componentErrors, error];
  }

  onMount(async () => {
    // Check system status for enhanced processing services
    const services = [
      { name: 'PostgreSQL', url: 'http://localhost:5432', expected: false }, // Can't check directly
      { name: 'Enhanced RAG', url: 'http://localhost:8094/api/health', expected: true },
      { name: 'Semantic Architecture', url: 'http://localhost:8095/api/health', expected: true },
      { name: 'OCR Service', url: '/api/ocr/health', expected: true },
      { name: 'LegalBERT', url: '/api/ai/legal-analysis/health', expected: true }
    ];

    const statusChecks = await Promise.allSettled(
      services.map(async (service) => {
        if (!service.expected) return { name: service.name, status: 'unknown' };
        
        try {
          const response = await fetch(service.url, { 
            method: 'GET',
            timeout: 5000 
          });
          return { 
            name: service.name, 
            status: response.ok ? 'online' : 'offline',
            url: service.url 
          };
        } catch (error) {
          return { 
            name: service.name, 
            status: 'offline', 
            error: error.message,
            url: service.url 
          };
        }
      })
    );

    systemStatus = {
      services: statusChecks.map(result => 
        result.status === 'fulfilled' ? result.value : 
        { name: 'unknown', status: 'error' }
      ),
      timestamp: new Date().toISOString()
    };
  });
</script>

<svelte:head>
  <title>Enhanced Legal Upload Test - Preserves OCR + LegalBERT + RAG</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
  <div class="container mx-auto p-6 max-w-6xl">
    
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-gray-800 mb-2">
        🚀 Enhanced Legal Upload Test
      </h1>
      <p class="text-lg text-gray-600">
        Testing Superforms + OCR + LegalBERT + Enhanced RAG Flow
      </p>
      <div class="text-sm text-gray-500 mt-2">
        This component preserves your existing enhanced processing pipeline created yesterday
      </div>
    </div>

    <!-- System Status Dashboard -->
    <div class="bg-white rounded-lg shadow-sm border p-6 mb-8">
      <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
        <span class="text-2xl">🔧</span>
        System Status
      </h2>
      
      {#if systemStatus.services}
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {#each systemStatus.services as service}
            <div class="text-center p-3 rounded-lg {service.status === 'online' ? 'bg-green-50 border border-green-200' : service.status === 'offline' ? 'bg-red-50 border border-red-200' : 'bg-gray-50 border border-gray-200'}">
              <div class="font-medium text-sm">{service.name}</div>
              <div class="text-xs mt-1 {service.status === 'online' ? 'text-green-600' : service.status === 'offline' ? 'text-red-600' : 'text-gray-500'}">
                {service.status === 'online' ? '✅ Online' : 
                 service.status === 'offline' ? '❌ Offline' : 
                 service.status === 'unknown' ? '❓ Unknown' : '🚫 Error'}
              </div>
              {#if service.error}
                <div class="text-xs text-red-500 mt-1">{service.error}</div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="text-center text-gray-500 py-4">
          <div class="animate-spin inline-block w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full"></div>
          <p class="mt-2">Checking system status...</p>
        </div>
      {/if}
      
      <div class="mt-4 text-xs text-gray-500">
        Last checked: {systemStatus.timestamp ? new Date(systemStatus.timestamp).toLocaleString() : 'Never'}
      </div>
    </div>

    <!-- Enhanced Upload Component Test -->
    <div class="bg-white rounded-lg shadow-sm border mb-8">
      <div class="border-b px-6 py-4">
        <h2 class="text-xl font-semibold flex items-center gap-2">
          <span class="text-2xl">📤</span>
          Enhanced Legal Upload Component
        </h2>
        <p class="text-sm text-gray-600 mt-1">
          Superforms + Zod validation + Preserves existing OCR → LegalBERT → RAG pipeline
        </p>
      </div>
      
      <div class="p-6">
        <!-- Test different case IDs -->
        <div class="mb-4 p-4 bg-blue-50 rounded-lg">
          <h3 class="font-medium text-blue-800 mb-2">Test Configuration</h3>
          <div class="text-sm text-blue-700">
            <p><strong>Case ID:</strong> {data.form.caseId}</p>
            <p><strong>Enhanced Flow:</strong> ✅ Enabled (preserveExistingFlow = true)</p>
            <p><strong>Processing Pipeline:</strong> OCR → LegalBERT → Semantic Architecture → Enhanced RAG</p>
          </div>
        </div>

        <!-- The Enhanced Upload Component -->
        <EnhancedLegalUpload 
          data={data}
          caseId={data.form.caseId}
          preserveExistingFlow={true}
          onUploadComplete={handleUploadComplete}
          onUploadError={handleUploadError}
        />
      </div>
    </div>

    <!-- Error Display -->
    {#if componentErrors.length > 0}
      <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
        <h3 class="text-red-800 font-semibold mb-2">❌ Component Errors</h3>
        <ul class="text-sm text-red-700">
          {#each componentErrors as error}
            <li class="mb-1">• {error}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- Upload Results -->
    {#if uploadResults.length > 0}
      <div class="bg-white rounded-lg shadow-sm border">
        <div class="border-b px-6 py-4">
          <h2 class="text-xl font-semibold flex items-center gap-2">
            <span class="text-2xl">📊</span>
            Upload Results ({uploadResults.length})
          </h2>
        </div>
        
        <div class="p-6">
          <div class="space-y-4">
            {#each uploadResults as result, index}
              <div class="border rounded-lg p-4 bg-gradient-to-r from-green-50 to-blue-50">
                <div class="flex justify-between items-start mb-3">
                  <h3 class="font-semibold text-lg">
                    #{index + 1}: {result.filename || 'Unknown file'}
                  </h3>
                  <div class="flex gap-2">
                    <span class="px-3 py-1 text-xs rounded-full {result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                      {result.success ? '✅ Success' : '❌ Failed'}
                    </span>
                    {#if result.enhancedProcessing}
                      <span class="px-3 py-1 text-xs rounded-full bg-purple-100 text-purple-800">
                        🧠 Enhanced
                      </span>
                    {/if}
                  </div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <!-- Basic Info -->
                  <div>
                    <h4 class="font-medium mb-2">📄 File Information</h4>
                    <div class="space-y-1 text-gray-600">
                      {#if result.documentId}
                        <p><strong>Document ID:</strong> {result.documentId}</p>
                      {/if}
                      {#if result.caseId}
                        <p><strong>Case ID:</strong> {result.caseId}</p>
                      {/if}
                      {#if result.size}
                        <p><strong>Size:</strong> {(result.size / 1024).toFixed(1)} KB</p>
                      {/if}
                      {#if result.type}
                        <p><strong>Type:</strong> {result.type}</p>
                      {/if}
                    </div>
                  </div>
                  
                  <!-- Enhanced Analysis Results -->
                  <div>
                    <h4 class="font-medium mb-2">🧠 Enhanced Analysis</h4>
                    <div class="space-y-1 text-gray-600">
                      {#if result.analysis?.ocr}
                        <p>✅ OCR: {result.analysis.ocr.pages} pages, {result.analysis.ocr.averageConfidence}% confidence</p>
                      {/if}
                      {#if result.analysis?.legal}
                        <p>✅ LegalBERT: {result.analysis.legal.concepts?.length || 0} concepts</p>
                      {/if}
                      {#if result.analysis?.semantic}
                        <p>✅ Semantic: Embeddings generated</p>
                      {/if}
                      {#if result.webhookTriggered}
                        <p>✅ Webhook: Enhanced RAG pipeline triggered</p>
                      {/if}
                    </div>
                  </div>
                </div>
                
                <!-- Error Display -->
                {#if result.error}
                  <div class="mt-3 p-3 bg-red-100 rounded-md">
                    <p class="text-red-700 text-sm"><strong>Error:</strong> {result.error}</p>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}

    <!-- Debug Information -->
    <div class="mt-8 bg-gray-50 rounded-lg p-6 text-sm">
      <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
        <span class="text-xl">🔍</span>
        Debug Information
      </h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 class="font-medium mb-2">🏗️ Architecture</h3>
          <div class="space-y-1 text-gray-600">
            <p><strong>Frontend:</strong> SvelteKit 2 + Svelte 5 + Superforms</p>
            <p><strong>Validation:</strong> Zod schema with automatic error handling</p>
            <p><strong>Enhanced Flow:</strong> OCR → LegalBERT → Semantic Architecture</p>
            <p><strong>RAG Integration:</strong> Enhanced RAG service + webhooks</p>
          </div>
        </div>
        <div>
          <h3 class="font-medium mb-2">🌐 Service Endpoints</h3>
          <div class="space-y-1 text-gray-600">
            <p><strong>OCR:</strong> /api/ocr/extract</p>
            <p><strong>LegalBERT:</strong> /api/ai/legal-analysis</p>
            <p><strong>Semantic:</strong> http://localhost:8095/api/intelligent-todos</p>
            <p><strong>Enhanced RAG:</strong> http://localhost:8094/api/rag/document-ingest</p>
            <p><strong>Webhooks:</strong> /api/webhooks/legal-processing</p>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<style>
  /* Component uses Tailwind/UnoCSS classes - no additional styles needed */
  :global(.enhanced-legal-upload) {
    /* Ensure the component fits well in the test environment */
    max-width: none;
    margin: 0;
    background: white;
    border: none;
  }
</style>