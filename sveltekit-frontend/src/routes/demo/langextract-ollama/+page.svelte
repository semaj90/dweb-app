<script lang="ts">
  import { onMount } from 'svelte';
  import { $state } from 'svelte';
  
  let serviceStatus = $state<any>(null);
  let extractionResult = $state<any>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Form state
  let selectedAction = $state('extract');
  let selectedDocumentType = $state('contract');
  let selectedExtractionType = $state('entities');
  let selectedModel = $state('gemma2:2b');
  let inputText = $state(`This Software License Agreement ("Agreement") is entered into on January 15, 2024, between TechCorp Inc., a Delaware corporation ("Licensor"), and DataSystems LLC, a California limited liability company ("Licensee").

WHEREAS, Licensor owns certain proprietary software and related documentation; and
WHEREAS, Licensee desires to obtain a license to use such software;

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

1. GRANT OF LICENSE. Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software for internal business purposes only.

2. TERM. This Agreement shall commence on the Effective Date and shall continue for a period of three (3) years, unless earlier terminated in accordance with the provisions hereof.

3. FEES. In consideration for the rights granted hereunder, Licensee shall pay Licensor a license fee of $50,000 per year, payable annually in advance.

4. TERMINATION. Either party may terminate this Agreement upon thirty (30) days' written notice to the other party.`);

  const actions = [
    { value: 'extract', label: 'General Extraction' },
    { value: 'contract_terms', label: 'Contract Terms' },
    { value: 'case_citations', label: 'Case Citations' },
    { value: 'dates', label: 'Legal Dates' },
    { value: 'summary', label: 'Document Summary' },
    { value: 'risks', label: 'Risk Analysis' }
  ];

  const documentTypes = [
    { value: 'contract', label: 'Contract' },
    { value: 'case_law', label: 'Case Law' },
    { value: 'statute', label: 'Statute' },
    { value: 'evidence', label: 'Evidence' },
    { value: 'motion', label: 'Motion' },
    { value: 'brief', label: 'Brief' }
  ];

  const extractionTypes = [
    { value: 'entities', label: 'Legal Entities' },
    { value: 'summary', label: 'Summary' },
    { value: 'key_terms', label: 'Key Terms' },
    { value: 'obligations', label: 'Obligations' },
    { value: 'risks', label: 'Risk Factors' },
    { value: 'dates', label: 'Dates & Deadlines' }
  ];

  onMount(async () => {
    await fetchServiceStatus();
  });

  async function fetchServiceStatus() {
    try {
      const response = await fetch('/api/legal-ai/langextract');
      serviceStatus = await response.json();
    } catch (err) {
      console.error('Failed to fetch service status:', err);
      error = 'Failed to connect to LangExtract service';
    }
  }

  async function performExtraction() {
    if (!inputText.trim()) {
      error = 'Please enter some text to extract from';
      return;
    }

    loading = true;
    error = null;
    extractionResult = null;

    try {
      const requestBody: any = {
        action: selectedAction,
        text: inputText,
        documentType: selectedDocumentType,
        extractionType: selectedExtractionType,
        model: selectedModel
      };

      const response = await fetch('/api/legal-ai/langextract', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      const result = await response.json();
      
      if (result.success) {
        extractionResult = result.data;
      } else {
        error = result.error || 'Extraction failed';
      }
    } catch (err) {
      console.error('Extraction error:', err);
      error = 'Failed to perform extraction';
    } finally {
      loading = false;
    }
  }

  function loadSampleText(type: string) {
    const samples = {
      contract: `This Software License Agreement ("Agreement") is entered into on January 15, 2024, between TechCorp Inc., a Delaware corporation ("Licensor"), and DataSystems LLC, a California limited liability company ("Licensee").`,
      case_law: `In Smith v. Jones, 123 F.3d 456 (9th Cir. 2023), the Court of Appeals held that software license agreements must include clear termination provisions. The court noted that ambiguous termination clauses favor the licensee under California law.`,
      statute: `15 U.S.C. § 1692 et seq. The Fair Debt Collection Practices Act (FDCPA) prohibits debt collectors from engaging in abusive, deceptive, or unfair practices. Violations may result in civil penalties up to $1,000 per violation.`,
      evidence: `On March 15, 2024, at approximately 2:30 PM, Detective Johnson recovered a laptop computer (Evidence Item #2024-001) from the defendant's residence at 123 Main Street. The device contained encrypted files relevant to the securities fraud investigation.`
    };
    inputText = samples[type as keyof typeof samples] || samples.contract;
  }
</script>

<svelte:head>
  <title>LangExtract + Ollama Demo | Legal AI Platform</title>
</svelte:head>

<div class="container mx-auto p-6 space-y-8">
  <div class="header">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">LangExtract + Ollama Integration</h1>
    <p class="text-gray-600">Local LLM processing for legal document extraction using Ollama</p>
  </div>

  <!-- Service Status -->
  <div class="bg-white rounded-lg shadow-sm border p-6">
    <h2 class="text-xl font-semibold mb-4">Service Status</h2>
    {#if serviceStatus}
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="status-item">
          <span class="font-medium">Ollama Status:</span>
          <span class="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {serviceStatus.status?.ollama_available ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
            {serviceStatus.status?.ollama_available ? 'Available' : 'Unavailable'}
          </span>
        </div>
        <div class="status-item">
          <span class="font-medium">Available Models:</span>
          <span class="ml-2 text-sm text-gray-600">
            {serviceStatus.status?.available_models?.length || 0} models
          </span>
        </div>
      </div>
      
      {#if serviceStatus.status?.available_models?.length > 0}
        <div class="mt-4">
          <h3 class="font-medium text-sm text-gray-700 mb-2">Models:</h3>
          <div class="flex flex-wrap gap-2">
            {#each serviceStatus.status.available_models as model}
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {model}
              </span>
            {/each}
          </div>
        </div>
      {/if}
    {:else}
      <div class="text-gray-500">Loading service status...</div>
    {/if}
  </div>

  <!-- Extraction Form -->
  <div class="bg-white rounded-lg shadow-sm border p-6">
    <h2 class="text-xl font-semibold mb-4">Document Extraction</h2>
    
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <!-- Action Selection -->
      <div>
        <label for="action" class="block text-sm font-medium text-gray-700 mb-2">Action</label>
        <select id="action" bind:value={selectedAction} class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500">
          {#each actions as action}
            <option value={action.value}>{action.label}</option>
          {/each}
        </select>
      </div>

      <!-- Document Type -->
      <div>
        <label for="documentType" class="block text-sm font-medium text-gray-700 mb-2">Document Type</label>
        <select id="documentType" bind:value={selectedDocumentType} class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500">
          {#each documentTypes as docType}
            <option value={docType.value}>{docType.label}</option>
          {/each}
        </select>
      </div>

      <!-- Extraction Type -->
      <div>
        <label for="extractionType" class="block text-sm font-medium text-gray-700 mb-2">Extraction Type</label>
        <select id="extractionType" bind:value={selectedExtractionType} class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500">
          {#each extractionTypes as extType}
            <option value={extType.value}>{extType.label}</option>
          {/each}
        </select>
      </div>

      <!-- Model Selection -->
      <div>
        <label for="model" class="block text-sm font-medium text-gray-700 mb-2">Model</label>
        <select id="model" bind:value={selectedModel} class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500">
          {#if serviceStatus?.status?.available_models}
            {#each serviceStatus.status.available_models as model}
              <option value={model}>{model}</option>
            {/each}
          {:else}
            <option value="gemma2:2b">gemma2:2b</option>
            <option value="llama3.2:1b">llama3.2:1b</option>
          {/if}
        </select>
      </div>
    </div>

    <!-- Sample Text Buttons -->
    <div class="mb-4">
      <label class="block text-sm font-medium text-gray-700 mb-2">Load Sample Text:</label>
      <div class="flex flex-wrap gap-2">
        <button onclick={() => loadSampleText('contract')} class="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200">Contract</button>
        <button onclick={() => loadSampleText('case_law')} class="px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200">Case Law</button>
        <button onclick={() => loadSampleText('statute')} class="px-3 py-1 text-sm bg-purple-100 text-purple-700 rounded hover:bg-purple-200">Statute</button>
        <button onclick={() => loadSampleText('evidence')} class="px-3 py-1 text-sm bg-orange-100 text-orange-700 rounded hover:bg-orange-200">Evidence</button>
      </div>
    </div>

    <!-- Text Input -->
    <div class="mb-6">
      <label for="inputText" class="block text-sm font-medium text-gray-700 mb-2">Document Text</label>
      <textarea 
        id="inputText" 
        bind:value={inputText} 
        rows="8" 
        class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
        placeholder="Enter legal document text here..."
      ></textarea>
    </div>

    <!-- Extract Button -->
    <button 
      onclick={performExtraction}
      disabled={loading || !serviceStatus?.status?.ollama_available}
      class="w-full py-3 px-6 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
    >
      {loading ? 'Extracting...' : 'Extract Legal Information'}
    </button>

    {#if error}
      <div class="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
        <p class="text-red-800">{error}</p>
      </div>
    {/if}
  </div>

  <!-- Results -->
  {#if extractionResult}
    <div class="bg-white rounded-lg shadow-sm border p-6">
      <h2 class="text-xl font-semibold mb-4">Extraction Results</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div class="metric">
          <span class="text-sm text-gray-600">Confidence:</span>
          <span class="ml-2 font-semibold text-lg">{(extractionResult.confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="metric">
          <span class="text-sm text-gray-600">Processing Time:</span>
          <span class="ml-2 font-semibold">{extractionResult.processing_time}ms</span>
        </div>
        <div class="metric">
          <span class="text-sm text-gray-600">Model Used:</span>
          <span class="ml-2 font-semibold">{extractionResult.model_used}</span>
        </div>
      </div>

      <div class="bg-gray-50 rounded-lg p-4">
        <h3 class="font-medium text-gray-900 mb-3">Extracted Data:</h3>
        <pre class="text-sm text-gray-800 whitespace-pre-wrap overflow-auto max-h-96">{JSON.stringify(extractionResult.extracted_data, null, 2)}</pre>
      </div>
    </div>
  {/if}

  <!-- API Documentation -->
  <div class="bg-white rounded-lg shadow-sm border p-6">
    <h2 class="text-xl font-semibold mb-4">API Documentation</h2>
    
    <div class="space-y-4">
      <div>
        <h3 class="font-medium text-gray-900">Endpoint:</h3>
        <code class="text-sm bg-gray-100 px-2 py-1 rounded">POST /api/legal-ai/langextract</code>
      </div>
      
      <div>
        <h3 class="font-medium text-gray-900">Supported Actions:</h3>
        <ul class="list-disc list-inside text-sm text-gray-600 mt-2 space-y-1">
          <li><code>extract</code> - General legal entity extraction</li>
          <li><code>contract_terms</code> - Extract contract terms and obligations</li>
          <li><code>case_citations</code> - Extract case law citations and holdings</li>
          <li><code>dates</code> - Extract legal dates and deadlines</li>
          <li><code>summary</code> - Generate document summary</li>
          <li><code>risks</code> - Analyze risk factors</li>
        </ul>
      </div>

      <div>
        <h3 class="font-medium text-gray-900">Example Request:</h3>
        <pre class="text-sm bg-gray-100 p-3 rounded mt-2 overflow-auto"><code>{JSON.stringify({
          action: "contract_terms",
          text: "This Agreement is entered into...",
          documentType: "contract",
          extractionType: "obligations",
          model: "gemma2:2b"
        }, null, 2)}</code></pre>
      </div>
    </div>
  </div>
</div>

<style>
  .container {
    max-width: 1200px;
  }
  
  .status-item {
    display: flex;
    align-items: center;
  }
  
  .metric {
    display: flex;
    flex-direction: column;
  }
  
  @media (min-width: 768px) {
    .metric {
      flex-direction: row;
      align-items: center;
    }
  }
</style>