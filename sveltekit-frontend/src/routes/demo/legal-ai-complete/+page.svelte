<!--
  Complete Legal AI System Demo
  Demonstrates the full integration of:
  - Enhanced Context7 MCP Server
  - Advanced RAG with Custom Reranker
  - Qdrant + PGVector Hybrid Storage
  - XState Document Processing Workflows
-->
<script lang="ts">
  import LegalDocumentProcessor from '$lib/components/legal/LegalDocumentProcessor.svelte';
  import type { LegalDocument } from '$lib/services/legalRAGEngine';
  
  // Sample legal documents for testing
  const sampleDocuments: Partial<LegalDocument>[] = [
    {
      title: 'Software License Agreement',
      content: `SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into on January 15, 2024, between TechCorp Inc., a Delaware corporation ("Licensor"), and DataSoft LLC, a California limited liability company ("Licensee").

1. GRANT OF LICENSE
Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software solely for Licensee's internal business purposes.

2. LIABILITY LIMITATIONS
IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING WITHOUT LIMITATION, DAMAGES FOR LOSS OF PROFITS, GOODWILL, USE, DATA, OR OTHER INTANGIBLE LOSSES.

3. INDEMNIFICATION
Licensee agrees to indemnify, defend, and hold harmless Licensor from and against any and all claims, damages, liabilities, costs, and expenses arising out of or relating to Licensee's use of the Software.

4. TERMINATION
This Agreement may be terminated by either party upon thirty (30) days written notice to the other party. Upon termination, Licensee shall immediately cease all use of the Software and return or destroy all copies.

5. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.`,
      caseType: 'contract',
      jurisdiction: 'federal',
      caseId: 'CASE-2024-001'
    },
    {
      title: 'Employment Termination Dispute',
      content: `COMPLAINT FOR WRONGFUL TERMINATION

Plaintiff John Smith brings this action against Defendant MegaCorp Inc. for wrongful termination, discrimination, and retaliation in violation of state and federal employment laws.

FACTUAL ALLEGATIONS

1. Plaintiff was employed by Defendant as a Senior Software Engineer from March 2020 through December 2023.

2. Throughout his employment, Plaintiff received excellent performance reviews and was promoted twice.

3. In November 2023, Plaintiff reported safety violations and potential regulatory compliance issues to company management.

4. Within thirty days of making these reports, Plaintiff was terminated allegedly for "performance issues" despite his exemplary record.

5. Plaintiff believes his termination was in retaliation for his protected whistleblower activities.

CAUSES OF ACTION

Count I: Wrongful Termination in Violation of Public Policy
Count II: Retaliation under the Whistleblower Protection Act  
Count III: Breach of Implied Contract

PRAYER FOR RELIEF

Plaintiff seeks compensatory damages in excess of $250,000, punitive damages, attorney's fees, and such other relief as the Court deems just and proper.`,
      caseType: 'litigation',
      jurisdiction: 'state',
      caseId: 'CASE-2024-002'
    },
    {
      title: 'Corporate Compliance Audit Report',
      content: `CORPORATE COMPLIANCE AUDIT REPORT
Q4 2023 Regulatory Review

EXECUTIVE SUMMARY

This report presents the findings of our comprehensive compliance audit conducted for GlobalTech Corporation during Q4 2023. The audit evaluated adherence to federal securities regulations, data privacy requirements, and industry-specific compliance standards.

KEY FINDINGS

1. DATA PRIVACY COMPLIANCE
- GDPR compliance rate: 87% (Improvement needed)
- CCPA implementation: Complete
- Data breach response procedures: Adequate

2. SECURITIES REGULATIONS
- Form 10-K filing: Compliant
- Insider trading policies: Requires update
- Disclosure controls: Partially effective

3. RISK ASSESSMENT
- High-risk areas identified: 3
- Medium-risk areas: 7
- Regulatory penalties exposure: $150,000 potential

RECOMMENDATIONS

1. Immediate Actions Required:
   - Update data retention policies by March 2024
   - Enhance employee training on GDPR requirements
   - Implement additional internal controls for financial reporting

2. Long-term Improvements:
   - Establish dedicated compliance monitoring system
   - Quarterly risk assessment reviews
   - Enhanced documentation procedures

CONCLUSION

While the company demonstrates strong commitment to compliance, several areas require immediate attention to reduce regulatory risk and ensure continued adherence to applicable laws and regulations.`,
      caseType: 'compliance',
      jurisdiction: 'federal',
      caseId: 'CASE-2024-003'
    }
  ];

  // Component state
  let selectedDocument = $state<Partial<LegalDocument> | undefined>(undefined);
  let processingResults = $state<any[]>([]);
  let isShowingResults = $state(false);

  // Event handlers
  function selectDocument(doc: Partial<LegalDocument>) {
    selectedDocument = doc;
    isShowingResults = false;
  }

  function onProcessingComplete(result: any) {
    processingResults = [...processingResults, {
      ...result,
      timestamp: new Date().toISOString(),
      document: selectedDocument
    }];
    isShowingResults = true;
  }

  function onProcessingError(errors: string[]) {
    console.error('Processing failed:', errors);
    // You could show a toast notification here
  }

  function clearResults() {
    processingResults = [];
    isShowingResults = false;
  }
</script>

<svelte:head>
  <title>Legal AI System Demo - Complete Integration</title>
  <meta name="description" content="Demonstration of the complete legal AI system with Context7 MCP, RAG, Qdrant, and XState integration." />
</svelte:head>

<div class="container mx-auto px-4 py-8 max-w-7xl">
  <!-- Header -->
  <header class="mb-8">
    <h1 class="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
      Legal AI System Demo
    </h1>
    <p class="text-xl text-gray-600 dark:text-gray-400 mb-6">
      Complete integration of Context7 MCP, Advanced RAG, Qdrant + PGVector, and XState workflows
    </p>
    
    <!-- System Architecture Overview -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <div class="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg text-center">
        <div class="text-2xl mb-2">🧠</div>
        <div class="font-semibold text-blue-900 dark:text-blue-100">Context7 MCP</div>
        <div class="text-sm text-blue-700 dark:text-blue-300">Stack-aware analysis</div>
      </div>
      <div class="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg text-center">
        <div class="text-2xl mb-2">🔍</div>
        <div class="font-semibold text-green-900 dark:text-green-100">Advanced RAG</div>
        <div class="text-sm text-green-700 dark:text-green-300">Custom reranker</div>
      </div>
      <div class="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg text-center">
        <div class="text-2xl mb-2">🗄️</div>
        <div class="font-semibold text-purple-900 dark:text-purple-100">Hybrid Storage</div>
        <div class="text-sm text-purple-700 dark:text-purple-300">Qdrant + PGVector</div>
      </div>
      <div class="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg text-center">
        <div class="text-2xl mb-2">⚡</div>
        <div class="font-semibold text-yellow-900 dark:text-yellow-100">XState Flow</div>
        <div class="text-sm text-yellow-700 dark:text-yellow-300">State machines</div>
      </div>
    </div>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
    <!-- Document Selection Panel -->
    <div class="space-y-6">
      <div class="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
        <h2 class="text-2xl font-semibold mb-4">Select Document to Process</h2>
        <div class="space-y-4">
          {#each sampleDocuments as doc, index}
            <button
              onclick={() => selectDocument(doc)}
              class="w-full text-left p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors {selectedDocument === doc ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700'}"
            >
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <div class="font-semibold text-gray-900 dark:text-gray-100">
                    {doc.title}
                  </div>
                  <div class="text-sm text-gray-500 mt-1">
                    {doc.caseType} • {doc.jurisdiction} • {doc.caseId}
                  </div>
                  <div class="text-sm text-gray-600 dark:text-gray-400 mt-2">
                    {doc.content?.slice(0, 150)}...
                  </div>
                </div>
                <div class="ml-4">
                  <div class="w-3 h-3 rounded-full {selectedDocument === doc ? 'bg-blue-500' : 'bg-gray-300'}"></div>
                </div>
              </div>
            </button>
          {/each}
        </div>
      </div>

      <!-- Processing Results Summary -->
      {#if processingResults.length > 0}
        <div class="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-2xl font-semibold">Processing History</h2>
            <button
              onclick={clearResults}
              class="text-sm text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
            >
              Clear History
            </button>
          </div>
          <div class="space-y-3">
            {#each processingResults as result, index}
              <div class="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="flex items-center justify-between mb-2">
                  <div class="font-medium text-gray-900 dark:text-gray-100">
                    {result.document?.title}
                  </div>
                  <div class="text-xs text-gray-500">
                    {new Date(result.timestamp).toLocaleTimeString()}
                  </div>
                </div>
                <div class="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span class="text-gray-600 dark:text-gray-400">Risk Score:</span>
                    <span class="font-medium ml-1">{result.riskScore || 'N/A'}</span>
                  </div>
                  <div>
                    <span class="text-gray-600 dark:text-gray-400">Duration:</span>
                    <span class="font-medium ml-1">{result.processingDuration ? `${(result.processingDuration / 1000).toFixed(1)}s` : 'N/A'}</span>
                  </div>
                </div>
                {#if result.entities}
                  <div class="mt-2 text-xs">
                    <span class="text-gray-600 dark:text-gray-400">Entities:</span>
                    <span class="ml-1">{result.entities.parties.length} parties, {result.entities.clauses.length} clauses</span>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- Document Processor Panel -->
    <div class="space-y-6">
      {#if selectedDocument}
        <LegalDocumentProcessor
          bind:document={selectedDocument}
          onComplete={onProcessingComplete}
          onError={onProcessingError}
          autoStart={false}
        />
      {:else}
        <div class="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6 text-center">
          <div class="text-6xl mb-4">📄</div>
          <h3 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Select a Document
          </h3>
          <p class="text-gray-600 dark:text-gray-400">
            Choose a sample document from the left panel to begin processing
          </p>
        </div>
      {/if}
    </div>
  </div>

  <!-- System Integration Details -->
  <div class="mt-12 bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
    <h2 class="text-2xl font-semibold mb-6">System Integration Details</h2>
    
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <!-- Context7 MCP Integration -->
      <div class="p-4 border rounded-lg">
        <h3 class="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-100">
          🧠 Context7 MCP
        </h3>
        <ul class="text-sm space-y-2 text-gray-600 dark:text-gray-400">
          <li>• Legal document analysis</li>
          <li>• Stack-aware recommendations</li>
          <li>• Compliance report generation</li>
          <li>• Legal precedent suggestions</li>
          <li>• Entity extraction</li>
        </ul>
      </div>

      <!-- Advanced RAG -->
      <div class="p-4 border rounded-lg">
        <h3 class="font-semibold text-lg mb-3 text-green-900 dark:text-green-100">
          🔍 Advanced RAG
        </h3>
        <ul class="text-sm space-y-2 text-gray-600 dark:text-gray-400">
          <li>• Custom legal reranker</li>
          <li>• Jurisdiction-aware scoring</li>
          <li>• Precedent matching</li>
          <li>• Risk-based relevance</li>
          <li>• Confidence scoring</li>
        </ul>
      </div>

      <!-- Hybrid Storage -->
      <div class="p-4 border rounded-lg">
        <h3 class="font-semibold text-lg mb-3 text-purple-900 dark:text-purple-100">
          🗄️ Hybrid Storage
        </h3>
        <ul class="text-sm space-y-2 text-gray-600 dark:text-gray-400">
          <li>• Qdrant vector search</li>
          <li>• PGVector ACID compliance</li>
          <li>• Metadata filtering</li>
          <li>• Auto-sync mechanisms</li>
          <li>• Performance optimization</li>
        </ul>
      </div>

      <!-- XState Workflows -->
      <div class="p-4 border rounded-lg">
        <h3 class="font-semibold text-lg mb-3 text-yellow-900 dark:text-yellow-100">
          ⚡ XState Workflows
        </h3>
        <ul class="text-sm space-y-2 text-gray-600 dark:text-gray-400">
          <li>• Parallel processing</li>
          <li>• Error recovery</li>
          <li>• Progress tracking</li>
          <li>• State visualization</li>
          <li>• Retry logic</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Technical Implementation Notes -->
  <div class="mt-8 bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
    <h3 class="font-semibold text-lg mb-4">Technical Implementation</h3>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
      <div>
        <h4 class="font-medium mb-2">Frontend Stack</h4>
        <ul class="space-y-1 text-gray-600 dark:text-gray-400">
          <li>• SvelteKit 2 with Svelte 5 runes</li>
          <li>• XState for state management</li>
          <li>• UnoCSS for styling</li>
          <li>• TypeScript for type safety</li>
        </ul>
      </div>
      <div>
        <h4 class="font-medium mb-2">Backend Integration</h4>
        <ul class="space-y-1 text-gray-600 dark:text-gray-400">
          <li>• Context7 MCP server (port 40000)</li>
          <li>• Qdrant vector database</li>
          <li>• PostgreSQL with pgvector</li>
          <li>• Ollama for local LLM inference</li>
        </ul>
      </div>
    </div>
  </div>
</div>