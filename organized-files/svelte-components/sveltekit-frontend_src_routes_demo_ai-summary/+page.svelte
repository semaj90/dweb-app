<script lang="ts">
  import AISummaryReader from '$lib/components/legal/AISummaryReader.svelte';
  import EvidenceReportSummary from '$lib/components/legal/EvidenceReportSummary.svelte';
  import CaseSynthesisWorkflow from '$lib/components/legal/CaseSynthesisWorkflow.svelte';
  import { Brain, FileText, GitMerge, Play } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let activeTab = 'summary-reader';
  let demoMode = 'interactive';

  // Mock data for demonstrations
  const mockEvidenceReport = {
    id: 'EVID-2024-001',
    title: 'Digital Forensics Analysis - Data Breach Investigation',
    type: 'digital_forensics' as const,
    status: 'completed' as const,
    priority: 'critical' as const,
    createdAt: '2024-01-15T10:00:00Z',
    updatedAt: '2024-01-20T16:30:00Z',
    analyst: {
      name: 'Dr. Sarah Chen',
      credentials: 'CISSP, EnCE, GCFA',
      department: 'Digital Forensics Unit'
    },
    evidence: {
      itemNumber: 'ITEM-2024-0056',
      description: 'Laptop computer seized from suspect\'s residence containing encrypted data and deleted files',
      chainOfCustody: ['Detective J. Rodriguez', 'Evidence Technician M. Williams', 'Forensic Analyst S. Chen'],
      dateCollected: '2024-01-10T14:30:00Z',
      location: '1247 Oak Street, Apartment 3B'
    },
    methodology: {
      procedures: ['Forensic imaging using write-blockers', 'Hash verification', 'Deleted file recovery', 'Encryption analysis'],
      tools: ['FTK Imager', 'EnCase Forensic', 'Autopsy', 'Hashcat'],
      standards: ['NIST SP 800-86', 'ISO/IEC 27037', 'ACPO Guidelines']
    },
    findings: {
      summary: 'Analysis revealed extensive evidence of unauthorized access to corporate networks, data exfiltration activities, and attempts to cover digital tracks. Recovered deleted files contain customer databases and financial records totaling over 50,000 individual records.',
      keyPoints: [
        'Recovered 127 deleted files containing sensitive customer data',
        'Network logs show unauthorized access to 15 different corporate systems',
        'Evidence of data encryption and transfer to external servers',
        'Browser history reveals research into data selling marketplaces',
        'Cryptocurrency wallet traces linked to payments for stolen data'
      ],
      confidence: 0.94,
      limitations: [
        'Some encrypted partitions remain inaccessible without additional decryption keys',
        'Network activity logs are incomplete due to VPN usage',
        'Potential evidence may exist on cloud storage accounts not yet identified'
      ]
    },
    legalImplications: {
      charges: [
        'Computer Fraud and Abuse Act (18 U.S.C. § 1030)',
        'Identity Theft (18 U.S.C. § 1028)',
        'Wire Fraud (18 U.S.C. § 1343)',
        'Money Laundering (18 U.S.C. § 1956)'
      ],
      precedents: [
        'United States v. Morris (1991)',
        'United States v. Nosal (2016)',
        'United States v. Valle (2014)'
      ],
      challengePoints: [
        'Defense may challenge forensic methodology',
        'Encryption may be claimed as privacy protection',
        'Attribution to specific individual may be disputed'
      ]
    },
    attachments: [
      { id: 'att1', name: 'Forensic_Image_Hash_Report.pdf', type: 'PDF', size: 245760 },
      { id: 'att2', name: 'Deleted_Files_Analysis.xlsx', type: 'Excel', size: 1048576 },
      { id: 'att3', name: 'Network_Traffic_Logs.csv', type: 'CSV', size: 2097152 },
      { id: 'att4', name: 'Encryption_Analysis_Report.docx', type: 'Word', size: 524288 }
    ]
  };

  const mockDocuments = [
    {
      id: 'DOC-001',
      title: 'Initial Incident Report',
      type: 'report' as const,
      content: 'On January 10, 2024, XYZ Corporation reported a suspected data breach affecting their customer database...',
      metadata: {
        dateCreated: '2024-01-10T09:00:00Z',
        author: 'Detective J. Rodriguez',
        relevanceScore: 0.95
      }
    },
    {
      id: 'DOC-002',
      title: 'Witness Statement - IT Administrator',
      type: 'witness_statement' as const,
      content: 'I first noticed unusual network activity on January 8th when our monitoring systems detected...',
      metadata: {
        dateCreated: '2024-01-12T14:00:00Z',
        author: 'Michael Thompson (Witness)',
        relevanceScore: 0.88
      }
    },
    {
      id: 'DOC-003',
      title: 'Financial Analysis Report',
      type: 'expert_testimony' as const,
      content: 'Forensic accounting analysis reveals systematic extraction of funds through cryptocurrency...',
      metadata: {
        dateCreated: '2024-01-18T11:00:00Z',
        author: 'CPA Jennifer Walsh',
        relevanceScore: 0.92
      }
    }
  ];

  const mockEvidenceReports = [mockEvidenceReport];

  const sampleLegalContent = `# Legal Analysis Report: Cybercrime Investigation

## Case Overview
This comprehensive analysis examines evidence collected in the investigation of a sophisticated data breach targeting XYZ Corporation. The investigation reveals a systematic cyber attack involving unauthorized access, data theft, and financial fraud.

## Executive Summary
Digital forensics analysis has uncovered substantial evidence of criminal activity including:
- Unauthorized access to corporate networks affecting 15 different systems
- Theft of customer data involving over 50,000 individual records
- Financial fraud through cryptocurrency transactions
- Sophisticated attempts to destroy digital evidence

## Key Evidence Points

### Digital Forensics Findings
The forensic examination of the suspect's laptop computer revealed extensive evidence of criminal activity. Advanced forensic techniques were employed to recover deleted files and analyze encrypted partitions.

### Network Analysis
Comprehensive analysis of network logs demonstrates unauthorized access patterns consistent with advanced persistent threat (APT) tactics. The investigation revealed:
- Multiple points of entry into corporate networks
- Data exfiltration over extended periods
- Use of sophisticated evasion techniques

### Financial Investigation
Forensic accounting analysis traced cryptocurrency transactions linking stolen data sales to the suspect's digital wallets. Evidence includes:
- Transaction records showing payments for stolen data
- Connection to underground marketplace activities
- Quantifiable financial damages exceeding $2.5 million

## Legal Implications

### Federal Charges
The evidence supports multiple federal charges under:
- Computer Fraud and Abuse Act (CFAA)
- Wire Fraud statutes
- Identity Theft provisions
- Money Laundering regulations

### Prosecution Strategy
Recommended prosecution approach focuses on the systematic nature of the violations and quantifiable damages. The strong digital evidence chain provides an excellent foundation for successful prosecution.

## Risk Assessment
Potential defense challenges include:
- Technical complexity requiring expert testimony
- Encryption and privacy arguments
- Attribution and identity verification

## Recommendations
1. Prepare comprehensive expert testimony for technical evidence
2. Develop clear visualizations for jury presentation
3. Coordinate with financial crimes specialists
4. Ensure robust chain of custody documentation
`;
</script>

<svelte:head>
  <title>AI Summary & Analysis Demo - Legal AI System</title>
  <meta name="description" content="Demonstration of AI-powered legal document summary and analysis components" />
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <div class="max-w-7xl mx-auto py-8 px-4">
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">
        AI Summary & Analysis Demo
      </h1>
      <p class="text-xl text-gray-600 max-w-3xl mx-auto">
        Experience advanced AI-powered legal document analysis with XState-driven workflows for evidence reports, 
        case synthesis, and intelligent summary generation.
      </p>
    </div>

    <!-- Navigation -->
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm mb-8">
      <div class="border-b border-gray-200">
        <nav class="flex space-x-8 px-6" aria-label="Tabs">
          <button
            class="border-b-2 py-4 px-1 text-sm font-medium transition-colors"
            class:border-blue-500={activeTab === 'summary-reader'}
            class:text-blue-600={activeTab === 'summary-reader'}
            class:border-transparent={activeTab !== 'summary-reader'}
            class:text-gray-500={activeTab !== 'summary-reader'}
            onclick={() => activeTab = 'summary-reader'}
          >
            <div class="flex items-center gap-2">
              <Brain class="w-4 h-4" />
              AI Summary Reader
            </div>
          </button>
          
          <button
            class="border-b-2 py-4 px-1 text-sm font-medium transition-colors"
            class:border-blue-500={activeTab === 'evidence-report'}
            class:text-blue-600={activeTab === 'evidence-report'}
            class:border-transparent={activeTab !== 'evidence-report'}
            class:text-gray-500={activeTab !== 'evidence-report'}
            onclick={() => activeTab = 'evidence-report'}
          >
            <div class="flex items-center gap-2">
              <FileText class="w-4 h-4" />
              Evidence Report
            </div>
          </button>
          
          <button
            class="border-b-2 py-4 px-1 text-sm font-medium transition-colors"
            class:border-blue-500={activeTab === 'case-synthesis'}
            class:text-blue-600={activeTab === 'case-synthesis'}
            class:border-transparent={activeTab !== 'case-synthesis'}
            class:text-gray-500={activeTab !== 'case-synthesis'}
            onclick={() => activeTab = 'case-synthesis'}
          >
            <div class="flex items-center gap-2">
              <GitMerge class="w-4 h-4" />
              Case Synthesis
            </div>
          </button>
        </nav>
      </div>

      <!-- Tab Content -->
      <div class="p-6">
        {#if activeTab === 'summary-reader'}
          <div class="space-y-6">
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 class="text-lg font-semibold text-blue-900 mb-2">AI Summary Reader</h3>
              <p class="text-blue-800">
                Intelligent document analysis with XState-powered reading workflows. Features voice synthesis, 
                section navigation, entity extraction, and comprehensive legal analysis.
              </p>
            </div>

            <AISummaryReader
              initialContent={sampleLegalContent}
              documentType="report"
              caseId="CASE-2024-001"
            />
          </div>

        {:else if activeTab === 'evidence-report'}
          <div class="space-y-6">
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
              <h3 class="text-lg font-semibold text-green-900 mb-2">Evidence Report Analysis</h3>
              <p class="text-green-800">
                Specialized component for evidence report summaries with forensic analysis details, 
                chain of custody tracking, and legal implications assessment.
              </p>
            </div>

            <EvidenceReportSummary
              evidenceId="EVID-2024-001"
              caseId="CASE-2024-001"
              reportData={mockEvidenceReport}
              allowExport={true}
            />
          </div>

        {:else if activeTab === 'case-synthesis'}
          <div class="space-y-6">
            <div class="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h3 class="text-lg font-semibold text-purple-900 mb-2">Case Synthesis Workflow</h3>
              <p class="text-purple-800">
                Comprehensive case analysis combining multiple documents and evidence reports into 
                strategic legal assessments with timeline reconstruction and prosecution recommendations.
              </p>
            </div>

            <CaseSynthesisWorkflow
              caseId="CASE-2024-001"
              documents={mockDocuments}
              evidenceReports={mockEvidenceReports}
            />
          </div>
        {/if}
      </div>
    </div>

    <!-- Features Overview -->
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
      <h2 class="text-2xl font-bold text-gray-900 mb-6">Key Features</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="text-center">
          <div class="w-12 h-12 mx-auto bg-blue-100 rounded-lg flex items-center justify-center mb-4">
            <Brain class="w-6 h-6 text-blue-600" />
          </div>
          <h3 class="text-lg font-semibold text-gray-900 mb-2">XState Workflows</h3>
          <p class="text-gray-600">
            Robust state management for complex reading and analysis workflows with error handling and progress tracking.
          </p>
        </div>

        <div class="text-center">
          <div class="w-12 h-12 mx-auto bg-green-100 rounded-lg flex items-center justify-center mb-4">
            <FileText class="w-6 h-6 text-green-600" />
          </div>
          <h3 class="text-lg font-semibold text-gray-900 mb-2">Legal Context Awareness</h3>
          <p class="text-gray-600">
            Specialized analysis for legal documents with entity extraction, precedent identification, and compliance checking.
          </p>
        </div>

        <div class="text-center">
          <div class="w-12 h-12 mx-auto bg-purple-100 rounded-lg flex items-center justify-center mb-4">
            <GitMerge class="w-6 h-6 text-purple-600" />
          </div>
          <h3 class="text-lg font-semibold text-gray-900 mb-2">Synthesis Engine</h3>
          <p class="text-gray-600">
            Advanced case synthesis combining multiple sources into comprehensive strategic assessments and recommendations.
          </p>
        </div>
      </div>

      <div class="mt-8 p-4 bg-gray-50 rounded-lg">
        <h3 class="text-lg font-semibold text-gray-900 mb-3">Technical Implementation</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700">
          <div>
            <h4 class="font-medium text-gray-900 mb-2">State Management</h4>
            <ul class="space-y-1">
              <li>• XState machines for complex workflows</li>
              <li>• Hierarchical state management</li>
              <li>• Error boundaries and recovery</li>
              <li>• Progress tracking and resumption</li>
            </ul>
          </div>
          <div>
            <h4 class="font-medium text-gray-900 mb-2">AI Integration</h4>
            <ul class="space-y-1">
              <li>• RAG-powered content analysis</li>
              <li>• Entity extraction and classification</li>
              <li>• Confidence scoring</li>
              <li>• Legal context understanding</li>
            </ul>
          </div>
          <div>
            <h4 class="font-medium text-gray-900 mb-2">User Experience</h4>
            <ul class="space-y-1">
              <li>• Voice synthesis for accessibility</li>
              <li>• Interactive section navigation</li>
              <li>• Real-time progress tracking</li>
              <li>• Export and sharing capabilities</li>
            </ul>
          </div>
          <div>
            <h4 class="font-medium text-gray-900 mb-2">Legal Features</h4>
            <ul class="space-y-1">
              <li>• Evidence chain tracking</li>
              <li>• Legal precedent identification</li>
              <li>• Risk assessment and mitigation</li>
              <li>• Strategic recommendations</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <!-- Integration Notes -->
    <div class="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold text-yellow-800 mb-3">Integration with Legal AI System</h3>
      <div class="text-yellow-700 space-y-2">
        <p>
          <strong>RAG Integration:</strong> These components integrate seamlessly with the RAG-MCP system for document retrieval and analysis.
        </p>
        <p>
          <strong>Case Management:</strong> Designed to work with the legal case management database schema and evidence tracking systems.
        </p>
        <p>
          <strong>Multi-Agent AI:</strong> Compatible with Autogen and CrewAI workflows for enhanced analysis capabilities.
        </p>
        <p>
          <strong>API Endpoints:</strong> Connects to <code>/api/legal/</code> endpoints for real-time data and analysis.
        </p>
      </div>
    </div>
  </div>
</div>

<style>
  /* Custom styles for demo components */
  :global(.prose) {
    max-width: none
  }
  
  :global(.prose p) {
    margin-bottom: 1rem;
  }
  
  :global(.prose h1, .prose h2, .prose h3) {
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
  }
</style>