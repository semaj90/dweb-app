<script lang="ts">
  interface Props {
    evidenceId: string;
    caseId: string;
    reportData: EvidenceReport;
    allowExport: boolean
  }
  let {
    evidenceId,
    caseId,
    reportData,
    allowExport = true
  } = $props();



  import { aiSummaryMachine } from "$lib/machines/aiSummaryMachine";
  import { useMachine } from "@xstate/svelte";
  import {
    AlertTriangle,
    CheckCircle,
    Download,
    Eye,
    FileText,
    Scale,
    Target,
  } from "lucide-svelte";
  import AISummaryReader from "./AISummaryReader.svelte";


  interface EvidenceReport {
    id: string
    title: string
    type:
      | "digital_forensics"
      | "dna_analysis"
      | "ballistics"
      | "financial"
      | "document_analysis"
      | "witness_statement";
    status: "pending" | "in_progress" | "completed" | "reviewed" | "challenged";
    priority: "low" | "medium" | "high" | "critical";
    createdAt: string
    updatedAt: string
    analyst: {
      name: string
      credentials: string
      department: string
    };
    evidence: {
      itemNumber: string
      description: string
      chainOfCustody: string[];
      dateCollected: string
      location: string
    };
    methodology: {
      procedures: string[];
      tools: string[];
      standards: string[];
    };
    findings: {
      summary: string
      keyPoints: string[];
      confidence: number
      limitations: string[];
    };
    legalImplications: {
      charges: string[];
      precedents: string[];
      challengePoints: string[];
    };
    attachments: {
      id: string
      name: string
      type: string
      size: number
    }[];
  }

  const { state, send } = useMachine(aiSummaryMachine);

  // Generate comprehensive content for AI analysis
  let analysisContent = $derived(generateAnalysisContent(reportData));

  function generateAnalysisContent(report: EvidenceReport): string {
    return `
EVIDENCE ANALYSIS REPORT

Case ID: ${caseId}
Evidence Item: ${report.evidence.itemNumber}
Report Type: ${report.type.replace("_", " ").toUpperCase()}
Priority Level: ${report.priority.toUpperCase()}
Status: ${report.status.toUpperCase()}

ANALYST INFORMATION
Name: ${report.analyst.name}
Credentials: ${report.analyst.credentials}
Department: ${report.analyst.department}

EVIDENCE DETAILS
Description: ${report.evidence.description}
Collection Date: ${report.evidence.dateCollected}
Collection Location: ${report.evidence.location}
Chain of Custody: ${report.evidence.chainOfCustody.join(" → ")}

METHODOLOGY
Procedures: ${report.methodology.procedures.join(", ")}
Tools Used: ${report.methodology.tools.join(", ")}
Standards Applied: ${report.methodology.standards.join(", ")}

FINDINGS
${report.findings.summary}

Key Points:
${report.findings.keyPoints.map((point) => `• ${point}`).join("\n")}

Confidence Level: ${Math.round(report.findings.confidence * 100)}%

Limitations:
${report.findings.limitations.map((limitation) => `• ${limitation}`).join("\n")}

LEGAL IMPLICATIONS
Potential Charges: ${report.legalImplications.charges.join(", ")}
Relevant Precedents: ${report.legalImplications.precedents.join(", ")}
Challenge Points: ${report.legalImplications.challengePoints.join(", ")}

ATTACHMENTS
${report.attachments.map((att) => `• ${att.name} (${att.type})`).join("\n")}
    `.trim();
  }

  function getStatusColor(status: string) {
    switch (status) {
      case "completed":
        return "text-green-600 bg-green-100";
      case "reviewed":
        return "text-blue-600 bg-blue-100";
      case "in_progress":
        return "text-yellow-600 bg-yellow-100";
      case "challenged":
        return "text-red-600 bg-red-100";
      case "pending":
        return "text-gray-600 bg-gray-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  }

  function getPriorityColor(priority: string) {
    switch (priority) {
      case "critical":
        return "text-red-600 bg-red-100 border-red-200";
      case "high":
        return "text-orange-600 bg-orange-100 border-orange-200";
      case "medium":
        return "text-yellow-600 bg-yellow-100 border-yellow-200";
      case "low":
        return "text-gray-600 bg-gray-100 border-gray-200";
      default:
        return "text-gray-600 bg-gray-100 border-gray-200";
    }
  }

  function getTypeIcon(type: string) {
    switch (type) {
      case "digital_forensics":
        return "💻";
      case "dna_analysis":
        return "🧬";
      case "ballistics":
        return "🔫";
      case "financial":
        return "💰";
      case "document_analysis":
        return "📄";
      case "witness_statement":
        return "👤";
      default:
        return "📋";
    }
  }

  function exportReport() {
    const content = `# Evidence Analysis Report Export\n\n${analysisContent}`;
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `evidence-report-${evidenceId}-${new Date().toISOString().split("T")[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
</script>

<div class="evidence-report-summary space-y-6">
  <!-- Report Header -->
  <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
    <div class="flex items-start justify-between mb-4">
      <div class="flex items-start gap-4">
        <div class="text-4xl">{getTypeIcon(reportData.type)}</div>
        <div>
          <h2 class="text-2xl font-bold text-gray-900 mb-2">
            {reportData.title}
          </h2>
          <div class="flex items-center gap-4 text-sm text-gray-600">
            <span>Case: {caseId}</span>
            <span>Evidence: {reportData.evidence.itemNumber}</span>
            <span
              >Updated: {new Date(
                reportData.updatedAt
              ).toLocaleDateString()}</span
            >
          </div>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <div class="text-right">
          <div class="flex items-center gap-2 mb-1">
            <span
              class="px-3 py-1 rounded-full text-sm font-medium {getStatusColor(
                reportData.status
              )}"
            >
              {reportData.status.replace("_", " ").toUpperCase()}
            </span>
          </div>
          <div
            class="px-3 py-1 border rounded-full text-sm font-medium {getPriorityColor(
              reportData.priority
            )}"
          >
            {reportData.priority.toUpperCase()} PRIORITY
          </div>
        </div>

        {#if allowExport}
          <button
            on:click={exportReport}
            class="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors"
            title="Export Report"
          >
            <Download class="w-5 h-5" />
          </button>
        {/if}
      </div>
    </div>

    <!-- Quick Stats -->
    <div
      class="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg"
    >
      <div class="text-center">
        <div class="text-lg font-semibold text-gray-900">
          {Math.round(reportData.findings.confidence * 100)}%
        </div>
        <div class="text-sm text-gray-600">Confidence</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-semibold text-gray-900">
          {reportData.evidence.chainOfCustody.length}
        </div>
        <div class="text-sm text-gray-600">Chain Links</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-semibold text-gray-900">
          {reportData.methodology.procedures.length}
        </div>
        <div class="text-sm text-gray-600">Procedures</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-semibold text-gray-900">
          {reportData.attachments.length}
        </div>
        <div class="text-sm text-gray-600">Attachments</div>
      </div>
    </div>
  </div>

  <!-- Evidence Details -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Evidence Information -->
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3
        class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2"
      >
        <Eye class="w-5 h-5" />
        Evidence Details
      </h3>

      <div class="space-y-4">
        <div>
          <label class="text-sm font-medium text-gray-700">Item Number</label>
          <p class="mt-1 text-gray-900 font-mono">
            {reportData.evidence.itemNumber}
          </p>
        </div>

        <div>
          <label class="text-sm font-medium text-gray-700">Description</label>
          <p class="mt-1 text-gray-900">{reportData.evidence.description}</p>
        </div>

        <div>
          <label class="text-sm font-medium text-gray-700"
            >Collection Details</label
          >
          <div class="mt-1 text-sm text-gray-900">
            <p>
              <strong>Date:</strong>
              {new Date(reportData.evidence.dateCollected).toLocaleString()}
            </p>
            <p><strong>Location:</strong> {reportData.evidence.location}</p>
          </div>
        </div>

        <div>
          <label class="text-sm font-medium text-gray-700"
            >Chain of Custody</label
          >
          <div class="mt-2 space-y-2">
            {#each reportData.evidence.chainOfCustody as custodian, index}
              <div class="flex items-center gap-2">
                <span
                  class="flex items-center justify-center w-6 h-6 bg-blue-100 text-blue-600 text-xs rounded-full"
                >
                  {index + 1}
                </span>
                <span class="text-sm text-gray-900">{custodian}</span>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>

    <!-- Analyst Information -->
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3
        class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2"
      >
        <Target class="w-5 h-5" />
        Analysis Details
      </h3>

      <div class="space-y-4">
        <div>
          <label class="text-sm font-medium text-gray-700">Analyst</label>
          <div class="mt-1">
            <p class="font-medium text-gray-900">{reportData.analyst.name}</p>
            <p class="text-sm text-gray-600">
              {reportData.analyst.credentials}
            </p>
            <p class="text-sm text-gray-600">{reportData.analyst.department}</p>
          </div>
        </div>

        <div>
          <label class="text-sm font-medium text-gray-700">Methodology</label>
          <div class="mt-2 space-y-3">
            <div>
              <p class="text-sm font-medium text-gray-600">Procedures</p>
              <ul class="mt-1 text-sm text-gray-900 list-disc list-inside">
                {#each reportData.methodology.procedures as procedure}
                  <li>{procedure}</li>
                {/each}
              </ul>
            </div>

            <div>
              <p class="text-sm font-medium text-gray-600">Tools</p>
              <div class="mt-1 flex flex-wrap gap-1">
                {#each reportData.methodology.tools as tool}
                  <span
                    class="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
                    >{tool}</span
                  >
                {/each}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Legal Impact -->
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3
        class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2"
      >
        <Scale class="w-5 h-5" />
        Legal Implications
      </h3>

      <div class="space-y-4">
        <div>
          <label class="text-sm font-medium text-gray-700"
            >Potential Charges</label
          >
          <div class="mt-2 space-y-1">
            {#each reportData.legalImplications.charges as charge}
              <div class="flex items-center gap-2">
                <CheckCircle class="w-4 h-4 text-green-600 flex-shrink-0" />
                <span class="text-sm text-gray-900">{charge}</span>
              </div>
            {/each}
          </div>
        </div>

        <div>
          <label class="text-sm font-medium text-gray-700"
            >Challenge Points</label
          >
          <div class="mt-2 space-y-1">
            {#each reportData.legalImplications.challengePoints as challenge}
              <div class="flex items-start gap-2">
                <AlertTriangle
                  class="w-4 h-4 text-yellow-600 flex-shrink-0 mt-0.5"
                />
                <span class="text-sm text-gray-900">{challenge}</span>
              </div>
            {/each}
          </div>
        </div>

        <div>
          <label class="text-sm font-medium text-gray-700"
            >Relevant Precedents</label
          >
          <div class="mt-2 space-y-1">
            {#each reportData.legalImplications.precedents as precedent}
              <div
                class="text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
              >
                {precedent}
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Key Findings -->
  <div class="bg-white border border-gray-200 rounded-lg p-6">
    <h3 class="text-lg font-semibold text-gray-900 mb-4">Key Findings</h3>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div>
        <h4 class="font-medium text-gray-900 mb-3">Summary</h4>
        <p class="text-gray-700 leading-relaxed">
          {reportData.findings.summary}
        </p>

        <div class="mt-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-gray-700"
              >Confidence Level</span
            >
            <span class="text-sm font-semibold text-gray-900"
              >{Math.round(reportData.findings.confidence * 100)}%</span
            >
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div
              class="h-2 rounded-full transition-all duration-300"
              class:bg-green-500={reportData.findings.confidence >= 0.8}
              class:bg-yellow-500={reportData.findings.confidence >= 0.6 &&
                reportData.findings.confidence < 0.8}
              class:bg-red-500={reportData.findings.confidence < 0.6}
              style="width: {reportData.findings.confidence * 100}%"
            ></div>
          </div>
        </div>
      </div>

      <div>
        <h4 class="font-medium text-gray-900 mb-3">Key Points</h4>
        <ul class="space-y-2">
          {#each reportData.findings.keyPoints as point}
            <li class="flex items-start gap-2">
              <CheckCircle
                class="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5"
              />
              <span class="text-gray-700">{point}</span>
            </li>
          {/each}
        </ul>

        {#if reportData.findings.limitations.length > 0}
          <div class="mt-4">
            <h4 class="font-medium text-gray-900 mb-3">Limitations</h4>
            <ul class="space-y-2">
              {#each reportData.findings.limitations as limitation}
                <li class="flex items-start gap-2">
                  <AlertTriangle
                    class="w-4 h-4 text-yellow-600 flex-shrink-0 mt-0.5"
                  />
                  <span class="text-gray-700">{limitation}</span>
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- Attachments -->
  {#if reportData.attachments.length > 0}
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold text-gray-900 mb-4">Attachments</h3>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {#each reportData.attachments as attachment}
          <div
            class="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div class="flex items-center gap-3">
              <div class="p-2 bg-blue-100 rounded-lg">
                <FileText class="w-5 h-5 text-blue-600" />
              </div>
              <div class="flex-1 min-w-0">
                <p class="font-medium text-gray-900 truncate">
                  {attachment.name}
                </p>
                <p class="text-sm text-gray-600">
                  {attachment.type} • {(attachment.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- AI Summary Reader -->
  <div class="mt-8">
    <AISummaryReader
      documentId={evidenceId}
      {caseId}
      initialContent={analysisContent}
      documentType="evidence"
    />
  </div>
</div>

<style>
  .evidence-report-summary {
    max-width: 80rem;
    margin-left: auto
    margin-right: auto
  }
</style>
