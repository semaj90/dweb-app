<script lang="ts">
  import { $derived } from 'svelte';
  import { browser } from "$app/environment";
  import { page } from "$app/stores";
  import EvidenceUploadModal from "$lib/components/modals/EvidenceUploadModal.svelte";
  import EvidenceValidationModal from "$lib/components/modals/EvidenceValidationModal.svelte";
  import { Button } from "$lib/components/ui/button";
  import { Card } from "$lib/components/ui/card";
  import Tooltip from "$lib/components/ui/Tooltip.svelte";
  import AdvancedFileUpload from "$lib/components/upload/AdvancedFileUpload.svelte";
  import ThinkingStyleToggle from "$lib/components/ai/ThinkingStyleToggle.svelte";
  import {
    evidenceActions,
    evidenceGrid,
    uploadActions,
    type Evidence,
  } from "$lib/stores/evidence-store";
  import { notifications } from "$lib/stores/notification";
  import { ThinkingProcessor } from "$lib/ai/thinking-processor";
  import { logSecurityEvent } from "$lib/utils/security";
  
  // Feedback Integration
  import FeedbackIntegration from '$lib/components/feedback/FeedbackIntegration.svelte';
  import {
    Activity,
    AlertTriangle,
    Archive,
    Brain,
    Calendar,
    CheckCircle,
    CheckSquare,
    Clock,
    Download,
    Edit,
    Eye,
    FileCheck,
    FileText,
    Filter,
    Grid,
    Hash,
    Image,
    List,
    Mic,
    MoreHorizontal,
    Plus,
    RefreshCw,
    Search,
    Shield,
    SortAsc,
    SortDesc,
    Square,
    Trash2,
    Upload,
    User,
    Video,
    XCircle,
    Zap,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  import type { PageData } from "./$types";
  // ... other imports ...
  export let data: PageData;

  // State management
  let validationModal = {
    open: false,
    evidence: null as Evidence | null,
    aiEvent: null as any,
  };

  let analysisModal = {
    open: false,
    evidence: null as Evidence | null,
    result: null as any,
    loading: false
  };

  let searchQuery = "";
  let showFilters = false;
  let showBulkActions = false;
  let viewMode: "grid" | "list" = "grid";
  let sortBy = "createdAt";
  let sortOrder: "asc" | "desc" = "desc";

  // Enhanced AI analysis state
  let thinkingStyleEnabled = false;
  let bulkAnalysisMode = false;
  let analysisInProgress = new Set<string>();

  // Feedback integration references
  let evidencePageFeedback: any;
  let evidenceSearchFeedback: any;
  let evidenceUploadFeedback: any;

  // Filtering and selection
  let selectedEvidence = new Set<string>();
  let selectedType = "";
  let selectedStatus = "";
  let selectedCollector = "";
  let dateFrom = "";
  let dateTo = "";
  let showAdvancedUpload = false;

  // Pagination
  let currentPage = 1;
  let itemsPerPage = 12;
  let totalPages = 1;

  // Bulk operations
  let bulkOperationLoading = false;

  // Get case ID from URL if available
  let caseId = $derived($page.url.searchParams.get("caseId") || undefined);

  // Reactive values from SSR data and store
  $: ({ isLoading: loading, error } = $evidenceGrid);
  let allEvidence = $derived(data.evidence || []);
  let filteredEvidence = $derived(filterAndSortEvidence(allEvidence));
  let visibleEvidence = $derived(getPaginatedEvidence());

  onMount(() => {
    // Initialize store with SSR data
    evidenceActions.setItems(data.evidence || []);
    // Set case context if available
    if (caseId) {
      evidenceActions.loadEvidence(caseId);
    }
  });

  function filterAndSortEvidence(evidence: Evidence[]) {
    let filtered = [...evidence];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (e) =>
          e.title?.toLowerCase().includes(query) ||
          e.description?.toLowerCase().includes(query) ||
          e.collectedBy?.toLowerCase().includes(query) ||
          e.tags?.some((tag) => tag.toLowerCase().includes(query))
      );
    }

    // Apply type filter
    if (selectedType) {
      filtered = filtered.filter((e) => e.evidenceType === selectedType);
    }

    // Apply status filter - using isAdmissible as a simple status indicator
    if (selectedStatus) {
      if (selectedStatus === "admissible") {
        filtered = filtered.filter((e) => e.isAdmissible);
      } else if (selectedStatus === "pending") {
        filtered = filtered.filter((e) => !e.isAdmissible);
      }
    }

    // Apply collector filter
    if (selectedCollector) {
      filtered = filtered.filter((e) =>
        e.collectedBy?.toLowerCase().includes(selectedCollector.toLowerCase())
      );
    }

    // Apply date filters
    if (dateFrom) {
      const fromDate = new Date(dateFrom);
      filtered = filtered.filter(
        (e) => new Date(e.uploadedAt || 0) >= fromDate
      );
    }

    if (dateTo) {
      const toDate = new Date(dateTo);
      toDate.setHours(23, 59, 59, 999); // End of day
      filtered = filtered.filter((e) => new Date(e.uploadedAt || 0) <= toDate);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue = a[sortBy as keyof Evidence];
      let bValue = b[sortBy as keyof Evidence];

      if (sortBy === "createdAt" || sortBy === "updatedAt") {
        aValue = new Date(aValue || 0).getTime();
        bValue = new Date(bValue || 0).getTime();
      } else if (typeof aValue === "string") {
        aValue = aValue.toLowerCase();
        bValue = (bValue as string)?.toLowerCase();
      }

      if (sortOrder === "asc") {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    totalPages = Math.ceil(filtered.length / itemsPerPage);
    currentPage = Math.min(currentPage, totalPages || 1);

    return filtered;
  }

  function getPaginatedEvidence() {
    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    return filteredEvidence.slice(start, end);
  }

  // Enhanced AI Analysis Functions
  async function analyzeEvidence(evidence: Evidence) {
    if (analysisInProgress.has(evidence.id)) return;
    
    analysisInProgress.add(evidence.id);
    analysisInProgress = analysisInProgress;

    try {
      const analysis = await ThinkingProcessor.analyzeEvidence(evidence.id, {
        analysisType: 'reasoning',
        useThinkingStyle: thinkingStyleEnabled,
        documentType: 'evidence'
      });

      analysisModal = {
        open: true,
        evidence,
        result: analysis,
        loading: false
      };

      notifications.add({
        type: "success",
        title: `Evidence Analysis Complete`,
        message: `${thinkingStyleEnabled ? 'Detailed thinking' : 'Quick'} analysis completed for ${evidence.title}`,
      });

    } catch (error) {
      console.error('Evidence analysis failed:', error);
      notifications.add({
        type: "error",
        title: "Analysis Failed",
        message: `Failed to analyze evidence: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    } finally {
      analysisInProgress.delete(evidence.id);
      analysisInProgress = analysisInProgress;
    }
  }

  async function bulkAnalyzeEvidence() {
    if (selectedEvidence.size === 0) {
      notifications.add({
        type: "warning",
        title: "No Evidence Selected",
        message: "Please select evidence items to analyze.",
      });
      return;
    }

    bulkOperationLoading = true;
    const evidenceIds = Array.from(selectedEvidence);
    let successCount = 0;
    let failureCount = 0;

    try {
      for (const evidenceId of evidenceIds) {
        try {
          await ThinkingProcessor.analyzeEvidence(evidenceId, {
            analysisType: 'classification',
            useThinkingStyle: thinkingStyleEnabled,
            documentType: 'evidence'
          });
          successCount++;
        } catch (error) {
          console.error(`Failed to analyze evidence ${evidenceId}:`, error);
          failureCount++;
        }
      }

      notifications.add({
        type: successCount > 0 ? "success" : "error",
        title: "Bulk Analysis Complete",
        message: `${successCount} evidence items analyzed successfully${failureCount > 0 ? `, ${failureCount} failed` : ''}.`,
      });

      if (successCount > 0) {
        await refreshEvidence();
      }

    } finally {
      bulkOperationLoading = false;
      selectedEvidence.clear();
      selectedEvidence = selectedEvidence;
      showBulkActions = false;
    }
  }

  function handleThinkingToggle(event: CustomEvent<{ enabled: boolean }>) {
    thinkingStyleEnabled = event.detail.enabled;
    
    notifications.add({
      type: "info",
      title: "Analysis Mode Changed",
      message: thinkingStyleEnabled 
        ? "🧠 Thinking Style enabled - detailed reasoning will be shown"
        : "⚡ Quick Mode enabled - concise analysis results",
    });
  }

  function closeAnalysisModal() {
    analysisModal = {
      open: false,
      evidence: null,
      result: null,
      loading: false
    };
  }

  function formatAnalysisForDisplay(analysis: any): string {
    if (!analysis) return "No analysis available";
    
    let display = "";
    
    if (analysis.thinking && thinkingStyleEnabled) {
      display += `**🧠 AI Reasoning Process:**\n${analysis.thinking}\n\n---\n\n`;
    }
    
    if (analysis.analysis) {
      display += `**📋 Analysis Results:**\n`;
      const analysisData = analysis.analysis;
      
      if (analysisData.key_findings) {
        display += `\n**Key Findings:**\n`;
        analysisData.key_findings.forEach((finding: string) => {
          display += `• ${finding}\n`;
        });
      }
      
      if (analysisData.legal_implications) {
        display += `\n**Legal Implications:**\n`;
        analysisData.legal_implications.forEach((implication: string) => {
          display += `• ${implication}\n`;
        });
      }
      
      if (analysisData.recommendations) {
        display += `\n**Recommendations:**\n`;
        analysisData.recommendations.forEach((rec: string) => {
          display += `• ${rec}\n`;
        });
      }
    }
    
    display += `\n**Confidence:** ${Math.round(analysis.confidence * 100)}%`;
    
    return display;
  }

  function toggleEvidenceSelection(evidenceId: string) {
    if (selectedEvidence.has(evidenceId)) {
      selectedEvidence.delete(evidenceId);
    } else {
      selectedEvidence.add(evidenceId);
    }
    selectedEvidence = selectedEvidence;
    showBulkActions = selectedEvidence.size > 0;
  }

  function selectAllEvidence() {
    if (selectedEvidence.size === visibleEvidence.length) {
      selectedEvidence.clear();
    } else {
      visibleEvidence.forEach((e) => selectedEvidence.add(e.id));
    }
    selectedEvidence = selectedEvidence;
    showBulkActions = selectedEvidence.size > 0;
  }

  async function bulkOperation(operation: string) {
    if (selectedEvidence.size === 0) return;

    bulkOperationLoading = true;
    try {
      const evidenceIds = Array.from(selectedEvidence);

      switch (operation) {
        case "analyze":
          await bulkAnalyzeEvidence();
          return; // bulkAnalyzeEvidence handles its own completion
        case "archive":
          await Promise.all(
            evidenceIds.map((id) => updateEvidenceStatus(id, "Archived"))
          );
          notifications.add({
            type: "success",
            title: "Evidence Archived",
            message: `${evidenceIds.length} evidence items archived successfully.`,
          });
          break;
        case "verify":
          await Promise.all(
            evidenceIds.map((id) => updateEvidenceStatus(id, "Verified"))
          );
          notifications.add({
            type: "success",
            title: "Evidence Verified",
            message: `${evidenceIds.length} evidence items verified successfully.`,
          });
          break;
        case "export":
          await exportEvidence(evidenceIds);
          notifications.add({
            type: "success",
            title: "Evidence Exported",
            message: `${evidenceIds.length} evidence items exported successfully.`,
          });
          break;
        case "delete":
          await deleteEvidence(evidenceIds);
          notifications.add({
            type: "success",
            title: "Evidence Deleted",
            message: `${evidenceIds.length} evidence items deleted successfully.`,
          });
          break;
      }

      selectedEvidence.clear();
      selectedEvidence = selectedEvidence;
      showBulkActions = false;
      await refreshEvidence();
    } catch (err) {
      console.error("Bulk operation failed:", err);
      notifications.add({
        type: "error",
        title: "Bulk Operation Failed",
        message: "Failed to perform bulk operation. Please try again.",
        duration: 5000,
      });
    } finally {
      bulkOperationLoading = false;
    }
  }

  async function updateEvidenceStatus(evidenceId: string, status: string) {
    // Implementation would call API
    console.log("Updating evidence status:", evidenceId, status);
  }

  async function exportEvidence(evidenceIds: string[]) {
    const evidenceToExport = allEvidence.filter((e) =>
      evidenceIds.includes(e.id)
    );
    const dataStr = JSON.stringify(evidenceToExport, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });

    if (browser) {
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `evidence_export_${new Date().toISOString().split("T")[0]}.json`;
      link.click();
      URL.revokeObjectURL(url);
    }

    // Log security event
    logSecurityEvent({
      type: "data_export",
      details: { evidenceIds, exportType: "bulk" },
      severity: "medium",
    });
  }

  async function deleteEvidence(evidenceIds: string[]) {
    // Implementation would call API
    console.log("Deleting evidence:", evidenceIds);
  }

  async function refreshEvidence() {
    if (caseId) {
      await evidenceActions.loadEvidence(caseId);
    }
  }

  function openUploadModal() {
    uploadActions.openModal(caseId);
  }

  function handleEvidenceValidation(event: CustomEvent) {
    const { evidence, aiEvent } = event.detail;
    validationModal = {
      open: true,
      evidence,
      aiEvent: aiEvent || null,
    };
  }

  function handleValidationComplete(event: CustomEvent) {
    validationModal.open = false;
    // Refresh evidence grid to show updated analysis
    refreshEvidence();
  }

  function handleAdvancedUpload() {
    showAdvancedUpload = true;
  }

  function handleFileUpload(event: CustomEvent) {
    const { files } = event.detail;
    // Process uploaded files
    console.log("Files uploaded:", files);
    showAdvancedUpload = false;
    refreshEvidence();
  }

  function getEvidenceTypeIcon(type: string) {
    switch (type?.toLowerCase()) {
      case "document":
        return FileText;
      case "image":
        return Image;
      case "video":
        return Video;
      case "audio":
        return Mic;
      default:
        return FileCheck;
    }
  }

  function getEvidenceStatusColor(status: string) {
    switch (status?.toLowerCase()) {
      case "pending":
        return "badge-warning";
      case "verified":
        return "badge-success";
      case "archived":
        return "badge-neutral";
      case "flagged":
        return "badge-error";
      default:
        return "badge-ghost";
    }
  }

  function getEvidenceStatusIcon(status: string) {
    switch (status?.toLowerCase()) {
      case "pending":
        return Clock;
      case "verified":
        return CheckCircle;
      case "archived":
        return Archive;
      case "flagged":
        return AlertTriangle;
      default:
        return FileCheck;
    }
  }

  // Reactive statements
  $: if (
    searchQuery ||
    selectedType ||
    selectedStatus ||
    selectedCollector ||
    dateFrom ||
    dateTo ||
    sortBy ||
    sortOrder
  ) {
    filteredEvidence = filterAndSortEvidence(allEvidence);
  }
</script>

<svelte:head>
  <title>Evidence Management - WardenNet Detective Mode</title>
  <meta
    name="description"
    content="Advanced evidence management with secure file handling, chain of custody, and AI-powered analysis"
  />
</svelte:head>

<div class="container-nes-main">
  <!-- Header Section -->
  <div class="container-nes-panel mb-6">
    <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
      <div>
        <h1 class="nes-text-pixelated text-3xl font-bold text-gray-900 dark:text-white">
          Evidence Management
        </h1>
        <p class="text-gray-600 dark:text-gray-400 mt-2">
          {#if caseId}
            <span class="nes-legal-priority-high px-2 py-1 rounded">
              Evidence for Case #{caseId}
            </span>
          {:else}
            All Evidence Files
          {/if}
          <span class="badge ml-2">({filteredEvidence.length} of {allEvidence.length} items)</span>
        </p>
      </div>

      <!-- Enhanced Action Buttons with AI Analysis -->
      <div class="flex flex-wrap gap-2 items-center">
        <!-- AI Analysis Toggle -->
        <div class="neural-sprite-active">
          <ThinkingStyleToggle 
            bind:enabled={thinkingStyleEnabled}
            premium={true}
            size="sm"
            ontoggle={handleThinkingToggle}
          />
        </div>

        <Tooltip content="Refresh evidence list">
          <Button
            variant="outline"
            size="sm"
            onclick={() => refreshEvidence()}
            disabled={loading}
            aria-label="Refresh evidence"
          >
            <span class:animate-spin={loading} class:neural-sprite-loading={loading}>
              <RefreshCw class="w-4 h-4" />
            </span>
          </Button>
        </Tooltip>

        <Tooltip content="Toggle filters">
          <Button
            variant="outline"
            size="sm"
            onclick={() => (showFilters = !showFilters)}
            class={showFilters ? 'nes-legal-priority-high' : ''}
            aria-label="Toggle filters"
            aria-expanded={showFilters}
          >
            <Filter class="w-4 h-4 mr-2" />
            Filters
          </Button>
        </Tooltip>

        <Tooltip content="Toggle view mode">
          <Button
            variant="outline"
            size="sm"
            onclick={() => (viewMode = viewMode === "grid" ? "list" : "grid")}
            aria-label="Toggle view mode"
            class="yorha-3d-button"
          >
            {#if viewMode === "grid"}
              <List class="w-4 h-4" />
            {:else}
              <Grid class="w-4 h-4" />
            {/if}
          </Button>
        </Tooltip>

        <Tooltip content="Advanced file upload">
          <Button
            variant="outline"
            size="sm"
            onclick={() => handleAdvancedUpload()}
            class="neural-sprite-cached"
          >
            <Upload class="w-4 h-4 mr-2" />
            Advanced Upload
          </Button>
        </Tooltip>

        <Tooltip content="Standard evidence upload">
          <Button 
            variant="evidence"
            onclick={() => openUploadModal()}
          >
            <Plus class="w-4 h-4 mr-2" />
            Upload Evidence
          </Button>
        </Tooltip>
      </div>
    </div>
  </div>

  <!-- Search and Filters -->
  <div class="mx-auto px-4 max-w-7xl">
    <!-- Search Bar -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <Search
          class="mx-auto px-4 max-w-7xl"
        />
        <input
          type="text"
          placeholder="Search evidence by title, description, collector, or tags..."
          class="mx-auto px-4 max-w-7xl"
          bind:value={searchQuery}
          aria-label="Search evidence"
        />
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <select
          class="mx-auto px-4 max-w-7xl"
          bind:value={sortBy}
          aria-label="Sort by field"
        >
          <option value="createdAt">Created Date</option>
          <option value="updatedAt">Updated Date</option>
          <option value="title">Title</option>
          <option value="type">Type</option>
          <option value="status">Status</option>
          <option value="collectedBy">Collector</option>
        </select>

        <Button
          variant="outline"
          size="sm"
          onclick={() => (sortOrder = sortOrder === "asc" ? "desc" : "asc")}
          aria-label="Toggle sort order"
        >
          {#if sortOrder === "asc"}
            <SortAsc class="mx-auto px-4 max-w-7xl" />
          {:else}
            <SortDesc class="mx-auto px-4 max-w-7xl" />
          {/if}
        </Button>
      </div>
    </div>

    <!-- Advanced Filters -->
    {#if showFilters}
      <div
        class="mx-auto px-4 max-w-7xl"
      >
        <div>
          <label for="type-filter" class="mx-auto px-4 max-w-7xl">Evidence Type</label
          >
          <select
            id="type-filter"
            class="mx-auto px-4 max-w-7xl"
            bind:value={selectedType}
            aria-label="Filter by evidence type"
          >
            <option value="">All Types</option>
            <option value="document">Documents</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
            <option value="audio">Audio</option>
            <option value="physical">Physical</option>
            <option value="digital">Digital</option>
          </select>
        </div>

        <div>
          <label for="status-filter" class="mx-auto px-4 max-w-7xl">Status</label>
          <select
            id="status-filter"
            class="mx-auto px-4 max-w-7xl"
            bind:value={selectedStatus}
            aria-label="Filter by status"
          >
            <option value="">All Statuses</option>
            <option value="Pending">Pending</option>
            <option value="Verified">Verified</option>
            <option value="Archived">Archived</option>
            <option value="Flagged">Flagged</option>
          </select>
        </div>

        <div>
          <label for="collector-filter" class="mx-auto px-4 max-w-7xl"
            >Collected By</label
          >
          <input
            id="collector-filter"
            type="text"
            placeholder="Officer name..."
            class="mx-auto px-4 max-w-7xl"
            bind:value={selectedCollector}
            aria-label="Filter by collector"
          />
        </div>

        <div>
          <label for="date-from" class="mx-auto px-4 max-w-7xl">Date From</label>
          <input
            id="date-from"
            type="date"
            class="mx-auto px-4 max-w-7xl"
            bind:value={dateFrom}
            aria-label="Filter from date"
          />
        </div>

        <div>
          <label for="date-to" class="mx-auto px-4 max-w-7xl">Date To</label>
          <input
            id="date-to"
            type="date"
            class="mx-auto px-4 max-w-7xl"
            bind:value={dateTo}
            aria-label="Filter to date"
          />
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <Button
          variant="outline"
          size="sm"
          onclick={() => {
            selectedType = "";
            selectedStatus = "";
            selectedCollector = "";
            dateFrom = "";
            dateTo = "";
            searchQuery = "";
          }
        >
          Clear Filters
        </Button>
      </div>
    {/if}
  </div>

  <!-- Enhanced Bulk Actions with AI Analysis -->
  {#if showBulkActions}
    <div class="mx-auto px-4 max-w-7xl">
      <div
        class="mx-auto px-4 max-w-7xl"
      >
        <div class="mx-auto px-4 max-w-7xl">
          <CheckSquare class="mx-auto px-4 max-w-7xl" />
          <span class="mx-auto px-4 max-w-7xl"
            >{selectedEvidence.size} evidence item(s) selected</span
          >
        </div>

        <div class="mx-auto px-4 max-w-7xl">
          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("analyze")}
            disabled={bulkOperationLoading}
            class="mx-auto px-4 max-w-7xl"
          >
            {#if thinkingStyleEnabled}
              <Brain class="mx-auto px-4 max-w-7xl" />
              Analyze (Thinking)
            {:else}
              <Zap class="mx-auto px-4 max-w-7xl" />
              Quick Analyze
            {/if}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("verify")}
            disabled={bulkOperationLoading}
            class="mx-auto px-4 max-w-7xl"
          >
            <CheckCircle class="mx-auto px-4 max-w-7xl" />
            Verify
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("archive")}
            disabled={bulkOperationLoading}
            class="mx-auto px-4 max-w-7xl"
          >
            <Archive class="mx-auto px-4 max-w-7xl" />
            Archive
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("export")}
            disabled={bulkOperationLoading}
            class="mx-auto px-4 max-w-7xl"
          >
            <Download class="mx-auto px-4 max-w-7xl" />
            Export
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("delete")}
            disabled={bulkOperationLoading}
            class="mx-auto px-4 max-w-7xl"
          >
            <Trash2 class="mx-auto px-4 max-w-7xl" />
            Delete
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => {
              selectedEvidence.clear();
              selectedEvidence = selectedEvidence;
              showBulkActions = false;
            "
            disabled={bulkOperationLoading}
          >
            Cancel
          </Button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Evidence List/Grid -->
  {#if loading}
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl"></div>
      <span class="mx-auto px-4 max-w-7xl">Loading evidence...</span>
    </div>
  {:else if error}
    <div class="mx-auto px-4 max-w-7xl" role="alert">
      <XCircle class="mx-auto px-4 max-w-7xl" />
      <div>
        <h3 class="mx-auto px-4 max-w-7xl">Error Loading Evidence</h3>
        <div class="mx-auto px-4 max-w-7xl">{error}</div>
      </div>
      <Button variant="outline" size="sm" onclick={() => refreshEvidence()}>
        <RefreshCw class="mx-auto px-4 max-w-7xl" />
        Retry
      </Button>
    </div>
  {:else if filteredEvidence.length === 0}
    <div class="mx-auto px-4 max-w-7xl">
      <FileCheck class="mx-auto px-4 max-w-7xl" />
      <h3 class="mx-auto px-4 max-w-7xl">
        {searchQuery ||
        selectedType ||
        selectedStatus ||
        selectedCollector ||
        dateFrom ||
        dateTo
          ? "No matching evidence found"
          : "No evidence found"}
      </h3>
      <p class="mx-auto px-4 max-w-7xl">
        {searchQuery ||
        selectedType ||
        selectedStatus ||
        selectedCollector ||
        dateFrom ||
        dateTo
          ? "Try adjusting your search criteria or filters"
          : "Upload files, documents, and digital evidence to get started"}
      </p>
      {#if !searchQuery && !selectedType && !selectedStatus && !selectedCollector && !dateFrom && !dateTo}
        <div class="mx-auto px-4 max-w-7xl">
          <Button onclick={() => openUploadModal()} class="mx-auto px-4 max-w-7xl">
            <Plus class="mx-auto px-4 max-w-7xl" />
            Upload Evidence
          </Button>
          <Button
            variant="outline"
            onclick={() => handleAdvancedUpload()}
            class="mx-auto px-4 max-w-7xl"
          >
            <Upload class="mx-auto px-4 max-w-7xl" />
            Advanced Upload
          </Button>
        </div>
      {/if}
    </div>
  {:else}
    <!-- Evidence Header Controls -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <span class="mx-auto px-4 max-w-7xl">
          Showing {(currentPage - 1) * itemsPerPage + 1}-{Math.min(
            currentPage * itemsPerPage,
            filteredEvidence.length
          )} of {filteredEvidence.length} evidence items
        </span>

        {#if visibleEvidence.length > 0}
          <Button
            variant="ghost"
            size="sm"
            onclick={() => selectAllEvidence()}
            class="mx-auto px-4 max-w-7xl"
            aria-label="Select all visible evidence"
          >
            {#if selectedEvidence.size === visibleEvidence.length}
              <CheckSquare class="mx-auto px-4 max-w-7xl" />
            {:else}
              <Square class="mx-auto px-4 max-w-7xl" />
            {/if}
            Select All
          </Button>
        {/if}
      </div>
    </div>

    <!-- Enhanced Evidence Grid/List View with AI Analysis -->
    <div class="space-y-6">
      {#if viewMode === "grid"}
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {#each visibleEvidence as evidence}
            <Card 
              variant="evidence" 
              priority={evidence.isAdmissible ? "critical" : "medium"}
              loading={analysisInProgress.has(evidence.id)}
              interactive={true}
              class="group hover:shadow-xl transition-all duration-300"
            >
              <!-- Card Header with Selection -->
              <div class="flex justify-between items-start mb-4">
                <input
                  type="checkbox"
                  class="input-nes-primary w-4 h-4"
                  checked={selectedEvidence.has(evidence.id)}
                  onchange={() => toggleEvidenceSelection(evidence.id)}
                  aria-label="Select evidence {evidence.title || 'Untitled Evidence'}"
                />
                
                <div class="dropdown dropdown-end">
                  <Tooltip content="Evidence actions">
                    <Button variant="ghost" size="sm" class="yorha-3d-button">
                      <MoreHorizontal class="w-4 h-4" />
                    </Button>
                  </Tooltip>
                </div>
              </div>

              <!-- Evidence Content -->
              <div class="space-y-4">
                <!-- Header with Icon -->
                <div class="flex items-center gap-3">
                  <svelte:component
                    this={getEvidenceTypeIcon(evidence.evidenceType)}
                    class="w-8 h-8 text-blue-500 nes-memory-active"
                  />
                  <div>
                    <h2 class="nes-text-pixelated font-bold text-lg line-clamp-2">
                      {evidence.title || "Untitled Evidence"}
                    </h2>
                  </div>
                </div>

                <!-- Status Badges -->
                <div class="flex flex-wrap gap-2">
                  <span class="badge bg-blue-100 text-blue-800">
                    {evidence.evidenceType || "Unknown"}
                  </span>
                  <span class={evidence.isAdmissible ? 'nes-legal-priority-critical' : 'nes-legal-priority-medium'}>
                    {evidence.isAdmissible ? "Admissible" : "Pending"}
                  </span>
                  {#if evidence.hash}
                    <span class="nes-legal-priority-high flex items-center gap-1">
                      <Shield class="w-3 h-3" />
                      Verified
                    </span>
                  {/if}
                  {#if evidence.aiAnalysis && Object.keys(evidence.aiAnalysis).length > 0}
                    <span class="neural-sprite-active flex items-center gap-1">
                      <Brain class="w-3 h-3" />
                      AI Analyzed
                    </span>
                  {/if}
                </div>

                <!-- Description -->
                <p class="text-gray-600 dark:text-gray-300 text-sm line-clamp-3">
                  {evidence.description
                    ? evidence.description.length > 120
                      ? evidence.description.substring(0, 120) + "..."
                      : evidence.description
                    : "No description available"}
                </p>

                <!-- Metadata -->
                <div class="grid grid-cols-1 gap-2 text-xs text-gray-500">
                  <div class="flex items-center gap-1">
                    <Calendar class="w-3 h-3" />
                    Collected: {evidence.uploadedAt
                      ? new Date(evidence.uploadedAt).toLocaleDateString()
                      : "Unknown"}
                  </div>
                  {#if evidence.collectedBy}
                    <div class="flex items-center gap-1">
                      <User class="w-3 h-3" />
                      By: {evidence.collectedBy}
                    </div>
                  {/if}
                  {#if evidence.fileSize}
                    <div class="flex items-center gap-1">
                      <Activity class="w-3 h-3" />
                      Size: {(evidence.fileSize / 1024 / 1024).toFixed(2)} MB
                    </div>
                  {/if}
                </div>
              </div>

              <!-- Enhanced Actions with AI Analysis -->
              <div class="flex gap-2 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <Button 
                  size="sm" 
                  variant={thinkingStyleEnabled ? "neural" : "yorha"}
                  onclick={() => analyzeEvidence(evidence)}
                  disabled={analysisInProgress.has(evidence.id)}
                  class="flex-1"
                >
                  {#if analysisInProgress.has(evidence.id)}
                    <div class="neural-sprite-loading w-3 h-3"></div>
                    Analyzing...
                  {:else if thinkingStyleEnabled}
                    <Brain class="w-3 h-3 mr-1" />
                    Think
                  {:else}
                    <Zap class="w-3 h-3 mr-1" />
                    Analyze
                  {/if}
                </Button>
                
                <a href="/evidence/{evidence.id}">
                  <Button size="sm" variant="evidence" class="flex-1">
                    <Eye class="w-3 h-3 mr-1" />
                    View
                  </Button>
                </a>
              </div>
            </Card>
          {/each}
        </div>
      {:else}
        <!-- Enhanced List View -->
        <div class="mx-auto px-4 max-w-7xl">
          {#each visibleEvidence as evidence}
            <div
              class="mx-auto px-4 max-w-7xl"
            >
              <div class="mx-auto px-4 max-w-7xl">
                <input
                  type="checkbox"
                  class="mx-auto px-4 max-w-7xl"
                  checked={selectedEvidence.has(evidence.id)}
                  onchange={() => toggleEvidenceSelection(evidence.id)}
                  aria-label="Select evidence {evidence.title ||
                    'Untitled Evidence'}"
                />

                <svelte:component
                  this={getEvidenceTypeIcon(evidence.evidenceType)}
                  class="mx-auto px-4 max-w-7xl"
                />

                <div class="mx-auto px-4 max-w-7xl">
                  <div class="mx-auto px-4 max-w-7xl">
                    <div class="mx-auto px-4 max-w-7xl">
                      <h3 class="mx-auto px-4 max-w-7xl">
                        {evidence.title || "Untitled Evidence"}
                      </h3>
                      <p class="mx-auto px-4 max-w-7xl">
                        {evidence.description || "No description available"}
                      </p>
                    </div>

                    <div class="mx-auto px-4 max-w-7xl">
                      <div class="mx-auto px-4 max-w-7xl">
                        {evidence.evidenceType || "Unknown"}
                      </div>
                      <div
                        class="mx-auto px-4 max-w-7xl"
                      >
                        {evidence.isAdmissible ? "Admissible" : "Pending"}
                      </div>
                      {#if evidence.hash}
                        <div class="mx-auto px-4 max-w-7xl">
                          <Shield class="mx-auto px-4 max-w-7xl" />
                          Verified
                        </div>
                      {/if}
                      {#if evidence.aiAnalysis && Object.keys(evidence.aiAnalysis).length > 0}
                        <div class="mx-auto px-4 max-w-7xl">
                          <Brain class="mx-auto px-4 max-w-7xl" />
                          AI Analyzed
                        </div>
                      {/if}
                    </div>
                  </div>

                  <div
                    class="mx-auto px-4 max-w-7xl"
                  >
                    <div class="mx-auto px-4 max-w-7xl">
                      <Calendar class="mx-auto px-4 max-w-7xl" />
                      {evidence.uploadedAt
                        ? new Date(evidence.uploadedAt).toLocaleDateString()
                        : "Unknown"}
                    </div>
                    {#if evidence.collectedBy}
                      <div class="mx-auto px-4 max-w-7xl">
                        <User class="mx-auto px-4 max-w-7xl" />
                        {evidence.collectedBy}
                      </div>
                    {/if}
                    {#if evidence.fileSize}
                      <div class="mx-auto px-4 max-w-7xl">
                        <Activity class="mx-auto px-4 max-w-7xl" />
                        {(evidence.fileSize / 1024 / 1024).toFixed(2)} MB
                      </div>
                    {/if}
                  </div>
                </div>

                <div class="mx-auto px-4 max-w-7xl">
                  <Button 
                    size="sm" 
                    variant="outline"
                    onclick={() => analyzeEvidence(evidence)}
                    disabled={analysisInProgress.has(evidence.id)}
                    class="mx-auto px-4 max-w-7xl"
                  >
                    {#if analysisInProgress.has(evidence.id)}
                      <div class="mx-auto px-4 max-w-7xl"></div>
                      Analyzing...
                    {:else if thinkingStyleEnabled}
                      <Brain class="mx-auto px-4 max-w-7xl" />
                      Think
                    {:else}
                      <Zap class="mx-auto px-4 max-w-7xl" />
                      Analyze
                    {/if}
                  </Button>

                  <a href="/evidence/{evidence.id}" class="mx-auto px-4 max-w-7xl">
                    <Button size="sm" variant="outline">
                      <Eye class="mx-auto px-4 max-w-7xl" />
                      View
                    </Button>
                  </a>

                  <div class="mx-auto px-4 max-w-7xl">
                    <Tooltip content="More actions">
                      <Button
                        variant="ghost"
                        size="sm"
                        tabindex={0}
                        role="button"
                        aria-label="More actions for {evidence.title ||
                          'Untitled Evidence'}"
                      >
                        <MoreHorizontal class="mx-auto px-4 max-w-7xl" />
                      </Button>
                    </Tooltip>
                    <ul
                      tabindex={0}
                      role="menu"
                      class="mx-auto px-4 max-w-7xl"
                    >
                      <li>
                        <a href="/evidence/{evidence.id}/edit" class="mx-auto px-4 max-w-7xl">
                          <Edit class="mx-auto px-4 max-w-7xl" />
                          Edit Evidence
                        </a>
                      </li>
                      <li>
                        <button class="mx-auto px-4 max-w-7xl">
                          <Hash class="mx-auto px-4 max-w-7xl" />
                          Verify Hash
                        </button>
                      </li>
                      <li>
                        <button class="mx-auto px-4 max-w-7xl">
                          <Trash2 class="mx-auto px-4 max-w-7xl" />
                          Delete
                        </button>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Pagination -->
    {#if totalPages > 1}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Button
            variant="outline"
            size="sm"
            class="mx-auto px-4 max-w-7xl"
            disabled={currentPage === 1}
            onclick={() => (currentPage = Math.max(1, currentPage - 1))}
            aria-label="Previous page"
          >
            Previous
          </Button>

          {#each Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            const start = Math.max(1, currentPage - 2);
            return start + i;
          }).filter((p) => p <= totalPages) as page}
            <Button
              variant={page === currentPage ? "default" : "outline"}
              size="sm"
              class="mx-auto px-4 max-w-7xl"
              onclick={() => (currentPage = page)}
              aria-label="Go to page {page}"
              aria-current={page === currentPage ? "page" : undefined}
            >
              {page}
            </Button>
          {/each}

          <Button
            variant="outline"
            size="sm"
            class="mx-auto px-4 max-w-7xl"
            disabled={currentPage === totalPages}
            onclick={() =>
              (currentPage = Math.min(totalPages, currentPage + 1))}
            aria-label="Next page"
          >
            Next
          </Button>
        </div>
      </div>
    {/if}
  {/if}
</div>

<!-- Modals -->
<EvidenceUploadModal />

<EvidenceValidationModal
  bind:open={validationModal.open}
  evidence={validationModal.evidence}
  aiEvent={validationModal.aiEvent}
  oncomplete={handleValidationComplete}
/>

<!-- AI Analysis Results Modal -->
{#if analysisModal.open && analysisModal.evidence && analysisModal.result}
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h3 class="mx-auto px-4 max-w-7xl">
          {#if thinkingStyleEnabled}
            <Brain class="mx-auto px-4 max-w-7xl" />
            Thinking Style Analysis
          {:else}
            <Zap class="mx-auto px-4 max-w-7xl" />
            Quick Analysis
          {/if}
          - {analysisModal.evidence.title}
        </h3>
        <Button variant="ghost" size="sm" onclick={closeAnalysisModal}>
          ✕
        </Button>
      </div>
      
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">{formatAnalysisForDisplay(analysisModal.result)}</div>
        </div>
        
        {#if analysisModal.result.reasoning_steps && analysisModal.result.reasoning_steps.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <h4 class="mx-auto px-4 max-w-7xl">Reasoning Steps:</h4>
            <ol class="mx-auto px-4 max-w-7xl">
              {#each analysisModal.result.reasoning_steps as step}
                <li class="mx-auto px-4 max-w-7xl">{step}</li>
              {/each}
            </ol>
          </div>
        {/if}
      </div>
      
      <div class="mx-auto px-4 max-w-7xl">
        <Button variant="outline" onclick={closeAnalysisModal}>Close</Button>
        <Button onclick={() => {
          // Save analysis or perform other actions
          closeAnalysisModal();
        ">Save Analysis</Button>
      </div>
    </div>
    <div class="mx-auto px-4 max-w-7xl" onclick={closeAnalysisModal}></div>
  </div>
{/if}

{#if showAdvancedUpload}
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <h3 class="mx-auto px-4 max-w-7xl">Advanced Evidence Upload</h3>
      <AdvancedFileUpload
        accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.json"
        multiple={true}
        maxFiles={10}
        maxFileSize={50 * 1024 * 1024}
        onupload={handleFileUpload}
        oncancel={() => (showAdvancedUpload = false)}
      />
    </div>
    <div
      class="mx-auto px-4 max-w-7xl"
      role="button"
      tabindex={0}
      aria-label="Close modal"
      onclick={() => (showAdvancedUpload = false)}
      onkeydown={(e) => e.key === "Escape" && (showAdvancedUpload = false)}
    ></div>
  </div>
{/if}

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>

<!-- Feedback Integration Components -->
<FeedbackIntegration
  bind:this={evidencePageFeedback}
  interactionType="page_visit"
  ratingType="ui_experience"
  priority="low"
  context={{ 
    page: 'evidence',
    viewMode,
    evidenceCount: $evidenceGrid.length,
    hasFilters: searchQuery.trim() || selectedType || selectedStatus
  }}
  trackOnMount={true}
  let:feedback
/>

<FeedbackIntegration
  bind:this={evidenceSearchFeedback}
  interactionType="evidence_search"
  ratingType="search_relevance"
  priority="medium"
  context={{ component: 'EvidenceSearch', legalDomain: 'evidence_management' }}
  let:feedback
/>

<FeedbackIntegration
  bind:this={evidenceUploadFeedback}
  interactionType="evidence_upload"
  ratingType="ui_experience"
  priority="high"
  context={{ component: 'EvidenceUpload' }}
  let:feedback
/>
