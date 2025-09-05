<script lang="ts">
  interface Props {
    data: PageData
  }
  let {
    data
  } = $props();



  import type { Evidence } from '$lib/types';
  import { browser } from "$app/environment";
  import { page } from "$app/stores";
  import EvidenceUploadModal from "$lib/components/modals/EvidenceUploadModal.svelte";
  import EvidenceValidationModal from "$lib/components/modals/EvidenceValidationModal.svelte";
  import { Button } from "$lib/components/ui/button";
  import Tooltip from "$lib/components/ui/Tooltip.svelte";
  import AdvancedFileUpload from "$lib/components/upload/AdvancedFileUpload.svelte";
  import ThinkingStyleToggle from "$lib/components/ai/ThinkingStyleToggle.svelte";
  import {
    evidenceActions,
    evidenceGrid,
    uploadActions,
  } from "$lib/stores/evidence-store";
  import { notifications } from "$lib/stores/notification";
  import { ThinkingProcessor } from "$lib/ai/thinking-processor";
  import { logSecurityEvent } from "$lib/utils/security";
  import { Activity, AlertTriangle, Archive, Brain, Calendar, CheckCircle, CheckSquare, Clock, Download, Edit, Eye, FileCheck, FileText, Filter, Grid, Hash, Image, List, Mic, MoreHorizontal, Plus, RefreshCw, Search, Shield, SortAsc, SortDesc, Square, Trash2, Upload, User as UserIcon, Video, XCircle, Zap } from "lucide-svelte";
  import { onMount } from "svelte";

  import type { PageData } from "./$types";
  // ... other imports ...
  
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
  let caseId = $derived($page.url.searchParams.get("caseId") || undefined;);

  // Reactive values from SSR data and store
  $effect(() => { ({ isLoading: loading, error } = $evidenceGrid);
  let allEvidence = $derived(data.evidence || [])
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
  $effect(() => { if (
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

<div class="space-y-4">
  <!-- Header Section -->
  <div
    class="space-y-4"
  >
    <div>
      <h1 class="space-y-4">Evidence Management</h1>
      <p class="space-y-4">
        {#if caseId}
          Evidence for Case #{caseId}
        {:else}
          All Evidence Files
        {/if}
        ({filteredEvidence.length} of {allEvidence.length} items)
      </p>
    </div>

    <!-- Enhanced Action Buttons with AI Analysis -->
    <div class="space-y-4">
      <!-- AI Analysis Toggle -->
      <div class="space-y-4">
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
          <span class:animate-spin={loading}>
            <RefreshCw class="space-y-4" />
          </span>
        </Button>
      </Tooltip>

      <Tooltip content="Toggle filters">
        <Button
          variant="outline"
          size="sm"
          onclick={() => (showFilters = !showFilters)}
          class="space-y-4"
          aria-label="Toggle filters"
          aria-expanded={showFilters}
        >
          <Filter class="space-y-4" />
          Filters
        </Button>
      </Tooltip>

      <Tooltip content="Toggle view mode">
        <Button
          variant="outline"
          size="sm"
          onclick={() => (viewMode = viewMode === "grid" ? "list" : "grid")}
          aria-label="Toggle view mode"
        >
          {#if viewMode === "grid"}
            <List class="space-y-4" />
          {:else}
            <Grid class="space-y-4" />
          {/if}
        </Button>
      </Tooltip>

      <Tooltip content="Advanced file upload">
        <Button
          variant="outline"
          size="sm"
          onclick={() => handleAdvancedUpload()}
          class="space-y-4"
        >
          <Upload class="space-y-4" />
          Advanced Upload
        </Button>
      </Tooltip>

      <Tooltip content="Standard evidence upload">
        <Button onclick={() => openUploadModal()} class="space-y-4">
          <Plus class="space-y-4" />
          Upload Evidence
        </Button>
      </Tooltip>
    </div>
  </div>

  <!-- Search and Filters -->
  <div class="space-y-4">
    <!-- Search Bar -->
    <div class="space-y-4">
      <div class="space-y-4">
        <Search
          class="space-y-4"
        />
        <input
          type="text"
          placeholder="Search evidence by title, description, collector, or tags..."
          class="space-y-4"
          bind:value={searchQuery}
          aria-label="Search evidence"
        />
      </div>

      <div class="space-y-4">
        <select
          class="space-y-4"
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
            <SortAsc class="space-y-4" />
          {:else}
            <SortDesc class="space-y-4" />
          {/if}
        </Button>
      </div>
    </div>

    <!-- Advanced Filters -->
    {#if showFilters}
      <div
        class="space-y-4"
      >
        <div>
          <label for="type-filter" class="space-y-4">Evidence Type</label
          >
          <select
            id="type-filter"
            class="space-y-4"
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
          <label for="status-filter" class="space-y-4">Status</label>
          <select
            id="status-filter"
            class="space-y-4"
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
          <label for="collector-filter" class="space-y-4"
            >Collected By</label
          >
          <input
            id="collector-filter"
            type="text"
            placeholder="Officer name..."
            class="space-y-4"
            bind:value={selectedCollector}
            aria-label="Filter by collector"
          />
        </div>

        <div>
          <label for="date-from" class="space-y-4">Date From</label>
          <input
            id="date-from"
            type="date"
            class="space-y-4"
            bind:value={dateFrom}
            aria-label="Filter from date"
          />
        </div>

        <div>
          <label for="date-to" class="space-y-4">Date To</label>
          <input
            id="date-to"
            type="date"
            class="space-y-4"
            bind:value={dateTo}
            aria-label="Filter to date"
          />
        </div>
      </div>

      <div class="space-y-4">
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
          }}
        >
          Clear Filters
        </Button>
      </div>
    {/if}
  </div>

  <!-- Enhanced Bulk Actions with AI Analysis -->
  {#if showBulkActions}
    <div class="space-y-4">
      <div
        class="space-y-4"
      >
        <div class="space-y-4">
          <CheckSquare class="space-y-4" />
          <span class="space-y-4"
            >{selectedEvidence.size} evidence item(s) selected</span
          >
        </div>

        <div class="space-y-4">
          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("analyze")}
            disabled={bulkOperationLoading}
            class="space-y-4"
          >
            {#if thinkingStyleEnabled}
              <Brain class="space-y-4" />
              Analyze (Thinking)
            {:else}
              <Zap class="space-y-4" />
              Quick Analyze
            {/if}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("verify")}
            disabled={bulkOperationLoading}
            class="space-y-4"
          >
            <CheckCircle class="space-y-4" />
            Verify
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("archive")}
            disabled={bulkOperationLoading}
            class="space-y-4"
          >
            <Archive class="space-y-4" />
            Archive
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("export")}
            disabled={bulkOperationLoading}
            class="space-y-4"
          >
            <Download class="space-y-4" />
            Export
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => bulkOperation("delete")}
            disabled={bulkOperationLoading}
            class="space-y-4"
          >
            <Trash2 class="space-y-4" />
            Delete
          </Button>

          <Button
            variant="outline"
            size="sm"
            onclick={() => {
              selectedEvidence.clear();
              selectedEvidence = selectedEvidence;
              showBulkActions = false;
            }}
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
    <div class="space-y-4">
      <div class="space-y-4"></div>
      <span class="space-y-4">Loading evidence...</span>
    </div>
  {:else if error}
    <div class="space-y-4" role="alert">
      <XCircle class="space-y-4" />
      <div>
        <h3 class="space-y-4">Error Loading Evidence</h3>
        <div class="space-y-4">{error}</div>
      </div>
      <Button variant="outline" size="sm" onclick={() => refreshEvidence()}>
        <RefreshCw class="space-y-4" />
        Retry
      </Button>
    </div>
  {:else if filteredEvidence.length === 0}
    <div class="space-y-4">
      <FileCheck class="space-y-4" />
      <h3 class="space-y-4">
        {searchQuery ||
        selectedType ||
        selectedStatus ||
        selectedCollector ||
        dateFrom ||
        dateTo
          ? "No matching evidence found"
          : "No evidence found"}
      </h3>
      <p class="space-y-4">
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
        <div class="space-y-4">
          <Button onclick={() => openUploadModal()} class="space-y-4">
            <Plus class="space-y-4" />
            Upload Evidence
          </Button>
          <Button
            variant="outline"
            onclick={() => handleAdvancedUpload()}
            class="space-y-4"
          >
            <Upload class="space-y-4" />
            Advanced Upload
          </Button>
        </div>
      {/if}
    </div>
  {:else}
    <!-- Evidence Header Controls -->
    <div class="space-y-4">
      <div class="space-y-4">
        <span class="space-y-4">
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
            class="space-y-4"
            aria-label="Select all visible evidence"
          >
            {#if selectedEvidence.size === visibleEvidence.length}
              <CheckSquare class="space-y-4" />
            {:else}
              <Square class="space-y-4" />
            {/if}
            Select All
          </Button>
        {/if}
      </div>
    </div>

    <!-- Enhanced Evidence Grid/List View with AI Analysis -->
    <div class="space-y-4">
      {#if viewMode === "grid"}
        <div class="space-y-4">
          {#each visibleEvidence as evidence}
            <div
              class="space-y-4"
            >
              <div class="space-y-4">
                <!-- Selection Checkbox -->
                <div class="space-y-4">
                  <input
                    type="checkbox"
                    class="space-y-4"
                    checked={selectedEvidence.has(evidence.id)}
                    onchange={() => toggleEvidenceSelection(evidence.id)}
                    aria-label="Select evidence {evidence.title ||
                      'Untitled Evidence'}"
                  />

                  <div class="space-y-4">
                    <Tooltip content="Evidence actions">
                      <Button
                        variant="ghost"
                        size="sm"
                        tabindex={0}
                        role="button"
                        aria-label="Evidence actions menu"
                      >
                        <MoreHorizontal class="space-y-4" />
                      </Button>
                    </Tooltip>
                    <ul
                      tabindex={0}
                      role="menu"
                      class="space-y-4"
                    >
                      <li>
                        <a href="/evidence/{evidence.id}" class="space-y-4">
                          <Eye class="space-y-4" />
                          View Details
                        </a>
                      </li>
                      <li>
                        <button 
                          class="space-y-4"
                          onclick={() => analyzeEvidence(evidence)}
                          disabled={analysisInProgress.has(evidence.id)}
                        >
                          {#if thinkingStyleEnabled}
                            <Brain class="space-y-4" />
                            Analyze (Thinking)
                          {:else}
                            <Zap class="space-y-4" />
                            Quick Analyze
                          {/if}
                        </button>
                      </li>
                      <li>
                        <a href="/evidence/{evidence.id}/edit" class="space-y-4">
                          <Edit class="space-y-4" />
                          Edit Evidence
                        </a>
                      </li>
                      <li>
                        <button class="space-y-4">
                          <Hash class="space-y-4" />
                          Verify Hash
                        </button>
                      </li>
                      <li>
                        <button class="space-y-4">
                          <Trash2 class="space-y-4" />
                          Delete
                        </button>
                      </li>
                    </ul>
                  </div>
                </div>

                <!-- Evidence Content -->
                <div class="space-y-4">
                  <div class="space-y-4">
                    <svelte:component
                      this={getEvidenceTypeIcon(evidence.evidenceType)}
                      class="space-y-4"
                    />
                    <h2 class="space-y-4">
                      {evidence.title || "Untitled Evidence"}
                    </h2>
                  </div>

                  <div class="space-y-4">
                    <div class="space-y-4">
                      {evidence.evidenceType || "Unknown"}
                    </div>
                    <div
                      class="space-y-4"
                    >
                      {evidence.isAdmissible ? "Admissible" : "Pending"}
                    </div>
                    {#if evidence.hash}
                      <div class="space-y-4">
                        <Shield class="space-y-4" />
                        Verified
                      </div>
                    {/if}
                    {#if evidence.aiAnalysis && Object.keys(evidence.aiAnalysis).length > 0}
                      <div class="space-y-4">
                        <Brain class="space-y-4" />
                        AI Analyzed
                      </div>
                    {/if}
                  </div>

                  <p class="space-y-4">
                    {evidence.description
                      ? evidence.description.length > 120
                        ? evidence.description.substring(0, 120) + "..."
                        : evidence.description
                      : "No description available"}
                  </p>

                  <div class="space-y-4">
                    <div class="space-y-4">
                      <Calendar class="space-y-4" />
                      Collected: {evidence.uploadedAt
                        ? new Date(evidence.uploadedAt).toLocaleDateString()
                        : "Unknown"}
                    </div>
                    {#if evidence.collectedBy}
                      <div class="space-y-4">
                        <UserIcon class="space-y-4" />
                        By: {evidence.collectedBy}
                      </div>
                    {/if}
                    {#if evidence.fileSize}
                      <div class="space-y-4">
                        <Activity class="space-y-4" />
                        Size: {(evidence.fileSize / 1024 / 1024).toFixed(2)} MB
                      </div>
                    {/if}
                  </div>
                </div>

                <!-- Enhanced Actions with AI Analysis -->
                <div class="space-y-4">
                  <Button 
                    size="sm" 
                    variant="outline"
                    onclick={() => analyzeEvidence(evidence)}
                    disabled={analysisInProgress.has(evidence.id)}
                    class="space-y-4"
                  >
                    {#if analysisInProgress.has(evidence.id)}
                      <div class="space-y-4"></div>
                      Analyzing...
                    {:else if thinkingStyleEnabled}
                      <Brain class="space-y-4" />
                      Think
                    {:else}
                      <Zap class="space-y-4" />
                      Analyze
                    {/if}
                  </Button>
                  
                  <a href="/evidence/{evidence.id}" class="space-y-4">
                    <Button size="sm">
                      <Eye class="space-y-4" />
                      View
                    </Button>
                  </a>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <!-- Enhanced List View -->
        <div class="space-y-4">
          {#each visibleEvidence as evidence}
            <div
              class="space-y-4"
            >
              <div class="space-y-4">
                <input
                  type="checkbox"
                  class="space-y-4"
                  checked={selectedEvidence.has(evidence.id)}
                  onchange={() => toggleEvidenceSelection(evidence.id)}
                  aria-label="Select evidence {evidence.title ||
                    'Untitled Evidence'}"
                />

                <svelte:component
                  this={getEvidenceTypeIcon(evidence.evidenceType)}
                  class="space-y-4"
                />

                <div class="space-y-4">
                  <div class="space-y-4">
                    <div class="space-y-4">
                      <h3 class="space-y-4">
                        {evidence.title || "Untitled Evidence"}
                      </h3>
                      <p class="space-y-4">
                        {evidence.description || "No description available"}
                      </p>
                    </div>

                    <div class="space-y-4">
                      <div class="space-y-4">
                        {evidence.evidenceType || "Unknown"}
                      </div>
                      <div
                        class="space-y-4"
                      >
                        {evidence.isAdmissible ? "Admissible" : "Pending"}
                      </div>
                      {#if evidence.hash}
                        <div class="space-y-4">
                          <Shield class="space-y-4" />
                          Verified
                        </div>
                      {/if}
                      {#if evidence.aiAnalysis && Object.keys(evidence.aiAnalysis).length > 0}
                        <div class="space-y-4">
                          <Brain class="space-y-4" />
                          AI Analyzed
                        </div>
                      {/if}
                    </div>
                  </div>

                  <div
                    class="space-y-4"
                  >
                    <div class="space-y-4">
                      <Calendar class="space-y-4" />
                      {evidence.uploadedAt
                        ? new Date(evidence.uploadedAt).toLocaleDateString()
                        : "Unknown"}
                    </div>
                    {#if evidence.collectedBy}
                      <div class="space-y-4">
                        <UserIcon class="space-y-4" />
                        {evidence.collectedBy}
                      </div>
                    {/if}
                    {#if evidence.fileSize}
                      <div class="space-y-4">
                        <Activity class="space-y-4" />
                        {(evidence.fileSize / 1024 / 1024).toFixed(2)} MB
                      </div>
                    {/if}
                  </div>
                </div>

                <div class="space-y-4">
                  <Button 
                    size="sm" 
                    variant="outline"
                    onclick={() => analyzeEvidence(evidence)}
                    disabled={analysisInProgress.has(evidence.id)}
                    class="space-y-4"
                  >
                    {#if analysisInProgress.has(evidence.id)}
                      <div class="space-y-4"></div>
                      Analyzing...
                    {:else if thinkingStyleEnabled}
                      <Brain class="space-y-4" />
                      Think
                    {:else}
                      <Zap class="space-y-4" />
                      Analyze
                    {/if}
                  </Button>

                  <a href="/evidence/{evidence.id}" class="space-y-4">
                    <Button size="sm" variant="outline">
                      <Eye class="space-y-4" />
                      View
                    </Button>
                  </a>

                  <div class="space-y-4">
                    <Tooltip content="More actions">
                      <Button
                        variant="ghost"
                        size="sm"
                        tabindex={0}
                        role="button"
                        aria-label="More actions for {evidence.title ||
                          'Untitled Evidence'}"
                      >
                        <MoreHorizontal class="space-y-4" />
                      </Button>
                    </Tooltip>
                    <ul
                      tabindex={0}
                      role="menu"
                      class="space-y-4"
                    >
                      <li>
                        <a href="/evidence/{evidence.id}/edit" class="space-y-4">
                          <Edit class="space-y-4" />
                          Edit Evidence
                        </a>
                      </li>
                      <li>
                        <button class="space-y-4">
                          <Hash class="space-y-4" />
                          Verify Hash
                        </button>
                      </li>
                      <li>
                        <button class="space-y-4">
                          <Trash2 class="space-y-4" />
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
      <div class="space-y-4">
        <div class="space-y-4">
          <Button
            variant="outline"
            size="sm"
            class="space-y-4"
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
              class="space-y-4"
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
            class="space-y-4"
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
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <h3 class="space-y-4">
          {#if thinkingStyleEnabled}
            <Brain class="space-y-4" />
            Thinking Style Analysis
          {:else}
            <Zap class="space-y-4" />
            Quick Analysis
          {/if}
          - {analysisModal.evidence.title}
        </h3>
        <Button variant="ghost" size="sm" onclick={closeAnalysisModal}>
          ✕
        </Button>
      </div>
      
      <div class="space-y-4">
        <div class="space-y-4">
          <div class="space-y-4">{formatAnalysisForDisplay(analysisModal.result)}</div>
        </div>
        
        {#if analysisModal.result.reasoning_steps && analysisModal.result.reasoning_steps.length > 0}
          <div class="space-y-4">
            <h4 class="space-y-4">Reasoning Steps:</h4>
            <ol class="space-y-4">
              {#each analysisModal.result.reasoning_steps as step}
                <li class="space-y-4">{step}</li>
              {/each}
            </ol>
          </div>
        {/if}
      </div>
      
      <div class="space-y-4">
        <Button variant="outline" onclick={closeAnalysisModal}>Close</Button>
        <Button onclick={() => {
          // Save analysis or perform other actions
          closeAnalysisModal();
        }}>Save Analysis</Button>
      </div>
    </div>
    <div class="space-y-4" onclick={closeAnalysisModal}></div>
  </div>
{/if}

{#if showAdvancedUpload}
  <div class="space-y-4">
    <div class="space-y-4">
      <h3 class="space-y-4">Advanced Evidence Upload</h3>
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
      class="space-y-4"
      role="button"
      tabindex={0}
      aria-label="Close modal"
      onclick={() => (showAdvancedUpload = false)}
      onkeydown={(e) => e.key === "Escape" && (showAdvancedUpload = false)}
    ></div>
  </div>
{/if}

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical
    overflow: hidden
}
  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical
    overflow: hidden
}
</style>
