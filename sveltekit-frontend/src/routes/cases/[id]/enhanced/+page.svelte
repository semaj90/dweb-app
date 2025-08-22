<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from "svelte/store";
  import type { PageData } from "./$types";
// Import our enhanced components
  import AdvancedRichTextEditor from "$lib/components/AdvancedRichTextEditor.svelte";
  import EnhancedCanvasEditor from "$lib/components/EnhancedCanvasEditor.svelte";
  import EvidenceUploader from "$lib/components/EvidenceUploader.svelte";
  import { aiSummarizationService } from "$lib/services/aiSummarizationService";

  export let data: PageData;

  // Use correct SvelteKit param for caseId
  const caseId = data.case.id;
  let caseData = data.case;
  let evidenceList = writable(data.evidence || []);

  // UI state management
  let activeTab = writable("canvas");
  let sidebarOpen = writable(true);
  let aiGenerating = false;
  let aiReports = writable<unknown[]>([]);

  // Component references for integration
  let canvasEditor: any;
  let reportEditor: any;

  // Case summary and reports
  let caseSummary = writable("");
  let aiAnalysisComplete = false;

  // Enhanced state management
  let loadingStates = {
    evidence: false,
    reports: false,
    aiAnalysis: false,
    canvasSave: false,
    reportSave: false,
  };

  let errorMessages = writable<string[]>([]);
  let successMessages = writable<string[]>([]);

  // Helper functions for user feedback
  function addErrorMessage(message: string) {
    errorMessages.update((msgs) => [...msgs, message]);
    setTimeout(() => {
      errorMessages.update((msgs) => msgs.slice(1));
    }, 5000);}
  function addSuccessMessage(message: string) {
    successMessages.update((msgs) => [...msgs, message]);
    setTimeout(() => {
      successMessages.update((msgs) => msgs.slice(1));
    }, 3000);}
  onMount(() => {
    loadEvidenceList();
    loadAIReports();

    // Add keyboard shortcuts
    const handleKeyPress = (e: CustomEvent<any>) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "b":
            e.preventDefault();
            sidebarOpen.update((open) => !open);
            break;
          case "1":
            e.preventDefault();
            activeTab.set("canvas");
            break;
          case "2":
            e.preventDefault();
            activeTab.set("editor");
            break;
          case "3":
            e.preventDefault();
            activeTab.set("evidence");
            break;
          case "4":
            e.preventDefault();
            activeTab.set("reports");
            break;}}
    };

    window.addEventListener("keydown", handleKeyPress);

    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  });

  async function loadEvidenceList() {
    loadingStates.evidence = true;
    try {
      const response = await fetch(`/api/evidence?caseId=${caseId}`);
      if (response.ok) {
        const data = await response.json();
        evidenceList.set(data.evidence || []);
      } else {
        addErrorMessage("Failed to load evidence: " + response.statusText);}
    } catch (error) {
      console.error("Failed to load evidence:", error);
      addErrorMessage("Error loading evidence. Please try again later.");
    } finally {
      loadingStates.evidence = false;}}
  async function loadAIReports() {
    loadingStates.reports = true;
    try {
      const response = await fetch(`/api/reports?caseId=${caseId}`);
      if (response.ok) {
        const data = await response.json();
        aiReports.set(data.reports || []);
      } else {
        addErrorMessage("Failed to load AI reports: " + response.statusText);}
    } catch (error) {
      console.error("Failed to load AI reports:", error);
      addErrorMessage("Error loading AI reports. Please try again later.");
    } finally {
      loadingStates.reports = false;}}
  async function handleEvidenceUploaded(event: any) {
    const { evidence } = event.detail;

    // Add to evidence list
    evidenceList.update((list) => [...list, evidence]);

    // Add to canvas automatically
    if (canvasEditor) {
      canvasEditor.addEvidenceToCanvas(evidence);}
    // Trigger AI analysis of new evidence
    await analyzeNewEvidence(evidence);}
  async function analyzeNewEvidence(evidence: any) {
    try {
      aiGenerating = true;
      addSuccessMessage("AI analysis started for new evidence.");

      // Get AI analysis for the evidence
      const analysisResponse = await fetch(`/api/ai/analyze-evidence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evidenceId: evidence.id, caseId }),
      });

      if (!analysisResponse.ok) {
        throw new Error("Failed to generate AI analysis");}
      const analysis = await analysisResponse.json();

      // Update evidence with AI analysis
      const response = await fetch(`/api/evidence/${evidence.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          aiAnalysis: analysis,
          tags: analysis.extractedTags || [],
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to update evidence with AI analysis.");}
      // Refresh evidence list
      await loadEvidenceList();
      addSuccessMessage("AI analysis completed successfully.");
    } catch (error) {
      console.error("AI analysis failed:", error);
      addErrorMessage("AI analysis failed. Please try again.");
    } finally {
      aiGenerating = false;}}
  async function generateCaseSummary() {
    try {
      aiGenerating = true;
      addSuccessMessage("Generating AI case summary...");

      const caseReport = await aiSummarizationService.generateCaseSummaryReport(
        {
          id: caseId,
          title: caseData.title,
          description: caseData.description,
          evidence: $evidenceList,
          activities: [], // Could load case activities
          metadata: caseData.metadata || {},}
      );

      // Save the report
      const response = await fetch("/api/reports/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          caseId,
          reportType: "case_overview",
          title: `Case Summary - ${caseData.title}`,
          content: caseReport.content,
          richTextContent: caseReport.richTextContent,
          metadata: caseReport.metadata,
          canvasElements: caseReport.canvasElements,
        }),
      });

      if (response.ok) {
        const savedReport = await response.json();
        aiReports.update((reports) => [...reports, savedReport]);

        // Add canvas elements to the canvas
        if (canvasEditor && caseReport.canvasElements) {
          canvasEditor.addElementsToCanvas(caseReport.canvasElements);}
        // Load rich text content into report editor
        if (reportEditor && caseReport.richTextContent) {
          reportEditor.setContent(caseReport.richTextContent);}
        caseSummary.set(caseReport.content);
        aiAnalysisComplete = true;
        addSuccessMessage("Case summary generated successfully.");
      } else {
        addErrorMessage(
          "Failed to save case summary report: " + response.statusText
        );}
    } catch (error) {
      console.error("Failed to generate case summary:", error);
      addErrorMessage("Error generating case summary. Please try again.");
    } finally {
      aiGenerating = false;}}
  async function generateProsecutionStrategy() {
    try {
      aiGenerating = true;
      addSuccessMessage("Generating AI prosecution strategy...");

      const strategy = await aiSummarizationService.generateProsecutionStrategy(
        {
          id: caseId,
          title: caseData.title,
          description: caseData.description,
          evidence: $evidenceList,
          activities: [],
          metadata: caseData.metadata || {},}
      );

      const response = await fetch("/api/reports/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          caseId,
          reportType: "prosecution_strategy",
          title: `Prosecution Strategy - ${caseData.title}`,
          content: strategy.content,
          richTextContent: strategy.richTextContent,
          metadata: strategy.metadata,
          canvasElements: strategy.canvasElements,
        }),
      });

      if (response.ok) {
        const savedReport = await response.json();
        aiReports.update((reports) => [...reports, savedReport]);

        if (canvasEditor && strategy.canvasElements) {
          canvasEditor.addElementsToCanvas(strategy.canvasElements);}
        addSuccessMessage("Prosecution strategy generated successfully.");
      } else {
        addErrorMessage(
          "Failed to save prosecution strategy report: " + response.statusText
        );}
    } catch (error) {
      console.error("Failed to generate prosecution strategy:", error);
      addErrorMessage(
        "Error generating prosecution strategy. Please try again."
      );
    } finally {
      aiGenerating = false;}}
  function handleCanvasChange(event: any) {
    // Auto-save canvas state
    const canvasState = event.detail;
    saveCanvasState(canvasState);}
  async function saveCanvasState(canvasState: any) {
    loadingStates.canvasSave = true;
    try {
      await fetch("/api/canvas/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          caseId,
          canvasState,
          timestamp: new Date().toISOString(),
        }),
      });
      addSuccessMessage("Canvas state saved successfully.");
    } catch (error) {
      console.error("Failed to save canvas state:", error);
      addErrorMessage("Error saving canvas state. Please try again.");
    } finally {
      loadingStates.canvasSave = false;}}
  function handleReportChange(event: any) {
    // Auto-save report content
    const content = event.detail;
    saveReportContent(content);}
  async function saveReportContent(content: any) {
    loadingStates.reportSave = true;
    try {
      await fetch("/api/reports/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          caseId,
          reportType: "case_notes",
          title: `Case Notes - ${caseData.title}`,
          richTextContent: content,
          timestamp: new Date().toISOString(),
        }),
      });
      addSuccessMessage("Report content saved successfully.");
    } catch (error) {
      console.error("Failed to save report content:", error);
      addErrorMessage("Error saving report content. Please try again.");
    } finally {
      loadingStates.reportSave = false;}}
</script>

<svelte:head>
  <title>Case: {caseData.title} - Legal Case Management</title>
</svelte:head>

<div class="container mx-auto px-4">
  <!-- Header with case info and AI controls -->
  <header class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <h1 class="container mx-auto px-4">{caseData.title}</h1>
      <div class="container mx-auto px-4">
        <span class="container mx-auto px-4">#{caseData.caseNumber}</span>
        <span
          class="container mx-auto px-4"
          class:status-open={caseData.status === "open"}
        >
          {caseData.status}
        </span>
        <span class="container mx-auto px-4"
          >{$evidenceList.length} pieces of evidence</span
        >
      </div>
    </div>

    <div class="container mx-auto px-4">
      <button
        class="container mx-auto px-4"
        onclick={() => sidebarOpen.update((open) => !open)}
      >
        {$sidebarOpen ? "◀" : "▶"}
        {$sidebarOpen ? "Hide" : "Show"} Sidebar
      </button>

      <button
        class="container mx-auto px-4"
        onclick={() => generateCaseSummary()}
        disabled={aiGenerating}
      >
        {#if aiGenerating}
          <span class="container mx-auto px-4">⏳</span>
        {:else}
          🤖
        {/if}
        Generate AI Summary
      </button>

      <button
        class="container mx-auto px-4"
        onclick={() => generateProsecutionStrategy()}
        disabled={aiGenerating}
      >
        📋 Prosecution Strategy
      </button>
    </div>
  </header>

  <!-- Notification Bar -->
  {#if $errorMessages.length > 0 || $successMessages.length > 0}
    <div class="container mx-auto px-4">
      {#each $errorMessages as error}
        <div class="container mx-auto px-4">
          ❌ {error}
        </div>
      {/each}
      {#each $successMessages as success}
        <div class="container mx-auto px-4">
          ✅ {success}
        </div>
      {/each}
    </div>
  {/if}

  <div class="container mx-auto px-4">
    <!-- Sidebar with evidence and tools -->
    <aside class="container mx-auto px-4" class:open={$sidebarOpen}>
      <div class="container mx-auto px-4">
        <button
          class="container mx-auto px-4"
          class:active={$activeTab === "evidence"}
          onclick={() => activeTab.set("evidence")}
          title="Evidence List (Ctrl/Cmd + 3)"
        >
          📁 Evidence
        </button>
        <button
          class="container mx-auto px-4"
          class:active={$activeTab === "reports"}
          onclick={() => activeTab.set("reports")}
          title="AI Reports (Ctrl/Cmd + 4)"
        >
          📊 AI Reports
        </button>
      </div>

      <div class="container mx-auto px-4">
        {#if $activeTab === "evidence"}
          <div class="container mx-auto px-4">
            <EvidenceUploader {caseId} onuploaded={handleEvidenceUploaded} />

            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4">
                <h3>Evidence ({$evidenceList.length})</h3>
                {#if loadingStates.evidence}
                  <div class="container mx-auto px-4">
                    <span class="container mx-auto px-4"></span>
                    Loading...
                  </div>
                {/if}
              </div>
              {#each $evidenceList as evidence}
                <div class="container mx-auto px-4" draggable={true}>
                  <div class="container mx-auto px-4">
                    {#if evidence.fileUrl}
                      <img
                        src="/api/upload?file={evidence.fileUrl}&thumbnail=true"
                        alt="Thumbnail"
                      />
                    {:else}
                      <div class="container mx-auto px-4">
                        {evidence.evidenceType === "image"
                          ? "🖼️"
                          : evidence.evidenceType === "video"
                            ? "🎥"
                            : "📄"}
                      </div>
                    {/if}
                  </div>
                  <div class="container mx-auto px-4">
                    <div class="container mx-auto px-4">{evidence.title}</div>
                    <div class="container mx-auto px-4">
                      {evidence.evidenceType} • {evidence.fileSize
                        ? (evidence.fileSize / 1024).toFixed(1) + "KB"
                        : ""}
                    </div>
                    {#if Array.isArray(evidence.tags) && evidence.tags.length > 0}
                      <div class="container mx-auto px-4">
                        {#each evidence.tags.slice(0, 2) as tag}
                          <span class="container mx-auto px-4">{tag}</span>
                        {/each}
                      </div>
                    {/if}
                  </div>
                  <button
                    class="container mx-auto px-4"
                    onclick={() => canvasEditor?.addEvidenceToCanvas(evidence)}
                  >
                    ➕
                  </button>
                </div>
              {/each}
            </div>
          </div>
        {:else if $activeTab === "reports"}
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              <h3>AI Generated Reports</h3>
              {#if loadingStates.reports}
                <div class="container mx-auto px-4">
                  <span class="container mx-auto px-4"></span>
                  Loading reports...
                </div>
              {/if}
            </div>
            {#each $aiReports as report}
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">{report.title}</div>
                <div class="container mx-auto px-4">
                  {report.reportType} • {new Date(
                    report.generatedAt
                  ).toLocaleDateString()}
                </div>
                <button
                  class="container mx-auto px-4"
                  onclick={() =>
                    reportEditor?.setContent(report.richTextContent)}
                >
                  Load into Editor
                </button>
              </div>
            {/each}

            {#if aiAnalysisComplete}
              <div class="container mx-auto px-4">
                <h4>Case Summary</h4>
                <p>{$caseSummary}</p>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    </aside>

    <!-- Main content area with tabs -->
    <main class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <button
          class="container mx-auto px-4"
          class:active={$activeTab === "canvas"}
          onclick={() => activeTab.set("canvas")}
          title="Interactive Canvas (Ctrl/Cmd + 1)"
        >
          🎨 Interactive Canvas
        </button>
        <button
          class="container mx-auto px-4"
          class:active={$activeTab === "editor"}
          onclick={() => activeTab.set("editor")}
          title="Report Editor (Ctrl/Cmd + 2)"
        >
          📝 Report Editor
        </button>

        <div class="container mx-auto px-4">
          <span class="container mx-auto px-4"
            >Shortcuts: Ctrl/Cmd + B (toggle sidebar), 1-4 (switch tabs)</span
          >
        </div>
      </div>

      <div class="container mx-auto px-4">
        {#if $activeTab === "canvas"}
          <div class="container mx-auto px-4">
            {#if loadingStates.canvasSave}
              <div class="container mx-auto px-4">
                <span class="container mx-auto px-4"></span>
                Saving canvas...
              </div>
            {/if}
            <EnhancedCanvasEditor
              bind:this={canvasEditor}
              {caseId}
              oncanvaschange={handleCanvasChange}
              width={1200}
              height={600}
            />
          </div>
        {:else if $activeTab === "editor"}
          <div class="container mx-auto px-4">
            {#if loadingStates.reportSave}
              <div class="container mx-auto px-4">
                <span class="container mx-auto px-4"></span>
                Saving report...
              </div>
            {/if}
            <AdvancedRichTextEditor
              bind:this={reportEditor}
              oncontentchange={handleReportChange}
              placeholder="Write your case report, notes, or analysis here..."
            />
          </div>
        {/if}
      </div>
    </main>
  </div>
</div>

<style>
  /* @unocss-include */
  .case-workspace {
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;}
  .notification-bar {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--background-alt, #f8f9fa);
    border-bottom: 1px solid var(--border, #dee2e6);}
  .notification {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-size: 0.875rem;
    animation: slideIn 0.3s ease-out;}
  .notification.error {
    background: var(--error-light, #f8d7da);
    color: var(--error-dark, #721c24);
    border: 1px solid var(--error, #dc3545);}
  .notification.success {
    background: var(--success-light, #d4edda);
    color: var(--success-dark, #155724);
    border: 1px solid var(--success, #28a745);}
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-10px);}
    to {
      opacity: 1;
      transform: translateY(0);}}
  .case-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: var(--surface, #fff);
    border-bottom: 1px solid var(--border, #dee2e6);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);}
  .case-title {
    margin: 0;
    font-size: 1.5rem;
    color: var(--text-primary, #333);}
  .case-meta {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;}
  .case-number {
    font-weight: 600;
    color: var(--primary, #007bff);}
  .case-status {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    background: var(--warning, #ffc107);
    color: white;}
  .case-status.status-open {
    background: var(--success, #28a745);}
  .evidence-count {
    color: var(--text-secondary, #666);
    font-size: 0.875rem;}
  .ai-controls {
    display: flex;
    gap: 1rem;}
  .btn-ai {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s ease;}
  .btn-ai:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);}
  .btn-ai:disabled {
    opacity: 0.6;
    cursor: not-allowed;}
  .btn-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--surface, #fff);
    color: var(--text-primary, #333);
    border: 1px solid var(--border, #dee2e6);
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s ease;}
  .btn-toggle:hover {
    background: var(--background-alt, #f8f9fa);
    border-color: var(--primary, #007bff);}
@keyframes spin {
    from {
      transform: rotate(0deg);}
    to {
      transform: rotate(360deg);}}
  .workspace-layout {
    display: flex;
    flex: 1;
    overflow: hidden;}
  .sidebar {
    width: 350px;
    background: var(--background-alt, #f8f9fa);
    border-right: 1px solid var(--border, #dee2e6);
    display: flex;
    flex-direction: column;
    transition: transform 0.3s ease;}
  .sidebar:not(.open) {
    transform: translateX(-100%);}
  .sidebar-tabs {
    display: flex;
    background: var(--surface, #fff);
    border-bottom: 1px solid var(--border, #dee2e6);}
  .tab-btn {
    flex: 1;
    padding: 0.75rem 1rem;
    border: none;
    background: transparent;
    cursor: pointer;
    transition: background-color 0.3s ease;}
  .tab-btn.active {
    background: var(--primary, #007bff);
    color: white;}
.evidence-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background: var(--surface, #fff);
    border-radius: 8px;
    border: 1px solid var(--border, #dee2e6);
    cursor: grab;
    transition: all 0.3s ease;}
  .evidence-item:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);}
  .evidence-thumbnail {
    width: 40px;
    height: 40px;
    border-radius: 4px;
    overflow: hidden;
    background: var(--background-alt, #f8f9fa);
    display: flex;
    align-items: center;
    justify-content: center;}
  .evidence-thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;}
  .file-icon {
    font-size: 1.5rem;}
  .evidence-info {
    flex: 1;}
  .evidence-title {
    font-weight: 500;
    font-size: 0.875rem;
    color: var(--text-primary, #333);}
  .evidence-meta {
    font-size: 0.75rem;
    color: var(--text-secondary, #666);
    margin-top: 0.25rem;}
  .ai-tags {
    display: flex;
    gap: 0.25rem;
    margin-top: 0.5rem;}
  .tag {
    padding: 0.125rem 0.375rem;
    background: var(--primary-light, #e7f3ff);
    color: var(--primary, #007bff);
    border-radius: 4px;
    font-size: 0.75rem;}
  .add-to-canvas-btn {
    padding: 0.25rem 0.5rem;
    background: var(--success, #28a745);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;}
  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;}
  .content-tabs {
    display: flex;
    align-items: center;
    background: var(--surface, #fff);
    border-bottom: 1px solid var(--border, #dee2e6);}
  .tab-shortcuts {
    margin-left: auto;
    padding: 0.75rem 1rem;}
  .shortcuts-hint {
    font-size: 0.75rem;
    color: var(--text-secondary, #666);
    font-style: italic;}
  .content-area {
    flex: 1;
    overflow: hidden;}
  .canvas-container,
  .editor-container {
    height: 100%;
    padding: 1rem;}
  .report-item {
    padding: 1rem;
    background: var(--surface, #fff);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border, #dee2e6);}
  .report-title {
    font-weight: 500;
    margin-bottom: 0.5rem;}
  .report-meta {
    font-size: 0.875rem;
    color: var(--text-secondary, #666);
    margin-bottom: 0.5rem;}
  .btn-small {
    padding: 0.25rem 0.75rem;
    background: var(--primary, #007bff);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;}
  .ai-summary {
    padding: 1rem;
    background: var(--success-light, #d4edda);
    border-radius: 8px;
    margin-top: 1rem;}
  .ai-summary h4 {
    margin: 0 0 0.5rem 0;
    color: var(--success-dark, #155724);}
  .ai-summary p {
    margin: 0;
    color: var(--success-dark, #155724);
    font-size: 0.875rem;}
  /* Loading indicators */
  .loading-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-secondary, #666);
    font-size: 0.875rem;}
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border, #dee2e6);
    border-left: 2px solid var(--primary, #007bff);
    border-radius: 50%;
    animation: spin 1s linear infinite;}
  .evidence-list-header,
  .reports-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;}
  .evidence-list-header h3,
  .reports-header h3 {
    margin: 0;}
  .canvas-saving-indicator,
  .editor-saving-indicator {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: var(--surface, #fff);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    border: 1px solid var(--border, #dee2e6);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: var(--text-secondary, #666);
    z-index: 1000;}
  .canvas-container,
  .editor-container {
    position: relative;}
  /* Responsive design */
  @media (max-width: 768px) {
    .case-header {
      flex-direction: column;
      gap: 1rem;
      align-items: flex-start;}
    .case-meta {
      flex-direction: column;
      gap: 0.5rem;}
    .ai-controls {
      flex-direction: column;
      gap: 0.5rem;
      width: 100%;}
    .btn-ai,
    .btn-toggle {
      width: 100%;
      justify-content: center;}
    .sidebar {
      position: fixed;
      top: 0;
      left: 0;
      height: 100vh;
      z-index: 1000;
      background: var(--surface, #fff);
      box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);}
    .sidebar:not(.open) {
      transform: translateX(-100%);}
    .workspace-layout {
      position: relative;}
    .main-content {
      width: 100%;}
    .content-tabs {
      flex-direction: column;
      gap: 0.5rem;}
    .tab-shortcuts {
      display: none;}
    .shortcuts-hint {
      display: none;}}
  @media (max-width: 480px) {
    .case-header {
      padding: 0.5rem 1rem;}
    .case-title {
      font-size: 1.25rem;}
    .sidebar {
      width: 100%;}
    .evidence-item {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.5rem;}
    .evidence-thumbnail {
      width: 100%;
      height: 80px;}
    .add-to-canvas-btn {
      align-self: flex-end;}}
</style>
