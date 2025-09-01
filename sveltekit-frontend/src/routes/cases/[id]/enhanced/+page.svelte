<script lang="ts">
  import { onMount } from "svelte";
  import type { PageData } from "./$types";
  // Import our enhanced components
  import AdvancedRichTextEditor from "$lib/components/AdvancedRichTextEditor.svelte";
  import EnhancedCanvasEditor from "$lib/components/EnhancedCanvasEditor.svelte";
  import EvidenceUploader from "$lib/components/EvidenceUploader.svelte";
  import LocalImageGenerator from "$lib/components/ai/LocalImageGenerator.svelte";
  import { aiSummarizationService } from "$lib/services/aiSummarizationService";

  // ===== PHASE 1: CACHE-FIRST INTEGRATION =====
  import { cacheFirstService, createCacheQuery } from "$lib/services/cache-first-architecture";
  import { formManager } from "$lib/forms/enhanced-cache-forms";
  import { derived } from "svelte/store";

  // Receive load() data from SvelteKit - Svelte 5 props pattern
  let { data }: { data: PageData } = $props();

  // Use correct SvelteKit param for caseId
  const caseId = data.case.id;
  let caseData = data.case;

  // ===== CACHE-FIRST DATA MANAGEMENT =====
  // Cache-first queries with automatic background sync
  const caseQuery = createCacheQuery(
    () => cacheFirstService.getCaseById(caseId),
    `case-${caseId}`,
    { refetchInterval: 30000, staleTime: 10000 } // Refresh every 30s, stale after 10s
  );

  const evidenceQuery = createCacheQuery(
    () => cacheFirstService.getEvidenceForCase(caseId),
    `evidence-${caseId}`,
    { refetchInterval: 15000, staleTime: 5000 } // More frequent updates for evidence
  );

  // Reactive stores from cache service
  let evidenceList = $state<any[]>(data.evidence || []);
  let cachedCase = $state<any>(null);

  // Subscribe to cache updates
  $: if ($cacheQuery.data) {
    cachedCase = $cacheQuery.data;
    if (cachedCase) caseData = cachedCase;
  }

  $: if ($evidenceQuery.data) {
    evidenceList = $evidenceQuery.data;
  }

  // UI state management - Svelte 5 runes with cache-aware states
  let activeTab = $state("canvas");
  let sidebarOpen = $state(true);
  let aiGenerating = $state(false);
  let aiReports = $state<any[]>([]);

  // Cache statistics display
  const cacheStats = derived([cacheFirstService.stats], ([$stats]) => $stats);

  // Component references for integration
  let canvasEditor: any;
  let reportEditor: any;

  // Case summary and reports - Svelte 5 runes
  let caseSummary = $state("");
  let aiAnalysisComplete = $state(false);

  // Enhanced state management
  let loadingStates = $state({
    evidence: false,
    reports: false,
    aiAnalysis: false,
    canvasSave: false,
    reportSave: false,
  });

  let errorMessages = $state<string[]>([]);
  let successMessages = $state<string[]>([]);

  // Helper functions for user feedback
  function addErrorMessage(message: string) {
    errorMessages = [...errorMessages, message];
    setTimeout(() => {
      errorMessages = errorMessages.slice(1);
    }, 5000);
  }
  function addSuccessMessage(message: string) {
    successMessages = [...successMessages, message];
    setTimeout(() => {
      successMessages = successMessages.slice(1);
    }, 3000);
  }

  onMount(() => {
    loadEvidenceList();
    loadAIReports();

    // Add keyboard shortcuts
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "b":
            e.preventDefault();
            sidebarOpen = !sidebarOpen;
            break;
          case "1":
            e.preventDefault();
            activeTab = "canvas";
            break;
          case "2":
            e.preventDefault();
            activeTab = "editor";
            break;
          case "3":
            e.preventDefault();
            activeTab = "evidence";
            break;
          case "4":
            e.preventDefault();
            activeTab = "reports";
            break;
          case "5":
            e.preventDefault();
            activeTab = "images";
            break;
        }
      }
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
        const responseData = await response.json();
        evidenceList = responseData.evidence || [];
      } else {
        addErrorMessage("Failed to load evidence: " + response.statusText);
      }
    } catch (err) {
      console.error("Failed to load evidence:", err);
      addErrorMessage("Error loading evidence. Please try again later.");
    } finally {
      loadingStates.evidence = false;
    }
  }

  async function loadAIReports() {
    loadingStates.reports = true;
    try {
      const response = await fetch(`/api/reports?caseId=${caseId}`);
      if (response.ok) {
        const responseData = await response.json();
        aiReports = responseData.reports || [];
      } else {
        addErrorMessage("Failed to load AI reports: " + response.statusText);
      }
    } catch (err) {
      console.error("Failed to load AI reports:", err);
      addErrorMessage("Error loading AI reports. Please try again later.");
    } finally {
      loadingStates.reports = false;
    }
  }

  async function handleEvidenceUploaded(event: any) {
    const { evidence } = event.detail;
    evidenceList = [...evidenceList, evidence];
    if (canvasEditor) {
      canvasEditor.addEvidenceToCanvas(evidence);
    }
    await analyzeNewEvidence(evidence);
  }

  async function analyzeNewEvidence(evidence: any) {
    try {
      aiGenerating = true;
      addSuccessMessage("AI analysis started for new evidence.");
      const analysisResponse = await fetch(`/api/ai/analyze-evidence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evidenceId: evidence.id, caseId }),
      });

      if (!analysisResponse.ok) {
        throw new Error("Failed to generate AI analysis");
      }
      const analysis = await analysisResponse.json();

      const response = await fetch(`/api/evidence/${evidence.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          aiAnalysis: analysis,
          tags: analysis.extractedTags || [],
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to update evidence with AI analysis.");
      }
      await loadEvidenceList();
      addSuccessMessage("AI analysis completed successfully.");
    } catch (err) {
      console.error("AI analysis failed:", err);
      addErrorMessage("AI analysis failed. Please try again.");
    } finally {
      aiGenerating = false;
    }
  }

  async function generateCaseSummary() {
    try {
      aiGenerating = true;
      addSuccessMessage("Generating AI case summary...");
      const caseReport = await aiSummarizationService.generateCaseSummaryReport({
        id: caseId,
        title: caseData.title,
        description: caseData.description,
        evidence: evidenceList,
        activities: [],
        metadata: caseData.metadata || {},
      });

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
        aiReports = [...aiReports, savedReport];
        if (canvasEditor && caseReport.canvasElements) {
          canvasEditor.addElementsToCanvas(caseReport.canvasElements);
        }
        if (reportEditor && caseReport.richTextContent) {
          reportEditor.setContent(caseReport.richTextContent);
        }
        caseSummary = caseReport.content;
        aiAnalysisComplete = true;
        addSuccessMessage("Case summary generated successfully.");
      } else {
        addErrorMessage("Failed to save case summary report: " + response.statusText);
      }
    } catch (err) {
      console.error("Failed to generate case summary:", err);
      addErrorMessage("Error generating case summary. Please try again.");
    } finally {
      aiGenerating = false;
    }
  }

  async function generateProsecutionStrategy() {
    try {
      aiGenerating = true;
      addSuccessMessage("Generating AI prosecution strategy...");
      const strategy = await aiSummarizationService.generateProsecutionStrategy({
        id: caseId,
        title: caseData.title,
        description: caseData.description,
        evidence: evidenceList,
        activities: [],
        metadata: caseData.metadata || {},
      });

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
        aiReports = [...aiReports, savedReport];
        if (canvasEditor && strategy.canvasElements) {
          canvasEditor.addElementsToCanvas(strategy.canvasElements);
        }
        addSuccessMessage("Prosecution strategy generated successfully.");
      } else {
        addErrorMessage("Failed to save prosecution strategy report: " + response.statusText);
      }
    } catch (err) {
      console.error("Failed to generate prosecution strategy:", err);
      addErrorMessage("Error generating prosecution strategy. Please try again.");
    } finally {
      aiGenerating = false;
    }
  }

  function handleCanvasChange(event: any) {
    saveCanvasState(event.detail);
  }

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
    } catch (err) {
      console.error("Failed to save canvas state:", err);
      addErrorMessage("Error saving canvas state. Please try again.");
    } finally {
      loadingStates.canvasSave = false;
    }
  }

  function handleReportChange(event: any) {
    saveReportContent(event.detail);
  }

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
    } catch (err) {
      console.error("Failed to save report content:", err);
      addErrorMessage("Error saving report content. Please try again.");
    } finally {
      loadingStates.reportSave = false;
    }
  }

  // Handle AI-generated images
  async function handleImageGenerated(result: any) {
    try {
      // Create evidence record for generated image
      const evidence = {
        caseId: caseId,
        title: `AI Generated: ${result.prompt.substring(0, 50)}...`,
        description: `Generated image from prompt: ${result.prompt}\n\nProvider: ${result.provider}\nStyle: ${result.parameters.style || 'realistic'}`,
        evidenceType: 'image',
        fileUrl: result.imageUrl,
        metadata: {
          aiGenerated: true,
          provider: result.provider,
          parameters: result.parameters,
          generatedAt: result.timestamp,
          prompt: result.prompt
        },
        tags: ['ai-generated', result.provider, result.parameters.style || 'realistic']
      };

      // Add to evidence list immediately for responsive UI
      evidenceList = [...evidenceList, evidence];

      // Add to canvas if it's available
      if (canvasEditor) {
        canvasEditor.addEvidenceToCanvas(evidence);
      }

      addSuccessMessage(`AI-generated image added as evidence: "${evidence.title}"`);
      
      // Trigger cache refresh
      if (evidenceQuery) {
        evidenceQuery.invalidate();
      }
      
    } catch (error) {
      console.error('Failed to handle generated image:', error);
      addErrorMessage('Failed to save generated image as evidence.');
    }
  }
</script>

<svelte:head>
  <title>Case: {caseData.title} - Legal Case Management</title>
</svelte:head>

<div class="case-workspace container mx-auto px-4">
  <header class="case-header">
    <div>
      <h1 class="case-title">{caseData.title}</h1>
      <div class="case-meta">
        <span class="case-number">#{caseData.caseNumber}</span>
        <span class="case-status" class:status-open={caseData.status === "open"}>
          {caseData.status}
        </span>
        <span class="evidence-count">{evidenceList.length} pieces of evidence</span>
      </div>
    </div>

    <div class="ai-controls">
      <button class="btn-toggle" on:on:onclick={() => sidebarOpen = !sidebarOpen}>
        {sidebarOpen ? "◀ Hide Sidebar" : "▶ Show Sidebar"}
      </button>

      <button class="btn-ai" on:on:onclick={generateCaseSummary} disabled={aiGenerating}>
        {#if aiGenerating}
          <span class="spinner" /> Generating...
        {:else}
          🤖 Generate AI Summary
        {/if}
      </button>

      <button class="btn-ai" on:on:onclick={generateProsecutionStrategy} disabled={aiGenerating}>
        📋 Prosecution Strategy
      </button>

      <a href="/cases/{caseId}/images" class="btn-ai nes-btn is-warning">
        🎨 Image Gallery
      </a>
    </div>
  </header>

  {#if errorMessages.length > 0 || successMessages.length > 0}
    <div class="notification-bar">
      {#each errorMessages as err}
        <div class="notification error">❌ {err}</div>
      {/each}
      {#each successMessages as msg}
        <div class="notification success">✅ {msg}</div>
      {/each}
    </div>
  {/if}

  <div class="workspace-layout ssr-layout bits-ui">
    <aside class="sidebar nes-container with-title" class:open={sidebarOpen} aria-hidden={!sidebarOpen} title="Sidebar">
      <p class="title">Case Tools</p>
      <div class="sidebar-tabs">
        <button class="tab-btn nes-btn" class:active={activeTab === 'evidence'} on:on:onclick={() => activeTab = "evidence"} title="Evidence List (Ctrl/Cmd + 3)">📁 Evidence</button>
        <button class="tab-btn nes-btn" class:active={activeTab === 'reports'} on:on:onclick={() => activeTab = "reports"} title="AI Reports (Ctrl/Cmd + 4)">📊 AI Reports</button>
        <button class="tab-btn nes-btn" class:active={activeTab === 'images'} on:on:onclick={() => activeTab = "images"} title="Image Generator (Ctrl/Cmd + 5)">🎨 Image Gen</button>
      </div>

      <div class="p-4">
        {#if activeTab === "evidence"}
          <EvidenceUploader {caseId} uploaded={handleEvidenceUploaded} />
          <div class="evidence-list-header nes-container is-rounded">
            <h3>Evidence ({evidenceList.length})</h3>
            {#if loadingStates.evidence}
              <div class="loading-indicator"><span class="spinner"></span> Loading...</div>
            {/if}
          </div>

          <div>
            {#each evidenceList as evidence}
              <div class="evidence-item nes-container is-rounded" draggable={true}>
                <div class="evidence-thumbnail">
                  {#if evidence.fileUrl}
                    <img src={`/api/upload?file=${encodeURIComponent(evidence.fileUrl)}&thumbnail=true`} alt="Thumbnail" />
                  {:else}
                    <div class="file-icon">
                      {evidence.evidenceType === "image" ? "🖼️" : evidence.evidenceType === "video" ? "🎥" : "📄"}
                    </div>
                  {/if}
                </div>

                <div class="evidence-info">
                  <div class="evidence-title">{evidence.title}</div>
                  <div class="evidence-meta">{evidence.evidenceType} • {evidence.fileSize ? (evidence.fileSize / 1024).toFixed(1) + "KB" : ""}</div>

                  {#if Array.isArray(evidence.tags) && evidence.tags.length > 0}
                    <div class="ai-tags">
                      {#each evidence.tags.slice(0, 2) as tag}
                        <span class="tag nes-badge is-primary">{tag}</span>
                      {/each}
                    </div>
                  {/if}
                </div>

                <div class="evidence-actions">
                  <button class="add-to-canvas-btn nes-btn is-success" on:on:onclick={() => canvasEditor?.addEvidenceToCanvas(evidence)}>➕ Add</button>

                  <!-- lightweight, SSR-friendly "modal" preview using details/summary (no extra JS state) -->
                  <details class="nes-container is-rounded" style="display:inline-block; margin-left:0.5rem;">
                    <summary class="nes-btn">Preview</summary>
                    <div style="padding:0.5rem; max-width:320px;">
                      {#if evidence.fileUrl && evidence.evidenceType === 'image'}
                        <img src={`/api/upload?file=${encodeURIComponent(evidence.fileUrl)}`} alt="Full preview" style="max-width:100%; height:auto; border-radius:4px;" />
                      {:else}
                        <div style="white-space:pre-wrap; font-size:.9rem;">{evidence.description || "No preview available."}</div>
                      {/if}
                      <div style="margin-top:0.5rem;">
                        <button class="nes-btn is-primary" on:on:onclick={() => canvasEditor?.addEvidenceToCanvas(evidence)}>Add to Canvas</button>
                        <a class="nes-btn is-warning" href={`/api/upload?file=${encodeURIComponent(evidence.fileUrl)}`} target="_blank" rel="noopener">Open</a>
                      </div>
                    </div>
                  </details>
                </div>
              </div>
            {/each}
          </div>
        {:else if activeTab === "reports"}
          <div class="reports-header nes-container is-rounded">
            <h3>AI Generated Reports</h3>
            {#if loadingStates.reports}
              <div class="loading-indicator"><span class="spinner"></span> Loading reports...</div>
            {/if}
          </div>

          <div>
            {#each aiReports as report}
              <div class="report-item nes-container is-rounded">
                <div class="report-title">{report.title}</div>
                <div class="report-meta">{report.reportType} • {report.generatedAt ? new Date(report.generatedAt).toLocaleString() : ""}</div>

                <div style="display:flex; gap:0.5rem; margin-top:0.5rem;">
                  <button class="btn-small nes-btn is-primary" on:on:onclick={() => reportEditor?.setContent(report.richTextContent)}>Load into Editor</button>

                  <!-- SRR-friendly inline preview using details -->
                  <details class="nes-container is-rounded" style="display:inline-block;">
                    <summary class="nes-btn is-success">Preview</summary>
                    <div style="padding:0.75rem; max-height:320px; overflow:auto;">
                      {#if report.richTextContent}
                        <div>{@html report.richTextContent}</div>
                      {:else}
                        <div style="white-space:pre-wrap;">{report.content || "No content available."}</div>
                      {/if}
                      <div style="margin-top:0.5rem;">
                        <a class="nes-btn is-warning" href={`/api/reports/download?id=${encodeURIComponent(report.id)}`} target="_blank" rel="noopener">Download</a>
                      </div>
                    </div>
                  </details>
                </div>
              </div>
            {/each}
          </div>

          {#if aiAnalysisComplete}
            <div class="ai-summary nes-container is-rounded" style="margin-top:1rem;">
              <h4>Case Summary</h4>
              <p>{caseSummary}</p>
            </div>
          {/if}
        {:else if activeTab === "images"}
          <div class="image-generator-section">
            <h3>🎨 AI Image Generation</h3>
            <p class="nes-text">Generate images for evidence recreation, crime scene visualization, and legal documentation.</p>
            <LocalImageGenerator 
              {caseId} 
              onImageGenerated={handleImageGenerated}
              compact={true}
            />
          </div>
        {/if}
      </div>
    </aside>

    <main class="main-content">
      <noscript class="nes-container is-rounded" style="margin:0.5rem;">
        <strong>Note:</strong> JavaScript is recommended for the full interactive experience.
      </noscript>

      <div class="content-tabs nes-container">
        <div>
          <button class="tab-btn nes-btn" class:active={activeTab === "canvas"} on:on:onclick={() => activeTab = "canvas"} title="Interactive Canvas (Ctrl/Cmd + 1)">🎨 Canvas</button>
          <button class="tab-btn nes-btn" class:active={activeTab === "editor"} on:on:onclick={() => activeTab = "editor"} title="Report Editor (Ctrl/Cmd + 2)">📝 Editor</button>
        </div>
        <div class="tab-shortcuts">
          <span class="shortcuts-hint">Shortcuts: Ctrl/Cmd + B (toggle sidebar), 1-4 (switch tabs)</span>
        </div>
      </div>

      <div class="content-area">
        {#if activeTab === "canvas"}
          <div class="canvas-container nes-container is-dark with-title">
            <p class="title">Interactive Canvas</p>
            {#if loadingStates.canvasSave}
              <div class="canvas-saving-indicator"><span class="spinner"></span> Saving canvas...</div>
            {/if}
            <EnhancedCanvasEditor bind:this={canvasEditor} {caseId} canvaschange={handleCanvasChange} width={1200} height={600} />
          </div>
        {:else if activeTab === "editor"}
          <div class="editor-container nes-container is-rounded">
            {#if loadingStates.reportSave}
              <div class="editor-saving-indicator"><span class="spinner"></span> Saving report...</div>
            {/if}
            <AdvancedRichTextEditor bind:this={reportEditor} contentchange={handleReportChange} placeholder="Write your case report, notes, or analysis here..." />
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
    overflow: hidden;
  }
  .notification-bar {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--background-alt, #f8f9fa);
    border-bottom: 1px solid var(--border, #dee2e6);
  }
  .notification {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-size: 0.875rem;
    animation: slideIn 0.3s ease-out;
  }
  .notification.error {
    background: var(--error-light, #f8d7da);
    color: var(--error-dark, #721c24);
    border: 1px solid var(--error, #dc3545);
  }
  .notification.success {
    background: var(--success-light, #d4edda);
    color: var(--success-dark, #155724);
    border: 1px solid var(--success, #28a745);
  }
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  .case-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: var(--surface, #fff);
    border-bottom: 1px solid var(--border, #dee2e6);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  .case-title {
    margin: 0;
    font-size: 1.5rem;
    color: var(--text-primary, #333);
  }
  .case-meta {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
  }
  .case-number {
    font-weight: 600;
    color: var(--primary, #007bff);
  }
  .case-status {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    background: var(--warning, #ffc107);
    color: white;
  }
  .case-status.status-open {
    background: var(--success, #28a745);
  }
  .evidence-count {
    color: var(--text-secondary, #666);
    font-size: 0.875rem;
  }
  .ai-controls {
    display: flex;
    gap: 1rem;
  }
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
    transition: all 0.3s ease;
  }
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
    transition: all 0.3s ease;
  }
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border, #dee2e6);
    border-left: 2px solid var(--primary, #007bff);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  .workspace-layout {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  .sidebar {
    width: 350px;
    background: var(--background-alt, #f8f9fa);
    border-right: 1px solid var(--border, #dee2e6);
    display: flex;
    flex-direction: column;
    transition: transform 0.3s ease;
  }
  .sidebar:not(.open) {
    transform: translateX(-100%);
  }
  .sidebar-tabs {
    display: flex;
    background: var(--surface, #fff);
    border-bottom: 1px solid var(--border, #dee2e6);
  }
  .tab-btn {
    flex: 1;
    padding: 0.75rem 1rem;
    border: none;
    background: transparent;
    cursor: pointer;
    transition: background-color 0.3s ease;
  }
  .tab-btn.active {
    background: var(--primary, #007bff);
    color: white;
  }
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
    transition: all 0.3s ease;
  }
  .evidence-thumbnail {
    width: 40px;
    height: 40px;
    border-radius: 4px;
    overflow: hidden;
    background: var(--background-alt, #f8f9fa);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .evidence-thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .file-icon {
    font-size: 1.5rem;
  }
  .evidence-info {
    flex: 1;
  }
  .evidence-title {
    font-weight: 500;
    font-size: 0.875rem;
    color: var(--text-primary, #333);
  }
  .evidence-meta {
    font-size: 0.75rem;
    color: var(--text-secondary, #666);
    margin-top: 0.25rem;
  }
  .ai-tags {
    display: flex;
    gap: 0.25rem;
    margin-top: 0.5rem;
  }
  .tag {
    padding: 0.125rem 0.375rem;
    background: var(--primary-light, #e7f3ff);
    color: var(--primary, #007bff);
    border-radius: 4px;
    font-size: 0.75rem;
  }
  .add-to-canvas-btn {
    padding: 0.25rem 0.5rem;
    background: var(--success, #28a745);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
  }
  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .content-tabs {
    display: flex;
    align-items: center;
    background: var(--surface, #fff);
    border-bottom: 1px solid var(--border, #dee2e6);
    padding: 0.5rem 1rem;
  }
  .tab-shortcuts {
    margin-left: auto;
    padding: 0.75rem 1rem;
  }
  .shortcuts-hint {
    font-size: 0.75rem;
    color: var(--text-secondary, #666);
    font-style: italic;
  }
  .content-area {
    flex: 1;
    overflow: hidden;
  }
  .canvas-container,
  .editor-container {
    height: 100%;
    padding: 1rem;
    position: relative;
  }
  .report-item {
    padding: 1rem;
    background: var(--surface, #fff);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border, #dee2e6);
  }
  .report-title {
    font-weight: 500;
    margin-bottom: 0.5rem;
  }
  .report-meta {
    font-size: 0.875rem;
    color: var(--text-secondary, #666);
    margin-bottom: 0.5rem;
  }
  .btn-small {
    padding: 0.25rem 0.75rem;
    background: var(--primary, #007bff);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
  }
  .ai-summary {
    padding: 1rem;
    background: var(--success-light, #d4edda);
    border-radius: 8px;
    margin-top: 1rem;
  }
  .ai-summary h4 {
    margin: 0 0 0.5rem 0;
    color: var(--success-dark, #155724);
  }
  .ai-summary p {
    margin: 0;
    color: var(--success-dark, #155724);
    font-size: 0.875rem;
  }
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
    z-index: 1000;
  }
  @media (max-width: 768px) {
    .case-header {
      flex-direction: column;
      gap: 1rem;
      align-items: flex-start;
    }
    .case-meta {
      flex-direction: column;
      gap: 0.5rem;
    }
    .ai-controls {
      flex-direction: column;
      gap: 0.5rem;
      width: 100%;
    }
    .btn-ai,
    .btn-toggle {
      width: 100%;
      justify-content: center;
    }
    .sidebar {
      position: fixed;
      top: 0;
      left: 0;
      height: 100vh;
      z-index: 1000;
      background: var(--surface, #fff);
      box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
    }
    .sidebar:not(.open) {
      transform: translateX(-100%);
    }
    .workspace-layout {
      position: relative;
    }
    .main-content {
      width: 100%;
    }
    .content-tabs {
      flex-direction: column;
      gap: 0.5rem;
    }
    .tab-shortcuts {
      display: none;
    }
    .shortcuts-hint {
      display: none;
    }
  }
</style>
