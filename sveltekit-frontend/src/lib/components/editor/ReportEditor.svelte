<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<script lang="ts">
  import { $derived } from 'svelte';

  import { onDestroy, onMount } from "svelte";
  import { quintOut } from "svelte/easing";
  import { fly } from "svelte/transition";
  import {
    editorState,
    report,
    reportActions,
    reportUI,
    setupAutoSave,
  } from '$lib/stores/report';
  // Components
  import EvidenceCard from "../evidence/EvidenceCard.svelte";
  import EvidenceForm from "../forms/EvidenceForm.svelte";
  import MasonryGrid from "../layout/MasonryGrid.svelte";
  import AdvancedSearch from "../search/AdvancedSearch.svelte";
  import Modal from "../ui/Modal.svelte";
  import ReportToolbar from "./ReportToolbar.svelte";
  import RichTextEditor from "./RichTextEditor.svelte";
  // Icons
  import { invalidateAll } from "$app/navigation";
  import {
    Columns,
    Grid,
    Layout,
    Maximize2,
    Minimize2,
    PanelLeftOpen,
    Plus,
    Settings,
  } from "lucide-svelte";

  // State
let editorComponent = $state<RichTextEditor;
  let cleanupAutoSave: (() >(> void) | undefined);
let evidenceSearchResults = $state<any[] >([]);
let selectedEvidence = $state<any >(null);
let showEvidenceModal = $state(false);
let showSettingsModal = $state(false);
let evidenceFormData = $state<any >(null); // For the evidence form

  // Reactive layout classes
  let layoutClass = $derived(
    ($report && $report.settings)
    ? {
        single: "layout-single",
        dual: "layout-dual",
        masonry: "layout-masonry",
      }[$report.settings.layout]
    : "layout-single"
  );

  // Reactive editor height
  let editorHeight = $derived($reportUI && $reportUI.fullscreen ? window.innerHeight - 200 : 500);

  function updateEditorHeight() {
    editorHeight = $reportUI.fullscreen ? window.innerHeight - 200 : 500;
  }

  onMount(() => {
    window.addEventListener('resize', updateEditorHeight);
  });

  onDestroy(() => {
    window.removeEventListener('resize', updateEditorHeight);
  });

  // Initialize auto-save
  onMount(() => {
    if ($report.settings.autoSave) {
      cleanupAutoSave = setupAutoSave();
    }
  });

  onDestroy(() => {
    if (cleanupAutoSave) {
      cleanupAutoSave();
    }
  });

  // Handle evidence actions
  const handleViewEvidence = (evidence: any) => {
    selectedEvidence = evidence;
    showEvidenceModal = true;
  };

  const handleEditEvidence = (evidence: any) => {
    selectedEvidence = evidence;
    showEvidenceModal = true;
  };

  const handleDeleteEvidence = async (evidence: any) => {
    if (confirm(`Are you sure you want to delete "${evidence.title}"?`)) {
      try {
        const formData = new FormData();
        formData.append("id", evidence.id);

        const response = await fetch("/api/evidence/delete", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          reportActions.removeEvidence(evidence.id);
          await invalidateAll(); // Refresh the page data
        } else {
          alert("Failed to delete evidence");
        }
      } catch (error) {
        console.error("Error deleting evidence:", error);
        alert("Error deleting evidence");
      }
    }
  };

  const handleDownloadEvidence = (evidence: any) => {
    if (evidence.url) {
      window.open(evidence.url, "_blank");
    }
  };
  const handleShareEvidence = (evidence: any) => {
    // Implementation for sharing evidence
    console.log('Sharing evidence:', evidence);
  };
  const handleInsertEvidence = (evidence: any) => {
    if (editorComponent) {
      editorComponent.insertEvidence(evidence);
    }
  };

  const handleAddNewEvidence = () => {
    selectedEvidence = null;
    showEvidenceModal = true;
  };

  // Layout switching
  const switchLayout = () => {
    const layouts = ["single", "dual", "masonry"] as const;
    const currentIndex = layouts.indexOf($report.settings.layout);
    const nextLayout = layouts[(currentIndex + 1) % layouts.length];
    reportActions.updateSettings({ layout: nextLayout });
  };

  // Sidebar toggle
  const toggleSidebar = () => {
    reportUI.update((ui) => ({ ...ui, sidebarOpen: !ui.sidebarOpen }));
  };

  // Fullscreen toggle
  const toggleFullscreen = () => {
    reportUI.update((ui) => ({ ...ui, fullscreen: !ui.fullscreen }));
    if (!$reportUI.fullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  };

  // Keyboard shortcuts
  const handleKeydown = (e: CustomEvent<any>) => {
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case "s":
          e.preventDefault();
          reportActions.save();
          break;
        case "b":
          e.preventDefault();
          toggleSidebar();
          break;
        case "n":
          e.preventDefault();
          reportActions.reset();
          break;
      }
    }
    if (e.key === "F11") {
      e.preventDefault();
      toggleFullscreen();
    }
  };
</script>

<svelte:window keydown={handleKeydown} />

<div
  class="report-editor {layoutClass}"
  class:fullscreen={$reportUI.fullscreen}
  class:sidebar-closed={!$reportUI.sidebarOpen}
>
  <!-- Toolbar -->
  <header class="editor-toolbar">
    <ReportToolbar />
  </header>

  <!-- Main Content Area -->
  <div class="editor-content">
    <!-- Sidebar -->
    {#if $reportUI.sidebarOpen}
      <aside
        class="editor-sidebar"
        style="width: {$reportUI.sidebarWidth}px"
        transitifly={{ x: -300, duration: 300, easing: quintOut }}
      >
        <!-- Evidence Search -->
        <section class="sidebar-section">
          <div class="section-header">
        <section class="space-y-4">
          <div>
            <h3>Evidence Library</h3>
            <button
              on:onclick={() => handleAddNewEvidence()}
              title="Add new evidence"
            >
              <Plus size={16} />
            </button>
          </div>
            items={$report.attachedEvidence}
            results={(results) => (evidenceSearchResults = results)}
            select={handleInsertEvidence}
            placeholder="Search evidence..."
          />
        </section>

        <!-- Evidence Grid -->
        <section class="evidence-section">
          {#if $report.settings.layout === "masonry"}
        <section class="space-y-4">
          {#if $report.settings.layout === "masonry"}
            <MasonryGrid
              items={evidenceSearchResults}
              columnWidth={250}
              gutter={12}
              let:item
            >
              <EvidenceCard
                evidence={item}
                view={handleViewEvidence}
                edit={handleEditEvidence}
                delete={handleDeleteEvidence}
                download={handleDownloadEvidence}
                compact={true}
              />
            </MasonryGrid>
          {:else}
            <div>
              {#each evidenceSearchResults as evidence (evidence.id)}
                <EvidenceCard
                  {evidence}
                  view={handleViewEvidence}
                  edit={handleEditEvidence}
                  delete={handleDeleteEvidence}
                  download={handleDownloadEvidence}
                  compact={true}
                />
              {/each}
            </div>
          {/if}

          {#if evidenceSearchResults.length === 0}
            <div>
              <p>No evidence found</p>
              <small>Add evidence to enhance your report</small>
            </div>
          {/if}
        </section>
        <section class="stats-section sidebar-section">
          <div class="stats-grid">
        <section class="space-y-4">
          <div>
            <div>
              <span>Words</span>
              <span>{$editorState.wordCount}</span>
            </div>
            <div>
              <span>Evidence</span>
              <span>{$report.attachedEvidence.length}</span>
            </div>
            <div>
              <span>Status</span>
              <span>
                {$report.metadata.status}
              </span>
            </div>
            <div>
              <span>Modified</span>
              <span>
                {$report.metadata.updatedAt.toLocaleDateString()}
              </span>
            </div>
          </div>
        </section>

    <!-- Main Editor Area -->
    <main class="editor-main">
      <!-- Editor Header -->
    <main class="space-y-4">
      <!-- Editor Header -->
      <div>
        <div>
          {#if !$reportUI.sidebarOpen}
            <button
              on:onclick={() => toggleSidebar()}
              title="Show sidebar"
            >
              <PanelLeftOpen size={20} />
            </button>
          {/if}

          <input
            type="text"
            value={$report.title}
            input={(e) => reportActions.updateTitle(e.currentTarget.value)}
            placeholder="Report title..."
          />
        </div>

        <div>
          <button
            on:onclick={() => switchLayout()}
            title="Switch layout ({$report.settings.layout})"
          >
            {#if $report.settings.layout === "single"}
              <Layout size={18} />
            {:else if $report.settings.layout === "dual"}
              <Columns size={18} />
            {:else}
              <Grid size={18} />
            {/if}
          </button>

          <button
            on:onclick={() => toggleFullscreen()}
            title="Toggle fullscreen"
          >
            {#if $reportUI.fullscreen}
              <Minimize2 size={18} />
            {:else}
              <Maximize2 size={18} />
            {/if}
          </button>

          <button
            on:onclick={() => (showSettingsModal = true)}
            title="Settings"
          >
            <Settings size={18} />
          </button>
        </div>
      </div>

      <RichTextEditor
        bind:this={editorComponent}
        height={editorHeight}
      />
    </main>

    <!-- Evidence Panel (for dual layout) -->
    {#if $report.settings.layout === "dual"}
      <aside
        class="evidence-panel"
        transitifly={{ x: 300, duration: 300, easing: quintOut }}
      >
        <div class="panel-header"></div>
          <h3>Evidence</h3>
          <button class="add-evidence-btn" on:onclick={() => handleAddNewEvidence()}>
            <Plus size={16} />
          </button>
        </div>

        <div class="evidence-grid-panel"></div>
          <MasonryGrid
            items={$report.attachedEvidence}
            columnWidth={200}
            gutter={8}
            let:item
          >
            <EvidenceCard
              evidence={item}
              view={handleViewEvidence}
              edit={handleEditEvidence}
              delete={handleDeleteEvidence}
              download={handleDownloadEvidence}
              compact={true}
            />
  {#if showEvidenceModal}
    <EvidenceForm
      data={evidenceFormData}
      evidence={selectedEvidence}
      success={() => {
        showEvidenceModal = false;
        selectedEvidence = null;
      }}
      error={(e) => {
        console.error("Evidence form error:", e.detail);
        alert("Error saving evidence");
      }}
      cancel={() => {
        showEvidenceModal = false;
        selectedEvidence = null;
      }}
    />
  {/if}
        showEvidenceModal = false;
        selectedEvidence = null;
      }}
      error={(e) => {
        console.error("Evidence form error:", e.detail);
        alert("Error saving evidence");
      }}
      cancel={() => {
        showEvidenceModal = false;
        selectedEvidence = null;
      }}
    />
  {/if}
</Modal>

<!-- Settings Modal -->
<Modal bind:open={showSettingsModal}>
  <div slot="title">Report Settings</div>
  <!-- Settings form would go here -->
  <div class="settings-form">
    <!-- TODO: Implement settings form for report options such as auto-save, layout selection, evidence preferences, and other report configurations -->
    <p>Settings panel - TODO: Implement settings form</p>
  </div>
</Modal>

<style>
  .report-editor {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #ffffff;
    transition: all 0.3s ease;
  }
  .report-editor.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
  }
  .editor-toolbar {
    flex-shrink: 0;
    border-bottom: 1px solid #e2e8f0;
  }
  .editor-content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  .editor-sidebar {
    flex-shrink: 0;
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-section {
    padding: 1rem;
    border-bottom: 1px solid #e2e8f0;
  }
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
  }
  .section-header h3 {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 600;
    color: #374151;
  }
  .add-evidence-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1.75rem;
    height: 1.75rem;
    border: none;
    background: #3b82f6;
    color: white;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: background-color 0.15s ease;
  }
  .add-evidence-btn:hover {
    background: #2563eb;
  }
  .evidence-section {
    flex: 1;
    overflow-y: auto;
  }
  .evidence-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .empty-evidence {
    text-align: center;
    padding: 2rem 1rem;
    color: #6b7280;
  }
  .empty-evidence p {
    margin: 0 0 0.25rem;
    font-weight: 500;
  }
  .empty-evidence small {
    font-size: 0.75rem;
    opacity: 0.8;
  }
  .stats-section {
    flex-shrink: 0;
    background: #ffffff;
  }
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }
  .stat-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .stat-label {
    font-size: 0.75rem;
    color: #6b7280;
    font-weight: 500;
  }
  .stat-value {
    font-size: 0.875rem;
    font-weight: 600;
    color: #111827;
  }
  .stat-value.status-draft {
    color: #3b82f6;
  }
  .stat-value.status-review {
    color: #f59e0b;
  }
  .stat-value.status-final {
    color: #10b981;
  }
  .editor-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .editor-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem;
    border-bottom: 1px solid #e2e8f0;
    background: #ffffff;
  }
  .editor-title-section {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex: 1;
  }
  .sidebar-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border: none;
    background: none;
    color: #6b7280;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .sidebar-toggle:hover {
    background: #f3f4f6;
    color: #3b82f6;
  }
  .report-title-input {
    flex: 1;
    max-width: 30rem;
    padding: 0.5rem 0.75rem;
    border: 1px solid transparent;
    background: none;
    font-size: 1.25rem;
    font-weight: 600;
    color: #111827;
    border-radius: 0.375rem;
    transition: border-color 0.15s ease;
  }
  .report-title-input:focus {
    outline: none;
    border-color: #3b82f6;
    background: #ffffff;
  }
  .editor-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .layout-toggle,
  .fullscreen-toggle,
  .settings-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    height: 2.25rem;
    border: none;
    background: none;
    color: #6b7280;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .layout-toggle:hover,
  .fullscreen-toggle:hover,
  .settings-btn:hover {
    background: #f3f4f6;
    color: #3b82f6;
  }
  .editor-wrapper {
    flex: 1;
    overflow: hidden;
    padding: 1rem;
  }
  .evidence-panel {
    width: 20rem;
    background: #f8fafc;
    border-left: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem;
    border-bottom: 1px solid #e2e8f0;
  }
  .panel-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: #374151;
  }
  .evidence-grid-panel {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
  }
  /* Layout variations */
  .layout-single .evidence-panel {
    display: none;
  }
  .layout-dual .editor-sidebar {
    width: 16rem !important;
  }
  .layout-masonry .evidence-section {
    padding: 0.5rem;
  }
  /* Modal content */
  .settings-form {
    padding: 1rem;
    text-align: center;
    color: #6b7280;
  }
  /* Responsive design */
  @media (max-width: 1024px) {
    .editor-sidebar {
      width: 16rem !important;
    }
    .evidence-panel {
      width: 16rem;
    }
  }
  @media (max-width: 768px) {
    .layout-dual .evidence-panel {
      display: none;
    }
    .editor-sidebar {
      width: 14rem !important;
    }
  }
</style>


