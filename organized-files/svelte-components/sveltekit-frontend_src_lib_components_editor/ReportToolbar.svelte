<script lang="ts">
  
  import {
    Download,
    Eye,
    FileText,
    Layout,
    Maximize,
    Minimize,
    Redo,
    Replace,
    Save,
    Search,
    Sidebar,
    Undo,
    Upload,
  } from "lucide-svelte";
  import { fly, slide } from "svelte/transition";
  import {
    editorState,
    report,
    reportActions,
    reportUI,
  } from '$lib/stores/report';

  // File menu dropdown
  const {
    elements: { trigger: fileTrigger, menu: fileMenu, item: fileItem },
    states: { open: fileOpen },
  } = createDropdownMenu({
    positioning: { placement: "bottom-start" },
  });

  // Edit menu dropdown
  const {
    elements: { trigger: editTrigger, menu: editMenu, item: editItem },
    states: { open: editOpen },
  } = createDropdownMenu({
    positioning: { placement: "bottom-start" },
  });

  // View menu dropdown
  const {
    elements: { trigger: viewTrigger, menu: viewMenu, item: viewItem },
    states: { open: viewOpen },
  } = createDropdownMenu({
    positioning: { placement: "bottom-start" },
  });

  // Toolbar
  const {
    elements: { root: toolbarRoot, button: toolbarButton, link: toolbarLink },
  } = createToolbar();

  // Actions
  const handleSave = () => {
    reportActions.save();
  };

  const handleExport = () => {
    // TODO: Implement export functionality
    console.log("Export report");
  };

  const handlePreview = () => {
    // TODO: Implement preview functionality
    console.log("Preview report");
  };

  const toggleSidebar = () => {
    reportUI.update((ui) => ({ ...ui, sidebarOpen: !ui.sidebarOpen }));
  };

  const toggleFullscreen = () => {
    reportUI.update((ui) => ({ ...ui, fullscreen: !ui.fullscreen }));
  };

  const toggleLayout = () => {
    const layouts = ["single", "dual", "masonry"] as const;
    const currentIndex = layouts.indexOf($report.settings.layout);
    const nextLayout = layouts[(currentIndex + 1) % layouts.length];
    reportActions.updateSettings({ layout: nextLayout });
  };
</script>

<div class="space-y-4" use:melt={$toolbarRoot}>
  <!-- Main Menu Bar -->
  <div class="space-y-4">
    <!-- File Menu -->
    <div class="space-y-4">
      <button
        use:melt={$fileTrigger}
        class="space-y-4"
        class:active={$fileOpen}
      >
        File
      </button>

      {#if $fileOpen}
        <div
          use:melt={$fileMenu}
          class="space-y-4"
          transitionfly={{ y: -5, duration: 150 }}
        >
          <button
            use:melt={$fileItem}
            class="space-y-4"
            onclick={() => handleSave()}
          >
            <Save size={16} />
            Save Report
            <span class="space-y-4">Ctrl+S</span>
          </button>
          <button use:melt={$fileItem} class="space-y-4">
            <FileText size={16} />
            New Report
            <span class="space-y-4">Ctrl+N</span>
          </button>
          <div class="space-y-4"></div>
          <button use:melt={$fileItem} class="space-y-4">
            <Upload size={16} />
            Import
          </button>
          <button
            use:melt={$fileItem}
            class="space-y-4"
            onclick={() => handleExport()}
          >
            <Download size={16} />
            Export
          </button>
          <div class="space-y-4"></div>
          <button
            use:melt={$fileItem}
            class="space-y-4"
            onclick={() => handlePreview()}
          >
            <Eye size={16} />
            Preview
          </button>
        </div>
      {/if}
    </div>

    <!-- Edit Menu -->
    <div class="space-y-4">
      <button
        use:melt={$editTrigger}
        class="space-y-4"
        class:active={$editOpen}
      >
        Edit
      </button>

      {#if $editOpen}
        <div
          use:melt={$editMenu}
          class="space-y-4"
          transitionfly={{ y: -5, duration: 150 "
        >
          <button use:melt={$editItem} class="space-y-4">
            <Undo size={16} />
            Undo
            <span class="space-y-4">Ctrl+Z</span>
          </button>
          <button use:melt={$editItem} class="space-y-4">
            <Redo size={16} />
            Redo
            <span class="space-y-4">Ctrl+Y</span>
          </button>
          <div class="space-y-4"></div>
          <button use:melt={$editItem} class="space-y-4">
            <Search size={16} />
            Find
            <span class="space-y-4">Ctrl+F</span>
          </button>
          <button use:melt={$editItem} class="space-y-4">
            <Replace size={16} />
            Replace
            <span class="space-y-4">Ctrl+H</span>
          </button>
        </div>
      {/if}
    </div>

    <!-- View Menu -->
    <div class="space-y-4">
      <button
        use:melt={$viewTrigger}
        class="space-y-4"
        class:active={$viewOpen}
      >
        View
      </button>

      {#if $viewOpen}
        <div
          use:melt={$viewMenu}
          class="space-y-4"
          transitionfly={{ y: -5, duration: 150 "
        >
          <button
            use:melt={$viewItem}
            class="space-y-4"
            onclick={() => toggleSidebar()}
          >
            <Sidebar size={16} />
            Toggle Sidebar
            <span class="space-y-4">Ctrl+B</span>
          </button>
          <button
            use:melt={$viewItem}
            class="space-y-4"
            onclick={() => toggleLayout()}
          >
            <Layout size={16} />
            Switch Layout ({$report.settings.layout})
          </button>
          <button
            use:melt={$viewItem}
            class="space-y-4"
            onclick={() => toggleFullscreen()}
          >
            {#if $reportUI.fullscreen}
              <Minimize size={16} />
              Exit Fullscreen
            {:else}
              <Maximize size={16} />
              Fullscreen
            {/if}
            <span class="space-y-4">F11</span>
          </button>
        </div>
      {/if}
    </div>
  </div>

  <!-- Quick Actions -->
  <div class="space-y-4">
    <button
      use:melt={$toolbarButton}
      class="space-y-4"
      class:unsaved={$editorState.hasUnsavedChanges}
      onclick={() => handleSave()}
      title="Save Report"
    >
      <Save size={16} />
    </button>

    <div class="space-y-4"></div>

    <button
      use:melt={$toolbarButton}
      class="space-y-4"
      onclick={() => toggleSidebar()}
      title="Toggle Sidebar"
    >
      <Sidebar size={16} />
    </button>

    <button
      use:melt={$toolbarButton}
      class="space-y-4"
      onclick={() => toggleLayout()}
      title="Switch Layout"
    >
      <Layout size={16} />
    </button>

    <div class="space-y-4"></div>

    <button
      use:melt={$toolbarButton}
      class="space-y-4"
      onclick={() => handlePreview()}
      title="Preview Report"
    >
      <Eye size={16} />
    </button>
  </div>

  <!-- Status Info -->
  <div class="space-y-4">
    <span class="space-y-4">
      {$editorState.wordCount} words
    </span>

    {#if $editorState.hasUnsavedChanges}
      <span class="space-y-4" transitionslide={{ duration: 200 ">
        Unsaved changes
      </span>
    {:else}
      <span class="space-y-4" transitionslide={{ duration: 200 ">
        Saved {$editorState.lastSaved.toLocaleTimeString()}
      </span>
    {/if}
  </div>
</div>

<style>
  /* @unocss-include */
  .report-toolbar {
    display: flex
    align-items: center
    justify-content: space-between;
    background: var(--pico-background-color, #ffffff);
    border-bottom: 1px solid var(--pico-border-color, #e2e8f0);
    padding: 0.5rem 1rem;
    min-height: 3rem;
    position: sticky
    top: 0;
    z-index: 40;
}
  .menu-bar {
    display: flex
    align-items: center
    gap: 0.5rem;
}
  .menu-item {
    position: relative
}
  .menu-trigger {
    background: none
    border: none
    padding: 0.5rem 0.75rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    color: var(--pico-color, #374151);
    cursor: pointer
    transition: all 0.15s ease;
}
  .menu-trigger:hover,
  .menu-trigger.active {
    background: var(--pico-primary-background, #f3f4f6);
}
  .dropdown-menu {
    position: absolute
    top: 100%;
    left: 0;
    min-width: 12rem;
    background: var(--pico-card-background-color, #ffffff);
    border: 1px solid var(--pico-border-color, #e2e8f0);
    border-radius: 0.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    padding: 0.5rem;
    z-index: 50;
}
  .dropdown-item {
    display: flex
    align-items: center
    gap: 0.5rem;
    width: 100%;
    padding: 0.5rem 0.75rem;
    border: none
    background: none
    text-align: left
    border-radius: 0.25rem;
    font-size: 0.875rem;
    color: var(--pico-color, #374151);
    cursor: pointer
    transition: background-color 0.15s ease;
}
  .dropdown-item:hover {
    background: var(--pico-primary-background, #f3f4f6);
}
  .dropdown-separator {
    height: 1px;
    background: var(--pico-border-color, #e2e8f0);
    margin: 0.5rem 0;
}
  .shortcut {
    margin-left: auto
    font-size: 0.75rem;
    color: var(--pico-muted-color, #6b7280);
    opacity: 0.7;
}
  .quick-actions {
    display: flex
    align-items: center
    gap: 0.25rem;
}
  .action-button {
    display: flex
    align-items: center
    justify-content: center
    width: 2rem;
    height: 2rem;
    border: none
    background: none
    border-radius: 0.25rem;
    color: var(--pico-color, #6b7280);
    cursor: pointer
    transition: all 0.15s ease;
}
  .action-button:hover {
    background: var(--pico-primary-background, #f3f4f6);
    color: var(--pico-primary, #3b82f6);
}
  .action-button.unsaved {
    color: var(--pico-del-color, #ef4444);
}
  .separator {
    width: 1px;
    height: 1.5rem;
    background: var(--pico-border-color, #e2e8f0);
    margin: 0 0.5rem;
}
  .status-info {
    display: flex
    align-items: center
    gap: 1rem;
    font-size: 0.75rem;
    color: var(--pico-muted-color, #6b7280);
}
  .word-count {
    font-weight: 500;
}
  .unsaved-indicator {
    color: var(--pico-del-color, #ef4444);
    font-weight: 500;
}
  .saved-indicator {
    color: var(--pico-ins-color, #10b981);
}
</style>
