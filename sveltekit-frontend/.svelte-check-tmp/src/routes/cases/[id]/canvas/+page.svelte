<script lang="ts">
  import CanvasEditor from '$lib/components/CanvasEditor.svelte';
  import EvidencePanel from '$lib/components/EvidencePanel.svelte';

  // receive load() data from SvelteKit
  export let data: any;

  // try common places where the id/details might be supplied by the load function
  const caseId = data?.caseId ?? data?.params?.id;
  const caseDetails = data?.caseDetails;
  const evidence = caseDetails?.evidence ?? [];

  let canvasState: any = null;

  let showSidebar = false;
  let sidebarHovered = false;

  function handleSidebarMouseEnter() {
    sidebarHovered = true;
    showSidebar = true;
  }
  function handleSidebarMouseLeave() {
    sidebarHovered = false;
    setTimeout(() => {
      if (!sidebarHovered) showSidebar = false;
    }, 300);
  }
  function handleEvidenceDrop(evd: any) {
    // Forward to CanvasEditor (could push to a store or call a method)
    // For now, just log
    console.log("Dropped on canvas:", evd);
  }

  // use an any-typed props object to avoid strict $$ComponentProps errors when passing caseId
  const canvasProps: any = {
    caseId,
    evidence,
    onEvidenceDrop: handleEvidenceDrop,
    width: undefined,
    height: undefined,
    canvasState
  };

</script>

<section class="flex h-screen bg-gray-50">
  <!-- Sidebar Toggle -->
  <div class="relative">
    <div
      class="fixed left-0 top-1/2 transform -translate-y-1/2 z-50 sidebar-trigger"
      role="button"
      tabindex={0}
      onmouseenter={handleSidebarMouseEnter}
      onmouseleave={handleSidebarMouseLeave}
      onkeydown={(e: any) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleSidebarMouseEnter();
        }
      }}
    >
      <div class="sidebar-tab bg-blue-600 text-white rounded-r px-2 py-4 shadow hover:bg-blue-700 transition cursor-pointer">
        &#9776;
      </div>
    </div>

    {#if showSidebar}
      <div
        class="sidebar-panel bg-white shadow-lg w-64 h-full fixed left-0 top-0 z-30 flex flex-col"
        role="complementary"
        onmouseenter={handleSidebarMouseEnter}
        onmouseleave={handleSidebarMouseLeave}
        onkeydown={(e: any) => {
          if (e.key === "Escape") {
            handleSidebarMouseLeave();
          }
        }}
      >
        <!-- pass callback prop expected by EvidencePanel -->
        <EvidencePanel {caseId} onEvidenceDrop={handleEvidenceDrop} />
      </div>
    {/if}
  </div>

  <!-- Main Content Area -->
  <div class="flex-1 p-4">
      <CanvasEditor
        {...canvasProps}
      />

      <!-- AI FAB Button -->
      <button
        class="ai-fab fixed right-8 bottom-8 bg-blue-600 text-white rounded-full shadow-lg p-3 hover:bg-blue-700 transition z-40 cursor-pointer flex items-center justify-center"
        aria-label="Ask AI"
      >
        <svg width="32" height="32" fill="currentColor">
          <circle
            cx="16"
            cy="16"
            r="16"
            fill="currentColor"
            opacity=".1"
          />
          <path
            d="M16 8a8 8 0 1 1 0 16 8 8 0 0 1 0-16zm0 2a6 6 0 1 0 0 12A6 6 0 0 0 16 10zm1 3v2h2v2h-2v2h-2v-2h-2v-2h2v-2h2z"
            fill="currentColor"
          />
        </svg>
      </button>

        <div class="infinite-scroll-list bits-container bits-scroll flex-1 overflow-y-auto mt-4" role="list" aria-label="Evidence and canvas items">
          <!-- Bits-UI friendly infinite scroll area:
           - Use bits-container for consistent padding/width
           - bits-scroll enables better scroll UX and custom scrollbars
           - Implement load-more-on-scroll in component or store (not shown)
          -->
        </div>
          </div>
        </section>

        <style>
          /* UnoCSS helper (keep for scanner) */
          /* @unocss-include */

          /* Fallback NES.css (keep as fallback UI primitives) */
          @import "nes.css/css/nes.min.css";

          /* Bits-UI / container helper (primary source of truth for layout) */
          .bits-container {
        display: block;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        border-radius: 0.75rem;
        background: var(--bits-bg, #fff);
        box-shadow: 0 6px 24px rgba(7, 12, 20, 0.06);
          }

          /* Canvas layout: use bits-ui variable instead of pico.css */
          .canvas-stretch-container {
        position: relative;
        width: 100%;
        height: 80vh;
        min-height: 500px;
        display: flex;
        flex-direction: column;
        background: var(--bits-bg, #fff);
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
          }

          /* Sidebar trigger/tab */
          .sidebar-trigger {
        left: 0;
        top: 50%;
        transform: translateY(-50%);
          }
          .sidebar-tab {
        --uno: bg-primary text-white rounded-r px-2 py-4 shadow
          hover: bg-primary-600 transition;
          }
          .sidebar-panel {
        --uno: bg-background-alt shadow-lg w-64 h-full fixed left-0 top-0 z-30 flex
          flex-col;
        animation: slideInSidebar 0.3s;
          }

          @keyframes slideInSidebar {
        from {
          transform: translateX(-100%);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
          }

          /* Floating AI FAB */
          .ai-fab {
        position: absolute;
        right: 2rem;
        bottom: 2rem;
        --uno: bg-primary text-white rounded-full shadow-lg p-3
          hover: bg-primary-600 transition;
        border: none;
        z-index: 40;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
          }

          /* Bits-UI scrolling helpers for infinite lists */
          .bits-scroll {
        overscroll-behavior: contain;
        -webkit-overflow-scrolling: touch; /* smooth scrolling on iOS */
        scrollbar-width: thin; /* Firefox */
        scrollbar-color: rgba(100, 100, 100, 0.35) transparent;
          }
          /* WebKit custom scrollbar */
          .bits-scroll::-webkit-scrollbar {
        width: 10px;
        height: 10px;
          }
          .bits-scroll::-webkit-scrollbar-track {
        background: transparent;
          }
          .bits-scroll::-webkit-scrollbar-thumb {
        background-color: rgba(100, 100, 100, 0.25);
        border-radius: 999px;
        border: 2px solid transparent;
        background-clip: padding-box;
          }
          .bits-scroll:focus {
        outline: none;
          }

          /* Infinite scroll list area small tweaks */
          .infinite-scroll-list {
        flex: 1;
        overflow-y: auto;
        margin-top: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
          }

          /* Dialog (native <dialog> with backdrop styles + NES.css friendly tweaks) */
          dialog {
        border: none;
        border-radius: 0.6rem;
        padding: 1.25rem;
        background: var(--dialog-bg, #fff);
        box-shadow: 0 12px 40px rgba(2, 6, 23, 0.3);
        width: min(90%, 720px);
        max-width: 95%;
          }
          dialog::backdrop {
        background: rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(2px);
          }
          .dialog-content {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
          }

          /* NES.css integration examples (small overrides to match app theme) */
          .nes-container.is-rounded {
        border-radius: 0.75rem;
          }
          .nes-btn.is-primary {
        background: var(--primary, #2563eb);
        color: white;
          }

          /* Accessibility: focus outlines for keyboard users */
          .sidebar-tab:focus,
          .ai-fab:focus,
          .nes-btn:focus,
          dialog:focus {
        outline: 3px solid rgba(37, 99, 235, 0.25);
        outline-offset: 2px;
          }
        </style>
