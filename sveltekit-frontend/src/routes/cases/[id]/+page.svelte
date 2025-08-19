<script lang="ts">
  import { page } from "$app/state";
  import CanvasEditor from '$lib/components/CanvasEditor.svelte';
  import EvidencePanel from '$lib/components/EvidencePanel.svelte';

  let caseId = page.params.id;
  let caseDetails = page.data.caseDetails;
  let evidence = caseDetails?.evidence || [];
  let canvasState: any = $state(null);

  let showSidebar = $state(false);
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
      onkeydown={(e) => {
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
        onkeydown={(e) => {
          if (e.key === "Escape") {
            handleSidebarMouseLeave();
          }
        }}
      >
        <EvidencePanel {caseId} onEvidenceDrop={handleEvidenceDrop} />
      </div>
    {/if}
  </div>

  <!-- Main Content Area -->
  <div class="flex-1 p-4">
    <div class="canvas-stretch-container relative w-full h-full min-h-500 flex flex-col bg-white rounded-lg shadow">
      <CanvasEditor
        bind:canvasState
        reportId={caseId}
        {evidence}
        onevidencedrop={handleEvidenceDrop}
        width={undefined}
        height={undefined}
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
      
      <div class="infinite-scroll-list flex-1 overflow-y-auto mt-4">
        <!-- Infinite scroll logic: load more on scroll bottom, show evidence or canvas items -->
      </div>
    </div>
  </div>
</section>

<style>
  /* @unocss-include */
  .case-layout {
    margin-top: 2rem;
    margin-bottom: 2rem;
}
  .canvas-stretch-container {
    position: relative;
    width: 100%;
    height: 80vh;
    min-height: 500px;
    display: flex;
    flex-direction: column;
    background: var(--pico-background, #fff);
    border-radius: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
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
}}
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
  .infinite-scroll-list {
    flex: 1;
    overflow-y: auto;
    margin-top: 1rem;
}
</style>
