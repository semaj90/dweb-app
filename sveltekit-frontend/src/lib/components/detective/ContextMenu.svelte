
<script lang="ts">
  import DropdownMenuContent from "$lib/components/ui/dropdown-menu/DropdownMenuContent.svelte";
  import DropdownMenuItem from "$lib/components/ui/dropdown-menu/DropdownMenuItem.svelte";
  import DropdownMenuRoot from "$lib/components/ui/dropdown-menu/DropdownMenuRoot.svelte";
  import DropdownMenuSeparator from "$lib/components/ui/dropdown-menu/DropdownMenuSeparator.svelte";
  import type { Case, Evidence } from "$lib/types/index";
  import { createEventDispatcher, onMount, tick } from "svelte";



  // --- Phase 10: Context7 Evidence Actions ---
  // Trigger semantic audit, agent review, or vector search for this evidence
  async function auditEvidence() {
    if (!item) return;
    try {
      const res = await fetch('/api/audit/semantic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: `Audit evidence ${item.id}` })
      });
      if (!res.ok) throw new Error('Failed to audit evidence');
      const data = await res.json();
      dispatch('auditResults', { results: data.results, evidence: item });
    } catch (error) {
      dispatch('auditError', { error: (error as Error).message, evidence: item });
    }
    closeMenu();
  }

  async function triggerAgentReview() {
    if (!item) return;
    try {
      const res = await fetch('/api/agent/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ evidenceId: item.id })
      });
      if (!res.ok) throw new Error('Failed to trigger agent review');
      const data = await res.json();
      dispatch('agentReviewResult', { result: data, evidence: item });
    } catch (error) {
      dispatch('agentReviewError', { error: (error as Error).message, evidence: item });
    }
    closeMenu();
  }

  export let x: number;
  export let y: number;
  export let item: Evidence | null;

  const dispatch = createEventDispatcher();
  let cases: Case[] = [];
  let menuOpen = true;

  onMount(async () => {
    // Load available cases
    try {
      const response = await fetch("/api/cases");
      if (response.ok) {
        cases = await response.json();
}
    } catch (error) {
      console.error("Failed to load cases:", error);
}
    // Open menu after mount
    await tick();
  });

  function sendToCase(caseId: string) {
    dispatch("sendToCase", { caseId });
    closeMenu();
}
  function viewEvidence() {
    if (item) window.open(`/evidence/${item.id}`, "_blank");
    closeMenu();
}
  function editEvidence() {
    if (item) window.location.href = `/evidence/${item.id}/edit`;
    closeMenu();
}
  function downloadEvidence() {
    if (item && item.fileUrl) {
      const link = document.createElement("a");
      link.href = item.fileUrl;
      link.download = item.fileName || "evidence";
      link.click();
}
    closeMenu();
}
  function duplicateEvidence() {
    // Implementation for duplicating evidence
    console.log("Duplicate evidence:", item?.id);
    closeMenu();
}
  function deleteEvidence() {
    if (item && confirm("Are you sure you want to delete this evidence?")) {
      // Implementation for deleting evidence
      console.log("Delete evidence:", item.id);
}
    closeMenu();
}
  function closeMenu() {
    menuOpen = false;
    dispatch("close");
}
</script>
<DropdownMenuRoot
  let:trigger
  let:states
  on:openChange={(e) => {
    if (!e.detail.open) closeMenu();
  }}
  <!-- Hidden trigger for programmatic open -->
  <button
    style="position:fixed;left:-9999px;top:-9999px;"
    aria-label="Open context menu"
    tabindex={-1}
  ></button>
  {#if menuOpen}
    <DropdownMenuContent
      class="container mx-auto px-4"
      style="position:fixed;left:{x}px;top:{y}px;"
      on:keydown={(e) => {
        if (e.detail && e.detail.key === "Escape") closeMenu();
      }}
      aria-label="Evidence context menu"
    >
      <div class="container mx-auto px-4">
        <p class="container mx-auto px-4">
          Evidence Actions
        </p>
      </div>
      <DropdownMenuItem on:select={viewEvidence}>
        <i class="container mx-auto px-4"></i>
        <span class="container mx-auto px-4">View Details</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={editEvidence}>
        <i class="container mx-auto px-4"></i>
        <span class="container mx-auto px-4">Edit</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={downloadEvidence}>
        <i class="container mx-auto px-4"></i>
        <span class="container mx-auto px-4">Download</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={duplicateEvidence}>
        <i class="container mx-auto px-4"></i>
        <span class="container mx-auto px-4">Duplicate</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={auditEvidence}>
        <i class="container mx-auto px-4"></i>
        <span class="container mx-auto px-4">Audit (Semantic/Vector)</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={triggerAgentReview}>
        <i class="container mx-auto px-4"></i>
        <span class="container mx-auto px-4">Trigger Agent Review</span>
      </DropdownMenuItem>
      {#if cases.length > 0}
        <DropdownMenuSeparator />
        <div class="container mx-auto px-4">
          <p class="container mx-auto px-4">
            Send to Case
          </p>
        </div>
        {#each cases as case_}
          <DropdownMenuItem on:select={() => sendToCase(case_.id)}>
            <i class="container mx-auto px-4"></i>
            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4">{case_.title}</div>
              <div class="container mx-auto px-4">
                {case_.caseNumber}
              </div>
            </div>
          </DropdownMenuItem>
        {/each}
      {/if}
      <DropdownMenuSeparator />
      <div class="container mx-auto px-4">
        <p class="container mx-auto px-4">
          Danger Zone
        </p>
      </div>
      <DropdownMenuItem
        on:select={deleteEvidence}
        class="container mx-auto px-4"
      >
        <i class="container mx-auto px-4"></i>
        <span class="container mx-auto px-4">Delete</span>
      </DropdownMenuItem>
    </DropdownMenuContent>
  {/if}
</DropdownMenuRoot>

<style>
  /* @unocss-include */
  @keyframes contextMenuFadeIn {
    from {
      opacity: 0;
      transform: scale(0.95);
}
    to {
      opacity: 1;
      transform: scale(1);
}}
</style>
