
<script lang="ts">
  interface Props {
    onauditResults?: (event?: any) => void;
    onauditError?: (event?: any) => void;
    onagentReviewResult?: (event?: any) => void;
    onagentReviewError?: (event?: any) => void;
    onsendToCase?: (event?: any) => void;
    onclose?: (event?: any) => void;
  }
  let {
    x,
    y,
    item
  } = $props();



  import DropdownMenuContent from "$lib/components/ui/dropdown-menu/DropdownMenuContent.svelte";
  import DropdownMenuItem from "$lib/components/ui/dropdown-menu/DropdownMenuItem.svelte";
  import DropdownMenuRoot from "$lib/components/ui/dropdown-menu/DropdownMenuRoot.svelte";
  import DropdownMenuSeparator from "$lib/components/ui/dropdown-menu/DropdownMenuSeparator.svelte";
  import type { Case, Evidence } from "$lib/types/index";
  


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
      onauditResults?.();
    } catch (error) {
      onauditError?.().message, evidence: item });
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
      onagentReviewResult?.();
    } catch (error) {
      onagentReviewError?.().message, evidence: item });
    }
    closeMenu();
  }

      
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
    onsendToCase?.();
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
    onclose?.();
}
</script>
<DropdownMenuRoot
  on:openChange={(e) => {
    if (!e.detail.open) closeMenu();
  }}
>
  <!-- Hidden trigger for programmatic open -->
  <button
    style="position: fixedleft:-9999px;top:-9999px;"
    aria-label="Open context menu"
    tabindex={-1}
  ></button>
  {#if menuOpen}
    <DropdownMenuContent
      menu={menuOpen}
      class="space-y-4"
      style="position: fixedleft:{x}px;top:{y}px;"
      onkeydown={(e) => {
        if (e.detail && e.detail.key === "Escape") closeMenu();
      }}
      aria-label="Evidence context menu"
    >
      <div class="space-y-4">
        <p class="space-y-4">
          Evidence Actions
        </p>
      </div>
      <DropdownMenuItem onselect={viewEvidence}>
        <i class="space-y-4"></i>
        <span class="space-y-4">View Details</span>
      </DropdownMenuItem>
      <DropdownMenuItem onselect={editEvidence}>
        <i class="space-y-4"></i>
        <span class="space-y-4">Edit</span>
      </DropdownMenuItem>
      <DropdownMenuItem onselect={downloadEvidence}>
        <i class="space-y-4"></i>
        <span class="space-y-4">Download</span>
      </DropdownMenuItem>
      <DropdownMenuItem onselect={duplicateEvidence}>
        <i class="space-y-4"></i>
        <span class="space-y-4">Duplicate</span>
      </DropdownMenuItem>
      <DropdownMenuItem onselect={auditEvidence}>
        <i class="space-y-4"></i>
        <span class="space-y-4">Audit (Semantic/Vector)</span>
      </DropdownMenuItem>
      <DropdownMenuItem onselect={triggerAgentReview}>
        <i class="space-y-4"></i>
        <span class="space-y-4">Trigger Agent Review</span>
      </DropdownMenuItem>
      {#if cases.length > 0}
        <DropdownMenuSeparator />
        <div class="space-y-4">
          <p class="space-y-4">
            Send to Case
          </p>
        </div>
        {#each cases as case_}
          <DropdownMenuItem onselect={() => sendToCase(case_.id)}>
            <i class="space-y-4"></i>
            <div class="space-y-4">
              <div class="space-y-4">{case_.title}</div>
              <div class="space-y-4">
                {case_.caseNumber}
              </div>
            </div>
          </DropdownMenuItem>
        {/each}
      {/if}
      <DropdownMenuSeparator />
      <div class="space-y-4">
        <p class="space-y-4">
          Danger Zone
        </p>
      </div>
      <DropdownMenuItem
        onselect={deleteEvidence}
        class="space-y-4"
      >
        <i class="space-y-4"></i>
        <span class="space-y-4">Delete</span>
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
