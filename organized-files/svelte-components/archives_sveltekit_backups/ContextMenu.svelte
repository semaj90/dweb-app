<script lang="ts">
  import DropdownMenuContent from "$lib/components/ui/dropdown-menu/DropdownMenuContent.svelte";
  import DropdownMenuItem from "$lib/components/ui/dropdown-menu/DropdownMenuItem.svelte";
  import DropdownMenuRoot from "$lib/components/ui/dropdown-menu/DropdownMenuRoot.svelte";
  import DropdownMenuSeparator from "$lib/components/ui/dropdown-menu/DropdownMenuSeparator.svelte";
  import type { Case, Evidence } from "$lib/types/index";
  import { createEventDispatcher, onMount, tick } from "svelte";

  export let x: number;
  export let y: number;
  export let item: Evidence | null;

  const dispatch = createEventDispatcher();
  let cases: Case[] = [];
  let menuOpen = true;
  let triggerEl: HTMLButtonElement;

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
    triggerEl?.focus();
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
  let:menu
  let:trigger
  let:states
  on:openChange={(e) => {
    if (!e.detail.open) closeMenu();
  "
>
  <!-- Hidden trigger for programmatic open -->
  <button
    bind:this={triggerEl}
    use:trigger
    style="position:fixed;left:-9999px;top:-9999px;"
    aria-label="Open context menu"
    tabindex={-1}
  ></button>
  {#if menuOpen}
    <DropdownMenuContent
      {menu}
      class="mx-auto px-4 max-w-7xl"
      style="position:fixed;left:{x}px;top:{y}px;"
      onkeydown={(e) => {
        if (e.detail && e.detail.key === "Escape") closeMenu();
      "
      aria-label="Evidence context menu"
    >
      <div class="mx-auto px-4 max-w-7xl">
        <p class="mx-auto px-4 max-w-7xl">
          Evidence Actions
        </p>
      </div>
      <DropdownMenuItem on:select={viewEvidence}>
        <i class="mx-auto px-4 max-w-7xl"></i>
        <span class="mx-auto px-4 max-w-7xl">View Details</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={editEvidence}>
        <i class="mx-auto px-4 max-w-7xl"></i>
        <span class="mx-auto px-4 max-w-7xl">Edit</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={downloadEvidence}>
        <i class="mx-auto px-4 max-w-7xl"></i>
        <span class="mx-auto px-4 max-w-7xl">Download</span>
      </DropdownMenuItem>
      <DropdownMenuItem on:select={duplicateEvidence}>
        <i class="mx-auto px-4 max-w-7xl"></i>
        <span class="mx-auto px-4 max-w-7xl">Duplicate</span>
      </DropdownMenuItem>
      {#if cases.length > 0}
        <DropdownMenuSeparator />
        <div class="mx-auto px-4 max-w-7xl">
          <p class="mx-auto px-4 max-w-7xl">
            Send to Case
          </p>
        </div>
        {#each cases as case_}
          <DropdownMenuItem on:select={() => sendToCase(case_.id)}>
            <i class="mx-auto px-4 max-w-7xl"></i>
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">{case_.title}</div>
              <div class="mx-auto px-4 max-w-7xl">
                {case_.caseNumber}
              </div>
            </div>
          </DropdownMenuItem>
        {/each}
      {/if}
      <DropdownMenuSeparator />
      <div class="mx-auto px-4 max-w-7xl">
        <p class="mx-auto px-4 max-w-7xl">
          Danger Zone
        </p>
      </div>
      <DropdownMenuItem
        on:select={deleteEvidence}
        class="mx-auto px-4 max-w-7xl"
      >
        <i class="mx-auto px-4 max-w-7xl"></i>
        <span class="mx-auto px-4 max-w-7xl">Delete</span>
      </DropdownMenuItem>
    </DropdownMenuContent>
  {/if}
</DropdownMenuRoot>

<style>
  @keyframes contextMenuFadeIn {
    from {
      opacity: 0;
      transform: scale(0.95);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }
</style>
