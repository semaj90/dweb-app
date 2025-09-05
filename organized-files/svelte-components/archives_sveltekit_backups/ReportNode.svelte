<script lang="ts">
  import { createEventDispatcher, onMount } from "svelte";
  import type { Report } from "../../../lib/data/types";
// UI Components
  import * as ContextMenu from "../../../lib/components/ui/context-menu";
  // Icons
  import { Link, Sparkles } from "lucide-svelte";

  export let report: Report;

  const dispatch = createEventDispatcher();

  let nodeElement: HTMLDivElement;
  let editorRef: HTMLElement | null;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;

  // Add local position state for drag-and-drop
  let position = { x: 100, y: 100 };

  function handleMouseDown(event: MouseEvent) {
    if (
      event.target === nodeElement ||
      (event.target as Element)?.classList?.contains("node-header")
    ) {
      isDragging = true;
      dragStartX = event.clientX - position.x;
      dragStartY = event.clientY - position.y;
    }
  }

  function handleMouseMove(event: MouseEvent) {
    if (isDragging) {
      position.x = event.clientX - dragStartX;
      position.y = event.clientY - dragStartY;
    }
  }

  function handleMouseUp() {
    isDragging = false;
  }

  async function saveCitation(text: string) {
    if (!text.trim()) return;

    // Implementation for saving citation
    console.log("Saving citation:", text);
  }

  async function summarizeReport() {
    // Implementation for AI summary
    console.log("Summarizing report");
  }

  onMount(() => {
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  });
</script>

<ContextMenu.Root>
  <ContextMenu.Trigger asChild={false}>
    <div
      bind:this={nodeElement}
      class="mx-auto px-4 max-w-7xl"
      style="left: {position.x}px; top: {position.y}px; z-index: 10;"
      onmousedown={handleMouseDown}
      onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleMouseDown(e);
        }
      "
      role="button"
      tabindex={0}
      aria-label="Drag report node"
    >
      <div
        class="mx-auto px-4 max-w-7xl"
      >
        <h3 class="mx-auto px-4 max-w-7xl">{report.title}</h3>
      </div>
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          {report.content}
        </div>
      </div>
    </div>
  </ContextMenu.Trigger>
  <ContextMenu.Content menu={true}>
    <ContextMenu.Item
      on:select={() => saveCitation(editorRef?.getSelectedText?.() || "")}
    >
      <Link class="mx-auto px-4 max-w-7xl" />
      Save as Citation
    </ContextMenu.Item>
    <ContextMenu.Separator />
    <ContextMenu.Item on:select={summarizeReport}>
      <Sparkles class="mx-auto px-4 max-w-7xl" />
      AI Summary
    </ContextMenu.Item>
  </ContextMenu.Content>
</ContextMenu.Root>

<style>
  .report-node {
    user-select: none;
  }

  .node-header {
    cursor: move;
  }
</style>